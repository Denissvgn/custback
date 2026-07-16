"""End-to-end pipeline tests using the synthetic camera and null output —
no hardware, no mediapipe, no virtual camera module required."""

import time
import threading

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import custback.pipeline as pipeline_mod
from custback.capture import CaptureHealth
from custback.config import AppConfig, RuntimeConfig
from custback.hub import FrameHub
from custback.pipeline import (
    ActivationError,
    Pipeline,
    ReconfigurationUnavailable,
    RestartRequiredError,
)
from custback.pipeline import ConfigConflictError, _restart_only_changes
from custback.vcam import NullOutput


def make_runtime(mode="color", **bg_overrides) -> RuntimeConfig:
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 128, "height": 72, "fps": 60},
            "background": {"mode": mode, **bg_overrides},
            "segmentation": {"backend": "heuristic", "temporal_smoothing": 0.0},
            "output": {"backend": "null", "fps": 60},
            "api": {"enabled": False},
        }
    )
    return RuntimeConfig(cfg)


def wait_for_frame(hub: FrameHub, seq=-1, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame, new_seq = hub.output.get(seq, 0.2)
        if frame is not None:
            return frame, new_seq
    raise AssertionError("no frame produced in time")


def run_pipeline(runtime):
    hub = FrameHub()
    pipeline = Pipeline(runtime, hub)
    pipeline.start()
    return pipeline, hub


def test_empty_slot_waits_through_clear_until_a_frame_arrives():
    hub = FrameHub()
    result = []
    waiter = threading.Thread(target=lambda: result.append(hub.output.get(-1, 1.0)))
    waiter.start()
    time.sleep(0.03)
    hub.output.clear()
    time.sleep(0.03)
    assert waiter.is_alive()
    expected = np.zeros((2, 2, 3), np.uint8)
    hub.publish_output(expected)
    waiter.join(1.0)
    assert not waiter.is_alive()
    assert np.array_equal(result[0][0], expected)


def test_color_mode_replaces_background():
    runtime = make_runtime(mode="color", color=[255, 0, 0])
    pipeline, hub = run_pipeline(runtime)
    try:
        frame, _ = wait_for_frame(hub)
        assert frame.shape == (72, 128, 3)
        # corner = background -> pure blue; center = synthetic "person"
        assert tuple(frame[2, 2]) == (255, 0, 0)
        assert tuple(frame[36, 64]) != (255, 0, 0)
    finally:
        pipeline.stop()


def test_passthrough_mode_is_identity():
    runtime = make_runtime(mode="passthrough")
    pipeline, hub = run_pipeline(runtime)
    try:
        out, _ = wait_for_frame(hub)
        raw, _ = hub.raw.latest()
        assert raw is not None
        # passthrough publishes the capture frame unmodified
        assert out.shape == raw.shape
    finally:
        pipeline.stop()


def test_hot_mode_switch():
    runtime = make_runtime(mode="passthrough")
    pipeline, hub = run_pipeline(runtime)
    try:
        _, seq = wait_for_frame(hub)
        state = pipeline.apply_config_patch(
            {"background": {"mode": "color", "color": [0, 0, 255]}}
        )
        assert state.version == 1
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame, seq = wait_for_frame(hub, seq)
            if tuple(frame[2, 2]) == (0, 0, 255):
                break
        else:
            raise AssertionError("mode switch did not take effect")
        assert hub.stats_dict()["mode"] == "color"
    finally:
        pipeline.stop()


def test_timed_out_candidate_build_does_not_block_frames_or_leak(monkeypatch):
    runtime = make_runtime(mode="color")
    pipeline, hub = run_pipeline(runtime)
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    construction_threads = []
    errors = []

    class CandidateSegmenter:
        device = "cpu"
        last_foreground = None

        def segment(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

        def close(self):
            closed.set()

    def build(_cfg, **_kwargs):
        construction_threads.append(threading.current_thread().name)
        entered.set()
        release.wait(1.0)
        return CandidateSegmenter()

    monkeypatch.setattr(pipeline_mod, "create_segmenter", build)

    def apply():
        try:
            pipeline.apply_config_patch(
                {"segmentation": {"threshold": 0.61}}, timeout=0.15
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=apply)
    try:
        _, sequence = wait_for_frame(hub)
        worker.start()
        assert entered.wait(1.0)
        # Candidate construction is blocked, but the latest frame lane keeps
        # publishing rather than waiting behind model acquisition.
        _, next_sequence = wait_for_frame(hub, sequence, timeout=0.5)
        assert next_sequence != sequence
        worker.join(1.0)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], ReconfigurationUnavailable)
        assert runtime.version == 0

        release.set()
        assert closed.wait(1.0)
        assert construction_threads[0].startswith("custback-segmentation-prepare")
        assert runtime.version == 0
    finally:
        release.set()
        worker.join(1.0)
        pipeline.stop()


def test_stop_owns_timed_out_candidate_preparation(monkeypatch):
    runtime = make_runtime(mode="color")
    pipeline, _hub = run_pipeline(runtime)
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class CandidateSegmenter:
        device = "cpu"
        last_foreground = None

        def segment(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

        def close(self):
            closed.set()

    def build(_cfg, **_kwargs):
        entered.set()
        release.wait(1.0)
        return CandidateSegmenter()

    monkeypatch.setattr(pipeline_mod, "create_segmenter", build)
    try:
        with pytest.raises(
            ReconfigurationUnavailable, match="candidate preparation exceeded"
        ):
            pipeline.apply_config_patch(
                {"segmentation": {"threshold": 0.61}}, timeout=0.05
            )
        assert entered.is_set()
        with pytest.raises(
            ReconfigurationUnavailable,
            match="candidate preparation worker did not stop",
        ):
            pipeline.stop(timeout=0.02)

        release.set()
        assert closed.wait(1.0)
        pipeline.stop(timeout=1.0)
        assert pipeline._preparation_executor is None
        assert runtime.version == 0
    finally:
        release.set()
        if pipeline._preparation_executor is not None:
            pipeline.stop(timeout=1.0)


def test_stop_deadline_bounds_abandoned_candidate_close(monkeypatch):
    runtime = make_runtime(mode="color")
    pipeline, _hub = run_pipeline(runtime)
    construction_entered = threading.Event()
    release_construction = threading.Event()
    close_entered = threading.Event()
    release_close = threading.Event()
    close_done = threading.Event()

    class CandidateSegmenter:
        device = "cpu"
        last_foreground = None

        def segment(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

        def close(self):
            close_entered.set()
            release_close.wait()
            close_done.set()

    def build(_cfg, **_kwargs):
        construction_entered.set()
        release_construction.wait()
        return CandidateSegmenter()

    monkeypatch.setattr(pipeline_mod, "create_segmenter", build)
    safety_release = threading.Timer(1.0, release_close.set)
    try:
        with pytest.raises(
            ReconfigurationUnavailable, match="candidate preparation exceeded"
        ):
            pipeline.apply_config_patch(
                {"segmentation": {"threshold": 0.61}}, timeout=0.03
            )
        assert construction_entered.is_set()

        # The timed-out Future owns its eventual result. Its completion
        # callback starts deterministic cleanup, whose backend close blocks.
        release_construction.set()
        assert close_entered.wait(1.0)
        safety_release.start()

        started = time.monotonic()
        with pytest.raises(
            ReconfigurationUnavailable,
            match="candidate preparation worker did not stop",
        ):
            pipeline.stop(timeout=0.05)
        assert time.monotonic() - started < 0.5
        assert pipeline._preparation_executor is not None

        release_close.set()
        assert close_done.wait(1.0)
        pipeline.stop(timeout=1.0)
        assert pipeline._preparation_executor is None
        assert pipeline._preparation_futures == set()
        assert runtime.version == 0
    finally:
        release_construction.set()
        release_close.set()
        safety_release.cancel()
        if pipeline._preparation_executor is not None:
            pipeline.stop(timeout=1.0)


def test_expired_deadline_before_future_wait_still_owns_candidate(monkeypatch):
    runtime = make_runtime(mode="color")
    pipeline, _hub = run_pipeline(runtime)
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class CandidateSegmenter:
        device = "cpu"
        last_foreground = None

        def segment(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

        def close(self):
            closed.set()

    def build(_cfg, **_kwargs):
        entered.set()
        release.wait(1.0)
        return CandidateSegmenter()

    monkeypatch.setattr(pipeline_mod, "create_segmenter", build)
    original_track = pipeline._track_preparation_future

    def track_after_constructor_starts(future):
        original_track(future)
        assert entered.wait(1.0)

    monkeypatch.setattr(
        pipeline, "_track_preparation_future", track_after_constructor_starts
    )
    current = runtime.read().config
    candidate = current.patched({"segmentation": {"threshold": 0.61}})
    request = pipeline_mod._PatchRequest(candidate, runtime.version)
    try:
        with pytest.raises(
            ReconfigurationUnavailable, match="preparation exceeded"
        ):
            pipeline._prepare_patch_request(
                request,
                current,
                time.monotonic() - 1.0,
            )
        assert request.prepared_activation is None

        release.set()
        assert closed.wait(1.0)
        assert runtime.version == 0
    finally:
        release.set()
        pipeline.stop()


def test_expired_deadline_after_successful_prep_discards_before_enqueue(
    monkeypatch,
):
    runtime = make_runtime(mode="color")
    pipeline, _hub = run_pipeline(runtime)
    closed = threading.Event()

    class CandidateSegmenter:
        device = "cpu"
        last_foreground = None

        def segment(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

        def close(self):
            closed.set()

    monkeypatch.setattr(
        pipeline_mod,
        "create_segmenter",
        lambda _cfg, **_kwargs: CandidateSegmenter(),
    )
    remaining_calls = 0

    def cross_deadline(_deadline):
        nonlocal remaining_calls
        remaining_calls += 1
        if remaining_calls == 1:
            return 1.0  # preparation future receives ownership and completes
        raise ReconfigurationUnavailable(
            "pipeline candidate preparation exceeded the reconfiguration deadline"
        )

    monkeypatch.setattr(pipeline, "_remaining", cross_deadline)
    try:
        with pytest.raises(
            ReconfigurationUnavailable, match="preparation exceeded"
        ):
            pipeline.apply_config_patch(
                {"segmentation": {"threshold": 0.61}}, timeout=1.0
            )

        assert remaining_calls == 2
        assert closed.wait(1.0)
        assert pipeline._requests.empty()
        assert runtime.version == 0
    finally:
        pipeline.stop()


def test_stop_between_preparation_and_enqueue_discards_candidate(monkeypatch):
    runtime = make_runtime(mode="color")
    pipeline, _hub = run_pipeline(runtime)
    prepared = threading.Event()
    allow_enqueue = threading.Event()
    closed = threading.Event()
    errors = []

    class CandidateSegmenter:
        device = "cpu"
        last_foreground = None

        def segment(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

        def close(self):
            closed.set()

    monkeypatch.setattr(
        pipeline_mod,
        "create_segmenter",
        lambda _cfg, **_kwargs: CandidateSegmenter(),
    )
    original_prepare = pipeline._prepare_patch_request

    def pause_after_prepare(*args, **kwargs):
        original_prepare(*args, **kwargs)
        prepared.set()
        assert allow_enqueue.wait(1.0)

    monkeypatch.setattr(pipeline, "_prepare_patch_request", pause_after_prepare)

    def apply():
        try:
            pipeline.apply_config_patch({"segmentation": {"threshold": 0.61}})
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=apply)
    try:
        worker.start()
        assert prepared.wait(1.0)
        pipeline.stop()
        allow_enqueue.set()
        worker.join(1.0)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], ReconfigurationUnavailable)
        assert closed.is_set()
        assert pipeline._requests.empty()
        assert runtime.version == 0
    finally:
        allow_enqueue.set()
        worker.join(1.0)
        if pipeline._preparation_executor is not None:
            pipeline.stop()


def test_storage_mutation_rejects_enqueue_after_stop(monkeypatch):
    runtime = make_runtime(mode="color")
    pipeline, _hub = run_pipeline(runtime)
    enqueue_entered = threading.Event()
    allow_enqueue = threading.Event()
    mutated = threading.Event()
    errors = []
    original_enqueue = pipeline._enqueue_request

    def pause_before_enqueue(request):
        enqueue_entered.set()
        assert allow_enqueue.wait(1.0)
        return original_enqueue(request)

    monkeypatch.setattr(pipeline, "_enqueue_request", pause_before_enqueue)

    def mutate():
        try:
            pipeline.apply_storage_mutation(lambda _cfg: mutated.set())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=mutate)
    try:
        worker.start()
        assert enqueue_entered.wait(1.0)
        pipeline.stop()
        allow_enqueue.set()
        worker.join(1.0)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], ReconfigurationUnavailable)
        assert not mutated.is_set()
        assert pipeline._requests.empty()
    finally:
        allow_enqueue.set()
        worker.join(1.0)


def test_storage_mutation_invalidates_off_lane_candidate(monkeypatch):
    runtime = make_runtime(mode="color")
    pipeline, _hub = run_pipeline(runtime)
    prepared = threading.Event()
    allow_enqueue = threading.Event()
    closed = threading.Event()
    errors = []

    class CandidateSegmenter:
        device = "cpu"
        last_foreground = None

        def segment(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

        def close(self):
            closed.set()

    monkeypatch.setattr(
        pipeline_mod,
        "create_segmenter",
        lambda _cfg, **_kwargs: CandidateSegmenter(),
    )
    original_prepare = pipeline._prepare_patch_request

    def pause_after_prepare(*args, **kwargs):
        original_prepare(*args, **kwargs)
        prepared.set()
        assert allow_enqueue.wait(1.0)

    monkeypatch.setattr(pipeline, "_prepare_patch_request", pause_after_prepare)

    def apply():
        try:
            pipeline.apply_config_patch({"segmentation": {"threshold": 0.61}})
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=apply)
    try:
        worker.start()
        assert prepared.wait(1.0)
        mutation = pipeline.apply_storage_mutation(lambda _cfg: None)
        assert mutation.version == 0
        allow_enqueue.set()
        worker.join(2.0)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], ActivationError)
        assert "assets changed" in str(errors[0])
        assert closed.wait(1.0)
        assert runtime.version == 0
    finally:
        allow_enqueue.set()
        worker.join(1.0)
        pipeline.stop()


def test_config_change_audit_has_origin_version_fields_and_safe_summary(caplog):
    runtime = make_runtime(mode="passthrough")
    pipeline, hub = run_pipeline(runtime)
    try:
        wait_for_frame(hub)
        with caplog.at_level("INFO", logger="custback.pipeline"):
            pipeline.apply_config_patch(
                {"background": {"mode": "color", "color": [0, 0, 255]}},
                origin="api",
            )
        assert "origin=api" in caplog.text
        assert "version=1" in caplog.text
        assert "fields=background.color,background.mode" in caplog.text
        assert "background.mode=color" in caplog.text
        assert "background.color=[0,0,255]" in caplog.text
    finally:
        pipeline.stop()


def test_bad_backdrop_update_keeps_running():
    runtime = make_runtime(mode="color", color=[255, 0, 0])
    pipeline, hub = run_pipeline(runtime)
    try:
        _, seq = wait_for_frame(hub)
        # switching to a nonexistent image must not kill the pipeline
        with pytest.raises(ActivationError):
            pipeline.apply_config_patch(
                {"background": {"mode": "image", "image_path": "/nope.png"}}
            )
        frame, _ = wait_for_frame(hub, seq)
        assert frame is not None
        assert pipeline.running
        assert runtime.snapshot().background.mode == "color"
    finally:
        pipeline.stop()


def test_staged_asset_promotion_commits_atomically_and_rolls_back_install_failure(
    tmp_path, monkeypatch
):
    runtime = make_runtime(mode="color", color=[255, 0, 0])
    pipeline, hub = run_pipeline(runtime)
    final = tmp_path / (("a" * 32) + ".png")
    hidden = tmp_path / (".upload-" + ("a" * 32) + ".png")
    image = np.full((72, 128, 3), 31, np.uint8)
    assert cv2.imwrite(str(hidden), image)
    promotions = []

    def promote():
        promotions.append("promote")
        hidden.replace(final)

    def rollback():
        promotions.append("rollback")
        final.replace(hidden)

    try:
        wait_for_frame(hub)
        state = pipeline.apply_staged_config_patch(
            {"background": {"mode": "image", "image_path": str(final)}},
            {"background": {"mode": "image", "image_path": str(hidden)}},
            promote,
            rollback,
        )
        assert state.version == 1
        assert state.config.background.image_path == str(final)
        assert final.is_file() and not hidden.exists()
        assert promotions == ["promote"]

        # A second candidate is promoted only inside the CAS activation. If
        # pointer installation fails, the path and effective config both roll
        # back before the caller observes the failure.
        hidden2 = tmp_path / (".upload-" + ("b" * 32) + ".png")
        final2 = tmp_path / (("b" * 32) + ".png")
        assert cv2.imwrite(str(hidden2), image)
        events = []

        def promote2():
            events.append("promote")
            hidden2.replace(final2)

        def rollback2():
            events.append("rollback")
            final2.replace(hidden2)

        monkeypatch.setattr(
            pipeline,
            "_install_activation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected pointer install failure")
            ),
        )
        with pytest.raises(RuntimeError, match="pointer install failure"):
            pipeline.apply_staged_config_patch(
                {"background": {"mode": "image", "image_path": str(final2)}},
                {"background": {"mode": "image", "image_path": str(hidden2)}},
                promote2,
                rollback2,
            )
        current = runtime.read()
        assert current.version == 1
        assert current.config.background.image_path == str(final)
        assert hidden2.is_file() and not final2.exists()
        assert events == ["promote", "rollback"]
    finally:
        pipeline.stop()


def test_staged_patch_rejects_a_resource_that_does_not_match_final_config():
    runtime = make_runtime(mode="passthrough")
    pipeline, _hub = run_pipeline(runtime)
    promoted = []
    try:
        with pytest.raises(ValueError, match="active image/video asset path"):
            pipeline.apply_staged_config_patch(
                {"background": {"mode": "color"}},
                {"background": {"mode": "blur"}},
                lambda: promoted.append("promote"),
                lambda: promoted.append("rollback"),
            )
        assert promoted == []
        assert runtime.version == 0
        assert runtime.snapshot().background.mode == "passthrough"
    finally:
        pipeline.stop()


def test_remote_mode_uses_pushed_frames_and_falls_back():
    runtime = make_runtime(
        mode="remote", remote_fallback_mode="color", color=[1, 2, 3]
    )
    pipeline, hub = run_pipeline(runtime)
    remote_session = None
    try:
        slate = Pipeline._privacy_slate((72, 128, 3))
        # No remote client yet -> fixed input-independent privacy slate.
        frame, seq = wait_for_frame(hub)
        assert np.array_equal(frame, slate)
        assert hub.stats_dict()["remote_fallback_reason"] == "no-client"
        assert hub.stats_dict()["remote_fallback_mode"] == "privacy-slate"

        # push an "avatar" frame; it must become the output while fresh
        remote_session = hub.remote_client_connected()
        avatar = np.full((72, 128, 3), (7, 8, 9), dtype=np.uint8)
        deadline = time.monotonic() + 5.0
        used = False
        while time.monotonic() < deadline:
            assert hub.push_remote_frame(avatar, remote_session)
            frame, seq = wait_for_frame(hub, seq)
            if tuple(frame[0, 0]) == (7, 8, 9):
                used = True
                break
        assert used, "remote frame was never used as output"
        assert hub.stats_dict()["remote_frames_used"] >= 1

        # stop pushing -> output must fall back after remote_timeout_ms
        time.sleep(0.5)
        frame, _ = wait_for_frame(hub, seq)
        assert np.array_equal(frame, slate)
        stats = hub.stats_dict()
        assert stats["remote_fallback_reason"] == "stale"
        assert stats["remote_fallback_count"] >= 1
    finally:
        if remote_session is not None:
            hub.remote_client_disconnected(remote_session)
        pipeline.stop()


@pytest.mark.parametrize("value", [0, 127, 255])
def test_emergency_privacy_fallback_never_equals_uniform_raw(value):
    raw = np.full((24, 32, 3), value, np.uint8)
    fallback = Pipeline._emergency_blur(raw)
    assert fallback.shape == raw.shape
    assert fallback.dtype == np.uint8
    assert not np.array_equal(fallback, raw)


def test_privacy_slate_is_independent_of_camera_pixels():
    first = np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3)
    first = (first % 251).astype(np.uint8)
    second = ((first.astype(np.uint16) + 97) % 251).astype(np.uint8)

    assert np.array_equal(
        Pipeline._emergency_blur(first),
        Pipeline._emergency_blur(second),
    )


def test_privacy_gate_rejects_current_and_delayed_near_raw_echoes():
    pipeline = Pipeline(make_runtime(mode="remote"), FrameHub())
    prior = np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3)
    prior = (prior % 251).astype(np.uint8)
    current = ((prior.astype(np.uint16) + 97) % 251).astype(np.uint8)

    near_current = current.copy()
    near_current[0, 0, 0] ^= np.uint8(1)
    guarded, reason = pipeline._guard_remote_output(
        near_current,
        current,
        privacy_safe=True,
    )
    assert reason == "privacy-raw-echo"
    assert np.array_equal(guarded, Pipeline._privacy_slate(current.shape))

    pipeline._remember_raw_frame(prior)
    delayed = prior.copy()
    delayed[-1, -1, -1] ^= np.uint8(1)
    guarded, reason = pipeline._guard_remote_output(
        delayed,
        current,
        privacy_safe=True,
    )
    assert reason == "privacy-delayed-raw-echo"
    assert np.array_equal(guarded, Pipeline._privacy_slate(current.shape))


@pytest.mark.parametrize("quality", [50, 60])
def test_privacy_gate_rejects_current_and_delayed_jpeg_raw_echoes(quality):
    pipeline = Pipeline(make_runtime(mode="remote"), FrameHub())
    raw = np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3)
    raw = (raw % 251).astype(np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg",
        raw,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    assert ok
    jpeg_echo = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert jpeg_echo is not None

    pipeline._recent_raw_fingerprints.clear()
    guarded, reason = pipeline._guard_remote_output(
        jpeg_echo,
        raw,
        privacy_safe=True,
    )
    assert reason == "privacy-raw-echo"
    assert np.array_equal(guarded, Pipeline._privacy_slate(raw.shape))

    current = ((raw.astype(np.uint16) + 97) % 251).astype(np.uint8)
    pipeline._recent_raw_fingerprints.clear()
    pipeline._remember_raw_frame(raw)
    guarded, reason = pipeline._guard_remote_output(
        jpeg_echo,
        current,
        privacy_safe=True,
    )
    assert reason == "privacy-delayed-raw-echo"
    assert np.array_equal(guarded, Pipeline._privacy_slate(raw.shape))


def test_privacy_gate_allows_a_materially_transformed_renderer_frame():
    pipeline = Pipeline(make_runtime(mode="remote"), FrameHub())
    raw = np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3)
    raw = (raw % 251).astype(np.uint8)
    rendered = (raw // np.uint8(64)) * np.uint8(64)

    guarded, reason = pipeline._guard_remote_output(
        rendered,
        raw,
        privacy_safe=True,
    )

    assert reason == ""
    assert np.array_equal(guarded, rendered)


@pytest.mark.parametrize(
    "bad_mask",
    [
        pytest.param(np.full((24, 32), np.nan, np.float32), id="nan"),
        pytest.param(np.full((24, 32), np.inf, np.float32), id="infinite"),
        pytest.param(np.full((24, 32), 1.1, np.float32), id="out-of-range"),
        pytest.param(np.ones((24, 32), np.float32), id="all-foreground"),
        pytest.param(np.ones((23, 32), np.float32), id="wrong-shape"),
    ],
)
def test_remote_mask_validation_rejects_unsafe_masks(bad_mask):
    raw = np.zeros((24, 32, 3), np.uint8)
    with pytest.raises(ValueError):
        Pipeline._validate_mask(bad_mask, raw, privacy_safe=True)


@pytest.mark.parametrize(
    "refined",
    [
        pytest.param(np.zeros((24, 32), np.float64), id="wrong-dtype"),
        pytest.param(np.full((24, 32), np.nan, np.float32), id="nan"),
        pytest.param(np.full((24, 32), -0.1, np.float32), id="out-of-range"),
    ],
)
def test_refined_mask_is_revalidated_before_composition(refined):
    raw = np.zeros((24, 32, 3), np.uint8)

    class Segmenter:
        def segment(self, frame):
            return np.zeros(frame.shape[:2], np.float32)

    class Refiner:
        def refine(self, _mask, _frame):
            return refined

    with pytest.raises(ValueError):
        Pipeline._segment_and_refine_mask(
            Segmenter(), Refiner(), raw, privacy_safe=False
        )


def test_runtime_privacy_gate_protects_vcam_and_preview(monkeypatch):
    raw = np.arange(72 * 128 * 3, dtype=np.uint32).reshape(72, 128, 3)
    raw = (raw % 251).astype(np.uint8)

    class FixedCapture:
        def read(self):
            return raw.copy()

        def close(self):
            pass

    class RecordingOutput:
        paces = False
        fallback_active = False
        fallback_reason = ""

        def __init__(self):
            self.frames = []

        def send(self, frame):
            self.frames.append(frame.copy())

        def close(self):
            pass

    output = RecordingOutput()
    monkeypatch.setattr(pipeline_mod, "open_capture", lambda _cfg: FixedCapture())
    monkeypatch.setattr(
        pipeline_mod,
        "open_output",
        lambda *_args, **_kwargs: output,
    )
    pipeline, hub = run_pipeline(
        make_runtime(mode="remote", remote_fallback_mode="color")
    )
    session = hub.remote_client_connected()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            assert hub.push_remote_frame(raw.copy(), session)
            if hub.stats_dict()["remote_fallback_reason"] in {
                "privacy-raw-echo",
                "privacy-delayed-raw-echo",
            }:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("raw echo did not reach the privacy gate")

        slate = Pipeline._privacy_slate(raw.shape)
        preview, _timestamp = hub.output.latest()
        assert preview is not None
        assert np.array_equal(preview, slate)
        assert output.frames
        assert np.array_equal(output.frames[-1], slate)
        assert all(np.array_equal(frame, slate) for frame in output.frames)
    finally:
        hub.remote_client_disconnected(session)
        pipeline.stop()


def test_remote_sessions_clear_frames_and_reject_prior_session_replay():
    hub = FrameHub()
    first_session = hub.remote_client_connected()
    frame = np.full((4, 6, 3), 9, np.uint8)
    assert hub.push_remote_frame(frame, first_session)
    assert hub.remote_frame_status(1.0)[0] is not None
    hub.remote_client_disconnected(first_session)
    assert hub.remote_frame_status(1.0) == (None, "no-client")
    assert hub.remote_in.latest()[0] is None

    next_session = hub.remote_client_connected()
    assert next_session != first_session
    assert not hub.push_remote_frame(frame, first_session)
    assert hub.remote_frame_status(1.0) == (None, "stale")
    hub.remote_client_disconnected(next_session)


def test_malformed_and_wrong_sized_remote_frames_use_privacy_slate():
    runtime = make_runtime(
        mode="remote", remote_fallback_mode="color", color=[11, 22, 33]
    )
    pipeline, hub = run_pipeline(runtime)
    session = hub.remote_client_connected()
    try:
        slate = Pipeline._privacy_slate((72, 128, 3))
        _, seq = wait_for_frame(hub)
        assert hub.push_remote_frame(np.zeros((72, 128, 3), np.float32), session)
        frame, seq = wait_for_frame(hub, seq)
        assert np.array_equal(frame, slate)
        assert hub.stats_dict()["remote_fallback_reason"] == "invalid"

        assert hub.push_remote_frame(np.zeros((10, 10, 3), np.uint8), session)
        frame, _ = wait_for_frame(hub, seq)
        assert np.array_equal(frame, slate)
        assert hub.stats_dict()["remote_fallback_reason"] == "wrong-size"
    finally:
        hub.remote_client_disconnected(session)
        pipeline.stop()


def test_segmentation_none_remote_fallback_is_input_independent_slate():
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 128, "height": 72, "fps": 60},
            "background": {"mode": "remote", "remote_fallback_mode": "blur"},
            "segmentation": {"backend": "none"},
            "output": {"backend": "null", "fps": 60},
            "api": {"enabled": False},
        }
    )
    pipeline, hub = run_pipeline(RuntimeConfig(cfg))
    try:
        output, _ = wait_for_frame(hub)
        raw, _timestamp = hub.raw.latest()
        assert raw is not None
        assert np.array_equal(output, Pipeline._privacy_slate(raw.shape))
        assert hub.stats_dict()["remote_fallback_reason"] == "no-client"
    finally:
        pipeline.stop()


def test_local_remote_fallback_failure_fails_closed_to_nonraw_frame():
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 32, "height": 24},
            "background": {
                "mode": "remote",
                "remote_fallback_mode": "color",
                "color": [5, 6, 7],
            },
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
            "api": {"enabled": False},
        }
    )

    class BrokenSegmenter:
        last_foreground = None
        device = "cpu"

        def segment(self, _frame):
            raise RuntimeError("local segmentation failed")

    class Refiner:
        def refine(self, mask, _frame):
            return mask

    class Backdrop:
        def frame(self, width, height):
            return np.zeros((height, width, 3), np.uint8)

    resources = pipeline_mod._Resources(
        cfg, 0, None, BrokenSegmenter(), Refiner(), Backdrop(), None
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    raw = np.full((24, 32, 3), 80, np.uint8)
    fallback, reason = pipeline._local_composite(
        resources, raw, privacy_safe=True
    )
    assert reason == "local-failure"
    assert not np.array_equal(fallback, raw)


def test_failed_segmenter_hot_swap_is_rolled_back(monkeypatch):
    runtime = make_runtime(mode="color")
    pipeline, hub = run_pipeline(runtime)
    try:
        wait_for_frame(hub)
        monkeypatch.setitem(__import__("sys").modules, "onnxruntime", None)
        with pytest.raises(ActivationError):
            pipeline.apply_config_patch(
                {
                    "segmentation": {
                        "backend": "rvm",
                        "model_path": "/fake/model.onnx",
                    }
                }
            )
        assert pipeline.running
        assert runtime.snapshot().segmentation.backend == "heuristic"
        assert hub.stats_dict()["segmentation_backend"] == "HeuristicSegmenter"
    finally:
        pipeline.stop()


def test_restart_only_patch_rejected_and_noop_does_not_bump_version():
    runtime = make_runtime(mode="color")
    pipeline, _ = run_pipeline(runtime)
    try:
        state = pipeline.apply_config_patch({"background": {"mode": "color"}})
        assert state.version == 0
        with pytest.raises(RestartRequiredError) as caught:
            pipeline.apply_config_patch({"output": {"fps": 30}})
        # make_runtime uses 60 FPS, so this is a real restart-only change.
        assert caught.value.fields == ("output.fps",)
        assert runtime.version == 0
    finally:
        pipeline.stop()


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"camera": {"device": 1}}, ["camera.device"]),
        ({"camera": {"width": 640}}, ["camera.width"]),
        ({"camera": {"height": 480}}, ["camera.height"]),
        ({"camera": {"fps": 31}}, ["camera.fps"]),
        ({"camera": {"synthetic": True}}, ["camera.synthetic"]),
        ({"camera": {"mirror": True}}, ["camera.mirror"]),
        ({"output": {"backend": "null"}}, ["output.backend"]),
        ({"output": {"device": "camera"}}, ["output.device"]),
        ({"output": {"fps": 31}}, ["output.fps"]),
        ({"output": {"preview": True}}, ["output.preview"]),
        ({"api": {"enabled": False}}, ["api.enabled"]),
        ({"api": {"host": "localhost"}}, ["api.host"]),
        ({"api": {"port": 8711}}, ["api.port"]),
        ({"api": {"allow_non_loopback": True}}, ["api.allow_non_loopback"]),
        (
            {"api": {"allowed_origins": ["http://localhost:8710"]}},
            ["api.allowed_origins"],
        ),
        ({"api": {"token_file": "/tmp/custback-token"}}, ["api.token_file"]),
        ({"api": {"session_ttl_s": 60}}, ["api.session_ttl_s"]),
        (
            {"api": {"tls_certfile": "cert.pem", "tls_keyfile": "key.pem"}},
            ["api.tls_certfile", "api.tls_keyfile"],
        ),
        ({"api": {"ws_max_bytes": 2048}}, ["api.ws_max_bytes"]),
        (
            {"api": {"uploads": {"image_max_bytes": 10 * 1024 * 1024}}},
            ["api.uploads.image_max_bytes"],
        ),
        (
            {"api": {"uploads": {"video_max_bytes": 128 * 1024 * 1024}}},
            ["api.uploads.video_max_bytes"],
        ),
        (
            {"api": {"uploads": {"image_max_pixels": 1_000_000}}},
            ["api.uploads.image_max_pixels"],
        ),
        (
            {"api": {"uploads": {"video_max_width": 1920}}},
            ["api.uploads.video_max_width"],
        ),
        (
            {"api": {"uploads": {"video_max_height": 1080}}},
            ["api.uploads.video_max_height"],
        ),
        (
            {"api": {"uploads": {"storage_max_bytes": 1024**3}}},
            ["api.uploads.storage_max_bytes"],
        ),
        ({"api": {"uploads": {"max_files": 50}}}, ["api.uploads.max_files"]),
        (
            {"avatar": {"url": "https://avatar.example:8711"}},
            ["avatar.url"],
        ),
        (
            {"avatar": {"token_file": "/tmp/avatar-client-token"}},
            ["avatar.token_file"],
        ),
    ],
)
def test_every_restart_only_field_is_classified(patch, expected):
    current = AppConfig()
    candidate = current.patched(patch)
    assert _restart_only_changes(current, candidate) == expected


def test_backdrop_preparation_key_includes_decode_limits(monkeypatch):
    cfg = AppConfig.from_dict(
        {
            "background": {"mode": "color"},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
        }
    )
    candidate = cfg.patched(
        {"api": {"uploads": {"image_max_pixels": 1_024}}}
    )
    observed = []

    class Backdrop:
        def close(self):
            pass

    staged_backdrop = Backdrop()

    def record_create(_cfg, **kwargs):
        observed.append(dict(kwargs))
        return staged_backdrop

    monkeypatch.setattr(pipeline_mod, "create_backdrop", record_create)
    prepared = Pipeline._prepare_activation_off_lane(cfg, candidate)
    resources = pipeline_mod._Resources(
        cfg, 0, None, None, object(), object(), None
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    staged = pipeline._stage_activation(resources, candidate, prepared)

    assert staged.replace_backdrop is True
    assert staged.backdrop is staged_backdrop
    assert observed == [
        {
            "image_max_pixels": 1_024,
            "video_max_width": cfg.api.uploads.video_max_width,
            "video_max_height": cfg.api.uploads.video_max_height,
        }
    ]


def test_concurrent_patches_are_serialized_and_one_conflicts():
    class SynchronizedRuntime(RuntimeConfig):
        def __init__(self, config):
            super().__init__(config)
            self.callers = threading.Barrier(2)

        def read(self):
            state = super().read()
            if threading.current_thread().name.startswith("patch-caller"):
                self.callers.wait(2.0)
            return state

    runtime = SynchronizedRuntime(make_runtime(mode="color").snapshot())
    pipeline, _hub = run_pipeline(runtime)
    outcomes = []

    def apply(color):
        try:
            outcomes.append(
                pipeline.apply_config_patch({"background": {"color": color}})
            )
        except BaseException as exc:
            outcomes.append(exc)

    callers = [
        threading.Thread(target=apply, args=([10, 20, 30],), name="patch-caller-a"),
        threading.Thread(target=apply, args=([40, 50, 60],), name="patch-caller-b"),
    ]
    try:
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(3.0)
        assert all(not caller.is_alive() for caller in callers)
        assert sum(not isinstance(item, BaseException) for item in outcomes) == 1
        assert sum(isinstance(item, ConfigConflictError) for item in outcomes) == 1
        assert runtime.version == 1
    finally:
        pipeline.stop()


def test_noop_patch_rechecks_generation_before_returning(monkeypatch):
    runtime = make_runtime(mode="blur")
    pipeline = Pipeline(runtime, FrameHub())
    writer = runtime._coordinator_writer()
    original_read = runtime.read
    calls = 0

    def racing_read():
        nonlocal calls
        calls += 1
        if calls == 2:
            current = original_read()
            candidate = current.config.patched(
                {"background": {"mode": "color", "color": [4, 5, 6]}}
            )
            writer.commit(candidate, current.version)
        return original_read()

    monkeypatch.setattr(runtime, "read", racing_read)
    state = pipeline.apply_config_patch({})
    assert state.version == 1
    assert state.config.background.mode == "color"
    assert runtime.read().version == state.version


def test_teardown_thread_start_failure_is_deferred_without_raising(monkeypatch):
    pipeline = Pipeline(make_runtime(mode="color"), FrameHub())

    class Resource:
        closes = 0

        def close(self):
            self.closes += 1

    resource = Resource()
    real_start = threading.Thread.start

    def fail_teardown_start(worker):
        if worker.name.startswith("teardown-"):
            raise RuntimeError("thread quota exhausted")
        return real_start(worker)

    monkeypatch.setattr(threading.Thread, "start", fail_teardown_start)
    pipeline._schedule_close(resource, "old backdrop")
    assert resource.closes == 0
    assert pipeline._error is None
    assert len(pipeline._deferred_closes) == 1

    pipeline._drain_deferred_closes()
    assert resource.closes == 1
    assert pipeline._deferred_closes == []


def test_success_ack_precedes_exactly_once_old_resource_close(monkeypatch):
    cfg = AppConfig.from_dict(
        {
            "background": {"mode": "color", "color": [1, 1, 1]},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
        }
    )
    runtime = RuntimeConfig(cfg)
    hub = FrameHub()
    pipeline = Pipeline(runtime, hub)
    close_started = threading.Event()
    release_close = threading.Event()

    class Segmenter:
        device = "cpu"
        last_foreground = None

        def segment(self, _frame):
            raise AssertionError("working segmenter must not be used by background trial")

        def close(self):
            pass

    class Refiner:
        def refine(self, *_args):
            raise AssertionError("working refiner must not be used by background trial")

    class OldBackdrop:
        closes = 0

        def close(self):
            self.closes += 1
            close_started.set()
            release_close.wait(1.0)
            raise RuntimeError("close failure must not revoke commit")

    class NewBackdrop:
        def __init__(self):
            self.closes = 0

        def frame(self, width, height):
            return np.zeros((height, width, 3), np.uint8)

        def close(self):
            self.closes += 1

    class Output:
        paces = False

        def close(self):
            pass

    old = OldBackdrop()
    new = NewBackdrop()
    resources = pipeline_mod._Resources(
        cfg, 0, None, Segmenter(), Refiner(), old, Output()
    )
    monkeypatch.setattr(
        pipeline_mod, "create_backdrop", lambda _cfg, **_kwargs: new
    )
    # Hub post-install failures are non-critical and must not roll back or
    # strand newly installed resource pointers.
    monkeypatch.setattr(
        hub, "clear_remote_frames", lambda: (_ for _ in ()).throw(RuntimeError("clear"))
    )
    monkeypatch.setattr(
        hub, "update_stats", lambda **_kw: (_ for _ in ()).throw(RuntimeError("stats"))
    )
    candidate = cfg.patched({"background": {"mode": "blur"}})
    request = pipeline_mod._PatchRequest(
        candidate,
        0,
        prepared_activation=pipeline._prepare_activation_off_lane(cfg, candidate),
    )
    handler = threading.Thread(
        target=pipeline._handle_patch_request,
        args=(resources, request, np.zeros((72, 128, 3), np.uint8)),
    )
    handler.start()
    assert request.done.wait(1.0)
    assert request.result is not None
    assert runtime.read().config.background.mode == "blur"
    assert close_started.wait(1.0)
    handler.join(1.0)
    assert not handler.is_alive()  # teardown cannot freeze the frame worker
    assert old.closes == 1
    with pytest.raises(ReconfigurationUnavailable, match="teardown worker"):
        pipeline.stop(timeout=0.03)
    release_close.set()
    pipeline.stop(timeout=1.0)
    assert old.closes == 1
    assert resources.backdrop is new


def test_failed_background_trial_preserves_working_processing_state(monkeypatch):
    cfg = AppConfig.from_dict(
        {
            "background": {"mode": "color", "color": [1, 1, 1]},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
        }
    )

    class WorkingSegmenter:
        device = "cpu"
        last_foreground = None
        calls = 0

        def segment(self, _frame):
            self.calls += 1
            return np.ones((72, 128), np.float32)

    class WorkingRefiner:
        calls = 0

        def refine(self, mask, _frame):
            self.calls += 1
            return mask

    class WorkingBackdrop:
        calls = 0

        def frame(self, width, height):
            self.calls += 1
            return np.zeros((height, width, 3), np.uint8)

    class BadBackdrop:
        closes = 0

        def frame(self, _width, _height):
            raise RuntimeError("candidate decode failed")

        def close(self):
            self.closes += 1

    segmenter = WorkingSegmenter()
    refiner = WorkingRefiner()
    backdrop = WorkingBackdrop()
    bad = BadBackdrop()
    resources = pipeline_mod._Resources(
        cfg, 0, None, segmenter, refiner, backdrop, None
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    monkeypatch.setattr(
        pipeline_mod, "create_backdrop", lambda _cfg, **_kwargs: bad
    )
    candidate = cfg.patched({"background": {"color": [2, 2, 2]}})
    prepared = pipeline._prepare_activation_off_lane(cfg, candidate)
    activation = pipeline._stage_activation(resources, candidate, prepared)
    with pytest.raises(ActivationError):
        pipeline._trial_activation(
            resources, activation, np.zeros((72, 128, 3), np.uint8)
        )
    activation.discard()
    assert segmenter.calls == 0
    assert refiner.calls == 0
    assert backdrop.calls == 0
    assert bad.closes == 1


def test_startup_timeout_reports_worker_that_survives_shutdown_request(monkeypatch):
    runtime = make_runtime(mode="passthrough")
    pipeline = Pipeline(runtime, FrameHub())
    entered = threading.Event()
    release = threading.Event()
    real_open_capture = pipeline_mod.open_capture

    def blocked_open(cfg):
        entered.set()
        release.wait(1.0)
        return real_open_capture(cfg)

    monkeypatch.setattr(pipeline_mod, "open_capture", blocked_open)
    with pytest.raises(pipeline_mod.ReconfigurationUnavailable, match="still running"):
        pipeline.start(timeout=0.02)
    assert entered.is_set()
    release.set()
    deadline = time.monotonic() + 2.0
    while pipeline.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not pipeline.running


def test_passthrough_startup_preflights_segmenter_inference(monkeypatch):
    runtime = make_runtime(mode="passthrough")
    pipeline = Pipeline(runtime, FrameHub())

    class BrokenSegmenter:
        device = "cpu"
        last_foreground = None
        produces_matte = False

        def segment(self, _frame):
            raise RuntimeError("inference is broken")

        def close(self):
            pass

    monkeypatch.setattr(
        pipeline_mod, "create_segmenter", lambda _cfg: BrokenSegmenter()
    )
    with pytest.raises(ActivationError, match="inference is broken"):
        pipeline.start()
    assert not pipeline.running
    assert runtime.version == 0


def test_stats_populated():
    runtime = make_runtime(mode="color")
    pipeline, hub = run_pipeline(runtime)
    try:
        wait_for_frame(hub)
        stats = hub.stats_dict()
        assert stats["frames_out"] >= 1
        assert stats["mode"] == "color"
        assert stats["segmentation_backend"] == "HeuristicSegmenter"
        assert stats["output_backend"] == "NullOutput"
    finally:
        pipeline.stop()


def test_slow_capture_repeats_last_safe_output_without_backlog(monkeypatch):
    frame = np.full((72, 128, 3), 90, np.uint8)

    class SlowLatestCapture:
        calls = 0
        frames_read = 0

        def read(self):
            self.calls += 1
            if self.calls == 1 or self.calls % 3 == 1:
                self.frames_read += 1
                return frame.copy()
            return None

        def health_snapshot(self):
            return CaptureHealth(
                backend="slow-fake",
                width=128,
                height=72,
                fps_reported=60.0,
                frames_read=self.frames_read,
            )

        def close(self):
            pass

    capture = SlowLatestCapture()
    monkeypatch.setattr(pipeline_mod, "open_capture", lambda _cfg: capture)
    runtime = make_runtime(mode="color")
    pipeline, hub = run_pipeline(runtime)
    try:
        deadline = time.monotonic() + 2.0
        while hub.stats_dict()["frames_out"] < 12 and time.monotonic() < deadline:
            time.sleep(0.01)
        stats = hub.stats_dict()
        assert stats["frames_out"] >= 12
        assert stats["output_repeated_frames"] > 0
        assert stats["frames_out"] == (
            stats["frames_in"] + stats["output_repeated_frames"]
        )
        assert stats["capture_frames_read"] == stats["frames_in"]
        assert stats["fps_attainment_pct"] is not None
    finally:
        pipeline.stop()


def test_slow_processing_counts_deadline_misses_without_send_pacing(monkeypatch):
    class SlowSegmenter:
        device = "cpu"
        last_foreground = None
        produces_matte = False

        def segment(self, frame):
            time.sleep(0.025)
            return np.ones(frame.shape[:2], np.float32)

        def close(self):
            pass

    monkeypatch.setattr(
        pipeline_mod,
        "create_segmenter",
        lambda _cfg: SlowSegmenter(),
    )
    runtime = make_runtime(mode="color")
    pipeline, hub = run_pipeline(runtime)
    try:
        deadline = time.monotonic() + 2.0
        while hub.stats_dict()["frames_out"] < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        stats = hub.stats_dict()
        assert stats["processing_deadline_misses"] > 0
        assert stats["frame_processing_ms"] >= 16.0
        # NullOutput send/pacing is outside the processing deadline sample.
        assert stats["output_send_ms"] < stats["frame_processing_ms"]
    finally:
        pipeline.stop()


def test_video_counters_reset_when_leaving_the_provider(monkeypatch):
    class VideoStatsBackdrop:
        def frame(self, width, height):
            return np.zeros((height, width, 3), np.uint8)

        def stats_dict(self):
            return {
                "background_video_source_fps": 24.0,
                "background_video_timing_mode": "nominal",
                "background_video_frames_displayed": 20,
                "background_video_frames_skipped": 5,
                "background_video_frames_reused": 2,
                "background_video_skip_ratio": 0.2,
                "background_video_seek_count": 1,
                "background_video_decode_failures": 0,
            }

        def close(self):
            pass

    monkeypatch.setattr(
        pipeline_mod,
        "create_backdrop",
        lambda cfg, **_kwargs: VideoStatsBackdrop()
        if cfg.mode == "video"
        else None,
    )
    runtime = make_runtime(mode="video", video_path="fake.mp4")
    pipeline, hub = run_pipeline(runtime)
    try:
        wait_for_frame(hub)
        assert hub.stats_dict()["background_video_frames_skipped"] == 5
        pipeline.apply_config_patch(
            {"background": {"mode": "color", "color": [0, 0, 0]}}
        )
        deadline = time.monotonic() + 1.0
        while (
            hub.stats_dict()["background_video_frames_skipped"] != 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        stats = hub.stats_dict()
        assert stats["background_video_source_fps"] is None
        assert stats["background_video_timing_mode"] is None
        assert stats["background_video_frames_displayed"] == 0
        assert stats["background_video_frames_skipped"] == 0
        assert stats["background_video_frames_reused"] == 0
    finally:
        pipeline.stop()


def test_fallback_logs_only_transitions_and_recovery(monkeypatch, caplog):
    runtime = make_runtime(mode="color")
    cfg = runtime.snapshot()
    cfg.segmentation.backend = "auto"
    runtime = RuntimeConfig(cfg)
    monkeypatch.setattr(
        pipeline_mod,
        "open_output",
        lambda *_args, **_kwargs: NullOutput(
            fallback_active=True,
            fallback_reason="virtual-camera-unavailable",
        ),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "create_segmenter",
        lambda cfg: pipeline_mod.HeuristicSegmenter(cfg),
    )
    with caplog.at_level("INFO", logger="custback.pipeline"):
        pipeline, hub = run_pipeline(runtime)
        try:
            wait_for_frame(hub)
            pipeline.apply_config_patch(
                {"background": {"color": [10, 20, 30]}},
                origin="api",
            )
            pipeline.apply_config_patch(
                {"segmentation": {"backend": "heuristic"}},
                origin="api",
            )
        finally:
            pipeline.stop()
    assert caplog.text.count("output fallback active") == 1
    assert caplog.text.count("segmentation fallback active") == 1
    assert caplog.text.count("segmentation fallback recovered") == 1
