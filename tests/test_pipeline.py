"""End-to-end pipeline tests using the synthetic camera and null output —
no hardware, no mediapipe, no virtual camera module required."""

import time
import threading
from typing import Any, cast

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import custback.compositor as compositor_mod
import custback.pipeline as pipeline_mod
import custback.runtime_performance as runtime_performance_mod
from custback.capture import CapturedFrame, CaptureHealth
from custback.color import (
    ANALYSIS_LONG_EDGE,
    ColorBehavior,
    ColorEstimate,
    ColorHarmonizer,
    ColorReason,
    ColorSceneSignature,
    ColorTransform,
    HarmonizerPhase,
    IDENTITY_TRANSFORM,
    apply_color_transform,
    bgr_u8_to_linear_rgb,
    linear_rgb_to_bgr_u8,
)
from custback.config import AppConfig, RuntimeConfig
from custback.hub import FrameHub
from custback.matte_diagnostics import (
    MatteCaptureMetadata,
    MatteDiagnosticRecorder,
    MatteFrameEvidence,
    MatteReplayBundle,
)
from custback.matte_live_diagnostics import LocalMatteDiagnosticMonitor
from custback.matte_policy import MatteBackendKind
from custback.light_wrap import LightWrapFrameContext, LightWrapStabilizer
from custback.pipeline import (
    ActivationError,
    Pipeline,
    ReconfigurationUnavailable,
    RestartRequiredError,
)
from custback.pipeline import ConfigConflictError, _restart_only_changes
from custback.segmentation import (
    HeuristicSegmenter,
    MediaPipeSegmenter,
    RVMTelemetry,
    Segmenter,
    segmenter_selection_status,
)
from custback.vcam import NullOutput, OutputSendTiming, VideoOutput


def _captured(
    pixels: np.ndarray,
    sequence: int = 1,
    *,
    captured_at_ns: int | None = None,
    generation: int = 1,
    geometry_generation: int = 1,
    content_rect: tuple[int, int, int, int] | None = None,
) -> CapturedFrame:
    height, width = pixels.shape[:2]
    return CapturedFrame(
        pixels=pixels,
        sequence=sequence,
        captured_at_ns=(
            1_000_000_000 + sequence if captured_at_ns is None else captured_at_ns
        ),
        generation=generation,
        geometry_generation=geometry_generation,
        content_rect=content_rect or (0, 0, width, height),
    )


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


def wait_for_stats(hub: FrameHub, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = hub.stats_dict()
        if predicate(stats):
            return stats
        time.sleep(0.01)
    raise AssertionError("status did not reach the expected state")


def push_remote_for_latest_raw(
    hub: FrameHub,
    frame: np.ndarray,
    session_id: int,
    *,
    timeout: float = 5.0,
) -> bool:
    """Echo the exact epoch attached to the newest eligible raw slot."""

    deadline = time.monotonic() + timeout
    sequence = -1
    while time.monotonic() < deadline:
        _raw, next_sequence = hub.raw.get(
            sequence,
            min(0.1, max(0.0, deadline - time.monotonic())),
        )
        if next_sequence == sequence:
            continue
        sequence = next_sequence
        raw_epoch = hub.remote_raw_epoch_for_sequence(sequence)
        if raw_epoch is None or raw_epoch <= 0:
            continue
        if hub.push_remote_frame(
            frame,
            raw_epoch=raw_epoch,
            session_id=session_id,
        ):
            return True
    return False


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


def test_open_resources_configure_canvas_failure_closes_each_owner_once(
    monkeypatch,
):
    cfg = make_runtime(mode="color").snapshot()
    close_counts = {
        "capture": 0,
        "segmenter": 0,
        "backdrop": 0,
        "output": 0,
    }
    executor_shutdowns = []

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            close_counts[self.name] += 1

    class AnalysisExecutor:
        def __init__(self, *, max_workers, thread_name_prefix):
            assert max_workers == 3
            assert thread_name_prefix == "custback-color-analysis"

        def shutdown(self, *, wait, cancel_futures):
            executor_shutdowns.append((wait, cancel_futures))

    capture = Resource("capture")
    segmenter = Resource("segmenter")
    backdrop = Resource("backdrop")
    output = Resource("output")
    monkeypatch.setattr(
        pipeline_mod,
        "open_capture",
        lambda *_args, **_kwargs: capture,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "create_segmenter",
        lambda *_args, **_kwargs: segmenter,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "refiner_for",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(pipeline_mod, "_build_backdrop", lambda _cfg: backdrop)
    monkeypatch.setattr(
        pipeline_mod,
        "open_output",
        lambda *_args, **_kwargs: output,
    )
    monkeypatch.setattr(pipeline_mod, "ThreadPoolExecutor", AnalysisExecutor)

    class FailingHub:
        @staticmethod
        def configure_canvas(_size):
            raise RuntimeError("configure canvas failed")

    pipeline = Pipeline(RuntimeConfig(cfg), cast(FrameHub, FailingHub()))

    with pytest.raises(RuntimeError, match="configure canvas failed"):
        pipeline._open_resources(pipeline_mod.ConfigState(cfg, 0))

    assert executor_shutdowns == [(True, True)]
    assert close_counts == {
        "capture": 1,
        "segmenter": 1,
        "backdrop": 1,
        "output": 1,
    }


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


def test_hot_patch_repeat_keeps_committed_and_pixel_config_versions_distinct(
    monkeypatch,
):
    runtime = make_runtime(mode="color")
    hub = FrameHub()
    pipeline = Pipeline(runtime, hub)
    original_local_composite = Pipeline._local_composite
    post_commit_processing = threading.Event()
    release_processing = threading.Event()

    def block_first_post_commit_composite(self, *args, **kwargs):
        if runtime.version == 1:
            post_commit_processing.set()
            assert release_processing.wait(2.0)
        return original_local_composite(self, *args, **kwargs)

    monkeypatch.setattr(
        Pipeline,
        "_local_composite",
        block_first_post_commit_composite,
    )
    pipeline.start()
    try:
        before = wait_for_stats(
            hub,
            lambda value: (
                value["frames_out"] >= 3 and value["output_base_config_version"] == 0
            ),
        )
        committed = pipeline.apply_config_patch(
            {
                "background": {"color": [0, 0, 255]},
                "segmentation": {"mask_blur": 3},
                "compositing": {
                    "color_correction": {"mode": "auto"},
                    "light_wrap": 0.0,
                },
            },
            origin="test",
        )
        assert committed.version == 1
        assert post_commit_processing.wait(1.0)

        repeated = wait_for_stats(
            hub,
            lambda value: (
                value["frames_out"] > before["frames_out"]
                and value["config_version"] == committed.version
                and value["output_base_config_version"] == 0
            ),
        )
        assert repeated["matte_rollout"]["config_version"] == committed.version
        assert repeated["color_correction_mode"] == "auto"
        assert repeated["effective_light_wrap"] == 0.0
        assert repeated["effective_mask_blur"] == 3
        # The committed policy is current, while the actual pixel-producing
        # segmenter generation remains tied to the repeated base.
        assert repeated["segmentation_generation"] == 0
        assert repeated["matte_policy"]["controls"]["mask_blur"]["configured"] == 3

        release_processing.set()
        adopted = wait_for_stats(
            hub,
            lambda value: (
                value["config_version"] == committed.version
                and value["output_base_config_version"] == committed.version
            ),
        )
        assert adopted["frames_in"] > repeated["frames_in"]
    finally:
        release_processing.set()
        pipeline.stop()


def test_operator_matte_mitigations_apply_confirm_and_rollback(monkeypatch):
    baseline = (
        make_runtime(mode="color")
        .snapshot()
        .patched({"segmentation": {"backend": "rvm"}})
    )
    runtime = RuntimeConfig(baseline)

    class FakeRvm:
        produces_matte = True
        device = "cuda"
        last_foreground = None

        def __init__(self, cfg):
            self.cfg = cfg
            self.last_downsample_ratio = None

        def segment(self, frame):
            self.last_downsample_ratio = self.cfg.rvm_downsample or 0.4
            self.last_foreground = frame.copy()
            return np.full(frame.shape[:2], 0.5, np.float32)

        def close(self):
            pass

    class FakeMediaPipe:
        produces_matte = False
        matte_backend_kind = MatteBackendKind.CONFIDENCE_MASK_VIDEO
        last_foreground = None

        def __init__(self, cfg):
            self.device = "gpu" if cfg.delegate == "gpu" else "cpu"

        def segment(self, frame):
            return np.full(frame.shape[:2], 0.5, np.float32)

        def close(self):
            pass

    def create(cfg, **_kwargs):
        return FakeMediaPipe(cfg) if cfg.backend == "mediapipe" else FakeRvm(cfg)

    monkeypatch.setattr(pipeline_mod, "create_segmenter", create)
    pipeline, hub = run_pipeline(runtime)

    def patch(value):
        state = pipeline.apply_config_patch(value)
        return wait_for_stats(
            hub,
            lambda stats: (
                stats["config_version"] == state.version
                and stats["output_base_config_version"] == state.version
            ),
        )

    try:
        stats = wait_for_stats(
            hub,
            lambda value: value["effective_rvm_downsample_ratio"] == 0.4,
        )
        assert stats["segmentation_backend"] == "FakeRvm"
        assert stats["segmentation_device"] == "cuda"
        assert stats["segmentation_generation"] == 0
        assert stats["segmentation_produces_matte"] is True
        assert stats["effective_mask_blur"] == 0
        assert stats["effective_edge_refine"] is False
        assert stats["effective_edge_refinement_mode"] == "off"
        assert stats["effective_edge_refinement_radius_px"] == 0
        assert stats["effective_mask_shift"] == 0
        assert stats["effective_temporal_smoothing"] == 0.0
        assert stats["effective_boundary_stabilization_mode"] == "off"
        assert stats["effective_boundary_stabilization_time_constant_s"] == 0.1
        assert stats["effective_boundary_stabilization_max_motion_px_per_s"] == 720.0
        assert stats["effective_use_model_foreground"] is True
        assert stats["effective_light_wrap"] == 0.25
        assert stats["segmentation_selection"] == {
            "schema": "custback.backend-selection",
            "version": 1,
            "requested_backend": "rvm",
            "selected_backend": "rvm",
            "quality_tier": "matting",
            "selection_mode": "explicit",
            "fallback_active": False,
            "fallback_category": "none",
            "fallback_reason": "",
            "guidance": "",
            "active_device": "cuda",
            "active_provider": "cuda",
            "attempts": [
                {
                    "backend": "rvm",
                    "quality_tier": "matting",
                    "preparation_result": "not-run",
                    "activation_result": "selected",
                    "reason_category": "none",
                    "reason": "",
                    "guidance": "",
                }
            ],
        }
        assert stats["matte_policy"]["schema"] == "custback.matte-policy"
        assert stats["matte_policy"]["version"] == 1
        assert stats["matte_policy"]["blend_space"] == (
            baseline.compositing.blend_space
        )
        assert stats["matte_policy"]["selected_backend_kind"] == (
            MatteBackendKind.TRUE_ALPHA_RECURRENT
        )
        assert stats["matte_policy"]["effective"]["rvm_downsample_ratio"] == 0.4
        assert stats["matte_policy"]["effective"]["raw_alpha_mode"] == (
            "native_soft_alpha"
        )
        assert stats["matte_policy"]["controls"]["mask_blur"]["state"] == "bypassed"

        stats = patch({"compositing": {"light_wrap": 0.0}})
        assert stats["effective_light_wrap"] == 0.0
        assert stats["effective_use_model_foreground"] is True
        assert stats["segmentation_generation"] == 0
        stats = patch(
            {
                "compositing": {
                    "light_wrap": baseline.compositing.light_wrap,
                }
            }
        )
        assert stats["effective_light_wrap"] == baseline.compositing.light_wrap
        assert stats["segmentation_generation"] == 0

        stats = patch({"compositing": {"use_model_foreground": False}})
        assert stats["effective_use_model_foreground"] is False
        assert stats["effective_light_wrap"] == baseline.compositing.light_wrap
        assert stats["segmentation_generation"] == 0
        stats = patch(
            {
                "compositing": {
                    "use_model_foreground": (baseline.compositing.use_model_foreground),
                }
            }
        )
        assert stats["effective_use_model_foreground"] is True
        assert stats["segmentation_generation"] == 0

        stats = patch({"segmentation": {"rvm_downsample": 0.5}})
        assert stats["effective_rvm_downsample_ratio"] == 0.5
        assert stats["segmentation_generation"] == 1
        stats = patch(
            {
                "segmentation": {
                    "rvm_downsample": baseline.segmentation.rvm_downsample,
                }
            }
        )
        assert stats["effective_rvm_downsample_ratio"] == 0.4
        assert stats["segmentation_generation"] == 2

        stats = patch({"segmentation": {"mask_shift": -1}})
        assert stats["effective_mask_shift"] == -1
        assert stats["segmentation_generation"] == 3
        stats = patch(
            {"segmentation": {"mask_shift": baseline.segmentation.mask_shift}}
        )
        assert stats["effective_mask_shift"] == 0
        assert stats["segmentation_generation"] == 4

        stats = patch(
            {
                "segmentation": {
                    "backend": "mediapipe",
                    "delegate": "cpu",
                }
            }
        )
        assert stats["segmentation_backend"] == "FakeMediaPipe"
        assert stats["segmentation_produces_matte"] is False
        assert stats["effective_rvm_downsample_ratio"] is None
        assert stats["effective_edge_refine"] is True
        assert stats["effective_edge_refinement_mode"] == "legacy_watershed"
        assert stats["effective_edge_refinement_radius_px"] == 8
        assert stats["effective_mask_blur"] == baseline.segmentation.mask_blur
        assert stats["effective_boundary_stabilization_mode"] == "off"
        assert stats["effective_use_model_foreground"] is False
        assert stats["segmentation_generation"] == 5
        assert stats["segmentation_selection"]["requested_backend"] == "mediapipe"
        assert stats["segmentation_selection"]["selected_backend"] == "mediapipe"
        assert stats["segmentation_selection"]["quality_tier"] == "segmentation"
        assert stats["segmentation_selection"]["fallback_active"] is False
        assert stats["matte_policy"]["selected_backend_kind"] == (
            MatteBackendKind.CONFIDENCE_MASK_VIDEO
        )
        assert stats["matte_policy"]["effective"]["rvm_downsample_ratio"] is None
        assert stats["matte_policy"]["effective"]["raw_alpha_mode"] == (
            "confidence_soft_mask"
        )

        stats = patch({"segmentation": {"edge_refine": False}})
        assert stats["effective_edge_refine"] is False
        assert stats["effective_edge_refinement_mode"] == "off"
        assert stats["effective_edge_refinement_radius_px"] == 0
        assert stats["segmentation_generation"] == 6
        stats = patch(
            {"segmentation": {"edge_refine": baseline.segmentation.edge_refine}}
        )
        assert stats["effective_edge_refine"] is True
        assert stats["effective_edge_refinement_mode"] == "legacy_watershed"
        assert stats["effective_edge_refinement_radius_px"] == 8
        assert stats["segmentation_generation"] == 7

        stats = patch(
            {"segmentation": {"spatial_edge_refinement": {"mode": "stable_guided"}}}
        )
        assert stats["effective_edge_refine"] is True
        assert stats["effective_edge_refinement_mode"] == "stable_guided"
        assert stats["effective_edge_refinement_radius_px"] == 2
        assert stats["segmentation_generation"] == 8
    finally:
        pipeline.stop()


def test_motion_boundary_policy_is_reported_separately_from_legacy_ema(
    monkeypatch,
):
    configured = (
        make_runtime(mode="color")
        .snapshot()
        .patched(
            {
                "segmentation": {
                    "backend": "mediapipe",
                    "temporal_smoothing": 0.8,
                    "boundary_stabilization": {
                        "mode": "motion_aware",
                        "time_constant_s": 0.06,
                        "max_motion_px_per_s": 420.0,
                    },
                }
            }
        )
    )

    class FakeMediaPipe:
        produces_matte = False
        device = "cpu"
        last_foreground = None

        def segment(self, frame):
            return np.full(frame.shape[:2], 0.5, np.float32)

        def close(self):
            pass

    monkeypatch.setattr(
        pipeline_mod,
        "create_segmenter",
        lambda _cfg, **_kwargs: FakeMediaPipe(),
    )
    pipeline, hub = run_pipeline(RuntimeConfig(configured))
    try:
        stats = wait_for_stats(
            hub,
            lambda value: (
                value["effective_boundary_stabilization_mode"] == "motion_aware"
            ),
        )
        assert stats["effective_temporal_smoothing"] == 0.0
        assert stats["effective_boundary_stabilization_time_constant_s"] == 0.06
        assert stats["effective_boundary_stabilization_max_motion_px_per_s"] == 420.0
    finally:
        pipeline.stop()


def test_remote_privacy_slate_survives_matte_mitigation_apply_and_rollback():
    runtime = make_runtime(mode="remote", remote_fallback_mode="blur")
    baseline = runtime.snapshot()
    pipeline, hub = run_pipeline(runtime)
    try:
        frame, sequence = wait_for_frame(hub)
        assert np.array_equal(frame, Pipeline._privacy_slate(frame.shape))
        assert hub.stats_dict()["remote_fallback_mode"] == "privacy-slate"

        for patch in (
            {"compositing": {"light_wrap": 0.0}},
            {"compositing": {"light_wrap": baseline.compositing.light_wrap}},
            {"segmentation": {"mask_shift": -1}},
            {"segmentation": {"mask_shift": baseline.segmentation.mask_shift}},
            {"segmentation": {"spatial_edge_refinement": {"mode": "stable_guided"}}},
            {
                "segmentation": {
                    "spatial_edge_refinement": {
                        "mode": (baseline.segmentation.spatial_edge_refinement.mode)
                    }
                }
            },
        ):
            pipeline.apply_config_patch(patch)
            frame, sequence = wait_for_frame(hub, sequence)
            assert np.array_equal(frame, Pipeline._privacy_slate(frame.shape))
            assert runtime.snapshot().background.mode == "remote"
            assert hub.stats_dict()["remote_fallback_mode"] == "privacy-slate"
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
        with pytest.raises(ReconfigurationUnavailable, match="preparation exceeded"):
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
        with pytest.raises(ReconfigurationUnavailable, match="preparation exceeded"):
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


def test_committed_patch_ack_survives_post_commit_side_effect_failure(monkeypatch):
    runtime = make_runtime(mode="color", color=[255, 0, 0])
    pipeline, hub = run_pipeline(runtime)
    try:
        wait_for_frame(hub)
        monkeypatch.setattr(
            pipeline,
            "_post_install_activation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected post-commit failure")
            ),
        )

        state = pipeline.apply_config_patch(
            {"background": {"color": [0, 0, 255]}},
        )

        assert state.version == 1
        assert state.config.background.color == (0, 0, 255)
        assert runtime.read() == state
        assert pipeline.running
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

        # If promotion mutates storage and then reports a failure, the paired
        # rollback is still armed. No candidate pointer or config generation
        # becomes live.
        partial_hidden = tmp_path / (".upload-" + ("c" * 32) + ".png")
        partial_final = tmp_path / (("c" * 32) + ".png")
        assert cv2.imwrite(str(partial_hidden), image)
        partial_events = []

        def partial_promote():
            partial_events.append("promote")
            partial_hidden.replace(partial_final)
            raise RuntimeError("promotion failed after rename")

        def partial_rollback():
            partial_events.append("rollback")
            partial_final.replace(partial_hidden)

        with pytest.raises(RuntimeError, match="promotion failed after rename"):
            pipeline.apply_staged_config_patch(
                {"background": {"mode": "image", "image_path": str(partial_final)}},
                {
                    "background": {
                        "mode": "image",
                        "image_path": str(partial_hidden),
                    }
                },
                partial_promote,
                partial_rollback,
            )
        current = runtime.read()
        assert current.version == 1
        assert current.config.background.image_path == str(final)
        assert partial_hidden.is_file() and not partial_final.exists()
        assert partial_events == ["promote", "rollback"]

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
    runtime = make_runtime(mode="remote", remote_fallback_mode="color", color=[1, 2, 3])
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
            assert push_remote_for_latest_raw(hub, avatar, remote_session)
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
        assert stats["remote_fallback_reason"] == "awaiting-renderer"
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
        sequence = 0

        def read(self):
            self.sequence += 1
            return _captured(raw.copy(), self.sequence)

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
    monkeypatch.setattr(
        pipeline_mod,
        "open_capture",
        lambda _cfg, _canvas: FixedCapture(),
    )
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
            assert push_remote_for_latest_raw(hub, raw.copy(), session)
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


@pytest.mark.parametrize(
    "mask_path",
    [
        pytest.param("motion-aware-refiner", id="motion-aware-refiner"),
        pytest.param("mediapipe-resized-mask", id="mediapipe-resized-mask"),
        pytest.param("rvm-raw-alpha", id="rvm-raw-alpha"),
        pytest.param("rvm-post-shift-alpha", id="rvm-post-shift-alpha"),
        pytest.param("rvm-model-foreground", id="rvm-model-foreground"),
    ],
)
def test_new_matte_paths_fail_closed_at_sink_and_hub(mask_path):
    """Every shipped matte path retains the final remote privacy boundary."""

    raw = np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3)
    raw = (raw % 251).astype(np.uint8)
    backdrop = np.full(raw.shape, (3, 5, 7), np.uint8)
    valid_alpha = np.full(raw.shape[:2], 0.5, np.float32)
    cfg = _color_integration_config("remote")
    segmentation_patch: dict[str, Any] = {
        "backend": "rvm" if mask_path.startswith("rvm-") else "mediapipe",
        "mask_blur": 0,
        "edge_refine": False,
        "temporal_smoothing": 0.0,
    }
    if mask_path == "motion-aware-refiner":
        segmentation_patch["boundary_stabilization"] = {"mode": "motion_aware"}
    elif mask_path == "rvm-post-shift-alpha":
        segmentation_patch["mask_shift"] = 2
    cfg = cfg.patched(
        {
            "segmentation": segmentation_patch,
            "compositing": {
                "use_model_foreground": mask_path == "rvm-model-foreground"
            },
        }
    )

    class PathSegmenter(Segmenter):
        device = "cpu"
        last_downsample_ratio = 0.5

        def __init__(self):
            super().__init__()
            self.calls = 0
            self.produces_matte = mask_path.startswith("rvm-")
            self.matte_backend_kind = (
                MatteBackendKind.TRUE_ALPHA_RECURRENT
                if self.produces_matte
                else MatteBackendKind.CONFIDENCE_MASK_VIDEO
            )
            self.last_foreground = (
                np.zeros((23, 32, 3), np.uint8)
                if mask_path == "rvm-model-foreground"
                else None
            )

        def segment(self, _frame, *, context=None):
            self._accept_frame_context(context, _frame.shape[:2])
            self.calls += 1
            if mask_path == "rvm-raw-alpha":
                return np.full(raw.shape[:2], np.nan, np.float32)
            if mask_path == "motion-aware-refiner":
                return np.ones(raw.shape[:2], np.float32)
            if mask_path == "mediapipe-resized-mask":
                resized, interpolation = MediaPipeSegmenter._resize_soft_mask(
                    np.ones((12, 16), np.float32),
                    raw.shape[:2],
                )
                assert interpolation == "linear"
                return resized
            if mask_path == "rvm-post-shift-alpha":
                alpha = np.ones(raw.shape[:2], np.float32)
                alpha[[0, -1], :] = 0.0
                alpha[:, [0, -1]] = 0.0
                return alpha
            return valid_alpha.copy()

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

    segmenter = PathSegmenter()
    refiner = pipeline_mod.refiner_for(cfg.segmentation, segmenter)
    assert (refiner.cfg.boundary_stabilization.mode == "motion_aware") is (
        mask_path == "motion-aware-refiner"
    )
    output = RecordingOutput()
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        segmenter,
        refiner,
        _FixedBackdrop(backdrop),
        output,
    )
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    expected_reason = (
        "local-failure"
        if mask_path == "rvm-model-foreground"
        else (
            "segmentation-invalid-mask"
            if mask_path == "rvm-raw-alpha"
            else "segmentation-all-foreground"
        )
    )
    slate = Pipeline._privacy_slate(raw.shape)

    try:
        guarded, reason = pipeline._local_composite(
            resources,
            raw,
            privacy_safe=True,
        )
        assert reason == expected_reason
        np.testing.assert_array_equal(guarded, slate)
        if mask_path == "rvm-raw-alpha":
            # Raw validation happens before the real refiner, so rejected
            # non-finite backend alpha cannot enter temporal history.
            assert refiner._prev is None

        pipeline_mod._send_output_with_timing(
            output,
            guarded,
            copy_frame=True,
        )
        hub.publish_output(guarded)

        assert segmenter.calls == 1
        assert output.frames
        np.testing.assert_array_equal(output.frames[-1], slate)
        np.testing.assert_array_equal(hub.output.latest()[0], slate)
        assert not np.array_equal(raw, slate)
    finally:
        resources.close()


def test_privacy_capacity_exhaustion_revokes_renderer_and_slates_all_sinks(
    monkeypatch,
):
    raw = np.arange(72 * 128 * 3, dtype=np.uint32).reshape(72, 128, 3)
    raw = (raw % 251).astype(np.uint8)

    class FixedCapture:
        sequence = 0

        def read(self):
            self.sequence += 1
            return _captured(raw.copy(), self.sequence)

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
    monkeypatch.setattr(
        pipeline_mod,
        "open_capture",
        lambda _cfg, _canvas: FixedCapture(),
    )
    monkeypatch.setattr(pipeline_mod, "open_output", lambda *_args, **_kwargs: output)
    runtime = make_runtime(mode="remote", remote_fallback_mode="color")
    hub = FrameHub()
    pipeline = Pipeline(runtime, hub, raw_fingerprint_capacity=2)
    pipeline.start()
    first_session = hub.remote_client_connected()
    second_session = None
    try:
        deadline = time.monotonic() + 2.0
        while hub.remote_session_valid(first_session) and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not hub.remote_session_valid(first_session)
        assert pipeline._privacy_history_exhausted
        assert not hub.push_remote_frame(
            raw.copy(),
            raw_epoch=1,
            session_id=first_session,
        )

        # Reauthentication cannot turn exhausted evidence into an allow.
        second_session = hub.remote_client_connected()
        pushed = push_remote_for_latest_raw(
            hub,
            raw.copy(),
            second_session,
            timeout=0.5,
        )
        assert pushed or not hub.remote_session_valid(second_session)
        deadline = time.monotonic() + 2.0
        while hub.remote_session_valid(second_session) and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not hub.remote_session_valid(second_session)

        slate = Pipeline._privacy_slate(raw.shape)
        preview, _timestamp = hub.output.latest()
        assert preview is not None
        assert np.array_equal(preview, slate)
        assert output.frames
        assert all(np.array_equal(frame, slate) for frame in output.frames)
        assert hub.stats_dict()["remote_fallback_reason"] == "privacy-history-exhausted"
    finally:
        if second_session is not None:
            hub.remote_client_disconnected(second_session)
        hub.remote_client_disconnected(first_session)
        pipeline.stop()


def test_prior_session_raw_echo_remains_rejected():
    hub = FrameHub()
    pipeline = Pipeline(make_runtime(mode="remote"), hub)
    prior = np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3)
    prior = (prior % 251).astype(np.uint8)
    current = ((prior.astype(np.uint16) + 97) % 251).astype(np.uint8)

    first_session = hub.remote_client_connected()
    pipeline._record_remote_raw_frame(prior)
    hub.remote_client_disconnected(first_session)
    second_session = hub.remote_client_connected()
    pipeline._record_remote_raw_frame(current)
    try:
        guarded, reason = pipeline._guard_remote_output(
            prior.copy(), current, privacy_safe=True
        )
        assert reason == "privacy-delayed-raw-echo"
        assert np.array_equal(guarded, Pipeline._privacy_slate(current.shape))
    finally:
        hub.remote_client_disconnected(second_session)


def test_remote_sessions_clear_frames_and_reject_prior_session_replay():
    hub = FrameHub()
    first_session = hub.remote_client_connected()
    frame = np.full((4, 6, 3), 9, np.uint8)
    hub.publish_remote_raw(frame, 1)
    assert hub.push_remote_frame(frame, raw_epoch=1, session_id=first_session)
    assert hub.remote_frame_status(1.0)[0] is not None
    hub.remote_client_disconnected(first_session)
    assert hub.remote_frame_status(1.0) == (None, "no-client")
    assert hub.remote_in.latest()[0] is None

    next_session = hub.remote_client_connected()
    assert next_session != first_session
    hub.publish_remote_raw(frame, 2)
    assert not hub.push_remote_frame(
        frame,
        raw_epoch=2,
        session_id=first_session,
    )
    assert hub.remote_frame_status(1.0) == (None, "stale")
    hub.remote_client_disconnected(next_session)


def test_remote_session_invalidation_is_linearized_with_frame_ownership():
    hub = FrameHub()
    session = hub.remote_client_connected()
    frame = np.full((4, 6, 3), 17, np.uint8)
    hub.publish_remote_raw(frame, 1)
    assert hub.push_remote_frame(frame, raw_epoch=1, session_id=session)

    assert hub.invalidate_remote_session(session)
    assert not hub.remote_session_valid(session)
    assert hub.remote_frame_status(1.0) == (None, "no-client")
    assert not hub.push_remote_frame(frame, raw_epoch=1, session_id=session)

    replacement = hub.remote_client_connected()
    assert replacement != session
    assert hub.remote_session_valid(replacement)
    hub.publish_remote_raw(frame, 2)
    assert hub.push_remote_frame(frame, raw_epoch=2, session_id=replacement)
    hub.remote_client_disconnected(replacement)


def test_remote_session_lifecycle_notifications_are_synchronous():
    hub = FrameHub()
    events: list[tuple[str, int]] = []
    hub.set_remote_lifecycle_listener(
        lambda event, session: events.append((event, session))
    )

    first = hub.remote_client_connected()
    assert events == [("connected", first)]
    hub.remote_client_disconnected(first)
    assert events[-1] == ("disconnected", first)

    replacement = hub.remote_client_connected()
    assert events[-1] == ("connected", replacement)
    assert hub.invalidate_remote_session(replacement)
    assert events[-1] == ("invalidated", replacement + 1)


def test_remote_session_connect_aborts_when_privacy_fence_fails():
    hub = FrameHub()

    def fail_fence(_event: str, _session: int) -> None:
        raise RuntimeError("fence unavailable")

    hub.set_remote_lifecycle_listener(fail_fence)
    with pytest.raises(RuntimeError, match="fence unavailable"):
        hub.remote_client_connected()

    assert hub.active_remote_session() is None
    assert not hub.stats_dict()["remote_connected"]


def test_malformed_and_wrong_sized_remote_frames_use_privacy_slate():
    runtime = make_runtime(
        mode="remote", remote_fallback_mode="color", color=[11, 22, 33]
    )
    pipeline, hub = run_pipeline(runtime)
    session = hub.remote_client_connected()
    try:
        slate = Pipeline._privacy_slate((72, 128, 3))
        _, seq = wait_for_frame(hub)
        assert push_remote_for_latest_raw(
            hub,
            np.zeros((72, 128, 3), np.float32),
            session,
        )
        wait_for_stats(
            hub,
            lambda status: status["remote_fallback_reason"] == "invalid",
        )
        frame, seq = wait_for_frame(hub, seq)
        assert np.array_equal(frame, slate)

        assert push_remote_for_latest_raw(
            hub,
            np.zeros((10, 10, 3), np.uint8),
            session,
        )
        wait_for_stats(
            hub,
            lambda status: status["remote_fallback_reason"] == "wrong-size",
        )
        frame, _ = wait_for_frame(hub, seq)
        assert np.array_equal(frame, slate)
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
    fallback, reason = pipeline._local_composite(resources, raw, privacy_safe=True)
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


def test_invalid_and_restart_visual_patches_never_prepare_resources(monkeypatch):
    runtime = make_runtime(mode="color")
    pipeline, hub = run_pipeline(runtime)

    def unexpected_preparation(*_args, **_kwargs):
        pytest.fail("rejected patch reached resource preparation")

    monkeypatch.setattr(pipeline, "_prepare_patch_request", unexpected_preparation)
    try:
        _, sequence = wait_for_frame(hub)
        with pytest.raises(ValueError):
            pipeline.apply_config_patch({"background": {"anchor_x": 1.01}})
        with pytest.raises(RestartRequiredError):
            pipeline.apply_config_patch({"camera": {"rotation": 90}})

        assert runtime.version == 0
        assert runtime.snapshot().background.anchor_x == 0.5
        assert runtime.snapshot().camera.rotation == 0
        wait_for_frame(hub, sequence)
        assert pipeline.running
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
        ({"camera": {"fit_mode": "cover"}}, ["camera.fit_mode"]),
        ({"camera": {"anchor_x": 0.25}}, ["camera.anchor_x"]),
        ({"camera": {"anchor_y": 0.75}}, ["camera.anchor_y"]),
        ({"camera": {"rotation": 90}}, ["camera.rotation"]),
        ({"camera": {"mirror": True}}, ["camera.mirror"]),
        (
            {"output": {"width": 1920, "height": 1080}},
            ["output.height", "output.width"],
        ),
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
            {"background": {"camera_device": 2}},
            ["background.camera_device"],
        ),
        (
            {"backdrop_targets": {"side-camera": {"source": 2}}},
            ["backdrop_targets.side-camera"],
        ),
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


@pytest.mark.parametrize(
    "patch",
    [
        {"background": {"fit_mode": "contain"}},
        {"background": {"anchor_x": 0.25, "anchor_y": 0.75}},
        {"compositing": {"blend_space": "linear_srgb"}},
        {"compositing": {"color_correction": {"mode": "auto"}}},
        {"compositing": {"color_correction": {"strength": 0.7}}},
    ],
)
def test_background_geometry_and_color_policy_remain_hot(patch):
    current = AppConfig()
    candidate = current.patched(patch)
    assert _restart_only_changes(current, candidate) == []


def test_backdrop_provider_and_visual_keys_separate_video_lifetime_from_geometry(
    monkeypatch,
):
    current = AppConfig.from_dict(
        {
            "background": {
                "mode": "video",
                "video_path": "/operator/background.mp4",
            },
            "segmentation": {"backend": "heuristic"},
        }
    )
    candidate = current.patched(
        {
            "background": {
                "fit_mode": "contain",
                "anchor_x": 0.25,
                "anchor_y": 0.75,
            }
        }
    )
    monkeypatch.setattr(
        pipeline_mod,
        "create_backdrop",
        lambda *_args, **_kwargs: pytest.fail(
            "presentation-only geometry must not reopen the video provider"
        ),
    )

    activation = Pipeline._prepare_activation_off_lane(current, candidate)

    assert pipeline_mod._backdrop_provider_key(
        current
    ) == pipeline_mod._backdrop_provider_key(candidate)
    assert pipeline_mod._backdrop_visual_key(
        current
    ) != pipeline_mod._backdrop_visual_key(candidate)
    hash(pipeline_mod._visual_state_key(candidate))
    assert activation.replace_backdrop is False
    assert activation.visual_state_changed is True


def test_hot_geometry_commit_refits_reused_provider_and_failed_trial_is_inert(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "anchored.png"
    pixels = np.zeros((64, 32, 3), np.uint8)
    pixels[:16] = (10, 20, 30)
    pixels[-16:] = (210, 220, 230)
    assert cv2.imwrite(str(path), pixels)
    cfg = AppConfig.from_dict(
        {
            "camera": {"width": 64, "height": 32},
            "background": {
                "mode": "image",
                "image_path": str(path),
                "fit_mode": "cover",
                "anchor_y": 0.0,
            },
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
        }
    )
    runtime = RuntimeConfig(cfg)
    pipeline = Pipeline(runtime, FrameHub())
    backdrop = pipeline_mod._build_backdrop(cfg)
    assert backdrop is not None

    class Segmenter:
        device = "cpu"
        last_foreground = None

        def close(self):
            pass

    class Refiner:
        pass

    resources = pipeline_mod._Resources(
        cfg,
        0,
        None,
        Segmenter(),
        Refiner(),
        backdrop,
        NullOutput(64, 32, 30),
    )
    trial_frame = _captured(np.zeros((32, 64, 3), np.uint8))
    top_anchored = backdrop.frame(64, 32).copy()

    committed_cfg = cfg.patched({"background": {"anchor_y": 1.0}})
    request = pipeline_mod._PatchRequest(
        committed_cfg,
        0,
        prepared_activation=pipeline._prepare_activation_off_lane(
            cfg,
            committed_cfg,
        ),
    )
    pipeline._handle_patch_request(resources, request, trial_frame)

    assert request.error is None
    assert request.result is not None and request.result.version == 1
    assert resources.backdrop is backdrop
    assert resources.visual_generation == 1
    assert backdrop.geometry.anchor_y == 1.0
    bottom_anchored = backdrop.frame(64, 32).copy()
    assert not np.array_equal(top_anchored, bottom_anchored)

    failed_cfg = committed_cfg.patched({"background": {"anchor_y": 0.25}})
    failed = pipeline_mod._PatchRequest(
        failed_cfg,
        1,
        prepared_activation=pipeline._prepare_activation_off_lane(
            committed_cfg,
            failed_cfg,
        ),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "apply_transform",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("detached geometry trial failed")
        ),
    )
    pipeline._handle_patch_request(resources, failed, trial_frame)

    assert failed.result is None
    assert isinstance(failed.error, ActivationError)
    assert runtime.read().version == resources.version == 1
    assert resources.cfg.background.anchor_y == 1.0
    assert resources.visual_generation == 1
    assert backdrop.geometry.anchor_y == 1.0
    assert np.array_equal(backdrop.frame(64, 32), bottom_anchored)
    backdrop.close()


def test_visual_generation_changes_without_replacing_reused_backdrop():
    current = AppConfig.from_dict(
        {
            "background": {
                "mode": "video",
                "video_path": "/operator/background.mp4",
            },
            "segmentation": {"backend": "heuristic"},
        }
    )
    candidate = current.patched(
        {
            "compositing": {
                "color_correction": {
                    "mode": "auto",
                    "adaptation_time_s": 1.5,
                }
            }
        }
    )
    backdrop = object()
    refiner = object()
    resources = pipeline_mod._Resources(
        current,
        0,
        None,
        None,
        refiner,
        backdrop,
        None,
        visual_generation=4,
    )
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    pipeline._active_state = pipeline_mod.ConfigState(current, 0)
    prepared = Pipeline._prepare_activation_off_lane(current, candidate)
    staged = pipeline._stage_activation(resources, candidate, prepared)

    old_backdrop, _old_segmenter, _old_cfg = pipeline._install_activation(
        resources, staged, 1
    )

    assert old_backdrop is None
    assert resources.backdrop is backdrop
    assert resources.visual_generation == 5
    assert resources.cfg.compositing.color_correction.mode == "auto"


def test_light_wrap_policy_hot_activation_installs_fresh_generation() -> None:
    current = AppConfig.from_dict(
        {
            "camera": {"width": 32, "height": 24},
            "background": {
                "mode": "video",
                "video_path": "/operator/background.mp4",
            },
            "segmentation": {"backend": "heuristic"},
            "compositing": {
                "light_wrap": 0.8,
                "light_wrap_stabilization": {
                    "mode": "temporal_bounded",
                    "time_constant_s": 0.12,
                },
            },
            "output": {"backend": "null"},
        }
    )
    candidate = current.patched(
        {"compositing": {"light_wrap_stabilization": {"time_constant_s": 0.2}}}
    )

    class Segmenter:
        device = "cpu"
        matte_backend_kind = MatteBackendKind.BINARY_COARSE
        last_foreground = None
        last_downsample_ratio = None

        def close(self) -> None:
            pass

    class Refiner:
        def close(self) -> None:
            pass

    class Backdrop:
        def close(self) -> None:
            pass

    resources = pipeline_mod._Resources(
        current,
        0,
        None,
        Segmenter(),
        Refiner(),
        Backdrop(),
        NullOutput(32, 24, 30),
    )
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    pipeline._active_state = pipeline_mod.ConfigState(current, 0)
    original = resources.light_wrap_stabilizer
    assert original is not None
    original.update(
        np.full((4, 4, 3), 64.0, np.float32),
        LightWrapFrameContext(0, 0, ("installed-video",)),
        value_scale=255.0,
        channel_order="bgr",
    )
    original_snapshot = original.snapshot()

    prepared = Pipeline._prepare_activation_off_lane(current, candidate)
    assert prepared.replace_light_wrap_stabilizer is True
    assert prepared.light_wrap_stabilizer is not None
    assert prepared.light_wrap_stabilizer.snapshot().updates == 0
    staged = pipeline._stage_activation(resources, candidate, prepared)
    pipeline._trial_activation(
        resources,
        staged,
        _captured(np.full((24, 32, 3), 90, np.uint8)),
    )

    assert original.snapshot() == original_snapshot
    assert staged.light_wrap_stabilizer is not None
    assert staged.light_wrap_stabilizer.snapshot().updates == 0
    pipeline._install_activation(resources, staged, 1)

    assert resources.light_wrap_stabilizer is staged.light_wrap_stabilizer
    assert resources.light_wrap_stabilizer is not original
    installed_stabilizer = resources.light_wrap_stabilizer
    assert installed_stabilizer is not None
    assert installed_stabilizer.snapshot().updates == 0
    assert resources.light_wrap_generation == 1
    assert resources.segmentation_generation == 0
    assert resources.cfg.compositing.light_wrap_stabilization.time_constant_s == 0.2
    resources.close()


def test_light_wrap_state_key_tracks_only_dynamic_wrap_semantics() -> None:
    active = AppConfig.from_dict(
        {
            "background": {
                "mode": "video",
                "video_path": "/operator/background.mp4",
            },
            "compositing": {
                "light_wrap": 0.8,
                "light_wrap_stabilization": {
                    "mode": "temporal_bounded",
                    "time_constant_s": 0.12,
                },
            },
        }
    )
    key = pipeline_mod._light_wrap_state_key(active)

    assert key != ("off",)
    assert (
        pipeline_mod._light_wrap_state_key(
            active.patched({"segmentation": {"threshold": 0.73}})
        )
        == key
    )
    assert (
        pipeline_mod._light_wrap_state_key(
            active.patched({"compositing": {"color_correction": {"strength": 0.75}}})
        )
        == key
    )
    for patch in (
        {"background": {"anchor_x": 0.25}},
        {"background": {"video_path": "/operator/replacement.mp4"}},
        {"compositing": {"blend_space": "linear_srgb"}},
        {"compositing": {"light_wrap_stabilization": {"time_constant_s": 0.2}}},
    ):
        assert pipeline_mod._light_wrap_state_key(active.patched(patch)) != key
    assert pipeline_mod._light_wrap_state_key(
        active.patched({"compositing": {"light_wrap": 0.0}})
    ) == ("off",)
    assert pipeline_mod._light_wrap_state_key(
        active.patched({"compositing": {"light_wrap_stabilization": {"mode": "off"}}})
    ) == ("off",)
    assert pipeline_mod._light_wrap_state_key(
        active.patched(
            {
                "background": {
                    "mode": "image",
                    "image_path": "/operator/background.png",
                }
            }
        )
    ) == ("off",)

    class Backdrop:
        @staticmethod
        def temporal_frame_timing():
            return pipeline_mod.BackdropFrameTiming(3, 100_000_000)

    class Resources:
        backdrop = Backdrop()
        canvas_size = (1280, 720)
        light_wrap_generation = 4
        visual_generation = 1

    resources = Resources()
    first_context = Pipeline._light_wrap_frame_context(
        cast(pipeline_mod._Resources, resources)
    )
    resources.visual_generation += 1
    second_context = Pipeline._light_wrap_frame_context(
        cast(pipeline_mod._Resources, resources)
    )
    assert first_context is not None
    assert second_context is not None
    assert first_context.source_token == second_context.source_token
    assert Pipeline._backdrop_diagnostic_identity(
        cast(pipeline_mod._Resources, resources)
    ) == {
        "provider": "Backdrop",
        "visual_generation": 2,
    }


def test_failed_light_wrap_trial_preserves_live_state_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = AppConfig.from_dict(
        {
            "camera": {"width": 32, "height": 24},
            "background": {
                "mode": "video",
                "video_path": "/operator/background.mp4",
            },
            "segmentation": {"backend": "heuristic"},
            "compositing": {
                "light_wrap": 0.8,
                "light_wrap_stabilization": {
                    "mode": "temporal_bounded",
                    "time_constant_s": 0.12,
                },
            },
            "output": {"backend": "null"},
        }
    )
    candidate = current.patched(
        {"compositing": {"light_wrap_stabilization": {"time_constant_s": 0.2}}}
    )

    class Segmenter:
        device = "cpu"
        matte_backend_kind = MatteBackendKind.BINARY_COARSE
        last_foreground = None
        last_downsample_ratio = None

        def close(self) -> None:
            pass

    class Closable:
        def close(self) -> None:
            pass

    resources = pipeline_mod._Resources(
        current,
        0,
        None,
        Segmenter(),
        Closable(),
        Closable(),
        NullOutput(32, 24, 30),
    )
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    live = resources.light_wrap_stabilizer
    assert live is not None
    live.update(
        np.full((4, 4, 3), 64.0, np.float32),
        LightWrapFrameContext(0, 0, ("installed-video",)),
        value_scale=255.0,
        channel_order="bgr",
    )
    before = live.snapshot()
    prepared = Pipeline._prepare_activation_off_lane(current, candidate)
    request = pipeline_mod._PatchRequest(
        candidate,
        0,
        prepared_activation=prepared,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "prepare_light_wrap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("candidate wrap trial failed")
        ),
    )

    pipeline._handle_patch_request(
        resources,
        request,
        _captured(np.full((24, 32, 3), 90, np.uint8)),
    )

    assert isinstance(request.error, ActivationError)
    assert request.result is None
    assert resources.light_wrap_stabilizer is live
    assert live.snapshot() == before
    assert resources.light_wrap_generation == 0
    assert resources.segmentation_generation == 0
    assert resources.version == 0
    assert resources.cfg == current
    assert pipeline.runtime.read().config == current
    resources.close()


def test_failed_visual_install_rolls_back_generation_and_reused_resources():
    current = AppConfig()
    candidate = current.patched(
        {
            "background": {"anchor_x": 0.25},
            "compositing": {"color_correction": {"mode": "auto"}},
        }
    )
    old_backdrop = object()
    old_segmenter = object()
    old_refiner = object()
    old_harmonizer = ColorHarmonizer(0.8, mode=current.background.mode)
    old_harmonizer.reset(5.0, reason=ColorReason.INVALID)
    old_harmonizer_snapshot = old_harmonizer.snapshot()
    replacement_harmonizer = ColorHarmonizer(0.8, mode=candidate.background.mode)
    old_light_wrap = LightWrapStabilizer(0.12)
    old_light_wrap.update(
        np.full((4, 4, 3), 32.0, np.float32),
        LightWrapFrameContext(0, 0, ("old-wrap",)),
        value_scale=255.0,
        channel_order="bgr",
    )
    old_light_wrap_snapshot = old_light_wrap.snapshot()
    replacement_light_wrap = LightWrapStabilizer(0.2)

    class FailingResources:
        version = 7
        capture = None
        segmenter = old_segmenter
        refiner = old_refiner
        backdrop = old_backdrop
        output = None
        visual_generation = 3
        harmonizer = old_harmonizer
        light_wrap_stabilizer = old_light_wrap
        light_wrap_generation = 4
        color_reset_token = ("stable",)

        def __init__(self):
            self._cfg = current

        @property
        def cfg(self):
            return self._cfg

        @cfg.setter
        def cfg(self, value):
            if value is candidate:
                raise RuntimeError("injected install failure")
            self._cfg = value

    resources = FailingResources()
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    old_state = pipeline_mod.ConfigState(current, 7)
    pipeline._active_state = old_state
    activation = pipeline_mod._Activation(
        candidate=candidate,
        refiner=old_refiner,
        backdrop=old_backdrop,
        visual_state_changed=True,
        replace_harmonizer=True,
        harmonizer=replacement_harmonizer,
        replace_light_wrap_stabilizer=True,
        light_wrap_stabilizer=replacement_light_wrap,
    )

    with pytest.raises(RuntimeError, match="injected install failure"):
        pipeline._install_activation(
            cast(pipeline_mod._Resources, resources), activation, 8
        )

    assert resources.cfg is current
    assert resources.version == 7
    assert resources.segmenter is old_segmenter
    assert resources.refiner is old_refiner
    assert resources.backdrop is old_backdrop
    assert resources.harmonizer is old_harmonizer
    assert resources.harmonizer.snapshot() == old_harmonizer_snapshot
    assert resources.light_wrap_stabilizer is old_light_wrap
    assert resources.light_wrap_stabilizer.snapshot() == old_light_wrap_snapshot
    assert resources.light_wrap_generation == 4
    assert resources.color_reset_token == ("stable",)
    assert resources.visual_generation == 3
    assert pipeline._active_state is old_state


def test_backdrop_preparation_key_includes_decode_limits(monkeypatch):
    cfg = AppConfig.from_dict(
        {
            "background": {"mode": "image", "image_path": "/operator/image.png"},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
        }
    )
    candidate = cfg.patched({"api": {"uploads": {"image_max_pixels": 1_024}}})
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
    resources = pipeline_mod._Resources(cfg, 0, None, None, object(), object(), None)
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


def test_operator_backdrop_target_selection_is_transactional(monkeypatch):
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 128, "height": 72, "fps": 60},
            "background": {"mode": "color", "camera_target": "side-camera"},
            "backdrop_targets": {
                "side-camera": {"source": 2},
                "broken-camera": {"source": 3},
            },
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null", "fps": 60},
            "api": {"enabled": False},
        }
    )
    created = []

    class Backdrop:
        def __init__(self, identifier):
            self.identifier = identifier
            self.closed = False

        def frame(self, width, height):
            return np.zeros((height, width, 3), np.uint8)

        def reset_stats(self):
            pass

        def close(self):
            self.closed = True

    def create(cfg, **kwargs):
        target = kwargs.get("camera_target")
        identifier = target.identifier if target is not None else cfg.mode
        if identifier == "broken-camera":
            raise RuntimeError("configured target failed")
        backdrop = Backdrop(identifier)
        created.append(backdrop)
        return backdrop

    monkeypatch.setattr(pipeline_mod, "create_backdrop", create)
    runtime = RuntimeConfig(cfg)
    pipeline, _hub = run_pipeline(runtime)
    try:
        initial = created[-1]
        selected_state = pipeline.apply_config_patch({"background": {"mode": "camera"}})
        selected = created[-1]
        assert selected_state.version == 1
        assert selected.identifier == "side-camera"
        assert selected is not initial

        # An unused presentation setting must not reopen startup authority.
        unrelated = pipeline.apply_config_patch({"background": {"color": [9, 8, 7]}})
        assert unrelated.version == 2
        assert created[-1] is selected

        with pytest.raises(ActivationError, match="configured target failed"):
            pipeline.apply_config_patch(
                {"background": {"camera_target": "broken-camera"}}
            )
        current = runtime.read()
        assert current.version == 2
        assert current.config.background.camera_target == "side-camera"
        assert not selected.closed
    finally:
        pipeline.stop()


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
            "camera": {"width": 128, "height": 72},
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
            raise AssertionError(
                "working segmenter must not be used by background trial"
            )

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
    monkeypatch.setattr(pipeline_mod, "create_backdrop", lambda _cfg, **_kwargs: new)
    # Hub post-install failures are non-critical and must not roll back or
    # strand newly installed resource pointers.
    monkeypatch.setattr(
        hub,
        "invalidate_remote_session",
        lambda: (_ for _ in ()).throw(RuntimeError("invalidate")),
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
        args=(
            resources,
            request,
            _captured(np.zeros((72, 128, 3), np.uint8)),
        ),
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
    monkeypatch.setattr(pipeline_mod, "create_backdrop", lambda _cfg, **_kwargs: bad)
    candidate = cfg.patched({"background": {"color": [2, 2, 2]}})
    prepared = pipeline._prepare_activation_off_lane(cfg, candidate)
    activation = pipeline._stage_activation(resources, candidate, prepared)
    with pytest.raises(ActivationError):
        pipeline._trial_activation(
            resources,
            activation,
            _captured(np.zeros((72, 128, 3), np.uint8)),
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

    def blocked_open(cfg, canvas_size):
        entered.set()
        release.wait(1.0)
        return real_open_capture(cfg, canvas_size)

    monkeypatch.setattr(pipeline_mod, "open_capture", blocked_open)
    with pytest.raises(pipeline_mod.ReconfigurationUnavailable, match="still running"):
        pipeline.start(timeout=0.02)
    assert entered.is_set()
    release.set()
    deadline = time.monotonic() + 2.0
    while (
        pipeline._thread is not None and pipeline._thread.is_alive()
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pipeline._thread is not None and not pipeline._thread.is_alive()
    assert not pipeline.running


def test_blocked_output_shutdown_rejects_restart_until_owner_exits(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class StableCapture:
        def __init__(self) -> None:
            self.sequence = 0

        def read(self) -> CapturedFrame:
            self.sequence += 1
            return _captured(
                np.zeros((72, 128, 3), np.uint8),
                self.sequence,
                captured_at_ns=time.monotonic_ns(),
            )

        def close(self) -> None:
            pass

    class BlockingOutput(VideoOutput):
        paces = False

        def __init__(self, *, block_third_send: bool) -> None:
            self.block_third_send = block_third_send
            self.send_count = 0
            self.close_count = 0
            self.close_thread: int | None = None

        def send(self, _frame_bgr: np.ndarray) -> None:
            self.send_count += 1
            if self.block_third_send and self.send_count == 3:
                entered.set()
                assert release.wait(2.0)

        def close(self) -> None:
            self.close_count += 1
            self.close_thread = threading.get_ident()

    outputs: list[BlockingOutput] = []

    def open_blocking_output(*_args, **_kwargs):
        output = BlockingOutput(block_third_send=not outputs)
        outputs.append(output)
        return output

    monkeypatch.setattr(
        pipeline_mod,
        "open_capture",
        lambda *_args, **_kwargs: StableCapture(),
    )
    monkeypatch.setattr(pipeline_mod, "open_output", open_blocking_output)
    pipeline = Pipeline(make_runtime(mode="color"), FrameHub())
    pipeline.start()
    first = outputs[0]
    try:
        assert entered.wait(1.0)
        with pytest.raises(
            ReconfigurationUnavailable,
            match="pipeline worker did not stop",
        ):
            pipeline.stop(timeout=0.05)
        assert first.close_count == 0

        with pytest.raises(ReconfigurationUnavailable, match="already running"):
            pipeline.start(timeout=0.05)
        assert len(outputs) == 1

        release.set()
        deadline = time.monotonic() + 2.0
        while (
            pipeline._thread is not None and pipeline._thread.is_alive()
        ) and time.monotonic() < deadline:
            time.sleep(0.005)
        assert pipeline._thread is not None and not pipeline._thread.is_alive()
        assert first.close_count == 1
        assert first.close_thread is not None

        # Once the surviving sink owner has released all run resources, a
        # fresh run is allowed and owns a different backend instance.
        pipeline.start(timeout=1.0)
        assert len(outputs) == 2
        pipeline.stop(timeout=1.0)
        assert first.close_count == 1
        assert outputs[1].close_count == 1
    finally:
        release.set()
        thread = pipeline._thread
        if thread is not None and thread.is_alive():
            try:
                pipeline.stop(timeout=2.0)
            except BaseException:
                pass


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
        pipeline_mod, "create_segmenter", lambda _cfg, **_kwargs: BrokenSegmenter()
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
        assert stats["output_send_count"] >= 2
        assert stats["frames_out"] == stats["output_send_count"]
        assert stats["frames_in"] == stats["base_composite_update_count"]
        assert (
            stats["segmentation_update_count"] == stats["base_composite_update_count"]
        )
        assert (
            stats["base_composite_update_count"] + stats["base_composite_reuse_count"]
            == stats["output_send_count"]
        )
        assert stats["output_repeated_frames"] == stats["base_composite_reuse_count"]
        assert stats["last_unique_frame_age_ms"] is not None
        assert stats["timing_schema_version"] == 1
        assert set(stats["timing_ms"]) == set(pipeline_mod.TIMING_FIELD_NAMES)
        assert stats["mode"] == "color"
        assert stats["segmentation_backend"] == "HeuristicSegmenter"
        assert stats["output_backend"] == "NullOutput"
    finally:
        pipeline.stop()


def test_terminal_capture_health_is_sampled_after_capture_stops(monkeypatch):
    cfg = _color_integration_config("color")
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    events: list[str] = []

    class FinalHealthCapture:
        closed = False
        close_calls = 0

        def health_snapshot(self):
            events.append("health-closed" if self.closed else "health-open")
            return CaptureHealth(
                backend="terminal-fake",
                frames_read=1,
                dropped_frames=7 if self.closed else 3,
            )

        def close(self):
            self.close_calls += 1
            self.closed = True
            events.append("capture-close")

    capture = FinalHealthCapture()
    output = NullOutput(32, 24, cfg.output.fps)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        capture,
        _FixedMaskSegmenter(np.ones((24, 32), np.float32)),
        _IdentityRefiner(),
        None,
        None,
        output_factory=lambda: output,
    )
    now_ns = time.monotonic_ns()
    captured = _captured(
        np.full((24, 32, 3), 60, np.uint8),
        captured_at_ns=now_ns - 3_000_000,
    )
    preflight = pipeline_mod._PreflightResult(
        output=captured.pixels.copy(),
        captured=captured,
        send_timing=OutputSendTiming(
            submitted_at_ns=now_ns - 1_000_000,
            completed_at_ns=now_ns,
            submission_ms=1.0,
            pacing_wait_ms=1.0,
        ),
        base_ready_at_ns=now_ns - 2_000_000,
        segmentation_updated=False,
        frame_processing_ms=1.0,
        new_frame_service_ms=2.0,
        new_frame_serialized_loop_ms=3.0,
        timings={
            "segmentation_ms": 0.0,
            "background_ms": 0.0,
            "color_correction_ms": 0.0,
            "composite_ms": 0.0,
        },
        color_status={},
        remote_fallback_active=False,
        remote_fallback_reason="",
    )
    monkeypatch.setattr(
        pipeline,
        "_open_resources",
        lambda _state, *, defer_output=False: resources,
    )
    monkeypatch.setattr(
        pipeline,
        "_preflight",
        lambda _resources, *, send_output=True: preflight,
    )
    monkeypatch.setattr(pipeline, "_loop", lambda *_args, **_kwargs: None)

    pipeline._run()

    assert capture.close_calls == 1
    assert events[-2:] == ["capture-close", "health-closed"]
    assert hub.stats_dict()["capture_dropped_frames"] == 7


def test_slow_capture_repeats_last_safe_output_without_backlog(monkeypatch):
    frame = np.full((72, 128, 3), 90, np.uint8)

    class SlowLatestCapture:
        calls = 0
        frames_read = 0

        def read(self):
            self.calls += 1
            if self.calls == 1 or self.calls % 3 == 1:
                self.frames_read += 1
                return _captured(frame.copy(), self.frames_read)
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
    monkeypatch.setattr(
        pipeline_mod,
        "open_capture",
        lambda _cfg, _canvas: capture,
    )
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
        assert stats["output_send_count"] == stats["frames_out"]
        assert stats["base_composite_update_count"] == stats["frames_in"]
        assert stats["base_composite_reuse_count"] == stats["output_repeated_frames"]
        assert stats["base_composite_reuse_ratio"] == pytest.approx(
            stats["base_composite_reuse_count"] / stats["output_send_count"],
            abs=1e-4,
        )
        assert stats["cadence_mismatch_active"] is True
        assert stats["capture_frames_read"] == stats["frames_in"]
        assert stats["fps_attainment_pct"] is not None
    finally:
        pipeline.stop()


def test_nonpacing_output_samples_capture_after_application_wait():
    cfg = _color_integration_config("color")
    raw_first = np.full((24, 32, 3), 40, np.uint8)
    raw_second = np.full((24, 32, 3), 90, np.uint8)
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)

    class BoundaryCapture:
        def __init__(self):
            self.first_sent = False
            self.second_sent = False
            self.second_ready_at = 0.0

        def read(self):
            if not self.first_sent:
                self.first_sent = True
                self.second_ready_at = time.monotonic() + 0.005
                return _captured(raw_first.copy(), 1)
            if not self.second_sent and time.monotonic() >= self.second_ready_at:
                self.second_sent = True
                return _captured(raw_second.copy(), 2)
            return None

        def health_snapshot(self):
            return CaptureHealth(
                backend="boundary-fake",
                frames_read=int(self.first_sent) + int(self.second_sent),
            )

        def close(self):
            pass

    class NonPacingStopAfterOutput(_StopAfterOutput):
        paces = False

    output = NonPacingStopAfterOutput(pipeline, 2)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        BoundaryCapture(),
        _FixedMaskSegmenter(np.ones((24, 32), np.float32)),
        _IdentityRefiner(),
        None,
        output,
    )
    try:
        pipeline._loop(resources)
        stats = hub.stats_dict()
        assert len(output.frames) == 2
        assert stats["base_composite_update_count"] == 2
        assert stats["base_composite_reuse_count"] == 0
        assert stats["application_pacing_events"] == 1
        assert np.array_equal(output.frames[1], raw_second)
    finally:
        resources.close()


def test_nonpacing_preflight_waits_one_slot_before_first_steady_capture(
    monkeypatch,
):
    cfg = _color_integration_config("color")
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    clock_ns = [0]
    waits: list[float] = []
    interval_ns = round(1_000_000_000 / cfg.output.fps)

    class VirtualStop:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, timeout):
            waits.append(timeout)
            clock_ns[0] += round(timeout * 1_000_000_000)
            return self.stopped

    class NextCapture:
        def read(self):
            assert clock_ns[0] >= interval_ns
            return _captured(
                np.full((24, 32, 3), 90, np.uint8),
                2,
                captured_at_ns=clock_ns[0],
            )

        def health_snapshot(self):
            return CaptureHealth(backend="startup-slot-fake", frames_read=2)

        def close(self):
            pass

    class OneShotOutput:
        paces = False
        fallback_active = False
        fallback_reason = ""

        def send(self, _frame):
            pipeline._stop.set()

        def close(self):
            pass

    virtual_stop = VirtualStop()
    pipeline._stop = cast(Any, virtual_stop)
    monkeypatch.setattr(pipeline_mod.time, "monotonic_ns", lambda: clock_ns[0])
    first = _captured(np.full((24, 32, 3), 40, np.uint8), 1, captured_at_ns=0)
    preflight = pipeline_mod._PreflightResult(
        output=first.pixels.copy(),
        captured=first,
        send_timing=OutputSendTiming(
            submitted_at_ns=0,
            completed_at_ns=0,
            submission_ms=0.0,
            pacing_wait_ms=0.0,
        ),
        base_ready_at_ns=0,
        segmentation_updated=False,
        frame_processing_ms=0.0,
        new_frame_service_ms=0.0,
        new_frame_serialized_loop_ms=0.0,
        timings={
            "segmentation_ms": 0.0,
            "background_ms": 0.0,
            "color_correction_ms": 0.0,
            "composite_ms": 0.0,
        },
        color_status={},
        remote_fallback_active=False,
        remote_fallback_reason="",
    )
    tracker = pipeline_mod.CadenceTracker(cfg.output.fps)
    tracker.record_send(
        sent_at_ns=0,
        capture_sequence=1,
        captured_at_ns=0,
        base_ready_at_ns=0,
        base_updated=True,
        segmentation_updated=False,
        exact_final_repeat=False,
    )
    resources = pipeline_mod._Resources(
        cfg,
        0,
        NextCapture(),
        _FixedMaskSegmenter(np.ones((24, 32), np.float32)),
        _IdentityRefiner(),
        None,
        OneShotOutput(),
    )
    try:
        pipeline._loop(
            resources,
            preflight=preflight,
            cadence_tracker=tracker,
        )
        assert waits == [pytest.approx(interval_ns / 1_000_000_000.0)]
        stats = hub.stats_dict()
        assert stats["output_send_count"] == 2
        assert stats["base_composite_update_count"] == 2
        assert stats["base_composite_reuse_count"] == 0
        assert stats["application_pacing_events"] == 1
        assert stats["output_send_delta_p50_ms"] == pytest.approx(
            interval_ns / 1_000_000.0,
            abs=0.001,
        )
    finally:
        resources.close()


def test_nonpacing_output_does_not_add_wait_after_over_budget_cycle(monkeypatch):
    cfg = _color_integration_config("color")
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    clock_ns = [0]
    waits: list[float] = []

    class VirtualStop:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, timeout):
            waits.append(timeout)
            clock_ns[0] += round(timeout * 1_000_000_000)
            return self.stopped

    class SlowCapture:
        sequence = 0

        def read(self):
            self.sequence += 1
            clock_ns[0] += 40_000_000
            return _captured(
                np.full((24, 32, 3), self.sequence, np.uint8),
                self.sequence,
                captured_at_ns=clock_ns[0] - 1_000_000,
            )

        def health_snapshot(self):
            return CaptureHealth(
                backend="slow-service-fake",
                frames_read=self.sequence,
            )

        def close(self):
            pass

    class NonPacingOutput:
        paces = False
        fallback_active = False
        fallback_reason = ""

        def __init__(self):
            self.frames = 0

        def send(self, _frame):
            self.frames += 1
            if self.frames == 3:
                pipeline._stop.set()

        def close(self):
            pass

    virtual_stop = VirtualStop()
    pipeline._stop = cast(Any, virtual_stop)
    monkeypatch.setattr(pipeline_mod.time, "monotonic_ns", lambda: clock_ns[0])
    output = NonPacingOutput()
    resources = pipeline_mod._Resources(
        cfg,
        0,
        SlowCapture(),
        _FixedMaskSegmenter(np.ones((24, 32), np.float32)),
        _IdentityRefiner(),
        None,
        output,
    )
    try:
        pipeline._loop(resources)
        assert output.frames == 3
        assert waits == []
        assert hub.stats_dict()["application_pacing_events"] == 0
    finally:
        resources.close()


def test_preflight_handoff_ignores_an_already_consumed_capture_sequence(
    monkeypatch,
):
    startup_pixels = np.full((72, 128, 3), 40, np.uint8)
    live_pixels = np.full((72, 128, 3), 90, np.uint8)
    startup = _captured(startup_pixels, 7, captured_at_ns=7_000_000_000)
    live = _captured(live_pixels, 8, captured_at_ns=8_000_000_000)

    class DuplicateStartupCapture:
        def __init__(self):
            # Model a faulty handoff that exposes the already-sent startup
            # capture once more before the first genuinely new input.
            self.frames = [startup, startup, live]

        def read(self):
            return self.frames.pop(0) if self.frames else None

        def health_snapshot(self):
            return CaptureHealth(
                sequence=8,
                captured_monotonic_ns=8_000_000_000,
                generation=1,
                geometry_generation=1,
                content_rect=(0, 0, 128, 72),
                backend="duplicate-startup-fake",
                normalized_width=128,
                normalized_height=72,
                frames_read=2,
            )

        def close(self):
            pass

    monkeypatch.setattr(
        pipeline_mod,
        "open_capture",
        lambda _cfg, _canvas: DuplicateStartupCapture(),
    )
    pipeline, hub = run_pipeline(make_runtime(mode="color"))
    try:
        stats = wait_for_stats(hub, lambda value: value["frames_in"] >= 2)
        # Preflight is input one and sequence 8 is input two. The duplicate
        # sequence 7 becomes only an output repeat and never a temporal input.
        assert stats["frames_in"] == 2
        raw, _sequence = hub.raw.latest()
        assert raw is not None
        np.testing.assert_array_equal(raw, live_pixels)
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
        lambda _cfg, **_kwargs: SlowSegmenter(),
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


def test_fifteen_fps_capture_at_thirty_fps_transport_is_intentional_repeat(
    monkeypatch,
):
    monkeypatch.setattr(
        runtime_performance_mod,
        "RUNTIME_PERFORMANCE_WARMUP_NS",
        200_000_000,
    )
    monkeypatch.setattr(
        runtime_performance_mod,
        "RUNTIME_PERFORMANCE_DEGRADE_NS",
        200_000_000,
    )
    monkeypatch.setattr(
        runtime_performance_mod,
        "RUNTIME_PERFORMANCE_WINDOW_NS",
        1_000_000_000,
    )
    original_local_composite = Pipeline._local_composite

    def forty_ms_local_composite(self, *args, **kwargs):
        result = original_local_composite(self, *args, **kwargs)
        time.sleep(0.04)
        return result

    monkeypatch.setattr(Pipeline, "_local_composite", forty_ms_local_composite)
    cfg = (
        make_runtime(mode="color")
        .snapshot()
        .patched({"camera": {"fps": 15}, "output": {"fps": 30}})
    )
    pipeline, hub = run_pipeline(RuntimeConfig(cfg))
    try:
        stats = wait_for_stats(
            hub,
            lambda value: (
                value["frames_out"] >= 30
                and value["runtime_performance"]["state"] == "healthy"
            ),
            timeout=4.0,
        )
        performance = stats["runtime_performance"]
        assert performance["schema_version"] == 2
        assert performance["target_fps"] == 30.0
        assert performance["transport_target_fps"] == 30.0
        assert performance["unique_target_fps"] == 15.0
        assert performance["transport_deadline_ms"] == pytest.approx(33.333, abs=0.001)
        assert performance["processing_deadline_ms"] == pytest.approx(66.667, abs=0.001)
        assert performance["cadence_status"] == "intentional-repeat"
        assert performance["output_healthy"] is True
        assert performance["unique_healthy"] is True
        assert performance["processing_deadline_miss_ratio"] == 0.0
        assert stats["processing_deadline_misses"] == 0
        assert stats["base_composite_reuse_count"] > 0
        assert (
            stats["exact_final_output_repeat_count"]
            >= stats["base_composite_reuse_count"]
        )
    finally:
        pipeline.stop()


def test_target_paced_publisher_stays_at_thirty_during_120ms_processing(
    monkeypatch,
    caplog,
):
    # The tracker thresholds themselves have exact fake-clock unit coverage.
    # Shorten only their wall-clock hysteresis here so this integration test
    # proves the production lane wiring without adding six seconds to the suite.
    monkeypatch.setattr(
        runtime_performance_mod,
        "RUNTIME_PERFORMANCE_WARMUP_NS",
        200_000_000,
    )
    monkeypatch.setattr(
        runtime_performance_mod,
        "RUNTIME_PERFORMANCE_DEGRADE_NS",
        200_000_000,
    )
    monkeypatch.setattr(
        runtime_performance_mod,
        "RUNTIME_PERFORMANCE_RECOVER_NS",
        200_000_000,
    )
    monkeypatch.setattr(
        runtime_performance_mod,
        "RUNTIME_PERFORMANCE_WINDOW_NS",
        1_000_000_000,
    )
    original_local_composite = Pipeline._local_composite
    original_segment_and_refine = Pipeline._segment_and_refine_masks
    original_refine = pipeline_mod.MaskRefiner.refine
    original_backdrop_frame = pipeline_mod.ColorBackdrop.frame
    original_harmonizer_update = pipeline_mod.ColorHarmonizer.update
    original_timeline_observe = pipeline_mod.SegmentationTimeline.observe
    stage_calls = {
        "segmentation": 0,
        "refinement": 0,
        "backdrop": 0,
        "harmonizer": 0,
        "matte_timeline": 0,
    }

    def count_segment_and_refine(cls, *args, **kwargs):
        stage_calls["segmentation"] += 1
        return original_segment_and_refine(*args, **kwargs)

    def count_refine(self, *args, **kwargs):
        stage_calls["refinement"] += 1
        return original_refine(self, *args, **kwargs)

    def count_backdrop_frame(self, *args, **kwargs):
        stage_calls["backdrop"] += 1
        return original_backdrop_frame(self, *args, **kwargs)

    def count_harmonizer_update(self, *args, **kwargs):
        stage_calls["harmonizer"] += 1
        return original_harmonizer_update(self, *args, **kwargs)

    def count_timeline_observe(self, *args, **kwargs):
        stage_calls["matte_timeline"] += 1
        return original_timeline_observe(self, *args, **kwargs)

    monkeypatch.setattr(
        Pipeline,
        "_segment_and_refine_masks",
        classmethod(count_segment_and_refine),
    )
    monkeypatch.setattr(pipeline_mod.MaskRefiner, "refine", count_refine)
    monkeypatch.setattr(pipeline_mod.ColorBackdrop, "frame", count_backdrop_frame)
    monkeypatch.setattr(
        pipeline_mod.ColorHarmonizer,
        "update",
        count_harmonizer_update,
    )
    monkeypatch.setattr(
        pipeline_mod.SegmentationTimeline,
        "observe",
        count_timeline_observe,
    )

    class RecordingOutput(VideoOutput):
        paces = False

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.frames: list[np.ndarray] = []

        def send(self, frame_bgr: np.ndarray) -> None:
            with self._lock:
                self.frames.append(frame_bgr.copy())

        def snapshot(self) -> list[np.ndarray]:
            with self._lock:
                return [frame.copy() for frame in self.frames]

    output = RecordingOutput()
    monkeypatch.setattr(
        pipeline_mod,
        "open_output",
        lambda *_args, **_kwargs: output,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "create_backdrop",
        lambda *_args, **_kwargs: pipeline_mod.ColorBackdrop((17, 31, 47)),
    )
    slow_processing = threading.Event()
    slow_processing.set()

    def slow_local_composite(self, *args, **kwargs):
        result = original_local_composite(self, *args, **kwargs)
        if slow_processing.is_set():
            time.sleep(0.12)
        return result

    monkeypatch.setattr(Pipeline, "_local_composite", slow_local_composite)
    cfg = (
        make_runtime(mode="camera", camera_device=1)
        .snapshot()
        .patched(
            {
                "camera": {"fps": 30},
                "output": {"fps": 30},
                "compositing": {"color_correction": {"mode": "auto", "strength": 0.5}},
            }
        )
    )
    pipeline, hub = run_pipeline(RuntimeConfig(cfg))
    initial_stats = hub.stats_dict()
    initial_stage_calls = dict(stage_calls)
    initial_frames = output.snapshot()
    try:
        stats = wait_for_stats(
            hub,
            lambda value: (
                value["frames_out"] >= 30
                and value["runtime_performance"]["state"] == "degraded"
            ),
            timeout=4.0,
        )
        performance = stats["runtime_performance"]
        assert performance["stage_p50_ms"]["capture.read"] is not None
        assert performance["stage_p95_ms"]["capture.read"] is not None
        assert performance["output_send_fps"] >= 27.0
        assert 6.0 <= performance["sent_unique_base_fps"] <= 10.0
        assert performance["output_healthy"] is True
        assert performance["unique_healthy"] is False
        assert performance["reason"] in {
            "unique-attainment",
            "multiple-performance-gates",
        }
        assert stats["output_repeated_frames"] > 0
        assert stats["frames_out"] == (
            stats["frames_in"] + stats["output_repeated_frames"]
        )
        assert (
            stats["exact_final_output_repeat_count"]
            >= stats["base_composite_reuse_count"]
        )
        assert performance["publisher"]["pending_depth"] in {0, 1}
        # Repeated presentation never re-enters the processing graph.
        assert stats["segmentation_update_count"] <= stats["processing_completed_count"]
        assert stats["processing_completed_count"] < stats["frames_out"]

        final_frames = output.snapshot()
        observed_frames = final_frames[max(0, len(initial_frames) - 1) :]
        exact_adjacent_pairs = sum(
            np.array_equal(previous, current)
            for previous, current in zip(observed_frames, observed_frames[1:])
        )
        repeat_delta = (
            stats["output_repeated_frames"] - initial_stats["output_repeated_frames"]
        )
        exact_repeat_delta = (
            stats["exact_final_output_repeat_count"]
            - initial_stats["exact_final_output_repeat_count"]
        )
        # The one-frame baseline snapshot can race one publisher tick. Apart
        # from that boundary, the frames physically accepted by the sink and
        # the publisher's byte-digest classification agree exactly.
        assert abs(exact_adjacent_pairs - exact_repeat_delta) <= 1
        assert exact_adjacent_pairs >= repeat_delta - 1

        processed_delta = (
            stats["processing_completed_count"]
            - initial_stats["processing_completed_count"]
        )
        send_delta = stats["frames_out"] - initial_stats["frames_out"]
        for name, count in stage_calls.items():
            call_delta = count - initial_stage_calls[name]
            assert 0 < call_delta <= processed_delta + 1, (name, stage_calls)
            assert call_delta < send_delta, (name, stage_calls)

        slow_processing.clear()
        wait_for_stats(
            hub,
            lambda value: value["runtime_performance"]["state"] == "healthy",
            timeout=4.0,
        )
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelname == "WARNING"
            and record.getMessage().startswith("runtime performance")
        ]
        assert sum("degraded" in message for message in warnings) == 1
        assert sum("recovered" in message for message in warnings) == 1
    finally:
        slow_processing.clear()
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
        lambda cfg, **_kwargs: VideoStatsBackdrop() if cfg.mode == "video" else None,
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
        lambda cfg, **_kwargs: HeuristicSegmenter(cfg),
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


# VIS-2.5 deterministic color-integration fixtures and regressions.


def _color_integration_config(
    mode: str = "image",
    *,
    correction_mode: str = "auto",
    blend_space: str = "linear_srgb",
    use_model_foreground: bool = False,
) -> AppConfig:
    background: dict[str, Any] = {"mode": mode}
    if mode == "image":
        background["image_path"] = "/deterministic/background.png"
    elif mode == "video":
        background["video_path"] = "/deterministic/background.mp4"
    elif mode == "camera":
        background["camera_device"] = 1
    return AppConfig.from_dict(
        {
            "camera": {"width": 32, "height": 24, "fps": 60},
            "background": background,
            "segmentation": {
                "backend": "heuristic",
                "temporal_smoothing": 0.0,
            },
            "compositing": {
                "blend_space": blend_space,
                "light_wrap": 0.0,
                "use_model_foreground": use_model_foreground,
                "color_correction": {"mode": correction_mode},
            },
            "output": {"backend": "null", "fps": 60},
            "api": {"enabled": False},
        }
    )


def _seeded_color_harmonizer(exposure_ev: float = 0.7) -> ColorHarmonizer:
    harmonizer = ColorHarmonizer(0.8, mode="image")
    estimate = ColorEstimate(
        transform=ColorTransform(exposure_ev, (1.10, 1.0, 0.90)),
        behavior=ColorBehavior.EXPOSURE_WHITE_BALANCE,
        reason=ColorReason.OK,
        confidence=0.9,
        exposure_confidence=0.95,
        white_balance_confidence=0.9,
        usable_source=512,
        usable_target=512,
        neutral_source=256,
        neutral_target=256,
        target_is_local=True,
        reliable=True,
        signature=ColorSceneSignature(
            source_log_luminance=-2.0,
            target_log_luminance=-1.5,
            source_chroma_log2=(0.0, 0.0, 0.0),
            target_chroma_log2=(0.0, 0.0, 0.0),
        ),
    )
    for index in range(121):
        harmonizer.update(estimate, index / 30.0, source_generation=1)
    return harmonizer


class _FixedMaskSegmenter:
    device = "cpu"
    produces_matte = False

    def __init__(
        self,
        mask: np.ndarray,
        edge_foreground: np.ndarray | None = None,
    ):
        self.mask = mask
        self.last_foreground = edge_foreground
        self.matte_backend_kind = (
            MatteBackendKind.TRUE_ALPHA_RECURRENT
            if edge_foreground is not None
            else MatteBackendKind.BINARY_COARSE
        )
        self.produces_matte = edge_foreground is not None
        self.closed = False

    def segment(self, _frame):
        return self.mask.copy()

    def close(self):
        self.closed = True


class _TelemetryMaskSegmenter(_FixedMaskSegmenter):
    def telemetry_snapshot(self):
        return {
            "input_frame_shape": (24, 32),
            "model_mask_shape": (12, 16),
            "output_mask_shape": (24, 32),
            "effective_timestamp_ms": 127,
            "effective_timestamp_delta_ms": 34,
            "timestamp_adjustment_count": 1,
            "timestamp_adjustment_ms": 0,
            "last_timestamp_adjusted": False,
            "resize_interpolation": "linear",
        }


class _IdentityRefiner:
    def refine(self, mask, _frame):
        return np.ascontiguousarray(mask, dtype=np.float32)


class _FixedBackdrop:
    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels

    def frame(self, width, height):
        assert self.pixels.shape == (height, width, 3)
        return self.pixels.copy()

    def close(self):
        pass


class _GeometryTokenBackdrop(_FixedBackdrop):
    def __init__(self, pixels: np.ndarray, token: object):
        super().__init__(pixels)
        self.token = token

    def content_rect(self, width, height):
        return (0, 0, width, height)

    def transform_plan(self, _width, _height):
        return self.token


class _SpyHarmonizer(ColorHarmonizer):
    def __init__(self, transform: ColorTransform = IDENTITY_TRANSFORM):
        super().__init__(0.8, mode="image")
        self.returned_transform = transform
        self.reset_calls = []
        self.update_calls = []
        self.error_calls = []

    def reset(self, now_s, *, reason=ColorReason.INVALID, source_generation=None):
        self.reset_calls.append((now_s, reason, source_generation))
        return IDENTITY_TRANSFORM

    def update(self, estimate, now_s, *, source_generation=None):
        self.update_calls.append((estimate, now_s, source_generation))
        return self.returned_transform

    def reset_and_update(self, estimate, now_s, *, source_generation=None):
        self.reset(
            now_s,
            reason=ColorReason.INVALID,
            source_generation=source_generation,
        )
        return self.update(
            estimate,
            now_s,
            source_generation=source_generation,
        )

    def on_error(self, now_s, *, source_generation=None):
        self.error_calls.append((now_s, source_generation))
        return IDENTITY_TRANSFORM


class _SequenceCapture:
    def __init__(self, frames: list[np.ndarray | None]):
        self.frames = list(frames)
        self.frames_read = 0

    def read(self):
        value = self.frames.pop(0) if self.frames else None
        if value is not None:
            self.frames_read += 1
            return _captured(value.copy(), self.frames_read)
        return None

    def health_snapshot(self):
        return CaptureHealth(
            generation=0,
            geometry_generation=0,
            backend="deterministic",
            width=32,
            height=24,
            normalized_width=32,
            normalized_height=24,
            frames_read=self.frames_read,
        )

    def close(self):
        pass


class _StopAfterOutput:
    paces = True
    fallback_active = False
    fallback_reason = ""

    def __init__(self, pipeline: Pipeline, count: int):
        self.pipeline = pipeline
        self.count = count
        self.frames = []

    def send(self, frame):
        self.frames.append(frame.copy())
        if len(self.frames) >= self.count:
            self.pipeline._stop.set()

    def close(self):
        pass


@pytest.mark.parametrize(
    ("background_mode", "correction_mode", "eligible"),
    [
        ("image", "auto", True),
        ("video", "auto", True),
        ("camera", "auto", True),
        ("image", "off", False),
        ("blur", "auto", False),
        ("color", "auto", False),
        ("passthrough", "auto", False),
        ("remote", "auto", False),
    ],
)
def test_color_correction_eligibility_is_explicit_and_bypasses_to_identity(
    monkeypatch,
    background_mode,
    correction_mode,
    eligible,
):
    cfg = _color_integration_config(
        background_mode,
        correction_mode=correction_mode,
    )
    transform = ColorTransform(exposure_ev=0.25)
    harmonizer = _SpyHarmonizer(transform)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        None,
        harmonizer=harmonizer,
    )
    estimator_calls = []
    estimate_token = object()

    def estimate(*args, **kwargs):
        estimator_calls.append((args, kwargs))
        return estimate_token

    monkeypatch.setattr(pipeline_mod, "estimate_color_transform_linear", estimate)
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    prepared = pipeline._prepare_color_frame(
        resources,
        np.full((24, 32, 3), 80, np.uint8),
        np.full((24, 32, 3), 20, np.uint8),
        np.full((24, 32), 0.5, np.float32),
        None,
        now_s=10.0,
        captured=_captured(
            np.full((24, 32, 3), 80, np.uint8),
            captured_at_ns=10_000_000_000,
        ),
    )

    if eligible:
        assert prepared.transform == transform
        assert prepared.foreground_linear_bgr is not None
        assert prepared.backdrop_linear_bgr is not None
        assert len(estimator_calls) == len(harmonizer.update_calls) == 1
        assert estimator_calls[0][1]["mode"] == background_mode
        assert len(harmonizer.reset_calls) == 1
    else:
        assert prepared == pipeline_mod._PreparedColorFrame()
        assert not estimator_calls
        assert not harmonizer.reset_calls
        assert not harmonizer.update_calls
        assert not harmonizer.error_calls


def test_legacy_prepared_color_seam_preserves_bgr_channel_order_with_wb(
    monkeypatch,
):
    cfg = _color_integration_config("image", blend_space="srgb_legacy")
    y, x = np.indices((24, 32), dtype=np.uint8)
    frame = np.stack(
        (
            20 + x,
            70 + y,
            130 + ((x.astype(np.uint16) + y.astype(np.uint16)) % 40).astype(np.uint8),
        ),
        axis=2,
    ).astype(np.uint8)
    backdrop = np.full_like(frame, (11, 37, 83))
    mask = np.ones((24, 32), np.float32)
    transform = ColorTransform(0.25, (1.12, 0.97, 0.88))
    estimate = ColorEstimate(
        transform=transform,
        behavior=ColorBehavior.EXPOSURE_WHITE_BALANCE,
        reason=ColorReason.OK,
        confidence=0.9,
        exposure_confidence=0.9,
        white_balance_confidence=0.9,
        usable_source=512,
        usable_target=512,
        neutral_source=256,
        neutral_target=256,
        target_is_local=True,
        reliable=True,
        signature=ColorSceneSignature(
            source_log_luminance=-2.0,
            target_log_luminance=-1.5,
            source_chroma_log2=(0.0, 0.0, 0.0),
            target_chroma_log2=(0.0, 0.0, 0.0),
        ),
    )
    estimator_inputs = []

    def estimate_predecoded(foreground_linear_bgr, backdrop_linear_bgr, *_args, **_kw):
        estimator_inputs.append((foreground_linear_bgr, backdrop_linear_bgr))
        return estimate

    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        estimate_predecoded,
    )
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(backdrop),
        None,
        harmonizer=_SpyHarmonizer(transform),
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    try:
        prepared = pipeline._prepare_color_frame(
            resources,
            frame,
            backdrop,
            mask,
            None,
            now_s=10.0,
            captured=_captured(frame, captured_at_ns=10_000_000_000),
        )
        rendered = pipeline._composite_prepared_color(
            cfg,
            frame,
            backdrop,
            mask,
            None,
            prepared,
            light_wrap=0.0,
        )
        expected = compositor_mod.composite(
            frame,
            backdrop,
            mask,
            light_wrap=0.0,
            blend_space="srgb_legacy",
            color_transform=transform,
        )

        assert len(estimator_inputs) == 1
        observed_foreground_bgr, observed_backdrop_bgr = estimator_inputs[0]
        np.testing.assert_array_equal(
            observed_foreground_bgr[..., ::-1],
            bgr_u8_to_linear_rgb(frame),
        )
        np.testing.assert_array_equal(
            observed_backdrop_bgr[..., ::-1],
            bgr_u8_to_linear_rgb(backdrop),
        )
        assert not np.array_equal(
            observed_foreground_bgr,
            bgr_u8_to_linear_rgb(frame),
        )
        np.testing.assert_array_equal(rendered, expected)
    finally:
        resources.close()


def test_image_backdrop_analysis_cache_is_bounded_and_generation_scoped(
    monkeypatch,
    tmp_path,
):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    y, x = np.indices((256, 384), dtype=np.uint16)
    first_source = np.stack(
        (
            (x % 256).astype(np.uint8),
            (y % 256).astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ),
        axis=2,
    )
    second_source = np.ascontiguousarray(255 - first_source)
    assert cv2.imwrite(str(first_path), first_source)
    assert cv2.imwrite(str(second_path), second_source)

    cfg = AppConfig.from_dict(
        {
            "camera": {"width": 384, "height": 216, "fps": 60},
            "background": {
                "mode": "image",
                "image_path": str(first_path),
                "fit_mode": "cover",
                "anchor_x": 0.5,
            },
            "segmentation": {"backend": "heuristic"},
            "compositing": {
                "blend_space": "linear_srgb",
                "color_correction": {"mode": "auto"},
            },
            "output": {"backend": "null", "fps": 60},
            "api": {"enabled": False},
        }
    )
    provider = pipeline_mod.ImageBackdrop(str(first_path))
    harmonizer = _SpyHarmonizer()
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(np.full((216, 384), 0.5, np.float32)),
        _IdentityRefiner(),
        provider,
        None,
        harmonizer=harmonizer,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    pipeline._active_state = pipeline_mod.ConfigState(cfg, 0)
    foreground = np.full((216, 384, 3), 80, np.uint8)
    mask = np.full((216, 384), 0.5, np.float32)
    estimate_token = object()
    estimator_analyses = []
    analysis_builds = []
    real_analysis_builder = pipeline_mod._linear_bgr_analysis_raster_prevalidated

    def estimate(*_args, **kwargs):
        estimator_analyses.append(kwargs["backdrop_analysis_linear_bgr"])
        return estimate_token

    def build_analysis(value):
        result = real_analysis_builder(value)
        analysis_builds.append(result)
        return result

    monkeypatch.setattr(pipeline_mod, "estimate_color_transform_linear", estimate)
    monkeypatch.setattr(
        pipeline_mod,
        "_linear_bgr_analysis_raster_prevalidated",
        build_analysis,
    )

    try:
        backdrop = provider.frame(384, 216)
        for timestamp in (1.0, 2.0):
            pipeline._prepare_color_frame(
                resources,
                foreground,
                backdrop,
                mask,
                None,
                now_s=timestamp,
                captured=_captured(
                    foreground,
                    int(timestamp),
                    captured_at_ns=int(timestamp * 1_000_000_000),
                ),
            )
        assert len(analysis_builds) == 1
        assert estimator_analyses[0] is estimator_analyses[1]
        assert estimator_analyses[0].dtype == np.float32
        assert max(estimator_analyses[0].shape[:2]) == ANALYSIS_LONG_EDGE

        geometry_cfg = cfg.patched({"background": {"anchor_x": 0.25}})
        geometry_activation = pipeline_mod._Activation(
            candidate=geometry_cfg,
            refiner=resources.refiner,
            backdrop=provider,
            visual_state_changed=True,
        )
        old_backdrop, _, _ = pipeline._install_activation(
            resources,
            geometry_activation,
            1,
        )
        assert old_backdrop is None
        assert resources.color_backdrop_analysis_linear_bgr is None
        backdrop = provider.frame(384, 216)
        pipeline._prepare_color_frame(
            resources,
            foreground,
            backdrop,
            mask,
            None,
            now_s=3.0,
            captured=_captured(
                foreground,
                3,
                captured_at_ns=3_000_000_000,
            ),
        )
        assert len(analysis_builds) == 2
        assert estimator_analyses[2] is not estimator_analyses[1]

        replacement_cfg = geometry_cfg.patched(
            {"background": {"image_path": str(second_path)}}
        )
        replacement = pipeline_mod.ImageBackdrop(
            str(second_path),
            fit_mode=replacement_cfg.background.fit_mode,
            anchor_x=replacement_cfg.background.anchor_x,
            anchor_y=replacement_cfg.background.anchor_y,
        )
        replacement_activation = pipeline_mod._Activation(
            candidate=replacement_cfg,
            refiner=resources.refiner,
            replace_backdrop=True,
            backdrop=replacement,
            visual_state_changed=True,
        )
        old_backdrop, _, _ = pipeline._install_activation(
            resources,
            replacement_activation,
            2,
        )
        assert old_backdrop is provider
        assert resources.color_backdrop_analysis_linear_bgr is None
        backdrop = replacement.frame(384, 216)
        pipeline._prepare_color_frame(
            resources,
            foreground,
            backdrop,
            mask,
            None,
            now_s=4.0,
            captured=_captured(
                foreground,
                4,
                captured_at_ns=4_000_000_000,
            ),
        )
        assert len(analysis_builds) == 3
        assert estimator_analyses[3] is not estimator_analyses[2]
        old_backdrop.close()
    finally:
        resources.close()

    assert resources.color_backdrop_analysis_token is None
    assert resources.color_backdrop_analysis_linear_bgr is None


def test_live_reset_consumes_first_reliable_estimate_with_production_harmonizer(
    monkeypatch,
):
    cfg = _color_integration_config("image")
    harmonizer = ColorHarmonizer(0.8, mode="image")
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        None,
        harmonizer=harmonizer,
    )
    signature = ColorSceneSignature(
        source_log_luminance=-2.0,
        target_log_luminance=-1.5,
        source_chroma_log2=(0.0, 0.0, 0.0),
        target_chroma_log2=(0.0, 0.0, 0.0),
    )
    estimate = ColorEstimate(
        transform=ColorTransform(exposure_ev=0.4),
        behavior=ColorBehavior.EXPOSURE_ONLY,
        reason=ColorReason.OK,
        confidence=0.9,
        exposure_confidence=0.9,
        white_balance_confidence=0.0,
        usable_source=512,
        usable_target=512,
        neutral_source=0,
        neutral_target=0,
        target_is_local=True,
        reliable=True,
        signature=signature,
        exposure_clamped=True,
        white_balance_clamped=False,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: estimate,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    foreground = np.full((24, 32, 3), 80, np.uint8)
    backdrop = np.full((24, 32, 3), 20, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)

    first = pipeline._prepare_color_frame(
        resources,
        foreground,
        backdrop,
        mask,
        None,
        now_s=10.0,
        captured=_captured(
            foreground,
            captured_at_ns=10_000_000_000,
        ),
    )
    snapshot = harmonizer.snapshot()
    assert first.transform.is_identity
    assert first.exposure_clamped is True
    assert first.white_balance_clamped is False
    assert snapshot.reliable
    assert snapshot.signature == signature
    assert snapshot.phase is HarmonizerPhase.WARMING
    assert snapshot.last_timestamp_s == 10.0
    assert resources.color_reset_token is not None

    second = pipeline._prepare_color_frame(
        resources,
        foreground,
        backdrop,
        mask,
        None,
        now_s=10.0 + 1.0 / 30.0,
        captured=_captured(
            foreground,
            2,
            captured_at_ns=10_000_000_000 + 1_000_000_000 // 30,
        ),
    )
    assert second.transform.exposure_ev > 0.0


def test_live_backdrop_geometry_token_change_hard_resets_harmonizer(monkeypatch):
    cfg = _color_integration_config("video")
    harmonizer = ColorHarmonizer(0.8, mode="video")
    backdrop_provider = _GeometryTokenBackdrop(
        np.full((24, 32, 3), 20, np.uint8),
        ("source-size", 640, 480),
    )
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        backdrop_provider,
        None,
        harmonizer=harmonizer,
    )
    signature = ColorSceneSignature(
        source_log_luminance=-2.0,
        target_log_luminance=-1.5,
        source_chroma_log2=(0.0, 0.0, 0.0),
        target_chroma_log2=(0.0, 0.0, 0.0),
    )
    estimate = ColorEstimate(
        transform=ColorTransform(exposure_ev=0.4),
        behavior=ColorBehavior.EXPOSURE_ONLY,
        reason=ColorReason.OK,
        confidence=0.9,
        exposure_confidence=0.9,
        white_balance_confidence=0.0,
        usable_source=512,
        usable_target=512,
        neutral_source=0,
        neutral_target=0,
        target_is_local=True,
        reliable=True,
        signature=signature,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: estimate,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    foreground = np.full((24, 32, 3), 80, np.uint8)
    backdrop = np.full((24, 32, 3), 20, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)

    pipeline._prepare_color_frame(
        resources,
        foreground,
        backdrop,
        mask,
        None,
        now_s=10.0,
        captured=_captured(
            foreground,
            captured_at_ns=10_000_000_000,
        ),
    )
    active = pipeline._prepare_color_frame(
        resources,
        foreground,
        backdrop,
        mask,
        None,
        now_s=10.0 + 1.0 / 30.0,
        captured=_captured(
            foreground,
            2,
            captured_at_ns=10_000_000_000 + 1_000_000_000 // 30,
        ),
    )
    assert active.transform.exposure_ev > 0.0

    backdrop_provider.token = ("source-size", 720, 1280)
    reset = pipeline._prepare_color_frame(
        resources,
        foreground,
        backdrop,
        mask,
        None,
        now_s=10.0 + 2.0 / 30.0,
        captured=_captured(
            foreground,
            3,
            captured_at_ns=10_000_000_000 + 2_000_000_000 // 30,
        ),
    )
    snapshot = harmonizer.snapshot()
    assert reset.transform.is_identity
    assert snapshot.reliable
    assert snapshot.signature == signature
    assert snapshot.phase is HarmonizerPhase.WARMING


def test_active_correction_preserves_raw_hub_bytes_and_is_mask_local(monkeypatch):
    cfg = _color_integration_config("image")
    raw = np.full((24, 32, 3), 64, np.uint8)
    background = np.full((24, 32, 3), 20, np.uint8)
    mask = np.zeros((24, 32), np.float32)
    mask[:, :16] = 1.0
    transform = ColorTransform(exposure_ev=1.0)
    harmonizer = _SpyHarmonizer(transform)
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    output = _StopAfterOutput(pipeline, 1)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([raw]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(background),
        output,
        harmonizer=harmonizer,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: object(),
    )

    pipeline._loop(resources)

    published_raw, _ = hub.raw.latest()
    published_output, _ = hub.output.latest()
    assert published_raw is not None and published_output is not None
    assert np.array_equal(published_raw, raw)
    assert np.array_equal(published_output[:, 16:], background[:, 16:])
    expected_foreground = linear_rgb_to_bgr_u8(
        apply_color_transform(bgr_u8_to_linear_rgb(raw), transform)
    )
    assert np.array_equal(published_output[:, :16], expected_foreground[:, :16])
    assert not np.array_equal(published_output[:, :16], raw[:, :16])
    assert len(harmonizer.update_calls) == 1


def test_auto_legacy_reuses_estimator_decodes_for_foreground_and_rvm_edge(
    monkeypatch,
):
    cfg = _color_integration_config(
        "image",
        blend_space="srgb_legacy",
        use_model_foreground=True,
    )
    frame = np.full((24, 32, 3), 40, np.uint8)
    clean_edge = np.full((24, 32, 3), 80, np.uint8)
    background = np.full((24, 32, 3), 20, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask, clean_edge),
        _IdentityRefiner(),
        _FixedBackdrop(background),
        None,
        harmonizer=_SpyHarmonizer(ColorTransform(exposure_ev=0.5)),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: object(),
    )
    calls = {"decode": 0, "legacy_predecoded": 0}
    real_decode = pipeline_mod.bgr_u8_to_linear_bgr
    real_legacy_predecoded = pipeline_mod.composite_legacy_predecoded

    def counted_decode(value):
        calls["decode"] += 1
        return real_decode(value)

    def counted_legacy_predecoded(*args, **kwargs):
        calls["legacy_predecoded"] += 1
        return real_legacy_predecoded(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "bgr_u8_to_linear_bgr", counted_decode)
    monkeypatch.setattr(
        pipeline_mod,
        "composite_legacy_predecoded",
        counted_legacy_predecoded,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())

    rendered, reason = pipeline._local_composite(
        resources,
        frame,
        privacy_safe=False,
    )

    assert reason == ""
    assert rendered.dtype == np.uint8
    assert calls == {"decode": 3, "legacy_predecoded": 1}


def test_rvm_edge_foreground_receives_the_same_color_transform(monkeypatch):
    cfg = _color_integration_config("image", use_model_foreground=True)
    frame = np.full((24, 32, 3), 40, np.uint8)
    clean_edge = np.full((24, 32, 3), 80, np.uint8)
    background = np.zeros((24, 32, 3), np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    transform = ColorTransform(exposure_ev=1.0)
    harmonizer = _SpyHarmonizer(transform)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask, clean_edge),
        _IdentityRefiner(),
        _FixedBackdrop(background),
        None,
        harmonizer=harmonizer,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: object(),
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())

    rendered, reason = pipeline._local_composite(
        resources,
        frame,
        privacy_safe=False,
    )

    transformed_edge = apply_color_transform(
        bgr_u8_to_linear_rgb(clean_edge),
        transform,
    )
    expected = linear_rgb_to_bgr_u8(transformed_edge * np.float32(0.5))
    assert reason == ""
    assert np.array_equal(rendered, expected)
    assert len(harmonizer.update_calls) == 1


def test_estimator_exception_is_fail_soft_with_identity_render(monkeypatch):
    cfg = _color_integration_config("image")
    frame = np.full((24, 32, 3), 80, np.uint8)
    background = np.full((24, 32, 3), 20, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    harmonizer = _SpyHarmonizer(ColorTransform(exposure_ev=0.5))
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(background),
        None,
        harmonizer=harmonizer,
    )

    def estimator_failure(*_args, **_kwargs):
        raise RuntimeError("estimator failed")

    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        estimator_failure,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    rendered, reason = pipeline._local_composite(
        resources,
        frame,
        privacy_safe=False,
    )
    expected = pipeline_mod.composite(
        frame,
        background,
        mask,
        blend_space="linear_srgb",
        color_transform=IDENTITY_TRANSFORM,
    )

    assert reason == ""
    assert np.array_equal(rendered, expected)
    assert not harmonizer.update_calls
    assert len(harmonizer.error_calls) == 1


def test_live_light_wrap_preparation_error_falls_back_without_advancing_state(
    monkeypatch,
):
    cfg = _color_integration_config(
        "video",
        correction_mode="off",
        blend_space="linear_srgb",
    ).patched(
        {
            "compositing": {
                "light_wrap": 0.8,
                "light_wrap_stabilization": {
                    "mode": "temporal_bounded",
                    "time_constant_s": 0.2,
                },
            }
        }
    )
    frame = np.full((24, 32, 3), 80, np.uint8)
    background = np.full((24, 32, 3), 20, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)

    class TimedBackdrop(_FixedBackdrop):
        @staticmethod
        def temporal_frame_timing():
            return pipeline_mod.BackdropFrameTiming(1, 1_000_000_000)

    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        TimedBackdrop(background),
        None,
    )
    stabilizer = resources.light_wrap_stabilizer
    assert stabilizer is not None
    before = stabilizer.snapshot()
    monkeypatch.setattr(
        pipeline_mod,
        "prepare_light_wrap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            compositor_mod.ColorError("forced wrap preparation failure")
        ),
    )

    rendered, reason = Pipeline(RuntimeConfig(cfg), FrameHub())._local_composite(
        resources,
        frame,
        privacy_safe=False,
    )
    expected = pipeline_mod.composite(
        frame,
        background,
        mask,
        light_wrap=0.8,
        blend_space="linear_srgb",
        color_transform=IDENTITY_TRANSFORM,
    )

    assert reason == ""
    np.testing.assert_array_equal(rendered, expected)
    assert resources.light_wrap_stabilizer is stabilizer
    assert stabilizer.snapshot() == before


@pytest.mark.parametrize("blend_space", ["srgb_legacy", "linear_srgb"])
def test_immutable_backdrop_light_wrap_is_cached_per_visual_generation(
    monkeypatch,
    blend_space,
):
    cfg = _color_integration_config(
        "color",
        correction_mode="off",
        blend_space=blend_space,
    ).patched(
        {
            "compositing": {
                "light_wrap": 0.8,
                "light_wrap_stabilization": {"mode": "off"},
            }
        }
    )
    foreground = np.full((24, 32, 3), 80, np.uint8)
    backdrop = np.full((24, 32, 3), (10, 30, 90), np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    provider = pipeline_mod.ColorBackdrop((10, 30, 90))
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        provider,
        None,
    )
    prepare_calls = 0
    original_prepare = pipeline_mod.prepare_static_light_wrap

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "prepare_static_light_wrap", counted_prepare)
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    first, first_reason = pipeline._local_composite(
        resources,
        foreground,
        privacy_safe=False,
    )
    second, second_reason = pipeline._local_composite(
        resources,
        foreground,
        privacy_safe=False,
    )
    expected = pipeline_mod.composite(
        foreground,
        backdrop,
        mask,
        light_wrap=0.8,
        blend_space=blend_space,
    )

    assert first_reason == second_reason == ""
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert prepare_calls == 1
    assert resources.static_light_wrap is not None
    assert resources.static_light_wrap.pixels_bgr.flags.writeable is False

    resources.visual_generation += 1
    third, third_reason = pipeline._local_composite(
        resources,
        foreground,
        privacy_safe=False,
    )
    assert third_reason == ""
    np.testing.assert_array_equal(third, expected)
    assert prepare_calls == 2
    resources.close()


def test_private_live_evidence_retains_exact_consumed_prepared_light_wrap(
    monkeypatch,
):
    cfg = _color_integration_config(
        "video",
        correction_mode="off",
        blend_space="linear_srgb",
    ).patched(
        {
            "compositing": {
                "light_wrap": 0.8,
                "light_wrap_stabilization": {
                    "mode": "temporal_bounded",
                    "time_constant_s": 0.2,
                },
            }
        }
    )
    frame = np.full((24, 32, 3), 80, np.uint8)
    background = np.full((24, 32, 3), 20, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)

    class TimedBackdrop(_FixedBackdrop):
        @staticmethod
        def temporal_frame_timing():
            return pipeline_mod.BackdropFrameTiming(1, 1_000_000_000)

    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        TimedBackdrop(background),
        None,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    evidence = MatteFrameEvidence(
        metadata=MatteCaptureMetadata(
            bundle_sequence=0,
            capture_sequence=1,
            capture_monotonic_ns=1_000_000_000,
            timestamp_source="test",
            capture_generation=0,
            geometry_generation=0,
        ),
        raw_frame=frame,
    )
    consumed: dict[str, object] = {}
    real_composite = pipeline._composite_prepared_color

    def capture_prepared(*args, **kwargs):
        consumed["prepared"] = kwargs["prepared_light_wrap"]
        return real_composite(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_composite_prepared_color", capture_prepared)

    rendered, reason = pipeline._local_composite(
        resources,
        frame,
        privacy_safe=False,
        matte_evidence=evidence,
    )

    assert reason == ""
    assert rendered.shape == frame.shape
    assert isinstance(consumed["prepared"], compositor_mod.PreparedLightWrap)
    assert evidence.prepared_light_wrap is consumed["prepared"]
    assert evidence.color_transform.is_identity


@pytest.mark.parametrize("operation", ["transform", "encode"])
def test_opencv_photometric_error_retries_identity_exactly_once(
    monkeypatch,
    operation,
):
    cfg = _color_integration_config("image")
    frame = np.full((24, 32, 3), 80, np.uint8)
    background = np.full((24, 32, 3), 20, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(background),
        None,
        harmonizer=_SpyHarmonizer(ColorTransform(exposure_ev=0.5)),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: object(),
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    real_identity_composite = pipeline_mod.composite
    expected = real_identity_composite(
        frame,
        background,
        mask,
        blend_space="linear_srgb",
        color_transform=IDENTITY_TRANSFORM,
    )
    calls = {"photometric_failure": 0, "identity_retry": 0}

    def fail(*_args, **_kwargs):
        calls["photometric_failure"] += 1
        raise compositor_mod.cv2.error(f"forced {operation} failure")

    if operation == "transform":
        monkeypatch.setattr(compositor_mod.cv2, "transform", fail)
    else:
        monkeypatch.setattr(
            compositor_mod,
            "_consume_linear_bgr_to_bgr_u8_prevalidated",
            fail,
        )

    def identity_retry(*args, **kwargs):
        calls["identity_retry"] += 1
        assert kwargs["color_transform"].is_identity
        return real_identity_composite(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "composite", identity_retry)
    evidence = MatteFrameEvidence(
        metadata=MatteCaptureMetadata(
            bundle_sequence=0,
            capture_sequence=1,
            capture_monotonic_ns=1_000_000_000,
            timestamp_source="test",
            capture_generation=0,
            geometry_generation=0,
        ),
        raw_frame=frame,
    )
    rendered, _ = pipeline._local_composite(
        resources,
        frame,
        privacy_safe=False,
        matte_evidence=evidence,
    )

    assert np.array_equal(rendered, expected)
    assert calls == {"photometric_failure": 1, "identity_retry": 1}
    assert evidence.prepared_light_wrap is None
    assert evidence.color_transform.is_identity


@pytest.mark.parametrize("failure", ["structural", "invalid-output"])
def test_malformed_compositor_or_output_remains_strict(monkeypatch, failure):
    cfg = _color_integration_config("image")
    frame = np.full((24, 32, 3), 80, np.uint8)
    background = np.full((24, 32, 3), 20, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(background),
        None,
        harmonizer=_SpyHarmonizer(),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: object(),
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    if failure == "structural":
        monkeypatch.setattr(
            pipeline,
            "_composite_prepared_color",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("structural compositor failure")
            ),
        )
    else:
        monkeypatch.setattr(
            pipeline,
            "_composite_prepared_color",
            lambda *_args, **_kwargs: np.zeros((24, 32, 3), np.float32),
        )

    with pytest.raises(ValueError):
        pipeline._local_composite(resources, frame, privacy_safe=False)


@pytest.mark.parametrize(
    ("second_capture", "processed_frames", "repeated_frames"),
    [
        pytest.param("pixel-identical", 2, 0, id="successful-identical-read"),
        pytest.param("missing", 1, 1, id="synthesized-output-repeat"),
    ],
)
def test_repeat_output_does_not_advance_harmonizer(
    monkeypatch,
    second_capture,
    processed_frames,
    repeated_frames,
):
    cfg = _color_integration_config("image")
    raw = np.full((24, 32, 3), 64, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    harmonizer = _SpyHarmonizer(ColorTransform(exposure_ev=0.5))
    estimator_calls = []
    capture_frames = [raw, raw.copy() if second_capture == "pixel-identical" else None]
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    output = _StopAfterOutput(pipeline, 2)

    class CountingSegmenter(_FixedMaskSegmenter):
        def __init__(self):
            super().__init__(mask)
            self.calls = 0

        def segment(self, frame):
            self.calls += 1
            return super().segment(frame)

    segmenter = CountingSegmenter()
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture(capture_frames),
        segmenter,
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        output,
        harmonizer=harmonizer,
    )

    def estimate(*_args, **_kwargs):
        estimator_calls.append(True)
        return object()

    monkeypatch.setattr(pipeline_mod, "estimate_color_transform_linear", estimate)
    pipeline._loop(resources)

    assert len(output.frames) == 2
    assert np.array_equal(output.frames[1], output.frames[0])
    assert segmenter.calls == processed_frames
    assert len(estimator_calls) == processed_frames
    assert len(harmonizer.update_calls) == processed_frames
    assert len(harmonizer.reset_calls) == 1
    stats = hub.stats_dict()
    assert stats["frames_in"] == processed_frames
    assert stats["capture_frames_read"] == processed_frames
    assert stats["output_repeated_frames"] == repeated_frames
    assert stats["base_composite_update_count"] == processed_frames
    assert stats["segmentation_update_count"] == processed_frames
    assert stats["base_composite_reuse_count"] == repeated_frames
    assert stats["output_send_count"] == 2
    assert stats["exact_final_output_repeat_count"] == 1
    assert stats["exact_final_output_repeat_ratio"] == 1.0


def test_temporal_timeline_status_counts_gaps_and_boundary_resets(caplog):
    caplog.set_level("INFO", logger="custback.pipeline")
    cfg = _color_integration_config(
        "image",
        correction_mode="off",
        blend_space="srgb_legacy",
    )
    raw = np.full((24, 32, 3), 64, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    captures = [
        _captured(raw.copy(), 1, captured_at_ns=1_000_000_000),
        _captured(raw.copy(), 3, captured_at_ns=1_100_000_000),
        None,
        _captured(raw.copy(), 6, captured_at_ns=2_300_000_000),
    ]

    class TimelineCapture:
        def __init__(self):
            self.frames = list(captures)
            self.frames_read = 0

        def read(self):
            captured = self.frames.pop(0) if self.frames else None
            if captured is not None:
                self.frames_read += 1
            return captured

        def health_snapshot(self):
            return CaptureHealth(
                sequence=6,
                captured_monotonic_ns=2_300_000_000,
                generation=1,
                geometry_generation=1,
                backend="timeline-fake",
                width=32,
                height=24,
                normalized_width=32,
                normalized_height=24,
                frames_read=self.frames_read,
                dropped_frames=3,
            )

    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    output = _StopAfterOutput(pipeline, 4)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        TimelineCapture(),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        output,
    )

    pipeline._loop(resources)

    stats = hub.stats_dict()
    assert stats["capture_sequence"] == 6
    assert stats["capture_sequence_gap_count"] == 2
    assert stats["capture_missing_input_count"] == 3
    assert stats["capture_dropped_frames"] == 3
    assert stats["matte_reset_count"] == 2
    assert stats["matte_last_reset_reason"] == "timestamp-gap"
    assert stats["output_repeated_frames"] == 1
    reset_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("matte temporal reset ")
    ]
    assert len(reset_messages) == 2
    assert "reason=initial sequence_gap=0 elapsed_gap_ms=none" in reset_messages[0]
    assert (
        "reason=timestamp-gap sequence_gap=2 elapsed_gap_ms=1200.000"
        in reset_messages[1]
    )
    assert "capture_generation=1 geometry_generation=1" in reset_messages[1]
    assert "segmentation_generation=0 config_version=0" in reset_messages[1]


@pytest.mark.parametrize(
    ("mode", "segmentation_backend", "background_overrides"),
    [
        pytest.param("passthrough", "heuristic", {}, id="passthrough"),
        pytest.param(
            "remote",
            "none",
            {"remote_fallback_mode": "color"},
            id="remote-no-segmenter",
        ),
    ],
)
def test_capture_status_advances_when_output_mode_bypasses_segmentation(
    mode,
    segmentation_backend,
    background_overrides,
):
    cfg = (
        make_runtime(mode=mode, **background_overrides)
        .snapshot()
        .patched({"segmentation": {"backend": segmentation_backend}})
    )
    pipeline, hub = run_pipeline(RuntimeConfig(cfg))
    try:
        stats = wait_for_stats(
            hub,
            lambda value: (
                value["frames_out"] >= 5
                if mode == "remote"
                else value["frames_in"] >= 5
            ),
        )
        assert stats["capture_sequence"] == stats["frames_in"]
        assert stats["capture_frames_read"] == stats["frames_in"]
        if mode == "remote":
            assert stats["frames_in"] == 1
            assert stats["output_repeated_frames"] >= 4
        assert stats["capture_sequence_gap_count"] == 0
        assert stats["capture_missing_input_count"] == 0
    finally:
        pipeline.stop()


def test_remote_to_local_transition_reports_capture_health_truthfully():
    cfg = (
        make_runtime(mode="remote", remote_fallback_mode="color")
        .snapshot()
        .patched({"segmentation": {"backend": "none"}})
    )
    pipeline, hub = run_pipeline(RuntimeConfig(cfg))
    try:
        before = wait_for_stats(hub, lambda value: value["frames_out"] >= 5)
        assert before["capture_sequence"] == before["frames_in"]
        assert before["frames_in"] == 1
        assert before["capture_sequence_gap_count"] == 0
        assert before["capture_missing_input_count"] == 0

        committed = pipeline.apply_config_patch(
            {"background": {"mode": "color"}},
            origin="test",
        )
        after = wait_for_stats(
            hub,
            lambda value: (
                value["config_version"] == committed.version
                and value["capture_sequence"] > before["capture_sequence"]
            ),
        )
        assert after["capture_sequence"] == after["capture_frames_read"]
        assert after["frames_in"] > before["frames_in"]
        missing = after["capture_missing_input_count"]
        gap_events = after["capture_sequence_gap_count"]
        assert missing >= gap_events
        assert (missing == 0) == (gap_events == 0)
    finally:
        pipeline.stop()


@pytest.mark.parametrize("remote_candidate", [False, True])
def test_auto_correction_leaves_remote_candidate_or_privacy_slate_untouched(
    monkeypatch,
    remote_candidate,
):
    cfg = _color_integration_config("remote")
    raw_values = np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3)
    raw = (raw_values % 251).astype(np.uint8)
    candidate = np.full((24, 32, 3), (7, 101, 223), np.uint8)
    harmonizer = _SpyHarmonizer(ColorTransform(exposure_ev=1.0))
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    output = _StopAfterOutput(pipeline, 1)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([raw]),
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        output,
        harmonizer=harmonizer,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: pytest.fail(
            "remote output reached local color estimation"
        ),
    )
    session = hub.remote_client_connected() if remote_candidate else None
    if session is not None:
        hub.publish_remote_raw(raw, 1)
        assert hub.push_remote_frame(
            candidate,
            raw_epoch=1,
            session_id=session,
        )
    try:
        pipeline._loop(resources)
    finally:
        if session is not None:
            hub.remote_client_disconnected(session)

    expected = candidate if remote_candidate else Pipeline._privacy_slate(raw.shape)
    published, _ = hub.output.latest()
    assert published is not None
    assert np.array_equal(published, expected)
    assert np.array_equal(output.frames[0], expected)
    assert not harmonizer.reset_calls
    assert not harmonizer.update_calls
    assert not harmonizer.error_calls


def test_color_correction_has_a_separate_deterministic_timing_bucket(monkeypatch):
    cfg = _color_integration_config("image")
    mask = np.full((24, 32), 0.5, np.float32)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        None,
        harmonizer=_SpyHarmonizer(),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: object(),
    )
    ticks = iter(
        [
            0,
            1_000_000,
            2_000_000,
            4_000_000,
            5_000_000,
            12_000_000,
            13_000_000,
            14_000_000,
            14_000_000,
            16_000_000,
            16_000_000,
        ]
    )
    monkeypatch.setattr(pipeline_mod.time, "monotonic_ns", lambda: next(ticks))
    timings = {}
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())

    pipeline._local_composite(
        resources,
        np.full((24, 32, 3), 80, np.uint8),
        privacy_safe=False,
        timings=timings,
    )

    assert timings == {
        "segmentation_ms": 1.0,
        "background_ms": 2.0,
        "color_correction_ms": 7.0,
        "composite_prepare_ms": 1.0,
        "composite_blend_ms": 2.0,
        "composite_ms": 3.0,
    }


def test_inactive_live_matte_monitor_keeps_evidence_fast_path_off():
    cfg = _color_integration_config("image")
    monitor = LocalMatteDiagnosticMonitor()
    pipeline = Pipeline(
        RuntimeConfig(cfg),
        FrameHub(),
        matte_monitor=monitor,
    )
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        None,
    )
    raw = np.full((24, 32, 3), 80, np.uint8)
    try:
        assert not monitor.accepting
        assert pipeline._new_matte_evidence(resources, _captured(raw)) is None
    finally:
        monitor.close()
        resources.close()


def test_live_matte_view_never_reaches_output_or_normal_hub_slots():
    cfg = _color_integration_config("image")
    source = np.empty((24, 32, 3), np.uint8)
    source[..., 0] = 11
    source[..., 1] = 73
    source[..., 2] = 191
    mask = np.full((24, 32), 0.25, np.float32)
    backdrop = np.full((24, 32, 3), (201, 31, 7), np.uint8)
    hub = FrameHub()
    monitor = LocalMatteDiagnosticMonitor()
    assert monitor.select("raw_alpha") == "raw_alpha"
    pipeline = Pipeline(
        RuntimeConfig(cfg),
        hub,
        matte_monitor=monitor,
    )
    output = _StopAfterOutput(pipeline, 1)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([source]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(backdrop),
        output,
    )

    try:
        pipeline._loop(resources)
        diagnostic, _sequence = monitor.get(-1, timeout=2.0)
        assert diagnostic is not None
        assert diagnostic.view == "raw_alpha"
        assert np.all(diagnostic.pixels == 64)
        assert diagnostic.status["capture_sequence"] == diagnostic.capture_sequence

        published_output, _timestamp = hub.output.latest()
        published_raw, _timestamp = hub.raw.latest()
        assert published_output is not None
        assert published_raw is not None
        assert len(output.frames) == 1
        np.testing.assert_array_equal(published_output, output.frames[0])
        np.testing.assert_array_equal(published_raw, source)
        assert not np.array_equal(output.frames[0], diagnostic.pixels)
        assert not np.array_equal(published_raw, diagnostic.pixels)
        assert "matte_diagnostic" not in repr(hub.stats_dict())
    finally:
        monitor.close()
        resources.close()


def test_live_matte_view_does_not_force_segmentation_in_passthrough():
    cfg = _color_integration_config("passthrough")
    source = np.full((24, 32, 3), (17, 83, 149), np.uint8)

    class ForbiddenSegmenter(_FixedMaskSegmenter):
        def segment(self, *_args, **_kwargs):
            pytest.fail("diagnostic view advanced bypassed segmentation state")

    hub = FrameHub()
    monitor = LocalMatteDiagnosticMonitor()
    monitor.select("raw_alpha")
    pipeline = Pipeline(
        RuntimeConfig(cfg),
        hub,
        matte_monitor=monitor,
    )
    output = _StopAfterOutput(pipeline, 1)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([source]),
        ForbiddenSegmenter(np.zeros((24, 32), np.float32)),
        _IdentityRefiner(),
        None,
        output,
    )

    try:
        pipeline._loop(resources)
        diagnostic, _sequence = monitor.get(-1, timeout=2.0)
        assert diagnostic is not None
        assert not diagnostic.available
        assert diagnostic.unavailable_reason == "raw model alpha is unavailable"
        np.testing.assert_array_equal(output.frames[0], source)
        np.testing.assert_array_equal(hub.output.latest()[0], source)
        np.testing.assert_array_equal(hub.raw.latest()[0], source)
    finally:
        monitor.close()
        resources.close()


def test_private_matte_frame_total_includes_sink_send_and_samples_rss(
    tmp_path,
    monkeypatch,
):
    cfg = _color_integration_config("image")
    raw = np.full((24, 32, 3), 80, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    bundle_root = tmp_path / "bundle"
    recorder = MatteDiagnosticRecorder(bundle_root, max_bytes=4_000_000)
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub, matte_recorder=recorder)

    class SlowOutput(_StopAfterOutput):
        def send(self, frame):
            time.sleep(0.01)
            super().send(frame)

    output = SlowOutput(pipeline, 1)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([raw]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        output,
    )
    monkeypatch.setattr(pipeline_mod, "process_rss_bytes", lambda: 123_456_789)

    try:
        pipeline._loop(resources)
    finally:
        recorder.close()

    bundle = MatteReplayBundle(bundle_root)
    assert len(bundle.frames) == 1
    frame = bundle.frames[0]
    timings = frame["timings_ms"]
    assert timings["output_send_ms"] >= 8.0
    assert timings["frame_total_ms"] >= timings["output_send_ms"]
    assert timings["frame_total_ms"] >= timings["frame_processing_ms"]
    assert set(frame["compositor_substages_ms"]) == set(
        compositor_mod.COMPOSITOR_SUBSTAGE_NAMES
    )
    assert all(
        isinstance(value, float) and np.isfinite(value) and value >= 0.0
        for value in frame["compositor_substages_ms"].values()
    )
    for name in (
        "post_composite_validation_ms",
        "guard_output_validation_ms",
        "output_submission_ms",
        "output_sink_pacing_wait_ms",
        "application_pacing_wait_ms",
        "output_schedule_lateness_ms",
        "new_frame_service_ms",
        "new_frame_serialized_loop_ms",
    ):
        assert isinstance(timings[name], float)
        assert np.isfinite(timings[name])
        assert timings[name] >= 0.0
    assert timings["output_send_ms"] >= timings["output_submission_ms"]
    assert frame["resource_samples"]["rss_bytes"] == 123_456_789


def test_private_matte_evidence_records_late_provider_fallback_without_reason_path():
    cfg = _color_integration_config("image")
    frame = np.full((24, 32, 3), 80, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)

    class FallbackStatus:
        def __init__(self, fallback):
            self.requested_mode = "auto"
            self.requested_provider = "cuda"
            self.device_id = 2
            self.state = "cpu_fallback" if fallback else "gpu_active"
            self.active_provider = "cpu" if fallback else "cuda"
            self.fallback_active = fallback
            self.fallback_reason = "/private/models/rvm.onnx failed" if fallback else ""
            self.fallback_count = int(fallback)
            self.last_transition_ms = 0.1

    class LateFallbackAcceleration:
        def __init__(self):
            self.fallback = False

        def status(self):
            return FallbackStatus(self.fallback)

    class LateFallbackSegmenter(_FixedMaskSegmenter):
        def __init__(self):
            super().__init__(mask, np.full_like(frame, 90))
            self.device = "cuda"
            self.accel = LateFallbackAcceleration()

        def segment(self, _frame):
            self.device = "cpu"
            self.accel.fallback = True
            return super().segment(_frame)

    segmenter = LateFallbackSegmenter()
    assert segmenter.accel.status().active_provider == "cuda"
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        segmenter,
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        None,
    )
    evidence = MatteFrameEvidence(
        metadata=MatteCaptureMetadata(
            bundle_sequence=0,
            capture_sequence=1,
            capture_monotonic_ns=1_000_000_000,
            timestamp_source="capture-completion",
            capture_generation=0,
            geometry_generation=0,
        ),
        raw_frame=frame,
    )

    rendered, reason = Pipeline(RuntimeConfig(cfg), FrameHub())._local_composite(
        resources,
        frame,
        privacy_safe=False,
        matte_evidence=evidence,
    )

    assert reason == ""
    assert rendered.shape == frame.shape
    assert evidence.matte_authoritative is True
    assert evidence.effective_controls["segmentation_device"] == "cpu"
    assert evidence.effective_controls["acceleration"] == {
        "applicable": True,
        "requested_mode": "auto",
        "requested_provider": "cuda",
        "device_id": 2,
        "state": "cpu_fallback",
        "active_provider": "cpu",
        "fallback_active": True,
        "fallback_count": 1,
        "fallback_reason_code": "provider-fallback",
    }
    selection = segmenter_selection_status(segmenter, "auto")
    assert selection["selected_backend"] == "rvm"
    assert selection["quality_tier"] == "matting"
    assert selection["fallback_active"] is False
    assert selection["active_device"] == "cpu"
    assert selection["active_provider"] == "cpu"
    assert "/private" not in repr(evidence.effective_controls)


def test_private_matte_evidence_records_rvm_identity_ratio_and_stage_timings():
    cfg = _color_integration_config("image")
    frame = np.full((24, 32, 3), 80, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)

    class TelemetrySegmenter(_FixedMaskSegmenter):
        def __init__(self):
            super().__init__(mask, np.full_like(frame, 90))
            self.last_downsample_ratio = 1.0
            self.snapshot_count = 0

        def rvm_telemetry_snapshot(self):
            self.snapshot_count += 1
            return RVMTelemetry(
                input_frame_shape=(24, 32),
                output_alpha_shape=(24, 32),
                output_foreground_shape=(24, 32, 3),
                configured_downsample_mode="auto",
                configured_downsample_ratio=0.0,
                resolved_downsample_ratio=1.0,
                preprocess_ms=1.25,
                session_run_ms=2.5,
                postprocess_ms=0.75,
                model_builtin=True,
                model_identity="rvm_mobilenetv3_fp32.onnx",
                model_sha256="a" * 64,
                model_bytes=1_234,
                acceleration_state="cpu_fallback",
                acceleration_active_provider="cpu",
                acceleration_fallback_active=False,
                acceleration_fallback_count=0,
            )

    segmenter = TelemetrySegmenter()
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        segmenter,
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        None,
    )
    evidence = MatteFrameEvidence(
        metadata=MatteCaptureMetadata(
            bundle_sequence=0,
            capture_sequence=1,
            capture_monotonic_ns=1_000_000_000,
            timestamp_source="capture-completion",
            capture_generation=0,
            geometry_generation=0,
        ),
        raw_frame=frame,
    )

    rendered, reason = Pipeline(RuntimeConfig(cfg), FrameHub())._local_composite(
        resources,
        frame,
        privacy_safe=False,
        matte_evidence=evidence,
    )

    assert reason == ""
    assert rendered.shape == frame.shape
    assert segmenter.snapshot_count == 1
    assert evidence.effective_controls["rvm_telemetry"] == {
        "applicable": True,
        "input_frame_shape": [24, 32],
        "output_alpha_shape": [24, 32],
        "output_foreground_shape": [24, 32, 3],
        "configured_downsample_mode": "auto",
        "configured_downsample_ratio": 0.0,
        "resolved_downsample_ratio": 1.0,
        "preprocess_ms": 1.25,
        "session_run_ms": 2.5,
        "postprocess_ms": 0.75,
        "model_builtin": True,
        "model_identity": "rvm_mobilenetv3_fp32.onnx",
        "model_sha256": "a" * 64,
        "model_bytes": 1_234,
        "acceleration_state": "cpu_fallback",
        "acceleration_active_provider": "cpu",
        "acceleration_fallback_active": False,
        "acceleration_fallback_count": 0,
    }
    assert evidence.timings_ms["rvm_preprocess_ms"] == 1.25
    assert evidence.timings_ms["rvm_session_run_ms"] == 2.5
    assert evidence.timings_ms["rvm_postprocess_ms"] == 0.75
    assert "/private" not in repr(evidence.effective_controls)


def test_rvm_evidence_rejects_untyped_hostile_or_failing_snapshots():
    class UntypedSnapshot:
        @staticmethod
        def rvm_telemetry_snapshot():
            return {"model_identity": "/private/models/rvm.onnx"}

    class FailingSnapshot:
        @staticmethod
        def rvm_telemetry_snapshot():
            raise RuntimeError("/private/models/rvm.onnx")

    class HostileTypedSnapshot:
        @staticmethod
        def rvm_telemetry_snapshot():
            return RVMTelemetry(
                input_frame_shape=(24, 32),
                output_alpha_shape=(24, 32),
                output_foreground_shape=(24, 32, 3),
                configured_downsample_mode="auto",
                configured_downsample_ratio=0.0,
                resolved_downsample_ratio=1.0,
                preprocess_ms=float("nan"),
                session_run_ms=2.5,
                postprocess_ms=0.75,
                model_builtin=False,
                model_identity="/private/models/rvm.onnx",
                model_sha256="a" * 64,
                model_bytes=1_234,
                acceleration_state="cpu_fallback",
                acceleration_active_provider="cpu",
                acceleration_fallback_active=False,
                acceleration_fallback_count=0,
            )

    assert pipeline_mod._rvm_telemetry_evidence(UntypedSnapshot()) == {
        "applicable": False
    }
    assert pipeline_mod._rvm_telemetry_evidence(FailingSnapshot()) == {
        "applicable": False
    }
    assert pipeline_mod._rvm_telemetry_evidence(HostileTypedSnapshot()) == {
        "applicable": False
    }


def test_output_sink_evidence_uses_actual_bounded_pacing_contract():
    assert pipeline_mod._output_sink_evidence(NullOutput(32, 24, 30)) == {
        "applicable": True,
        "backend": "null",
        "paces": False,
    }

    class UnknownPacingSink:
        paces = True

    assert pipeline_mod._output_sink_evidence(UnknownPacingSink()) == {
        "applicable": True,
        "backend": "unknown",
        "paces": True,
    }
    assert pipeline_mod._output_sink_evidence(None) == {
        "applicable": False,
        "backend": "unknown",
        "paces": None,
    }


def test_mediapipe_telemetry_is_attached_only_to_private_matte_evidence():
    cfg = _color_integration_config("color")
    frame = np.full((24, 32, 3), 80, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _TelemetryMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        None,
    )
    evidence = MatteFrameEvidence(
        metadata=MatteCaptureMetadata(
            bundle_sequence=0,
            capture_sequence=1,
            capture_monotonic_ns=1_000_000_000,
            timestamp_source="capture-completion",
            capture_generation=0,
            geometry_generation=0,
        ),
        raw_frame=frame,
    )

    rendered, reason = Pipeline(RuntimeConfig(cfg), FrameHub())._local_composite(
        resources,
        frame,
        privacy_safe=False,
        matte_evidence=evidence,
    )

    assert reason == ""
    assert rendered.shape == frame.shape
    assert evidence.segmentation_diagnostics == {
        "backend": "mediapipe",
        "input_frame_shape": [24, 32],
        "model_mask_shape": [12, 16],
        "output_mask_shape": [24, 32],
        "effective_timestamp_delta_ms": 34,
        "timestamp_adjustment_count": 1,
        "timestamp_adjustment_ms": 0,
        "last_timestamp_adjusted": False,
        "resize_interpolation": "linear",
    }
    assert "effective_timestamp_ms" not in evidence.segmentation_diagnostics
    assert not (
        set(evidence.segmentation_diagnostics)
        & (set(evidence.effective_controls) | set(evidence.timings_ms))
    )


@pytest.mark.parametrize("change", ["visual", "segmentation"])
def test_successful_color_state_commit_installs_pristine_harmonizer(
    monkeypatch,
    change,
):
    current = _color_integration_config("image")
    if change == "visual":
        candidate = current.patched({"background": {"anchor_x": 0.25}})
    else:
        candidate = current.patched({"segmentation": {"threshold": 0.61}})
        monkeypatch.setattr(
            pipeline_mod,
            "create_segmenter",
            lambda *_args, **_kwargs: _FixedMaskSegmenter(
                np.full((24, 32), 0.5, np.float32)
            ),
        )
        monkeypatch.setattr(
            pipeline_mod,
            "refiner_for",
            lambda *_args, **_kwargs: _IdentityRefiner(),
        )

    live = ColorHarmonizer(0.8, mode="image")
    live.reset(5.0, reason=ColorReason.INVALID)
    resources = pipeline_mod._Resources(
        current,
        0,
        None,
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.zeros((24, 32, 3), np.uint8)),
        None,
        harmonizer=live,
        color_reset_token=("live",),
    )
    runtime = RuntimeConfig(current)
    pipeline = Pipeline(runtime, FrameHub())
    activation = pipeline._prepare_activation_off_lane(current, candidate)
    staged_harmonizer = activation.harmonizer
    assert staged_harmonizer is not None
    pristine = staged_harmonizer.snapshot()
    request = pipeline_mod._PatchRequest(
        candidate,
        0,
        prepared_activation=activation,
    )

    pipeline._handle_patch_request(
        resources,
        request,
        _captured(np.zeros((24, 32, 3), np.uint8)),
    )

    assert request.error is None
    assert request.result is not None
    assert resources.harmonizer is staged_harmonizer
    assert staged_harmonizer.snapshot() == pristine
    assert resources.color_reset_token is None


def test_unrelated_hot_commit_preserves_harmonizer_object_and_snapshot():
    current = _color_integration_config("image")
    candidate = current.patched({"api": {"remote_timeout_ms": 750}})
    live = ColorHarmonizer(0.8, mode="image")
    live.reset(5.0, reason=ColorReason.INVALID)
    snapshot = live.snapshot()
    resources = pipeline_mod._Resources(
        current,
        0,
        None,
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.zeros((24, 32, 3), np.uint8)),
        None,
        harmonizer=live,
        color_reset_token=("stable",),
    )
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    request = pipeline_mod._PatchRequest(
        candidate,
        0,
        prepared_activation=pipeline._prepare_activation_off_lane(
            current,
            candidate,
        ),
    )

    pipeline._handle_patch_request(
        resources,
        request,
        _captured(np.zeros((24, 32, 3), np.uint8)),
    )

    assert request.error is None
    assert resources.harmonizer is live
    assert live.snapshot() == snapshot
    assert resources.color_reset_token == ("stable",)


@pytest.mark.parametrize(
    ("patch", "expected_adaptation"),
    [
        ({"compositing": {"color_correction": {"strength": 0.25}}}, 0.8),
        (
            {"compositing": {"color_correction": {"exposure_limit_ev": 0.95}}},
            0.8,
        ),
        (
            {"compositing": {"color_correction": {"white_balance_strength": 0.25}}},
            0.8,
        ),
        (
            {"compositing": {"color_correction": {"adaptation_time_s": 1.7}}},
            1.7,
        ),
    ],
)
def test_safe_color_scalar_commit_preserves_applied_temporal_state(
    patch,
    expected_adaptation,
):
    current = _color_integration_config("image")
    candidate = current.patched(patch)
    live = _seeded_color_harmonizer()
    before = live.snapshot()
    resources = pipeline_mod._Resources(
        current,
        0,
        None,
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.zeros((24, 32, 3), np.uint8)),
        None,
        harmonizer=live,
        color_reset_token=("stable",),
        visual_generation=4,
    )
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    activation = pipeline._prepare_activation_off_lane(current, candidate)
    assert activation.replace_harmonizer is False
    assert activation.reconfigure_harmonizer is True
    request = pipeline_mod._PatchRequest(
        candidate,
        0,
        prepared_activation=activation,
    )

    pipeline._handle_patch_request(
        resources,
        request,
        _captured(
            np.zeros((24, 32, 3), np.uint8),
            captured_at_ns=10_000_000_000,
        ),
    )

    assert request.error is None
    assert request.result is not None
    assert resources.harmonizer is not live
    assert resources.harmonizer is not None
    assert resources.harmonizer.snapshot() == before
    assert resources.harmonizer.adaptation_time_s == expected_adaptation
    assert resources.color_reset_token == ("stable",)
    assert resources.visual_generation == 4
    resources.close()


@pytest.mark.parametrize(
    ("initial_exposure", "new_limit", "expected_exposure"),
    [(0.7, 0.3, 0.3), (0.2, 0.3, 0.2)],
)
def test_lowered_exposure_bound_clamps_only_when_current_state_requires_it(
    initial_exposure,
    new_limit,
    expected_exposure,
):
    current = _color_integration_config("image")
    candidate = current.patched(
        {"compositing": {"color_correction": {"exposure_limit_ev": new_limit}}}
    )
    live = _seeded_color_harmonizer(initial_exposure)
    before = live.snapshot()
    resources = pipeline_mod._Resources(
        current,
        0,
        None,
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.zeros((24, 32, 3), np.uint8)),
        None,
        harmonizer=live,
        color_reset_token=("stable",),
    )
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    request = pipeline_mod._PatchRequest(
        candidate,
        0,
        prepared_activation=pipeline._prepare_activation_off_lane(
            current,
            candidate,
        ),
    )

    pipeline._handle_patch_request(
        resources,
        request,
        _captured(
            np.zeros((24, 32, 3), np.uint8),
            captured_at_ns=10_000_000_000,
        ),
    )

    assert request.error is None
    assert resources.harmonizer is not None
    after = resources.harmonizer.snapshot()
    assert after.transform.exposure_ev == pytest.approx(expected_exposure, abs=0.01)
    assert after.transform.wb_gains == before.transform.wb_gains
    for name in (
        "phase",
        "reason",
        "confidence",
        "reliable",
        "last_timestamp_s",
        "last_reliable_s",
        "low_confidence_since_s",
        "fast_until_s",
        "source_generation",
        "signature",
    ):
        assert getattr(after, name) == getattr(before, name)
    assert resources.color_reset_token == ("stable",)
    resources.close()


@pytest.mark.parametrize(
    "patch",
    [
        {"compositing": {"color_correction": {"mode": "off"}}},
        {"background": {"image_path": "/deterministic/other.png"}},
        {"background": {"anchor_x": 0.25}},
        {"segmentation": {"threshold": 0.61}},
        {"compositing": {"blend_space": "srgb_legacy"}},
    ],
)
def test_color_state_key_retains_every_hard_reset_boundary(patch):
    current = _color_integration_config("image")
    candidate = current.patched(patch)
    assert pipeline_mod._color_state_key(candidate) != pipeline_mod._color_state_key(
        current
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"compositing": {"color_correction": {"strength": 0.25}}},
        {"compositing": {"color_correction": {"exposure_limit_ev": 0.5}}},
        {"compositing": {"color_correction": {"white_balance_strength": 0.25}}},
        {"compositing": {"color_correction": {"adaptation_time_s": 1.7}}},
    ],
)
def test_color_scalar_policy_does_not_expand_the_hard_reset_key(patch):
    current = _color_integration_config("image")
    candidate = current.patched(patch)
    assert pipeline_mod._color_state_key(candidate) == pipeline_mod._color_state_key(
        current
    )
    assert pipeline_mod._color_scalar_policy_key(
        candidate
    ) != pipeline_mod._color_scalar_policy_key(current)


def test_color_clamp_telemetry_advances_only_on_unique_processed_frames():
    cfg = _color_integration_config("image")
    resources = pipeline_mod._Resources(
        cfg,
        0,
        None,
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.zeros((24, 32, 3), np.uint8)),
        None,
    )
    status = {
        "color_correction_active": True,
        "color_correction_state": "active",
        "color_correction_exposure_clamped": True,
        "color_correction_wb_clamped": True,
    }

    Pipeline._record_color_output(resources, status, processed=True, observed_at_s=10.0)
    Pipeline._record_color_output(
        resources, status, processed=False, observed_at_s=10.4
    )
    Pipeline._record_color_output(
        resources,
        {**status, "color_correction_wb_clamped": False},
        processed=True,
        observed_at_s=10.5,
    )
    Pipeline._record_color_output(
        resources,
        {
            **status,
            "color_correction_exposure_clamped": False,
            "color_correction_wb_clamped": False,
        },
        processed=True,
        observed_at_s=11.0,
    )

    assert resources.color_correction_applied_frames == 4
    assert resources.color_correction_exposure_clamped is False
    assert resources.color_correction_exposure_clamp_count == 2
    assert resources.color_correction_exposure_clamp_time_s == pytest.approx(1.0)
    assert resources.color_correction_wb_clamped is False
    assert resources.color_correction_wb_clamp_count == 1
    assert resources.color_correction_wb_clamp_time_s == pytest.approx(0.5)
    resources.close()


@pytest.mark.parametrize("outcome", ["trial", "conflict", "cancel"])
def test_rejected_activation_paths_do_not_mutate_live_harmonizer(
    monkeypatch,
    outcome,
):
    current = _color_integration_config("image")
    candidate = current.patched(
        {"compositing": {"color_correction": {"adaptation_time_s": 1.7}}}
    )
    live = ColorHarmonizer(0.8, mode="image")
    live.reset(5.0, reason=ColorReason.INVALID)
    snapshot = live.snapshot()
    resources = pipeline_mod._Resources(
        current,
        0,
        None,
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.zeros((24, 32, 3), np.uint8)),
        None,
        harmonizer=live,
        color_reset_token=("stable",),
    )
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    request = pipeline_mod._PatchRequest(
        candidate,
        1 if outcome == "conflict" else 0,
        prepared_activation=pipeline._prepare_activation_off_lane(
            current,
            candidate,
        ),
    )
    if outcome == "cancel":
        request.cancelled = True
    elif outcome == "trial":
        monkeypatch.setattr(
            pipeline,
            "_trial_activation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ActivationError("trial failed")
            ),
        )

    pipeline._handle_patch_request(
        resources,
        request,
        _captured(np.zeros((24, 32, 3), np.uint8)),
    )

    assert resources.harmonizer is live
    assert live.snapshot() == snapshot
    assert resources.color_reset_token == ("stable",)
    assert resources.cfg is current
    assert resources.version == 0


def test_preparation_timeout_does_not_mutate_live_harmonizer(monkeypatch):
    real_factory = pipeline_mod._new_color_harmonizer
    created = []

    def tracked_factory(cfg):
        harmonizer = real_factory(cfg)
        created.append(harmonizer)
        return harmonizer

    monkeypatch.setattr(pipeline_mod, "_new_color_harmonizer", tracked_factory)
    runtime = make_runtime(mode="color")
    pipeline, _hub = run_pipeline(runtime)
    live = created[0]
    live.reset(5.0, reason=ColorReason.INVALID)
    snapshot = live.snapshot()
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class CandidateSegmenter:
        device = "cpu"
        last_foreground = None
        produces_matte = False

        def segment(self, frame):
            return np.full(frame.shape[:2], 0.5, np.float32)

        def close(self):
            closed.set()

    def blocked_build(*_args, **_kwargs):
        entered.set()
        release.wait(1.0)
        return CandidateSegmenter()

    monkeypatch.setattr(pipeline_mod, "create_segmenter", blocked_build)
    try:
        with pytest.raises(ReconfigurationUnavailable, match="preparation exceeded"):
            pipeline.apply_config_patch(
                {"segmentation": {"threshold": 0.61}},
                timeout=0.03,
            )
        assert entered.is_set()
        assert live.snapshot() == snapshot

        release.set()
        assert closed.wait(1.0)
        assert len(created) == 2
        assert created[0] is live
        assert created[1] is not live
        assert live.snapshot() == snapshot
        assert runtime.version == 0
    finally:
        release.set()
        pipeline.stop()


def test_queued_ack_timeout_and_late_cancel_ack_preserve_live_harmonizer(
    monkeypatch,
):
    current = _color_integration_config("image")
    candidate = current.patched({"background": {"anchor_x": 0.25}})
    live = ColorHarmonizer(0.8, mode="image")
    live.reset(5.0, reason=ColorReason.INVALID)
    snapshot = live.snapshot()
    resources = pipeline_mod._Resources(
        current,
        0,
        None,
        _FixedMaskSegmenter(np.full((24, 32), 0.5, np.float32)),
        _IdentityRefiner(),
        _FixedBackdrop(np.zeros((24, 32, 3), np.uint8)),
        None,
        harmonizer=live,
        color_reset_token=("stable",),
    )
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    request = pipeline_mod._PatchRequest(
        candidate,
        0,
        prepared_activation=pipeline._prepare_activation_off_lane(
            current,
            candidate,
        ),
    )

    class ImmediateTimeoutEvent:
        def __init__(self):
            self.set_calls = 0

        def wait(self, _timeout=None):
            return False

        def set(self):
            self.set_calls += 1

    request.done = cast(Any, ImmediateTimeoutEvent())
    monkeypatch.setattr(
        pipeline,
        "_enqueue_request",
        lambda queued: pipeline._requests.put(queued),
    )

    with pytest.raises(
        ReconfigurationUnavailable,
        match="did not acknowledge",
    ):
        pipeline._submit_patch(request, timeout=1.0)

    assert request.cancelled
    assert request.prepared_activation is None
    assert resources.harmonizer is live
    assert live.snapshot() == snapshot
    queued = pipeline._requests.get_nowait()
    assert queued is request

    pipeline._handle_patch_request(
        resources,
        request,
        _captured(np.zeros((24, 32, 3), np.uint8)),
    )

    assert resources.harmonizer is live
    assert live.snapshot() == snapshot
    assert resources.color_reset_token == ("stable",)
    assert resources.cfg is current
    assert resources.version == 0


def test_legacy_compositor_workspace_is_lazy_reused_and_generation_owned():
    frame = np.full((24, 32, 3), 80, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    cfg = _color_integration_config(
        "image",
        correction_mode="off",
        blend_space="srgb_legacy",
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        None,
    )
    assert resources._legacy_compositor_workspace is None

    try:
        first, first_reason = pipeline._local_composite(
            resources,
            frame,
            privacy_safe=False,
        )
        workspace = resources._legacy_compositor_workspace
        assert workspace is not None
        first_snapshot = workspace.snapshot()

        second, second_reason = pipeline._local_composite(
            resources,
            frame,
            privacy_safe=False,
        )

        assert first_reason == second_reason == ""
        assert np.array_equal(first, second)
        assert resources._legacy_compositor_workspace is workspace
        assert first_snapshot.calls == 1
        assert workspace.snapshot().calls == 2
    finally:
        workspace = resources._legacy_compositor_workspace
        resources.close()

    assert workspace is not None
    assert workspace.snapshot().closed is True
    assert workspace.snapshot().retained_bytes == 0
    assert resources._legacy_compositor_workspace is None
    resources.close()


def test_linear_compositor_does_not_allocate_legacy_workspace():
    frame = np.full((24, 32, 3), 80, np.uint8)
    mask = np.full((24, 32), 0.5, np.float32)
    cfg = _color_integration_config(
        "image",
        correction_mode="off",
        blend_space="linear_srgb",
    )
    resources = pipeline_mod._Resources(
        cfg,
        0,
        _SequenceCapture([]),
        _FixedMaskSegmenter(mask),
        _IdentityRefiner(),
        _FixedBackdrop(np.full((24, 32, 3), 20, np.uint8)),
        None,
    )

    try:
        rendered, reason = Pipeline(RuntimeConfig(cfg), FrameHub())._local_composite(
            resources,
            frame,
            privacy_safe=False,
        )
        assert reason == ""
        assert rendered.shape == frame.shape
        assert resources._legacy_compositor_workspace is None
    finally:
        resources.close()


def test_timing_projection_populates_reserved_compositor_buckets():
    stage_ewma = {
        "composite_ms": 4.0,
        "composite_prepare_ms": 1.25,
        "composite_blend_ms": 2.75,
    }

    projected = pipeline_mod._timing_fields(
        stage_ewma,
        capture_health=CaptureHealth(),
        segmenter=object(),
    )

    assert projected["compositor.total"] == 4.0
    assert projected["compositor.prepare"] == 1.25
    assert projected["compositor.blend"] == 2.75
