"""RVM matting backend tests with a fake onnxruntime — no model download,
no GPU, no network. The fake mimics the RVM ONNX interface: src + 4 recurrent
states + downsample_ratio in, (fgr, pha, 4 states) out."""

import hashlib
import json
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import custback.pipeline as pipeline_mod
import custback.segmentation as segmentation_mod
from custback.capture import CapturedFrame
from custback.config import AppConfig, RuntimeConfig, SegmentationConfig
from custback.hub import FrameHub
from custback.matte_diagnostics import MatteDiagnosticRecorder, MatteReplayBundle
from custback.pipeline import (
    ActivationError,
    Pipeline,
    ReconfigurationUnavailable,
    _PatchRequest,
    _Resources,
)
from custback.segmentation import (
    SEGMENTATION_TIMESTAMP_GAP_RESET_NS,
    HeuristicSegmenter,
    MaskRefiner,
    RVMSegmenter,
    RVMTelemetry,
    SegmentationFrameContext,
    TemporalResetReason,
    create_segmenter,
    refiner_for,
)

_FAKE_MODEL_BYTES = b"custback-rvm-test-model"


def fake_ort(
    available_providers,
    *,
    run_error=None,
    gpu_fail_after=None,
    profile_provider=True,
):
    """Fake onnxruntime.

    ``run_error`` (a callable) is raised from a *production* GPU session's
    ``run`` once more than ``gpu_fail_after`` production GPU inferences have run
    (the warm-up is the first), to exercise inference-time recovery without
    tripping the startup proof or warm-up. ``profile_provider`` controls whether
    the profiling proof shows a node executed on the bound provider — set False
    to simulate a registered-but-not-executing GPU provider.
    """

    mod = cast(Any, types.ModuleType("onnxruntime"))
    mod.get_available_providers = lambda: list(available_providers)
    mod.sessions = []

    class SessionOptions:
        def __init__(self):
            self.log_severity_level = 2
            self.enable_profiling = False
            self.profile_file_prefix = ""
            self._entries = {}

        def add_session_config_entry(self, key, value):
            self._entries[key] = value

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            avail = set(available_providers)
            requested = []
            for p in providers or []:
                requested.append(p[0] if isinstance(p, tuple) else p)
            self._providers = [p for p in requested if p in avail] or [
                "CPUExecutionProvider"
            ]
            self._profiling = bool(getattr(sess_options, "enable_profiling", False))
            self._gpu_runs = 0
            self.feeds = []
            mod.sessions.append(self)

        def get_providers(self):
            return self._providers

        def run(self, _outputs, feeds):
            self.feeds.append(feeds)
            active = self._providers[0]
            # Only a production (non-profiling) GPU session fails, and only after
            # the warm-up so the startup proof and warm-up both succeed first.
            if active != "CPUExecutionProvider" and not self._profiling:
                self._gpu_runs += 1
                if run_error is not None and (
                    gpu_fail_after is None or self._gpu_runs > gpu_fail_after
                ):
                    raise run_error()
            src = feeds["src"]  # (1, 3, H, W) float32 RGB in [0, 1]
            gray = src[0].mean(axis=0)
            pha = (gray > 0.5).astype(np.float32)[None, None]
            fgr = src  # "clean foreground" prediction = the source itself
            rec = [feeds[k] + 1.0 for k in ("r1i", "r2i", "r3i", "r4i")]
            return [fgr, pha, *rec]

        def end_profiling(self):
            if not self._profiling:
                return ""
            provider = (
                self._providers[0] if profile_provider else "CPUExecutionProvider"
            )
            events = [
                {"cat": "Node", "name": "MatteNode", "args": {"provider": provider}}
            ]
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, prefix="fake-ort-profile-"
            )
            with handle:
                json.dump(events, handle)
            return handle.name

    mod.SessionOptions = SessionOptions
    mod.InferenceSession = InferenceSession
    return mod


@pytest.fixture(autouse=True)
def readable_fake_model(monkeypatch):
    """Give the conventional fake path one immutable, hashable byte image."""

    original = Path.read_bytes

    def read_bytes(path):
        if path == Path("/fake/model.onnx"):
            return _FAKE_MODEL_BYTES
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)


@pytest.fixture()
def cpu_ort(monkeypatch):
    mod = fake_ort(["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


def rvm_cfg(**kw):
    # model_path set -> no download attempt
    return SegmentationConfig(backend="rvm", model_path="/fake/model.onnx", **kw)


def bright_center_frame(h=72, w=128):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    img[20:52, 44:84] = 220
    return img


class _ResetInspectingRVM(RVMSegmenter):
    """Record the concrete state immediately after each production reset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.clean_reset_snapshots: list[tuple[object, ...]] = []

    def reset_temporal_state(
        self,
        reason: TemporalResetReason,
        timestamp_ns: int | None,
    ) -> None:
        super().reset_temporal_state(reason, timestamp_ns)
        self.clean_reset_snapshots.append(
            (
                reason,
                timestamp_ns,
                self._rec,
                self._size,
                self.last_downsample_ratio,
                self.last_foreground,
                self.last_input_sequence,
                self.last_input_timestamp_ns,
            )
        )


class _InspectingMaskRefiner(MaskRefiner):
    """Record whether previous alpha existed at each refine boundary."""

    def __init__(self, cfg: SegmentationConfig) -> None:
        super().__init__(cfg)
        self.previous_at_entry: list[np.ndarray | None] = []

    def refine(
        self,
        mask: np.ndarray,
        frame_bgr: np.ndarray | None = None,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        self.previous_at_entry.append(None if self._prev is None else self._prev.copy())
        return super().refine(mask, frame_bgr, context=context)


class _FailingInspectingMaskRefiner(_InspectingMaskRefiner):
    """Advance only candidate-owned state, then fail its hidden trial."""

    def refine(
        self,
        mask: np.ndarray,
        frame_bgr: np.ndarray | None = None,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        super().refine(mask, frame_bgr, context=context)
        raise RuntimeError("candidate refinement failed")


def _temporal_app_config(*, background_mode: str = "color") -> AppConfig:
    return AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 16, "height": 16},
            "background": {"mode": background_mode},
            "segmentation": {
                "backend": "rvm",
                "model_path": "/fake/model.onnx",
                "mask_blur": 0,
                "edge_refine": False,
                "temporal_smoothing": 0.8,
            },
            "output": {"backend": "null"},
            "api": {"enabled": False},
        }
    )


def _captured_frame(
    sequence: int,
    timestamp_ns: int,
    *,
    shape: tuple[int, int] = (16, 16),
    generation: int = 1,
    geometry_generation: int = 1,
) -> CapturedFrame:
    height, width = shape
    return CapturedFrame(
        pixels=np.full((height, width, 3), 160, dtype=np.uint8),
        sequence=sequence,
        captured_at_ns=timestamp_ns,
        generation=generation,
        geometry_generation=geometry_generation,
        content_rect=(0, 0, width, height),
    )


def _install_alpha_sequence(
    segmenter: RVMSegmenter,
    values: list[float],
) -> None:
    session = segmenter._session
    original_run = session.run
    remaining = iter(values)

    def run_with_alpha(_outputs, feeds):
        result = original_run(_outputs, feeds)
        value = next(remaining)
        height, width = feeds["src"].shape[2:]
        result[1] = np.full((1, 1, height, width), value, dtype=np.float32)
        return result

    session.run = run_with_alpha


def _active_rvm_resources(
    alpha_values: list[float],
    *,
    backdrop: object | None = None,
    background_mode: str = "color",
) -> tuple[
    AppConfig,
    Pipeline,
    _Resources,
    _ResetInspectingRVM,
    _InspectingMaskRefiner,
]:
    cfg = _temporal_app_config(background_mode=background_mode)
    segmenter = _ResetInspectingRVM(cfg.segmentation)
    _install_alpha_sequence(segmenter, alpha_values)
    refiner = _InspectingMaskRefiner(cfg.segmentation)
    resources = _Resources(
        cfg,
        0,
        None,
        segmenter,
        refiner,
        backdrop,
        None,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    pipeline._segment_resource_masks(
        resources,
        _captured_frame(1, 1_000_000_000),
        privacy_safe=False,
    )
    return cfg, pipeline, resources, segmenter, refiner


def _pair_state(
    resources: _Resources,
    segmenter: _ResetInspectingRVM,
    refiner: _InspectingMaskRefiner,
) -> dict[str, Any]:
    assert segmenter._rec is not None
    assert segmenter.last_foreground is not None
    assert refiner._prev is not None
    feed_names = ("src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio")
    return {
        "rec": tuple(state.copy() for state in segmenter._rec),
        "foreground": segmenter.last_foreground.copy(),
        "previous_alpha": refiner._prev.copy(),
        "feeds": tuple(
            tuple((name, feed[name].copy()) for name in feed_names)
            for feed in segmenter._session.feeds
        ),
        "refiner_entries": tuple(
            None if previous is None else previous.copy()
            for previous in refiner.previous_at_entry
        ),
        "segmenter_scalars": (
            segmenter._size,
            segmenter.last_downsample_ratio,
            segmenter.last_input_sequence,
            segmenter.last_input_timestamp_ns,
            segmenter.temporal_reset_count,
            segmenter.last_temporal_reset_reason,
            segmenter.last_temporal_reset_timestamp_ns,
            tuple(segmenter.clean_reset_snapshots),
        ),
        "refiner_scalars": (
            refiner.last_input_sequence,
            refiner.last_input_timestamp_ns,
            refiner.temporal_reset_count,
            refiner.last_temporal_reset_reason,
            refiner.last_temporal_reset_timestamp_ns,
        ),
        "timeline": resources.segmentation_timeline.checkpoint(),
        "segmentation_generation": resources.segmentation_generation,
    }


def _assert_pair_state(
    resources: _Resources,
    segmenter: _ResetInspectingRVM,
    refiner: _InspectingMaskRefiner,
    expected: dict[str, Any],
    *,
    include_resource_state: bool = True,
) -> None:
    current = _pair_state(resources, segmenter, refiner)
    assert current["segmenter_scalars"] == expected["segmenter_scalars"]
    assert current["refiner_scalars"] == expected["refiner_scalars"]
    if include_resource_state:
        assert current["timeline"] == expected["timeline"]
        assert current["segmentation_generation"] == expected["segmentation_generation"]
    for actual, wanted in zip(current["rec"], expected["rec"], strict=True):
        np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_equal(current["foreground"], expected["foreground"])
    np.testing.assert_array_equal(
        current["previous_alpha"],
        expected["previous_alpha"],
    )
    for actual_feed, wanted_feed in zip(
        current["feeds"],
        expected["feeds"],
        strict=True,
    ):
        assert tuple(name for name, _value in actual_feed) == tuple(
            name for name, _value in wanted_feed
        )
        for (_actual_name, actual), (_wanted_name, wanted) in zip(
            actual_feed,
            wanted_feed,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, wanted)
    for actual, wanted in zip(
        current["refiner_entries"],
        expected["refiner_entries"],
        strict=True,
    ):
        if wanted is None:
            assert actual is None
        else:
            assert actual is not None
            np.testing.assert_array_equal(actual, wanted)


def _prepare_concrete_rvm_candidate(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: Pipeline,
    current: AppConfig,
    candidate: AppConfig,
    alpha_values: list[float],
    *,
    refiner_type: type[_InspectingMaskRefiner] = _InspectingMaskRefiner,
):
    def create_candidate(cfg, *, acceleration=None, **_kwargs):
        return _ResetInspectingRVM(cfg, acceleration=acceleration)

    refiner = refiner_type(candidate.segmentation)
    monkeypatch.setattr(pipeline_mod, "create_segmenter", create_candidate)
    monkeypatch.setattr(
        pipeline_mod,
        "refiner_for",
        lambda _cfg, _segmenter: refiner,
    )
    activation = pipeline._prepare_activation_off_lane(current, candidate)
    segmenter = cast(_ResetInspectingRVM, activation.segmenter)
    _install_alpha_sequence(segmenter, alpha_values)
    assert activation.refiner is refiner
    return activation, segmenter, refiner


def _track_candidate_close(
    segmenter: _ResetInspectingRVM,
    refiner: _InspectingMaskRefiner,
) -> tuple[threading.Event, dict[str, Any]]:
    closed = threading.Event()
    observed: dict[str, Any] = {}
    original_close = segmenter.close

    def close() -> None:
        try:
            observed["feed_count"] = len(segmenter._session.feeds)
            observed["rec"] = (
                None
                if segmenter._rec is None
                else tuple(state.copy() for state in segmenter._rec)
            )
            observed["previous_alpha"] = (
                None if refiner._prev is None else refiner._prev.copy()
            )
            observed["reset_reasons"] = tuple(
                snapshot[0] for snapshot in segmenter.clean_reset_snapshots
            )
            original_close()
        finally:
            closed.set()

    setattr(segmenter, "close", close)
    return closed, observed


def test_segment_returns_mask_and_clean_foreground(cpu_ort):
    seg = RVMSegmenter(rvm_cfg())
    frame = bright_center_frame()
    frame_before = frame.copy()
    mask = seg.segment(frame)
    expected_src = frame[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    expected_mask = (expected_src[0].mean(axis=0) > 0.5).astype(np.float32)
    assert mask.shape == (72, 128)
    assert mask.dtype == np.float32
    assert mask[36, 64] == 1.0  # bright center = person
    assert mask[2, 2] == 0.0
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(frame, frame_before)
    np.testing.assert_array_equal(seg._session.feeds[-1]["src"], expected_src)
    assert seg.last_foreground is not None
    assert seg.last_foreground.shape == frame.shape
    assert seg.last_foreground.dtype == np.uint8
    np.testing.assert_array_equal(seg.last_foreground, frame)
    assert not np.shares_memory(
        seg.last_foreground,
        seg._session.feeds[-1]["src"],
    )


def test_feed_conversion_is_exact_contiguous_and_fresh(cpu_ort):
    seg = RVMSegmenter(rvm_cfg())
    values = np.arange(17 * 23 * 3, dtype=np.uint16).reshape(17, 23, 3)
    first_frame = np.asarray(values % 256, dtype=np.uint8)
    second_frame = np.ascontiguousarray(255 - first_frame)
    seg._start_temporal_epoch(first_frame.shape[:2])

    first = seg._feeds(first_frame, 0.5)["src"]
    first_before = first.copy()
    second = seg._feeds(second_frame, 0.5)["src"]
    expected_first = (
        first_frame[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    )
    expected_second = (
        second_frame[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    )

    assert first.shape == (1, 3, 17, 23)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert first.flags.owndata
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(second, expected_second)
    np.testing.assert_array_equal(first, first_before)
    assert not np.shares_memory(first, second)


def test_alpha_output_is_detached_from_aliased_feed_and_recurrent_state(cpu_ort):
    seg = RVMSegmenter(rvm_cfg())
    observed = {}

    def run_with_aliased_alpha(_outputs, feeds):
        src = feeds["src"]
        observed["src"] = src
        alpha = src[:, :1]
        recurrent = [
            alpha,
            *(feeds[name] + 1.0 for name in ("r2i", "r3i", "r4i")),
        ]
        return [src, alpha, *recurrent]

    seg._session.run = run_with_aliased_alpha
    mask = seg.segment(np.full((8, 12, 3), 128, dtype=np.uint8))
    source = observed["src"]
    assert seg._rec is not None
    recurrent = seg._rec[0]

    assert mask.flags.c_contiguous
    assert mask.flags.owndata
    assert np.shares_memory(source, recurrent)
    assert not np.shares_memory(mask, source)
    assert not np.shares_memory(mask, recurrent)
    np.testing.assert_array_equal(mask, source[0, 0])
    retained_source = source.copy()
    retained = recurrent.copy()
    mask.fill(0.0)
    np.testing.assert_array_equal(source, retained_source)
    np.testing.assert_array_equal(recurrent, retained)


def test_success_publishes_typed_content_free_rvm_telemetry(
    cpu_ort,
    monkeypatch,
):
    seg = RVMSegmenter(rvm_cfg())
    initial = seg.rvm_telemetry_snapshot()
    assert isinstance(initial, RVMTelemetry)
    assert initial.input_frame_shape is None
    assert initial.resolved_downsample_ratio is None
    assert initial.configured_downsample_mode == "auto"
    assert initial.configured_downsample_ratio == 0.0

    ticks = iter(
        [
            0,
            1_000_000,
            2_000_000,
            3_000_000,
            4_000_000,
            5_000_000,
        ]
    )
    monkeypatch.setattr(segmentation_mod.time, "monotonic_ns", lambda: next(ticks))
    frame = bright_center_frame()
    alpha = seg.segment(frame)

    snapshot = seg.rvm_telemetry_snapshot()
    assert isinstance(snapshot, RVMTelemetry)
    assert snapshot.input_frame_shape == frame.shape[:2]
    assert snapshot.output_alpha_shape == alpha.shape
    assert snapshot.output_foreground_shape == frame.shape
    assert snapshot.configured_downsample_mode == "auto"
    assert snapshot.configured_downsample_ratio == 0.0
    assert snapshot.resolved_downsample_ratio == pytest.approx(1.0)
    assert snapshot.preprocess_ms == 1.0
    assert snapshot.session_run_ms == 1.0
    assert snapshot.postprocess_ms == 1.0
    assert snapshot.model_builtin is False
    assert snapshot.model_identity == "model.onnx"
    assert snapshot.model_sha256 == hashlib.sha256(_FAKE_MODEL_BYTES).hexdigest()
    assert snapshot.model_bytes == len(_FAKE_MODEL_BYTES)
    assert snapshot.acceleration_state
    assert snapshot.acceleration_active_provider == "cpu"
    assert snapshot.acceleration_fallback_active is False
    assert snapshot.acceleration_fallback_count == 0


def test_builtin_model_telemetry_uses_pinned_identity(
    cpu_ort,
    monkeypatch,
    tmp_path,
):
    payload = b"managed-rvm-model"
    model = tmp_path / "rvm_mobilenetv3_fp32.onnx"
    model.write_bytes(payload)
    spec = segmentation_mod.ModelSpec(
        backend="rvm",
        url="https://example.invalid/rvm.onnx",
        filename=model.name,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(segmentation_mod, "RVM_MODEL", spec)
    monkeypatch.setattr(
        segmentation_mod,
        "acquire_model",
        lambda *_args, **_kwargs: model,
    )
    seg = RVMSegmenter(SegmentationConfig(backend="rvm"))

    snapshot = seg.rvm_telemetry_snapshot()
    assert snapshot.model_builtin is True
    assert snapshot.model_identity == spec.filename
    assert snapshot.model_sha256 == spec.sha256
    assert snapshot.model_bytes == spec.size
    assert snapshot.input_frame_shape is None
    assert snapshot.resolved_downsample_ratio is None


def test_custom_model_identity_is_bound_to_the_bytes_loaded_by_every_session(
    cpu_ort,
    tmp_path,
):
    model = tmp_path / "alternative.onnx"
    original = b"immutable-rvm-model-snapshot"
    model.write_bytes(original)
    seg = RVMSegmenter(
        SegmentationConfig(backend="rvm", model_path=str(model)),
        acceleration=AccelerationConfig(mode="cpu"),
    )

    snapshot = seg.rvm_telemetry_snapshot()
    assert snapshot.model_identity == "alternative.onnx"
    assert snapshot.model_sha256 == hashlib.sha256(original).hexdigest()
    assert snapshot.model_bytes == len(original)
    assert seg._session.path == original

    model.write_bytes(b"replacement")
    rebuilt = seg._build_cpu_session()
    assert rebuilt.path == original
    seg.close()
    assert seg._model_source == b""


def test_unreadable_model_fails_closed_before_session_construction(
    cpu_ort,
    tmp_path,
):
    missing = tmp_path / "missing.onnx"

    with pytest.raises(RuntimeError, match="model bytes could not be read"):
        RVMSegmenter(
            SegmentationConfig(backend="rvm", model_path=str(missing)),
            acceleration=AccelerationConfig(mode="cpu"),
        )

    assert cpu_ort.sessions == []


def test_concrete_rvm_telemetry_round_trips_through_private_replay_bundle(
    cpu_ort,
    tmp_path,
):
    cfg = _temporal_app_config()
    segmenter = RVMSegmenter(cfg.segmentation)
    refiner = refiner_for(cfg.segmentation, segmenter)
    captured = _captured_frame(1, 1_000_000_000)

    class Backdrop:
        @staticmethod
        def frame(width, height):
            return np.full((height, width, 3), 20, dtype=np.uint8)

        @staticmethod
        def close():
            return None

    resources = _Resources(
        cfg,
        0,
        cast(Any, object()),
        segmenter,
        refiner,
        Backdrop(),
        None,
    )
    recorder = MatteDiagnosticRecorder(
        tmp_path / "rvm-replay",
        max_bytes=4_000_000,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub(), matte_recorder=recorder)
    evidence = pipeline._new_matte_evidence(resources, captured)
    assert evidence is not None
    try:
        rendered, reason = pipeline._local_composite(
            resources,
            captured.pixels,
            captured=captured,
            privacy_safe=False,
            matte_evidence=evidence,
        )
        assert reason == ""
        assert recorder.submit(evidence, rendered)
    finally:
        recorder.close()

    bundle = MatteReplayBundle(tmp_path / "rvm-replay")
    frame = bundle.frames[0]
    telemetry = frame["effective_controls"]["rvm_telemetry"]
    assert telemetry["applicable"] is True
    assert telemetry["input_frame_shape"] == [16, 16]
    assert telemetry["output_alpha_shape"] == [16, 16]
    assert telemetry["output_foreground_shape"] == [16, 16, 3]
    assert telemetry["resolved_downsample_ratio"] == 1.0
    assert telemetry["model_builtin"] is False
    assert telemetry["model_identity"] == "model.onnx"
    assert telemetry["model_sha256"] == hashlib.sha256(_FAKE_MODEL_BYTES).hexdigest()
    assert telemetry["model_bytes"] == len(_FAKE_MODEL_BYTES)
    assert telemetry["acceleration_active_provider"] == "cpu"
    assert telemetry["acceleration_fallback_active"] is False
    assert telemetry["acceleration_fallback_count"] == 0
    assert frame["timings_ms"]["rvm_preprocess_ms"] == telemetry["preprocess_ms"]
    assert frame["timings_ms"]["rvm_session_run_ms"] == telemetry["session_run_ms"]
    assert frame["timings_ms"]["rvm_postprocess_ms"] == telemetry["postprocess_ms"]


def test_concrete_rvm_evidence_separates_native_alpha_from_post_shift_policy(
    cpu_ort,
    tmp_path,
):
    cfg = _temporal_app_config().patched(
        {
            "segmentation": {
                "rvm_downsample": 0.5,
                "mask_shift": 1,
            }
        }
    )
    segmenter = RVMSegmenter(cfg.segmentation)
    refiner = refiner_for(cfg.segmentation, segmenter)
    captured = _captured_frame(7, 1_750_000_000)
    native_alpha = np.zeros(captured.pixels.shape[:2], dtype=np.float32)
    native_alpha[5:11, 6:10] = 0.25
    native_alpha[7:9, 7:9] = 0.75
    original_run = segmenter._session.run

    def run_with_native_alpha(outputs, feeds):
        result = original_run(outputs, feeds)
        result[1] = native_alpha[None, None].copy()
        return result

    segmenter._session.run = run_with_native_alpha

    class Backdrop:
        @staticmethod
        def frame(width, height):
            return np.full((height, width, 3), 20, dtype=np.uint8)

        @staticmethod
        def close():
            return None

    resources = _Resources(
        cfg,
        0,
        cast(Any, object()),
        segmenter,
        refiner,
        Backdrop(),
        None,
    )
    recorder = MatteDiagnosticRecorder(
        tmp_path / "rvm-alpha-attribution",
        max_bytes=4_000_000,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub(), matte_recorder=recorder)
    evidence = pipeline._new_matte_evidence(resources, captured)
    assert evidence is not None
    try:
        rendered, reason = pipeline._local_composite(
            resources,
            captured.pixels,
            captured=captured,
            privacy_safe=False,
            matte_evidence=evidence,
        )
    finally:
        recorder.close()
        resources.close()

    assert reason == ""
    assert rendered.shape == captured.pixels.shape
    assert evidence.metadata.capture_sequence == captured.sequence
    assert evidence.metadata.capture_monotonic_ns == captured.captured_at_ns
    assert evidence.raw_mask is not None
    assert evidence.refined_mask is not None
    expected_refined = segmentation_mod.cv2.dilate(
        native_alpha,
        segmentation_mod.cv2.getStructuringElement(
            segmentation_mod.cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
    )
    np.testing.assert_array_equal(evidence.raw_mask, native_alpha)
    np.testing.assert_array_equal(evidence.refined_mask, expected_refined)
    assert not np.array_equal(evidence.raw_mask, evidence.refined_mask)

    effective_controls = evidence.effective_controls
    assert effective_controls["rvm_downsample_ratio"] == 0.5
    assert effective_controls["mask_shift"] == 1
    telemetry = effective_controls["rvm_telemetry"]
    assert isinstance(telemetry, dict)
    assert telemetry["resolved_downsample_ratio"] == 0.5
    matte_policy = effective_controls["matte_policy"]
    assert isinstance(matte_policy, dict)
    effective_policy = matte_policy["effective"]
    assert isinstance(effective_policy, dict)
    assert effective_policy["raw_alpha_mode"] == "native_soft_alpha"
    assert effective_policy["rvm_downsample_ratio"] == 0.5
    assert effective_policy["mask_shift"] == 1


def test_device_cpu_and_cuda(cpu_ort, monkeypatch):
    assert RVMSegmenter(rvm_cfg()).device == "cpu"
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    assert RVMSegmenter(rvm_cfg()).device == "cuda"


def test_recurrent_state_is_fed_back_and_reset_on_resize(cpu_ort):
    seg = RVMSegmenter(rvm_cfg())
    seg.segment(bright_center_frame())
    seg.segment(bright_center_frame())
    feeds = seg._session.feeds
    assert len({id(feeds[0][name]) for name in ("r1i", "r2i", "r3i", "r4i")}) == 4
    assert float(feeds[0]["r1i"].ravel()[0]) == 0.0
    assert float(feeds[1]["r1i"].ravel()[0]) == 1.0  # previous output reused
    # resolution change -> state must reset (it is resolution-bound)
    seg.segment(bright_center_frame(h=36, w=64))
    assert float(feeds[2]["r1i"].ravel()[0]) == 0.0


def test_downsample_ratio_auto_and_explicit(cpu_ort):
    seg = RVMSegmenter(rvm_cfg())
    seg.segment(np.zeros((720, 1280, 3), np.uint8))
    auto = float(seg._session.feeds[0]["downsample_ratio"][0])
    assert auto == pytest.approx(512 / 1280)
    auto_telemetry = seg.rvm_telemetry_snapshot()
    assert auto_telemetry.configured_downsample_mode == "auto"
    assert auto_telemetry.configured_downsample_ratio == 0.0
    assert auto_telemetry.resolved_downsample_ratio == pytest.approx(512 / 1280)

    seg = RVMSegmenter(rvm_cfg(rvm_downsample=0.5))
    seg.segment(np.zeros((720, 1280, 3), np.uint8))
    assert float(seg._session.feeds[0]["downsample_ratio"][0]) == pytest.approx(0.5)
    explicit_telemetry = seg.rvm_telemetry_snapshot()
    assert explicit_telemetry.configured_downsample_mode == "explicit"
    assert explicit_telemetry.configured_downsample_ratio == 0.5
    assert explicit_telemetry.resolved_downsample_ratio == 0.5


def test_auto_backend_prefers_rvm_when_onnxruntime_present(cpu_ort):
    cfg = SegmentationConfig(backend="auto", model_path="/fake/model.onnx")
    assert isinstance(create_segmenter(cfg), RVMSegmenter)


def test_auto_backend_without_onnxruntime_falls_back(monkeypatch):
    # None in sys.modules makes `import x` raise ImportError; block both ML
    # backends so the test never hits the network for model downloads.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    seg = create_segmenter(SegmentationConfig(backend="auto"))
    assert isinstance(seg, HeuristicSegmenter)


def test_explicit_rvm_backend_raises_without_onnxruntime(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError):
        create_segmenter(SegmentationConfig(backend="rvm"))


def test_refiner_skips_redundant_work_for_matting_backends(cpu_ort, monkeypatch):
    cfg = rvm_cfg(mask_blur=7, edge_refine=True, temporal_smoothing=0.5, mask_shift=-1)
    matte_refiner = refiner_for(cfg, RVMSegmenter(cfg))
    assert matte_refiner.cfg.mask_blur == 0
    assert matte_refiner.cfg.edge_refine is False
    assert matte_refiner.cfg.temporal_smoothing == 0.0
    assert matte_refiner.cfg.mask_shift == -1  # user's halo control survives

    stable_cfg = rvm_cfg(
        edge_refine=True,
        spatial_edge_refinement={"mode": "stable_guided"},
    )
    stable_matte_refiner = refiner_for(stable_cfg, RVMSegmenter(stable_cfg))
    assert stable_matte_refiner.cfg.edge_refine is False
    assert stable_matte_refiner.cfg.spatial_edge_refinement.mode == "stable_guided"
    monkeypatch.setattr(
        segmentation_mod,
        "_stable_guided_edge_refine",
        lambda *_args, **_kwargs: pytest.fail(
            "RVM must bypass generic stable spatial refinement"
        ),
    )
    stable_matte_refiner.refine(
        np.full((12, 16), 0.5, dtype=np.float32),
        np.zeros((12, 16, 3), dtype=np.uint8),
    )

    plain_refiner = refiner_for(cfg, HeuristicSegmenter(cfg))
    assert isinstance(plain_refiner, MaskRefiner)
    assert plain_refiner.cfg.mask_blur == 7

    explicit_motion_cfg = rvm_cfg(
        mask_blur=7,
        edge_refine=True,
        temporal_smoothing=0.5,
        boundary_stabilization={
            "mode": "motion_aware",
            "time_constant_s": 0.05,
            "max_motion_px_per_s": 360.0,
        },
    )
    motion_refiner = refiner_for(
        explicit_motion_cfg,
        RVMSegmenter(explicit_motion_cfg),
    )
    assert motion_refiner.cfg.boundary_stabilization.mode == "motion_aware"
    assert motion_refiner.cfg.boundary_stabilization.time_constant_s == 0.05
    assert motion_refiner.cfg.boundary_stabilization.max_motion_px_per_s == 360.0
    assert motion_refiner.cfg.temporal_smoothing == 0.0


# -- acceleration policy (WIN-4.x) --------------------------------------------

from custback.acceleration import GpuRequiredError  # noqa: E402
from custback.config import AccelerationConfig  # noqa: E402


def gpu_ort(monkeypatch, **kwargs):
    mod = fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"], **kwargs)
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


def test_cpu_mode_never_touches_gpu(monkeypatch):
    gpu_ort(monkeypatch)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="cpu"))
    assert seg.device == "cpu"
    status = seg.accel.status()
    assert status.active_provider == "cpu"
    assert status.fallback_active is False
    # The production session was asked for CPU only.
    assert seg._session.get_providers() == ["CPUExecutionProvider"]


def test_auto_uses_proven_gpu(monkeypatch):
    gpu_ort(monkeypatch)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig())
    assert seg.device == "cuda"
    assert seg.accel.status().active_provider == "cuda"
    assert seg.accel.on_gpu is True


def test_auto_gpu_registered_but_unprovable_falls_back(monkeypatch):
    gpu_ort(monkeypatch, profile_provider=False)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig())
    assert seg.device == "cpu"
    status = seg.accel.status()
    assert status.fallback_active is True
    assert status.fallback_count == 1


def test_auto_gpu_proof_success_but_production_construction_failure_falls_back(
    monkeypatch,
):
    mod = gpu_ort(monkeypatch)
    inference_session = mod.InferenceSession

    def fail_production_gpu(path, sess_options=None, providers=None):
        requested = [
            item[0] if isinstance(item, tuple) else item for item in providers or []
        ]
        if (
            requested
            and requested[0] == "CUDAExecutionProvider"
            and not bool(getattr(sess_options, "enable_profiling", False))
        ):
            raise RuntimeError("production provider construction failed")
        return inference_session(path, sess_options=sess_options, providers=providers)

    mod.InferenceSession = fail_production_gpu
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig())

    assert seg.device == "cpu"
    status = seg.accel.status()
    assert status.fallback_active is True
    assert status.fallback_count == 1


def test_gpu_required_without_gpu_raises(cpu_ort):
    with pytest.raises(GpuRequiredError):
        RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="gpu_required"))


def test_gpu_required_registered_but_unprovable_raises(monkeypatch):
    gpu_ort(monkeypatch, profile_provider=False)
    with pytest.raises(GpuRequiredError):
        RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="gpu_required"))


def test_gpu_required_with_proven_gpu_succeeds(monkeypatch):
    gpu_ort(monkeypatch)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="gpu_required"))
    assert seg.device == "cuda"


def test_create_segmenter_threads_acceleration(monkeypatch):
    gpu_ort(monkeypatch)
    seg = create_segmenter(
        SegmentationConfig(backend="rvm", model_path="/fake/model.onnx"),
        acceleration=AccelerationConfig(mode="cpu"),
    )
    assert isinstance(seg, RVMSegmenter)
    assert seg.device == "cpu"


def test_gpu_required_not_swallowed_by_auto_backend(monkeypatch):
    # backend=auto would normally fall back to another backend on RVM failure,
    # but gpu_required is an explicit demand that must propagate.
    gpu_ort(monkeypatch, profile_provider=False)
    with pytest.raises(GpuRequiredError):
        create_segmenter(
            SegmentationConfig(backend="auto", model_path="/fake/model.onnx"),
            acceleration=AccelerationConfig(mode="gpu_required"),
        )


def test_inference_time_recovery_rebuilds_cpu_and_retries_once(monkeypatch):
    mod = gpu_ort(monkeypatch, run_error=RuntimeError, gpu_fail_after=1)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig())
    assert seg.device == "cuda"  # warm-up (1st production GPU run) succeeded
    frame = bright_center_frame()
    failed_gpu_session = seg._session
    context = SegmentationFrameContext(
        sequence=1,
        timestamp_ns=1_000_000_000,
        generation=1,
        geometry_generation=1,
        shape=frame.shape[:2],
    )
    # First real frame fails on GPU, recovers to CPU, retries once, succeeds.
    mask = seg.segment(frame, context=context)
    assert mask.shape == frame.shape[:2]
    assert seg.device == "cpu"
    status = seg.accel.status()
    assert status.fallback_active is True
    assert status.fallback_count == 1
    assert seg.accel.on_gpu is False
    assert seg.temporal_reset_count == 1
    assert seg.last_temporal_reset_reason is TemporalResetReason.BACKEND_RECOVERY
    assert seg.last_temporal_reset_timestamp_ns == context.timestamp_ns
    assert seg.last_downsample_ratio == pytest.approx(1.0)
    telemetry = seg.rvm_telemetry_snapshot()
    assert telemetry.acceleration_active_provider == "cpu"
    assert telemetry.acceleration_fallback_active is True
    assert telemetry.acceleration_fallback_count == 1
    assert telemetry.resolved_downsample_ratio == pytest.approx(1.0)
    assert telemetry.preprocess_ms is not None and telemetry.preprocess_ms >= 0.0
    assert telemetry.session_run_ms is not None and telemetry.session_run_ms >= 0.0
    assert telemetry.postprocess_ms is not None and telemetry.postprocess_ms >= 0.0
    assert len(failed_gpu_session.feeds) == 2  # warm-up plus the failed real input
    # Recovery builds one CPU session and retries exactly once from zero state.
    assert len(seg._session.feeds) == 1
    for name in ("r1i", "r2i", "r3i", "r4i"):
        assert not np.any(seg._session.feeds[0][name])

    # The next frame stays on CPU and consumes the retry's recurrent outputs.
    seg.segment(
        frame,
        context=SegmentationFrameContext(
            sequence=2,
            timestamp_ns=1_000_000_001,
            generation=1,
            geometry_generation=1,
            shape=frame.shape[:2],
        ),
    )
    assert seg.device == "cpu"
    assert len(seg._session.feeds) == 2
    for name in ("r1i", "r2i", "r3i", "r4i"):
        assert np.all(seg._session.feeds[1][name] == 1.0)
    assert mod.sessions
    assert all(session.path == _FAKE_MODEL_BYTES for session in mod.sessions)


def test_gpu_required_inference_failure_never_degrades_to_cpu(monkeypatch):
    gpu_ort(monkeypatch, run_error=RuntimeError, gpu_fail_after=1)
    seg = RVMSegmenter(
        rvm_cfg(),
        acceleration=AccelerationConfig(mode="gpu_required"),
    )
    failed_gpu_session = seg._session
    initial_telemetry = seg.rvm_telemetry_snapshot()

    with pytest.raises(
        GpuRequiredError,
        match="gpu_required RVM inference failed on the active accelerator",
    ):
        seg.segment(bright_center_frame())

    assert seg._session is failed_gpu_session
    assert seg.device == "cuda"
    status = seg.accel.status()
    assert status.active_provider == "cuda"
    assert status.fallback_active is False
    assert status.fallback_count == 0
    assert seg.temporal_reset_count == 0
    assert seg.last_downsample_ratio is None
    assert seg.last_foreground is None
    assert seg.rvm_telemetry_snapshot() == initial_telemetry


def test_cpu_inference_failure_is_not_retried(cpu_ort):
    # A pure-CPU session that fails is a real error, not a GPU fallback: it must
    # surface rather than loop rebuilding CPU sessions.
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="cpu"))
    assert seg.device == "cpu"
    initial_telemetry = seg.rvm_telemetry_snapshot()

    def boom(*_a, **_k):
        raise RuntimeError("cpu inference error")

    seg._session.run = boom
    with pytest.raises(RuntimeError, match="cpu inference error"):
        seg.segment(bright_center_frame())
    assert seg.last_downsample_ratio is None
    assert seg.last_foreground is None
    assert seg.rvm_telemetry_snapshot() == initial_telemetry


def test_failed_cpu_inference_preserves_prior_successful_telemetry(cpu_ort):
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="cpu"))
    frame = bright_center_frame()
    seg.segment(frame)
    previous_telemetry = seg.rvm_telemetry_snapshot()
    previous_foreground = seg.last_foreground

    def boom(*_a, **_k):
        raise RuntimeError("later cpu inference error")

    seg._session.run = boom
    with pytest.raises(RuntimeError, match="later cpu inference error"):
        seg.segment(frame)

    assert seg.last_downsample_ratio == previous_telemetry.resolved_downsample_ratio
    assert seg.last_foreground is previous_foreground
    assert seg.rvm_telemetry_snapshot() == previous_telemetry


def test_malformed_output_does_not_consume_context_and_exact_retry_succeeds(cpu_ort):
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="cpu"))
    frame = bright_center_frame()
    first_context = SegmentationFrameContext(
        sequence=10,
        timestamp_ns=1_000_000_000,
        generation=1,
        geometry_generation=1,
        shape=frame.shape[:2],
    )
    retry_context = SegmentationFrameContext(
        sequence=11,
        timestamp_ns=1_033_333_333,
        generation=1,
        geometry_generation=1,
        shape=frame.shape[:2],
    )
    seg.segment(frame, context=first_context)
    previous_recurrent = seg._rec
    previous_foreground = seg.last_foreground
    previous_ratio = seg.last_downsample_ratio
    previous_telemetry = seg.rvm_telemetry_snapshot()
    valid_run = seg._session.run

    def malformed(_outputs, feeds):
        height, width = feeds["src"].shape[2:]
        return [
            np.zeros((1, 3, height, width), dtype=np.float32),
            np.full((1, 1, height, width), np.nan, dtype=np.float32),
            *(feeds[name] for name in ("r1i", "r2i", "r3i", "r4i")),
        ]

    seg._session.run = malformed
    with pytest.raises(ValueError, match="finite"):
        seg.segment(frame, context=retry_context)

    assert seg.last_input_sequence == first_context.sequence
    assert seg.last_input_timestamp_ns == first_context.timestamp_ns
    assert seg._rec is previous_recurrent
    assert seg.last_foreground is previous_foreground
    assert seg.last_downsample_ratio == previous_ratio
    assert seg.rvm_telemetry_snapshot() == previous_telemetry

    seg._session.run = valid_run
    seg.segment(frame, context=retry_context)
    assert seg.last_input_sequence == retry_context.sequence
    assert seg.last_input_timestamp_ns == retry_context.timestamp_ns


def test_invalid_rvm_outputs_do_not_publish_ratio_or_frame_telemetry(cpu_ort):
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="cpu"))
    initial_telemetry = seg.rvm_telemetry_snapshot()

    def malformed(_outputs, feeds):
        return [
            np.zeros((1, 3, 1, 1), dtype=np.float32),
            np.zeros((1, 1, 1, 1), dtype=np.float32),
            *(feeds[name] for name in ("r1i", "r2i", "r3i", "r4i")),
        ]

    seg._session.run = malformed
    with pytest.raises(ValueError, match="invalid foreground"):
        seg.segment(bright_center_frame())

    assert seg.last_downsample_ratio is None
    assert seg.last_foreground is None
    assert seg.rvm_telemetry_snapshot() == initial_telemetry


@pytest.mark.parametrize(
    ("invalid_part", "message"),
    [
        ("output_count", "invalid output set"),
        ("integer_foreground", "invalid foreground"),
        ("nonfinite_foreground", "invalid foreground"),
        ("integer_alpha", "invalid alpha"),
        ("nonfinite_alpha", "finite"),
        ("out_of_range_alpha", "finite"),
        ("nonfinite_recurrent", "invalid recurrent state"),
        ("integer_recurrent", "invalid recurrent state"),
        ("rank3_recurrent", "invalid recurrent state"),
        ("wrong_batch_recurrent", "invalid recurrent state"),
        ("empty_recurrent", "invalid recurrent state"),
    ],
)
def test_nonfloating_or_malformed_rvm_outputs_never_publish_success(
    cpu_ort,
    invalid_part,
    message,
):
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="cpu"))
    initial_telemetry = seg.rvm_telemetry_snapshot()

    def malformed(_outputs, feeds):
        frame = feeds["src"]
        height, width = frame.shape[2:]
        foreground = np.zeros((1, 3, height, width), dtype=np.float32)
        alpha = np.zeros((1, 1, height, width), dtype=np.float32)
        recurrent = [feeds[name].copy() for name in ("r1i", "r2i", "r3i", "r4i")]
        if invalid_part == "output_count":
            return [foreground, alpha, *recurrent[:3]]
        if invalid_part == "integer_foreground":
            foreground = foreground.astype(np.uint8)
        elif invalid_part == "nonfinite_foreground":
            foreground[0, 0, 0, 0] = np.nan
        elif invalid_part == "integer_alpha":
            alpha = alpha.astype(np.uint8)
        elif invalid_part == "nonfinite_alpha":
            alpha[0, 0, 0, 0] = np.nan
        elif invalid_part == "out_of_range_alpha":
            alpha[0, 0, 0, 0] = 1.1
        elif invalid_part == "nonfinite_recurrent":
            recurrent[0][0, 0, 0, 0] = np.nan
        elif invalid_part == "integer_recurrent":
            recurrent[0] = recurrent[0].astype(np.int32)
        elif invalid_part == "rank3_recurrent":
            recurrent[0] = recurrent[0][0]
        elif invalid_part == "wrong_batch_recurrent":
            recurrent[0] = np.zeros((2, 1, 1, 1), dtype=np.float32)
        else:
            recurrent[0] = np.zeros((1, 1, 0, 1), dtype=np.float32)
        return [foreground, alpha, *recurrent]

    seg._session.run = malformed
    with pytest.raises(ValueError, match=message):
        seg.segment(bright_center_frame())

    assert seg.last_downsample_ratio is None
    assert seg.last_foreground is None
    assert seg.rvm_telemetry_snapshot() == initial_telemetry


@pytest.mark.parametrize(
    "reason",
    list(TemporalResetReason),
    ids=lambda reason: reason.value,
)
def test_every_reset_reason_clears_all_rvm_temporal_state(
    cpu_ort,
    reason: TemporalResetReason,
):
    cfg = _temporal_app_config()
    segmenter = _ResetInspectingRVM(cfg.segmentation)
    frame = np.full((16, 16, 3), 160, dtype=np.uint8)
    context = SegmentationFrameContext(
        sequence=7,
        timestamp_ns=1_000_000_000,
        generation=3,
        geometry_generation=4,
        shape=frame.shape[:2],
    )

    segmenter.segment(frame, context=context)
    assert segmenter._rec is not None
    assert len(segmenter._rec) == 4
    assert all(np.all(state == 1.0) for state in segmenter._rec)
    assert segmenter._size == frame.shape[:2]
    assert segmenter.last_downsample_ratio == pytest.approx(1.0)
    assert segmenter.last_foreground is not None
    assert segmenter.last_input_sequence == context.sequence
    assert segmenter.last_input_timestamp_ns == context.timestamp_ns
    populated_telemetry = segmenter.rvm_telemetry_snapshot()
    assert populated_telemetry.input_frame_shape == frame.shape[:2]
    assert populated_telemetry.resolved_downsample_ratio == pytest.approx(1.0)

    segmenter.reset_temporal_state(reason, 2_000_000_000)

    assert segmenter.clean_reset_snapshots[-1] == (
        reason,
        2_000_000_000,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert segmenter._rec is None
    assert segmenter._size is None
    assert segmenter.last_downsample_ratio is None
    assert segmenter.last_foreground is None
    assert segmenter.last_input_sequence is None
    assert segmenter.last_input_timestamp_ns is None
    reset_telemetry = segmenter.rvm_telemetry_snapshot()
    assert reset_telemetry.input_frame_shape is None
    assert reset_telemetry.output_alpha_shape is None
    assert reset_telemetry.output_foreground_shape is None
    assert reset_telemetry.resolved_downsample_ratio is None
    assert reset_telemetry.preprocess_ms is None
    assert reset_telemetry.session_run_ms is None
    assert reset_telemetry.postprocess_ms is None
    assert reset_telemetry.model_identity == populated_telemetry.model_identity
    assert segmenter.temporal_reset_count == 1
    assert segmenter.last_temporal_reset_reason is reason
    assert segmenter.last_temporal_reset_timestamp_ns == 2_000_000_000


def test_close_clears_rvm_frame_telemetry(cpu_ort):
    seg = RVMSegmenter(rvm_cfg())
    seg.segment(bright_center_frame())
    assert seg.rvm_telemetry_snapshot().input_frame_shape is not None

    seg.close()

    snapshot = seg.rvm_telemetry_snapshot()
    assert snapshot.input_frame_shape is None
    assert snapshot.output_alpha_shape is None
    assert snapshot.output_foreground_shape is None
    assert snapshot.resolved_downsample_ratio is None
    assert snapshot.preprocess_ms is None
    assert snapshot.session_run_ms is None
    assert snapshot.postprocess_ms is None
    assert snapshot.model_identity == "model.onnx"


@pytest.mark.parametrize(
    (
        "second_timestamp_ns",
        "second_generation",
        "second_geometry_generation",
        "second_shape",
        "expected_reason",
    ),
    [
        pytest.param(
            1_100_000_000,
            2,
            1,
            (16, 16),
            TemporalResetReason.CAPTURE_GENERATION,
            id="same-shape-capture-generation",
        ),
        pytest.param(
            1_100_000_000,
            1,
            2,
            (16, 16),
            TemporalResetReason.GEOMETRY,
            id="geometry-generation",
        ),
        pytest.param(
            1_100_000_000,
            1,
            1,
            (12, 20),
            TemporalResetReason.GEOMETRY,
            id="pixel-shape",
        ),
        pytest.param(
            999_999_999,
            1,
            1,
            (16, 16),
            TemporalResetReason.NON_MONOTONIC_TIMESTAMP,
            id="decreasing-time",
        ),
        pytest.param(
            1_000_000_000 + SEGMENTATION_TIMESTAMP_GAP_RESET_NS + 1,
            1,
            1,
            (16, 16),
            TemporalResetReason.TIMESTAMP_GAP,
            id="qualified-long-gap",
        ),
    ],
)
def test_discontinuity_resets_rvm_and_refiner_before_exact_boundary_frame(
    cpu_ort,
    second_timestamp_ns: int,
    second_generation: int,
    second_geometry_generation: int,
    second_shape: tuple[int, int],
    expected_reason: TemporalResetReason,
):
    cfg = _temporal_app_config()
    segmenter = _ResetInspectingRVM(cfg.segmentation)
    _install_alpha_sequence(segmenter, [0.4, 0.45])
    refiner = _InspectingMaskRefiner(cfg.segmentation)
    resources = _Resources(
        cfg,
        0,
        None,
        segmenter,
        refiner,
        None,
        None,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    try:
        first_raw, first_refined = pipeline._segment_resource_masks(
            resources,
            _captured_frame(1, 1_000_000_000),
            privacy_safe=False,
        )
        np.testing.assert_allclose(first_raw, 0.4)
        np.testing.assert_allclose(first_refined, first_raw)
        assert segmenter.last_foreground is not None
        assert refiner._prev is not None

        second_raw, second_refined = pipeline._segment_resource_masks(
            resources,
            _captured_frame(
                2,
                second_timestamp_ns,
                shape=second_shape,
                generation=second_generation,
                geometry_generation=second_geometry_generation,
            ),
            privacy_safe=False,
        )

        assert segmenter.clean_reset_snapshots[-1] == (
            expected_reason,
            second_timestamp_ns,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        assert len(segmenter._session.feeds) == 2
        for name in ("r1i", "r2i", "r3i", "r4i"):
            assert not np.any(segmenter._session.feeds[1][name])
        assert refiner.previous_at_entry == [None, None]
        np.testing.assert_allclose(second_raw, 0.45)
        np.testing.assert_allclose(second_refined, second_raw)
        assert segmenter._rec is not None
        assert len(segmenter._rec) == 4
        assert all(np.all(state == 1.0) for state in segmenter._rec)
        assert segmenter._size == second_shape
        assert segmenter.last_downsample_ratio is not None
        assert segmenter.last_foreground is not None
        assert segmenter.last_input_sequence == 2
        assert segmenter.last_input_timestamp_ns == second_timestamp_ns
        assert refiner.last_input_sequence == 2
        assert refiner.last_input_timestamp_ns == second_timestamp_ns
        assert segmenter.temporal_reset_count == 2
        assert refiner.temporal_reset_count == 2
        assert segmenter.last_temporal_reset_reason is expected_reason
        assert refiner.last_temporal_reset_reason is expected_reason
        timeline = resources.segmentation_timeline.snapshot()
        assert timeline.reset_count == 2
        assert timeline.last_reset_reason is expected_reason
    finally:
        resources.close()


def test_irregular_fifteen_fps_sequence_gaps_preserve_rvm_and_refiner_state(
    cpu_ort,
):
    cfg = _temporal_app_config()
    segmenter = _ResetInspectingRVM(cfg.segmentation)
    _install_alpha_sequence(segmenter, [0.40, 0.42, 0.44, 0.46])
    refiner = _InspectingMaskRefiner(cfg.segmentation)
    resources = _Resources(
        cfg,
        0,
        None,
        segmenter,
        refiner,
        None,
        None,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    inputs = (
        (1, 1_000_000_000),
        (2, 1_066_666_667),
        (5, 1_135_000_000),
        (6, 1_201_000_000),
    )
    try:
        for sequence, timestamp_ns in inputs:
            pipeline._segment_resource_masks(
                resources,
                _captured_frame(sequence, timestamp_ns),
                privacy_safe=False,
            )

        assert len(segmenter._session.feeds) == 4
        for feed_index, expected_state in enumerate((0.0, 1.0, 2.0, 3.0)):
            for name in ("r1i", "r2i", "r3i", "r4i"):
                assert np.all(
                    segmenter._session.feeds[feed_index][name] == expected_state
                )
        assert refiner.previous_at_entry[0] is None
        assert all(previous is not None for previous in refiner.previous_at_entry[1:])
        assert [snapshot[0] for snapshot in segmenter.clean_reset_snapshots] == [
            TemporalResetReason.INITIAL
        ]
        assert segmenter.temporal_reset_count == 1
        assert refiner.temporal_reset_count == 1
        timeline = resources.segmentation_timeline.snapshot()
        assert timeline.reset_count == 1
        assert timeline.last_reset_reason is TemporalResetReason.INITIAL
        assert timeline.sequence_gap_events == 1
        assert timeline.sequence_gap_frames == 2
    finally:
        resources.close()


def test_context_free_rvm_resize_pairs_internal_reset_with_refiner(cpu_ort):
    """Compatibility callers still clear both owners on RVM-discovered geometry."""

    cfg = _temporal_app_config()
    segmenter = _ResetInspectingRVM(cfg.segmentation)
    _install_alpha_sequence(segmenter, [0.4, 0.45])
    refiner = _InspectingMaskRefiner(cfg.segmentation)

    first_raw, first_refined = Pipeline._segment_and_refine_masks(
        segmenter,
        refiner,
        np.full((16, 16, 3), 160, dtype=np.uint8),
        privacy_safe=False,
    )
    second_raw, second_refined = Pipeline._segment_and_refine_masks(
        segmenter,
        refiner,
        np.full((12, 20, 3), 160, dtype=np.uint8),
        privacy_safe=False,
    )

    np.testing.assert_allclose(first_refined, first_raw)
    np.testing.assert_allclose(second_refined, second_raw)
    assert refiner.previous_at_entry == [None, None]
    assert segmenter.temporal_reset_count == 1
    assert refiner.temporal_reset_count == 1
    assert segmenter.last_temporal_reset_reason is TemporalResetReason.GEOMETRY
    assert refiner.last_temporal_reset_reason is TemporalResetReason.GEOMETRY


def test_backend_recovery_resets_rvm_and_refiner_once_before_clean_cpu_retry(
    monkeypatch,
):
    gpu_ort(monkeypatch, run_error=RuntimeError, gpu_fail_after=2)
    cfg = _temporal_app_config()
    segmenter = _ResetInspectingRVM(cfg.segmentation)
    assert segmenter.device == "cuda"
    refiner = _InspectingMaskRefiner(cfg.segmentation)
    resources = _Resources(
        cfg,
        0,
        None,
        segmenter,
        refiner,
        None,
        None,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    try:
        pipeline._segment_resource_masks(
            resources,
            _captured_frame(1, 1_000_000_000),
            privacy_safe=False,
        )
        failed_gpu_session = segmenter._session
        assert refiner._prev is not None
        assert segmenter.last_foreground is not None
        segmenter_resets_before = segmenter.temporal_reset_count
        refiner_resets_before = refiner.temporal_reset_count
        timeline_resets_before = resources.segmentation_timeline.snapshot().reset_count

        pipeline._segment_resource_masks(
            resources,
            _captured_frame(2, 1_066_666_667),
            privacy_safe=False,
        )

        assert segmenter.device == "cpu"
        assert segmenter.temporal_reset_count == segmenter_resets_before + 1
        assert refiner.temporal_reset_count == refiner_resets_before + 1
        timeline = resources.segmentation_timeline.snapshot()
        assert timeline.reset_count == timeline_resets_before + 1
        assert (
            segmenter.last_temporal_reset_reason is TemporalResetReason.BACKEND_RECOVERY
        )
        assert (
            refiner.last_temporal_reset_reason is TemporalResetReason.BACKEND_RECOVERY
        )
        assert timeline.last_reset_reason is TemporalResetReason.BACKEND_RECOVERY
        assert [snapshot[0] for snapshot in segmenter.clean_reset_snapshots] == [
            TemporalResetReason.INITIAL,
            TemporalResetReason.BACKEND_RECOVERY,
        ]
        assert segmenter.clean_reset_snapshots[-1] == (
            TemporalResetReason.BACKEND_RECOVERY,
            1_066_666_667,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        assert refiner.previous_at_entry == [None, None]
        assert len(failed_gpu_session.feeds) == 3
        assert len(segmenter._session.feeds) == 1
        for name in ("r1i", "r2i", "r3i", "r4i"):
            assert not np.any(segmenter._session.feeds[0][name])
        assert segmenter._rec is not None
        assert len(segmenter._rec) == 4
        assert all(np.all(state == 1.0) for state in segmenter._rec)
        assert segmenter.last_foreground is not None
    finally:
        resources.close()


def test_failed_candidate_trial_preserves_live_rvm_refiner_and_timeline(
    cpu_ort,
    monkeypatch,
):
    cfg, pipeline, resources, live_segmenter, live_refiner = _active_rvm_resources(
        [0.4]
    )
    live_state = _pair_state(resources, live_segmenter, live_refiner)
    live_temporal_owner = resources.temporal_state_owner
    candidate_cfg = cfg.patched({"segmentation": {"rvm_downsample": 0.5}})
    activation, candidate_segmenter, candidate_refiner = (
        _prepare_concrete_rvm_candidate(
            monkeypatch,
            pipeline,
            cfg,
            candidate_cfg,
            [0.45],
            refiner_type=_FailingInspectingMaskRefiner,
        )
    )
    closed, discarded = _track_candidate_close(
        candidate_segmenter,
        candidate_refiner,
    )
    request = _PatchRequest(
        candidate_cfg,
        0,
        prepared_activation=activation,
    )
    try:
        pipeline._handle_patch_request(
            resources,
            request,
            _captured_frame(2, 1_066_666_667),
        )

        assert request.result is None
        assert isinstance(request.error, ActivationError)
        assert "candidate refinement failed" in str(request.error)
        assert closed.wait(1.0)
        assert resources.segmenter is live_segmenter
        assert resources.refiner is live_refiner
        assert resources.temporal_state_owner is live_temporal_owner
        assert resources.cfg is cfg
        assert resources.version == 0
        assert pipeline.runtime.version == 0
        _assert_pair_state(
            resources,
            live_segmenter,
            live_refiner,
            live_state,
        )
        assert discarded["feed_count"] == 1
        assert candidate_segmenter._session is None
    finally:
        resources.close()
        pipeline.stop(timeout=1.0)


def test_successful_passthrough_commit_restarts_trial_pair_on_exact_live_frame_and_defers_old_close(
    cpu_ort,
    monkeypatch,
):
    cfg, pipeline, resources, old_segmenter, old_refiner = _active_rvm_resources(
        [0.4],
        background_mode="passthrough",
    )
    old_state = _pair_state(resources, old_segmenter, old_refiner)
    timeline_before = resources.segmentation_timeline.snapshot()
    old_temporal_owner = resources.temporal_state_owner
    candidate_cfg = cfg.patched(
        {
            "segmentation": {
                "rvm_downsample": 0.5,
                "boundary_stabilization": {
                    "mode": "motion_aware",
                    "time_constant_s": 0.08,
                    "max_motion_px_per_s": 480.0,
                },
            }
        }
    )
    activation, candidate_segmenter, candidate_refiner = (
        _prepare_concrete_rvm_candidate(
            monkeypatch,
            pipeline,
            cfg,
            candidate_cfg,
            [0.45, 0.45],
        )
    )
    request = _PatchRequest(
        candidate_cfg,
        0,
        prepared_activation=activation,
    )
    real_thread_start = threading.Thread.start

    def defer_replaced_segmenter_close(worker: threading.Thread) -> None:
        if worker.name == "teardown-replaced-segmenter":
            raise RuntimeError("teardown thread unavailable")
        real_thread_start(worker)

    monkeypatch.setattr(
        threading.Thread,
        "start",
        defer_replaced_segmenter_close,
    )
    boundary_frame = _captured_frame(2, 1_066_666_667)
    try:
        pipeline._handle_patch_request(resources, request, boundary_frame)

        assert request.error is None
        assert request.result is not None
        assert pipeline.runtime.version == 1
        assert resources.segmenter is candidate_segmenter
        assert resources.refiner is candidate_refiner
        assert resources.segmentation_generation == 1
        assert resources.temporal_state_owner is not old_temporal_owner
        assert resources.temporal_state_owner.matches(
            policy=pipeline_mod._segmenter_key(candidate_cfg),
            generation=1,
            segmenter=candidate_segmenter,
            refiner=candidate_refiner,
        )
        assert activation.promoted
        assert activation.segmenter is None
        assert activation.refiner is None
        activation.discard()
        assert candidate_segmenter._session is not None
        assert resources.segmentation_timeline.snapshot() == timeline_before
        _assert_pair_state(
            resources,
            old_segmenter,
            old_refiner,
            old_state,
            include_resource_state=False,
        )
        assert pipeline._deferred_closes == [(old_segmenter, "replaced segmenter")]

        assert len(candidate_segmenter._session.feeds) == 1
        for name in ("r1i", "r2i", "r3i", "r4i"):
            assert not np.any(candidate_segmenter._session.feeds[0][name])
        assert [
            snapshot[0] for snapshot in candidate_segmenter.clean_reset_snapshots
        ] == [
            TemporalResetReason.SEGMENTATION_CONFIG,
            TemporalResetReason.SEGMENTATION_CONFIG,
        ]
        assert candidate_refiner.previous_at_entry == [None]
        assert candidate_segmenter._rec is None
        assert candidate_segmenter._size is None
        assert candidate_segmenter.last_downsample_ratio is None
        assert candidate_segmenter.last_foreground is None
        assert candidate_segmenter.last_input_sequence is None
        assert candidate_segmenter.last_input_timestamp_ns is None
        assert candidate_refiner._prev is None
        assert candidate_refiner._prev_guide is None
        assert candidate_refiner._prev_motion_timestamp_ns is None
        assert candidate_refiner._motion_hold_age is None
        assert candidate_refiner._motion_direction is None
        assert candidate_refiner.last_input_sequence is None
        assert candidate_refiner.last_input_timestamp_ns is None
        assert candidate_segmenter.temporal_reset_count == 2
        assert candidate_refiner.temporal_reset_count == 2

        raw, refined = pipeline._segment_resource_masks(
            resources,
            boundary_frame,
            privacy_safe=False,
        )

        # Policy: trial state is deliberately discarded. The promoted pair is
        # reset and reprocesses the exact same capture identity as its first
        # live input, so neither recurrent tensors nor previous alpha cross.
        assert len(candidate_segmenter._session.feeds) == 2
        for name in ("r1i", "r2i", "r3i", "r4i"):
            assert not np.any(candidate_segmenter._session.feeds[1][name])
        assert [
            snapshot[0] for snapshot in candidate_segmenter.clean_reset_snapshots
        ] == [
            TemporalResetReason.SEGMENTATION_CONFIG,
            TemporalResetReason.SEGMENTATION_CONFIG,
            TemporalResetReason.SEGMENTATION_CONFIG,
        ]
        assert candidate_refiner.previous_at_entry == [None, None]
        np.testing.assert_allclose(raw, 0.45)
        np.testing.assert_allclose(refined, raw)
        assert candidate_refiner._prev_guide is not None
        assert (
            candidate_refiner._prev_motion_timestamp_ns == boundary_frame.captured_at_ns
        )
        assert candidate_segmenter.last_input_sequence == boundary_frame.sequence
        assert (
            candidate_segmenter.last_input_timestamp_ns == boundary_frame.captured_at_ns
        )
        timeline = resources.segmentation_timeline.snapshot()
        assert timeline.reset_count == timeline_before.reset_count + 1
        assert timeline.last_reset_reason is TemporalResetReason.SEGMENTATION_CONFIG
        assert timeline.last_sequence == boundary_frame.sequence

        pipeline._drain_deferred_closes()
        assert old_segmenter._session is None
        assert pipeline._deferred_closes == []
        assert candidate_segmenter._session is not None
    finally:
        resources.close()
        pipeline.stop(timeout=1.0)


def test_background_only_commit_preserves_pair_identity_and_temporal_state(
    cpu_ort,
):
    cfg, pipeline, resources, segmenter, refiner = _active_rvm_resources([0.4, 0.42])
    state_before = _pair_state(resources, segmenter, refiner)
    owner_before = resources.temporal_state_owner
    candidate_cfg = cfg.patched({"background": {"anchor_x": 0.25}})
    activation = pipeline._prepare_activation_off_lane(cfg, candidate_cfg)
    assert activation.replace_segmenter is False
    request = _PatchRequest(
        candidate_cfg,
        0,
        prepared_activation=activation,
    )
    boundary_frame = _captured_frame(2, 1_066_666_667)
    try:
        pipeline._handle_patch_request(resources, request, boundary_frame)

        assert request.error is None
        assert request.result is not None
        assert resources.segmenter is segmenter
        assert resources.refiner is refiner
        assert resources.temporal_state_owner is owner_before
        assert resources.segmentation_generation == 0
        _assert_pair_state(
            resources,
            segmenter,
            refiner,
            state_before,
        )

        raw, refined = pipeline._segment_resource_masks(
            resources,
            boundary_frame,
            privacy_safe=False,
        )

        assert resources.segmenter is segmenter
        assert resources.refiner is refiner
        assert len(segmenter._session.feeds) == 2
        for name in ("r1i", "r2i", "r3i", "r4i"):
            assert np.all(segmenter._session.feeds[1][name] == 1.0)
        assert refiner.previous_at_entry[1] is not None
        np.testing.assert_allclose(raw, 0.42)
        assert not np.allclose(refined, raw)
        assert segmenter.temporal_reset_count == 1
        assert refiner.temporal_reset_count == 1
        timeline = resources.segmentation_timeline.snapshot()
        assert timeline.reset_count == 1
        assert timeline.last_reset_reason is TemporalResetReason.INITIAL
        assert resources.segmentation_generation == 0
    finally:
        resources.close()
        pipeline.stop(timeout=1.0)


def test_install_failure_rolls_back_concrete_pair_and_discards_trial_state(
    cpu_ort,
    monkeypatch,
):
    class RollbackBackdrop:
        def __init__(self) -> None:
            self.geometry = types.SimpleNamespace(
                fit_mode="cover",
                anchor_x=0.5,
                anchor_y=0.5,
            )
            self.set_calls: list[tuple[str, float, float]] = []

        def set_geometry(
            self,
            fit_mode: str,
            anchor_x: float,
            anchor_y: float,
        ) -> None:
            self.set_calls.append((fit_mode, anchor_x, anchor_y))
            if len(self.set_calls) == 1:
                raise RuntimeError("candidate pointer install failed")
            self.geometry = types.SimpleNamespace(
                fit_mode=fit_mode,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
            )

        def close(self) -> None:
            pass

    backdrop = RollbackBackdrop()
    cfg, pipeline, resources, live_segmenter, live_refiner = _active_rvm_resources(
        [0.4],
        backdrop=backdrop,
    )
    live_state = _pair_state(resources, live_segmenter, live_refiner)
    live_temporal_owner = resources.temporal_state_owner
    candidate_cfg = cfg.patched(
        {
            "background": {"anchor_x": 0.25},
            "segmentation": {"rvm_downsample": 0.5},
        }
    )
    activation, candidate_segmenter, candidate_refiner = (
        _prepare_concrete_rvm_candidate(
            monkeypatch,
            pipeline,
            cfg,
            candidate_cfg,
            [0.45],
        )
    )
    closed, discarded = _track_candidate_close(
        candidate_segmenter,
        candidate_refiner,
    )
    request = _PatchRequest(
        candidate_cfg,
        0,
        prepared_activation=activation,
    )
    try:
        pipeline._handle_patch_request(
            resources,
            request,
            _captured_frame(2, 1_066_666_667),
        )

        assert request.result is None
        assert isinstance(request.error, RuntimeError)
        assert "candidate pointer install failed" in str(request.error)
        assert closed.wait(1.0)
        assert resources.segmenter is live_segmenter
        assert resources.refiner is live_refiner
        assert resources.temporal_state_owner is live_temporal_owner
        assert resources.backdrop is backdrop
        assert resources.cfg is cfg
        assert resources.version == 0
        assert pipeline.runtime.version == 0
        _assert_pair_state(
            resources,
            live_segmenter,
            live_refiner,
            live_state,
        )
        assert backdrop.set_calls == [
            ("cover", 0.25, 0.5),
            ("cover", 0.5, 0.5),
        ]
        assert discarded["feed_count"] == 1
        assert discarded["rec"] is None
        assert discarded["previous_alpha"] is None
        assert discarded["reset_reasons"] == (
            TemporalResetReason.SEGMENTATION_CONFIG,
            TemporalResetReason.SEGMENTATION_CONFIG,
        )
    finally:
        resources.close()
        pipeline.stop(timeout=1.0)


def test_ack_timeout_discards_untried_candidate_and_keeps_live_pair_unchanged(
    cpu_ort,
    monkeypatch,
):
    cfg, pipeline, resources, live_segmenter, live_refiner = _active_rvm_resources(
        [0.4]
    )
    live_state = _pair_state(resources, live_segmenter, live_refiner)
    candidate_cfg = cfg.patched({"segmentation": {"rvm_downsample": 0.5}})
    activation, candidate_segmenter, candidate_refiner = (
        _prepare_concrete_rvm_candidate(
            monkeypatch,
            pipeline,
            cfg,
            candidate_cfg,
            [0.45],
        )
    )
    closed, discarded = _track_candidate_close(
        candidate_segmenter,
        candidate_refiner,
    )
    request = _PatchRequest(
        candidate_cfg,
        0,
        prepared_activation=activation,
    )

    class ImmediateTimeoutEvent:
        def __init__(self) -> None:
            self.set_calls = 0

        def wait(self, _timeout=None) -> bool:
            return False

        def set(self) -> None:
            self.set_calls += 1

    request.done = cast(Any, ImmediateTimeoutEvent())
    monkeypatch.setattr(
        pipeline,
        "_enqueue_request",
        lambda queued: pipeline._requests.put(queued),
    )
    try:
        with pytest.raises(ReconfigurationUnavailable, match="did not acknowledge"):
            pipeline._submit_patch(request, timeout=0.01)

        assert request.cancelled
        assert request.prepared_activation is None
        assert closed.wait(1.0)
        assert resources.segmenter is live_segmenter
        assert resources.refiner is live_refiner
        _assert_pair_state(
            resources,
            live_segmenter,
            live_refiner,
            live_state,
        )
        assert discarded["feed_count"] == 0
        assert discarded["rec"] is None
        assert discarded["previous_alpha"] is None
        assert discarded["reset_reasons"] == ()

        queued = pipeline._requests.get_nowait()
        assert queued is request
        pipeline._handle_patch_request(
            resources,
            request,
            _captured_frame(2, 1_066_666_667),
        )
        _assert_pair_state(
            resources,
            live_segmenter,
            live_refiner,
            live_state,
        )
        assert resources.segmentation_generation == 0
        assert pipeline.runtime.version == 0
    finally:
        resources.close()
        pipeline.stop(timeout=1.0)


def test_failed_cpu_retry_preserves_error_and_pends_one_clean_boundary(monkeypatch):
    """A failed recovery attempt is not published as an accepted-frame reset."""

    gpu_ort(monkeypatch, run_error=RuntimeError, gpu_fail_after=2)
    cfg = _temporal_app_config()
    segmenter = _ResetInspectingRVM(cfg.segmentation)
    refiner = _InspectingMaskRefiner(cfg.segmentation)
    resources = _Resources(
        cfg,
        0,
        None,
        segmenter,
        refiner,
        None,
        None,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())

    class _FailingCpuSession:
        def __init__(self) -> None:
            self.feeds: list[dict[str, np.ndarray]] = []

        def run(self, _outputs, feeds):
            self.feeds.append(feeds)
            raise RuntimeError("cpu retry failed")

    failing_cpu = _FailingCpuSession()
    try:
        pipeline._segment_resource_masks(
            resources,
            _captured_frame(1, 1_000_000_000),
            privacy_safe=False,
        )
        previous_refined = refiner._prev
        assert previous_refined is not None
        monkeypatch.setattr(segmenter, "_build_cpu_session", lambda: failing_cpu)

        with pytest.raises(RuntimeError, match="cpu retry failed"):
            pipeline._segment_resource_masks(
                resources,
                _captured_frame(2, 1_066_666_667),
                privacy_safe=False,
            )

        assert len(failing_cpu.feeds) == 1
        for name in ("r1i", "r2i", "r3i", "r4i"):
            assert not np.any(failing_cpu.feeds[0][name])
        assert refiner._prev is previous_refined
        assert refiner.previous_at_entry == [None]
        assert refiner.temporal_reset_count == 1
        failed_timeline = resources.segmentation_timeline.snapshot()
        assert failed_timeline.reset_count == 1
        assert failed_timeline.last_reset_reason is TemporalResetReason.INITIAL
        assert failed_timeline.last_sequence == 1

        cpu_mod = fake_ort(["CPUExecutionProvider"])
        segmenter._session = cpu_mod.InferenceSession(
            "/fake/model.onnx",
            providers=["CPUExecutionProvider"],
        )
        pipeline._segment_resource_masks(
            resources,
            _captured_frame(3, 1_133_333_334),
            privacy_safe=False,
        )

        assert len(segmenter._session.feeds) == 1
        for name in ("r1i", "r2i", "r3i", "r4i"):
            assert not np.any(segmenter._session.feeds[0][name])
        assert refiner.previous_at_entry == [None, None]
        assert refiner.temporal_reset_count == 2
        timeline = resources.segmentation_timeline.snapshot()
        assert timeline.reset_count == 2
        assert timeline.last_reset_reason is TemporalResetReason.BACKEND_RECOVERY
        assert timeline.last_sequence == 3
    finally:
        resources.close()
