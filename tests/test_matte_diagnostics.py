"""Privacy, round-trip, and attribution tests for MATTE-0.1 bundles."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import numpy as np
import pytest

from custback.__main__ import build_parser
from custback.compositor import composite
from custback.matte_diagnostics import (
    MatteCaptureMetadata,
    MatteDiagnosticRecorder,
    MatteDiagnosticsError,
    MatteFrameEvidence,
    MatteReplayBundle,
    ReplayOptions,
    replay_bundle,
)
from custback.segmentation import SegmentationFrameContext, TemporalResetReason


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    mask = np.linspace(0.0, 1.0, 12 * 16, dtype=np.float32).reshape(12, 16)
    backdrop = np.full_like(raw, 23)
    foreground = np.full_like(raw, 177)
    return raw, mask, backdrop, foreground


def _controls() -> dict[str, object]:
    return {
        "segmentation": {
            "backend": "heuristic",
            "model_path": "",
            "delegate": "cpu",
            "rvm_downsample": 0.0,
            "threshold": 0.5,
            "mask_blur": 0,
            "edge_refine": False,
            "mask_shift": 0,
            "temporal_smoothing": 0.0,
        },
        "acceleration": {
            "mode": "cpu",
            "provider": "auto",
            "device_id": 0,
        },
        "compositing": {
            "light_wrap": 0.0,
            "use_model_foreground": True,
            "blend_space": "srgb_legacy",
            "color_correction": {
                "mode": "off",
                "strength": 0.5,
                "exposure_limit_ev": 0.85,
                "white_balance_strength": 0.5,
                "adaptation_time_s": 0.8,
            },
        },
        "background": {
            "mode": "color",
            "fit_mode": "cover",
            "anchor_x": 0.5,
            "anchor_y": 0.5,
        },
    }


def _mediapipe_diagnostics(
    *,
    delta_ms: int | None = 33,
    adjustment_count: int = 0,
    adjustment_ms: int = 0,
) -> dict[str, object]:
    return {
        "backend": "mediapipe",
        "input_frame_shape": [12, 16],
        "model_mask_shape": [6, 8],
        "output_mask_shape": [12, 16],
        "effective_timestamp_delta_ms": delta_ms,
        "timestamp_adjustment_count": adjustment_count,
        "timestamp_adjustment_ms": adjustment_ms,
        "last_timestamp_adjusted": adjustment_ms > 0,
        "resize_interpolation": "linear",
    }


def _evidence(
    sequence: int, timestamp_ns: int
) -> tuple[MatteFrameEvidence, np.ndarray]:
    raw, mask, backdrop, foreground = _arrays()
    rendered = composite(
        raw,
        backdrop,
        mask,
        edge_foreground=foreground,
    )
    evidence = MatteFrameEvidence(
        MatteCaptureMetadata(
            bundle_sequence=sequence,
            capture_sequence=sequence + 10,
            capture_monotonic_ns=timestamp_ns,
            timestamp_source="capture-completion",
            capture_generation=2,
            geometry_generation=3,
        ),
        raw,
        raw_mask=mask,
        refined_mask=mask.copy(),
        clean_foreground=foreground,
        backdrop_frame=backdrop,
        base_composite=rendered,
        configured_controls=_controls(),
        effective_controls={
            "segmentation_backend": "RVMSegmenter",
            "rvm_downsample_ratio": 0.4,
            "mask_shift": 0,
            "use_model_foreground": True,
            "light_wrap": 0.0,
            "blend_space": "srgb_legacy",
        },
        timings_ms={
            "backend_inference_ms": 4.0,
            "refinement_ms": 0.2,
            "background_ms": 0.1,
            "composite_ms": 0.3,
        },
        backdrop_identity={
            "provider": "VideoBackdrop",
            "source_index": 7,
            "logical_index": 9,
            "pts_s": 0.25,
        },
    )
    return evidence, rendered


def test_full_bundle_round_trip_and_frozen_replay_are_exact(tmp_path):
    bundle_dir = tmp_path / "bundle"
    recorder = MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000)
    first, first_rendered = _evidence(0, 1_000_000_000)
    second, second_rendered = _evidence(1, 1_033_000_000)
    first.segmentation_diagnostics = _mediapipe_diagnostics(delta_ms=None)
    assert recorder.submit(first, first_rendered)
    recorder._queue.join()
    assert recorder.submit(second, second_rendered)
    recorder.close()

    bundle = MatteReplayBundle(bundle_dir)
    assert [frame["sequence"] for frame in bundle.frames] == [0, 1]
    assert [frame["capture_sequence"] for frame in bundle.frames] == [10, 11]
    assert [frame["capture_monotonic_ns"] for frame in bundle.frames] == [
        1_000_000_000,
        1_033_000_000,
    ]
    assert (
        bundle.frames[0]["segmentation_diagnostics"] == first.segmentation_diagnostics
    )
    assert "effective_timestamp_ms" not in bundle.frames[0]["segmentation_diagnostics"]
    assert "segmentation_diagnostics" not in bundle.frames[1]
    np.testing.assert_array_equal(
        bundle.load_array(bundle.frames[0], "raw_mask"),
        first.raw_mask,
    )
    np.testing.assert_array_equal(
        bundle.load_array(bundle.frames[0], "refined_mask"),
        first.refined_mask,
    )
    np.testing.assert_array_equal(
        bundle.load_array(bundle.frames[0], "clean_foreground"),
        first.clean_foreground,
    )

    report = replay_bundle(bundle_dir, tmp_path / "replay")
    assert [frame["reference_exact"] for frame in report["frames"]] == [True, True]
    assert report["frames"][0]["reference_max_channel_delta"] == 0
    assert report["frames"][0]["segmentation_diagnostics"] == {
        "recorded": first.segmentation_diagnostics,
        "rerun": None,
    }
    assert "segmentation_diagnostics" not in report["frames"][1]


def test_v1_resource_samples_accept_legacy_and_optional_rss_vram_fields(tmp_path):
    bundle_dir = tmp_path / "bundle"
    first, first_rendered = _evidence(0, 1_000_000_000)
    second, second_rendered = _evidence(1, 1_033_000_000)
    first.resource_samples = {
        "allocation_bytes": 1_024,
        "memory_bytes": 2_048,
    }
    second.resource_samples = {
        "rss_bytes": 3_072,
        "vram_bytes": 4_096,
    }
    with MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000) as recorder:
        assert recorder.submit(first, first_rendered)
        recorder._queue.join()
        assert recorder.submit(second, second_rendered)

    bundle = MatteReplayBundle(bundle_dir)
    assert bundle.manifest["version"] == 1
    assert bundle.frames[0]["resource_samples"] == first.resource_samples
    assert bundle.frames[1]["resource_samples"] == second.resource_samples


@pytest.mark.parametrize(
    "resources",
    (
        {"rss_bytes": True},
        {"vram_bytes": -1},
        {"private_gpu_path": 1},
    ),
)
def test_bundle_rejects_invalid_extended_resource_samples(tmp_path, resources):
    bundle_dir = tmp_path / "bundle"
    evidence, rendered = _evidence(0, 1)
    with MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000) as recorder:
        assert recorder.submit(evidence, rendered)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["resource_samples"] = resources
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MatteDiagnosticsError, match="resource samples"):
        MatteReplayBundle(bundle_dir)


@pytest.mark.parametrize(
    "invalid_diagnostics",
    [
        pytest.param(
            {**_mediapipe_diagnostics(), "effective_timestamp_ms": 33},
            id="absolute-timestamp-is-not-persisted",
        ),
        pytest.param(
            {
                **_mediapipe_diagnostics(),
                "output_mask_shape": [6, 8],
            },
            id="output-shape-does-not-match-input",
        ),
        pytest.param(
            {
                **_mediapipe_diagnostics(),
                "resize_interpolation": "area",
            },
            id="resize-policy-contradicts-shapes",
        ),
        pytest.param(
            {
                **_mediapipe_diagnostics(),
                "timestamp_adjustment_ms": 1,
            },
            id="adjustment-flag-is-inconsistent",
        ),
    ],
)
def test_bundle_rejects_malformed_segmentation_diagnostics(
    tmp_path,
    invalid_diagnostics,
):
    bundle_dir = tmp_path / "bundle"
    evidence, rendered = _evidence(0, 1)
    with MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000) as recorder:
        assert recorder.submit(evidence, rendered)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["segmentation_diagnostics"] = invalid_diagnostics
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MatteDiagnosticsError, match="segmentation diagnostic"):
        MatteReplayBundle(bundle_dir)


def test_scalar_output_timeline_keeps_repeats_out_of_unique_input_track(tmp_path):
    bundle_dir = tmp_path / "bundle"
    recorder = MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000)
    evidence, rendered = _evidence(0, 1_000_000_000)
    assert recorder.submit(evidence, rendered)
    recorder._queue.join()
    assert recorder.submit_output_event(
        sent_monotonic_ns=1_001_000_000,
        source_bundle_sequence=0,
        base_updated=True,
        exact_final_repeat=False,
    )
    assert recorder.submit_output_event(
        sent_monotonic_ns=1_011_000_000,
        source_bundle_sequence=0,
        base_updated=False,
        exact_final_repeat=True,
    )
    recorder.close()

    bundle = MatteReplayBundle(bundle_dir)
    assert len(bundle.frames) == 1
    assert len(bundle.output_events) == 2
    assert bundle.output_events[1]["source_bundle_sequence"] == 0
    assert bundle.output_events[1]["base_updated"] is False
    assert bundle.output_events[1]["exact_final_repeat"] is True


def test_frozen_attribution_swaps_compositor_without_mutating_upstream(tmp_path):
    bundle_dir = tmp_path / "bundle"
    evidence, rendered = _evidence(0, 10)
    with MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000) as recorder:
        assert recorder.submit(evidence, rendered)
    source = MatteReplayBundle(bundle_dir)
    raw_before = source.load_array(source.frames[0], "raw_frame")
    mask_before = source.load_array(source.frames[0], "refined_mask")

    report = replay_bundle(
        bundle_dir,
        tmp_path / "variant",
        options=ReplayOptions(model_foreground="off"),
    )
    assert report["frames"][0]["reference_exact"] is False
    source_after = MatteReplayBundle(bundle_dir)
    np.testing.assert_array_equal(
        source_after.load_array(source_after.frames[0], "raw_frame"),
        raw_before,
    )
    np.testing.assert_array_equal(
        source_after.load_array(source_after.frames[0], "refined_mask"),
        mask_before,
    )


def test_model_rerun_uses_recorded_selection_without_live_capture(tmp_path):
    bundle_dir = tmp_path / "bundle"
    evidence, rendered = _evidence(0, 10)
    with MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000) as recorder:
        assert recorder.submit(evidence, rendered)

    report = replay_bundle(
        bundle_dir,
        tmp_path / "rerun",
        options=ReplayOptions(mode="rerun", model_foreground="off"),
    )
    assert report["mode"] == "rerun"
    assert len(report["frames"]) == 1
    assert (tmp_path / "rerun" / "frames" / "00000000" / "composite.npy").is_file()


def test_model_rerun_preserves_recorded_timeline_and_config_resets(
    tmp_path,
    monkeypatch,
):
    events: list[tuple[object, ...]] = []
    created = 0

    class SpySegmenter:
        last_foreground = None

        def __init__(self, identifier: int):
            self.identifier = identifier
            self.temporal_reset_count = 0
            self.last_temporal_reset_reason = None
            self.last_shape = (12, 16)

        def reset_temporal_state(self, reason, timestamp_ns):
            self.temporal_reset_count += 1
            self.last_temporal_reset_reason = reason
            events.append(("segment-reset", self.identifier, reason, timestamp_ns))

        def segment(self, frame, *, context=None):
            events.append(("segment", self.identifier, context))
            self.last_shape = frame.shape[:2]
            return np.full(frame.shape[:2], 0.5, dtype=np.float32)

        def telemetry_snapshot(self):
            return {
                **_mediapipe_diagnostics(delta_ms=None),
                "input_frame_shape": self.last_shape,
                "model_mask_shape": self.last_shape,
                "output_mask_shape": self.last_shape,
                "resize_interpolation": "none",
                # Snapshot consumers must explicitly discard this private
                # submitted-clock value from persisted evidence.
                "effective_timestamp_ms": 123,
            }

        def close(self):
            events.append(("close", self.identifier))

    class SpyRefiner:
        def __init__(self, identifier: int):
            self.identifier = identifier

        def reset_temporal_state(self, reason, timestamp_ns):
            events.append(("refiner-reset", self.identifier, reason, timestamp_ns))

        def refine(self, mask, _frame, *, context=None):
            events.append(("refine", self.identifier, context))
            return mask

    def create_spy(*_args, **_kwargs):
        nonlocal created
        result = SpySegmenter(created)
        created += 1
        return result

    def create_refiner(_cfg, segmenter):
        return SpyRefiner(segmenter.identifier)

    monkeypatch.setattr("custback.matte_diagnostics.create_segmenter", create_spy)
    monkeypatch.setattr("custback.matte_diagnostics.refiner_for", create_refiner)

    bundle_dir = tmp_path / "bundle"
    first, first_rendered = _evidence(0, 1_000_000_000)
    second, second_rendered = _evidence(1, 1_033_000_000)
    second_segmentation = second.configured_controls["segmentation"]
    assert isinstance(second_segmentation, dict)
    second_segmentation["threshold"] = 0.6
    recorder = MatteDiagnosticRecorder(bundle_dir, max_bytes=4_000_000)
    assert recorder.submit(first, first_rendered)
    recorder._queue.join()
    assert recorder.submit(second, second_rendered)
    recorder.close()

    report = replay_bundle(
        bundle_dir,
        tmp_path / "rerun",
        options=ReplayOptions(mode="rerun", model_foreground="off"),
    )

    resets = [event for event in events if str(event[0]).endswith("reset")]
    assert resets == [
        (
            "segment-reset",
            0,
            TemporalResetReason.INITIAL,
            1_000_000_000,
        ),
        (
            "refiner-reset",
            0,
            TemporalResetReason.INITIAL,
            1_000_000_000,
        ),
        (
            "segment-reset",
            1,
            TemporalResetReason.SEGMENTATION_CONFIG,
            1_033_000_000,
        ),
        (
            "refiner-reset",
            1,
            TemporalResetReason.SEGMENTATION_CONFIG,
            1_033_000_000,
        ),
    ]
    segment_contexts = [event[2] for event in events if event[0] == "segment"]
    refine_contexts = [event[2] for event in events if event[0] == "refine"]
    expected = [
        SegmentationFrameContext(10, 1_000_000_000, 2, 3, (12, 16)),
        SegmentationFrameContext(11, 1_033_000_000, 2, 3, (12, 16)),
    ]
    assert segment_contexts == expected
    assert refine_contexts == expected
    assert [
        frame["segmentation_diagnostics"]["rerun"] for frame in report["frames"]
    ] == [
        {
            **_mediapipe_diagnostics(delta_ms=None),
            "input_frame_shape": [12, 16],
            "model_mask_shape": [12, 16],
            "output_mask_shape": [12, 16],
            "resize_interpolation": "none",
        },
        {
            **_mediapipe_diagnostics(delta_ms=None),
            "input_frame_shape": [12, 16],
            "model_mask_shape": [12, 16],
            "output_mask_shape": [12, 16],
            "resize_interpolation": "none",
        },
    ]


def test_composite_only_is_explicitly_insufficient_for_matte_metrics(tmp_path):
    bundle_dir = tmp_path / "bundle"
    evidence, rendered = _evidence(0, 10)
    with MatteDiagnosticRecorder(
        bundle_dir,
        capture_mode="composite_only",
        max_bytes=1_000_000,
    ) as recorder:
        assert recorder.submit(evidence, rendered)
    bundle = MatteReplayBundle(bundle_dir)
    assert bundle.manifest["matte_metrics_authoritative"] is False
    assert bundle.frames[0]["matte_metrics_authoritative"] is False
    assert "raw_frame" not in bundle.frames[0]["artifacts"]
    with pytest.raises(MatteDiagnosticsError, match="cannot run matte attribution"):
        replay_bundle(bundle_dir, tmp_path / "frozen")
    report = replay_bundle(
        bundle_dir,
        tmp_path / "reference",
        options=ReplayOptions(mode="reference"),
    )
    assert report["frames"][0]["reference_exact"] is True


def test_duration_and_size_limits_stop_cleanly(tmp_path):
    duration_dir = tmp_path / "duration"
    recorder = MatteDiagnosticRecorder(
        duration_dir,
        duration_s=0.01,
        max_bytes=2_000_000,
    )
    first, rendered = _evidence(0, 1_000_000_000)
    late, _ = _evidence(1, 1_020_000_000)
    assert recorder.submit(first, rendered)
    recorder._queue.join()
    assert recorder.submit(late, rendered) is False
    recorder.close()
    duration_manifest = json.loads(
        (duration_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert duration_manifest["stop_reason"] == "duration-limit"
    assert duration_manifest["frame_count"] == 1

    size_dir = tmp_path / "size"
    recorder = MatteDiagnosticRecorder(size_dir, max_bytes=64 * 1024)
    evidence, _rendered = _evidence(0, 1)
    large_raw = np.zeros((128, 128, 3), np.uint8)
    large_mask = np.zeros((128, 128), np.float32)
    evidence.raw_frame = large_raw
    evidence.raw_mask = large_mask
    evidence.refined_mask = large_mask
    evidence.clean_foreground = large_raw
    evidence.backdrop_frame = large_raw
    evidence.base_composite = large_raw
    assert recorder.submit(evidence, large_raw)
    recorder.close()
    size_manifest = json.loads((size_dir / "manifest.json").read_text(encoding="utf-8"))
    assert size_manifest["stop_reason"] == "size-limit"
    assert size_manifest["frame_count"] == 0


def test_bundle_and_artifact_permissions_are_owner_only(tmp_path):
    bundle_dir = tmp_path / "bundle"
    evidence, rendered = _evidence(0, 1)
    with MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000) as recorder:
        assert recorder.submit(evidence, rendered)
    assert _mode(bundle_dir) == 0o700
    assert _mode(bundle_dir / "frames") == 0o700
    assert _mode(bundle_dir / "frames" / "00000000") == 0o700
    assert _mode(bundle_dir / "manifest.json") == 0o600
    assert all(
        _mode(path) == 0o600 for path in (bundle_dir / "frames" / "00000000").iterdir()
    )


def test_interrupted_write_and_partial_manifest_are_rejected(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "write-error"
    recorder = MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000)
    evidence, rendered = _evidence(0, 1)

    def fail_write(_path, _payload):
        raise OSError("forced write interruption")

    monkeypatch.setattr(recorder, "_write_artifact", fail_write)
    assert recorder.submit(evidence, rendered)
    recorder.close()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "incomplete"
    assert manifest["stop_reason"] == "write-error"
    with pytest.raises(MatteDiagnosticsError, match="incomplete"):
        MatteReplayBundle(bundle_dir)

    partial_dir = tmp_path / "partial"
    partial_dir.mkdir(mode=0o700)
    (partial_dir / "manifest.partial.json").write_text("{}", encoding="utf-8")
    (partial_dir / "manifest.partial.json").chmod(0o600)
    with pytest.raises(MatteDiagnosticsError, match="interrupted"):
        MatteReplayBundle(partial_dir)


def test_malformed_manifest_and_traversal_are_rejected(tmp_path):
    bundle_dir = tmp_path / "bundle"
    evidence, rendered = _evidence(0, 1)
    with MatteDiagnosticRecorder(bundle_dir, max_bytes=2_000_000) as recorder:
        assert recorder.submit(evidence, rendered)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["artifacts"]["raw_frame"]["path"] = "../outside.npy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MatteDiagnosticsError, match="escapes"):
        MatteReplayBundle(bundle_dir)

    malformed_dir = tmp_path / "malformed"
    malformed_dir.mkdir(mode=0o700)
    malformed = malformed_dir / "manifest.json"
    malformed.write_bytes(b"{not-json")
    malformed.chmod(0o600)
    with pytest.raises(MatteDiagnosticsError, match="malformed"):
        MatteReplayBundle(malformed_dir)


def test_existing_or_symlink_output_paths_are_refused(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(MatteDiagnosticsError, match="already exists"):
        MatteDiagnosticRecorder(existing)

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    os.symlink(target, link)
    with pytest.raises(MatteDiagnosticsError, match="already exists"):
        MatteDiagnosticRecorder(link)


def test_cli_requires_explicit_directory_for_diagnostic_overrides():
    args = build_parser().parse_args([])
    assert args.matte_diagnostics_dir is None
    assert args.matte_diagnostics_duration is None
    assert args.matte_diagnostics_max_bytes is None
    assert args.matte_diagnostics_mode == "full"

    opted_in = build_parser().parse_args(
        [
            "--matte-diagnostics-dir",
            "new-private-dir",
            "--matte-diagnostics-mode",
            "composite-only",
            "--matte-diagnostics-duration",
            "12",
            "--matte-diagnostics-max-bytes",
            "123456",
        ]
    )
    assert opted_in.matte_diagnostics_dir == "new-private-dir"
    assert opted_in.matte_diagnostics_mode == "composite-only"
    assert opted_in.matte_diagnostics_duration == 12.0
    assert opted_in.matte_diagnostics_max_bytes == 123456


def test_recording_is_not_named_in_normal_public_stats():
    # The diagnostic recorder is constructor-only authority and does not add a
    # config/status field. This guard catches accidental public exposure later.
    from custback.config import AppConfig
    from custback.hub import FrameHub
    from custback.pipeline import Pipeline
    from custback.config import RuntimeConfig

    pipeline = Pipeline(RuntimeConfig(AppConfig()), FrameHub())
    assert all(
        "matte_diagnostic" not in key for key in pipeline.hub.stats_dict().keys()
    )


def test_pipeline_records_only_unique_full_composites_after_output_send(tmp_path):
    from custback.config import AppConfig, RuntimeConfig
    from custback.hub import FrameHub
    from custback.pipeline import Pipeline

    cfg = AppConfig.from_dict(
        {
            "camera": {
                "synthetic": True,
                "width": 64,
                "height": 48,
                "fps": 30,
            },
            "background": {"mode": "color", "color": [1, 2, 3]},
            "segmentation": {
                "backend": "heuristic",
                "mask_blur": 0,
                "edge_refine": False,
                "temporal_smoothing": 0.0,
            },
            "output": {"backend": "null", "fps": 30},
            "api": {"enabled": False},
        }
    )
    bundle_dir = tmp_path / "pipeline-bundle"
    recorder = MatteDiagnosticRecorder(bundle_dir, max_bytes=10_000_000)
    hub = FrameHub()
    pipeline = Pipeline(
        RuntimeConfig(cfg),
        hub,
        matte_recorder=recorder,
    )
    pipeline.start()
    time.sleep(0.08)
    pipeline.stop()

    bundle = MatteReplayBundle(bundle_dir)
    assert len(bundle.frames) >= 2
    assert bundle.manifest["matte_metrics_authoritative"] is True
    assert [frame["sequence"] for frame in bundle.frames] == list(
        range(len(bundle.frames))
    )
    capture_sequences = [int(frame["capture_sequence"]) for frame in bundle.frames]
    capture_timestamps = [int(frame["capture_monotonic_ns"]) for frame in bundle.frames]
    assert all(
        current > previous
        for previous, current in zip(capture_sequences, capture_sequences[1:])
    )
    assert all(timestamp > 0 for timestamp in capture_timestamps)
    assert capture_timestamps == sorted(capture_timestamps)
    assert all(
        frame["timestamp_source"] == "capture-completion" for frame in bundle.frames
    )
    assert all("output_send_ms" in frame["timings_ms"] for frame in bundle.frames)
    assert all(
        frame["effective_controls"]["segmentation_backend"] == "HeuristicSegmenter"
        for frame in bundle.frames
    )
    assert all(
        frame["effective_controls"]["edge_refinement_mode"] == "off"
        and frame["effective_controls"]["edge_refinement_radius_px"] == 0
        for frame in bundle.frames
    )
    assert len(bundle.output_events) >= len(bundle.frames)
    assert sorted(
        {event["source_bundle_sequence"] for event in bundle.output_events}
    ) == list(range(len(bundle.frames)))
    assert not any("matte_diagnostic" in key for key in hub.stats_dict())
