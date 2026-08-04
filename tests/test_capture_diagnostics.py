"""Deterministic capture-only diagnosis tests for MATTE-3.1."""

from __future__ import annotations

import hashlib
import json
import stat
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import custback.__main__ as core_main
import custback.capture as capture_mod
import custback.capture_diagnostics as diagnostics
import custback.diagnostics as process_diagnostics
from custback.capture import (
    CameraControlObservation,
    CameraControlReport,
    CapturedFrame,
    CaptureError,
    CaptureHealth,
    CaptureSource,
    CaptureTimingSample,
    OpenCVCapture,
)
from custback.config import AppConfig, CameraConfig


DEVICE_DIGEST = "a" * 64
HARDWARE_DIGEST = "b" * 64
PRIVATE_DEVICE = "/private/operator/camera0?token=do-not-persist"
MEASUREMENT_STARTED_NS = 9_876_543_210_000_000
MEASUREMENT_SECONDS = 5.0
MEASUREMENT_FINISHED_NS = MEASUREMENT_STARTED_NS + int(MEASUREMENT_SECONDS * 1e9)


def _config(
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    pixel_format: str = "auto",
    device: int | str = 0,
) -> AppConfig:
    return AppConfig.from_dict(
        {
            "camera": {
                "device": device,
                "width": width,
                "height": height,
                "fps": fps,
                "pixel_format": pixel_format,
            }
        }
    )


def _health(
    *,
    width: int = 1280,
    height: int = 720,
    fps_reported: float | None = 30.0,
    frames_read: int = 28,
    delivered_width: int | None = None,
    delivered_height: int | None = None,
    backend: str = "V4L2",
    fourcc: str | None = "MJPG",
    generation: int = 1,
    geometry_generation: int = 1,
    geometry_transitions: int = 0,
    read_failures: int = 0,
    restarts: int = 0,
    stalled: bool = False,
) -> CaptureHealth:
    return CaptureHealth(
        sequence=frames_read,
        captured_monotonic_ns=9_876_543_210_000_000,
        generation=generation,
        geometry_generation=geometry_generation,
        content_rect=(0, 0, width, height),
        backend=backend,
        fourcc=fourcc,
        width=width,
        height=height,
        delivered_width=width if delivered_width is None else delivered_width,
        delivered_height=height if delivered_height is None else delivered_height,
        oriented_width=width,
        oriented_height=height,
        normalized_width=width,
        normalized_height=height,
        geometry_transitions=geometry_transitions,
        fps_reported=fps_reported,
        capture_fps=0.0,
        target_met=None,
        frames_read=frames_read,
        read_failures=read_failures,
        restarts=restarts,
        stalled=stalled,
        camera_controls=CameraControlReport(
            backend_family="v4l2",
            generation=generation,
            properties=(
                CameraControlObservation(
                    name="auto_exposure",
                    status="reported",
                    value=0.75,
                ),
            ),
        ),
    )


def _sample(
    sequence: int,
    completion_ns: int,
    *,
    generation: int = 1,
    geometry_generation: int = 1,
    read_ms: float = 2.0,
    negotiation_ms: float = 0.1,
    normalization_ms: float = 1.0,
    publish_ms: float = 0.1,
    reader_cpu_ms: float | None = 0.5,
) -> CaptureTimingSample:
    return CaptureTimingSample(
        sequence=sequence,
        generation=generation,
        geometry_generation=geometry_generation,
        captured_at_ns=completion_ns,
        read_ms=read_ms,
        negotiation_ms=negotiation_ms,
        normalization_ms=normalization_ms,
        publish_ms=publish_ms,
        total_ms=read_ms + negotiation_ms + normalization_ms + publish_ms,
        reader_cpu_ms=reader_cpu_ms,
    )


def _rate_samples(
    fps: float,
    *,
    read_ms: float = 2.0,
    normalization_ms: float = 1.0,
    start_ns: int = MEASUREMENT_STARTED_NS,
    first_sequence: int = 1,
) -> tuple[CaptureTimingSample, ...]:
    successes = int(fps * MEASUREMENT_SECONDS)
    return tuple(
        _sample(
            first_sequence + index,
            start_ns + round((index + 1) / fps * 1_000_000_000),
            read_ms=read_ms,
            normalization_ms=normalization_ms,
        )
        for index in range(successes)
    )


def _window_samples(
    count: int,
    *,
    first_offset_ms: float,
    interval_ms: float = 1000.0 / 30.0,
) -> tuple[CaptureTimingSample, ...]:
    return tuple(
        _sample(
            index + 1,
            MEASUREMENT_STARTED_NS
            + round((first_offset_ms + index * interval_ms) * 1_000_000.0),
        )
        for index in range(count)
    )


def _report(
    *,
    fps: float = 30.0,
    fps_reported: float | None = 30.0,
    read_ms: float = 2.0,
    normalization_ms: float = 1.0,
    health: CaptureHealth | None = None,
    hardware_verified: bool = False,
    native_evidence: diagnostics._NativeEvidence | None = None,
    runtime_evidence: diagnostics._RuntimeEvidence | None = None,
    device_identity_sha256: str = "",
    hardware_identity_sha256: str = "",
    cfg: AppConfig | None = None,
    baseline_health: CaptureHealth | None = None,
    samples: tuple[CaptureTimingSample, ...] | None = None,
    measurement_started_ns: int = MEASUREMENT_STARTED_NS,
    measurement_finished_ns: int = MEASUREMENT_FINISHED_NS,
    warmup_seconds: float = 2.0,
) -> dict[str, Any]:
    timing_samples = (
        samples
        if samples is not None
        else _rate_samples(
            fps,
            read_ms=read_ms,
            normalization_ms=normalization_ms,
        )
    )
    final_health = health or _health(fps_reported=fps_reported)
    actual_seconds = (
        measurement_finished_ns - measurement_started_ns
    ) / 1_000_000_000.0
    return cast(
        dict[str, Any],
        diagnostics.build_capture_report(
            cfg=cfg or _config(),
            condition_id="reported-light",
            hardware_verified=hardware_verified,
            device_identity_sha256=device_identity_sha256,
            hardware_identity_sha256=hardware_identity_sha256,
            warmup_seconds=warmup_seconds,
            measurement_seconds=MEASUREMENT_SECONDS,
            actual_measurement_seconds=actual_seconds,
            measurement_started_ns=measurement_started_ns,
            measurement_finished_ns=measurement_finished_ns,
            baseline_health=(
                baseline_health if baseline_health is not None else CaptureHealth()
            ),
            final_health=final_health,
            samples=timing_samples,
            harness_deliveries=len(timing_samples),
            process_cpu_ms=125.0,
            capture_error=None,
            close_error=None,
            native_evidence=native_evidence,
            runtime_evidence=runtime_evidence,
        ),
    )


def _private_json(tmp_path: Path, name: str, value: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    return path


def _private_text(tmp_path: Path, name: str, value: str) -> Path:
    path = tmp_path / name
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    return path


def _native_payload(*, fps: float = 30.0) -> dict[str, object]:
    interval_ms = 1000.0 / fps
    count = int(5000.0 / interval_ms) + 1
    offsets = [index * interval_ms for index in range(count)]
    if offsets[-1] < 5000.0:
        offsets.append(5000.0)
    else:
        offsets[-1] = 5000.0
    return {
        "schema": diagnostics.NATIVE_EVIDENCE_SCHEMA,
        "version": diagnostics.NATIVE_EVIDENCE_VERSION,
        "device_identity_sha256": DEVICE_DIGEST,
        "hardware_identity_sha256": HARDWARE_DIGEST,
        "condition_id": "reported-light",
        "hardware_verified": True,
        "tool": {
            "name": "native-probe",
            "version": "1.0",
            "capture_api": "V4L2",
        },
        "requested": {
            "width": 1280,
            "height": 720,
            "fps": 30,
            "pixel_format": "auto",
        },
        "delivered": {
            "width": 1280,
            "height": 720,
            "pixel_format": "MJPG",
        },
        "timestamp_kind": "host-read-completion",
        "one_uninterrupted_run": True,
        "output_rate_conversion": False,
        "camera_control_writes": False,
        "completion_offsets_ms": offsets,
        "failures": 0,
    }


def _runtime_payload() -> dict[str, object]:
    return {
        "schema": diagnostics.RUNTIME_EVIDENCE_SCHEMA,
        "version": diagnostics.RUNTIME_EVIDENCE_VERSION,
        "device_identity_sha256": DEVICE_DIGEST,
        "hardware_identity_sha256": HARDWARE_DIGEST,
        "condition_id": "reported-light",
        "hardware_verified": True,
        "one_uninterrupted_run": True,
        "requested": {
            "width": 1280,
            "height": 720,
            "fps": 30,
            "pixel_format": "auto",
            "canvas_width": 1280,
            "canvas_height": 720,
        },
        "negotiated": {
            "backend": "V4L2",
            "pixel_format": "MJPG",
            "width": 1280,
            "height": 720,
            "fps_reported": 30.0,
            "delivered_width": 1280,
            "delivered_height": 720,
        },
        "start": {
            "run_id": "capture-run-01",
            "uptime_s": 10.0,
            "frames_in": 100,
            "frames_out": 200,
            "capture_frames_read": 300,
            "capture_dropped_frames": 20,
            "capture_read_failures": 2,
            "capture_restarts": 1,
            "processing_deadline_misses": 4,
        },
        "end": {
            "run_id": "capture-run-01",
            "uptime_s": 20.0,
            "frames_in": 250,
            "frames_out": 500,
            "capture_frames_read": 450,
            "capture_dropped_frames": 30,
            "capture_read_failures": 2,
            "capture_restarts": 1,
            "processing_deadline_misses": 7,
        },
        "timings_ms": {
            "capture_read": 2.0,
            "segmentation": 42.0,
            "background": 2.0,
            "color_correction": 1.0,
            "composite": 9.0,
            "output_send": 5.3,
            "frame_processing": 61.3,
        },
    }


def test_timing_report_uses_sequence_deltas_and_separates_recovery_outage() -> None:
    origin_ns = 1_000_000_000
    report = cast(
        dict[str, Any],
        diagnostics._timing_report(
            (
                _sample(1, origin_ns, read_ms=90.0),
                _sample(3, origin_ns + 66_666_666, read_ms=2.0),
                _sample(
                    4,
                    origin_ns + 1_066_666_666,
                    generation=2,
                    geometry_generation=2,
                    read_ms=80.0,
                ),
                _sample(
                    6,
                    origin_ns + 1_133_333_332,
                    generation=2,
                    geometry_generation=2,
                    read_ms=3.0,
                ),
            )
        ),
    )

    assert report["source_success_count"] == 6
    assert report["observer_missing_sample_count"] == 2
    assert report["generation_count"] == 2
    assert report["geometry_generation_count"] == 2
    assert report["active_capture_fps"] == pytest.approx(30.0, abs=0.001)
    assert report["wall_completion_fps"] == pytest.approx(4.4118, abs=0.0001)
    assert report["interval_ms"]["p50"] == pytest.approx(33.3333, abs=0.0001)
    assert report["cross_generation_outage_ms"]["p50"] == 1000.0
    # The first frame of each generation includes negotiation and is omitted
    # from steady-state service percentiles.
    assert report["read_ms"] == {
        "count": 2,
        "mean": 2.5,
        "p50": 2.0,
        "p95": 3.0,
        "p99": 3.0,
        "max": 3.0,
    }


@pytest.mark.parametrize(
    "samples",
    [
        (_sample(1, 1_000_000_000), _sample(1, 2_000_000_000)),
        (_sample(1, 1_000_000_000), _sample(2, 1_000_000_000)),
    ],
    ids=["non-increasing-sequence", "non-increasing-timestamp"],
)
def test_timing_report_rejects_non_increasing_identity(
    samples: tuple[CaptureTimingSample, CaptureTimingSample],
) -> None:
    with pytest.raises(
        diagnostics.CaptureDiagnosticError,
        match="must increase strictly",
    ):
        diagnostics._timing_report(samples)


def test_exact_ninety_percent_capture_threshold_is_inclusive() -> None:
    passing = _report(
        fps=27.0,
        hardware_verified=True,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )
    failing = _report(fps=26.99, read_ms=30.0, hardware_verified=True)

    assert passing["diagnosis"]["code"] == "target-sustained"
    assert passing["diagnosis"]["capture_only_fps"] == 27.0
    assert passing["qualification"]["acceptance_satisfied"] is True
    assert failing["diagnosis"]["code"] == "capture-read-path-paced"
    assert failing["diagnosis"]["capture_only_fps"] == 26.8
    assert failing["qualification"]["acceptance_satisfied"] is False


@pytest.mark.parametrize(
    ("samples", "expected_leading_ms", "expected_trailing_ms"),
    [
        (_window_samples(2, first_offset_ms=2000.0), 2000.0, 2966.6667),
        (_window_samples(145, first_offset_ms=200.0), 200.0, 0.0),
        (
            _window_samples(145, first_offset_ms=1000.0 / 30.0),
            33.3333,
            166.6667,
        ),
    ],
    ids=["two-sample-burst", "leading-starvation", "trailing-starvation"],
)
def test_full_measurement_window_is_required_for_target_sustained(
    samples: tuple[CaptureTimingSample, ...],
    expected_leading_ms: float,
    expected_trailing_ms: float,
) -> None:
    report = _report(
        samples=samples,
        hardware_verified=True,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    window = report["timing"]["measurement_window"]
    assert window["leading_gap_ms"] == pytest.approx(expected_leading_ms, abs=0.0001)
    assert window["trailing_gap_ms"] == pytest.approx(expected_trailing_ms, abs=0.0001)
    assert report["diagnosis"]["code"] == "capture-window-starvation"
    assert report["diagnosis"]["measurement_window_complete"] is False
    assert report["diagnosis"]["target_sustained"] is False
    assert report["qualification"]["acceptance_satisfied"] is False


def test_full_window_availability_count_cannot_be_replaced_by_fast_inner_rate() -> None:
    samples = _window_samples(
        134,
        first_offset_ms=100.0,
        interval_ms=4800.0 / 133.0,
    )

    report = _report(
        samples=samples,
        hardware_verified=True,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["timing"]["active_capture_fps"] > 27.0
    assert report["timing"]["wall_completion_fps"] > 27.0
    assert report["timing"]["measurement_window"]["availability_fps"] == 26.8
    assert report["diagnosis"]["measurement_window_complete"] is True
    assert report["diagnosis"]["target_sustained"] is False
    assert report["qualification"]["acceptance_satisfied"] is False


def test_long_mid_window_outage_cannot_be_hidden_by_a_fast_burst() -> None:
    samples = tuple(
        [
            _sample(
                index + 1,
                MEASUREMENT_STARTED_NS + (index + 1) * 1_000_000,
            )
            for index in range(135)
        ]
        + [_sample(136, MEASUREMENT_FINISHED_NS)]
    )

    report = _report(
        samples=samples,
        hardware_verified=True,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["timing"]["measurement_window"]["availability_fps"] == 27.2
    assert report["timing"]["interval_ms"]["p95"] == 1.0
    assert report["timing"]["maximum_completion_gap_ms"] > 4800.0
    assert report["diagnosis"]["measurement_window_complete"] is True
    assert report["diagnosis"]["measurement_window_sustained"] is False
    assert report["diagnosis"]["target_sustained"] is False
    assert report["qualification"]["acceptance_satisfied"] is False


def test_warmup_events_are_reported_without_poisoning_stable_measurement() -> None:
    baseline = _health(
        frames_read=12,
        read_failures=3,
        restarts=2,
        geometry_transitions=4,
    )
    final = _health(
        frames_read=162,
        read_failures=3,
        restarts=2,
        geometry_transitions=4,
    )

    report = _report(
        health=final,
        baseline_health=baseline,
        hardware_verified=True,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["measurement"]["read_failures"] == 0
    assert report["measurement"]["restarts"] == 0
    assert report["measurement"]["geometry_transitions"] == 0
    assert report["measurement"]["warmup_read_failures"] == 3
    assert report["measurement"]["warmup_restarts"] == 2
    assert report["measurement"]["warmup_geometry_transitions"] == 4
    assert report["diagnosis"]["code"] == "target-sustained"


def test_zero_warmup_first_geometry_establishment_is_not_instability() -> None:
    report = _report(
        warmup_seconds=0.0,
        baseline_health=CaptureHealth(),
        health=_health(geometry_transitions=1),
    )

    assert report["measurement"]["geometry_transitions"] == 0
    assert report["measurement"]["warmup_geometry_transitions"] == 0
    assert report["diagnosis"]["code"] == "target-sustained"


@pytest.mark.parametrize(
    ("fps_reported", "read_ms", "normalization_ms", "expected"),
    [
        (15.0, 2.0, 1.0, "driver-reported-mode-mismatch"),
        (30.0, 2.0, 10.0, "normalization-budget-limited"),
        (30.0, 60.0, 1.0, "capture-read-path-paced"),
        (30.0, 2.0, 1.0, "unresolved-capture-scheduling-or-backend-limit"),
    ],
)
def test_under_rate_classification_does_not_overclaim_causality(
    fps_reported: float,
    read_ms: float,
    normalization_ms: float,
    expected: str,
) -> None:
    report = _report(
        fps=15.0,
        fps_reported=fps_reported,
        read_ms=read_ms,
        normalization_ms=normalization_ms,
    )

    assert report["diagnosis"]["code"] == expected
    assert report["diagnosis"]["segmentation_executed"] is False
    assert report["diagnosis"]["segmentation_causal_for_capture_only_result"] is False
    assert (
        "auto-exposure-observation-does-not-prove-low-light-causality"
        in report["diagnosis"]["non_claims"]
    )


def test_native_evidence_parser_is_strict_private_and_digest_bound(
    tmp_path: Path,
) -> None:
    payload = _native_payload(fps=30.0)
    path = _private_json(tmp_path, "native.json", payload)

    evidence = diagnostics.load_native_evidence(path)

    assert evidence.fps == pytest.approx(30.0)
    assert evidence.device_identity_sha256 == DEVICE_DIGEST
    assert evidence.hardware_identity_sha256 == HARDWARE_DIGEST
    assert evidence.digest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence.hardware_verified is True


def test_native_evidence_accepts_common_backend_fourcc(
    tmp_path: Path,
) -> None:
    payload = _native_payload()
    cast(dict[str, object], payload["delivered"])["pixel_format"] = "YUYV"
    evidence = diagnostics.load_native_evidence(
        _private_json(tmp_path, "native-yuyv.json", payload)
    )

    report = _report(
        health=_health(fourcc="YUYV"),
        native_evidence=evidence,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["native_comparison"]["compatible"] is True
    assert report["native_comparison"]["summary"]["delivered"]["pixel_format"] == (
        "YUYV"
    )


@pytest.mark.parametrize("native_format", ["unknown", "MJPG"])
def test_native_evidence_requires_exact_observed_delivery(
    tmp_path: Path,
    native_format: str,
) -> None:
    payload = _native_payload()
    cast(dict[str, object], payload["delivered"])["pixel_format"] = native_format
    evidence = diagnostics.load_native_evidence(
        _private_json(tmp_path, f"native-delivered-{native_format}.json", payload)
    )
    local_health = (
        _health(fourcc=None)
        if native_format == "unknown"
        else _health(delivered_width=640)
    )

    report = _report(
        health=local_health,
        native_evidence=evidence,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["native_comparison"]["compatible"] is False
    reasons = report["native_comparison"]["compatibility_reasons"]
    expected = (
        "delivered-pixel-format-unverifiable"
        if native_format == "unknown"
        else "delivered-width-mismatch"
    )
    assert expected in reasons


def test_evidence_json_rejects_duplicate_members_and_huge_numbers(
    tmp_path: Path,
) -> None:
    native_json = json.dumps(_native_payload())
    duplicate = native_json.replace(
        '"hardware_verified": true',
        '"hardware_verified": false, "hardware_verified": true',
        1,
    )
    duplicate_path = _private_text(tmp_path, "native-duplicate.json", duplicate)
    with pytest.raises(diagnostics.CaptureDiagnosticError, match="is malformed"):
        diagnostics.load_native_evidence(duplicate_path)

    runtime_json = json.dumps(_runtime_payload())
    nested_duplicate = runtime_json.replace(
        '"run_id": "capture-run-01"',
        '"run_id": "other-run", "run_id": "capture-run-01"',
        1,
    )
    runtime_path = _private_text(
        tmp_path,
        "runtime-nested-duplicate.json",
        nested_duplicate,
    )
    with pytest.raises(diagnostics.CaptureDiagnosticError, match="is malformed"):
        diagnostics.load_runtime_evidence(runtime_path)

    huge_payload = _native_payload()
    cast(dict[str, object], huge_payload["requested"])["fps"] = 10**1000
    huge_path = _private_json(tmp_path, "native-huge-number.json", huge_payload)
    with pytest.raises(diagnostics.CaptureDiagnosticError, match="fps is invalid"):
        diagnostics.load_native_evidence(huge_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("camera_control_writes", True, "camera_control_writes"),
        ("timestamp_kind", "container-pts", "host-read-completion"),
        ("completion_offsets_ms", [0.0, 5000.0, 4999.0], "increase strictly"),
    ],
)
def test_native_evidence_rejects_unsafe_or_non_authoritative_inputs(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _native_payload()
    payload[field] = value
    path = _private_json(tmp_path, f"{field}.json", payload)

    with pytest.raises(diagnostics.CaptureDiagnosticError, match=match):
        diagnostics.load_native_evidence(path)


@pytest.mark.parametrize(
    ("native_fps", "expected"),
    [
        (30.0, "opencv-capture-path-limited"),
        (15.0, "shared-device-environment-or-backend-limit"),
    ],
)
def test_compatible_native_timestamps_corroborate_only_the_matching_boundary(
    tmp_path: Path,
    native_fps: float,
    expected: str,
) -> None:
    evidence = diagnostics.load_native_evidence(
        _private_json(
            tmp_path, f"native-{native_fps}.json", _native_payload(fps=native_fps)
        )
    )

    report = _report(
        fps=15.0,
        native_evidence=evidence,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["native_comparison"]["compatible"] is True
    assert report["diagnosis"]["code"] == expected
    assert report["diagnosis"]["confidence"] == "high"


def test_native_burst_with_long_outage_does_not_corroborate_sustained_rate(
    tmp_path: Path,
) -> None:
    payload = _native_payload()
    payload["completion_offsets_ms"] = [
        *[float(index) for index in range(135)],
        5000.0,
    ]
    evidence = diagnostics.load_native_evidence(
        _private_json(tmp_path, "native-burst.json", payload)
    )

    report = _report(
        fps=15.0,
        native_evidence=evidence,
        hardware_verified=True,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["native_comparison"]["compatible"] is True
    assert report["native_comparison"]["summary"]["capture_fps"] == 27.0
    assert report["native_comparison"]["summary"]["maximum_completion_gap_ms"] > 4800.0
    assert report["diagnosis"]["code"] == ("shared-device-environment-or-backend-limit")
    assert report["qualification"]["acceptance_satisfied"] is False


def test_shared_under_rate_evidence_does_not_claim_a_specific_limitation(
    tmp_path: Path,
) -> None:
    evidence = diagnostics.load_native_evidence(
        _private_json(tmp_path, "native-under-rate.json", _native_payload(fps=15.0))
    )

    report = _report(
        fps=15.0,
        native_evidence=evidence,
        hardware_verified=True,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["diagnosis"]["code"] == ("shared-device-environment-or-backend-limit")
    assert report["qualification"]["acceptance_satisfied"] is False
    assert report["qualification"]["outcome"] == "hardware-evidence-required"


def test_native_read_failures_make_evidence_incompatible(
    tmp_path: Path,
) -> None:
    payload = _native_payload(fps=30.0)
    payload["failures"] = 1
    evidence = diagnostics.load_native_evidence(
        _private_json(tmp_path, "native-failures.json", payload)
    )

    report = _report(
        fps=15.0,
        native_evidence=evidence,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["native_comparison"]["compatible"] is False
    assert (
        "native-read-failures" in report["native_comparison"]["compatibility_reasons"]
    )
    assert report["diagnosis"]["code"] == (
        "unresolved-capture-scheduling-or-backend-limit"
    )
    assert report["diagnosis"]["confidence"] == "medium"


def test_native_requested_pixel_format_mismatch_is_not_comparable(
    tmp_path: Path,
) -> None:
    payload = _native_payload()
    cast(dict[str, object], payload["requested"])["pixel_format"] = "mjpeg"
    evidence = diagnostics.load_native_evidence(
        _private_json(tmp_path, "native-requested-format.json", payload)
    )

    report = _report(
        native_evidence=evidence,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["native_comparison"]["compatible"] is False
    assert (
        "requested-pixel_format-mismatch"
        in report["native_comparison"]["compatibility_reasons"]
    )


def test_runtime_evidence_uses_two_snapshot_deltas_and_separates_cadence(
    tmp_path: Path,
) -> None:
    evidence_path = _private_json(tmp_path, "runtime.json", _runtime_payload())
    runtime = diagnostics.load_runtime_evidence(evidence_path)

    report = _report(
        fps=30.0,
        runtime_evidence=runtime,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert runtime.capture_fps == 15.0
    assert runtime.processed_unique_fps == 15.0
    assert runtime.output_fps == 30.0
    assert runtime.run_id == "capture-run-01"
    assert report["diagnosis"]["code"] == "target-sustained"
    assert report["diagnosis"]["runtime_comparison"] == "full-runtime-capture-regressed"
    assert report["pacing"]["capture"]["active_source_fps"] == pytest.approx(
        30.0, abs=0.0001
    )
    assert report["pacing"]["processed_frames"] == {
        "measured": True,
        "reason": "delta-unique-inputs-consumed-over-bounded-runtime-window",
        "unique_fps": 15.0,
        "output_fps": 30.0,
        "processing_deadline_misses": 3,
        "frame_processing_ms": 61.3,
        "stages_ms": {
            "capture_read": 2.0,
            "segmentation": 42.0,
            "background": 2.0,
            "color_correction": 1.0,
            "composite": 9.0,
            "output_send": 5.3,
        },
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing-start-run-id", "runtime start snapshot fields are invalid"),
        ("mismatched-run-id", "snapshots must share one run ID"),
    ],
)
def test_runtime_evidence_requires_one_bound_run(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    payload = _runtime_payload()
    if mutation == "missing-start-run-id":
        del cast(dict[str, object], payload["start"])["run_id"]
    else:
        cast(dict[str, object], payload["end"])["run_id"] = "capture-run-02"

    path = _private_json(tmp_path, f"runtime-{mutation}.json", payload)
    with pytest.raises(diagnostics.CaptureDiagnosticError, match=match):
        diagnostics.load_runtime_evidence(path)


@pytest.mark.parametrize(
    ("counter", "reason"),
    [
        ("capture_read_failures", "runtime-read-failures"),
        ("capture_restarts", "runtime-capture-restarts"),
    ],
)
def test_runtime_capture_instability_prevents_causal_comparison(
    tmp_path: Path,
    counter: str,
    reason: str,
) -> None:
    payload = _runtime_payload()
    start = cast(dict[str, object], payload["start"])
    end = cast(dict[str, object], payload["end"])
    end[counter] = cast(int, start[counter]) + 1
    runtime = diagnostics.load_runtime_evidence(
        _private_json(tmp_path, f"runtime-{counter}.json", payload)
    )

    report = _report(
        runtime_evidence=runtime,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["full_runtime_comparison"]["compatible"] is False
    assert reason in report["full_runtime_comparison"]["compatibility_reasons"]
    assert report["diagnosis"]["runtime_comparison"] == "not-compared"


@pytest.mark.parametrize("mutation", ["missing-end", "decreasing-counter"])
def test_runtime_evidence_rejects_non_comparable_snapshots(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _runtime_payload()
    if mutation == "missing-end":
        del payload["end"]
        match = "runtime cadence evidence fields are invalid"
    else:
        cast(dict[str, object], payload["end"])["frames_in"] = 99
        match = "counter frames_in decreased"

    path = _private_json(tmp_path, f"runtime-{mutation}.json", payload)
    with pytest.raises(diagnostics.CaptureDiagnosticError, match=match):
        diagnostics.load_runtime_evidence(path)


@pytest.mark.parametrize(
    ("boundary", "field", "value", "reason"),
    [
        (
            "requested",
            "pixel_format",
            "mjpeg",
            "runtime-requested-pixel_format-mismatch",
        ),
        (
            "negotiated",
            "fps_reported",
            15.0,
            "runtime-negotiated-fps_reported-mismatch",
        ),
        (
            "requested",
            "canvas_width",
            640,
            "runtime-requested-canvas_width-mismatch",
        ),
        (
            "negotiated",
            "delivered_width",
            640,
            "runtime-negotiated-delivered_width-mismatch",
        ),
    ],
)
def test_runtime_mode_mismatch_is_not_comparable(
    tmp_path: Path,
    boundary: str,
    field: str,
    value: object,
    reason: str,
) -> None:
    payload = _runtime_payload()
    cast(dict[str, object], payload[boundary])[field] = value
    runtime = diagnostics.load_runtime_evidence(
        _private_json(tmp_path, f"runtime-{field}-mismatch.json", payload)
    )

    report = _report(
        runtime_evidence=runtime,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["full_runtime_comparison"]["compatible"] is False
    assert reason in report["full_runtime_comparison"]["compatibility_reasons"]


def test_runtime_reported_fps_matches_two_decimal_status_precision(
    tmp_path: Path,
) -> None:
    runtime = diagnostics.load_runtime_evidence(
        _private_json(tmp_path, "runtime-rounded-fps.json", _runtime_payload())
    )

    report = _report(
        health=_health(fps_reported=30.004),
        runtime_evidence=runtime,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert report["full_runtime_comparison"]["compatible"] is True
    assert (
        "runtime-negotiated-fps_reported-mismatch"
        not in report["full_runtime_comparison"]["compatibility_reasons"]
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("backend", "RTSP://user:secret@private.example/camera", "backend is invalid"),
        ("pixel_format", "../x", "pixel format is invalid"),
    ],
)
def test_runtime_evidence_rejects_unsafe_native_labels(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    payload = _runtime_payload()
    cast(dict[str, object], payload["negotiated"])[field] = value

    path = _private_json(tmp_path, f"runtime-{field}.json", payload)
    with pytest.raises(diagnostics.CaptureDiagnosticError, match=match):
        diagnostics.load_runtime_evidence(path)


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        (0, "camera-index"),
        ("0", "camera-index"),
        ("/dev/video12", "camera-device-path"),
        ("/dev/v4l/by-id/usb-Camera_1.0:2", "camera-device-path"),
        ("recording.mp4", "unsupported-file-stream-or-device-string"),
        ("rtsp://private.example/camera", "unsupported-file-stream-or-device-string"),
    ],
)
def test_physical_source_classification_is_bounded(
    device: int | str,
    expected: str,
) -> None:
    assert diagnostics._physical_source_kind(device) == expected


def test_hardware_qualification_requires_physical_source_and_both_digests() -> None:
    unbound = _report(hardware_verified=True)
    bound = _report(
        hardware_verified=True,
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )
    unsupported = _report(
        hardware_verified=True,
        cfg=_config(device="recording.mp4"),
        device_identity_sha256=DEVICE_DIGEST,
        hardware_identity_sha256=HARDWARE_DIGEST,
    )

    assert unbound["qualification"]["acceptance_satisfied"] is False
    assert unbound["condition"]["device_identity_bound"] is False
    assert unbound["condition"]["hardware_identity_bound"] is False
    assert bound["qualification"]["acceptance_satisfied"] is True
    assert bound["condition"]["source_kind"] == "camera-index"
    assert bound["condition"]["physical_source_eligible"] is True
    assert unsupported["qualification"]["acceptance_satisfied"] is False
    assert unsupported["condition"]["physical_source_eligible"] is False


def test_run_capture_only_rejects_file_or_stream_before_opening() -> None:
    opened = False

    def factory(_camera: CameraConfig, _canvas: tuple[int, int]) -> CaptureSource:
        nonlocal opened
        opened = True
        raise AssertionError("unsupported source must be rejected before open")

    with pytest.raises(
        diagnostics.CaptureDiagnosticError,
        match="media files and streams",
    ):
        diagnostics.run_capture_only(
            _config(device=PRIVATE_DEVICE),
            warmup_seconds=0.0,
            measurement_seconds=5.0,
            capture_factory=factory,
        )

    assert opened is False


def test_report_redacts_device_native_labels_and_absolute_monotonic_time() -> None:
    hostile = _health(
        backend="RTSP://user:secret@private.example/camera0",
        fourcc="../x",
    )

    report = _report(health=hostile)
    serialized = json.dumps(report, sort_keys=True)

    assert PRIVATE_DEVICE not in serialized
    assert "private.example" not in serialized
    assert "user:secret" not in serialized
    assert "9876543210000000" not in serialized
    assert report["negotiated"]["backend"] == "unknown"
    assert report["negotiated"]["pixel_format"] == "unknown"
    assert report["privacy"] == {
        "contains_pixels": False,
        "contains_frame_hashes": False,
        "contains_wall_clock_timestamps": False,
        "contains_device_path_or_index": False,
        "contains_only_opaque_identity_bindings": True,
        "contains_credentials": False,
        "timing_trace_uses_relative_monotonic_offsets": True,
    }
    assert report["capture_only_contract"]["camera_control_policy"] == "preserve"
    assert report["capture_only_contract"]["camera_control_writes"] is False


def test_report_writer_uses_new_owner_only_directory(tmp_path: Path) -> None:
    report = _report()
    output = tmp_path / "capture-report"

    diagnostics.write_report(report, output)

    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "capture.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "capture.md").stat().st_mode) == 0o600
    assert json.loads((output / "capture.json").read_text()) == report
    assert PRIVATE_DEVICE not in (output / "capture.md").read_text()
    with pytest.raises(diagnostics.MatteDiagnosticsError, match="already exists"):
        diagnostics.write_report(report, output)


def _fourcc(code: str) -> int:
    return sum(ord(character) << (8 * index) for index, character in enumerate(code))


class _TimingCap:
    def __init__(self) -> None:
        self.values = {
            _TimingCV2.CAP_PROP_FRAME_WIDTH: 32.0,
            _TimingCV2.CAP_PROP_FRAME_HEIGHT: 18.0,
            _TimingCV2.CAP_PROP_FPS: 30.0,
            _TimingCV2.CAP_PROP_FOURCC: float(_fourcc("YUYV")),
            _TimingCV2.CAP_PROP_BACKEND: float(_TimingCV2.CAP_V4L2),
        }
        self.frame = np.full((18, 32, 3), 17, np.uint8)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def getBackendName(self) -> str:
        return "V4L2"

    def set(self, prop: int, value: float) -> bool:
        self.values[prop] = float(value)
        return True

    def get(self, prop: int) -> float:
        return self.values.get(prop, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(0.001)
        if self.released:
            return False, None
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True


class _TimingCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6
    CAP_PROP_BACKEND = 42
    CAP_V4L2 = 200

    def __init__(self, capture: _TimingCap) -> None:
        self.capture = capture

    def VideoCapture(
        self,
        _device: int | str,
        _api_preference: int | None = None,
    ) -> _TimingCap:
        return self.capture

    @staticmethod
    def VideoWriter_fourcc(*characters: str) -> int:
        return _fourcc("".join(characters))

    @staticmethod
    def resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        width, height = size
        return np.full((height, width, 3), frame[0, 0], np.uint8)

    @staticmethod
    def flip(frame: np.ndarray, axis: int) -> np.ndarray:
        return np.flip(frame, axis=axis).copy()


@pytest.mark.parametrize("collect_timing", [False, True])
def test_opencv_capture_timing_collection_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    collect_timing: bool,
) -> None:
    native_capture = _TimingCap()
    monkeypatch.setattr(capture_mod, "cv2", _TimingCV2(native_capture))
    cfg = CameraConfig(device=0, width=32, height=18, fps=30)
    capture = (
        OpenCVCapture(cfg, collect_timing=True)
        if collect_timing
        else OpenCVCapture(cfg)
    )
    try:
        deadline = time.monotonic() + 1.0
        while capture.health_snapshot().frames_read < 1:
            if time.monotonic() >= deadline:
                raise AssertionError("fake OpenCV capture produced no frame")
            time.sleep(0.003)
        samples = capture.timing_samples()
    finally:
        capture.close()

    assert bool(samples) is collect_timing
    assert native_capture.released is True


def test_timing_ring_covers_maximum_warmup_and_measurement_envelope() -> None:
    maximum_configured_frames = (15 + 60) * 240
    assert OpenCVCapture._TIMING_SAMPLE_CAPACITY >= maximum_configured_frames


def test_diagnostic_capture_opens_production_reader_with_timing_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _ScriptedSource(())
    calls: list[tuple[CameraConfig, tuple[int, int], bool]] = []

    def open_capture(
        cfg: CameraConfig,
        canvas: tuple[int, int],
        *,
        collect_timing: bool = False,
    ) -> CaptureSource:
        calls.append((cfg, canvas, collect_timing))
        return source

    monkeypatch.setattr(diagnostics, "open_capture", open_capture)

    assert (
        diagnostics._open_diagnostic_capture(
            CameraConfig(device=0),
            (1280, 720),
        )
        is source
    )
    assert calls[0][2] is True


class _FakeTime:
    def __init__(self, *, initial_ns: int = 0) -> None:
        self.wall_ns = initial_ns
        self.cpu_ns = 0

    def monotonic(self) -> float:
        return self.wall_ns / 1_000_000_000.0

    def monotonic_ns(self) -> int:
        return self.wall_ns

    def process_time_ns(self) -> int:
        return self.cpu_ns

    def sleep(self, seconds: float) -> None:
        self.wall_ns += round(seconds * 1_000_000_000)
        self.cpu_ns += 1_000_000


class _ScriptedSource(CaptureSource):
    def __init__(
        self,
        samples: tuple[CaptureTimingSample, ...],
        *,
        read_error: CaptureError | KeyboardInterrupt | None = None,
    ) -> None:
        self.samples = samples
        self.read_error = read_error
        self.read_calls = 0
        self.health_calls = 0
        self.close_calls = 0
        self.timing_after_sequence: int | None = None

    def read(self) -> CapturedFrame | None:
        self.read_calls += 1
        if self.read_error is not None:
            error = self.read_error
            self.read_error = None
            raise error
        return None

    def health_snapshot(self) -> CaptureHealth:
        self.health_calls += 1
        return CaptureHealth() if self.health_calls == 1 else _health()

    def timing_samples(
        self,
        *,
        after_sequence: int = 0,
    ) -> tuple[CaptureTimingSample, ...]:
        self.timing_after_sequence = after_sequence
        return self.samples

    def close(self) -> None:
        self.close_calls += 1


def test_run_capture_only_uses_only_injected_capture_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeTime()
    source = _ScriptedSource(_rate_samples(30.0, start_ns=0))
    opened: list[tuple[CameraConfig, tuple[int, int]]] = []

    def factory(camera: CameraConfig, canvas: tuple[int, int]) -> CaptureSource:
        opened.append((camera, canvas))
        return source

    monkeypatch.setattr(diagnostics, "time", clock)
    monkeypatch.setattr(diagnostics, "POLL_SECONDS", 1.0)

    report = cast(
        dict[str, Any],
        diagnostics.run_capture_only(
            _config(),
            warmup_seconds=0.0,
            measurement_seconds=5.0,
            capture_factory=factory,
        ),
    )

    assert len(opened) == 1
    assert opened[0][1] == (1280, 720)
    assert source.read_calls == 5
    assert source.close_calls == 1
    assert source.timing_after_sequence == 0
    assert report["measurement"]["measurement_seconds_actual"] == 5.0
    assert report["capture_only_contract"] == {
        "production_capture_reader": True,
        "camera_acquisition": True,
        "canonical_normalization": True,
        "segmentation": False,
        "backdrop": False,
        "compositor": False,
        "preview": False,
        "api": False,
        "output_sink": False,
        "camera_control_policy": "preserve",
        "camera_control_writes": False,
    }


def test_run_capture_only_filters_samples_before_measurement_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurement_start_ns = 10_000_000_000
    clock = _FakeTime(initial_ns=measurement_start_ns)
    warmup_sample = _sample(1, measurement_start_ns - 1)
    measurement_samples = _rate_samples(
        30.0,
        start_ns=measurement_start_ns,
        first_sequence=2,
    )
    source = _ScriptedSource((warmup_sample, *measurement_samples))
    monkeypatch.setattr(diagnostics, "time", clock)
    monkeypatch.setattr(diagnostics, "POLL_SECONDS", 1.0)

    report = cast(
        dict[str, Any],
        diagnostics.run_capture_only(
            _config(),
            warmup_seconds=0.0,
            measurement_seconds=5.0,
            capture_factory=lambda _camera, _canvas: source,
        ),
    )

    assert source.timing_after_sequence == 0
    assert report["timing"]["sample_count"] == 150
    assert report["timing"]["source_success_count"] == 150
    assert report["timing"]["trace"][0]["source_sequence_offset"] == 0
    assert report["timing"]["measurement_window"]["leading_gap_ms"] == pytest.approx(
        33.3333, abs=0.0001
    )
    assert report["diagnosis"]["code"] == "target-sustained"


def test_run_capture_only_redacts_capture_error_and_closes_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeTime()
    source = _ScriptedSource(
        (),
        read_error=CaptureError(f"failed {PRIVATE_DEVICE}"),
    )
    monkeypatch.setattr(diagnostics, "time", clock)
    monkeypatch.setattr(diagnostics, "POLL_SECONDS", 1.0)

    report = cast(
        dict[str, Any],
        diagnostics.run_capture_only(
            _config(),
            warmup_seconds=0.0,
            measurement_seconds=5.0,
            capture_factory=lambda _camera, _canvas: source,
        ),
    )

    assert source.close_calls == 1
    assert report["measurement"]["capture_error"] == "CaptureError"
    assert report["diagnosis"]["code"] == "unstable-capture"
    assert PRIVATE_DEVICE not in json.dumps(report)


def test_run_capture_only_closes_before_propagating_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _ScriptedSource((), read_error=KeyboardInterrupt())
    monkeypatch.setattr(diagnostics, "time", _FakeTime())
    monkeypatch.setattr(diagnostics, "POLL_SECONDS", 1.0)

    with pytest.raises(KeyboardInterrupt):
        diagnostics.run_capture_only(
            _config(),
            warmup_seconds=0.0,
            measurement_seconds=5.0,
            capture_factory=lambda _camera, _canvas: source,
        )

    assert source.close_calls == 1


def test_run_capture_only_propagates_keyboard_interrupt_from_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _ScriptedSource(_rate_samples(30.0, start_ns=0))

    def interrupting_close() -> None:
        source.close_calls += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(source, "close", interrupting_close)
    monkeypatch.setattr(diagnostics, "time", _FakeTime())
    monkeypatch.setattr(diagnostics, "POLL_SECONDS", 1.0)

    with pytest.raises(KeyboardInterrupt):
        diagnostics.run_capture_only(
            _config(),
            warmup_seconds=0.0,
            measurement_seconds=5.0,
            capture_factory=lambda _camera, _canvas: source,
        )

    assert source.close_calls == 1


def test_core_cli_dispatches_capture_diagnosis_before_normal_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str] | None, str]] = []

    def capture_main(
        argv: list[str] | None = None,
        *,
        prog: str,
    ) -> int:
        calls.append((argv, prog))
        return 23

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("normal runtime diagnostics must not initialize")

    monkeypatch.setattr(diagnostics, "main", capture_main)
    monkeypatch.setattr(process_diagnostics, "configure_logging", forbidden)
    monkeypatch.setattr(core_main, "Pipeline", forbidden)

    result = core_main.main(["capture-diagnose", "--output", "private-report"])

    assert result == 23
    assert calls == [(["--output", "private-report"], "custback capture-diagnose")]


def test_capture_cli_redacts_hostile_configuration_value(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _private_json(
        tmp_path,
        "invalid-config.json",
        {"camera": {"device": {"credential": PRIVATE_DEVICE}}},
    )

    result = diagnostics.main(
        [
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "unused-output"),
        ]
    )
    captured = capsys.readouterr()

    assert result == 2
    assert PRIVATE_DEVICE not in captured.err
    assert "token=do-not-persist" not in captured.err
    assert "camera.device" in captured.err


def test_capture_cli_does_not_echo_custom_validator_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_origin = "https://private.internal/secret?token=do-not-persist"
    config_path = _private_json(
        tmp_path,
        "invalid-origin-config.json",
        {"api": {"allowed_origins": [private_origin]}},
    )

    result = diagnostics.main(
        [
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "unused-output"),
        ]
    )
    captured = capsys.readouterr()

    assert result == 2
    assert private_origin not in captured.err
    assert "private.internal" not in captured.err
    assert "token=do-not-persist" not in captured.err
    assert "api: configuration value is invalid" in captured.err


def test_capture_cli_redacts_hostile_io_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "evidence-token=do-not-persist.json"

    result = diagnostics.main(
        [
            "--runtime-evidence",
            str(missing_path),
            "--output",
            str(tmp_path / "unused-output"),
        ]
    )
    captured = capsys.readouterr()

    assert result == 2
    assert str(missing_path) not in captured.err
    assert "token=do-not-persist" not in captured.err
    assert "bundle artifact is missing" in captured.err
