"""Bounded, privacy-safe capture-only cadence diagnosis.

This command exercises the production :class:`~custback.capture.OpenCVCapture`
reader and canonical-frame normalization without constructing a pipeline,
segmenter, backdrop, compositor, preview, API, or output sink. Its report
keeps camera acquisition and processed-frame pacing as separate evidence
boundaries and never records pixels, frame hashes, wall-clock timestamps, or
the configured device/path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from pydantic import ValidationError

from .capture import (
    CaptureError,
    CaptureHealth,
    CaptureSource,
    CaptureTimingSample,
    open_capture,
)
from .config import AppConfig, CameraConfig, resolved_output_size
from .matte_diagnostics import (
    MatteDiagnosticsError,
    _atomic_private_write,
    _json_bytes,
    _private_directory,
    _read_private_file,
)

REPORT_SCHEMA = "custback.capture-diagnostic-report"
REPORT_VERSION = 1
NATIVE_EVIDENCE_SCHEMA = "custback.native-capture-evidence"
NATIVE_EVIDENCE_VERSION = 1
RUNTIME_EVIDENCE_SCHEMA = "custback.capture-runtime-evidence"
RUNTIME_EVIDENCE_VERSION = 1

MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
MAX_TIMING_SAMPLES = 16_384
MIN_MEASUREMENT_SECONDS = 5.0
MAX_MEASUREMENT_SECONDS = 60.0
MAX_WARMUP_SECONDS = 15.0
POLL_SECONDS = 0.001
TARGET_RATIO = 0.9

_SAFE_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_SAFE_TOOL_RE = re.compile(r"[A-Za-z0-9_.+ -]{1,64}")
_SAFE_RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

CaptureFactory = Callable[[CameraConfig, tuple[int, int]], CaptureSource]


class CaptureDiagnosticError(ValueError):
    """Capture evidence is malformed, unsafe, or insufficiently specified."""


class _DuplicateJsonKeyError(ValueError):
    """Internal marker for contradictory JSON object members."""


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError
        result[key] = value
    return result


def _load_strict_json(payload: bytes, *, name: str) -> object:
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
        RecursionError,
    ) as exc:
        raise CaptureDiagnosticError(f"{name} is malformed") from exc


def _open_diagnostic_capture(
    cfg: CameraConfig,
    canvas_size: tuple[int, int],
) -> CaptureSource:
    return open_capture(cfg, canvas_size, collect_timing=True)


@dataclass(frozen=True)
class _NativeEvidence:
    digest_sha256: str
    device_identity_sha256: str
    hardware_identity_sha256: str
    condition_id: str
    tool: dict[str, str]
    requested: dict[str, object]
    delivered: dict[str, object]
    completion_offsets_ms: tuple[float, ...]
    failures: int
    hardware_verified: bool

    @property
    def fps(self) -> float:
        span_ms = self.completion_offsets_ms[-1] - self.completion_offsets_ms[0]
        return (len(self.completion_offsets_ms) - 1) * 1000.0 / span_ms


@dataclass(frozen=True)
class _RuntimeEvidence:
    digest_sha256: str
    device_identity_sha256: str
    hardware_identity_sha256: str
    condition_id: str
    run_id: str
    requested: dict[str, object]
    negotiated: dict[str, object]
    duration_seconds: float
    capture_fps: float
    processed_unique_fps: float
    output_fps: float
    counter_deltas: dict[str, int]
    timings_ms: dict[str, float | None]


def _strict_mapping(
    value: object,
    keys: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CaptureDiagnosticError(f"{name} fields are invalid")
    return cast(dict[str, Any], value)


def _bounded_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise CaptureDiagnosticError(f"{name} is invalid")
    return value


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureDiagnosticError(f"{name} is invalid")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CaptureDiagnosticError(f"{name} is invalid") from exc
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise CaptureDiagnosticError(f"{name} is invalid")
    return result


def _safe_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise CaptureDiagnosticError(f"{name} is invalid")
    return value


def _safe_tool(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SAFE_TOOL_RE.fullmatch(value) is None:
        raise CaptureDiagnosticError(f"{name} is invalid")
    return value


def _safe_backend_label(value: object) -> str:
    """Bound the native backend label before it enters durable evidence."""

    if not isinstance(value, str) or _SAFE_TOOL_RE.fullmatch(value) is None:
        return "unknown"
    return value


def _safe_fourcc(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Z0-9 ]{1,4}", value) is None:
        return "unknown"
    return value


def _physical_source_kind(device: int | str) -> str:
    """Classify source authority without serializing its value."""

    if isinstance(device, int) and not isinstance(device, bool) and device >= 0:
        return "camera-index"
    if isinstance(device, str) and device.isdecimal():
        return "camera-index"
    if isinstance(device, str) and (
        re.fullmatch(r"/dev/video[0-9]+", device)
        or re.fullmatch(r"/dev/v4l/by-id/[A-Za-z0-9_.:+-]+", device)
    ):
        return "camera-device-path"
    return "unsupported-file-stream-or-device-string"


def _sha256_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CaptureDiagnosticError(f"{name} is invalid")
    return value


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _rate_meets(value: float, threshold: float) -> bool:
    """Compare measured rates without penalizing nanosecond quantization."""

    tolerance = max(1e-6, abs(threshold) * 1e-9)
    return value >= threshold - tolerance


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    """Return a deterministic nearest-rank percentile."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def _summary(values: Sequence[float]) -> dict[str, object]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": _round(sum(finite) / len(finite)),
        "p50": _round(_nearest_rank(finite, 0.50)),
        "p95": _round(_nearest_rank(finite, 0.95)),
        "p99": _round(_nearest_rank(finite, 0.99)),
        "max": _round(max(finite)),
    }


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_energy = sum(value * value for value in left_centered)
    right_energy = sum(value * value for value in right_centered)
    if left_energy <= 0.0 or right_energy <= 0.0:
        return None
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    )
    return _round(numerator / math.sqrt(left_energy * right_energy), 6)


def _timing_report(
    samples: Sequence[CaptureTimingSample],
    *,
    measurement_started_ns: int | None = None,
    measurement_finished_ns: int | None = None,
) -> dict[str, object]:
    """Summarize and serialize one bounded content-free completion trace."""

    if len(samples) > MAX_TIMING_SAMPLES:
        raise CaptureDiagnosticError("capture timing sample bound was exceeded")
    if (measurement_started_ns is None) != (measurement_finished_ns is None):
        raise CaptureDiagnosticError("capture measurement window is incomplete")
    window_seconds: float | None = None
    if measurement_started_ns is not None and measurement_finished_ns is not None:
        if (
            type(measurement_started_ns) is not int
            or type(measurement_finished_ns) is not int
            or measurement_started_ns < 0
            or measurement_finished_ns < measurement_started_ns
        ):
            raise CaptureDiagnosticError("capture measurement window is invalid")
        window_seconds = (
            measurement_finished_ns - measurement_started_ns
        ) / 1_000_000_000.0
    if not samples:
        return {
            "sample_count": 0,
            "source_success_count": 0,
            "observer_missing_sample_count": 0,
            "generation_count": 0,
            "geometry_generation_count": 0,
            "active_capture_fps": None,
            "wall_completion_fps": None,
            "maximum_completion_gap_ms": None,
            "interval_ms": _summary(()),
            "cross_generation_outage_ms": _summary(()),
            "read_ms": _summary(()),
            "pre_normalization_ms": _summary(()),
            "normalization_ms": _summary(()),
            "publish_ms": _summary(()),
            "total_ms": _summary(()),
            "reader_cpu_ms": _summary(()),
            "measurement_window": {
                "duration_seconds": _round(window_seconds),
                "availability_fps": 0.0 if window_seconds else None,
                "leading_gap_ms": None,
                "trailing_gap_ms": None,
                "completion_span_ms": None,
            },
            "correlation": {
                "interval_vs_read": None,
                "interval_vs_previous_normalization": None,
            },
            "trace": [],
        }

    previous: CaptureTimingSample | None = None
    intervals: list[float] = []
    interval_reads: list[float] = []
    previous_normalizations: list[float] = []
    outages: list[float] = []
    active_successes = 0
    active_span_ms = 0.0
    observer_missing = 0
    for sample in samples:
        if (
            type(sample.sequence) is not int
            or sample.sequence <= 0
            or type(sample.generation) is not int
            or sample.generation <= 0
            or type(sample.captured_at_ns) is not int
            or sample.captured_at_ns < 0
        ):
            raise CaptureDiagnosticError("capture timing identity is invalid")
        if previous is not None:
            sequence_delta = sample.sequence - previous.sequence
            timestamp_delta_ns = sample.captured_at_ns - previous.captured_at_ns
            if sequence_delta <= 0 or timestamp_delta_ns <= 0:
                raise CaptureDiagnosticError(
                    "capture timing samples must increase strictly"
                )
            observer_missing += max(0, sequence_delta - 1)
            elapsed_ms = timestamp_delta_ns / 1_000_000.0
            if sample.generation == previous.generation:
                per_success_ms = elapsed_ms / sequence_delta
                intervals.append(per_success_ms)
                interval_reads.append(sample.read_ms)
                previous_normalizations.append(previous.normalization_ms)
                active_successes += sequence_delta
                active_span_ms += elapsed_ms
            else:
                outages.append(elapsed_ms)
        previous = sample

    first = samples[0]
    last = samples[-1]
    if (
        measurement_started_ns is not None
        and measurement_finished_ns is not None
        and (
            first.captured_at_ns < measurement_started_ns
            or last.captured_at_ns > measurement_finished_ns
        )
    ):
        raise CaptureDiagnosticError(
            "capture samples fall outside the measurement window"
        )
    total_sequence_delta = last.sequence - first.sequence
    total_span_ms = (last.captured_at_ns - first.captured_at_ns) / 1_000_000.0
    wall_completion_fps = (
        total_sequence_delta * 1000.0 / total_span_ms
        if total_sequence_delta > 0 and total_span_ms > 0.0
        else None
    )
    active_capture_fps = (
        active_successes * 1000.0 / active_span_ms
        if active_successes > 0 and active_span_ms > 0.0
        else None
    )

    first_per_generation: set[int] = set()
    steady_samples: list[CaptureTimingSample] = []
    for sample in samples:
        if sample.generation in first_per_generation:
            steady_samples.append(sample)
        else:
            first_per_generation.add(sample.generation)
    if not steady_samples:
        steady_samples = list(samples)

    origin_ns = first.captured_at_ns
    origin_sequence = first.sequence
    origin_generation = first.generation
    origin_geometry = first.geometry_generation
    trace: list[dict[str, object]] = []
    for sample in samples:
        trace.append(
            {
                "source_sequence_offset": sample.sequence - origin_sequence,
                "generation_offset": sample.generation - origin_generation,
                "geometry_generation_offset": (
                    sample.geometry_generation - origin_geometry
                ),
                "completion_offset_ms": _round(
                    (sample.captured_at_ns - origin_ns) / 1_000_000.0
                ),
                "read_ms": _round(sample.read_ms),
                "pre_normalization_ms": _round(sample.negotiation_ms),
                "normalization_ms": _round(sample.normalization_ms),
                "publish_ms": _round(sample.publish_ms),
                "total_ms": _round(sample.total_ms),
                "reader_cpu_ms": _round(sample.reader_cpu_ms),
            }
        )

    reader_cpu = [
        float(sample.reader_cpu_ms)
        for sample in steady_samples
        if sample.reader_cpu_ms is not None
    ]
    source_success_count = total_sequence_delta + 1
    availability_fps = (
        source_success_count / window_seconds
        if window_seconds is not None and window_seconds > 0.0
        else None
    )
    leading_gap_ms = (
        (first.captured_at_ns - measurement_started_ns) / 1_000_000.0
        if measurement_started_ns is not None
        else None
    )
    trailing_gap_ms = (
        (measurement_finished_ns - last.captured_at_ns) / 1_000_000.0
        if measurement_finished_ns is not None
        else None
    )
    return {
        "sample_count": len(samples),
        "source_success_count": source_success_count,
        "observer_missing_sample_count": observer_missing,
        "generation_count": len({sample.generation for sample in samples}),
        "geometry_generation_count": len(
            {sample.geometry_generation for sample in samples}
        ),
        # Retain full finite precision for the 90% acceptance comparison.
        # Presentation layers may round, but classification must not turn a
        # just-below-threshold observation into a pass.
        "active_capture_fps": active_capture_fps,
        "wall_completion_fps": wall_completion_fps,
        "maximum_completion_gap_ms": max(intervals) if intervals else None,
        "interval_ms": _summary(intervals),
        "cross_generation_outage_ms": _summary(outages),
        "read_ms": _summary([sample.read_ms for sample in steady_samples]),
        "pre_normalization_ms": _summary(
            [sample.negotiation_ms for sample in steady_samples]
        ),
        "normalization_ms": _summary(
            [sample.normalization_ms for sample in steady_samples]
        ),
        "publish_ms": _summary([sample.publish_ms for sample in steady_samples]),
        "total_ms": _summary([sample.total_ms for sample in steady_samples]),
        "reader_cpu_ms": _summary(reader_cpu),
        "measurement_window": {
            "duration_seconds": _round(window_seconds),
            "availability_fps": availability_fps,
            "leading_gap_ms": _round(leading_gap_ms),
            "trailing_gap_ms": _round(trailing_gap_ms),
            "completion_span_ms": _round(total_span_ms),
        },
        "correlation": {
            "interval_vs_read": _correlation(intervals, interval_reads),
            "interval_vs_previous_normalization": _correlation(
                intervals,
                previous_normalizations,
            ),
        },
        "trace": trace,
    }


def _requested_mode(
    value: object,
    *,
    name: str,
    include_canvas: bool = False,
) -> dict[str, object]:
    keys = {"width", "height", "fps", "pixel_format"}
    if include_canvas:
        keys.update(("canvas_width", "canvas_height"))
    raw = _strict_mapping(
        value,
        keys,
        name=name,
    )
    result: dict[str, object] = {
        "width": _bounded_int(
            raw["width"], name=f"{name} width", minimum=1, maximum=16_384
        ),
        "height": _bounded_int(
            raw["height"], name=f"{name} height", minimum=1, maximum=16_384
        ),
    }
    pixel_format = raw["pixel_format"]
    if pixel_format not in ("auto", "mjpeg", "backend"):
        raise CaptureDiagnosticError(f"{name} pixel format is invalid")
    result["pixel_format"] = pixel_format
    result["fps"] = _finite(
        raw["fps"],
        name=f"{name} fps",
        minimum=1.0,
        maximum=240.0,
    )
    if include_canvas:
        for field in ("canvas_width", "canvas_height"):
            result[field] = _bounded_int(
                raw[field],
                name=f"{name} {field.replace('_', ' ')}",
                minimum=1,
                maximum=16_384,
            )
    return result


def _delivered_mode(value: object, *, name: str) -> dict[str, object]:
    raw = _strict_mapping(
        value,
        {"width", "height", "pixel_format"},
        name=name,
    )
    pixel_format = _safe_fourcc(raw["pixel_format"])
    if pixel_format == "unknown" and raw["pixel_format"] != "unknown":
        raise CaptureDiagnosticError(f"{name} pixel format is invalid")
    return {
        "width": _bounded_int(
            raw["width"], name=f"{name} width", minimum=1, maximum=16_384
        ),
        "height": _bounded_int(
            raw["height"], name=f"{name} height", minimum=1, maximum=16_384
        ),
        "pixel_format": pixel_format,
    }


def load_native_evidence(path: Path | str) -> _NativeEvidence:
    """Load strict same-device native-tool host-completion timestamps."""

    payload = _read_private_file(Path(path), max_bytes=MAX_EVIDENCE_BYTES)
    value = _load_strict_json(payload, name="native capture evidence")
    root = _strict_mapping(
        value,
        {
            "schema",
            "version",
            "device_identity_sha256",
            "hardware_identity_sha256",
            "condition_id",
            "hardware_verified",
            "tool",
            "requested",
            "delivered",
            "timestamp_kind",
            "one_uninterrupted_run",
            "output_rate_conversion",
            "camera_control_writes",
            "completion_offsets_ms",
            "failures",
        },
        name="native capture evidence",
    )
    if (
        root["schema"] != NATIVE_EVIDENCE_SCHEMA
        or root["version"] != NATIVE_EVIDENCE_VERSION
    ):
        raise CaptureDiagnosticError("native capture evidence schema is unsupported")
    for name, expected in (
        ("hardware_verified", True),
        ("one_uninterrupted_run", True),
        ("output_rate_conversion", False),
        ("camera_control_writes", False),
    ):
        if type(root[name]) is not bool or root[name] is not expected:
            raise CaptureDiagnosticError(f"native capture evidence {name} is invalid")
    if root["timestamp_kind"] != "host-read-completion":
        raise CaptureDiagnosticError(
            "native evidence must use host-read-completion timestamps"
        )
    tool_raw = _strict_mapping(
        root["tool"],
        {"name", "version", "capture_api"},
        name="native capture tool",
    )
    tool = {
        name: _safe_tool(tool_raw[name], name=f"native tool {name}")
        for name in ("name", "version", "capture_api")
    }
    requested = _requested_mode(
        root["requested"],
        name="native requested mode",
    )
    delivered = _delivered_mode(root["delivered"], name="native delivered mode")
    offsets_raw = root["completion_offsets_ms"]
    if (
        not isinstance(offsets_raw, list)
        or not 2 <= len(offsets_raw) <= MAX_TIMING_SAMPLES
    ):
        raise CaptureDiagnosticError("native completion timestamp count is invalid")
    offsets = tuple(
        _finite(
            value,
            name="native completion offset",
            minimum=0.0,
            maximum=600_000.0,
        )
        for value in offsets_raw
    )
    if offsets[0] != 0.0 or any(
        current <= previous for previous, current in zip(offsets, offsets[1:])
    ):
        raise CaptureDiagnosticError(
            "native completion offsets must start at zero and increase strictly"
        )
    if offsets[-1] < MIN_MEASUREMENT_SECONDS * 1000.0:
        raise CaptureDiagnosticError(
            "native capture evidence is shorter than five seconds"
        )
    failures = _bounded_int(
        root["failures"],
        name="native failure count",
        minimum=0,
        maximum=1_000_000,
    )
    return _NativeEvidence(
        digest_sha256=hashlib.sha256(payload).hexdigest(),
        device_identity_sha256=_sha256_id(
            root["device_identity_sha256"],
            name="native device identity digest",
        ),
        hardware_identity_sha256=_sha256_id(
            root["hardware_identity_sha256"],
            name="native hardware identity digest",
        ),
        condition_id=_safe_id(root["condition_id"], name="native condition ID"),
        tool=tool,
        requested=requested,
        delivered=delivered,
        completion_offsets_ms=offsets,
        failures=failures,
        hardware_verified=True,
    )


def _native_summary(
    evidence: _NativeEvidence | None,
    *,
    requested: Mapping[str, object],
    negotiated: Mapping[str, object],
    condition_id: str,
    device_identity_sha256: str,
    hardware_identity_sha256: str,
) -> dict[str, object]:
    if evidence is None:
        return {
            "provided": False,
            "compatible": None,
            "compatibility_reasons": ["native-comparison-not-provided"],
            "summary": None,
        }
    reasons: list[str] = []
    if not device_identity_sha256 or (
        evidence.device_identity_sha256 != device_identity_sha256
    ):
        reasons.append("device-identity-mismatch-or-missing")
    if not hardware_identity_sha256 or (
        evidence.hardware_identity_sha256 != hardware_identity_sha256
    ):
        reasons.append("hardware-identity-mismatch-or-missing")
    if evidence.condition_id != condition_id:
        reasons.append("condition-mismatch")
    if evidence.failures:
        reasons.append("native-read-failures")
    for field in ("width", "height", "fps", "pixel_format"):
        if evidence.requested[field] != requested[field]:
            reasons.append(f"requested-{field}-mismatch")
    for field in ("width", "height"):
        if evidence.delivered[field] != negotiated.get(f"delivered_{field}"):
            reasons.append(f"delivered-{field}-mismatch")
    negotiated_format = negotiated.get("pixel_format")
    native_format = evidence.delivered["pixel_format"]
    if negotiated_format in (None, "unknown") or native_format == "unknown":
        reasons.append("delivered-pixel-format-unverifiable")
    elif native_format != negotiated_format:
        reasons.append("delivered-pixel-format-mismatch")

    intervals = [
        current - previous
        for previous, current in zip(
            evidence.completion_offsets_ms,
            evidence.completion_offsets_ms[1:],
        )
    ]
    return {
        "provided": True,
        "compatible": not reasons,
        "compatibility_reasons": reasons,
        "summary": {
            "evidence_sha256": evidence.digest_sha256,
            "tool": evidence.tool,
            "requested": evidence.requested,
            "delivered": evidence.delivered,
            "condition_id": evidence.condition_id,
            "sample_count": len(evidence.completion_offsets_ms),
            "capture_fps": evidence.fps,
            "interval_ms": _summary(intervals),
            "maximum_completion_gap_ms": max(intervals),
            "failures": evidence.failures,
            "hardware_verified": evidence.hardware_verified,
        },
    }


_RUNTIME_COUNTER_KEYS = {
    "frames_in",
    "frames_out",
    "capture_frames_read",
    "capture_dropped_frames",
    "capture_read_failures",
    "capture_restarts",
    "processing_deadline_misses",
}
_RUNTIME_TIMING_KEYS = {
    "capture_read",
    "segmentation",
    "background",
    "color_correction",
    "composite",
    "output_send",
    "frame_processing",
}


def _runtime_snapshot(
    value: object,
    *,
    name: str,
) -> dict[str, float | int | str]:
    raw = _strict_mapping(
        value,
        {"run_id", "uptime_s", *_RUNTIME_COUNTER_KEYS},
        name=name,
    )
    run_id = raw["run_id"]
    if not isinstance(run_id, str) or _SAFE_RUN_ID_RE.fullmatch(run_id) is None:
        raise CaptureDiagnosticError(f"{name} run ID is invalid")
    result: dict[str, float | int | str] = {
        "run_id": run_id,
        "uptime_s": _finite(
            raw["uptime_s"],
            name=f"{name} uptime",
            minimum=0.0,
            maximum=10_000_000_000.0,
        ),
    }
    for field in sorted(_RUNTIME_COUNTER_KEYS):
        result[field] = _bounded_int(
            raw[field],
            name=f"{name} {field}",
            minimum=0,
            maximum=10**15,
        )
    return result


def load_runtime_evidence(path: Path | str) -> _RuntimeEvidence:
    """Load a strict two-snapshot full-pipeline cadence observation."""

    payload = _read_private_file(Path(path), max_bytes=MAX_EVIDENCE_BYTES)
    value = _load_strict_json(payload, name="runtime cadence evidence")
    root = _strict_mapping(
        value,
        {
            "schema",
            "version",
            "device_identity_sha256",
            "hardware_identity_sha256",
            "condition_id",
            "hardware_verified",
            "one_uninterrupted_run",
            "requested",
            "negotiated",
            "start",
            "end",
            "timings_ms",
        },
        name="runtime cadence evidence",
    )
    if (
        root["schema"] != RUNTIME_EVIDENCE_SCHEMA
        or root["version"] != RUNTIME_EVIDENCE_VERSION
    ):
        raise CaptureDiagnosticError("runtime cadence evidence schema is unsupported")
    if root["hardware_verified"] is not True:
        raise CaptureDiagnosticError(
            "runtime cadence evidence hardware_verified must be true"
        )
    if root["one_uninterrupted_run"] is not True:
        raise CaptureDiagnosticError(
            "runtime cadence evidence one_uninterrupted_run must be true"
        )
    requested = _requested_mode(
        root["requested"],
        name="runtime requested mode",
        include_canvas=True,
    )
    negotiated_raw = _strict_mapping(
        root["negotiated"],
        {
            "backend",
            "pixel_format",
            "width",
            "height",
            "fps_reported",
            "delivered_width",
            "delivered_height",
        },
        name="runtime negotiated mode",
    )
    runtime_backend = _safe_tool(
        negotiated_raw["backend"],
        name="runtime negotiated backend",
    )
    runtime_fourcc = _safe_fourcc(negotiated_raw["pixel_format"])
    if runtime_fourcc == "unknown" and negotiated_raw["pixel_format"] != "unknown":
        raise CaptureDiagnosticError("runtime negotiated pixel format is invalid")
    negotiated = {
        "backend": runtime_backend,
        "pixel_format": runtime_fourcc,
        "width": _bounded_int(
            negotiated_raw["width"],
            name="runtime negotiated width",
            minimum=1,
            maximum=16_384,
        ),
        "height": _bounded_int(
            negotiated_raw["height"],
            name="runtime negotiated height",
            minimum=1,
            maximum=16_384,
        ),
        "fps_reported": _finite(
            negotiated_raw["fps_reported"],
            name="runtime negotiated FPS",
            minimum=0.0,
            maximum=240.0,
        ),
        "delivered_width": _bounded_int(
            negotiated_raw["delivered_width"],
            name="runtime delivered width",
            minimum=1,
            maximum=16_384,
        ),
        "delivered_height": _bounded_int(
            negotiated_raw["delivered_height"],
            name="runtime delivered height",
            minimum=1,
            maximum=16_384,
        ),
    }
    start = _runtime_snapshot(root["start"], name="runtime start snapshot")
    end = _runtime_snapshot(root["end"], name="runtime end snapshot")
    if start["run_id"] != end["run_id"]:
        raise CaptureDiagnosticError("runtime cadence snapshots must share one run ID")
    duration_seconds = float(end["uptime_s"]) - float(start["uptime_s"])
    if duration_seconds < MIN_MEASUREMENT_SECONDS:
        raise CaptureDiagnosticError(
            "runtime cadence evidence window is shorter than five seconds"
        )
    deltas: dict[str, int] = {}
    for field in sorted(_RUNTIME_COUNTER_KEYS):
        delta = int(end[field]) - int(start[field])
        if delta < 0:
            raise CaptureDiagnosticError(f"runtime cadence counter {field} decreased")
        deltas[field] = delta
    timings_raw = _strict_mapping(
        root["timings_ms"],
        _RUNTIME_TIMING_KEYS,
        name="runtime timing summary",
    )
    timings: dict[str, float | None] = {}
    for field in sorted(_RUNTIME_TIMING_KEYS):
        raw = timings_raw[field]
        timings[field] = (
            None
            if raw is None
            else _finite(
                raw,
                name=f"runtime timing {field}",
                minimum=0.0,
                maximum=10_000_000.0,
            )
        )
    return _RuntimeEvidence(
        digest_sha256=hashlib.sha256(payload).hexdigest(),
        device_identity_sha256=_sha256_id(
            root["device_identity_sha256"],
            name="runtime device identity digest",
        ),
        hardware_identity_sha256=_sha256_id(
            root["hardware_identity_sha256"],
            name="runtime hardware identity digest",
        ),
        condition_id=_safe_id(root["condition_id"], name="runtime condition ID"),
        run_id=cast(str, start["run_id"]),
        requested=requested,
        negotiated=negotiated,
        duration_seconds=duration_seconds,
        capture_fps=deltas["capture_frames_read"] / duration_seconds,
        processed_unique_fps=deltas["frames_in"] / duration_seconds,
        output_fps=deltas["frames_out"] / duration_seconds,
        counter_deltas=deltas,
        timings_ms=timings,
    )


def _runtime_summary(
    runtime: _RuntimeEvidence | None,
    *,
    requested: Mapping[str, object],
    negotiated: Mapping[str, object],
    condition_id: str,
    device_identity_sha256: str,
    hardware_identity_sha256: str,
) -> dict[str, object]:
    if runtime is None:
        return {
            "provided": False,
            "compatible": None,
            "compatibility_reasons": ["full-runtime-evidence-not-provided"],
            "capture_pacing": None,
            "processed_frame_pacing": {
                "measured": False,
                "reason": "not-run-by-capture-only-harness",
                "unique_fps": None,
                "frame_processing_ms": None,
            },
        }
    reasons: list[str] = []
    if (
        not device_identity_sha256
        or runtime.device_identity_sha256 != device_identity_sha256
    ):
        reasons.append("device-identity-mismatch-or-missing")
    if (
        not hardware_identity_sha256
        or runtime.hardware_identity_sha256 != hardware_identity_sha256
    ):
        reasons.append("hardware-identity-mismatch-or-missing")
    if runtime.condition_id != condition_id:
        reasons.append("condition-mismatch")
    if runtime.counter_deltas["capture_read_failures"]:
        reasons.append("runtime-read-failures")
    if runtime.counter_deltas["capture_restarts"]:
        reasons.append("runtime-capture-restarts")
    for field in (
        "width",
        "height",
        "fps",
        "pixel_format",
        "canvas_width",
        "canvas_height",
    ):
        if runtime.requested[field] != requested[field]:
            reasons.append(f"runtime-requested-{field}-mismatch")
    for field in (
        "width",
        "height",
        "backend",
        "pixel_format",
        "delivered_width",
        "delivered_height",
    ):
        if runtime.negotiated[field] != negotiated[field]:
            reasons.append(f"runtime-negotiated-{field}-mismatch")
    runtime_reported_fps = cast(float, runtime.negotiated["fps_reported"])
    local_reported_fps = negotiated.get("fps_reported")
    if not isinstance(local_reported_fps, (int, float)) or not math.isclose(
        runtime_reported_fps,
        float(local_reported_fps),
        rel_tol=1e-6,
        # Normal status evidence exposes this property to two decimals.
        abs_tol=0.005,
    ):
        reasons.append("runtime-negotiated-fps_reported-mismatch")
    return {
        "provided": True,
        "compatible": not reasons,
        "compatibility_reasons": reasons,
        "evidence_sha256": runtime.digest_sha256,
        "condition_id": runtime.condition_id,
        "run_id": runtime.run_id,
        "requested": runtime.requested,
        "negotiated": runtime.negotiated,
        "window_seconds": _round(runtime.duration_seconds),
        "capture_pacing": {
            "fps": runtime.capture_fps,
            "target_fps": runtime.requested["fps"],
            "successful_reads": runtime.counter_deltas["capture_frames_read"],
            "slot_overwrites": runtime.counter_deltas["capture_dropped_frames"],
            "read_failures": runtime.counter_deltas["capture_read_failures"],
            "restarts": runtime.counter_deltas["capture_restarts"],
        },
        "processed_frame_pacing": {
            "measured": True,
            "reason": "delta-unique-inputs-consumed-over-bounded-runtime-window",
            "unique_fps": runtime.processed_unique_fps,
            "output_fps": runtime.output_fps,
            "processing_deadline_misses": runtime.counter_deltas[
                "processing_deadline_misses"
            ],
            "frame_processing_ms": runtime.timings_ms["frame_processing"],
            "stages_ms": {
                "capture_read": runtime.timings_ms["capture_read"],
                "segmentation": runtime.timings_ms["segmentation"],
                "background": runtime.timings_ms["background"],
                "color_correction": runtime.timings_ms["color_correction"],
                "composite": runtime.timings_ms["composite"],
                "output_send": runtime.timings_ms["output_send"],
            },
        },
    }


def _diagnosis(
    *,
    requested: Mapping[str, object],
    negotiated: Mapping[str, object],
    timing: Mapping[str, object],
    measurement_read_failures: int,
    measurement_restarts: int,
    measurement_geometry_transitions: int,
    stalled_at_end: bool,
    capture_error: str | None,
    close_error: str | None,
    native: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    target_fps = float(cast(int, requested["fps"]))
    threshold_fps = target_fps * TARGET_RATIO
    active_fps_raw = timing["active_capture_fps"]
    wall_fps_raw = timing["wall_completion_fps"]
    active_fps = (
        float(cast(float, active_fps_raw)) if active_fps_raw is not None else None
    )
    wall_fps = float(cast(float, wall_fps_raw)) if wall_fps_raw is not None else None
    window = cast(Mapping[str, object], timing["measurement_window"])
    availability_raw = window.get("availability_fps")
    availability_fps = (
        float(cast(float, availability_raw)) if availability_raw is not None else None
    )
    measured_rates = [
        value for value in (availability_fps, active_fps, wall_fps) if value is not None
    ]
    measured_fps = min(measured_rates) if measured_rates else None
    reported_fps = negotiated.get("fps_reported")
    reported_mismatch = isinstance(reported_fps, (int, float)) and target_fps - float(
        reported_fps
    ) > max(1.0, target_fps * 0.1)
    dimensions_match = (
        negotiated.get("width") == requested["width"]
        and negotiated.get("height") == requested["height"]
        and negotiated.get("delivered_width") == requested["width"]
        and negotiated.get("delivered_height") == requested["height"]
    )
    unstable = bool(
        capture_error
        or close_error
        or measurement_read_failures
        or measurement_restarts
        or measurement_geometry_transitions
        or stalled_at_end
        or cast(int, timing["observer_missing_sample_count"]) > 0
        or cast(int, timing["generation_count"]) > 1
        or cast(int, timing["geometry_generation_count"]) > 1
    )
    duration_raw = window.get("duration_seconds")
    duration_seconds = (
        float(cast(float, duration_raw)) if duration_raw is not None else None
    )
    leading_gap_raw = window.get("leading_gap_ms")
    trailing_gap_raw = window.get("trailing_gap_ms")
    leading_gap_ms = (
        float(cast(float, leading_gap_raw)) if leading_gap_raw is not None else None
    )
    trailing_gap_ms = (
        float(cast(float, trailing_gap_raw)) if trailing_gap_raw is not None else None
    )
    interval_summary = cast(Mapping[str, object], timing["interval_ms"])
    interval_p95_raw = interval_summary.get("p95")
    interval_p95 = (
        float(cast(float, interval_p95_raw)) if interval_p95_raw is not None else None
    )
    maximum_gap_raw = timing.get("maximum_completion_gap_ms")
    maximum_gap_ms = (
        float(cast(float, maximum_gap_raw)) if maximum_gap_raw is not None else None
    )
    frame_budget_ms = 1000.0 / target_fps
    boundary_tolerance_ms = max(100.0, frame_budget_ms * 3.0)
    minimum_samples = (
        math.ceil(
            threshold_fps * duration_seconds
            - max(1e-9, threshold_fps * duration_seconds * 1e-9)
        )
        if duration_seconds is not None
        else 0
    )
    window_complete = bool(
        duration_seconds is not None
        and duration_seconds >= MIN_MEASUREMENT_SECONDS
        and cast(int, timing["sample_count"]) >= 2
        and leading_gap_ms is not None
        and leading_gap_ms <= boundary_tolerance_ms
        and trailing_gap_ms is not None
        and trailing_gap_ms <= boundary_tolerance_ms
    )
    sustained_window = bool(
        window_complete
        and availability_fps is not None
        and _rate_meets(availability_fps, threshold_fps)
        and cast(int, timing["source_success_count"]) >= minimum_samples
        and cast(int, timing["observer_missing_sample_count"]) == 0
        and interval_p95 is not None
        and interval_p95 <= frame_budget_ms * 2.0
        and maximum_gap_ms is not None
        and maximum_gap_ms <= boundary_tolerance_ms
    )
    target_sustained = bool(
        availability_fps is not None
        and _rate_meets(availability_fps, threshold_fps)
        and active_fps is not None
        and _rate_meets(active_fps, threshold_fps)
        and wall_fps is not None
        and _rate_meets(wall_fps, threshold_fps)
        and sustained_window
        and dimensions_match
        and not unstable
    )

    normalization = cast(Mapping[str, object], timing["normalization_ms"])
    read_timing = cast(Mapping[str, object], timing["read_ms"])
    normalization_p95 = normalization.get("p95")
    read_p50 = read_timing.get("p50")
    native_compatible = native.get("compatible") is True
    native_summary = native.get("summary")
    native_fps = (
        cast(Mapping[str, object], native_summary).get("capture_fps")
        if isinstance(native_summary, dict)
        else None
    )
    native_maximum_gap_ms = (
        cast(Mapping[str, object], native_summary).get("maximum_completion_gap_ms")
        if isinstance(native_summary, dict)
        else None
    )

    code: str
    confidence: str
    actionable = False
    actions: list[str]
    if unstable:
        code = "unstable-capture"
        confidence = "high"
        actions = [
            "resolve-read-failures-restarts-or-worker-shutdown-before-rate-attribution"
        ]
    elif reported_mismatch or not dimensions_match:
        code = "driver-reported-mode-mismatch"
        confidence = "high"
        actionable = True
        actions = [
            "select-a-hardware-supported-mode-or-explicit-mjpg-policy",
            "repeat-with-the-same-mode-in-a-native-capture-tool",
        ]
    elif target_sustained:
        code = "target-sustained"
        confidence = "high"
        actionable = True
        actions = [
            "compare-a-matched-full-processing-run-with-capture-pacing-kept-separate"
        ]
    elif (
        measured_fps is None
        or duration_seconds is None
        or duration_seconds < MIN_MEASUREMENT_SECONDS
        or cast(int, timing["sample_count"]) < 2
    ):
        code = "insufficient-capture-samples"
        confidence = "high"
        actions = ["repeat-a-five-second-or-longer-uninterrupted-measurement"]
    elif not window_complete:
        code = "capture-window-starvation"
        confidence = "high"
        actions = [
            "inspect-leading-or-trailing-capture-gaps-and-repeat-the-matched-run"
        ]
    elif native_compatible and isinstance(native_fps, (int, float)):
        actionable = True
        if (
            _rate_meets(float(native_fps), threshold_fps)
            and isinstance(native_maximum_gap_ms, (int, float))
            and float(native_maximum_gap_ms) <= boundary_tolerance_ms
        ):
            code = "opencv-capture-path-limited"
            confidence = "high"
            actions = [
                "compare-qualified-opencv-backend-and-pixel-format-rows",
                "inspect-read-normalization-and-host-cpu-timings",
            ]
        else:
            code = "shared-device-environment-or-backend-limit"
            confidence = "high"
            actions = [
                "repeat-under-adequate-diffuse-lighting-with-controls-preserved",
                "test-a-lower-bandwidth-mode-and-the-supported-mjpg-mode",
                "inspect-native-backend-and-usb-device-evidence",
            ]
    elif (
        isinstance(normalization_p95, (int, float))
        and float(normalization_p95) >= frame_budget_ms * 0.25
    ):
        code = "normalization-budget-limited"
        confidence = "medium"
        actionable = True
        actions = [
            "use-matching-capture-and-canvas-dimensions-or-reduce-qualified-resolution"
        ]
    elif (
        isinstance(read_p50, (int, float)) and float(read_p50) >= frame_budget_ms * 0.8
    ):
        code = "capture-read-path-paced"
        confidence = "high"
        actions = [
            "supply-same-device-native-host-completion-timestamps",
            "repeat-reported-and-adequate-lighting-conditions-with-controls-preserved",
            "compare-mjpg-and-backend-default-without-changing-processing",
        ]
    else:
        code = "unresolved-capture-scheduling-or-backend-limit"
        confidence = "medium"
        actions = [
            "supply-same-device-native-host-completion-timestamps",
            "compare-capture-only-against-a-matched-full-processing-status",
        ]

    runtime_capture = cast(Mapping[str, object] | None, runtime.get("capture_pacing"))
    runtime_capture_fps = (
        runtime_capture.get("fps") if runtime_capture is not None else None
    )
    runtime_comparison = "not-compared"
    if runtime.get("compatible") is True and isinstance(
        runtime_capture_fps, (int, float)
    ):
        runtime_target_met = _rate_meets(
            float(runtime_capture_fps),
            threshold_fps,
        )
        if target_sustained and not runtime_target_met:
            runtime_comparison = "full-runtime-capture-regressed"
        elif (
            measured_fps is not None
            and not _rate_meets(measured_fps, threshold_fps)
            and not runtime_target_met
        ):
            runtime_comparison = "capture-under-rate-with-processing-absent"
        elif target_sustained and runtime_target_met:
            runtime_comparison = "capture-target-sustained-in-both"
        else:
            runtime_comparison = "mixed-or-insufficient"

    return {
        "code": code,
        "confidence": confidence,
        "actionable": actionable,
        "target_sustained": target_sustained,
        "target_threshold_fps": _round(threshold_fps),
        "capture_only_fps": _round(measured_fps),
        "availability_fps": _round(availability_fps),
        "measurement_window_complete": window_complete,
        "measurement_window_sustained": sustained_window,
        "measurement_boundary_tolerance_ms": _round(boundary_tolerance_ms),
        "minimum_successful_reads": minimum_samples,
        "segmentation_executed": False,
        "segmentation_causal_for_capture_only_result": False,
        "runtime_comparison": runtime_comparison,
        "actions": actions,
        "non_claims": [
            "opencv-read-time-does-not-separate-device-transfer-and-decode",
            "auto-exposure-observation-does-not-prove-low-light-causality",
            "capture-recovery-does-not-prove-processed-frame-budget-recovery",
        ],
    }


def _health_mode(health: CaptureHealth) -> dict[str, object]:
    return {
        "backend": _safe_backend_label(health.backend),
        "pixel_format": _safe_fourcc(health.fourcc),
        "width": health.width,
        "height": health.height,
        "fps_reported": _round(health.fps_reported),
        "delivered_width": health.delivered_width,
        "delivered_height": health.delivered_height,
        "oriented_width": health.oriented_width,
        "oriented_height": health.oriented_height,
        "normalized_width": health.normalized_width,
        "normalized_height": health.normalized_height,
    }


def build_capture_report(
    *,
    cfg: AppConfig,
    condition_id: str,
    hardware_verified: bool,
    device_identity_sha256: str,
    hardware_identity_sha256: str,
    warmup_seconds: float,
    measurement_seconds: float,
    actual_measurement_seconds: float,
    measurement_started_ns: int,
    measurement_finished_ns: int,
    baseline_health: CaptureHealth,
    final_health: CaptureHealth,
    samples: Sequence[CaptureTimingSample],
    harness_deliveries: int,
    process_cpu_ms: float | None,
    capture_error: str | None,
    close_error: str | None,
    native_evidence: _NativeEvidence | None,
    runtime_evidence: _RuntimeEvidence | None,
) -> dict[str, object]:
    window_seconds = (
        measurement_finished_ns - measurement_started_ns
    ) / 1_000_000_000.0
    if (
        not math.isfinite(window_seconds)
        or window_seconds < 0.0
        or abs(window_seconds - actual_measurement_seconds) > 0.001
    ):
        raise CaptureDiagnosticError("capture measurement duration is inconsistent")
    actual_measurement_seconds = window_seconds
    timing = _timing_report(
        samples,
        measurement_started_ns=measurement_started_ns,
        measurement_finished_ns=measurement_finished_ns,
    )
    requested = {
        "width": cfg.camera.width,
        "height": cfg.camera.height,
        "fps": cfg.camera.fps,
        "pixel_format": cfg.camera.pixel_format,
        "mode_mismatch": cfg.camera.mode_mismatch,
        "canvas_width": resolved_output_size(cfg)[0],
        "canvas_height": resolved_output_size(cfg)[1],
    }
    negotiated = _health_mode(final_health)
    native = _native_summary(
        native_evidence,
        requested=requested,
        negotiated=negotiated,
        condition_id=condition_id,
        device_identity_sha256=device_identity_sha256,
        hardware_identity_sha256=hardware_identity_sha256,
    )
    runtime = _runtime_summary(
        runtime_evidence,
        requested=requested,
        negotiated=negotiated,
        condition_id=condition_id,
        device_identity_sha256=device_identity_sha256,
        hardware_identity_sha256=hardware_identity_sha256,
    )
    measurement_read_failures = max(
        0,
        final_health.read_failures - baseline_health.read_failures,
    )
    measurement_restarts = max(
        0,
        final_health.restarts - baseline_health.restarts,
    )
    measurement_geometry_transitions = max(
        0,
        final_health.geometry_transitions
        - max(1, baseline_health.geometry_transitions),
    )
    diagnosis = _diagnosis(
        requested=requested,
        negotiated=negotiated,
        timing=timing,
        measurement_read_failures=measurement_read_failures,
        measurement_restarts=measurement_restarts,
        measurement_geometry_transitions=measurement_geometry_transitions,
        stalled_at_end=final_health.stalled,
        capture_error=capture_error,
        close_error=close_error,
        native=native,
        runtime=runtime,
    )
    exact_acceptance_mode = (
        cfg.camera.width == 1280 and cfg.camera.height == 720 and cfg.camera.fps == 30
    )
    exact_mode_verified = (
        negotiated["width"] == 1280
        and negotiated["height"] == 720
        and negotiated["delivered_width"] == 1280
        and negotiated["delivered_height"] == 720
        and isinstance(negotiated["fps_reported"], (int, float))
        and _rate_meets(float(cast(float, negotiated["fps_reported"])), 27.0)
    )
    actionable_limitation = diagnosis["actionable"] and diagnosis["code"] in {
        "driver-reported-mode-mismatch",
        "opencv-capture-path-limited",
        "normalization-budget-limited",
    }
    native_corroborated = native["compatible"] is True
    source_kind = _physical_source_kind(cfg.camera.device)
    identity_bound = bool(device_identity_sha256 and hardware_identity_sha256)
    acceptance_satisfied = bool(
        hardware_verified
        and identity_bound
        and source_kind != "unsupported-file-stream-or-device-string"
        and exact_acceptance_mode
        and (
            (
                exact_mode_verified
                and diagnosis["code"] == "target-sustained"
                and actual_measurement_seconds >= MIN_MEASUREMENT_SECONDS
            )
            or (
                actionable_limitation
                and (
                    diagnosis["code"] == "driver-reported-mode-mismatch"
                    or native_corroborated
                )
            )
        )
    )
    if acceptance_satisfied and diagnosis["code"] == "target-sustained":
        qualification_outcome = "hardware-target-sustained"
    elif acceptance_satisfied:
        qualification_outcome = "hardware-actionable-limitation"
    else:
        qualification_outcome = "hardware-evidence-required"

    process_cpu_percent = (
        process_cpu_ms / (actual_measurement_seconds * 1000.0) * 100.0
        if process_cpu_ms is not None and actual_measurement_seconds > 0.0
        else None
    )
    measured_successes = cast(int, timing["source_success_count"])
    return {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "privacy": {
            "contains_pixels": False,
            "contains_frame_hashes": False,
            "contains_wall_clock_timestamps": False,
            "contains_device_path_or_index": False,
            "contains_only_opaque_identity_bindings": True,
            "contains_credentials": False,
            "timing_trace_uses_relative_monotonic_offsets": True,
        },
        "capture_only_contract": {
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
        },
        "condition": {
            "id": condition_id,
            "hardware_verified": hardware_verified,
            "source_kind": source_kind,
            "physical_source_eligible": (
                source_kind != "unsupported-file-stream-or-device-string"
            ),
            "device_identity_sha256": device_identity_sha256 or None,
            "hardware_identity_sha256": hardware_identity_sha256 or None,
            "device_identity_bound": bool(device_identity_sha256),
            "hardware_identity_bound": bool(hardware_identity_sha256),
        },
        "requested": requested,
        "negotiated": negotiated,
        "camera_controls": final_health.camera_controls.as_dict(),
        "measurement": {
            "warmup_seconds_requested": _round(warmup_seconds),
            "measurement_seconds_requested": _round(measurement_seconds),
            "measurement_seconds_actual": _round(actual_measurement_seconds),
            "successful_reads": measured_successes,
            "harness_deliveries": harness_deliveries,
            "latest_slot_overwrites": max(
                0,
                final_health.dropped_frames - baseline_health.dropped_frames,
            ),
            "read_failures": measurement_read_failures,
            "restarts": measurement_restarts,
            "geometry_transitions": measurement_geometry_transitions,
            "warmup_read_failures": baseline_health.read_failures,
            "warmup_restarts": baseline_health.restarts,
            "warmup_geometry_transitions": baseline_health.geometry_transitions,
            "stalled_at_end": final_health.stalled,
            "capture_error": capture_error,
            "close_error": close_error,
            "process_cpu_ms": _round(process_cpu_ms),
            "process_cpu_percent_of_one_core": _round(process_cpu_percent),
        },
        "timing": timing,
        "pacing": {
            "capture": {
                "measured": cast(int, timing["sample_count"]) >= 2,
                "active_source_fps": timing["active_capture_fps"],
                "wall_completion_fps": timing["wall_completion_fps"],
                "target_fps": cfg.camera.fps,
            },
            "processed_frames": runtime["processed_frame_pacing"],
            "output": {
                "measured_by_capture_only_harness": False,
                "reason": "no-output-sink-was-opened",
            },
        },
        "native_comparison": native,
        "full_runtime_comparison": runtime,
        "diagnosis": diagnosis,
        "qualification": {
            "backlog_acceptance_mode": "1280x720@30",
            "minimum_unique_fps": 27.0,
            "exact_acceptance_mode_requested": exact_acceptance_mode,
            "exact_mode_verified": exact_mode_verified,
            "hardware_verified": hardware_verified,
            "acceptance_satisfied": acceptance_satisfied,
            "outcome": qualification_outcome,
            "note": (
                "Synthetic or CI timing validates the harness but cannot qualify "
                "physical-camera cadence."
            ),
        },
    }


def _error_code(exc: BaseException) -> str:
    """Return a bounded error category without native exception text."""

    name = type(exc).__name__
    return name if _SAFE_TOOL_RE.fullmatch(name) else "CaptureRuntimeError"


def _cli_error(exc: BaseException) -> str:
    """Return one path/value-free CLI failure line."""

    if isinstance(exc, (CaptureDiagnosticError, MatteDiagnosticsError)):
        return str(exc).splitlines()[0]
    if isinstance(exc, PermissionError):
        return "an evidence or output path is not private to its owner"
    if isinstance(exc, OSError):
        return f"I/O operation failed ({type(exc).__name__})"
    if isinstance(exc, ValidationError):
        errors = exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        location = ""
        if errors:
            raw_location = errors[0].get("loc", ())
            if raw_location:
                first = raw_location[0]
                if isinstance(first, str) and first in AppConfig.model_fields:
                    location = first
                    if first == "camera" and len(raw_location) > 1:
                        second = raw_location[1]
                        if (
                            isinstance(second, str)
                            and second in CameraConfig.model_fields
                        ):
                            location = f"{first}.{second}"
        prefix = f"{location}: " if location else ""
        remaining = len(errors) - 1
        suffix = f" (+{remaining} more error(s))" if remaining > 0 else ""
        return f"{prefix}configuration value is invalid{suffix}"
    return "configuration is invalid"


def run_capture_only(
    cfg: AppConfig,
    *,
    condition_id: str = "default",
    hardware_verified: bool = False,
    device_identity_sha256: str = "",
    hardware_identity_sha256: str = "",
    warmup_seconds: float = 2.0,
    measurement_seconds: float = 10.0,
    native_evidence: _NativeEvidence | None = None,
    runtime_evidence: _RuntimeEvidence | None = None,
    capture_factory: CaptureFactory = _open_diagnostic_capture,
) -> dict[str, object]:
    """Run the production capture reader with every processing stage absent."""

    condition_id = _safe_id(condition_id, name="condition ID")
    warmup_seconds = _finite(
        warmup_seconds,
        name="warmup seconds",
        minimum=0.0,
        maximum=MAX_WARMUP_SECONDS,
    )
    measurement_seconds = _finite(
        measurement_seconds,
        name="measurement seconds",
        minimum=MIN_MEASUREMENT_SECONDS,
        maximum=MAX_MEASUREMENT_SECONDS,
    )
    if device_identity_sha256:
        device_identity_sha256 = _sha256_id(
            device_identity_sha256,
            name="device identity digest",
        )
    if hardware_identity_sha256:
        hardware_identity_sha256 = _sha256_id(
            hardware_identity_sha256,
            name="hardware identity digest",
        )
    if type(hardware_verified) is not bool:
        raise CaptureDiagnosticError("hardware_verified must be boolean")
    if cfg.camera.synthetic:
        raise CaptureDiagnosticError(
            "capture diagnosis requires a physical camera; synthetic timing "
            "cannot qualify hardware"
        )
    source_kind = _physical_source_kind(cfg.camera.device)
    if source_kind == "unsupported-file-stream-or-device-string":
        raise CaptureDiagnosticError(
            "capture diagnosis accepts only a numeric camera index or a "
            "recognized local camera-device path; media files and streams are "
            "not physical-camera evidence"
        )

    source: CaptureSource | None = None
    baseline_health = CaptureHealth()
    final_health = CaptureHealth()
    measurement_health: CaptureHealth | None = None
    measurement_started_ns = 0
    measurement_finished_ns = 0
    process_cpu_started_ns: int | None = None
    process_cpu_finished_ns: int | None = None
    harness_deliveries = 0
    capture_error: str | None = None
    close_error: str | None = None
    samples: tuple[CaptureTimingSample, ...] = ()
    try:
        source = capture_factory(cfg.camera, resolved_output_size(cfg))

        warmup_deadline = time.monotonic() + warmup_seconds
        while time.monotonic() < warmup_deadline:
            try:
                source.read()
            except CaptureError as exc:
                capture_error = _error_code(exc)
                break
            time.sleep(POLL_SECONDS)
        baseline_health = source.health_snapshot()

        measurement_started_ns = time.monotonic_ns()
        process_cpu_started_ns = time.process_time_ns()
        deadline_ns = measurement_started_ns + int(measurement_seconds * 1e9)
        while capture_error is None and time.monotonic_ns() < deadline_ns:
            try:
                frame = source.read()
            except CaptureError as exc:
                capture_error = _error_code(exc)
                break
            if frame is not None:
                harness_deliveries += 1
            time.sleep(POLL_SECONDS)
        measurement_finished_ns = time.monotonic_ns()
        process_cpu_finished_ns = time.process_time_ns()
        measurement_health = source.health_snapshot()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        capture_error = _error_code(exc)
        if measurement_started_ns == 0:
            measurement_started_ns = time.monotonic_ns()
        measurement_finished_ns = time.monotonic_ns()
    finally:
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                close_error = _error_code(exc)
            try:
                closed_health = source.health_snapshot()
                final_health = measurement_health or closed_health
                samples = tuple(
                    sample
                    for sample in source.timing_samples(
                        after_sequence=baseline_health.frames_read
                    )
                    if (
                        measurement_started_ns
                        <= sample.captured_at_ns
                        <= measurement_finished_ns
                    )
                )
            except Exception as exc:
                if capture_error is None:
                    capture_error = _error_code(exc)

    if measurement_finished_ns == 0:
        measurement_finished_ns = time.monotonic_ns()
    actual_seconds = max(
        0.0,
        (measurement_finished_ns - measurement_started_ns) / 1_000_000_000.0,
    )
    process_cpu_ms = (
        None
        if process_cpu_started_ns is None
        or process_cpu_finished_ns is None
        or process_cpu_finished_ns < process_cpu_started_ns
        else (process_cpu_finished_ns - process_cpu_started_ns) / 1_000_000.0
    )
    return build_capture_report(
        cfg=cfg,
        condition_id=condition_id,
        hardware_verified=hardware_verified,
        device_identity_sha256=device_identity_sha256,
        hardware_identity_sha256=hardware_identity_sha256,
        warmup_seconds=warmup_seconds,
        measurement_seconds=measurement_seconds,
        actual_measurement_seconds=actual_seconds,
        measurement_started_ns=measurement_started_ns,
        measurement_finished_ns=measurement_finished_ns,
        baseline_health=baseline_health,
        final_health=final_health,
        samples=samples,
        harness_deliveries=harness_deliveries,
        process_cpu_ms=process_cpu_ms,
        capture_error=capture_error,
        close_error=close_error,
        native_evidence=native_evidence,
        runtime_evidence=runtime_evidence,
    )


def report_markdown(report: Mapping[str, object]) -> str:
    diagnosis = cast(Mapping[str, object], report["diagnosis"])
    qualification = cast(Mapping[str, object], report["qualification"])
    requested = cast(Mapping[str, object], report["requested"])
    negotiated = cast(Mapping[str, object], report["negotiated"])
    timing = cast(Mapping[str, object], report["timing"])
    pacing = cast(Mapping[str, object], report["pacing"])
    processed = cast(Mapping[str, object], pacing["processed_frames"])
    measurement = cast(Mapping[str, object], report["measurement"])
    native = cast(Mapping[str, object], report["native_comparison"])
    lines = [
        "# Capture cadence diagnostic",
        "",
        f"- Diagnosis: `{diagnosis['code']}` ({diagnosis['confidence']} confidence)",
        f"- Capture-only FPS: `{diagnosis['capture_only_fps']}` "
        f"(target `{requested['fps']}`, threshold "
        f"`{diagnosis['target_threshold_fps']}`)",
        f"- Requested: `{requested['pixel_format']} "
        f"{requested['width']}x{requested['height']}@{requested['fps']}`",
        f"- Negotiated: `{negotiated['backend']}/{negotiated['pixel_format']} "
        f"{negotiated['width']}x{negotiated['height']}@"
        f"{negotiated['fps_reported']}`",
        f"- Successful reads: `{measurement['successful_reads']}`",
        f"- Native comparison compatible: `{native['compatible']}`",
        f"- Processed-frame pacing measured: `{processed['measured']}`",
        f"- MATTE-3.1 hardware outcome: `{qualification['outcome']}`",
        "",
        "## Timing boundaries",
        "",
        "| Boundary | p50 ms | p95 ms |",
        "| --- | ---: | ---: |",
    ]
    for label, key in (
        ("OpenCV read (pacing/transfer/decode)", "read_ms"),
        ("Pre-normalization/first-frame negotiation", "pre_normalization_ms"),
        ("Canonical normalization", "normalization_ms"),
        ("Slot publication", "publish_ms"),
        ("Reader total", "total_ms"),
    ):
        summary = cast(Mapping[str, object], timing[key])
        lines.append(f"| {label} | {summary['p50']} | {summary['p95']} |")
    lines.extend(
        [
            "",
            "Capture-only runs do not execute segmentation, backdrop, compositing, "
            "preview, API, or any output sink. The JSON report is authoritative. "
            "OpenCV `read()` combines device pacing, transfer, and backend decode; "
            "it does not prove low-light, USB, or decode causality by itself.",
            "",
            "## Next actions",
            "",
        ]
    )
    lines.extend(f"- `{action}`" for action in cast(list[str], diagnosis["actions"]))
    lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, object], output: Path | str) -> None:
    root = Path(output)
    _private_directory(root, create=True)
    _atomic_private_write(root / "capture.json", _json_bytes(report))
    _atomic_private_write(
        root / "capture.md",
        report_markdown(report).encode("utf-8"),
    )


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.load(args.config)
    values = cfg.to_dict()
    camera = cast(dict[str, object], values["camera"])
    for argument, field in (
        ("device", "device"),
        ("width", "width"),
        ("height", "height"),
        ("fps", "fps"),
        ("pixel_format", "pixel_format"),
        ("mode_mismatch", "mode_mismatch"),
    ):
        value = getattr(args, argument)
        if value is not None:
            camera[field] = value
    return AppConfig.from_dict(values)


def build_parser(*, prog: str = "custback capture-diagnose") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Measure production camera acquisition/normalization with every "
            "processing and output stage absent"
        ),
    )
    parser.add_argument("--config", help="runtime YAML used only for camera/canvas")
    parser.add_argument("--device", help="camera device override (never serialized)")
    parser.add_argument("--width", type=int, help="camera width override")
    parser.add_argument("--height", type=int, help="camera height override")
    parser.add_argument("--fps", type=int, help="camera FPS override")
    parser.add_argument(
        "--pixel-format",
        choices=("auto", "mjpeg", "backend"),
        help="camera pixel-format override",
    )
    parser.add_argument(
        "--mode-mismatch",
        choices=("warn", "error"),
        help="negotiated-mode mismatch policy override",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="discarded reader warm-up (0..15; default 2)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=10.0,
        help="measurement duration (5..60; default 10)",
    )
    parser.add_argument(
        "--condition-id",
        default="default",
        help="bounded comparison condition ID",
    )
    parser.add_argument(
        "--hardware-verified",
        action="store_true",
        help="attest that the requested physical-camera mode is hardware-supported",
    )
    parser.add_argument(
        "--device-identity-sha256",
        default="",
        help="opaque same-device binding for native comparison",
    )
    parser.add_argument(
        "--hardware-identity-sha256",
        default="",
        help="opaque same-host binding for native comparison",
    )
    parser.add_argument(
        "--native-evidence",
        help="owner-only native-tool host-completion evidence JSON",
    )
    parser.add_argument(
        "--runtime-evidence",
        help="owner-only two-snapshot full-runtime cadence evidence JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="new owner-only content-free report directory",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback capture-diagnose",
) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        cfg = _config_from_args(args)
        native = (
            load_native_evidence(args.native_evidence) if args.native_evidence else None
        )
        runtime = (
            load_runtime_evidence(args.runtime_evidence)
            if args.runtime_evidence
            else None
        )
        report = run_capture_only(
            cfg,
            condition_id=args.condition_id,
            hardware_verified=args.hardware_verified,
            device_identity_sha256=args.device_identity_sha256,
            hardware_identity_sha256=args.hardware_identity_sha256,
            warmup_seconds=args.warmup_seconds,
            measurement_seconds=args.duration_seconds,
            native_evidence=native,
            runtime_evidence=runtime,
        )
        write_report(report, args.output)
    except (
        CaptureDiagnosticError,
        MatteDiagnosticsError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"{prog}: {_cli_error(exc)}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"{prog}: capture diagnosis failed ({_error_code(exc)})",
            file=sys.stderr,
        )
        return 2
    diagnosis = cast(dict[str, Any], report["diagnosis"])
    print(
        f"capture-only diagnosis {diagnosis['code']}; "
        f"measured {diagnosis['capture_only_fps']} fps"
    )
    return 0 if diagnosis["code"] == "target-sustained" else 1


if __name__ == "__main__":
    raise SystemExit(main())
