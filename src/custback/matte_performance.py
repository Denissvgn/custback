"""Fixed-replay MATTE-3.4 compositor and complete-service qualification.

This module deliberately separates two kinds of evidence:

* a fixed 1280x720 compositor matrix that exercises every foreground/wrap
  combination in both supported blend spaces; and
* a content-free, model-backed RVM/CUDA service sidecar collected around the
  complete non-pacing unique-frame boundary.

Generated samples are useful for testing this validator, but can never qualify
the advertised path.  Output repeats are not accepted as unique-frame work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import queue
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence, cast

import numpy as np

from .acceleration import collect_cuda_device_identity
from .color import _bgr_u8_to_linear_bgr_prevalidated
from .compositor import (
    COMPOSITOR_P95_SUB_BUDGET_MS,
    COMPOSITOR_SUBSTAGE_NAMES,
    LegacyCompositorWorkspace,
    PreparedLightWrap,
    _composite_linear_bgr_prevalidated,
    composite,
    prepare_light_wrap,
)
from .light_wrap import LightWrapFrameContext, LightWrapStabilizer
from .segmentation import RVM_MODEL

REPORT_SCHEMA = "custback.matte-performance-report"
REPORT_VERSION = 1
ROW_SCHEMA = "custback.matte-compositor-performance-row"
ROW_VERSION = 1
FULL_PATH_SCHEMA = "custback.matte-full-path-performance-evidence"
FULL_PATH_VERSION = 1

FIXED_WIDTH = 1280
FIXED_HEIGHT = 720
FIXED_FPS = 30
COMPLETE_SERVICE_BUDGET_MS = 33.333334
SERIALIZED_DEADLINE_GRACE_MS = 1.0
MIN_WARMUP_FRAMES = 30
MIN_MEASURED_FRAMES = 300
MIN_UNIQUE_COMPOSITES_PER_S = 27.0
ARRIVAL_SWEEP_FPS = (30, 27, 24, 20, 15)
MAX_SAMPLES = 100_000
MAX_DURATION_MS = 3_600_000.0
MAX_RESOURCE_BYTES = 1 << 40
MAX_BANDWIDTH_BYTES_PER_S = 1 << 55
MAX_TIMESTAMP_NS = (1 << 63) - 1
# Default qualifying collection retains 330 raw/backdrop frame pairs
# (1,824,768,000 bytes) so disk reads cannot thermally pace the model. Keep a
# fixed ceiling above that default and fail before loading an oversized run.
MAX_FULL_PATH_PRELOAD_BYTES = 2 * 1024 * 1024 * 1024
# A default 330-frame matrix preload retains raw/backdrop/clean uint8 tracks
# plus float32 masks (3,953,664,000 bytes). Keep artifact I/O and correctness
# renders out of the measured tight loop while retaining a fixed failure bound.
MAX_MATRIX_PRELOAD_BYTES = 5 * 1024 * 1024 * 1024

TIMING_SAMPLE_NAMES = ("total", *COMPOSITOR_SUBSTAGE_NAMES)
ALLOCATION_SAMPLE_NAMES = (
    "known_transient_allocation_bytes",
    "retained_workspace_bytes",
)
SOURCE_KINDS = (
    "single-frame",
    "explicit-frame-sequence",
    "privacy-replay-bundle",
)
FULL_PATH_SAMPLE_NAMES = (
    "service_samples_ms",
    "non_pacing_cycle_samples_ms",
    "frame_processing_samples_ms",
    "serialized_new_frame_samples_ms",
    "rvm_preprocess_samples_ms",
    "rvm_inference_samples_ms",
    "rvm_postprocess_samples_ms",
    "background_selection_samples_ms",
    "compositor_samples_ms",
    "post_composite_validation_samples_ms",
    "sink_submission_samples_ms",
    "pacing_wait_samples_ms",
    "schedule_lateness_samples_ms",
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SERVICE_BOUNDARY = "unique-dequeue-through-sink-submit-excluding-deliberate-pacing"
_QUALIFICATION_SINK_IDS = ("pyvirtualcam", "native")
_BALANCED_MATRIX_ROW_ID = "srgb_legacy_both"
_BALANCED_EFFECTIVE_POLICY: dict[str, object] = {
    "resolved_rvm_downsample_ratio": 0.4,
    "raw_alpha_mode": "native_soft_alpha",
    "halo_mode": "mask_shift_only",
    "mask_shift_px": 0,
    "boundary_stabilization_mode": "off",
    "use_model_foreground": True,
    "light_wrap": 0.25,
    "light_wrap_stabilization_mode": "off",
    "blend_space": "srgb_legacy",
    "color_correction_mode": "off",
}


class MattePerformanceError(ValueError):
    """A performance evidence document is malformed or self-contradictory."""


@dataclass(frozen=True)
class CompositorVariant:
    """One required matrix cell."""

    id: str
    blend_space: str
    use_model_foreground: bool
    light_wrap: float


@dataclass(frozen=True)
class CompositorFrameSample:
    """One owner-provided frame quartet in a fixed replay sequence."""

    foreground_bgr: np.ndarray
    backdrop_bgr: np.ndarray
    mask: np.ndarray
    clean_foreground_bgr: np.ndarray
    capture_sequence: int
    capture_timestamp_ns: int


@dataclass(frozen=True)
class _ModelReplayFrame:
    """Only the resident inputs needed to rerun model-backed full service."""

    foreground_bgr: np.ndarray
    backdrop_bgr: np.ndarray
    capture_sequence: int
    capture_timestamp_ns: int
    capture_generation: int
    geometry_generation: int


@dataclass(frozen=True)
class _ReplaySource:
    kind: str
    frame_count: int
    source_sha256: str
    bundle_manifest_sha256: str | None
    lineage: tuple[tuple[int, int, int, int], ...]
    load: Callable[[int], CompositorFrameSample]
    load_model_frame: Callable[[int], _ModelReplayFrame]


_FEATURES = (
    ("plain", False, 0.0),
    ("model_foreground", True, 0.0),
    ("light_wrap", False, 0.25),
    ("both", True, 0.25),
)
COMPOSITOR_VARIANTS = tuple(
    CompositorVariant(
        id=f"{blend_space}_{name}",
        blend_space=blend_space,
        use_model_foreground=use_foreground,
        light_wrap=light_wrap,
    )
    for blend_space in ("srgb_legacy", "linear_srgb")
    for name, use_foreground, light_wrap in _FEATURES
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _strict_keys(
    value: object,
    expected: set[str] | frozenset[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise MattePerformanceError(
            f"{name} does not match the strict version-1 schema"
        )
    return cast(dict[str, Any], value)


def _safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise MattePerformanceError(f"{name} must be a bounded safe identifier")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise MattePerformanceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_RESOURCE_BYTES:
        raise MattePerformanceError(f"{name} must be a bounded non-negative integer")
    return int(value)


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise MattePerformanceError(f"{name} must be positive")
    return result


def _timestamp_ns(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_TIMESTAMP_NS:
        raise MattePerformanceError(
            f"{name} must be a bounded non-negative nanosecond timestamp"
        )
    return int(value)


def _finite_ms(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= MAX_DURATION_MS
    ):
        raise MattePerformanceError(
            f"{name} must be a bounded finite non-negative duration"
        )
    return float(value)


def _sample_values(
    value: object,
    name: str,
    *,
    expected_count: int | None = None,
) -> list[float]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_SAMPLES
        or (expected_count is not None and len(value) != expected_count)
    ):
        raise MattePerformanceError(f"{name} has an invalid sample count")
    return [_finite_ms(item, f"{name} sample") for item in value]


def _byte_samples(
    value: object,
    name: str,
    *,
    expected_count: int,
) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != expected_count
        or not value
        or len(value) > MAX_SAMPLES
    ):
        raise MattePerformanceError(f"{name} has an invalid sample count")
    return [_nonnegative_int(item, f"{name} sample") for item in value]


def _source_scope(
    value: object,
    *,
    warmup: int,
    measured: int,
) -> tuple[dict[str, object], bool]:
    scope = _strict_keys(
        value,
        {
            "kind",
            "source_frame_count",
            "profiled_frame_count",
            "unique_capture_sequence",
            "privacy_guarded_bundle",
            "bundle_manifest_sha256",
            "measured_frame_lineage_sha256",
            "execution_pacing",
        },
        "compositor source scope",
    )
    kind = scope["kind"]
    source_frames = _positive_int(
        scope["source_frame_count"],
        "compositor source frame count",
    )
    profiled_frames = _positive_int(
        scope["profiled_frame_count"],
        "compositor profiled frame count",
    )
    if (
        kind not in SOURCE_KINDS
        or profiled_frames != warmup + measured
        or type(scope["unique_capture_sequence"]) is not bool
        or type(scope["privacy_guarded_bundle"]) is not bool
        or not isinstance(scope["measured_frame_lineage_sha256"], str)
        or _DIGEST.fullmatch(scope["measured_frame_lineage_sha256"]) is None
        or scope["execution_pacing"] != "unpaced-tight-loop"
        or (kind == "single-frame" and source_frames != 1)
        or (scope["privacy_guarded_bundle"] is not (kind == "privacy-replay-bundle"))
        or (
            kind == "privacy-replay-bundle"
            and (
                not isinstance(scope["bundle_manifest_sha256"], str)
                or _DIGEST.fullmatch(scope["bundle_manifest_sha256"]) is None
            )
        )
        or (
            kind != "privacy-replay-bundle"
            and scope["bundle_manifest_sha256"] is not None
        )
    ):
        raise MattePerformanceError("compositor source scope is malformed")
    authoritative_sequence = (
        kind != "single-frame"
        and scope["unique_capture_sequence"] is True
        and source_frames >= 2
    )
    return dict(scope), authoritative_sequence


def _memory_bandwidth(
    value: object,
    name: str,
    *,
    expected_source_sha256: str,
) -> dict[str, object]:
    bandwidth = _strict_keys(
        value,
        {
            "available",
            "bytes_per_second",
            "counter_source",
            "source_sha256",
            "hardware_identity_sha256",
            "provider_environment_sha256",
            "measurement_run_sha256",
        },
        name,
    )
    available = bandwidth["available"]
    if type(available) is not bool:
        raise MattePerformanceError(f"{name}.available must be boolean")
    if not available:
        if (
            bandwidth["bytes_per_second"] is not None
            or bandwidth["counter_source"] is not None
            or bandwidth["source_sha256"] is not None
            or bandwidth["hardware_identity_sha256"] is not None
            or bandwidth["provider_environment_sha256"] is not None
            or bandwidth["measurement_run_sha256"] is not None
        ):
            raise MattePerformanceError(
                f"{name} must use null values when counters are unavailable"
            )
        return {
            "available": False,
            "bytes_per_second": None,
            "counter_source": None,
            "source_sha256": None,
            "hardware_identity_sha256": None,
            "provider_environment_sha256": None,
            "measurement_run_sha256": None,
        }
    bytes_per_second = bandwidth["bytes_per_second"]
    if (
        isinstance(bytes_per_second, bool)
        or not isinstance(bytes_per_second, (int, float))
        or not math.isfinite(float(bytes_per_second))
        or not 0.0 < float(bytes_per_second) <= float(MAX_BANDWIDTH_BYTES_PER_S)
    ):
        raise MattePerformanceError(
            f"{name}.bytes_per_second must be a bounded positive native measurement"
        )
    counter_source = _safe_id(
        bandwidth["counter_source"],
        f"{name}.counter_source",
    )
    source_sha256 = _digest(bandwidth["source_sha256"], f"{name}.source_sha256")
    hardware_identity_sha256 = _digest(
        bandwidth["hardware_identity_sha256"],
        f"{name}.hardware_identity_sha256",
    )
    provider_environment_sha256 = _digest(
        bandwidth["provider_environment_sha256"],
        f"{name}.provider_environment_sha256",
    )
    measurement_run_sha256 = _digest(
        bandwidth["measurement_run_sha256"],
        f"{name}.measurement_run_sha256",
    )
    if source_sha256 != expected_source_sha256:
        raise MattePerformanceError(f"{name} is not bound to its matrix source")
    return {
        "available": True,
        "bytes_per_second": round(float(bytes_per_second), 3),
        "counter_source": counter_source,
        "source_sha256": source_sha256,
        "hardware_identity_sha256": hardware_identity_sha256,
        "provider_environment_sha256": provider_environment_sha256,
        "measurement_run_sha256": measurement_run_sha256,
    }


def _balanced_effective_policy(value: object) -> dict[str, object]:
    policy = _strict_keys(
        value,
        set(_BALANCED_EFFECTIVE_POLICY),
        "full-path balanced effective policy",
    )
    ratio = policy["resolved_rvm_downsample_ratio"]
    mask_shift = policy["mask_shift_px"]
    light_wrap = policy["light_wrap"]
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or not 0.0 < float(ratio) <= 1.0
        or type(mask_shift) is not int
        or not -20 <= mask_shift <= 20
        or type(policy["use_model_foreground"]) is not bool
        or isinstance(light_wrap, bool)
        or not isinstance(light_wrap, (int, float))
        or not math.isfinite(float(light_wrap))
        or not 0.0 <= float(light_wrap) <= 1.0
    ):
        raise MattePerformanceError("full-path effective policy is malformed")
    for name in (
        "raw_alpha_mode",
        "halo_mode",
        "boundary_stabilization_mode",
        "light_wrap_stabilization_mode",
        "blend_space",
        "color_correction_mode",
    ):
        _safe_id(policy[name], f"full-path effective policy {name}")
    return {
        **dict(policy),
        "resolved_rvm_downsample_ratio": float(ratio),
        "mask_shift_px": mask_shift,
        "use_model_foreground": policy["use_model_foreground"],
        "light_wrap": float(light_wrap),
    }


def _nearest_rank(values: Sequence[float] | Sequence[int], percentile: float) -> float:
    if not values:
        raise MattePerformanceError("a percentile requires at least one sample")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def nearest_rank_summary(
    values: Sequence[float] | Sequence[int],
    *,
    digits: int = 6,
) -> dict[str, float | int]:
    """Summarize raw samples without relying on an interpolated percentile."""

    if not values:
        raise MattePerformanceError("performance summaries require samples")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": round(sum(numeric) / len(numeric), digits),
        "p50": round(_nearest_rank(numeric, 0.50), digits),
        "p95": round(_nearest_rank(numeric, 0.95), digits),
        "p99": round(_nearest_rank(numeric, 0.99), digits),
        "max": round(max(numeric), digits),
    }


def _variant_by_id(variant_id: str) -> CompositorVariant:
    for variant in COMPOSITOR_VARIANTS:
        if variant.id == variant_id:
            return variant
    raise MattePerformanceError("compositor row id is not in the required matrix")


def _normalize_equivalence(
    value: object,
    *,
    variant: CompositorVariant,
) -> tuple[dict[str, object], bool]:
    equivalence = _strict_keys(
        value,
        {
            "reference_contract",
            "max_channel_delta",
            "foreground_endpoint_exact",
            "background_endpoint_exact",
            "output_shape",
            "output_dtype",
            "output_c_contiguous",
            "deterministic_repeat_exact",
        },
        f"{variant.id} equivalence",
    )
    expected_reference = (
        "frozen-srgb-legacy-v1"
        if variant.blend_space == "srgb_legacy"
        else "linear-srgb-reference-v1"
    )
    delta = _nonnegative_int(
        equivalence["max_channel_delta"],
        f"{variant.id} maximum channel delta",
    )
    shape = equivalence["output_shape"]
    if (
        equivalence["reference_contract"] != expected_reference
        or not isinstance(shape, list)
        or shape != [FIXED_HEIGHT, FIXED_WIDTH, 3]
        or equivalence["output_dtype"] != "uint8"
        or any(
            type(equivalence[name]) is not bool
            for name in (
                "foreground_endpoint_exact",
                "background_endpoint_exact",
                "output_c_contiguous",
                "deterministic_repeat_exact",
            )
        )
    ):
        raise MattePerformanceError(
            f"{variant.id} output equivalence contract is malformed"
        )
    delta_limit = 0 if variant.blend_space == "srgb_legacy" else 1
    passed = (
        delta <= delta_limit
        and equivalence["foreground_endpoint_exact"] is True
        and equivalence["background_endpoint_exact"] is True
        and equivalence["output_c_contiguous"] is True
        and equivalence["deterministic_repeat_exact"] is True
    )
    return dict(equivalence), passed


def _normalize_matrix_row(value: object) -> dict[str, object]:
    row = _strict_keys(
        value,
        {
            "schema",
            "version",
            "id",
            "source_sha256",
            "source_scope",
            "evidence_kind",
            "blend_space",
            "use_model_foreground",
            "light_wrap",
            "warmup_frame_count",
            "measured_frame_count",
            "timing_samples_ms",
            "allocation_samples_bytes",
            "memory_bandwidth",
            "equivalence",
        },
        "compositor performance row",
    )
    if row["schema"] != ROW_SCHEMA or row["version"] != ROW_VERSION:
        raise MattePerformanceError("unsupported compositor performance row")
    variant_id = _safe_id(row["id"], "compositor row id")
    variant = _variant_by_id(variant_id)
    _digest(row["source_sha256"], f"{variant_id} source digest")
    if row["evidence_kind"] not in ("generated-proxy", "fixed-replay-measured"):
        raise MattePerformanceError(f"{variant_id} evidence kind is invalid")
    if (
        row["blend_space"] != variant.blend_space
        or row["use_model_foreground"] is not variant.use_model_foreground
        or isinstance(row["light_wrap"], bool)
        or not isinstance(row["light_wrap"], (int, float))
        or float(row["light_wrap"]) != variant.light_wrap
    ):
        raise MattePerformanceError(f"{variant_id} controls do not match its matrix id")
    warmup = _nonnegative_int(
        row["warmup_frame_count"], f"{variant_id} warm-up frame count"
    )
    measured = _positive_int(
        row["measured_frame_count"], f"{variant_id} measured frame count"
    )
    if measured > MAX_SAMPLES:
        raise MattePerformanceError(f"{variant_id} has too many measured frames")
    source_scope, source_sequence_complete = _source_scope(
        row["source_scope"],
        warmup=warmup,
        measured=measured,
    )

    raw_timings = _strict_keys(
        row["timing_samples_ms"],
        set(TIMING_SAMPLE_NAMES),
        f"{variant_id} timing samples",
    )
    timing_samples = {
        name: _sample_values(
            raw_timings[name],
            f"{variant_id}.{name}",
            expected_count=measured,
        )
        for name in TIMING_SAMPLE_NAMES
    }
    for index in range(measured):
        substage_total = sum(
            timing_samples[name][index] for name in COMPOSITOR_SUBSTAGE_NAMES
        )
        if substage_total > timing_samples["total"][index] + 1e-6:
            raise MattePerformanceError(
                f"{variant_id} substages exceed their enclosing total"
            )

    raw_allocations = _strict_keys(
        row["allocation_samples_bytes"],
        set(ALLOCATION_SAMPLE_NAMES),
        f"{variant_id} allocation samples",
    )
    allocation_samples = {
        name: _byte_samples(
            raw_allocations[name],
            f"{variant_id}.{name}",
            expected_count=measured,
        )
        for name in ALLOCATION_SAMPLE_NAMES
    }
    equivalence, equivalence_passed = _normalize_equivalence(
        row["equivalence"],
        variant=variant,
    )
    timing_summary = {
        name: nearest_rank_summary(samples) for name, samples in timing_samples.items()
    }
    allocation_summary = {
        name: nearest_rank_summary(samples, digits=3)
        for name, samples in allocation_samples.items()
    }
    memory_bandwidth = _memory_bandwidth(
        row["memory_bandwidth"],
        f"{variant_id} memory bandwidth",
        expected_source_sha256=str(row["source_sha256"]),
    )
    compositor_p95 = float(timing_summary["total"]["p95"])
    signed_headroom = round(
        COMPOSITOR_P95_SUB_BUDGET_MS - compositor_p95,
        6,
    )
    sample_floor_passed = (
        warmup >= MIN_WARMUP_FRAMES and measured >= MIN_MEASURED_FRAMES
    )
    measured_authoritative = row["evidence_kind"] == "fixed-replay-measured"
    sub_budget_passed = compositor_p95 <= COMPOSITOR_P95_SUB_BUDGET_MS
    outcome = (
        "rejected"
        if not equivalence_passed
        else (
            "qualified"
            if sample_floor_passed
            and measured_authoritative
            and source_sequence_complete
            and sub_budget_passed
            else (
                "rejected"
                if sample_floor_passed
                and measured_authoritative
                and source_sequence_complete
                and not sub_budget_passed
                else "not_decidable"
            )
        )
    )
    return {
        "schema": ROW_SCHEMA,
        "version": ROW_VERSION,
        "id": variant.id,
        "source_sha256": row["source_sha256"],
        "source_scope": source_scope,
        "evidence_kind": row["evidence_kind"],
        "blend_space": variant.blend_space,
        "use_model_foreground": variant.use_model_foreground,
        "light_wrap": variant.light_wrap,
        "warmup_frame_count": warmup,
        "measured_frame_count": measured,
        "timing_samples_ms": timing_samples,
        "timing_summary_ms": timing_summary,
        "allocation_samples_bytes": allocation_samples,
        "allocation_summary_bytes": allocation_summary,
        "allocation_measurement_scope": (
            "known-lower-bound-not-native-memory-bandwidth"
        ),
        "memory_bandwidth": memory_bandwidth,
        "equivalence": equivalence,
        "checks": {
            "sample_floor": sample_floor_passed,
            "fixed_replay_measured": measured_authoritative,
            "ordered_multi_frame_source_sequence": source_sequence_complete,
            "output_equivalence": equivalence_passed,
            "compositor_p95_sub_budget": sub_budget_passed,
        },
        "compositor_p95_ms": round(compositor_p95, 6),
        "signed_compositor_headroom_ms": signed_headroom,
        "outcome": outcome,
    }


def logical_arrival_sweep(
    cycle_samples_ms: Sequence[float],
) -> list[dict[str, object]]:
    """Project complete non-pacing cycles through explicit latest-only arrivals.

    The projection consumes the samples in their recorded order.  It never
    estimates capacity from ``1000 / p95``.  When processing is busy, the next
    cycle selects the newest logical arrival and reports skipped
    capture sequences separately.
    """

    samples = [
        _finite_ms(value, "logical arrival non-pacing cycle sample")
        for value in cycle_samples_ms
    ]
    if not samples or len(samples) > MAX_SAMPLES:
        raise MattePerformanceError("logical arrival sweep requires bounded samples")
    rows: list[dict[str, object]] = []
    for arrival_fps in ARRIVAL_SWEEP_FPS:
        interval_ms = 1000.0 / arrival_fps
        now_ms = 0.0
        previous_sequence = -1
        completion_times: list[float] = []
        queue_ages: list[float] = []
        sequence_gap_count = 0
        missing_input_count = 0
        for duration_ms in samples:
            newest_sequence = math.floor((now_ms + 1e-9) / interval_ms)
            if newest_sequence <= previous_sequence:
                newest_sequence = previous_sequence + 1
                now_ms = newest_sequence * interval_ms
            arrival_ms = newest_sequence * interval_ms
            queue_ages.append(max(0.0, now_ms - arrival_ms))
            if previous_sequence >= 0:
                missing = max(0, newest_sequence - previous_sequence - 1)
                if missing:
                    sequence_gap_count += 1
                    missing_input_count += missing
            now_ms += duration_ms
            completion_times.append(now_ms)
            previous_sequence = newest_sequence
        span_ms = (
            0.0
            if len(completion_times) < 2
            else completion_times[-1] - completion_times[0]
        )
        unique_fps = (
            None if span_ms <= 0.0 else 1000.0 * (len(completion_times) - 1) / span_ms
        )
        queue_growth = queue_ages[-1] - queue_ages[0]
        queue_age_not_growing = queue_growth <= 1e-6
        rows.append(
            {
                "arrival_fps": arrival_fps,
                "unique_composite_fps": (
                    None if unique_fps is None else round(unique_fps, 6)
                ),
                "logical_capture_sequence_gap_count": sequence_gap_count,
                "logical_capture_missing_input_count": missing_input_count,
                "maximum_queue_age_ms": round(max(queue_ages), 6),
                "ending_minus_starting_queue_age_ms": round(queue_growth, 6),
                "queue_age_bounded_to_one_arrival_interval": (
                    max(queue_ages) <= interval_ms + 1e-6
                ),
                "queue_age_not_growing": queue_age_not_growing,
                "sustained_without_sequence_gaps": (
                    sequence_gap_count == 0 and queue_age_not_growing
                ),
            }
        )
    return rows


def _logical_schedule_lateness_samples(
    service_samples_ms: Sequence[float],
    cycle_samples_ms: Sequence[float],
) -> list[float]:
    """Derive sink-submission lateness from non-pacing serialized cycles."""

    if len(service_samples_ms) != len(cycle_samples_ms):
        raise MattePerformanceError(
            "schedule lateness service/cycle sample counts do not match"
        )
    cycle_elapsed_ms = 0.0
    lateness: list[float] = []
    for index, (service_ms, cycle_ms) in enumerate(
        zip(service_samples_ms, cycle_samples_ms)
    ):
        deadline_ms = (index + 1) * 1000.0 / FIXED_FPS
        lateness.append(max(0.0, cycle_elapsed_ms + service_ms - deadline_ms))
        cycle_elapsed_ms += cycle_ms
    return lateness


def _normalize_full_path(
    value: object | None,
    *,
    expected_source_sha256: str,
    expected_source_scope: Mapping[str, object],
    matrix_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if value is None:
        return {
            "status": "missing",
            "outcome": "not_decidable",
            "reason": "model-backed fixed-replay full-service evidence is required",
            "evidence": None,
            "timing_summary_ms": None,
            "arrival_sweep": [],
            "sustainable_unique_frame_profile_fps": None,
            "signed_service_headroom_ms": None,
            "signed_compositor_headroom_ms": None,
            "checks": {},
        }
    evidence = _strict_keys(
        value,
        {
            "schema",
            "version",
            "evidence_kind",
            "source_sha256",
            "hardware_id",
            "hardware_identity_sha256",
            "provider_environment_sha256",
            "measurement_run_sha256",
            "rvm_model_sha256",
            "bundle_manifest_sha256",
            "measured_frame_lineage_sha256",
            "matrix_row_id",
            "effective_policy",
            "background_provider_scope",
            "sink_id",
            "sink_identity_sha256",
            "sink_submission_copy_in_service_boundary",
            "canvas",
            "backend",
            "provider",
            "fallback_count",
            "boundary",
            "warmup_frame_count",
            *FULL_PATH_SAMPLE_NAMES,
            "unique_composite_count",
            "model_invocation_count",
            "output_send_count",
            "output_repeat_count",
            "no_unread_repeat_count",
            "capture_slot_overwrite_count",
            "capture_sequence_gap_count",
            "capture_missing_input_count",
            "processing_deadline_miss_count",
            "serialized_new_frame_deadline_miss_count",
            "sink_recovery_count",
        },
        "full-path performance evidence",
    )
    if (
        evidence["schema"] != FULL_PATH_SCHEMA
        or evidence["version"] != FULL_PATH_VERSION
    ):
        raise MattePerformanceError("unsupported full-path performance evidence")
    evidence_kind = evidence["evidence_kind"]
    if evidence_kind not in ("generated-proxy", "model-backed"):
        raise MattePerformanceError("full-path evidence kind is invalid")
    source_sha256 = _digest(
        evidence["source_sha256"],
        "full-path source digest",
    )
    hardware_id = _safe_id(evidence["hardware_id"], "full-path hardware id")
    hardware_identity_sha256 = _digest(
        evidence["hardware_identity_sha256"],
        "full-path hardware identity",
    )
    provider_environment_sha256 = _digest(
        evidence["provider_environment_sha256"],
        "full-path provider environment",
    )
    measurement_run_sha256 = _digest(
        evidence["measurement_run_sha256"],
        "full-path measurement run",
    )
    rvm_model_sha256 = _digest(
        evidence["rvm_model_sha256"],
        "full-path RVM model",
    )
    expected_manifest = expected_source_scope["bundle_manifest_sha256"]
    bundle_manifest_value = evidence["bundle_manifest_sha256"]
    if expected_manifest is None:
        if bundle_manifest_value is not None:
            raise MattePerformanceError(
                "non-bundle full-path evidence cannot claim a bundle manifest"
            )
        bundle_manifest_sha256 = None
    else:
        bundle_manifest_sha256 = _digest(
            bundle_manifest_value,
            "full-path bundle manifest",
        )
    measured_frame_lineage_sha256 = _digest(
        evidence["measured_frame_lineage_sha256"],
        "full-path measured frame lineage",
    )
    matrix_row_id = _safe_id(
        evidence["matrix_row_id"],
        "full-path matrix row id",
    )
    _variant_by_id(matrix_row_id)
    effective_policy = _balanced_effective_policy(evidence["effective_policy"])
    if evidence["background_provider_scope"] != "resident-recorded-frame-copy":
        raise MattePerformanceError(
            "full-path background provider scope is not the fixed replay contract"
        )
    sink_id = _safe_id(evidence["sink_id"], "full-path sink id")
    if sink_id not in _QUALIFICATION_SINK_IDS:
        raise MattePerformanceError(
            "full-path sink id must name an explicit qualified production sink"
        )
    sink_identity_sha256 = _digest(
        evidence["sink_identity_sha256"],
        "full-path sink identity",
    )
    if type(evidence["sink_submission_copy_in_service_boundary"]) is not bool:
        raise MattePerformanceError(
            "full-path sink submission/copy boundary flag must be boolean"
        )
    canvas = _strict_keys(
        evidence["canvas"],
        {"width", "height", "nominal_fps"},
        "full-path canvas",
    )
    if canvas != {
        "width": FIXED_WIDTH,
        "height": FIXED_HEIGHT,
        "nominal_fps": FIXED_FPS,
    }:
        raise MattePerformanceError("full-path evidence is not fixed 1280x720@30")
    warmup = _nonnegative_int(
        evidence["warmup_frame_count"],
        "full-path warm-up frame count",
    )
    service_samples = _sample_values(
        evidence["service_samples_ms"],
        "full-path service",
    )
    measured = len(service_samples)
    normalized_samples: dict[str, list[float]] = {"service_samples_ms": service_samples}
    for name in FULL_PATH_SAMPLE_NAMES[1:]:
        normalized_samples[name] = _sample_values(
            evidence[name],
            f"full-path {name}",
            expected_count=measured,
        )
    cycle_samples = normalized_samples["non_pacing_cycle_samples_ms"]
    expected_lateness = _logical_schedule_lateness_samples(
        service_samples,
        cycle_samples,
    )
    if any(
        abs(observed - expected) > 1e-6
        for observed, expected in zip(
            normalized_samples["schedule_lateness_samples_ms"],
            expected_lateness,
        )
    ):
        raise MattePerformanceError(
            "full-path schedule lateness does not match ordered cycle samples"
        )
    for index, service_ms in enumerate(service_samples):
        timed_work = sum(
            normalized_samples[name][index]
            for name in (
                "rvm_preprocess_samples_ms",
                "rvm_inference_samples_ms",
                "rvm_postprocess_samples_ms",
                "background_selection_samples_ms",
                "compositor_samples_ms",
                "post_composite_validation_samples_ms",
                "sink_submission_samples_ms",
            )
        )
        if timed_work > service_ms + 1e-6:
            raise MattePerformanceError(
                "full-path substages exceed the enclosing service boundary"
            )
        frame_processing_ms = normalized_samples["frame_processing_samples_ms"][index]
        model_and_compositor_ms = sum(
            normalized_samples[name][index]
            for name in (
                "rvm_preprocess_samples_ms",
                "rvm_inference_samples_ms",
                "rvm_postprocess_samples_ms",
                "background_selection_samples_ms",
                "compositor_samples_ms",
            )
        )
        if (
            frame_processing_ms > service_ms + 1e-6
            or model_and_compositor_ms > frame_processing_ms + 1e-6
            or service_ms > cycle_samples[index] + 1e-6
            or service_ms
            > normalized_samples["serialized_new_frame_samples_ms"][index] + 1e-6
            or (
                normalized_samples["serialized_new_frame_samples_ms"][index]
                - normalized_samples["pacing_wait_samples_ms"][index]
                > cycle_samples[index] + 1e-6
            )
            or service_ms + normalized_samples["pacing_wait_samples_ms"][index]
            > normalized_samples["serialized_new_frame_samples_ms"][index] + 1e-6
        ):
            raise MattePerformanceError(
                "full-path frame-processing/service/serialized timing boundaries "
                "are inconsistent"
            )

    counters = {
        name: _nonnegative_int(evidence[name], f"full-path {name}")
        for name in (
            "unique_composite_count",
            "model_invocation_count",
            "output_send_count",
            "output_repeat_count",
            "no_unread_repeat_count",
            "capture_slot_overwrite_count",
            "capture_sequence_gap_count",
            "capture_missing_input_count",
            "processing_deadline_miss_count",
            "serialized_new_frame_deadline_miss_count",
            "sink_recovery_count",
            "fallback_count",
        )
    }
    expected_deadline_misses = sum(
        sample > COMPLETE_SERVICE_BUDGET_MS
        for sample in normalized_samples["frame_processing_samples_ms"]
    )
    if counters["processing_deadline_miss_count"] != expected_deadline_misses:
        raise MattePerformanceError(
            "full-path processing deadline misses do not match frame-processing samples"
        )
    expected_serialized_deadline_misses = sum(
        sample > COMPLETE_SERVICE_BUDGET_MS + SERIALIZED_DEADLINE_GRACE_MS
        for sample in normalized_samples["serialized_new_frame_samples_ms"]
    )
    if (
        counters["serialized_new_frame_deadline_miss_count"]
        != expected_serialized_deadline_misses
    ):
        raise MattePerformanceError(
            "full-path serialized deadline misses do not match serialized samples"
        )
    if (
        counters["unique_composite_count"] != measured
        or counters["model_invocation_count"] != measured
    ):
        raise MattePerformanceError(
            "fixed replay requires one model invocation per unique composite sample"
        )
    if (
        counters["output_send_count"]
        != counters["unique_composite_count"] + counters["output_repeat_count"]
        or counters["no_unread_repeat_count"] > counters["output_repeat_count"]
    ):
        raise MattePerformanceError(
            "full-path output repeat counters are internally inconsistent"
        )
    timing_summary = {
        name: nearest_rank_summary(samples)
        for name, samples in normalized_samples.items()
    }
    service_p95 = float(timing_summary["service_samples_ms"]["p95"])
    compositor_p95 = float(timing_summary["compositor_samples_ms"]["p95"])
    arrival_sweep = logical_arrival_sweep(cycle_samples)
    arrival_30 = next(row for row in arrival_sweep if row["arrival_fps"] == FIXED_FPS)
    simulated_sustainable = next(
        (
            int(cast(Any, row["arrival_fps"]))
            for row in arrival_sweep
            if row["sustained_without_sequence_gaps"] is True
        ),
        None,
    )
    sample_floor = warmup >= MIN_WARMUP_FRAMES and measured >= MIN_MEASURED_FRAMES
    source_identity_passed = (
        source_sha256 == expected_source_sha256
        and bundle_manifest_sha256 == expected_manifest
        and measured_frame_lineage_sha256
        == expected_source_scope["measured_frame_lineage_sha256"]
        and warmup + measured == expected_source_scope["profiled_frame_count"]
    )
    selected_matrix_row = matrix_rows.get(matrix_row_id)
    selected_matrix_checks = (
        {}
        if selected_matrix_row is None
        else cast(Mapping[str, object], selected_matrix_row["checks"])
    )
    selected_matrix_authoritative = selected_matrix_row is not None and all(
        selected_matrix_checks.get(name) is True
        for name in (
            "sample_floor",
            "fixed_replay_measured",
            "ordered_multi_frame_source_sequence",
            "output_equivalence",
        )
    )
    selected_matrix_qualified = (
        selected_matrix_row is not None
        and selected_matrix_row["outcome"] == "qualified"
    )
    exact_profile_span = (
        selected_matrix_row is not None
        and warmup == selected_matrix_row["warmup_frame_count"]
        and measured == selected_matrix_row["measured_frame_count"]
    )
    distinct_source_frames = (
        type(expected_source_scope.get("source_frame_count")) is int
        and cast(int, expected_source_scope["source_frame_count"]) >= warmup + measured
    )
    backend_passed = (
        evidence["backend"] == "rvm"
        and evidence["provider"] == "cuda"
        and rvm_model_sha256 == RVM_MODEL.sha256
        and counters["fallback_count"] == 0
    )
    opaque_identity_passed = (
        len(
            {
                hardware_identity_sha256,
                provider_environment_sha256,
                sink_identity_sha256,
            }
        )
        == 3
    )
    boundary_passed = evidence["boundary"] == _SERVICE_BOUNDARY
    balanced_policy_passed = (
        matrix_row_id == _BALANCED_MATRIX_ROW_ID
        and effective_policy == _BALANCED_EFFECTIVE_POLICY
        and evidence["sink_submission_copy_in_service_boundary"] is True
    )
    no_recovery = counters["sink_recovery_count"] == 0
    service_p95_passed = service_p95 <= COMPLETE_SERVICE_BUDGET_MS
    compositor_p95_passed = compositor_p95 <= COMPOSITOR_P95_SUB_BUDGET_MS
    unique_fps = arrival_30["unique_composite_fps"]
    cadence_alternative = (
        isinstance(unique_fps, (int, float))
        and float(unique_fps) >= MIN_UNIQUE_COMPOSITES_PER_S
        and arrival_30["queue_age_bounded_to_one_arrival_interval"] is True
        and arrival_30["queue_age_not_growing"] is True
    )
    cycle_rate_floor_passed = (
        simulated_sustainable is not None
        and simulated_sustainable >= MIN_UNIQUE_COMPOSITES_PER_S
    )
    model_backed = evidence_kind == "model-backed"
    checks = {
        "model_backed": model_backed,
        "sample_floor": sample_floor,
        "same_fixed_source_and_authoritative_matrix_row": (
            source_identity_passed and selected_matrix_authoritative
        ),
        "same_fixed_source_and_qualified_matrix_row": (
            source_identity_passed and selected_matrix_qualified
        ),
        "exact_matrix_warmup_and_measured_span": exact_profile_span,
        "distinct_source_frame_per_full_path_invocation": distinct_source_frames,
        "selected_matrix_row_p95_sub_budget": (
            selected_matrix_checks.get("compositor_p95_sub_budget") is True
        ),
        "builtin_rvm_cuda_without_fallback": backend_passed,
        "opaque_hardware_provider_and_sink_digests_present": (opaque_identity_passed),
        "non_pacing_service_boundary": boundary_passed,
        "balanced_effective_policy_and_sink": balanced_policy_passed,
        "sink_recovery_absent": no_recovery,
        "full_path_compositor_p95_sub_budget": compositor_p95_passed,
        "non_pacing_cycle_sustains_minimum_tier": cycle_rate_floor_passed,
        "service_p95_budget": service_p95_passed,
        "arrival_30_unique_fps_alternative": cadence_alternative,
        "output_repeats_reported_separately": True,
    }
    qualification_prerequisites = (
        model_backed
        and sample_floor
        and source_identity_passed
        and selected_matrix_qualified
        and exact_profile_span
        and distinct_source_frames
        and backend_passed
        and opaque_identity_passed
        and boundary_passed
        and balanced_policy_passed
        and no_recovery
        and compositor_p95_passed
    )
    classification_prerequisites = (
        model_backed
        and sample_floor
        and source_identity_passed
        and selected_matrix_authoritative
        and exact_profile_span
        and distinct_source_frames
        and backend_passed
        and opaque_identity_passed
        and boundary_passed
        and balanced_policy_passed
        and no_recovery
    )
    performance_passed = cycle_rate_floor_passed and (
        service_p95_passed or cadence_alternative
    )
    sustainable = simulated_sustainable if classification_prerequisites else None
    outcome = (
        "qualified"
        if qualification_prerequisites and performance_passed
        else (
            "rejected"
            if model_backed
            and sample_floor
            and (
                not source_identity_passed
                or not selected_matrix_qualified
                or not exact_profile_span
                or not distinct_source_frames
                or not backend_passed
                or not opaque_identity_passed
                or not boundary_passed
                or not balanced_policy_passed
                or not no_recovery
                or not compositor_p95_passed
                or not performance_passed
            )
            else "not_decidable"
        )
    )
    normalized_evidence = {
        "schema": FULL_PATH_SCHEMA,
        "version": FULL_PATH_VERSION,
        "evidence_kind": evidence_kind,
        "source_sha256": source_sha256,
        "hardware_id": hardware_id,
        "hardware_identity_sha256": hardware_identity_sha256,
        "provider_environment_sha256": provider_environment_sha256,
        "measurement_run_sha256": measurement_run_sha256,
        "rvm_model_sha256": rvm_model_sha256,
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "measured_frame_lineage_sha256": measured_frame_lineage_sha256,
        "matrix_row_id": matrix_row_id,
        "effective_policy": effective_policy,
        "background_provider_scope": evidence["background_provider_scope"],
        "sink_id": sink_id,
        "sink_identity_sha256": sink_identity_sha256,
        "sink_submission_copy_in_service_boundary": evidence[
            "sink_submission_copy_in_service_boundary"
        ],
        "canvas": dict(canvas),
        "backend": evidence["backend"],
        "provider": evidence["provider"],
        "fallback_count": counters["fallback_count"],
        "boundary": evidence["boundary"],
        "warmup_frame_count": warmup,
        **normalized_samples,
        **{
            name: counters[name]
            for name in (
                "unique_composite_count",
                "model_invocation_count",
                "output_send_count",
                "output_repeat_count",
                "no_unread_repeat_count",
                "capture_slot_overwrite_count",
                "capture_sequence_gap_count",
                "capture_missing_input_count",
                "processing_deadline_miss_count",
                "serialized_new_frame_deadline_miss_count",
                "sink_recovery_count",
            )
        },
    }
    return {
        "status": "evaluated",
        "outcome": outcome,
        "reason": (
            ""
            if outcome == "qualified"
            else (
                "complete fixed-replay service missed its performance gate"
                if model_backed and sample_floor and not performance_passed
                else "model-backed fixed-replay qualification is incomplete"
            )
        ),
        "evidence": normalized_evidence,
        "timing_summary_ms": timing_summary,
        "arrival_sweep": arrival_sweep,
        "sustainable_unique_frame_profile_fps": sustainable,
        "signed_service_headroom_ms": round(
            COMPLETE_SERVICE_BUDGET_MS - service_p95,
            6,
        ),
        "signed_compositor_headroom_ms": round(
            COMPOSITOR_P95_SUB_BUDGET_MS - compositor_p95,
            6,
        ),
        "checks": checks,
    }


def build_performance_report(
    rows: Sequence[Mapping[str, object]],
    *,
    full_path_evidence: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate raw evidence and build a deterministic content-free report."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise MattePerformanceError("compositor matrix rows must be a sequence")
    normalized_rows = [_normalize_matrix_row(dict(row)) for row in rows]
    expected_ids = [variant.id for variant in COMPOSITOR_VARIANTS]
    actual_ids = [str(row["id"]) for row in normalized_rows]
    if actual_ids != expected_ids:
        raise MattePerformanceError(
            "compositor rows must contain the exact ordered eight-cell matrix"
        )
    source_digests = {str(row["source_sha256"]) for row in normalized_rows}
    if len(source_digests) != 1:
        raise MattePerformanceError("compositor matrix rows do not share one source")
    source_sha256 = next(iter(source_digests))
    source_scopes = {_json_bytes(row["source_scope"]) for row in normalized_rows}
    if len(source_scopes) != 1:
        raise MattePerformanceError(
            "compositor matrix rows do not share one source scope"
        )
    source_scope = dict(cast(Mapping[str, object], normalized_rows[0]["source_scope"]))
    row_outcomes = {str(row["id"]): str(row["outcome"]) for row in normalized_rows}
    matrix_rows = {
        str(row["id"]): cast(Mapping[str, object], row) for row in normalized_rows
    }
    worst_p95 = max(
        float(cast(Any, row["compositor_p95_ms"])) for row in normalized_rows
    )
    matrix_outcome = (
        "rejected"
        if any(outcome == "rejected" for outcome in row_outcomes.values())
        else (
            "qualified"
            if all(outcome == "qualified" for outcome in row_outcomes.values())
            else "not_decidable"
        )
    )
    matrix = {
        "outcome": matrix_outcome,
        "required_row_count": len(COMPOSITOR_VARIANTS),
        "completed_row_count": len(normalized_rows),
        "source_sha256": source_sha256,
        "worst_compositor_p95_ms": round(worst_p95, 6),
        "signed_worst_compositor_headroom_ms": round(
            COMPOSITOR_P95_SUB_BUDGET_MS - worst_p95,
            6,
        ),
        "rows": normalized_rows,
    }
    full_path = _normalize_full_path(
        full_path_evidence,
        expected_source_sha256=source_sha256,
        expected_source_scope=source_scope,
        matrix_rows=matrix_rows,
    )
    measured_bandwidth = [
        cast(Mapping[str, object], row["memory_bandwidth"])
        for row in normalized_rows
        if cast(Mapping[str, object], row["memory_bandwidth"])["available"] is True
    ]
    if measured_bandwidth:
        bandwidth_provenance = {
            (
                bandwidth["source_sha256"],
                bandwidth["hardware_identity_sha256"],
                bandwidth["provider_environment_sha256"],
                bandwidth["measurement_run_sha256"],
            )
            for bandwidth in measured_bandwidth
        }
        if len(bandwidth_provenance) != 1:
            raise MattePerformanceError(
                "native memory-bandwidth rows do not share one source/hardware run"
            )
        normalized_full_evidence = full_path["evidence"]
        if normalized_full_evidence is not None:
            full_evidence = cast(Mapping[str, object], normalized_full_evidence)
            expected_bandwidth_provenance = (
                full_evidence["source_sha256"],
                full_evidence["hardware_identity_sha256"],
                full_evidence["provider_environment_sha256"],
                full_evidence["measurement_run_sha256"],
            )
            if next(iter(bandwidth_provenance)) != expected_bandwidth_provenance:
                raise MattePerformanceError(
                    "native memory-bandwidth evidence is not bound to the "
                    "full-path measurement run"
                )
    overall_outcome = (
        "rejected"
        if matrix_outcome == "rejected" or full_path["outcome"] == "rejected"
        else (
            "qualified"
            if matrix_outcome == "qualified" and full_path["outcome"] == "qualified"
            else "not_decidable"
        )
    )
    reasons: list[str] = []
    if matrix_outcome != "qualified":
        reasons.append("the complete measured compositor matrix is not qualified")
    if full_path["outcome"] != "qualified":
        reasons.append(
            "model-backed RVM/CUDA complete-service evidence is not qualified"
        )
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "scope": {
            "width": FIXED_WIDTH,
            "height": FIXED_HEIGHT,
            "nominal_fps": FIXED_FPS,
            "fixed_replay": True,
            "reactions_enabled": False,
            "source": source_scope,
            "source_sha256": source_sha256,
        },
        "policy": {
            "compositor_p95_sub_budget_ms": COMPOSITOR_P95_SUB_BUDGET_MS,
            "complete_service_p95_budget_ms": COMPLETE_SERVICE_BUDGET_MS,
            "minimum_unique_composites_per_s": MIN_UNIQUE_COMPOSITES_PER_S,
            "minimum_warmup_frames": MIN_WARMUP_FRAMES,
            "minimum_measured_frames": MIN_MEASURED_FRAMES,
            "arrival_sweep_fps": list(ARRIVAL_SWEEP_FPS),
            "percentile_method": "nearest-rank",
        },
        "matrix": matrix,
        "full_path": full_path,
        "decision": {
            "outcome": overall_outcome,
            "reasons": reasons,
            "declared_sustainable_unique_frame_profile_fps": full_path[
                "sustainable_unique_frame_profile_fps"
            ],
            "output_repeats_credited_as_unique_work": False,
            "reaction_qualification_must_use_signed_residual_headroom": True,
            "reaction_ready": False,
        },
    }
    report["evidence_sha256"] = _sha256(report)
    return report


def validate_performance_report(report: Mapping[str, object]) -> dict[str, Any]:
    """Rebuild and byte-compare a report so summaries cannot be hand edited."""

    root = _strict_keys(
        dict(report),
        {
            "schema",
            "version",
            "scope",
            "policy",
            "matrix",
            "full_path",
            "decision",
            "evidence_sha256",
        },
        "matte performance report",
    )
    if root["schema"] != REPORT_SCHEMA or root["version"] != REPORT_VERSION:
        raise MattePerformanceError("unsupported matte performance report")
    matrix = cast(Mapping[str, object], root["matrix"])
    report_rows = cast(Sequence[Mapping[str, object]], matrix["rows"])
    raw_rows = [
        {
            name: row[name]
            for name in (
                "schema",
                "version",
                "id",
                "source_sha256",
                "source_scope",
                "evidence_kind",
                "blend_space",
                "use_model_foreground",
                "light_wrap",
                "warmup_frame_count",
                "measured_frame_count",
                "timing_samples_ms",
                "allocation_samples_bytes",
                "memory_bandwidth",
                "equivalence",
            )
        }
        for row in report_rows
    ]
    full_path = cast(Mapping[str, object], root["full_path"])
    evidence = full_path.get("evidence")
    rebuilt = build_performance_report(
        raw_rows,
        full_path_evidence=(
            None if evidence is None else cast(Mapping[str, object], evidence)
        ),
    )
    if _json_bytes(rebuilt) != _json_bytes(root):
        raise MattePerformanceError(
            "matte performance report does not match its raw evidence"
        )
    return rebuilt


def _fixed_inputs(
    foreground: object,
    backdrop: object,
    mask: object,
    clean_foreground: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    for value, name in (
        (foreground, "foreground"),
        (backdrop, "backdrop"),
        (clean_foreground, "clean foreground"),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.uint8
            or value.shape != (FIXED_HEIGHT, FIXED_WIDTH, 3)
            or not value.flags.c_contiguous
        ):
            raise MattePerformanceError(f"{name} must be contiguous 1280x720 uint8 BGR")
        frames.append(value)
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.float32
        or mask.shape != (FIXED_HEIGHT, FIXED_WIDTH)
        or not mask.flags.c_contiguous
        or not np.isfinite(mask).all()
        or float(mask.min()) < 0.0
        or float(mask.max()) > 1.0
    ):
        raise MattePerformanceError(
            "mask must be contiguous finite 1280x720 float32 alpha in [0, 1]"
        )
    return frames[0], frames[1], mask, frames[2]


def _validated_frame_sample(
    sample: object,
    *,
    name: str,
) -> CompositorFrameSample:
    if not isinstance(sample, CompositorFrameSample):
        raise MattePerformanceError(f"{name} must be a CompositorFrameSample")
    foreground, backdrop, mask, clean = _fixed_inputs(
        sample.foreground_bgr,
        sample.backdrop_bgr,
        sample.mask,
        sample.clean_foreground_bgr,
    )
    sequence = _nonnegative_int(
        sample.capture_sequence,
        f"{name} capture sequence",
    )
    timestamp_ns = _timestamp_ns(
        sample.capture_timestamp_ns,
        f"{name} capture timestamp",
    )
    return CompositorFrameSample(
        foreground_bgr=foreground,
        backdrop_bgr=backdrop,
        mask=mask,
        clean_foreground_bgr=clean,
        capture_sequence=sequence,
        capture_timestamp_ns=timestamp_ns,
    )


def _explicit_replay_source(
    samples: Sequence[CompositorFrameSample],
    *,
    kind: str,
) -> _ReplaySource:
    if (
        not isinstance(samples, Sequence)
        or isinstance(samples, (str, bytes))
        or not samples
        or len(samples) > MAX_SAMPLES
        or kind not in ("single-frame", "explicit-frame-sequence")
    ):
        raise MattePerformanceError("explicit compositor replay samples are invalid")
    validated = tuple(
        _validated_frame_sample(sample, name=f"source frame {index}")
        for index, sample in enumerate(samples)
    )
    if kind == "single-frame" and len(validated) != 1:
        raise MattePerformanceError("single-frame source must contain one frame")
    for previous, current in zip(validated, validated[1:]):
        if (
            current.capture_sequence <= previous.capture_sequence
            or current.capture_timestamp_ns <= previous.capture_timestamp_ns
        ):
            raise MattePerformanceError(
                "explicit replay capture sequence and timestamps must increase"
            )
    digest = hashlib.sha256()
    digest.update(b"custback-matte-performance-explicit-source-v1\0")
    digest.update(kind.encode("ascii"))
    for sample in validated:
        digest.update(
            _json_bytes(
                {
                    "capture_sequence": sample.capture_sequence,
                    "capture_timestamp_ns": sample.capture_timestamp_ns,
                }
            )
        )
        for array in (
            sample.foreground_bgr,
            sample.backdrop_bgr,
            sample.mask,
            sample.clean_foreground_bgr,
        ):
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(_json_bytes(list(array.shape)))
            digest.update(array.tobytes())
    return _ReplaySource(
        kind=kind,
        frame_count=len(validated),
        source_sha256=digest.hexdigest(),
        bundle_manifest_sha256=None,
        lineage=tuple(
            (
                sample.capture_sequence,
                sample.capture_timestamp_ns,
                1,
                1,
            )
            for sample in validated
        ),
        load=lambda index: validated[index],
        load_model_frame=lambda index: _ModelReplayFrame(
            foreground_bgr=validated[index].foreground_bgr,
            backdrop_bgr=validated[index].backdrop_bgr,
            capture_sequence=validated[index].capture_sequence,
            capture_timestamp_ns=validated[index].capture_timestamp_ns,
            capture_generation=1,
            geometry_generation=1,
        ),
    )


def _private_bundle_source(root: Path | str) -> _ReplaySource:
    """Open a privacy-guarded full replay without exposing its path in output."""

    from .matte_diagnostics import MatteReplayBundle

    bundle = MatteReplayBundle(root)
    if bundle.manifest.get("capture_mode") != "full" or not bundle.frames:
        raise MattePerformanceError(
            "MATTE-3.4 requires a non-empty private full-capture replay bundle"
        )
    contract: list[dict[str, object]] = []
    required_tracks = (
        "raw_frame",
        "backdrop_frame",
        "refined_mask",
        "clean_foreground",
    )
    for frame in bundle.frames:
        artifacts = frame.get("artifacts")
        if not isinstance(artifacts, dict) or any(
            name not in artifacts for name in required_tracks
        ):
            raise MattePerformanceError(
                "private replay lacks a required raw/backdrop/mask/foreground track"
            )
        tracks: dict[str, object] = {}
        for name in required_tracks:
            descriptor = artifacts[name]
            if not isinstance(descriptor, dict) or "alias_of" in descriptor:
                raise MattePerformanceError(
                    "private replay performance tracks must be concrete artifacts"
                )
            tracks[name] = {
                "sha256": descriptor.get("sha256"),
                "shape": descriptor.get("shape"),
                "dtype": descriptor.get("dtype"),
            }
        contract.append(
            {
                "capture_sequence": frame["capture_sequence"],
                "capture_timestamp_ns": frame["capture_monotonic_ns"],
                "capture_generation": frame["capture_generation"],
                "geometry_generation": frame["geometry_generation"],
                "tracks": tracks,
            }
        )
    for previous, current in zip(bundle.frames, bundle.frames[1:]):
        if int(current["capture_sequence"]) <= int(previous["capture_sequence"]) or int(
            current["capture_monotonic_ns"]
        ) <= int(previous["capture_monotonic_ns"]):
            raise MattePerformanceError(
                "private replay capture sequence and timestamps must increase"
            )
    source_sha256 = _sha256(
        {
            "schema": "custback.matte-performance-private-source",
            "version": 1,
            "frames": contract,
        }
    )

    def load(index: int) -> CompositorFrameSample:
        frame = bundle.frames[index]
        return _validated_frame_sample(
            CompositorFrameSample(
                foreground_bgr=bundle.load_array(frame, "raw_frame"),
                backdrop_bgr=bundle.load_array(frame, "backdrop_frame"),
                mask=np.ascontiguousarray(
                    bundle.load_array(frame, "refined_mask"),
                    dtype=np.float32,
                ),
                clean_foreground_bgr=bundle.load_array(
                    frame,
                    "clean_foreground",
                ),
                capture_sequence=int(frame["capture_sequence"]),
                capture_timestamp_ns=int(frame["capture_monotonic_ns"]),
            ),
            name=f"private replay frame {index}",
        )

    def load_model_frame(index: int) -> _ModelReplayFrame:
        frame = bundle.frames[index]
        foreground = bundle.load_array(frame, "raw_frame")
        backdrop = bundle.load_array(frame, "backdrop_frame")
        for name, value in (
            ("raw frame", foreground),
            ("backdrop frame", backdrop),
        ):
            if (
                value.dtype != np.uint8
                or value.shape != (FIXED_HEIGHT, FIXED_WIDTH, 3)
                or not value.flags.c_contiguous
            ):
                raise MattePerformanceError(
                    f"private replay {name} must be contiguous 1280x720 uint8 BGR"
                )
        return _ModelReplayFrame(
            foreground_bgr=foreground,
            backdrop_bgr=backdrop,
            capture_sequence=int(frame["capture_sequence"]),
            capture_timestamp_ns=int(frame["capture_monotonic_ns"]),
            capture_generation=int(frame["capture_generation"]),
            geometry_generation=int(frame["geometry_generation"]),
        )

    return _ReplaySource(
        kind="privacy-replay-bundle",
        frame_count=len(bundle.frames),
        source_sha256=source_sha256,
        bundle_manifest_sha256=bundle.manifest_sha256,
        lineage=tuple(
            (
                int(frame["capture_sequence"]),
                int(frame["capture_monotonic_ns"]),
                int(frame["capture_generation"]),
                int(frame["geometry_generation"]),
            )
            for frame in bundle.frames
        ),
        load=load,
        load_model_frame=load_model_frame,
    )


def _open_balanced_rvm_segmenter(device_id: int) -> object:
    """Open the exact built-in RVM/CUDA policy owned by this qualification."""

    from .config import AccelerationConfig, SegmentationConfig
    from .segmentation import create_segmenter

    if type(device_id) is not int or not 0 <= device_id <= 64:
        raise MattePerformanceError("CUDA device id must be an integer in [0, 64]")
    return create_segmenter(
        SegmentationConfig(
            backend="rvm",
            rvm_downsample=0.0,
            mask_shift=0,
        ),
        acceleration=AccelerationConfig(
            mode="gpu_required",
            provider="cuda",
            device_id=device_id,
        ),
    )


def _open_qualification_sink(sink_backend: str) -> object:
    """Open an explicit non-fallback 720p30 production output sink."""

    from .config import OutputConfig
    from .vcam import NullOutput, open_output

    if sink_backend not in _QUALIFICATION_SINK_IDS:
        raise MattePerformanceError(
            "full-path qualification sink must be pyvirtualcam or native"
        )
    output = open_output(
        OutputConfig(backend=cast(Any, sink_backend), fps=FIXED_FPS),
        FIXED_WIDTH,
        FIXED_HEIGHT,
    )
    if isinstance(output, NullOutput) or bool(
        getattr(output, "fallback_active", False)
    ):
        try:
            output.close()
        finally:
            raise MattePerformanceError(
                "full-path qualification cannot use a null or fallback sink"
            )
    actual_mode = (
        getattr(output, "width", None),
        getattr(output, "height", None),
        getattr(output, "fps", None),
    )
    if any(type(value) is not int for value in actual_mode) or actual_mode != (
        FIXED_WIDTH,
        FIXED_HEIGHT,
        FIXED_FPS,
    ):
        try:
            output.close()
        finally:
            raise MattePerformanceError(
                "full-path qualification sink did not retain exact 1280x720@30 mode"
            )
    return output


def _send_unpaced_qualification_frame(
    output: object,
    frame: np.ndarray,
) -> object:
    """Submit to a real sink without allowing deliberate pacing to hide work."""

    from .vcam import OutputSendTiming, PyVirtualCamOutput

    if isinstance(output, PyVirtualCamOutput):
        # Production ``send_with_timing`` delegates to this exact validation
        # and submission seam before sleeping. Deliberately omit only the
        # pacing wait so replay capacity remains observable.
        return output.submit_unpaced_with_timing(frame)
    if bool(getattr(output, "paces", False)):
        raise MattePerformanceError(
            "qualification sink has no explicit unpaced submission seam"
        )
    sender = getattr(output, "send_with_timing", None)
    if not callable(sender):
        raise MattePerformanceError("qualification sink has no timed submission seam")
    timing = sender(frame)
    if (
        not isinstance(timing, OutputSendTiming)
        or timing.pacing_wait_ms != 0.0
        or timing.pacing_events != 0
    ):
        raise MattePerformanceError(
            "qualification sink performed pacing inside the unpaced replay"
        )
    return timing


def _qualification_identity(
    *,
    segmenter: object,
    output: object,
    source: _ReplaySource,
    hardware_id: str,
    device_id: int,
    sink_backend: str,
    run_started_ns: int,
) -> tuple[str, str, str, str]:
    """Return opaque hardware/provider/sink/run identities without raw paths."""

    try:
        ort = importlib.import_module("onnxruntime")

        ort_version = str(getattr(ort, "__version__", "unknown"))
        available_providers = sorted(
            str(value) for value in ort.get_available_providers()
        )
        ort_device = str(ort.get_device())
        build_info_getter = getattr(ort, "get_build_info", None)
        ort_build_info = (
            str(build_info_getter()) if callable(build_info_getter) else "unavailable"
        )
    except Exception as exc:  # pragma: no cover - RVM dependency invariant
        raise MattePerformanceError(
            "ONNX Runtime identity is unavailable for full-path evidence"
        ) from exc
    try:
        cuda_identity = collect_cuda_device_identity(device_id)
        cuda_provider_environment = {
            name: cuda_identity[name]
            for name in (
                "identity_source",
                "ordinal",
                "cuda_driver_version",
                "cuda_runtime_version",
                "nvidia_driver_version",
                "nvml_version",
            )
        }
    except Exception:
        raise MattePerformanceError(
            "exact CUDA hardware identity is unavailable for full-path evidence"
        ) from None
    snapshotter = getattr(segmenter, "rvm_telemetry_snapshot", None)
    if not callable(snapshotter):
        raise MattePerformanceError("RVM telemetry is unavailable")
    telemetry = snapshotter()
    active_provider = str(getattr(telemetry, "acceleration_active_provider", ""))
    hardware_identity = _sha256(
        {
            "schema": "custback.matte-performance-hardware-identity",
            "version": 1,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "operator_hardware_tier": hardware_id,
            "cuda_device_id": device_id,
            "ort_device": ort_device,
            "cuda_device_identity": cuda_identity,
        }
    )
    provider_identity = _sha256(
        {
            "schema": "custback.matte-performance-provider-environment",
            "version": 1,
            "python": list(sys.version_info[:3]),
            "numpy": np.__version__,
            "onnxruntime": ort_version,
            "onnxruntime_build_info": ort_build_info,
            "available_providers": available_providers,
            "active_provider": active_provider,
            "cuda_environment": cuda_provider_environment,
        }
    )
    sink_identity = _sha256(
        {
            "schema": "custback.matte-performance-sink-identity",
            "version": 1,
            "backend": sink_backend,
            "class": f"{type(output).__module__}.{type(output).__qualname__}",
            "width": getattr(output, "width", None),
            "height": getattr(output, "height", None),
            "fps": getattr(output, "fps", None),
            "device": str(getattr(getattr(output, "cam", None), "device", "")),
        }
    )
    measurement_run = _sha256(
        {
            "schema": "custback.matte-performance-measurement-run",
            "version": 1,
            "source_sha256": source.source_sha256,
            "bundle_manifest_sha256": source.bundle_manifest_sha256,
            "operator_hardware_tier": hardware_id,
            "hardware_identity_sha256": hardware_identity,
            "provider_environment_sha256": provider_identity,
            "sink_identity_sha256": sink_identity,
            "background_provider_scope": "resident-recorded-frame-copy",
            "run_started_ns": run_started_ns,
        }
    )
    return (
        hardware_identity,
        provider_identity,
        sink_identity,
        measurement_run,
    )


def collect_private_720p_full_path_evidence(
    bundle_root: Path | str,
    *,
    warmup_frame_count: int = MIN_WARMUP_FRAMES,
    measured_frame_count: int = MIN_MEASURED_FRAMES,
    hardware_id: str = "local_cuda",
    cuda_device_id: int = 0,
    sink_backend: str = "pyvirtualcam",
) -> dict[str, Any]:
    """Run an unpaced built-in-RVM/CUDA replay through a real output sink.

    This is the producer for the strict full-path sidecar consumed by
    :func:`build_performance_report`. It performs no capture and never writes
    source pixels. A short run is useful for diagnostics but remains below the
    validator's qualification floor.
    """

    from .cadence import CadenceTracker
    from .capture import CaptureHealth, CapturedFrame
    from .config import (
        AccelerationConfig,
        AppConfig,
        BackgroundConfig,
        CameraConfig,
        CompositingConfig,
        OutputConfig,
        RuntimeConfig,
        SegmentationConfig,
    )
    from .hub import TIMING_SCHEMA_VERSION, FrameHub
    from .pipeline import (
        Pipeline,
        _Resources,
        _cadence_status,
        _capture_health_stats,
        _ewma,
        _timing_fields,
    )
    from .segmentation import (
        RVMTelemetry,
        refiner_for,
    )
    from .vcam import OutputSendTiming

    warmup = _nonnegative_int(warmup_frame_count, "full-path warm-up frame count")
    measured = _positive_int(
        measured_frame_count,
        "full-path measured frame count",
    )
    if measured > MAX_SAMPLES or warmup + measured > MAX_SAMPLES:
        raise MattePerformanceError("full-path profiled frame count exceeds the bound")
    _safe_id(hardware_id, "full-path hardware id")
    source = _private_bundle_source(bundle_root)
    required_frames = warmup + measured
    if source.frame_count < required_frames:
        raise MattePerformanceError(
            "full-path qualification requires one distinct replay frame per "
            "warm-up and measured invocation"
        )
    preload_bytes = required_frames * (
        FIXED_HEIGHT * FIXED_WIDTH * 3 * np.dtype(np.uint8).itemsize * 2
    )
    if preload_bytes > MAX_FULL_PATH_PRELOAD_BYTES:
        raise MattePerformanceError(
            "full-path resident raw/backdrop replay exceeds the fixed memory bound"
        )

    segmenter = _open_balanced_rvm_segmenter(cuda_device_id)
    output: object | None = None
    refiner: object | None = None
    resources: _Resources | None = None
    pipeline: Pipeline | None = None
    hub: FrameHub | None = None
    run_started_ns = time.monotonic_ns()
    try:
        output = _open_qualification_sink(sink_backend)
        typed_segmenter = cast(Any, segmenter)
        segmentation_cfg = SegmentationConfig(
            backend="rvm",
            rvm_downsample=0.0,
            mask_shift=0,
        )
        compositing_cfg = CompositingConfig(
            light_wrap=0.25,
            use_model_foreground=True,
            blend_space="srgb_legacy",
        )
        refiner = refiner_for(
            segmentation_cfg,
            cast(Any, segmenter),
            compositing_cfg,
        )

        class _ReplayBackdrop:
            """Deliver one exact recorded frame with bounded provider ownership."""

            def __init__(self) -> None:
                self.current: np.ndarray | None = None
                self.output: np.ndarray | None = np.empty(
                    (FIXED_HEIGHT, FIXED_WIDTH, 3),
                    dtype=np.uint8,
                )

            def select(self, frame: np.ndarray) -> None:
                self.current = frame

            def frame(self, width: int, height: int) -> np.ndarray:
                current = self.current
                if current is None or current.shape != (height, width, 3):
                    raise MattePerformanceError(
                        "full-path replay backdrop selection is incomplete"
                    )
                output = self.output
                if output is None:
                    raise MattePerformanceError("full-path replay backdrop is closed")
                np.copyto(output, current)
                return output

            def close(self) -> None:
                self.current = None
                self.output = None

        replay_backdrop = _ReplayBackdrop()

        class _ReplayCapture:
            """One-slot resident source with the production capture interface."""

            def __init__(self) -> None:
                self.current: CapturedFrame | None = None
                self.last: CapturedFrame | None = None
                self.frames_read = 0

            def select(self, frame: CapturedFrame) -> None:
                if self.current is not None:
                    raise MattePerformanceError(
                        "full-path replay capture slot was not consumed"
                    )
                self.current = frame

            def read(self) -> CapturedFrame | None:
                current = self.current
                self.current = None
                if current is not None:
                    self.last = current
                    self.frames_read += 1
                return current

            def health_snapshot(self) -> CaptureHealth:
                current = self.last
                return CaptureHealth(
                    sequence=0 if current is None else current.sequence,
                    captured_monotonic_ns=(
                        None if current is None else current.captured_at_ns
                    ),
                    generation=0 if current is None else current.generation,
                    geometry_generation=(
                        0 if current is None else current.geometry_generation
                    ),
                    content_rect=(None if current is None else current.content_rect),
                    backend="resident-replay",
                    width=FIXED_WIDTH,
                    height=FIXED_HEIGHT,
                    delivered_width=FIXED_WIDTH,
                    delivered_height=FIXED_HEIGHT,
                    oriented_width=FIXED_WIDTH,
                    oriented_height=FIXED_HEIGHT,
                    normalized_width=FIXED_WIDTH,
                    normalized_height=FIXED_HEIGHT,
                    geometry_transitions=(
                        0 if current is None else current.geometry_generation
                    ),
                    fps_reported=float(FIXED_FPS),
                    capture_fps=float(FIXED_FPS),
                    target_met=True,
                    frames_read=self.frames_read,
                    frame_age_ms=0.0,
                    read_ms=0.0,
                    worker_alive=True,
                )

            def close(self) -> None:
                self.current = None
                self.last = None

        replay_capture = _ReplayCapture()

        cfg = AppConfig(
            camera=CameraConfig(
                width=FIXED_WIDTH,
                height=FIXED_HEIGHT,
                fps=FIXED_FPS,
                synthetic=True,
            ),
            background=BackgroundConfig(mode="color"),
            segmentation=segmentation_cfg,
            acceleration=AccelerationConfig(
                mode="gpu_required",
                provider="cuda",
                device_id=cuda_device_id,
            ),
            compositing=compositing_cfg,
            output=OutputConfig(
                width=FIXED_WIDTH,
                height=FIXED_HEIGHT,
                backend=cast(Any, sink_backend),
                fps=FIXED_FPS,
            ),
        )
        hub = FrameHub()
        hub.configure_canvas((FIXED_WIDTH, FIXED_HEIGHT))
        pipeline = Pipeline(RuntimeConfig(cfg), hub)
        resources = _Resources(
            cfg=cfg,
            version=0,
            capture=replay_capture,
            segmenter=segmenter,
            refiner=refiner,
            backdrop=replay_backdrop,
            output=output,
        )

        (
            hardware_identity_sha256,
            provider_environment_sha256,
            sink_identity_sha256,
            measurement_run_sha256,
        ) = _qualification_identity(
            segmenter=segmenter,
            output=output,
            source=source,
            hardware_id=hardware_id,
            device_id=cuda_device_id,
            sink_backend=sink_backend,
            run_started_ns=run_started_ns,
        )

        try:
            resident_frames = tuple(
                source.load_model_frame(ordinal) for ordinal in range(required_frames)
            )
        except MemoryError as exc:
            raise MattePerformanceError(
                "full-path resident replay allocation failed within its bound"
            ) from exc
        if (
            sum(
                frame.foreground_bgr.nbytes + frame.backdrop_bgr.nbytes
                for frame in resident_frames
            )
            != preload_bytes
        ):
            raise MattePerformanceError(
                "full-path resident replay byte accounting is inconsistent"
            )

        cadence_tracker = CadenceTracker(FIXED_FPS)
        stage_ewma: dict[str, float | None] = {
            "segmentation_ms": None,
            "background_ms": None,
            "color_correction_ms": None,
            "composite_prepare_ms": None,
            "composite_blend_ms": None,
            "composite_ms": None,
            "output_send_ms": None,
            "output_submission_ms": None,
            "output_sink_pacing_wait_ms": None,
            "application_pacing_wait_ms": None,
            "output_schedule_lateness_ms": None,
            "frame_processing_ms": None,
            "new_frame_service_ms": None,
            "new_frame_serialized_loop_ms": None,
        }
        last_output: np.ndarray | None = None

        def process_one(
            sample: _ModelReplayFrame,
        ) -> tuple[dict[str, float], OutputSendTiming, int, int]:
            nonlocal last_output
            assert pipeline is not None
            assert resources is not None
            assert hub is not None
            assert output is not None

            captured = CapturedFrame(
                pixels=sample.foreground_bgr,
                sequence=sample.capture_sequence,
                captured_at_ns=sample.capture_timestamp_ns,
                generation=sample.capture_generation,
                geometry_generation=sample.geometry_generation,
                content_rect=(0, 0, FIXED_WIDTH, FIXED_HEIGHT),
            )
            replay_capture.select(captured)
            # Privacy-checked replay artifact I/O and envelope construction
            # replace device acquisition and stay outside the clock. The
            # resident latest-slot dequeue through sink submission remains in
            # the service sample; the live synchronous post-submit tail is
            # measured by the enclosing non-pacing cycle below.
            service_started_ns = time.monotonic_ns()
            dequeued = resources.capture.read()
            if dequeued is not captured:
                raise MattePerformanceError(
                    "full-path replay did not dequeue the selected unique frame"
                )
            captured = dequeued
            replay_backdrop.select(sample.backdrop_bgr)
            frame = Pipeline._validate_canvas_frame(
                captured.pixels,
                resources.canvas_size,
                boundary="capture",
            )
            resources.capture_sequence_timeline.observe(captured.sequence)
            hub.publish_raw(frame)
            segmentation_sequence_before = (
                resources.segmentation_timeline.snapshot().last_sequence
            )
            try:
                request = pipeline._requests.get_nowait()
            except queue.Empty:
                request = None
            if request is not None:  # pragma: no cover - private empty queue invariant
                raise MattePerformanceError(
                    "qualification pipeline unexpectedly received a mutation"
                )
            if pipeline._new_matte_evidence(resources, captured) is not None:
                raise MattePerformanceError(
                    "qualification unexpectedly enabled pixel diagnostics"
                )

            process_started_ns = time.monotonic_ns()
            pipeline._latest_raw_frame = frame.copy()
            stage_timings = {
                "segmentation_ms": 0.0,
                "background_ms": 0.0,
                "color_correction_ms": 0.0,
                "composite_prepare_ms": 0.0,
                "composite_blend_ms": 0.0,
                "composite_ms": 0.0,
            }
            color_outcome: dict[str, object] = {}
            rendered, render_reason = pipeline._local_composite(
                resources,
                frame,
                captured=captured,
                privacy_safe=False,
                timings=stage_timings,
                color_outcome=color_outcome,
            )
            if render_reason:
                raise MattePerformanceError(
                    "local qualification unexpectedly activated a render fallback"
                )
            # Preserve the production frame-processing tail: resolve the
            # accepted timeline state and consume the color outcome before the
            # narrower deadline boundary closes.
            if (
                resources.segmentation_timeline.snapshot().last_sequence
                == segmentation_sequence_before
            ):
                raise MattePerformanceError(
                    "full-path replay did not advance one unique segmentation input"
                )
            if not color_outcome:
                raise MattePerformanceError(
                    "full-path replay did not resolve the production color policy"
                )
            frame_processing_ms = (
                time.monotonic_ns() - process_started_ns
            ) / 1_000_000.0

            telemetry = typed_segmenter.rvm_telemetry_snapshot()
            if not isinstance(telemetry, RVMTelemetry):
                raise MattePerformanceError("RVM telemetry snapshot is malformed")
            rvm_timings = (
                telemetry.preprocess_ms,
                telemetry.session_run_ms,
                telemetry.postprocess_ms,
            )
            if any(value is None for value in rvm_timings):
                raise MattePerformanceError("RVM timing telemetry is incomplete")

            # Exercise the exact final guarded-base/output validation and
            # deterministic-repeat scan used by the production worker.
            guard_started_ns = time.monotonic_ns()
            safe_base, guard_reason = pipeline._guard_remote_output(
                rendered,
                frame,
                privacy_safe=False,
            )
            if guard_reason:
                raise MattePerformanceError(
                    "local qualification unexpectedly activated a privacy fallback"
                )
            Pipeline._validate_output_frame(safe_base, (FIXED_WIDTH, FIXED_HEIGHT))
            base_ready_at_ns = time.monotonic_ns()
            exact_final_repeat = last_output is not None and np.array_equal(
                safe_base,
                last_output,
            )
            guard_validation_ms = (time.monotonic_ns() - guard_started_ns) / 1_000_000.0

            send_started_ns = time.monotonic_ns()
            send_timing = _send_unpaced_qualification_frame(output, safe_base)
            if not isinstance(send_timing, OutputSendTiming):
                raise MattePerformanceError("qualification sink timing is malformed")
            output_send_ms = (
                send_timing.completed_at_ns - send_started_ns
            ) / 1_000_000.0
            service_ms = (
                send_timing.submitted_at_ns - service_started_ns
            ) / 1_000_000.0
            serialized_ms = (
                send_timing.completed_at_ns - service_started_ns
            ) / 1_000_000.0
            processing_deadline_missed = (
                frame_processing_ms > COMPLETE_SERVICE_BUDGET_MS
            )
            serialized_deadline_missed = (
                serialized_ms
                > COMPLETE_SERVICE_BUDGET_MS + SERIALIZED_DEADLINE_GRACE_MS
            )
            # Run the same synchronous post-submit accounting and publication
            # performed by the live worker. These operations are intentionally
            # outside the sink-submission service sample but inside the
            # non-pacing cycle used for arrival-capacity classification.
            cadence_tracker.record_send(
                sent_at_ns=send_timing.submitted_at_ns,
                capture_sequence=captured.sequence,
                captured_at_ns=min(captured.captured_at_ns, service_started_ns),
                base_ready_at_ns=base_ready_at_ns,
                base_updated=True,
                segmentation_updated=True,
                exact_final_repeat=exact_final_repeat,
                processing_deadline_missed=processing_deadline_missed,
                serialized_new_frame_deadline_missed=serialized_deadline_missed,
                output_sink_pacing_events=send_timing.pacing_events,
                output_sink_recovery_events=send_timing.recovery_events,
                application_pacing_events=0,
                output_schedule_late=False,
            )
            for name, value in (
                *stage_timings.items(),
                ("frame_processing_ms", frame_processing_ms),
                ("output_send_ms", output_send_ms),
                ("output_submission_ms", send_timing.submission_ms),
                ("output_sink_pacing_wait_ms", send_timing.pacing_wait_ms),
                ("application_pacing_wait_ms", 0.0),
                ("output_schedule_lateness_ms", 0.0),
                ("new_frame_service_ms", service_ms),
                ("new_frame_serialized_loop_ms", serialized_ms),
            ):
                stage_ewma[name] = _ewma(stage_ewma[name], value)
            pipeline._record_color_output(resources, color_outcome, processed=True)
            capture_health = replay_capture.health_snapshot()
            frame_stats = pipeline._identity_stats(
                resources,
                capture_health=capture_health,
                color_status=color_outcome,
            )
            cadence_snapshot = cadence_tracker.snapshot(
                now_ns=send_timing.completed_at_ns
            )
            frame_stats.update(
                {
                    **_cadence_status(
                        cadence_tracker,
                        now_ns=send_timing.completed_at_ns,
                        target_output_fps=FIXED_FPS,
                    ),
                    **_capture_health_stats(
                        capture_health,
                        fallback_frames_read=cadence_snapshot.unique_capture_count,
                    ),
                    "remote_frames_used": 0,
                    "remote_fallback_active": False,
                    "remote_fallback_count": 0,
                    "remote_fallback_reason": "",
                    "segmentation_ms": stage_ewma["segmentation_ms"],
                    "background_ms": stage_ewma["background_ms"],
                    "color_correction_ms": stage_ewma["color_correction_ms"],
                    "composite_ms": stage_ewma["composite_ms"],
                    "output_send_ms": stage_ewma["output_send_ms"],
                    "output_submission_ms": stage_ewma["output_submission_ms"],
                    "output_sink_pacing_wait_ms": stage_ewma[
                        "output_sink_pacing_wait_ms"
                    ],
                    "application_pacing_wait_ms": stage_ewma[
                        "application_pacing_wait_ms"
                    ],
                    "output_schedule_lateness_ms": stage_ewma[
                        "output_schedule_lateness_ms"
                    ],
                    "frame_processing_ms": stage_ewma["frame_processing_ms"],
                    "new_frame_service_ms": stage_ewma["new_frame_service_ms"],
                    "new_frame_serialized_loop_ms": stage_ewma[
                        "new_frame_serialized_loop_ms"
                    ],
                    "timing_schema_version": TIMING_SCHEMA_VERSION,
                    "timing_ms": _timing_fields(
                        stage_ewma,
                        capture_health=capture_health,
                        segmenter=segmenter,
                    ),
                }
            )
            hub.publish_output(safe_base, stats=frame_stats)
            cycle_completed_ns = time.monotonic_ns()
            last_output = safe_base
            return (
                {
                    "service_samples_ms": service_ms,
                    "frame_processing_samples_ms": frame_processing_ms,
                    "serialized_new_frame_samples_ms": serialized_ms,
                    "rvm_preprocess_samples_ms": float(
                        cast(float, telemetry.preprocess_ms)
                    ),
                    "rvm_inference_samples_ms": float(
                        cast(float, telemetry.session_run_ms)
                    ),
                    "rvm_postprocess_samples_ms": float(
                        cast(float, telemetry.postprocess_ms)
                    ),
                    "background_selection_samples_ms": stage_timings["background_ms"],
                    "compositor_samples_ms": stage_timings["composite_ms"],
                    "post_composite_validation_samples_ms": guard_validation_ms,
                    "sink_submission_samples_ms": send_timing.submission_ms,
                    "pacing_wait_samples_ms": send_timing.pacing_wait_ms,
                },
                send_timing,
                service_started_ns,
                cycle_completed_ns,
            )

        for ordinal in range(warmup):
            process_one(resident_frames[ordinal])

        samples: dict[str, list[float]] = {name: [] for name in FULL_PATH_SAMPLE_NAMES}
        sink_recovery_count = 0
        capture_timeline_before = resources.capture_sequence_timeline.snapshot()
        for measured_index in range(measured):
            ordinal = warmup + measured_index
            sample = resident_frames[ordinal]
            (
                frame_timings,
                send_timing,
                cycle_started_ns,
                cycle_completed_ns,
            ) = process_one(sample)
            for name, value in frame_timings.items():
                samples[name].append(value)
            samples["non_pacing_cycle_samples_ms"].append(
                (cycle_completed_ns - cycle_started_ns) / 1_000_000.0
            )
            sink_recovery_count += send_timing.recovery_events
        capture_timeline_after = resources.capture_sequence_timeline.snapshot()
        capture_sequence_gap_count = (
            capture_timeline_after.gap_events - capture_timeline_before.gap_events
        )
        capture_missing_input_count = (
            capture_timeline_after.missing_inputs
            - capture_timeline_before.missing_inputs
        )
        samples["schedule_lateness_samples_ms"] = _logical_schedule_lateness_samples(
            samples["service_samples_ms"],
            samples["non_pacing_cycle_samples_ms"],
        )

        telemetry = typed_segmenter.rvm_telemetry_snapshot()
        if (
            not isinstance(telemetry, RVMTelemetry)
            or telemetry.model_builtin is not True
            or telemetry.model_sha256 != RVM_MODEL.sha256
            or telemetry.acceleration_active_provider != "cuda"
            or telemetry.acceleration_fallback_active is not False
            or telemetry.resolved_downsample_ratio != 0.4
        ):
            raise MattePerformanceError(
                "full-path run did not retain the built-in balanced RVM/CUDA policy"
            )
        processing_deadline_misses = sum(
            value > COMPLETE_SERVICE_BUDGET_MS
            for value in samples["frame_processing_samples_ms"]
        )
        serialized_deadline_misses = sum(
            value > COMPLETE_SERVICE_BUDGET_MS + SERIALIZED_DEADLINE_GRACE_MS
            for value in samples["serialized_new_frame_samples_ms"]
        )
        return {
            "schema": FULL_PATH_SCHEMA,
            "version": FULL_PATH_VERSION,
            "evidence_kind": "model-backed",
            "source_sha256": source.source_sha256,
            "hardware_id": hardware_id,
            "hardware_identity_sha256": hardware_identity_sha256,
            "provider_environment_sha256": provider_environment_sha256,
            "measurement_run_sha256": measurement_run_sha256,
            "rvm_model_sha256": RVM_MODEL.sha256,
            "bundle_manifest_sha256": source.bundle_manifest_sha256,
            "measured_frame_lineage_sha256": _measured_lineage_sha256(
                source,
                warmup_frame_count=warmup,
                measured_frame_count=measured,
            ),
            "matrix_row_id": _BALANCED_MATRIX_ROW_ID,
            "effective_policy": dict(_BALANCED_EFFECTIVE_POLICY),
            "background_provider_scope": "resident-recorded-frame-copy",
            "sink_id": sink_backend,
            "sink_identity_sha256": sink_identity_sha256,
            "sink_submission_copy_in_service_boundary": True,
            "canvas": {
                "width": FIXED_WIDTH,
                "height": FIXED_HEIGHT,
                "nominal_fps": FIXED_FPS,
            },
            "backend": "rvm",
            "provider": "cuda",
            "fallback_count": telemetry.acceleration_fallback_count,
            "boundary": _SERVICE_BOUNDARY,
            "warmup_frame_count": warmup,
            **samples,
            "unique_composite_count": measured,
            "model_invocation_count": measured,
            "output_send_count": measured,
            "output_repeat_count": 0,
            "no_unread_repeat_count": 0,
            "capture_slot_overwrite_count": 0,
            "capture_sequence_gap_count": capture_sequence_gap_count,
            "capture_missing_input_count": capture_missing_input_count,
            "processing_deadline_miss_count": processing_deadline_misses,
            "serialized_new_frame_deadline_miss_count": (serialized_deadline_misses),
            "sink_recovery_count": sink_recovery_count,
        }
    finally:
        if pipeline is not None:
            pipeline._latest_raw_frame = None
        if hub is not None:
            hub.raw.clear()
            hub.output.clear()
        if resources is not None:
            resources.close()
        else:
            if refiner is not None:
                close_refiner = getattr(refiner, "close", None)
                if callable(close_refiner):
                    close_refiner()
            if output is not None:
                close_output = getattr(output, "close", None)
                if callable(close_output):
                    close_output()
            close_segmenter = getattr(segmenter, "close", None)
            if callable(close_segmenter):
                close_segmenter()


def _unavailable_memory_bandwidth() -> dict[str, object]:
    return {
        "available": False,
        "bytes_per_second": None,
        "counter_source": None,
        "source_sha256": None,
        "hardware_identity_sha256": None,
        "provider_environment_sha256": None,
        "measurement_run_sha256": None,
    }


def _linear_inputs(
    sample: CompositorFrameSample,
    *,
    use_model_foreground: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    return (
        _bgr_u8_to_linear_bgr_prevalidated(sample.foreground_bgr),
        _bgr_u8_to_linear_bgr_prevalidated(sample.backdrop_bgr),
        (
            _bgr_u8_to_linear_bgr_prevalidated(sample.clean_foreground_bgr)
            if use_model_foreground
            else None
        ),
    )


def _wrap_context(ordinal: int, source_sha256: str) -> LightWrapFrameContext:
    return LightWrapFrameContext(
        frame_id=ordinal,
        timestamp_ns=round(ordinal * 1_000_000_000 / FIXED_FPS),
        source_token=("matte-performance", source_sha256),
    )


def _prepare_profile_wrap(
    sample: CompositorFrameSample,
    variant: CompositorVariant,
    *,
    ordinal: int,
    source_sha256: str,
    stabilizer: LightWrapStabilizer | None,
    backdrop_linear_bgr: np.ndarray | None,
    diagnostics: MutableMapping[str, float] | None,
) -> PreparedLightWrap | None:
    if variant.light_wrap == 0.0 or stabilizer is None:
        # The balanced compatibility policy has temporal light-wrap
        # stabilization off. In that lane the compositor owns the stateless
        # downscale/blur/upscale work and its diagnostics.
        return None
    return prepare_light_wrap(
        sample.backdrop_bgr,
        blend_space=cast(Any, variant.blend_space),
        stabilizer=stabilizer,
        context=_wrap_context(ordinal, source_sha256),
        backdrop_linear_bgr=(
            backdrop_linear_bgr if variant.blend_space == "linear_srgb" else None
        ),
        diagnostics=diagnostics,
    )


def _production_composite(
    sample: CompositorFrameSample,
    variant: CompositorVariant,
    *,
    linear_inputs: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None,
    prepared_light_wrap: PreparedLightWrap | None,
    workspace: LegacyCompositorWorkspace | None,
    diagnostics: MutableMapping[str, float] | None,
) -> np.ndarray:
    edge = sample.clean_foreground_bgr if variant.use_model_foreground else None
    if variant.blend_space == "linear_srgb":
        if linear_inputs is None:  # pragma: no cover - construction invariant
            raise MattePerformanceError("linear matrix cell has no decoded inputs")
        foreground_linear, backdrop_linear, edge_linear = linear_inputs
        return _composite_linear_bgr_prevalidated(
            sample.foreground_bgr,
            sample.backdrop_bgr,
            sample.mask,
            foreground_linear_bgr=foreground_linear,
            backdrop_linear_bgr=backdrop_linear,
            light_wrap=variant.light_wrap,
            edge_foreground_bgr=edge,
            edge_foreground_linear_bgr=edge_linear,
            prepared_light_wrap=prepared_light_wrap,
            diagnostics=diagnostics,
        )
    return composite(
        sample.foreground_bgr,
        sample.backdrop_bgr,
        sample.mask,
        light_wrap=variant.light_wrap,
        edge_foreground=edge,
        blend_space="srgb_legacy",
        prepared_light_wrap=prepared_light_wrap,
        workspace=workspace,
        diagnostics=diagnostics,
    )


def _reference_composite(
    sample: CompositorFrameSample,
    variant: CompositorVariant,
    *,
    prepared_light_wrap: PreparedLightWrap | None,
) -> np.ndarray:
    """Render through the public frozen NumPy/reference contract."""

    return composite(
        sample.foreground_bgr,
        sample.backdrop_bgr,
        sample.mask,
        light_wrap=variant.light_wrap,
        edge_foreground=(
            sample.clean_foreground_bgr if variant.use_model_foreground else None
        ),
        blend_space=cast(Any, variant.blend_space),
        prepared_light_wrap=prepared_light_wrap,
    )


def _measured_lineage_sha256(
    source: _ReplaySource,
    *,
    warmup_frame_count: int,
    measured_frame_count: int,
) -> str:
    """Bind a sidecar to the exact ordered measured replay identities."""

    if len(source.lineage) != source.frame_count:  # pragma: no cover - invariant
        raise MattePerformanceError("replay source lineage is incomplete")
    frames = [
        {
            "capture_sequence": source.lineage[ordinal % source.frame_count][0],
            "capture_timestamp_ns": source.lineage[ordinal % source.frame_count][1],
            "capture_generation": source.lineage[ordinal % source.frame_count][2],
            "geometry_generation": source.lineage[ordinal % source.frame_count][3],
        }
        for ordinal in range(
            warmup_frame_count,
            warmup_frame_count + measured_frame_count,
        )
    ]
    return _sha256(
        {
            "schema": "custback.matte-performance-measured-lineage",
            "version": 1,
            "source_sha256": source.source_sha256,
            "warmup_frame_count": warmup_frame_count,
            "measured_frame_count": measured_frame_count,
            "frames": frames,
        }
    )


def _profile_replay_source(
    source: _ReplaySource,
    *,
    warmup_frame_count: int,
    measured_frame_count: int,
    evidence_kind: str,
    full_path_evidence: Mapping[str, object] | None,
    native_memory_bandwidth: Mapping[str, Mapping[str, object]] | None,
    clock_ns: Callable[[], int],
) -> dict[str, Any]:
    warmup = _nonnegative_int(warmup_frame_count, "warm-up frame count")
    measured = _positive_int(measured_frame_count, "measured frame count")
    if measured > MAX_SAMPLES or warmup + measured > MAX_SAMPLES:
        raise MattePerformanceError("profiled frame count exceeds the bound")
    if evidence_kind not in ("generated-proxy", "fixed-replay-measured"):
        raise MattePerformanceError("compositor evidence kind is invalid")
    if source.kind == "single-frame" and evidence_kind != "generated-proxy":
        raise MattePerformanceError(
            "single-frame matrices are generated proxies and cannot qualify"
        )
    if native_memory_bandwidth is not None:
        unknown_variants = set(native_memory_bandwidth) - {
            variant.id for variant in COMPOSITOR_VARIANTS
        }
        if unknown_variants:
            raise MattePerformanceError(
                "native memory bandwidth contains an unknown matrix row"
            )
    profiled_frame_count = warmup + measured
    measured_lineage_sha256 = _measured_lineage_sha256(
        source,
        warmup_frame_count=warmup,
        measured_frame_count=measured,
    )
    resident_frame_count = min(source.frame_count, profiled_frame_count)
    expected_resident_bytes = resident_frame_count * (
        FIXED_HEIGHT
        * FIXED_WIDTH
        * (3 * np.dtype(np.uint8).itemsize * 3 + np.dtype(np.float32).itemsize)
    )
    if expected_resident_bytes > MAX_MATRIX_PRELOAD_BYTES:
        raise MattePerformanceError(
            "matrix resident replay exceeds the fixed memory bound"
        )
    try:
        resident_samples = tuple(
            source.load(index) for index in range(resident_frame_count)
        )
    except MemoryError as exc:
        raise MattePerformanceError(
            "matrix resident replay allocation failed within its bound"
        ) from exc
    if (
        sum(
            sample.foreground_bgr.nbytes
            + sample.backdrop_bgr.nbytes
            + sample.mask.nbytes
            + sample.clean_foreground_bgr.nbytes
            for sample in resident_samples
        )
        != expected_resident_bytes
    ):
        raise MattePerformanceError(
            "matrix resident replay byte accounting is inconsistent"
        )
    source_scope = {
        "kind": source.kind,
        "source_frame_count": source.frame_count,
        "profiled_frame_count": profiled_frame_count,
        "unique_capture_sequence": source.kind != "single-frame",
        "privacy_guarded_bundle": source.kind == "privacy-replay-bundle",
        "bundle_manifest_sha256": source.bundle_manifest_sha256,
        "measured_frame_lineage_sha256": measured_lineage_sha256,
        "execution_pacing": "unpaced-tight-loop",
    }
    rows: list[dict[str, object]] = []
    for variant in COMPOSITOR_VARIANTS:
        workspace = (
            LegacyCompositorWorkspace((FIXED_HEIGHT, FIXED_WIDTH, 3))
            if variant.blend_space == "srgb_legacy"
            else None
        )
        stabilizer: LightWrapStabilizer | None = None
        max_delta = 0
        foreground_endpoint_exact = True
        background_endpoint_exact = True
        deterministic_repeat_exact = True
        output_contract_exact = True
        try:
            # Complete all reference, repeat, and endpoint checks before the
            # warm-up. The timed production calls below are then consecutive:
            # no artifact reads or extra correctness renders can pace/cool the
            # measured compositor.
            for measured_index in range(measured):
                ordinal = warmup + measured_index
                sample = resident_samples[ordinal % resident_frame_count]
                linear_inputs = (
                    _linear_inputs(
                        sample,
                        use_model_foreground=variant.use_model_foreground,
                    )
                    if variant.blend_space == "linear_srgb"
                    else None
                )
                prepared = _prepare_profile_wrap(
                    sample,
                    variant,
                    ordinal=ordinal,
                    source_sha256=source.source_sha256,
                    stabilizer=None,
                    backdrop_linear_bgr=(
                        None if linear_inputs is None else linear_inputs[1]
                    ),
                    diagnostics=None,
                )
                output = _production_composite(
                    sample,
                    variant,
                    linear_inputs=linear_inputs,
                    prepared_light_wrap=prepared,
                    workspace=workspace,
                    diagnostics=None,
                )
                reference = _reference_composite(
                    sample,
                    variant,
                    prepared_light_wrap=prepared,
                )
                repeated = _production_composite(
                    sample,
                    variant,
                    linear_inputs=linear_inputs,
                    prepared_light_wrap=prepared,
                    workspace=workspace,
                    diagnostics=None,
                )
                endpoint_mask = sample.mask.copy()
                endpoint_mask[0, 0] = np.float32(0.0)
                endpoint_mask[0, 1] = np.float32(1.0)
                endpoint_sample = CompositorFrameSample(
                    foreground_bgr=sample.foreground_bgr,
                    backdrop_bgr=sample.backdrop_bgr,
                    mask=endpoint_mask,
                    clean_foreground_bgr=sample.clean_foreground_bgr,
                    capture_sequence=sample.capture_sequence,
                    capture_timestamp_ns=sample.capture_timestamp_ns,
                )
                endpoint_output = _production_composite(
                    endpoint_sample,
                    variant,
                    linear_inputs=linear_inputs,
                    prepared_light_wrap=prepared,
                    workspace=workspace,
                    diagnostics=None,
                )
                endpoint_reference = _reference_composite(
                    endpoint_sample,
                    variant,
                    prepared_light_wrap=prepared,
                )
                max_delta = max(
                    max_delta,
                    int(
                        np.max(
                            np.abs(
                                output.astype(np.int16) - reference.astype(np.int16)
                            ),
                            initial=0,
                        )
                    ),
                    int(
                        np.max(
                            np.abs(
                                endpoint_output.astype(np.int16)
                                - endpoint_reference.astype(np.int16)
                            ),
                            initial=0,
                        )
                    ),
                )
                foreground_endpoint_exact = (
                    foreground_endpoint_exact
                    and np.array_equal(
                        endpoint_output[0, 1], sample.foreground_bgr[0, 1]
                    )
                )
                background_endpoint_exact = (
                    background_endpoint_exact
                    and np.array_equal(endpoint_output[0, 0], sample.backdrop_bgr[0, 0])
                )
                deterministic_repeat_exact = (
                    deterministic_repeat_exact and np.array_equal(output, repeated)
                )
                output_contract_exact = (
                    output_contract_exact
                    and output.dtype == np.uint8
                    and output.shape == (FIXED_HEIGHT, FIXED_WIDTH, 3)
                    and output.flags.c_contiguous
                    and endpoint_output.dtype == np.uint8
                    and endpoint_output.shape == (FIXED_HEIGHT, FIXED_WIDTH, 3)
                    and endpoint_output.flags.c_contiguous
                )

            for ordinal in range(warmup):
                sample = resident_samples[ordinal % resident_frame_count]
                linear_inputs = (
                    _linear_inputs(
                        sample,
                        use_model_foreground=variant.use_model_foreground,
                    )
                    if variant.blend_space == "linear_srgb"
                    else None
                )
                prepared = _prepare_profile_wrap(
                    sample,
                    variant,
                    ordinal=ordinal,
                    source_sha256=source.source_sha256,
                    stabilizer=stabilizer,
                    backdrop_linear_bgr=(
                        None if linear_inputs is None else linear_inputs[1]
                    ),
                    diagnostics=None,
                )
                _production_composite(
                    sample,
                    variant,
                    linear_inputs=linear_inputs,
                    prepared_light_wrap=prepared,
                    workspace=workspace,
                    diagnostics=None,
                )

            timing_samples = {name: [] for name in TIMING_SAMPLE_NAMES}
            allocation_samples = {name: [] for name in ALLOCATION_SAMPLE_NAMES}
            for measured_index in range(measured):
                ordinal = warmup + measured_index
                sample = resident_samples[ordinal % resident_frame_count]
                diagnostics: MutableMapping[str, float] = {}
                # Linear EOTF conversion remains production work unless an
                # upstream reuse contract proves otherwise, so it stays inside
                # ``total``.
                started_ns = clock_ns()
                linear_inputs = (
                    _linear_inputs(
                        sample,
                        use_model_foreground=variant.use_model_foreground,
                    )
                    if variant.blend_space == "linear_srgb"
                    else None
                )
                prepared = _prepare_profile_wrap(
                    sample,
                    variant,
                    ordinal=ordinal,
                    source_sha256=source.source_sha256,
                    stabilizer=stabilizer,
                    backdrop_linear_bgr=(
                        None if linear_inputs is None else linear_inputs[1]
                    ),
                    diagnostics=diagnostics,
                )
                output = _production_composite(
                    sample,
                    variant,
                    linear_inputs=linear_inputs,
                    prepared_light_wrap=prepared,
                    workspace=workspace,
                    diagnostics=diagnostics,
                )
                completed_ns = clock_ns()
                if completed_ns < started_ns:
                    raise MattePerformanceError("profiling clock moved backwards")
                timing_samples["total"].append(
                    (completed_ns - started_ns) / 1_000_000.0
                )
                for name in COMPOSITOR_SUBSTAGE_NAMES:
                    timing_samples[name].append(float(diagnostics.get(name, 0.0)))

                snapshot = None if workspace is None else workspace.snapshot()
                prepared_bytes = (
                    0 if prepared is None else int(prepared.pixels_bgr.nbytes)
                )
                known_output_bytes = (
                    output.nbytes
                    if snapshot is None
                    else max(
                        output.nbytes,
                        snapshot.last_known_allocation_bytes,
                    )
                )
                linear_input_bytes = (
                    0
                    if linear_inputs is None
                    else sum(
                        value.nbytes for value in linear_inputs if value is not None
                    )
                )
                allocation_samples["known_transient_allocation_bytes"].append(
                    known_output_bytes + prepared_bytes + linear_input_bytes
                )
                stabilizer_bytes = (
                    0 if stabilizer is None else stabilizer.snapshot().retained_bytes
                )
                allocation_samples["retained_workspace_bytes"].append(
                    stabilizer_bytes
                    + (0 if snapshot is None else snapshot.retained_bytes)
                )
            bandwidth = (
                _unavailable_memory_bandwidth()
                if native_memory_bandwidth is None
                or variant.id not in native_memory_bandwidth
                else dict(native_memory_bandwidth[variant.id])
            )
            rows.append(
                {
                    "schema": ROW_SCHEMA,
                    "version": ROW_VERSION,
                    "id": variant.id,
                    "source_sha256": source.source_sha256,
                    "source_scope": source_scope,
                    "evidence_kind": evidence_kind,
                    "blend_space": variant.blend_space,
                    "use_model_foreground": variant.use_model_foreground,
                    "light_wrap": variant.light_wrap,
                    "warmup_frame_count": warmup,
                    "measured_frame_count": measured,
                    "timing_samples_ms": timing_samples,
                    "allocation_samples_bytes": allocation_samples,
                    "memory_bandwidth": bandwidth,
                    "equivalence": {
                        "reference_contract": (
                            "frozen-srgb-legacy-v1"
                            if variant.blend_space == "srgb_legacy"
                            else "linear-srgb-reference-v1"
                        ),
                        "max_channel_delta": max_delta,
                        "foreground_endpoint_exact": bool(foreground_endpoint_exact),
                        "background_endpoint_exact": bool(background_endpoint_exact),
                        "output_shape": [FIXED_HEIGHT, FIXED_WIDTH, 3],
                        "output_dtype": (
                            "uint8" if output_contract_exact else "invalid"
                        ),
                        "output_c_contiguous": bool(output_contract_exact),
                        "deterministic_repeat_exact": bool(deterministic_repeat_exact),
                    },
                }
            )
        finally:
            if workspace is not None:
                workspace.close()
    return build_performance_report(rows, full_path_evidence=full_path_evidence)


def profile_fixed_720p_compositor_samples(
    samples: Sequence[CompositorFrameSample],
    *,
    warmup_frame_count: int = MIN_WARMUP_FRAMES,
    measured_frame_count: int = MIN_MEASURED_FRAMES,
    evidence_kind: str = "fixed-replay-measured",
    full_path_evidence: Mapping[str, object] | None = None,
    native_memory_bandwidth: Mapping[str, Mapping[str, object]] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Profile an explicitly supplied unique-frame sequence without pacing."""

    source = _explicit_replay_source(
        samples,
        kind="explicit-frame-sequence",
    )
    return _profile_replay_source(
        source,
        warmup_frame_count=warmup_frame_count,
        measured_frame_count=measured_frame_count,
        evidence_kind=evidence_kind,
        full_path_evidence=full_path_evidence,
        native_memory_bandwidth=native_memory_bandwidth,
        clock_ns=clock_ns,
    )


def profile_private_720p_replay_bundle(
    bundle_root: Path | str,
    *,
    warmup_frame_count: int = MIN_WARMUP_FRAMES,
    measured_frame_count: int = MIN_MEASURED_FRAMES,
    evidence_kind: str = "fixed-replay-measured",
    full_path_evidence: Mapping[str, object] | None = None,
    native_memory_bandwidth: Mapping[str, Mapping[str, object]] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Profile a validated owner-only replay bundle without copying its path."""

    return _profile_replay_source(
        _private_bundle_source(bundle_root),
        warmup_frame_count=warmup_frame_count,
        measured_frame_count=measured_frame_count,
        evidence_kind=evidence_kind,
        full_path_evidence=full_path_evidence,
        native_memory_bandwidth=native_memory_bandwidth,
        clock_ns=clock_ns,
    )


def profile_fixed_720p_compositor(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    clean_foreground: np.ndarray,
    *,
    warmup_frame_count: int = MIN_WARMUP_FRAMES,
    measured_frame_count: int = MIN_MEASURED_FRAMES,
    evidence_kind: str = "generated-proxy",
    full_path_evidence: Mapping[str, object] | None = None,
    native_memory_bandwidth: Mapping[str, Mapping[str, object]] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Execute a non-qualifying single-frame version of the eight-cell matrix.

    Use :func:`profile_fixed_720p_compositor_samples` or
    :func:`profile_private_720p_replay_bundle` for source-bound qualification.
    """

    foreground, backdrop, mask, clean_foreground = _fixed_inputs(
        foreground,
        backdrop,
        mask,
        clean_foreground,
    )
    source = _explicit_replay_source(
        (
            CompositorFrameSample(
                foreground_bgr=foreground,
                backdrop_bgr=backdrop,
                mask=mask,
                clean_foreground_bgr=clean_foreground,
                capture_sequence=0,
                capture_timestamp_ns=0,
            ),
        ),
        kind="single-frame",
    )
    return _profile_replay_source(
        source,
        warmup_frame_count=warmup_frame_count,
        measured_frame_count=measured_frame_count,
        evidence_kind=evidence_kind,
        full_path_evidence=full_path_evidence,
        native_memory_bandwidth=native_memory_bandwidth,
        clock_ns=clock_ns,
    )


def report_markdown(report: Mapping[str, object]) -> str:
    """Render a compact review summary from a validated report."""

    validated = validate_performance_report(report)
    matrix = cast(Mapping[str, object], validated["matrix"])
    full_path = cast(Mapping[str, object], validated["full_path"])
    decision = cast(Mapping[str, object], validated["decision"])
    scope = cast(Mapping[str, object], validated["scope"])
    source_scope = cast(Mapping[str, object], scope["source"])
    lines = [
        "# MATTE-3.4 fixed 720p performance report",
        "",
        f"- Outcome: `{decision['outcome']}`",
        (
            f"- Source: `{source_scope['kind']}` / "
            f"`{source_scope['source_frame_count']}` frame(s), SHA-256 "
            f"`{scope['source_sha256']}`"
        ),
        f"- Matrix outcome: `{matrix['outcome']}`",
        (
            "- Worst compositor p95 / signed 22 ms headroom: "
            f"`{matrix['worst_compositor_p95_ms']} ms` / "
            f"`{matrix['signed_worst_compositor_headroom_ms']} ms`"
        ),
        f"- Full-path outcome: `{full_path['outcome']}`",
        (
            "- Full-path signed 33.333334 ms headroom: "
            f"`{full_path['signed_service_headroom_ms']} ms`"
        ),
        (
            "- Full-path signed 22 ms compositor headroom: "
            f"`{full_path['signed_compositor_headroom_ms']} ms`"
        ),
        (
            "- Declared sustainable unique-frame profile: "
            f"`{decision['declared_sustainable_unique_frame_profile_fps']} FPS`"
        ),
        "- Output repeats credited as unique work: `false`",
        "- Reaction ready: `false` (later qualification must use residual headroom)",
        (
            "- Allocation bytes: known lower bounds; native memory bandwidth is "
            "reported only when a native counter is supplied"
        ),
        "",
        "| Variant | p50 ms | p95 ms | p99 ms | Headroom ms | Outcome |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in cast(Sequence[Mapping[str, object]], matrix["rows"]):
        total = cast(Mapping[str, object], row["timing_summary_ms"])["total"]
        summary = cast(Mapping[str, object], total)
        lines.append(
            f"| `{row['id']}` | {summary['p50']} | {summary['p95']} | "
            f"{summary['p99']} | {row['signed_compositor_headroom_ms']} | "
            f"`{row['outcome']}` |"
        )
    lines.extend(
        [
            "",
            "The arrival sweep is computed from ordered complete non-pacing "
            "cycle samples at "
            "30/27/24/20/15 FPS. It is not `1000 / p95`. Capture gaps, "
            "overwrites, deadline misses, pacing, and sink recovery remain "
            "separate evidence classes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_owner_json(path: Path | str, *, name: str) -> dict[str, object]:
    """Read one bounded owner-only JSON object used as local evidence."""

    from .matte_diagnostics import MAX_MANIFEST_BYTES, _read_private_file

    payload = _read_private_file(Path(path), max_bytes=MAX_MANIFEST_BYTES)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MattePerformanceError(f"{name} is malformed") from exc
    if not isinstance(value, dict):
        raise MattePerformanceError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def run_private_720p_profile(
    bundle_root: Path | str,
    output_root: Path | str,
    *,
    warmup_frame_count: int = MIN_WARMUP_FRAMES,
    measured_frame_count: int = MIN_MEASURED_FRAMES,
    full_path_evidence_path: Path | str | None = None,
    native_memory_bandwidth_path: Path | str | None = None,
    collect_full_path: bool = False,
    hardware_id: str = "local_cuda",
    cuda_device_id: int = 0,
    sink_backend: str = "pyvirtualcam",
) -> dict[str, Any]:
    """Profile one private replay and write a new content-free report directory."""

    from . import _platform as platform_fs
    from .matte_diagnostics import _atomic_private_write, _private_directory
    from .storage_tx import rename_noreplace

    if collect_full_path and full_path_evidence_path is not None:
        raise MattePerformanceError(
            "collect-full-path and full-path-evidence are mutually exclusive"
        )
    if collect_full_path and native_memory_bandwidth_path is not None:
        raise MattePerformanceError(
            "native bandwidth must be collected with and bound to the same "
            "full-path run before it can be joined"
        )
    full_path_evidence = (
        None
        if full_path_evidence_path is None
        else _read_owner_json(
            full_path_evidence_path,
            name="full-path performance evidence",
        )
    )
    native_memory_bandwidth: Mapping[str, Mapping[str, object]] | None = None
    if native_memory_bandwidth_path is not None:
        raw_bandwidth = _read_owner_json(
            native_memory_bandwidth_path,
            name="native memory-bandwidth evidence",
        )
        expected_ids = {variant.id for variant in COMPOSITOR_VARIANTS}
        if set(raw_bandwidth) != expected_ids or any(
            not isinstance(value, dict) for value in raw_bandwidth.values()
        ):
            raise MattePerformanceError(
                "native memory-bandwidth evidence must contain the exact "
                "eight compositor variants"
            )
        native_memory_bandwidth = cast(
            Mapping[str, Mapping[str, object]],
            raw_bandwidth,
        )

    bundle_path = Path(bundle_root).resolve()
    output_path = Path(output_root).resolve()
    if (
        bundle_path == output_path
        or bundle_path in output_path.parents
        or output_path in bundle_path.parents
    ):
        raise MattePerformanceError(
            "performance output must not overlap the private replay bundle"
        )
    try:
        output_path.lstat()
    except FileNotFoundError:
        pass
    else:
        raise MattePerformanceError("performance output directory already exists")

    # Finish all validation and expensive profiling before creating an output
    # artifact. Publishing uses a private sibling directory and one no-replace
    # rename, so interruption cannot expose an empty or half-written report.
    if collect_full_path:
        full_path_evidence = collect_private_720p_full_path_evidence(
            bundle_root,
            warmup_frame_count=warmup_frame_count,
            measured_frame_count=measured_frame_count,
            hardware_id=hardware_id,
            cuda_device_id=cuda_device_id,
            sink_backend=sink_backend,
        )
    report = profile_private_720p_replay_bundle(
        bundle_root,
        warmup_frame_count=warmup_frame_count,
        measured_frame_count=measured_frame_count,
        evidence_kind="fixed-replay-measured",
        full_path_evidence=full_path_evidence,
        native_memory_bandwidth=native_memory_bandwidth,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".custback-matte-performance-",
            dir=output_path.parent,
        )
    )
    published = False
    try:
        _private_directory(staging, create=False)
        _atomic_private_write(
            staging / "performance.json",
            _json_bytes(report),
        )
        _atomic_private_write(
            staging / "performance.md",
            report_markdown(report).encode("utf-8"),
        )
        rename_noreplace(staging, output_path)
        platform_fs.fsync_dir(output_path.parent)
        published = True
    finally:
        if not published:
            # The staging directory is process-owned and contains only these
            # two known filenames. Never recurse across a user-controlled path.
            for name in ("performance.json", "performance.md"):
                try:
                    (staging / name).unlink()
                except FileNotFoundError:
                    pass
            try:
                staging.rmdir()
            except FileNotFoundError:
                pass
    return report


def build_parser(
    *,
    prog: str = "custback matte-performance",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Profile the fixed unpaced 1280x720 compositor matrix and "
            "fail-closed RVM/CUDA service gate"
        ),
    )
    parser.add_argument(
        "bundle",
        help="complete owner-only privacy replay bundle",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="new owner-only content-free report directory",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=MIN_WARMUP_FRAMES,
        help=f"unreported warm-up calls per matrix cell (default {MIN_WARMUP_FRAMES})",
    )
    parser.add_argument(
        "--measured-frames",
        type=int,
        default=MIN_MEASURED_FRAMES,
        help=f"measured calls per matrix cell (default {MIN_MEASURED_FRAMES})",
    )
    full_path_group = parser.add_mutually_exclusive_group()
    full_path_group.add_argument(
        "--full-path-evidence",
        help=(
            "owner-only model-backed full-service JSON sidecar; omission keeps "
            "the complete path not_decidable"
        ),
    )
    full_path_group.add_argument(
        "--collect-full-path",
        action="store_true",
        help=(
            "run built-in RVM on required CUDA hardware and submit each replay "
            "frame through an explicit unpaced production sink"
        ),
    )
    parser.add_argument(
        "--hardware-id",
        default="local_cuda",
        help="bounded content-free label for the hardware qualification tier",
    )
    parser.add_argument(
        "--cuda-device-id",
        type=int,
        default=0,
        help="CUDA adapter index for full-path collection (default 0)",
    )
    parser.add_argument(
        "--sink-backend",
        choices=("pyvirtualcam", "native"),
        default="pyvirtualcam",
        help="explicit real output sink for full-path collection",
    )
    parser.add_argument(
        "--memory-bandwidth-evidence",
        help=(
            "optional owner-only native-counter JSON for the exact eight "
            "matrix variants"
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback matte-performance",
) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        report = run_private_720p_profile(
            args.bundle,
            args.output,
            warmup_frame_count=args.warmup_frames,
            measured_frame_count=args.measured_frames,
            full_path_evidence_path=args.full_path_evidence,
            native_memory_bandwidth_path=args.memory_bandwidth_evidence,
            collect_full_path=args.collect_full_path,
            hardware_id=args.hardware_id,
            cuda_device_id=args.cuda_device_id,
            sink_backend=args.sink_backend,
        )
    except (OSError, ValueError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2
    except RuntimeError:
        # GPU/provider and sink libraries often include host paths or device
        # details in their exception text. Keep the CLI failure actionable
        # without copying those private dependency diagnostics to stderr.
        print(f"{prog}: full-path performance collection failed", file=sys.stderr)
        return 2
    decision = cast(Mapping[str, object], report["decision"])
    matrix = cast(Mapping[str, object], report["matrix"])
    print(
        f"profiled {matrix['completed_row_count']} compositor matrix cell(s); "
        f"outcome {decision['outcome']}"
    )
    return 0 if decision["outcome"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
