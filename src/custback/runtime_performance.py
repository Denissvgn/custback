"""Bounded, path-free runtime performance health accounting.

The tracker intentionally accepts only scalar timings, bounded enum values, and
integer generation identities.  Public snapshots therefore cannot accidentally
contain camera devices, backdrop paths, model paths, URLs, or traceback text.

All timestamps use a monotonic clock.  The public contract contains only
relative durations and rates; absolute monotonic timestamps remain private.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, cast

RUNTIME_PERFORMANCE_SCHEMA_VERSION = 1
RUNTIME_PERFORMANCE_WINDOW_NS = 5_000_000_000
RUNTIME_PERFORMANCE_SAMPLE_LIMIT = 2048
RUNTIME_PERFORMANCE_WARMUP_NS = 3_000_000_000
RUNTIME_PERFORMANCE_DEGRADE_NS = 3_000_000_000
RUNTIME_PERFORMANCE_RECOVER_NS = 5_000_000_000
RUNTIME_PERFORMANCE_MIN_ATTAINMENT = 0.90
RUNTIME_PERFORMANCE_MAX_DEADLINE_MISS_RATIO = 0.05

# Names are a versioned public contract.  Keep the map fixed even when a stage
# has no samples so clients never have to infer schema from the active backend.
RUNTIME_STAGE_NAMES = (
    "capture.read",
    "segmentation.total",
    "background.total",
    "color_correction.estimate",
    "color_correction.apply",
    "compositor.input_mask_validation",
    "compositor.color_transform_application",
    "compositor.edge_band",
    "compositor.model_foreground_replacement",
    "compositor.backdrop_blur_resize",
    "compositor.light_wrap_temporal_filter",
    "compositor.light_wrap_interpolation",
    "compositor.final_blend_conversion",
    "compositor.internal_output_validation",
    "compositor.light_wrap",
    "compositor.prepare",
    "compositor.blend",
    "compositor.total",
    "pipeline.processing_only",
    "output.submission",
)

PerformanceState = Literal["warming", "healthy", "degraded", "failed"]
PerformanceReason = Literal[
    "none",
    "output-attainment",
    "unique-attainment",
    "output-and-unique-attainment",
    "processing-deadline-miss",
    "multiple-performance-gates",
    "publisher-failed",
    "pipeline-failed",
]
PublisherMode = Literal["uninitialized", "sink-paced", "deadline-paced"]
PublisherState = Literal["starting", "running", "stopped", "failed"]
ColorCorrectionMode = Literal["off", "auto"]

_FAILURE_REASONS = {"publisher-failed", "pipeline-failed"}
_PUBLISHER_MODES = {"uninitialized", "sink-paced", "deadline-paced"}
_PUBLISHER_STATES = {"starting", "running", "stopped", "failed"}
_COLOR_CORRECTION_MODES = {"off", "auto"}
_MAX_PUBLIC_DURATION_MS = 3_600_000.0
_MAX_PUBLIC_COUNT = 2**63 - 1
_MAX_PUBLIC_RATE = 10_000.0
_MAX_PUBLIC_RATIO = 10_000.0
_PERFORMANCE_STATES = {"warming", "healthy", "degraded", "failed"}
_PERFORMANCE_REASONS = {
    "none",
    "output-attainment",
    "unique-attainment",
    "output-and-unique-attainment",
    "processing-deadline-miss",
    "multiple-performance-gates",
    "publisher-failed",
    "pipeline-failed",
}
_DEGRADED_REASONS = _PERFORMANCE_REASONS - _FAILURE_REASONS - {"none"}
_COUNTER_NAMES = (
    "processing_completed_count",
    "output_send_count",
    "sent_unique_base_count",
    "processing_deadline_miss_count",
    "output_schedule_late_count",
)
_METRIC_NAMES = (
    "window_duration_s",
    "window_sample_count",
    "output_send_fps",
    "processing_completed_fps",
    "sent_unique_base_fps",
    "output_attainment",
    "unique_attainment",
    "processing_deadline_miss_ratio",
    "output_schedule_late_ratio",
    "stage_p50_ms",
    "stage_p95_ms",
    "dominant_stage",
)
_EPOCH_KEY_NAMES = frozenset(
    {
        "config_version",
        "capture_generation",
        "segmentation_generation",
        "backdrop_generation",
    }
)
_EPOCH_NAMES = frozenset(
    {
        "key",
        "state",
        "reason",
        "duration_s",
        *_COUNTER_NAMES,
        *_METRIC_NAMES,
    }
)
_PUBLISHER_NAMES = frozenset(
    {
        "mode",
        "state",
        "handoff_overwrite_count",
        "missed_slot_count",
        "slate_send_count",
        "pending_depth",
        "output_base_config_version",
    }
)
_TOP_LEVEL_NAMES = frozenset(
    {
        "schema_version",
        "state",
        "reason",
        "target_fps",
        *_METRIC_NAMES,
        "output_healthy",
        "unique_healthy",
        "current_epoch",
        "last_closed_epoch",
        "startup",
        "publisher",
        "recommended_mitigation",
    }
)


def _non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{name} must be a finite number in range")
    return result


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _rounded(value: float) -> float:
    return round(value, 3)


def _strict_mapping(
    value: object,
    fields: frozenset[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly the version-1 fields")
    return cast(Mapping[str, object], value)


def _validated_count(
    value: object, name: str, *, maximum: int = _MAX_PUBLIC_COUNT
) -> int:
    result = _non_negative_int(value, name)
    if result > maximum:
        raise ValueError(f"{name} exceeds the public bound")
    return result


def _validated_state_reason(
    state_value: object,
    reason_value: object,
    name: str,
) -> tuple[PerformanceState, PerformanceReason]:
    if not isinstance(state_value, str) or state_value not in _PERFORMANCE_STATES:
        raise ValueError(f"{name} state is invalid")
    if not isinstance(reason_value, str) or reason_value not in _PERFORMANCE_REASONS:
        raise ValueError(f"{name} reason is invalid")
    state = cast(PerformanceState, state_value)
    reason = cast(PerformanceReason, reason_value)
    if state in {"warming", "healthy"} and reason != "none":
        raise ValueError(f"{name} healthy/warming state requires reason none")
    if state == "degraded" and reason not in _DEGRADED_REASONS:
        raise ValueError(f"{name} degraded state requires a performance reason")
    if state == "failed" and reason not in _FAILURE_REASONS:
        raise ValueError(f"{name} failed state requires a failure reason")
    return state, reason


def _validated_stage_map(value: object, name: str) -> dict[str, float | None]:
    raw = _strict_mapping(value, frozenset(RUNTIME_STAGE_NAMES), name)
    validated: dict[str, float | None] = {}
    for stage in RUNTIME_STAGE_NAMES:
        sample = raw[stage]
        if sample is None:
            validated[stage] = None
        else:
            validated[stage] = _finite_float(
                sample,
                f"{name} {stage}",
                minimum=0.0,
                maximum=_MAX_PUBLIC_DURATION_MS,
            )
    return validated


def _validated_metrics(value: Mapping[str, object], name: str) -> dict[str, object]:
    dominant = value["dominant_stage"]
    if dominant is not None and dominant not in RUNTIME_STAGE_NAMES:
        raise ValueError(f"{name} dominant stage is invalid")
    return {
        "window_duration_s": _finite_float(
            value["window_duration_s"],
            f"{name} window duration",
            minimum=0.0,
            maximum=RUNTIME_PERFORMANCE_WINDOW_NS / 1_000_000_000.0,
        ),
        "window_sample_count": _validated_count(
            value["window_sample_count"],
            f"{name} window sample count",
            maximum=RUNTIME_PERFORMANCE_SAMPLE_LIMIT,
        ),
        "output_send_fps": _finite_float(
            value["output_send_fps"],
            f"{name} output send FPS",
            minimum=0.0,
            maximum=_MAX_PUBLIC_RATE,
        ),
        "processing_completed_fps": _finite_float(
            value["processing_completed_fps"],
            f"{name} processing completed FPS",
            minimum=0.0,
            maximum=_MAX_PUBLIC_RATE,
        ),
        "sent_unique_base_fps": _finite_float(
            value["sent_unique_base_fps"],
            f"{name} sent unique base FPS",
            minimum=0.0,
            maximum=_MAX_PUBLIC_RATE,
        ),
        "output_attainment": _finite_float(
            value["output_attainment"],
            f"{name} output attainment",
            minimum=0.0,
            maximum=_MAX_PUBLIC_RATIO,
        ),
        "unique_attainment": _finite_float(
            value["unique_attainment"],
            f"{name} unique attainment",
            minimum=0.0,
            maximum=_MAX_PUBLIC_RATIO,
        ),
        "processing_deadline_miss_ratio": _finite_float(
            value["processing_deadline_miss_ratio"],
            f"{name} processing deadline miss ratio",
            minimum=0.0,
            maximum=1.0,
        ),
        "output_schedule_late_ratio": _finite_float(
            value["output_schedule_late_ratio"],
            f"{name} output schedule late ratio",
            minimum=0.0,
            maximum=1.0,
        ),
        "stage_p50_ms": _validated_stage_map(
            value["stage_p50_ms"], f"{name} stage p50"
        ),
        "stage_p95_ms": _validated_stage_map(
            value["stage_p95_ms"], f"{name} stage p95"
        ),
        "dominant_stage": dominant,
    }


def _validated_epoch(value: object, name: str) -> dict[str, object]:
    raw = _strict_mapping(value, _EPOCH_NAMES, name)
    raw_key = _strict_mapping(raw["key"], _EPOCH_KEY_NAMES, f"{name} key")
    state, reason = _validated_state_reason(raw["state"], raw["reason"], name)
    result: dict[str, object] = {
        "key": {
            field_name: _validated_count(
                raw_key[field_name], f"{name} key {field_name}"
            )
            for field_name in _EPOCH_KEY_NAMES
        },
        "state": state,
        "reason": reason,
        "duration_s": _finite_float(
            raw["duration_s"],
            f"{name} duration",
            minimum=0.0,
            maximum=1_000_000_000.0,
        ),
    }
    result.update(
        {
            field_name: _validated_count(raw[field_name], f"{name} {field_name}")
            for field_name in _COUNTER_NAMES
        }
    )
    result.update(_validated_metrics(raw, name))
    return result


def _validated_publisher(value: object) -> dict[str, object]:
    raw = _strict_mapping(value, _PUBLISHER_NAMES, "runtime publisher")
    mode = raw["mode"]
    state = raw["state"]
    if not isinstance(mode, str) or mode not in _PUBLISHER_MODES:
        raise ValueError("runtime publisher mode is invalid")
    if not isinstance(state, str) or state not in _PUBLISHER_STATES:
        raise ValueError("runtime publisher state is invalid")
    pending_depth = raw["pending_depth"]
    if type(pending_depth) is not int or pending_depth not in (0, 1):
        raise ValueError("runtime publisher pending depth must be zero or one")
    return {
        "mode": mode,
        "state": state,
        "handoff_overwrite_count": _validated_count(
            raw["handoff_overwrite_count"], "runtime publisher handoff overwrites"
        ),
        "missed_slot_count": _validated_count(
            raw["missed_slot_count"], "runtime publisher missed slots"
        ),
        "slate_send_count": _validated_count(
            raw["slate_send_count"], "runtime publisher slate sends"
        ),
        "pending_depth": pending_depth,
        "output_base_config_version": _validated_count(
            raw["output_base_config_version"],
            "runtime publisher output base config version",
        ),
    }


def _validated_mitigation(
    value: object,
    *,
    config_version: int,
    state: PerformanceState,
) -> dict[str, object] | None:
    if value is None:
        return None
    if state != "degraded" or not isinstance(value, Mapping):
        raise ValueError("runtime mitigation is only valid while degraded")
    raw = _strict_mapping(
        value,
        frozenset({"config_version", "kind", "patch"}),
        "runtime mitigation",
    )
    kind = raw["kind"]
    candidates: dict[str, dict[str, object]] = {
        "disable-color-and-light-wrap": {
            "compositing": {
                "color_correction": {"mode": "off"},
                "light_wrap": 0.0,
            }
        },
        "disable-color-correction": {
            "compositing": {"color_correction": {"mode": "off"}}
        },
        "disable-light-wrap": {"compositing": {"light_wrap": 0.0}},
        "review-backend-or-diagnostic-target": {},
    }
    if not isinstance(kind, str) or kind not in candidates:
        raise ValueError("runtime mitigation kind is invalid")
    version = _validated_count(
        raw["config_version"], "runtime mitigation config version"
    )
    if version != config_version or raw["patch"] != candidates[kind]:
        raise ValueError("runtime mitigation is not bound to its configuration")
    return {
        "config_version": version,
        "kind": kind,
        "patch": copy.deepcopy(candidates[kind]),
    }


def empty_runtime_performance_status() -> dict[str, object]:
    """Return the deterministic pre-ready version-1 public snapshot."""

    stages = dict.fromkeys(RUNTIME_STAGE_NAMES)
    metrics: dict[str, object] = {
        "window_duration_s": 0.0,
        "window_sample_count": 0,
        "output_send_fps": 0.0,
        "processing_completed_fps": 0.0,
        "sent_unique_base_fps": 0.0,
        "output_attainment": 0.0,
        "unique_attainment": 0.0,
        "processing_deadline_miss_ratio": 0.0,
        "output_schedule_late_ratio": 0.0,
        "stage_p50_ms": stages,
        "stage_p95_ms": dict(stages),
        "dominant_stage": None,
    }
    counters = dict.fromkeys(_COUNTER_NAMES, 0)
    current_epoch: dict[str, object] = {
        "key": dict.fromkeys(_EPOCH_KEY_NAMES, 0),
        "state": "warming",
        "reason": "none",
        "duration_s": 0.0,
        **counters,
        **metrics,
    }
    public = {
        "schema_version": RUNTIME_PERFORMANCE_SCHEMA_VERSION,
        "state": "warming",
        "reason": "none",
        "target_fps": 0.0,
        **metrics,
        "output_healthy": False,
        "unique_healthy": False,
        "current_epoch": current_epoch,
        "last_closed_epoch": None,
        "startup": counters,
        "publisher": PublisherTelemetry().as_dict(),
        "recommended_mitigation": None,
    }
    return validate_runtime_performance_status(public)


def validate_runtime_performance_status(value: object) -> dict[str, object]:
    """Validate and defensively copy one strict path-free v1 snapshot."""

    raw = _strict_mapping(value, _TOP_LEVEL_NAMES, "runtime performance")
    if raw["schema_version"] != RUNTIME_PERFORMANCE_SCHEMA_VERSION:
        raise ValueError(
            f"runtime performance schema version must be "
            f"{RUNTIME_PERFORMANCE_SCHEMA_VERSION}"
        )
    state, reason = _validated_state_reason(
        raw["state"], raw["reason"], "runtime performance"
    )
    metrics = _validated_metrics(raw, "runtime performance")
    current = _validated_epoch(raw["current_epoch"], "runtime current epoch")
    if current["state"] != state or current["reason"] != reason:
        raise ValueError("runtime current epoch health does not match top level")
    for field_name in _METRIC_NAMES:
        if current[field_name] != metrics[field_name]:
            raise ValueError("runtime current epoch metrics do not match top level")

    last_raw = raw["last_closed_epoch"]
    last = (
        None
        if last_raw is None
        else _validated_epoch(last_raw, "runtime last closed epoch")
    )
    startup_raw = _strict_mapping(
        raw["startup"], frozenset(_COUNTER_NAMES), "runtime startup"
    )
    startup = {
        field_name: _validated_count(
            startup_raw[field_name], f"runtime startup {field_name}"
        )
        for field_name in _COUNTER_NAMES
    }
    publisher = _validated_publisher(raw["publisher"])
    current_key = cast(dict[str, int], current["key"])
    mitigation = _validated_mitigation(
        raw["recommended_mitigation"],
        config_version=current_key["config_version"],
        state=state,
    )
    output_healthy = raw["output_healthy"]
    unique_healthy = raw["unique_healthy"]
    if type(output_healthy) is not bool or type(unique_healthy) is not bool:
        raise ValueError("runtime performance health flags must be boolean")
    if output_healthy != (
        cast(float, metrics["output_attainment"]) >= RUNTIME_PERFORMANCE_MIN_ATTAINMENT
    ) or unique_healthy != (
        cast(float, metrics["unique_attainment"]) >= RUNTIME_PERFORMANCE_MIN_ATTAINMENT
    ):
        raise ValueError("runtime performance health flags are inconsistent")

    return {
        "schema_version": RUNTIME_PERFORMANCE_SCHEMA_VERSION,
        "state": state,
        "reason": reason,
        "target_fps": _finite_float(
            raw["target_fps"],
            "runtime performance target FPS",
            minimum=0.0,
            maximum=1000.0,
        ),
        **metrics,
        "output_healthy": output_healthy,
        "unique_healthy": unique_healthy,
        "current_epoch": current,
        "last_closed_epoch": last,
        "startup": startup,
        "publisher": publisher,
        "recommended_mitigation": mitigation,
    }


@dataclass(frozen=True)
class PerformanceEpochKey:
    """Scalar identities that bind samples to one effective configuration."""

    config_version: int
    capture_generation: int
    segmentation_generation: int
    backdrop_generation: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _non_negative_int(value, name.replace("_", " "))

    def as_dict(self) -> dict[str, int]:
        """Return a new JSON-compatible mapping."""

        return asdict(self)


@dataclass(frozen=True)
class PublisherTelemetry:
    """Run-lifetime output-publisher state supplied by its owning thread."""

    mode: PublisherMode = "uninitialized"
    state: PublisherState = "starting"
    handoff_overwrite_count: int = 0
    missed_slot_count: int = 0
    slate_send_count: int = 0
    pending_depth: int = 0
    output_base_config_version: int = 0

    def __post_init__(self) -> None:
        if self.mode not in _PUBLISHER_MODES:
            raise ValueError("publisher mode is invalid")
        if self.state not in _PUBLISHER_STATES:
            raise ValueError("publisher state is invalid")
        for name in (
            "handoff_overwrite_count",
            "missed_slot_count",
            "slate_send_count",
            "output_base_config_version",
        ):
            _non_negative_int(getattr(self, name), name.replace("_", " "))
        if type(self.pending_depth) is not int or self.pending_depth not in (0, 1):
            raise ValueError("publisher pending depth must be zero or one")

    def as_dict(self) -> dict[str, object]:
        """Return a new JSON-compatible mapping."""

        return asdict(self)


@dataclass(frozen=True)
class _Sample:
    at_ns: int
    processing_completed: bool = False
    output_sent: bool = False
    sent_unique_base: bool = False
    processing_deadline_missed: bool = False
    output_schedule_late: bool = False
    stages_ms: tuple[tuple[str, float], ...] = ()


@dataclass
class _Counters:
    processing_completed_count: int = 0
    output_send_count: int = 0
    sent_unique_base_count: int = 0
    processing_deadline_miss_count: int = 0
    output_schedule_late_count: int = 0

    def record(self, sample: _Sample) -> None:
        self.processing_completed_count += int(sample.processing_completed)
        self.output_send_count += int(sample.output_sent)
        self.sent_unique_base_count += int(sample.sent_unique_base)
        self.processing_deadline_miss_count += int(sample.processing_deadline_missed)
        self.output_schedule_late_count += int(sample.output_schedule_late)

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class _ClosedEpoch:
    """Private bounded accounting retained for one producer generation.

    Processing closes at the generation boundary, but an already queued base
    from that generation can still reach the sink afterward.  Keeping this
    bucket mutable while any publisher envelope references it lets every late
    accepted send remain attributed to its immutable producer key without
    contaminating the active epoch.  Its effective end advances only for such
    a late accepted send.
    """

    key: PerformanceEpochKey
    state: PerformanceState
    reason: PerformanceReason
    started_ns: int
    ended_ns: int
    samples: deque[_Sample]
    counters: _Counters
    unhealthy_since_ns: int | None
    healthy_since_ns: int | None
    failed_reason: PerformanceReason | None
    steady_state: bool
    finalized: bool = False


class RuntimePerformanceTracker:
    """Track runtime performance health in one bounded five-second window.

    Callers mark the transition to steady state with :meth:`mark_ready`.
    Processing and output are recorded independently so a paced publisher can
    remain healthy while unique visual updates correctly report degradation.
    Generation changes must be supplied through :meth:`bind_epoch`. Publisher
    envelopes call :meth:`retain_epoch` before handoff and :meth:`release_epoch`
    exactly once when they leave the depth-one publisher. Closed buckets still
    referenced by current or pending pixels remain private and mutable; only
    the current and most recently closed summaries are public.
    """

    def __init__(
        self,
        target_output_fps: float,
        epoch_key: PerformanceEpochKey,
        *,
        color_correction_mode: ColorCorrectionMode = "off",
        light_wrap: float = 0.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._target_fps = _finite_float(
            target_output_fps, "target output FPS", minimum=0.001, maximum=1000.0
        )
        if not isinstance(epoch_key, PerformanceEpochKey):
            raise TypeError("epoch key must be a PerformanceEpochKey")
        if color_correction_mode not in _COLOR_CORRECTION_MODES:
            raise ValueError("color correction mode is invalid")
        self._color_correction_mode = color_correction_mode
        self._light_wrap = _finite_float(
            light_wrap, "light wrap", minimum=0.0, maximum=1.0
        )
        if not callable(clock_ns):
            raise TypeError("runtime performance clock must be callable")

        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        initial_ns = _non_negative_int(clock_ns(), "initial timestamp")
        self._last_observed_ns = initial_ns
        self._run_started_ns = initial_ns
        self._epoch_started_ns = initial_ns
        self._ready_at_ns: int | None = None
        self._epoch_key = epoch_key
        self._samples: deque[_Sample] = deque(maxlen=RUNTIME_PERFORMANCE_SAMPLE_LIMIT)
        self._current = _Counters()
        self._startup = _Counters()
        self._closed_epochs: dict[PerformanceEpochKey, _ClosedEpoch] = {}
        self._last_closed_epoch_key: PerformanceEpochKey | None = None
        self._epoch_references: dict[PerformanceEpochKey, int] = {}
        self._finalized_epoch_summaries: deque[dict[str, object]] = deque(maxlen=4)

        self._state: PerformanceState = "warming"
        self._reason: PerformanceReason = "none"
        self._unhealthy_since_ns: int | None = None
        self._healthy_since_ns: int | None = None
        self._failed_reason: PerformanceReason | None = None

        self._publisher = PublisherTelemetry()

    @property
    def retained_sample_count(self) -> int:
        """Return private ring occupancy for boundedness assertions."""

        with self._lock:
            return len(self._samples)

    @property
    def current_epoch_key(self) -> PerformanceEpochKey:
        """Return the immutable scalar identity receiving new samples."""

        with self._lock:
            return self._epoch_key

    @property
    def retained_closed_epoch_count(self) -> int:
        """Return the bounded number of private closed producer buckets."""

        with self._lock:
            return len(self._closed_epochs)

    def retain_epoch(self, epoch_key: PerformanceEpochKey) -> bool:
        """Retain one producer epoch while a publisher envelope owns it.

        The output publisher owns at most a current and a pending envelope.
        Registration happens immediately before handoff, so a third distinct
        key is permitted transiently while submitting a replacement. Returning
        ``False`` rejects an unknown/finalized identity without reviving it.
        """

        if not isinstance(epoch_key, PerformanceEpochKey):
            raise TypeError("retained epoch must be a PerformanceEpochKey")
        with self._lock:
            if epoch_key != self._epoch_key:
                closed = self._closed_epochs.get(epoch_key)
                if closed is None or closed.finalized:
                    return False
            if epoch_key not in self._epoch_references:
                distinct_references = sum(
                    count > 0 for count in self._epoch_references.values()
                )
                if distinct_references >= 3:
                    raise RuntimeError(
                        "publisher retained more than three producer epochs"
                    )
            self._epoch_references[epoch_key] = (
                self._epoch_references.get(epoch_key, 0) + 1
            )
            return True

    def release_epoch(
        self,
        epoch_key: PerformanceEpochKey,
    ) -> dict[str, object] | None:
        """Release one publisher reference and finalize a closed epoch at zero.

        The optional return value is the same path-free summary queued for
        :meth:`take_finalized_epoch_summaries`. A missing reference is an
        invariant violation and is rejected so double-release cannot silently
        erase accounting.
        """

        if not isinstance(epoch_key, PerformanceEpochKey):
            raise TypeError("released epoch must be a PerformanceEpochKey")
        with self._lock:
            count = self._epoch_references.get(epoch_key, 0)
            if count <= 0:
                raise RuntimeError("publisher released an unretained epoch")
            if count == 1:
                self._epoch_references.pop(epoch_key, None)
            else:
                self._epoch_references[epoch_key] = count - 1
            summary: dict[str, object] | None = None
            closed = self._closed_epochs.get(epoch_key)
            if count == 1 and closed is not None and not closed.finalized:
                summary = self._finalize_closed_epoch_locked(epoch_key)
            self._cleanup_closed_epochs_locked()
            return copy.deepcopy(summary)

    def take_finalized_epoch_summaries(self) -> tuple[dict[str, object], ...]:
        """Drain path-free summaries that became final since the prior call."""

        with self._lock:
            summaries = tuple(
                copy.deepcopy(item) for item in self._finalized_epoch_summaries
            )
            self._finalized_epoch_summaries.clear()
            return summaries

    def _resolve_time_locked(self, at_ns: int | None) -> int:
        value = self._clock_ns() if at_ns is None else at_ns
        now_ns = _non_negative_int(value, "event timestamp")
        if now_ns < self._last_observed_ns:
            raise ValueError("runtime performance clock moved backwards")
        self._last_observed_ns = now_ns
        return now_ns

    def _prune_locked(self, now_ns: int) -> None:
        cutoff_ns = now_ns - RUNTIME_PERFORMANCE_WINDOW_NS
        while self._samples and self._samples[0].at_ns < cutoff_ns:
            self._samples.popleft()

    def mark_ready(self, *, at_ns: int | None = None) -> None:
        """Start steady-state measurement, excluding all startup samples."""

        with self._lock:
            now_ns = self._resolve_time_locked(at_ns)
            if self._ready_at_ns is not None:
                return
            self._ready_at_ns = now_ns
            self._epoch_started_ns = now_ns
            self._samples.clear()
            self._current = _Counters()
            self._state = "warming"
            self._reason = "none"
            self._unhealthy_since_ns = None
            self._healthy_since_ns = None

    def record_processing(
        self,
        *,
        deadline_missed: bool,
        stages_ms: Mapping[str, float | None] | None = None,
        at_ns: int | None = None,
    ) -> None:
        """Record one completed processing result, sent or overwritten."""

        if type(deadline_missed) is not bool:
            raise TypeError("processing deadline flag must be boolean")
        validated_stages = self._validate_stages(stages_ms)
        with self._lock:
            now_ns = self._resolve_time_locked(at_ns)
            sample = _Sample(
                at_ns=now_ns,
                processing_completed=True,
                processing_deadline_missed=deadline_missed,
                stages_ms=validated_stages,
            )
            self._record_locked(sample)

    def record_output(
        self,
        *,
        unique_base: bool,
        schedule_late: bool,
        submission_ms: float | None = None,
        expected_epoch: PerformanceEpochKey | None = None,
        at_ns: int | None = None,
    ) -> bool:
        """Record one successfully accepted output submission.

        When ``expected_epoch`` is supplied, the immutable producer identity
        is compared with the active and publisher-retained closed epochs under
        the same lock that records the sample. An unknown or already-finalized
        generation is ignored and returns ``False`` rather than contaminating
        the active epoch.
        """

        if type(unique_base) is not bool:
            raise TypeError("unique-base flag must be boolean")
        if type(schedule_late) is not bool:
            raise TypeError("output schedule-late flag must be boolean")
        if expected_epoch is not None and not isinstance(
            expected_epoch, PerformanceEpochKey
        ):
            raise TypeError("expected output epoch must be a PerformanceEpochKey")
        stages: tuple[tuple[str, float], ...] = ()
        if submission_ms is not None:
            stages = (
                (
                    "output.submission",
                    _finite_float(
                        submission_ms,
                        "output submission duration",
                        minimum=0.0,
                        maximum=_MAX_PUBLIC_DURATION_MS,
                    ),
                ),
            )
        with self._lock:
            now_ns = self._resolve_time_locked(at_ns)
            sample = _Sample(
                at_ns=now_ns,
                output_sent=True,
                sent_unique_base=unique_base,
                output_schedule_late=schedule_late,
                stages_ms=stages,
            )
            if expected_epoch is not None and expected_epoch != self._epoch_key:
                closed = self._closed_epochs.get(expected_epoch)
                if closed is None or closed.finalized:
                    return False
                closed.samples.append(sample)
                closed.counters.record(sample)
                closed.ended_ns = max(closed.ended_ns, now_ns)
                self._prune_samples_locked(closed.samples, closed.ended_ns)
                self._evaluate_closed_locked(closed, closed.ended_ns)
                return True
            self._record_locked(sample)
            return True

    @staticmethod
    def _validate_stages(
        stages_ms: Mapping[str, float | None] | None,
    ) -> tuple[tuple[str, float], ...]:
        if stages_ms is None:
            return ()
        if not isinstance(stages_ms, Mapping):
            raise TypeError("runtime stage timings must be a mapping")
        unknown = set(stages_ms) - set(RUNTIME_STAGE_NAMES)
        if unknown:
            raise ValueError("runtime stage timings contain an unknown key")
        result: list[tuple[str, float]] = []
        for name in RUNTIME_STAGE_NAMES:
            value = stages_ms.get(name)
            if value is None:
                continue
            result.append(
                (
                    name,
                    _finite_float(
                        value,
                        f"{name} duration",
                        minimum=0.0,
                        maximum=_MAX_PUBLIC_DURATION_MS,
                    ),
                )
            )
        return tuple(result)

    def _record_locked(self, sample: _Sample) -> None:
        if self._ready_at_ns is None:
            self._startup.record(sample)
            return
        self._samples.append(sample)
        self._current.record(sample)
        self._prune_locked(sample.at_ns)
        self._evaluate_locked(sample.at_ns)

    def update_publisher(
        self,
        telemetry: PublisherTelemetry,
        *,
        at_ns: int | None = None,
    ) -> None:
        """Publish an atomic, monotonic output-publisher scalar snapshot."""

        if not isinstance(telemetry, PublisherTelemetry):
            raise TypeError("publisher telemetry must be PublisherTelemetry")
        with self._lock:
            now_ns = self._resolve_time_locked(at_ns)
            for field_name in (
                "handoff_overwrite_count",
                "missed_slot_count",
                "slate_send_count",
            ):
                if getattr(telemetry, field_name) < getattr(
                    self._publisher, field_name
                ):
                    raise ValueError("publisher lifetime counters cannot decrease")
            self._publisher = telemetry
            if telemetry.state == "failed" and self._failed_reason is None:
                self._failed_reason = "publisher-failed"
            self._evaluate_locked(now_ns)

    def mark_failed(
        self,
        reason: Literal["publisher-failed", "pipeline-failed"],
        *,
        at_ns: int | None = None,
    ) -> None:
        """Latch a bounded fatal reason for the current run."""

        if reason not in _FAILURE_REASONS:
            raise ValueError("runtime performance failure reason is invalid")
        with self._lock:
            now_ns = self._resolve_time_locked(at_ns)
            if self._failed_reason is None:
                self._failed_reason = reason
            self._evaluate_locked(now_ns)

    def bind_epoch(
        self,
        epoch_key: PerformanceEpochKey,
        *,
        color_correction_mode: ColorCorrectionMode,
        light_wrap: float,
        at_ns: int | None = None,
    ) -> bool:
        """Close and replace the epoch when any supplied identity changes.

        Returns ``True`` when a new epoch was opened.  Rebinding the same key
        only refreshes its already-scalar mitigation controls and returns
        ``False``.
        """

        if not isinstance(epoch_key, PerformanceEpochKey):
            raise TypeError("epoch key must be a PerformanceEpochKey")
        if color_correction_mode not in _COLOR_CORRECTION_MODES:
            raise ValueError("color correction mode is invalid")
        validated_wrap = _finite_float(
            light_wrap, "light wrap", minimum=0.0, maximum=1.0
        )
        with self._lock:
            now_ns = self._resolve_time_locked(at_ns)
            if epoch_key == self._epoch_key:
                self._color_correction_mode = color_correction_mode
                self._light_wrap = validated_wrap
                return False
            prior = self._closed_epochs.get(epoch_key)
            if prior is not None and not prior.finalized:
                raise ValueError("cannot reopen a retained performance epoch")
            self._closed_epochs.pop(epoch_key, None)
            self._evaluate_locked(now_ns)
            closed = _ClosedEpoch(
                key=self._epoch_key,
                state=self._state,
                reason=self._reason,
                started_ns=self._epoch_started_ns,
                ended_ns=now_ns,
                samples=deque(
                    self._samples,
                    maxlen=RUNTIME_PERFORMANCE_SAMPLE_LIMIT,
                ),
                counters=_Counters(**self._current.as_dict()),
                unhealthy_since_ns=self._unhealthy_since_ns,
                healthy_since_ns=self._healthy_since_ns,
                failed_reason=self._failed_reason,
                steady_state=self._ready_at_ns is not None,
            )
            self._closed_epochs[closed.key] = closed
            self._last_closed_epoch_key = closed.key
            if self._epoch_references.get(closed.key, 0) == 0:
                self._finalize_closed_epoch_locked(closed.key)
            self._cleanup_closed_epochs_locked()
            self._epoch_key = epoch_key
            self._color_correction_mode = color_correction_mode
            self._light_wrap = validated_wrap
            self._epoch_started_ns = now_ns
            self._samples.clear()
            self._current = _Counters()
            self._unhealthy_since_ns = None
            self._healthy_since_ns = None
            if self._failed_reason is None:
                self._state = "warming"
                self._reason = "none"
            else:
                # Output ambiguity is fatal for the run.  A concurrent config
                # generation change must not accidentally clear that latch.
                self._state = "failed"
                self._reason = self._failed_reason
            return True

    def _cleanup_closed_epochs_locked(self) -> None:
        """Discard only finalized non-public buckets with no live envelope."""

        for key, closed in tuple(self._closed_epochs.items()):
            if (
                key != self._last_closed_epoch_key
                and closed.finalized
                and self._epoch_references.get(key, 0) == 0
            ):
                self._closed_epochs.pop(key, None)

    def _finalize_closed_epoch_locked(
        self,
        epoch_key: PerformanceEpochKey,
    ) -> dict[str, object]:
        closed = self._closed_epochs[epoch_key]
        if closed.finalized:  # pragma: no cover - caller invariant
            raise RuntimeError("performance epoch was finalized twice")
        self._evaluate_closed_locked(closed, closed.ended_ns)
        closed.finalized = True
        summary = self._summary_for_closed_epoch_locked(closed)
        self._finalized_epoch_summaries.append(copy.deepcopy(summary))
        return summary

    @staticmethod
    def _prune_samples_locked(samples: deque[_Sample], now_ns: int) -> None:
        cutoff_ns = now_ns - RUNTIME_PERFORMANCE_WINDOW_NS
        while samples and samples[0].at_ns < cutoff_ns:
            samples.popleft()

    def _window_metrics_for_locked(
        self,
        samples_source: deque[_Sample],
        epoch_started_ns: int,
        now_ns: int,
    ) -> dict[str, object]:
        self._prune_samples_locked(samples_source, now_ns)
        window_start_ns = self._effective_window_start_ns(
            samples_source,
            epoch_started_ns,
            now_ns,
        )
        duration_ns = max(0, now_ns - window_start_ns)
        duration_s = duration_ns / 1_000_000_000.0
        # The left edge is exclusive: an event exactly on the boundary starts
        # the first interval and must not add a fictitious extra frame to a
        # stable 30 Hz five-second window.
        samples = tuple(
            sample for sample in samples_source if sample.at_ns > window_start_ns
        )
        output_count = sum(sample.output_sent for sample in samples)
        processing_count = sum(sample.processing_completed for sample in samples)
        unique_count = sum(sample.sent_unique_base for sample in samples)
        deadline_misses = sum(sample.processing_deadline_missed for sample in samples)
        late_sends = sum(sample.output_schedule_late for sample in samples)

        if duration_s <= 0.0:
            output_fps = processing_fps = unique_fps = 0.0
        else:
            output_fps = output_count / duration_s
            processing_fps = processing_count / duration_s
            unique_fps = unique_count / duration_s
        output_attainment = output_fps / self._target_fps
        unique_attainment = unique_fps / self._target_fps
        miss_ratio = deadline_misses / processing_count if processing_count else 0.0
        late_ratio = late_sends / output_count if output_count else 0.0

        stage_values: dict[str, list[float]] = {
            name: [] for name in RUNTIME_STAGE_NAMES
        }
        for sample in samples:
            for name, value in sample.stages_ms:
                stage_values[name].append(value)
        p50 = {
            name: (
                None
                if (value := _nearest_rank(stage_values[name], 0.50)) is None
                else _rounded(value)
            )
            for name in RUNTIME_STAGE_NAMES
        }
        p95 = {
            name: (
                None
                if (value := _nearest_rank(stage_values[name], 0.95)) is None
                else _rounded(value)
            )
            for name in RUNTIME_STAGE_NAMES
        }
        # The whole-pipeline aggregate is useful as an end-to-end gate but not
        # actionable as a dominant stage.  Every other fixed key is eligible.
        candidates = {
            name: value
            for name, value in p95.items()
            if name != "pipeline.processing_only" and value is not None
        }
        dominant_stage = (
            max(candidates, key=lambda name: (candidates[name], name))
            if candidates
            else None
        )
        return {
            "window_duration_s": _rounded(duration_s),
            "window_sample_count": len(samples),
            "output_send_fps": _rounded(output_fps),
            "processing_completed_fps": _rounded(processing_fps),
            "sent_unique_base_fps": _rounded(unique_fps),
            # Qualification compares the unrounded ratios.  Preserve that
            # precision publicly so a value just below 90% cannot serialize as
            # 0.900 while its state truthfully reports an attainment failure.
            "output_attainment": output_attainment,
            "unique_attainment": unique_attainment,
            # These ratios have exact code-owned thresholds or diagnose
            # scheduler boundary behavior. Preserve full precision so a value
            # just above 5% can never render as the passing boundary.
            "processing_deadline_miss_ratio": miss_ratio,
            "output_schedule_late_ratio": late_ratio,
            "stage_p50_ms": p50,
            "stage_p95_ms": p95,
            "dominant_stage": dominant_stage,
        }

    def _window_metrics_locked(self, now_ns: int) -> dict[str, object]:
        return self._window_metrics_for_locked(
            self._samples,
            self._epoch_started_ns,
            now_ns,
        )

    @staticmethod
    def _effective_window_start_ns(
        samples: deque[_Sample],
        epoch_started_ns: int,
        now_ns: int,
    ) -> int:
        """Return the complete retained horizon used by rates and gates.

        At the hard event cap, the ring can cover less than the nominal five
        seconds (notably at the supported 240 Hz maximum, where processing and
        output are independent events).  Dividing retained counts by time that
        the ring no longer represents would manufacture a rate collapse.  In
        that case the oldest retained timestamp becomes the exclusive window
        boundary for both public metrics and raw qualification.
        """

        nominal_start_ns = max(
            epoch_started_ns,
            now_ns - RUNTIME_PERFORMANCE_WINDOW_NS,
        )
        if (
            len(samples) >= RUNTIME_PERFORMANCE_SAMPLE_LIMIT
            and samples
            and samples[0].at_ns > nominal_start_ns
        ):
            return samples[0].at_ns
        return nominal_start_ns

    def _raw_reason_for_locked(
        self,
        samples_source: deque[_Sample],
        epoch_started_ns: int,
        now_ns: int,
    ) -> PerformanceReason:
        """Evaluate gates without public rounding changing a threshold."""

        window_start_ns = self._effective_window_start_ns(
            samples_source,
            epoch_started_ns,
            now_ns,
        )
        duration_s = max(0, now_ns - window_start_ns) / 1_000_000_000.0
        samples = tuple(
            sample for sample in samples_source if sample.at_ns > window_start_ns
        )
        output_count = sum(sample.output_sent for sample in samples)
        unique_count = sum(sample.sent_unique_base for sample in samples)
        processing_count = sum(sample.processing_completed for sample in samples)
        deadline_misses = sum(sample.processing_deadline_missed for sample in samples)
        output_fps = output_count / duration_s if duration_s > 0.0 else 0.0
        unique_fps = unique_count / duration_s if duration_s > 0.0 else 0.0
        output_bad = output_fps / self._target_fps < RUNTIME_PERFORMANCE_MIN_ATTAINMENT
        unique_bad = unique_fps / self._target_fps < RUNTIME_PERFORMANCE_MIN_ATTAINMENT
        miss_ratio = deadline_misses / processing_count if processing_count else 0.0
        deadline_bad = miss_ratio > RUNTIME_PERFORMANCE_MAX_DEADLINE_MISS_RATIO
        failed_gates = int(output_bad) + int(unique_bad) + int(deadline_bad)
        if failed_gates > 1 and not (output_bad and unique_bad and not deadline_bad):
            return "multiple-performance-gates"
        if output_bad and unique_bad:
            return "output-and-unique-attainment"
        if output_bad:
            return "output-attainment"
        if unique_bad:
            return "unique-attainment"
        if deadline_bad:
            return "processing-deadline-miss"
        return "none"

    def _raw_reason_locked(self, now_ns: int) -> PerformanceReason:
        return self._raw_reason_for_locked(
            self._samples,
            self._epoch_started_ns,
            now_ns,
        )

    def _evaluate_locked(self, now_ns: int) -> None:
        if self._failed_reason is not None:
            self._state = "failed"
            self._reason = self._failed_reason
            return
        if self._ready_at_ns is None:
            self._state = "warming"
            self._reason = "none"
            return
        if now_ns - self._epoch_started_ns < RUNTIME_PERFORMANCE_WARMUP_NS:
            self._state = "warming"
            self._reason = "none"
            return

        raw_reason = self._raw_reason_locked(now_ns)
        if raw_reason != "none":
            self._healthy_since_ns = None
            if self._unhealthy_since_ns is None:
                self._unhealthy_since_ns = now_ns
            if now_ns - self._unhealthy_since_ns >= RUNTIME_PERFORMANCE_DEGRADE_NS:
                self._state = "degraded"
                self._reason = raw_reason
            return

        self._unhealthy_since_ns = None
        if self._state == "degraded":
            if self._healthy_since_ns is None:
                self._healthy_since_ns = now_ns
            if now_ns - self._healthy_since_ns < RUNTIME_PERFORMANCE_RECOVER_NS:
                return
        self._healthy_since_ns = None
        self._state = "healthy"
        self._reason = "none"

    def _evaluate_closed_locked(self, closed: _ClosedEpoch, now_ns: int) -> None:
        """Re-evaluate a mutable closed bucket after a late accepted send."""

        if closed.failed_reason is not None:
            closed.state = "failed"
            closed.reason = closed.failed_reason
            return
        if not closed.steady_state or (
            now_ns - closed.started_ns < RUNTIME_PERFORMANCE_WARMUP_NS
        ):
            closed.state = "warming"
            closed.reason = "none"
            return

        raw_reason = self._raw_reason_for_locked(
            closed.samples,
            closed.started_ns,
            now_ns,
        )
        if raw_reason != "none":
            closed.healthy_since_ns = None
            if closed.unhealthy_since_ns is None:
                closed.unhealthy_since_ns = now_ns
            if now_ns - closed.unhealthy_since_ns >= RUNTIME_PERFORMANCE_DEGRADE_NS:
                closed.state = "degraded"
                closed.reason = raw_reason
            return

        closed.unhealthy_since_ns = None
        if closed.state == "degraded":
            if closed.healthy_since_ns is None:
                closed.healthy_since_ns = now_ns
            if now_ns - closed.healthy_since_ns < RUNTIME_PERFORMANCE_RECOVER_NS:
                return
        closed.healthy_since_ns = None
        closed.state = "healthy"
        closed.reason = "none"

    def _epoch_summary_locked(self, now_ns: int) -> dict[str, object]:
        metrics = self._window_metrics_locked(now_ns)
        return {
            "key": self._epoch_key.as_dict(),
            "state": self._state,
            "reason": self._reason,
            "duration_s": _rounded(
                max(0, now_ns - self._epoch_started_ns) / 1_000_000_000.0
            ),
            **self._current.as_dict(),
            **metrics,
        }

    def _summary_for_closed_epoch_locked(
        self,
        closed: _ClosedEpoch,
    ) -> dict[str, object]:
        metrics = self._window_metrics_for_locked(
            closed.samples,
            closed.started_ns,
            closed.ended_ns,
        )
        return {
            "key": closed.key.as_dict(),
            "state": closed.state,
            "reason": closed.reason,
            "duration_s": _rounded(
                max(0, closed.ended_ns - closed.started_ns) / 1_000_000_000.0
            ),
            **closed.counters.as_dict(),
            **metrics,
        }

    def _closed_epoch_summary_locked(self) -> dict[str, object] | None:
        key = self._last_closed_epoch_key
        if key is None:
            return None
        closed = self._closed_epochs.get(key)
        if closed is None:  # pragma: no cover - retention invariant
            return None
        return self._summary_for_closed_epoch_locked(closed)

    def _mitigation_locked(self) -> dict[str, object] | None:
        if self._state != "degraded":
            return None
        if self._color_correction_mode == "auto" and self._light_wrap > 0.0:
            kind = "disable-color-and-light-wrap"
            patch: dict[str, object] = {
                "compositing": {
                    "color_correction": {"mode": "off"},
                    "light_wrap": 0.0,
                }
            }
        elif self._color_correction_mode == "auto":
            kind = "disable-color-correction"
            patch = {"compositing": {"color_correction": {"mode": "off"}}}
        elif self._light_wrap > 0.0:
            kind = "disable-light-wrap"
            patch = {"compositing": {"light_wrap": 0.0}}
        else:
            kind = "review-backend-or-diagnostic-target"
            patch = {}
        return {
            "config_version": self._epoch_key.config_version,
            "kind": kind,
            "patch": patch,
        }

    def snapshot(self, *, at_ns: int | None = None) -> dict[str, object]:
        """Return the strict version-1 JSON-compatible public snapshot."""

        with self._lock:
            now_ns = self._resolve_time_locked(at_ns)
            self._evaluate_locked(now_ns)
            metrics = self._window_metrics_locked(now_ns)
            current_epoch = self._epoch_summary_locked(now_ns)
            public = {
                "schema_version": RUNTIME_PERFORMANCE_SCHEMA_VERSION,
                "state": self._state,
                "reason": self._reason,
                "target_fps": _rounded(self._target_fps),
                **metrics,
                "output_healthy": (
                    cast(float, metrics["output_attainment"])
                    >= RUNTIME_PERFORMANCE_MIN_ATTAINMENT
                ),
                "unique_healthy": (
                    cast(float, metrics["unique_attainment"])
                    >= RUNTIME_PERFORMANCE_MIN_ATTAINMENT
                ),
                "current_epoch": current_epoch,
                "last_closed_epoch": self._closed_epoch_summary_locked(),
                "startup": self._startup.as_dict(),
                "publisher": self._publisher.as_dict(),
                "recommended_mitigation": self._mitigation_locked(),
            }
            return validate_runtime_performance_status(public)


__all__ = [
    "RUNTIME_PERFORMANCE_DEGRADE_NS",
    "RUNTIME_PERFORMANCE_MAX_DEADLINE_MISS_RATIO",
    "RUNTIME_PERFORMANCE_MIN_ATTAINMENT",
    "RUNTIME_PERFORMANCE_RECOVER_NS",
    "RUNTIME_PERFORMANCE_SAMPLE_LIMIT",
    "RUNTIME_PERFORMANCE_SCHEMA_VERSION",
    "RUNTIME_PERFORMANCE_WARMUP_NS",
    "RUNTIME_PERFORMANCE_WINDOW_NS",
    "RUNTIME_STAGE_NAMES",
    "PerformanceEpochKey",
    "PublisherTelemetry",
    "RuntimePerformanceTracker",
    "empty_runtime_performance_status",
    "validate_runtime_performance_status",
]
