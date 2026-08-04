"""Bounded, pixel-free cadence accounting for successful output sends.

The tracker records one scalar provenance event only after an output submission
has succeeded.  It deliberately keeps camera/base/segmentation work distinct
from safe-base reuse and from byte-identical final output.  In particular, a
new captured frame may update the base while producing pixels identical to the
previous final output.

All timestamps use the process monotonic clock.  Public snapshots contain only
relative durations and rates; absolute monotonic timestamps remain private.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

CADENCE_WINDOW_NS = 2_000_000_000
CADENCE_EVENT_LIMIT = 1024
CADENCE_SAMPLE_LIMIT = 256

# Relative status remains useful across long stalls without exposing an
# unbounded integer.  The cap is telemetry-only and never alters timeline
# decisions or lifetime counters.
CADENCE_MAX_REPORTED_MS = 86_400_000.0


@dataclass(frozen=True)
class CadenceSnapshot:
    """Immutable scalar cadence state suitable for status publication."""

    capture_sequence: int
    capture_sequence_gap_count: int
    capture_missing_input_count: int
    unique_capture_count: int
    unique_capture_fps: float
    segmentation_update_count: int
    segmentation_update_fps: float
    base_composite_update_count: int
    base_composite_update_fps: float
    base_composite_reuse_count: int
    base_composite_reuse_fps: float
    base_composite_reuse_ratio: float
    exact_final_output_repeat_count: int
    exact_final_output_repeat_fps: float
    exact_final_output_repeat_ratio: float
    output_send_count: int
    output_send_fps: float
    last_unique_frame_age_ms: float | None
    capture_timestamp_delta_p50_ms: float | None
    capture_timestamp_delta_p95_ms: float | None
    output_send_delta_p50_ms: float | None
    output_send_delta_p95_ms: float | None
    output_send_jitter_p50_ms: float | None
    output_send_jitter_p95_ms: float | None
    base_composite_delta_p50_ms: float | None
    base_composite_delta_p95_ms: float | None
    processing_deadline_misses: int
    serialized_new_frame_deadline_misses: int
    output_sink_pacing_events: int
    output_sink_recovery_events: int
    application_pacing_events: int
    output_schedule_late_events: int
    cadence_mismatch_active: bool

    def as_dict(self) -> dict[str, object]:
        """Return a new flat mapping without exposing retained events."""

        return asdict(self)


@dataclass(frozen=True)
class _SuccessfulSend:
    sent_at_ns: int
    capture_sequence: int | None
    captured_at_ns: int | None
    base_ready_at_ns: int | None
    base_updated: bool
    segmentation_updated: bool
    exact_final_repeat: bool


def _non_negative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _bounded_ms(delta_ns: int) -> float:
    return min(max(0.0, delta_ns / 1_000_000.0), CADENCE_MAX_REPORTED_MS)


def _recent_deltas_ms(timestamps_ns: Sequence[int]) -> list[float]:
    values = [
        _bounded_ms(current - previous)
        for previous, current in zip(timestamps_ns, timestamps_ns[1:])
        if current >= previous
    ]
    return values[-CADENCE_SAMPLE_LIMIT:]


def _rate(count: int, first_ns: int, last_ns: int) -> float:
    span_ns = last_ns - first_ns
    if count <= 0 or span_ns <= 0:
        return 0.0
    return count * 1_000_000_000.0 / span_ns


class CadenceTracker:
    """Track lifetime provenance and a bounded two-second cadence window.

    ``record_send`` must be called only after the sink has accepted the output.
    A base update requires the matching capture sequence/timestamp.  A
    no-unread-frame send omits capture identity and is counted as safe-base
    reuse.  ``exact_final_repeat`` is independent byte-equality provenance:
    both a new pixel-identical capture and a synthesized reuse can be exact.
    """

    def __init__(
        self,
        target_output_fps: float,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if (
            isinstance(target_output_fps, bool)
            or not isinstance(target_output_fps, (int, float))
            or not math.isfinite(float(target_output_fps))
            or float(target_output_fps) <= 0.0
        ):
            raise ValueError("target output FPS must be finite and positive")
        if not callable(clock_ns):
            raise TypeError("cadence clock must be callable")

        target_interval_ns = 1_000_000_000.0 / float(target_output_fps)
        if not math.isfinite(target_interval_ns):
            raise ValueError("target output FPS produces an invalid interval")
        self._target_interval_ns = target_interval_ns
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._events: deque[_SuccessfulSend] = deque(maxlen=CADENCE_EVENT_LIMIT)

        self._capture_sequence = 0
        self._capture_sequence_gap_count = 0
        self._capture_missing_input_count = 0
        self._unique_capture_count = 0
        self._segmentation_update_count = 0
        self._base_composite_update_count = 0
        self._base_composite_reuse_count = 0
        self._exact_final_output_repeat_count = 0
        self._output_send_count = 0
        self._last_captured_at_ns: int | None = None
        self._last_base_ready_at_ns: int | None = None
        self._last_sent_at_ns: int | None = None

        self._processing_deadline_misses = 0
        self._serialized_new_frame_deadline_misses = 0
        self._output_sink_pacing_events = 0
        self._output_sink_recovery_events = 0
        self._application_pacing_events = 0
        self._output_schedule_late_events = 0

    @property
    def retained_event_count(self) -> int:
        """Return the current private ring size for boundedness tests."""

        with self._lock:
            return len(self._events)

    def _prune_locked(self, now_ns: int) -> None:
        cutoff_ns = now_ns - CADENCE_WINDOW_NS
        while self._events and self._events[0].sent_at_ns < cutoff_ns:
            self._events.popleft()

    def record_send(
        self,
        *,
        sent_at_ns: int,
        base_updated: bool,
        segmentation_updated: bool,
        exact_final_repeat: bool,
        capture_sequence: int | None = None,
        captured_at_ns: int | None = None,
        base_ready_at_ns: int | None = None,
        processing_deadline_missed: bool = False,
        serialized_new_frame_deadline_missed: bool = False,
        output_sink_pacing_events: int = 0,
        output_sink_recovery_events: int = 0,
        application_pacing_events: int = 0,
        output_schedule_late: bool = False,
    ) -> None:
        """Atomically record one successfully submitted final output."""

        sent_at_ns = _non_negative_int(sent_at_ns, "send timestamp")
        for value, label in (
            (base_updated, "base-updated flag"),
            (segmentation_updated, "segmentation-updated flag"),
            (exact_final_repeat, "exact-final-repeat flag"),
            (processing_deadline_missed, "processing-deadline flag"),
            (
                serialized_new_frame_deadline_missed,
                "serialized-new-frame-deadline flag",
            ),
            (output_schedule_late, "output-schedule-late flag"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{label} must be boolean")
        pacing_events = _non_negative_int(
            output_sink_pacing_events,
            "output sink pacing events",
        )
        recovery_events = _non_negative_int(
            output_sink_recovery_events,
            "output sink recovery events",
        )
        app_pacing_events = _non_negative_int(
            application_pacing_events,
            "application pacing events",
        )

        if segmentation_updated and not base_updated:
            raise ValueError("segmentation update requires a base update")
        if (
            processing_deadline_missed or serialized_new_frame_deadline_missed
        ) and not base_updated:
            raise ValueError("new-frame deadline miss requires a base update")
        has_sequence = capture_sequence is not None
        has_timestamp = captured_at_ns is not None
        has_base_ready = base_ready_at_ns is not None
        if (
            has_sequence != has_timestamp
            or has_sequence != has_base_ready
            or has_sequence != base_updated
        ):
            raise ValueError(
                "base update requires matching capture, timestamp, and ready time"
            )
        if capture_sequence is not None:
            capture_sequence = _non_negative_int(
                capture_sequence,
                "capture sequence",
            )
        if captured_at_ns is not None:
            captured_at_ns = _non_negative_int(
                captured_at_ns,
                "capture timestamp",
            )
        if base_ready_at_ns is not None:
            base_ready_at_ns = _non_negative_int(
                base_ready_at_ns,
                "base-ready timestamp",
            )
            assert captured_at_ns is not None
            if captured_at_ns > base_ready_at_ns or base_ready_at_ns > sent_at_ns:
                raise ValueError(
                    "capture, base-ready, and send timestamps must be ordered"
                )

        event = _SuccessfulSend(
            sent_at_ns=sent_at_ns,
            capture_sequence=capture_sequence,
            captured_at_ns=captured_at_ns,
            base_ready_at_ns=base_ready_at_ns,
            base_updated=base_updated,
            segmentation_updated=segmentation_updated,
            exact_final_repeat=exact_final_repeat,
        )
        with self._lock:
            if self._last_sent_at_ns is not None and sent_at_ns < self._last_sent_at_ns:
                raise ValueError("successful send timestamps must not decrease")
            if self._output_send_count == 0:
                if not base_updated:
                    raise ValueError("the first successful send must establish a base")
                if exact_final_repeat:
                    raise ValueError(
                        "the first successful send has no comparable final output"
                    )
            if capture_sequence is not None:
                previous_sequence = (
                    None if self._unique_capture_count == 0 else self._capture_sequence
                )
                if (
                    previous_sequence is not None
                    and capture_sequence <= previous_sequence
                ):
                    raise ValueError(
                        "capture sequence must increase for every base update"
                    )
                if (
                    self._last_captured_at_ns is not None
                    and captured_at_ns is not None
                    and captured_at_ns < self._last_captured_at_ns
                ):
                    raise ValueError("capture timestamps must not decrease")
                if (
                    self._last_base_ready_at_ns is not None
                    and base_ready_at_ns is not None
                    and base_ready_at_ns < self._last_base_ready_at_ns
                ):
                    raise ValueError("base-ready timestamps must not decrease")
                if previous_sequence is not None:
                    missing = capture_sequence - previous_sequence - 1
                    if missing:
                        self._capture_sequence_gap_count += 1
                        self._capture_missing_input_count += missing
                self._capture_sequence = capture_sequence
                self._last_captured_at_ns = captured_at_ns
                self._last_base_ready_at_ns = base_ready_at_ns
                self._unique_capture_count += 1
                self._base_composite_update_count += 1
                if segmentation_updated:
                    self._segmentation_update_count += 1
            else:
                self._base_composite_reuse_count += 1
            if exact_final_repeat:
                self._exact_final_output_repeat_count += 1
            if processing_deadline_missed:
                self._processing_deadline_misses += 1
            if serialized_new_frame_deadline_missed:
                self._serialized_new_frame_deadline_misses += 1
            self._output_sink_pacing_events += pacing_events
            self._output_sink_recovery_events += recovery_events
            self._application_pacing_events += app_pacing_events
            if output_schedule_late:
                self._output_schedule_late_events += 1

            self._events.append(event)
            self._output_send_count += 1
            self._last_sent_at_ns = sent_at_ns
            self._prune_locked(sent_at_ns)

    def snapshot(self, *, now_ns: int | None = None) -> CadenceSnapshot:
        """Return lifetime counters plus a bounded rolling cadence view."""

        if now_ns is None:
            now_ns = self._clock_ns()
        now_ns = _non_negative_int(now_ns, "snapshot timestamp")

        with self._lock:
            self._prune_locked(now_ns)
            events = tuple(self._events)

            if len(events) >= 2 and events[-1].sent_at_ns > events[0].sent_at_ns:
                first_send_ns = events[0].sent_at_ns
                last_send_ns = events[-1].sent_at_ns
                send_transition_count = len(events) - 1
                output_send_fps = _rate(
                    send_transition_count,
                    first_send_ns,
                    last_send_ns,
                )
                base_reuse_fps = _rate(
                    sum(not event.base_updated for event in events[1:]),
                    first_send_ns,
                    last_send_ns,
                )
                base_update_fps = _rate(
                    sum(event.base_updated for event in events[1:]),
                    first_send_ns,
                    last_send_ns,
                )
                segmentation_update_fps = _rate(
                    sum(event.segmentation_updated for event in events[1:]),
                    first_send_ns,
                    last_send_ns,
                )
                exact_repeat_fps = _rate(
                    sum(event.exact_final_repeat for event in events[1:]),
                    first_send_ns,
                    last_send_ns,
                )
            else:
                output_send_fps = 0.0
                base_update_fps = 0.0
                base_reuse_fps = 0.0
                segmentation_update_fps = 0.0
                exact_repeat_fps = 0.0

            capture_timestamps = [
                event.captured_at_ns
                for event in events
                if event.captured_at_ns is not None
            ]
            if (
                len(capture_timestamps) >= 2
                and capture_timestamps[-1] > capture_timestamps[0]
            ):
                unique_capture_fps = _rate(
                    len(capture_timestamps) - 1,
                    capture_timestamps[0],
                    capture_timestamps[-1],
                )
            else:
                unique_capture_fps = 0.0

            base_timestamps = [
                event.base_ready_at_ns
                for event in events
                if event.base_ready_at_ns is not None
            ]
            send_timestamps = [event.sent_at_ns for event in events]
            capture_deltas = _recent_deltas_ms(capture_timestamps)
            base_deltas = _recent_deltas_ms(base_timestamps)
            send_deltas = _recent_deltas_ms(send_timestamps)
            target_interval_ms = self._target_interval_ns / 1_000_000.0
            send_jitter = [
                min(
                    abs(delta_ms - target_interval_ms),
                    CADENCE_MAX_REPORTED_MS,
                )
                for delta_ms in send_deltas
            ]

            last_unique_age_ms = (
                None
                if self._last_base_ready_at_ns is None
                else _bounded_ms(now_ns - self._last_base_ready_at_ns)
            )
            reuse_ratio = (
                self._base_composite_reuse_count / self._output_send_count
                if self._output_send_count
                else 0.0
            )
            comparable_transitions = max(1, self._output_send_count - 1)
            exact_repeat_ratio = (
                self._exact_final_output_repeat_count / comparable_transitions
                if self._output_send_count
                else 0.0
            )
            cadence_mismatch_active = (
                len(events) >= 3
                and output_send_fps > 0.0
                and base_update_fps < output_send_fps * 0.9
            )

            return CadenceSnapshot(
                capture_sequence=self._capture_sequence,
                capture_sequence_gap_count=self._capture_sequence_gap_count,
                capture_missing_input_count=self._capture_missing_input_count,
                unique_capture_count=self._unique_capture_count,
                unique_capture_fps=unique_capture_fps,
                segmentation_update_count=self._segmentation_update_count,
                segmentation_update_fps=segmentation_update_fps,
                base_composite_update_count=self._base_composite_update_count,
                base_composite_update_fps=base_update_fps,
                base_composite_reuse_count=self._base_composite_reuse_count,
                base_composite_reuse_fps=base_reuse_fps,
                base_composite_reuse_ratio=reuse_ratio,
                exact_final_output_repeat_count=(self._exact_final_output_repeat_count),
                exact_final_output_repeat_fps=exact_repeat_fps,
                exact_final_output_repeat_ratio=exact_repeat_ratio,
                output_send_count=self._output_send_count,
                output_send_fps=output_send_fps,
                last_unique_frame_age_ms=last_unique_age_ms,
                capture_timestamp_delta_p50_ms=_nearest_rank(
                    capture_deltas,
                    0.50,
                ),
                capture_timestamp_delta_p95_ms=_nearest_rank(
                    capture_deltas,
                    0.95,
                ),
                output_send_delta_p50_ms=_nearest_rank(send_deltas, 0.50),
                output_send_delta_p95_ms=_nearest_rank(send_deltas, 0.95),
                output_send_jitter_p50_ms=_nearest_rank(send_jitter, 0.50),
                output_send_jitter_p95_ms=_nearest_rank(send_jitter, 0.95),
                base_composite_delta_p50_ms=_nearest_rank(base_deltas, 0.50),
                base_composite_delta_p95_ms=_nearest_rank(base_deltas, 0.95),
                processing_deadline_misses=self._processing_deadline_misses,
                serialized_new_frame_deadline_misses=(
                    self._serialized_new_frame_deadline_misses
                ),
                output_sink_pacing_events=self._output_sink_pacing_events,
                output_sink_recovery_events=self._output_sink_recovery_events,
                application_pacing_events=self._application_pacing_events,
                output_schedule_late_events=self._output_schedule_late_events,
                cadence_mismatch_active=cadence_mismatch_active,
            )
