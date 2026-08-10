"""Focused tests for the bounded runtime performance health tracker."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from custback.runtime_performance import (
    RUNTIME_PERFORMANCE_SAMPLE_LIMIT,
    RUNTIME_STAGE_NAMES,
    PerformanceEpochKey,
    PublisherTelemetry,
    RuntimePerformanceTracker,
)


class _Clock:
    def __init__(self) -> None:
        self.now_ns = 0

    def __call__(self) -> int:
        return self.now_ns

    def set_seconds(self, value: float) -> int:
        self.now_ns = round(value * 1_000_000_000)
        return self.now_ns


def _key(version: int = 1) -> PerformanceEpochKey:
    return PerformanceEpochKey(
        config_version=version,
        capture_generation=2,
        segmentation_generation=3,
        backdrop_generation=4,
    )


def _healthy_frame(
    tracker: RuntimePerformanceTracker,
    clock: _Clock,
    at_s: float,
    *,
    deadline_missed: bool = False,
    schedule_late: bool = False,
) -> None:
    at_ns = clock.set_seconds(at_s)
    tracker.record_processing(
        at_ns=at_ns,
        deadline_missed=deadline_missed,
        stages_ms={
            "segmentation.total": 12.0,
            "compositor.total": 5.0,
            "pipeline.processing_only": 20.0,
        },
    )
    tracker.record_output(
        at_ns=at_ns,
        unique_base=True,
        schedule_late=schedule_late,
        submission_ms=0.25,
    )


def test_snapshot_is_strict_path_free_and_separates_startup() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(
        30.0,
        _key(),
        color_correction_mode="auto",
        light_wrap=0.25,
        clock_ns=clock,
    )
    tracker.record_processing(
        at_ns=clock.set_seconds(0.25),
        deadline_missed=True,
        stages_ms={"segmentation.total": 17.0},
    )
    tracker.record_output(
        at_ns=clock.set_seconds(0.3),
        unique_base=True,
        schedule_late=True,
    )
    tracker.mark_ready(at_ns=clock.set_seconds(1.0))

    snapshot = tracker.snapshot(at_ns=clock.set_seconds(2.0))
    assert set(snapshot) == {
        "schema_version",
        "state",
        "reason",
        "target_fps",
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
        "output_healthy",
        "unique_healthy",
        "current_epoch",
        "last_closed_epoch",
        "startup",
        "publisher",
        "recommended_mitigation",
    }
    assert snapshot["schema_version"] == 1
    assert snapshot["state"] == "warming"
    assert snapshot["startup"] == {
        "processing_completed_count": 1,
        "output_send_count": 1,
        "sent_unique_base_count": 1,
        "processing_deadline_miss_count": 1,
        "output_schedule_late_count": 1,
    }
    assert snapshot["window_sample_count"] == 0
    stage_p50 = cast(dict[str, float | None], snapshot["stage_p50_ms"])
    stage_p95 = cast(dict[str, float | None], snapshot["stage_p95_ms"])
    assert set(stage_p50) == set(RUNTIME_STAGE_NAMES)
    assert set(stage_p95) == set(RUNTIME_STAGE_NAMES)
    assert snapshot["last_closed_epoch"] is None
    assert snapshot["recommended_mitigation"] is None

    encoded = json.dumps(snapshot, sort_keys=True)
    assert "/" not in encoded
    assert "\\" not in encoded
    assert "://" not in encoded


def test_degradation_hysteresis_and_sanitized_mitigation() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(
        30,
        _key(11),
        color_correction_mode="auto",
        light_wrap=1.0,
        clock_ns=clock,
    )
    tracker.mark_ready(at_ns=clock.set_seconds(0.0))

    assert tracker.snapshot(at_ns=clock.set_seconds(2.999))["state"] == "warming"
    # The first eligible bad observation starts, but does not bypass, the
    # continuous three-second degradation interval.
    assert tracker.snapshot(at_ns=clock.set_seconds(3.0))["state"] == "warming"
    assert tracker.snapshot(at_ns=clock.set_seconds(5.999))["state"] == "warming"
    degraded = tracker.snapshot(at_ns=clock.set_seconds(6.0))
    assert degraded["state"] == "degraded"
    assert degraded["reason"] == "output-and-unique-attainment"
    assert degraded["output_healthy"] is False
    assert degraded["unique_healthy"] is False
    assert degraded["recommended_mitigation"] == {
        "config_version": 11,
        "kind": "disable-color-and-light-wrap",
        "patch": {
            "compositing": {
                "color_correction": {"mode": "off"},
                "light_wrap": 0.0,
            }
        },
    }

    # Ten seconds of sustained 30 fps output and unique processing fills the
    # rolling window, then satisfies the separate five-second recovery hold.
    for index in range(1, 301):
        _healthy_frame(tracker, clock, 6.0 + index / 30.0)
    recovered = tracker.snapshot(at_ns=clock.set_seconds(16.0))
    assert recovered["state"] == "healthy"
    assert recovered["reason"] == "none"
    assert recovered["output_send_fps"] == pytest.approx(30.0, abs=0.01)
    assert recovered["sent_unique_base_fps"] == pytest.approx(30.0, abs=0.01)
    assert recovered["recommended_mitigation"] is None


def test_output_and_unique_health_are_independent() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(30, _key(), clock_ns=clock)
    tracker.mark_ready(at_ns=clock.set_seconds(0.0))

    # Publisher repeats maintain 30 fps, while only every fourth send adopts a
    # newly processed base.  Continue beyond the degradation hold.
    for index in range(1, 241):
        at_s = index / 30.0
        at_ns = clock.set_seconds(at_s)
        unique = index % 4 == 0
        if unique:
            tracker.record_processing(
                at_ns=at_ns,
                deadline_missed=True,
                stages_ms={"compositor.total": 91.6},
            )
        tracker.record_output(
            at_ns=at_ns,
            unique_base=unique,
            schedule_late=False,
        )

    snapshot = tracker.snapshot(at_ns=clock.set_seconds(8.0))
    assert snapshot["state"] == "degraded"
    assert snapshot["reason"] == "multiple-performance-gates"
    assert snapshot["output_healthy"] is True
    assert snapshot["unique_healthy"] is False
    assert snapshot["output_attainment"] == pytest.approx(1.0, abs=0.01)
    assert snapshot["unique_attainment"] == pytest.approx(0.25, abs=0.01)


def test_exact_qualification_thresholds_remain_healthy() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(20, _key(), clock_ns=clock)
    tracker.mark_ready(at_ns=0)
    events: list[tuple[float, str, bool]] = []
    for index in range(1, 91):
        events.append((index * 5.0 / 90.0, "output", False))
    for index in range(1, 101):
        # Exactly 5%, not at least 5%, is the allowed deadline boundary.
        events.append((index * 5.0 / 100.0, "processing", index <= 5))
    for at_s, kind, missed in sorted(events):
        at_ns = clock.set_seconds(at_s)
        if kind == "processing":
            tracker.record_processing(at_ns=at_ns, deadline_missed=missed)
        else:
            tracker.record_output(
                at_ns=at_ns,
                unique_base=True,
                schedule_late=False,
            )

    snapshot = tracker.snapshot(at_ns=clock.set_seconds(5.0))
    assert snapshot["state"] == "healthy"
    assert snapshot["output_attainment"] == 0.9
    assert snapshot["unique_attainment"] == 0.9
    assert snapshot["processing_deadline_miss_ratio"] == 0.05
    assert snapshot["output_healthy"] is True
    assert snapshot["unique_healthy"] is True


def test_attainment_precision_cannot_publish_false_healthy_flags() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(30.01, _key(), clock_ns=clock)
    tracker.mark_ready(at_ns=0)
    for index in range(1, 163):
        at_ns = clock.set_seconds(index / 27.0)
        tracker.record_output(
            at_ns=at_ns,
            unique_base=True,
            schedule_late=False,
        )
        if index == 81:
            tracker.snapshot(at_ns=clock.set_seconds(3.0))

    snapshot = tracker.snapshot(at_ns=clock.set_seconds(6.0))
    assert snapshot["state"] == "degraded"
    assert snapshot["reason"] == "output-and-unique-attainment"
    assert snapshot["output_send_fps"] == 27.0
    assert snapshot["sent_unique_base_fps"] == 27.0
    output_attainment = cast(float, snapshot["output_attainment"])
    unique_attainment = cast(float, snapshot["unique_attainment"])
    assert 0.899 < output_attainment < 0.9
    assert 0.899 < unique_attainment < 0.9
    assert snapshot["output_healthy"] is False
    assert snapshot["unique_healthy"] is False


def test_epoch_rotation_retains_only_final_scalar_summary() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(30, _key(7), clock_ns=clock)
    tracker.mark_ready(at_ns=clock.set_seconds(0.0))
    _healthy_frame(tracker, clock, 0.5)
    tracker.update_publisher(
        PublisherTelemetry(
            mode="sink-paced",
            state="running",
            handoff_overwrite_count=3,
            missed_slot_count=2,
            slate_send_count=1,
            pending_depth=1,
            output_base_config_version=7,
        ),
        at_ns=clock.set_seconds(0.75),
    )

    changed = tracker.bind_epoch(
        PerformanceEpochKey(8, 2, 3, 5),
        color_correction_mode="off",
        light_wrap=0.0,
        at_ns=clock.set_seconds(1.0),
    )
    assert changed is True
    snapshot = tracker.snapshot(at_ns=clock.set_seconds(1.25))
    current = cast(dict[str, Any], snapshot["current_epoch"])
    assert current["key"] == {
        "config_version": 8,
        "capture_generation": 2,
        "segmentation_generation": 3,
        "backdrop_generation": 5,
    }
    assert current["processing_completed_count"] == 0
    closed = cast(dict[str, Any], snapshot["last_closed_epoch"])
    closed_key = cast(dict[str, int], closed["key"])
    assert closed_key["config_version"] == 7
    assert closed["processing_completed_count"] == 1
    assert closed["output_send_count"] == 1
    publisher = cast(dict[str, Any], snapshot["publisher"])
    assert publisher["handoff_overwrite_count"] == 3
    assert publisher["missed_slot_count"] == 2
    assert publisher["slate_send_count"] == 1
    closed_p95 = cast(dict[str, float | None], closed["stage_p95_ms"])
    assert closed_p95["segmentation.total"] == 12.0

    assert (
        tracker.bind_epoch(
            PerformanceEpochKey(8, 2, 3, 5),
            color_correction_mode="off",
            light_wrap=0.0,
            at_ns=clock.set_seconds(1.5),
        )
        is False
    )
    assert tracker.snapshot(at_ns=clock.set_seconds(1.5))["last_closed_epoch"] == closed


def test_output_record_is_atomically_bound_to_expected_epoch() -> None:
    clock = _Clock()
    old_key = _key(7)
    tracker = RuntimePerformanceTracker(30, old_key, clock_ns=clock)
    tracker.mark_ready(at_ns=clock.set_seconds(0.0))
    assert tracker.retain_epoch(old_key)

    assert tracker.record_output(
        at_ns=clock.set_seconds(0.25),
        unique_base=True,
        schedule_late=False,
        expected_epoch=old_key,
    )
    new_key = PerformanceEpochKey(8, 2, 3, 5)
    assert tracker.bind_epoch(
        new_key,
        color_correction_mode="off",
        light_wrap=0.0,
        at_ns=clock.set_seconds(0.5),
    )

    assert tracker.record_output(
        at_ns=clock.set_seconds(0.75),
        unique_base=False,
        schedule_late=True,
        submission_ms=0.75,
        expected_epoch=old_key,
    )
    after_stale = tracker.snapshot(at_ns=clock.set_seconds(1.0))
    current = cast(dict[str, Any], after_stale["current_epoch"])
    closed = cast(dict[str, Any], after_stale["last_closed_epoch"])
    assert current["output_send_count"] == 0
    assert current["output_schedule_late_count"] == 0
    assert current["stage_p95_ms"]["output.submission"] is None
    assert closed["output_send_count"] == 2
    assert closed["output_schedule_late_count"] == 1
    assert closed["output_schedule_late_ratio"] == 0.5
    assert closed["stage_p95_ms"]["output.submission"] == 0.75
    finalized = tracker.release_epoch(old_key)
    assert finalized is not None
    assert finalized["output_send_count"] == 2

    assert tracker.record_output(
        at_ns=clock.set_seconds(1.25),
        unique_base=True,
        schedule_late=False,
        expected_epoch=new_key,
    )
    current = cast(
        dict[str, Any],
        tracker.snapshot(at_ns=clock.set_seconds(1.5))["current_epoch"],
    )
    assert current["output_send_count"] == 1


def test_referenced_a_survives_a_to_b_to_c_until_late_acceptance() -> None:
    clock = _Clock()
    key_a = _key(1)
    key_b = _key(2)
    key_c = _key(3)
    tracker = RuntimePerformanceTracker(30, key_a, clock_ns=clock)
    tracker.mark_ready(at_ns=clock.set_seconds(0.0))

    assert tracker.retain_epoch(key_a)
    assert tracker.bind_epoch(
        key_b,
        color_correction_mode="off",
        light_wrap=0.0,
        at_ns=clock.set_seconds(0.5),
    )
    assert tracker.retain_epoch(key_b)
    assert tracker.bind_epoch(
        key_c,
        color_correction_mode="off",
        light_wrap=0.0,
        at_ns=clock.set_seconds(1.0),
    )
    assert tracker.retained_closed_epoch_count == 2
    assert tracker.take_finalized_epoch_summaries() == ()

    assert tracker.record_output(
        unique_base=True,
        schedule_late=True,
        submission_ms=1.25,
        expected_epoch=key_a,
        at_ns=clock.set_seconds(1.25),
    )
    public = tracker.snapshot(at_ns=clock.set_seconds(1.5))
    current = cast(dict[str, Any], public["current_epoch"])
    last_closed = cast(dict[str, Any], public["last_closed_epoch"])
    assert current["key"]["config_version"] == 3
    assert current["output_send_count"] == 0
    assert last_closed["key"]["config_version"] == 2
    assert last_closed["output_send_count"] == 0

    finalized_a = tracker.release_epoch(key_a)
    assert finalized_a is not None
    finalized_a_key = cast(dict[str, int], finalized_a["key"])
    assert finalized_a_key["config_version"] == 1
    assert finalized_a["output_send_count"] == 1
    assert finalized_a["output_schedule_late_count"] == 1
    assert tracker.retained_closed_epoch_count == 1

    finalized_b = tracker.release_epoch(key_b)
    assert finalized_b is not None
    finalized_b_key = cast(dict[str, int], finalized_b["key"])
    assert finalized_b_key["config_version"] == 2
    assert tracker.retained_closed_epoch_count == 1


def test_late_receipts_recompute_closed_health_before_finalization() -> None:
    clock = _Clock()
    key_a = _key(1)
    tracker = RuntimePerformanceTracker(0.2, key_a, clock_ns=clock)
    tracker.mark_ready(at_ns=clock.set_seconds(0.0))
    assert tracker.retain_epoch(key_a)

    tracker.snapshot(at_ns=clock.set_seconds(3.0))
    degraded = tracker.snapshot(at_ns=clock.set_seconds(6.0))
    assert degraded["state"] == "degraded"
    assert tracker.bind_epoch(
        _key(2),
        color_correction_mode="off",
        light_wrap=0.0,
        at_ns=clock.set_seconds(6.0),
    )

    assert tracker.record_output(
        unique_base=True,
        schedule_late=False,
        expected_epoch=key_a,
        at_ns=clock.set_seconds(6.1),
    )
    still_recovering = cast(
        dict[str, Any],
        tracker.snapshot(at_ns=clock.set_seconds(6.2))["last_closed_epoch"],
    )
    assert still_recovering["state"] == "degraded"

    assert tracker.record_output(
        unique_base=True,
        schedule_late=False,
        expected_epoch=key_a,
        at_ns=clock.set_seconds(11.2),
    )
    recovered = cast(
        dict[str, Any],
        tracker.snapshot(at_ns=clock.set_seconds(11.3))["last_closed_epoch"],
    )
    assert recovered["state"] == "healthy"
    assert recovered["reason"] == "none"
    finalized = tracker.release_epoch(key_a)
    assert finalized is not None
    assert finalized["state"] == "healthy"
    assert finalized["reason"] == "none"


def test_fixed_percentiles_dominant_stage_and_bounded_window() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(30, _key(), clock_ns=clock)
    tracker.mark_ready(at_ns=0)
    for index, duration in enumerate((1.0, 2.0, 100.0, 3.0), start=1):
        tracker.record_processing(
            at_ns=clock.set_seconds(index * 0.1),
            deadline_missed=False,
            stages_ms={
                "segmentation.total": duration,
                "compositor.total": duration / 2.0,
                "pipeline.processing_only": duration * 10.0,
            },
        )
    snapshot = tracker.snapshot(at_ns=clock.set_seconds(0.5))
    stage_p50 = cast(dict[str, float | None], snapshot["stage_p50_ms"])
    stage_p95 = cast(dict[str, float | None], snapshot["stage_p95_ms"])
    assert stage_p50["segmentation.total"] == 2.0
    assert stage_p95["segmentation.total"] == 100.0
    assert snapshot["dominant_stage"] == "segmentation.total"

    for index in range(RUNTIME_PERFORMANCE_SAMPLE_LIMIT + 100):
        tracker.record_output(
            at_ns=clock.set_seconds(0.6 + index / 10_000.0),
            unique_base=False,
            schedule_late=False,
        )
    assert tracker.retained_sample_count == RUNTIME_PERFORMANCE_SAMPLE_LIMIT

    tracker.snapshot(at_ns=clock.set_seconds(6.0))
    assert tracker.retained_sample_count == 0


def test_hard_sample_cap_shortens_window_without_false_rate_collapse() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(240, _key(), clock_ns=clock)
    tracker.mark_ready(at_ns=0)
    for index in range(1, 1_921):
        at_ns = clock.set_seconds(index / 240.0)
        tracker.record_processing(at_ns=at_ns, deadline_missed=False)
        tracker.record_output(
            at_ns=at_ns,
            unique_base=True,
            schedule_late=False,
        )

    snapshot = tracker.snapshot(at_ns=clock.set_seconds(8.0))
    assert tracker.retained_sample_count <= RUNTIME_PERFORMANCE_SAMPLE_LIMIT
    assert (
        cast(int, snapshot["window_sample_count"]) <= RUNTIME_PERFORMANCE_SAMPLE_LIMIT
    )
    assert cast(float, snapshot["window_duration_s"]) < 5.0
    assert snapshot["output_send_fps"] == pytest.approx(240.0, abs=0.25)
    assert snapshot["processing_completed_fps"] == pytest.approx(240.0, abs=0.25)
    assert snapshot["sent_unique_base_fps"] == pytest.approx(240.0, abs=0.25)
    assert snapshot["output_attainment"] == pytest.approx(1.0, abs=0.002)
    assert snapshot["unique_attainment"] == pytest.approx(1.0, abs=0.002)
    assert snapshot["state"] == "healthy"
    assert snapshot["output_healthy"] is True
    assert snapshot["unique_healthy"] is True


def test_strict_scalar_validation_and_failed_state() -> None:
    clock = _Clock()
    with pytest.raises(ValueError, match="non-negative"):
        PerformanceEpochKey(-1, 0, 0, 0)
    with pytest.raises(ValueError, match="publisher mode"):
        PublisherTelemetry(mode=cast(Any, "/private/device"))

    tracker = RuntimePerformanceTracker(30, _key(), clock_ns=clock)
    with pytest.raises(ValueError, match="unknown key"):
        tracker.record_processing(
            at_ns=0,
            deadline_missed=False,
            stages_ms={"/private/model.onnx": 1.0},
        )
    tracker.update_publisher(
        PublisherTelemetry(mode="deadline-paced", state="failed"),
        at_ns=clock.set_seconds(1.0),
    )
    snapshot = tracker.snapshot(at_ns=clock.set_seconds(1.1))
    assert snapshot["state"] == "failed"
    assert snapshot["reason"] == "publisher-failed"
    assert snapshot["recommended_mitigation"] is None
    tracker.bind_epoch(
        PerformanceEpochKey(2, 0, 0, 0),
        color_correction_mode="off",
        light_wrap=0.0,
        at_ns=clock.set_seconds(1.2),
    )
    after_rotation = tracker.snapshot(at_ns=clock.set_seconds(1.3))
    assert after_rotation["state"] == "failed"
    assert after_rotation["reason"] == "publisher-failed"


def test_first_fatal_reason_wins_across_failure_sources() -> None:
    first_clock = _Clock()
    publisher_first = RuntimePerformanceTracker(30, _key(), clock_ns=first_clock)
    publisher_first.mark_failed(
        "publisher-failed",
        at_ns=first_clock.set_seconds(0.1),
    )
    publisher_first.mark_failed(
        "pipeline-failed",
        at_ns=first_clock.set_seconds(0.2),
    )
    first = publisher_first.snapshot(at_ns=first_clock.set_seconds(0.3))
    assert first["state"] == "failed"
    assert first["reason"] == "publisher-failed"

    second_clock = _Clock()
    pipeline_first = RuntimePerformanceTracker(30, _key(), clock_ns=second_clock)
    pipeline_first.mark_failed(
        "pipeline-failed",
        at_ns=second_clock.set_seconds(0.1),
    )
    pipeline_first.update_publisher(
        PublisherTelemetry(mode="deadline-paced", state="failed"),
        at_ns=second_clock.set_seconds(0.2),
    )
    second = pipeline_first.snapshot(at_ns=second_clock.set_seconds(0.3))
    assert second["state"] == "failed"
    assert second["reason"] == "pipeline-failed"
    assert cast(dict[str, object], second["publisher"])["state"] == "failed"


def test_publisher_counters_cannot_move_backwards() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(30, _key(), clock_ns=clock)
    tracker.update_publisher(
        PublisherTelemetry(handoff_overwrite_count=2),
        at_ns=clock.set_seconds(0.1),
    )
    with pytest.raises(ValueError, match="cannot decrease"):
        tracker.update_publisher(
            PublisherTelemetry(handoff_overwrite_count=1),
            at_ns=clock.set_seconds(0.2),
        )
