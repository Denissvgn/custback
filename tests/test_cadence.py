from __future__ import annotations

import math

import pytest

from custback.cadence import (
    CADENCE_EVENT_LIMIT,
    CADENCE_MAX_REPORTED_MS,
    CADENCE_SAMPLE_LIMIT,
    CADENCE_WINDOW_NS,
    CadenceTracker,
)


def _record_base(
    tracker: CadenceTracker,
    sequence: int,
    timestamp_ns: int,
    *,
    ready_at_ns: int | None = None,
    sent_at_ns: int | None = None,
    segmentation_updated: bool = True,
    exact_final_repeat: bool = False,
    **counters: object,
) -> None:
    ready = timestamp_ns if ready_at_ns is None else ready_at_ns
    sent = ready if sent_at_ns is None else sent_at_ns
    tracker.record_send(
        sent_at_ns=sent,
        capture_sequence=sequence,
        captured_at_ns=timestamp_ns,
        base_ready_at_ns=ready,
        base_updated=True,
        segmentation_updated=segmentation_updated,
        exact_final_repeat=exact_final_repeat,
        **counters,  # type: ignore[arg-type]
    )


def _record_reuse(
    tracker: CadenceTracker,
    sent_at_ns: int,
    *,
    exact_final_repeat: bool = True,
    **counters: object,
) -> None:
    tracker.record_send(
        sent_at_ns=sent_at_ns,
        base_updated=False,
        segmentation_updated=False,
        exact_final_repeat=exact_final_repeat,
        **counters,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "target",
    [0, -1, 1e-320, float("nan"), float("inf"), True, "30"],
)
def test_target_output_fps_must_be_finite_and_positive(target):
    with pytest.raises(ValueError):
        CadenceTracker(target)  # type: ignore[arg-type]


def test_empty_snapshot_has_bounded_scalar_defaults():
    tracker = CadenceTracker(30, clock_ns=lambda: 123)

    snapshot = tracker.snapshot()
    public = snapshot.as_dict()

    assert snapshot.capture_sequence == 0
    assert snapshot.unique_capture_count == 0
    assert snapshot.base_composite_update_count == 0
    assert snapshot.base_composite_reuse_count == 0
    assert snapshot.exact_final_output_repeat_count == 0
    assert snapshot.output_send_count == 0
    assert snapshot.base_composite_reuse_ratio == 0.0
    assert snapshot.exact_final_output_repeat_ratio == 0.0
    assert snapshot.last_unique_frame_age_ms is None
    assert snapshot.capture_timestamp_delta_p50_ms is None
    assert snapshot.output_send_delta_p95_ms is None
    assert snapshot.cadence_mismatch_active is False
    assert tracker.retained_event_count == 0
    assert not any(key.endswith("_ns") for key in public)


def test_pixel_identical_successful_capture_is_unique_and_exact():
    tracker = CadenceTracker(30)
    _record_base(
        tracker,
        1,
        0,
        ready_at_ns=10_000_000,
        sent_at_ns=20_000_000,
    )
    # Pixel equality is supplied as final-output provenance. It does not turn
    # this successful second capture into safe-base reuse.
    _record_base(
        tracker,
        2,
        33_000_000,
        ready_at_ns=50_000_000,
        sent_at_ns=60_000_000,
        exact_final_repeat=True,
    )

    snapshot = tracker.snapshot(now_ns=70_000_000)

    assert snapshot.unique_capture_count == 2
    assert snapshot.segmentation_update_count == 2
    assert snapshot.base_composite_update_count == 2
    assert snapshot.base_composite_reuse_count == 0
    assert snapshot.exact_final_output_repeat_count == 1
    assert snapshot.output_send_count == 2
    assert snapshot.base_composite_reuse_ratio == 0.0
    assert snapshot.exact_final_output_repeat_ratio == 1.0
    assert snapshot.last_unique_frame_age_ms == 20.0


def test_new_visual_base_may_reuse_exact_capture_provenance():
    tracker = CadenceTracker(30)
    _record_base(
        tracker,
        7,
        10_000_000,
        ready_at_ns=20_000_000,
        sent_at_ns=30_000_000,
    )
    # A renderer response or committed scalar policy can produce a second
    # immutable visual base from the same capture. It is a base update (and
    # may be pixel-identical), but it must not invent a second camera sample.
    _record_base(
        tracker,
        7,
        10_000_000,
        ready_at_ns=40_000_000,
        sent_at_ns=50_000_000,
        segmentation_updated=False,
        exact_final_repeat=True,
    )

    snapshot = tracker.snapshot(now_ns=50_000_000)
    assert snapshot.capture_sequence == 7
    assert snapshot.unique_capture_count == 1
    assert snapshot.base_composite_update_count == 2
    assert snapshot.exact_final_output_repeat_count == 1
    assert snapshot.capture_sequence_gap_count == 0
    assert snapshot.capture_missing_input_count == 0


def test_no_unread_send_is_safe_base_reuse_independent_of_final_equality():
    tracker = CadenceTracker(30)
    _record_base(tracker, 1, 0)
    _record_reuse(tracker, 33_333_333, exact_final_repeat=False)
    assert tracker.snapshot(now_ns=33_333_333).cadence_mismatch_active is False
    _record_reuse(tracker, 66_666_666, exact_final_repeat=True)

    snapshot = tracker.snapshot(now_ns=66_666_666)

    assert snapshot.unique_capture_count == 1
    assert snapshot.segmentation_update_count == 1
    assert snapshot.base_composite_update_count == 1
    assert snapshot.base_composite_reuse_count == 2
    assert snapshot.exact_final_output_repeat_count == 1
    assert snapshot.output_send_count == 3
    assert snapshot.base_composite_reuse_ratio == pytest.approx(2 / 3)
    assert snapshot.exact_final_output_repeat_ratio == pytest.approx(1 / 2)
    assert snapshot.base_composite_reuse_fps == pytest.approx(30.0, rel=1e-6)
    assert snapshot.exact_final_output_repeat_fps == pytest.approx(15.0, rel=1e-6)
    assert snapshot.cadence_mismatch_active is True


def test_irregular_timestamps_report_nearest_rank_deltas_and_jitter():
    tracker = CadenceTracker(25)
    _record_base(
        tracker,
        1,
        0,
        ready_at_ns=10_000_000,
        sent_at_ns=20_000_000,
    )
    _record_reuse(tracker, 50_000_000)
    _record_base(
        tracker,
        2,
        40_000_000,
        ready_at_ns=60_000_000,
        sent_at_ns=80_000_000,
        segmentation_updated=False,
        exact_final_repeat=True,
    )
    _record_base(
        tracker,
        3,
        140_000_000,
        ready_at_ns=180_000_000,
        sent_at_ns=200_000_000,
    )

    snapshot = tracker.snapshot(now_ns=230_000_000)

    assert snapshot.unique_capture_fps == pytest.approx(2 / 0.14)
    assert snapshot.base_composite_update_fps == pytest.approx(2 / 0.18)
    assert snapshot.segmentation_update_fps == pytest.approx(1 / 0.18)
    assert snapshot.output_send_fps == pytest.approx(3 / 0.18)
    assert snapshot.capture_timestamp_delta_p50_ms == 40.0
    assert snapshot.capture_timestamp_delta_p95_ms == 100.0
    assert snapshot.base_composite_delta_p50_ms == 50.0
    assert snapshot.base_composite_delta_p95_ms == 120.0
    assert snapshot.output_send_delta_p50_ms == 30.0
    assert snapshot.output_send_delta_p95_ms == 120.0
    assert snapshot.output_send_jitter_p50_ms == 10.0
    assert snapshot.output_send_jitter_p95_ms == 80.0
    assert snapshot.last_unique_frame_age_ms == 50.0


def test_run_b_lifetime_arithmetic_and_mismatch_are_exact():
    tracker = CadenceTracker(30)
    send_count = 1105
    base_count = 541
    # Spread 541 successful base updates across all 1,105 send opportunities.
    base_indices = {
        round(index * (send_count - 1) / (base_count - 1))
        for index in range(base_count)
    }
    assert len(base_indices) == base_count
    capture_sequence = 0

    for index in range(send_count):
        sent_at_ns = index * 33_333_333
        if index in base_indices:
            capture_sequence += 1
            ready_at_ns = max(0, sent_at_ns - 1_000_000)
            captured_at_ns = max(0, ready_at_ns - 5_000_000)
            _record_base(
                tracker,
                capture_sequence,
                captured_at_ns,
                ready_at_ns=ready_at_ns,
                sent_at_ns=sent_at_ns,
            )
        else:
            _record_reuse(tracker, sent_at_ns)

    snapshot = tracker.snapshot(now_ns=(send_count - 1) * 33_333_333)

    assert snapshot.unique_capture_count == 541
    assert snapshot.segmentation_update_count == 541
    assert snapshot.base_composite_update_count == 541
    assert snapshot.base_composite_reuse_count == 564
    assert snapshot.exact_final_output_repeat_count == 564
    assert snapshot.output_send_count == 1105
    assert snapshot.base_composite_reuse_ratio == pytest.approx(564 / 1105)
    assert snapshot.exact_final_output_repeat_ratio == pytest.approx(564 / 1104)
    assert snapshot.output_send_fps == pytest.approx(30.0, rel=1e-6)
    assert 14.0 < snapshot.base_composite_update_fps < 16.0
    assert snapshot.cadence_mismatch_active is True


def test_sequence_gaps_and_operational_counters_commit_with_send():
    tracker = CadenceTracker(30)
    _record_base(tracker, 5, 0)
    _record_base(
        tracker,
        8,
        30_000_000,
        ready_at_ns=31_000_000,
        sent_at_ns=32_000_000,
        processing_deadline_missed=True,
        serialized_new_frame_deadline_missed=True,
        output_sink_pacing_events=2,
        output_sink_recovery_events=1,
        application_pacing_events=3,
        output_schedule_late=True,
    )

    snapshot = tracker.snapshot(now_ns=32_000_000)

    assert snapshot.capture_sequence == 8
    assert snapshot.capture_sequence_gap_count == 1
    assert snapshot.capture_missing_input_count == 2
    assert snapshot.processing_deadline_misses == 1
    assert snapshot.serialized_new_frame_deadline_misses == 1
    assert snapshot.output_sink_pacing_events == 2
    assert snapshot.output_sink_recovery_events == 1
    assert snapshot.application_pacing_events == 3
    assert snapshot.output_schedule_late_events == 1


def test_invalid_event_is_rejected_without_partial_counter_mutation():
    tracker = CadenceTracker(30)
    _record_base(
        tracker,
        1,
        10,
        ready_at_ns=20,
        sent_at_ns=30,
    )
    before = tracker.snapshot(now_ns=30)

    with pytest.raises(ValueError, match="capture timestamps"):
        _record_base(
            tracker,
            3,
            9,
            ready_at_ns=40,
            sent_at_ns=50,
        )
    with pytest.raises(ValueError, match="capture sequence"):
        _record_base(
            tracker,
            1,
            40,
            ready_at_ns=50,
            sent_at_ns=60,
        )
    with pytest.raises(ValueError, match="must be ordered"):
        _record_base(
            tracker,
            2,
            50,
            ready_at_ns=40,
            sent_at_ns=60,
        )

    after = tracker.snapshot(now_ns=60)
    assert after.output_send_count == before.output_send_count
    assert after.unique_capture_count == before.unique_capture_count
    assert after.capture_sequence_gap_count == before.capture_sequence_gap_count
    assert after.capture_missing_input_count == before.capture_missing_input_count


def test_first_send_and_provenance_shape_are_validated():
    tracker = CadenceTracker(30)
    with pytest.raises(ValueError, match="first successful send"):
        _record_reuse(tracker, 0)
    with pytest.raises(ValueError, match="no comparable final output"):
        _record_base(tracker, 1, 0, exact_final_repeat=True)
    with pytest.raises(ValueError, match="requires a base update"):
        tracker.record_send(
            sent_at_ns=0,
            base_updated=False,
            segmentation_updated=True,
            exact_final_repeat=False,
        )
    with pytest.raises(ValueError, match="deadline miss requires a base update"):
        tracker.record_send(
            sent_at_ns=0,
            base_updated=False,
            segmentation_updated=False,
            exact_final_repeat=False,
            serialized_new_frame_deadline_missed=True,
        )
    with pytest.raises(ValueError, match="matching capture"):
        tracker.record_send(
            sent_at_ns=0,
            capture_sequence=1,
            captured_at_ns=0,
            base_updated=True,
            segmentation_updated=True,
            exact_final_repeat=False,
        )
    assert tracker.snapshot(now_ns=0).output_send_count == 0


def test_window_is_time_pruned_and_event_ring_is_hard_bounded():
    tracker = CadenceTracker(1000)
    _record_base(tracker, 1, 0)
    for index in range(1, CADENCE_EVENT_LIMIT + 1):
        _record_reuse(tracker, index * 1_000_000)

    assert tracker.retained_event_count == CADENCE_EVENT_LIMIT
    snapshot = tracker.snapshot(now_ns=CADENCE_EVENT_LIMIT * 1_000_000)
    assert snapshot.output_send_count == CADENCE_EVENT_LIMIT + 1
    assert snapshot.base_composite_reuse_count == CADENCE_EVENT_LIMIT
    assert snapshot.exact_final_output_repeat_ratio == 1.0

    tracker.snapshot(now_ns=CADENCE_EVENT_LIMIT * 1_000_000 + CADENCE_WINDOW_NS + 1)
    assert tracker.retained_event_count == 0
    expired = tracker.snapshot(
        now_ns=CADENCE_EVENT_LIMIT * 1_000_000 + CADENCE_WINDOW_NS + 1
    )
    assert expired.output_send_fps == 0.0
    assert expired.output_send_delta_p50_ms is None
    # Lifetime provenance is not erased with the rolling window.
    assert expired.output_send_count == CADENCE_EVENT_LIMIT + 1


def test_percentiles_use_only_the_latest_bounded_sample_set():
    tracker = CadenceTracker(1000)
    timestamp_ns = 0
    _record_base(tracker, 1, timestamp_ns)
    sequence = 1
    # More than five percent of the complete history are long outliers. All of
    # them precede the most recent 256 interval samples.
    for _ in range(20):
        sequence += 1
        timestamp_ns += 50_000_000
        _record_base(tracker, sequence, timestamp_ns)
    for _ in range(CADENCE_SAMPLE_LIMIT):
        sequence += 1
        timestamp_ns += 1_000_000
        _record_base(tracker, sequence, timestamp_ns)

    snapshot = tracker.snapshot(now_ns=timestamp_ns)

    assert snapshot.capture_timestamp_delta_p50_ms == 1.0
    assert snapshot.capture_timestamp_delta_p95_ms == 1.0
    assert snapshot.base_composite_delta_p95_ms == 1.0
    assert snapshot.output_send_delta_p95_ms == 1.0
    assert snapshot.output_send_jitter_p95_ms == 0.0


def test_relative_age_is_finite_bounded_and_contains_no_absolute_timestamp():
    tracker = CadenceTracker(30)
    private_timestamp = 1_234_567_890_123_456
    _record_base(tracker, 1, private_timestamp)

    snapshot = tracker.snapshot(
        now_ns=private_timestamp + math.ceil(CADENCE_MAX_REPORTED_MS * 1_000_000) + 1,
    )
    public = snapshot.as_dict()

    assert snapshot.last_unique_frame_age_ms == CADENCE_MAX_REPORTED_MS
    assert private_timestamp not in public.values()
    assert not any(key.endswith("_ns") for key in public)
