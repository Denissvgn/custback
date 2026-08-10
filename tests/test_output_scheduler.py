from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from custback.output_scheduler import (
    OutputPublisher,
    OutputPublisherError,
    OutputPublisherTimeout,
    OutputSendReceipt,
    PublicationPolicy,
    RemoteOutputProof,
    SafeBaseFrame,
    scheduled_slot,
    thaw_status,
)
from custback.vcam import OutputSendTiming, VideoOutput


def _pixels(value: int) -> np.ndarray:
    return np.full((6, 8, 3), value, np.uint8)


def _base(
    value: int,
    base_id: int,
    *,
    policy_epoch: int = 0,
    remote_proof: RemoteOutputProof | None = None,
    privacy_slate: bool = False,
    privacy_reason: str = "",
) -> SafeBaseFrame:
    return SafeBaseFrame.create(
        _pixels(value),
        base_id=base_id,
        config_version=3,
        capture_sequence=base_id + 10,
        captured_at_ns=base_id * 1_000_000,
        base_ready_at_ns=base_id * 1_000_000 + 500_000,
        policy_epoch=policy_epoch,
        segmentation_updated=True,
        processing_deadline_missed=False,
        status={"config_version": 3, "nested": {"safe": True}},
        remote_proof=remote_proof,
        privacy_slate=privacy_slate,
        privacy_reason=privacy_reason,
    )


class _PacedOutput(VideoOutput):
    paces = True

    def __init__(self, clock: list[int], interval_ns: int) -> None:
        self.clock = clock
        self.interval_ns = interval_ns
        self.frames: list[np.ndarray] = []
        self.closed = 0

    def send(self, frame_bgr: np.ndarray) -> None:
        self.send_with_timing(frame_bgr)

    def send_with_timing(self, frame_bgr: np.ndarray) -> OutputSendTiming:
        submitted = self.clock[0]
        self.frames.append(frame_bgr.copy())
        self.clock[0] += self.interval_ns
        return OutputSendTiming(
            submitted_at_ns=submitted,
            completed_at_ns=self.clock[0],
            submission_ms=0.0,
            pacing_wait_ms=self.interval_ns / 1_000_000.0,
            pacing_events=1,
        )

    def close(self) -> None:
        self.closed += 1


def test_safe_base_owns_read_only_pixels_and_frozen_status() -> None:
    source = _pixels(20)
    status = {"config_version": 3, "nested": {"safe": True}}

    base = SafeBaseFrame.create(
        source,
        base_id=1,
        config_version=3,
        capture_sequence=4,
        captured_at_ns=10,
        base_ready_at_ns=11,
        policy_epoch=0,
        segmentation_updated=True,
        processing_deadline_missed=False,
        status=status,
    )
    source[:] = 99
    status["config_version"] = 9

    assert np.all(base.pixels == 20)
    assert not base.pixels.flags.writeable
    assert thaw_status(base.status) == {
        "config_version": 3,
        "nested": {"safe": True},
    }
    with pytest.raises(ValueError):
        base.pixels[0, 0, 0] = 1


@pytest.mark.parametrize(
    ("deadline", "now", "interval", "expected"),
    [
        (100, 100, 10, (100, 0)),
        (100, 109, 10, (100, 0)),
        (100, 110, 10, (110, 1)),
        (100, 111, 10, (110, 1)),
        (100, 145, 10, (140, 4)),
    ],
)
def test_scheduled_slot_skips_late_slots_without_catch_up(
    deadline: int,
    now: int,
    interval: int,
    expected: tuple[int, int],
) -> None:
    assert scheduled_slot(deadline, now, interval) == expected


def test_paced_publisher_repeats_exact_base_at_absolute_cadence() -> None:
    interval_ns = round(1_000_000_000 / 30)
    clock = [0]
    output = _PacedOutput(clock, interval_ns)
    receipts: list[OutputSendReceipt] = []
    holder: dict[str, OutputPublisher] = {}

    def receive(receipt: OutputSendReceipt) -> None:
        receipts.append(receipt)
        if len(receipts) == 4:
            holder["publisher"].request_stop()

    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=30,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=receive,
        clock_ns=lambda: clock[0],
    )
    holder["publisher"] = publisher
    assert publisher.submit(_base(50, 1))
    publisher.start()

    assert publisher.wait_terminated(1.0)
    publisher.close()

    assert [receipt.timing.submitted_at_ns for receipt in receipts] == [
        0,
        interval_ns,
        interval_ns * 2,
        interval_ns * 3,
    ]
    assert [receipt.base_updated for receipt in receipts] == [True, False, False, False]
    assert [receipt.exact_final_repeat for receipt in receipts] == [
        False,
        True,
        True,
        True,
    ]
    assert output.closed == 1


def test_acceptance_aware_paced_sink_publishes_before_pacing_wait_finishes() -> None:
    pacing_started = threading.Event()
    release_pacing = threading.Event()
    receipt_published = threading.Event()
    events: list[str] = []

    class AcceptanceAwareOutput(VideoOutput):
        paces = True

        def __init__(self) -> None:
            self.closed = 0

        def send(self, _frame_bgr: np.ndarray) -> None:  # pragma: no cover
            raise AssertionError("publisher must use the acceptance-aware seam")

        def send_with_acceptance_timing(
            self,
            _frame_bgr: np.ndarray,
            on_accepted,
        ) -> OutputSendTiming:
            submitted_at_ns = time.monotonic_ns()
            events.append("sink-accepted")
            on_accepted(
                OutputSendTiming(
                    submitted_at_ns=submitted_at_ns,
                    completed_at_ns=submitted_at_ns,
                    submission_ms=0.0,
                    pacing_wait_ms=0.0,
                    pacing_events=1,
                )
            )
            events.append("pacing-started")
            pacing_started.set()
            assert release_pacing.wait(1.0)
            events.append("pacing-finished")
            completed_at_ns = time.monotonic_ns()
            return OutputSendTiming(
                submitted_at_ns=submitted_at_ns,
                completed_at_ns=completed_at_ns,
                submission_ms=0.0,
                pacing_wait_ms=(completed_at_ns - submitted_at_ns) / 1_000_000.0,
                pacing_events=1,
            )

        def close(self) -> None:
            self.closed += 1

    output = AcceptanceAwareOutput()

    def publish_receipt(_receipt: OutputSendReceipt) -> None:
        events.append("receipt-published")
        receipt_published.set()

    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=30,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=publish_receipt,
    )
    assert publisher.submit(_base(50, 1))
    publisher.start()

    assert pacing_started.wait(1.0)
    assert receipt_published.is_set()
    assert events == ["sink-accepted", "receipt-published", "pacing-started"]
    publisher.request_stop()
    release_pacing.set()
    assert publisher.wait_terminated(1.0)
    publisher.close()

    assert events[-1] == "pacing-finished"
    assert output.closed == 1


def test_nonpaced_publisher_uses_absolute_deadlines_with_fake_clock() -> None:
    interval_ns = round(1_000_000_000 / 30)
    clock = [0]
    owner_threads: list[int] = []
    receipts: list[OutputSendReceipt] = []
    holder: dict[str, OutputPublisher] = {}

    class NonPacedOutput(VideoOutput):
        paces = False

        def __init__(self) -> None:
            owner_threads.append(threading.get_ident())
            self.closed = 0

        def send(self, _frame_bgr: np.ndarray) -> None:
            owner_threads.append(threading.get_ident())

        def send_with_timing(self, frame_bgr: np.ndarray) -> OutputSendTiming:
            self.send(frame_bgr)
            return OutputSendTiming(
                submitted_at_ns=clock[0],
                completed_at_ns=clock[0],
                submission_ms=0.0,
                pacing_wait_ms=0.0,
            )

        def close(self) -> None:
            owner_threads.append(threading.get_ident())
            self.closed += 1

    output: NonPacedOutput | None = None

    def factory() -> NonPacedOutput:
        nonlocal output
        output = NonPacedOutput()
        return output

    def receive(receipt: OutputSendReceipt) -> None:
        receipts.append(receipt)
        if len(receipts) == 4:
            holder["publisher"].request_stop()

    publisher = OutputPublisher(
        factory,
        width=8,
        height=6,
        target_fps=30,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=receive,
        clock_ns=lambda: clock[0],
    )
    holder["publisher"] = publisher

    def advance_to(deadline_ns: int) -> bool:
        clock[0] = deadline_ns
        return True

    publisher._wait_until = advance_to
    assert publisher.submit(_base(60, 1))
    publisher.start()
    assert publisher.wait_terminated(1.0)
    publisher.close()

    assert [receipt.timing.submitted_at_ns for receipt in receipts] == [
        0,
        interval_ns,
        interval_ns * 2,
        interval_ns * 3,
    ]
    assert [receipt.application_pacing_events for receipt in receipts] == [0, 1, 1, 1]
    assert output is not None and output.closed == 1
    assert len(set(owner_threads)) == 1
    assert owner_threads[0] != threading.get_ident()


def test_late_wake_never_causes_a_catch_up_microburst() -> None:
    interval_ns = 10_000_000
    clock = [0]
    receipts: list[OutputSendReceipt] = []
    holder: dict[str, OutputPublisher] = {}

    class NonPacedOutput(VideoOutput):
        paces = False

        def send(self, _frame_bgr: np.ndarray) -> None:
            pass

        def send_with_timing(self, frame_bgr: np.ndarray) -> OutputSendTiming:
            self.send(frame_bgr)
            return OutputSendTiming(
                submitted_at_ns=clock[0],
                completed_at_ns=clock[0],
                submission_ms=0.0,
                pacing_wait_ms=0.0,
            )

    def receive(receipt: OutputSendReceipt) -> None:
        receipts.append(receipt)
        if len(receipts) == 3:
            holder["publisher"].request_stop()

    publisher = OutputPublisher(
        NonPacedOutput,
        width=8,
        height=6,
        target_fps=100,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=receive,
        clock_ns=lambda: clock[0],
    )
    holder["publisher"] = publisher
    wait_count = 0

    def advance_with_one_late_wake(deadline_ns: int) -> bool:
        nonlocal wait_count
        wait_count += 1
        clock[0] = (
            deadline_ns + (2 * interval_ns) + (interval_ns // 2)
            if wait_count == 1
            else deadline_ns
        )
        return True

    publisher._wait_until = advance_with_one_late_wake
    assert publisher.submit(_base(60, 1))
    publisher.start()
    assert publisher.wait_terminated(1.0)
    publisher.close()

    submissions = [receipt.timing.submitted_at_ns for receipt in receipts]
    assert submissions == [0, 35_000_000, 45_000_000]
    assert all(
        later - earlier >= interval_ns
        for earlier, later in zip(submissions, submissions[1:])
    )
    assert receipts[1].schedule_skipped_slots == 2


def test_slow_nonpaced_sink_counts_elapsed_slots_without_catch_up() -> None:
    interval_ns = 10_000_000
    clock = [0]
    receipts: list[OutputSendReceipt] = []
    holder: dict[str, OutputPublisher] = {}

    class SlowNonPacedOutput(VideoOutput):
        paces = False

        def send(self, _frame_bgr: np.ndarray) -> None:
            clock[0] += 35_000_000

        def send_with_timing(self, frame_bgr: np.ndarray) -> OutputSendTiming:
            started_at_ns = clock[0]
            self.send(frame_bgr)
            return OutputSendTiming(
                submitted_at_ns=clock[0],
                completed_at_ns=clock[0],
                submission_ms=(clock[0] - started_at_ns) / 1_000_000.0,
                pacing_wait_ms=0.0,
            )

    def receive(receipt: OutputSendReceipt) -> None:
        receipts.append(receipt)
        if len(receipts) == 3:
            holder["publisher"].request_stop()

    publisher = OutputPublisher(
        SlowNonPacedOutput,
        width=8,
        height=6,
        target_fps=100,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=receive,
        clock_ns=lambda: clock[0],
    )
    holder["publisher"] = publisher

    def advance_to(deadline_ns: int) -> bool:
        clock[0] = deadline_ns
        return True

    publisher._wait_until = advance_to
    assert publisher.submit(_base(60, 1))
    publisher.start()
    assert publisher.wait_terminated(1.0)
    publisher.close()

    submissions = [receipt.timing.submitted_at_ns for receipt in receipts]
    assert submissions == [35_000_000, 80_000_000, 125_000_000]
    assert all(
        later - earlier >= interval_ns
        for earlier, later in zip(submissions, submissions[1:])
    )
    assert [receipt.schedule_skipped_slots for receipt in receipts] == [3, 3, 3]
    assert publisher.snapshot().schedule_skipped_slots == 9


def test_depth_one_handoff_replaces_unsent_base() -> None:
    interval_ns = 10_000_000
    clock = [0]
    output = _PacedOutput(clock, interval_ns)
    receipts: list[OutputSendReceipt] = []
    holder: dict[str, OutputPublisher] = {}

    def receive(receipt: OutputSendReceipt) -> None:
        receipts.append(receipt)
        holder["publisher"].request_stop()

    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=100,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=receive,
        clock_ns=lambda: clock[0],
    )
    holder["publisher"] = publisher
    assert publisher.submit(_base(20, 1))
    assert publisher.submit(_base(80, 2))
    assert publisher.snapshot().handoff_overwrite_count == 1
    publisher.start()

    assert publisher.wait_terminated(1.0)
    publisher.close()

    assert len(receipts) == 1
    assert receipts[0].base is not None and receipts[0].base.base_id == 2
    assert np.all(receipts[0].pixels == 80)


def test_remote_policy_accepts_only_matching_proof_or_declared_slate() -> None:
    publisher = OutputPublisher(
        lambda: _PacedOutput([0], 1),
        width=8,
        height=6,
        target_fps=30,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(4, "remote", raw_epoch=7, renderer_session=2),
        on_send=lambda _receipt: None,
        clock_ns=lambda: 100,
    )
    wrong = _base(
        40,
        1,
        policy_epoch=4,
        remote_proof=RemoteOutputProof(6, 2, 200),
    )
    valid = _base(
        50,
        2,
        policy_epoch=4,
        remote_proof=RemoteOutputProof(7, 2, 200),
    )
    slate = _base(
        0,
        3,
        policy_epoch=4,
        privacy_slate=True,
        privacy_reason="renderer-missing",
    )

    assert not publisher.submit(wrong)
    assert publisher.submit(valid)
    assert publisher.submit(slate)
    assert publisher.snapshot().handoff_overwrite_count == 1
    publisher.close()


def test_send_failure_latches_error_and_closes_exactly_once() -> None:
    clock = [0]

    class FailingOutput(_PacedOutput):
        calls = 0

        def send_with_timing(self, frame_bgr: np.ndarray) -> OutputSendTiming:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("send failed")
            return super().send_with_timing(frame_bgr)

    output = FailingOutput(clock, 10_000_000)
    failures: list[type[BaseException]] = []
    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=100,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=lambda _receipt: None,
        on_failure=lambda exc: failures.append(type(exc)),
        clock_ns=lambda: clock[0],
    )
    assert publisher.submit(_base(30, 1))
    publisher.start()

    assert publisher.wait_terminated(1.0)
    with pytest.raises(OutputPublisherError):
        publisher.close()

    assert failures == [RuntimeError]
    assert output.closed == 1
    assert publisher.snapshot().state == "failed"


def test_first_sink_send_failure_never_publishes_and_closes_exactly_once() -> None:
    class FirstSendFailure(VideoOutput):
        paces = False

        def __init__(self) -> None:
            self.calls = 0
            self.closed = 0

        def send(self, _frame_bgr: np.ndarray) -> None:
            self.calls += 1
            raise RuntimeError("first sink send failed")

        def close(self) -> None:
            self.closed += 1

    output = FirstSendFailure()
    receipts: list[OutputSendReceipt] = []
    failures: list[type[BaseException]] = []
    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=30,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=receipts.append,
        on_failure=lambda exc: failures.append(type(exc)),
    )
    assert publisher.submit(_base(30, 1))
    publisher.start()
    assert publisher.wait_terminated(1.0)

    with pytest.raises(OutputPublisherError, match="failed during startup") as ready:
        publisher.wait_ready(0.1)
    assert isinstance(ready.value.__cause__, RuntimeError)
    assert str(ready.value.__cause__) == "first sink send failed"
    with pytest.raises(OutputPublisherError, match="output publisher failed") as closed:
        publisher.close()
    assert isinstance(closed.value.__cause__, RuntimeError)
    assert str(closed.value.__cause__) == "first sink send failed"
    assert output.calls == 1
    assert output.closed == 1
    assert receipts == []
    assert failures == [RuntimeError]
    assert publisher.snapshot().state == "failed"


def test_hub_publication_failure_is_fatal_after_one_ambiguous_send() -> None:
    clock = [0]
    output = _PacedOutput(clock, 10_000_000)
    failures: list[type[BaseException]] = []

    def reject_publication(_receipt: OutputSendReceipt) -> None:
        raise RuntimeError("hub publication failed")

    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=100,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=reject_publication,
        on_failure=lambda exc: failures.append(type(exc)),
        clock_ns=lambda: clock[0],
    )
    assert publisher.submit(_base(30, 1))
    publisher.start()
    assert publisher.wait_terminated(1.0)

    with pytest.raises(OutputPublisherError):
        publisher.close()
    assert len(output.frames) == 1
    assert output.closed == 1
    assert failures == [RuntimeError]


def test_blocked_sink_makes_close_bounded_then_owner_closes_once() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingOutput(VideoOutput):
        paces = False

        def __init__(self) -> None:
            self.closed = 0

        def send(self, _frame_bgr: np.ndarray) -> None:
            entered.set()
            release.wait()

        def close(self) -> None:
            self.closed += 1

    output = BlockingOutput()
    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=30,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=lambda _receipt: None,
    )
    assert publisher.submit(_base(30, 1))
    publisher.start()
    assert entered.wait(1.0)

    with pytest.raises(OutputPublisherTimeout):
        publisher.close(0.01)
    assert output.closed == 0
    release.set()
    assert publisher.wait_terminated(1.0)
    publisher.close()
    assert output.closed == 1


def test_slate_fence_waits_for_send_gate_and_clears_retained_base() -> None:
    sent = threading.Event()

    class RealTimeOutput(VideoOutput):
        paces = False

        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []

        def send(self, frame_bgr: np.ndarray) -> None:
            self.frames.append(frame_bgr.copy())
            sent.set()

    output = RealTimeOutput()
    receipts: list[OutputSendReceipt] = []
    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=100,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=receipts.append,
    )
    assert publisher.submit(_base(90, 1))
    publisher.start()
    assert sent.wait(1.0)

    publisher.fence_to_slate(
        PublicationPolicy(1, "remote", raw_epoch=1, renderer_session=1),
        reason="mode-transition",
    )
    deadline = time.monotonic() + 1.0
    while not any(receipt.privacy_slate for receipt in receipts):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    publisher.close()

    first_slate = next(receipt for receipt in receipts if receipt.privacy_slate)
    assert first_slate.base is None
    assert first_slate.privacy_reason == "mode-transition"
    assert np.array_equal(first_slate.pixels, _pixels(0))


def test_privacy_fault_reason_survives_next_raw_epoch_fence() -> None:
    first_send = threading.Event()

    class RealTimeOutput(VideoOutput):
        paces = False

        def send(self, _frame_bgr: np.ndarray) -> None:
            first_send.set()

    output = RealTimeOutput()
    receipts: list[OutputSendReceipt] = []
    holder: dict[str, OutputPublisher] = {}

    def receive(receipt: OutputSendReceipt) -> None:
        receipts.append(receipt)
        if receipt.privacy_reason == "privacy-raw-echo":
            holder["publisher"].request_stop()

    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=100,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(1, "remote", raw_epoch=1, renderer_session=4),
        on_send=receive,
    )
    holder["publisher"] = publisher
    assert publisher.submit(
        _base(
            0,
            1,
            policy_epoch=1,
            privacy_slate=True,
            privacy_reason="renderer-missing",
        )
    )
    publisher.start()
    assert first_send.wait(1.0)
    assert publisher.submit(
        _base(
            0,
            2,
            policy_epoch=1,
            privacy_slate=True,
            privacy_reason="privacy-raw-echo",
        )
    )
    publisher.fence_to_slate(
        PublicationPolicy(2, "remote", raw_epoch=2, renderer_session=4),
        reason="awaiting-renderer",
    )
    assert publisher.wait_terminated(1.0)
    publisher.close()

    assert receipts[-1].base is None
    assert receipts[-1].privacy_reason == "privacy-raw-echo"
    np.testing.assert_array_equal(receipts[-1].pixels, _pixels(0))


def test_slate_fence_discards_unsent_pending_evidence() -> None:
    discarded: list[int] = []
    publisher = OutputPublisher(
        lambda: _PacedOutput([0], 1),
        width=8,
        height=6,
        target_fps=30,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=lambda _receipt: None,
        on_overwrite=discarded.append,
    )
    assert publisher.submit(_base(90, 41))

    publisher.fence_to_slate(
        PublicationPolicy(1, "remote", raw_epoch=1, renderer_session=1),
        reason="mode-transition",
    )
    publisher.close()

    assert discarded == [41]
    assert not publisher.snapshot().pending


def test_stop_discards_unsent_pending_evidence() -> None:
    discarded: list[int] = []
    publisher = OutputPublisher(
        lambda: _PacedOutput([0], 1),
        width=8,
        height=6,
        target_fps=30,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=lambda _receipt: None,
        on_overwrite=discarded.append,
    )
    assert publisher.submit(_base(90, 42))

    publisher.request_stop()
    publisher.close()

    assert discarded == [42]


def test_release_callback_is_exact_for_pending_overwrite_and_stop() -> None:
    overwritten: list[int] = []
    released: list[int] = []
    publisher = OutputPublisher(
        lambda: _PacedOutput([0], 1),
        width=8,
        height=6,
        target_fps=30,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=lambda _receipt: None,
        on_overwrite=overwritten.append,
        on_release=released.append,
    )

    assert publisher.submit(_base(10, 51))
    assert publisher.submit(_base(20, 52))
    assert overwritten == [51]
    assert released == [51]
    publisher.request_stop()
    publisher.close()

    assert overwritten == [51, 52]
    assert released == [51, 52]


def test_release_callback_tracks_current_displacement_and_final_shutdown() -> None:
    interval_ns = 10_000_000
    clock = [0]
    output = _PacedOutput(clock, interval_ns)
    released: list[int] = []
    receipts: list[OutputSendReceipt] = []
    holder: dict[str, OutputPublisher] = {}

    def receive(receipt: OutputSendReceipt) -> None:
        receipts.append(receipt)
        if len(receipts) == 1:
            assert holder["publisher"].submit(_base(50, 62))
        elif len(receipts) == 2:
            holder["publisher"].request_stop()

    publisher = OutputPublisher(
        lambda: output,
        width=8,
        height=6,
        target_fps=100,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=receive,
        on_release=released.append,
        clock_ns=lambda: clock[0],
    )
    holder["publisher"] = publisher
    assert publisher.submit(_base(50, 61))
    publisher.start()
    assert publisher.wait_terminated(1.0)
    publisher.close()

    assert [receipt.base_updated for receipt in receipts] == [True, True]
    assert receipts[1].exact_final_repeat is True
    assert released == [61, 62]
    assert not hasattr(publisher, "_last_sent")
    assert isinstance(publisher._last_sent_digest, bytes)
    assert len(publisher._last_sent_digest) == 32


def test_privacy_fence_releases_current_base_once() -> None:
    sent = threading.Event()
    released: list[int] = []

    class RealTimeOutput(VideoOutput):
        paces = False

        def send(self, _frame_bgr: np.ndarray) -> None:
            sent.set()

    publisher = OutputPublisher(
        RealTimeOutput,
        width=8,
        height=6,
        target_fps=100,
        slate_pixels=_pixels(0),
        initial_policy=PublicationPolicy(0, "local"),
        on_send=lambda _receipt: None,
        on_release=released.append,
    )
    assert publisher.submit(_base(90, 71))
    publisher.start()
    assert sent.wait(1.0)

    publisher.fence_to_slate(
        PublicationPolicy(1, "remote", raw_epoch=1, renderer_session=1),
        reason="mode-transition",
    )
    publisher.close()

    assert released == [71]
