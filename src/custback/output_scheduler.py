"""Target-paced, depth-one publication of already guarded output frames.

The publisher is intentionally ignorant of capture, matte, and renderer pixel
state.  Its producer may hand it only an owned final BGR frame plus bounded
scalar provenance.  That separation lets output cadence continue while an
expensive base is computed without allowing the output lane to advance any
temporal image-processing state.
"""

from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

import numpy as np

from .geometry import validate_bgr_frame
from .vcam import OutputSendTiming, VideoOutput


PublisherMode = Literal["local", "remote"]
PublisherState = Literal["starting", "running", "stopping", "stopped", "failed"]


class OutputPublisherError(RuntimeError):
    """The output owner failed to open, send, publish, or close its sink."""


class OutputPublisherTimeout(OutputPublisherError):
    """The output owner did not acknowledge an operation before its deadline."""


def _non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _freeze_scalars(value: object) -> object:
    """Freeze a path-free JSON-like scalar snapshot without retaining aliases."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Enum) and isinstance(value.value, str):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("status values must be finite")
        return float(value)
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("status keys must be strings")
            frozen[key] = _freeze_scalars(item)
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_scalars(item) for item in value)
    raise TypeError("status snapshots may contain only JSON-like scalar values")


def thaw_status(value: object) -> object:
    """Return a detached mutable projection of a frozen scalar snapshot."""

    if isinstance(value, Mapping):
        return {str(key): thaw_status(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_status(item) for item in value]
    return copy.deepcopy(value)


def scheduled_slot(
    deadline_ns: int,
    now_ns: int,
    interval_ns: int,
) -> tuple[int, int]:
    """Return the current absolute slot and slots skipped without catch-up."""

    _non_negative_int(deadline_ns, "schedule deadline")
    _non_negative_int(now_ns, "schedule timestamp")
    _positive_int(interval_ns, "schedule interval")
    if now_ns < deadline_ns + interval_ns:
        return deadline_ns, 0
    skipped = (now_ns - deadline_ns) // interval_ns
    return deadline_ns + skipped * interval_ns, skipped


@dataclass(frozen=True)
class PublicationPolicy:
    """The exact privacy epoch a base must satisfy before it may be sent."""

    epoch: int
    mode: PublisherMode
    raw_epoch: int = 0
    renderer_session: int = 0

    def __post_init__(self) -> None:
        _non_negative_int(self.epoch, "publication policy epoch")
        if self.mode not in {"local", "remote"}:
            raise ValueError("publication policy mode must be local or remote")
        _non_negative_int(self.raw_epoch, "publication raw epoch")
        _non_negative_int(self.renderer_session, "renderer session")
        if self.mode == "local" and (self.raw_epoch or self.renderer_session):
            raise ValueError("local publication policy cannot carry remote identity")


@dataclass(frozen=True)
class RemoteOutputProof:
    """Bounded scalar proof for one guarded renderer output candidate."""

    raw_epoch: int
    renderer_session: int
    valid_until_ns: int

    def __post_init__(self) -> None:
        _non_negative_int(self.raw_epoch, "remote proof raw epoch")
        _non_negative_int(self.renderer_session, "remote proof renderer session")
        _non_negative_int(self.valid_until_ns, "remote proof expiry")


@dataclass(frozen=True)
class SafeBaseFrame:
    """One owned final frame eligible for a first send and exact repeats."""

    pixels: np.ndarray
    base_id: int
    config_version: int
    capture_sequence: int
    captured_at_ns: int
    base_ready_at_ns: int
    policy_epoch: int
    segmentation_updated: bool
    processing_deadline_missed: bool
    status: Mapping[str, object]
    performance_epoch: tuple[int, int, int, int] = (0, 0, 0, 0)
    remote_proof: RemoteOutputProof | None = None
    resolved_remote_raw_epoch: int = 0
    privacy_slate: bool = False
    privacy_reason: str = ""
    matte_bundle_sequence: int | None = None
    _pixel_digest: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_pixel_digest",
            hashlib.blake2b(
                self.pixels,
                digest_size=32,
                person=b"custback-out-v1",
            ).digest(),
        )

    @property
    def pixel_digest(self) -> bytes:
        """Return the bounded content token used for repeat classification."""

        return self._pixel_digest

    @classmethod
    def create(
        cls,
        pixels: np.ndarray,
        *,
        base_id: int,
        config_version: int,
        capture_sequence: int,
        captured_at_ns: int,
        base_ready_at_ns: int,
        policy_epoch: int,
        segmentation_updated: bool,
        processing_deadline_missed: bool,
        status: Mapping[str, object],
        performance_epoch: tuple[int, int, int, int] = (0, 0, 0, 0),
        remote_proof: RemoteOutputProof | None = None,
        resolved_remote_raw_epoch: int = 0,
        privacy_slate: bool = False,
        privacy_reason: str = "",
        matte_bundle_sequence: int | None = None,
    ) -> "SafeBaseFrame":
        frame = validate_bgr_frame(
            pixels,
            name="safe output base",
            require_contiguous=True,
        ).copy()
        frame.setflags(write=False)
        for value, name in (
            (base_id, "base id"),
            (config_version, "config version"),
            (capture_sequence, "capture sequence"),
            (captured_at_ns, "capture timestamp"),
            (base_ready_at_ns, "base-ready timestamp"),
            (policy_epoch, "policy epoch"),
        ):
            _non_negative_int(value, name)
        if captured_at_ns > base_ready_at_ns:
            raise ValueError("capture timestamp must not follow base-ready timestamp")
        if type(segmentation_updated) is not bool:
            raise TypeError("segmentation_updated must be boolean")
        if type(processing_deadline_missed) is not bool:
            raise TypeError("processing_deadline_missed must be boolean")
        if (
            not isinstance(performance_epoch, tuple)
            or len(performance_epoch) != 4
            or any(type(value) is not int or value < 0 for value in performance_epoch)
        ):
            raise ValueError(
                "performance epoch must contain four non-negative integers"
            )
        if type(privacy_slate) is not bool:
            raise TypeError("privacy_slate must be boolean")
        if not isinstance(privacy_reason, str) or len(privacy_reason) > 64:
            raise ValueError("privacy reason must be a bounded string")
        if privacy_slate != bool(privacy_reason):
            raise ValueError("privacy slate and reason must be declared together")
        if privacy_reason and any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for char in privacy_reason
        ):
            raise ValueError("privacy reason must be a lowercase reason code")
        if privacy_slate and remote_proof is not None:
            raise ValueError("privacy slate cannot carry renderer proof")
        _non_negative_int(
            resolved_remote_raw_epoch,
            "resolved remote raw epoch",
        )
        if remote_proof is not None and (
            resolved_remote_raw_epoch not in {0, remote_proof.raw_epoch}
        ):
            raise ValueError("resolved remote epoch must match renderer proof")
        if matte_bundle_sequence is not None:
            _non_negative_int(matte_bundle_sequence, "matte bundle sequence")
        frozen_status = _freeze_scalars(status)
        if not isinstance(frozen_status, Mapping):  # pragma: no cover - type invariant
            raise TypeError("safe base status must be a mapping")
        return cls(
            pixels=frame,
            base_id=base_id,
            config_version=config_version,
            capture_sequence=capture_sequence,
            captured_at_ns=captured_at_ns,
            base_ready_at_ns=base_ready_at_ns,
            policy_epoch=policy_epoch,
            segmentation_updated=segmentation_updated,
            processing_deadline_missed=processing_deadline_missed,
            status=frozen_status,
            performance_epoch=performance_epoch,
            remote_proof=remote_proof,
            resolved_remote_raw_epoch=resolved_remote_raw_epoch,
            privacy_slate=privacy_slate,
            privacy_reason=privacy_reason,
            matte_bundle_sequence=matte_bundle_sequence,
        )


@dataclass(frozen=True)
class OutputPublisherSnapshot:
    state: PublisherState
    send_count: int
    adopted_base_count: int
    reuse_count: int
    handoff_overwrite_count: int
    schedule_skipped_slots: int
    privacy_slate_send_count: int
    pacing_wait_events: int
    pending: bool
    current_base_id: int | None
    error_type: str


@dataclass(frozen=True)
class OutputSendReceipt:
    pixels: np.ndarray
    base: SafeBaseFrame | None
    timing: OutputSendTiming
    base_updated: bool
    exact_final_repeat: bool
    privacy_slate: bool
    privacy_reason: str
    schedule_lateness_ms: float
    schedule_skipped_slots: int
    application_pacing_wait_ms: float
    application_pacing_events: int
    snapshot: OutputPublisherSnapshot


class OutputPublisher:
    """Own one output backend and submit the newest safe base at target cadence."""

    def __init__(
        self,
        output_factory: Callable[[], VideoOutput],
        *,
        width: int,
        height: int,
        target_fps: int,
        slate_pixels: np.ndarray,
        initial_policy: PublicationPolicy,
        on_send: Callable[[OutputSendReceipt], None],
        on_pacing_complete: Callable[[OutputSendTiming], None] | None = None,
        on_failure: Callable[[BaseException], None] | None = None,
        on_overwrite: Callable[[int], None] | None = None,
        on_release: Callable[[int], None] | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not callable(output_factory):
            raise TypeError("output_factory must be callable")
        if not callable(on_send):
            raise TypeError("on_send must be callable")
        if on_failure is not None and not callable(on_failure):
            raise TypeError("on_failure must be callable")
        if on_pacing_complete is not None and not callable(on_pacing_complete):
            raise TypeError("on_pacing_complete must be callable")
        if on_overwrite is not None and not callable(on_overwrite):
            raise TypeError("on_overwrite must be callable")
        if on_release is not None and not callable(on_release):
            raise TypeError("on_release must be callable")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self.width = _positive_int(width, "output width")
        self.height = _positive_int(height, "output height")
        self.target_fps = _positive_int(target_fps, "target FPS")
        interval = round(1_000_000_000 / target_fps)
        self._interval_ns = max(1, interval)
        slate = validate_bgr_frame(
            slate_pixels,
            name="publisher privacy slate",
            require_contiguous=True,
        )
        if slate.shape != (height, width, 3):
            raise ValueError("publisher privacy slate does not match output canvas")
        self._slate = slate.copy()
        self._slate.setflags(write=False)
        self._slate_digest = hashlib.blake2b(
            self._slate,
            digest_size=32,
            person=b"custback-out-v1",
        ).digest()
        self._output_factory = output_factory
        self._on_send = on_send
        self._on_pacing_complete = on_pacing_complete
        self._on_failure = on_failure
        self._on_overwrite = on_overwrite
        self._on_release = on_release
        self._clock_ns = clock_ns
        self._condition = threading.Condition()
        self._send_gate = threading.Lock()
        self._stop = False
        self._state: PublisherState = "starting"
        self._policy = initial_policy
        self._force_slate = initial_policy.mode == "remote"
        self._privacy_reason = "awaiting-renderer" if self._force_slate else ""
        self._pending: SafeBaseFrame | None = None
        self._current: SafeBaseFrame | None = None
        self._last_sent_digest: bytes | None = None
        self._output: VideoOutput | None = None
        self._error: BaseException | None = None
        self._send_count = 0
        self._adopted_base_count = 0
        self._reuse_count = 0
        self._handoff_overwrite_count = 0
        self._schedule_skipped_slots = 0
        self._privacy_slate_send_count = 0
        self._pacing_wait_events = 0
        self._opened = threading.Event()
        self._first_send = threading.Event()
        self._followup_send = threading.Event()
        self._terminated = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="custback-output-publisher",
            daemon=True,
        )
        self._started = False
        self._last_submitted_base_id: int | None = None
        self._last_adopted_base_id: int | None = None

    def _notify_released(self, *base_ids: int | None) -> None:
        """Notify once for every distinct envelope leaving publisher custody."""

        if self._on_release is None:
            return
        for base_id in dict.fromkeys(value for value in base_ids if value is not None):
            self._on_release(base_id)

    @property
    def error(self) -> BaseException | None:
        with self._condition:
            return self._error

    @property
    def output(self) -> VideoOutput | None:
        with self._condition:
            return self._output

    def start(self) -> None:
        with self._condition:
            if self._started or self._terminated.is_set():
                raise OutputPublisherError("output publisher cannot be restarted")
            self._started = True
            self._thread.start()

    def wait_opened(self, timeout: float) -> VideoOutput:
        if not self._opened.wait(timeout):
            raise OutputPublisherTimeout(
                "output publisher did not open before deadline"
            )
        error = self.error
        if error is not None:
            raise OutputPublisherError("output publisher failed to open") from error
        output = self.output
        if output is None:  # pragma: no cover - event ordering invariant
            raise OutputPublisherError("output publisher reported no output")
        return output

    def wait_ready(self, timeout: float) -> None:
        if not self._followup_send.wait(timeout):
            error = self.error
            if error is not None:
                raise OutputPublisherError(
                    "output publisher failed during startup"
                ) from error
            raise OutputPublisherTimeout(
                "output publisher did not complete two sends before deadline"
            )
        error = self.error
        if error is not None:
            raise OutputPublisherError(
                "output publisher failed during startup"
            ) from error

    def _eligible_locked(self, base: SafeBaseFrame, now_ns: int) -> bool:
        if base.policy_epoch != self._policy.epoch:
            return False
        if self._policy.mode == "local":
            return base.remote_proof is None
        if base.privacy_slate:
            return True
        proof = base.remote_proof
        return bool(
            proof is not None
            and proof.raw_epoch == self._policy.raw_epoch
            and proof.renderer_session == self._policy.renderer_session
            and now_ns <= proof.valid_until_ns
        )

    def submit(self, base: SafeBaseFrame) -> bool:
        if not isinstance(base, SafeBaseFrame):
            raise TypeError("publisher base must be SafeBaseFrame")
        if base.pixels.shape != (self.height, self.width, 3):
            raise ValueError("publisher base does not match output canvas")
        overwritten_base_id: int | None = None
        with self._condition:
            if self._stop or self._state in {"stopping", "stopped", "failed"}:
                return False
            if not self._eligible_locked(base, self._clock_ns()):
                return False
            if (
                self._last_submitted_base_id is not None
                and base.base_id <= self._last_submitted_base_id
            ):
                raise ValueError("safe base ids must increase")
            if self._pending is not None:
                self._handoff_overwrite_count += 1
                overwritten_base_id = self._pending.base_id
            self._pending = base
            self._last_submitted_base_id = base.base_id
            # A processed privacy rejection is itself an eligible safe base.
            # Keep the lane fail-closed while retaining its bounded reason
            # until either that base is adopted or a valid candidate clears
            # the latch. This prevents a faster producer from erasing a
            # security transition before the next paced tick.
            self._force_slate = base.privacy_slate
            self._privacy_reason = base.privacy_reason if base.privacy_slate else ""
            self._condition.notify_all()
        if overwritten_base_id is not None and self._on_overwrite is not None:
            self._on_overwrite(overwritten_base_id)
        self._notify_released(overwritten_base_id)
        return True

    def fence_to_slate(
        self,
        policy: PublicationPolicy,
        *,
        reason: str,
        timeout: float = 0.25,
    ) -> None:
        if not isinstance(policy, PublicationPolicy):
            raise TypeError("publisher policy must be PublicationPolicy")
        if not isinstance(reason, str) or not reason or len(reason) > 64:
            raise ValueError("privacy reason must be a bounded non-empty string")
        if any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in reason):
            raise ValueError("privacy reason must be a lowercase reason code")
        if not self._send_gate.acquire(timeout=max(0.0, timeout)):
            raise OutputPublisherTimeout("output publisher privacy fence timed out")
        discarded_base_id: int | None = None
        released_current_base_id: int | None = None
        try:
            with self._condition:
                if self._error is not None:
                    raise OutputPublisherError(
                        "output publisher has failed"
                    ) from self._error
                previous_reason = self._privacy_reason
                self._policy = policy
                self._force_slate = True
                self._privacy_reason = (
                    previous_reason
                    if previous_reason.startswith("privacy-")
                    and reason == "awaiting-renderer"
                    else reason
                )
                if self._pending is not None:
                    discarded_base_id = self._pending.base_id
                if self._current is not None:
                    released_current_base_id = self._current.base_id
                self._pending = None
                self._current = None
                self._condition.notify_all()
        finally:
            self._send_gate.release()
        if discarded_base_id is not None and self._on_overwrite is not None:
            # The publisher envelope is pixels-only, but the producer may hold
            # a diagnostic raster keyed by the scalar base id until first
            # adoption. A privacy fence must retire that unsent evidence just
            # as a normal depth-one handoff overwrite does.
            self._on_overwrite(discarded_base_id)
        self._notify_released(discarded_base_id, released_current_base_id)

    def snapshot(self) -> OutputPublisherSnapshot:
        with self._condition:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> OutputPublisherSnapshot:
        return OutputPublisherSnapshot(
            state=self._state,
            send_count=self._send_count,
            adopted_base_count=self._adopted_base_count,
            reuse_count=self._reuse_count,
            handoff_overwrite_count=self._handoff_overwrite_count,
            schedule_skipped_slots=self._schedule_skipped_slots,
            privacy_slate_send_count=self._privacy_slate_send_count,
            pacing_wait_events=self._pacing_wait_events,
            pending=self._pending is not None,
            current_base_id=(None if self._current is None else self._current.base_id),
            error_type="" if self._error is None else type(self._error).__name__,
        )

    def close(self, timeout: float = 5.0) -> None:
        self.request_stop()
        if not self._started:
            with self._condition:
                self._state = "stopped"
            return
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            raise OutputPublisherTimeout(
                "output publisher did not stop before deadline"
            )
        error = self.error
        if error is not None:
            raise OutputPublisherError("output publisher failed") from error

    def request_stop(self) -> None:
        """Request an interruptible stop without joining the calling thread."""

        discarded_base_id: int | None = None
        with self._condition:
            if self._state not in {"stopped", "failed"}:
                self._state = "stopping"
            self._stop = True
            if self._pending is not None:
                discarded_base_id = self._pending.base_id
            self._pending = None
            self._condition.notify_all()
        if discarded_base_id is not None and self._on_overwrite is not None:
            self._on_overwrite(discarded_base_id)
        self._notify_released(discarded_base_id)

    def wait_terminated(self, timeout: float) -> bool:
        return self._terminated.wait(timeout)

    @property
    def is_alive(self) -> bool:
        """Return whether the sink-owning worker still retains run resources."""

        return self._thread.is_alive()

    def _wait_for_first_base(self) -> SafeBaseFrame | None:
        with self._condition:
            while not self._stop and self._pending is None:
                self._condition.wait()
            if self._stop:
                return None
            base = self._pending
            self._pending = None
            self._current = base
            return base

    def _wait_until(self, deadline_ns: int) -> bool:
        with self._condition:
            while not self._stop:
                remaining_ns = deadline_ns - self._clock_ns()
                if remaining_ns <= 0:
                    return True
                self._condition.wait(remaining_ns / 1_000_000_000.0)
            return False

    def _select_for_send(
        self, now_ns: int
    ) -> tuple[np.ndarray, SafeBaseFrame | None, bool, str, tuple[int, ...]]:
        with self._condition:
            released: list[int] = []
            if self._force_slate:
                candidate = (
                    self._pending if self._pending is not None else self._current
                )
                if (
                    candidate is not None
                    and candidate.privacy_slate
                    and self._eligible_locked(candidate, now_ns)
                ):
                    if self._pending is candidate:
                        self._pending = None
                    if self._current is not None and self._current is not candidate:
                        released.append(self._current.base_id)
                    self._current = candidate
                    return (
                        candidate.pixels,
                        candidate,
                        True,
                        candidate.privacy_reason,
                        tuple(released),
                    )
                return self._slate, None, True, self._privacy_reason, ()
            if self._pending is not None:
                candidate = self._pending
                self._pending = None
                if self._eligible_locked(candidate, now_ns):
                    if self._current is not None and self._current is not candidate:
                        released.append(self._current.base_id)
                    self._current = candidate
                    return (
                        candidate.pixels,
                        candidate,
                        candidate.privacy_slate,
                        candidate.privacy_reason,
                        tuple(released),
                    )
                released.append(candidate.base_id)
            current = self._current
            if current is not None and self._eligible_locked(current, now_ns):
                return (
                    current.pixels,
                    current,
                    current.privacy_slate,
                    current.privacy_reason,
                    tuple(released),
                )
            if current is not None:
                released.append(current.base_id)
                self._current = None
            reason = (
                "remote-proof-expired"
                if self._policy.mode == "remote"
                else "no-safe-base"
            )
            return self._slate, None, True, reason, tuple(released)

    def _record_receipt(
        self,
        *,
        pixels: np.ndarray,
        base: SafeBaseFrame | None,
        adopted: bool,
        privacy_slate: bool,
        privacy_reason: str,
        timing: OutputSendTiming,
        schedule_lateness_ms: float,
        skipped_slots: int,
        application_pacing_wait_ms: float,
        application_pacing_events: int,
    ) -> None:
        pixels_digest = self._slate_digest if base is None else base.pixel_digest
        exact_repeat = (
            self._last_sent_digest is not None
            and pixels_digest == self._last_sent_digest
        )
        with self._condition:
            self._send_count += 1
            if adopted:
                self._adopted_base_count += 1
            else:
                self._reuse_count += 1
            if privacy_slate:
                self._privacy_slate_send_count += 1
            if timing.pacing_wait_ms > 0.05:
                self._pacing_wait_events += 1
            self._schedule_skipped_slots += skipped_slots
            snapshot = self._snapshot_locked()
        receipt = OutputSendReceipt(
            pixels=pixels,
            base=base,
            timing=timing,
            base_updated=adopted,
            exact_final_repeat=exact_repeat,
            privacy_slate=privacy_slate,
            privacy_reason=privacy_reason,
            schedule_lateness_ms=schedule_lateness_ms,
            schedule_skipped_slots=skipped_slots,
            application_pacing_wait_ms=application_pacing_wait_ms,
            application_pacing_events=application_pacing_events,
            snapshot=snapshot,
        )
        self._on_send(receipt)
        self._last_sent_digest = pixels_digest
        self._first_send.set()
        if snapshot.send_count >= 2:
            self._followup_send.set()

    def _send_once(
        self,
        output: VideoOutput,
        *,
        deadline_ns: int,
        skipped_slots: int,
        application_pacing_wait_ms: float = 0.0,
        application_pacing_events: int = 0,
    ) -> OutputSendTiming:
        with self._send_gate:
            now_ns = self._clock_ns()
            (
                pixels,
                base,
                privacy_slate,
                privacy_reason,
                released_base_ids,
            ) = self._select_for_send(now_ns)
            self._notify_released(*released_base_ids)
            adopted = bool(
                base is not None and base.base_id != self._last_adopted_base_id
            )
            accepted = False
            accepted_timing: OutputSendTiming | None = None

            def record_accepted(
                timing: OutputSendTiming,
                *,
                before_pacing: bool = True,
            ) -> None:
                nonlocal accepted, accepted_timing
                if accepted:
                    raise OutputPublisherError(
                        "output reported duplicate frame acceptance"
                    )
                if not isinstance(timing, OutputSendTiming):
                    raise OutputPublisherError(
                        "output returned invalid acceptance timing"
                    )
                if before_pacing and timing.completed_at_ns != timing.submitted_at_ns:
                    raise OutputPublisherError(
                        "acceptance timing cannot include sink pacing"
                    )
                if before_pacing and timing.pacing_wait_ms != 0.0:
                    raise OutputPublisherError(
                        "acceptance timing pacing wait must be zero"
                    )
                # A sink may itself spend several target intervals inside
                # submission.  Those elapsed wall-clock slots are just as
                # unavailable to consumers as slots skipped before entering
                # the sink, so account for them at the acceptance boundary.
                # The following deadline is also kept at least one complete
                # interval after acceptance below, preventing a catch-up
                # burst after this slow call.
                sink_elapsed_slots = max(
                    0,
                    (timing.submitted_at_ns - deadline_ns) // self._interval_ns,
                )
                total_skipped_slots = skipped_slots + sink_elapsed_slots
                if adopted and base is not None:
                    self._last_adopted_base_id = base.base_id
                self._record_receipt(
                    pixels=pixels,
                    base=base,
                    adopted=adopted,
                    privacy_slate=privacy_slate,
                    privacy_reason=privacy_reason,
                    timing=timing,
                    schedule_lateness_ms=max(
                        0.0,
                        (timing.submitted_at_ns - deadline_ns) / 1_000_000.0,
                    ),
                    skipped_slots=total_skipped_slots,
                    application_pacing_wait_ms=application_pacing_wait_ms,
                    application_pacing_events=application_pacing_events,
                )
                accepted = True
                accepted_timing = timing

            acceptance_sender = getattr(
                output,
                "send_with_acceptance_timing",
                None,
            )
            if bool(getattr(output, "paces", False)) and callable(acceptance_sender):
                timing = acceptance_sender(pixels, record_accepted)
                if not isinstance(timing, OutputSendTiming):
                    raise OutputPublisherError("output returned invalid send timing")
                if not accepted:
                    raise OutputPublisherError(
                        "paced output returned without accepting the frame"
                    )
                assert accepted_timing is not None
                if timing.submitted_at_ns != accepted_timing.submitted_at_ns:
                    raise OutputPublisherError(
                        "paced output changed its accepted submission timestamp"
                    )
                with self._condition:
                    if timing.pacing_wait_ms > 0.05:
                        self._pacing_wait_events += 1
                if self._on_pacing_complete is not None:
                    self._on_pacing_complete(timing)
                return timing
            sender = getattr(output, "send_with_timing", None)
            if callable(sender):
                timing = sender(pixels)
            else:  # pragma: no cover - VideoOutput compatibility contract
                started = self._clock_ns()
                output.send(pixels)
                completed = max(started, self._clock_ns())
                timing = OutputSendTiming(
                    submitted_at_ns=completed,
                    completed_at_ns=completed,
                    submission_ms=(completed - started) / 1_000_000.0,
                    pacing_wait_ms=0.0,
                )
            if not isinstance(timing, OutputSendTiming):
                raise OutputPublisherError("output returned invalid send timing")
            # Legacy/third-party paced sinks expose only a completion
            # boundary.  Their hub publication necessarily follows the wait,
            # but remains supported for one compatibility release.  The
            # production pyvirtualcam backend uses the acceptance-aware seam
            # above and publishes before pacing.
            record_accepted(timing, before_pacing=False)
            return timing

    def _run(self) -> None:
        output: VideoOutput | None = None
        try:
            output = self._output_factory()
            if not isinstance(output, VideoOutput):
                # Test doubles commonly duck-type VideoOutput. Require only its
                # actual lifecycle boundary rather than nominal inheritance.
                if not callable(getattr(output, "send", None)):
                    raise TypeError("output factory returned no send-capable output")
            with self._condition:
                self._output = output
                self._state = "running"
                self._opened.set()
            first = self._wait_for_first_base()
            if first is None:
                return
            first_deadline = self._clock_ns()
            timing = self._send_once(
                output,
                deadline_ns=first_deadline,
                skipped_slots=0,
            )
            next_deadline_ns = timing.submitted_at_ns + self._interval_ns
            while True:
                with self._condition:
                    if self._stop:
                        break
                application_wait_started_ns = self._clock_ns()
                application_pacing_events = 0
                application_pacing_wait_ms = 0.0
                if not bool(getattr(output, "paces", False)):
                    if not self._wait_until(next_deadline_ns):
                        break
                    application_wait_completed_ns = self._clock_ns()
                    application_pacing_wait_ms = max(
                        0.0,
                        (application_wait_completed_ns - application_wait_started_ns)
                        / 1_000_000.0,
                    )
                    if application_pacing_wait_ms > 0.05:
                        application_pacing_events = 1
                now_ns = self._clock_ns()
                next_deadline_ns, skipped_slots = scheduled_slot(
                    next_deadline_ns,
                    now_ns,
                    self._interval_ns,
                )
                timing = self._send_once(
                    output,
                    deadline_ns=next_deadline_ns,
                    skipped_slots=skipped_slots,
                    application_pacing_wait_ms=application_pacing_wait_ms,
                    application_pacing_events=application_pacing_events,
                )
                # Never schedule a catch-up send less than one full target
                # interval after the frame the sink actually accepted. A late
                # wake may shift the monotonic phase, but it cannot create a
                # micro-burst to reclaim an already missed wall-clock slot.
                sink_elapsed_slots = max(
                    0,
                    (timing.submitted_at_ns - next_deadline_ns) // self._interval_ns,
                )
                next_deadline_ns = max(
                    next_deadline_ns + ((sink_elapsed_slots + 1) * self._interval_ns),
                    timing.submitted_at_ns + self._interval_ns,
                )
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._state = "failed"
                self._stop = True
                self._opened.set()
                self._condition.notify_all()
            self._first_send.set()
            self._followup_send.set()
            if self._on_failure is not None:
                try:
                    self._on_failure(exc)
                except Exception:
                    pass
        finally:
            if output is not None:
                try:
                    output.close()
                except BaseException as exc:
                    with self._condition:
                        if self._error is None:
                            self._error = exc
                            self._state = "failed"
            released_pending_base_id: int | None = None
            released_current_base_id: int | None = None
            with self._condition:
                if self._pending is not None:
                    released_pending_base_id = self._pending.base_id
                if self._current is not None:
                    released_current_base_id = self._current.base_id
                self._pending = None
                self._current = None
                if self._state != "failed":
                    self._state = "stopped"
                self._output = None
                self._opened.set()
                self._condition.notify_all()
            try:
                self._notify_released(
                    released_pending_base_id,
                    released_current_base_id,
                )
            finally:
                self._terminated.set()


__all__ = [
    "OutputPublisher",
    "OutputPublisherError",
    "OutputPublisherSnapshot",
    "OutputPublisherTimeout",
    "OutputSendReceipt",
    "PublicationPolicy",
    "RemoteOutputProof",
    "SafeBaseFrame",
    "scheduled_slot",
    "thaw_status",
]
