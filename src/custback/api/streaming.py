"""Latest-only asynchronous frame delivery for authenticated API streams."""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
from collections.abc import Callable
from typing import Any

from starlette.responses import StreamingResponse

from ..hub import _AsyncSlotSubscription, _Slot


class _DaemonExecutor:
    """Tiny executor whose workers cannot hold interpreter shutdown open."""

    def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
        self._jobs: queue.SimpleQueue[
            tuple[
                concurrent.futures.Future[bytes],
                Callable[[Any], bytes],
                Any,
            ]
        ] = queue.SimpleQueue()
        self._threads = tuple(
            threading.Thread(
                target=self._worker,
                name=f"{thread_name_prefix}_{index}",
                daemon=True,
            )
            for index in range(max_workers)
        )
        for thread in self._threads:
            thread.start()

    def submit(
        self, function: Callable[[Any], bytes], argument: Any
    ) -> concurrent.futures.Future[bytes]:
        future: concurrent.futures.Future[bytes] = concurrent.futures.Future()
        self._jobs.put((future, function, argument))
        return future

    def _worker(self) -> None:
        while True:
            future, function, argument = self._jobs.get()
            try:
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = function(argument)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
                    del result
            finally:
                # A worker blocked on the next queue read must not retain the
                # previous broadcaster, frame, or completed Future.
                del future, function, argument


_JPEG_EXECUTOR = _DaemonExecutor(
    max_workers=2,
    thread_name_prefix="custback-stream-jpeg",
)


class JpegBroadcaster:
    """Share JPEG work while retaining only the newest pending frame."""

    def __init__(self, slot: _Slot, encoder: Callable[[Any], bytes]) -> None:
        self._slot = slot
        self._encoder = encoder
        self._lock = threading.Lock()
        self._latest_seq = -1
        self._latest_data: bytes | None = None
        self._latest_error: BaseException | None = None
        self._running_seq = -1
        self._running_job: concurrent.futures.Future[bytes] | None = None
        self._running_delivery: concurrent.futures.Future[tuple[bytes, int]] | None = (
            None
        )
        self._pending_seq = -1
        self._pending_frame: Any | None = None
        self._pending_delivery: concurrent.futures.Future[tuple[bytes, int]] | None = (
            None
        )

    def subscribe(self) -> "JpegSubscription":
        return JpegSubscription(self, self._slot.subscribe())

    def _start_locked(
        self,
        frame: Any,
        seq: int,
        delivery: concurrent.futures.Future[tuple[bytes, int]],
    ) -> concurrent.futures.Future[bytes]:
        job = _JPEG_EXECUTOR.submit(self._encoder, frame)
        self._running_seq = seq
        self._running_job = job
        self._running_delivery = delivery
        return job

    def _completed(self, future: concurrent.futures.Future[bytes]) -> None:
        try:
            data = future.result()
            error = None
        except BaseException as exc:  # retain the newest terminal result
            data = None
            error = exc

        next_job = None
        with self._lock:
            # Only the job recorded as running can own these delivery fields.
            # This guard is defensive against a future executor implementation
            # invoking a completion callback more than once.
            if future is not self._running_job:  # pragma: no cover
                return
            seq = self._running_seq
            delivery = self._running_delivery
            if delivery is None:  # pragma: no cover - invariant guard
                return

            if seq > self._latest_seq:
                self._latest_seq = seq
                self._latest_data = data
                self._latest_error = error

            self._running_seq = -1
            self._running_job = None
            self._running_delivery = None

            if self._pending_delivery is not None:
                pending_delivery = self._pending_delivery
                pending_frame = self._pending_frame
                pending_seq = self._pending_seq
                self._pending_delivery = None
                self._pending_frame = None
                self._pending_seq = -1
                next_job = self._start_locked(
                    pending_frame,
                    pending_seq,
                    pending_delivery,
                )

        # Wake clients before registering the next callback. If the next
        # encoder is extremely fast, add_done_callback may run inline.
        if error is not None:
            delivery.set_exception(error)
        elif data is not None:
            delivery.set_result((data, seq))
        else:  # pragma: no cover - Future[bytes] cannot normally produce None
            delivery.set_exception(RuntimeError("JPEG encoder completed without data"))
        if next_job is not None:
            next_job.add_done_callback(self._completed)

    def _cached_locked(self) -> tuple[bytes, int] | None:
        if self._latest_error is not None:
            raise self._latest_error
        if self._latest_data is None:
            return None
        return self._latest_data, self._latest_seq

    async def _encode(self, frame: Any, seq: int) -> tuple[bytes, int]:
        created_job = None
        with self._lock:
            if self._latest_seq >= seq:
                cached = self._cached_locked()
                if cached is not None:
                    return cached

            if self._running_job is None:
                delivery: concurrent.futures.Future[tuple[bytes, int]] = (
                    concurrent.futures.Future()
                )
                created_job = self._start_locked(frame, seq, delivery)
            elif seq <= self._running_seq:
                running_delivery = self._running_delivery
                if running_delivery is None:  # pragma: no cover - invariant guard
                    raise RuntimeError("JPEG encoder delivery is unavailable")
                delivery = running_delivery
            else:
                pending_delivery = self._pending_delivery
                if pending_delivery is None:
                    pending_delivery = concurrent.futures.Future()
                    self._pending_delivery = pending_delivery
                if seq > self._pending_seq:
                    # One pending frame is retained. Every newer sequence
                    # replaces it while all callers share the same delivery.
                    self._pending_seq = seq
                    self._pending_frame = frame
                delivery = pending_delivery

        # A suspended caller must not retain its superseded input frame. The
        # running worker and the single pending slot now own the only frames
        # needed by the scheduler.
        del frame

        if created_job is not None:
            # Register outside ``_lock``: add_done_callback executes inline
            # when a very fast encoder has already completed.
            created_job.add_done_callback(self._completed)

        # The delivery Future is shared by all clients waiting on the running
        # or coalesced-pending encode. Shield it so one disconnected client
        # cannot cancel work still owned by the others.
        return await asyncio.shield(asyncio.wrap_future(delivery))


class JpegSubscription:
    """Latest-only JPEG subscription backed by an async frame subscription."""

    def __init__(
        self,
        broadcaster: JpegBroadcaster,
        frames: _AsyncSlotSubscription,
    ) -> None:
        self._broadcaster = broadcaster
        self._frames = frames

    async def get(
        self, last_seq: int = -1, timeout: float | None = None
    ) -> tuple[bytes | None, int]:
        frame, seq = await self._frames.get(last_seq, timeout)
        if frame is None:
            return None, last_seq
        return await self._broadcaster._encode(frame, seq)

    def close(self) -> None:
        self._frames.close()

    async def __aenter__(self) -> "JpegSubscription":
        return self

    async def __aexit__(self, *_exc_info) -> None:
        self.close()


class ConnectionLease:
    """Idempotent ownership token for one long-lived authenticated stream."""

    def __init__(self, limiter: "ConnectionLimiter") -> None:
        self._limiter = limiter
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._limiter._release()


class ConnectionLimiter:
    """Small thread-safe admission counter shared by all app stream routes."""

    def __init__(self, maximum: int) -> None:
        if maximum < 1:
            raise ValueError("stream connection limit must be positive")
        self.maximum = maximum
        self._active = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    def try_acquire(self) -> ConnectionLease | None:
        with self._lock:
            if self._active >= self.maximum:
                return None
            self._active += 1
        return ConnectionLease(self)

    def _release(self) -> None:
        with self._lock:
            if self._active <= 0:  # pragma: no cover - lease guards this path
                raise RuntimeError("stream connection lease released twice")
            self._active -= 1


class LeasedStreamingResponse(StreamingResponse):
    """Hold a stream lease for the response's complete ASGI lifecycle."""

    def __init__(self, *args, lease: ConnectionLease, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_lease = lease

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # This also covers cancellation while sending response headers,
            # before Starlette ever starts or closes the body iterator.
            self._stream_lease.release()
