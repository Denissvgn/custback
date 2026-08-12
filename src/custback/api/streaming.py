"""Latest-only asynchronous frame delivery for authenticated API streams."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import queue
import threading
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from starlette.responses import StreamingResponse

from ..hub import _AsyncSlotSubscription, _Slot


log = logging.getLogger(__name__)

StreamKind = Literal[
    "core_mjpeg",
    "management_websocket",
    "renderer_websocket",
    "avatar_proxy",
    "avatar_mjpeg",
]
_STREAM_KINDS: tuple[StreamKind, ...] = (
    "core_mjpeg",
    "management_websocket",
    "renderer_websocket",
    "avatar_proxy",
    "avatar_mjpeg",
)
_STREAM_SHUTDOWN_TIMEOUT_S = 1.0
_STREAM_CLOSE_GRACE_S = 0.25


@dataclass(frozen=True)
class StreamShutdownResult:
    """Content-free result of one bounded stream shutdown request."""

    requested: int
    remaining: int
    close_timeouts: int
    task_timeouts: int


@dataclass(frozen=True)
class _StreamEntry:
    kind: StreamKind
    task: asyncio.Task[Any]
    close: Callable[[], Awaitable[None]] | None


class StreamRegistration:
    """Idempotent ownership token for one app-owned long-lived stream."""

    def __init__(self, registry: "StreamLifecycleRegistry", sequence: int) -> None:
        self._registry = registry
        self._sequence = sequence
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._registry._release(self._sequence)

    def set_close(
        self,
        close: Callable[[], Awaitable[None]],
    ) -> None:
        """Install a closer after a streamed upstream resource is acquired."""

        with self._lock:
            if self._released:
                return
        self._registry._set_close(self._sequence, close)


class StreamLifecycleRegistry:
    """App-owned registry that can interrupt every long-lived API stream.

    Registrations and diagnostics are protected by a normal lock so the main
    coordination thread can atomically reject new streams before asking the API
    loop to close existing ones.  No request path, peer address, credential, or
    frame-derived value is retained.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, _StreamEntry] = {}
        self._next_sequence = 1
        self._closing = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_future: (
            concurrent.futures.Future[StreamShutdownResult] | None
        ) = None
        self._shutdown_calls = 0
        self._shutdown_timeouts = 0
        self._last_result = StreamShutdownResult(0, 0, 0, 0)

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the registry to the live application event loop."""

        with self._lock:
            current = self._loop
            if current is not None and current is not loop:
                raise RuntimeError("stream lifecycle registry is already bound")
            self._loop = loop

    def unbind(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            if self._loop is loop:
                self._loop = None

    @property
    def closing(self) -> bool:
        with self._lock:
            return self._closing

    def register(
        self,
        kind: StreamKind,
        *,
        close: Callable[[], Awaitable[None]] | None = None,
    ) -> StreamRegistration | None:
        """Register the current ASGI task, or reject it during shutdown."""

        if kind not in _STREAM_KINDS:
            raise ValueError("unknown API stream kind")
        if close is not None and not inspect.iscoroutinefunction(close):
            raise TypeError("API stream close callback must be async")
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("API stream registration requires an asyncio task")
        with self._lock:
            if self._closing:
                return None
            sequence = self._next_sequence
            self._next_sequence += 1
            self._entries[sequence] = _StreamEntry(kind, task, close)
        task.add_done_callback(lambda _task: self._release(sequence))
        return StreamRegistration(self, sequence)

    def _set_close(
        self,
        sequence: int,
        close: Callable[[], Awaitable[None]],
    ) -> None:
        if not inspect.iscoroutinefunction(close):
            raise TypeError("API stream close callback must be async")
        with self._lock:
            entry = self._entries.get(sequence)
            if entry is not None:
                self._entries[sequence] = _StreamEntry(entry.kind, entry.task, close)

    def _release(self, sequence: int) -> None:
        with self._lock:
            self._entries.pop(sequence, None)

    def snapshot(self) -> dict[str, object]:
        """Return fixed-schema, content-free lifecycle diagnostics."""

        with self._lock:
            counts = {kind: 0 for kind in _STREAM_KINDS}
            for entry in self._entries.values():
                counts[entry.kind] += 1
            return {
                "closing": self._closing,
                "active": len(self._entries),
                "active_tasks": len({entry.task for entry in self._entries.values()}),
                "by_kind": counts,
                "shutdown_calls": self._shutdown_calls,
                "shutdown_timeouts": self._shutdown_timeouts,
                "last_requested": self._last_result.requested,
                "last_remaining": self._last_result.remaining,
                "last_close_timeouts": self._last_result.close_timeouts,
                "last_task_timeouts": self._last_result.task_timeouts,
            }

    def request_shutdown(
        self,
        timeout_s: float = _STREAM_SHUTDOWN_TIMEOUT_S,
    ) -> concurrent.futures.Future[StreamShutdownResult] | None:
        """Reject new streams and schedule bounded cleanup on the API loop."""

        if timeout_s <= 0.0:
            raise ValueError("stream shutdown timeout must be positive")
        with self._lock:
            self._closing = True
            existing = self._shutdown_future
            if existing is not None and not existing.done():
                return existing
            loop = self._loop
            if loop is None or not loop.is_running():
                return None
            future = asyncio.run_coroutine_threadsafe(self.shutdown(timeout_s), loop)
            self._shutdown_future = future
            return future

    async def shutdown(
        self,
        timeout_s: float = _STREAM_SHUTDOWN_TIMEOUT_S,
    ) -> StreamShutdownResult:
        """Close callbacks, then cancel and join registered stream tasks."""

        if timeout_s <= 0.0:
            raise ValueError("stream shutdown timeout must be positive")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        with self._lock:
            self._closing = True
            self._shutdown_calls += 1
            entries = tuple(self._entries.values())

        close_tasks: list[asyncio.Task[None]] = []

        async def invoke_close(entry: _StreamEntry) -> None:
            if entry.close is None:
                return
            await entry.close()

        for entry in entries:
            if entry.close is not None:
                close_tasks.append(asyncio.create_task(invoke_close(entry)))

        close_timeouts = 0
        if close_tasks:
            close_budget = min(
                _STREAM_CLOSE_GRACE_S,
                max(0.0, deadline - loop.time()),
            )
            done_close, pending_close = await asyncio.wait(
                close_tasks,
                timeout=close_budget,
            )
            close_timeouts = len(pending_close)
            for task in pending_close:
                task.cancel()
                # A closer can suppress cancellation. Never await it without a
                # second deadline: retain only a callback that consumes its
                # eventual result while shutdown proceeds to stream-task
                # cancellation within the original total budget.
                task.add_done_callback(_consume_task_result)
            if done_close:
                await asyncio.gather(*done_close, return_exceptions=True)

        current = asyncio.current_task()
        stream_tasks = {
            entry.task
            for entry in entries
            if entry.task is not current and not entry.task.done()
        }
        for task in stream_tasks:
            task.cancel("API stream lifecycle shutdown")

        task_timeouts = 0
        if stream_tasks:
            _done, pending_tasks = await asyncio.wait(
                stream_tasks,
                timeout=max(0.0, deadline - loop.time()),
            )
            task_timeouts = len(pending_tasks)

        with self._lock:
            remaining = len(self._entries)
            result = StreamShutdownResult(
                requested=len(entries),
                remaining=remaining,
                close_timeouts=close_timeouts,
                task_timeouts=task_timeouts,
            )
            self._last_result = result
            if close_timeouts or task_timeouts or remaining:
                self._shutdown_timeouts += 1

        if close_timeouts or task_timeouts or remaining:
            log.warning(
                "API stream shutdown incomplete requested=%d remaining=%d "
                "close_timeouts=%d task_timeouts=%d",
                len(entries),
                remaining,
                close_timeouts,
                task_timeouts,
            )
        return result


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """Consume a detached cleanup task's terminal exception without blocking."""

    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


def stream_lifecycle_for(app: Any) -> StreamLifecycleRegistry:
    """Return the one registry owned by ``app``, creating it if necessary."""

    registry = getattr(app.state, "stream_lifecycle", None)
    if registry is None:
        registry = StreamLifecycleRegistry()
        app.state.stream_lifecycle = registry
    elif not isinstance(registry, StreamLifecycleRegistry):
        raise TypeError("app stream_lifecycle state has an invalid owner")
    return registry


def install_stream_lifecycle(app: Any) -> StreamLifecycleRegistry:
    """Compose bounded registry cleanup into an app lifespan exactly once."""

    registry = stream_lifecycle_for(app)
    if getattr(app.state, "stream_lifecycle_installed", False):
        return registry
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_stream_shutdown(wrapped_app: Any):
        loop = asyncio.get_running_loop()
        registry.bind(loop)
        try:
            async with previous_lifespan(wrapped_app):
                try:
                    yield
                finally:
                    await registry.shutdown()
        finally:
            registry.unbind(loop)

    app.router.lifespan_context = lifespan_with_stream_shutdown
    app.state.stream_lifecycle_installed = True
    return registry


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

    def __init__(
        self,
        *args,
        lease: ConnectionLease,
        registration: StreamRegistration | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._stream_lease = lease
        self._stream_registration = registration

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # This also covers cancellation while sending response headers,
            # before Starlette ever starts or closes the body iterator.
            if self._stream_registration is not None:
                self._stream_registration.release()
            self._stream_lease.release()
