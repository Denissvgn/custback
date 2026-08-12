"""Async latest-only API frame subscriptions and admission control."""

import asyncio
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from custback.api.streaming import (
    ConnectionLimiter,
    JpegBroadcaster,
    LeasedStreamingResponse,
    StreamLifecycleRegistry,
)
from custback.hub import FrameHub


def test_jpeg_subscriptions_work_while_default_executor_is_saturated():
    hub = FrameHub()
    calls = []
    release = threading.Event()
    started = threading.Event()

    def block_default_executor():
        started.set()
        release.wait()

    def encode(frame):
        calls.append(int(frame[0, 0, 0]))
        return bytes([calls[-1]])

    async def scenario():
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        blocked = loop.run_in_executor(None, block_default_executor)

        async def heartbeat():
            # Restricted selectors cannot be woken by worker callback fds.
            while True:
                await asyncio.sleep(0.01)

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            deadline = loop.time() + 1.0
            while not started.is_set():
                assert loop.time() < deadline
                await asyncio.sleep(0.005)

            broadcaster = JpegBroadcaster(hub.output, encode)
            async with (
                broadcaster.subscribe() as first,
                broadcaster.subscribe() as second,
            ):
                hub.publish_output(np.full((1, 1, 3), 17, np.uint8))
                results = await asyncio.wait_for(
                    asyncio.gather(first.get(), second.get()),
                    1.0,
                )
                assert results == [(b"\x11", 1), (b"\x11", 1)]
                assert calls == [17]
        finally:
            release.set()
            await blocked
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_jpeg_broadcaster_coalesces_newer_frames_while_encoder_is_busy():
    hub = FrameHub()
    calls = []
    first_started = threading.Event()
    release_first = threading.Event()

    def encode(frame):
        value = int(frame[0, 0, 0])
        calls.append(value)
        if value == 1:
            first_started.set()
            assert release_first.wait(2.0)
        return bytes([value])

    async def scenario():
        broadcaster = JpegBroadcaster(hub.output, encode)
        first = asyncio.create_task(
            broadcaster._encode(np.full((1, 1, 3), 1, np.uint8), 1)
        )
        deadline = asyncio.get_running_loop().time() + 1.0
        while not first_started.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.005)

        newer = []
        superseded = []
        for value in range(2, 33):
            frame = np.full((1, 1, 3), value, np.uint8)
            if value < 32:
                superseded.append(weakref.ref(frame))
            newer.append(asyncio.create_task(broadcaster._encode(frame, value)))
            del frame
            # Ensure each distinct sequence reaches the scheduler before the
            # next one replaces it.
            await asyncio.sleep(0)
            assert broadcaster._pending_seq == value

        assert calls == [1]
        gc.collect()
        assert all(reference() is None for reference in superseded)
        release_first.set()
        deadline = asyncio.get_running_loop().time() + 1.0
        while not all(task.done() for task in (first, *newer)):
            # Restricted selectors cannot always be woken by worker callback
            # file descriptors, so keep the loop polling during this test.
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.005)
        first_result, *newer_results = await asyncio.gather(first, *newer)
        assert first_result == (b"\x01", 1)
        assert newer_results == [(b"\x20", 32)] * len(newer)
        assert calls == [1, 32]

    asyncio.run(scenario())


def test_cancelled_jpeg_waiter_does_not_cancel_shared_encode():
    hub = FrameHub()
    started = threading.Event()
    release = threading.Event()

    def encode(frame):
        started.set()
        assert release.wait(2.0)
        return b"encoded"

    async def scenario():
        broadcaster = JpegBroadcaster(hub.output, encode)
        frame = np.zeros((1, 1, 3), np.uint8)
        cancelled = asyncio.create_task(broadcaster._encode(frame, 1))
        survivor = asyncio.create_task(broadcaster._encode(frame, 1))

        deadline = asyncio.get_running_loop().time() + 1.0
        while not started.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.005)
        cancelled.cancel()
        try:
            await cancelled
        except asyncio.CancelledError:
            pass
        else:  # pragma: no cover - cancellation must propagate to the caller
            raise AssertionError("cancelled JPEG waiter unexpectedly completed")

        release.set()
        deadline = asyncio.get_running_loop().time() + 1.0
        while not survivor.done():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.005)
        assert await survivor == (b"encoded", 1)

    asyncio.run(scenario())


def test_async_subscription_drops_backlog_and_delivers_latest_frame():
    hub = FrameHub()

    async def scenario():
        async with hub.output.subscribe() as subscription:
            for value in (1, 2, 3):
                hub.publish_output(np.full((1, 1, 3), value, np.uint8))
            frame, seq = await subscription.get(-1, 0.1)
            assert seq == 3
            assert frame is not None
            assert int(frame[0, 0, 0]) == 3

    asyncio.run(scenario())


def test_connection_limiter_caps_and_releases_idempotently():
    limiter = ConnectionLimiter(2)
    first = limiter.try_acquire()
    second = limiter.try_acquire()
    assert first is not None and second is not None
    assert limiter.active == 2
    assert limiter.try_acquire() is None

    first.release()
    first.release()
    assert limiter.active == 1
    replacement = limiter.try_acquire()
    assert replacement is not None
    second.release()
    replacement.release()
    assert limiter.active == 0


def test_stream_lease_releases_when_headers_fail_before_body_iteration():
    limiter = ConnectionLimiter(1)
    lease = limiter.try_acquire()
    assert lease is not None
    body_entered = False

    async def body():
        nonlocal body_entered
        body_entered = True
        yield b"never sent"

    async def scenario():
        response = LeasedStreamingResponse(body(), lease=lease)
        blocked_receive = asyncio.Event()

        async def receive():
            await blocked_receive.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            assert message["type"] == "http.response.start"
            raise RuntimeError("header transport failed")

        caught = None
        try:
            await response(
                {"type": "http", "asgi": {"spec_version": "2.3"}},
                receive,
                send,
            )
        except BaseException as exc:  # Starlette may wrap this in ExceptionGroup.
            caught = exc
        assert caught is not None

    asyncio.run(scenario())
    assert body_entered is False
    assert limiter.active == 0


def test_stream_lifecycle_shutdown_is_thread_safe_bounded_and_idempotent():
    registry = StreamLifecycleRegistry()

    async def scenario():
        loop = asyncio.get_running_loop()
        registry.bind(loop)
        started = {
            kind: asyncio.Event()
            for kind in (
                "core_mjpeg",
                "management_websocket",
                "renderer_websocket",
                "avatar_proxy",
            )
        }
        closed: list[str] = []

        async def stream(kind):
            async def close():
                closed.append(kind)

            registration = registry.register(kind, close=close)
            assert registration is not None
            started[kind].set()
            try:
                await asyncio.Event().wait()
            finally:
                registration.release()

        tasks = [asyncio.create_task(stream(kind)) for kind in started]
        await asyncio.gather(*(event.wait() for event in started.values()))
        snapshot = registry.snapshot()
        assert snapshot["active"] == 4
        assert snapshot["by_kind"] == {
            **{kind: 1 for kind in started},
            "avatar_mjpeg": 0,
        }

        requested = []

        def request_from_coordinator_thread():
            requested.append(registry.request_shutdown(0.5))

        thread = threading.Thread(target=request_from_coordinator_thread)
        thread.start()
        thread.join()
        assert requested[0] is not None
        result = await asyncio.wrap_future(requested[0])
        await asyncio.gather(*tasks, return_exceptions=True)

        assert result.requested == 4
        assert result.remaining == 0
        assert result.close_timeouts == 0
        assert result.task_timeouts == 0
        assert sorted(closed) == sorted(started)
        assert registry.snapshot()["active"] == 0
        assert registry.register("core_mjpeg") is None

        repeated = await registry.shutdown(0.5)
        assert repeated.requested == 0
        assert repeated.remaining == 0
        registry.unbind(loop)

    asyncio.run(scenario())


def test_stream_lifecycle_does_not_join_a_cancellation_resistant_closer():
    registry = StreamLifecycleRegistry()

    async def scenario():
        started = asyncio.Event()

        async def resistant_close():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.2)

        async def stream():
            registration = registry.register("core_mjpeg", close=resistant_close)
            assert registration is not None
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                registration.release()

        stream_task = asyncio.create_task(stream())
        await started.wait()
        before = asyncio.get_running_loop().time()
        result = await registry.shutdown(0.02)
        elapsed = asyncio.get_running_loop().time() - before
        await asyncio.gather(stream_task, return_exceptions=True)
        assert elapsed < 0.1
        assert result.close_timeouts == 1
        await asyncio.sleep(0.21)

    asyncio.run(scenario())


def test_stream_lifecycle_rejects_synchronous_close_callbacks():
    registry = StreamLifecycleRegistry()

    async def scenario():
        with pytest.raises(TypeError, match="close callback must be async"):
            registry.register("core_mjpeg", close=lambda: None)  # type: ignore[arg-type]

    asyncio.run(scenario())
