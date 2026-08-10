"""Phase 5 regressions for terminal renderer and Audio2Face ownership."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import custback.avatar.audio2face as audio2face_mod
from custback.avatar.audio2face import Audio2FaceDriver
from custback.avatar.config import Audio2FaceConfig, AvatarConfig, AvatarRuntime
from custback.avatar.drivers import DriverUnavailableError
from custback.avatar.service import AvatarService, _RenderPublication
from custback.remote_protocol import encode_remote_frame


async def _wait_for_thread_event(event: threading.Event, message: str) -> None:
    deadline = asyncio.get_running_loop().time() + 1.0
    while not event.is_set():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(message)
        await asyncio.sleep(0.001)


# LIFE-01: a completed send cannot authorize a queued publication after stop.
def test_LIFE_01_stop_after_completed_send_discards_queued_publication_same_version(
    monkeypatch,
):
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    service.activate_initial()
    version = runtime.read().version
    prior_preview = np.full((8, 8, 3), 19, dtype=np.uint8)
    service.output.put(prior_preview)

    frame = np.full((32, 32, 3), 70, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame)
    assert ok

    lane_blocked = threading.Event()
    lane_release = threading.Event()
    send_returned = threading.Event()
    publication_queued = threading.Event()
    original_submit = service._submit_lane

    def occupy_lane() -> None:
        lane_blocked.set()
        lane_release.wait()

    def tracked_submit(callback, *args):
        future = original_submit(callback, *args)
        if (
            getattr(callback, "__func__", None)
            is AvatarService._publish_render_if_current
        ):
            publication_queued.set()
        return future

    monkeypatch.setattr(service, "_submit_lane", tracked_submit)

    class WebSocket:
        def __init__(self):
            self.first = True
            self.never = asyncio.Event()
            self.sent: list[bytes] = []

        async def recv(self):
            if self.first:
                self.first = False
                return encode_remote_frame("raw-input", 1, jpeg.tobytes())
            await self.never.wait()

        async def send(self, payload):
            self.sent.append(payload)
            service._submit_lane(occupy_lane)
            await _wait_for_thread_event(lane_blocked, "lane blocker never started")
            send_returned.set()

    websocket = WebSocket()

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(service._session(websocket, stop))
        try:
            await _wait_for_thread_event(send_returned, "WebSocket send did not return")
            await _wait_for_thread_event(
                publication_queued, "render publication was not queued"
            )
            assert service.stats_dict()["frames_sent"] == 1

            stop.set()
            deadline = asyncio.get_running_loop().time() + 1.0
            while service._active_session_lease is not None:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("renderer lease was not invalidated")
                await asyncio.sleep(0.001)
            assert not task.done()  # the queued authoritative future is retained

            lane_release.set()
            await asyncio.wait_for(task, 1.0)
        finally:
            lane_release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
        latest, _timestamp = service.output.latest()
        assert latest is prior_preview
        assert runtime.read().version == version
        assert len(websocket.sent) == 1
        stats = service.stats_dict()
        assert stats["frames_sent"] == 1
        assert stats["frames_rendered"] == 0
    finally:
        lane_release.set()
        service.close()


# LIFE-01: epochs, not component versions, authorize local side effects.
def test_LIFE_01_stale_epoch_cannot_publish_into_new_same_version_session():
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    service.activate_initial()
    prior_preview = np.full((8, 8, 3), 23, dtype=np.uint8)
    service.output.put(prior_preview)

    first = service._begin_render_session()
    publication = _RenderPublication(
        frame=np.full((8, 8, 3), 91, dtype=np.uint8),
        version=runtime.read().version,
        driver_ms=1.0,
        render_ms=2.0,
        face_present=True,
        width=8,
        height=8,
        session_epoch=first.epoch,
        frame_sequence=0,
    )
    assert service._account_completed_send(first, 0)
    service._invalidate_render_session(first)
    second = service._begin_render_session()
    try:
        assert second.epoch != first.epoch
        assert not service._publish_render_if_current(publication)
        latest, _timestamp = service.output.latest()
        assert latest is prior_preview
        assert service.stats_dict()["frames_rendered"] == 0
    finally:
        service._invalidate_render_session(second)
        service.close()


# LIFE-01: a publication that wins the lease lock commits exactly once.
def test_LIFE_01_publication_winner_is_accounted_exactly_once():
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    service.activate_initial()
    lease = service._begin_render_session()
    publication = _RenderPublication(
        frame=np.full((8, 8, 3), 47, dtype=np.uint8),
        version=runtime.read().version,
        driver_ms=1.0,
        render_ms=2.0,
        face_present=True,
        width=8,
        height=8,
        session_epoch=lease.epoch,
        frame_sequence=0,
    )
    try:
        assert service._account_completed_send(lease, 0)
        assert not service._account_completed_send(lease, 0)
        assert service._publish_render_if_current(publication)
        assert not service._publish_render_if_current(publication)
        assert service.stats_dict()["frames_sent"] == 1
        assert service.stats_dict()["frames_rendered"] == 1
    finally:
        service._invalidate_render_session(lease)
        service.close()


# LIFE-02: successful native interruption is not terminal worker ownership.
def test_LIFE_02_successful_cancel_without_wakeup_retains_complete_generation(
    monkeypatch,
):
    iterator_entered = threading.Event()
    iterator_release = threading.Event()

    class Call:
        def __init__(self):
            self.cancel_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            iterator_entered.set()
            iterator_release.wait()
            raise StopIteration

        def cancel(self):
            self.cancel_calls += 1
            return True  # deliberately does not wake __next__

    class Channel:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class Source:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    call = Call()
    channel = Channel()
    source = Source()
    channels_created = 0

    def create_channel(_target, **_kwargs):
        nonlocal channels_created
        channels_created += 1
        return channel

    class Stub:
        def __init__(self, _channel):
            pass

        def ProcessAudioStream(self, _requests):
            return call

    protocol = SimpleNamespace(
        grpc=SimpleNamespace(insecure_channel=create_channel),
        stub_class=Stub,
    )
    monkeypatch.setattr(audio2face_mod, "_load_protocol", lambda: protocol)
    monkeypatch.setattr(audio2face_mod, "create_audio_source", lambda _cfg: source)
    monkeypatch.setattr(audio2face_mod, "_CLOSE_TIMEOUT_S", 0.1)

    driver = Audio2FaceDriver(Audio2FaceConfig(url="grpc://127.0.0.1:52000"))
    driver.start()
    assert iterator_entered.wait(1.0)
    worker = driver._worker
    assert worker is not None

    try:
        with pytest.raises(DriverUnavailableError, match="worker did not stop"):
            driver.close()

        generation = driver._owned_generation
        assert generation is not None
        assert generation.source is source
        assert generation.call is call
        assert generation.channel is channel
        assert generation.worker is worker
        assert worker.is_alive()
        assert driver._worker is worker
        assert [item.resource for item in generation.interruptions] == [
            source,
            call,
            channel,
        ]
        assert generation.completed_interruptions == {
            ("source", id(source)),
            ("call", id(call)),
            ("channel", id(channel)),
        }

        with pytest.raises(DriverUnavailableError, match="already closed"):
            driver.start()
        assert channels_created == 1

        iterator_release.set()
        driver.close()
        assert driver._owned_generation is None
        assert driver._worker is None
        assert not worker.is_alive()
        assert source.close_calls == 1
        assert call.cancel_calls == 1
        assert channel.close_calls == 1

        driver.close()
        assert source.close_calls == 1
        assert call.cancel_calls == 1
        assert channel.close_calls == 1
    finally:
        iterator_release.set()
        worker.join(1.0)
