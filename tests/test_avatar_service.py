"""End-to-end avatar service tests against a fake custback frame API.

A real ``websockets`` server on loopback stands in for custback: it
authenticates the Bearer token, streams synthetic camera JPEGs, and collects
the frames the avatar service sends back — the exact stage-2 contract.
"""

import asyncio
import contextlib
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
websockets = pytest.importorskip("websockets")

from custback.avatar.config import AvatarConfig, AvatarRuntime
from custback.avatar.service import AvatarService

TOKEN = "avatar-service-test-token-0123456789abcdef"
WIDTH, HEIGHT = 160, 120


def run_async(awaitable):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(awaitable)
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        loop.close()
        asyncio.set_event_loop(None)


def _camera_jpeg(index: int) -> bytes:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, : (index * 13) % WIDTH] = (0, 128, 255)
    ok, jpeg = cv2.imencode(".jpg", frame)
    assert ok
    return jpeg.tobytes()


def _request_headers(ws):
    request = getattr(ws, "request", None)
    if request is not None and hasattr(request, "headers"):
        return request.headers
    return ws.request_headers  # websockets < 14 legacy protocol


class FakeCustback:
    """Loopback stand-in for custback's /ws/frames?stream=raw endpoint."""

    def __init__(self, *, close_after: int | None = None):
        self.received: list[bytes] = []
        self.rejected = 0
        self.sessions = 0
        self.close_after = close_after
        self._server = None
        self.port = 0

    async def _handler(self, ws):
        if _request_headers(ws).get("Authorization") != f"Bearer {TOKEN}":
            self.rejected += 1
            await ws.close(code=4401, reason="valid API token required")
            return
        self.sessions += 1

        async def send_frames():
            index = 0
            while True:
                await ws.send(_camera_jpeg(index))
                index += 1
                if self.close_after is not None and index >= self.close_after:
                    await ws.close()
                    return
                await asyncio.sleep(0.02)

        async def receive_frames():
            while True:
                data = await ws.recv()
                if isinstance(data, bytes):
                    self.received.append(data)

        tasks = {
            asyncio.create_task(send_frames()),
            asyncio.create_task(receive_frames()),
        }
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def __aenter__(self):
        self._server = await websockets.serve(self._handler, "127.0.0.1", 0)
        self.port = next(iter(self._server.sockets)).getsockname()[1]
        return self

    async def __aexit__(self, *_exc):
        self._server.close()
        with contextlib.suppress(Exception):
            await self._server.wait_closed()


@pytest.fixture()
def token_file(tmp_path):
    path = tmp_path / "api-token"
    path.write_text(TOKEN + "\n")
    os.chmod(path, 0o600)
    return path


def _service(token_file, port, **overrides) -> AvatarService:
    config = {
        "source": {
            "url": f"ws://127.0.0.1:{port}",
            "token_file": str(token_file),
            "connect_timeout_s": 2.0,
            "reconnect_min_s": 0.1,
            "reconnect_max_s": 0.5,
        },
        "driver": {"backend": "idle", "smoothing": 0.0},
        "background": {"mode": "color", "color": [10, 20, 30]},
        "render": {"max_fps": 60},
        "api": {"enabled": False},
    }
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    runtime = AvatarRuntime(AvatarConfig.from_dict(config))
    return AvatarService(runtime)


async def _wait_for(predicate, timeout=8.0, message="condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"timed out waiting for {message}")
        await asyncio.sleep(0.02)


def test_service_renders_and_returns_camera_sized_frames(token_file):
    async def scenario():
        async with FakeCustback() as fake:
            service = _service(token_file, fake.port)
            stop = asyncio.Event()
            runner = asyncio.create_task(service.run(stop))
            await _wait_for(lambda: len(fake.received) >= 3, message="returned frames")
            stats = service.stats_dict()
            stop.set()
            await asyncio.wait_for(runner, 5.0)
            return fake, stats, service

    fake, stats, service = run_async(scenario())
    assert fake.rejected == 0
    for data in fake.received[:3]:
        assert data.startswith(b"\xff\xd8")  # custback requires JPEG
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        # custback rejects any frame that does not match the camera size.
        assert frame.shape == (HEIGHT, WIDTH, 3)
    assert stats["connected"] is True
    assert stats["frames_sent"] >= 3
    assert stats["frames_received"] >= 3
    assert stats["driver_backend"] == "idle"
    assert stats["output_width"] == WIDTH
    assert stats["output_height"] == HEIGHT
    snapshot, _ = service.output.latest()
    assert snapshot is not None and snapshot.shape == (HEIGHT, WIDTH, 3)


def test_service_reconnects_after_source_drops(token_file):
    async def scenario():
        async with FakeCustback(close_after=2) as fake:
            service = _service(token_file, fake.port)
            stop = asyncio.Event()
            runner = asyncio.create_task(service.run(stop))
            await _wait_for(
                lambda: (
                    fake.sessions >= 3
                    and service.stats_dict()["reconnects"] >= 2
                ),
                message="reconnections",
            )
            stats = service.stats_dict()
            stop.set()
            await asyncio.wait_for(runner, 5.0)
            return stats

    stats = run_async(scenario())
    assert stats["reconnects"] >= 2
    assert stats["connect_attempts"] >= 3


def test_service_backs_off_while_source_is_down(token_file):
    async def scenario():
        service = _service(token_file, 1)  # nothing listens on port 1
        stop = asyncio.Event()
        runner = asyncio.create_task(service.run(stop))
        await _wait_for(
            lambda: service.stats_dict()["connect_attempts"] >= 2,
            message="retry attempts",
        )
        stats = service.stats_dict()
        stop.set()
        await asyncio.wait_for(runner, 5.0)
        return stats

    stats = run_async(scenario())
    assert stats["connected"] is False
    assert stats["last_error"] != ""
    assert stats["frames_sent"] == 0


def test_service_passes_verified_ssl_context_to_wss(monkeypatch, token_file):
    async def scenario():
        cfg = AvatarConfig.from_dict(
            {
                "source": {
                    "url": "wss://renderer.example:8710",
                    "token_file": str(token_file),
                },
                "driver": {"backend": "idle"},
                "api": {"enabled": False},
            }
        )
        service = AvatarService(AvatarRuntime(cfg))
        stop = asyncio.Event()
        captured = {}

        class Connection:
            async def __aenter__(self):
                stop.set()
                return object()

            async def __aexit__(self, *_exc):
                return False

        def connect(url, *, proxy=True, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            captured["proxy"] = proxy
            return Connection()

        monkeypatch.setattr("custback.avatar.service.websockets.connect", connect)
        await service.run(stop)
        return service, captured

    service, captured = run_async(scenario())
    context = captured["kwargs"]["ssl"]
    assert context is service._source_ssl
    assert context.check_hostname
    assert context.verify_mode.name == "CERT_REQUIRED"
    assert captured["proxy"] is None
    assert captured["url"].startswith("wss://renderer.example:8710/")


def test_service_switches_avatar_style_and_framing_live(token_file):
    async def scenario():
        async with FakeCustback() as fake:
            service = _service(token_file, fake.port)
            stop = asyncio.Event()
            runner = asyncio.create_task(service.run(stop))
            await _wait_for(lambda: len(fake.received) >= 2, message="first frames")
            service.runtime.apply_patch(
                {
                    "appearance": {
                        "avatar": "robin",
                        "style": "sketch",
                        "framing": "closeup",
                    }
                }
            )
            before = len(fake.received)
            await _wait_for(
                lambda: len(fake.received) >= before + 4, message="restyled frames"
            )
            stop.set()
            await asyncio.wait_for(runner, 5.0)
            frame = cv2.imdecode(
                np.frombuffer(fake.received[-1], np.uint8), cv2.IMREAD_COLOR
            )
            return frame

    frame = run_async(scenario())
    # closeup + sketch: the face fills the frame center as near-grayscale
    # pencil strokes, unlike the colored cartoon skin it replaced.
    center = frame[55:65, 75:85].reshape(-1, 3).mean(axis=0)
    assert abs(float(center[0]) - float(center[2])) < 25
    assert center.mean() > 140


def test_service_applies_hot_config_between_frames(token_file):
    async def scenario():
        async with FakeCustback() as fake:
            service = _service(token_file, fake.port)
            stop = asyncio.Event()
            runner = asyncio.create_task(service.run(stop))
            await _wait_for(lambda: len(fake.received) >= 2, message="first frames")
            service.runtime.apply_patch(
                {"background": {"mode": "color", "color": [0, 255, 0]}}
            )
            before = len(fake.received)
            await _wait_for(
                lambda: len(fake.received) >= before + 3, message="recolored frames"
            )
            stop.set()
            await asyncio.wait_for(runner, 5.0)
            frame = cv2.imdecode(
                np.frombuffer(fake.received[-1], np.uint8), cv2.IMREAD_COLOR
            )
            return frame

    frame = run_async(scenario())
    corner = frame[2, 2]  # background pixel, away from the avatar
    assert corner[1] > 180 and corner[0] < 80 and corner[2] < 80
