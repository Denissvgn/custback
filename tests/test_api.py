"""Authenticated API tests with a live synthetic pipeline.

HTTP requests use an async ASGI transport directly.  This covers both the
old-httpx and new-httpx2 Starlette dependency paths without relying on
AnyIO's blocking portal, which is unavailable in some restricted runtimes.
"""

import asyncio
import contextlib
import errno
import os
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import numpy as np
import pytest

pytest.importorskip("fastapi")
cv2 = pytest.importorskip("cv2")

import starlette
from fastapi import HTTPException, Request

if int(starlette.__version__.split(".", 1)[0]) >= 1:
    from httpx2 import ASGITransport as _ASGITransport
    from httpx2 import AsyncClient as _AsyncClient
else:  # Starlette < 1 uses the original httpx client contract.
    from httpx import ASGITransport as _ASGITransport
    from httpx import AsyncClient as _AsyncClient

from custback.api.security import SESSION_COOKIE, SecurityPolicy
import custback.api.server as server_mod
from custback.api.server import _UploadLimits, _UploadStore, create_app
from custback.config import AppConfig, ConfigState, RuntimeConfig
from custback.hub import FrameHub
from custback.pipeline import Pipeline


TOKEN = "test-api-token-which-is-at-least-32-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
RENDERER_TOKEN = "test-renderer-token-which-is-at-least-32-characters"
RENDERER_AUTH = {"Authorization": f"Bearer {RENDERER_TOKEN}"}
ORIGIN = "http://testserver"


def _async_client(app: object) -> Any:
    """Return a client across the incompatible httpx/httpx2 transport types."""

    transport = cast(Any, _ASGITransport)(app=app)
    return cast(Any, _AsyncClient)(
        transport=transport,
        base_url="http://testserver",
    )


def _error_code(error: HTTPException) -> object:
    """Read the structured detail payload asserted by upload validation tests."""

    return cast(dict[str, object], error.detail)["code"]


async def _with_event_loop_heartbeat(awaitable):
    """Keep restricted selectors polling while worker-thread callbacks finish."""

    async def heartbeat():
        while True:
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    try:
        return await awaitable
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def run_async(awaitable):
    # Avoid asyncio.run()'s blocking default-executor shutdown: the managed
    # sandbox cannot wake a selector that is waiting for that cross-thread
    # shutdown callback. CI and production do not need this harness.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_with_event_loop_heartbeat(awaitable))
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        asyncio.set_event_loop(None)


@dataclass
class Stack:
    app: object
    runtime: RuntimeConfig
    hub: FrameHub
    pipeline: Pipeline
    upload_dir: object

    async def arequest(self, method: str, path: str, **kwargs):
        async with _async_client(self.app) as client:
            return await client.request(method, path, **kwargs)

    def request(self, method: str, path: str, **kwargs):
        return run_async(self.arequest(method, path, **kwargs))

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)

    def patch(self, path: str, **kwargs):
        return self.request("PATCH", path, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.request("DELETE", path, **kwargs)


class WebSocketClosed(Exception):
    def __init__(self, code: int, reason: str = ""):
        self.code = code
        self.reason = reason
        super().__init__(f"WebSocket closed with {code}: {reason}")


class ASGIWebSocket:
    """Small deterministic WebSocket peer for exercising the ASGI app."""

    def __init__(self, app, url: str, headers: dict[str, str] | None = None):
        self.app = app
        self.url = url
        self.headers = headers or {}
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.outgoing: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None

    async def _receive(self):
        return await self.incoming.get()

    async def _send(self, message):
        await self.outgoing.put(message)

    async def connect(self):
        parsed = urlsplit(self.url)
        headers = {"host": "testserver", **self.headers}
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
            "path": parsed.path,
            "raw_path": parsed.path.encode(),
            "query_string": parsed.query.encode(),
            "headers": [
                (name.lower().encode(), value.encode())
                for name, value in headers.items()
            ],
            "subprotocols": [],
            "state": {},
            "extensions": {},
        }
        self.task = asyncio.create_task(self.app(scope, self._receive, self._send))
        await self.incoming.put({"type": "websocket.connect"})
        message = await asyncio.wait_for(self.outgoing.get(), 2.0)
        if message["type"] == "websocket.close":
            await self._finish()
            raise WebSocketClosed(message.get("code", 1000), message.get("reason", ""))
        assert message["type"] == "websocket.accept", message
        return self

    async def send_bytes(self, data: bytes):
        await self.incoming.put({"type": "websocket.receive", "bytes": data})

    async def send_text(self, data: str):
        await self.incoming.put({"type": "websocket.receive", "text": data})

    async def receive_bytes(self) -> bytes:
        message = await asyncio.wait_for(self.outgoing.get(), 3.0)
        if message["type"] == "websocket.close":
            raise WebSocketClosed(message.get("code", 1000), message.get("reason", ""))
        assert message["type"] == "websocket.send", message
        return message["bytes"]

    async def receive_close(self) -> WebSocketClosed:
        # The independent sender may already have queued a camera frame before
        # the receiver validates our bad payload. Drain those frames until the
        # protocol close arrives.
        for _ in range(10):
            try:
                await self.receive_bytes()
            except WebSocketClosed as closed:
                return closed
        raise AssertionError("server did not close after an invalid payload")

    async def _finish(self):
        if self.task is None:
            return
        if not self.task.done():
            await self.incoming.put({"type": "websocket.disconnect", "code": 1000})
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(self.task, 2.0)

    async def close(self):
        await self._finish()


async def _first_mjpeg_frame(app: Any) -> bytes:
    """Read one encoded frame through the real streaming ASGI route."""

    incoming: asyncio.Queue = asyncio.Queue()
    outgoing: asyncio.Queue = asyncio.Queue()

    async def receive():
        return await incoming.get()

    async def send(message):
        await outgoing.put(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "GET",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 12345),
        "root_path": "",
        "path": "/video/mjpeg",
        "raw_path": b"/video/mjpeg",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"authorization", AUTH["Authorization"].encode()),
        ],
        "state": {},
        "extensions": {},
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await incoming.put({"type": "http.request", "body": b"", "more_body": False})
        started = await asyncio.wait_for(outgoing.get(), 2.0)
        assert started["type"] == "http.response.start"
        assert started["status"] == 200
        body = await asyncio.wait_for(outgoing.get(), 3.0)
        assert body["type"] == "http.response.body"
        header, payload = body["body"].split(b"\r\n\r\n", 1)
        length_line = next(
            line
            for line in header.split(b"\r\n")
            if line.startswith(b"Content-Length:")
        )
        length = int(length_line.split(b":", 1)[1].strip())
        return payload[:length]
    finally:
        await incoming.put({"type": "http.disconnect"})
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, 2.0)


@pytest.fixture()
def stack(tmp_path):
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 128, "height": 72, "fps": 60},
            "background": {"mode": "color", "color": [200, 30, 30]},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null", "fps": 60},
            "api": {"ws_max_bytes": 1024},
        }
    )
    runtime = RuntimeConfig(cfg)
    hub = FrameHub()
    pipeline = Pipeline(runtime, hub)
    pipeline.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and hub.output.latest()[0] is None:
        time.sleep(0.02)
    security = SecurityPolicy.for_bind(
        TOKEN,
        "testserver",
        80,
        allowed_origins=[ORIGIN],
        extra_hosts=["testserver"],
        renderer_token=RENDERER_TOKEN,
    )
    upload_dir = tmp_path / "uploads"
    app = create_app(runtime, hub, pipeline, security=security, upload_dir=upload_dir)
    yield Stack(app, runtime, hub, pipeline, upload_dir)
    pipeline.stop()


def test_every_data_route_requires_auth(stack):
    for path in ("/status", "/config", "/backgrounds", "/video/snapshot.jpg", "/docs"):
        response = stack.get(path)
        assert response.status_code == 401, path
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.headers["x-config-version"] == "0"

    upload = stack.post(
        "/background/image",
        files={"file": ("private.png", b"not-public", "image/png")},
    )
    assert upload.status_code == 401
    assert not stack.upload_dir.exists()


def test_unauthenticated_root_exposes_only_login_shell(stack):
    response = stack.get("/")
    assert response.status_code == 401
    assert "custback login" in response.text
    assert "/video/mjpeg" not in response.text


def test_status_and_config_with_bearer(stack):
    status = stack.get("/status", headers=AUTH)
    assert status.status_code == 200
    body = status.json()
    assert body["frames_out"] >= 1
    assert body["mode"] == "color"
    assert {
        "run_id",
        "capture_backend",
        "capture_target_fps",
        "capture_fps",
        "capture_frames_read",
        "capture_dropped_frames",
        "capture_frame_age_ms",
        "output_target_fps",
        "output_repeated_frames",
        "processing_deadline_misses",
        "color_correction_ms",
        "frame_processing_ms",
        "output_fallback_active",
        "segmentation_fallback_active",
        "background_video_frames_skipped",
    } <= body.keys()
    assert body["capture_backend"] == "synthetic"
    assert body["output_fallback_active"] is False
    if sys.platform == "win32":
        assert body["native_ring"] in {"section absent", "section present"}
    else:
        assert body["native_ring"] == "unsupported"
    config = stack.get("/config", headers=AUTH)
    assert config.json()["background"]["mode"] == "color"
    assert {
        "token",
        "token_file",
        "renderer_token_file",
        "tls_keyfile",
    }.isdisjoint(config.json()["api"])
    assert "token_file" not in config.json()["avatar"]


def test_public_config_exposes_only_backdrop_target_ids():
    secret_source = "/run/operator/private/camera-device"
    cfg = AppConfig.from_dict(
        {
            "background": {"mode": "camera", "camera_target": "side-camera"},
            "backdrop_targets": {
                "side-camera": {"source": secret_source},
                "desk-camera": {"source": 2},
            },
        }
    )

    public = server_mod._state_body(ConfigState(cfg, 7))
    background = public["background"]
    assert background["camera_target"] == "side-camera"
    assert background["camera_targets"] == ("desk-camera", "side-camera")
    assert background["camera_source_configured"] is True
    assert background["fit_mode"] == "cover"
    assert background["anchor_x"] == 0.5
    assert background["anchor_y"] == 0.5
    assert "camera_device" not in background
    assert "backdrop_targets" not in public
    assert secret_source not in repr(public)


def test_exact_origin_is_enforced_even_with_token(stack):
    rejected = stack.get("/status", headers={**AUTH, "Origin": "https://evil.example"})
    assert rejected.status_code == 403
    allowed = stack.get("/status", headers={**AUTH, "Origin": ORIGIN})
    assert allowed.status_code == 200

    duplicate_host = stack.get(
        "/status",
        headers=[
            ("Host", "testserver"),
            ("Host", "evil.example"),
            ("Authorization", f"Bearer {TOKEN}"),
        ],
    )
    assert duplicate_host.status_code == 403


def test_browser_session_cookie(stack):
    async def scenario():
        async with _async_client(stack.app) as client:
            denied = await client.post(
                "/auth/session",
                json={"token": "x" * 40},
                headers={"Origin": ORIGIN},
            )
            assert denied.status_code == 401
            created = await client.post(
                "/auth/session",
                json={"token": TOKEN},
                headers={"Origin": ORIGIN},
            )
            assert created.status_code == 204
            assert created.cookies.get("custback_session")
            assert (await client.get("/status")).status_code == 200
            assert (await client.delete("/auth/session")).status_code == 204
            assert (await client.get("/status")).status_code == 401

    run_async(scenario())


def test_browser_session_rejects_undeclared_body_fields(stack):
    response = stack.post(
        "/auth/session",
        json={"token": TOKEN, "unexpected": "not accepted"},
        headers={"Origin": ORIGIN},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_content"


def test_hot_patch_returns_committed_version_header(stack):
    response = stack.patch(
        "/config", json={"background": {"mode": "passthrough"}}, headers=AUTH
    )
    assert response.status_code == 200
    assert response.json()["config"]["background"]["mode"] == "passthrough"
    assert response.json()["config_version"] == 1
    assert response.headers["x-config-version"] == "1"
    assert stack.runtime.read().config.background.mode == "passthrough"

    config = stack.get("/config", headers=AUTH)
    assert config.headers["x-config-version"] == "1"
    assert config.json()["background"]["mode"] == "passthrough"


def test_noop_patch_preserves_config_version(stack):
    response = stack.patch(
        "/config", json={"background": {"mode": "color"}}, headers=AUTH
    )
    assert response.status_code == 200
    assert response.json()["config_version"] == 0
    assert response.headers["x-config-version"] == "0"


def test_restart_only_patch_is_409_and_atomic(stack):
    response = stack.patch(
        "/config",
        json={"camera": {"width": 64}, "background": {"mode": "passthrough"}},
        headers=AUTH,
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "restart_required"
    assert response.json()["detail"]["fields"] == ["camera.width"]
    current = stack.runtime.read()
    assert current.version == 0
    assert current.config.camera.width == 128
    assert current.config.background.mode == "color"


def test_visual_policy_patch_is_hot_atomic_and_public(stack):
    response = stack.patch(
        "/config",
        json={
            "background": {
                "fit_mode": "contain",
                "anchor_x": 0.25,
                "anchor_y": 0.75,
            },
            "compositing": {
                "blend_space": "linear_srgb",
                "color_correction": {
                    "mode": "auto",
                    "strength": 0.7,
                    "exposure_limit_ev": 0.9,
                    "white_balance_strength": 0.4,
                    "adaptation_time_s": 1.2,
                },
            },
        },
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.json()["config_version"] == 1
    public = response.json()["config"]
    assert public["schema_version"] == 1
    assert public["background"]["fit_mode"] == "contain"
    assert public["background"]["anchor_x"] == 0.25
    assert public["background"]["anchor_y"] == 0.75
    assert public["compositing"]["blend_space"] == "linear_srgb"
    assert public["compositing"]["color_correction"] == {
        "mode": "auto",
        "strength": 0.7,
        "exposure_limit_ev": 0.9,
        "white_balance_strength": 0.4,
        "adaptation_time_s": 1.2,
    }


def test_schema_version_patch_is_restart_required_even_when_unchanged(stack):
    response = stack.patch(
        "/config",
        json={"schema_version": 1, "background": {"anchor_x": 0.25}},
        headers=AUTH,
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "restart_required"
    assert detail["fields"] == ["schema_version"]
    assert detail["current_version"] == 0
    current = stack.runtime.read()
    assert current.version == 0
    assert current.config.schema_version == 1
    assert current.config.background.anchor_x == 0.5


def test_canvas_and_camera_geometry_patch_is_restart_required_and_atomic(stack):
    response = stack.patch(
        "/config",
        json={
            "camera": {"fit_mode": "cover", "rotation": 90},
            "output": {"width": 1920, "height": 1080},
            "background": {"anchor_x": 0.25},
        },
        headers=AUTH,
    )

    assert response.status_code == 409
    assert response.json()["detail"]["fields"] == [
        "camera.fit_mode",
        "camera.rotation",
        "output.height",
        "output.width",
    ]
    current = stack.runtime.read()
    assert current.version == 0
    assert current.config.camera.fit_mode == "stretch"
    assert (current.config.output.width, current.config.output.height) == (None, None)
    assert current.config.background.anchor_x == 0.5


def test_hot_camera_backdrop_source_patch_is_restart_required(stack):
    response = stack.patch(
        "/config",
        json={
            "background": {
                "mode": "camera",
                "camera_device": "/run/secrets/operator-camera",
            }
        },
        headers=AUTH,
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "restart_required"
    assert response.json()["detail"]["fields"] == ["background.camera_device"]
    assert stack.runtime.version == 0
    assert stack.runtime.snapshot().background.mode == "color"


def test_invalid_patch_is_422(stack):
    response = stack.patch(
        "/config", json={"background": {"mode": "bogus"}}, headers=AUTH
    )
    assert response.status_code == 422
    assert stack.runtime.read().config.background.mode == "color"

    malformed = stack.patch("/config", json=["not", "an", "object"], headers=AUTH)
    assert malformed.status_code == 422
    assert malformed.json()["detail"]["code"] == "invalid_content"


def test_invalid_config_response_never_echoes_sensitive_input(stack):
    secret = "do-not-reflect-this-token-value-0123456789"
    response = stack.patch(
        "/config",
        json={"api": {"token": secret}},
        headers=AUTH,
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_config"
    assert secret not in response.text
    assert "configuration validation failed" in response.text


def test_pathological_json_number_is_stable_422(stack):
    body = b'{"background":{"blur_kernel":' + (b"9" * 5000) + b"}}"
    response = stack.patch(
        "/config",
        content=body,
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_content"


def test_unavailable_reconfiguration_is_standardized_503(stack, monkeypatch):
    # The class is defined in pipeline.py; expose the same class-name contract
    # through the coordinator method without mutating runtime state.
    from custback.pipeline import ReconfigurationUnavailable

    def unavailable(*_args, **_kwargs):
        raise ReconfigurationUnavailable("pipeline unavailable")

    monkeypatch.setattr(stack.pipeline, "apply_config_patch", unavailable)
    response = stack.patch(
        "/config", json={"background": {"mode": "passthrough"}}, headers=AUTH
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "reconfiguration_unavailable"
    assert response.headers["x-config-version"] == "0"


def test_snapshot_returns_jpeg(stack):
    response = stack.get("/video/snapshot.jpg", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


def test_active_auto_frame_is_consistent_across_raw_output_snapshot_and_mjpeg(
    stack,
    tmp_path,
):
    height, width = 72, 128
    y, x = np.indices((height, width), dtype=np.int16)
    backdrop = np.stack(
        (
            125 + (x % 17),
            145 + (y % 13),
            165 + ((x + y) % 11),
        ),
        axis=-1,
    ).astype(np.uint8)
    image_path = tmp_path / "auto-backdrop.png"
    assert cv2.imwrite(str(image_path), backdrop)
    with image_path.open("rb") as handle:
        uploaded = stack.post(
            "/background/image",
            files={"file": ("auto-backdrop.png", handle, "image/png")},
            headers=AUTH,
        )
    assert uploaded.status_code == 201
    configured = stack.patch(
        "/config",
        json={
            "compositing": {
                "blend_space": "linear_srgb",
                "light_wrap": 0.0,
                "color_correction": {
                    "mode": "auto",
                    "strength": 0.7,
                },
            }
        },
        headers=AUTH,
    )
    assert configured.status_code == 200

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        stats = stack.hub.stats_dict()
        if (
            stats["color_correction_active"] is True
            and stats["color_correction_applied_frames"] >= 1
        ):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("automatic correction never became active")

    # Freeze a single generation so every transport must encode the same raw
    # or composited pixels rather than merely a nearby synthetic-camera frame.
    stack.pipeline.stop()
    raw_exact = stack.hub.raw.latest()[0]
    output_exact = stack.hub.output.latest()[0]
    assert raw_exact is not None and output_exact is not None
    assert not np.array_equal(raw_exact, output_exact)

    async def read_websockets() -> tuple[bytes, bytes]:
        raw_socket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=raw",
            headers={**AUTH, "Origin": ORIGIN},
        ).connect()
        output_socket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=output",
            headers={**AUTH, "Origin": ORIGIN},
        ).connect()
        try:
            return (
                await raw_socket.receive_bytes(),
                await output_socket.receive_bytes(),
            )
        finally:
            await raw_socket.close()
            await output_socket.close()

    raw_jpeg, output_ws_jpeg = run_async(read_websockets())
    mjpeg = run_async(_first_mjpeg_frame(stack.app))
    snapshot = stack.get("/video/snapshot.jpg", headers=AUTH)
    assert snapshot.status_code == 200

    def decode(jpeg: bytes) -> np.ndarray:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert frame is not None
        return frame

    decoded_raw = decode(raw_jpeg)
    decoded_outputs = [
        decode(value) for value in (output_ws_jpeg, mjpeg, snapshot.content)
    ]
    assert (
        float(
            np.mean(np.abs(decoded_raw.astype(np.int16) - raw_exact.astype(np.int16)))
        )
        < 4.0
    )
    for decoded in decoded_outputs:
        assert (
            float(
                np.mean(
                    np.abs(decoded.astype(np.int16) - output_exact.astype(np.int16))
                )
            )
            < 4.0
        )
    np.testing.assert_array_equal(decoded_outputs[0], decoded_outputs[1])
    np.testing.assert_array_equal(decoded_outputs[0], decoded_outputs[2])
    assert not np.array_equal(decoded_raw, decoded_outputs[0])


def test_remote_near_raw_echo_reaches_mjpeg_only_as_privacy_slate(stack):
    configured = stack.patch(
        "/config",
        json={
            "background": {
                "mode": "remote",
                "remote_fallback_mode": "color",
            },
            "api": {"remote_timeout_ms": 5_000},
        },
        headers=AUTH,
    )
    assert configured.status_code == 200

    async def trigger_privacy_gate() -> tuple[np.ndarray, np.ndarray, dict]:
        websocket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=raw",
            headers={**AUTH, "Origin": ORIGIN},
        ).connect()
        try:
            # Drain the frame which may predate renderer-session activation.
            await websocket.receive_bytes()
            raw_jpeg = await websocket.receive_bytes()
            raw = cv2.imdecode(
                np.frombuffer(raw_jpeg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            assert raw is not None and raw.shape == (72, 128, 3)

            # The fixture intentionally limits renderer frames to 1 KiB. Use
            # the highest quality that fits, preserving a recognizable JPEG
            # near-echo while still traversing the real WebSocket decoder.
            echo_jpeg = None
            for quality in (60, 50, 40, 30, 25, 20, 15, 10):
                ok, encoded = cv2.imencode(
                    ".jpg",
                    raw,
                    [cv2.IMWRITE_JPEG_QUALITY, quality],
                )
                assert ok
                if encoded.nbytes <= stack.runtime.snapshot().api.ws_max_bytes:
                    echo_jpeg = encoded.tobytes()
                    break
            assert echo_jpeg is not None
            near_echo = cv2.imdecode(
                np.frombuffer(echo_jpeg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            assert near_echo is not None
            assert (
                float(
                    np.mean(np.abs(near_echo.astype(np.int16) - raw.astype(np.int16)))
                )
                < 8.0
            )
            await websocket.send_bytes(echo_jpeg)

            slate = Pipeline._privacy_slate(raw.shape)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                stats = stack.hub.stats_dict()
                protected = stack.hub.output.latest()[0]
                if (
                    stats["remote_fallback_reason"]
                    in {"privacy-raw-echo", "privacy-delayed-raw-echo"}
                    and protected is not None
                    and np.array_equal(protected, slate)
                ):
                    # Stop while the renderer session and echo classification
                    # are current, freezing one exact protected generation for
                    # every transport assertion below.
                    stack.pipeline.stop()
                    frozen = stack.hub.output.latest()[0]
                    assert frozen is not None
                    return raw, frozen.copy(), stats
                await asyncio.sleep(0.02)
            raise AssertionError(
                "near-raw renderer echo never reached the privacy gate"
            )
        finally:
            await websocket.close()

    echoed_raw, frozen, privacy_stats = run_async(trigger_privacy_gate())
    slate = Pipeline._privacy_slate(frozen.shape)
    np.testing.assert_array_equal(frozen, slate)
    assert not np.array_equal(echoed_raw, frozen)
    assert privacy_stats["remote_fallback_active"] is True
    assert privacy_stats["remote_fallback_mode"] == "privacy-slate"
    assert privacy_stats["remote_fallback_reason"] in {
        "privacy-raw-echo",
        "privacy-delayed-raw-echo",
    }

    mjpeg = run_async(_first_mjpeg_frame(stack.app))
    snapshot = stack.get("/video/snapshot.jpg", headers=AUTH)
    assert snapshot.status_code == 200
    for jpeg in (mjpeg, snapshot.content):
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded is not None and decoded.shape == slate.shape
        error = np.abs(decoded.astype(np.int16) - slate.astype(np.int16))
        assert float(np.mean(error)) < 4.0


def test_upload_image_validates_and_switches_mode(stack, tmp_path):
    image_path = tmp_path / "office.png"
    assert cv2.imwrite(str(image_path), np.full((40, 40, 3), 77, np.uint8))
    with image_path.open("rb") as handle:
        response = stack.post(
            "/background/image",
            files={"file": ("office.png", handle, "image/png")},
            headers=AUTH,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["width"] == body["height"] == 40
    assert body["original_name"] == "office.png"
    assert (stack.upload_dir / body["id"]).is_file()
    background = stack.runtime.read().config.background
    assert background.mode == "image"
    assert Path(background.image_path).name == body["id"]


def test_upload_is_hidden_until_preflight_and_atomic_activation(
    stack, tmp_path, monkeypatch
):
    image_path = tmp_path / "staged.png"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), np.uint8))
    payload = image_path.read_bytes()
    entered = threading.Event()
    release = threading.Event()
    original = stack.pipeline.apply_staged_config_patch

    def delayed_apply(*args, **kwargs):
        entered.set()
        assert release.wait(2.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(stack.pipeline, "apply_staged_config_patch", delayed_apply)

    async def scenario():
        request = asyncio.create_task(
            stack.arequest(
                "POST",
                "/background/image",
                files={"file": ("staged.png", payload, "image/png")},
                headers=AUTH,
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0.01)
        names = sorted(path.name for path in stack.upload_dir.iterdir())
        assert len(names) == 1
        assert names[0].startswith(".upload-")
        assert names[0].endswith(".png")
        release.set()
        response = await request
        assert response.status_code == 201
        assert sorted(path.name for path in stack.upload_dir.iterdir()) == [
            response.json()["id"]
        ]

    run_async(scenario())


def test_cancelled_upload_does_not_delete_successfully_activated_asset(
    stack, tmp_path, monkeypatch
):
    image_path = tmp_path / "cancelled.png"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), np.uint8))
    payload = image_path.read_bytes()
    entered = threading.Event()
    release = threading.Event()
    original = stack.pipeline.apply_staged_config_patch

    def delayed_apply(*args, **kwargs):
        entered.set()
        assert release.wait(2.0)
        return original(*args, **kwargs)

    monkeypatch.setattr(stack.pipeline, "apply_staged_config_patch", delayed_apply)

    async def scenario():
        request = asyncio.create_task(
            stack.arequest(
                "POST",
                "/background/image",
                files={"file": ("cancelled.png", payload, "image/png")},
                headers=AUTH,
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0.01)
        request.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request

    run_async(scenario())
    deadline = time.monotonic() + 2.0
    while (
        time.monotonic() < deadline
        and stack.runtime.read().config.background.mode != "image"
    ):
        time.sleep(0.01)
    active = Path(stack.runtime.read().config.background.image_path)
    assert stack.runtime.read().config.background.mode == "image"
    assert active.is_file()


def test_upload_rejects_extension_and_fake_content(stack):
    bad_extension = stack.post(
        "/background/image",
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
        headers=AUTH,
    )
    assert bad_extension.status_code == 415
    fake_image = stack.post(
        "/background/image",
        files={"file": ("fake.png", b"not an image", "image/png")},
        headers=AUTH,
    )
    assert fake_image.status_code == 422
    assert not list(stack.upload_dir.glob(".upload-*"))
    assert stack.runtime.read().config.background.mode == "color"


def test_valid_upload_is_removed_when_activation_fails(stack, tmp_path, monkeypatch):
    image_path = tmp_path / "candidate.png"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), np.uint8))

    def fail_candidate(_cfg, **_kwargs):
        raise RuntimeError("candidate backdrop failed")

    monkeypatch.setattr("custback.pipeline.create_backdrop", fail_candidate)
    with image_path.open("rb") as handle:
        response = stack.post(
            "/background/image",
            files={"file": ("candidate.png", handle, "image/png")},
            headers=AUTH,
        )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "activation_failed"
    assert list(stack.upload_dir.glob("*")) == []
    assert stack.runtime.read().config.background.mode == "color"
    assert stack.runtime.read().version == 0


def test_failed_promotion_rollback_reclaims_uncommitted_final_asset(
    stack, tmp_path, monkeypatch
):
    image_path = tmp_path / "rollback.png"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), np.uint8))
    monkeypatch.setattr(
        stack.pipeline,
        "_install_activation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected install failure")
        ),
    )

    def fail_rollback(_store, _staged, _final):
        raise OSError(errno.ENOSPC, "rollback failed")

    monkeypatch.setattr(_UploadStore, "rollback_promotion", fail_rollback)
    with image_path.open("rb") as handle:
        response = stack.post(
            "/background/image",
            files={"file": ("rollback.png", handle, "image/png")},
            headers=AUTH,
        )
    assert response.status_code == 507
    assert response.json()["detail"]["code"] == "insufficient_storage"
    assert list(stack.upload_dir.glob("*")) == []
    assert stack.runtime.read().version == 0
    assert stack.runtime.read().config.background.mode == "color"


def test_failed_upload_cleanup_retries_after_reconfiguration_recovers(
    stack, tmp_path, monkeypatch
):
    from custback.pipeline import ReconfigurationUnavailable

    image_path = tmp_path / "retry-cleanup.png"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), np.uint8))
    original_mutation = stack.pipeline.apply_storage_mutation

    def unavailable(*_args, **_kwargs):
        raise ReconfigurationUnavailable("pipeline temporarily stalled")

    monkeypatch.setattr(stack.pipeline, "apply_staged_config_patch", unavailable)
    monkeypatch.setattr(stack.pipeline, "apply_storage_mutation", unavailable)
    with image_path.open("rb") as handle:
        response = stack.post(
            "/background/image",
            files={"file": ("retry-cleanup.png", handle, "image/png")},
            headers=AUTH,
        )
    assert response.status_code == 503
    assert list(stack.upload_dir.glob(".upload-*"))
    assert stack.runtime.read().version == 0

    monkeypatch.setattr(stack.pipeline, "apply_storage_mutation", original_mutation)
    deadline = time.monotonic() + 3.0
    while list(stack.upload_dir.glob(".upload-*")) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert list(stack.upload_dir.glob("*")) == []
    assert stack.runtime.read().version == 0


def test_delete_rejects_active_then_removes_inactive_background(stack, tmp_path):
    image_path = tmp_path / "office.png"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), np.uint8))
    with image_path.open("rb") as handle:
        uploaded = stack.post(
            "/background/image",
            files={"file": ("office.png", handle, "image/png")},
            headers=AUTH,
        ).json()
    identifier = uploaded["id"]
    assert stack.delete(f"/backgrounds/{identifier}", headers=AUTH).status_code == 409
    assert (
        stack.patch(
            "/config", json={"background": {"mode": "passthrough"}}, headers=AUTH
        ).status_code
        == 200
    )
    deleted = stack.delete(f"/backgrounds/{identifier}", headers=AUTH)
    assert deleted.status_code == 204
    assert deleted.headers["x-config-version"] == "2"
    assert stack.runtime.read().version == 2
    assert not (stack.upload_dir / identifier).exists()


def test_delete_and_reactivation_are_serialized_without_dangling_config(
    stack, tmp_path
):
    image_path = tmp_path / "race.png"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), np.uint8))
    with image_path.open("rb") as handle:
        body = stack.post(
            "/background/image",
            files={"file": ("race.png", handle, "image/png")},
            headers=AUTH,
        ).json()
    identifier = body["id"]
    stored = stack.upload_dir / identifier
    assert (
        stack.patch(
            "/config", json={"background": {"mode": "passthrough"}}, headers=AUTH
        ).status_code
        == 200
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        deletion = pool.submit(stack.delete, f"/backgrounds/{identifier}", headers=AUTH)
        activation = pool.submit(
            stack.patch,
            "/config",
            json={
                "background": {
                    "mode": "image",
                    "image_path": str(stored),
                }
            },
            headers=AUTH,
        )
        responses = (deletion.result(5.0), activation.result(5.0))

    assert responses[0].status_code in (204, 409)
    assert responses[1].status_code in (200, 409, 422)
    effective = stack.runtime.read().config.background
    assert not (effective.mode == "image" and not Path(effective.image_path).is_file())


def test_websocket_requires_auth_and_origin(stack):
    async def scenario():
        with pytest.raises(WebSocketClosed) as unauthenticated:
            await ASGIWebSocket(stack.app, "/ws/frames?stream=raw").connect()
        assert unauthenticated.value.code == 4401

        with pytest.raises(WebSocketClosed) as bad_origin:
            await ASGIWebSocket(
                stack.app,
                "/ws/frames?stream=raw",
                headers={**AUTH, "Origin": "https://evil.example"},
            ).connect()
        assert bad_origin.value.code == 4403

    run_async(scenario())


def test_websocket_stream_limit_is_enforced_after_authentication(stack):
    limiter = stack.app.state.stream_connections
    leases = [limiter.try_acquire() for _ in range(limiter.maximum)]
    assert all(lease is not None for lease in leases)
    try:
        response = stack.get("/video/mjpeg", headers=AUTH)
        assert response.status_code == 429
        assert response.json()["detail"]["code"] == "stream_limit"

        async def scenario():
            with pytest.raises(WebSocketClosed) as unauthenticated:
                await ASGIWebSocket(
                    stack.app,
                    "/ws/frames?stream=raw",
                ).connect()
            assert unauthenticated.value.code == 4401

            with pytest.raises(WebSocketClosed) as saturated:
                await ASGIWebSocket(
                    stack.app,
                    "/ws/frames?stream=raw",
                    headers={**AUTH, "Origin": ORIGIN},
                ).connect()
            assert saturated.value.code == 4429

        run_async(scenario())
    finally:
        for lease in leases:
            lease.release()
    assert limiter.active == 0


def test_renderer_credential_is_scoped_to_raw_frame_websocket(stack):
    assert stack.get("/status", headers=RENDERER_AUTH).status_code == 401

    async def scenario():
        websocket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=raw",
            headers={**RENDERER_AUTH, "Origin": ORIGIN},
        ).connect()
        await websocket.close()

        with pytest.raises(WebSocketClosed) as output_stream:
            await ASGIWebSocket(
                stack.app,
                "/ws/frames?stream=output",
                headers={**RENDERER_AUTH, "Origin": ORIGIN},
            ).connect()
        assert output_stream.value.code == 4401

    run_async(scenario())


def test_output_websocket_is_read_only_and_does_not_create_renderer_session(stack):
    async def scenario():
        websocket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=output",
            headers={**AUTH, "Origin": ORIGIN},
        ).connect()
        assert stack.hub.stats_dict()["remote_connected"] is False
        frame = np.zeros((72, 128, 3), dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", frame)
        assert ok
        await websocket.send_bytes(jpeg.tobytes())
        closed = await websocket.receive_close()
        assert closed.code == 1008
        assert stack.hub.remote_in.latest()[0] is None
        assert stack.hub.stats_dict()["remote_connected"] is False
        await websocket.close()

    run_async(scenario())


def test_openapi_documents_bodies_and_local_docs_have_no_cdn(stack):
    docs = stack.get("/docs", headers=AUTH)
    assert docs.status_code == 200
    assert "cdn" not in docs.text.lower()
    assert "<script" not in docs.text.lower()
    assert docs.headers["content-security-policy"]

    schema = stack.get("/openapi.json", headers=AUTH).json()
    assert "requestBody" in schema["paths"]["/auth/session"]["post"]
    assert "requestBody" in schema["paths"]["/config"]["patch"]
    patch_content = schema["paths"]["/config"]["patch"]["requestBody"]["content"]
    expected_visual_examples = {
        "camera-cover-restart",
        "backdrop-contain",
        "linear-compositing",
        "automatic-color-correction",
        "visual-quality-hot",
    }
    for media_type in ("application/json", "application/merge-patch+json"):
        assert set(patch_content[media_type]["examples"]) == expected_visual_examples
        examples = patch_content[media_type]["examples"]
        assert examples["camera-cover-restart"]["value"] == {
            "camera": {"fit_mode": "cover"}
        }
        assert examples["linear-compositing"]["value"] == {
            "compositing": {"blend_space": "linear_srgb"}
        }
        assert examples["automatic-color-correction"]["value"] == {
            "compositing": {"color_correction": {"mode": "auto", "strength": 0.5}}
        }
        assert examples["visual-quality-hot"]["value"] == {
            "background": {
                "fit_mode": "cover",
                "anchor_x": 0.5,
                "anchor_y": 0.25,
            },
            "compositing": {
                "blend_space": "linear_srgb",
                "color_correction": {"mode": "auto", "strength": 0.5},
            },
        }
    get_config = schema["paths"]["/config"]["get"]
    public_config_ref = get_config["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    assert public_config_ref.endswith("/PublicAppConfig")
    schemas = schema["components"]["schemas"]
    assert "schema_version" in schemas["PublicAppConfig"]["properties"]
    assert {
        "token_file",
        "renderer_token_file",
        "tls_keyfile",
    }.isdisjoint(schemas["PublicApiConfig"]["properties"])
    assert {
        "token_file",
        "tls_ca_file",
        "tls_certfile",
        "tls_keyfile",
    }.isdisjoint(schemas["PublicAvatarRemoteConfig"]["properties"])
    assert "camera_device" not in schemas["PublicBackgroundConfig"]["properties"]
    assert {"fit_mode", "anchor_x", "anchor_y"} <= set(
        schemas["PublicBackgroundConfig"]["properties"]
    )
    assert {"fit_mode", "anchor_x", "anchor_y", "rotation"} <= set(
        schemas["CameraConfig"]["properties"]
    )
    assert schemas["CameraConfig"]["properties"]["fit_mode"]["default"] == "stretch"
    assert {"width", "height"} <= set(schemas["OutputConfig"]["properties"])
    assert {"blend_space", "color_correction"} <= set(
        schemas["CompositingConfig"]["properties"]
    )
    assert (
        schemas["CompositingConfig"]["properties"]["blend_space"]["default"]
        == "srgb_legacy"
    )
    assert {
        "mode",
        "strength",
        "exposure_limit_ev",
        "white_balance_strength",
        "adaptation_time_s",
    } <= set(schemas["ColorCorrectionConfig"]["properties"])
    assert schemas["ColorCorrectionConfig"]["properties"]["mode"]["default"] == "off"
    executable_example = patch_content["application/merge-patch+json"]["examples"][
        "visual-quality-hot"
    ]["value"]
    applied = stack.patch(
        "/config",
        headers={**AUTH, "content-type": "application/merge-patch+json"},
        json=executable_example,
    )
    assert applied.status_code == 200
    effective = applied.json()["config"]
    assert effective["background"]["fit_mode"] == "cover"
    assert effective["background"]["anchor_y"] == 0.25
    assert effective["compositing"]["blend_space"] == "linear_srgb"
    assert effective["compositing"]["color_correction"]["mode"] == "auto"
    patch_schema = schema["paths"]["/config"]["patch"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert patch_schema["$ref"].endswith("/_ConfigPatchResponse")
    for path in ("/background/image", "/background/video"):
        operation = schema["paths"][path]["post"]
        assert "multipart/form-data" in operation["requestBody"]["content"]
        assert "201" in operation["responses"]
        response_schema = operation["responses"]["201"]["content"]["application/json"][
            "schema"
        ]
        assert response_schema["$ref"].endswith("/_UploadResponse")
    assert schema["paths"]["/status"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/_StatusResponse")
    assert schema["paths"]["/backgrounds"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/_BackgroundListResponse")
    snapshot_content = schema["paths"]["/video/snapshot.jpg"]["get"]["responses"][
        "200"
    ]["content"]
    assert set(snapshot_content) == {"image/jpeg"}
    assert snapshot_content["image/jpeg"]["schema"]["format"] == "binary"
    mjpeg_content = schema["paths"]["/video/mjpeg"]["get"]["responses"]["200"][
        "content"
    ]
    assert set(mjpeg_content) == {"multipart/x-mixed-replace"}
    assert mjpeg_content["multipart/x-mixed-replace"]["schema"]["format"] == "binary"


def test_websocket_round_trip(stack):
    stack.pipeline.apply_config_patch({"background": {"mode": "remote"}})

    async def scenario():
        websocket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=raw",
            headers={**AUTH, "Origin": ORIGIN},
        ).connect()
        try:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            assert frame is not None and frame.shape == (72, 128, 3)

            rendered = np.full_like(frame, (9, 9, 9))
            ok, jpeg = cv2.imencode(".jpg", rendered)
            assert ok
            await websocket.send_bytes(jpeg.tobytes())

            deadline = time.monotonic() + 5.0
            while (
                time.monotonic() < deadline and stack.hub.remote_in.latest()[0] is None
            ):
                await asyncio.sleep(0.02)
            assert stack.hub.remote_in.latest()[0] is not None
        finally:
            await websocket.close()

    run_async(scenario())


def test_invalidated_renderer_epoch_closes_websocket(stack):
    async def scenario():
        websocket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=raw",
            headers={**RENDERER_AUTH, "Origin": ORIGIN},
        ).connect()
        try:
            await websocket.receive_bytes()
            session = stack.hub.active_remote_session()
            assert session is not None
            assert stack.hub.invalidate_remote_session(session)
            closed = await websocket.receive_close()
            assert closed.code == 1012
            assert closed.reason == "renderer session invalidated"
        finally:
            await websocket.close()

    run_async(scenario())


def test_websocket_rejects_text_and_wrong_dimensions(stack):
    async def scenario():
        websocket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=raw",
            headers={**AUTH, "Origin": ORIGIN},
        ).connect()
        await websocket.send_text("not jpeg")
        closed = await websocket.receive_close()
        assert closed.code == 1003
        await websocket.close()

        wrong = np.zeros((10, 10, 3), np.uint8)
        ok, jpeg = cv2.imencode(".jpg", wrong)
        assert ok
        websocket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=raw",
            headers={**AUTH, "Origin": ORIGIN},
        ).connect()
        await websocket.send_bytes(jpeg.tobytes())
        closed = await websocket.receive_close()
        assert closed.code == 1007
        await websocket.close()

    run_async(scenario())


def test_websocket_rejects_frame_over_configured_limit(stack):
    async def scenario():
        websocket = await ASGIWebSocket(
            stack.app,
            "/ws/frames?stream=raw",
            headers={**AUTH, "Origin": ORIGIN},
        ).connect()
        await websocket.send_bytes(b"x" * 1025)
        closed = await websocket.receive_close()
        assert closed.code == 1009
        await websocket.close()

    run_async(scenario())


def _multipart_request(data: bytes, *, filename: str = "exact.png") -> Request:
    boundary = b"custback-test-boundary"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="'
        + filename.encode()
        + b'"\r\nContent-Type: image/png\r\n\r\n'
        + data
        + b"\r\n--"
        + boundary
        + b"--\r\n"
    )
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/background/image",
        "raw_path": b"/background/image",
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 1),
        "headers": [
            (b"content-type", b"multipart/form-data; boundary=" + boundary),
            (b"content-length", str(len(body)).encode()),
        ],
    }
    return Request(scope, receive)


def test_upload_store_enforces_exact_byte_limit_and_private_mode(tmp_path):
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    payload = encoded.tobytes()
    limits = _UploadLimits(
        image_max_bytes=len(payload),
        video_max_bytes=len(payload),
        image_max_pixels=100,
        storage_max_bytes=len(payload) * 2,
        max_files=2,
    )
    store = _UploadStore(tmp_path / "exact", limits)
    saved = run_async(store.save(_multipart_request(payload), "image"))
    assert saved.size == len(payload)
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(saved.path.stat().st_mode) == 0o600

    too_small = _UploadStore(
        tmp_path / "small",
        _UploadLimits(
            image_max_bytes=len(payload) - 1,
            video_max_bytes=len(payload),
            image_max_pixels=100,
            storage_max_bytes=len(payload) * 2,
            max_files=2,
        ),
    )
    with pytest.raises(HTTPException) as caught:
        run_async(too_small.save(_multipart_request(payload), "image"))
    assert caught.value.status_code == 413
    assert not list((tmp_path / "small").glob("*"))


def test_upload_store_establishes_mode_0600_with_restrictive_umask(tmp_path):
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    payload = encoded.tobytes()
    directory = tmp_path / "private-mode"
    directory.mkdir(mode=0o700)
    store = _UploadStore(
        directory,
        _UploadLimits(
            image_max_bytes=len(payload),
            storage_max_bytes=len(payload),
            max_files=1,
        ),
    )
    previous = os.umask(0o777)
    try:
        saved = run_async(store.save(_multipart_request(payload), "image"))
    finally:
        os.umask(previous)
    assert stat.S_IMODE(saved.path.stat().st_mode) == 0o600


def test_upload_store_establishes_exact_private_modes_with_permissive_umask(
    tmp_path,
):
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    payload = encoded.tobytes()
    store = _UploadStore(
        tmp_path / "permissive-mode",
        _UploadLimits(
            image_max_bytes=len(payload),
            storage_max_bytes=len(payload),
            max_files=1,
        ),
    )
    previous = os.umask(0)
    try:
        saved = run_async(store.save(_multipart_request(payload), "image"))
    finally:
        os.umask(previous)
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(saved.path.stat().st_mode) == 0o600


def test_upload_store_revalidates_managed_directory_after_path_swap(tmp_path):
    directory = tmp_path / "managed"
    store = _UploadStore(directory, _UploadLimits())
    store.ensure_directory()
    original = tmp_path / "original"
    directory.rename(original)
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o755)
    directory.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(OSError):
        store.ensure_directory()
    assert stat.S_IMODE(replacement.stat().st_mode) == 0o755


def test_upload_store_accounts_actual_bytes_and_rejects_quota(tmp_path):
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    payload = encoded.tobytes()
    store = _UploadStore(
        tmp_path / "quota",
        _UploadLimits(
            image_max_bytes=len(payload) * 4,
            video_max_bytes=len(payload) * 4,
            image_max_pixels=100,
            storage_max_bytes=len(payload),
            max_files=5,
        ),
    )
    run_async(store.save(_multipart_request(payload), "image"))
    with pytest.raises(HTTPException) as caught:
        run_async(store.save(_multipart_request(payload), "image"))
    assert caught.value.status_code == 507
    staged = list((tmp_path / "quota").glob(".upload-*.png"))
    assert len(staged) == 1
    assert not list((tmp_path / "quota").glob(".upload-*.part"))
    assert not [
        path
        for path in (tmp_path / "quota").glob("*.png")
        if not path.name.startswith(".upload-")
    ]


def test_unmarked_staged_lookalike_counts_toward_quota_but_is_not_deleted(tmp_path):
    directory = tmp_path / "crash-quota"
    directory.mkdir()
    orphan = directory / (".upload-" + ("a" * 32) + ".png")
    orphan.write_bytes(b"occupied")
    store = _UploadStore(
        directory,
        _UploadLimits(
            image_max_bytes=100,
            video_max_bytes=100,
            image_max_pixels=100,
            storage_max_bytes=len(b"occupied"),
            max_files=2,
        ),
    )
    assert store._usage() == (len(b"occupied"), 1)
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    with pytest.raises(HTTPException) as caught:
        run_async(store.save(_multipart_request(encoded.tobytes()), "image"))
    assert caught.value.status_code == 507
    assert orphan.exists()
    store.cleanup_staged(AppConfig())
    # A filename pattern is not an ownership capability.  Restart recovery
    # only removes inodes named by a durable custback transaction record.
    assert orphan.exists()


def test_upload_quota_reservation_is_atomic_under_race(tmp_path):
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    payload = encoded.tobytes()
    store = _UploadStore(
        tmp_path / "race-quota",
        _UploadLimits(
            image_max_bytes=len(payload),
            video_max_bytes=len(payload),
            image_max_pixels=100,
            storage_max_bytes=len(payload),
            max_files=2,
        ),
    )

    def attempt():
        try:
            return run_async(store.save(_multipart_request(payload), "image"))
        except HTTPException as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result(5.0)
            for future in (pool.submit(attempt), pool.submit(attempt))
        ]
    assert sorted(
        201 if not isinstance(result, HTTPException) else result.status_code
        for result in results
    ) == [201, 507]
    assert len(list((tmp_path / "race-quota").glob("*.png"))) == 1


def test_upload_store_maps_disk_full_and_cleans_reservations(tmp_path, monkeypatch):
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    payload = encoded.tobytes()
    store = _UploadStore(
        tmp_path / "disk",
        _UploadLimits(
            image_max_bytes=len(payload),
            video_max_bytes=len(payload),
            image_max_pixels=100,
            storage_max_bytes=len(payload),
            max_files=1,
        ),
    )
    monkeypatch.setattr(
        server_mod.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")),
    )
    with pytest.raises(HTTPException) as caught:
        run_async(store.save(_multipart_request(payload), "image"))
    assert caught.value.status_code == 507
    assert store._reserved_files == 0
    assert store._reserved_bytes == 0


def test_upload_store_closes_raw_fd_when_private_mode_setup_fails(
    tmp_path, monkeypatch
):
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    payload = encoded.tobytes()
    store = _UploadStore(
        tmp_path / "fd-failure",
        _UploadLimits(
            image_max_bytes=len(payload),
            video_max_bytes=len(payload),
            image_max_pixels=100,
            storage_max_bytes=len(payload),
            max_files=1,
        ),
    )
    opened = []

    def fail_fchmod(descriptor, _mode):
        opened.append(descriptor)
        raise PermissionError(errno.EPERM, "mode change denied")

    monkeypatch.setattr(server_mod.os, "fchmod", fail_fchmod)
    with pytest.raises(HTTPException) as caught:
        run_async(store.save(_multipart_request(payload), "image"))
    assert caught.value.status_code == 507
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(opened[0])
    assert closed.value.errno == errno.EBADF
    assert store._reserved_files == 0
    assert store._reserved_bytes == 0


def test_upload_fsync_does_not_block_the_event_loop(tmp_path, monkeypatch):
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    payload = encoded.tobytes()
    store = _UploadStore(
        tmp_path / "nonblocking-fsync",
        _UploadLimits(
            image_max_bytes=len(payload),
            video_max_bytes=len(payload),
            image_max_pixels=100,
            storage_max_bytes=len(payload),
            max_files=1,
        ),
    )
    entered = threading.Event()
    release = threading.Event()
    original_fsync = server_mod.os.fsync

    def slow_fsync(descriptor):
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError("the event loop could not release the fsync worker")
        return original_fsync(descriptor)

    monkeypatch.setattr(server_mod.os, "fsync", slow_fsync)

    async def scenario():
        upload = asyncio.create_task(store.save(_multipart_request(payload), "image"))
        try:
            deadline = time.monotonic() + 1.0
            while not entered.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert entered.is_set()
            # Reaching another scheduling point while fsync is stalled proves
            # the synchronous disk operation is not running on the ASGI loop.
            await asyncio.sleep(0.01)
            release.set()
            return await upload
        finally:
            release.set()

    saved = run_async(scenario())
    assert saved.size == len(payload)


@pytest.mark.parametrize("stage", ("reserve", "write", "fsync", "commit"))
def test_upload_cancellation_waits_for_mutating_worker_and_cleans_ownership(
    tmp_path, monkeypatch, stage
):
    ok, encoded = cv2.imencode(".png", np.zeros((2, 2, 3), np.uint8))
    assert ok
    payload = encoded.tobytes()
    directory = tmp_path / f"cancel-{stage}"
    store = _UploadStore(
        directory,
        _UploadLimits(
            image_max_bytes=len(payload),
            video_max_bytes=len(payload),
            image_max_pixels=100,
            storage_max_bytes=len(payload),
            max_files=1,
        ),
    )
    entered = threading.Event()
    release = threading.Event()
    if stage == "fsync":
        owner, attribute = server_mod.os, "fsync"
    else:
        owner = store
        attribute = {
            "reserve": "_reserve_file",
            "write": "_reserve_bytes",
            "commit": "_commit",
        }[stage]
    original = getattr(owner, attribute)

    def blocked(*args, **kwargs):
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError(f"cancelled upload did not release {stage} worker")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, attribute, blocked)

    async def scenario():
        upload = asyncio.create_task(store.save(_multipart_request(payload), "image"))
        try:
            deadline = time.monotonic() + 1.0
            while not entered.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert entered.is_set(), f"upload did not reach {stage} worker"
            upload.cancel()
            await asyncio.sleep(0.02)
            # Cancellation is deferred until the authoritative worker finishes;
            # otherwise finally can race its quota and filesystem mutations.
            assert not upload.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await upload
        finally:
            release.set()

    run_async(scenario())
    assert store._reserved_files == 0
    assert store._reserved_bytes == 0
    assert store._active_temps == set()
    assert list(directory.iterdir()) == []


def test_failed_upload_unlink_keeps_ownership_until_retry(tmp_path, monkeypatch):
    directory = tmp_path / "cleanup-retry"
    store = _UploadStore(directory, _UploadLimits())
    store.ensure_directory()
    temporary = directory / ".upload-retry.part"
    store._reserve_file(temporary)
    temporary.write_bytes(b"owned staging bytes")
    store._bind_created(temporary)
    store._reserve_bytes(temporary.stat().st_size)
    size = temporary.stat().st_size
    original_unlink = Path.unlink
    failures = 3

    def fail_first_cleanup_attempts(path, *args, **kwargs):
        nonlocal failures
        if (
            path.parent == directory
            and path.name.startswith(".custback-cleanup-")
            and failures
        ):
            failures -= 1
            raise PermissionError(errno.EACCES, "temporary unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_cleanup_attempts)
    store._cleanup_save(None, temporary, None, size, True)

    pending_paths = store._cleanup_pending[temporary][0]
    assert len(pending_paths) == 1 and pending_paths[0].exists()
    assert not temporary.exists()
    assert temporary in store._cleanup_pending
    assert pending_paths[0] in store._active_temps
    assert store._reserved_files == 1
    assert store._reserved_bytes == size

    store._retry_pending_cleanup()
    assert not pending_paths[0].exists()
    assert store._cleanup_pending == {}
    assert store._active_temps == set()
    assert store._reserved_files == 0
    assert store._reserved_bytes == 0


def test_image_header_rejection_happens_before_opencv_decode(tmp_path, monkeypatch):
    path = tmp_path / "forged.png"
    path.write_bytes(b"not a real image")
    store = _UploadStore(tmp_path, _UploadLimits())
    monkeypatch.setattr(
        server_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCV must not see a Pillow-rejected image")
        ),
    )
    with pytest.raises(HTTPException) as caught:
        store._validate(path, "image")
    assert caught.value.status_code == 422


def test_shared_image_decoder_rejects_before_opencv(tmp_path, monkeypatch):
    path = tmp_path / "corrupt-compressed.png"
    path.write_bytes(b"container accepted by the test decoder")
    calls = 0

    def reject_image(*_args):
        nonlocal calls
        calls += 1
        raise server_mod.ColorError("invalid image")

    store = _UploadStore(tmp_path, _UploadLimits(image_max_pixels=100))
    monkeypatch.setattr(server_mod, "decode_image_to_srgb_bgr", reject_image)
    monkeypatch.setattr(
        server_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCV must not see corrupt compressed pixels")
        ),
    )

    with pytest.raises(HTTPException) as caught:
        store._validate(path, "image")
    assert caught.value.status_code == 422
    assert _error_code(caught.value) == "invalid_media"
    assert calls == 1


def test_image_dimension_limit_is_checked_before_opencv_decode(tmp_path, monkeypatch):
    path = tmp_path / "large.png"
    assert cv2.imwrite(str(path), np.zeros((20, 20, 3), np.uint8))
    store = _UploadStore(
        tmp_path,
        _UploadLimits(image_max_pixels=100),
    )
    monkeypatch.setattr(
        server_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dimension rejection must precede OpenCV decode")
        ),
    )
    with pytest.raises(HTTPException) as caught:
        store._validate(path, "image")
    assert caught.value.status_code == 422
    assert _error_code(caught.value) == "image_dimensions_exceeded"


def test_pillow_decompression_bomb_is_mapped_without_opencv_decode(
    tmp_path, monkeypatch
):
    path = tmp_path / "bomb.png"
    path.write_bytes(b"forged image header")
    store = _UploadStore(tmp_path, _UploadLimits())
    monkeypatch.setattr(
        server_mod,
        "decode_image_to_srgb_bgr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            server_mod.ColorError(
                f"image exceeds {store.limits.image_max_pixels} pixels"
            )
        ),
    )
    monkeypatch.setattr(
        server_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCV must not decode a decompression bomb")
        ),
    )
    with pytest.raises(HTTPException) as caught:
        store._validate(path, "image")
    assert caught.value.status_code == 422
    assert _error_code(caught.value) == "image_dimensions_exceeded"


def test_websocket_jpeg_header_bomb_is_rejected_before_opencv(monkeypatch):
    monkeypatch.setattr(
        server_mod.Image,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            server_mod.Image.DecompressionBombError("too many pixels")
        ),
    )
    monkeypatch.setattr(
        server_mod.cv2,
        "imdecode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCV must not decode a JPEG header bomb")
        ),
    )
    assert server_mod._decode_jpeg(b"\xff\xd8forged", (128, 72)) is None


def test_video_metadata_limit_is_checked_before_first_frame_decode(
    tmp_path, monkeypatch
):
    path = tmp_path / "forged.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypisomcontainer")

    class Capture:
        read_called = False
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == server_mod.cv2.CAP_PROP_FRAME_WIDTH:
                return 4096.0
            if prop == server_mod.cv2.CAP_PROP_FRAME_HEIGHT:
                return 2160.0
            return 0.0

        def read(self):
            self.read_called = True
            return True, np.zeros((1, 1, 3), np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(server_mod.cv2, "VideoCapture", lambda _path: capture)
    store = _UploadStore(tmp_path, _UploadLimits(video_max_width=3840))
    with pytest.raises(HTTPException) as caught:
        store._validate(path, "video")
    assert caught.value.status_code == 422
    assert _error_code(caught.value) == "video_dimensions_exceeded"
    assert not capture.read_called
    assert capture.released


def test_video_later_frame_limit_rejects_the_whole_upload(tmp_path, monkeypatch):
    path = tmp_path / "changing.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypisomcontainer")

    class Capture:
        def __init__(self):
            self.frames = iter(
                [
                    np.zeros((4, 6, 3), np.uint8),
                    np.zeros((5, 7, 3), np.uint8),
                ]
            )
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == server_mod.cv2.CAP_PROP_FRAME_WIDTH:
                return 6.0
            if prop == server_mod.cv2.CAP_PROP_FRAME_HEIGHT:
                return 4.0
            return 0.0

        def read(self):
            try:
                return True, next(self.frames)
            except StopIteration:
                return False, None

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(server_mod.cv2, "VideoCapture", lambda _path: capture)
    store = _UploadStore(tmp_path, _UploadLimits(video_max_width=6, video_max_height=4))
    with pytest.raises(HTTPException) as caught:
        store._validate(path, "video")
    assert caught.value.status_code == 422
    assert _error_code(caught.value) == "video_dimensions_exceeded"
    assert capture.released


def _shutdown_stack(stack, on_shutdown):
    """A second app over the fixture's live pipeline with a shutdown channel."""
    security = SecurityPolicy.for_bind(
        TOKEN,
        "testserver",
        80,
        allowed_origins=[ORIGIN],
        extra_hosts=["testserver"],
        renderer_token=RENDERER_TOKEN,
    )
    app = create_app(
        stack.runtime,
        stack.hub,
        stack.pipeline,
        security=security,
        upload_dir=stack.upload_dir,
        on_shutdown=on_shutdown,
    )
    return Stack(app, stack.runtime, stack.hub, stack.pipeline, stack.upload_dir)


def test_lifecycle_shutdown_requires_bearer_and_invokes_handler(stack):
    # WIN-5.3: the supervising shell's private lifecycle channel.
    calls = []
    wired = _shutdown_stack(stack, lambda: calls.append(True))

    unauth = wired.post("/lifecycle/shutdown")
    assert unauth.status_code == 401
    assert calls == []

    ok = wired.post("/lifecycle/shutdown", headers=AUTH)
    assert ok.status_code == 202
    assert calls == [True]


def test_lifecycle_shutdown_rejects_browser_session(stack):
    # A WebView holds only the HttpOnly session cookie, never the bearer; it
    # must not be able to stop the engine.
    calls = []
    wired = _shutdown_stack(stack, lambda: calls.append(True))
    issued = wired.post("/auth/session", json={"token": TOKEN})
    assert issued.status_code == 204
    session = issued.cookies.get(SESSION_COOKIE)
    assert session
    denied = wired.post(
        "/lifecycle/shutdown", headers={"Cookie": f"{SESSION_COOKIE}={session}"}
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "forbidden"
    assert calls == []


def test_lifecycle_shutdown_unavailable_without_supervisor(stack):
    # The default (source/signal-driven) app wires no handler; the route then
    # reports the channel is unavailable rather than crashing or 404-ing.
    resp = stack.post("/lifecycle/shutdown", headers=AUTH)
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "shutdown_unavailable"
