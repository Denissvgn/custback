"""The /avatar/* reverse proxy: auth boundary, forwarding, and failures."""

import asyncio
import contextlib
import io
import ssl
import zipfile
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("fastapi")
cv2 = pytest.importorskip("cv2")

import httpx
import starlette
from fastapi import FastAPI

if int(starlette.__version__.split(".", 1)[0]) >= 1:
    from httpx2 import ASGITransport as _ASGITransport
    from httpx2 import AsyncClient as _AsyncClient
else:  # Starlette < 1 uses the original httpx client contract.
    from httpx import ASGITransport as _ASGITransport
    from httpx import AsyncClient as _AsyncClient

import custback.api.avatar_proxy as avatar_proxy_module
from custback.api.avatar_proxy import register_avatar_proxy
from custback.api.security import SecurityPolicy
from custback.api.server import create_app
from custback.avatar.api import create_avatar_app
from custback.avatar.config import AvatarConfig, AvatarRuntime
from custback.avatar.service import AvatarService
from custback.config import AppConfig, RuntimeConfig
from custback.hub import FrameHub
from custback.pipeline import Pipeline

CORE_TOKEN = "core-api-test-token-which-is-long-enough-000"
AVATAR_TOKEN = "avatar-api-test-token-which-is-long-enough-1"
AUTH = {"Authorization": f"Bearer {CORE_TOKEN}"}
ORIGIN = "http://testserver"


def _async_client(app: object) -> Any:
    """Return a client across the incompatible httpx/httpx2 transport types."""

    transport = cast(Any, _ASGITransport)(app=app)
    return cast(Any, _AsyncClient)(
        transport=transport,
        base_url="http://testserver",
    )


async def _with_event_loop_heartbeat(awaitable):
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


def layer_png() -> bytes:
    layer = np.zeros((96, 64, 4), dtype=np.uint8)
    layer[8:-8, 8:-8] = (90, 140, 200, 255)
    ok, data = cv2.imencode(".png", layer)
    assert ok
    return data.tobytes()


def rig_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("torso.png", layer_png())
        archive.writestr("head.png", layer_png())
    return buffer.getvalue()


@dataclass
class Stack:
    app: object
    runtime: RuntimeConfig
    pipeline: Pipeline
    avatar_runtime: AvatarRuntime
    avatar_service: AvatarService

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


def make_stack(tmp_path, monkeypatch, avatar_url="https://avatar-host:8711"):
    monkeypatch.setenv("CUSTBACK_AVATAR_API_TOKEN", AVATAR_TOKEN)
    avatar_runtime = AvatarRuntime(
        AvatarConfig.from_dict(
            {
                "driver": {"backend": "idle"},
                "storage": {
                    "rigs_dir": str(tmp_path / "rigs"),
                    "backgrounds_dir": str(tmp_path / "avatar-media"),
                },
            }
        )
    )
    avatar_service = AvatarService(avatar_runtime)
    avatar_security = SecurityPolicy.for_bind(
        AVATAR_TOKEN,
        "avatar-host",
        8711,
        allowed_origins=[],
        extra_hosts=["avatar-host"],
    )
    avatar_app = create_avatar_app(
        avatar_runtime, avatar_service, security=avatar_security
    )

    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 128, "height": 72, "fps": 60},
            "background": {"mode": "color", "color": [200, 30, 30]},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null", "fps": 60},
            "avatar": {"url": avatar_url},
        }
    )
    runtime = RuntimeConfig(cfg)
    hub = FrameHub()
    pipeline = Pipeline(runtime, hub)
    pipeline.start()
    security = SecurityPolicy.for_bind(
        CORE_TOKEN,
        "testserver",
        80,
        allowed_origins=[ORIGIN],
        extra_hosts=["testserver"],
    )

    def avatar_client_factory():
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=avatar_app))

    app = create_app(
        runtime,
        hub,
        pipeline,
        security=security,
        upload_dir=tmp_path / "uploads",
        avatar_client_factory=avatar_client_factory,
    )
    return Stack(app, runtime, pipeline, avatar_runtime, avatar_service)


@pytest.fixture()
def stack(tmp_path, monkeypatch):
    built = make_stack(tmp_path, monkeypatch)
    yield built
    try:
        built.pipeline.stop()
    finally:
        built.avatar_service.close()


def test_proxy_requires_core_auth(stack):
    response = stack.get("/avatar/status")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_proxy_forwards_status_and_config(stack):
    response = stack.get("/avatar/status", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["connected"] is False

    response = stack.get("/avatar/config", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["appearance"]["rig"] == "builtin"

    response = stack.patch(
        "/avatar/config",
        headers={**AUTH, "content-type": "application/json"},
        json={"appearance": {"scale": 0.5}},
    )
    assert response.status_code == 200
    assert response.headers["x-config-version"] == "1"
    assert stack.avatar_runtime.read().config.appearance.scale == 0.5


def test_proxy_forwards_driver_modes(stack):
    response = stack.get("/avatar/avatars", headers=AUTH)
    assert response.status_code == 200
    modes = {mode["id"]: mode for mode in response.json()["modes"]}
    assert modes["motion"]["backend"] == "vision"
    assert modes["voice"]["backend"] == "audio2face"


def test_proxy_streams_rig_upload_and_thumbnail(stack):
    response = stack.post(
        "/avatar/rigs",
        params={"name": "viarig"},
        content=rig_zip(),
        headers={**AUTH, "content-type": "application/zip"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["name"] == "viarig"

    response = stack.get("/avatar/rigs/viarig/thumbnail.jpg", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")

    assert stack.delete("/avatar/rigs/viarig", headers=AUTH).status_code == 204


def test_proxy_rejects_unknown_paths(stack):
    response = stack.get("/avatar/auth/session", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "unknown_avatar_path"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/avatar/status"),
        ("DELETE", "/avatar/config"),
        ("PATCH", "/avatar/status"),
        ("GET", "/avatar/config/internal"),
        ("POST", "/avatar/backgrounds/custom"),
        ("GET", "/avatar/rigs/name/private"),
    ],
)
def test_proxy_rejects_unpublished_method_path_pairs(stack, method, path):
    response = stack.request(method, path, headers=AUTH)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "unknown_avatar_path"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "status"),
        ("GET", "avatars"),
        ("GET", "avatars/casey/thumbnail.jpg"),
        ("GET", "rigs"),
        ("POST", "rigs"),
        ("DELETE", "rigs/custom"),
        ("GET", "rigs/custom/thumbnail.jpg"),
        ("GET", "backgrounds"),
        ("POST", "backgrounds/image"),
        ("POST", "backgrounds/video"),
        ("DELETE", "backgrounds/office.png"),
        ("GET", "backgrounds/office.png/thumbnail.jpg"),
        ("GET", "config"),
        ("PATCH", "config"),
        ("GET", "video/snapshot.jpg"),
        ("GET", "video/mjpeg"),
    ],
)
def test_proxy_route_contract_includes_each_public_avatar_operation(method, path):
    assert avatar_proxy_module._is_allowed_avatar_route(method, path)


@pytest.mark.parametrize(
    "path",
    [
        "avatars/../thumbnail.jpg",
        "avatars/%2e%2e/thumbnail.jpg",
        "avatars/casey?/thumbnail.jpg",
        "avatars/casey#/thumbnail.jpg",
        r"avatars/casey\escape/thumbnail.jpg",
        "rigs/.hidden",
        "backgrounds/../thumbnail.jpg",
    ],
)
def test_proxy_route_contract_rejects_ambiguous_parameter_segments(path):
    assert not avatar_proxy_module._is_allowed_avatar_route("GET", path)


@pytest.mark.parametrize(
    "path",
    [
        "/avatar/avatars/%252e%252e/thumbnail.jpg",
        "/avatar/avatars/casey%3Fignored/thumbnail.jpg",
        "/avatar/avatars/casey%23ignored/thumbnail.jpg",
        "/avatar/avatars/casey%5Cescape/thumbnail.jpg",
    ],
)
def test_encoded_proxy_path_ambiguity_performs_no_upstream_io(path, monkeypatch):
    monkeypatch.setenv("CUSTBACK_AVATAR_API_TOKEN", AVATAR_TOKEN)
    runtime = RuntimeConfig(
        AppConfig.from_dict({"avatar": {"url": "https://trusted-avatar.example:8711"}})
    )
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200)

    app = _proxy_only_app(runtime, handler=handler)
    response = run_async(_request_proxy_app(app, path))

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "unknown_avatar_path"
    assert calls == []


def test_proxy_does_not_leak_core_credentials(stack):
    # The avatar app rejects the core Bearer token; the proxy must inject
    # the avatar token instead, so this request succeeds end to end.
    response = stack.get("/avatar/status", headers=AUTH)
    assert response.status_code == 200
    # And the avatar app itself refuses the core token directly.
    direct = run_async(_direct_avatar_request(stack))
    assert direct == 401


async def _direct_avatar_request(stack) -> int:
    factory_client = getattr(stack.app.state, "avatar_proxy_client", None)
    assert factory_client is not None
    response = await factory_client.get("https://avatar-host:8711/status", headers=AUTH)
    return response.status_code


def test_unconfigured_proxy_returns_503(tmp_path, monkeypatch):
    stack = make_stack(tmp_path, monkeypatch, avatar_url="")
    try:
        response = stack.get("/avatar/status", headers=AUTH)
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "avatar_unconfigured"
    finally:
        stack.pipeline.stop()


def test_avatar_destination_and_token_path_require_restart(stack):
    response = stack.patch(
        "/config",
        headers=AUTH,
        json={
            "avatar": {
                "url": "https://other-host:9000",
                "token_file": "/tmp/other-avatar-token",
                "tls_ca_file": "/tmp/other-avatar-ca.pem",
                "tls_certfile": "/tmp/other-avatar-client.pem",
                "tls_keyfile": "/tmp/other-avatar-client.key",
            }
        },
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "restart_required"
    assert set(response.json()["detail"]["fields"]) == {
        "avatar.url",
        "avatar.token_file",
        "avatar.tls_ca_file",
        "avatar.tls_certfile",
        "avatar.tls_keyfile",
    }
    assert stack.runtime.read().config.avatar.url == "https://avatar-host:8711"


def test_avatar_url_validation():
    with pytest.raises(ValueError):
        AppConfig.from_dict({"avatar": {"url": "ftp://nope:21"}})
    with pytest.raises(ValueError):
        AppConfig.from_dict({"avatar": {"url": "http://user:pw@host:8711"}})
    with pytest.raises(ValueError):
        AppConfig.from_dict({"avatar": {"url": "http://host:8711/path"}})
    with pytest.raises(ValueError):
        AppConfig.from_dict({"avatar": {"url": "http://avatar.example:8711"}})
    loopback = AppConfig.from_dict({"avatar": {"url": "http://127.0.0.1:8711"}})
    assert loopback.avatar.url == "http://127.0.0.1:8711"
    cfg = AppConfig.from_dict({"avatar": {"url": "https://gb10.local:8711/"}})
    assert cfg.avatar.url == "https://gb10.local:8711"


def test_avatar_proxy_tls_configuration_is_structural_and_secure(tmp_path):
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("placeholder")
    certfile = tmp_path / "client.pem"
    certfile.write_text("placeholder")
    keyfile = tmp_path / "client.key"
    keyfile.write_text("placeholder")

    with pytest.raises(ValueError, match="together"):
        AppConfig.from_dict(
            {
                "avatar": {
                    "url": "https://avatar.example:8711",
                    "tls_certfile": str(certfile),
                }
            }
        )
    with pytest.raises(ValueError, match="configured secure endpoint"):
        AppConfig.from_dict({"avatar": {"url": "", "tls_ca_file": str(ca_file)}})
    with pytest.raises(ValueError, match="plaintext"):
        AppConfig.from_dict(
            {
                "avatar": {
                    "url": "http://127.0.0.1:8711",
                    "tls_ca_file": str(ca_file),
                }
            }
        )
    configured = AppConfig.from_dict(
        {
            "avatar": {
                "url": "https://avatar.example:8711",
                "tls_ca_file": str(ca_file),
                "tls_certfile": str(certfile),
                "tls_keyfile": str(keyfile),
            }
        }
    )
    assert configured.avatar.tls_ca_file == str(ca_file)

    missing_ca = tmp_path / "not-created-ca.pem"
    deferred = AppConfig.from_dict(
        {
            "avatar": {
                "url": "https://avatar.example:8711",
                "tls_ca_file": str(missing_ca),
            }
        }
    )
    with pytest.raises(ValueError, match="does not exist"):
        avatar_proxy_module._build_proxy_target(deferred.avatar)


def test_default_proxy_client_uses_immutable_tls_context(monkeypatch):
    monkeypatch.setenv("CUSTBACK_AVATAR_API_TOKEN", AVATAR_TOKEN)
    default_ca = ssl.get_default_verify_paths().cafile
    if not default_ca:
        pytest.skip("the test interpreter has no default CA bundle")
    cfg = AppConfig.from_dict(
        {
            "avatar": {
                "url": "https://avatar.example:8711",
                "tls_ca_file": default_ca,
            }
        }
    ).avatar
    target = avatar_proxy_module._build_proxy_target(cfg)
    assert isinstance(target.verify, ssl.SSLContext)
    assert target.verify.check_hostname is True
    assert target.verify.verify_mode == ssl.CERT_REQUIRED
    captured = {}

    class CapturingClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(avatar_proxy_module.httpx, "AsyncClient", CapturingClient)
    client = avatar_proxy_module._default_proxy_client(target)

    assert isinstance(client, CapturingClient)
    assert captured == {
        "verify": target.verify,
        "follow_redirects": False,
        "trust_env": False,
    }


async def _request_proxy_app(app: FastAPI, path: str = "/avatar/status"):
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(path)
    finally:
        proxy_client = getattr(app.state, "avatar_proxy_client", None)
        if proxy_client is not None and not proxy_client.is_closed:
            await proxy_client.aclose()


def _proxy_only_app(runtime, handler=None, client_factory=None):
    app = FastAPI()
    if client_factory is None:
        assert handler is not None

        def mock_client_factory():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        client_factory = mock_client_factory

    register_avatar_proxy(app, runtime, client_factory=client_factory)
    return app


class _AsyncBody(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content

    async def __aiter__(self):
        yield self.content


def test_proxy_freezes_destination_and_credential_at_registration(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CUSTBACK_AVATAR_API_TOKEN", raising=False)
    trusted_token = tmp_path / "trusted-avatar-token"
    trusted_token.write_text(AVATAR_TOKEN + "\n")
    trusted_token.chmod(0o600)
    attacker_token = tmp_path / "core-api-token"
    attacker_token.write_text(CORE_TOKEN + "\n")
    attacker_token.chmod(0o600)
    runtime = RuntimeConfig(
        AppConfig.from_dict(
            {
                "avatar": {
                    "url": "https://trusted-avatar.example:8711",
                    "token_file": str(trusted_token),
                }
            }
        )
    )
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=_AsyncBody(b'{"ok":true}'))

    app = _proxy_only_app(runtime, handler=handler)
    base = runtime.read()
    runtime._coordinator_writer().commit(
        base.config.patched(
            {
                "avatar": {
                    "url": "https://attacker.example:443",
                    "token_file": str(attacker_token),
                }
            }
        ),
        base.version,
    )
    # Removing the original file proves requests use the startup credential
    # snapshot and perform no request-time credential file read.
    trusted_token.unlink()

    response = run_async(_request_proxy_app(app))

    assert response.status_code == 200
    assert len(seen) == 1
    assert str(seen[0].url) == "https://trusted-avatar.example:8711/status"
    assert seen[0].headers["authorization"] == f"Bearer {AVATAR_TOKEN}"


def test_proxy_missing_client_token_is_not_created(tmp_path, monkeypatch):
    monkeypatch.delenv("CUSTBACK_AVATAR_API_TOKEN", raising=False)
    missing = tmp_path / "missing" / "avatar-token"
    runtime = RuntimeConfig(
        AppConfig.from_dict(
            {
                "avatar": {
                    "url": "https://trusted-avatar.example:8711",
                    "token_file": str(missing),
                }
            }
        )
    )
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200)

    app = _proxy_only_app(runtime, handler=handler)
    response = run_async(_request_proxy_app(app))

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "avatar_token_unavailable"
    assert not missing.exists()
    assert not missing.parent.exists()
    assert calls == []


@pytest.mark.parametrize("upstream_status", [401, 403])
def test_proxy_maps_upstream_auth_rejection_without_forwarding_body(
    upstream_status, monkeypatch
):
    monkeypatch.setenv("CUSTBACK_AVATAR_API_TOKEN", AVATAR_TOKEN)
    runtime = RuntimeConfig(
        AppConfig.from_dict({"avatar": {"url": "https://trusted-avatar.example:8711"}})
    )
    upstream_responses = []

    def handler(_request):
        response = httpx.Response(
            upstream_status,
            content=b'upstream secret: {"internal":"detail"}',
            headers={"content-type": "text/plain", "x-config-version": "99"},
        )
        upstream_responses.append(response)
        return response

    app = _proxy_only_app(runtime, handler=handler)
    response = run_async(_request_proxy_app(app))

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "avatar_auth_failed"
    assert "upstream secret" not in response.text
    assert "x-config-version" not in response.headers
    assert upstream_responses[0].is_closed


def test_proxy_maps_request_construction_error(monkeypatch):
    monkeypatch.setenv("CUSTBACK_AVATAR_API_TOKEN", AVATAR_TOKEN)
    runtime = RuntimeConfig(
        AppConfig.from_dict({"avatar": {"url": "https://trusted-avatar.example:8711"}})
    )

    class InvalidBuildClient:
        is_closed = False

        def build_request(self, *_args, **_kwargs):
            raise httpx.InvalidURL("invalid test URL")

        async def aclose(self):
            self.is_closed = True

    app = _proxy_only_app(runtime, client_factory=InvalidBuildClient)
    response = run_async(_request_proxy_app(app))

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "avatar_request_invalid"


def test_proxy_never_follows_upstream_redirects(monkeypatch):
    monkeypatch.setenv("CUSTBACK_AVATAR_API_TOKEN", AVATAR_TOKEN)
    runtime = RuntimeConfig(
        AppConfig.from_dict({"avatar": {"url": "https://trusted-avatar.example:8711"}})
    )
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(
            302,
            headers={"location": "https://attacker.example/capture"},
            stream=_AsyncBody(b""),
        )

    def redirecting_client_factory():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        )

    app = _proxy_only_app(runtime, client_factory=redirecting_client_factory)
    response = run_async(_request_proxy_app(app))

    assert response.status_code == 302
    assert calls == ["https://trusted-avatar.example:8711/status"]
    assert "location" not in response.headers


def test_proxy_stream_closes_upstream_when_sending_headers_fails():
    closed = False
    body_entered = False

    async def body():
        nonlocal body_entered
        body_entered = True
        yield b"never sent"

    async def close():
        nonlocal closed
        closed = True

    async def scenario():
        response = avatar_proxy_module._ClosingStreamingResponse(body(), close=close)
        blocked_receive = asyncio.Event()

        async def receive():
            await blocked_receive.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            assert message["type"] == "http.response.start"
            raise RuntimeError("header transport failed")

        with pytest.raises(BaseException):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.3"}},
                receive,
                send,
            )

    asyncio.run(scenario())
    assert body_entered is False
    assert closed is True
