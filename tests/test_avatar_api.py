"""Avatar control-plane API: authentication boundary and config PATCH."""

import asyncio
import contextlib
import io
import ssl
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
cv2 = pytest.importorskip("cv2")

import starlette

if int(starlette.__version__.split(".", 1)[0]) >= 1:
    from httpx2 import ASGITransport, AsyncClient
else:  # Starlette < 1 uses the original httpx client contract.
    from httpx import ASGITransport, AsyncClient

import custback.avatar.api as avatar_api_mod
from custback.api.security import SecurityPolicy
from custback.avatar.api import create_avatar_app, resolve_avatar_api_token
from custback.avatar.config import AvatarConfig, AvatarRuntime
from custback.avatar.service import AvatarService

TOKEN = "avatar-api-test-token-which-is-long-enough-0123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ORIGIN = "http://testserver"


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


@dataclass
class Stack:
    app: object
    runtime: AvatarRuntime
    service: AvatarService

    async def arequest(self, method: str, path: str, **kwargs):
        transport = ASGITransport(app=self.app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)

    def request(self, method: str, path: str, **kwargs):
        return run_async(self.arequest(method, path, **kwargs))

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def patch(self, path: str, **kwargs):
        return self.request("PATCH", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.request("DELETE", path, **kwargs)


def layer_png() -> bytes:
    layer = np.zeros((96, 64, 4), dtype=np.uint8)
    layer[8:-8, 8:-8] = (90, 140, 200, 255)
    ok, data = cv2.imencode(".png", layer)
    assert ok
    return data.tobytes()


def rig_zip(members: dict[str, bytes] | None = None) -> bytes:
    if members is None:
        members = {"torso.png": layer_png(), "head.png": layer_png()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def image_png() -> bytes:
    frame = np.full((120, 160, 3), (30, 90, 160), dtype=np.uint8)
    ok, data = cv2.imencode(".png", frame)
    assert ok
    return data.tobytes()


@pytest.fixture()
def stack(tmp_path):
    runtime = AvatarRuntime(
        AvatarConfig.from_dict(
            {
                "driver": {"backend": "idle"},
                "storage": {
                    "rigs_dir": str(tmp_path / "rigs"),
                    "backgrounds_dir": str(tmp_path / "media"),
                    "rig_zip_max_bytes": 65536,
                },
            }
        )
    )
    service = AvatarService(runtime)
    security = SecurityPolicy.for_bind(
        TOKEN,
        "testserver",
        80,
        allowed_origins=[ORIGIN],
        extra_hosts=["testserver"],
    )
    app = create_avatar_app(runtime, service, security=security)
    stack = Stack(app, runtime, service)
    try:
        yield stack
    finally:
        service.close()


def test_all_routes_require_bearer_auth(stack):
    for path in (
        "/status", "/config", "/avatars", "/video/snapshot.jpg", "/openapi.json",
        "/rigs", "/backgrounds", "/avatars/casey/thumbnail.jpg",
    ):
        response = stack.get(path)
        assert response.status_code == 401, path
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.headers["x-config-version"] == "0"
    response = stack.patch("/config", json={"appearance": {"scale": 0.4}})
    assert response.status_code == 401
    assert stack.runtime.read().config.appearance.scale == 1.0


def test_foreign_origin_is_rejected(stack):
    response = stack.get(
        "/status", headers={**AUTH, "Origin": "http://evil.example"}
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "forbidden_origin"


def test_status_reports_service_stats(stack):
    response = stack.get("/status", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is False
    assert body["frames_sent"] == 0
    assert body["config_version"] == 0
    assert "uptime_s" in body


def test_avatar_options_are_listed_for_pickers(stack):
    response = stack.get("/avatars", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["avatars"] == ["casey", "robin", "alex", "nova"]
    assert body["styles"] == ["cartoon", "realistic", "sketch"]
    assert body["framings"] == ["full", "bust", "closeup"]
    assert body["parts"][0] == "torso"
    # Every advertised option is accepted by a live PATCH.
    response = stack.patch(
        "/config",
        headers=AUTH,
        json={
            "appearance": {
                "avatar": body["avatars"][-1],
                "style": body["styles"][-1],
                "framing": body["framings"][-1],
            }
        },
    )
    assert response.status_code == 200
    assert response.json()["config"]["appearance"]["avatar"] == "nova"


def test_get_and_patch_config_with_versions(stack):
    response = stack.get("/config", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["x-config-version"] == "0"
    assert response.json()["appearance"]["scale"] == 1.0
    assert {
        "token_file",
        "tls_ca_file",
        "tls_certfile",
        "tls_keyfile",
    }.isdisjoint(response.json()["source"])
    assert {"token_file", "tls_certfile", "tls_keyfile"}.isdisjoint(
        response.json()["api"]
    )
    assert {"tls_ca_file", "tls_certfile", "tls_keyfile"}.isdisjoint(
        response.json()["driver"]["audio2face"]
    )

    schema = stack.get("/openapi.json", headers=AUTH).json()
    schemas = schema["components"]["schemas"]
    assert schema["paths"]["/config"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/PublicAvatarConfig")
    assert "token_file" not in schemas["PublicAvatarSourceConfig"]["properties"]
    assert "tls_keyfile" not in schemas["PublicAudio2FaceConfig"]["properties"]
    assert "token_file" not in schemas["PublicAvatarApiConfig"]["properties"]

    response = stack.patch(
        "/config",
        headers=AUTH,
        json={"appearance": {"scale": 0.6, "parts": ["head", "eyes"]}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["config_version"] == 1
    assert body["config"]["appearance"]["parts"] == ["head", "eyes"]
    assert stack.runtime.read().config.appearance.scale == 0.6


def test_patch_restart_sections_return_409(stack):
    response = stack.patch(
        "/config", headers=AUTH, json={"api": {"port": 9999}}
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "restart_required"
    assert "api.port" in detail["fields"]
    assert stack.runtime.version == 0


def test_patch_invalid_values_return_422(stack):
    response = stack.patch(
        "/config", headers=AUTH, json={"appearance": {"parts": ["wings"]}}
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_config"
    response = stack.patch("/config", headers=AUTH, content=b"[]")
    assert response.status_code in (415, 422)


def test_noop_patch_activates_startup_generation_or_reports_failure(tmp_path):
    runtime = AvatarRuntime(
        AvatarConfig.from_dict(
            {
                "driver": {"backend": "idle"},
                "background": {
                    "mode": "image",
                    "image_path": str(tmp_path / "missing.png"),
                },
            }
        )
    )
    service = AvatarService(runtime)
    security = SecurityPolicy.for_bind(
        TOKEN,
        "testserver",
        80,
        allowed_origins=[ORIGIN],
        extra_hosts=["testserver"],
    )
    app = create_avatar_app(runtime, service, security=security)
    try:
        response = Stack(app, runtime, service).patch(
            "/config", headers=AUTH, json={}
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "activation_failed"
        assert runtime.version == 0
    finally:
        service.close()


def test_hot_patch_validation_does_not_disclose_operator_tls_paths(tmp_path):
    system_ca = ssl.get_default_verify_paths().cafile
    if not system_ca:
        pytest.skip("the test interpreter has no default CA bundle")
    ca_file = tmp_path / "operator-private" / "renderer-ca.pem"
    ca_file.parent.mkdir()
    ca_file.write_bytes(Path(system_ca).read_bytes())
    runtime = AvatarRuntime(
        AvatarConfig.from_dict(
            {
                "source": {
                    "url": "wss://renderer.example:8710",
                    "tls_ca_file": str(ca_file),
                },
                "driver": {"backend": "idle"},
            }
        )
    )
    service = AvatarService(runtime)
    security = SecurityPolicy.for_bind(
        TOKEN,
        "testserver",
        80,
        allowed_origins=[ORIGIN],
        extra_hosts=["testserver"],
    )
    app = create_avatar_app(runtime, service, security=security)
    ca_file.unlink()

    try:
        response = Stack(app, runtime, service).patch(
            "/config",
            headers=AUTH,
            json={"appearance": {"scale": 0.9}},
        )

        assert response.status_code == 422
        assert "does not exist" in response.text
        assert str(ca_file) not in response.text
    finally:
        service.close()


def test_snapshot_unavailable_then_served(stack):
    response = stack.get("/video/snapshot.jpg", headers=AUTH)
    assert response.status_code == 503
    frame = np.full((12, 16, 3), 200, dtype=np.uint8)
    stack.service.output.put(frame)
    response = stack.get("/video/snapshot.jpg", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_mjpeg_stream_connection_limit_returns_429(stack):
    limiter = stack.app.state.stream_connections
    leases = [limiter.try_acquire() for _ in range(limiter.maximum)]
    assert all(lease is not None for lease in leases)
    try:
        response = stack.get("/video/mjpeg", headers=AUTH)
        assert response.status_code == 429
        assert response.json()["detail"]["code"] == "stream_limit"
    finally:
        for lease in leases:
            lease.release()
    assert limiter.active == 0


def test_mjpeg_stream_encodes_async_frame_and_releases_lease(stack):
    route = next(
        route for route in stack.app.routes if route.path == "/video/mjpeg"
    )
    limiter = stack.app.state.stream_connections

    async def scenario():
        response = await route.endpoint()
        assert limiter.active == 1
        try:
            stack.service.output.put(np.full((12, 16, 3), 123, dtype=np.uint8))
            chunk = await asyncio.wait_for(
                response.body_iterator.__anext__(),
                1.0,
            )
            assert b"Content-Type: image/jpeg" in chunk
            assert b"\xff\xd8" in chunk
        finally:
            await response.body_iterator.aclose()
        assert limiter.active == 0

    run_async(scenario())


def test_avatar_token_env_is_distinct(tmp_path, monkeypatch):
    token_file = tmp_path / "avatar-token"
    monkeypatch.setenv("CUSTBACK_API_TOKEN", "custback-token-should-not-be-used-here!")
    monkeypatch.delenv("CUSTBACK_AVATAR_API_TOKEN", raising=False)
    resolved = resolve_avatar_api_token(str(token_file))
    assert resolved.created  # generated fresh instead of borrowing custback's
    assert resolved.value != "custback-token-should-not-be-used-here!"

    monkeypatch.setenv("CUSTBACK_AVATAR_API_TOKEN", TOKEN)
    assert resolve_avatar_api_token(str(token_file)).value == TOKEN


def test_avatars_reports_rigs_and_driver_modes(stack):
    response = stack.get("/avatars", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["rigs"] == []
    modes = {mode["id"]: mode for mode in body["modes"]}
    assert set(modes) == {"auto", "motion", "voice", "presence"}
    assert modes["motion"]["backend"] == "vision"
    assert modes["voice"]["backend"] == "audio2face"
    assert modes["presence"]["available"] is True
    assert modes["presence"]["active"] is True  # fixture runs the idle backend
    assert modes["voice"]["configured"] is False  # no audio2face.url yet
    for mode in body["modes"]:
        assert set(mode) >= {
            "id", "backend", "label", "available", "reason", "configured", "active"
        }


def test_builtin_thumbnails_served_and_validated(stack):
    response = stack.get("/avatars/casey/thumbnail.jpg", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")
    assert stack.get("/avatars/ghost/thumbnail.jpg", headers=AUTH).status_code == 404
    response = stack.get(
        "/avatars/casey/thumbnail.jpg", params={"style": "neon"}, headers=AUTH
    )
    assert response.status_code == 422


def test_rig_upload_select_and_delete_lifecycle(stack):
    response = stack.post(
        "/rigs",
        params={"name": "myrig"},
        content=rig_zip(),
        headers={**AUTH, "content-type": "application/zip"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["name"] == "myrig"
    assert body["parts"] == ["torso", "head"]

    listed = stack.get("/rigs", headers=AUTH).json()
    assert [rig["name"] for rig in listed["rigs"]] == ["myrig"]
    assert listed["active"] == "builtin"

    response = stack.get("/rigs/myrig/thumbnail.jpg", headers=AUTH)
    assert response.status_code == 200
    assert response.content.startswith(b"\xff\xd8")

    response = stack.patch(
        "/config", headers=AUTH, json={"appearance": {"rig": "myrig"}}
    )
    assert response.status_code == 200
    assert response.json()["config"]["appearance"]["rig"] == "myrig"

    response = stack.delete("/rigs/myrig", headers=AUTH)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "rig_in_use"

    stack.patch("/config", headers=AUTH, json={"appearance": {"rig": "builtin"}})
    assert stack.delete("/rigs/myrig", headers=AUTH).status_code == 204
    assert stack.get("/rigs", headers=AUTH).json()["rigs"] == []


def test_rig_thumbnail_forwards_configured_decode_limits(monkeypatch, stack):
    response = stack.post(
        "/rigs",
        params={"name": "thumbnail-limits"},
        content=rig_zip(),
        headers={**AUTH, "content-type": "application/zip"},
    )
    assert response.status_code == 201, response.text
    observed = []
    original = avatar_api_mod.render_avatar_thumbnail

    def recording_thumbnail(selector, **kwargs):
        observed.append(dict(kwargs))
        return original(selector, **kwargs)

    monkeypatch.setattr(
        avatar_api_mod,
        "render_avatar_thumbnail",
        recording_thumbnail,
    )

    response = stack.get("/rigs/thumbnail-limits/thumbnail.jpg", headers=AUTH)

    assert response.status_code == 200
    storage = stack.runtime.read().config.storage
    assert observed == [
        {
            "style": "cartoon",
            "rig_layer_max_pixels": storage.rig_layer_max_pixels,
            "rig_total_max_pixels": storage.rig_total_max_pixels,
            "rig_manifest_max_bytes": storage.rig_manifest_max_bytes,
        }
    ]


def test_rig_upload_rejections(stack):
    response = stack.post(
        "/rigs",
        params={"name": "Bad Name"},
        content=rig_zip(),
        headers={**AUTH, "content-type": "application/zip"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_rig_name"

    response = stack.post(
        "/rigs",
        params={"name": "notzip"},
        content=b"junk bytes",
        headers={**AUTH, "content-type": "application/zip"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_rig"

    response = stack.post(
        "/rigs",
        params={"name": "wrongtype"},
        content=rig_zip(),
        headers={**AUTH, "content-type": "text/html"},
    )
    assert response.status_code == 415

    response = stack.post(
        "/rigs",
        params={"name": "toolarge"},
        content=b"x" * 70000,  # over the fixture's 64 KiB zip cap
        headers={**AUTH, "content-type": "application/zip"},
    )
    assert response.status_code == 413
    assert stack.get("/rigs", headers=AUTH).json()["rigs"] == []


def test_patch_unknown_rig_selector_is_rejected(stack):
    response = stack.patch(
        "/config", headers=AUTH, json={"appearance": {"rig": "ghost"}}
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_config"
    assert stack.runtime.read().config.appearance.rig == "builtin"


def test_background_media_lifecycle(stack):
    response = stack.post(
        "/backgrounds/image",
        params={"name": "Wall Art.png"},
        content=image_png(),
        headers={**AUTH, "content-type": "image/png"},
    )
    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["name"] == "wall-art.png"
    assert saved["kind"] == "image"

    listed = stack.get("/backgrounds", headers=AUTH).json()
    assert [media["name"] for media in listed["files"]] == ["wall-art.png"]
    assert listed["modes"] == ["color", "image", "video", "blur"]
    assert listed["active"]["mode"] == "color"

    response = stack.get("/backgrounds/wall-art.png/thumbnail.jpg", headers=AUTH)
    assert response.status_code == 200
    assert response.content.startswith(b"\xff\xd8")

    response = stack.patch(
        "/config",
        headers=AUTH,
        json={"background": {"mode": "image", "image_path": saved["path"]}},
    )
    assert response.status_code == 200

    response = stack.delete("/backgrounds/wall-art.png", headers=AUTH)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "media_in_use"

    stack.patch("/config", headers=AUTH, json={"background": {"mode": "color"}})
    assert stack.delete("/backgrounds/wall-art.png", headers=AUTH).status_code == 204
    assert stack.get("/backgrounds", headers=AUTH).json()["files"] == []


def test_background_upload_rejections(stack):
    response = stack.post(
        "/backgrounds/image",
        params={"name": "x.png"},
        content=image_png(),
        headers={**AUTH, "content-type": "text/plain"},
    )
    assert response.status_code == 415

    response = stack.post(
        "/backgrounds/image",
        params={"name": "x.png"},
        content=b"not an image",
        headers={**AUTH, "content-type": "image/png"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_media"
    assert stack.get("/backgrounds", headers=AUTH).json()["files"] == []
    response = stack.get("/backgrounds/../etc/thumbnail.jpg", headers=AUTH)
    assert response.status_code == 404
