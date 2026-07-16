"""Authenticated control plane for a running avatar service.

A deliberately small sibling of :mod:`custback.api.server`: status, live
configuration of the avatar's look (parts, scale, position, background,
driver), rig and scene-media stores with tile thumbnails, and a
rendered-output preview. It reuses custback's security boundary — Bearer
tokens, exact Host/Origin checks, loopback-by-default — with its own token
file so a remote avatar host never has to hold the custback control token
at all; its source connection uses the separate renderer-scoped credential.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, get_args

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ValidationError

from .. import __version__
from ..api.security import SecurityPolicy, resolve_api_token
from ..api.server import (
    _encode_jpeg,
    _error,
    _limited_json,
    _to_thread_terminal,
)
from ..api.streaming import (
    ConnectionLimiter,
    JpegBroadcaster,
    LeasedStreamingResponse,
)
from .audio2face import microphone_available, protocol_available
from .config import (
    AVATAR_PARTS,
    BUILTIN_AVATARS,
    AvatarConfig,
    AvatarBackgroundConfig,
    AvatarFraming,
    AvatarRuntime,
    AvatarStyle,
    AppearanceConfig,
    RenderConfig,
    RestartRequiredError,
    StorageConfig,
    VisionConfig,
)
from .rig import RigError
from .service import (
    ActivationError,
    AvatarService,
    ConfigConflictError,
    ReconfigurationUnavailable,
)
from .store import (
    MediaStore,
    RigStore,
    StoreError,
    ThumbnailCache,
    UploadReservation,
    is_rig_directory,
    render_avatar_thumbnail,
    render_media_thumbnail,
    resolve_rig_selector,
)

log = logging.getLogger(__name__)

CONFIG_REQUEST_MAX_BYTES = 64 * 1024
AVATAR_TOKEN_ENV = "CUSTBACK_AVATAR_API_TOKEN"

_ZIP_MEDIA_TYPES = frozenset(
    {"application/zip", "application/x-zip-compressed", "application/octet-stream"}
)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken parent pkg
        return False


def driver_modes(cfg: AvatarConfig) -> list[dict]:
    """UI-selectable animation modes mapped onto ``driver.backend``.

    ``motion`` follows the user's movements (camera tracking), ``voice``
    follows only the user's voice (Audio2Face-3D), ``presence`` is the
    synthetic idle animator, ``auto`` picks motion when available.
    """
    vision_ok = _module_available("mediapipe")
    a2f_reason = ""
    if not protocol_available():
        a2f_reason = "install the [audio2face] extra for voice-driven animation"
    elif (
        cfg.driver.audio2face.audio_source == "microphone"
        and not microphone_available()
    ):
        a2f_reason = "the microphone audio source needs the sounddevice package"
    active = cfg.driver.backend
    return [
        {
            "id": "auto",
            "backend": "auto",
            "label": "Auto",
            "description": "follow movements when tracking is available, else idle",
            "available": True,
            "reason": "",
            "configured": True,
            "active": active == "auto",
        },
        {
            "id": "motion",
            "backend": "vision",
            "label": "Follow my movements",
            "description": "face tracking on the camera frames",
            "available": vision_ok,
            "reason": "" if vision_ok else "install the [mediapipe] extra",
            "configured": True,
            "active": active == "vision",
        },
        {
            "id": "voice",
            "backend": "audio2face",
            "label": "Follow my voice only",
            "description": "audio-driven animation via an Audio2Face-3D endpoint",
            "available": not a2f_reason,
            "reason": a2f_reason,
            "configured": bool(cfg.driver.audio2face.url),
            "active": active == "audio2face",
        },
        {
            "id": "presence",
            "backend": "idle",
            "label": "Idle presence",
            "description": "no tracking; blinks and gentle sway only",
            "available": True,
            "reason": "",
            "configured": True,
            "active": active == "idle",
        },
    ]


async def _receive_body_to(
    request: Request,
    destination: UploadReservation,
    *,
    max_bytes: int,
    media_types: frozenset[str],
    kind: str,
) -> int:
    """Stream the raw request body to ``destination`` under a hard cap."""
    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if content_type and content_type not in media_types:
        prefix_ok = any(
            allowed.endswith("/") and content_type.startswith(allowed)
            for allowed in media_types
        )
        if not prefix_ok:
            raise _error(
                415,
                "unsupported_media_type",
                f"{kind} uploads must be sent as one of: "
                f"{', '.join(sorted(media_types))}",
            )
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise _error(413, "upload_too_large", f"{kind} exceeds {max_bytes} bytes")
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise _error(413, "upload_too_large", f"{kind} exceeds {max_bytes} bytes")
        result = await _to_thread_terminal(destination.write, chunk)
        if result.cancellation is not None:
            raise result.cancellation
    return total


def resolve_avatar_api_token(
    token_file: str, *, environ: Mapping[str, str] | None = None
):
    """Resolve the avatar control-plane token (its own env var and file)."""
    env = os.environ if environ is None else environ
    from_env = env.get(AVATAR_TOKEN_ENV, "")
    # resolve_api_token consults CUSTBACK_API_TOKEN, which is the core
    # management credential; remap so the avatar control secret stays separate.
    return resolve_api_token(
        token_file, environ={"CUSTBACK_API_TOKEN": from_env} if from_env else {}
    )


class PublicAvatarSourceConfig(BaseModel):
    url: str
    connect_timeout_s: float
    reconnect_min_s: float
    reconnect_max_s: float
    frame_max_bytes: int


class PublicAudio2FaceConfig(BaseModel):
    url: str
    audio_source: str
    sample_rate: int
    chunk_ms: int


class PublicAvatarDriverConfig(BaseModel):
    backend: str
    smoothing: float
    vision: VisionConfig
    audio2face: PublicAudio2FaceConfig


class PublicAvatarApiConfig(BaseModel):
    enabled: bool
    host: str
    port: int
    allow_non_loopback: bool
    allowed_origins: tuple[str, ...]
    session_ttl_s: int
    ws_max_bytes: int
    max_stream_connections: int


class PublicAvatarConfig(BaseModel):
    source: PublicAvatarSourceConfig
    driver: PublicAvatarDriverConfig
    appearance: AppearanceConfig
    background: AvatarBackgroundConfig
    render: RenderConfig
    storage: StorageConfig
    api: PublicAvatarApiConfig


class PublicAvatarConfigPatchResponse(BaseModel):
    config: PublicAvatarConfig
    config_version: int


def _public_config(config: AvatarConfig) -> dict[str, Any]:
    """Serialize only fields in the browser-safe public contract."""

    return PublicAvatarConfig.model_validate(config.to_dict()).model_dump(mode="python")


def create_avatar_app(
    runtime: AvatarRuntime,
    service: AvatarService,
    *,
    security: SecurityPolicy,
) -> FastAPI:
    app = FastAPI(
        title="custback-avatar",
        version=__version__,
        docs_url=None,
        redoc_url=None,
    )
    startup_api = runtime.read().config.api
    stream_connections = ConnectionLimiter(startup_api.max_stream_connections)
    output_jpegs = JpegBroadcaster(service.output, _encode_jpeg)
    app.state.stream_connections = stream_connections
    # Storage is a restart-required section, so the startup snapshot is
    # authoritative for the process lifetime.
    storage_cfg = runtime.read().config.storage
    rig_store = RigStore(storage_cfg)
    media_store = MediaStore(storage_cfg)
    thumbnails = ThumbnailCache()

    @app.exception_handler(StoreError)
    async def store_error(_request: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse(
            {"detail": {"code": exc.code, "message": str(exc)}},
            status_code=exc.status,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError):
        return JSONResponse(
            {
                "detail": {
                    "code": "invalid_content",
                    "message": "request content does not match the API contract",
                }
            },
            status_code=422,
        )

    @app.middleware("http")
    async def security_boundary(request: Request, call_next):
        hosts = request.headers.getlist("host")
        origins = request.headers.getlist("origin")
        authorizations = request.headers.getlist("authorization")
        valid_header_shape = (
            len(hosts) == 1 and len(origins) <= 1 and len(authorizations) <= 1
        )
        try:
            if not valid_header_shape or not security.context_allowed(
                hosts[0] if len(hosts) == 1 else None,
                origins[0] if len(origins) == 1 else None,
            ):
                response: Response = JSONResponse(
                    {
                        "detail": {
                            "code": "forbidden_origin",
                            "message": "request context rejected",
                        }
                    },
                    status_code=403,
                )
            elif not security.bearer_valid(
                authorizations[0] if len(authorizations) == 1 else None
            ):
                response = JSONResponse(
                    {
                        "detail": {
                            "code": "unauthorized",
                            "message": "valid API token required",
                        }
                    },
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                response = await call_next(request)
        except Exception:
            log.exception("unhandled avatar API request failure")
            response = JSONResponse(
                {"detail": {"code": "internal_error", "message": "internal API error"}},
                status_code=500,
            )
        if "X-Config-Version" not in response.headers:
            response.headers["X-Config-Version"] = str(runtime.version)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/status")
    async def status() -> JSONResponse:
        body = service.stats_dict()
        return JSONResponse(
            body, headers={"X-Config-Version": str(body["config_version"])}
        )

    @app.get("/avatars")
    async def avatars() -> JSONResponse:
        """Selectable appearance options, for UI pickers.

        The current selection lives in ``GET /config`` under ``appearance``;
        everything here is hot-switchable through ``PATCH /config``.
        ``rigs`` lists installed PNG-layer rigs (select by name via
        ``appearance.rig``); ``modes`` maps UI animation choices onto
        ``driver.backend`` with availability for this host.
        """
        cfg = runtime.read().config
        rigs = await asyncio.to_thread(rig_store.list)
        return JSONResponse(
            {
                "avatars": list(BUILTIN_AVATARS),
                "styles": list(get_args(AvatarStyle)),
                "framings": list(get_args(AvatarFraming)),
                "parts": list(AVATAR_PARTS),
                "rigs": [asdict(rig) for rig in rigs],
                "modes": driver_modes(cfg),
            }
        )

    @app.get(
        "/avatars/{name}/thumbnail.jpg",
        response_class=Response,
        responses={200: {"content": {"image/jpeg": {}}}},
    )
    async def builtin_thumbnail(name: str, style: str | None = None) -> Response:
        if name not in BUILTIN_AVATARS:
            raise _error(404, "avatar_not_found", f"no builtin avatar {name!r}")
        cfg = runtime.read().config
        chosen = style if style is not None else cfg.appearance.style
        if chosen not in get_args(AvatarStyle):
            raise _error(422, "invalid_config", f"unknown style {chosen!r}")
        key = ("builtin", name, chosen)
        cached = thumbnails.get(key)
        if cached is None:
            cached = await asyncio.to_thread(
                render_avatar_thumbnail, "builtin", avatar=name, style=chosen
            )
            thumbnails.put(key, cached)
        return Response(content=cached, media_type="image/jpeg")

    @app.get("/rigs")
    async def list_rigs() -> JSONResponse:
        cfg = runtime.read().config
        rigs = await asyncio.to_thread(rig_store.list)
        return JSONResponse(
            {
                "rigs": [asdict(rig) for rig in rigs],
                "active": cfg.appearance.rig,
            }
        )

    @app.post(
        "/rigs",
        status_code=201,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def install_rig(request: Request, name: str) -> JSONResponse:
        # Fails fast on a bad name before the body is consumed.
        rig_store.rig_path(name)
        opened = await _to_thread_terminal(rig_store.open_staging)
        staging = opened.value
        if opened.cancellation is not None:
            await _to_thread_terminal(staging.abort)
            raise opened.cancellation
        try:
            await _receive_body_to(
                request,
                staging,
                max_bytes=storage_cfg.rig_zip_max_bytes,
                media_types=_ZIP_MEDIA_TYPES,
                kind="rig archive",
            )
            result = await _to_thread_terminal(rig_store.install_zip, name, staging)
            installed = result.value
            if result.cancellation is not None:
                raise result.cancellation
        finally:
            cleanup = await _to_thread_terminal(staging.abort)
            if cleanup.cancellation is not None:
                raise cleanup.cancellation
        return JSONResponse(asdict(installed), status_code=201)

    @app.delete("/rigs/{name}", status_code=204)
    async def delete_rig(name: str) -> Response:
        try:
            await asyncio.to_thread(
                service.apply_storage_mutation,
                lambda cfg: rig_store.remove(name, cfg.appearance.rig),
            )
        except ConfigConflictError as exc:
            raise _error(409, "config_conflict", str(exc)) from exc
        except (ReconfigurationUnavailable, TimeoutError) as exc:
            raise _error(503, "reconfiguration_unavailable", str(exc)) from exc
        return Response(status_code=204)

    @app.get(
        "/rigs/{name}/thumbnail.jpg",
        response_class=Response,
        responses={200: {"content": {"image/jpeg": {}}}},
    )
    async def rig_thumbnail(name: str) -> Response:
        directory = rig_store.stored_path(name)
        cfg = runtime.read().config
        key = (
            "rig",
            name,
            cfg.appearance.style,
            cfg.storage.rig_layer_max_pixels,
            cfg.storage.rig_total_max_pixels,
            cfg.storage.rig_manifest_max_bytes,
            directory.stat().st_mtime_ns,
        )
        cached = thumbnails.get(key)
        if cached is None:
            try:
                cached = await asyncio.to_thread(
                    render_avatar_thumbnail,
                    str(directory),
                    style=cfg.appearance.style,
                    rig_layer_max_pixels=cfg.storage.rig_layer_max_pixels,
                    rig_total_max_pixels=cfg.storage.rig_total_max_pixels,
                    rig_manifest_max_bytes=cfg.storage.rig_manifest_max_bytes,
                )
            except RigError as exc:
                raise _error(422, "invalid_rig", str(exc)) from exc
            thumbnails.put(key, cached)
        return Response(content=cached, media_type="image/jpeg")

    @app.get("/backgrounds")
    async def list_backgrounds() -> JSONResponse:
        cfg = runtime.read().config
        files = await asyncio.to_thread(media_store.list)
        return JSONResponse(
            {
                "files": [asdict(media) for media in files],
                "modes": ["color", "image", "video", "blur"],
                "active": {
                    "mode": cfg.background.mode,
                    "image_path": cfg.background.image_path,
                    "video_path": cfg.background.video_path,
                },
            }
        )

    async def upload_background(request: Request, name: str, kind: str) -> JSONResponse:
        media_types = (
            frozenset({"image/", "application/octet-stream"})
            if kind == "image"
            else frozenset({"video/", "image/gif", "application/octet-stream"})
        )
        opened = await _to_thread_terminal(media_store.open_staging, kind)
        staging = opened.value
        if opened.cancellation is not None:
            await _to_thread_terminal(staging.abort)
            raise opened.cancellation
        try:
            await _receive_body_to(
                request,
                staging,
                max_bytes=media_store.max_bytes(kind),
                media_types=media_types,
                kind=kind,
            )
            result = await _to_thread_terminal(media_store.commit, staging, name, kind)
            saved = result.value
            if result.cancellation is not None:
                raise result.cancellation
        finally:
            cleanup = await _to_thread_terminal(staging.abort)
            if cleanup.cancellation is not None:
                raise cleanup.cancellation
        return JSONResponse(asdict(saved), status_code=201)

    upload_openapi = {
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        }
    }

    @app.post("/backgrounds/image", status_code=201, openapi_extra=upload_openapi)
    async def upload_background_image(request: Request, name: str) -> JSONResponse:
        return await upload_background(request, name, "image")

    @app.post("/backgrounds/video", status_code=201, openapi_extra=upload_openapi)
    async def upload_background_video(request: Request, name: str) -> JSONResponse:
        return await upload_background(request, name, "video")

    @app.delete("/backgrounds/{name}", status_code=204)
    async def delete_background(name: str) -> Response:
        try:
            await asyncio.to_thread(
                service.apply_storage_mutation,
                lambda cfg: media_store.remove(name, cfg.background),
            )
        except ConfigConflictError as exc:
            raise _error(409, "config_conflict", str(exc)) from exc
        except (ReconfigurationUnavailable, TimeoutError) as exc:
            raise _error(503, "reconfiguration_unavailable", str(exc)) from exc
        return Response(status_code=204)

    @app.get(
        "/backgrounds/{name}/thumbnail.jpg",
        response_class=Response,
        responses={200: {"content": {"image/jpeg": {}}}},
    )
    async def background_thumbnail(name: str) -> Response:
        path = await asyncio.to_thread(media_store.stored_path, name)
        kind = media_store.kind_of(path)
        stat = path.stat()
        key = ("media", str(path), stat.st_mtime_ns, stat.st_size)
        cached = thumbnails.get(key)
        if cached is None:
            cached = await asyncio.to_thread(render_media_thumbnail, path, kind)
            thumbnails.put(key, cached)
        return Response(content=cached, media_type="image/jpeg")

    @app.get("/config", response_model=PublicAvatarConfig)
    async def get_config() -> JSONResponse:
        state = runtime.read()
        return JSONResponse(
            _public_config(state.config),
            headers={"X-Config-Version": str(state.version)},
        )

    @app.patch("/config", response_model=PublicAvatarConfigPatchResponse)
    async def patch_config(request: Request) -> JSONResponse:
        patch = await _limited_json(
            request,
            CONFIG_REQUEST_MAX_BYTES,
            media_types=frozenset({"application/json", "application/merge-patch+json"}),
        )
        if not isinstance(patch, dict):
            raise _error(422, "invalid_content", "config patch must be a JSON object")
        appearance = patch.get("appearance")
        if isinstance(appearance, dict) and isinstance(appearance.get("rig"), str):
            selector = appearance["rig"]
            resolved = await asyncio.to_thread(
                resolve_rig_selector, selector, storage_cfg.rigs_dir
            )
            if resolved != "builtin" and not is_rig_directory(resolved):
                raise _error(
                    422,
                    "invalid_config",
                    f"unknown rig {selector!r}: not an installed rig name or "
                    "an existing directory of PNG layers",
                )
        try:
            state = await asyncio.to_thread(service.apply_config_patch, patch)
        except RestartRequiredError as exc:
            raise _error(
                409,
                "restart_required",
                str(exc),
                fields=list(exc.fields),
                current_version=exc.current_version,
            ) from exc
        except ConfigConflictError as exc:
            raise _error(
                409,
                "config_conflict",
                str(exc),
                expected_version=exc.expected_version,
                current_version=exc.current_version,
            ) from exc
        except ActivationError as exc:
            raise _error(422, "activation_failed", str(exc)) from exc
        except (ReconfigurationUnavailable, TimeoutError) as exc:
            raise _error(503, "reconfiguration_unavailable", str(exc)) from exc
        except ValidationError as exc:
            errors = [
                {
                    "field": ".".join(str(part) for part in error.get("loc", ())),
                    "message": error.get("msg", "invalid value"),
                    "type": error.get("type", "value_error"),
                }
                for error in exc.errors(
                    include_url=False, include_context=False, include_input=False
                )
            ]
            raise _error(
                422, "invalid_config", "configuration validation failed", errors=errors
            ) from exc
        except (ValueError, TypeError) as exc:
            raise _error(422, "invalid_config", str(exc)) from exc
        return JSONResponse(
            {"config": _public_config(state.config), "config_version": state.version},
            headers={"X-Config-Version": str(state.version)},
        )

    @app.get(
        "/video/snapshot.jpg",
        response_class=Response,
        responses={
            200: {
                "description": "Latest rendered avatar frame as JPEG",
                "content": {
                    "image/jpeg": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
    )
    async def snapshot() -> Response:
        frame, _ = service.output.latest()
        if frame is None:
            raise _error(503, "frame_unavailable", "no rendered frame yet")
        return Response(
            content=await asyncio.to_thread(_encode_jpeg, frame),
            media_type="image/jpeg",
        )

    @app.get(
        "/video/mjpeg",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "Continuous multipart JPEG stream of rendered output",
                "content": {
                    "multipart/x-mixed-replace": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def mjpeg() -> StreamingResponse:
        boundary = "custbackavatarframe"
        lease = stream_connections.try_acquire()
        if lease is None:
            raise _error(
                429,
                "stream_limit",
                "authenticated stream connection limit reached",
            )

        async def gen():
            try:
                async with output_jpegs.subscribe() as subscription:
                    seq = -1
                    while True:
                        jpeg, seq = await subscription.get(seq, 1.0)
                        if jpeg is None:
                            continue
                        yield (
                            (
                                f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                                f"Content-Length: {len(jpeg)}\r\n\r\n"
                            ).encode()
                            + jpeg
                            + b"\r\n"
                        )
            finally:
                lease.release()

        return LeasedStreamingResponse(
            gen(),
            lease=lease,
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        )

    return app
