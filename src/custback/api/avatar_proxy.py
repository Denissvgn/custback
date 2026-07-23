"""Reverse proxy from custback's API to the avatar control plane.

The web UI talks to one origin: custback (which owns the browser session).
Requests under ``/avatar/*`` are forwarded to the configured
``custback-avatar`` control API with the avatar Bearer token injected
server-side, so the avatar secret never reaches the browser and a remote
avatar host (e.g. a GB10 box) only needs to be reachable from custback.

Only exact method/path pairs in the avatar API's public contract are
forwarded; the client's own credentials (Authorization header, cookies) are
never passed upstream, and upstream ``Set-Cookie`` headers are never passed
back.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import ssl
import stat
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import _platform as platform_fs

from .security import (
    MAX_TOKEN_FILE_BYTES,
    SecurityConfigurationError,
    _validate_token,
    create_client_ssl_context,
    validate_outbound_endpoint,
)

log = logging.getLogger(__name__)


class _ClosingStreamingResponse(StreamingResponse):
    """Close the upstream response across the complete ASGI lifecycle."""

    def __init__(self, *args, close: Callable[[], Any], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._close_upstream = close

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette background tasks are not run when sending response
            # headers raises.  The upstream connection is still ours then.
            await self._close_upstream()


_STATIC_ROUTES = frozenset(
    {
        ("GET", "status"),
        ("GET", "avatars"),
        ("GET", "rigs"),
        ("POST", "rigs"),
        ("GET", "backgrounds"),
        ("POST", "backgrounds/image"),
        ("POST", "backgrounds/video"),
        ("GET", "config"),
        ("PATCH", "config"),
        ("GET", "video/snapshot.jpg"),
        ("GET", "video/mjpeg"),
    }
)
_SAFE_SLUG = r"[a-z0-9][a-z0-9_-]{0,63}"
_SAFE_MEDIA_NAME = r"[a-z0-9][a-z0-9._-]{0,95}"
_PARAMETER_ROUTES = (
    ("GET", re.compile(rf"avatars/{_SAFE_SLUG}/thumbnail\.jpg\Z")),
    ("DELETE", re.compile(rf"rigs/{_SAFE_SLUG}\Z")),
    ("GET", re.compile(rf"rigs/{_SAFE_SLUG}/thumbnail\.jpg\Z")),
    ("DELETE", re.compile(rf"backgrounds/{_SAFE_MEDIA_NAME}\Z")),
    ("GET", re.compile(rf"backgrounds/{_SAFE_MEDIA_NAME}/thumbnail\.jpg\Z")),
)
_FORWARD_REQUEST_HEADERS = ("content-type", "accept")
_FORWARD_RESPONSE_HEADERS = ("content-type", "x-config-version")
_BODY_METHODS = frozenset({"POST", "PATCH"})
_AVATAR_TOKEN_ENV = "CUSTBACK_AVATAR_API_TOKEN"


@dataclass(frozen=True)
class _AvatarProxyTarget:
    """Immutable destination, credential path, and TLS trust selected at startup."""

    url: str
    token_file: str
    verify: ssl.SSLContext | bool


def _read_avatar_client_token(token_file: str) -> str:
    """Load an existing avatar client credential without provisioning one.

    The avatar service owns creation of its control token.  The core process
    is only a client, so a missing client credential is a configuration error
    and must never create a new, unrelated token on the core host.
    """

    from_env = os.environ.get(_AVATAR_TOKEN_ENV, "")
    if from_env:
        return _validate_token(from_env)

    path = Path(token_file).expanduser()
    try:
        existing = path.lstat()
    except FileNotFoundError as exc:
        raise SecurityConfigurationError(
            f"avatar API token file {path} does not exist"
        ) from exc
    if not stat.S_ISREG(existing.st_mode) or stat.S_ISLNK(existing.st_mode):
        raise SecurityConfigurationError(
            f"avatar API token path {path} must be a regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = platform_fs.open_nofollow(path, flags)
    except OSError as exc:
        raise SecurityConfigurationError(
            f"cannot safely open avatar API token file {path}"
        ) from exc
    with os.fdopen(descriptor, "rb") as token_stream:
        opened = os.fstat(token_stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise SecurityConfigurationError(
                f"avatar API token path {path} must be a regular file"
            )
        if (opened.st_dev, opened.st_ino) != (existing.st_dev, existing.st_ino):
            raise SecurityConfigurationError(
                f"avatar API token file {path} changed while it was being opened"
            )
        if not platform_fs.is_private_to_owner(token_stream.fileno()):
            mode = stat.S_IMODE(opened.st_mode)
            raise SecurityConfigurationError(
                f"avatar API token file {path} must not be accessible by group "
                f"or others (current mode {mode:04o}; run chmod 600 {path})"
            )
        encoded = token_stream.read(MAX_TOKEN_FILE_BYTES + 1)
    if len(encoded) > MAX_TOKEN_FILE_BYTES:
        raise SecurityConfigurationError(
            f"avatar API token file {path} exceeds {MAX_TOKEN_FILE_BYTES} bytes"
        )
    try:
        value = encoded.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SecurityConfigurationError(
            f"avatar API token file {path} must be ASCII"
        ) from exc
    return _validate_token(value)


def _build_proxy_target(cfg: Any) -> _AvatarProxyTarget:
    """Resolve a validated destination, TLS trust, and credential path once."""

    if not cfg.url:
        return _AvatarProxyTarget(url="", token_file=cfg.token_file, verify=True)
    endpoint = validate_outbound_endpoint(
        cfg.url,
        kind="http",
        label="avatar.url",
    )
    assert endpoint is not None
    ssl_context = create_client_ssl_context(
        endpoint,
        ca_file=cfg.tls_ca_file,
        certfile=cfg.tls_certfile,
        keyfile=cfg.tls_keyfile,
        label="avatar",
    )
    return _AvatarProxyTarget(
        url=endpoint.url,
        token_file=cfg.token_file,
        verify=ssl_context if ssl_context is not None else True,
    )


def _default_proxy_client(target: _AvatarProxyTarget) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        verify=target.verify,
        follow_redirects=False,
        trust_env=False,
    )


def _is_allowed_avatar_route(method: str, path: str) -> bool:
    """Return whether an exact public avatar API operation is proxyable."""

    method = method.upper()
    if (method, path) in _STATIC_ROUTES:
        return True
    return any(
        method == allowed_method and pattern.fullmatch(path) is not None
        for allowed_method, pattern in _PARAMETER_ROUTES
    )


def _proxy_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"detail": {"code": code, "message": message}}, status_code=status
    )


def _avatar_config(runtime: Any):
    read = getattr(runtime, "read", None)
    if read is not None:
        return read().config.avatar
    return runtime.snapshot().avatar


def register_avatar_proxy(
    app: FastAPI,
    runtime: Any,
    *,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> None:
    """Mount ``/avatar/{path}`` forwarding on an already-secured app.

    The caller's security middleware protects these routes like any other;
    ``client_factory`` exists for tests (e.g. an ASGI-transport client).
    """

    target = _build_proxy_target(_avatar_config(runtime))

    state_lock = asyncio.Lock()

    async def _client() -> httpx.AsyncClient:
        async with state_lock:
            client = getattr(app.state, "avatar_proxy_client", None)
            if client is None or client.is_closed:
                if client_factory is not None:
                    client = client_factory()
                else:
                    client = _default_proxy_client(target)
                app.state.avatar_proxy_client = client
            return client

    # Compose the existing lifespan instead of the deprecated on_event hook.
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_proxy_client(wrapped_app: FastAPI):
        async with previous_lifespan(wrapped_app):
            try:
                yield
            finally:
                client = getattr(app.state, "avatar_proxy_client", None)
                if client is not None and not client.is_closed:
                    await client.aclose()

    app.router.lifespan_context = lifespan_with_proxy_client

    @app.api_route(
        "/avatar/{path:path}",
        methods=["GET", "POST", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    async def avatar_proxy(request: Request, path: str):
        if not target.url:
            return _proxy_error(
                503,
                "avatar_unconfigured",
                "no avatar service configured; set the avatar.url config "
                "section to the custback-avatar control API",
            )
        if not _is_allowed_avatar_route(request.method, path):
            return _proxy_error(
                404, "unknown_avatar_path", f"no proxied avatar route: /{path}"
            )
        try:
            # The path is the immutable, restart-only startup selection, but
            # the supervised avatar owns token creation and starts after the
            # core API. Read its value lazily without ever provisioning it.
            token = _read_avatar_client_token(target.token_file)
        except (OSError, ValueError) as exc:
            log.warning(
                "avatar API client credential unavailable at request time (%s)",
                type(exc).__name__,
            )
            return _proxy_error(
                502,
                "avatar_token_unavailable",
                "the avatar API client credential is unavailable",
            )
        # ``path`` has already passed an ASCII-only route grammar. Preserve
        # the ASGI query as raw query bytes so delimiters decoded by a web
        # framework can never be reinterpreted as path or fragment syntax.
        url = httpx.URL(f"{target.url}/{path}")
        raw_query = request.scope.get("query_string", b"")
        if raw_query:
            url = url.copy_with(query=raw_query)
        headers = {
            "authorization": f"Bearer {token}",
            # Keep upstream bodies un-compressed so they stream through 1:1.
            "accept-encoding": "identity",
        }
        for name in _FORWARD_REQUEST_HEADERS:
            value = request.headers.get(name)
            if value is not None:
                headers[name] = value
        # Only harmless timeout values remain hot. Destination, credential
        # path, and TLS trust all come exclusively from the startup target
        # above; only the contents at that path are refreshed.
        # The MJPEG stream is intentionally endless; everything else gets the
        # configured read deadline.
        cfg = _avatar_config(runtime)
        endless = request.method == "GET" and path == "video/mjpeg"
        timeout = httpx.Timeout(
            connect=cfg.connect_timeout_s,
            read=None if endless else cfg.read_timeout_s,
            write=cfg.read_timeout_s,
            pool=cfg.connect_timeout_s,
        )
        try:
            client = await _client()
            upstream_request = client.build_request(
                request.method,
                url,
                headers=headers,
                content=request.stream() if request.method in _BODY_METHODS else None,
                timeout=timeout,
            )
            upstream = await client.send(
                upstream_request,
                stream=True,
                follow_redirects=False,
            )
        except (httpx.InvalidURL, httpx.UnsupportedProtocol, ValueError) as exc:
            log.warning("avatar proxy request was invalid: %s", type(exc).__name__)
            return _proxy_error(
                502,
                "avatar_request_invalid",
                "the configured avatar request could not be constructed",
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            log.warning("avatar service unreachable: %s", type(exc).__name__)
            return _proxy_error(
                502,
                "avatar_unreachable",
                "the avatar service did not accept the connection",
            )
        except httpx.TimeoutException:
            return _proxy_error(
                504, "avatar_timeout", "the avatar service did not respond in time"
            )
        except httpx.HTTPError as exc:
            log.warning("avatar proxy request failed: %s", type(exc).__name__)
            return _proxy_error(502, "avatar_unreachable", "the avatar request failed")
        if upstream.status_code in {401, 403}:
            await upstream.aclose()
            log.warning(
                "avatar service rejected its configured client credential: status=%d",
                upstream.status_code,
            )
            return _proxy_error(
                502,
                "avatar_auth_failed",
                "the avatar service rejected its configured client credential",
            )
        response_headers = {
            name: upstream.headers[name]
            for name in _FORWARD_RESPONSE_HEADERS
            if name in upstream.headers
        }
        return _ClosingStreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=response_headers,
            close=upstream.aclose,
        )
