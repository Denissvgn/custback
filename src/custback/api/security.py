"""Authentication and network-boundary helpers for the custback API.

The API deliberately keeps the long-lived bearer token out of ``AppConfig``
snapshots returned to clients.  A short-lived, opaque browser session cookie
is derived from that token so the preview page and WebSocket can authenticate
without putting secrets in URLs.
"""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
import ssl
import stat
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping
from urllib.parse import urlsplit

from .. import _platform as platform_fs


TOKEN_ENV = "CUSTBACK_API_TOKEN"
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "custback" / "api-token"
RENDERER_TOKEN_ENV = "CUSTBACK_RENDERER_TOKEN"
DEFAULT_RENDERER_TOKEN_FILE = Path.home() / ".config" / "custback" / "renderer-token"
SESSION_COOKIE = "custback_session"
MIN_TOKEN_LENGTH = 32
MAX_TOKEN_FILE_BYTES = 4096
MAX_BROWSER_SESSIONS = 1024

_BEARER_TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/\-]+=*\Z")
_DNS_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_DEFAULT_PORTS = {"http": 80, "https": 443}
_OUTBOUND_SCHEMES = {
    "http": (frozenset({"http", "https"}), frozenset({"http"})),
    "websocket": (frozenset({"ws", "wss"}), frozenset({"ws"})),
    "grpc": (frozenset({"grpc", "grpcs"}), frozenset({"grpc"})),
}

OutboundKind = Literal["http", "websocket", "grpc"]


class SecurityConfigurationError(ValueError):
    """Raised before startup when the API security boundary is unsafe."""


@dataclass(frozen=True)
class ResolvedToken:
    value: str
    path: Path | None = None
    created: bool = False


@dataclass(frozen=True)
class OutboundEndpoint:
    """Canonical outbound service endpoint validated at configuration time."""

    url: str
    scheme: str
    host: str
    port: int | None
    authority: str
    secure: bool


def _validate_token(token: str) -> str:
    if not isinstance(token, str):
        raise SecurityConfigurationError("API token must be text")
    token = token.strip()
    if len(token) < MIN_TOKEN_LENGTH:
        raise SecurityConfigurationError(
            f"API token must contain at least {MIN_TOKEN_LENGTH} characters"
        )
    if not token.isascii() or _BEARER_TOKEN_RE.fullmatch(token) is None:
        raise SecurityConfigurationError(
            "API token must use the RFC 6750 Bearer-token character set"
        )
    return token


def _resolve_token(
    token_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    env_name: str,
    default_file: Path,
    create: bool,
) -> ResolvedToken:
    env = os.environ if environ is None else environ
    from_env = env.get(env_name)
    if from_env:
        return ResolvedToken(_validate_token(from_env))

    path = Path(token_file).expanduser() if token_file else default_file
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if not stat.S_ISREG(existing.st_mode) or stat.S_ISLNK(existing.st_mode):
            raise SecurityConfigurationError(
                f"API token path {path} must be a regular non-symlink file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = platform_fs.open_nofollow(path, flags)
        except OSError as exc:
            raise SecurityConfigurationError(
                f"cannot safely open API token file {path}"
            ) from exc
        with os.fdopen(descriptor, "rb") as token_stream:
            opened = os.fstat(token_stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise SecurityConfigurationError(
                    f"API token path {path} must be a regular file"
                )
            if (opened.st_dev, opened.st_ino) != (existing.st_dev, existing.st_ino):
                raise SecurityConfigurationError(
                    f"API token file {path} changed while it was being opened"
                )
            if not platform_fs.is_private_to_owner(token_stream.fileno()):
                mode = stat.S_IMODE(opened.st_mode)
                raise SecurityConfigurationError(
                    f"API token file {path} must not be accessible by group or others "
                    f"(current mode {mode:04o}; run chmod 600 {path})"
                )
            encoded = token_stream.read(MAX_TOKEN_FILE_BYTES + 1)
        if len(encoded) > MAX_TOKEN_FILE_BYTES:
            raise SecurityConfigurationError(
                f"API token file {path} exceeds {MAX_TOKEN_FILE_BYTES} bytes"
            )
        try:
            value = encoded.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SecurityConfigurationError(
                f"API token file {path} must be ASCII"
            ) from exc
        return ResolvedToken(_validate_token(value), path=path)

    if not create:
        raise SecurityConfigurationError(
            f"required token file {path} does not exist and will not be created by a client"
        )

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = secrets.token_urlsafe(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = platform_fs.open_nofollow(path, flags, 0o600)
    except FileExistsError:  # another process won the first-start race
        return _resolve_token(
            path,
            environ={},
            env_name=env_name,
            default_file=default_file,
            create=False,
        )
    try:
        # A restrictive process umask may remove owner-write permission.  The
        # token contract is an exact private mode, so establish it explicitly.
        platform_fs.set_private_mode(fd, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(token + "\n")
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return ResolvedToken(token, path=path, created=True)


def resolve_api_token(
    token_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ResolvedToken:
    """Resolve or provision the core management token.

    This retains the original server-oriented first-start behavior. Outbound
    clients should use a purpose-specific loader such as
    :func:`resolve_renderer_token`, which never provisions a credential by
    default.
    """

    return _resolve_token(
        token_file,
        environ=environ,
        env_name=TOKEN_ENV,
        default_file=DEFAULT_TOKEN_FILE,
        create=True,
    )


def resolve_renderer_token(
    token_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    create: bool = False,
) -> ResolvedToken:
    """Resolve the frame-renderer credential, provisioning only when requested.

    Avatar clients use the default fail-closed behavior. The core server may
    pass ``create=True`` during explicit credential provisioning.
    """

    return _resolve_token(
        token_file,
        environ=environ,
        env_name=RENDERER_TOKEN_ENV,
        default_file=DEFAULT_RENDERER_TOKEN_FILE,
        create=create,
    )


def _canonical_hostname(value: str) -> tuple[str, bool] | None:
    """Return ``(host, is_ipv6)`` for an IP literal or strict DNS name."""

    if not value or value != value.strip() or any(ch.isspace() for ch in value):
        return None
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return None
    if "%" in value:  # scoped IPv6 literals are not valid web origins
        return None
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        try:
            hostname = value.rstrip(".").encode("idna").decode("ascii").lower()
        except (UnicodeError, ValueError):
            return None
        if not hostname or len(hostname) > 253:
            return None
        if any(_DNS_LABEL_RE.fullmatch(label) is None for label in hostname.split(".")):
            return None
        return hostname, False
    return address.compressed.lower(), address.version == 6


def normalize_bind_host(host: str) -> str | None:
    """Canonicalize a bind host, rejecting URL/authority syntax and controls."""

    if not isinstance(host, str) or host != host.strip():
        return None
    candidate = host
    bracketed = candidate.startswith("[")
    if bracketed:
        if not candidate.endswith("]"):
            return None
        candidate = candidate[1:-1]
    elif any(char in candidate for char in ("/", "\\", "@", "?", "#")):
        return None
    parsed = _canonical_hostname(candidate)
    if parsed is not None and bracketed and not parsed[1]:
        return None
    return parsed[0] if parsed is not None else None


def is_loopback_host(host: str) -> bool:
    """Return true only for literal loopback addresses or ``localhost``."""

    normalized = normalize_bind_host(host)
    if normalized == "localhost":
        return True
    if normalized is None:
        return False
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def is_numeric_loopback_host(host: str) -> bool:
    """Return true only for an IPv4 or IPv6 loopback literal.

    DNS names such as ``localhost`` are intentionally excluded.  Outbound
    plaintext policy must not rely on DNS resolution because an entry can be
    rebound to a non-loopback destination after validation.
    """

    normalized = normalize_bind_host(host)
    if normalized is None:
        return False
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_bind_security(
    host: str,
    *,
    allow_non_loopback: bool,
    tls_certfile: str = "",
    tls_keyfile: str = "",
) -> None:
    """Reject accidental plaintext exposure outside the loopback boundary."""

    if normalize_bind_host(host) is None:
        raise SecurityConfigurationError(f"invalid API bind host: {host!r}")
    cert, key = bool(tls_certfile), bool(tls_keyfile)
    if cert != key:
        raise SecurityConfigurationError(
            "api.tls_certfile and api.tls_keyfile must be configured together"
        )
    if cert:
        for label, value in (
            ("certificate", tls_certfile),
            ("private key", tls_keyfile),
        ):
            if not Path(value).is_file():
                raise SecurityConfigurationError(f"TLS {label} does not exist: {value}")
    if is_loopback_host(host):
        return
    if not allow_non_loopback:
        raise SecurityConfigurationError(
            f"refusing non-loopback API bind {host!r}; set "
            "api.allow_non_loopback=true explicitly"
        )
    if not (cert and key):
        raise SecurityConfigurationError(
            "non-loopback API binds require api.tls_certfile and api.tls_keyfile"
        )


def _parse_authority(value: str) -> tuple[str, bool, int | None] | None:
    """Parse and canonicalize an HTTP Host/origin authority.

    IPv6 must use RFC 3986 brackets.  Ports are decimal and bounded to the
    range Uvicorn can actually bind.
    """

    if not value or value != value.strip():
        return None
    if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return None
    if any(char in value for char in ("@", "/", "\\", "?", "#")):
        return None

    port: int | None = None
    if value.startswith("["):
        end = value.find("]")
        if end <= 1:
            return None
        host_text = value[1:end]
        remainder = value[end + 1 :]
        if remainder:
            if not remainder.startswith(":") or not remainder[1:].isdigit():
                return None
            port = int(remainder[1:])
        parsed = _canonical_hostname(host_text)
        if parsed is None or not parsed[1]:
            return None
    else:
        if value.count(":") > 1:  # IPv6 Host values must use brackets.
            return None
        if ":" in value:
            host_text, port_text = value.rsplit(":", 1)
            if not port_text.isdigit():
                return None
            port = int(port_text)
        else:
            host_text = value
        parsed = _canonical_hostname(host_text)
        if parsed is None or parsed[1]:
            return None

    if port is not None and not 1 <= port <= 65535:
        return None
    return parsed[0], parsed[1], port


def _format_authority(host: str, is_ipv6: bool, port: int | None = None) -> str:
    authority = f"[{host}]" if is_ipv6 else host
    return f"{authority}:{port}" if port is not None else authority


def validate_outbound_endpoint(
    value: str,
    *,
    kind: OutboundKind,
    label: str = "outbound endpoint",
    allow_empty: bool = False,
    require_port: bool = False,
) -> OutboundEndpoint | None:
    """Validate and canonicalize a root HTTP, WebSocket, or gRPC URL.

    Plaintext endpoints are permitted only for numeric loopback literals.
    Secure endpoints may use strict DNS names or usable IP literals.  The
    returned authority is suitable for clients such as gRPC that take
    ``host:port`` separately from the transport scheme.
    """

    if not isinstance(value, str) or value != value.strip():
        raise SecurityConfigurationError(f"{label} must be a trimmed URL")
    if not value:
        if allow_empty:
            return None
        raise SecurityConfigurationError(f"{label} must not be empty")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise SecurityConfigurationError(f"{label} contains control characters")
    try:
        parsed = urlsplit(value)
        # Accessing ``port`` is deliberately eager: urllib otherwise leaves a
        # malformed or out-of-range port latent until the network call.
        parsed_port = parsed.port
    except ValueError as exc:
        raise SecurityConfigurationError(
            f"{label} has an invalid port or host"
        ) from exc

    allowed, plaintext = _OUTBOUND_SCHEMES[kind]
    scheme = parsed.scheme.lower()
    if scheme not in allowed:
        choices = " or ".join(f"{candidate}://" for candidate in sorted(allowed))
        raise SecurityConfigurationError(f"{label} must use {choices}")
    if (
        not parsed.netloc
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise SecurityConfigurationError(
            f"{label} must be a root URL without credentials, path, query, or fragment"
        )
    authority = _parse_authority(parsed.netloc)
    if authority is None:
        raise SecurityConfigurationError(f"{label} has an invalid host or port")
    host, is_ipv6, authority_port = authority
    if parsed_port != authority_port:  # defensive consistency check
        raise SecurityConfigurationError(f"{label} has an invalid port")
    if require_port and authority_port is None:
        raise SecurityConfigurationError(f"{label} must include an explicit port")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (
        address.is_unspecified
        or address.is_multicast
        or address.is_link_local
        or (address.is_reserved and not address.is_loopback)
    ):
        raise SecurityConfigurationError(f"{label} uses an unsafe destination address")
    if scheme in plaintext and not is_numeric_loopback_host(host):
        raise SecurityConfigurationError(
            f"{label} may use {scheme}:// only with a numeric loopback address"
        )

    normalized_authority = _format_authority(host, is_ipv6, authority_port)
    return OutboundEndpoint(
        url=f"{scheme}://{normalized_authority}",
        scheme=scheme,
        host=host,
        port=authority_port,
        authority=normalized_authority,
        secure=scheme not in plaintext,
    )


def validate_client_tls(
    endpoint: OutboundEndpoint | None,
    *,
    ca_file: str = "",
    certfile: str = "",
    keyfile: str = "",
    label: str = "outbound endpoint",
) -> None:
    """Validate TLS trust and optional mutual-TLS file configuration."""

    if bool(certfile) != bool(keyfile):
        raise SecurityConfigurationError(
            f"{label} TLS certificate and private key must be configured together"
        )
    configured = bool(ca_file or certfile or keyfile)
    if endpoint is None:
        if configured:
            raise SecurityConfigurationError(
                f"{label} TLS files require a configured secure endpoint"
            )
        return
    if configured and not endpoint.secure:
        raise SecurityConfigurationError(
            f"{label} TLS files cannot be used with a plaintext endpoint"
        )
    for path_label, value in (
        ("CA bundle", ca_file),
        ("client certificate", certfile),
        ("client private key", keyfile),
    ):
        if value and not Path(value).expanduser().is_file():
            raise SecurityConfigurationError(f"{label} TLS {path_label} does not exist")


def create_client_ssl_context(
    endpoint: OutboundEndpoint,
    *,
    ca_file: str = "",
    certfile: str = "",
    keyfile: str = "",
    label: str = "outbound endpoint",
) -> ssl.SSLContext | None:
    """Build a verified client context, or return ``None`` for loopback plaintext."""

    validate_client_tls(
        endpoint,
        ca_file=ca_file,
        certfile=certfile,
        keyfile=keyfile,
        label=label,
    )
    if not endpoint.secure:
        return None
    try:
        context = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH,
            cafile=str(Path(ca_file).expanduser()) if ca_file else None,
        )
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        if certfile:
            context.load_cert_chain(
                certfile=str(Path(certfile).expanduser()),
                keyfile=str(Path(keyfile).expanduser()),
            )
    except (OSError, ssl.SSLError) as exc:
        # These helpers are also reached while validating authenticated PATCH
        # candidates. Never reflect operator filesystem paths through a public
        # ValidationError response.
        raise SecurityConfigurationError(
            f"{label} TLS material could not be loaded or verified"
        ) from exc
    return context


def canonical_origin(origin: str) -> str | None:
    """Return the unique serialization of an exact HTTP(S) origin."""

    if not isinstance(origin, str) or origin != origin.strip():
        return None
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in origin):
        return None
    if "?" in origin or "#" in origin:
        return None
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if (
        scheme not in _DEFAULT_PORTS
        or not parsed.netloc
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    authority = _parse_authority(parsed.netloc)
    if authority is None or authority[0] in ("*", "null"):
        return None
    host, is_ipv6, port = authority
    if port == _DEFAULT_PORTS[scheme]:
        port = None
    return f"{scheme}://{_format_authority(host, is_ipv6, port)}"


def default_origins(host: str, port: int, *, tls: bool) -> frozenset[str]:
    scheme = "https" if tls else "http"
    normalized = normalize_bind_host(host)
    if normalized is None:
        raise SecurityConfigurationError(f"invalid API bind host: {host!r}")
    hosts = {normalized}
    if is_loopback_host(host):
        hosts.update(("127.0.0.1", "localhost", "::1"))
    origins: set[str] = set()
    for name in hosts:
        parsed = _canonical_hostname(name)
        if parsed is None:  # normalize_bind_host and constants make this defensive.
            continue
        origin = canonical_origin(
            f"{scheme}://{_format_authority(parsed[0], parsed[1], port)}"
        )
        if origin is not None:
            origins.add(origin)
    return frozenset(origins)


class SessionStore:
    """Small in-memory store for browser sessions derived from the bearer."""

    def __init__(
        self,
        ttl_s: int = 8 * 60 * 60,
        max_sessions: int = MAX_BROWSER_SESSIONS,
    ):
        self.ttl_s = ttl_s
        if (
            not isinstance(max_sessions, int)
            or isinstance(max_sessions, bool)
            or max_sessions < 1
        ):
            raise ValueError("max_sessions must be a positive integer")
        self.max_sessions = max_sessions
        self._sessions: dict[str, float] = {}
        self._lock = threading.Lock()

    def issue(self) -> str:
        session = secrets.token_urlsafe(32)
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            while len(self._sessions) >= self.max_sessions:
                oldest = min(self._sessions, key=self._sessions.get)
                self._sessions.pop(oldest, None)
            self._sessions[session] = now + self.ttl_s
        return session

    def valid(self, session: str | None) -> bool:
        if not session:
            return False
        now = time.monotonic()
        with self._lock:
            expires = self._sessions.get(session, 0.0)
            if expires <= now:
                self._sessions.pop(session, None)
                return False
            return True

    def revoke(self, session: str | None) -> None:
        if not session:
            return
        with self._lock:
            self._sessions.pop(session, None)

    def _purge(self, now: float) -> None:
        for key, expires in list(self._sessions.items()):
            if expires <= now:
                self._sessions.pop(key, None)


@dataclass
class SecurityPolicy:
    token: str
    allowed_origins: frozenset[str]
    allowed_hosts: frozenset[str]
    renderer_token: str | None = None
    secure_cookie: bool = False
    session_ttl_s: int = 8 * 60 * 60
    sessions: SessionStore = field(init=False)

    def __post_init__(self) -> None:
        self.token = _validate_token(self.token)
        if self.renderer_token is not None:
            self.renderer_token = _validate_token(self.renderer_token)
            if secrets.compare_digest(self.renderer_token, self.token):
                raise SecurityConfigurationError(
                    "renderer token must be distinct from the management token"
                )
        if (
            not isinstance(self.session_ttl_s, int)
            or isinstance(self.session_ttl_s, bool)
            or self.session_ttl_s < 60
        ):
            raise SecurityConfigurationError(
                "session TTL must be an integer of at least 60 seconds"
            )
        if not self.allowed_origins:
            raise SecurityConfigurationError(
                "at least one exact API origin is required"
            )
        normalized_origins: set[str] = set()
        for origin in self.allowed_origins:
            canonical = canonical_origin(origin)
            if canonical is None:
                raise SecurityConfigurationError(
                    f"invalid exact API origin: {origin!r}"
                )
            normalized_origins.add(canonical)
        if len(normalized_origins) != len(self.allowed_origins):
            raise SecurityConfigurationError(
                "API origins must be unique after canonicalization"
            )
        self.allowed_origins = frozenset(normalized_origins)
        if not self.allowed_hosts:
            raise SecurityConfigurationError("at least one API Host value is required")
        normalized_hosts: set[str] = set()
        for host in self.allowed_hosts:
            normalized = _normalize_authority(host)
            if not normalized:
                raise SecurityConfigurationError(f"invalid exact API Host: {host!r}")
            normalized_hosts.add(normalized)
        self.allowed_hosts = frozenset(normalized_hosts)
        self.sessions = SessionStore(self.session_ttl_s)

    @classmethod
    def for_bind(
        cls,
        token: str,
        host: str,
        port: int,
        *,
        allowed_origins: tuple[str, ...] | list[str] = (),
        tls: bool = False,
        session_ttl_s: int = 8 * 60 * 60,
        extra_hosts: tuple[str, ...] | list[str] = (),
        renderer_token: str | None = None,
    ) -> "SecurityPolicy":
        bind_host = normalize_bind_host(host)
        if bind_host is None:
            raise SecurityConfigurationError(f"invalid API bind host: {host!r}")
        raw_origins = frozenset(allowed_origins) or default_origins(host, port, tls=tls)
        normalized_origins = {canonical_origin(origin) for origin in raw_origins}
        if None in normalized_origins:
            invalid = next(
                origin for origin in raw_origins if canonical_origin(origin) is None
            )
            raise SecurityConfigurationError(f"invalid exact API origin: {invalid!r}")
        origins = frozenset(normalized_origins)
        expected_scheme = "https" if tls else "http"
        if any(urlsplit(origin).scheme != expected_scheme for origin in origins):
            raise SecurityConfigurationError(
                f"API origins must use {expected_scheme} for this bind"
            )
        parsed_bind = _canonical_hostname(bind_host)
        assert parsed_bind is not None  # normalize_bind_host already established this.
        bind_authority = _format_authority(parsed_bind[0], parsed_bind[1], port)
        hosts = {bind_authority, *extra_hosts}
        if port == _DEFAULT_PORTS[expected_scheme]:
            hosts.add(_format_authority(parsed_bind[0], parsed_bind[1]))
        for origin in origins:
            origin_parts = urlsplit(origin)
            origin_authority = _parse_authority(origin_parts.netloc)
            assert origin_authority is not None
            hosts.add(_format_authority(*origin_authority))
            if origin_authority[2] is None:
                hosts.add(
                    _format_authority(
                        origin_authority[0],
                        origin_authority[1],
                        _DEFAULT_PORTS[origin_parts.scheme],
                    )
                )
        if is_loopback_host(host):
            for loopback in ("127.0.0.1", "localhost", "::1"):
                parsed = _canonical_hostname(loopback)
                assert parsed is not None
                hosts.add(_format_authority(parsed[0], parsed[1], port))
                if port == _DEFAULT_PORTS[expected_scheme]:
                    hosts.add(_format_authority(parsed[0], parsed[1]))
        return cls(
            token=token,
            allowed_origins=origins,
            allowed_hosts=frozenset(hosts),
            renderer_token=renderer_token,
            secure_cookie=tls,
            session_ttl_s=session_ttl_s,
        )

    def context_allowed(self, host_header: str | None, origin: str | None) -> bool:
        host = _normalize_authority(host_header or "")
        if host not in self.allowed_hosts:
            return False
        if origin is not None:
            normalized_origin = canonical_origin(origin)
            if (
                normalized_origin is None
                or normalized_origin not in self.allowed_origins
            ):
                return False
        return True

    def bearer_valid(self, authorization: str | None) -> bool:
        if not authorization:
            return False
        scheme, separator, candidate = authorization.partition(" ")
        return bool(
            separator
            and scheme.lower() == "bearer"
            and secrets.compare_digest(candidate, self.token)
        )

    def renderer_bearer_valid(self, authorization: str | None) -> bool:
        """Validate only the renderer-scoped credential.

        Route code must call this exclusively on the raw frame WebSocket; it
        deliberately doesn't participate in general API authentication.
        """

        if not authorization or self.renderer_token is None:
            return False
        scheme, separator, candidate = authorization.partition(" ")
        return bool(
            separator
            and scheme.lower() == "bearer"
            and secrets.compare_digest(candidate, self.renderer_token)
        )

    def authenticated(
        self,
        authorization: str | None,
        session_cookie: str | None,
    ) -> bool:
        return self.bearer_valid(authorization) or self.sessions.valid(session_cookie)


def _normalize_authority(value: str) -> str:
    parsed = _parse_authority(value)
    if parsed is None:
        return ""
    return _format_authority(*parsed)
