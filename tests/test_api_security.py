import os
import stat

import pytest

from custback.api.security import (
    SESSION_COOKIE,
    SessionStore,
    SecurityConfigurationError,
    SecurityPolicy,
    canonical_origin,
    create_client_ssl_context,
    default_origins,
    is_numeric_loopback_host,
    normalize_bind_host,
    resolve_api_token,
    resolve_renderer_token,
    validate_outbound_endpoint,
    validate_bind_security,
)
from custback.__main__ import _security_policy
from custback.config import AppConfig


TOKEN = "a-secure-test-token-with-more-than-32-characters"
RENDERER_TOKEN = "a-distinct-renderer-token-with-more-than-32-characters"


def test_token_environment_takes_precedence(tmp_path):
    missing = tmp_path / "missing-token"
    resolved = resolve_api_token(missing, environ={"CUSTBACK_API_TOKEN": TOKEN})
    assert resolved.value == TOKEN
    assert resolved.path is None
    assert not missing.exists()


def test_token_file_is_created_once_with_private_permissions(tmp_path):
    path = tmp_path / "config" / "api-token"
    first = resolve_api_token(path, environ={})
    assert first.created
    assert first.path == path
    assert len(first.value) >= 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    second = resolve_api_token(path, environ={})
    assert second.value == first.value
    assert not second.created


def test_token_creation_establishes_mode_0600_even_with_restrictive_umask(tmp_path):
    path = tmp_path / "token"
    previous = os.umask(0o777)
    try:
        resolve_api_token(path, environ={})
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_insecure_or_weak_token_file_is_rejected(tmp_path):
    path = tmp_path / "token"
    path.write_text("short\n")
    path.chmod(0o600)
    with pytest.raises(SecurityConfigurationError, match="at least"):
        resolve_api_token(path, environ={})
    path.write_text(TOKEN)
    path.chmod(0o644)
    with pytest.raises(SecurityConfigurationError, match="chmod 600"):
        resolve_api_token(path, environ={})


def test_token_file_rejects_symlinks_including_dangling_links(tmp_path):
    target = tmp_path / "target"
    target.write_text(TOKEN)
    target.chmod(0o600)
    link = tmp_path / "token-link"
    link.symlink_to(target)
    with pytest.raises(SecurityConfigurationError, match="non-symlink"):
        resolve_api_token(link, environ={})

    link.unlink()
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(SecurityConfigurationError, match="non-symlink"):
        resolve_api_token(link, environ={})


def test_renderer_client_token_is_never_provisioned_implicitly(tmp_path):
    path = tmp_path / "renderer-token"
    with pytest.raises(SecurityConfigurationError, match="will not be created"):
        resolve_renderer_token(path, environ={})
    assert not path.exists()

    provisioned = resolve_renderer_token(path, environ={}, create=True)
    assert provisioned.created
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert resolve_renderer_token(path, environ={}).value == provisioned.value


def test_renderer_token_uses_its_own_environment_variable(tmp_path):
    resolved = resolve_renderer_token(
        tmp_path / "missing",
        environ={"CUSTBACK_RENDERER_TOKEN": RENDERER_TOKEN, "CUSTBACK_API_TOKEN": TOKEN},
    )
    assert resolved.value == RENDERER_TOKEN


def test_core_provisions_distinct_management_and_renderer_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CUSTBACK_API_TOKEN", raising=False)
    monkeypatch.delenv("CUSTBACK_RENDERER_TOKEN", raising=False)
    management_path = tmp_path / "management-token"
    renderer_path = tmp_path / "renderer-token"
    cfg = AppConfig.from_dict(
        {
            "api": {
                "token_file": str(management_path),
                "renderer_token_file": str(renderer_path),
            }
        }
    )

    policy = _security_policy(cfg)

    assert management_path.is_file() and renderer_path.is_file()
    assert stat.S_IMODE(management_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(renderer_path.stat().st_mode) == 0o600
    assert policy.bearer_valid(f"Bearer {management_path.read_text().strip()}")
    assert policy.renderer_bearer_valid(
        f"Bearer {renderer_path.read_text().strip()}"
    )
    assert not policy.bearer_valid(f"Bearer {renderer_path.read_text().strip()}")


def test_token_file_is_bounded_ascii_and_token_has_header_safe_characters(tmp_path):
    path = tmp_path / "token"
    path.write_bytes(b"x" * 4097)
    path.chmod(0o600)
    with pytest.raises(SecurityConfigurationError, match="exceeds"):
        resolve_api_token(path, environ={})

    path.write_bytes(("x" * 32 + "\N{SNOWMAN}").encode())
    with pytest.raises(SecurityConfigurationError, match="ASCII"):
        resolve_api_token(path, environ={})

    with pytest.raises(SecurityConfigurationError, match="Bearer-token"):
        resolve_api_token(
            path,
            environ={"CUSTBACK_API_TOKEN": "x" * 32 + "\nInjected: value"},
        )


def test_non_loopback_requires_opt_in_and_tls_pair(tmp_path):
    with pytest.raises(SecurityConfigurationError, match="allow_non_loopback"):
        validate_bind_security("0.0.0.0", allow_non_loopback=False)
    with pytest.raises(SecurityConfigurationError, match="require"):
        validate_bind_security("0.0.0.0", allow_non_loopback=True)
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert.write_text("cert")
    key.write_text("key")
    validate_bind_security(
        "0.0.0.0",
        allow_non_loopback=True,
        tls_certfile=str(cert),
        tls_keyfile=str(key),
    )


def test_configured_tls_files_are_checked_even_on_loopback(tmp_path):
    with pytest.raises(SecurityConfigurationError, match="does not exist"):
        validate_bind_security(
            "127.0.0.1",
            allow_non_loopback=False,
            tls_certfile=str(tmp_path / "missing-cert"),
            tls_keyfile=str(tmp_path / "missing-key"),
        )


def test_loopback_does_not_require_tls():
    for host in ("localhost", "127.0.0.1", "127.9.8.7", "::1"):
        validate_bind_security(host, allow_non_loopback=False)


@pytest.mark.parametrize(
    "kind,scheme",
    [("http", "http"), ("websocket", "ws"), ("grpc", "grpc")],
)
def test_outbound_plaintext_is_numeric_loopback_only(kind, scheme):
    endpoint = validate_outbound_endpoint(
        f"{scheme}://[::1]:8710",
        kind=kind,
        require_port=True,
    )
    assert endpoint is not None
    assert endpoint.host == "::1"
    assert endpoint.authority == "[::1]:8710"
    for host in ("localhost", "service.example", "192.168.1.4", "0.0.0.0"):
        with pytest.raises(SecurityConfigurationError):
            validate_outbound_endpoint(f"{scheme}://{host}:8710", kind=kind)


@pytest.mark.parametrize(
    "url",
    [
        "wss://example.test:bad",
        "wss://example.test:65536",
        "wss://user@example.test:443",
        "wss://example.test:443/path",
        "wss://[fe80::1]:443",
        "wss://240.0.0.1:443",
    ],
)
def test_outbound_endpoints_reject_latent_or_unsafe_authorities(url):
    with pytest.raises(SecurityConfigurationError):
        validate_outbound_endpoint(url, kind="websocket")


def test_secure_outbound_context_requires_hostname_verification():
    endpoint = validate_outbound_endpoint(
        "wss://renderer.example:8710", kind="websocket"
    )
    assert endpoint is not None
    context = create_client_ssl_context(endpoint)
    assert context is not None
    assert context.check_hostname
    assert context.verify_mode.name == "CERT_REQUIRED"


def test_numeric_loopback_helper_excludes_dns_names():
    assert is_numeric_loopback_host("127.0.0.1")
    assert is_numeric_loopback_host("::1")
    assert not is_numeric_loopback_host("localhost")


def test_policy_uses_exact_origin_and_host_and_constant_bearer_contract():
    policy = SecurityPolicy.for_bind(
        TOKEN,
        "127.0.0.1",
        8710,
        allowed_origins=["http://127.0.0.1:8710"],
    )
    assert policy.context_allowed("127.0.0.1:8710", None)
    assert policy.context_allowed("127.0.0.1:8710", "http://127.0.0.1:8710")
    assert not policy.context_allowed("evil.example", "http://127.0.0.1:8710")
    assert not policy.context_allowed("127.0.0.1:8710", "https://evil.example")
    assert policy.bearer_valid(f"Bearer {TOKEN}")
    assert not policy.bearer_valid(f"Basic {TOKEN}")
    assert not policy.bearer_valid("Bearer wrong")


def test_renderer_credential_is_distinct_and_not_general_authentication():
    policy = SecurityPolicy.for_bind(
        TOKEN,
        "127.0.0.1",
        8710,
        renderer_token=RENDERER_TOKEN,
    )
    authorization = f"Bearer {RENDERER_TOKEN}"
    assert policy.renderer_bearer_valid(authorization)
    assert not policy.bearer_valid(authorization)
    assert not policy.authenticated(authorization, None)
    assert not policy.renderer_bearer_valid(f"Bearer {TOKEN}")

    with pytest.raises(SecurityConfigurationError, match="distinct"):
        SecurityPolicy.for_bind(
            TOKEN,
            "127.0.0.1",
            8710,
            renderer_token=TOKEN,
        )


def test_browser_session_expires_and_revokes(monkeypatch):
    policy = SecurityPolicy.for_bind(TOKEN, "localhost", 8710)
    session = policy.sessions.issue()
    assert policy.authenticated(None, session)
    policy.sessions.revoke(session)
    assert not policy.authenticated(None, session)


def test_browser_session_store_has_a_hard_cardinality_bound():
    sessions = SessionStore(ttl_s=3600, max_sessions=2)
    first = sessions.issue()
    second = sessions.issue()
    third = sessions.issue()
    assert not sessions.valid(first)
    assert sessions.valid(second)
    assert sessions.valid(third)
    assert len(sessions._sessions) == 2


def test_ipv6_default_origin_is_well_formed():
    origins = default_origins("::1", 8710, tls=False)
    assert "http://[::1]:8710" in origins

    policy = SecurityPolicy.for_bind(TOKEN, "::1", 8710)
    assert policy.context_allowed("[::1]:8710", "http://[::1]:8710")


@pytest.mark.parametrize(
    ("port", "tls", "origin"),
    [(80, False, "http://localhost"), (443, True, "https://localhost")],
)
def test_default_port_host_authorities_allow_omitted_or_explicit_port(
    port, tls, origin
):
    policy = SecurityPolicy.for_bind(TOKEN, "localhost", port, tls=tls)
    assert policy.context_allowed("localhost", origin)
    assert policy.context_allowed(f"localhost:{port}", f"{origin}:{port}")


def test_origins_and_bind_hosts_are_canonical_and_structurally_valid():
    assert canonical_origin("HTTPS://Example.COM:443/") == "https://example.com"
    assert canonical_origin("http://[0:0:0:0:0:0:0:1]:80") == "http://[::1]"
    assert normalize_bind_host("[0:0:0:0:0:0:0:1]") == "::1"
    assert normalize_bind_host("[localhost]") is None
    assert normalize_bind_host("localhost:8710") is None
    for invalid in (
        "https://example.com:0",
        "https://example.com:65536",
        "https://example.com:abc",
        "https://user@example.com",
        "https://example.com/path",
        "https://example.com?",
        "https://example.com#",
        "https://example.com\\@evil.example",
    ):
        assert canonical_origin(invalid) is None


def test_wildcard_origin_is_rejected():
    with pytest.raises(SecurityConfigurationError):
        SecurityPolicy(
            token=TOKEN,
            allowed_origins=frozenset({"https://*"}),
            allowed_hosts=frozenset({"localhost"}),
        )


def test_origin_scheme_must_match_tls_mode():
    with pytest.raises(SecurityConfigurationError, match="must use https"):
        SecurityPolicy.for_bind(
            TOKEN,
            "camera.example",
            8710,
            allowed_origins=["http://camera.example:8710"],
            tls=True,
        )
