"""Permanent regression contracts for security and remote-mode privacy."""

from __future__ import annotations

import asyncio
import builtins
import socket
from collections.abc import Callable

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

import custback.backgrounds as backgrounds_mod
from custback.api.avatar_proxy import register_avatar_proxy
from custback.api.streaming import ConnectionLimiter
from custback.avatar.config import Audio2FaceConfig, AvatarConfig
from custback.capture import CapturedFrame
from custback.config import AppConfig, RuntimeConfig
from custback.hub import FrameHub
from custback.pipeline import Pipeline, RestartRequiredError, _Resources


# Registry links retained after the strict expected-failure markers were
# retired; every item below is now enforced by the permanent tests in this file.
RESOLVED_PHASE_1_BLOCKERS = frozenset({"SEC-01", "TOKEN-01", "TRANS-01", "PRIV-01"})


def _run(awaitable):
    return asyncio.run(awaitable)


async def _get_from_app(app: FastAPI, path: str) -> httpx.Response:
    """Issue an in-process request and close any proxy client it creates."""

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


def _proxy_app(
    cfg: AppConfig,
    handler: Callable[[httpx.Request], httpx.Response],
) -> FastAPI:
    app = FastAPI()
    runtime = RuntimeConfig(cfg)

    def client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    stream_connections = ConnectionLimiter(cfg.api.max_stream_connections)
    app.state.stream_connections = stream_connections
    register_avatar_proxy(
        app,
        runtime,
        stream_connections=stream_connections,
        client_factory=client_factory,
    )
    return app


def test_proxy_destination_and_credential_path_are_restart_only(tmp_path):
    """An API patch cannot turn the proxy into a file-exfiltration client."""

    cfg = AppConfig.from_dict(
        {
            "avatar": {
                "url": "https://trusted-avatar.example:8711",
                "token_file": str(tmp_path / "trusted-avatar-token"),
            }
        }
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())

    with pytest.raises(RestartRequiredError) as caught:
        pipeline.apply_config_patch(
            {
                "avatar": {
                    "url": "https://attacker.example:443",
                    "token_file": str(tmp_path / "core-api-token"),
                }
            }
        )

    assert set(caught.value.fields) == {"avatar.url", "avatar.token_file"}


@pytest.mark.parametrize(
    "build_config",
    [
        pytest.param(
            lambda: AppConfig.from_dict(
                {"avatar": {"url": "http://avatar.example:8711"}}
            ),
            id="avatar-http",
        ),
        pytest.param(
            lambda: AvatarConfig.from_dict(
                {"source": {"url": "ws://camera.example:8710"}}
            ),
            id="source-ws",
        ),
    ],
)
def test_non_loopback_plaintext_endpoints_are_rejected(build_config):
    with pytest.raises(ValueError):
        build_config()


def test_numeric_loopback_plaintext_endpoints_remain_available():
    core = AppConfig.from_dict({"avatar": {"url": "http://127.0.0.1:8711"}})
    avatar = AvatarConfig.from_dict({"source": {"url": "ws://[::1]:8710"}})

    assert core.avatar.url == "http://127.0.0.1:8711"
    assert avatar.source.url == "ws://[::1]:8710"


def test_remote_audio2face_insecure_transport_is_rejected():
    with pytest.raises(ValueError):
        Audio2FaceConfig(url="grpc://audio2face.example:52000")


class _Capture:
    def __init__(self, frame: np.ndarray):
        self._frame = frame
        self._sequence = 0

    def read(self) -> CapturedFrame:
        self._sequence += 1
        frame = self._frame.copy()
        height, width = frame.shape[:2]
        return CapturedFrame(
            pixels=frame,
            sequence=self._sequence,
            captured_at_ns=self._sequence * 1_000_000,
            generation=1,
            geometry_generation=1,
            content_rect=(0, 0, width, height),
        )


class _AllForegroundSegmenter:
    last_foreground = None

    def segment(self, frame: np.ndarray) -> np.ndarray:
        return np.ones(frame.shape[:2], dtype=np.float32)


class _IdentityRefiner:
    def refine(self, mask: np.ndarray, _frame: np.ndarray) -> np.ndarray:
        return mask


class _BlackBackdrop:
    def frame(self, width: int, height: int) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)


class _RecordingOutput:
    def __init__(self):
        self.frames: list[np.ndarray] = []

    def send(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())


def _remote_pipeline() -> tuple[Pipeline, AppConfig]:
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 32, "height": 24},
            "background": {
                "mode": "remote",
                "remote_fallback_mode": "color",
                "color": [0, 0, 0],
            },
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
            "api": {"enabled": False},
        }
    )
    return Pipeline(RuntimeConfig(cfg), FrameHub()), cfg


def _patterned_frame(offset: int = 0) -> np.ndarray:
    values = np.arange(24 * 32 * 3, dtype=np.uint32).reshape(24, 32, 3)
    return ((values + offset) % 251).astype(np.uint8)


def test_SEC_02_hot_backdrop_authority_is_io_free(monkeypatch):
    """Hostile hot source strings fail before capture, file, DNS, or network I/O."""

    calls: list[tuple[str, object]] = []

    def forbidden(kind):
        def fail(*args, **_kwargs):
            calls.append((kind, args[0] if args else None))
            raise AssertionError(f"unexpected {kind} I/O")

        return fail

    monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", forbidden("capture"))
    monkeypatch.setattr(socket, "getaddrinfo", forbidden("dns"))
    monkeypatch.setattr(socket, "create_connection", forbidden("network"))
    monkeypatch.setattr(builtins, "open", forbidden("file"))

    runtime = RuntimeConfig(AppConfig.from_dict({"background": {"mode": "color"}}))
    pipeline = Pipeline(runtime, FrameHub())
    payloads = (
        "http://camera.example.test/live",
        "rtsp://camera.example.test/live",
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:2375/containers/json",
        "/dev/video9",
        "/run/secrets/core-api-token",
    )

    for source in payloads:
        with pytest.raises((RestartRequiredError, ValidationError)):
            pipeline.apply_config_patch(
                {
                    "background": {
                        "mode": "camera",
                        "camera_device": source,
                    }
                }
            )
        state = runtime.read()
        assert state.version == 0
        assert state.config.background.mode == "color"

    legacy_runtime = RuntimeConfig(
        AppConfig.from_dict(
            {
                "background": {
                    "mode": "color",
                    "camera_device": "/dev/video9",
                }
            }
        )
    )
    with pytest.raises(RestartRequiredError):
        Pipeline(legacy_runtime, FrameHub()).apply_config_patch(
            {"background": {"camera_device": "/dev/video9"}}
        )

    target_runtime = RuntimeConfig(
        AppConfig.from_dict({"backdrop_targets": {"side-camera": {"source": 2}}})
    )
    with pytest.raises(RestartRequiredError):
        Pipeline(target_runtime, FrameHub()).apply_config_patch(
            {"backdrop_targets": {"side-camera": {"source": 2}}}
        )

    assert calls == []


def test_PRIV_01_replay_is_rejected_after_window_and_capacity(monkeypatch):
    """Elapsed time and full bounded history can never make raw replay allowable."""

    now = [0.0]
    monkeypatch.setattr("custback.pipeline.time.monotonic", lambda: now[0])
    raw = np.random.default_rng(78).integers(0, 256, (72, 128, 3), dtype=np.uint8)
    current = np.random.default_rng(79).integers(0, 256, raw.shape, dtype=np.uint8)
    pipeline = Pipeline(
        RuntimeConfig(AppConfig()),
        FrameHub(),
        raw_fingerprint_capacity=2,
    )
    assert pipeline._remember_raw_frame(raw)

    now[0] = 60.0
    exact, exact_reason = pipeline._guard_remote_output(
        raw.copy(), current, privacy_safe=True
    )
    ok, encoded = backgrounds_mod.cv2.imencode(
        ".jpg", raw, [backgrounds_mod.cv2.IMWRITE_JPEG_QUALITY, 50]
    )
    assert ok
    jpeg = backgrounds_mod.cv2.imdecode(encoded, backgrounds_mod.cv2.IMREAD_COLOR)
    assert jpeg is not None
    altered, altered_reason = pipeline._guard_remote_output(
        jpeg, current, privacy_safe=True
    )

    assert pipeline._remember_raw_frame(
        np.random.default_rng(80).integers(0, 256, raw.shape, dtype=np.uint8)
    )
    assert not pipeline._remember_raw_frame(
        np.random.default_rng(81).integers(0, 256, raw.shape, dtype=np.uint8)
    )
    capacity, capacity_reason = pipeline._guard_remote_output(
        raw.copy(),
        np.random.default_rng(82).integers(0, 256, raw.shape, dtype=np.uint8),
        privacy_safe=True,
    )
    capacity_jpeg, capacity_jpeg_reason = pipeline._guard_remote_output(
        jpeg,
        np.random.default_rng(83).integers(0, 256, raw.shape, dtype=np.uint8),
        privacy_safe=True,
    )

    slate = Pipeline._privacy_slate(raw.shape)
    assert exact_reason == "privacy-delayed-raw-echo"
    assert altered_reason == "privacy-delayed-raw-echo"
    assert capacity_reason == "privacy-history-exhausted"
    assert capacity_jpeg_reason == "privacy-history-exhausted"
    assert np.array_equal(exact, slate)
    assert np.array_equal(altered, slate)
    assert np.array_equal(capacity, slate)
    assert np.array_equal(capacity_jpeg, slate)


def test_remote_startup_preflight_never_sends_the_captured_frame():
    pipeline, cfg = _remote_pipeline()
    raw = _patterned_frame()
    output = _RecordingOutput()
    resources = _Resources(
        cfg=cfg,
        version=0,
        capture=_Capture(raw),
        segmenter=_AllForegroundSegmenter(),
        refiner=_IdentityRefiner(),
        backdrop=_BlackBackdrop(),
        output=output,
    )

    pipeline._preflight(resources)

    assert output.frames
    assert all(not np.array_equal(frame, raw) for frame in output.frames)


def test_remote_privacy_gate_rejects_a_near_raw_frame():
    pipeline, _cfg = _remote_pipeline()
    raw = _patterned_frame()
    near_raw = raw.copy()
    near_raw[0, 0, 0] ^= np.uint8(1)

    guarded = pipeline._privacy_checked(near_raw, raw, privacy_safe=True)

    assert not np.array_equal(guarded, near_raw)


def test_remote_emergency_output_is_independent_of_camera_input():
    pipeline, _cfg = _remote_pipeline()

    first = pipeline._emergency_blur(_patterned_frame())
    second = pipeline._emergency_blur(_patterned_frame(offset=97))

    assert np.array_equal(first, second)


def test_missing_proxy_client_token_is_not_created(tmp_path, monkeypatch):
    monkeypatch.delenv("CUSTBACK_AVATAR_API_TOKEN", raising=False)
    missing_token = tmp_path / "client" / "avatar-token"
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"connected": False})

    app = _proxy_app(
        AppConfig.from_dict(
            {
                "avatar": {
                    "url": "https://avatar.example:8711",
                    "token_file": str(missing_token),
                }
            }
        ),
        upstream,
    )

    response = _run(_get_from_app(app, "/avatar/status"))

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "avatar_token_unavailable"
    assert not missing_token.exists()
    assert calls == []


def test_upstream_avatar_unauthorized_is_mapped_to_proxy_auth_failure(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CUSTBACK_AVATAR_API_TOKEN", raising=False)
    token_file = tmp_path / "avatar-token"
    token_file.write_text("avatar-client-token-that-is-long-enough-000\n")
    token_file.chmod(0o600)

    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad avatar credential"})

    app = _proxy_app(
        AppConfig.from_dict(
            {
                "avatar": {
                    "url": "https://avatar.example:8711",
                    "token_file": str(token_file),
                }
            }
        ),
        upstream,
    )

    response = _run(_get_from_app(app, "/avatar/status"))

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "avatar_auth_failed"
