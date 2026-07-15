"""Phase-0 executable specifications for runtime remediation work.

The strict xfails in this module are intentional.  They preserve deterministic
reproductions of known defects while production fixes are delivered in later
phases.  Once a defect is fixed, its XPASS is a hard failure so the owning PR
must remove the marker and keep the assertion as a permanent regression test.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import stat
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import custback.avatar.audio2face as audio2face_mod
import custback.backgrounds as backgrounds_mod
import custback.pipeline as pipeline_mod
import custback.segmentation as segmentation_mod
from custback.api import server as server_mod
from custback.__main__ import build_parser, config_from_args
from custback.api.security import SecurityPolicy
from custback.avatar.api import create_avatar_app
from custback.avatar.audio2face import Audio2FaceDriver
from custback.avatar.config import (
    Audio2FaceConfig,
    AvatarConfig,
    AvatarRuntime,
    StorageConfig,
)
from custback.avatar.rig import alpha_over
from custback.avatar.service import AvatarService
from custback.avatar.state import FaceState
from custback.avatar.store import MediaStore, RigStore, StoreError
from custback.config import AppConfig, RuntimeConfig, SegmentationConfig
from custback.hub import FrameHub
from custback.pipeline import ActivationError, Pipeline, _Resources


def _known(issue: str, summary: str):
    return pytest.mark.xfail(
        strict=True,
        reason=f"{issue}: {summary}",
    )


def _png(*, width: int = 32, height: int = 32, alpha: bool = False) -> bytes:
    channels = 4 if alpha else 3
    image = np.zeros((height, width, channels), dtype=np.uint8)
    if alpha:
        image[2:-2, 2:-2] = (40, 120, 220, 255)
    else:
        image[:] = (40, 120, 220)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _avatar_storage(tmp_path: Path, **overrides) -> StorageConfig:
    values = {
        "rigs_dir": str(tmp_path / "rigs"),
        "backgrounds_dir": str(tmp_path / "media"),
    }
    values.update(overrides)
    return StorageConfig.model_validate(values)


@_known("CFG-01", "failed avatar activation publishes configuration early")
def test_CFG_01_failed_avatar_activation_does_not_publish_version(tmp_path):
    """A PATCH is acknowledged only after its referenced resources activate."""
    try:
        from httpx import ASGITransport, AsyncClient
    except ImportError:  # pragma: no cover - development dependency
        pytest.skip("httpx is required for the in-process API regression")

    token = "phase-zero-avatar-token-0123456789abcdef"
    runtime = AvatarRuntime(
        AvatarConfig.from_dict(
            {
                "driver": {"backend": "idle"},
                "storage": {
                    "rigs_dir": str(tmp_path / "rigs"),
                    "backgrounds_dir": str(tmp_path / "media"),
                },
            }
        )
    )
    service = AvatarService(runtime)
    policy = SecurityPolicy.for_bind(
        token,
        "testserver",
        80,
        allowed_origins=["http://testserver"],
        extra_hosts=["testserver"],
    )
    app = create_avatar_app(runtime, service, security=policy)

    async def patch_invalid_background():
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.patch(
                "/config",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "background": {
                        "mode": "image",
                        "image_path": str(tmp_path / "does-not-exist.png"),
                    }
                },
            )

    response = asyncio.run(patch_invalid_background())
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "activation_failed"
    state = runtime.read()
    assert state.version == 0
    assert state.config.background.mode == "color"


@_known("CFG-02", "core merge-patch replaces a nested object with defaults")
def test_CFG_02_core_nested_patch_preserves_upload_siblings():
    original = AppConfig.from_dict(
        {
            "api": {
                "uploads": {
                    "image_max_bytes": 1_111,
                    "video_max_bytes": 2_222,
                    "image_max_pixels": 3_333,
                    "video_max_width": 444,
                    "video_max_height": 333,
                    "storage_max_bytes": 9_999,
                    "max_files": 8,
                }
            }
        }
    )
    candidate = original.patched({"api": {"uploads": {"max_files": 7}}})
    expected = original.api.uploads.model_copy(update={"max_files": 7})
    assert candidate.api.uploads == expected


@_known("CFG-02", "avatar merge-patch replaces Audio2Face siblings with defaults")
def test_CFG_02_avatar_nested_patch_preserves_audio2face_siblings():
    original = AvatarConfig.from_dict(
        {
            "driver": {
                "audio2face": {
                    "url": "old-host:52000",
                    "audio_source": "voice.wav",
                    "sample_rate": 48_000,
                    "chunk_ms": 100,
                }
            }
        }
    )
    candidate = original.patched(
        {"driver": {"audio2face": {"url": "new-host:52000"}}}
    )
    assert candidate.driver.audio2face.url == "new-host:52000"
    assert candidate.driver.audio2face.audio_source == "voice.wav"
    assert candidate.driver.audio2face.sample_rate == 48_000
    assert candidate.driver.audio2face.chunk_ms == 100


@_known("LIFE-02", "blocked Audio2Face RPC is not cancelled during close")
def test_LIFE_02_audio2face_close_cancels_blocked_rpc(monkeypatch):
    entered = threading.Event()
    released = threading.Event()
    cancelled = threading.Event()
    source_closed = threading.Event()

    class BlockingCall:
        def __iter__(self):
            return self

        def __next__(self):
            entered.set()
            released.wait()
            raise StopIteration

        def cancel(self):
            cancelled.set()
            released.set()
            return True

    call = BlockingCall()

    class Stub:
        def __init__(self, _channel):
            pass

        def ProcessAudioStream(self, _requests):
            return call

    class Channel:
        def close(self):
            pass

    protocol = SimpleNamespace(
        grpc=SimpleNamespace(insecure_channel=lambda _url, **_kwargs: Channel()),
        stub_class=Stub,
    )

    class Source:
        def close(self):
            source_closed.set()

    monkeypatch.setattr(audio2face_mod, "_load_protocol", lambda: protocol)
    monkeypatch.setattr(audio2face_mod, "create_audio_source", lambda _cfg: Source())
    driver = Audio2FaceDriver(Audio2FaceConfig(url="127.0.0.1:52000"))
    driver.start()
    assert entered.wait(1.0)
    assert driver._worker is not None
    worker = driver._worker
    real_join = worker.join
    # Keep the known three-second timeout from making this regression slow.
    monkeypatch.setattr(worker, "join", lambda timeout=None: None)
    try:
        driver.close()
        deadline = time.monotonic() + 0.25
        while not source_closed.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert cancelled.is_set()
        assert source_closed.is_set()
        assert not worker.is_alive()
    finally:
        released.set()
        real_join(1.0)


@_known("LIFE-01", "cancelling a session abandons an active to_thread render")
def test_LIFE_01_session_waits_for_inflight_render_before_returning(monkeypatch):
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    render_started = threading.Event()
    render_release = threading.Event()

    def blocked_process(_payload):
        render_started.set()
        render_release.wait()
        return None

    monkeypatch.setattr(service, "_process", blocked_process)

    class WebSocket:
        def __init__(self):
            self._first = True
            self._never = asyncio.Event()

        async def recv(self):
            if self._first:
                self._first = False
                return b"jpeg-like-payload"
            await self._never.wait()

        async def send(self, _payload):
            pass

    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(service._session(WebSocket(), stop))
        deadline = asyncio.get_running_loop().time() + 1.0
        while not render_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("render worker never started")
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.sleep(0.05)
        try:
            assert not task.done()
        finally:
            render_release.set()
            await asyncio.wait_for(task, 1.0)

    asyncio.run(scenario())


@_known("STOR-01", "avatar stores inherit permissive process umask")
def test_STOR_01_media_and_rig_assets_have_exact_private_permissions(tmp_path):
    store_cfg = _avatar_storage(tmp_path)
    media_store = MediaStore(store_cfg)
    rig_store = RigStore(store_cfg)
    archive = _zip(tmp_path / "rig.zip", {"head.png": _png(alpha=True)})

    previous_umask = os.umask(0)
    try:
        staging = media_store.open_staging()
        staging.write_bytes(_png())
        media = media_store.commit(staging, "scene.png", "image")
        rig_store.install_zip("private-rig", archive)
    finally:
        os.umask(previous_umask)

    expected = {
        media_store.directory: 0o700,
        Path(media.path): 0o600,
        rig_store.directory: 0o700,
        rig_store.directory / "private-rig": 0o700,
        rig_store.directory / "private-rig" / "head.png": 0o600,
    }
    actual = {path: stat.S_IMODE(path.stat().st_mode) for path in expected}
    assert actual == expected


@_known("STOR-02", "rig store has no aggregate count or byte quota")
def test_STOR_02_rig_aggregate_quota_rejects_second_install(tmp_path):
    cfg = _avatar_storage(
        tmp_path,
        max_rigs=1,
        rig_storage_max_bytes=64 * 1024,
    )
    store = RigStore(cfg)
    archive = _zip(tmp_path / "rig.zip", {"head.png": _png(alpha=True)})
    store.install_zip("first", archive)
    with pytest.raises(StoreError) as caught:
        store.install_zip("second", archive)
    assert caught.value.code == "storage_full"


@_known("STOR-02", "rig layers are decoded without a configured pixel ceiling")
def test_STOR_02_rig_dimension_bomb_is_rejected_before_decode(tmp_path):
    cfg = _avatar_storage(
        tmp_path,
        rig_layer_max_pixels=1_024,
        rig_total_max_pixels=2_048,
    )
    store = RigStore(cfg)
    archive = _zip(
        tmp_path / "large-layer.zip",
        {"head.png": _png(width=64, height=64, alpha=True)},
    )
    with pytest.raises(StoreError) as caught:
        store.install_zip("large-layer", archive)
    assert caught.value.code == "rig_too_large"
    assert store.list() == []


@_known("SEG-02", "auto probes RVM before honoring a custom .tflite model")
def test_SEG_02_auto_tflite_selects_mediapipe(monkeypatch, tmp_path):
    calls: list[str] = []

    class FakeRVM:
        device = "cpu"

        def __init__(self, _cfg, **_kwargs):
            calls.append("rvm")

    class FakeMediaPipe:
        device = "cpu"

        def __init__(self, _cfg, **_kwargs):
            calls.append("mediapipe")

    monkeypatch.setattr(segmentation_mod, "RVMSegmenter", FakeRVM)
    monkeypatch.setattr(segmentation_mod, "MediaPipeSegmenter", FakeMediaPipe)
    cfg = SegmentationConfig(backend="auto", model_path=str(tmp_path / "custom.tflite"))
    segmenter = segmentation_mod.create_segmenter(cfg)
    assert isinstance(segmenter, FakeMediaPipe)
    assert calls == ["mediapipe"]


@_known("SEG-01", "segmenter construction and model acquisition run on the frame lane")
def test_SEG_01_segmenter_preparation_does_not_run_on_frame_worker(monkeypatch):
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 32, "height": 24, "fps": 60},
            "background": {"mode": "color"},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null", "fps": 60},
            "api": {"enabled": False},
        }
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    construction_threads: list[str] = []

    class PreparedSegmenter:
        device = "cpu"
        last_foreground = None

        def segment(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

        def close(self):
            pass

    def construct(_cfg, **_kwargs):
        construction_threads.append(threading.current_thread().name)
        return PreparedSegmenter()

    pipeline.start()
    assert pipeline._thread is not None
    frame_thread_name = pipeline._thread.name
    monkeypatch.setattr(pipeline_mod, "create_segmenter", construct)
    try:
        pipeline.apply_config_patch({"segmentation": {"threshold": 0.61}})
    finally:
        pipeline.stop()

    assert construction_threads
    assert all(name != frame_thread_name for name in construction_threads)


class _SingleFrameCapture:
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def read(self):
        return self.frame.copy()


class _IdentityRefiner:
    def refine(self, mask, _frame):
        return mask


class _NaNSegmenter:
    device = "test"
    last_foreground = None

    def segment(self, frame):
        return np.full(frame.shape[:2], np.nan, dtype=np.float32)


class _SolidBackdrop:
    def frame(self, width, height):
        return np.full((height, width, 3), (4, 5, 6), dtype=np.uint8)


class _RecordingOutput:
    def __init__(self):
        self.frames: list[np.ndarray] = []

    def send(self, frame):
        self.frames.append(frame.copy())


def _nan_resources(cfg: AppConfig, frame: np.ndarray) -> _Resources:
    return _Resources(
        cfg=cfg,
        version=0,
        capture=_SingleFrameCapture(frame),
        segmenter=_NaNSegmenter(),
        refiner=_IdentityRefiner(),
        backdrop=_SolidBackdrop(),
        output=_RecordingOutput(),
    )


# SEG-03 permanent regression: non-finite masks fail before output.
@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_SEG_03_startup_rejects_nan_mask_before_output():
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 16, "height": 16},
            "background": {"mode": "color"},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
            "api": {"enabled": False},
        }
    )
    output = _RecordingOutput()
    resources = _nan_resources(cfg, np.full((16, 16, 3), 90, dtype=np.uint8))
    resources.output = output
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    with pytest.raises(ActivationError, match="invalid mask"):
        pipeline._preflight(resources)
    assert output.frames == []


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_SEG_03_remote_nan_mask_uses_input_independent_fallback():
    cfg = AppConfig.from_dict(
        {
            "background": {
                "mode": "remote",
                "remote_fallback_mode": "color",
            },
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
            "api": {"enabled": False},
        }
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    outputs = []
    reasons = []
    for value in (20, 220):
        raw = np.full((16, 16, 3), value, dtype=np.uint8)
        resources = _nan_resources(cfg, raw)
        output, reason = pipeline._local_composite(
            resources, raw, privacy_safe=True
        )
        outputs.append(output)
        reasons.append(reason)
    assert all(reasons)
    assert np.array_equal(outputs[0], outputs[1])


@_known("API-01", "stream waits occupy the shared default asyncio executor")
def test_API_01_stream_routes_use_async_subscriptions_not_blocking_to_thread():
    source = inspect.getsource(server_mod.create_app)
    blocking_wait = re.compile(
        r"asyncio\.to_thread\(\s*(?:hub\.(?:raw|output)|slot)\.get",
        re.MULTILINE,
    )
    assert blocking_wait.search(source) is None


@_known("RENDER-01", "alpha_over stores premultiplied RGB consumed as straight alpha")
def test_RENDER_01_half_alpha_layer_is_not_double_multiplied():
    base = np.zeros((1, 1, 4), dtype=np.uint8)
    layer = np.array([[[200, 100, 50, 128]]], dtype=np.uint8)
    alpha_over(base, layer)
    # Straight-alpha storage retains the authored color over transparency;
    # compose_avatar applies the 50% coverage exactly once later.
    np.testing.assert_allclose(base[0, 0, :3], layer[0, 0, :3], atol=1)
    assert base[0, 0, 3] == pytest.approx(128, abs=1)


@_known("RENDER-02", "appearance.follow_pose is not consumed by rendering")
def test_RENDER_02_follow_pose_false_changes_rendered_output():
    class PoseDriver:
        name = "pose-test"
        device = "cpu"

        def update(self, _frame, timestamp):
            return FaceState(
                present=True,
                yaw=0.55,
                pitch=0.15,
                roll=0.12,
                timestamp=timestamp,
            )

        def close(self):
            pass

    frame = np.full((96, 96, 3), 30, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame)
    assert ok

    rendered = []
    for follow_pose in (True, False):
        cfg = AvatarConfig.from_dict(
            {
                "driver": {"backend": "idle", "smoothing": 0.0},
                "appearance": {"follow_pose": follow_pose},
                "background": {"mode": "color", "color": [10, 20, 30]},
            }
        )
        service = AvatarService(
            AvatarRuntime(cfg),
            driver_factory=lambda *_args, **_kwargs: PoseDriver(),
        )
        try:
            payload = service._process(jpeg.tobytes())
            assert payload is not None
            rendered.append(payload)
        finally:
            service._components.close()
    assert rendered[0] != rendered[1]


@_known("MISC-01", "failed CameraBackdrop construction leaks VideoCapture")
def test_MISC_01_camera_backdrop_releases_failed_capture(monkeypatch):
    captures = []

    class FailedCapture:
        def __init__(self, _device):
            self.release_calls = 0
            captures.append(self)

        def isOpened(self):
            return False

        def release(self):
            self.release_calls += 1

    monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", FailedCapture)
    with pytest.raises(RuntimeError, match="cannot open backdrop source"):
        backgrounds_mod.CameraBackdrop(7)
    assert captures[0].release_calls == 1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AppConfig.from_dict({"avatar": {"url": "http://host:abc"}}),
        lambda: AvatarConfig.from_dict({"source": {"url": "ws://host:abc"}}),
    ],
    ids=["core-avatar-proxy", "avatar-source"],
)
def test_MISC_02_malformed_url_ports_are_rejected_eagerly(factory):
    with pytest.raises(ValueError):
        factory()


@_known("MISC-02", "YAML parser errors escape the configuration error contract")
@pytest.mark.parametrize("loader", [AppConfig.load, AvatarConfig.load])
def test_MISC_02_malformed_yaml_is_reported_as_value_error(tmp_path, loader):
    path = tmp_path / "malformed.yaml"
    path.write_text("camera: [unterminated\n")
    with pytest.raises(ValueError):
        loader(path)


@_known("MISC-02", "sequential CLI assignments reject a valid combined override")
def test_MISC_02_combined_cli_overrides_are_validated_as_one_candidate(tmp_path):
    config_path = tmp_path / "camera.yaml"
    config_path.write_text(
        "camera:\n"
        "  fps: 60\n"
        "  recovery_timeout_s: 3.0\n"
    )
    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--fps",
            "1",
            "--camera-recovery-timeout",
            "6",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.camera.fps == 1
    assert cfg.output.fps == 1
    assert cfg.camera.recovery_timeout_s == 6.0
