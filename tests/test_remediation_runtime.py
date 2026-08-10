"""Permanent executable specifications for remediated runtime behavior."""

from __future__ import annotations

import asyncio
import contextlib
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
import custback.avatar.service as avatar_service_mod
import custback.backgrounds as backgrounds_mod
import custback.pipeline as pipeline_mod
import custback.segmentation as segmentation_mod
from custback.api import server as server_mod
from custback.__main__ import build_parser, config_from_args, main as core_main
from custback.api.security import SecurityPolicy
from custback.avatar.api import create_avatar_app
from custback.avatar.__main__ import main as avatar_main
from custback.avatar.audio2face import Audio2FaceDriver
from custback.avatar.config import (
    Audio2FaceConfig,
    AvatarConfig,
    AvatarRuntime,
    StorageConfig,
)
from custback.avatar.drivers import FaceDriver
from custback.avatar.rig import alpha_over
from custback.avatar.service import (
    ActivationError as AvatarActivationError,
    AvatarService,
    ConfigConflictError as AvatarConfigConflictError,
    ReconfigurationUnavailable as AvatarReconfigurationUnavailable,
)
from custback.avatar.state import FaceState
from custback.avatar.store import MediaStore, RigStore, StoreError
from custback.capture import CapturedFrame
from custback.config import AppConfig, RuntimeConfig, SegmentationConfig
from custback.hub import FrameHub
from custback.pipeline import ActivationError, Pipeline, _Activation, _Resources
from custback.remote_protocol import encode_remote_frame


async def _with_event_loop_heartbeat(awaitable):
    """Keep restricted selectors polling while worker callbacks complete."""

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


def _run_async(awaitable):
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


# CFG-01 permanent regressions: only a complete, trialed generation publishes.
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

    try:
        response = _run_async(patch_invalid_background())
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "activation_failed"
        state = runtime.read()
        assert state.version == 0
        assert state.config.background.mode == "color"
    finally:
        service.close()


# CFG-02 permanent regressions: nested merge patches preserve siblings.
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


def test_CFG_02_avatar_nested_patch_preserves_audio2face_siblings():
    original = AvatarConfig.from_dict(
        {
            "driver": {
                "audio2face": {
                    "url": "127.0.0.1:52000",
                    "audio_source": "voice.wav",
                    "sample_rate": 48_000,
                    "chunk_ms": 100,
                }
            }
        }
    )
    candidate = original.patched({"driver": {"audio2face": {"url": "127.0.0.2:52000"}}})
    assert candidate.driver.audio2face.url == "grpc://127.0.0.2:52000"
    assert candidate.driver.audio2face.audio_source == "voice.wav"
    assert candidate.driver.audio2face.sample_rate == 48_000
    assert candidate.driver.audio2face.chunk_ms == 100


# LIFE-02 permanent regression: close interrupts every owned gRPC resource.
def test_LIFE_02_audio2face_close_cancels_blocked_rpc(monkeypatch):
    entered = threading.Event()
    released = threading.Event()
    cancelled = threading.Event()
    source_closed = threading.Event()
    events = []

    class BlockingCall:
        def __iter__(self):
            return self

        def __next__(self):
            entered.set()
            released.wait()
            raise StopIteration

        def cancel(self):
            events.append("call.cancel")
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
            events.append("channel.close")

    protocol = SimpleNamespace(
        grpc=SimpleNamespace(insecure_channel=lambda _url, **_kwargs: Channel()),
        stub_class=Stub,
    )

    class Source:
        def close(self):
            events.append("source.close")
            source_closed.set()

    monkeypatch.setattr(audio2face_mod, "_load_protocol", lambda: protocol)
    monkeypatch.setattr(audio2face_mod, "create_audio_source", lambda _cfg: Source())
    driver = Audio2FaceDriver(Audio2FaceConfig(url="127.0.0.1:52000"))
    driver.start()
    assert entered.wait(1.0)
    assert driver._worker is not None
    worker = driver._worker
    try:
        driver.close()
        assert cancelled.is_set()
        assert source_closed.is_set()
        assert not worker.is_alive()
        assert events == ["source.close", "call.cancel", "channel.close"]
        driver.close()  # idempotent: native resources remain exactly-once owned
        assert events == ["source.close", "call.cancel", "channel.close"]
    finally:
        released.set()
        worker.join(1.0)


# LIFE-01 permanent regressions: cancellation retains terminal lane ownership.
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
                return encode_remote_frame("raw-input", 1, b"jpeg-like-payload")
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

    try:
        _run_async(scenario())
    finally:
        service.close()


def test_CFG_01_failed_candidate_preserves_complete_live_generation():
    created = []

    class Driver(FaceDriver):
        name = "tracked"
        device = "cpu"

        def __init__(self, *, fail=False):
            self.fail = fail
            self.close_calls = 0
            created.append(self)

        def update(self, _frame, timestamp):
            if self.fail:
                raise RuntimeError("candidate trial failed")
            return FaceState.neutral(timestamp=timestamp)

        def close(self):
            self.close_calls += 1

    def factory(cfg, **_kwargs):
        return Driver(fail=cfg.backend == "auto")

    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime, driver_factory=factory)
    try:
        service.activate_initial()
        before = service._components
        identities = (before.driver, before.rig, before.backdrop)

        with pytest.raises(AvatarActivationError, match="candidate trial failed"):
            service.apply_config_patch({"driver": {"backend": "auto"}})

        state = runtime.read()
        assert state.version == 0
        assert state.config.driver.backend == "idle"
        assert service._components is before
        assert (
            service._components.driver,
            service._components.rig,
            service._components.backdrop,
        ) == identities
        assert created[0].close_calls == 0
        assert created[1].close_calls == 1
        assert service.stats_dict()["config_version"] == 0
    finally:
        service.close()
    assert created[0].close_calls == 1


@pytest.mark.parametrize("failed_resource", ["rig", "background"])
def test_CFG_01_partial_candidate_construction_closes_staged_driver(
    monkeypatch, tmp_path, failed_resource
):
    created = []

    class Driver(FaceDriver):
        name = "partial-candidate"
        device = "cpu"

        def __init__(self):
            self.close_calls = 0
            created.append(self)

        def update(self, _frame, timestamp):
            return FaceState.neutral(timestamp=timestamp)

        def close(self):
            self.close_calls += 1

    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime, driver_factory=lambda *_args, **_kwargs: Driver())
    service.activate_initial()
    before = service._components
    patch = {"driver": {"backend": "auto"}}
    if failed_resource == "rig":
        original = avatar_service_mod.create_rig

        def fail_rig(selector, *, avatar, style, **limits):
            if style == "sketch":
                raise RuntimeError("rig candidate failed")
            return original(
                selector,
                avatar=avatar,
                style=style,
                **limits,
            )

        monkeypatch.setattr(avatar_service_mod, "create_rig", fail_rig)
        patch["appearance"] = {"style": "sketch"}
        message = "rig candidate failed"
    else:
        background = tmp_path / "candidate.png"
        background.write_bytes(_png())

        def fail_background(_cfg, **_limits):
            raise RuntimeError("background candidate failed")

        monkeypatch.setattr(
            avatar_service_mod, "create_avatar_backdrop", fail_background
        )
        patch["background"] = {
            "mode": "image",
            "image_path": str(background),
        }
        message = "background candidate failed"

    try:
        with pytest.raises(AvatarActivationError, match=message):
            service.apply_config_patch(patch)
        assert service._components is before
        assert runtime.version == 0
        assert created[0].close_calls == 0
        assert created[1].close_calls == 1
    finally:
        service.close()


def test_CFG_01_concurrent_avatar_patches_cas_or_conflict(monkeypatch):
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    service.activate_initial()
    original_prepare = service._prepare_activation
    prepared = threading.Barrier(2)

    def synchronize(*args, **kwargs):
        activation = original_prepare(*args, **kwargs)
        prepared.wait(1.0)
        return activation

    monkeypatch.setattr(service, "_prepare_activation", synchronize)
    results = []
    errors = []

    def apply(patch):
        try:
            results.append(service.apply_config_patch(patch))
        except BaseException as exc:
            errors.append(exc)

    workers = [
        threading.Thread(target=apply, args=({"appearance": {"scale": 0.6}},)),
        threading.Thread(target=apply, args=({"appearance": {"offset_x": 0.2}},)),
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(2.0)
        assert not any(worker.is_alive() for worker in workers)
        assert [state.version for state in results] == [1]
        assert len(errors) == 1
        assert isinstance(errors[0], AvatarConfigConflictError)
        state = runtime.read()
        generation = service._components
        assert generation.version == state.version == 1
        assert generation.config == state.config
        assert service.stats_dict()["config_version"] == 1
    finally:
        service.close()


def test_CFG_01_concurrent_driver_preparation_linearizes_to_conflict(monkeypatch):
    class Driver(FaceDriver):
        device = "cpu"

        def __init__(self, name):
            self.name = name

        def update(self, _frame, timestamp):
            return FaceState.neutral(timestamp=timestamp)

        def close(self):
            pass

    def factory(cfg, **_kwargs):
        return Driver(cfg.backend)

    block = False
    first_entered = threading.Event()
    first_release = threading.Event()
    second_entered = threading.Event()
    active = 0
    maximum_active = 0
    active_lock = threading.Lock()

    def prepare(cfg, **_kwargs):
        nonlocal active, maximum_active
        if not block:
            return avatar_service_mod.DriverPreparation(
                cfg.backend, cfg.vision.model_path
            )
        with active_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if cfg.backend == "auto":
                first_entered.set()
                assert first_release.wait(1.0)
            else:
                second_entered.set()
                deadline = time.monotonic() + 1.0
                while runtime.version == 0 and time.monotonic() < deadline:
                    time.sleep(0.005)
            return avatar_service_mod.DriverPreparation(
                cfg.backend, cfg.vision.model_path
            )
        finally:
            with active_lock:
                active -= 1

    monkeypatch.setattr(avatar_service_mod, "create_driver", factory)
    monkeypatch.setattr(avatar_service_mod, "prepare_driver", prepare)
    runtime = AvatarRuntime(
        AvatarConfig.from_dict(
            {
                "driver": {
                    "backend": "idle",
                    "audio2face": {"url": "127.0.0.1:52000"},
                }
            }
        )
    )
    service = AvatarService(runtime, driver_factory=factory)
    service.activate_initial()
    block = True
    results = []
    errors = []

    def apply(backend):
        try:
            patch = {"backend": backend}
            results.append(service.apply_config_patch({"driver": patch}))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=apply, args=("auto",))
    second = threading.Thread(target=apply, args=("audio2face",))
    try:
        first.start()
        assert first_entered.wait(1.0)
        second.start()
        time.sleep(0.03)
        assert not second_entered.is_set()
        assert maximum_active == 1

        first_release.set()
        first.join(2.0)
        second.join(2.0)
        assert not first.is_alive() and not second.is_alive()
        assert [state.version for state in results] == [1]
        assert len(errors) == 1
        assert isinstance(errors[0], AvatarConfigConflictError)
        assert maximum_active == 1
        assert runtime.read().config.driver.backend == "auto"
        assert service._components.version == runtime.version == 1
    finally:
        first_release.set()
        first.join(1.0)
        second.join(1.0)
        service.close()


def test_CFG_01_background_delete_linearizes_before_candidate_commit(
    monkeypatch, tmp_path
):
    background = tmp_path / "candidate.png"
    background.write_bytes(_png())
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    service.activate_initial()
    before = service._components
    original_prepare = service._prepare_activation
    candidate_prepared = threading.Event()
    allow_submit = threading.Event()

    def pause_after_prepare(*args, **kwargs):
        activation = original_prepare(*args, **kwargs)
        candidate_prepared.set()
        assert allow_submit.wait(1.0)
        return activation

    monkeypatch.setattr(service, "_prepare_activation", pause_after_prepare)
    errors = []

    def select_background():
        try:
            service.apply_config_patch(
                {
                    "background": {
                        "mode": "image",
                        "image_path": str(background),
                    }
                }
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=select_background)
    try:
        worker.start()
        assert candidate_prepared.wait(1.0)
        service.apply_storage_mutation(
            lambda cfg: (
                pytest.fail("candidate became active before deletion")
                if cfg.background.mode != "color"
                else background.unlink()
            )
        )
        allow_submit.set()
        worker.join(2.0)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], AvatarActivationError)
        assert runtime.version == 0
        assert runtime.read().config.background.mode == "color"
        assert service._components is before
        assert not background.exists()
    finally:
        allow_submit.set()
        worker.join(1.0)
        service.close()


def test_CFG_01_slow_asset_preparation_cannot_publish_after_timeout(monkeypatch):
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    service.activate_initial()
    entered = threading.Event()
    release = threading.Event()
    original = avatar_service_mod.prepare_driver

    def slow_prepare(cfg, **kwargs):
        entered.set()
        release.wait()
        return original(cfg, **kwargs)

    monkeypatch.setattr(avatar_service_mod, "prepare_driver", slow_prepare)
    started = time.monotonic()
    try:
        with pytest.raises(
            AvatarReconfigurationUnavailable, match="activation deadline"
        ):
            service.apply_config_patch({"driver": {"backend": "auto"}}, timeout=0.05)
        assert entered.is_set()
        assert time.monotonic() - started < 0.5
        assert runtime.version == 0
        assert service._components.version == 0
        release.set()
        deadline = time.monotonic() + 1.0
        while service._asset_slot.locked() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not service._asset_slot.locked()
        assert runtime.version == 0
        assert service._components.version == 0
    finally:
        release.set()
        service.close()


def test_LIFE_01_close_owns_timed_out_asset_preparation(monkeypatch):
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    service.activate_initial()
    entered = threading.Event()
    release = threading.Event()
    original = avatar_service_mod.prepare_driver

    def blocked_prepare(cfg, **kwargs):
        entered.set()
        release.wait()
        return original(cfg, **kwargs)

    monkeypatch.setattr(avatar_service_mod, "prepare_driver", blocked_prepare)
    try:
        with pytest.raises(AvatarReconfigurationUnavailable):
            service.apply_config_patch({"driver": {"backend": "auto"}}, timeout=0.05)
        assert entered.is_set()
        with pytest.raises(
            AvatarReconfigurationUnavailable, match="asset preparation worker"
        ):
            service.close(timeout=0.02)
        assert not service._closed

        release.set()
        deadline = time.monotonic() + 1.0
        while not service._closed and time.monotonic() < deadline:
            time.sleep(0.005)
        assert service._closed
        assert not service._live_asset_threads()
        assert not any(thread.is_alive() for thread in service._executor._threads)
    finally:
        release.set()
        if not service._closed:
            service.close()


def test_LIFE_01_async_close_bounds_and_owns_asset_preparation(monkeypatch):
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    service.activate_initial()
    entered = threading.Event()
    release = threading.Event()
    original = avatar_service_mod.prepare_driver

    def blocked_prepare(cfg, **kwargs):
        entered.set()
        release.wait()
        return original(cfg, **kwargs)

    monkeypatch.setattr(avatar_service_mod, "prepare_driver", blocked_prepare)
    try:
        with pytest.raises(AvatarReconfigurationUnavailable):
            service.apply_config_patch({"driver": {"backend": "auto"}}, timeout=0.05)
        assert entered.is_set()
        started = time.monotonic()
        with pytest.raises(
            AvatarReconfigurationUnavailable, match="asset preparation worker"
        ):
            _run_async(service.aclose(timeout=0.02))
        assert time.monotonic() - started < 0.5
        assert not service._closed

        release.set()
        deadline = time.monotonic() + 1.0
        while not service._closed and time.monotonic() < deadline:
            time.sleep(0.005)
        assert service._closed
        assert not service._live_asset_threads()
    finally:
        release.set()
        if not service._closed:
            service.close()


def test_LIFE_01_replacement_waits_for_terminal_render_ownership():
    render_started = threading.Event()
    render_release = threading.Event()
    old_closed = threading.Event()
    created = []

    class Driver(FaceDriver):
        name = "owned"
        device = "cpu"

        def __init__(self, blocking):
            self.blocking = blocking
            self.block_enabled = False
            created.append(self)

        def update(self, _frame, timestamp):
            if self.blocking and self.block_enabled:
                render_started.set()
                render_release.wait()
            return FaceState.neutral(timestamp=timestamp)

        def close(self):
            if self is created[0]:
                old_closed.set()

    def factory(cfg, **_kwargs):
        return Driver(blocking=cfg.backend == "idle")

    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime, driver_factory=factory)
    service.activate_initial()
    created[0].block_enabled = True
    frame = np.full((32, 32, 3), 60, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame)
    assert ok
    render = service._submit_lane(service._process, jpeg.tobytes())
    assert render_started.wait(1.0)
    patch_result = []
    patch_error = []

    def replace_driver():
        try:
            patch_result.append(
                service.apply_config_patch({"driver": {"backend": "auto"}})
            )
        except BaseException as exc:
            patch_error.append(exc)

    patcher = threading.Thread(target=replace_driver)
    try:
        patcher.start()
        time.sleep(0.05)
        assert patcher.is_alive()
        assert runtime.version == 0
        assert not old_closed.is_set()

        render_release.set()
        assert render.result(timeout=1.0) is not None
        patcher.join(2.0)
        assert not patcher.is_alive()
        assert not patch_error
        assert patch_result[0].version == 1
        assert old_closed.wait(1.0)
        assert service._components.driver is created[1]
    finally:
        render_release.set()
        patcher.join(1.0)
        service.close()


def test_LIFE_01_repeated_cancellation_still_waits_for_lane_terminal():
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    started = threading.Event()
    release = threading.Event()

    def blocked():
        started.set()
        release.wait()
        return "stale"

    async def scenario():
        future = service._submit_lane(blocked)
        task = asyncio.create_task(service._await_lane_future(future))
        await asyncio.sleep(0)  # enter the ownership helper before cancelling it
        deadline = asyncio.get_running_loop().time() + 1.0
        while not started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("lane callback never started")
            await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert not future.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert future.done()

    try:
        _run_async(scenario())
    finally:
        release.set()
        service.close()


def test_LIFE_01_cancelled_staging_close_failure_poison_is_terminal():
    trial_entered = threading.Event()
    trial_release = threading.Event()
    created = []

    class Driver(FaceDriver):
        name = "owned"
        device = "cpu"

        def __init__(self, block_trial):
            self.block_trial = block_trial
            self.close_calls = 0
            created.append(self)

        def update(self, _frame, timestamp):
            if self.block_trial:
                trial_entered.set()
                trial_release.wait()
            return FaceState.neutral(timestamp=timestamp)

        def close(self):
            self.close_calls += 1
            if self.block_trial and self.close_calls == 1:
                raise RuntimeError("staged worker survived close")

    def factory(cfg, **_kwargs):
        return Driver(block_trial=cfg.backend == "auto")

    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime, driver_factory=factory)
    service.activate_initial()
    try:
        with pytest.raises(AvatarReconfigurationUnavailable):
            service.apply_config_patch({"driver": {"backend": "auto"}}, timeout=0.05)
        assert trial_entered.is_set()
        assert runtime.version == 0

        trial_release.set()
        deadline = time.monotonic() + 1.0
        while service._lifecycle_error is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert service._lifecycle_error is not None
        assert service._components.version == -1
        assert created[0].close_calls == 1  # live generation was made terminal
        assert created[1].close_calls == 1  # failed staged identity is retained
        with pytest.raises(AvatarReconfigurationUnavailable):
            service.apply_config_patch({"appearance": {"scale": 0.7}})

        service.close()
        assert created[1].close_calls == 2
    finally:
        trial_release.set()
        if not service._closed:
            service.close()


def test_LIFE_01_close_timeout_finishes_executor_after_lane_releases():
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    started = threading.Event()
    release = threading.Event()

    def blocked():
        started.set()
        release.wait()

    future = service._submit_lane(blocked)
    assert started.wait(1.0)
    with pytest.raises(AvatarReconfigurationUnavailable, match="did not stop"):
        service.close(timeout=0.01)
    assert not service._closed
    release.set()
    future.result(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while not service._closed and time.monotonic() < deadline:
        time.sleep(0.005)
    assert service._closed
    assert not any(thread.is_alive() for thread in service._executor._threads)


def test_LIFE_01_cancelled_render_never_publishes_stale_preview(monkeypatch):
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    service.activate_initial()
    rendered = threading.Event()
    release = threading.Event()
    original = service._process_deferred

    def blocked_after_render(data):
        result = original(data)
        rendered.set()
        release.wait()
        return result

    monkeypatch.setattr(service, "_process_deferred", blocked_after_render)
    frame = np.full((32, 32, 3), 70, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame)
    assert ok

    class WebSocket:
        def __init__(self):
            self.first = True
            self.wait = asyncio.Event()
            self.sent = []

        async def recv(self):
            if self.first:
                self.first = False
                return encode_remote_frame("raw-input", 1, jpeg.tobytes())
            await self.wait.wait()

        async def send(self, payload):
            self.sent.append(payload)

    websocket = WebSocket()

    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(service._session(websocket, stop))
        deadline = asyncio.get_running_loop().time() + 1.0
        while not rendered.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("render never reached the cancellation barrier")
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.sleep(0.02)
        assert not task.done()
        assert service.output.latest()[0] is None
        release.set()
        await asyncio.wait_for(task, 1.0)

    try:
        _run_async(scenario())
        assert websocket.sent == []
        assert service.output.latest()[0] is None
        assert service.stats_dict()["frames_rendered"] == 0
    finally:
        release.set()
        service.close()


def test_LIFE_01_failed_component_close_is_owned_and_retryable():
    class FlakyDriver(FaceDriver):
        name = "flaky-close"
        device = "cpu"

        def __init__(self):
            self.close_calls = 0

        def update(self, _frame, timestamp):
            return FaceState.neutral(timestamp=timestamp)

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("still stopping")

    driver = FlakyDriver()
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime, driver_factory=lambda *_args, **_kwargs: driver)
    service.activate_initial()

    with pytest.raises(AvatarReconfigurationUnavailable, match="survived teardown"):
        service.close()
    assert not service._closed
    assert service._deferred_closes[0][0] is driver

    service.close()
    assert service._closed
    assert driver.close_calls == 2


# STOR-01 permanent regression: managed assets have exact private modes.
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


# STOR-02 permanent regressions: aggregate and decoded-image quotas apply.
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


# SEG-02 permanent regression: custom model suffix selects its backend.
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


# SEG-01 permanent regression: candidate construction stays off the frame lane.
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
        self.sequence = 0

    def read(self) -> CapturedFrame:
        self.sequence += 1
        frame = self.frame.copy()
        height, width = frame.shape[:2]
        return CapturedFrame(
            pixels=frame,
            sequence=self.sequence,
            captured_at_ns=self.sequence * 1_000_000,
            generation=1,
            geometry_generation=1,
            content_rect=(0, 0, width, height),
        )


class _FixedMaskSegmenter:
    device = "test"
    last_foreground = None

    def __init__(self, mask):
        self.mask = mask

    def segment(self, _frame):
        return self.mask.copy()


class _SolidBackdrop:
    def frame(self, width, height):
        return np.full((height, width, 3), (4, 5, 6), dtype=np.uint8)


class _RecordingOutput:
    def __init__(self):
        self.frames: list[np.ndarray] = []

    def send(self, frame):
        self.frames.append(frame.copy())


def _mask_resources(cfg: AppConfig, frame: np.ndarray, mask: np.ndarray) -> _Resources:
    return _Resources(
        cfg=cfg,
        version=0,
        capture=_SingleFrameCapture(frame),
        segmenter=_FixedMaskSegmenter(mask),
        # The production refiner sanitizes NaN/Inf and clips out-of-range
        # values. Raw validation must reject them before this boundary.
        refiner=segmentation_mod.MaskRefiner(cfg.segmentation),
        backdrop=_SolidBackdrop(),
        output=_RecordingOutput(),
    )


def _nan_resources(cfg: AppConfig, frame: np.ndarray) -> _Resources:
    return _mask_resources(
        cfg,
        frame,
        np.full(frame.shape[:2], np.nan, dtype=np.float32),
    )


# SEG-03 permanent regression: non-finite masks fail before output.
@pytest.mark.parametrize(
    "invalid_mask",
    [
        pytest.param(np.full((16, 16), np.nan, np.float32), id="nan"),
        pytest.param(np.full((16, 16), np.inf, np.float32), id="infinite"),
        pytest.param(np.full((16, 16), 1.1, np.float32), id="out-of-range"),
        pytest.param(np.zeros((16, 16), np.float64), id="wrong-dtype"),
        pytest.param(np.zeros((15, 16), np.float32), id="wrong-shape"),
    ],
)
def test_SEG_03_startup_rejects_invalid_raw_mask_before_refine_and_output(
    invalid_mask,
):
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
    frame = np.full((16, 16, 3), 90, dtype=np.uint8)
    resources = _mask_resources(cfg, frame, invalid_mask)
    resources.output = output
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    with pytest.raises(ActivationError, match="invalid mask"):
        pipeline._preflight(resources)
    assert output.frames == []


@pytest.mark.parametrize(
    "invalid_mask",
    [
        pytest.param(np.full((16, 16), np.nan, np.float32), id="nan"),
        pytest.param(np.full((16, 16), 1.1, np.float32), id="out-of-range"),
        pytest.param(np.zeros((16, 16), np.float64), id="wrong-dtype"),
    ],
)
def test_SEG_03_trial_rejects_invalid_raw_mask_before_refine(invalid_mask):
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 16, "height": 16},
            "background": {"mode": "color"},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
            "api": {"enabled": False},
        }
    )
    candidate = cfg.patched({"segmentation": {"threshold": 0.61}})
    frame = np.full((16, 16, 3), 90, dtype=np.uint8)
    current = _mask_resources(
        cfg,
        frame,
        np.zeros(frame.shape[:2], dtype=np.float32),
    )
    candidate_segmenter = _FixedMaskSegmenter(invalid_mask)
    candidate_refiner = segmentation_mod.MaskRefiner(candidate.segmentation)
    activation = _Activation(
        candidate=candidate,
        replace_segmenter=True,
        segmenter=candidate_segmenter,
        refiner=candidate_refiner,
        temporal_state_owner=pipeline_mod._TemporalStateOwner(
            policy=pipeline_mod._segmenter_key(candidate),
            generation=current.segmentation_generation + 1,
            segmenter=candidate_segmenter,
            refiner=candidate_refiner,
        ),
        backdrop=current.backdrop,
    )

    with pytest.raises(ActivationError, match="invalid mask|segmenter returned"):
        Pipeline(RuntimeConfig(cfg), FrameHub())._trial_activation(
            current,
            activation,
            CapturedFrame(
                pixels=frame,
                sequence=1,
                captured_at_ns=1_000_000,
                generation=1,
                geometry_generation=1,
                content_rect=(0, 0, frame.shape[1], frame.shape[0]),
            ),
        )


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
        output, reason = pipeline._local_composite(resources, raw, privacy_safe=True)
        outputs.append(output)
        reasons.append(reason)
    assert all(reasons)
    assert np.array_equal(outputs[0], outputs[1])


# API-01 permanent regression: stream waits are event-loop-native.
def test_API_01_stream_routes_use_async_subscriptions_not_blocking_to_thread():
    source = inspect.getsource(server_mod.create_app)
    blocking_wait = re.compile(
        r"asyncio\.to_thread\(\s*(?:hub\.(?:raw|output)|slot)\.get",
        re.MULTILINE,
    )
    assert blocking_wait.search(source) is None


# RENDER-01 permanent regression: authored alpha is applied exactly once.
def test_RENDER_01_half_alpha_layer_is_not_double_multiplied():
    base = np.zeros((1, 1, 4), dtype=np.uint8)
    layer = np.array([[[200, 100, 50, 128]]], dtype=np.uint8)
    alpha_over(base, layer)
    # Straight-alpha storage retains the authored color over transparency;
    # compose_avatar applies the 50% coverage exactly once later.
    np.testing.assert_allclose(base[0, 0, :3], layer[0, 0, :3], atol=1)
    assert base[0, 0, 3] == pytest.approx(128, abs=1)


# RENDER-02 permanent regression: follow_pose affects both render paths.
def test_RENDER_02_follow_pose_false_changes_rendered_output():
    class PoseDriver(FaceDriver):
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
            service.close()
    assert rendered[0] != rendered[1]


# MISC-01 permanent regression: failed construction releases its capture.
@pytest.mark.parametrize("probe_raises", [False, True])
def test_MISC_01_camera_backdrop_releases_failed_capture(
    monkeypatch,
    probe_raises,
):
    captures = []

    class FailedCapture:
        def __init__(self, _device):
            self.release_calls = 0
            captures.append(self)

        def isOpened(self):
            if probe_raises:
                raise RuntimeError("capture probe failed")
            return False

        def release(self):
            self.release_calls += 1

    monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", FailedCapture)
    with pytest.raises(RuntimeError):
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


# MISC-02 permanent regressions: configuration errors remain concise/atomic.
@pytest.mark.parametrize("loader", [AppConfig.load, AvatarConfig.load])
def test_MISC_02_malformed_yaml_is_reported_as_value_error(tmp_path, loader):
    path = tmp_path / "malformed.yaml"
    path.write_text("camera: [unterminated\n")
    with pytest.raises(ValueError) as caught:
        loader(path)
    assert str(caught.value).startswith("invalid YAML configuration at line ")
    assert "\n" not in str(caught.value)


@pytest.mark.parametrize("loader", [AppConfig.load, AvatarConfig.load])
def test_MISC_02_deep_yaml_is_reported_as_value_error(tmp_path, loader):
    path = tmp_path / "deep.yaml"
    path.write_text("value: " + "[" * 2_000 + "0" + "]" * 2_000)
    with pytest.raises(ValueError, match="invalid YAML configuration") as caught:
        loader(path)
    assert "\n" not in str(caught.value)


def test_MISC_02_combined_cli_overrides_are_validated_as_one_candidate(tmp_path):
    config_path = tmp_path / "camera.yaml"
    config_path.write_text("camera:\n  fps: 60\n  recovery_timeout_s: 3.0\n")
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


@pytest.mark.parametrize(
    ("entrypoint", "argv"),
    [
        (core_main, ["--no-file-log", "--api-port", "70000"]),
        (avatar_main, ["--api-port", "70000"]),
    ],
    ids=["core", "avatar"],
)
def test_MISC_02_cli_validation_errors_are_concise_and_value_free(
    entrypoint,
    argv,
    capsys,
):
    assert entrypoint(argv) == 2
    error = capsys.readouterr().err
    assert "invalid configuration: api.port:" in error
    assert "input_value" not in error
    assert "errors.pydantic.dev" not in error
    assert len(error.splitlines()) == 1
