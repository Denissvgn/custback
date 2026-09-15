"""Immutable system profile, preference-store, and capture-probe contracts."""

from __future__ import annotations

import asyncio
import contextlib
import json
import stat
import threading
from importlib.resources import files
from typing import Any, cast

import pytest
import yaml
import starlette

if int(starlette.__version__.split(".", 1)[0]) >= 1:
    from httpx2 import ASGITransport as _ASGITransport
    from httpx2 import AsyncClient as _AsyncClient
else:
    from httpx import ASGITransport as _ASGITransport
    from httpx import AsyncClient as _AsyncClient

import custback.system_profile_probe as probe_mod
from custback.__main__ import build_parser, resolve_config_from_args
from custback.api.security import SecurityPolicy
from custback.api.server import create_app
from custback.config import AppConfig, RuntimeConfig
from custback.hub import FrameHub
from custback.profile_preferences import (
    ProfilePreferences,
    ProfilePreferencesError,
    ProfilePreferencesStore,
)
from custback.profile_service import (
    ProfileService,
    ProfileServiceError,
    ProfileStartupContext,
)
from custback.system_profiles import (
    PROFILE_CATALOG,
    ProfileCatalogError,
    load_profile_catalog,
    profile_availability,
    public_catalog,
)


CUDA_FACTS = {
    "output_backend": "PyVirtualCamOutput",
    "output_fallback_active": False,
    "acceleration_state": "gpu_active",
    "acceleration_active_provider": "cuda",
    "acceleration_fallback_active": False,
    "segmentation_device": "cuda",
    "segmentation_fallback_active": False,
    "segmentation_selection": {
        "selected_backend": "rvm",
        "active_provider": "cuda",
    },
    "output_width": 1280,
    "output_height": 720,
    "output_fps": 30,
}
TOKEN = "profile-test-token-with-at-least-thirty-two-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _run_async(awaitable):
    async def wrapped():
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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(wrapped())
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        asyncio.set_event_loop(None)


class _Coordinator:
    def apply_config_patch(self, *_args, **_kwargs):  # pragma: no cover - not called
        raise AssertionError

    def apply_staged_config_patch(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError

    def apply_storage_mutation(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError


def _service(tmp_path, *, base=None, cli_values=None, facts=None):
    base = AppConfig() if base is None else base
    store = ProfilePreferencesStore(tmp_path / "config" / "profile-preferences.yaml")
    startup = ProfileStartupContext(
        base,
        ProfilePreferences(0, {}),
        store,
        {} if cli_values is None else cli_values,
    )
    runtime = RuntimeConfig(base)
    service = ProfileService(
        runtime,
        startup,
        runtime_facts=lambda: CUDA_FACTS if facts is None else facts,
    )
    return service, runtime, store


def test_packaged_catalog_is_immutable_validated_and_digest_bound():
    public = public_catalog()
    assert public["schema"] == "custback.system-profile-catalog"
    assert public["version"] == 1
    assert public["quality_claim"] is False
    assert set(public["axes"]) == {"quality", "framing"}
    assert set(public["axes"]["quality"]["profiles"]) == {
        "performance",
        "balanced",
        "quality",
        "motion_stable",
    }
    for axis in public["axes"].values():
        for definition in axis["profiles"].values():
            assert definition["evidence_state"] == "experimental"
            assert definition["quality_claim"] is False
            assert definition["lifecycle"] == "restart"
            assert len(definition["patch_digest"]) == 64
            assert "patch" not in definition
    assert "acceleration.device_id" not in PROFILE_CATALOG.manageable_paths
    assert "camera.device" not in PROFILE_CATALOG.manageable_paths
    assert "output.device" not in PROFILE_CATALOG.manageable_paths
    assert "segmentation.model_path" not in PROFILE_CATALOG.manageable_paths

    raw = files("custback").joinpath("system-profile-catalog.json").read_bytes()
    assert load_profile_catalog(raw).digest == PROFILE_CATALOG.digest


def test_catalog_rejects_overlapping_axes_and_quality_claims():
    source = files("custback").joinpath("system-profile-catalog.json").read_text()
    raw = json.loads(source)
    raw["quality_claim"] = True
    with pytest.raises(ProfileCatalogError, match="quality claim"):
        load_profile_catalog(json.dumps(raw).encode())

    raw["quality_claim"] = False
    raw["axes"]["framing"]["owned_paths"][0] = "camera.width"
    for definition in raw["axes"]["framing"]["profiles"].values():
        definition["patch"] = {
            "camera": {"width": 1280, "anchor_x": 0.5, "anchor_y": 0.5}
        }
    with pytest.raises(ProfileCatalogError, match="overlap"):
        load_profile_catalog(json.dumps(raw).encode())

    raw = json.loads(source)
    raw["axes"]["quality"]["owned_paths"].append("acceleration.device_id")
    for profile in raw["axes"]["quality"]["profiles"].values():
        profile["patch"]["acceleration"]["device_id"] = 0
    with pytest.raises(ProfileCatalogError, match="forbidden"):
        load_profile_catalog(json.dumps(raw).encode())

    raw = json.loads(source)
    raw["axes"]["quality"]["profiles"]["performance"]["requirements"]["output_mode"] = [
        1280,
        720,
        30,
    ]
    with pytest.raises(ProfileCatalogError, match="output-mode requirement"):
        load_profile_catalog(json.dumps(raw).encode())

    raw = json.loads(source)
    raw["axes"]["quality"]["profiles"]["balanced"]["patch"]["acceleration"]["mode"] = (
        "auto"
    )
    with pytest.raises(ProfileCatalogError, match="CUDA requirement"):
        load_profile_catalog(json.dumps(raw).encode())

    raw = json.loads(source)
    raw["axes"]["quality"]["profiles"]["balanced"]["evidence_state"] = (
        "locally_screened"
    )
    assert (
        load_profile_catalog(json.dumps(raw).encode())
        .definition("quality", "balanced")
        .evidence_state
        == "locally_screened"
    )


def test_preferences_are_private_concrete_and_revision_cas(tmp_path):
    service, runtime, store = _service(tmp_path)
    result = service.apply(
        {"quality": "balanced", "framing": "headroom"},
        expected_config_version=0,
        expected_preferences_revision=0,
        accept_experimental=True,
    )
    assert runtime.snapshot() == AppConfig()
    assert result["preference_revision"] == 1
    assert result["axes"]["quality"]["state"] == "saved_for_restart"
    assert result["axes"]["framing"]["state"] == "saved_for_restart"
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    encoded = store.path.read_text()
    assert "balanced" not in encoded
    assert "headroom" not in encoded
    assert "profile_id" not in encoded
    document = yaml.safe_load(encoded)
    assert document["schema"] == "custback.profile-preferences"
    assert document["values"]["camera"]["fps"] == 15
    assert document["values"]["camera"]["anchor_y"] == 0.35
    with pytest.raises(ProfileServiceError, match="current revision is 1"):
        service.apply(
            {"quality": "quality"},
            expected_config_version=0,
            expected_preferences_revision=0,
            accept_experimental=True,
        )


def test_preferences_reject_symlink_forbidden_path_and_insecure_mode(
    tmp_path, monkeypatch
):
    path = tmp_path / "config" / "profile-preferences.yaml"
    path.parent.mkdir()
    path.parent.chmod(0o700)
    target = tmp_path / "foreign"
    target.write_text("not preferences")
    path.symlink_to(target)
    with pytest.raises(ProfilePreferencesError, match="regular file"):
        ProfilePreferencesStore(path).read()

    path.unlink()
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "custback.profile-preferences",
                "version": 1,
                "revision": 1,
                "values": {"api": {"token_file": "/private/token"}},
            }
        )
    )
    path.chmod(0o600)
    with pytest.raises(ProfilePreferencesError, match="forbidden path"):
        ProfilePreferencesStore(path).read()

    path.write_text(
        yaml.safe_dump(
            {
                "schema": "custback.profile-preferences",
                "version": 1,
                "revision": 1,
                "values": {},
            }
        )
    )
    path.chmod(0o644)
    with pytest.raises(ProfilePreferencesError, match="owner-only"):
        ProfilePreferencesStore(path).read()

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    real_parent.chmod(0o700)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ProfilePreferencesError, match="reparse point"):
        ProfilePreferencesStore(linked_parent / "missing.yaml").read()

    duplicate = tmp_path / "duplicate" / "profile-preferences.yaml"
    duplicate.parent.mkdir()
    duplicate.parent.chmod(0o700)
    duplicate.write_text(
        "schema: custback.profile-preferences\n"
        "version: 1\n"
        "revision: 1\n"
        "revision: 2\n"
        "values: {}\n"
    )
    duplicate.chmod(0o600)
    with pytest.raises(ProfilePreferencesError, match="malformed"):
        ProfilePreferencesStore(duplicate).read()

    exhausted = tmp_path / "exhausted" / "profile-preferences.yaml"
    exhausted.parent.mkdir()
    exhausted.parent.chmod(0o700)
    exhausted.write_text(
        yaml.safe_dump(
            {
                "schema": "custback.profile-preferences",
                "version": 1,
                "revision": 2**63 - 1,
                "values": {},
            }
        )
    )
    exhausted.chmod(0o600)
    with pytest.raises(ProfilePreferencesError, match="revision is exhausted"):
        ProfilePreferencesStore(exhausted).update(2**63 - 1, lambda values: values)

    denied = tmp_path / "denied" / "profile-preferences.yaml"
    denied.parent.mkdir()
    denied.parent.chmod(0o700)
    denied.write_text("unreadable")
    denied.chmod(0o600)
    from custback import profile_preferences as preferences_mod

    real_open = preferences_mod.platform_fs.open_nofollow

    def reject_file(open_path, flags, mode=0o777, *, directory=False):
        if open_path == denied:
            raise PermissionError(13, "denied", "/private/preferences.yaml")
        return real_open(open_path, flags, mode, directory=directory)

    monkeypatch.setattr(preferences_mod.platform_fs, "open_nofollow", reject_file)
    with pytest.raises(ProfilePreferencesError) as captured:
        ProfilePreferencesStore(denied).read()
    assert "/private" not in str(captured.value)

    insecure_directory = tmp_path / "insecure" / "profile-preferences.yaml"
    insecure_directory.parent.mkdir()
    insecure_directory.parent.chmod(0o755)
    insecure_directory.write_text(
        yaml.safe_dump(
            {
                "schema": "custback.profile-preferences",
                "version": 1,
                "revision": 1,
                "values": {},
            }
        )
    )
    insecure_directory.chmod(0o600)
    with pytest.raises(ProfilePreferencesError, match="directory is not owner-only"):
        ProfilePreferencesStore(insecure_directory).read()


def test_experimental_availability_cli_locks_and_custom_model_fail_closed(tmp_path):
    service, _runtime, _store = _service(tmp_path)
    with pytest.raises(ProfileServiceError, match="explicit acknowledgment"):
        service.apply(
            {"quality": "balanced"},
            expected_config_version=0,
            expected_preferences_revision=0,
            accept_experimental=False,
        )

    locked, _runtime, _store = _service(
        tmp_path / "locked",
        cli_values={"camera.fps": 30, "output.fps": 30},
    )
    locked_status = locked.status()
    balanced = locked_status["catalog"]["axes"]["quality"]["profiles"]["balanced"]
    assert not balanced["available"]
    assert balanced["availability_reason"] == "Explicit CLI override locks camera.fps."
    with pytest.raises(ProfileServiceError, match="camera.fps"):
        locked.apply(
            {"quality": "balanced"},
            expected_config_version=0,
            expected_preferences_revision=0,
            accept_experimental=True,
        )

    custom = AppConfig.from_dict(
        {"segmentation": {"backend": "rvm", "model_path": "/models/custom.onnx"}}
    )
    custom_service, _runtime, _store = _service(tmp_path / "custom", base=custom)
    status = custom_service.status()
    quality = status["catalog"]["axes"]["quality"]["profiles"]
    assert all(not definition["available"] for definition in quality.values())
    assert all(
        "custom segmentation model" in definition["availability_reason"]
        for definition in quality.values()
    )

    custom_tflite = AppConfig.from_dict(
        {"segmentation": {"backend": "auto", "model_path": "/models/custom.tflite"}}
    )
    direct, _runtime, direct_store = _service(tmp_path / "direct", base=custom_tflite)
    with pytest.raises(ProfileServiceError) as captured:
        direct.apply(
            {"quality": "balanced"},
            expected_config_version=0,
            expected_preferences_revision=0,
            accept_experimental=True,
        )
    assert captured.value.status == 409
    assert captured.value.code == "profile_unavailable"
    assert direct_store.read().revision == 0


def test_installed_cuda_provider_does_not_substitute_for_active_cuda(tmp_path):
    facts = {
        **CUDA_FACTS,
        "acceleration_state": "cpu_fallback",
        "acceleration_active_provider": "cpu",
        "segmentation_device": "cpu",
        "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "segmentation_selection": {
            "selected_backend": "rvm",
            "active_provider": "cpu",
        },
    }
    service, _runtime, _store = _service(tmp_path, facts=facts)
    definition = service.status()["catalog"]["axes"]["quality"]["profiles"]["balanced"]
    assert definition["available"] is False
    assert "CUDA execution has not been proven" in definition["availability_reason"]


def test_restart_materialization_and_independent_axis_reset(tmp_path):
    service, _runtime, store = _service(tmp_path)
    service.apply(
        {"quality": "balanced", "framing": "show_all"},
        expected_config_version=0,
        expected_preferences_revision=0,
        accept_experimental=True,
    )
    preferences = store.read()
    desired = AppConfig().patched(preferences.values)
    restarted = ProfileService(
        RuntimeConfig(desired),
        ProfileStartupContext(AppConfig(), preferences, store, {}),
        runtime_facts=lambda: CUDA_FACTS,
    )
    status = restarted.status()
    assert status["axes"]["quality"]["state"] == "active"
    assert status["axes"]["framing"]["state"] == "active"
    reset = restarted.reset(
        ["framing"],
        expected_config_version=0,
        expected_preferences_revision=1,
    )
    assert reset["axes"]["quality"]["desired"] == "balanced"
    assert reset["axes"]["framing"]["desired"] is None
    assert "fit_mode" not in store.read().values["camera"]


def test_session_patches_do_not_create_managed_restart_state(tmp_path):
    balanced = AppConfig().patched(
        PROFILE_CATALOG.definition("quality", "balanced").detached_patch()
    )
    service, runtime, _store = _service(tmp_path / "managed", base=balanced)
    writer = runtime._coordinator_writer()
    writer.commit(
        balanced.patched({"compositing": {"light_wrap": 0.1}}),
        0,
    )
    managed_change = service.status()
    assert managed_change["axes"]["quality"]["state"] == "custom"
    assert managed_change["pending_restart_fields"] == []

    service, runtime, _store = _service(tmp_path / "unmanaged", base=balanced)
    runtime._coordinator_writer().commit(
        balanced.patched({"background": {"blur_strength": 43}}),
        0,
    )
    unrelated_change = service.status()
    assert unrelated_change["axes"]["quality"]["state"] == "active"
    assert unrelated_change["pending_restart_fields"] == []


def test_quality_availability_is_sink_capability_scoped_and_fallback_closed(tmp_path):
    service, _runtime, _store = _service(tmp_path)
    initial = service.status()
    performance = initial["catalog"]["axes"]["quality"]["profiles"]["performance"]
    assert not performance["available"]
    assert "exact output canvas" in performance["availability_reason"]

    proven_facts = {
        **CUDA_FACTS,
        "supported_output_modes": [[1280, 720, 30], [960, 540, 30]],
    }
    service, _runtime, _store = _service(tmp_path / "proven", facts=proven_facts)
    assert service.status()["catalog"]["axes"]["quality"]["profiles"]["performance"][
        "available"
    ]
    staged = service.apply(
        {"quality": "performance"},
        expected_config_version=0,
        expected_preferences_revision=0,
        accept_experimental=True,
    )
    assert staged["axes"]["quality"]["state"] == "saved_for_restart"

    balanced = AppConfig().patched(
        PROFILE_CATALOG.definition("quality", "balanced").detached_patch()
    )
    unsafe_facts = {
        **CUDA_FACTS,
        "output_backend": "NullOutput",
        "output_fallback_active": True,
        "segmentation_fallback_active": True,
        "segmentation_selection": {
            "selected_backend": "mediapipe",
            "active_provider": "cpu",
        },
    }
    unsafe, _runtime, _store = _service(
        tmp_path / "unsafe", base=balanced, facts=unsafe_facts
    )
    status = unsafe.status()
    assert status["axes"]["quality"]["active"] == "balanced"
    assert status["axes"]["quality"]["state"] == "configured_unavailable"
    assert not status["catalog"]["axes"]["quality"]["profiles"]["balanced"]["available"]


def test_profile_write_holds_runtime_revision_guard(tmp_path):
    entered_write = threading.Event()
    release_write = threading.Event()

    class PausingStore(ProfilePreferencesStore):
        def _write_locked(self, preferences):
            entered_write.set()
            assert release_write.wait(2.0)
            super()._write_locked(preferences)

    base = AppConfig()
    runtime = RuntimeConfig(base)
    store = PausingStore(tmp_path / "config" / "profile-preferences.yaml")
    service = ProfileService(
        runtime,
        ProfileStartupContext(base, ProfilePreferences(0, {}), store, {}),
        runtime_facts=lambda: CUDA_FACTS,
    )
    apply_errors = []
    commit_errors = []
    apply_done = threading.Event()
    commit_done = threading.Event()
    candidate = base.patched({"background": {"blur_strength": 43}})
    writer = runtime._coordinator_writer()

    def apply_profile():
        try:
            service.apply(
                {"quality": "balanced"},
                expected_config_version=0,
                expected_preferences_revision=0,
                accept_experimental=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            apply_errors.append(exc)
        finally:
            apply_done.set()

    def commit_config():
        try:
            writer.commit(candidate, 0)
        except BaseException as exc:  # pragma: no cover - asserted below
            commit_errors.append(exc)
        finally:
            commit_done.set()

    apply_thread = threading.Thread(target=apply_profile)
    apply_thread.start()
    assert entered_write.wait(2.0)
    commit_thread = threading.Thread(target=commit_config)
    commit_thread.start()
    assert not commit_done.wait(0.05)
    release_write.set()
    assert apply_done.wait(2.0)
    assert commit_done.wait(2.0)
    apply_thread.join()
    commit_thread.join()
    assert not apply_errors
    assert not commit_errors
    assert store.read().revision == 1
    assert runtime.version == 1


def test_startup_precedence_preferences_then_cli_and_recovery_bypass(tmp_path):
    config_path = tmp_path / "operator.yaml"
    config_path.write_text("camera:\n  width: 640\n  height: 360\n  fps: 24\n")
    store_path = tmp_path / "managed" / "profile-preferences.yaml"
    service, _runtime, source_store = _service(tmp_path / "source")
    service.apply(
        {"quality": "balanced"},
        expected_config_version=0,
        expected_preferences_revision=0,
        accept_experimental=True,
    )
    store_path.parent.mkdir()
    store_path.parent.chmod(0o700)
    store_path.write_bytes(source_store.path.read_bytes())
    store_path.chmod(0o600)

    args = build_parser().parse_args(["--config", str(config_path), "--width", "800"])
    resolved = resolve_config_from_args(args, preferences_path=store_path)
    assert resolved.config.camera.width == 800
    assert resolved.config.camera.height == 720
    assert resolved.config.camera.fps == 15
    assert resolved.config.output.fps == 30
    assert resolved.profile_startup.cli_locks == {"camera.width"}

    store_path.unlink()
    store_path.symlink_to(tmp_path / "missing")
    bypass = build_parser().parse_args(
        ["--config", str(config_path), "--no-profile-preferences"]
    )
    resolved = resolve_config_from_args(bypass, preferences_path=store_path)
    assert resolved.config.camera.width == 640
    assert resolved.profile_startup.store is None


def test_startup_reports_preference_operator_conflicts_as_recoverable(tmp_path):
    config_path = tmp_path / "operator.yaml"
    config_path.write_text(
        "segmentation:\n  backend: auto\n  model_path: /models/custom.tflite\n"
    )
    preferences_path = tmp_path / "managed" / "profile-preferences.yaml"
    preferences_path.parent.mkdir()
    preferences_path.parent.chmod(0o700)
    preferences_path.write_text(
        yaml.safe_dump(
            {
                "schema": "custback.profile-preferences",
                "version": 1,
                "revision": 1,
                "values": {"acceleration": {"mode": "gpu_required"}},
            }
        )
    )
    preferences_path.chmod(0o600)
    args = build_parser().parse_args(["--config", str(config_path)])
    with pytest.raises(
        ProfilePreferencesError,
        match="conflict with the operator configuration",
    ):
        resolve_config_from_args(args, preferences_path=preferences_path)


def test_native_output_rejects_performance_canvas():
    definition = PROFILE_CATALOG.definition("quality", "performance")
    available, reason = profile_availability(
        definition,
        AppConfig.from_dict({"output": {"backend": "native"}}),
        {
            "output_backend": "NativeVirtualCameraOutput",
            "output_fallback_active": False,
            "acceleration_state": "gpu_active",
            "acceleration_active_provider": "cuda",
            "segmentation_device": "cuda",
            "acceleration_fallback_active": False,
            "segmentation_fallback_active": False,
            "segmentation_selection": {
                "selected_backend": "rvm",
                "active_provider": "cuda",
            },
        },
        platform="win32",
    )
    assert not available
    assert "native output" in reason


def _probe_report(*, fps: float, close_error=None):
    return {
        "requested": {"width": 1280, "height": 720, "fps": 15},
        "negotiated": {
            "width": 1280,
            "height": 720,
            "delivered_width": 1280,
            "delivered_height": 720,
        },
        "measurement": {
            "capture_error": None,
            "close_error": close_error,
            "read_failures": 0,
            "restarts": 0,
            "geometry_transitions": 0,
            "stalled_at_end": False,
        },
        "timing": {"measurement_window": {"availability_fps": fps}},
        "pacing": {
            "capture": {
                "active_source_fps": fps,
                "wall_completion_fps": fps,
            }
        },
        "diagnosis": {
            "measurement_window_complete": True,
            "measurement_window_sustained": True,
            "target_sustained": True,
        },
    }


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda report: report["diagnosis"].update(
                measurement_window_complete=False
            ),
            "capture-window-incomplete",
        ),
        (
            lambda report: report["timing"]["measurement_window"].update(
                availability_fps=14.49
            ),
            "capture-cadence-shortfall",
        ),
        (
            lambda report: report["diagnosis"].update(target_sustained=False),
            "capture-target-not-sustained",
        ),
        (
            lambda report: report["measurement"].update(geometry_transitions=1),
            "capture-health-failure",
        ),
    ],
)
def test_system_profile_probe_suitability_requires_complete_stable_window(
    mutation,
    reason,
):
    report = _probe_report(fps=15.0)
    mutation(report)
    suitability = probe_mod._suitability(report, 15)
    assert suitability["capture_suitable"] is False
    assert reason in suitability["reasons"]


def test_system_profile_probe_accepts_locally_screened_with_acknowledgment(
    monkeypatch,
):
    source = files("custback").joinpath("system-profile-catalog.json").read_text()
    raw = json.loads(source)
    raw["axes"]["quality"]["profiles"]["balanced"]["evidence_state"] = (
        "locally_screened"
    )
    monkeypatch.setattr(
        probe_mod,
        "PROFILE_CATALOG",
        load_profile_catalog(json.dumps(raw).encode()),
    )
    monkeypatch.setattr(
        probe_mod,
        "run_capture_only",
        lambda cfg, **_kwargs: _probe_report(fps=float(cfg.camera.fps)),
    )

    report = probe_mod.run_system_profile_probe(
        AppConfig(),
        ["balanced"],
        accept_experimental=True,
    )

    assert report["completed_profiles"] == 1
    assert report["profiles"][0]["suitability"]["capture_suitable"] is True


def test_system_profile_probe_is_sequential_capture_only_and_aborts_on_close(
    monkeypatch,
):
    calls = []

    def fake_run(cfg, **_kwargs):
        calls.append((cfg.camera.width, cfg.camera.height, cfg.camera.fps))
        report = _probe_report(fps=float(cfg.camera.fps))
        report["requested"] = {"width": cfg.camera.width, "height": cfg.camera.height}
        report["negotiated"] = {
            "width": cfg.camera.width,
            "height": cfg.camera.height,
            "delivered_width": cfg.camera.width,
            "delivered_height": cfg.camera.height,
        }
        if len(calls) == 2:
            report["measurement"]["close_error"] = "CloseError"
        return report

    monkeypatch.setattr(probe_mod, "run_capture_only", fake_run)
    report = probe_mod.run_system_profile_probe(
        AppConfig(),
        ["performance", "balanced", "quality"],
        accept_experimental=True,
    )
    assert calls == [(960, 540, 30), (1280, 720, 15)]
    assert report["completed_profiles"] == 2
    assert report["aborted_reason"] == "reader-close-failure"
    assert report["preferences_mutated"] is False
    assert report["full_path_qualification_inferred"] is False


def test_profiles_api_is_authenticated_revision_bound_and_path_free(tmp_path):
    service, runtime, _store = _service(tmp_path)
    security = SecurityPolicy.for_bind(
        TOKEN,
        "testserver",
        80,
        allowed_origins=["http://testserver"],
        extra_hosts=["testserver"],
    )
    app = create_app(
        runtime,
        FrameHub(),
        _Coordinator(),
        security=security,
        upload_dir=tmp_path / "uploads",
        profile_service=service,
    )

    async def request(method: str, path: str, **kwargs):
        transport = cast(Any, _ASGITransport)(app=app)
        async with cast(Any, _AsyncClient)(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    assert _run_async(request("GET", "/profiles")).status_code == 401
    initial = _run_async(request("GET", "/profiles", headers=AUTH))
    assert initial.status_code == 200
    assert initial.json()["preference_revision"] == 0
    encoded = json.dumps(initial.json(), sort_keys=True)
    assert "/models/" not in encoded
    assert "token_file" not in encoded
    assert "device" not in initial.json()["catalog"]["axes"]["quality"]["owned_paths"]

    body = {
        "selections": {"quality": "balanced"},
        "expected_config_version": 0,
        "expected_preferences_revision": 0,
        "accept_experimental": False,
    }
    denied = _run_async(request("POST", "/profiles/apply", headers=AUTH, json=body))
    assert denied.status_code == 409
    assert denied.json()["detail"]["code"] == "experimental_ack_required"

    body["accept_experimental"] = True
    applied = _run_async(request("POST", "/profiles/apply", headers=AUTH, json=body))
    assert applied.status_code == 200
    assert applied.json()["axes"]["quality"]["state"] == "saved_for_restart"
    stale = _run_async(request("POST", "/profiles/apply", headers=AUTH, json=body))
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "stale_preference_revision"

    reset = _run_async(
        request(
            "POST",
            "/profiles/reset",
            headers=AUTH,
            json={
                "axes": ["quality"],
                "expected_config_version": 0,
                "expected_preferences_revision": 1,
            },
        )
    )
    assert reset.status_code == 200
    assert reset.json()["axes"]["quality"]["state"] == "custom"
