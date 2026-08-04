"""Acceleration policy: provider resolution, latched state, and real-RVM proof.

No GPU, no onnxruntime, no model download: the proof path is exercised with a
fake ORT that writes an ORT-shaped profile file, exactly as ``prove_rvm_provider``
inspects it.
"""

import ctypes
import json
import sys
import tempfile
import types
import uuid
from typing import Any

import numpy as np
import pytest

import custback.acceleration as acceleration
from custback.acceleration import (
    AccelerationState,
    AccelState,
    GpuRequiredError,
    ProviderCandidate,
    collect_cuda_device_identity,
    provider_label,
    preload_acceleration_dlls,
    prove_rvm_provider,
    resolve_provider_candidates,
    synthetic_rvm_feeds,
)
from custback.config import AccelerationConfig, AppConfig


# -- exact CUDA device identity ----------------------------------------------


class _FakeCFunction:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.argtypes: Any = None
        self.restype: Any = None

    def __call__(self, *args: Any) -> Any:
        return self.callback(*args)


def _set_int(pointer: Any, value: int) -> None:
    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = value


def _set_size(pointer: Any, value: int) -> None:
    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t)).contents.value = value


def _set_handle(pointer: Any, value: int) -> None:
    ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = value


def _write_bytes(buffer: Any, value: bytes) -> None:
    buffer.value = value


def _fake_cuda_identity_libraries(
    *,
    count: int = 2,
    uuid_kind: str = "gpu",
    nvml_uuid_override: str | None = None,
    pci_bus_id: bytes = b"0000:65:00.0",
    driver_pci_bus_id: bytes = b"0000:65:00.0",
    nvml_name: bytes = b"NVIDIA Test GPU",
    nvml_memory_bytes: int = 24 * 1024**3,
    other_uuid_result: int = 6,
    shutdown_result: int = 0,
) -> tuple[tuple[Any, Any, Any], dict[str, int]]:
    state = {"init": 0, "shutdown": 0}
    raw_uuid = uuid.UUID("00112233-4455-6677-8899-aabbccddeeff").bytes
    uuid_prefix = "GPU-" if uuid_kind == "gpu" else "MIG-"
    selected_uuid = f"{uuid_prefix}{uuid.UUID(bytes=raw_uuid)}"

    def device_get_count(pointer: Any) -> int:
        _set_int(pointer, count)
        return 0

    def device_get(pointer: Any, ordinal: int) -> int:
        if ordinal < 0 or ordinal >= count:
            return 101
        _set_int(pointer, 7)
        return 0

    def device_get_name(buffer: Any, _length: int, device: int) -> int:
        if device != 7:
            return 101
        _write_bytes(buffer, b"NVIDIA Test GPU")
        return 0

    def device_get_uuid(pointer: Any, device: int) -> int:
        if device != 7:
            return 101
        ctypes.memmove(pointer, raw_uuid, len(raw_uuid))
        return 0

    def device_total_memory(pointer: Any, device: int) -> int:
        if device != 7:
            return 101
        _set_size(pointer, 24 * 1024**3)
        return 0

    def device_get_attribute(pointer: Any, attribute: int, device: int) -> int:
        if device != 7:
            return 101
        values = {75: 8, 76: 9}
        if attribute not in values:
            return 1
        _set_int(pointer, values[attribute])
        return 0

    def driver_version(pointer: Any) -> int:
        _set_int(pointer, 12040)
        return 0

    def runtime_version(pointer: Any) -> int:
        _set_int(pointer, 12030)
        return 0

    def get_pci_bus_id(buffer: Any, _length: int, ordinal: int) -> int:
        if ordinal < 0 or ordinal >= count:
            return 101
        _write_bytes(buffer, pci_bus_id)
        return 0

    def driver_get_pci_bus_id(buffer: Any, _length: int, device: int) -> int:
        if device != 7:
            return 101
        _write_bytes(buffer, driver_pci_bus_id)
        return 0

    def nvml_init() -> int:
        state["init"] += 1
        return 0

    def nvml_shutdown() -> int:
        state["shutdown"] += 1
        return shutdown_result

    def nvml_handle_by_uuid(candidate: bytes, pointer: Any) -> int:
        if candidate.decode("ascii") != selected_uuid:
            return other_uuid_result
        _set_handle(pointer, 0x1234)
        return 0

    def nvml_device_uuid(
        handle: Any,
        buffer: Any,
        _length: int,
    ) -> int:
        if getattr(handle, "value", handle) != 0x1234:
            return 2
        _write_bytes(
            buffer,
            (nvml_uuid_override or selected_uuid).encode("ascii"),
        )
        return 0

    def nvml_device_name(
        handle: Any,
        buffer: Any,
        _length: int,
    ) -> int:
        if getattr(handle, "value", handle) != 0x1234:
            return 2
        _write_bytes(buffer, nvml_name)
        return 0

    def nvml_memory(handle: Any, pointer: Any) -> int:
        if getattr(handle, "value", handle) != 0x1234:
            return 2
        values = (ctypes.c_ulonglong * 3)(
            nvml_memory_bytes,
            nvml_memory_bytes,
            0,
        )
        ctypes.memmove(pointer, values, ctypes.sizeof(values))
        return 0

    def system_driver_version(buffer: Any, _length: int) -> int:
        _write_bytes(buffer, b"595.84")
        return 0

    def system_nvml_version(buffer: Any, _length: int) -> int:
        _write_bytes(buffer, b"13.595.84")
        return 0

    driver = types.SimpleNamespace(
        cuInit=_FakeCFunction(lambda _flags: 0),
        cuDeviceGetCount=_FakeCFunction(device_get_count),
        cuDeviceGet=_FakeCFunction(device_get),
        cuDeviceGetName=_FakeCFunction(device_get_name),
        cuDeviceGetUuid_v2=_FakeCFunction(device_get_uuid),
        cuDeviceTotalMem_v2=_FakeCFunction(device_total_memory),
        cuDeviceGetAttribute=_FakeCFunction(device_get_attribute),
        cuDeviceGetPCIBusId=_FakeCFunction(driver_get_pci_bus_id),
        cuDriverGetVersion=_FakeCFunction(driver_version),
    )
    runtime = types.SimpleNamespace(
        cudaDeviceGetPCIBusId=_FakeCFunction(get_pci_bus_id),
        cudaDriverGetVersion=_FakeCFunction(driver_version),
        cudaRuntimeGetVersion=_FakeCFunction(runtime_version),
    )
    nvml = types.SimpleNamespace(
        nvmlInit_v2=_FakeCFunction(nvml_init),
        nvmlShutdown=_FakeCFunction(nvml_shutdown),
        nvmlSystemGetDriverVersion=_FakeCFunction(system_driver_version),
        nvmlSystemGetNVMLVersion=_FakeCFunction(system_nvml_version),
        nvmlDeviceGetHandleByUUID=_FakeCFunction(nvml_handle_by_uuid),
        nvmlDeviceGetUUID=_FakeCFunction(nvml_device_uuid),
        nvmlDeviceGetName=_FakeCFunction(nvml_device_name),
        nvmlDeviceGetMemoryInfo=_FakeCFunction(nvml_memory),
    )
    return (driver, runtime, nvml), state


@pytest.mark.parametrize(
    ("uuid_kind", "expected_uuid"),
    [
        ("gpu", "GPU-00112233-4455-6677-8899-aabbccddeeff"),
        ("mig", "MIG-00112233-4455-6677-8899-aabbccddeeff"),
    ],
)
def test_collect_cuda_identity_returns_exact_json_safe_facts(
    monkeypatch: pytest.MonkeyPatch,
    uuid_kind: str,
    expected_uuid: str,
):
    libraries, state = _fake_cuda_identity_libraries(uuid_kind=uuid_kind)
    monkeypatch.setattr(
        acceleration,
        "_load_cuda_identity_libraries",
        lambda: libraries,
    )

    result = collect_cuda_device_identity(1)

    assert result == {
        "identity_source": "custback-cuda-driver-runtime-nvml-v1",
        "ordinal": 1,
        "uuid": expected_uuid,
        "uuid_kind": uuid_kind,
        "pci_bus_id": "00000000:65:00.0",
        "cuda_name": "NVIDIA Test GPU",
        "nvml_name": "NVIDIA Test GPU",
        "cuda_total_memory_bytes": 24 * 1024**3,
        "nvml_total_memory_bytes": 24 * 1024**3,
        "compute_capability_major": 8,
        "compute_capability_minor": 9,
        "cuda_driver_version": 12040,
        "cuda_runtime_version": 12030,
        "nvidia_driver_version": "595.84",
        "nvml_version": "13.595.84",
    }
    assert json.loads(json.dumps(result)) == result
    assert state == {"init": 1, "shutdown": 1}


@pytest.mark.parametrize("device_id", [True, -1, 65, 0.0, "0", None])
def test_collect_cuda_identity_rejects_invalid_ordinals(device_id: Any):
    with pytest.raises(ValueError, match=r"integer in \[0, 64\]"):
        collect_cuda_device_identity(device_id)


def test_collect_cuda_identity_rejects_unavailable_ordinal(
    monkeypatch: pytest.MonkeyPatch,
):
    libraries, state = _fake_cuda_identity_libraries(count=1)
    monkeypatch.setattr(
        acceleration,
        "_load_cuda_identity_libraries",
        lambda: libraries,
    )

    with pytest.raises(
        GpuRequiredError,
        match=r"^CUDA device identity is unavailable$",
    ):
        collect_cuda_device_identity(1)

    assert state == {"init": 0, "shutdown": 0}


@pytest.mark.parametrize(
    ("overrides", "expected_nvml_calls"),
    [
        (
            {"nvml_uuid_override": ("GPU-ffeeddcc-bbaa-9988-7766-554433221100")},
            {"init": 1, "shutdown": 1},
        ),
        ({"pci_bus_id": b"65:00.0"}, {"init": 0, "shutdown": 0}),
        (
            {"pci_bus_id": b"0000:66:00.0"},
            {"init": 0, "shutdown": 0},
        ),
        ({"nvml_name": b""}, {"init": 1, "shutdown": 1}),
        ({"nvml_memory_bytes": 0}, {"init": 1, "shutdown": 1}),
        ({"other_uuid_result": 999}, {"init": 1, "shutdown": 1}),
        ({"shutdown_result": 1}, {"init": 1, "shutdown": 1}),
    ],
)
def test_collect_cuda_identity_fails_closed_and_shuts_down_nvml(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    expected_nvml_calls: dict[str, int],
):
    libraries, state = _fake_cuda_identity_libraries(**overrides)
    monkeypatch.setattr(
        acceleration,
        "_load_cuda_identity_libraries",
        lambda: libraries,
    )

    with pytest.raises(
        GpuRequiredError,
        match=r"^CUDA device identity is unavailable$",
    ):
        collect_cuda_device_identity(0)

    assert state == expected_nvml_calls


def test_collect_cuda_identity_redacts_loader_errors_and_does_not_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    def fail_load() -> tuple[Any, Any, Any]:
        raise OSError("/private/host/path/libcuda.so: secret loader detail")

    monkeypatch.setattr(acceleration, "_load_cuda_identity_libraries", fail_load)

    with pytest.raises(
        GpuRequiredError,
        match=r"^CUDA device identity is unavailable$",
    ):
        collect_cuda_device_identity(0)

    assert not caplog.records


# -- config model -------------------------------------------------------------


def test_default_acceleration_is_auto():
    accel = AppConfig().acceleration
    assert (accel.mode, accel.provider, accel.device_id) == ("auto", "auto", 0)


def test_cpu_mode_forbids_pinned_gpu_provider():
    with pytest.raises(ValueError):
        AccelerationConfig(mode="cpu", provider="cuda")
    # cpu + auto is fine (auto simply resolves to no GPU under cpu mode).
    assert AccelerationConfig(mode="cpu").provider == "auto"


def test_config_without_acceleration_section_loads_with_default():
    # Backward compatibility: a pre-Phase-4 config has no acceleration section,
    # and extra="forbid" must still accept it, filling the safe default.
    cfg = AppConfig.from_dict({"segmentation": {"backend": "auto"}})
    assert cfg.acceleration == AccelerationConfig()
    # And an explicit section round-trips.
    loaded = AppConfig.from_dict(
        {"acceleration": {"mode": "gpu_required", "provider": "cuda", "device_id": 1}}
    )
    assert loaded.acceleration.mode == "gpu_required"
    assert loaded.acceleration.device_id == 1


@pytest.mark.parametrize(
    "segmentation",
    [
        {"backend": "mediapipe"},
        {"backend": "heuristic"},
        {"backend": "none"},
        {"backend": "auto", "model_path": "/private/custom.tflite"},
    ],
)
def test_gpu_required_rejects_non_rvm_eligible_segmentation(segmentation):
    with pytest.raises(ValueError, match="requires an RVM-eligible"):
        AppConfig.from_dict(
            {
                "segmentation": segmentation,
                "acceleration": {"mode": "gpu_required"},
            }
        )


def test_device_id_bounds():
    with pytest.raises(ValueError):
        AccelerationConfig(device_id=-1)
    with pytest.raises(ValueError):
        AccelerationConfig(device_id=65)


# -- provider resolution ------------------------------------------------------


def test_cpu_mode_resolves_no_gpu_candidates():
    got = resolve_provider_candidates(
        AccelerationConfig(mode="cpu"),
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert got == []


def test_auto_prefers_cuda_then_directml():
    got = resolve_provider_candidates(
        AccelerationConfig(),
        ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert [c.name for c in got] == ["CUDAExecutionProvider", "DmlExecutionProvider"]


def test_pinned_provider_absent_yields_no_candidate():
    got = resolve_provider_candidates(
        AccelerationConfig(provider="directml"),
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert got == []


def test_device_id_threaded_into_gpu_options():
    got = resolve_provider_candidates(
        AccelerationConfig(provider="cuda", device_id=2),
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert got[0].options == {"device_id": 2}


def test_provider_label():
    assert provider_label("CUDAExecutionProvider") == "cuda"
    assert provider_label("DmlExecutionProvider") == "directml"
    assert provider_label("CPUExecutionProvider") == "cpu"


# -- latched state machine ----------------------------------------------------


def test_state_transitions_and_status():
    state = AccelerationState(AccelerationConfig(provider="cuda", device_id=3))
    assert state.status().state == AccelState.STARTING.value
    state.mark_probing()
    assert state.status().state == AccelState.GPU_PROBING.value
    state.mark_gpu_active("CUDAExecutionProvider")
    status = state.status()
    assert status.state == AccelState.GPU_ACTIVE.value
    assert status.active_provider == "cuda"
    assert status.fallback_active is False
    assert status.device_id == 3
    assert state.on_gpu is True


def test_fallback_latches_and_counts_once():
    state = AccelerationState(AccelerationConfig())
    state.mark_gpu_active("CUDAExecutionProvider")
    state.latch_fallback("cuda out of memory in session.run")
    first = state.status()
    assert first.state == AccelState.CPU_FALLBACK.value
    assert first.active_provider == "cpu"
    assert first.fallback_active is True
    assert first.fallback_count == 1
    assert first.fallback_reason == "cuda out of memory in session.run"
    # A second latch does not re-count, and GPU can never re-activate.
    state.latch_fallback("still broken")
    state.mark_gpu_active("CUDAExecutionProvider")
    after = state.status()
    assert after.fallback_count == 1
    assert after.state == AccelState.CPU_FALLBACK.value
    assert state.on_gpu is False


def test_intended_cpu_is_not_a_fallback():
    state = AccelerationState(AccelerationConfig(mode="cpu"))
    state.mark_cpu_active()
    status = state.status()
    assert status.active_provider == "cpu"
    assert status.fallback_active is False
    assert status.fallback_count == 0
    assert state.on_gpu is False


def test_fallback_reason_is_bounded():
    state = AccelerationState(AccelerationConfig())
    state.mark_gpu_active("CUDAExecutionProvider")
    state.latch_fallback("x " * 500)
    assert len(state.status().fallback_reason) <= 200


# -- real-RVM provider proof --------------------------------------------------


def _fake_ort(*, active_provider_in_profile):
    mod: Any = types.ModuleType("onnxruntime")
    mod.proof_session_options = None
    mod.proof_session_providers = None
    mod.proof_model_source = None

    class SessionOptions:
        def __init__(self):
            self.enable_profiling = False
            self._entries = {}

        def add_session_config_entry(self, key, value):
            self._entries[key] = value

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            mod.proof_model_source = path
            names = [p[0] if isinstance(p, tuple) else p for p in providers or []]
            if bool(getattr(sess_options, "enable_profiling", False)):
                mod.proof_session_options = sess_options
                mod.proof_session_providers = names
                if (
                    getattr(sess_options, "_entries", {}).get(
                        "session.disable_cpu_ep_fallback"
                    )
                    == "1"
                    and "CPUExecutionProvider" in names
                ):
                    raise ValueError(
                        "explicit CPU EP conflicts with disabled CPU EP fallback"
                    )
            self._providers = names or ["CPUExecutionProvider"]
            self._profiling = bool(getattr(sess_options, "enable_profiling", False))

        def get_providers(self):
            return self._providers

        def run(self, _outputs, feeds):
            return [feeds["src"], np.zeros((1, 1, 4, 4), np.float32)]

        def end_profiling(self):
            if not self._profiling:
                return ""
            provider = (
                self._providers[0]
                if active_provider_in_profile
                else "CPUExecutionProvider"
            )
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, prefix="accel-test-"
            )
            with handle:
                json.dump([{"cat": "Node", "args": {"provider": provider}}], handle)
            return handle.name

    mod.SessionOptions = SessionOptions
    mod.InferenceSession = InferenceSession
    return mod


def test_prove_rvm_provider_confirms_gpu_execution():
    ort = _fake_ort(active_provider_in_profile=True)
    candidate = ProviderCandidate("CUDAExecutionProvider", {"device_id": 0})
    model_payload = b"immutable-model"
    result = prove_rvm_provider(ort, model_payload, candidate)
    assert result.proven is True
    assert ort.proof_model_source is model_payload
    assert "CUDAExecutionProvider" in result.active_providers
    assert ort.proof_session_providers == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert ort.proof_session_options is not None
    assert "session.disable_cpu_ep_fallback" not in ort.proof_session_options._entries


def test_prove_rvm_provider_rejects_registered_but_cpu_executed():
    # Registered provider, but the profile shows every node ran on CPU.
    ort = _fake_ort(active_provider_in_profile=False)
    candidate = ProviderCandidate("CUDAExecutionProvider", {"device_id": 0})
    result = prove_rvm_provider(ort, "/fake/model.onnx", candidate)
    assert result.proven is False
    assert result.error


def test_prove_rvm_provider_handles_construction_failure():
    ort: Any = types.ModuleType("onnxruntime")

    class SessionOptions:
        def add_session_config_entry(self, *a):
            pass

    def boom(*a, **k):
        raise RuntimeError("CUDA driver missing libcudart.so")

    ort.SessionOptions = SessionOptions
    ort.InferenceSession = boom
    candidate = ProviderCandidate("CUDAExecutionProvider", {"device_id": 0})
    result = prove_rvm_provider(ort, "/fake/model.onnx", candidate)
    assert result.proven is False
    assert "libcudart" in result.error


# -- misc helpers -------------------------------------------------------------


def test_synthetic_feeds_shape():
    feeds = synthetic_rvm_feeds(48, 80)
    assert feeds["src"].shape == (1, 3, 48, 80)
    assert feeds["src"].dtype == np.float32
    assert set(feeds) == {"src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX no-op path")
def test_preload_dlls_is_noop_off_windows():
    # Must not raise and must do nothing observable on POSIX.
    preload_acceleration_dlls("CUDAExecutionProvider")
