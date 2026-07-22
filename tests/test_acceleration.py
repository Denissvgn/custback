"""Acceleration policy: provider resolution, latched state, and real-RVM proof.

No GPU, no onnxruntime, no model download: the proof path is exercised with a
fake ORT that writes an ORT-shaped profile file, exactly as ``prove_rvm_provider``
inspects it.
"""

import json
import sys
import tempfile
import types

import numpy as np
import pytest

from custback.acceleration import (
    AccelerationState,
    AccelState,
    ProviderCandidate,
    provider_label,
    preload_acceleration_dlls,
    prove_rvm_provider,
    resolve_provider_candidates,
    synthetic_rvm_feeds,
)
from custback.config import AccelerationConfig, AppConfig


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
    mod = types.ModuleType("onnxruntime")

    class SessionOptions:
        def __init__(self):
            self.enable_profiling = False
            self._entries = {}

        def add_session_config_entry(self, key, value):
            self._entries[key] = value

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            names = [p[0] if isinstance(p, tuple) else p for p in providers or []]
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
    result = prove_rvm_provider(ort, "/fake/model.onnx", candidate)
    assert result.proven is True
    assert "CUDAExecutionProvider" in result.active_providers


def test_prove_rvm_provider_rejects_registered_but_cpu_executed():
    # Registered provider, but the profile shows every node ran on CPU.
    ort = _fake_ort(active_provider_in_profile=False)
    candidate = ProviderCandidate("CUDAExecutionProvider", {"device_id": 0})
    result = prove_rvm_provider(ort, "/fake/model.onnx", candidate)
    assert result.proven is False
    assert result.error


def test_prove_rvm_provider_handles_construction_failure():
    ort = types.ModuleType("onnxruntime")

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
