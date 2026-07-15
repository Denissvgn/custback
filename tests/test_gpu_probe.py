from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from custback.gpu_probe import _CUDA_PROBE_MODEL, _profile_uses_cuda, probe_cuda_inference


class FakeOptions:
    def __init__(self):
        self.enable_profiling = False
        self.profile_file_prefix = ""
        self.graph_optimization_level = None
        self.entries = {}

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class FakeSession:
    def __init__(self, options, *, provider="CUDAExecutionProvider", output=None):
        self.options = options
        self.provider = provider
        self.output = output if output is not None else np.asarray([3.0], dtype=np.float32)

    def get_providers(self):
        return [self.provider]

    def run(self, output_names, inputs):
        assert output_names is None
        assert np.array_equal(inputs["x"], np.asarray([1.0], dtype=np.float32))
        assert np.array_equal(inputs["y"], np.asarray([2.0], dtype=np.float32))
        return [self.output]

    def end_profiling(self):
        path = Path(f"{self.options.profile_file_prefix}.json")
        path.write_text(
            json.dumps(
                [
                    {
                        "cat": "Node",
                        "name": "cuda_probe_add_kernel_time",
                        "args": {"op_name": "Add", "provider": self.provider},
                    }
                ]
            ),
            encoding="utf-8",
        )
        return str(path)


class FakeOrt:
    class GraphOptimizationLevel:
        ORT_DISABLE_ALL = "disabled"

    SessionOptions = FakeOptions

    def __init__(self, *, providers=None, session_provider="CUDAExecutionProvider", output=None):
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session_provider = session_provider
        self.output = output
        self.observed_model = None
        self.observed_options = None

    def get_available_providers(self):
        return self.providers

    def InferenceSession(self, model, *, sess_options, providers):
        self.observed_model = model
        self.observed_options = sess_options
        assert providers == ["CUDAExecutionProvider"]
        return FakeSession(
            sess_options,
            provider=self.session_provider,
            output=self.output,
        )


def test_probe_requires_verified_cuda_node_and_output():
    ort = FakeOrt()
    result = probe_cuda_inference(ort)

    assert result["onnxruntime"] is True
    assert result["cuda_provider"] is True
    assert result["cuda_inference"] is True
    assert result["output_verified"] is True
    assert result["profile_verified"] is True
    assert result["error"] == ""
    assert ort.observed_model == _CUDA_PROBE_MODEL
    assert ort.observed_options.entries == {"session.disable_cpu_ep_fallback": "1"}


def test_advertised_cuda_with_cpu_execution_fails_closed():
    result = probe_cuda_inference(FakeOrt(session_provider="CPUExecutionProvider"))

    assert result["cuda_provider"] is True
    assert result["output_verified"] is True
    assert result["profile_verified"] is False
    assert result["cuda_inference"] is False
    assert "did not execute" in result["error"]


def test_malformed_probe_output_fails_closed():
    result = probe_cuda_inference(FakeOrt(output=np.asarray([4.0], dtype=np.float32)))

    assert result["output_verified"] is False
    assert result["cuda_inference"] is False


def test_unregistered_cuda_does_not_construct_session():
    ort = FakeOrt(providers=["CPUExecutionProvider"])
    result = probe_cuda_inference(ort)

    assert result["onnxruntime"] is True
    assert result["cuda_provider"] is False
    assert result["cuda_inference"] is False
    assert ort.observed_model is None


def test_profile_evidence_ignores_non_node_and_cpu_events():
    assert _profile_uses_cuda([
        {"cat": "Session", "args": {"provider": "CUDAExecutionProvider"}},
        {
            "cat": "Node",
            "name": "cuda_probe_add_kernel_time",
            "args": {"op_name": "Add", "provider": "CPUExecutionProvider"},
        },
    ]) is False
    assert _profile_uses_cuda([
        {
            "cat": "Node",
            "name": "cuda_probe_add_kernel_time",
            "args": {"op_name": "Add", "provider": "CUDAExecutionProvider"},
        },
    ]) is True


def test_profile_rejects_unrelated_cuda_node_when_probe_add_ran_on_cpu():
    assert _profile_uses_cuda(
        [
            {
                "cat": "Node",
                "name": "cuda_probe_add_kernel_time",
                "args": {"op_name": "Add", "provider": "CPUExecutionProvider"},
            },
            {
                "cat": "Node",
                "name": "Memcpy_kernel_time",
                "args": {
                    "op_name": "MemcpyFromHost",
                    "provider": "CUDAExecutionProvider",
                },
            },
        ]
    ) is False
