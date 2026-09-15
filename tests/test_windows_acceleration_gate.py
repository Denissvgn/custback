"""WIN-6.2 DirectML release-gate checker tests.

The `run` half of scripts/release/windows-acceleration-gate.py needs Windows
hardware; the `check` half is deterministic stdlib code and is fully verified
here so the go/no-go criteria are executable, not prose.  Any change to the
thresholds is a deliberate gate change and must update these tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

GATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "release"
    / "windows-acceleration-gate.py"
)

pytestmark = pytest.mark.skipif(
    not GATE_PATH.exists(),
    reason="release scripts are not present in this layout",
)


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location(
        "windows_acceleration_gate", GATE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adapter(gate, vendor: str, **overrides) -> dict:
    adapter = {
        "vendor": vendor,
        "name": f"{vendor} test adapter",
        "driver": "1.0",
        "device_id": 0,
        "proven": True,
        "proof_error": "",
        "correctness": {
            "frames": 30,
            "alpha_delta_mean_worst": 0.001,
            "alpha_delta_max": 0.01,
        },
        "performance": {
            "720p": {"median_ms": 20.0, "p95_ms": 25.0, "fps": 50.0},
            "1080p": {"median_ms": 40.0, "p95_ms": 50.0, "fps": 25.0},
        },
        "target_fps": 30,
    }
    adapter.update(overrides)
    assert set(adapter) == gate.ADAPTER_KEYS
    return adapter


def _evidence(gate, adapters: list[dict]) -> dict:
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "task": gate.TASK,
        "generated_at": "2026-07-23T00:00:00+00:00",
        "host": {"os": "Windows-11", "machine": "AMD64"},
        "runtime": {
            "onnxruntime_version": "1.22.0",
            "flavor": "directml",
            "model_file": "rvm_mobilenetv3_fp32.onnx",
            "model_sha256": "0" * 64,
        },
        "wheel_conflict": {
            "onnxruntime_gpu_installed": False,
            "onnxruntime_directml_installed": True,
        },
        "adapters": adapters,
    }


def test_go_requires_amd_and_intel_coverage(gate) -> None:
    assert gate.SCHEMA_VERSION == 2
    both = _evidence(gate, [_adapter(gate, "AMD"), _adapter(gate, "Intel")])
    go, reasons = gate.evaluate_evidence(both)
    assert go and reasons == []

    amd_only = _evidence(gate, [_adapter(gate, "AMD")])
    go, reasons = gate.evaluate_evidence(amd_only)
    assert not go
    assert any("INTEL" in reason for reason in reasons)


def test_unproven_adapter_is_no_go(gate) -> None:
    evidence = _evidence(
        gate,
        [
            _adapter(gate, "AMD", proven=False, proof_error="no node executed"),
            _adapter(gate, "Intel"),
        ],
    )
    go, reasons = gate.evaluate_evidence(evidence)
    assert not go
    assert any("no node executed" in reason for reason in reasons)


def test_alpha_drift_beyond_tolerance_is_no_go(gate) -> None:
    # The thresholds are the published gate contract; changing them must be a
    # deliberate update to these reviewed limits.
    assert gate.ALPHA_DELTA_MEAN_MAX == 0.005
    assert gate.ALPHA_DELTA_MAX_MAX == 0.02
    drifted = _adapter(gate, "AMD")
    drifted["correctness"] = {
        "frames": 30,
        "alpha_delta_mean_worst": 0.02,
        "alpha_delta_max": 0.2,
    }
    evidence = _evidence(gate, [drifted, _adapter(gate, "Intel")])
    go, reasons = gate.evaluate_evidence(evidence)
    assert not go
    assert any("alpha drift" in reason for reason in reasons)


def test_sub_target_720p_is_no_go(gate) -> None:
    slow = _adapter(gate, "Intel")
    slow["performance"]["720p"] = {"median_ms": 50.0, "p95_ms": 80.0, "fps": 20.0}
    evidence = _evidence(gate, [_adapter(gate, "AMD"), slow])
    go, reasons = gate.evaluate_evidence(evidence)
    assert not go
    assert any("below the 30 fps target" in reason for reason in reasons)


def test_wheel_coinstallation_is_no_go(gate) -> None:
    evidence = _evidence(gate, [_adapter(gate, "AMD"), _adapter(gate, "Intel")])
    evidence["wheel_conflict"]["onnxruntime_gpu_installed"] = True
    go, reasons = gate.evaluate_evidence(evidence)
    assert not go
    assert any("co-installed" in reason for reason in reasons)


def test_schema_is_exact_match(gate) -> None:
    evidence = _evidence(gate, [_adapter(gate, "AMD"), _adapter(gate, "Intel")])
    evidence["extra_key"] = True
    go, reasons = gate.evaluate_evidence(evidence)
    assert not go
    assert reasons == ["evidence does not match the exact WIN-6.2 schema"]

    stray = _adapter(gate, "AMD")
    stray["surprise"] = 1
    go, reasons = gate.evaluate_evidence(
        _evidence(gate, [stray, _adapter(gate, "Intel")])
    )
    assert not go


def test_schema_1_evidence_is_rejected_before_schema_2_evaluation(gate) -> None:
    legacy = _evidence(gate, [_adapter(gate, "AMD"), _adapter(gate, "Intel")])
    legacy["schema_version"] = 1
    for adapter in legacy["adapters"]:
        correctness = adapter["correctness"]
        correctness["alpha_delta_mean"] = correctness.pop("alpha_delta_mean_worst")

    go, reasons = gate.evaluate_evidence(legacy)
    assert not go
    assert reasons == ["wrong schema_version/task for the WIN-6.2 gate"]


def test_schema_2_rejects_the_old_correctness_field(gate) -> None:
    evidence = _evidence(gate, [_adapter(gate, "AMD"), _adapter(gate, "Intel")])
    correctness = evidence["adapters"][0]["correctness"]
    correctness["alpha_delta_mean"] = correctness.pop("alpha_delta_mean_worst")

    go, reasons = gate.evaluate_evidence(evidence)
    assert not go
    assert any(
        "correctness does not match the exact schema" in reason for reason in reasons
    )


def test_check_cli_exit_codes(gate, tmp_path, capsys) -> None:
    evidence = _evidence(gate, [_adapter(gate, "AMD"), _adapter(gate, "Intel")])
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    assert gate.main(["check", str(path)]) == 0
    assert "GO" in capsys.readouterr().out

    evidence["adapters"] = [_adapter(gate, "AMD")]
    path.write_text(json.dumps(evidence), encoding="utf-8")
    assert gate.main(["check", str(path), "--json"]) == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["go"] is False and verdict["reasons"]


def test_run_strict_exit_semantics(gate, capsys) -> None:
    no_go = _evidence(gate, [_adapter(gate, "AMD")])
    assert gate._report_local_verdict(no_go, strict=False) == 0
    captured = capsys.readouterr()
    assert "NO-GO" in captured.out
    assert "CI must gate on `check`" in captured.err

    assert gate._report_local_verdict(no_go, strict=True) == 1
    assert "NO-GO" in capsys.readouterr().out

    go = _evidence(gate, [_adapter(gate, "AMD"), _adapter(gate, "Intel")])
    assert gate._report_local_verdict(go, strict=True) == 0
    assert "GO" in capsys.readouterr().out


def test_run_cli_threads_strict_option(gate, monkeypatch) -> None:
    seen: list[bool] = []

    def fake_run(args) -> int:
        seen.append(args.strict)
        return int(args.strict)

    monkeypatch.setattr(gate, "run", fake_run)
    assert gate.main(["run", "--model", "model.onnx"]) == 0
    assert gate.main(["run", "--model", "model.onnx", "--strict"]) == 1
    assert seen == [False, True]


def test_run_mode_refuses_non_windows(gate, monkeypatch, capsys) -> None:
    monkeypatch.setattr(gate.sys, "platform", "linux")
    assert gate.main(["run", "--model", "missing.onnx"]) == 1
    assert "Windows hardware" in capsys.readouterr().err


def test_directml_extra_is_exclusive_with_cuda_in_build_script(gate) -> None:
    # The wheel-conflict rule the checker enforces at release time is also
    # enforced at freeze time by the PyInstaller build script.
    build = (
        Path(__file__).resolve().parents[1]
        / "packaging"
        / "windows"
        / "pyinstaller"
        / "build.ps1"
    ).read_text(encoding="utf-8")
    assert "mutually exclusive" in build
    assert "directml" in build
