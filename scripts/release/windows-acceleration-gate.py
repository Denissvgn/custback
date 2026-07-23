#!/usr/bin/env python3
"""WIN-6.2 generic-GPU (DirectML) release-gate harness and checker.

The WIN-4.7 spike (``WINDOWS_ACCELERATION_SPIKE.md``) defined *what* evidence
justifies a "GPU-capable on all Windows hardware" claim; this script makes
that evidence reproducible and its evaluation deterministic:

``run``
    Executes on Windows hardware with ``onnxruntime-directml`` installed:
    proves real RVM node execution on ``DmlExecutionProvider`` (the same
    profiling standard CUDA must meet, via ``custback.acceleration``),
    measures per-pixel alpha drift against the CPU reference on a fixed
    synthetic sequence, times 720p/1080p inference, and writes one evidence
    JSON per machine.

``check``
    Runs anywhere (stdlib only): validates an evidence file against the exact
    schema and the go/no-go criteria below.  Exit 0 = **go** (the WIN-6.2
    claim may be advertised for the tested adapters); exit 1 = **no-go** with
    reasons.  This checker is the validator the ``windows-acceleration`` gate
    slot (WINDOWS_DECISIONS.md Part B, item 3) will invoke once WIN-1.8 wires
    a windows evidence source; until then it gates nothing and the feature
    matrix keeps DirectML unadvertised.

Go criteria (from the spike, encoded here so they cannot drift in prose):
    * every listed adapter proves real RVM execution on DirectML,
    * at least one AMD *and* one Intel adapter are covered,
    * alpha drift vs CPU stays within tolerance (mean <= 0.005, max <= 0.02),
    * 720p sustains the configured target FPS (median),
    * the CUDA and DirectML wheels are not co-installed (either-or, mirroring
      the CPU/CUDA rule in packaging/npm/install.js).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 1
TASK = "WIN-6.2"

#: Vendors a "generic Windows GPU" claim must cover (lowercase substrings).
REQUIRED_VENDORS = ("amd", "intel")

#: Per-pixel alpha drift tolerated between DirectML and the CPU reference.
ALPHA_DELTA_MEAN_MAX = 0.005
ALPHA_DELTA_MAX_MAX = 0.02

#: Benchmark geometry: (label, width, height).
RESOLUTIONS = (("720p", 1280, 720), ("1080p", 1920, 1080))

EVIDENCE_KEYS = {
    "schema_version",
    "task",
    "generated_at",
    "host",
    "runtime",
    "wheel_conflict",
    "adapters",
}
ADAPTER_KEYS = {
    "vendor",
    "name",
    "driver",
    "device_id",
    "proven",
    "proof_error",
    "correctness",
    "performance",
    "target_fps",
}


# --------------------------------------------------------------------------- #
# check mode — deterministic, stdlib-only evaluation
# --------------------------------------------------------------------------- #
def evaluate_evidence(evidence: dict) -> tuple[bool, list[str]]:
    """Return (go, reasons); ``reasons`` is empty exactly when ``go``."""

    reasons: list[str] = []
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_KEYS:
        return False, ["evidence does not match the exact WIN-6.2 schema"]
    if evidence["schema_version"] != SCHEMA_VERSION or evidence["task"] != TASK:
        reasons.append("wrong schema_version/task for the WIN-6.2 gate")

    conflict = evidence["wheel_conflict"]
    if conflict.get("onnxruntime_gpu_installed") and conflict.get(
        "onnxruntime_directml_installed"
    ):
        reasons.append(
            "onnxruntime-gpu and onnxruntime-directml are co-installed; the "
            "installer profile must pick exactly one (spike question 4)"
        )
    if not conflict.get("onnxruntime_directml_installed"):
        reasons.append("evidence was not produced with onnxruntime-directml")

    adapters = evidence["adapters"]
    if not isinstance(adapters, list) or not adapters:
        return False, reasons + ["no adapters were tested"]

    covered: set[str] = set()
    for adapter in adapters:
        if set(adapter) != ADAPTER_KEYS:
            reasons.append("adapter entry does not match the exact schema")
            continue
        label = f"{adapter['vendor']} {adapter['name']}".strip()
        vendor = str(adapter["vendor"]).casefold()
        for required in REQUIRED_VENDORS:
            if required in vendor:
                covered.add(required)
        if not adapter["proven"]:
            reasons.append(
                f"{label}: DirectML did not prove real RVM execution "
                f"({adapter['proof_error'] or 'no error recorded'})"
            )
            continue
        correctness = adapter["correctness"]
        if correctness["alpha_delta_mean"] > ALPHA_DELTA_MEAN_MAX:
            reasons.append(
                f"{label}: mean alpha drift {correctness['alpha_delta_mean']:.4f} "
                f"exceeds {ALPHA_DELTA_MEAN_MAX}"
            )
        if correctness["alpha_delta_max"] > ALPHA_DELTA_MAX_MAX:
            reasons.append(
                f"{label}: max alpha drift {correctness['alpha_delta_max']:.4f} "
                f"exceeds {ALPHA_DELTA_MAX_MAX}"
            )
        hd = adapter["performance"]["720p"]
        if hd["fps"] < adapter["target_fps"]:
            reasons.append(
                f"{label}: 720p median {hd['fps']:.1f} fps is below the "
                f"{adapter['target_fps']} fps target"
            )

    for required in REQUIRED_VENDORS:
        if required not in covered:
            reasons.append(
                f"no {required.upper()} adapter in evidence; the generic-GPU "
                "claim requires both AMD and Intel coverage"
            )
    return not reasons, reasons


def check(path: Path, as_json: bool) -> int:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"windows-acceleration-gate: cannot read {path}: {exc}", file=sys.stderr)
        return 1
    go, reasons = evaluate_evidence(evidence)
    if as_json:
        print(json.dumps({"go": go, "reasons": reasons}, indent=2))
    elif go:
        print("WIN-6.2 gate: GO — DirectML evidence satisfies every criterion")
    else:
        print("WIN-6.2 gate: NO-GO", file=sys.stderr)
        for reason in reasons:
            print(f"  - {reason}", file=sys.stderr)
    return 0 if go else 1


# --------------------------------------------------------------------------- #
# run mode — Windows hardware harness
# --------------------------------------------------------------------------- #
def _detect_adapters() -> list[dict]:
    """Best-effort adapter identity via CIM; overridable with --adapter."""

    if sys.platform != "win32":
        return []
    query = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterCompatibility,DriverVersion | ConvertTo-Json"
    )
    try:
        raw = subprocess.run(
            ["powershell", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        parsed = json.loads(raw)
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    entries = parsed if isinstance(parsed, list) else [parsed]
    return [
        {
            "vendor": str(entry.get("AdapterCompatibility") or "unknown"),
            "name": str(entry.get("Name") or "unknown"),
            "driver": str(entry.get("DriverVersion") or "unknown"),
        }
        for entry in entries
    ]


def _installed(distribution: str) -> bool:
    from importlib import metadata

    try:
        metadata.version(distribution)
        return True
    except metadata.PackageNotFoundError:
        return False


def _alpha_output_index(session) -> int:
    for index, output in enumerate(session.get_outputs()):
        if output.name == "pha":
            return index
    raise SystemExit("model does not expose the RVM 'pha' alpha output")


def _measure(session, feeds, frames: int) -> list[float]:
    times_ms: list[float] = []
    for _ in range(frames):
        started = time.perf_counter()
        session.run(None, feeds)
        times_ms.append((time.perf_counter() - started) * 1000.0)
    return times_ms


def run(args: argparse.Namespace) -> int:
    if sys.platform != "win32":
        print(
            "windows-acceleration-gate: `run` needs Windows hardware; "
            "`check` works anywhere",
            file=sys.stderr,
        )
        return 1
    import numpy as np
    import onnxruntime as ort

    from custback.acceleration import (
        ProviderCandidate,
        prove_rvm_provider,
        synthetic_rvm_feeds,
    )

    model = Path(args.model).expanduser()
    if not model.is_file():
        print(f"windows-acceleration-gate: model not found: {model}", file=sys.stderr)
        return 1
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()

    detected = _detect_adapters()
    adapters: list[dict] = []
    for device_id in args.device_ids:
        identity = (
            detected[device_id]
            if device_id < len(detected)
            else {
                "vendor": "unknown",
                "name": f"adapter{device_id}",
                "driver": "unknown",
            }
        )
        if args.adapter_vendor:
            identity["vendor"] = args.adapter_vendor
        candidate = ProviderCandidate("DmlExecutionProvider", {"device_id": device_id})
        proof = prove_rvm_provider(ort, str(model), candidate)

        correctness = {"frames": 0, "alpha_delta_mean": 1.0, "alpha_delta_max": 1.0}
        performance = {
            label: {"median_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
            for label, _, _ in RESOLUTIONS
        }
        if proof.proven:
            dml = ort.InferenceSession(str(model), providers=[candidate.as_ort_arg()])
            cpu = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            alpha = _alpha_output_index(dml)
            deltas: list[float] = []
            maxima: list[float] = []
            for frame in range(args.frames):
                feeds = synthetic_rvm_feeds(288, 512)
                feeds["src"] = feeds["src"] + np.float32(
                    0.001 * frame
                )  # deterministic variation across the sequence
                delta = np.abs(
                    dml.run(None, feeds)[alpha].astype(np.float32)
                    - cpu.run(None, feeds)[alpha].astype(np.float32)
                )
                deltas.append(float(delta.mean()))
                maxima.append(float(delta.max()))
            correctness = {
                "frames": args.frames,
                "alpha_delta_mean": max(deltas),
                "alpha_delta_max": max(maxima),
            }
            for label, width, height in RESOLUTIONS:
                feeds = synthetic_rvm_feeds(height, width)
                _measure(dml, feeds, 3)  # warm-up excluded from statistics
                times = _measure(dml, feeds, args.frames)
                median = statistics.median(times)
                p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
                performance[label] = {
                    "median_ms": round(median, 2),
                    "p95_ms": round(p95, 2),
                    "fps": round(1000.0 / median, 1) if median else 0.0,
                }

        adapters.append(
            {
                **identity,
                "device_id": device_id,
                "proven": proof.proven,
                "proof_error": proof.error,
                "correctness": correctness,
                "performance": performance,
                "target_fps": args.target_fps,
            }
        )

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "host": {"os": platform.platform(), "machine": platform.machine()},
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "flavor": "directml",
            "model_file": model.name,
            "model_sha256": model_sha,
        },
        "wheel_conflict": {
            "onnxruntime_gpu_installed": _installed("onnxruntime-gpu"),
            "onnxruntime_directml_installed": _installed("onnxruntime-directml"),
        },
        "adapters": adapters,
    }
    out = Path(args.out)
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"evidence written to {out}")
    go, reasons = evaluate_evidence(evidence)
    print("verdict on this machine alone: " + ("GO" if go else "NO-GO"))
    for reason in reasons:
        print(f"  - {reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="windows-acceleration-gate")
    commands = parser.add_subparsers(dest="command", required=True)

    runner = commands.add_parser("run", help="produce evidence (Windows only)")
    runner.add_argument(
        "--model", required=True, help="path to the pinned rvm_mobilenetv3_fp32.onnx"
    )
    runner.add_argument("--out", default="windows-acceleration-evidence.json")
    runner.add_argument("--device-ids", type=int, nargs="+", default=[0])
    runner.add_argument("--frames", type=int, default=30)
    runner.add_argument("--target-fps", type=int, default=30)
    runner.add_argument(
        "--adapter-vendor", default="", help="override detected vendor (CI runners)"
    )

    checker = commands.add_parser("check", help="evaluate evidence (any OS)")
    checker.add_argument("evidence", type=Path)
    checker.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    return check(args.evidence, args.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
