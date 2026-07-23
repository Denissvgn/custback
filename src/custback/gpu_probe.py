"""Machine-readable proof that ONNX Runtime can execute a node on CUDA.

Provider registration alone is not sufficient: ONNX Runtime can advertise the
CUDA execution provider and then silently create a CPU-only session.  This
probe runs a tiny Add graph and verifies its provider in the ORT profile.
"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Audited ONNX ModelProto (IR 8, opset 13): z = Add(x, y), float32 shape [1].
# Keeping this 125-byte fixture in the package avoids adding the large ``onnx``
# authoring dependency merely to validate an installed execution provider.
_CUDA_PROBE_MODEL = base64.b64decode(
    "CAgSCGN1c3RiYWNrGgExOmgKHgoBeAoBeRIBehoOY3VkYV9wcm9iZV9hZGQiA0FkZBITY3VzdGJh"
    "Y2tfY3VkYV9wcm9iZVoPCgF4EgoKCAgBEgQKAggBWg8KAXkSCgoICAESBAoCCAFiDwoBehIKCggI"
    "ARIECgIIAUICEA0="
)


def _bounded_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:500]


def _profile_uses_cuda(events: Any) -> bool:
    if not isinstance(events, list):
        return False
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        args = event.get("args")
        name = event.get("name")
        if (
            isinstance(name, str)
            and "cuda_probe_add" in name
            and isinstance(args, dict)
            and args.get("op_name") == "Add"
            and args.get("provider") == "CUDAExecutionProvider"
        ):
            return True
    return False


def probe_cuda_inference(ort_module=None) -> dict[str, Any]:
    """Return JSON-safe CUDA provider and real-inference capability evidence."""

    result: dict[str, Any] = {
        "schema": 1,
        "onnxruntime": False,
        "cuda_provider": False,
        "cuda_inference": False,
        "active_providers": [],
        "output_verified": False,
        "profile_verified": False,
        "error": "",
    }
    try:
        if ort_module is None:
            import onnxruntime as ort_module  # pyright: ignore[reportMissingImports] - optional extra
    except Exception as exc:
        result["error"] = _bounded_error(exc)
        return result

    result["onnxruntime"] = True
    try:
        available = list(ort_module.get_available_providers())
    except Exception as exc:
        result["error"] = _bounded_error(exc)
        return result
    result["cuda_provider"] = "CUDAExecutionProvider" in available
    if not result["cuda_provider"]:
        result["error"] = "CUDAExecutionProvider is not registered"
        return result

    session = None
    profile_path = ""
    try:
        with tempfile.TemporaryDirectory(prefix="custback-cuda-probe-") as temporary:
            options = ort_module.SessionOptions()
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(temporary) / "profile")
            # Ensure node-assignment fallback is rejected where supported. We
            # still inspect the profile because EP initialization fallback can
            # happen above the session configuration layer.
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            if hasattr(ort_module, "GraphOptimizationLevel"):
                options.graph_optimization_level = (
                    ort_module.GraphOptimizationLevel.ORT_DISABLE_ALL
                )
            session = ort_module.InferenceSession(
                _CUDA_PROBE_MODEL,
                sess_options=options,
                providers=["CUDAExecutionProvider"],
            )
            result["active_providers"] = list(session.get_providers())
            output = session.run(
                None,
                {
                    "x": np.asarray([1.0], dtype=np.float32),
                    "y": np.asarray([2.0], dtype=np.float32),
                },
            )
            result["output_verified"] = bool(
                len(output) == 1
                and isinstance(output[0], np.ndarray)
                and output[0].shape == (1,)
                and np.allclose(output[0], np.asarray([3.0], dtype=np.float32))
            )
            profile_path = session.end_profiling()
            session = None
            with Path(profile_path).open(encoding="utf-8") as stream:
                result["profile_verified"] = _profile_uses_cuda(json.load(stream))
            result["cuda_inference"] = bool(
                result["output_verified"]
                and result["profile_verified"]
                and "CUDAExecutionProvider" in result["active_providers"]
            )
            if not result["cuda_inference"]:
                result["error"] = "probe did not execute on CUDAExecutionProvider"
    except Exception as exc:
        result["error"] = _bounded_error(exc)
    finally:
        if session is not None:
            try:
                session.end_profiling()
            except Exception:
                pass
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)
    result = probe_cuda_inference()
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    # A negative capability is valid probe output. Callers decide whether it
    # is required by the selected installation profile.
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by npm probes
    raise SystemExit(main())
