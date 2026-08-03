"""Fail-closed, cross-device RVM profile qualification.

The qualifier is deliberately an offline evidence joiner.  It does not open a
camera, model, execution provider, or output sink.  A MATTE-0.3 report limits
the candidates admitted to the matrix; private replay bundles provide the
pixel-derived quality and identity evidence; and a separately collected,
content-free run document supplies hardware, model-contract, startup, memory,
VRAM, and complete-service measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

from .config import CompositingConfig, SegmentationConfig
from .matte_ablation import (
    REPORT_SCHEMA as ABLATION_REPORT_SCHEMA,
    REPORT_VERSION as ABLATION_REPORT_VERSION,
)
from .matte_diagnostics import (
    MAX_MANIFEST_BYTES,
    MatteDiagnosticsError,
    MatteReplayBundle,
    _atomic_private_write,
    _controls,
    _json_bytes,
    _private_directory,
    _read_private_file,
)
from .matte_policy import MatteBackendKind, resolve_matte_policy
from .matte_quality import (
    EvaluationMetadata,
    MatteQualityAnnotations,
    MatteQualityError,
    _bundle_manifest_digest,
    _metric_path,
    _round,
    _summary,
    evaluate_bundle,
)
from .segmentation import RVM_MODEL

PLAN_SCHEMA = "custback.rvm-qualification-plan"
PLAN_VERSION = 1
RUN_SCHEMA = "custback.rvm-qualification-run"
RUN_VERSION = 1
REPORT_SCHEMA = "custback.rvm-qualification-report"
REPORT_VERSION = 1

MAX_CANDIDATES = 8
MAX_HARDWARE = 16
MAX_CANVASES = 8
MAX_CELLS = 4096
MAX_RUN_SAMPLES = 100_000

Cadence = Literal["native30", "decimated15"]
RenderMode = Literal["raw_model", "qualified_compositor"]
CellOutcome = Literal["qualified", "rejected", "not_decidable", "unavailable"]

_CADENCES: tuple[Cadence, ...] = ("native30", "decimated15")
_RENDER_MODES: tuple[RenderMode, ...] = (
    "raw_model",
    "qualified_compositor",
)
_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROVIDERS = frozenset({"cpu", "cuda", "directml"})
_ORT_PROVIDER_NAMES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
}
_EVIDENCE_KINDS = frozenset({"model-backed", "generated-proxy", "unavailable"})
_CADENCE_FPS_RANGES: dict[Cadence, tuple[float, float]] = {
    "native30": (29.0, 31.0),
    "decimated15": (14.5, 15.5),
}
_MIN_WARMUP_FRAMES = 30
_MIN_STEADY_FRAMES = 300
_MIN_OBSERVATION_S = 10.0
_MAX_RATIFIED_QUEUE_AGE_MS = 33.333334
_QUALITY_POLICY_ID = "matte-0.2-ratified-v1"
_REQUIRED_SEGMENT_KINDS = frozenset(
    {"stationary", "moving", "fast_motion", "occlusion"}
)

_ACCELERATION_KEYS = frozenset(
    {
        "applicable",
        "requested_mode",
        "requested_provider",
        "device_id",
        "state",
        "active_provider",
        "fallback_active",
        "fallback_count",
        "fallback_reason_code",
    }
)
_RVM_TELEMETRY_KEYS = frozenset(
    {
        "applicable",
        "input_frame_shape",
        "output_alpha_shape",
        "output_foreground_shape",
        "configured_downsample_mode",
        "configured_downsample_ratio",
        "resolved_downsample_ratio",
        "preprocess_ms",
        "session_run_ms",
        "postprocess_ms",
        "model_builtin",
        "model_identity",
        "model_sha256",
        "model_bytes",
        "acceleration_state",
        "acceleration_active_provider",
        "acceleration_fallback_active",
        "acceleration_fallback_count",
    }
)
_TIMING_KEYS = (
    "rvm_preprocess_ms",
    "rvm_session_run_ms",
    "rvm_postprocess_ms",
    "backend_inference_ms",
    "refinement_ms",
    "segmentation_ms",
    "background_ms",
    "color_correction_ms",
    "composite_ms",
    "frame_processing_ms",
    "output_send_ms",
    "frame_total_ms",
)
_OUTPUT_SINK_KEYS = frozenset({"applicable", "backend", "paces"})

_REQUIRED_GATE_PATHS: dict[str, frozenset[str]] = {
    "opaque_core": frozenset(
        {
            "aggregate.metrics.opaque_core_alpha_p05.p05",
            "aggregate.metrics.opaque_core_fraction_below_0_95.p95",
            "aggregate.metrics.foreground_hole_components.max",
        }
    ),
    "background": frozenset({"aggregate.metrics.background_alpha_mean.p95"}),
    "halo": frozenset(
        {
            "aggregate.metrics.exterior_halo_area_ratio.p95",
            "aggregate.metrics.exterior_halo_width_p95_px.p95",
        }
    ),
    "fine_detail": frozenset(
        {
            "aggregate.metrics.ground_truth_alpha_mse.p95",
            "aggregate.metrics.ground_truth_gradient_mae.p95",
            "aggregate.metrics.uncertain_pixel_fraction.p50",
        }
    ),
    "temporal": frozenset(
        {
            "aggregate.metrics.contour_displacement_p95_px.p95",
            "aggregate.metrics.compensated_alpha_temporal_abs_diff.p95",
            "aggregate.metrics.motion_trail_area_ratio.p95",
        }
    ),
}

_GATE_SPECS: dict[str, tuple[str, float, float | None]] = {
    "aggregate.metrics.opaque_core_alpha_p05.p05": (">=", 0.0, 1.0),
    "aggregate.metrics.opaque_core_fraction_below_0_95.p95": ("<=", 0.0, 1.0),
    "aggregate.metrics.foreground_hole_components.max": ("<=", 0.0, None),
    "aggregate.metrics.background_alpha_mean.p95": ("<=", 0.0, 1.0),
    "aggregate.metrics.exterior_halo_area_ratio.p95": ("<=", 0.0, 1.0),
    "aggregate.metrics.exterior_halo_width_p95_px.p95": ("<=", 0.0, None),
    "aggregate.metrics.ground_truth_alpha_mse.p95": ("<=", 0.0, 1.0),
    "aggregate.metrics.ground_truth_gradient_mae.p95": ("<=", 0.0, None),
    "aggregate.metrics.uncertain_pixel_fraction.p50": (">=", 0.0, 1.0),
    "aggregate.metrics.contour_displacement_p95_px.p95": ("<=", 0.0, None),
    "aggregate.metrics.compensated_alpha_temporal_abs_diff.p95": ("<=", 0.0, 1.0),
    "aggregate.metrics.motion_trail_area_ratio.p95": ("<=", 0.0, 1.0),
}

# MATTE-0.2 ratified the core absolute limits and required fixture/local bounds
# for the remaining independent families.  Version 1 fixes conservative
# qualification bounds for those families so a caller may strengthen, but
# cannot silently weaken, the meaning of a "qualified" result.
_RATIFIED_GATE_LIMITS: dict[str, float] = {
    "aggregate.metrics.opaque_core_alpha_p05.p05": 0.95,
    "aggregate.metrics.opaque_core_fraction_below_0_95.p95": 0.05,
    "aggregate.metrics.foreground_hole_components.max": 0.0,
    "aggregate.metrics.background_alpha_mean.p95": 0.01,
    "aggregate.metrics.exterior_halo_area_ratio.p95": 0.05,
    "aggregate.metrics.exterior_halo_width_p95_px.p95": 8.0,
    "aggregate.metrics.ground_truth_alpha_mse.p95": 0.01,
    "aggregate.metrics.ground_truth_gradient_mae.p95": 0.10,
    "aggregate.metrics.uncertain_pixel_fraction.p50": 0.05,
    "aggregate.metrics.contour_displacement_p95_px.p95": 1.5,
    "aggregate.metrics.compensated_alpha_temporal_abs_diff.p95": 0.10,
    "aggregate.metrics.motion_trail_area_ratio.p95": 0.10,
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_keys(
    value: object,
    expected: set[str] | frozenset[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise MatteQualityError(f"{name} does not match the strict version-1 schema")
    return cast(dict[str, Any], value)


def _safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise MatteQualityError(f"{name} must be a lowercase safe identifier")
    return value


def _model_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or _MODEL_ID.fullmatch(value) is None:
        raise MatteQualityError(f"{name} must be a bounded model filename/identity")
    return value


def _safe_label(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_LABEL.fullmatch(value) is None:
        raise MatteQualityError(f"{name} must be bounded and path-free")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MatteQualityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MatteQualityError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise MatteQualityError(f"{name} is below its minimum")
    if maximum is not None and result > maximum:
        raise MatteQualityError(f"{name} is above its maximum")
    return result


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or int(value) < (0 if allow_zero else 1):
        description = "non-negative" if allow_zero else "positive"
        raise MatteQualityError(f"{name} must be a {description} integer")
    return int(value)


def _read_json(path: Path | str, *, name: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_private_file(Path(path), max_bytes=MAX_MANIFEST_BYTES)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatteQualityError(f"{name} is malformed") from exc
    if not isinstance(value, dict):
        raise MatteQualityError(f"{name} must be an object")
    return value, payload


def _qualification_segmentation_digest(
    segmentation: Mapping[str, object],
) -> str:
    projected = dict(segmentation)
    projected.pop("model_path", None)
    return _sha256(_json_bytes(projected))


def _verified_ablation(path: Path | str) -> tuple[dict[str, Any], str, set[str], bool]:
    report, payload = _read_json(path, name="ablation report")
    if (
        report.get("schema") != ABLATION_REPORT_SCHEMA
        or report.get("version") != ABLATION_REPORT_VERSION
    ):
        raise MatteQualityError("unsupported ablation report")
    evidence = {
        name: report.get(name)
        for name in (
            "source",
            "scope",
            "coverage",
            "rows",
            "decisions",
            "cadence_conclusion",
            "required_decisions",
        )
    }
    if report.get("evidence_sha256") != _sha256(_json_bytes(evidence)):
        raise MatteQualityError("ablation evidence digest does not verify")
    decisions = report.get("decisions")
    rows = report.get("rows")
    if not isinstance(decisions, dict) or not isinstance(rows, list):
        raise MatteQualityError("ablation shortlist evidence is missing")
    shortlist = decisions.get("matte_2_5_rvm_candidates")
    if not isinstance(shortlist, list) or any(
        not isinstance(item, str) for item in shortlist
    ):
        raise MatteQualityError("ablation RVM shortlist is invalid")
    by_id = {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    admitted: set[str] = set()
    for candidate_id in shortlist:
        row = by_id.get(candidate_id)
        if (
            isinstance(row, dict)
            and row.get("lane") == "rvm"
            and row.get("status") == "completed"
            and row.get("evidence_kind") == "model-backed"
        ):
            admitted.add(candidate_id)
    scope = report.get("scope")
    model_backed = bool(
        isinstance(scope, dict)
        and scope.get("model_inference_executed") is True
        and admitted
    )
    return report, _sha256(payload), admitted, model_backed


def _verify_shortlist_binding(
    candidate: Mapping[str, object],
    ablation_row: object,
) -> None:
    if not isinstance(ablation_row, Mapping):
        raise MatteQualityError("shortlist candidate has no ablation row")
    configuration = ablation_row.get("configuration")
    if not isinstance(configuration, Mapping):
        raise MatteQualityError("shortlist ablation row lacks configuration evidence")
    segmentation = cast(Mapping[str, object], candidate["segmentation"])
    expected_digest = _qualification_segmentation_digest(segmentation)
    if configuration.get("qualification_segmentation_sha256") != expected_digest:
        raise MatteQualityError(
            "shortlist candidate segmentation is not bound to its ablation row"
        )
    if (
        configuration.get("qualification_model_identity") != candidate["model_id"]
        or configuration.get("qualification_model_sha256") != candidate["model_sha256"]
        or configuration.get("qualification_model_bytes") != candidate["model_bytes"]
    ):
        raise MatteQualityError(
            "shortlist candidate model is not bound to its ablation row"
        )


def _gate_policy(value: object) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    if not isinstance(value, list) or not value:
        raise MatteQualityError("qualification quality_gates must be a non-empty list")
    gates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    observed: dict[str, set[str]] = {name: set() for name in _REQUIRED_GATE_PATHS}
    for raw in value:
        gate = _strict_keys(
            raw,
            {"id", "family", "metric", "op", "value"},
            "qualification gate",
        )
        gate_id = _safe_id(gate["id"], "qualification gate id")
        if gate_id in seen_ids:
            raise MatteQualityError("qualification gate id is duplicated")
        seen_ids.add(gate_id)
        family = gate["family"]
        metric = gate["metric"]
        if family not in _REQUIRED_GATE_PATHS or not isinstance(metric, str):
            raise MatteQualityError("qualification gate family or metric is invalid")
        if metric not in _REQUIRED_GATE_PATHS[str(family)]:
            raise MatteQualityError(
                "qualification gate metric does not match its family"
            )
        if metric in observed[str(family)]:
            raise MatteQualityError("qualification gate metric is duplicated")
        expected_op, minimum, maximum = _GATE_SPECS[metric]
        if gate["op"] != expected_op:
            raise MatteQualityError(
                "qualification gate operation contradicts the metric direction"
            )
        threshold = _finite(
            gate["value"],
            "qualification gate threshold",
            minimum=minimum,
            maximum=maximum,
        )
        ratified_limit = _RATIFIED_GATE_LIMITS[metric]
        weaker = (
            threshold < ratified_limit
            if expected_op == ">="
            else threshold > ratified_limit
        )
        if weaker:
            raise MatteQualityError(
                "qualification gate is weaker than the ratified quality policy"
            )
        observed[str(family)].add(metric)
        gates.append(
            {
                "id": gate_id,
                "family": family,
                "metric": metric,
                "op": gate["op"],
                "value": threshold,
            }
        )
    missing = {
        family: sorted(required - observed[family])
        for family, required in _REQUIRED_GATE_PATHS.items()
        if required - observed[family]
    }
    if missing:
        raise MatteQualityError(f"qualification gates are incomplete: {missing}")
    return gates, observed


def load_plan(path: Path | str) -> tuple[dict[str, Any], str]:
    """Read and strictly normalize an owner-only qualification plan."""

    value, payload = _read_json(path, name="RVM qualification plan")
    plan = _strict_keys(
        value,
        {
            "schema",
            "version",
            "ablation_report",
            "provenance",
            "candidates",
            "hardware",
            "canvases",
            "qualified_compositor",
            "policy",
            "profile_proposals",
            "cells",
        },
        "RVM qualification plan",
    )
    if plan["schema"] != PLAN_SCHEMA or plan["version"] != PLAN_VERSION:
        raise MatteQualityError("unsupported RVM qualification plan")
    if not isinstance(plan["ablation_report"], str) or not plan["ablation_report"]:
        raise MatteQualityError("qualification plan ablation report path is missing")

    provenance = _strict_keys(
        plan["provenance"],
        {
            "qualification_id",
            "kind",
            "license_or_consent_reference",
            "contains_private_footage_in_repository",
        },
        "qualification provenance",
    )
    _safe_id(provenance["qualification_id"], "qualification id")
    if provenance["kind"] not in (
        "consented-local",
        "licensed-local",
        "generated",
    ):
        raise MatteQualityError("qualification provenance kind is invalid")
    if not isinstance(provenance["license_or_consent_reference"], str):
        raise MatteQualityError("qualification provenance reference is invalid")
    if type(provenance["contains_private_footage_in_repository"]) is not bool:
        raise MatteQualityError("qualification repository-footage flag is invalid")
    if (
        provenance["kind"] != "generated"
        and not provenance["license_or_consent_reference"].strip()
    ):
        raise MatteQualityError("model-backed qualification requires provenance")
    if provenance["contains_private_footage_in_repository"]:
        raise MatteQualityError(
            "private qualification footage cannot be repository data"
        )

    candidates = plan["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise MatteQualityError("qualification candidate count is invalid")
    candidate_ids: set[str] = set()
    normalized_candidates: list[dict[str, Any]] = []
    for raw in candidates:
        candidate = _strict_keys(
            raw,
            {
                "id",
                "role",
                "ablation_candidate_id",
                "model_id",
                "model_sha256",
                "model_bytes",
                "segmentation",
            },
            "qualification candidate",
        )
        candidate_id = _safe_id(candidate["id"], "candidate id")
        if candidate_id in candidate_ids:
            raise MatteQualityError("qualification candidate id is duplicated")
        candidate_ids.add(candidate_id)
        role = candidate["role"]
        if role not in ("shortlist", "auto", "compatibility_baseline"):
            raise MatteQualityError("qualification candidate role is invalid")
        ablation_id = candidate["ablation_candidate_id"]
        if role == "shortlist":
            _safe_id(ablation_id, "ablation candidate id")
        elif ablation_id != "":
            raise MatteQualityError("reserved candidates cannot claim an ablation id")
        model_id = _model_identity(candidate["model_id"], "model id")
        model_sha = _digest(candidate["model_sha256"], "candidate model digest")
        model_bytes = _positive_int(candidate["model_bytes"], "candidate model bytes")
        try:
            segmentation = SegmentationConfig.model_validate(
                candidate["segmentation"]
            ).model_dump(mode="json")
        except ValueError as exc:
            raise MatteQualityError("candidate segmentation policy is invalid") from exc
        if segmentation["backend"] not in ("auto", "rvm"):
            raise MatteQualityError("RVM qualification candidate must select RVM")
        if role == "auto" and float(segmentation["rvm_downsample"]) != 0.0:
            raise MatteQualityError("auto candidate must configure RVM downsample zero")
        normalized_candidates.append(
            {
                "id": candidate_id,
                "role": role,
                "ablation_candidate_id": ablation_id,
                "model_id": model_id,
                "model_sha256": model_sha,
                "model_bytes": model_bytes,
                "segmentation": segmentation,
                "segmentation_sha256": _qualification_segmentation_digest(segmentation),
            }
        )
    roles = [candidate["role"] for candidate in normalized_candidates]
    if roles.count("auto") != 1 or roles.count("compatibility_baseline") != 1:
        raise MatteQualityError(
            "qualification requires exactly one auto and one compatibility baseline"
        )
    if "shortlist" not in roles:
        raise MatteQualityError(
            "qualification requires at least one MATTE-0.3 shortlist candidate"
        )
    for candidate in normalized_candidates:
        if candidate["role"] in ("auto", "compatibility_baseline") and (
            candidate["model_id"] != RVM_MODEL.filename
            or candidate["model_sha256"] != RVM_MODEL.sha256
            or candidate["model_bytes"] != RVM_MODEL.size
        ):
            raise MatteQualityError(
                "reserved candidates must use the current built-in RVM model"
            )
    compatibility = SegmentationConfig().model_dump(mode="json")
    compatibility_candidate = next(
        candidate
        for candidate in normalized_candidates
        if candidate["role"] == "compatibility_baseline"
    )
    if compatibility_candidate["segmentation"] != compatibility:
        raise MatteQualityError(
            "compatibility baseline must match the current segmentation defaults"
        )

    canvases = plan["canvases"]
    if not isinstance(canvases, list) or not 1 <= len(canvases) <= MAX_CANVASES:
        raise MatteQualityError("qualification canvas count is invalid")
    canvas_ids: set[str] = set()
    normalized_canvases: list[dict[str, Any]] = []
    for raw in canvases:
        canvas = _strict_keys(raw, {"id", "width", "height"}, "qualification canvas")
        canvas_id = _safe_id(canvas["id"], "canvas id")
        if canvas_id in canvas_ids:
            raise MatteQualityError("qualification canvas id is duplicated")
        canvas_ids.add(canvas_id)
        normalized_canvases.append(
            {
                "id": canvas_id,
                "width": _positive_int(canvas["width"], "canvas width"),
                "height": _positive_int(canvas["height"], "canvas height"),
            }
        )

    hardware = plan["hardware"]
    if not isinstance(hardware, list) or not 1 <= len(hardware) <= MAX_HARDWARE:
        raise MatteQualityError("qualification hardware count is invalid")
    hardware_ids: set[str] = set()
    normalized_hardware: list[dict[str, Any]] = []
    for raw in hardware:
        item = _strict_keys(
            raw,
            {"id", "providers", "canvas_ids"},
            "qualification hardware",
        )
        hardware_id = _safe_id(item["id"], "hardware id")
        if hardware_id in hardware_ids:
            raise MatteQualityError("qualification hardware id is duplicated")
        hardware_ids.add(hardware_id)
        providers = item["providers"]
        assigned_canvases = item["canvas_ids"]
        if (
            not isinstance(providers, list)
            or not providers
            or any(provider not in _PROVIDERS for provider in providers)
            or len(set(providers)) != len(providers)
            or "cpu" not in providers
        ):
            raise MatteQualityError("hardware providers must include unique CPU")
        if (
            not isinstance(assigned_canvases, list)
            or not assigned_canvases
            or any(canvas not in canvas_ids for canvas in assigned_canvases)
            or len(set(assigned_canvases)) != len(assigned_canvases)
        ):
            raise MatteQualityError("hardware canvas coverage is invalid")
        normalized_hardware.append(
            {
                "id": hardware_id,
                "providers": list(providers),
                "canvas_ids": list(assigned_canvases),
            }
        )
    declared_providers = {
        provider for item in normalized_hardware for provider in item["providers"]
    }
    if declared_providers != _PROVIDERS:
        raise MatteQualityError(
            "qualification must declare CPU, CUDA, and DirectML coverage"
        )

    try:
        compositor = CompositingConfig.model_validate(
            plan["qualified_compositor"]
        ).model_dump(mode="json")
    except ValueError as exc:
        raise MatteQualityError("qualified compositor policy is invalid") from exc

    policy = _strict_keys(
        plan["policy"],
        {
            "quality_policy_id",
            "quality_gates",
            "native30_service_p95_ms",
            "native30_min_unique_fps",
            "decimated15_service_p95_ms",
            "decimated15_min_unique_fps",
            "min_warmup_frames",
            "min_steady_frames",
            "min_observation_s",
            "max_queue_age_ms",
            "max_queue_age_growth_ms",
            "minimum_profile_hardware_count",
        },
        "qualification policy",
    )
    if policy["quality_policy_id"] != _QUALITY_POLICY_ID:
        raise MatteQualityError("qualification quality policy id is unsupported")
    gates, _families = _gate_policy(policy["quality_gates"])
    normalized_policy = {
        "quality_policy_id": _QUALITY_POLICY_ID,
        "quality_gates": gates,
        "native30_service_p95_ms": _finite(
            policy["native30_service_p95_ms"],
            "native30 service budget",
            minimum=0.001,
        ),
        "native30_min_unique_fps": _finite(
            policy["native30_min_unique_fps"],
            "native30 unique FPS",
            minimum=0.001,
        ),
        "decimated15_service_p95_ms": _finite(
            policy["decimated15_service_p95_ms"],
            "decimated15 service budget",
            minimum=0.001,
        ),
        "decimated15_min_unique_fps": _finite(
            policy["decimated15_min_unique_fps"],
            "decimated15 unique FPS",
            minimum=0.001,
        ),
        "min_warmup_frames": _positive_int(
            policy["min_warmup_frames"], "minimum warm-up frames"
        ),
        "min_steady_frames": _positive_int(
            policy["min_steady_frames"], "minimum steady frames"
        ),
        "min_observation_s": _finite(
            policy["min_observation_s"], "minimum observation duration", minimum=0.001
        ),
        "max_queue_age_ms": _finite(
            policy["max_queue_age_ms"],
            "maximum queue age",
            minimum=0.0,
            maximum=_MAX_RATIFIED_QUEUE_AGE_MS,
        ),
        "max_queue_age_growth_ms": _finite(
            policy["max_queue_age_growth_ms"],
            "maximum queue-age growth",
            minimum=0.0,
        ),
        "minimum_profile_hardware_count": _positive_int(
            policy["minimum_profile_hardware_count"],
            "minimum profile hardware count",
        ),
    }
    if normalized_policy["minimum_profile_hardware_count"] < 2:
        raise MatteQualityError(
            "named profiles require at least two qualification hardware targets"
        )
    if (
        normalized_policy["native30_min_unique_fps"]
        < _CADENCE_FPS_RANGES["native30"][0]
        or normalized_policy["decimated15_min_unique_fps"]
        < _CADENCE_FPS_RANGES["decimated15"][0]
    ):
        raise MatteQualityError(
            "qualification unique-FPS minima are below the ratified cadence floors"
        )
    if (
        normalized_policy["min_warmup_frames"] < _MIN_WARMUP_FRAMES
        or normalized_policy["min_steady_frames"] < _MIN_STEADY_FRAMES
        or normalized_policy["min_observation_s"] < _MIN_OBSERVATION_S
    ):
        raise MatteQualityError(
            "qualification sample or duration minima are below the ratified floors"
        )

    proposals = plan["profile_proposals"]
    if not isinstance(proposals, list):
        raise MatteQualityError("profile proposals must be a list")
    proposal_names: set[str] = set()
    normalized_proposals: list[dict[str, str]] = []
    for raw in proposals:
        proposal = _strict_keys(raw, {"name", "candidate_id"}, "profile proposal")
        name = proposal["name"]
        if name not in ("performance", "balanced", "quality") or name in proposal_names:
            raise MatteQualityError("profile proposal name is invalid or duplicated")
        candidate_id = _safe_id(proposal["candidate_id"], "profile candidate id")
        if candidate_id not in candidate_ids:
            raise MatteQualityError("profile proposal references an unknown candidate")
        proposal_names.add(str(name))
        normalized_proposals.append({"name": str(name), "candidate_id": candidate_id})
    if proposals and proposal_names != {"performance", "balanced", "quality"}:
        raise MatteQualityError("profile proposals must define all three profile names")
    if len({item["candidate_id"] for item in normalized_proposals}) != len(
        normalized_proposals
    ):
        raise MatteQualityError("profile proposals require distinct candidate meanings")
    candidate_meanings = {
        candidate["id"]: (
            candidate["model_id"],
            candidate["model_sha256"],
            candidate["segmentation_sha256"],
        )
        for candidate in normalized_candidates
    }
    proposed_meanings = {
        candidate_meanings[item["candidate_id"]] for item in normalized_proposals
    }
    if normalized_proposals and len(proposed_meanings) != len(normalized_proposals):
        raise MatteQualityError(
            "profile proposals require distinct model/segmentation meanings"
        )

    cells = plan["cells"]
    if not isinstance(cells, list) or not 1 <= len(cells) <= MAX_CELLS:
        raise MatteQualityError("qualification cell count is invalid")
    cell_ids: set[str] = set()
    normalized_cells: list[dict[str, Any]] = []
    for raw in cells:
        cell = _strict_keys(
            raw,
            {
                "id",
                "candidate_id",
                "hardware_id",
                "provider",
                "canvas_id",
                "cadence",
                "render_mode",
                "status",
                "bundle",
                "annotations",
                "run",
                "availability_reason",
                "native30_cell_id",
                "paired_raw_cell_id",
            },
            "qualification cell",
        )
        cell_id = _safe_id(cell["id"], "cell id")
        if cell_id in cell_ids:
            raise MatteQualityError("qualification cell id is duplicated")
        cell_ids.add(cell_id)
        if cell["candidate_id"] not in candidate_ids:
            raise MatteQualityError("qualification cell candidate is unknown")
        if cell["hardware_id"] not in hardware_ids:
            raise MatteQualityError("qualification cell hardware is unknown")
        if cell["provider"] not in _PROVIDERS:
            raise MatteQualityError("qualification cell provider is invalid")
        if cell["canvas_id"] not in canvas_ids:
            raise MatteQualityError("qualification cell canvas is unknown")
        if cell["cadence"] not in _CADENCES or cell["render_mode"] not in _RENDER_MODES:
            raise MatteQualityError("qualification cell cadence/render mode is invalid")
        status = cell["status"]
        if status not in ("recorded", "unavailable"):
            raise MatteQualityError("qualification cell status is invalid")
        for name in ("bundle", "annotations", "run", "availability_reason"):
            if not isinstance(cell[name], str):
                raise MatteQualityError(f"qualification cell {name} is invalid")
        if status == "recorded":
            if not cell["bundle"] or not cell["annotations"] or not cell["run"]:
                raise MatteQualityError("recorded qualification cell paths are missing")
            if cell["availability_reason"]:
                raise MatteQualityError(
                    "recorded cell cannot have an availability reason"
                )
        elif (
            cell["bundle"]
            or cell["annotations"]
            or cell["run"]
            or not cell["availability_reason"].strip()
        ):
            raise MatteQualityError("unavailable cell must contain only a reason")
        if status == "unavailable":
            _safe_id(
                cell["availability_reason"],
                "availability reason code",
            )
        native_id = cell["native30_cell_id"]
        pair_id = cell["paired_raw_cell_id"]
        if not isinstance(native_id, str) or not isinstance(pair_id, str):
            raise MatteQualityError("qualification cell references are invalid")
        if cell["cadence"] == "native30" and native_id:
            raise MatteQualityError("native30 cell cannot name a native parent")
        if cell["cadence"] == "decimated15" and not native_id:
            raise MatteQualityError("decimated15 cell requires a native30 parent")
        if cell["render_mode"] == "raw_model" and pair_id:
            raise MatteQualityError("raw-model cell cannot name a raw pair")
        if cell["render_mode"] == "qualified_compositor" and not pair_id:
            raise MatteQualityError("qualified compositor cell requires a raw pair")
        normalized_cells.append(dict(cell))
    for cell in normalized_cells:
        for field in ("native30_cell_id", "paired_raw_cell_id"):
            reference = cell[field]
            if reference and reference not in cell_ids:
                raise MatteQualityError(f"qualification cell {field} is unknown")

    result = dict(plan)
    result["candidates"] = normalized_candidates
    result["hardware"] = normalized_hardware
    result["canvases"] = normalized_canvases
    result["qualified_compositor"] = compositor
    result["qualified_compositor_sha256"] = _sha256(_json_bytes(compositor))
    result["policy"] = normalized_policy
    result["profile_proposals"] = normalized_proposals
    result["cells"] = normalized_cells
    return result, _sha256(payload)


def load_run(path: Path | str) -> tuple[dict[str, Any], str]:
    """Read and strictly validate one content-free run sidecar."""

    value, payload = _read_json(path, name="RVM qualification run")
    run = _strict_keys(
        value,
        {
            "schema",
            "version",
            "bundle_manifest_sha256",
            "evidence_kind",
            "provenance",
            "source",
            "hardware",
            "model",
            "provider",
            "resource_evidence",
            "recurrence",
            "warmup_frame_count",
            "startup_ms",
            "memory_bytes",
            "vram_bytes",
            "service",
        },
        "RVM qualification run",
    )
    if run["schema"] != RUN_SCHEMA or run["version"] != RUN_VERSION:
        raise MatteQualityError("unsupported RVM qualification run")
    _digest(run["bundle_manifest_sha256"], "run bundle manifest digest")
    if run["evidence_kind"] not in _EVIDENCE_KINDS - {"unavailable"}:
        raise MatteQualityError("recorded run evidence kind is invalid")
    provenance = _strict_keys(
        run["provenance"],
        {"kind", "reference", "contains_private_pixels"},
        "run provenance",
    )
    if provenance["kind"] not in ("consented-local", "licensed-local", "generated"):
        raise MatteQualityError("run provenance kind is invalid")
    if (
        not isinstance(provenance["reference"], str)
        or type(provenance["contains_private_pixels"]) is not bool
    ):
        raise MatteQualityError("run provenance is invalid")
    if provenance["contains_private_pixels"]:
        raise MatteQualityError("content-free run evidence cannot contain pixels")
    if provenance["kind"] != "generated" and not provenance["reference"].strip():
        raise MatteQualityError("model-backed run provenance is missing")
    source = _strict_keys(
        run["source"],
        {"clip_sha256", "frame_sha256", "pixel_contract"},
        "run source identity",
    )
    _digest(source["clip_sha256"], "run source clip digest")
    frame_digests = source["frame_sha256"]
    if (
        not isinstance(frame_digests, list)
        or not frame_digests
        or len(frame_digests) > MAX_RUN_SAMPLES
    ):
        raise MatteQualityError("run source frame digests are invalid")
    for frame_digest in frame_digests:
        _digest(frame_digest, "run source frame digest")
    if source["pixel_contract"] != "canonical-pre-resize-rgb8":
        raise MatteQualityError("run source pixel contract is invalid")
    hardware = _strict_keys(
        run["hardware"],
        {
            "id",
            "label",
            "platform",
            "identity_sha256",
            "identity_source",
            "inventory",
        },
        "run hardware",
    )
    _safe_id(hardware["id"], "run hardware id")
    _digest(hardware["identity_sha256"], "run hardware identity")
    _safe_label(hardware["label"], "run hardware label")
    _safe_id(hardware["platform"], "run hardware platform")
    if hardware["identity_source"] != "qualification-hardware-inventory-v1":
        raise MatteQualityError("run hardware identity source is invalid")
    inventory = _strict_keys(
        hardware["inventory"],
        {
            "cpu_model",
            "accelerators",
            "memory_bytes",
            "os_name",
            "os_version",
            "architecture",
        },
        "run hardware inventory",
    )
    _safe_label(inventory["cpu_model"], "run hardware CPU model")
    accelerators = inventory["accelerators"]
    if (
        not isinstance(accelerators, list)
        or len(accelerators) > 8
        or any(
            not isinstance(item, str) or _SAFE_LABEL.fullmatch(item) is None
            for item in accelerators
        )
        or len(set(accelerators)) != len(accelerators)
    ):
        raise MatteQualityError("run hardware accelerator inventory is invalid")
    _positive_int(inventory["memory_bytes"], "run hardware memory bytes")
    for field in ("os_name", "os_version", "architecture"):
        _safe_label(inventory[field], f"run hardware {field}")
    if hardware["identity_sha256"] != _sha256(_json_bytes(inventory)):
        raise MatteQualityError("run hardware identity does not bind its inventory")
    model = _strict_keys(
        run["model"],
        {
            "id",
            "sha256",
            "bytes",
            "license",
            "license_reviewed",
            "packaging_supported",
            "download_integrity",
            "startup_succeeded",
        },
        "run model",
    )
    _model_identity(model["id"], "run model id")
    _digest(model["sha256"], "run model digest")
    _positive_int(model["bytes"], "run model bytes")
    if not isinstance(model["license"], str) or any(
        type(model[name]) is not bool
        for name in (
            "license_reviewed",
            "packaging_supported",
            "download_integrity",
            "startup_succeeded",
        )
    ):
        raise MatteQualityError("run model contract is invalid")
    provider = _strict_keys(
        run["provider"],
        {"requested", "execution_proven", "runtime"},
        "run provider",
    )
    if (
        provider["requested"] not in _PROVIDERS
        or type(provider["execution_proven"]) is not bool
    ):
        raise MatteQualityError("run provider contract is invalid")
    provider_runtime = _strict_keys(
        provider["runtime"],
        {
            "onnxruntime_version",
            "execution_provider",
            "provider_runtime_version",
            "driver_version",
        },
        "run provider runtime",
    )
    for field in (
        "onnxruntime_version",
        "execution_provider",
        "provider_runtime_version",
        "driver_version",
    ):
        _safe_label(provider_runtime[field], f"run provider runtime {field}")
    if (
        provider_runtime["execution_provider"]
        != _ORT_PROVIDER_NAMES[provider["requested"]]
    ):
        raise MatteQualityError("run provider runtime contradicts its request")
    resource_evidence = _strict_keys(
        run["resource_evidence"],
        {
            "sample_alignment",
            "rss_source",
            "rss_semantics",
            "sampling_point",
            "vram_applicable",
            "vram_source",
            "vram_semantics",
        },
        "run resource evidence",
    )
    if (
        resource_evidence["sample_alignment"] != "one-per-unique-frame"
        or resource_evidence["rss_source"] not in ("process-api", "external-sampler")
        or resource_evidence["rss_semantics"] != "process-current-resident-bytes"
        or resource_evidence["sampling_point"] != "post-sink-submit"
        or type(resource_evidence["vram_applicable"]) is not bool
        or resource_evidence["vram_source"]
        not in ("not-applicable", "provider-api", "external-sampler")
    ):
        raise MatteQualityError("run resource evidence contract is invalid")
    vram_applicable = resource_evidence["vram_applicable"]
    if (provider["requested"] == "cpu") == bool(vram_applicable):
        raise MatteQualityError("run VRAM applicability contradicts its provider")
    if (vram_applicable and resource_evidence["vram_source"] == "not-applicable") or (
        not vram_applicable and resource_evidence["vram_source"] != "not-applicable"
    ):
        raise MatteQualityError("run VRAM source contradicts applicability")
    expected_vram_semantics = (
        "process-current-allocated-bytes" if vram_applicable else "not-applicable"
    )
    if resource_evidence["vram_semantics"] != expected_vram_semantics:
        raise MatteQualityError("run VRAM semantics contradict applicability")
    recurrence = _strict_keys(
        run["recurrence"],
        {
            "input_selection",
            "model_invocation_count",
            "fresh_temporal_state",
            "output_projection_used",
        },
        "run recurrence evidence",
    )
    if recurrence["input_selection"] not in (
        "native-all",
        "every-other-native-preserved-timestamps",
    ):
        raise MatteQualityError("run recurrent input selection is invalid")
    _positive_int(
        recurrence["model_invocation_count"],
        "run model invocation count",
    )
    if (
        recurrence["fresh_temporal_state"] is not True
        or recurrence["output_projection_used"] is not False
    ):
        raise MatteQualityError("run recurrent execution proof is invalid")
    _positive_int(run["warmup_frame_count"], "run warm-up frame count", allow_zero=True)

    def samples(name: str, *, optional: bool = False) -> list[float] | None:
        raw = run[name]
        if optional and raw is None:
            return None
        if not isinstance(raw, list) or not raw or len(raw) > MAX_RUN_SAMPLES:
            raise MatteQualityError(f"run {name} samples are invalid")
        return [_finite(item, f"run {name} sample", minimum=0.0) for item in raw]

    def byte_samples(name: str, *, optional: bool = False) -> list[int] | None:
        raw = run[name]
        if optional and raw is None:
            return None
        if not isinstance(raw, list) or not raw or len(raw) > MAX_RUN_SAMPLES:
            raise MatteQualityError(f"run {name} samples are invalid")
        return [
            _positive_int(
                item,
                f"run {name} sample",
                allow_zero=True,
            )
            for item in raw
        ]

    startup = samples("startup_ms")
    memory = byte_samples("memory_bytes")
    vram = byte_samples("vram_bytes", optional=True)
    service = _strict_keys(
        run["service"],
        {
            "new_frame_service_ms",
            "queue_age_ms",
            "duration_s",
            "capture_fps",
            "unique_composite_fps",
            "output_fps",
            "output_repeat_count",
            "output_timeline_complete",
            "boundary",
            "bundle_frame_total_alignment",
        },
        "run service evidence",
    )
    service_samples = service["new_frame_service_ms"]
    queue = service["queue_age_ms"]
    if (
        not isinstance(service_samples, list)
        or not service_samples
        or len(service_samples) > MAX_RUN_SAMPLES
        or not isinstance(queue, list)
        or not queue
        or len(queue) > MAX_RUN_SAMPLES
    ):
        raise MatteQualityError("run service sample arrays are invalid")
    normalized = dict(run)
    normalized["startup_ms"] = startup
    normalized["memory_bytes"] = memory
    normalized["vram_bytes"] = vram
    if (vram is not None) != bool(vram_applicable):
        raise MatteQualityError("run VRAM samples contradict applicability")
    normalized["service"] = {
        "new_frame_service_ms": [
            _finite(item, "new-frame service sample", minimum=0.0)
            for item in service_samples
        ],
        "queue_age_ms": [
            _finite(item, "queue-age sample", minimum=0.0) for item in queue
        ],
        "duration_s": _finite(service["duration_s"], "service duration", minimum=0.0),
        "capture_fps": _finite(service["capture_fps"], "capture FPS", minimum=0.0),
        "unique_composite_fps": _finite(
            service["unique_composite_fps"], "unique composite FPS", minimum=0.0
        ),
        "output_fps": _finite(service["output_fps"], "output FPS", minimum=0.0),
        "output_repeat_count": _positive_int(
            service["output_repeat_count"],
            "output repeat count",
            allow_zero=True,
        ),
        "output_timeline_complete": service["output_timeline_complete"],
        "boundary": service["boundary"],
        "bundle_frame_total_alignment": service["bundle_frame_total_alignment"],
    }
    if type(normalized["service"]["output_timeline_complete"]) is not bool:
        raise MatteQualityError("output timeline completeness must be boolean")
    if (
        normalized["service"]["boundary"]
        != "unique-dequeue-through-sink-submit-excluding-deliberate-pacing"
    ):
        raise MatteQualityError("run service boundary is not qualification-safe")
    if normalized["service"]["bundle_frame_total_alignment"] not in (
        "exact-non-pacing-sink",
        "separate-pacing-boundary",
    ):
        raise MatteQualityError("bundle frame-total alignment is invalid")
    return normalized, _sha256(payload)


def _artifact_digest(
    artifacts: Mapping[str, object],
    name: str,
) -> str | None:
    descriptor = artifacts.get(name)
    if not isinstance(descriptor, Mapping):
        return None
    alias = descriptor.get("alias_of")
    if isinstance(alias, str):
        descriptor = cast(Mapping[str, object], artifacts.get(alias, {}))
    digest = descriptor.get("sha256")
    return str(digest) if isinstance(digest, str) else None


def _source_contract(bundle: MatteReplayBundle) -> list[dict[str, object]]:
    contract: list[dict[str, object]] = []
    for frame in bundle.frames:
        artifacts = cast(dict[str, Any], frame["artifacts"])
        contract.append(
            {
                "capture_sequence": frame["capture_sequence"],
                "capture_monotonic_ns": frame["capture_monotonic_ns"],
                "capture_generation": frame["capture_generation"],
                "geometry_generation": frame["geometry_generation"],
                "raw_frame_sha256": _artifact_digest(
                    artifacts,
                    "raw_frame",
                ),
            }
        )
    return contract


def _capture_lineage_contract(
    bundle: MatteReplayBundle,
) -> list[dict[str, object]]:
    """Return canvas-independent capture identity for cross-size comparison."""

    return [
        {
            "capture_sequence": frame["capture_sequence"],
            "capture_monotonic_ns": frame["capture_monotonic_ns"],
            "capture_generation": frame["capture_generation"],
            "geometry_generation": frame["geometry_generation"],
        }
        for frame in bundle.frames
    ]


def _annotation_descriptor_contract(
    descriptor: Mapping[str, object],
) -> dict[str, object]:
    """Strip local paths while retaining exact annotation-array identity."""

    return {
        "bytes": descriptor.get("bytes"),
        "sha256": descriptor.get("sha256"),
        "dtype": descriptor.get("dtype"),
        "shape": descriptor.get("shape"),
    }


def _annotation_contract(
    annotations: MatteQualityAnnotations,
) -> dict[str, object]:
    """Normalize every semantic annotation field independently of bundle path."""

    frames: list[dict[str, object]] = []
    for frame in annotations.frames:
        artifacts = cast(dict[str, Any], frame["artifacts"])
        normalized: dict[str, object] = {
            "sequence": frame["sequence"],
            "segment": frame["segment"],
            "registration_from_previous": frame["registration_from_previous"],
            "artifacts": {
                name: _annotation_descriptor_contract(descriptor)
                for name, descriptor in sorted(artifacts.items())
            },
        }
        regions = frame.get("regions", [])
        normalized["regions"] = [
            {
                "name": region["name"],
                "kind": region["kind"],
                "artifact": _annotation_descriptor_contract(region["artifact"]),
            }
            for region in cast(list[dict[str, Any]], regions)
        ]
        frames.append(normalized)
    return {
        "provenance": annotations.provenance,
        "segments": list(annotations.segments),
        "gates": list(annotations.gates),
        "frames": frames,
    }


def _annotation_projection_contract(
    annotations: MatteQualityAnnotations,
) -> dict[str, object]:
    """Return cadence-projectable annotation semantics without local IDs."""

    segment_kinds = {
        str(segment["id"]): str(segment["kind"]) for segment in annotations.segments
    }
    frames: list[dict[str, object]] = []
    for frame in annotations.frames:
        artifacts = cast(dict[str, Any], frame["artifacts"])
        frames.append(
            {
                "segment_kind": segment_kinds.get(str(frame["segment"])),
                "registration_from_previous": frame["registration_from_previous"],
                "artifacts": {
                    name: _annotation_descriptor_contract(descriptor)
                    for name, descriptor in sorted(artifacts.items())
                },
                "regions": [
                    {
                        "name": region["name"],
                        "kind": region["kind"],
                        "artifact": _annotation_descriptor_contract(region["artifact"]),
                    }
                    for region in cast(list[dict[str, Any]], frame.get("regions", []))
                ],
            }
        )
    return {
        "provenance": annotations.provenance,
        "gates": list(annotations.gates),
        "frames": frames,
    }


def _annotation_coverage_reasons(
    annotations: MatteQualityAnnotations,
) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    segment_kind_by_id = {
        str(segment["id"]): str(segment["kind"]) for segment in annotations.segments
    }
    observed_kinds = set(segment_kind_by_id.values())
    if observed_kinds != _REQUIRED_SEGMENT_KINDS:
        reasons.append(
            {
                "category": "evidence completeness",
                "detail": (
                    "annotation coverage must include stationary, moving, "
                    "fast-motion, and occlusion segments"
                ),
            }
        )
    required_artifacts = {
        "opaque_core",
        "background",
        "ground_truth_alpha",
        "ground_truth_foreground",
    }
    covered: dict[str, set[str]] = {kind: set() for kind in _REQUIRED_SEGMENT_KINDS}
    for frame in annotations.frames:
        segment_kind = segment_kind_by_id.get(str(frame["segment"]))
        if segment_kind in covered:
            covered[segment_kind].update(
                str(name) for name in cast(dict[str, Any], frame["artifacts"])
            )
    if any(required_artifacts - covered[kind] for kind in _REQUIRED_SEGMENT_KINDS):
        reasons.append(
            {
                "category": "evidence completeness",
                "detail": (
                    "every required segment lacks complete opaque, background, "
                    "and ground-truth coverage"
                ),
            }
        )
    return reasons


def _track_contract(
    bundle: MatteReplayBundle,
    names: Sequence[str],
) -> list[dict[str, str | None]]:
    result: list[dict[str, str | None]] = []
    for frame in bundle.frames:
        artifacts = cast(dict[str, Any], frame["artifacts"])
        entry: dict[str, str | None] = {}
        for name in names:
            entry[name] = _artifact_digest(artifacts, name)
        result.append(entry)
    return result


def _quality_cache_key(
    bundle: MatteReplayBundle,
    annotations: MatteQualityAnnotations,
) -> str:
    """Bind every input that can affect cached quality or cadence output."""

    return _sha256(
        _json_bytes(
            {
                "source": _source_contract(bundle),
                "output_timeline": bundle.manifest.get("output_timeline"),
                "tracks": _track_contract(
                    bundle,
                    (
                        "raw_frame",
                        "raw_mask",
                        "refined_mask",
                        "clean_foreground",
                        "backdrop_frame",
                        "base_composite",
                        "final_composite",
                    ),
                ),
                "annotations": _annotation_contract(annotations),
            }
        )
    )


def _expected_auto_ratio(width: int, height: int) -> float:
    return min(1.0, max(0.125, 512.0 / max(width, height)))


def _quality_gates(
    report: Mapping[str, object],
    gates: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, str]], bool]:
    results: list[dict[str, object]] = []
    reasons: list[dict[str, str]] = []
    decidable = True
    for gate in gates:
        actual = _metric_path(report, str(gate["metric"]))
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
        ):
            status = "not_evaluated"
            decidable = False
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": f"required {gate['family']} metric is unavailable",
                }
            )
            actual_value: object = actual
        else:
            actual_number = float(actual)
            threshold = float(cast(float, gate["value"]))
            passed = (
                actual_number <= threshold
                if gate["op"] == "<="
                else actual_number >= threshold
            )
            status = "pass" if passed else "fail"
            actual_value = _round(actual_number)
            if not passed:
                reasons.append(
                    {
                        "category": str(gate["family"]),
                        "detail": f"gate {gate['id']} failed",
                    }
                )
        results.append(
            {
                "id": gate["id"],
                "family": gate["family"],
                "metric": gate["metric"],
                "op": gate["op"],
                "actual": actual_value,
                "threshold": gate["value"],
                "status": status,
            }
        )
    return results, reasons, decidable


def _phase_timings(
    bundle: MatteReplayBundle,
    warmup_count: int,
) -> tuple[dict[str, object], list[dict[str, str]], bool]:
    warm: dict[str, list[float]] = {name: [] for name in _TIMING_KEYS}
    steady: dict[str, list[float]] = {name: [] for name in _TIMING_KEYS}
    missing: list[dict[str, str]] = []
    coherence_failed = False
    for sequence, frame in enumerate(bundle.frames):
        values = frame.get("timings_ms")
        target = warm if sequence < warmup_count else steady
        if not isinstance(values, Mapping):
            continue
        numeric: dict[str, float] = {}
        for name in _TIMING_KEYS:
            value = values.get(name)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
            ):
                numeric[name] = float(value)
                target[name].append(float(value))
        if len(numeric) == len(_TIMING_KEYS):
            checks = (
                (
                    numeric["rvm_preprocess_ms"]
                    + numeric["rvm_session_run_ms"]
                    + numeric["rvm_postprocess_ms"],
                    numeric["backend_inference_ms"],
                ),
                (
                    numeric["backend_inference_ms"] + numeric["refinement_ms"],
                    numeric["segmentation_ms"],
                ),
                (
                    numeric["segmentation_ms"]
                    + numeric["background_ms"]
                    + numeric["color_correction_ms"]
                    + numeric["composite_ms"],
                    numeric["frame_processing_ms"],
                ),
                (
                    numeric["frame_processing_ms"] + numeric["output_send_ms"],
                    numeric["frame_total_ms"],
                ),
            )
            if any(inner > outer + 1e-6 for inner, outer in checks):
                coherence_failed = True
    expected_warm = min(warmup_count, len(bundle.frames))
    expected_steady = max(0, len(bundle.frames) - warmup_count)
    for phase_name, values, expected_count in (
        ("warm_up", warm, expected_warm),
        ("steady_state", steady, expected_steady),
    ):
        for timing_name, series in values.items():
            if len(series) != expected_count:
                missing.append(
                    {
                        "category": "evidence completeness",
                        "detail": (
                            f"{phase_name} {timing_name} samples do not cover "
                            "every frame"
                        ),
                    }
                )
    if coherence_failed:
        missing.append(
            {
                "category": "evidence completeness",
                "detail": "per-frame timing components exceed an enclosing boundary",
            }
        )
    return (
        {
            "warm_up_ms": {
                name: _summary(values) for name, values in sorted(warm.items())
            },
            "steady_state_ms": {
                name: _summary(values) for name, values in sorted(steady.items())
            },
            "warm_up_frame_count": warmup_count,
            "steady_state_frame_count": max(0, len(bundle.frames) - warmup_count),
        },
        missing,
        not missing,
    )


def _serialized_service_reasons(
    bundle: MatteReplayBundle,
    queue_age_ms: Sequence[float],
) -> list[dict[str, str]]:
    """Bind non-pacing full-frame service to the monotonic send timeline."""

    if len(queue_age_ms) != len(bundle.frames):
        return [
            {
                "category": "evidence completeness",
                "detail": "queue-age samples do not cover the serialized timeline",
            }
        ]
    by_source: dict[int, list[dict[str, Any]]] = {}
    for event in bundle.output_events:
        if event["base_updated"] is True:
            by_source.setdefault(int(event["source_bundle_sequence"]), []).append(event)
    if any(
        len(by_source.get(sequence, [])) != 1 for sequence in range(len(bundle.frames))
    ):
        return [
            {
                "category": "evidence completeness",
                "detail": "output timeline lacks one unique base-update send per frame",
            }
        ]

    previous_send_ns: int | None = None
    for sequence, frame in enumerate(bundle.frames):
        timings = frame.get("timings_ms")
        frame_total = (
            timings.get("frame_total_ms") if isinstance(timings, Mapping) else None
        )
        if (
            isinstance(frame_total, bool)
            or not isinstance(frame_total, (int, float))
            or not math.isfinite(float(frame_total))
            or float(frame_total) < 0.0
        ):
            return [
                {
                    "category": "evidence completeness",
                    "detail": "serialized service lacks a finite frame-total sample",
                }
            ]
        event = by_source[sequence][0]
        if event["exact_final_repeat"] is not False:
            return [
                {
                    "category": "evidence completeness",
                    "detail": "a unique base update is mislabeled as an exact repeat",
                }
            ]
        sent_ns = int(event["sent_monotonic_ns"])
        capture_ns = int(frame["capture_monotonic_ns"])
        service_ns = int(round(float(frame_total) * 1_000_000.0))
        started_ns = sent_ns - service_ns
        measured_queue_ms = (started_ns - capture_ns) / 1_000_000.0
        if (
            sent_ns < capture_ns
            or started_ns < capture_ns
            or (previous_send_ns is not None and started_ns < previous_send_ns)
            or not math.isclose(
                float(queue_age_ms[sequence]),
                measured_queue_ms,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            return [
                {
                    "category": "evidence completeness",
                    "detail": (
                        "frame service or queue age contradicts the serialized "
                        "send timeline"
                    ),
                }
            ]
        previous_send_ns = sent_ns
    return []


def _bundle_resources(
    bundle: MatteReplayBundle,
    provider: str,
) -> tuple[dict[str, object], list[dict[str, str]], list[int], list[int] | None]:
    reasons: list[dict[str, str]] = []
    rss: list[int] = []
    vram: list[int] = []
    for frame in bundle.frames:
        resources = frame.get("resource_samples")
        rss_value = (
            resources.get("rss_bytes") if isinstance(resources, Mapping) else None
        )
        if rss_value is None:
            pass
        elif type(rss_value) is not int or int(rss_value) < 0:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "recorded RSS sample is malformed",
                }
            )
        else:
            rss.append(int(rss_value))
        vram_value = (
            resources.get("vram_bytes") if isinstance(resources, Mapping) else None
        )
        if provider == "cpu":
            if vram_value is not None:
                reasons.append(
                    {
                        "category": "evidence completeness",
                        "detail": "CPU VRAM must be explicitly non-applicable",
                    }
                )
        elif vram_value is None:
            pass
        elif type(vram_value) is not int or int(vram_value) < 0:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "recorded accelerator VRAM sample is malformed",
                }
            )
        else:
            vram.append(int(vram_value))
    expected = len(bundle.frames)
    if rss and len(rss) != expected:
        reasons.append(
            {
                "category": "evidence completeness",
                "detail": "optional bundle RSS is not all-frame aligned",
            }
        )
        rss = []
    normalized_vram: list[int] | None
    if provider == "cpu":
        normalized_vram = None
    elif not vram:
        normalized_vram = None
    elif len(vram) == expected:
        normalized_vram = vram
    else:
        reasons.append(
            {
                "category": "evidence completeness",
                "detail": "optional bundle VRAM is not all-frame aligned",
            }
        )
        normalized_vram = None
    return (
        {
            "bundle_rss_bytes": _summary(rss),
            "bundle_vram_bytes": _summary(normalized_vram or []),
        },
        reasons,
        rss,
        normalized_vram,
    )


def _stable_evidence(
    bundle: MatteReplayBundle,
    *,
    candidate: Mapping[str, object],
    provider: str,
    width: int,
    height: int,
    render_mode: RenderMode,
    expected_compositing: Mapping[str, object],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    reasons: list[dict[str, str]] = []
    candidate_segmentation = SegmentationConfig.model_validate(
        candidate["segmentation"]
    )
    expected_compositor = CompositingConfig.model_validate(expected_compositing)
    ratios: list[float] = []
    acceleration_contract: bytes | None = None
    active_provider: str | None = None
    device_id: int | None = None
    telemetry_model: tuple[object, ...] | None = None
    output_sink_contract: bytes | None = None
    output_sink_backend: str | None = None
    background_contract: bytes | None = None
    background_mode: str | None = None
    for frame in bundle.frames:
        artifacts = frame.get("artifacts")
        clean_foreground = (
            _artifact_digest(artifacts, "clean_foreground")
            if isinstance(artifacts, Mapping)
            else None
        )
        if clean_foreground is None:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "RVM clean foreground artifact is missing",
                }
            )
        configured_segmentation, configured_compositing = _controls(
            cast(dict[str, Any], frame)
        )
        try:
            segmentation = SegmentationConfig.model_validate(
                configured_segmentation
            ).model_dump(mode="json")
            compositing = CompositingConfig.model_validate(
                configured_compositing
            ).model_dump(mode="json")
        except ValueError:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "recorded configured controls are invalid",
                }
            )
            continue
        if (
            _qualification_segmentation_digest(segmentation)
            != candidate["segmentation_sha256"]
        ):
            reasons.append(
                {
                    "category": "configuration",
                    "detail": "recorded segmentation policy differs from candidate",
                }
            )
        if _json_bytes(compositing) != _json_bytes(expected_compositing):
            reasons.append(
                {
                    "category": "configuration",
                    "detail": f"{render_mode} compositor policy is not exact",
                }
            )
        configured_controls = frame.get("configured_controls")
        background = (
            configured_controls.get("background")
            if isinstance(configured_controls, Mapping)
            else None
        )
        if (
            not isinstance(background, Mapping)
            or set(background) != {"mode", "fit_mode", "anchor_x", "anchor_y"}
            or background.get("mode")
            not in {"blur", "image", "video", "color", "camera"}
        ):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "configured local background identity is missing",
                }
            )
            frame_background_mode = "color"
        else:
            encoded_background = _json_bytes(dict(background))
            frame_background_mode = str(background["mode"])
            if background_contract is None:
                background_contract = encoded_background
                background_mode = frame_background_mode
            elif background_contract != encoded_background:
                reasons.append(
                    {
                        "category": "configuration",
                        "detail": "configured background policy drifted during the run",
                    }
                )
        effective = frame.get("effective_controls")
        if not isinstance(effective, Mapping):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "effective controls are missing",
                }
            )
            continue
        output_sink = effective.get("output_sink")
        if (
            not isinstance(output_sink, Mapping)
            or set(output_sink) != _OUTPUT_SINK_KEYS
            or output_sink.get("applicable") is not True
            or output_sink.get("backend")
            not in ("null", "pyvirtualcam", "native", "unknown")
            or type(output_sink.get("paces")) is not bool
            or output_sink.get("paces") is not False
        ):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "non-pacing output sink proof is missing or invalid",
                }
            )
        else:
            encoded_sink = _json_bytes(dict(output_sink))
            if output_sink_contract is None:
                output_sink_contract = encoded_sink
                output_sink_backend = str(output_sink["backend"])
            elif output_sink_contract != encoded_sink:
                reasons.append(
                    {
                        "category": "evidence completeness",
                        "detail": "output sink identity drifted during the run",
                    }
                )
        if effective.get("segmentation_backend") != "RVMSegmenter":
            reasons.append(
                {"category": "backend", "detail": "effective backend is not RVM"}
            )
        ratio = effective.get("rvm_downsample_ratio")
        resolved_ratio: float | None = None
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or not 0.0 < float(ratio) <= 1.0
        ):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "effective RVM ratio is unavailable",
                }
            )
        else:
            resolved_ratio = float(ratio)
            ratios.append(resolved_ratio)
        expected_policy = resolve_matte_policy(
            candidate_segmentation,
            expected_compositor,
            MatteBackendKind.TRUE_ALPHA_RECURRENT,
            resolved_rvm_ratio=resolved_ratio,
            passthrough=False,
            canvas_shape=(height, width),
            light_wrap_stabilization_eligible=frame_background_mode
            in {"video", "camera"},
        )
        expected_refiner = expected_policy.effective_refiner_config(
            candidate_segmentation
        ).model_dump(mode="json")
        recorded_refiner = effective.get("refiner")
        refiner_matches = isinstance(recorded_refiner, Mapping)
        if refiner_matches:
            projected_refiner = dict(cast(Mapping[str, object], recorded_refiner))
            projected_refiner.pop("model_path", None)
            expected_refiner.pop("model_path", None)
            refiner_matches = projected_refiner == expected_refiner
        expected_flat = {
            "produces_matte": True,
            "edge_refinement_mode": expected_policy.effective.edge_refinement_mode,
            "edge_refinement_radius_px": (
                expected_policy.effective.edge_refinement_radius_px
            ),
            "mask_shift": expected_policy.effective.mask_shift,
            "use_model_foreground": expected_policy.effective.use_model_foreground,
            "light_wrap": expected_policy.effective.light_wrap,
            "blend_space": expected_compositor.blend_space,
        }
        if (
            any(effective.get(name) != value for name, value in expected_flat.items())
            or not refiner_matches
            or effective.get("matte_policy") != expected_policy.to_dict()
        ):
            reasons.append(
                {
                    "category": "attribution",
                    "detail": (
                        f"{render_mode} effective matte/compositor policy "
                        "does not match the qualified policy"
                    ),
                }
            )
        acceleration = effective.get("acceleration")
        if (
            not isinstance(acceleration, Mapping)
            or set(acceleration) != _ACCELERATION_KEYS
        ):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "effective acceleration snapshot is missing or malformed",
                }
            )
        else:
            if active_provider is None:
                active_provider = str(acceleration.get("active_provider", ""))
            if device_id is None and type(acceleration.get("device_id")) is int:
                device_id = int(cast(int, acceleration["device_id"]))
            encoded = _json_bytes(dict(acceleration))
            if acceleration_contract is None:
                acceleration_contract = encoded
            elif acceleration_contract != encoded:
                reasons.append(
                    {
                        "category": "provider fallback",
                        "detail": "acceleration state drifted during the run",
                    }
                )
            if (
                acceleration.get("applicable") is not True
                or acceleration.get("active_provider") != provider
                or acceleration.get("fallback_active") is not False
                or type(acceleration.get("fallback_count")) is not int
                or acceleration.get("fallback_count") != 0
                or acceleration.get("fallback_reason_code") != ""
                or type(acceleration.get("device_id")) is not int
                or int(cast(int, acceleration.get("device_id"))) < 0
                or int(cast(int, acceleration.get("device_id"))) > 64
            ):
                reasons.append(
                    {
                        "category": "provider fallback",
                        "detail": "requested provider was not stably active",
                    }
                )
            expected_mode = "cpu" if provider == "cpu" else "gpu_required"
            expected_requested_provider = "auto" if provider == "cpu" else provider
            expected_state = "cpu_fallback" if provider == "cpu" else "gpu_active"
            if (
                acceleration.get("requested_mode") != expected_mode
                or acceleration.get("requested_provider") != expected_requested_provider
                or acceleration.get("state") != expected_state
            ):
                reasons.append(
                    {
                        "category": "provider fallback",
                        "detail": "provider request policy does not match the cell",
                    }
                )
        telemetry = effective.get("rvm_telemetry")
        if not isinstance(telemetry, Mapping) or set(telemetry) != _RVM_TELEMETRY_KEYS:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "RVM telemetry snapshot is missing or malformed",
                }
            )
        else:
            shape = [height, width]
            if (
                telemetry.get("applicable") is not True
                or telemetry.get("input_frame_shape") != shape
                or telemetry.get("output_alpha_shape") != shape
                or telemetry.get("output_foreground_shape") != [height, width, 3]
            ):
                reasons.append(
                    {
                        "category": "evidence completeness",
                        "detail": "RVM telemetry shapes do not match the canvas",
                    }
                )
            configured_ratio = float(
                cast(dict[str, Any], candidate["segmentation"])["rvm_downsample"]
            )
            configured_mode = "auto" if configured_ratio == 0.0 else "explicit"
            telemetry_configured_ratio = telemetry.get("configured_downsample_ratio")
            if (
                telemetry.get("configured_downsample_mode") != configured_mode
                or isinstance(telemetry_configured_ratio, bool)
                or not isinstance(telemetry_configured_ratio, (int, float))
                or not math.isfinite(float(telemetry_configured_ratio))
                or float(telemetry_configured_ratio) != configured_ratio
            ):
                reasons.append(
                    {
                        "category": "configuration",
                        "detail": "RVM telemetry configured detail is contradictory",
                    }
                )
            frame_timings = frame.get("timings_ms")
            for telemetry_key, timing_key in (
                ("preprocess_ms", "rvm_preprocess_ms"),
                ("session_run_ms", "rvm_session_run_ms"),
                ("postprocess_ms", "rvm_postprocess_ms"),
            ):
                telemetry_timing = telemetry.get(telemetry_key)
                recorded_timing = (
                    frame_timings.get(timing_key)
                    if isinstance(frame_timings, Mapping)
                    else None
                )
                if (
                    isinstance(telemetry_timing, bool)
                    or not isinstance(telemetry_timing, (int, float))
                    or not math.isfinite(float(telemetry_timing))
                    or float(telemetry_timing) < 0.0
                    or isinstance(recorded_timing, bool)
                    or not isinstance(recorded_timing, (int, float))
                    or not math.isfinite(float(recorded_timing))
                    or float(recorded_timing) < 0.0
                    or float(telemetry_timing) != float(recorded_timing)
                ):
                    reasons.append(
                        {
                            "category": "evidence completeness",
                            "detail": (
                                f"RVM telemetry {telemetry_key} does not match "
                                f"{timing_key}"
                            ),
                        }
                    )
            identity = (
                telemetry.get("model_identity"),
                telemetry.get("model_sha256"),
                telemetry.get("model_bytes"),
                telemetry.get("model_builtin"),
            )
            if telemetry_model is None:
                telemetry_model = identity
            elif telemetry_model != identity:
                reasons.append(
                    {
                        "category": "model contract",
                        "detail": "RVM model identity drifted during the run",
                    }
                )
            telemetry_resolved_ratio = telemetry.get("resolved_downsample_ratio")
            expected_builtin = (
                candidate["model_id"] == RVM_MODEL.filename
                and candidate["model_sha256"] == RVM_MODEL.sha256
                and candidate["model_bytes"] == RVM_MODEL.size
            )
            if (
                telemetry.get("model_identity") != candidate["model_id"]
                or telemetry.get("model_sha256") != candidate["model_sha256"]
                or telemetry.get("model_bytes") != candidate["model_bytes"]
                or telemetry.get("model_builtin") is not expected_builtin
                or isinstance(telemetry_resolved_ratio, bool)
                or not isinstance(telemetry_resolved_ratio, (int, float))
                or not math.isfinite(float(telemetry_resolved_ratio))
                or telemetry_resolved_ratio != ratio
                or not isinstance(acceleration, Mapping)
                or telemetry.get("acceleration_state") != acceleration.get("state")
                or telemetry.get("acceleration_active_provider") != provider
                or telemetry.get("acceleration_fallback_active") is not False
                or type(telemetry.get("acceleration_fallback_count")) is not int
                or telemetry.get("acceleration_fallback_count") != 0
            ):
                reasons.append(
                    {
                        "category": "model contract",
                        "detail": "RVM telemetry contradicts the planned run",
                    }
                )
    unique_ratios = {round(value, 8) for value in ratios}
    if len(unique_ratios) > 1:
        reasons.append(
            {
                "category": "configuration",
                "detail": "effective RVM ratio drifted across frames",
            }
        )
    effective_ratio = ratios[0] if ratios else None
    configured = cast(dict[str, Any], candidate["segmentation"])["rvm_downsample"]
    expected = (
        _expected_auto_ratio(width, height)
        if float(configured) == 0.0
        else float(configured)
    )
    if effective_ratio is None or not math.isclose(
        effective_ratio, expected, rel_tol=0.0, abs_tol=1e-6
    ):
        reasons.append(
            {
                "category": "configuration",
                "detail": "resolved RVM ratio does not match configured/auto policy",
            }
        )
    return reasons, {
        "configured_rvm_downsample": configured,
        "effective_rvm_downsample_ratio": (
            _round(effective_ratio) if effective_ratio is not None else None
        ),
        "effective_internal_long_edge_px": (
            _round(max(width, height) * effective_ratio)
            if effective_ratio is not None
            else None
        ),
        "auto_formula": "min(1,max(0.125,512/max(width,height)))",
        "active_provider": active_provider,
        "device_id": device_id,
        "output_sink_backend": output_sink_backend,
        "background_mode": background_mode,
    }


@dataclass
class _LoadedCell:
    plan: dict[str, Any]
    result: dict[str, Any]
    bundle: MatteReplayBundle | None = None
    annotations: MatteQualityAnnotations | None = None
    run: dict[str, Any] | None = None
    source_contract: list[dict[str, object]] | None = None
    capture_lineage_contract: list[dict[str, object]] | None = None
    annotation_contract: dict[str, object] | None = None
    annotation_projection_contract: dict[str, object] | None = None


def _profile_semantic_reasons(
    proposals: Sequence[Mapping[str, str]],
    loaded: Mapping[str, _LoadedCell],
) -> list[str]:
    """Prove non-weighted performance/balanced/quality meaning in every scope."""

    if not proposals:
        return []
    candidate_by_name = {
        str(proposal["name"]): str(proposal["candidate_id"]) for proposal in proposals
    }
    if set(candidate_by_name) != {"performance", "balanced", "quality"}:
        return ["profile semantic roles are incomplete"]
    proposed = set(candidate_by_name.values())
    by_scope: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for item in loaded.values():
        candidate_id = str(item.plan["candidate_id"])
        if candidate_id not in proposed or item.bundle is None:
            continue
        key = _cell_key(item.plan)
        scope = (key[1], key[2], key[3], key[4], key[5])
        by_scope.setdefault(scope, {})[candidate_id] = item.result

    latency_failed = False
    quality_failed = False
    for rows in by_scope.values():
        if set(rows) != proposed or any(
            row.get("outcome") != "qualified" for row in rows.values()
        ):
            continue
        ordered = [
            rows[candidate_by_name[name]]
            for name in ("performance", "balanced", "quality")
        ]
        latency = [
            cast(
                dict[str, Any],
                cast(dict[str, Any], row["performance"])[
                    "new_frame_service_steady_state_ms"
                ],
            ).get("p95")
            for row in ordered
        ]
        if (
            any(not isinstance(value, (int, float)) for value in latency)
            or not float(cast(float, latency[0]))
            <= float(cast(float, latency[1]))
            <= float(cast(float, latency[2]))
            or not float(cast(float, latency[0])) < float(cast(float, latency[2]))
        ):
            latency_failed = True

        gate_maps = [
            {
                str(gate["metric"]): float(gate["actual"])
                for gate in cast(list[dict[str, Any]], row["quality_gates"])
                if isinstance(gate.get("actual"), (int, float))
            }
            for row in ordered
        ]
        strictly_better = False
        for metric, (operation, _minimum, _maximum) in _GATE_SPECS.items():
            if any(metric not in gate_map for gate_map in gate_maps):
                quality_failed = True
                continue
            performance_value, balanced_value, quality_value = (
                gate_map[metric] for gate_map in gate_maps
            )
            if operation == "<=":
                ordered_quality = performance_value >= balanced_value >= quality_value
                strictly_better |= quality_value < performance_value
            else:
                ordered_quality = performance_value <= balanced_value <= quality_value
                strictly_better |= quality_value > performance_value
            if not ordered_quality:
                quality_failed = True
        if not strictly_better:
            quality_failed = True

    expected_scope_count = len(
        {
            (
                _cell_key(item.plan)[1],
                _cell_key(item.plan)[2],
                _cell_key(item.plan)[3],
                _cell_key(item.plan)[4],
                _cell_key(item.plan)[5],
            )
            for item in loaded.values()
            if str(item.plan["candidate_id"]) in proposed
        }
    )
    if len(by_scope) != expected_scope_count or not by_scope:
        return ["profile semantic comparison scope is incomplete"]
    reasons: list[str] = []
    if latency_failed:
        reasons.append("profile latency ordering is not stable across every scope")
    if quality_failed:
        reasons.append("profile quality ordering is not stable across every scope")
    return reasons


def _cell_key(cell: Mapping[str, object]) -> tuple[str, str, str, str, str, str]:
    return (
        str(cell["candidate_id"]),
        str(cell["hardware_id"]),
        str(cell["provider"]),
        str(cell["canvas_id"]),
        str(cell["cadence"]),
        str(cell["render_mode"]),
    )


def _expected_keys(
    plan: Mapping[str, object],
) -> set[tuple[str, str, str, str, str, str]]:
    candidates = cast(list[dict[str, Any]], plan["candidates"])
    hardware = cast(list[dict[str, Any]], plan["hardware"])
    return {
        (candidate["id"], machine["id"], provider, canvas, cadence, render)
        for candidate in candidates
        for machine in hardware
        for provider in machine["providers"]
        for canvas in machine["canvas_ids"]
        for cadence in _CADENCES
        for render in _RENDER_MODES
    }


def _reasons_outcome(
    reasons: Sequence[Mapping[str, str]], *, proxy: bool
) -> CellOutcome:
    if proxy:
        return "not_decidable"
    if any(reason["category"] == "evidence completeness" for reason in reasons):
        return "not_decidable"
    return "rejected" if reasons else "qualified"


def qualify_plan(plan_path: Path | str) -> dict[str, Any]:
    """Validate and aggregate one complete RVM qualification matrix."""

    plan, plan_sha = load_plan(plan_path)
    ablation, ablation_file_sha, shortlist, ablation_model_backed = _verified_ablation(
        plan["ablation_report"]
    )
    candidates = {
        candidate["id"]: candidate
        for candidate in cast(list[dict[str, Any]], plan["candidates"])
    }
    ablation_rows = {
        str(row.get("id")): row
        for row in cast(list[object], ablation["rows"])
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    for candidate in candidates.values():
        if (
            candidate["role"] == "shortlist"
            and candidate["ablation_candidate_id"] not in shortlist
        ):
            raise MatteQualityError(
                "shortlist candidate lacks model-backed MATTE-0.3 admission"
            )
        if candidate["role"] == "shortlist":
            _verify_shortlist_binding(
                candidate,
                ablation_rows.get(candidate["ablation_candidate_id"]),
            )

    planned_cells = cast(list[dict[str, Any]], plan["cells"])
    actual_keys = [_cell_key(cell) for cell in planned_cells]
    duplicated = sorted(
        {"|".join(key) for key in actual_keys if actual_keys.count(key) > 1}
    )
    expected = _expected_keys(plan)
    actual = set(actual_keys)
    missing = sorted("|".join(key) for key in expected - actual)
    extra = sorted("|".join(key) for key in actual - expected)
    if duplicated or extra:
        raise MatteQualityError(
            "qualification matrix contains duplicate or extra cells"
        )

    canvas_by_id = {
        canvas["id"]: canvas for canvas in cast(list[dict[str, Any]], plan["canvases"])
    }
    compositor = cast(dict[str, Any], plan["qualified_compositor"])
    raw_compositor = dict(compositor)
    raw_compositor["use_model_foreground"] = False
    raw_compositor["light_wrap"] = 0.0
    raw_compositor = CompositingConfig.model_validate(raw_compositor).model_dump(
        mode="json"
    )
    policy = cast(dict[str, Any], plan["policy"])

    loaded: dict[str, _LoadedCell] = {}
    hardware_contracts: dict[str, tuple[object, ...]] = {}
    hardware_conflicts: set[str] = set()
    hardware_digest_ids: dict[str, set[str]] = {}
    environment_contracts: dict[tuple[str, str], tuple[object, ...]] = {}
    environment_conflicts: set[tuple[str, str]] = set()
    quality_cache: dict[str, dict[str, Any]] = {}
    for cell in planned_cells:
        if cell["status"] == "unavailable":
            loaded[cell["id"]] = _LoadedCell(
                plan=cell,
                result={
                    "id": cell["id"],
                    "key": list(_cell_key(cell)),
                    "outcome": "unavailable",
                    "availability_reason": cell["availability_reason"],
                    "evidence_kind": "unavailable",
                    "reasons": [
                        {
                            "category": "platform availability",
                            "detail": cell["availability_reason"],
                        }
                    ],
                },
            )
            continue
        bundle = MatteReplayBundle(cell["bundle"])
        annotations = MatteQualityAnnotations(cell["annotations"], bundle)
        _annotation_gates, _annotation_families = _gate_policy(list(annotations.gates))
        run, run_sha = load_run(cell["run"])
        bundle_sha = _bundle_manifest_digest(bundle)
        if run["bundle_manifest_sha256"] != bundle_sha:
            raise MatteQualityError("run evidence is bound to another replay bundle")
        if run["hardware"]["id"] != cell["hardware_id"]:
            raise MatteQualityError("run hardware does not match its matrix cell")
        if run["provider"]["requested"] != cell["provider"]:
            raise MatteQualityError("run provider does not match its matrix cell")
        hardware_id = str(run["hardware"]["id"])
        hardware_contract = (
            run["hardware"]["label"],
            run["hardware"]["platform"],
            run["hardware"]["identity_sha256"],
            run["hardware"]["identity_source"],
        )
        previous_hardware = hardware_contracts.setdefault(
            hardware_id,
            hardware_contract,
        )
        if previous_hardware != hardware_contract:
            hardware_conflicts.add(hardware_id)
        hardware_digest_ids.setdefault(
            str(run["hardware"]["identity_sha256"]),
            set(),
        ).add(hardware_id)
        candidate = candidates[cell["candidate_id"]]
        if (
            run["model"]["id"] != candidate["model_id"]
            or run["model"]["sha256"] != candidate["model_sha256"]
            or run["model"]["bytes"] != candidate["model_bytes"]
        ):
            raise MatteQualityError("run model identity does not match its candidate")
        canvas = canvas_by_id[cell["canvas_id"]]
        expected_compositing = (
            raw_compositor if cell["render_mode"] == "raw_model" else compositor
        )
        reasons, effective = _stable_evidence(
            bundle,
            candidate=candidate,
            provider=cell["provider"],
            width=canvas["width"],
            height=canvas["height"],
            render_mode=cast(RenderMode, cell["render_mode"]),
            expected_compositing=expected_compositing,
        )
        environment_key = (str(cell["hardware_id"]), str(cell["provider"]))
        environment_contract = (
            run["hardware"]["identity_sha256"],
            _json_bytes(run["provider"]["runtime"]),
            effective["device_id"],
            _json_bytes(run["resource_evidence"]),
        )
        environment_sha256 = _sha256(
            _json_bytes(
                {
                    "hardware_identity_sha256": run["hardware"]["identity_sha256"],
                    "provider_runtime": run["provider"]["runtime"],
                    "device_id": effective["device_id"],
                    "resource_evidence": run["resource_evidence"],
                }
            )
        )
        previous_environment = environment_contracts.setdefault(
            environment_key,
            environment_contract,
        )
        if previous_environment != environment_contract:
            environment_conflicts.add(environment_key)
        accelerators = run["hardware"]["inventory"]["accelerators"]
        if cell["provider"] != "cpu" and (
            not accelerators
            or type(effective["device_id"]) is not int
            or not 0 <= int(effective["device_id"]) < len(accelerators)
        ):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": (
                        "accelerator device id is not bound to hardware inventory"
                    ),
                }
            )
        reasons.extend(_annotation_coverage_reasons(annotations))
        frame_count = len(bundle.frames)
        service = run["service"]
        if len(run["source"]["frame_sha256"]) != frame_count:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": (
                        "pre-resize source digests do not cover every unique frame"
                    ),
                }
            )
        expected_input_selection = (
            "native-all"
            if cell["cadence"] == "native30"
            else "every-other-native-preserved-timestamps"
        )
        if (
            run["recurrence"]["input_selection"] != expected_input_selection
            or run["recurrence"]["model_invocation_count"] != frame_count
        ):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "recurrent execution does not match the cadence cell",
                }
            )
        aligned_series: list[tuple[str, Sequence[float] | None]] = [
            ("new-frame service", service["new_frame_service_ms"]),
            ("queue age", service["queue_age_ms"]),
            ("RSS", run["memory_bytes"]),
            ("VRAM", run["vram_bytes"]),
        ]
        for name, series in aligned_series:
            expected_count = (
                0 if name == "VRAM" and cell["provider"] == "cpu" else frame_count
            )
            actual_count = 0 if series is None else len(series)
            if actual_count != expected_count:
                reasons.append(
                    {
                        "category": "evidence completeness",
                        "detail": f"{name} samples do not cover every unique frame",
                    }
                )
        warmup_count = int(run["warmup_frame_count"])
        if warmup_count >= frame_count:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "warm-up count leaves no steady-state frame",
                }
            )
        resources, resource_reasons, bundle_rss, bundle_vram = _bundle_resources(
            bundle,
            cell["provider"],
        )
        reasons.extend(resource_reasons)
        if bundle_rss and bundle_rss != run["memory_bytes"]:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "run RSS samples do not match replay evidence",
                }
            )
        if bundle_vram is not None and bundle_vram != run["vram_bytes"]:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "run VRAM samples do not match replay evidence",
                }
            )
        if service["bundle_frame_total_alignment"] == "separate-pacing-boundary":
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": (
                        "separate pacing boundary cannot prove authoritative "
                        "per-frame service alignment"
                    ),
                }
            )
        if service["bundle_frame_total_alignment"] == "exact-non-pacing-sink":
            frame_totals = [
                cast(dict[str, Any], frame.get("timings_ms", {})).get("frame_total_ms")
                for frame in bundle.frames
            ]
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in frame_totals
                )
                or len(service["new_frame_service_ms"]) != frame_count
                or any(
                    not math.isclose(
                        float(cast(float, frame_total)),
                        float(service_total),
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    for frame_total, service_total in zip(
                        frame_totals,
                        service["new_frame_service_ms"],
                    )
                )
            ):
                reasons.append(
                    {
                        "category": "evidence completeness",
                        "detail": "bundle frame total does not match service evidence",
                    }
                )
        for name in (
            "license_reviewed",
            "packaging_supported",
            "download_integrity",
            "startup_succeeded",
        ):
            if run["model"][name] is not True:
                reasons.append(
                    {
                        "category": "model contract",
                        "detail": f"model {name} contract failed",
                    }
                )
        if not run["model"]["license"]:
            reasons.append(
                {"category": "model contract", "detail": "model license is missing"}
            )
        if run["provider"]["execution_proven"] is not True:
            reasons.append(
                {
                    "category": "provider fallback",
                    "detail": "provider execution was not proven",
                }
            )
        provenance_kinds = {
            str(plan["provenance"]["kind"]),
            str(run["provenance"]["kind"]),
            str(annotations.provenance.get("kind", "")),
        }
        if len(provenance_kinds) != 1:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "plan, run, and annotation provenance do not agree",
                }
            )
        quality_cache_key = _quality_cache_key(bundle, annotations)
        quality = quality_cache.get(quality_cache_key)
        if quality is None:
            quality = evaluate_bundle(
                bundle.root,
                annotations_root=annotations.root,
                metadata=EvaluationMetadata(
                    hardware_label=str(run["hardware"]["label"]),
                    backend="rvm",
                    device=str(cell["provider"]),
                    effective_detail=str(effective["effective_rvm_downsample_ratio"]),
                    configuration_label=str(cell["candidate_id"]),
                ),
            )
            quality_cache[quality_cache_key] = quality
        measured_cadence = cast(
            Mapping[str, object],
            cast(Mapping[str, object], quality["aggregate"])["cadence"],
        )
        measured_duration_s = (
            (
                int(bundle.frames[-1]["capture_monotonic_ns"])
                - int(bundle.frames[0]["capture_monotonic_ns"])
            )
            / 1_000_000_000.0
            if len(bundle.frames) >= 2
            else None
        )
        if measured_duration_s is None or not math.isclose(
            float(service["duration_s"]),
            measured_duration_s,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "attested service duration contradicts replay cadence",
                }
            )
        cadence_pairs = (
            ("capture_fps", "unique_input_fps"),
            ("unique_composite_fps", "base_composite_update_fps"),
            ("output_fps", "output_send_fps"),
        )
        for attested_name, measured_name in cadence_pairs:
            measured = measured_cadence.get(measured_name)
            if (
                isinstance(measured, bool)
                or not isinstance(measured, (int, float))
                or not math.isfinite(float(measured))
                or not math.isclose(
                    float(service[attested_name]),
                    float(measured),
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
            ):
                reasons.append(
                    {
                        "category": "evidence completeness",
                        "detail": (
                            f"attested {attested_name} contradicts replay cadence"
                        ),
                    }
                )
        if service["output_repeat_count"] != measured_cadence.get(
            "exact_final_output_repeat_count"
        ) or service["output_timeline_complete"] is not measured_cadence.get(
            "output_timeline_complete"
        ):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "attested output timeline contradicts replay evidence",
                }
            )
        annotation_gate_results = cast(list[dict[str, Any]], quality["gates"])
        annotation_gates_decidable = bool(annotation_gate_results)
        if not annotation_gate_results:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "digest-bound annotation release gates are missing",
                }
            )
        for annotation_gate in annotation_gate_results:
            status = annotation_gate.get("status")
            if status == "fail":
                reasons.append(
                    {
                        "category": "annotation gate",
                        "detail": (
                            f"annotation gate {annotation_gate.get('id')} failed"
                        ),
                    }
                )
            elif status != "pass":
                annotation_gates_decidable = False
                reasons.append(
                    {
                        "category": "evidence completeness",
                        "detail": (
                            f"annotation gate {annotation_gate.get('id')} "
                            "was not evaluated"
                        ),
                    }
                )
        gate_results, gate_reasons, gate_decidable = _quality_gates(
            quality,
            cast(list[dict[str, Any]], policy["quality_gates"]),
        )
        reasons.extend(gate_reasons)
        timings, timing_reasons, timing_complete = _phase_timings(bundle, warmup_count)
        reasons.extend(timing_reasons)
        serialized_reasons = _serialized_service_reasons(
            bundle,
            service["queue_age_ms"],
        )
        reasons.extend(serialized_reasons)
        timing_complete = timing_complete and not serialized_reasons
        if warmup_count < policy["min_warmup_frames"]:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "warm-up sample count is below policy",
                }
            )
        if len(bundle.frames) - warmup_count < policy["min_steady_frames"]:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "steady-state sample count is below policy",
                }
            )
        warm_service_summary = _summary(service["new_frame_service_ms"][:warmup_count])
        service_summary = _summary(service["new_frame_service_ms"][warmup_count:])
        budget = float(
            policy[
                "native30_service_p95_ms"
                if cell["cadence"] == "native30"
                else "decimated15_service_p95_ms"
            ]
        )
        minimum_fps = float(
            policy[
                "native30_min_unique_fps"
                if cell["cadence"] == "native30"
                else "decimated15_min_unique_fps"
            ]
        )
        p95 = cast(float | None, service_summary["p95"])
        budget_status = (
            "not_measured"
            if p95 is None
            else ("within_budget" if p95 <= budget else "degraded")
        )
        if budget_status == "not_measured":
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "new-frame service performance was not measured",
                }
            )
        elif budget_status == "degraded":
            reasons.append(
                {
                    "category": "performance",
                    "detail": "service p95 missed; only a lower-rate declaration is supported",
                }
            )
        queue_growth = service["queue_age_ms"][-1] - service["queue_age_ms"][0]
        capture_fps_min, capture_fps_max = _CADENCE_FPS_RANGES[cell["cadence"]]
        if not capture_fps_min <= service["capture_fps"] <= capture_fps_max:
            reasons.append(
                {
                    "category": "performance",
                    "detail": (
                        f"{cell['cadence']} source cadence is outside its "
                        "ratified qualification range"
                    ),
                }
            )
        if (
            service["unique_composite_fps"] < minimum_fps
            or max(service["queue_age_ms"]) > policy["max_queue_age_ms"]
            or queue_growth > policy["max_queue_age_growth_ms"]
        ):
            reasons.append(
                {
                    "category": "performance",
                    "detail": "sustained unique cadence or queue-age gate failed",
                }
            )
        if (
            service["duration_s"] < policy["min_observation_s"]
            or service["output_timeline_complete"] is not True
        ):
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "sustained cadence observation is incomplete",
                }
            )
        if not run["memory_bytes"]:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "process memory was not measured",
                }
            )
        if cell["provider"] != "cpu" and run["vram_bytes"] is None:
            reasons.append(
                {
                    "category": "evidence completeness",
                    "detail": "accelerator VRAM was not measured",
                }
            )
        proxy = (
            run["evidence_kind"] != "model-backed"
            or run["provenance"]["kind"] == "generated"
            or annotations.provenance.get("kind") == "generated"
            or plan["provenance"]["kind"] == "generated"
            or not ablation_model_backed
        )
        if proxy:
            reasons.append(
                {
                    "category": "evidence class",
                    "detail": "generated/proxy evidence cannot qualify a profile",
                }
            )
        outcome = _reasons_outcome(reasons, proxy=proxy)
        loaded[cell["id"]] = _LoadedCell(
            plan=cell,
            bundle=bundle,
            annotations=annotations,
            run=run,
            source_contract=_source_contract(bundle),
            capture_lineage_contract=_capture_lineage_contract(bundle),
            annotation_contract=_annotation_contract(annotations),
            annotation_projection_contract=_annotation_projection_contract(annotations),
            result={
                "id": cell["id"],
                "key": list(_cell_key(cell)),
                "outcome": outcome,
                "evidence_kind": run["evidence_kind"],
                "source": {
                    "bundle_manifest_sha256": bundle_sha,
                    "annotation_manifest_sha256": annotations.manifest_sha256,
                    "annotation_contract_sha256": _sha256(
                        _json_bytes(_annotation_contract(annotations))
                    ),
                    "run_sha256": run_sha,
                    "source_contract_sha256": _sha256(
                        _json_bytes(_source_contract(bundle))
                    ),
                    "pre_resize_clip_sha256": run["source"]["clip_sha256"],
                    "pre_resize_frame_contract_sha256": _sha256(
                        _json_bytes(run["source"]["frame_sha256"])
                    ),
                    "pre_resize_pixel_contract": run["source"]["pixel_contract"],
                },
                "effective": effective,
                "annotation_gates": annotation_gate_results,
                "quality_gates": gate_results,
                "performance": {
                    **timings,
                    "startup_ms": _summary(run["startup_ms"]),
                    "new_frame_service_warm_up_ms": warm_service_summary,
                    "new_frame_service_steady_state_ms": service_summary,
                    "budget_ms": budget,
                    "budget_status": budget_status,
                    **resources,
                    "rss_bytes": _summary(run["memory_bytes"]),
                    "vram_bytes": (
                        {"applicable": False, **_summary([])}
                        if run["vram_bytes"] is None
                        else {
                            "applicable": True,
                            **_summary(run["vram_bytes"]),
                        }
                    ),
                    "resource_evidence": run["resource_evidence"],
                    "frame_total_budget_authoritative": (
                        service["bundle_frame_total_alignment"]
                        == "exact-non-pacing-sink"
                    ),
                },
                "cadence": {
                    **service,
                    "new_frame_service_ms": None,
                    "queue_age_ms": None,
                    "queue_age_max_ms": _round(max(service["queue_age_ms"])),
                    "queue_age_growth_ms": _round(queue_growth),
                    "qualification_capture_fps_range": [
                        capture_fps_min,
                        capture_fps_max,
                    ],
                },
                "model": {
                    "id": run["model"]["id"],
                    "sha256": run["model"]["sha256"],
                    "bytes": run["model"]["bytes"],
                    "contract_complete": all(
                        run["model"][name] is True
                        for name in (
                            "license_reviewed",
                            "packaging_supported",
                            "download_integrity",
                            "startup_succeeded",
                        )
                    ),
                },
                "hardware": {
                    "id": run["hardware"]["id"],
                    "label": run["hardware"]["label"],
                    "platform": run["hardware"]["platform"],
                    "identity_sha256": run["hardware"]["identity_sha256"],
                    "inventory": run["hardware"]["inventory"],
                },
                "provider": {
                    "requested": run["provider"]["requested"],
                    "active": effective["active_provider"],
                    "device_id": effective["device_id"],
                    "execution_proven": run["provider"]["execution_proven"],
                    "runtime": run["provider"]["runtime"],
                    "environment_sha256": environment_sha256,
                    "fallback_observed": any(
                        reason["category"] == "provider fallback" for reason in reasons
                    ),
                },
                "gate_evidence_complete": (
                    gate_decidable and annotation_gates_decidable
                ),
                "timing_evidence_complete": timing_complete,
                "reasons": reasons,
            },
        )

    # Identity contracts are evaluated after every cell is loaded.  A mismatch
    # is evidence, not malformed JSON: retain the row and fail qualification.
    duplicate_hardware_ids = {
        hardware_id
        for hardware_ids in hardware_digest_ids.values()
        if len(hardware_ids) > 1
        for hardware_id in hardware_ids
    }
    invalid_hardware_ids = hardware_conflicts | duplicate_hardware_ids
    for item in loaded.values():
        if item.run is not None and item.plan["hardware_id"] in invalid_hardware_ids:
            item.result["reasons"].append(
                {
                    "category": "evidence completeness",
                    "detail": "hardware inventory identity is inconsistent",
                }
            )
        if (
            item.run is not None
            and (
                str(item.plan["hardware_id"]),
                str(item.plan["provider"]),
            )
            in environment_conflicts
        ):
            item.result["reasons"].append(
                {
                    "category": "evidence completeness",
                    "detail": "hardware/provider environment identity is inconsistent",
                }
            )
    native_pixels: dict[str, dict[str, list[_LoadedCell]]] = {}
    native_lineages: dict[str, list[_LoadedCell]] = {}
    native_pre_resize_sources: dict[str, list[_LoadedCell]] = {}
    annotation_families: dict[tuple[str, str], dict[str, list[_LoadedCell]]] = {}
    for item in loaded.values():
        cell = item.plan
        if item.bundle is None or item.source_contract is None:
            continue
        if cell["cadence"] == "native30":
            source_digest = _sha256(_json_bytes(item.source_contract))
            native_pixels.setdefault(cell["canvas_id"], {}).setdefault(
                source_digest, []
            ).append(item)
            if item.capture_lineage_contract is not None:
                lineage_digest = _sha256(_json_bytes(item.capture_lineage_contract))
                native_lineages.setdefault(lineage_digest, []).append(item)
            if item.run is not None:
                pre_resize_digest = _sha256(_json_bytes(item.run["source"]))
                native_pre_resize_sources.setdefault(
                    pre_resize_digest,
                    [],
                ).append(item)
        else:
            parent = loaded[cell["native30_cell_id"]]
            expected_parent_key = list(_cell_key(cell))
            expected_parent_key[4] = "native30"
            if (
                parent.bundle is None
                or list(_cell_key(parent.plan)) != expected_parent_key
                or item.source_contract
                != cast(list[dict[str, object]], parent.source_contract)[::2]
            ):
                item.result["reasons"].append(
                    {
                        "category": "source identity",
                        "detail": "decimated15 is not the exact every-other native30 subsequence",
                    }
                )
            if (
                item.run is None
                or parent.run is None
                or item.run["source"]["clip_sha256"]
                != parent.run["source"]["clip_sha256"]
                or item.run["source"]["pixel_contract"]
                != parent.run["source"]["pixel_contract"]
                or item.run["source"]["frame_sha256"]
                != parent.run["source"]["frame_sha256"][::2]
            ):
                item.result["reasons"].append(
                    {
                        "category": "source identity",
                        "detail": (
                            "decimated15 pre-resize source is not the exact "
                            "every-other native30 projection"
                        ),
                    }
                )
            parent_projection = parent.annotation_projection_contract
            item_projection = item.annotation_projection_contract
            if (
                not isinstance(parent_projection, Mapping)
                or not isinstance(item_projection, Mapping)
                or item_projection.get("provenance")
                != parent_projection.get("provenance")
                or item_projection.get("gates") != parent_projection.get("gates")
                or item_projection.get("frames")
                != cast(list[object], parent_projection.get("frames"))[::2]
            ):
                item.result["reasons"].append(
                    {
                        "category": "annotation identity",
                        "detail": (
                            "decimated15 annotations are not the exact "
                            "every-other native30 projection"
                        ),
                    }
                )
        if item.annotation_contract is not None:
            annotation_digest = _sha256(_json_bytes(item.annotation_contract))
            annotation_families.setdefault(
                (cell["canvas_id"], cell["cadence"]),
                {},
            ).setdefault(annotation_digest, []).append(item)
        if cell["render_mode"] == "qualified_compositor":
            pair = loaded[cell["paired_raw_cell_id"]]
            expected_pair_key = list(_cell_key(cell))
            expected_pair_key[5] = "raw_model"
            tracks = (
                "raw_frame",
                "raw_mask",
                "refined_mask",
                "clean_foreground",
                "backdrop_frame",
            )
            if (
                pair.bundle is None
                or list(_cell_key(pair.plan)) != expected_pair_key
                or _track_contract(item.bundle, tracks)
                != _track_contract(pair.bundle, tracks)
            ):
                item.result["reasons"].append(
                    {
                        "category": "attribution",
                        "detail": "qualified compositor lacks an exact raw-model pair",
                    }
                )

    for canvas_id, contracts in native_pixels.items():
        if len(contracts) > 1:
            for item in loaded.values():
                if item.bundle is not None and item.plan["canvas_id"] == canvas_id:
                    item.result["reasons"].append(
                        {
                            "category": "source identity",
                            "detail": (
                                "native30 source differs within its canvas family"
                            ),
                        }
                    )
    if len(native_lineages) > 1:
        for item in loaded.values():
            if item.bundle is not None:
                item.result["reasons"].append(
                    {
                        "category": "source identity",
                        "detail": (
                            "native30 capture lineage differs across canvas targets"
                        ),
                    }
                )
    if len(native_pre_resize_sources) > 1:
        for item in loaded.values():
            if item.bundle is not None:
                item.result["reasons"].append(
                    {
                        "category": "source identity",
                        "detail": (
                            "native30 pre-resize source differs across the matrix"
                        ),
                    }
                )
    for contracts in annotation_families.values():
        if len(contracts) > 1:
            for family in contracts.values():
                for item in family:
                    item.result["reasons"].append(
                        {
                            "category": "annotation identity",
                            "detail": (
                                "quality annotations differ within the source family"
                            ),
                        }
                    )

    # Recompute outcomes after cross-cell proofs.
    for item in loaded.values():
        if item.result["outcome"] == "unavailable":
            continue
        proxy = item.result["evidence_kind"] != "model-backed" or any(
            reason["category"] == "evidence class" for reason in item.result["reasons"]
        )
        item.result["outcome"] = _reasons_outcome(item.result["reasons"], proxy=proxy)

    candidate_decisions: list[dict[str, object]] = []
    for candidate_id in candidates:
        rows = [
            item.result
            for item in loaded.values()
            if item.plan["candidate_id"] == candidate_id
        ]
        outcomes = [row["outcome"] for row in rows]
        required_count = sum(key[0] == candidate_id for key in expected)
        complete = len(rows) == required_count
        candidate_decisions.append(
            {
                "candidate_id": candidate_id,
                "outcome": (
                    "qualified"
                    if complete
                    and rows
                    and all(outcome == "qualified" for outcome in outcomes)
                    else (
                        "rejected"
                        if any(outcome == "rejected" for outcome in outcomes)
                        else "not_decidable"
                    )
                ),
                "qualified_cell_count": sum(
                    outcome == "qualified" for outcome in outcomes
                ),
                "declared_cell_count": len(rows),
                "required_cell_count": required_count,
            }
        )

    decision_by_id = {str(item["candidate_id"]): item for item in candidate_decisions}
    proposals = cast(list[dict[str, str]], plan["profile_proposals"])
    profile_reasons: list[str] = []
    if missing:
        profile_reasons.append("required matrix cells are missing")
    for proposal in proposals:
        candidate_id = proposal["candidate_id"]
        if decision_by_id[candidate_id]["outcome"] != "qualified":
            profile_reasons.append(
                f"{proposal['name']} candidate is not fully qualified"
            )
        qualified_hardware = {
            item.plan["hardware_id"]
            for item in loaded.values()
            if item.plan["candidate_id"] == candidate_id
            and item.result["outcome"] == "qualified"
        }
        if len(qualified_hardware) < policy["minimum_profile_hardware_count"]:
            profile_reasons.append(
                f"{proposal['name']} cross-device hardware coverage is insufficient"
            )
    profile_reasons.extend(_profile_semantic_reasons(proposals, loaded))
    profile_reasons = list(dict.fromkeys(profile_reasons))
    profile_status = (
        "qualified"
        if proposals and not profile_reasons
        else ("not_proposed" if not proposals else "not_decidable")
    )
    profile_definitions: list[dict[str, object]] = []
    if profile_status == "qualified":
        for proposal in proposals:
            candidate = candidates[proposal["candidate_id"]]
            segmentation = dict(candidate["segmentation"])
            segmentation.pop("model_path", None)
            scope: list[dict[str, object]] = []
            environment_keys = sorted(
                {
                    (
                        item.plan["hardware_id"],
                        item.plan["provider"],
                        item.plan["canvas_id"],
                    )
                    for item in loaded.values()
                    if item.plan["candidate_id"] == proposal["candidate_id"]
                    and item.result["outcome"] == "qualified"
                }
            )
            for hardware_id, provider, canvas_id in environment_keys:
                representative = next(
                    item
                    for item in loaded.values()
                    if item.plan["candidate_id"] == proposal["candidate_id"]
                    and item.plan["hardware_id"] == hardware_id
                    and item.plan["provider"] == provider
                    and item.plan["canvas_id"] == canvas_id
                    and item.result["outcome"] == "qualified"
                )
                canvas = canvas_by_id[canvas_id]
                scope.append(
                    {
                        "hardware": representative.result["hardware"],
                        "provider": representative.result["provider"],
                        "canvas": {
                            "id": canvas_id,
                            "width": canvas["width"],
                            "height": canvas["height"],
                        },
                        "cadences": list(_CADENCES),
                        "render_modes": list(_RENDER_MODES),
                    }
                )
            profile_definitions.append(
                {
                    "name": proposal["name"],
                    "candidate_id": proposal["candidate_id"],
                    "model": {
                        "id": candidate["model_id"],
                        "sha256": candidate["model_sha256"],
                        "bytes": candidate["model_bytes"],
                    },
                    "segmentation": segmentation,
                    "segmentation_sha256": candidate["segmentation_sha256"],
                    "qualified_compositor": compositor,
                    "qualified_compositor_sha256": plan["qualified_compositor_sha256"],
                    "qualification_scope": scope,
                    "semantic_ordering": {
                        "performance": "lowest full-frame service p95",
                        "balanced": "non-worse intermediate latency and quality",
                        "quality": "non-worse gate metrics with a strict gain",
                        "required_in_every_scope": True,
                    },
                }
            )

    rows = [loaded[cell["id"]].result for cell in planned_cells]
    candidate_catalog: list[dict[str, object]] = []
    for candidate in cast(list[dict[str, Any]], plan["candidates"]):
        path_free_segmentation = dict(candidate["segmentation"])
        path_free_segmentation.pop("model_path", None)
        candidate_catalog.append(
            {
                "id": candidate["id"],
                "role": candidate["role"],
                "ablation_candidate_id": candidate["ablation_candidate_id"],
                "model": {
                    "id": candidate["model_id"],
                    "sha256": candidate["model_sha256"],
                    "bytes": candidate["model_bytes"],
                },
                "segmentation": path_free_segmentation,
                "segmentation_sha256": candidate["segmentation_sha256"],
            }
        )
    qualification_contract = {
        "quality_policy_id": policy["quality_policy_id"],
        "policy": policy,
        "candidates": candidate_catalog,
        "qualified_compositor": compositor,
        "qualified_compositor_sha256": plan["qualified_compositor_sha256"],
        "canvases": plan["canvases"],
        "declared_hardware_scope": plan["hardware"],
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "source": {
            "plan_sha256": plan_sha,
            "ablation_file_sha256": ablation_file_sha,
            "ablation_evidence_sha256": ablation["evidence_sha256"],
            "admitted_model_backed_shortlist": sorted(shortlist),
        },
        "provenance": {
            "qualification_id": plan["provenance"]["qualification_id"],
            "kind": plan["provenance"]["kind"],
            "contains_private_footage_in_repository": False,
        },
        "privacy": {
            "report_is_content_free": True,
            "input_paths_retained": False,
            "network_camera_model_or_sink_opened": False,
        },
        "qualification_contract": qualification_contract,
        "coverage": {
            "required_cell_count": len(expected),
            "declared_cell_count": len(actual),
            "missing_cells": missing,
            "unavailable_cells": [
                row["id"] for row in rows if row["outcome"] == "unavailable"
            ],
            "complete": not missing
            and all(row["outcome"] != "unavailable" for row in rows),
            "hardware": [
                {
                    "id": hardware_id,
                    "label": contract[0],
                    "platform": contract[1],
                    "identity_sha256": contract[2],
                    "inventory": next(
                        item.run["hardware"]["inventory"]
                        for item in loaded.values()
                        if item.run is not None
                        and item.plan["hardware_id"] == hardware_id
                    ),
                }
                for hardware_id, contract in sorted(hardware_contracts.items())
            ],
        },
        "rows": rows,
        "candidate_decisions": candidate_decisions,
        "profiles": {
            "status": profile_status,
            "definitions": profile_definitions,
            "reasons": profile_reasons,
            "stable_cross_device_meaning_required": True,
        },
        "production": {
            "default_changed": False,
            "high_detail_global_default_selected": False,
            "generated_proxy_can_select_profile": False,
        },
    }
    deterministic = {
        name: report[name]
        for name in (
            "source",
            "provenance",
            "privacy",
            "qualification_contract",
            "coverage",
            "rows",
            "candidate_decisions",
            "profiles",
            "production",
        )
    }
    report["evidence_sha256"] = _sha256(_json_bytes(deterministic))
    return report


def report_markdown(report: Mapping[str, object]) -> str:
    """Render a compact review companion; canonical JSON remains authoritative."""

    rows = cast(list[dict[str, Any]], report["rows"])
    profiles = cast(dict[str, Any], report["profiles"])
    lines = [
        "# RVM profile qualification report",
        "",
        f"- Evidence SHA-256: `{report['evidence_sha256']}`",
        f"- Profile status: **{profiles['status']}**",
        "- Production default changed: **no**",
        "",
        "| Cell | Candidate | Provider | Canvas | Cadence | Mode | Outcome | Budget |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        key = row["key"]
        performance = cast(dict[str, Any], row.get("performance", {}))
        lines.append(
            f"| `{row['id']}` | `{key[0]}` | `{key[2]}` | `{key[3]}` | "
            f"`{key[4]}` | `{key[5]}` | **{row['outcome']}** | "
            f"`{performance.get('budget_status')}` |"
        )
    lines.extend(
        [
            "",
            "Missing, unavailable, generated, fallback, unpaired, or null "
            "required evidence never qualifies a named profile.",
            "",
        ]
    )
    return "\n".join(lines)


def run_qualification(
    plan_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Evaluate a plan and write owner-only, path-free JSON and Markdown."""

    report = qualify_plan(plan_path)
    output = Path(output_root)
    _private_directory(output, create=True)
    _atomic_private_write(output / "qualification.json", _json_bytes(report))
    _atomic_private_write(
        output / "qualification.md",
        report_markdown(report).encode("utf-8"),
    )
    return report


def build_parser(
    *,
    prog: str = "custback matte-rvm-qualify",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Aggregate a strict cross-device RVM qualification matrix",
    )
    parser.add_argument("--plan", required=True, help="owner-only qualification plan")
    parser.add_argument(
        "--output",
        required=True,
        help="new owner-only content-free report directory",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback matte-rvm-qualify",
) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        report = run_qualification(args.plan, args.output)
    except (
        OSError,
        MatteDiagnosticsError,
        MatteQualityError,
        ValueError,
    ) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2
    profile_status = cast(dict[str, Any], report["profiles"])["status"]
    failed = (
        profile_status == "not_decidable"
        or not cast(dict[str, Any], report["coverage"])["complete"]
        or any(
            row["outcome"] != "qualified"
            for row in cast(list[dict[str, Any]], report["rows"])
        )
    )
    print(
        f"evaluated {len(report['rows'])} RVM matrix cell(s); "
        f"profile status {profile_status}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
