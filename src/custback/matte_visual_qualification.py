"""Offline, privacy-preserving MATTE-5.2 visual qualification authority.

The qualifier joins already-recorded private replay/annotation pairs, local
boundary captures, and a structured human review.  It deliberately opens no
camera, model, network service, preview, or output sink.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, cast

import cv2
import numpy as np

from .config import CompositingConfig, SegmentationConfig
from .matte_diagnostics import (
    MAX_MANIFEST_BYTES,
    MatteDiagnosticsError,
    MatteReplayBundle,
    _atomic_private_write,
    _json_bytes,
    _private_directory,
    _read_private_file,
)
from .matte_quality import (
    EvaluationMetadata,
    MatteQualityAnnotations,
    MatteQualityError,
    _metric_path,
    evaluate_bundle,
)
from .matte_policy import MatteBackendKind, resolve_matte_policy

PLAN_SCHEMA = "custback.matte-visual-qualification-plan"
PLAN_VERSION = 1
BOUNDARY_SCHEMA = "custback.matte-visual-boundary-evidence"
BOUNDARY_VERSION = 1
REVIEW_SCHEMA = "custback.matte-visual-human-review"
REVIEW_VERSION = 1
REPORT_SCHEMA = "custback.matte-visual-qualification-report"
REPORT_VERSION = 1

REQUIRED_BOUNDARIES = (
    "in_memory",
    "highgui_pre_overlay",
    "snapshot_jpeg",
    "mjpeg_jpeg",
    "websocket_jpeg",
    "pyvirtualcam_loopback",
    "windows_native_loopback",
)
REQUIRED_REVIEW_CONCERNS = (
    "halo",
    "edge_shimmer",
    "ghost_trail",
    "cutout_sharpness",
    "hair_retention",
    "opaque_core_backdrop_leakage",
    "accessory_coverage",
    "motion_cadence",
)
REQUIRED_COVERAGE: dict[str, tuple[str, ...]] = {
    "appearances": (
        "bald_or_short_hair",
        "long_or_fine_hair",
        "glasses",
        "facial_hair",
        "headphones_or_solid_accessories",
        "dark_opaque_clothing",
        "light_opaque_clothing",
        "skin_tone_diversity",
        "lighting_diversity",
    ),
    "motions": (
        "stationary",
        "speech_micro_motion",
        "slow_turn",
        "fast_turn",
        "hand_or_prop_crossing_face",
        "entering_frame",
        "leaving_frame",
    ),
    "source_conditions": (
        "bright",
        "dim",
        "compression_noise",
        "low_contrast",
        "clutter",
    ),
    "backgrounds": (
        "static_image",
        "dynamic_video",
        "blur",
        "solid_color",
        "live_camera",
    ),
    "cadences": (
        "fps_15",
        "fps_30",
        "fps_60",
        "irregular",
        "drops",
        "restart",
    ),
    "canvases": ("640x360", "1280x720", "1920x1080"),
}

MAX_CASES = 256
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_PIXELS = 4096 * 2160
MAX_FULL_FRAME_MAE = 4.0
MAX_EDGE_BAND_MAE = 8.0

_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_MATTE_POLICY_KEYS = (
    "selected_backend_kind",
    "backend_kind",
    "passthrough",
    "experimental_rvm_generic",
    "configured",
    "effective",
    "controls",
)


class MatteVisualQualificationError(MatteQualityError):
    """A MATTE-5.2 plan or one of its evidence inputs is invalid."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_keys(value: object, expected: Sequence[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise MatteVisualQualificationError(
            f"{label} does not match the strict version-1 schema"
        )
    return cast(dict[str, Any], value)


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise MatteVisualQualificationError(
            f"{label} must be a lowercase safe identifier"
        )
    return value


def _safe_label(value: object, label: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str) or _LABEL.fullmatch(value) is None:
        raise MatteVisualQualificationError(f"{label} must be bounded and path-free")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MatteVisualQualificationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _finite(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MatteVisualQualificationError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise MatteVisualQualificationError(f"{label} is below its minimum")
    if maximum is not None and result > maximum:
        raise MatteVisualQualificationError(f"{label} exceeds its maximum")
    return result


def _read_json(path: Path | str, *, label: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(path)
    _private_directory(candidate.parent, create=False)
    payload = _read_private_file(candidate, max_bytes=MAX_MANIFEST_BYTES)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatteVisualQualificationError(f"{label} is malformed") from exc
    if not isinstance(value, dict):
        raise MatteVisualQualificationError(f"{label} must contain an object")
    return value, payload


def _safe_input_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 4096:
        raise MatteVisualQualificationError(f"{label} path is invalid")
    return value


def _unique_enum_list(
    value: object,
    allowed: Sequence[str],
    label: str,
) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or item not in allowed for item in value)
        or len(set(value)) != len(value)
    ):
        raise MatteVisualQualificationError(f"{label} coverage values are invalid")
    return sorted(cast(list[str], value))


def _path_free_segmentation(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    model_path = result.pop("model_path", "")
    result["model_path_present"] = bool(model_path)
    return result


def _normalized_candidate(value: object) -> dict[str, Any]:
    candidate = _strict_keys(
        value,
        ("id", "segmentation", "compositing"),
        "visual candidate",
    )
    candidate_id = _safe_id(candidate["id"], "candidate id")
    try:
        segmentation = SegmentationConfig.model_validate(
            candidate["segmentation"]
        ).model_dump(mode="json")
        compositing = CompositingConfig.model_validate(
            candidate["compositing"]
        ).model_dump(mode="json")
    except ValueError as exc:
        raise MatteVisualQualificationError(
            f"candidate {candidate_id} has invalid matte controls"
        ) from exc
    return {
        "id": candidate_id,
        "segmentation": segmentation,
        "compositing": compositing,
    }


def _normalized_expected_effective(value: object, label: str) -> dict[str, Any]:
    effective = _strict_keys(
        value,
        (
            "segmentation_backend",
            "segmentation_device",
            "rvm_downsample_ratio",
            "matte_policy",
        ),
        label,
    )
    backend = _safe_label(effective["segmentation_backend"], f"{label} backend")
    device = _safe_label(effective["segmentation_device"], f"{label} device")
    ratio_value = effective["rvm_downsample_ratio"]
    ratio = (
        None
        if ratio_value is None
        else _finite(ratio_value, f"{label} RVM ratio", minimum=0.0, maximum=1.0)
    )
    if ratio == 0.0:
        raise MatteVisualQualificationError(
            f"{label} effective RVM ratio must be positive or null"
        )
    policy = _strict_keys(
        effective["matte_policy"],
        _MATTE_POLICY_KEYS,
        f"{label} matte policy",
    )
    if (
        type(policy["passthrough"]) is not bool
        or type(policy["experimental_rvm_generic"]) is not bool
    ):
        raise MatteVisualQualificationError(f"{label} matte policy flags are invalid")
    try:
        _json_bytes(policy)
    except (TypeError, ValueError) as exc:
        raise MatteVisualQualificationError(
            f"{label} matte policy is not canonical JSON"
        ) from exc
    return {
        "segmentation_backend": backend,
        "segmentation_device": device,
        "rvm_downsample_ratio": ratio,
        "matte_policy": policy,
    }


def _normalized_pair(value: object, label: str) -> dict[str, str]:
    pair = _strict_keys(value, ("bundle", "annotations"), label)
    return {
        "bundle": _safe_input_path(pair["bundle"], f"{label} bundle"),
        "annotations": _safe_input_path(pair["annotations"], f"{label} annotations"),
    }


def _normalized_baseline_expected(value: object, label: str) -> dict[str, Any]:
    expected = _strict_keys(
        value,
        ("segmentation", "compositing", "effective"),
        label,
    )
    try:
        segmentation = SegmentationConfig.model_validate(
            expected["segmentation"]
        ).model_dump(mode="json")
        compositing = CompositingConfig.model_validate(
            expected["compositing"]
        ).model_dump(mode="json")
    except ValueError as exc:
        raise MatteVisualQualificationError(
            f"{label} configured controls are invalid"
        ) from exc
    return {
        "segmentation": segmentation,
        "compositing": compositing,
        "effective": _normalized_expected_effective(
            expected["effective"], f"{label} effective"
        ),
    }


def _load_plan(path: Path | str) -> tuple[dict[str, Any], str]:
    value, payload = _read_json(path, label="matte visual qualification plan")
    plan = _strict_keys(
        value,
        ("schema", "version", "provenance", "policy", "candidates", "cases"),
        "matte visual qualification plan",
    )
    if (
        plan["schema"] != PLAN_SCHEMA
        or type(plan["version"]) is not int
        or plan["version"] != PLAN_VERSION
    ):
        raise MatteVisualQualificationError(
            "unsupported matte visual qualification plan"
        )

    provenance = _strict_keys(
        plan["provenance"],
        (
            "qualification_id",
            "kind",
            "license_or_consent_reference",
            "contains_private_footage_in_repository",
        ),
        "qualification provenance",
    )
    qualification_id = _safe_id(provenance["qualification_id"], "qualification id")
    kind = provenance["kind"]
    if kind not in ("generated", "consented-local", "licensed-local"):
        raise MatteVisualQualificationError("qualification provenance kind is invalid")
    reference = provenance["license_or_consent_reference"]
    if not isinstance(reference, str) or len(reference) > 512:
        raise MatteVisualQualificationError(
            "qualification provenance reference is invalid"
        )
    if kind != "generated" and not reference.strip():
        raise MatteVisualQualificationError(
            "local visual qualification requires a consent or license reference"
        )
    if provenance["contains_private_footage_in_repository"] is not False:
        raise MatteVisualQualificationError(
            "private qualification footage cannot be repository data"
        )

    policy = _strict_keys(
        plan["policy"],
        (
            "quality_policy_id",
            "maximum_full_frame_mae",
            "maximum_edge_band_mae",
        ),
        "visual qualification policy",
    )
    if policy["quality_policy_id"] != "matte-0.2-ratified-v1":
        raise MatteVisualQualificationError("quality policy id is not ratified")
    full_mae = _finite(
        policy["maximum_full_frame_mae"],
        "maximum full-frame MAE",
        minimum=0.0,
        maximum=MAX_FULL_FRAME_MAE,
    )
    edge_mae = _finite(
        policy["maximum_edge_band_mae"],
        "maximum edge-band MAE",
        minimum=0.0,
        maximum=MAX_EDGE_BAND_MAE,
    )
    raw_candidates = plan["candidates"]
    if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 32:
        raise MatteVisualQualificationError("candidate count is invalid")
    candidates = [_normalized_candidate(candidate) for candidate in raw_candidates]
    candidate_ids = [str(candidate["id"]) for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise MatteVisualQualificationError("candidate id is duplicated")

    raw_cases = plan["cases"]
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= MAX_CASES:
        raise MatteVisualQualificationError("qualification case count is invalid")
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    boundary_paths: set[Path] = set()
    review_paths: set[Path] = set()
    evidence_inputs: set[Path] = set()
    for raw_case in raw_cases:
        case = _strict_keys(
            raw_case,
            (
                "id",
                "candidate_id",
                "baseline",
                "baseline_expected",
                "candidate",
                "expected_effective",
                "coverage",
                "boundary_evidence",
                "review",
            ),
            "visual qualification case",
        )
        case_id = _safe_id(case["id"], "case id")
        if case_id in case_ids:
            raise MatteVisualQualificationError("case id is duplicated")
        case_ids.add(case_id)
        candidate_id = _safe_id(case["candidate_id"], "case candidate id")
        if candidate_id not in candidate_ids:
            raise MatteVisualQualificationError(
                f"case {case_id} references an unknown candidate"
            )
        baseline = _normalized_pair(case["baseline"], f"case {case_id} baseline")
        candidate = _normalized_pair(case["candidate"], f"case {case_id} candidate")
        normalized_baseline_bundle = Path(baseline["bundle"]).resolve(strict=False)
        normalized_candidate_bundle = Path(candidate["bundle"]).resolve(strict=False)
        if normalized_baseline_bundle == normalized_candidate_bundle:
            raise MatteVisualQualificationError(
                f"case {case_id} baseline and candidate bundles must be distinct"
            )
        for evidence_path in (
            normalized_baseline_bundle,
            Path(baseline["annotations"]).resolve(strict=False),
            normalized_candidate_bundle,
            Path(candidate["annotations"]).resolve(strict=False),
        ):
            if evidence_path in evidence_inputs:
                raise MatteVisualQualificationError(
                    "replay and annotation inputs cannot be reused across cases"
                )
            evidence_inputs.add(evidence_path)
        coverage_value = _strict_keys(
            case["coverage"], tuple(REQUIRED_COVERAGE), f"case {case_id} coverage"
        )
        coverage = {
            axis: _unique_enum_list(
                coverage_value[axis], REQUIRED_COVERAGE[axis], f"case {case_id} {axis}"
            )
            for axis in REQUIRED_COVERAGE
        }
        boundary_path = _safe_input_path(
            case["boundary_evidence"], f"case {case_id} boundary evidence"
        )
        review_path = _safe_input_path(case["review"], f"case {case_id} review")
        normalized_boundary_path = Path(boundary_path).resolve(strict=False)
        normalized_review_path = Path(review_path).resolve(strict=False)
        if (
            normalized_boundary_path in boundary_paths
            or normalized_review_path in review_paths
        ):
            raise MatteVisualQualificationError(
                "boundary and review sidecars cannot be reused across cases"
            )
        boundary_paths.add(normalized_boundary_path)
        review_paths.add(normalized_review_path)
        cases.append(
            {
                "id": case_id,
                "candidate_id": candidate_id,
                "baseline": baseline,
                "baseline_expected": _normalized_baseline_expected(
                    case["baseline_expected"], f"case {case_id} baseline expected"
                ),
                "candidate": candidate,
                "expected_effective": _normalized_expected_effective(
                    case["expected_effective"], f"case {case_id} expected effective"
                ),
                "coverage": coverage,
                "boundary_evidence": boundary_path,
                "review": review_path,
            }
        )

    referenced_candidates = {str(case["candidate_id"]) for case in cases}
    if referenced_candidates != set(candidate_ids):
        raise MatteVisualQualificationError(
            "every declared visual candidate must be exercised by a case"
        )

    return (
        {
            "schema": PLAN_SCHEMA,
            "version": PLAN_VERSION,
            "provenance": {
                "qualification_id": qualification_id,
                "kind": kind,
                "license_or_consent_reference": reference,
                "contains_private_footage_in_repository": False,
            },
            "policy": {
                "quality_policy_id": policy["quality_policy_id"],
                "maximum_full_frame_mae": full_mae,
                "maximum_edge_band_mae": edge_mae,
            },
            "candidates": candidates,
            "cases": cases,
        },
        _sha256(payload),
    )


def _artifact_digest(artifacts: Mapping[str, object], name: str) -> str | None:
    descriptor = _artifact_descriptor(artifacts, name)
    if descriptor is None:
        return None
    digest = descriptor.get("sha256")
    return str(digest) if isinstance(digest, str) else None


def _artifact_descriptor(
    artifacts: Mapping[str, object], name: str
) -> Mapping[str, object] | None:
    descriptor = artifacts.get(name)
    if not isinstance(descriptor, Mapping):
        return None
    alias = descriptor.get("alias_of")
    if isinstance(alias, str):
        descriptor = cast(Mapping[str, object], artifacts.get(alias, {}))
    return descriptor


def _source_contract(bundle: MatteReplayBundle) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for frame in bundle.frames:
        artifacts = cast(dict[str, Any], frame["artifacts"])
        result.append(
            {
                "capture_sequence": frame["capture_sequence"],
                "capture_monotonic_ns": frame["capture_monotonic_ns"],
                "timestamp_source": frame["timestamp_source"],
                "capture_generation": frame["capture_generation"],
                "geometry_generation": frame["geometry_generation"],
                "raw_frame_sha256": _artifact_digest(artifacts, "raw_frame"),
                "backdrop_frame_sha256": _artifact_digest(artifacts, "backdrop_frame"),
                "backdrop_identity": frame.get("backdrop_identity", {}),
            }
        )
    return result


def _annotation_descriptor(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "bytes": value.get("bytes"),
        "sha256": value.get("sha256"),
        "dtype": value.get("dtype"),
        "shape": value.get("shape"),
    }


def _annotation_contract(annotations: MatteQualityAnnotations) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for frame in annotations.frames:
        artifacts = cast(dict[str, Any], frame["artifacts"])
        frames.append(
            {
                "sequence": frame["sequence"],
                "segment": frame["segment"],
                "registration_from_previous": frame["registration_from_previous"],
                "artifacts": {
                    name: _annotation_descriptor(descriptor)
                    for name, descriptor in sorted(artifacts.items())
                },
                "regions": [
                    {
                        "name": region["name"],
                        "kind": region["kind"],
                        "artifact": _annotation_descriptor(region["artifact"]),
                    }
                    for region in cast(list[dict[str, Any]], frame.get("regions", []))
                ],
            }
        )
    return {
        "segments": list(annotations.segments),
        "frames": frames,
    }


def _assert_annotation_provenance(
    baseline: MatteQualityAnnotations,
    candidate: MatteQualityAnnotations,
    *,
    case_id: str,
    plan_kind: str,
    plan_reference: str,
) -> str:
    baseline_provenance = dict(baseline.provenance)
    candidate_provenance = dict(candidate.provenance)
    if baseline_provenance != candidate_provenance:
        raise MatteVisualQualificationError(
            f"case {case_id} baseline and candidate annotation provenance differ"
        )
    if baseline_provenance.get("kind") != plan_kind:
        raise MatteVisualQualificationError(
            f"case {case_id} annotation provenance differs from the plan"
        )
    license_reference = baseline_provenance.get("license")
    if not isinstance(license_reference, str):
        raise MatteVisualQualificationError(
            f"case {case_id} annotation license or consent reference is invalid"
        )
    if plan_kind != "generated" and (
        not license_reference.strip() or license_reference != plan_reference
    ):
        raise MatteVisualQualificationError(
            f"case {case_id} annotations are not bound to the plan consent or license"
        )
    return _sha256(_json_bytes(baseline_provenance))


_MOTION_SEGMENT_KINDS: dict[str, frozenset[str]] = {
    "stationary": frozenset(("stationary",)),
    "speech_micro_motion": frozenset(("moving",)),
    "slow_turn": frozenset(("moving",)),
    "fast_turn": frozenset(("fast_motion",)),
    "hand_or_prop_crossing_face": frozenset(("occlusion",)),
    "entering_frame": frozenset(("moving", "fast_motion")),
    "leaving_frame": frozenset(("moving", "fast_motion")),
}


def _assert_motion_coverage(
    annotations: MatteQualityAnnotations,
    claimed: Sequence[str],
    *,
    case_id: str,
) -> None:
    observed_kinds = {str(segment["kind"]) for segment in annotations.segments}
    unsupported = [
        motion
        for motion in claimed
        if observed_kinds.isdisjoint(_MOTION_SEGMENT_KINDS[motion])
    ]
    if unsupported:
        raise MatteVisualQualificationError(
            f"case {case_id} claimed motions lack matching annotation segments: "
            + ", ".join(unsupported)
        )


def _assert_reactions_disabled(bundle: MatteReplayBundle, label: str) -> None:
    extension_points = bundle.manifest.get("extension_points")
    post_base_extension = (
        extension_points.get("post_base_final_output_provenance")
        if isinstance(extension_points, Mapping)
        else None
    )
    if (
        not isinstance(post_base_extension, Mapping)
        or set(post_base_extension) != {"version", "present"}
        or type(post_base_extension.get("version")) is not int
        or post_base_extension.get("version") != 1
        or post_base_extension.get("present") is not False
    ):
        raise MatteVisualQualificationError(
            f"{label} does not attest a disabled post-base extension"
        )
    for event in bundle.output_events:
        if event.get("post_base_final_output_provenance") is not None:
            raise MatteVisualQualificationError(
                f"{label} contains post-base/reaction provenance"
            )
    for frame in bundle.frames:
        if frame.get("post_base_final_output_provenance") is not None:
            raise MatteVisualQualificationError(
                f"{label} contains frame-level post-base/reaction provenance"
            )
        artifacts = frame.get("artifacts")
        base_digest = (
            _artifact_digest(artifacts, "base_composite")
            if isinstance(artifacts, Mapping)
            else None
        )
        final_digest = (
            _artifact_digest(artifacts, "final_composite")
            if isinstance(artifacts, Mapping)
            else None
        )
        if base_digest is None or base_digest != final_digest:
            raise MatteVisualQualificationError(
                f"{label} does not prove base and final composites are identical"
            )
        base_composite = bundle.load_array(frame, "base_composite")
        final_composite = bundle.load_array(frame, "final_composite")
        if not np.array_equal(base_composite, final_composite):
            raise MatteVisualQualificationError(
                f"{label} base and final composite arrays differ"
            )
        configured = frame.get("configured_controls")
        if not isinstance(configured, Mapping):
            raise MatteVisualQualificationError(
                f"{label} configured controls are missing"
            )
        for key in ("reactions", "reaction"):
            value = configured.get(key)
            if isinstance(value, Mapping) and value.get("enabled") is not False:
                raise MatteVisualQualificationError(
                    f"{label} does not prove reactions were disabled"
                )
            if value is not None and not isinstance(value, Mapping):
                raise MatteVisualQualificationError(
                    f"{label} reaction controls are malformed"
                )


_BACKGROUND_COVERAGE = {
    "image": "static_image",
    "video": "dynamic_video",
    "blur": "blur",
    "color": "solid_color",
    "camera": "live_camera",
}

_BACKEND_KINDS = {
    "RVMSegmenter": MatteBackendKind.TRUE_ALPHA_RECURRENT,
    "MediaPipeSegmenter": MatteBackendKind.CONFIDENCE_MASK_VIDEO,
    "HeuristicSegmenter": MatteBackendKind.BINARY_COARSE,
    "NullSegmenter": MatteBackendKind.NULL_PASSTHROUGH,
}

_CONFIGURED_BACKENDS = {
    "rvm": frozenset(("RVMSegmenter",)),
    "mediapipe": frozenset(("MediaPipeSegmenter",)),
    "heuristic": frozenset(("HeuristicSegmenter",)),
    "none": frozenset(("NullSegmenter",)),
    "auto": frozenset(("RVMSegmenter", "MediaPipeSegmenter", "HeuristicSegmenter")),
}


def _configured_case_contract(
    bundle: MatteReplayBundle,
    candidate: Mapping[str, Any],
    expected_effective: Mapping[str, Any],
    *,
    case_id: str,
) -> tuple[list[str], list[str]]:
    candidate_segmentation = SegmentationConfig.model_validate(
        candidate["segmentation"]
    )
    candidate_compositing = CompositingConfig.model_validate(candidate["compositing"])
    expected_backend = str(expected_effective["segmentation_backend"])
    backend_kind = _BACKEND_KINDS.get(expected_backend)
    allowed_backends = _CONFIGURED_BACKENDS[candidate_segmentation.backend]
    if backend_kind is None or expected_backend not in allowed_backends:
        raise MatteVisualQualificationError(
            f"case {case_id} effective backend is incompatible with its candidate"
        )
    expected_ratio = expected_effective["rvm_downsample_ratio"]
    if (backend_kind is MatteBackendKind.TRUE_ALPHA_RECURRENT) != (
        expected_ratio is not None
    ):
        raise MatteVisualQualificationError(
            f"case {case_id} effective RVM ratio applicability is invalid"
        )
    canvases: set[str] = set()
    backgrounds: set[str] = set()
    for sequence, frame in enumerate(bundle.frames):
        artifacts = cast(dict[str, Any], frame["artifacts"])
        final_descriptor = _artifact_descriptor(artifacts, "final_composite")
        if final_descriptor is None:
            raise MatteVisualQualificationError(
                f"case {case_id} frame {sequence} final artifact is missing"
            )
        shape = final_descriptor.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(item) is not int for item in shape)
            or shape[2] != 3
        ):
            raise MatteVisualQualificationError(
                f"case {case_id} frame {sequence} final shape is invalid"
            )
        canvases.add(f"{shape[1]}x{shape[0]}")
        configured = frame.get("configured_controls")
        if not isinstance(configured, Mapping):
            raise MatteVisualQualificationError(
                f"case {case_id} frame {sequence} configured controls are missing"
            )
        try:
            recorded_segmentation = SegmentationConfig.model_validate(
                configured.get("segmentation")
            ).model_dump(mode="json")
            recorded_compositing = CompositingConfig.model_validate(
                configured.get("compositing")
            ).model_dump(mode="json")
        except ValueError as exc:
            raise MatteVisualQualificationError(
                f"case {case_id} frame {sequence} matte controls are invalid"
            ) from exc
        if recorded_segmentation != candidate["segmentation"]:
            raise MatteVisualQualificationError(
                f"case {case_id} configured segmentation differs from its candidate"
            )
        if recorded_compositing != candidate["compositing"]:
            raise MatteVisualQualificationError(
                f"case {case_id} configured compositing differs from its candidate"
            )
        background = configured.get("background")
        if not isinstance(background, Mapping):
            raise MatteVisualQualificationError(
                f"case {case_id} frame {sequence} background controls are missing"
            )
        mode = background.get("mode")
        if mode not in _BACKGROUND_COVERAGE:
            raise MatteVisualQualificationError(
                f"case {case_id} background is outside the qualification set"
            )
        backgrounds.add(_BACKGROUND_COVERAGE[str(mode)])

        try:
            expected_policy = resolve_matte_policy(
                candidate_segmentation,
                candidate_compositing,
                backend_kind,
                resolved_rvm_ratio=expected_effective["rvm_downsample_ratio"],
                passthrough=False,
                canvas_shape=(int(shape[0]), int(shape[1])),
                experimental_rvm_generic=bool(
                    expected_effective["matte_policy"]["experimental_rvm_generic"]
                ),
                light_wrap_stabilization_eligible=mode in {"video", "camera"},
            ).to_dict()
        except (TypeError, ValueError) as exc:
            raise MatteVisualQualificationError(
                f"case {case_id} matte policy cannot be resolved"
            ) from exc
        if expected_effective["matte_policy"] != expected_policy:
            raise MatteVisualQualificationError(
                f"case {case_id} planned matte policy is not the code-resolved policy"
            )

        effective = frame.get("effective_controls")
        if not isinstance(effective, Mapping):
            raise MatteVisualQualificationError(
                f"case {case_id} frame {sequence} effective controls are missing"
            )
        for name in ("segmentation_backend", "segmentation_device"):
            if effective.get(name) != expected_effective[name]:
                raise MatteVisualQualificationError(
                    f"case {case_id} effective {name} differs from the plan"
                )
        actual_ratio = effective.get("rvm_downsample_ratio")
        expected_ratio = expected_effective["rvm_downsample_ratio"]
        if expected_ratio is None:
            if actual_ratio is not None:
                raise MatteVisualQualificationError(
                    f"case {case_id} unexpectedly reports an RVM ratio"
                )
        elif (
            isinstance(actual_ratio, bool)
            or not isinstance(actual_ratio, (int, float))
            or not math.isclose(
                float(actual_ratio), float(expected_ratio), rel_tol=0.0, abs_tol=1e-6
            )
        ):
            raise MatteVisualQualificationError(
                f"case {case_id} effective RVM ratio differs from the plan"
            )
        if effective.get("matte_policy") != expected_effective["matte_policy"]:
            raise MatteVisualQualificationError(
                f"case {case_id} effective matte policy differs from the plan"
            )
    if len(canvases) != 1 or len(backgrounds) != 1:
        raise MatteVisualQualificationError(
            f"case {case_id} changes canvas or background inside one comparison"
        )
    canvas = next(iter(canvases))
    return (
        [canvas] if canvas in REQUIRED_COVERAGE["canvases"] else [],
        [next(iter(backgrounds))],
    )


def _derived_cadences(
    bundle: MatteReplayBundle,
    case_id: str,
    *,
    authoritative: bool,
) -> list[str]:
    if len(bundle.frames) < 2:
        raise MatteVisualQualificationError(
            f"case {case_id} requires at least two unique input frames"
        )
    if authoritative and any(
        frame.get("timestamp_source") != "capture-completion" for frame in bundle.frames
    ):
        raise MatteVisualQualificationError(
            f"case {case_id} local cadence lacks capture-completion timestamps"
        )
    timestamps = np.asarray(
        [int(frame["capture_monotonic_ns"]) for frame in bundle.frames],
        dtype=np.int64,
    )
    intervals = np.diff(timestamps).astype(np.float64) / 1_000_000_000.0
    if bool(np.any(intervals <= 0.0)):
        raise MatteVisualQualificationError(
            f"case {case_id} capture timestamps are not strictly increasing"
        )
    median = float(np.median(intervals))
    fps = 1.0 / median
    cadence: list[str] = []
    closest = min((15, 30, 60), key=lambda target: abs(fps - target))
    if abs(fps - closest) / closest <= 0.10:
        cadence.append(f"fps_{closest}")
    else:
        raise MatteVisualQualificationError(
            f"case {case_id} has no 15/30/60 median capture cadence"
        )
    if float(np.max(np.abs(intervals - median))) > max(0.001, median * 0.10):
        cadence.append("irregular")
    sequences = [int(frame["capture_sequence"]) for frame in bundle.frames]
    if authoritative and any(
        current <= previous for previous, current in zip(sequences, sequences[1:])
    ):
        raise MatteVisualQualificationError(
            f"case {case_id} local capture sequence is not strictly increasing"
        )
    if any(
        current - previous > 1 for previous, current in zip(sequences, sequences[1:])
    ):
        cadence.append("drops")
    generations = [
        (int(frame["capture_generation"]), int(frame["geometry_generation"]))
        for frame in bundle.frames
    ]
    if any(
        current != previous for previous, current in zip(generations, generations[1:])
    ):
        cadence.append("restart")
    return sorted(cadence)


_GENERATED_CAPTURE_METHODS = dict(
    zip(
        REQUIRED_BOUNDARIES,
        (
            "generated-memory-tap",
            "generated-highgui-pre-overlay",
            "generated-http-snapshot",
            "generated-mjpeg-part",
            "generated-output-websocket",
            "generated-pyvirtualcam-double",
            "generated-native-ring-double",
        ),
    )
)
_LOCAL_CAPTURE_METHODS = dict(
    zip(
        REQUIRED_BOUNDARIES,
        (
            "pipeline-memory-tap",
            "highgui-pre-overlay-tap",
            "authenticated-http-snapshot",
            "authenticated-mjpeg-part",
            "authenticated-output-websocket",
            "pyvirtualcam-consumer-recording",
            "windows-native-consumer-recording",
        ),
    )
)
_JPEG_BOUNDARIES = frozenset({"snapshot_jpeg", "mjpeg_jpeg", "websocket_jpeg"})


def _safe_artifact_filename(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise MatteVisualQualificationError(f"{label} filename is malformed")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise MatteVisualQualificationError(
            f"{label} filename must be one safe relative component"
        )
    return value


def _png_dimensions(payload: bytes, label: str) -> tuple[int, int]:
    if (
        len(payload) < 33
        or payload[:8] != b"\x89PNG\r\n\x1a\n"
        or payload[12:16] != b"IHDR"
    ):
        raise MatteVisualQualificationError(f"{label} is not a PNG")
    return int.from_bytes(payload[16:20], "big"), int.from_bytes(payload[20:24], "big")


def _jpeg_dimensions(payload: bytes, label: str) -> tuple[int, int]:
    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        raise MatteVisualQualificationError(f"{label} is not a JPEG")
    index = 2
    while index + 4 <= len(payload):
        if payload[index] != 0xFF:
            index += 1
            continue
        while index < len(payload) and payload[index] == 0xFF:
            index += 1
        if index >= len(payload):
            break
        marker = payload[index]
        index += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(payload):
            break
        length = int.from_bytes(payload[index : index + 2], "big")
        if length < 2 or index + length > len(payload):
            break
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if length < 7:
                break
            height = int.from_bytes(payload[index + 3 : index + 5], "big")
            width = int.from_bytes(payload[index + 5 : index + 7], "big")
            return width, height
        index += length
    raise MatteVisualQualificationError(f"{label} has no valid JPEG dimensions")


def _decode_artifact(
    root: Path,
    descriptor_value: object,
    *,
    boundary: str,
    label: str,
    seen_paths: set[Path],
) -> tuple[np.ndarray, dict[str, object]]:
    descriptor = _strict_keys(
        descriptor_value,
        ("filename", "sha256", "bytes", "media_type"),
        f"{label} artifact",
    )
    filename = _safe_artifact_filename(descriptor["filename"], f"{label} artifact")
    expected_digest = _digest(descriptor["sha256"], f"{label} artifact digest")
    byte_count = descriptor["bytes"]
    if type(byte_count) is not int or not 1 <= byte_count <= MAX_ARTIFACT_BYTES:
        raise MatteVisualQualificationError(f"{label} artifact byte count is invalid")
    expected_media = "image/jpeg" if boundary in _JPEG_BOUNDARIES else "image/png"
    if descriptor["media_type"] != expected_media:
        raise MatteVisualQualificationError(
            f"{label} artifact must use {expected_media}"
        )
    suffix = ".jpg" if expected_media == "image/jpeg" else ".png"
    if Path(filename).suffix.lower() not in (
        (".jpg", ".jpeg") if suffix == ".jpg" else (suffix,)
    ):
        raise MatteVisualQualificationError(f"{label} artifact extension is invalid")
    path = root / filename
    resolved = path.resolve(strict=False)
    if resolved in seen_paths:
        raise MatteVisualQualificationError("boundary artifact path is reused")
    seen_paths.add(resolved)
    payload = _read_private_file(path, max_bytes=MAX_ARTIFACT_BYTES)
    if len(payload) != byte_count or _sha256(payload) != expected_digest:
        raise MatteVisualQualificationError(
            f"{label} artifact size or digest differs from its descriptor"
        )
    width, height = (
        _jpeg_dimensions(payload, label)
        if expected_media == "image/jpeg"
        else _png_dimensions(payload, label)
    )
    if (
        width <= 0
        or height <= 0
        or width > MAX_IMAGE_DIMENSION
        or height > MAX_IMAGE_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise MatteVisualQualificationError(f"{label} dimensions exceed the bound")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if (
        decoded is None
        or decoded.dtype != np.uint8
        or decoded.shape != (height, width, 3)
    ):
        raise MatteVisualQualificationError(f"{label} artifact could not be decoded")
    return np.ascontiguousarray(decoded), {
        "sha256": expected_digest,
        "bytes": byte_count,
        "media_type": expected_media,
    }


def _edge_band(alpha: np.ndarray) -> np.ndarray:
    uncertain = (alpha > 0.05) & (alpha < 0.95)
    binary = (alpha >= 0.5).astype(np.uint8)
    contour = cv2.morphologyEx(
        binary,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    band = uncertain | contour
    if not bool(np.any(band)):
        raise MatteVisualQualificationError(
            "boundary comparison reference has no measurable edge band"
        )
    return band


def _load_boundary_evidence(
    path_value: str,
    *,
    case_id: str,
    plan_kind: str,
    bundle: MatteReplayBundle,
    policy: Mapping[str, Any],
    derived_cadences: Sequence[str],
    seen_artifact_paths: set[Path],
) -> tuple[dict[str, Any], str]:
    path = Path(path_value)
    value, payload = _read_json(path, label=f"case {case_id} boundary evidence")
    evidence = _strict_keys(
        value,
        ("schema", "version", "case_id", "authority", "source", "artifacts"),
        f"case {case_id} boundary evidence",
    )
    if (
        evidence["schema"] != BOUNDARY_SCHEMA
        or type(evidence["version"]) is not int
        or evidence["version"] != BOUNDARY_VERSION
        or evidence["case_id"] != case_id
    ):
        raise MatteVisualQualificationError(
            f"case {case_id} boundary evidence header is invalid"
        )
    expected_authority = (
        "generated-fake" if plan_kind == "generated" else "local-observed"
    )
    if evidence["authority"] != expected_authority:
        raise MatteVisualQualificationError(
            f"case {case_id} boundary authority differs from plan provenance"
        )
    source = _strict_keys(
        evidence["source"],
        (
            "bundle_manifest_sha256",
            "bundle_sequence",
            "capture_sequence",
            "capture_generation",
            "geometry_generation",
            "reference_final_composite_sha256",
            "identity_region",
        ),
        f"case {case_id} boundary source",
    )
    if source["bundle_manifest_sha256"] != bundle.manifest_sha256:
        raise MatteVisualQualificationError(
            f"case {case_id} boundary source bundle digest differs"
        )
    sequence = source["bundle_sequence"]
    if type(sequence) is not int or not 0 <= sequence < len(bundle.frames):
        raise MatteVisualQualificationError(
            f"case {case_id} boundary bundle sequence is invalid"
        )
    frame = bundle.frames[sequence]
    for name in ("capture_sequence", "capture_generation", "geometry_generation"):
        if (
            type(source[name]) is not int
            or source[name] < 0
            or source[name] != frame[name]
        ):
            raise MatteVisualQualificationError(
                f"case {case_id} boundary {name} differs from the replay frame"
            )
    frame_artifacts = cast(dict[str, Any], frame["artifacts"])
    reference_digest = _artifact_digest(frame_artifacts, "final_composite")
    if (
        source["reference_final_composite_sha256"] != reference_digest
        or _digest(
            source["reference_final_composite_sha256"],
            f"case {case_id} final-composite digest",
        )
        != reference_digest
    ):
        raise MatteVisualQualificationError(
            f"case {case_id} boundary reference digest differs"
        )
    reference = bundle.load_array(frame, "final_composite")
    identity_value = _strict_keys(
        source["identity_region"],
        ("x", "y", "width", "height", "reference_region_sha256"),
        f"case {case_id} boundary identity region",
    )
    identity_coordinates: dict[str, int] = {}
    for name in ("x", "y", "width", "height"):
        coordinate = identity_value[name]
        minimum = 0 if name in ("x", "y") else 8
        if type(coordinate) is not int or coordinate < minimum:
            raise MatteVisualQualificationError(
                f"case {case_id} boundary identity region {name} is invalid"
            )
        identity_coordinates[name] = coordinate
    identity_x = identity_coordinates["x"]
    identity_y = identity_coordinates["y"]
    identity_width = identity_coordinates["width"]
    identity_height = identity_coordinates["height"]
    if (
        identity_x + identity_width > reference.shape[1]
        or identity_y + identity_height > reference.shape[0]
    ):
        raise MatteVisualQualificationError(
            f"case {case_id} boundary identity region exceeds the reference frame"
        )
    reference_identity_region = np.ascontiguousarray(
        reference[
            identity_y : identity_y + identity_height,
            identity_x : identity_x + identity_width,
        ]
    )
    alpha = bundle.load_array(frame, "refined_mask").astype(np.float32, copy=False)
    band = _edge_band(alpha)
    identity_band = band[
        identity_y : identity_y + identity_height,
        identity_x : identity_x + identity_width,
    ]
    if bool(np.any(identity_band)):
        raise MatteVisualQualificationError(
            f"case {case_id} boundary identity region overlaps the matte edge band"
        )
    identity_luma = cv2.cvtColor(reference_identity_region, cv2.COLOR_BGR2GRAY)
    if float(np.std(identity_luma, dtype=np.float64)) < 4.0:
        raise MatteVisualQualificationError(
            f"case {case_id} boundary identity region lacks contrast"
        )
    identity_digest = _sha256(reference_identity_region.tobytes(order="C"))
    if (
        _digest(
            identity_value["reference_region_sha256"],
            f"case {case_id} boundary identity-region digest",
        )
        != identity_digest
    ):
        raise MatteVisualQualificationError(
            f"case {case_id} boundary identity region differs from the replay frame"
        )
    other_frame_maes: list[float] = []
    for other_sequence, other_frame in enumerate(bundle.frames):
        if other_sequence == sequence:
            continue
        other_composite = bundle.load_array(other_frame, "final_composite")
        if np.array_equal(other_composite, reference):
            continue
        other_region = other_composite[
            identity_y : identity_y + identity_height,
            identity_x : identity_x + identity_width,
        ]
        region_mae = float(
            np.mean(
                np.abs(
                    other_region.astype(np.int16)
                    - reference_identity_region.astype(np.int16)
                ),
                dtype=np.float64,
            )
        )
        other_frame_maes.append(region_mae)
        if region_mae <= policy["maximum_full_frame_mae"]:
            raise MatteVisualQualificationError(
                f"case {case_id} boundary identity region cannot distinguish "
                f"replay frame {sequence} from frame {other_sequence}"
            )

    artifacts = evidence["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED_BOUNDARIES):
        raise MatteVisualQualificationError(
            f"case {case_id} boundary set is incomplete"
        )
    results: dict[str, Any] = {}
    for boundary in REQUIRED_BOUNDARIES:
        entry = _strict_keys(
            artifacts[boundary],
            ("status", "artifact", "reason", "capture_method", "platform"),
            f"case {case_id} boundary {boundary}",
        )
        status = entry["status"]
        platform = entry["platform"]
        if platform not in ("generated", "linux", "macos", "windows"):
            raise MatteVisualQualificationError(
                f"case {case_id} boundary {boundary} platform is invalid"
            )
        if expected_authority == "generated-fake" and platform != "generated":
            raise MatteVisualQualificationError(
                f"case {case_id} generated boundary claims a real platform"
            )
        if expected_authority == "local-observed" and platform == "generated":
            raise MatteVisualQualificationError(
                f"case {case_id} local boundary claims generated evidence"
            )
        if boundary == "windows_native_loopback" and platform != (
            "generated" if expected_authority == "generated-fake" else "windows"
        ):
            raise MatteVisualQualificationError(
                f"case {case_id} native boundary has the wrong platform"
            )
        if status == "not_applicable":
            if (
                entry["artifact"] is not None
                or entry["capture_method"] != "not-applicable"
            ):
                raise MatteVisualQualificationError(
                    f"case {case_id} inapplicable boundary carries an artifact"
                )
            reason = _safe_id(
                entry["reason"], f"case {case_id} boundary {boundary} reason"
            )
            results[boundary] = {
                "status": "not_applicable",
                "reason": reason,
                "capture_method": "not-applicable",
                "platform": platform,
                "comparison": None,
            }
            continue
        if status != "captured" or entry["reason"] != "":
            raise MatteVisualQualificationError(
                f"case {case_id} boundary {boundary} status is invalid"
            )
        expected_method = (
            _GENERATED_CAPTURE_METHODS[boundary]
            if expected_authority == "generated-fake"
            else _LOCAL_CAPTURE_METHODS[boundary]
        )
        if entry["capture_method"] != expected_method:
            raise MatteVisualQualificationError(
                f"case {case_id} boundary {boundary} capture method is invalid"
            )
        if (
            boundary == "windows_native_loopback"
            and expected_authority == "local-observed"
        ):
            canvas = f"{reference.shape[1]}x{reference.shape[0]}"
            if (
                canvas not in ("1280x720", "1920x1080")
                or "fps_30" not in derived_cadences
            ):
                raise MatteVisualQualificationError(
                    f"case {case_id} native capture uses an unsupported mode"
                )
        decoded, descriptor = _decode_artifact(
            path.parent,
            entry["artifact"],
            boundary=boundary,
            label=f"case {case_id} boundary {boundary}",
            seen_paths=seen_artifact_paths,
        )
        if decoded.shape != reference.shape:
            raise MatteVisualQualificationError(
                f"case {case_id} boundary {boundary} introduced a resize"
            )
        difference = np.abs(decoded.astype(np.int16) - reference.astype(np.int16))
        full_mae = float(np.mean(difference, dtype=np.float64))
        edge_mae = float(np.mean(difference[band], dtype=np.float64))
        identity_mae = float(
            np.mean(
                difference[
                    identity_y : identity_y + identity_height,
                    identity_x : identity_x + identity_width,
                ],
                dtype=np.float64,
            )
        )
        exact_required = boundary in ("in_memory", "highgui_pre_overlay") or (
            expected_authority == "generated-fake"
            and boundary in ("pyvirtualcam_loopback", "windows_native_loopback")
        )
        passed = (
            full_mae == 0.0 and edge_mae == 0.0 and identity_mae == 0.0
            if exact_required
            else full_mae <= policy["maximum_full_frame_mae"]
            and edge_mae <= policy["maximum_edge_band_mae"]
            and identity_mae <= policy["maximum_full_frame_mae"]
        )
        results[boundary] = {
            "status": "passed" if passed else "failed",
            "reason": "",
            "capture_method": expected_method,
            "platform": platform,
            "artifact": descriptor,
            "comparison": {
                "width": int(decoded.shape[1]),
                "height": int(decoded.shape[0]),
                "full_frame_mae": round(full_mae, 8),
                "edge_band_mae": round(edge_mae, 8),
                "identity_region_mae": round(identity_mae, 8),
                "extra_resize": False,
                "exact_required": exact_required,
            },
        }
    return (
        {
            "authority": expected_authority,
            "source": {
                "bundle_manifest_sha256": bundle.manifest_sha256,
                "bundle_sequence": sequence,
                "capture_sequence": frame["capture_sequence"],
                "capture_generation": frame["capture_generation"],
                "geometry_generation": frame["geometry_generation"],
                "reference_final_composite_sha256": reference_digest,
                "identity_region": {
                    **identity_coordinates,
                    "reference_region_sha256": identity_digest,
                    "visually_distinct_other_frame_count": len(other_frame_maes),
                    "minimum_other_frame_mae": (
                        None
                        if not other_frame_maes
                        else round(min(other_frame_maes), 8)
                    ),
                },
            },
            "boundaries": results,
        },
        _sha256(payload),
    )


def _load_review(
    path_value: str,
    *,
    case_id: str,
    plan_kind: str,
    plan_reference: str,
    plan_sha256: str,
    baseline_manifest_sha256: str,
    candidate_manifest_sha256: str,
    baseline_annotation_manifest_sha256: str,
    candidate_annotation_manifest_sha256: str,
    baseline_quality_evidence_sha256: str,
    candidate_quality_evidence_sha256: str,
    boundary_evidence_sha256: str,
) -> dict[str, Any]:
    value, payload = _read_json(path_value, label=f"case {case_id} human review")
    review = _strict_keys(
        value,
        (
            "schema",
            "version",
            "case_id",
            "provenance",
            "bindings",
            "method",
            "blinding",
            "reviewer",
            "reviewed_at",
            "concerns",
            "overall",
            "notes",
        ),
        f"case {case_id} human review",
    )
    if (
        review["schema"] != REVIEW_SCHEMA
        or type(review["version"]) is not int
        or review["version"] != REVIEW_VERSION
        or review["case_id"] != case_id
    ):
        raise MatteVisualQualificationError(f"case {case_id} review header is invalid")
    provenance = _strict_keys(
        review["provenance"],
        ("kind", "reference"),
        f"case {case_id} review provenance",
    )
    if provenance["kind"] != plan_kind:
        raise MatteVisualQualificationError(
            f"case {case_id} review provenance differs from the plan"
        )
    reference = provenance["reference"]
    if not isinstance(reference, str) or len(reference) > 512:
        raise MatteVisualQualificationError(
            f"case {case_id} review provenance reference is invalid"
        )
    if plan_kind != "generated" and not reference.strip():
        raise MatteVisualQualificationError(
            f"case {case_id} local review lacks a provenance reference"
        )
    if plan_kind != "generated" and reference != plan_reference:
        raise MatteVisualQualificationError(
            f"case {case_id} review is not bound to the plan consent or license"
        )
    bindings = _strict_keys(
        review["bindings"],
        (
            "plan_sha256",
            "baseline_bundle_manifest_sha256",
            "candidate_bundle_manifest_sha256",
            "baseline_annotation_manifest_sha256",
            "candidate_annotation_manifest_sha256",
            "baseline_quality_evidence_sha256",
            "candidate_quality_evidence_sha256",
            "boundary_evidence_sha256",
        ),
        f"case {case_id} review bindings",
    )
    expected_bindings = {
        "plan_sha256": plan_sha256,
        "baseline_bundle_manifest_sha256": baseline_manifest_sha256,
        "candidate_bundle_manifest_sha256": candidate_manifest_sha256,
        "baseline_annotation_manifest_sha256": (baseline_annotation_manifest_sha256),
        "candidate_annotation_manifest_sha256": (candidate_annotation_manifest_sha256),
        "baseline_quality_evidence_sha256": baseline_quality_evidence_sha256,
        "candidate_quality_evidence_sha256": candidate_quality_evidence_sha256,
        "boundary_evidence_sha256": boundary_evidence_sha256,
    }
    for name, expected in expected_bindings.items():
        if _digest(bindings[name], f"case {case_id} review {name}") != expected:
            raise MatteVisualQualificationError(
                f"case {case_id} review {name} differs from its evidence"
            )
    if review["method"] not in ("blinded", "side_by_side"):
        raise MatteVisualQualificationError(f"case {case_id} review method is invalid")
    blinding = review["blinding"]
    if review["method"] == "side_by_side":
        if blinding is not None:
            raise MatteVisualQualificationError(
                f"case {case_id} side-by-side review cannot claim blinding"
            )
        normalized_blinding = None
    else:
        blinded = _strict_keys(
            blinding,
            ("assignment_sha256", "reveal_sha256", "revealed_after_decisions"),
            f"case {case_id} blinding evidence",
        )
        assignment = _digest(
            blinded["assignment_sha256"], f"case {case_id} blind assignment"
        )
        reveal = _digest(blinded["reveal_sha256"], f"case {case_id} blind reveal")
        if assignment == reveal or blinded["revealed_after_decisions"] is not True:
            raise MatteVisualQualificationError(
                f"case {case_id} blinding evidence is invalid"
            )
        normalized_blinding = {
            "assignment_sha256": assignment,
            "reveal_sha256": reveal,
            "revealed_after_decisions": True,
        }
    reviewer = _safe_label(review["reviewer"], f"case {case_id} reviewer")
    reviewed_at_value = review["reviewed_at"]
    if (
        not isinstance(reviewed_at_value, str)
        or _UTC_TIMESTAMP.fullmatch(reviewed_at_value) is None
    ):
        raise MatteVisualQualificationError(
            f"case {case_id} review time must be ISO-8601 UTC"
        )
    try:
        dt.datetime.fromisoformat(reviewed_at_value[:-1] + "+00:00")
    except ValueError as exc:
        raise MatteVisualQualificationError(
            f"case {case_id} review time is invalid"
        ) from exc
    concerns = review["concerns"]
    if not isinstance(concerns, dict) or set(concerns) != set(REQUIRED_REVIEW_CONCERNS):
        raise MatteVisualQualificationError(
            f"case {case_id} review concerns are incomplete"
        )
    if any(value not in ("pass", "fail") for value in concerns.values()):
        raise MatteVisualQualificationError(
            f"case {case_id} review concern status is invalid"
        )
    overall = review["overall"]
    if overall not in (
        "candidate_preferred",
        "candidate_acceptable",
        "candidate_worse",
    ):
        raise MatteVisualQualificationError(f"case {case_id} review outcome is invalid")
    notes = review["notes"]
    if not isinstance(notes, str) or len(notes) > 4096:
        raise MatteVisualQualificationError(f"case {case_id} review notes are invalid")
    passed = all(value == "pass" for value in concerns.values()) and overall in (
        "candidate_preferred",
        "candidate_acceptable",
    )
    return {
        "method": review["method"],
        "blinding": normalized_blinding,
        "reviewer_present": bool(reviewer),
        "reviewed_at_present": True,
        "concerns": dict(sorted(cast(dict[str, str], concerns).items())),
        "overall": overall,
        "status": "passed" if passed else "failed",
        "evidence_sha256": _sha256(payload),
    }


_ABSOLUTE_GATES: tuple[tuple[str, str, str, float], ...] = (
    (
        "opaque_core_p05",
        "aggregate.metrics.opaque_core_alpha_p05.p05",
        ">=",
        0.95,
    ),
    (
        "opaque_core_below_095",
        "aggregate.metrics.opaque_core_fraction_below_0_95.p95",
        "<=",
        0.05,
    ),
    (
        "foreground_holes",
        "aggregate.metrics.foreground_hole_components.max",
        "<=",
        0.0,
    ),
    (
        "background_alpha",
        "aggregate.metrics.background_alpha_mean.p95",
        "<=",
        0.01,
    ),
    (
        "exterior_halo_area",
        "aggregate.metrics.exterior_halo_area_ratio.p95",
        "<=",
        0.05,
    ),
    (
        "exterior_halo_width",
        "aggregate.metrics.exterior_halo_width_p95_px.p95",
        "<=",
        8.0,
    ),
    (
        "ground_truth_mse",
        "aggregate.metrics.ground_truth_alpha_mse.p95",
        "<=",
        0.01,
    ),
    (
        "ground_truth_gradient",
        "aggregate.metrics.ground_truth_gradient_mae.p95",
        "<=",
        0.10,
    ),
    (
        "fine_detail_uncertain_fraction",
        "aggregate.metrics.uncertain_pixel_fraction.p50",
        ">=",
        0.05,
    ),
    (
        "registered_contour",
        "aggregate.metrics.contour_displacement_p95_px.p95",
        "<=",
        1.5,
    ),
    (
        "compensated_alpha",
        "aggregate.metrics.compensated_alpha_temporal_abs_diff.p95",
        "<=",
        0.10,
    ),
    (
        "motion_trail",
        "aggregate.metrics.motion_trail_area_ratio.p95",
        "<=",
        0.10,
    ),
)

# direction, non-regression multiplier, material-improvement multiplier.  A
# lower-is-better value may grow by at most 10%; material improvement is the
# ratified 40% contour / 30% other-error reduction.  Opaque confidence is the
# one larger-is-better family.
_RELATIVE_METRICS: tuple[tuple[str, str, str, float, float], ...] = (
    (
        "registered_contour",
        "aggregate.metrics.contour_displacement_p95_px.p95",
        "lower",
        1.10,
        0.60,
    ),
    (
        "compensated_alpha",
        "aggregate.metrics.compensated_alpha_temporal_abs_diff.p95",
        "lower",
        1.10,
        0.70,
    ),
    (
        "motion_trail",
        "aggregate.metrics.motion_trail_area_ratio.p95",
        "lower",
        1.10,
        0.70,
    ),
    (
        "opaque_core_deficit",
        "aggregate.metrics.opaque_core_mean_deficit.p95",
        "lower",
        1.10,
        0.70,
    ),
    (
        "opaque_core_p05",
        "aggregate.metrics.opaque_core_alpha_p05.p05",
        "higher",
        0.95,
        1.05,
    ),
    (
        "background_alpha",
        "aggregate.metrics.background_alpha_mean.p95",
        "lower",
        1.10,
        0.70,
    ),
    (
        "halo_area",
        "aggregate.metrics.exterior_halo_area_ratio.p95",
        "lower",
        1.10,
        0.70,
    ),
    (
        "ground_truth_mse",
        "aggregate.metrics.ground_truth_alpha_mse.p95",
        "lower",
        1.10,
        0.70,
    ),
    (
        "ground_truth_gradient",
        "aggregate.metrics.ground_truth_gradient_mae.p95",
        "lower",
        1.10,
        0.70,
    ),
    (
        "edge_shimmer",
        "aggregate.metrics.edge_band_rgb_variation.p95",
        "lower",
        1.10,
        0.70,
    ),
)


def _metric_number(report: Mapping[str, object], path: str) -> float | None:
    value = _metric_path(report, path)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _absolute_gate_results(report: Mapping[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for gate_id, metric, operation, threshold in _ABSOLUTE_GATES:
        actual = _metric_number(report, metric)
        if actual is None:
            status = "not_evaluated"
        else:
            passed = actual <= threshold if operation == "<=" else actual >= threshold
            status = "pass" if passed else "fail"
        results.append(
            {
                "id": gate_id,
                "metric": metric,
                "op": operation,
                "threshold": threshold,
                "actual": None if actual is None else round(actual, 8),
                "status": status,
            }
        )
    return results


def _relative_results(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    epsilon = 1e-6
    for metric_id, path, direction, nonregression, material in _RELATIVE_METRICS:
        baseline_value = _metric_number(baseline, path)
        candidate_value = _metric_number(candidate, path)
        if baseline_value is None or candidate_value is None:
            results.append(
                {
                    "id": metric_id,
                    "metric": path,
                    "direction": direction,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "nonregression": "not_evaluated",
                    "material_improvement": False,
                }
            )
            continue
        if direction == "lower":
            allowed = max(epsilon, baseline_value * nonregression + epsilon)
            nonregressed = candidate_value <= allowed
            improved = baseline_value > epsilon and candidate_value <= (
                baseline_value * material
            )
        else:
            allowed = baseline_value * nonregression - epsilon
            nonregressed = candidate_value >= allowed
            improved = baseline_value < 1.0 - epsilon and candidate_value >= min(
                1.0, baseline_value * material + epsilon
            )
        results.append(
            {
                "id": metric_id,
                "metric": path,
                "direction": direction,
                "baseline": round(baseline_value, 8),
                "candidate": round(candidate_value, 8),
                "nonregression": "pass" if nonregressed else "fail",
                "material_improvement": bool(improved),
            }
        )
    return results


def _configured_optional_algorithms(
    candidate: Mapping[str, Any],
) -> dict[str, bool]:
    segmentation = cast(dict[str, Any], candidate["segmentation"])
    compositing = cast(dict[str, Any], candidate["compositing"])
    return {
        "MATTE-2.1": (segmentation["boundary_stabilization"]["mode"] == "motion_aware"),
        "MATTE-2.2": (
            segmentation["edge_refine"] is True
            and segmentation["spatial_edge_refinement"]["mode"] == "stable_guided"
        ),
        "MATTE-2.4": (
            float(compositing["light_wrap"]) > 0.0
            and compositing["light_wrap_stabilization"]["mode"] == "temporal_bounded"
        ),
    }


def _effective_optional_algorithms(case: Mapping[str, Any]) -> dict[str, bool]:
    effective = cast(
        dict[str, Any],
        case["algorithm_contract"]["expected_effective"]["matte_policy"]["effective"],
    )
    return {
        "MATTE-2.1": effective["boundary_stabilization_mode"] == "motion_aware",
        "MATTE-2.2": (
            effective["edge_refine"] is True
            and effective["edge_refinement_mode"] == "stable_guided"
        ),
        "MATTE-2.4": (
            float(effective["light_wrap"]) > 0.0
            and effective["light_wrap_stabilization_mode"] == "temporal_bounded"
        ),
    }


def _algorithm_manifest(
    candidates: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        segmentation = cast(dict[str, Any], candidate["segmentation"])
        compositing = cast(dict[str, Any], candidate["compositing"])
        path_free_segmentation = _path_free_segmentation(segmentation)
        configured = _configured_optional_algorithms(candidate)
        effective_case_ids: dict[str, list[str]] = {
            algorithm: [] for algorithm in configured
        }
        for case in cases:
            if case["candidate_id"] != candidate["id"]:
                continue
            active = _effective_optional_algorithms(case)
            for algorithm, is_active in active.items():
                if is_active:
                    effective_case_ids[algorithm].append(str(case["id"]))
        selected = {
            algorithm: is_configured and bool(effective_case_ids[algorithm])
            for algorithm, is_configured in configured.items()
        }
        ineffective = [
            algorithm
            for algorithm, is_configured in configured.items()
            if is_configured and not selected[algorithm]
        ]
        if ineffective:
            raise MatteVisualQualificationError(
                f"candidate {candidate['id']} configures optional algorithms that "
                "are ineffective in every case: " + ", ".join(ineffective)
            )
        contract = {
            "segmentation": path_free_segmentation,
            "compositing": compositing,
        }
        result.append(
            {
                "id": candidate["id"],
                **contract,
                "contract_sha256": _sha256(_json_bytes(contract)),
                "configured_optional_algorithms": configured,
                "selected_optional_algorithms": selected,
                "effective_case_ids": effective_case_ids,
            }
        )
    return result


def _aggregate_absolute(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, object]], bool, bool]:
    rows: list[dict[str, object]] = []
    complete = True
    passed = True
    required_case_count = len(cases)
    for gate_id, metric, operation, threshold in _ABSOLUTE_GATES:
        observed = [
            gate
            for case in cases
            for gate in cast(list[dict[str, object]], case["absolute_gates"])
            if gate["id"] == gate_id and gate["status"] != "not_evaluated"
        ]
        statuses = [str(gate["status"]) for gate in observed]
        row_complete = required_case_count > 0 and len(observed) == required_case_count
        status = (
            "fail"
            if "fail" in statuses
            else "not_evaluated"
            if not row_complete
            else "pass"
        )
        complete = complete and row_complete
        passed = passed and status != "fail"
        rows.append(
            {
                "id": gate_id,
                "metric": metric,
                "op": operation,
                "threshold": threshold,
                "status": status,
                "evaluated_case_count": len(observed),
                "required_case_count": required_case_count,
            }
        )
    return rows, complete, passed


def _aggregate_relative(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, object]], bool, bool, bool]:
    rows: list[dict[str, object]] = []
    complete = True
    passed = True
    any_material = False
    required_case_count = len(cases)
    for metric_id, path, direction, _nonregression, _material in _RELATIVE_METRICS:
        observed = [
            metric
            for case in cases
            for metric in cast(list[dict[str, object]], case["relative_metrics"])
            if metric["id"] == metric_id and metric["nonregression"] != "not_evaluated"
        ]
        statuses = [str(metric["nonregression"]) for metric in observed]
        row_complete = required_case_count > 0 and len(observed) == required_case_count
        status = (
            "fail"
            if "fail" in statuses
            else "not_evaluated"
            if not row_complete
            else "pass"
        )
        material = any(bool(metric["material_improvement"]) for metric in observed)
        any_material = any_material or material
        complete = complete and row_complete
        passed = passed and status != "fail"
        rows.append(
            {
                "id": metric_id,
                "metric": path,
                "direction": direction,
                "nonregression": status,
                "material_improvement": material,
                "evaluated_case_count": len(observed),
                "required_case_count": required_case_count,
            }
        )
    return rows, complete, passed, any_material


def _quality_by_candidate(
    cases: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate_cases = [
            case for case in cases if case["candidate_id"] == candidate_id
        ]
        absolute, absolute_complete, absolute_passed = _aggregate_absolute(
            candidate_cases
        )
        relative, relative_complete, relative_passed, material = _aggregate_relative(
            candidate_cases
        )
        result.append(
            {
                "candidate_id": candidate_id,
                "case_count": len(candidate_cases),
                "absolute_gates": absolute,
                "absolute_complete": absolute_complete,
                "absolute_passed": absolute_passed,
                "relative_metrics": relative,
                "relative_complete": relative_complete,
                "relative_nonregression_passed": relative_passed,
                "material_improvement": material,
            }
        )
    return result


def _case_metric_row(
    case: Mapping[str, Any],
    collection: str,
    metric_id: str,
) -> Mapping[str, object] | None:
    for row in cast(list[dict[str, object]], case[collection]):
        if row["id"] == metric_id:
            return row
    return None


def _optional_algorithm_gates(
    candidates: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        candidate_cases = [
            case for case in cases if case["candidate_id"] == candidate_id
        ]
        configured = _configured_optional_algorithms(candidate)
        algorithms: dict[str, dict[str, Any]] = {}

        temporal_cells: list[dict[str, Any]] = []
        if configured["MATTE-2.1"]:
            for cadence in ("fps_15", "fps_30", "fps_60"):
                applicable = [
                    case
                    for case in candidate_cases
                    if cadence in case["coverage"]["cadences"]
                    and "stationary" in case["coverage"]["motions"]
                    and "1280x720" in case["coverage"]["canvases"]
                    and _effective_optional_algorithms(case)["MATTE-2.1"]
                ]
                cell_status = "passed"
                if not applicable:
                    cell_status = "pending"
                for case in applicable:
                    metrics = cast(dict[str, object], case["algorithm_metrics"])
                    area = metrics["stationary_subject_area_drift_p95"]
                    dominance = metrics["maximum_dominant_previous_contour_intervals"]
                    baseline_contour = metrics[
                        "baseline_stationary_contour_displacement_p95_px"
                    ]
                    candidate_contour = metrics[
                        "candidate_stationary_contour_displacement_p95_px"
                    ]
                    if (
                        area is None
                        or dominance is None
                        or baseline_contour is None
                        or candidate_contour is None
                    ):
                        if cell_status != "failed":
                            cell_status = "pending"
                    elif (
                        float(cast(float, candidate_contour)) > 1.5
                        or float(cast(float, baseline_contour)) <= 1e-6
                        or float(cast(float, candidate_contour))
                        > float(cast(float, baseline_contour)) * 0.60
                        or float(cast(float, area)) > 0.01
                        or float(cast(float, dominance)) > 1.0
                    ):
                        cell_status = "failed"
                temporal_cells.append(
                    {
                        "cadence": cadence,
                        "canvas": "1280x720",
                        "case_ids": [str(case["id"]) for case in applicable],
                        "status": cell_status,
                    }
                )
        algorithms["MATTE-2.1"] = _algorithm_gate_summary(
            configured["MATTE-2.1"], temporal_cells
        )

        spatial_cells: list[dict[str, Any]] = []
        if configured["MATTE-2.2"]:
            for canvas in REQUIRED_COVERAGE["canvases"]:
                applicable = [
                    case
                    for case in candidate_cases
                    if canvas in case["coverage"]["canvases"]
                    and _effective_optional_algorithms(case)["MATTE-2.2"]
                ]
                cell_status = "passed"
                if not applicable:
                    cell_status = "pending"
                for case in applicable:
                    spatial_rows = [
                        _case_metric_row(case, "relative_metrics", metric_id)
                        for metric_id in (
                            "ground_truth_mse",
                            "ground_truth_gradient",
                        )
                    ]
                    if any(
                        row is None or row["nonregression"] == "not_evaluated"
                        for row in spatial_rows
                    ):
                        if cell_status != "failed":
                            cell_status = "pending"
                    elif any(
                        row["nonregression"] != "pass"
                        for row in spatial_rows
                        if row is not None
                    ) or not any(
                        row["material_improvement"] is True
                        for row in spatial_rows
                        if row is not None
                    ):
                        cell_status = "failed"
                spatial_cells.append(
                    {
                        "canvas": canvas,
                        "case_ids": [str(case["id"]) for case in applicable],
                        "status": cell_status,
                    }
                )
        algorithms["MATTE-2.2"] = _algorithm_gate_summary(
            configured["MATTE-2.2"], spatial_cells
        )

        wrap_cells: list[dict[str, Any]] = []
        if configured["MATTE-2.4"]:
            for background in ("dynamic_video", "live_camera"):
                applicable = [
                    case
                    for case in candidate_cases
                    if background in case["coverage"]["backgrounds"]
                    and _effective_optional_algorithms(case)["MATTE-2.4"]
                ]
                cell_status = "passed"
                if not applicable:
                    cell_status = "pending"
                for case in applicable:
                    shimmer = _case_metric_row(case, "relative_metrics", "edge_shimmer")
                    alpha_rows = [
                        _case_metric_row(case, "relative_metrics", metric_id)
                        for metric_id in (
                            "opaque_core_deficit",
                            "opaque_core_p05",
                            "background_alpha",
                            "halo_area",
                        )
                    ]
                    if (
                        shimmer is None
                        or shimmer["nonregression"] == "not_evaluated"
                        or any(
                            row is None or row["nonregression"] == "not_evaluated"
                            for row in alpha_rows
                        )
                    ):
                        if cell_status != "failed":
                            cell_status = "pending"
                    elif (
                        shimmer["nonregression"] != "pass"
                        or shimmer["material_improvement"] is not True
                        or any(
                            row["nonregression"] != "pass"
                            for row in alpha_rows
                            if row is not None
                        )
                    ):
                        cell_status = "failed"
                wrap_cells.append(
                    {
                        "background": background,
                        "case_ids": [str(case["id"]) for case in applicable],
                        "status": cell_status,
                    }
                )
        algorithms["MATTE-2.4"] = _algorithm_gate_summary(
            configured["MATTE-2.4"], wrap_cells
        )

        selected_rows = [
            row for row in algorithms.values() if row["status"] != "not_selected"
        ]
        result.append(
            {
                "candidate_id": candidate_id,
                "algorithms": algorithms,
                "complete": all(row["status"] != "pending" for row in selected_rows),
                "passed": all(row["status"] != "failed" for row in selected_rows),
            }
        )
    return result


def _algorithm_gate_summary(
    configured: bool,
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not configured:
        status = "not_selected"
    elif any(cell["status"] == "failed" for cell in cells):
        status = "failed"
    elif not cells or any(cell["status"] == "pending" for cell in cells):
        status = "pending"
    else:
        status = "passed"
    return {"status": status, "cells": list(cells)}


def _matrix_by_candidate(
    cases: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    *,
    authoritative: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate_cases = [
            case for case in cases if case["candidate_id"] == candidate_id
        ]
        proxy_observed = {
            axis: sorted(
                {
                    value
                    for case in candidate_cases
                    for value in cast(dict[str, list[str]], case["coverage"])[axis]
                }
            )
            for axis in REQUIRED_COVERAGE
        }
        observed = (
            proxy_observed
            if authoritative
            else {axis: [] for axis in REQUIRED_COVERAGE}
        )
        missing = {
            axis: sorted(set(required) - set(observed[axis]))
            for axis, required in REQUIRED_COVERAGE.items()
        }
        proxy_boundaries = sorted(
            {
                boundary_id
                for case in candidate_cases
                for boundary_id, boundary in cast(
                    dict[str, dict[str, object]],
                    case["boundaries"]["boundaries"],
                ).items()
                if boundary["status"] == "passed"
            }
        )
        observed_boundaries = proxy_boundaries if authoritative else []
        missing_boundaries = sorted(set(REQUIRED_BOUNDARIES) - set(observed_boundaries))
        proxy_review_passed = all(
            case["review"]["status"] == "passed" for case in candidate_cases
        )
        proxy_preferred = any(
            case["review"]["overall"] == "candidate_preferred"
            for case in candidate_cases
        )
        result.append(
            {
                "candidate_id": candidate_id,
                "case_count": len(candidate_cases),
                "observed": observed,
                "proxy_observed": proxy_observed,
                "missing": missing,
                "complete": not any(missing.values()),
                "observed_boundaries": observed_boundaries,
                "proxy_observed_boundaries": proxy_boundaries,
                "missing_boundaries": missing_boundaries,
                "boundaries_complete": not missing_boundaries,
                "review_all_cases_passed": (authoritative and proxy_review_passed),
                "review_candidate_preferred_at_least_once": (
                    authoritative and proxy_preferred
                ),
                "proxy_review_all_cases_passed": proxy_review_passed,
                "proxy_review_candidate_preferred_at_least_once": proxy_preferred,
            }
        )
    return result


def _qualify_plan(
    plan_path: Path | str,
    *,
    loaded: tuple[dict[str, Any], str] | None = None,
) -> dict[str, Any]:
    plan, plan_sha256 = _load_plan(plan_path) if loaded is None else loaded
    candidates = {
        str(candidate["id"]): candidate
        for candidate in cast(list[dict[str, Any]], plan["candidates"])
    }
    plan_kind = str(plan["provenance"]["kind"])
    plan_reference = str(plan["provenance"]["license_or_consent_reference"])
    case_results: list[dict[str, Any]] = []
    proxy_coverage: dict[str, set[str]] = {axis: set() for axis in REQUIRED_COVERAGE}
    captured_boundaries: set[str] = set()
    seen_artifact_paths: set[Path] = set()
    seen_bundle_manifests: set[str] = set()
    seen_annotation_manifests: set[str] = set()

    for case in cast(list[dict[str, Any]], plan["cases"]):
        case_id = str(case["id"])
        candidate_definition = candidates[str(case["candidate_id"])]
        baseline_bundle = MatteReplayBundle(case["baseline"]["bundle"])
        candidate_bundle = MatteReplayBundle(case["candidate"]["bundle"])
        if baseline_bundle.manifest_sha256 == candidate_bundle.manifest_sha256:
            raise MatteVisualQualificationError(
                f"case {case_id} baseline and candidate manifests are identical"
            )
        for manifest_sha256 in (
            baseline_bundle.manifest_sha256,
            candidate_bundle.manifest_sha256,
        ):
            if manifest_sha256 in seen_bundle_manifests:
                raise MatteVisualQualificationError(
                    "replay bundle content cannot be reused across qualification cases"
                )
            seen_bundle_manifests.add(manifest_sha256)
        baseline_annotations = MatteQualityAnnotations(
            case["baseline"]["annotations"], baseline_bundle
        )
        candidate_annotations = MatteQualityAnnotations(
            case["candidate"]["annotations"], candidate_bundle
        )
        for manifest_sha256 in (
            baseline_annotations.manifest_sha256,
            candidate_annotations.manifest_sha256,
        ):
            if manifest_sha256 in seen_annotation_manifests:
                raise MatteVisualQualificationError(
                    "annotation content cannot be reused across qualification cases"
                )
            seen_annotation_manifests.add(manifest_sha256)
        if _source_contract(baseline_bundle) != _source_contract(candidate_bundle):
            raise MatteVisualQualificationError(
                f"case {case_id} baseline and candidate do not share source, "
                "backdrop, timestamps, and generations"
            )
        if _annotation_contract(baseline_annotations) != _annotation_contract(
            candidate_annotations
        ):
            raise MatteVisualQualificationError(
                f"case {case_id} baseline and candidate annotations differ"
            )
        annotation_provenance_sha256 = _assert_annotation_provenance(
            baseline_annotations,
            candidate_annotations,
            case_id=case_id,
            plan_kind=plan_kind,
            plan_reference=plan_reference,
        )
        _assert_motion_coverage(
            candidate_annotations,
            case["coverage"]["motions"],
            case_id=case_id,
        )
        _assert_reactions_disabled(baseline_bundle, f"case {case_id} baseline")
        _assert_reactions_disabled(candidate_bundle, f"case {case_id} candidate")

        baseline_canvases, baseline_backgrounds = _configured_case_contract(
            baseline_bundle,
            case["baseline_expected"],
            case["baseline_expected"]["effective"],
            case_id=f"{case_id} baseline",
        )
        derived_canvases, derived_backgrounds = _configured_case_contract(
            candidate_bundle,
            candidate_definition,
            case["expected_effective"],
            case_id=case_id,
        )
        if (
            baseline_canvases != derived_canvases
            or baseline_backgrounds != derived_backgrounds
        ):
            raise MatteVisualQualificationError(
                f"case {case_id} baseline and candidate canvas/background differ"
            )
        derived_cadences = _derived_cadences(
            candidate_bundle,
            case_id,
            authoritative=plan_kind != "generated",
        )
        mechanical = {
            "canvases": derived_canvases,
            "backgrounds": derived_backgrounds,
            "cadences": derived_cadences,
        }
        for axis, derived in mechanical.items():
            if case["coverage"][axis] != derived:
                raise MatteVisualQualificationError(
                    f"case {case_id} claimed {axis} do not match replay evidence; "
                    f"claimed={case['coverage'][axis]!r} derived={derived!r}"
                )
        for axis, values in cast(dict[str, list[str]], case["coverage"]).items():
            proxy_coverage[axis].update(values)

        baseline_report = evaluate_bundle(
            baseline_bundle.root,
            annotations_root=baseline_annotations.root,
            metadata=EvaluationMetadata(
                backend="baseline",
                device="recorded",
                configuration_label=case_id,
            ),
        )
        candidate_report = evaluate_bundle(
            candidate_bundle.root,
            annotations_root=candidate_annotations.root,
            metadata=EvaluationMetadata(
                backend=str(case["expected_effective"]["segmentation_backend"]),
                device=str(case["expected_effective"]["segmentation_device"]),
                configuration_label=str(case["candidate_id"]),
            ),
            baseline=baseline_report,
        )
        absolute_gates = _absolute_gate_results(candidate_report)
        relative_metrics = _relative_results(baseline_report, candidate_report)
        stationary_area_drift = _metric_number(
            candidate_report,
            "aggregate.segment_kinds.stationary.metrics."
            "stationary_subject_area_drift.p95",
        )
        previous_contour_dominance = _metric_number(
            candidate_report,
            "aggregate.motion.maximum_dominant_previous_contour_intervals",
        )
        baseline_stationary_contour = _metric_number(
            baseline_report,
            "aggregate.segment_kinds.stationary.metrics."
            "contour_displacement_p95_px.p95",
        )
        candidate_stationary_contour = _metric_number(
            candidate_report,
            "aggregate.segment_kinds.stationary.metrics."
            "contour_displacement_p95_px.p95",
        )
        algorithm_metrics = {
            "stationary_subject_area_drift_p95": (
                None
                if stationary_area_drift is None
                else round(stationary_area_drift, 8)
            ),
            "maximum_dominant_previous_contour_intervals": (
                None
                if previous_contour_dominance is None
                else round(previous_contour_dominance, 8)
            ),
            "baseline_stationary_contour_displacement_p95_px": (
                None
                if baseline_stationary_contour is None
                else round(baseline_stationary_contour, 8)
            ),
            "candidate_stationary_contour_displacement_p95_px": (
                None
                if candidate_stationary_contour is None
                else round(candidate_stationary_contour, 8)
            ),
        }
        boundary, boundary_sha256 = _load_boundary_evidence(
            case["boundary_evidence"],
            case_id=case_id,
            plan_kind=plan_kind,
            bundle=candidate_bundle,
            policy=plan["policy"],
            derived_cadences=derived_cadences,
            seen_artifact_paths=seen_artifact_paths,
        )
        for boundary_id, result in cast(
            dict[str, dict[str, object]], boundary["boundaries"]
        ).items():
            if result["status"] == "passed":
                captured_boundaries.add(boundary_id)
        review = _load_review(
            case["review"],
            case_id=case_id,
            plan_kind=plan_kind,
            plan_reference=plan_reference,
            plan_sha256=plan_sha256,
            baseline_manifest_sha256=baseline_bundle.manifest_sha256,
            candidate_manifest_sha256=candidate_bundle.manifest_sha256,
            baseline_annotation_manifest_sha256=(baseline_annotations.manifest_sha256),
            candidate_annotation_manifest_sha256=(
                candidate_annotations.manifest_sha256
            ),
            baseline_quality_evidence_sha256=baseline_report["determinism"][
                "evidence_sha256"
            ],
            candidate_quality_evidence_sha256=candidate_report["determinism"][
                "evidence_sha256"
            ],
            boundary_evidence_sha256=boundary_sha256,
        )
        gate_failed = any(gate["status"] == "fail" for gate in absolute_gates)
        relative_failed = any(
            metric["nonregression"] == "fail" for metric in relative_metrics
        )
        quality_incomplete = any(
            gate["status"] == "not_evaluated" for gate in absolute_gates
        ) or any(
            metric["nonregression"] == "not_evaluated" for metric in relative_metrics
        )
        boundary_failed = any(
            result["status"] == "failed"
            for result in cast(
                dict[str, dict[str, object]], boundary["boundaries"]
            ).values()
        )
        outcome = (
            "failed"
            if gate_failed
            or relative_failed
            or boundary_failed
            or review["status"] == "failed"
            else "passed"
            if not quality_incomplete
            else "pending"
        )
        candidate_contract = {
            "segmentation": _path_free_segmentation(
                candidate_definition["segmentation"]
            ),
            "compositing": candidate_definition["compositing"],
            "expected_effective": case["expected_effective"],
        }
        baseline_contract = {
            "segmentation": _path_free_segmentation(
                case["baseline_expected"]["segmentation"]
            ),
            "compositing": case["baseline_expected"]["compositing"],
            "expected_effective": case["baseline_expected"]["effective"],
        }
        case_results.append(
            {
                "id": case_id,
                "candidate_id": case["candidate_id"],
                "outcome": outcome,
                "source": {
                    "baseline_bundle_manifest_sha256": (
                        baseline_bundle.manifest_sha256
                    ),
                    "candidate_bundle_manifest_sha256": (
                        candidate_bundle.manifest_sha256
                    ),
                    "baseline_annotation_manifest_sha256": (
                        baseline_annotations.manifest_sha256
                    ),
                    "candidate_annotation_manifest_sha256": (
                        candidate_annotations.manifest_sha256
                    ),
                    "annotation_provenance_sha256": annotation_provenance_sha256,
                    "source_contract_sha256": _sha256(
                        _json_bytes(_source_contract(candidate_bundle))
                    ),
                },
                "algorithm_contract": candidate_contract,
                "algorithm_contract_sha256": _sha256(_json_bytes(candidate_contract)),
                "baseline_algorithm_contract": baseline_contract,
                "baseline_algorithm_contract_sha256": _sha256(
                    _json_bytes(baseline_contract)
                ),
                "coverage": case["coverage"],
                "absolute_gates": absolute_gates,
                "relative_metrics": relative_metrics,
                "algorithm_metrics": algorithm_metrics,
                "quality_complete": not quality_incomplete,
                "quality_evidence": {
                    "baseline_sha256": baseline_report["determinism"][
                        "evidence_sha256"
                    ],
                    "candidate_sha256": candidate_report["determinism"][
                        "evidence_sha256"
                    ],
                },
                "boundaries": boundary,
                "boundary_evidence_sha256": boundary_sha256,
                "review": review,
                "reactions_disabled": True,
            }
        )

    absolute_rows, _global_absolute_complete, _global_absolute_passed = (
        _aggregate_absolute(case_results)
    )
    (
        relative_rows,
        _global_relative_complete,
        _global_relative_passed,
        _global_material_improvement,
    ) = _aggregate_relative(case_results)
    candidate_quality = _quality_by_candidate(
        case_results,
        [str(candidate["id"]) for candidate in plan["candidates"]],
    )
    absolute_complete = all(
        bool(candidate["absolute_complete"]) for candidate in candidate_quality
    )
    absolute_passed = all(
        bool(candidate["absolute_passed"]) for candidate in candidate_quality
    )
    relative_complete = all(
        bool(candidate["relative_complete"]) for candidate in candidate_quality
    )
    relative_passed = all(
        bool(candidate["relative_nonregression_passed"])
        for candidate in candidate_quality
    )
    material_improvement = all(
        bool(candidate["material_improvement"]) for candidate in candidate_quality
    )
    optional_algorithm_gates = _optional_algorithm_gates(
        plan["candidates"], case_results
    )
    optional_algorithms_complete = all(
        bool(candidate["complete"]) for candidate in optional_algorithm_gates
    )
    optional_algorithms_passed = all(
        bool(candidate["passed"]) for candidate in optional_algorithm_gates
    )
    candidate_matrix = _matrix_by_candidate(
        case_results,
        [str(candidate["id"]) for candidate in plan["candidates"]],
        authoritative=plan_kind != "generated",
    )
    proxy_observed = {axis: sorted(values) for axis, values in proxy_coverage.items()}
    authoritative_observed = {
        axis: sorted(
            set(REQUIRED_COVERAGE[axis]).intersection(
                *(set(candidate["observed"][axis]) for candidate in candidate_matrix)
            )
        )
        for axis in REQUIRED_COVERAGE
    }
    missing = {
        axis: sorted(set(required) - set(authoritative_observed[axis]))
        for axis, required in REQUIRED_COVERAGE.items()
    }
    coverage_complete = all(
        bool(candidate["complete"]) for candidate in candidate_matrix
    )
    authoritative_boundaries = set(REQUIRED_BOUNDARIES).intersection(
        *(set(candidate["observed_boundaries"]) for candidate in candidate_matrix)
    )
    missing_boundaries = sorted(set(REQUIRED_BOUNDARIES) - authoritative_boundaries)
    boundaries_complete = all(
        bool(candidate["boundaries_complete"]) for candidate in candidate_matrix
    )
    proxy_reviews_passed = all(
        bool(candidate["proxy_review_all_cases_passed"])
        for candidate in candidate_matrix
    )
    proxy_preferred_review = all(
        bool(candidate["proxy_review_candidate_preferred_at_least_once"])
        for candidate in candidate_matrix
    )
    reviews_passed = all(
        bool(candidate["review_all_cases_passed"]) for candidate in candidate_matrix
    )
    preferred_review = all(
        bool(candidate["review_candidate_preferred_at_least_once"])
        for candidate in candidate_matrix
    )
    case_failures = [case["id"] for case in case_results if case["outcome"] == "failed"]

    reasons: list[str] = []
    if plan_kind == "generated":
        reasons.append("generated evidence cannot qualify representative local video")
    if not coverage_complete:
        reasons.append("representative visual matrix coverage is incomplete")
    if not boundaries_complete:
        reasons.append("local output-boundary coverage is incomplete")
    if not absolute_complete:
        reasons.append("ratified absolute metric coverage is incomplete")
    if not relative_complete:
        reasons.append("baseline-relative metric coverage is incomplete")
    if not material_improvement:
        reasons.append("no material baseline-relative improvement was established")
    if not optional_algorithms_complete:
        reasons.append("selected optional-algorithm qualification cells are incomplete")
    if not optional_algorithms_passed:
        reasons.append("one or more selected optional-algorithm gates failed")
    if not preferred_review:
        reasons.append("no review preferred the candidate")
    if case_failures:
        reasons.append("one or more qualification cases failed")

    hard_failure = (
        bool(case_failures)
        or not absolute_passed
        or not relative_passed
        or not optional_algorithms_passed
    )
    if plan_kind != "generated" and relative_complete and not material_improvement:
        hard_failure = True
    if plan_kind != "generated" and reviews_passed and not preferred_review:
        hard_failure = True
    complete = (
        plan_kind != "generated"
        and coverage_complete
        and boundaries_complete
        and absolute_complete
        and relative_complete
        and material_improvement
        and optional_algorithms_complete
        and optional_algorithms_passed
        and reviews_passed
        and preferred_review
    )
    status = "failed" if hard_failure else "qualified" if complete else "pending"

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "status": status,
        "source": {"plan_sha256": plan_sha256},
        "provenance": {
            "qualification_id": plan["provenance"]["qualification_id"],
            "kind": plan_kind,
            "contains_private_footage_in_repository": False,
        },
        "privacy": {
            "report_is_path_free": True,
            "report_contains_pixels": False,
            "network_camera_model_preview_or_sink_opened": False,
            "private_inputs_owner_only": True,
            "local_boundary_method_and_platform_attested": plan_kind != "generated",
            "physical_capture_origin_cryptographically_proven": False,
        },
        "policy": plan["policy"],
        "algorithm_manifest": _algorithm_manifest(plan["candidates"], case_results),
        "coverage": {
            "appearance_and_source_condition_authority": (
                "generated-proxy"
                if plan_kind == "generated"
                else "owner-and-reviewer-attested"
            ),
            "required": {
                axis: list(values) for axis, values in REQUIRED_COVERAGE.items()
            },
            "observed": authoritative_observed,
            "proxy_observed": proxy_observed,
            "missing": missing,
            "complete": coverage_complete,
            "required_boundaries": list(REQUIRED_BOUNDARIES),
            "observed_boundaries": sorted(authoritative_boundaries),
            "proxy_observed_boundaries": sorted(captured_boundaries),
            "missing_boundaries": missing_boundaries,
            "boundaries_complete": boundaries_complete,
            "boundary_evidence_scope": (
                "same-generation-spatial-resize-and-contour-parity"
            ),
            "boundary_sink_cadence_qualified": False,
            "by_candidate": [
                {
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "case_count",
                        "observed",
                        "proxy_observed",
                        "missing",
                        "complete",
                        "observed_boundaries",
                        "proxy_observed_boundaries",
                        "missing_boundaries",
                        "boundaries_complete",
                    )
                }
                for candidate in candidate_matrix
            ],
        },
        "quality": {
            "absolute_gates": absolute_rows,
            "absolute_complete": absolute_complete,
            "absolute_passed": absolute_passed,
            "relative_metrics": relative_rows,
            "relative_complete": relative_complete,
            "relative_nonregression_passed": relative_passed,
            "material_improvement": material_improvement,
            "by_candidate": candidate_quality,
            "optional_algorithms_complete": optional_algorithms_complete,
            "optional_algorithms_passed": optional_algorithms_passed,
            "optional_algorithm_gates": optional_algorithm_gates,
            "optional_algorithm_evidence_scope": "combined-selected-policy",
            "independent_optional_algorithm_causality_claimed": False,
        },
        "review": {
            "required_concerns": list(REQUIRED_REVIEW_CONCERNS),
            "authority": (
                "generated-schema-proxy"
                if plan_kind == "generated"
                else "consented-or-licensed-local-human"
            ),
            "all_cases_passed": reviews_passed,
            "candidate_preferred_at_least_once": preferred_review,
            "proxy_all_cases_passed": proxy_reviews_passed,
            "proxy_candidate_preferred_at_least_once": proxy_preferred_review,
            "motion_cadence_evidence_scope": "digest-bound-replay-clip-sequence",
            "by_candidate": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "all_cases_passed": candidate["review_all_cases_passed"],
                    "candidate_preferred_at_least_once": candidate[
                        "review_candidate_preferred_at_least_once"
                    ],
                    "proxy_all_cases_passed": candidate[
                        "proxy_review_all_cases_passed"
                    ],
                    "proxy_candidate_preferred_at_least_once": candidate[
                        "proxy_review_candidate_preferred_at_least_once"
                    ],
                }
                for candidate in candidate_matrix
            ],
        },
        "cases": case_results,
        "production": {
            "quality_preset_selected": False,
            "default_changed": False,
            "generated_evidence_can_qualify": False,
            "reactions_enabled": False,
        },
        "reasons": reasons,
    }
    report["evidence_sha256"] = _sha256(_json_bytes(report))
    return report


def qualify_plan(plan_path: Path | str) -> dict[str, Any]:
    """Recompute and aggregate one strict owner-only MATTE-5.2 plan."""

    try:
        return _qualify_plan(plan_path)
    except MatteDiagnosticsError as exc:
        raise MatteVisualQualificationError(str(exc)) from exc


def report_markdown(report: Mapping[str, object]) -> str:
    """Render a compact, path-free companion to the authoritative JSON report."""

    coverage = cast(dict[str, Any], report["coverage"])
    quality = cast(dict[str, Any], report["quality"])
    review = cast(dict[str, Any], report["review"])
    reasons = cast(list[str], report["reasons"])
    cases = cast(list[dict[str, Any]], report["cases"])
    lines = [
        "# Matte visual qualification report",
        "",
        f"- Status: **{report['status']}**",
        f"- Evidence SHA-256: `{report['evidence_sha256']}`",
        f"- Representative matrix complete: **{coverage['complete']}**",
        f"- Output boundaries complete: **{coverage['boundaries_complete']}**",
        f"- Absolute gates passed: **{quality['absolute_passed']}**",
        (
            "- Baseline-relative non-regression passed: "
            f"**{quality['relative_nonregression_passed']}**"
        ),
        f"- Material improvement established: **{quality['material_improvement']}**",
        f"- Review authority: `{review['authority']}`",
        "- Production default changed: **False**",
        "",
        "| Case | Candidate | Outcome | Review |",
        "| --- | --- | --- | --- |",
    ]
    for case in cases:
        case_review = cast(dict[str, Any], case["review"])
        lines.append(
            f"| `{case['id']}` | `{case['candidate_id']}` | "
            f"**{case['outcome']}** | `{case_review['status']}` |"
        )
    lines.extend(["", "## Decision reasons", ""])
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- All qualification requirements were met.")
    lines.extend(
        [
            "",
            "Canonical JSON is authoritative. Generated proxy evidence can exercise "
            "the schema and comparisons but can never qualify representative local "
            "footage or physical output boundaries.",
            "",
        ]
    )
    return "\n".join(lines)


def _assert_output_separate(
    output: Path,
    plan_path: Path | str,
    plan: Mapping[str, Any],
) -> None:
    resolved_output = output.resolve(strict=False)
    protected = [Path(plan_path).resolve(strict=False)]
    for case in cast(list[dict[str, Any]], plan["cases"]):
        protected.extend(
            Path(value).resolve(strict=False)
            for value in (
                case["baseline"]["bundle"],
                case["baseline"]["annotations"],
                case["candidate"]["bundle"],
                case["candidate"]["annotations"],
                case["boundary_evidence"],
                case["review"],
            )
        )
    if any(
        resolved_output == item
        or resolved_output in item.parents
        or item in resolved_output.parents
        for item in protected
    ):
        raise MatteVisualQualificationError(
            "qualification output must be separate from every evidence input"
        )


def run_qualification(
    plan_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """Evaluate a plan and write a new owner-only, path-free report directory."""

    try:
        plan, plan_sha256 = _load_plan(plan_path)
        output = Path(output_root)
        _assert_output_separate(output, plan_path, plan)
        report = _qualify_plan(plan_path, loaded=(plan, plan_sha256))
        _private_directory(output, create=True)
        _atomic_private_write(output / "qualification.json", _json_bytes(report))
        _atomic_private_write(
            output / "qualification.md",
            report_markdown(report).encode("utf-8"),
        )
    except MatteDiagnosticsError as exc:
        raise MatteVisualQualificationError(str(exc)) from exc
    return report


def build_parser(
    *,
    prog: str = "custback matte-visual-qualify",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Recompute a private same-source matte visual qualification matrix"
        ),
    )
    parser.add_argument("plan", help="owner-only visual qualification plan JSON")
    parser.add_argument(
        "--output",
        required=True,
        help="new owner-only path-free report directory",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback matte-visual-qualify",
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
    status = str(report["status"])
    print(f"evaluated {len(report['cases'])} visual case(s); status {status}")
    return 0 if status == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
