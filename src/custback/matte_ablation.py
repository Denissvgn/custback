"""Bounded, same-source matte backend/postprocess/cadence ablation.

The runner executes frozen postprocess/compositor variants, projects output
repeat cadence without inventing model inputs, and joins separately recorded
backend/model/device runs after proving their source-frame identity. It never
turns a missing backend into a fallback result and never labels one-host
screening as a production preset.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import cv2
import numpy as np

from .compositor import composite
from .config import BlendSpace, SegmentationConfig
from .matte_diagnostics import (
    MAX_MANIFEST_BYTES,
    MIN_BUNDLE_BYTES,
    MatteCaptureMetadata,
    MatteDiagnosticRecorder,
    MatteDiagnosticsError,
    MatteFrameEvidence,
    MatteReplayBundle,
    _atomic_private_write,
    _controls,
    _json_bytes,
    _private_directory,
    _private_write,
    _read_private_file,
    _recorded_transform,
)
from .matte_quality import (
    EvaluationMetadata,
    MatteQualityAnnotations,
    MatteQualityError,
    QualityFrameAnnotations,
    _metric_path,
    _round,
    _summary,
    evaluate_bundle,
    write_quality_annotations,
)
from .segmentation import MaskRefiner

PLAN_SCHEMA = "custback.matte-ablation-plan"
PLAN_VERSION = 1
REPORT_SCHEMA = "custback.matte-ablation-report"
REPORT_VERSION = 1
DEFAULT_MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_VARIANTS = 64
MAX_SHORTLIST = 5
_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_AXIS = re.compile(r"^[a-z][a-z0-9_.:-]{0,95}$")

VariantKind = Literal["frozen", "recorded", "cadence_projection", "unavailable"]
Lane = Literal["rvm", "mediapipe", "compositor", "cadence"]

_POSTPROCESS_FIELDS = {
    "edge_refine",
    "mask_blur",
    "mask_shift",
    "temporal_smoothing",
}
_COMPOSITOR_FIELDS = {
    "use_model_foreground",
    "light_wrap",
    "blend_space",
}
_QUALITY_PATHS = {
    "opaque_core_alpha_p05": "aggregate.metrics.opaque_core_alpha_p05.p05",
    "opaque_core_mean_deficit": "aggregate.metrics.opaque_core_mean_deficit.p95",
    "foreground_hole_components": "aggregate.metrics.foreground_hole_components.max",
    "background_alpha_mean": "aggregate.metrics.background_alpha_mean.p95",
    "exterior_halo_area_ratio": "aggregate.metrics.exterior_halo_area_ratio.p95",
    "exterior_halo_width_p95_px": ("aggregate.metrics.exterior_halo_width_p95_px.p95"),
    "opaque_backdrop_leakage_coefficient": (
        "aggregate.metrics.opaque_backdrop_leakage_coefficient.p95"
    ),
    "compensated_contour_displacement_p95_px": (
        "aggregate.metrics.contour_displacement_p95_px.p95"
    ),
    "compensated_alpha_temporal_abs_diff": (
        "aggregate.metrics.compensated_alpha_temporal_abs_diff.p95"
    ),
    "motion_trail_area_ratio": "aggregate.metrics.motion_trail_area_ratio.p95",
    "edge_band_rgb_variation": "aggregate.metrics.edge_band_rgb_variation.p95",
    "ground_truth_alpha_mse": "aggregate.metrics.ground_truth_alpha_mse.p95",
    "ground_truth_gradient_mae": ("aggregate.metrics.ground_truth_gradient_mae.p95"),
    "uncertain_pixel_fraction": "aggregate.metrics.uncertain_pixel_fraction.p50",
}


@dataclass(frozen=True)
class AblationPolicy:
    shortlist_limit_per_lane: int = 2
    performance_budget_ms: float | None = None
    opaque_alpha_p05: float = 0.95
    background_alpha_mean: float = 0.01
    relative_nonregression: float = 1.10
    trail_absolute_tolerance: float = 0.01
    edge_improvement_ratio: float = 0.70
    contour_improvement_ratio: float = 0.60
    performance_improvement_ratio: float = 0.90
    leakage_absolute_improvement: float = 0.01


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise MatteQualityError(f"{field} must be a lowercase safe identifier")
    return value


def _axes(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise MatteQualityError("variant covers must be a list")
    result: list[str] = []
    for axis in value:
        if not isinstance(axis, str) or _AXIS.fullmatch(axis) is None or axis in result:
            raise MatteQualityError("variant coverage axis is invalid")
        result.append(axis)
    return tuple(result)


def _policy(value: object) -> AblationPolicy:
    if value is None:
        return AblationPolicy()
    if not isinstance(value, dict):
        raise MatteQualityError("ablation decision policy is invalid")
    allowed = set(AblationPolicy.__dataclass_fields__)
    if set(value) - allowed:
        raise MatteQualityError("ablation decision policy contains unknown fields")
    try:
        result = AblationPolicy(**value)
    except TypeError as exc:
        raise MatteQualityError("ablation decision policy is invalid") from exc
    if (
        type(result.shortlist_limit_per_lane) is not int
        or not 1 <= result.shortlist_limit_per_lane <= MAX_SHORTLIST
    ):
        raise MatteQualityError("shortlist limit is invalid")
    for name, setting in asdict(result).items():
        if name == "shortlist_limit_per_lane" or setting is None:
            continue
        if (
            isinstance(setting, bool)
            or not isinstance(setting, (int, float))
            or not math.isfinite(float(setting))
            or float(setting) < 0.0
        ):
            raise MatteQualityError(f"ablation policy {name} is invalid")
    return result


def load_plan(path: Path | str) -> tuple[dict[str, Any], str]:
    """Read and validate an owner-only plan that may contain private paths."""

    payload = _read_private_file(Path(path), max_bytes=MAX_MANIFEST_BYTES)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatteQualityError("ablation plan is malformed") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != PLAN_SCHEMA
        or value.get("version") != PLAN_VERSION
    ):
        raise MatteQualityError("unsupported ablation plan")
    scope = value.get("scope")
    if not isinstance(scope, dict):
        raise MatteQualityError("ablation scope is missing")
    for name in (
        "one_host_screening",
        "model_inference_executed",
        "reported_screenshots_used_as_ab",
    ):
        if type(scope.get(name)) is not bool:
            raise MatteQualityError(f"ablation scope {name} must be boolean")
    if scope["reported_screenshots_used_as_ab"]:
        raise MatteQualityError("reported screenshots cannot be used as an A/B test")
    required_axes = _axes(value.get("required_axes", []))
    variants = value.get("variants")
    if not isinstance(variants, list) or not variants or len(variants) > MAX_VARIANTS:
        raise MatteQualityError("ablation variant count is invalid")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw in variants:
        if not isinstance(raw, dict):
            raise MatteQualityError("ablation variant is invalid")
        variant_id = _safe_id(raw.get("id"), field="variant id")
        if variant_id in seen or variant_id == "baseline":
            raise MatteQualityError("ablation variant id is duplicated or reserved")
        seen.add(variant_id)
        kind = raw.get("kind")
        lane = raw.get("lane")
        if kind not in ("frozen", "recorded", "cadence_projection", "unavailable"):
            raise MatteQualityError("ablation variant kind is invalid")
        if lane not in ("rvm", "mediapipe", "compositor", "cadence"):
            raise MatteQualityError("ablation lane is invalid")
        covers = _axes(raw.get("covers", []))
        group = raw.get("group", "one_variable")
        if group not in (
            "one_variable",
            "compositor_factorial",
            "blend_space",
            "backend",
            "model_profile",
            "backdrop",
            "cadence",
        ):
            raise MatteQualityError("ablation group is invalid")
        normalized_row = dict(raw)
        evidence_kind = raw.get(
            "evidence_kind",
            "model-backed" if kind == "recorded" else "recorded-intermediate",
        )
        if evidence_kind not in (
            "model-backed",
            "generated-proxy",
            "recorded-intermediate",
            "projection",
            "unavailable",
        ):
            raise MatteQualityError("ablation evidence kind is invalid")
        if kind == "recorded" and evidence_kind not in (
            "model-backed",
            "generated-proxy",
        ):
            raise MatteQualityError(
                "recorded variants require model-backed or generated-proxy evidence"
            )
        if kind == "frozen" and evidence_kind not in (
            "recorded-intermediate",
            "generated-proxy",
        ):
            raise MatteQualityError(
                "frozen variants require recorded-intermediate or generated-proxy evidence"
            )
        if kind == "cadence_projection" and evidence_kind != "projection":
            raise MatteQualityError(
                "cadence projections must be labeled as projection evidence"
            )
        if kind == "unavailable" and evidence_kind != "unavailable":
            raise MatteQualityError(
                "unavailable variants must be labeled as unavailable evidence"
            )
        if evidence_kind == "model-backed" and not scope["model_inference_executed"]:
            raise MatteQualityError(
                "model-backed evidence requires model_inference_executed scope"
            )
        normalized_row.update(
            {
                "id": variant_id,
                "kind": kind,
                "lane": lane,
                "covers": list(covers),
                "group": group,
                "evidence_kind": evidence_kind,
            }
        )
        if kind == "frozen":
            postprocess = raw.get("postprocess", {})
            compositing = raw.get("compositing", {})
            if (
                not isinstance(postprocess, dict)
                or set(postprocess) - _POSTPROCESS_FIELDS
                or not isinstance(compositing, dict)
                or set(compositing) - _COMPOSITOR_FIELDS
            ):
                raise MatteQualityError("frozen variant overrides are invalid")
            changed = len(postprocess) + len(compositing)
            if group == "one_variable" and changed != 1:
                raise MatteQualityError(
                    "one-variable rows must override exactly one control"
                )
            if group == "compositor_factorial" and (
                set(compositing) != {"use_model_foreground", "light_wrap"}
                or postprocess
            ):
                raise MatteQualityError(
                    "compositor factorial rows must set foreground and wrap"
                )
            if lane == "rvm" and not bool(raw.get("explicit_rvm_policy", False)):
                if (
                    bool(postprocess.get("edge_refine", False))
                    or int(postprocess.get("mask_blur", 0) or 0) != 0
                    or float(postprocess.get("temporal_smoothing", 0.0) or 0.0) != 0.0
                ):
                    raise MatteQualityError(
                        "generic edge/blur/EMA controls require an explicit RVM policy"
                    )
        elif kind == "recorded":
            if not isinstance(raw.get("bundle"), str) or not isinstance(
                raw.get("annotations"), str
            ):
                raise MatteQualityError(
                    "recorded variants require bundle and annotation paths"
                )
            if raw.get("timestamp_policy", "exact") not in ("exact", "same_pixels"):
                raise MatteQualityError("recorded timestamp policy is invalid")
            if not isinstance(raw.get("expected_backend"), str) or not str(
                raw["expected_backend"]
            ):
                raise MatteQualityError("recorded variants require an expected backend")
            if not isinstance(raw.get("expected_device"), str) or not str(
                raw["expected_device"]
            ):
                raise MatteQualityError("recorded variants require an expected device")
            expected_ratio = raw.get("expected_rvm_downsample_ratio")
            expected_ratio_number = _finite(expected_ratio)
            if expected_ratio is not None and (
                expected_ratio_number is None or not 0.0 < expected_ratio_number <= 1.0
            ):
                raise MatteQualityError("recorded RVM ratio expectation is invalid")
            if raw["expected_backend"] == "RVMSegmenter" and expected_ratio is None:
                raise MatteQualityError(
                    "recorded RVM variants require an expected resolved ratio"
                )
        elif kind == "cadence_projection":
            _validate_schedule(raw.get("schedule"))
            multiplier = raw.get("output_multiplier", 1)
            if type(multiplier) is not int or multiplier not in (1, 2):
                raise MatteQualityError("output cadence multiplier must be 1 or 2")
        else:
            reason = raw.get("availability_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise MatteQualityError("unavailable variant requires a reason")
        normalized.append(normalized_row)
    result = dict(value)
    result["required_axes"] = list(required_axes)
    result["variants"] = normalized
    result["_policy"] = _policy(value.get("decision_policy"))
    return result, _sha256(payload)


def _validate_schedule(value: object) -> None:
    if not isinstance(value, dict):
        raise MatteQualityError("cadence schedule is missing")
    schedule_type = value.get("type")
    if schedule_type == "fixed":
        fps = value.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or not 1.0 <= float(fps) <= 240.0
        ):
            raise MatteQualityError("fixed cadence FPS is invalid")
    elif schedule_type == "decimate":
        factor = value.get("factor")
        if type(factor) is not int or not 2 <= factor <= 16:
            raise MatteQualityError("cadence decimation factor is invalid")
    elif schedule_type == "irregular":
        deltas = value.get("deltas_ns")
        if (
            not isinstance(deltas, list)
            or not deltas
            or any(type(item) is not int or item < 0 for item in deltas)
            or any(current <= previous for previous, current in zip(deltas, deltas[1:]))
        ):
            raise MatteQualityError("irregular cadence deltas are invalid")
    elif schedule_type != "native":
        raise MatteQualityError("cadence schedule type is invalid")


def _source_contract(
    bundle: MatteReplayBundle, *, timestamps: bool
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for frame in bundle.frames:
        artifacts = cast(dict[str, Any], frame["artifacts"])
        descriptor = cast(dict[str, Any], artifacts["raw_frame"])
        entry: dict[str, Any] = {
            "capture_sequence": frame["capture_sequence"],
            "raw_frame_sha256": descriptor["sha256"],
        }
        if timestamps:
            entry["capture_monotonic_ns"] = frame["capture_monotonic_ns"]
        result.append(entry)
    return result


def _source_digest(bundle: MatteReplayBundle, *, timestamps: bool) -> str:
    return _sha256(_json_bytes(_source_contract(bundle, timestamps=timestamps)))


def _artifact_volume(bundle: MatteReplayBundle) -> int:
    total = 0
    for frame in bundle.frames:
        artifacts = cast(dict[str, Any], frame["artifacts"])
        total += sum(
            int(descriptor["bytes"])
            for descriptor in artifacts.values()
            if isinstance(descriptor, dict) and "alias_of" not in descriptor
        )
    return total


def _directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise MatteQualityError("ablation output cannot contain symlinks")
        if path.is_file():
            total += path.stat().st_size
    return total


def _frame_controls(bundle: MatteReplayBundle) -> tuple[dict[str, Any], dict[str, Any]]:
    first_segmentation, first_compositing = _controls(
        cast(dict[str, Any], bundle.frames[0])
    )
    frozen = _json_bytes(
        {"segmentation": first_segmentation, "compositing": first_compositing}
    )
    for frame in bundle.frames[1:]:
        segmentation, compositing = _controls(cast(dict[str, Any], frame))
        if (
            _json_bytes({"segmentation": segmentation, "compositing": compositing})
            != frozen
        ):
            raise MatteQualityError(
                "frozen ablation requires stable recorded matte controls"
            )
    return copy.deepcopy(first_segmentation), copy.deepcopy(first_compositing)


def _clone_annotations(
    source: MatteQualityAnnotations,
    source_bundle: MatteReplayBundle,
    target_bundle: Path,
    target_annotations: Path,
) -> None:
    frames: list[QualityFrameAnnotations] = []
    for sequence, bundle_frame in enumerate(source_bundle.frames):
        refined = source_bundle.load_array(bundle_frame, "refined_mask")
        shape = refined.shape
        source_frame = source.frames[sequence]
        frames.append(
            QualityFrameAnnotations(
                segment=str(source_frame["segment"]),
                registration_from_previous=cast(
                    tuple[float, float, float, float, float, float],
                    tuple(
                        float(item)
                        for item in cast(
                            list[float],
                            source_frame["registration_from_previous"],
                        )
                    ),
                ),
                opaque_core=source.load_array(sequence, "opaque_core", shape=shape),
                background=source.load_array(sequence, "background", shape=shape),
                ground_truth_alpha=source.load_array(
                    sequence, "ground_truth_alpha", shape=shape
                ),
                ground_truth_foreground=source.load_array(
                    sequence, "ground_truth_foreground", shape=shape
                ),
                regions=source.load_regions(sequence, shape=shape),
            )
        )
    provenance = dict(source.provenance)
    provenance["copied_for_ablation"] = True
    write_quality_annotations(
        target_annotations,
        target_bundle,
        frames,
        segments=source.segments,
        provenance=provenance,
        gates=source.gates,
    )


def _frozen_variant(
    source_bundle: MatteReplayBundle,
    source_annotations: MatteQualityAnnotations,
    row: Mapping[str, object],
    row_root: Path,
    *,
    max_bytes: int,
) -> tuple[MatteReplayBundle, MatteQualityAnnotations]:
    segmentation, compositing_controls = _frame_controls(source_bundle)
    postprocess = cast(dict[str, Any], row.get("postprocess", {}))
    compositing_overrides = cast(dict[str, Any], row.get("compositing", {}))
    segmentation.update(postprocess)
    compositing_controls.update(compositing_overrides)
    segmentation_cfg = SegmentationConfig.model_validate(segmentation)
    if row["lane"] == "rvm" and not bool(row.get("explicit_rvm_policy", False)):
        segmentation_cfg = segmentation_cfg.model_copy(
            update={
                "edge_refine": False,
                "mask_blur": 0,
                "temporal_smoothing": 0.0,
            }
        )
    refiner = MaskRefiner(segmentation_cfg) if postprocess else None
    raw_wrap = compositing_controls.get("light_wrap", 0.0)
    raw_foreground = compositing_controls.get("use_model_foreground", False)
    blend_value = compositing_controls.get("blend_space", "srgb_legacy")
    if (
        isinstance(raw_wrap, bool)
        or not isinstance(raw_wrap, (int, float))
        or not math.isfinite(float(raw_wrap))
        or not 0.0 <= float(raw_wrap) <= 1.0
        or type(raw_foreground) is not bool
        or blend_value not in ("srgb_legacy", "linear_srgb")
    ):
        raise MatteQualityError("frozen compositor controls are invalid")
    blend_space = cast(BlendSpace, blend_value)

    bundle_root = row_root / "bundle"
    annotations_root = row_root / "annotations"
    recorder = MatteDiagnosticRecorder(
        bundle_root,
        duration_s=max(
            1.0,
            (
                int(source_bundle.frames[-1]["capture_monotonic_ns"])
                - int(source_bundle.frames[0]["capture_monotonic_ns"])
            )
            / 1_000_000_000.0
            + 1.0,
        ),
        max_bytes=max(max_bytes, MIN_BUNDLE_BYTES),
    )
    try:
        for sequence, frame in enumerate(source_bundle.frames):
            source = source_bundle.load_array(frame, "raw_frame")
            raw_alpha = source_bundle.load_array(frame, "raw_mask").astype(
                np.float32, copy=False
            )
            backdrop = source_bundle.load_array(frame, "backdrop_frame")
            artifacts = cast(dict[str, Any], frame["artifacts"])
            foreground = (
                source_bundle.load_array(frame, "clean_foreground")
                if "clean_foreground" in artifacts
                else None
            )
            refine_started = time.perf_counter_ns()
            refined = (
                np.ascontiguousarray(
                    refiner.refine(raw_alpha, source), dtype=np.float32
                )
                if refiner is not None
                else source_bundle.load_array(frame, "refined_mask").astype(
                    np.float32, copy=False
                )
            )
            refinement_ms = (time.perf_counter_ns() - refine_started) / 1_000_000.0
            composite_started = time.perf_counter_ns()
            rendered = composite(
                source,
                backdrop,
                np.ascontiguousarray(refined),
                light_wrap=float(raw_wrap),
                edge_foreground=(
                    foreground
                    if bool(raw_foreground) and foreground is not None
                    else None
                ),
                blend_space=blend_space,
                color_transform=_recorded_transform(cast(dict[str, Any], frame)),
            )
            composite_ms = (time.perf_counter_ns() - composite_started) / 1_000_000.0
            configured = copy.deepcopy(
                cast(dict[str, Any], frame["configured_controls"])
            )
            configured["segmentation"] = segmentation_cfg.model_dump(mode="json")
            configured["compositing"] = copy.deepcopy(compositing_controls)
            effective = copy.deepcopy(
                cast(dict[str, Any], frame.get("effective_controls", {}))
            )
            effective.update(
                {
                    "mask_shift": segmentation_cfg.mask_shift,
                    "use_model_foreground": bool(raw_foreground),
                    "light_wrap": float(raw_wrap),
                    "blend_space": blend_space,
                    "ablation_model_inference": "recorded-frozen",
                }
            )
            evidence = MatteFrameEvidence(
                metadata=MatteCaptureMetadata(
                    bundle_sequence=sequence,
                    capture_sequence=int(frame["capture_sequence"]),
                    capture_monotonic_ns=int(frame["capture_monotonic_ns"]),
                    timestamp_source=str(frame["timestamp_source"]),
                    capture_generation=int(frame["capture_generation"]),
                    geometry_generation=int(frame["geometry_generation"]),
                ),
                raw_frame=source,
                raw_mask=np.ascontiguousarray(raw_alpha),
                refined_mask=np.ascontiguousarray(refined),
                clean_foreground=foreground,
                backdrop_frame=backdrop,
                base_composite=rendered,
                configured_controls=configured,
                effective_controls=effective,
                timings_ms={
                    "refinement_ms": refinement_ms,
                    "composite_ms": composite_ms,
                    "frame_processing_ms": refinement_ms + composite_ms,
                },
                backdrop_identity=copy.deepcopy(
                    cast(dict[str, Any], frame.get("backdrop_identity", {}))
                ),
                color_transform=_recorded_transform(cast(dict[str, Any], frame)),
            )
            if not recorder.submit(evidence, rendered):
                raise MatteQualityError("frozen ablation recorder stopped early")
            recorder._queue.join()
            if not recorder.submit_output_event(
                sent_monotonic_ns=int(frame["capture_monotonic_ns"]) + 1_000_000,
                source_bundle_sequence=sequence,
                base_updated=True,
                exact_final_repeat=False,
            ):
                raise MatteQualityError("frozen ablation output timeline failed")
    finally:
        recorder.close()
    target_bundle = MatteReplayBundle(bundle_root)
    _clone_annotations(
        source_annotations,
        target_bundle,
        bundle_root,
        annotations_root,
    )
    return target_bundle, MatteQualityAnnotations(annotations_root, target_bundle)


def _quality_summary(report: Mapping[str, object]) -> dict[str, float | None]:
    return {
        name: (
            _round(number)
            if (number := _finite(_metric_path(report, path))) is not None
            else None
        )
        for name, path in _QUALITY_PATHS.items()
    }


def _timing_summary(bundle: MatteReplayBundle) -> dict[str, object]:
    series: dict[str, list[float]] = {}
    warm_up: dict[str, float] = {}
    for sequence, frame in enumerate(bundle.frames):
        combined: dict[str, object] = {}
        combined.update(cast(dict[str, Any], frame.get("timings_ms", {})))
        combined.update(
            {
                f"compositor.{name}": value
                for name, value in cast(
                    dict[str, Any],
                    frame.get("compositor_substages_ms", {}),
                ).items()
            }
        )
        for name, value in combined.items():
            number = _finite(value)
            if number is None:
                continue
            if sequence == 0:
                warm_up[name] = _round(number)
            else:
                series.setdefault(name, []).append(number)
    return {
        "warm_up_first_frame_ms": dict(sorted(warm_up.items())),
        "steady_state_ms": {
            name: _summary(values) for name, values in sorted(series.items())
        },
        "steady_state_frame_count": max(0, len(bundle.frames) - 1),
    }


def _contact_sheet(bundle: MatteReplayBundle, *, label: str) -> np.ndarray:
    images: list[np.ndarray] = []
    for frame in bundle.frames[:6]:
        image = bundle.load_array(frame, "final_composite")
        width = min(240, image.shape[1])
        height = max(1, int(round(image.shape[0] * width / image.shape[1])))
        thumb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        images.append(thumb)
    if not images:
        raise MatteQualityError("contact sheet requires at least one frame")
    cell_height = max(image.shape[0] for image in images) + 28
    cell_width = max(image.shape[1] for image in images)
    columns = min(3, len(images))
    rows = math.ceil(len(images) / columns)
    sheet = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y = row * cell_height
        x = column * cell_width
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
        cv2.putText(
            sheet,
            f"{label} #{index}",
            (x + 4, y + image.shape[0] + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return np.ascontiguousarray(sheet)


def _write_png(path: Path, relative: str, image: np.ndarray) -> dict[str, object]:
    success, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )
    if not success:
        raise MatteQualityError("could not encode ablation contact sheet")
    payload = encoded.tobytes()
    _private_write(path, payload)
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "encoding": "lossless-png",
        "shape": list(image.shape),
    }


def _scheduled_projection(
    source: MatteReplayBundle,
    row: Mapping[str, object],
) -> dict[str, object]:
    schedule = cast(dict[str, Any], row["schedule"])
    schedule_type = str(schedule["type"])
    source_indices = list(range(len(source.frames)))
    original = [int(frame["capture_monotonic_ns"]) for frame in source.frames]
    if schedule_type == "native":
        timestamps = original
    elif schedule_type == "fixed":
        fps = float(schedule["fps"])
        start = original[0]
        timestamps = [
            start + int(round(index * 1_000_000_000.0 / fps))
            for index in source_indices
        ]
    elif schedule_type == "decimate":
        factor = int(schedule["factor"])
        source_indices = source_indices[::factor]
        timestamps = [original[index] for index in source_indices]
    else:
        deltas = cast(list[int], schedule["deltas_ns"])
        if len(deltas) != len(source_indices):
            raise MatteQualityError(
                "irregular cadence schedule must match source frame count"
            )
        start = original[0]
        timestamps = [start + delta for delta in deltas]
    multiplier = cast(int, row.get("output_multiplier", 1))
    span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    unique_fps = (
        _round((len(timestamps) - 1) * 1_000_000_000.0 / span) if span > 0 else None
    )
    output_timestamps = list(timestamps)
    if multiplier == 2 and timestamps:
        intervals = [
            current - previous for previous, current in zip(timestamps, timestamps[1:])
        ]
        final_interval = (
            int(round(float(np.median(intervals))))
            if intervals
            else 1_000_000_000 // 30
        )
        output_timestamps = []
        for index, timestamp in enumerate(timestamps):
            interval = (
                timestamps[index + 1] - timestamp
                if index + 1 < len(timestamps)
                else final_interval
            )
            output_timestamps.extend((timestamp, timestamp + max(1, interval // 2)))
    output_count = len(output_timestamps)
    output_span = (
        output_timestamps[-1] - output_timestamps[0]
        if len(output_timestamps) > 1
        else 0
    )
    output_fps = (
        _round((output_count - 1) * 1_000_000_000.0 / output_span)
        if output_span > 0
        else None
    )
    repeat_count = len(timestamps) * (multiplier - 1)
    alpha_contract = [
        {
            "source_index": index,
            "raw_alpha_sha256": cast(
                dict[str, Any],
                cast(dict[str, Any], source.frames[index]["artifacts"])["raw_mask"],
            )["sha256"],
            "refined_alpha_sha256": cast(
                dict[str, Any],
                cast(dict[str, Any], source.frames[index]["artifacts"])["refined_mask"],
            )["sha256"],
        }
        for index in source_indices
    ]
    return {
        "schedule": copy.deepcopy(schedule),
        "source_indices": source_indices,
        "scheduled_timestamps_ns": timestamps,
        "unique_input_count": len(timestamps),
        "unique_input_fps": unique_fps,
        "model_invocation_count": len(timestamps),
        "output_multiplier": multiplier,
        "output_send_count": output_count,
        "output_send_fps_projection": output_fps,
        "exact_repeat_count": repeat_count,
        "output_repeat_ratio": (
            _round(repeat_count / max(1, output_count - 1)) if output_count > 1 else 0.0
        ),
        "introduces_new_alpha_values": False,
        "alpha_contract_sha256": _sha256(_json_bytes(alpha_contract)),
        "quality_effect": (
            "not evaluated: cadence projection does not rerun a recurrent model"
        ),
    }


def _row_result(
    row: Mapping[str, object],
    bundle: MatteReplayBundle,
    annotations: MatteQualityAnnotations,
    *,
    source: MatteReplayBundle,
    contact_dir: Path,
    metadata: EvaluationMetadata,
) -> dict[str, Any]:
    exact_timestamps = row.get("timestamp_policy", "exact") == "exact"
    pixels_match = _source_digest(bundle, timestamps=False) == _source_digest(
        source, timestamps=False
    )
    timestamps_match = _source_digest(bundle, timestamps=True) == _source_digest(
        source, timestamps=True
    )
    if not pixels_match or (exact_timestamps and not timestamps_match):
        raise MatteQualityError(
            f"variant {row['id']} does not satisfy its source identity contract"
        )
    quality_report = evaluate_bundle(
        bundle.root,
        annotations_root=annotations.root,
        metadata=metadata,
    )
    sheet = _contact_sheet(bundle, label=str(row["id"]))
    relative = f"contact-sheets/{row['id']}.png"
    contact = _write_png(contact_dir / f"{row['id']}.png", relative, sheet)
    configured_segmentation, configured_compositing = _controls(
        cast(dict[str, Any], bundle.frames[0])
    )
    effective = cast(dict[str, Any], bundle.frames[0].get("effective_controls", {}))
    if row["kind"] == "recorded" and row["id"] != "baseline":
        if effective.get("segmentation_backend") != row.get("expected_backend"):
            raise MatteQualityError(
                f"variant {row['id']} effective backend does not match the plan"
            )
        if str(effective.get("segmentation_device", "")) != str(
            row.get("expected_device")
        ):
            raise MatteQualityError(
                f"variant {row['id']} effective device does not match the plan"
            )
        expected_ratio = row.get("expected_rvm_downsample_ratio")
        if expected_ratio is not None:
            actual_ratio = _finite(effective.get("rvm_downsample_ratio"))
            expected_number = _finite(expected_ratio)
            if (
                actual_ratio is None
                or expected_number is None
                or not math.isclose(
                    actual_ratio, expected_number, rel_tol=0.0, abs_tol=1e-6
                )
            ):
                raise MatteQualityError(
                    f"variant {row['id']} effective RVM ratio does not match the plan"
                )
    raw_foreground_retained = all(
        cast(dict[str, Any], frame["artifacts"]).get("clean_foreground") is not None
        for frame in bundle.frames
    )
    return {
        "id": row["id"],
        "kind": row["kind"],
        "lane": row["lane"],
        "group": row["group"],
        "covers": row["covers"],
        "status": "completed",
        "same_source": {
            "pixels_and_order": pixels_match,
            "timestamps": timestamps_match,
            "timestamp_policy": row.get("timestamp_policy", "exact"),
        },
        "source_contract_sha256": _source_digest(bundle, timestamps=exact_timestamps),
        "bundle_manifest_sha256": quality_report["source"]["bundle_manifest_sha256"],
        "annotation_manifest_sha256": quality_report["source"][
            "annotation_manifest_sha256"
        ],
        "quality_evidence_sha256": quality_report["determinism"]["evidence_sha256"],
        "quality": _quality_summary(quality_report),
        "cadence": quality_report["aggregate"]["cadence"],
        "performance": _timing_summary(bundle),
        "track_integrity": {
            "raw_pha_retained": True,
            "post_refiner_alpha_retained": True,
            "clean_foreground_retained": raw_foreground_retained,
            "final_composite_retained": True,
        },
        "attribution": {
            "alpha_motion": _quality_summary(quality_report)[
                "compensated_alpha_temporal_abs_diff"
            ],
            "edge_color_motion": _quality_summary(quality_report)[
                "edge_band_rgb_variation"
            ],
            "metrics_are_separate": True,
        },
        "contact_sheet": contact,
        "quality_gates": quality_report["gates"],
        "configuration": {
            "backend_label": str(
                row.get(
                    "backend_label",
                    effective.get("segmentation_backend", row["lane"]),
                )
            ),
            "device_label": str(
                row.get("device_label", effective.get("segmentation_device", ""))
            ),
            "model_label": str(row.get("model_label", "")),
            "model_path_present": bool(configured_segmentation.get("model_path")),
            "configured_rvm_downsample": configured_segmentation.get("rvm_downsample"),
            "effective_rvm_downsample_ratio": effective.get("rvm_downsample_ratio"),
            "configured_mask_shift": configured_segmentation.get("mask_shift"),
            "effective_mask_shift": effective.get("mask_shift"),
            "edge_refine": configured_segmentation.get("edge_refine"),
            "mask_blur": configured_segmentation.get("mask_blur"),
            "temporal_smoothing": configured_segmentation.get("temporal_smoothing"),
            "use_model_foreground": configured_compositing.get("use_model_foreground"),
            "light_wrap": configured_compositing.get("light_wrap"),
            "blend_space": configured_compositing.get("blend_space"),
        },
        "model_inference": (
            "recorded-model-backed"
            if row["kind"] == "recorded" and row.get("evidence_kind") == "model-backed"
            else (
                "not-run-generated-proxy"
                if row.get("evidence_kind") == "generated-proxy"
                else "not-rerun-recorded-intermediates"
            )
        ),
        "evidence_kind": row.get(
            "evidence_kind",
            "model-backed" if row["kind"] == "recorded" else "recorded-intermediate",
        ),
        "variant_controls": {
            "postprocess": copy.deepcopy(row.get("postprocess", {})),
            "compositing": copy.deepcopy(row.get("compositing", {})),
        },
        "reference_in_group": bool(row.get("reference_in_group", False)),
        "experimental_complexity": bool(row.get("experimental_complexity", False)),
    }


def _value(row: Mapping[str, object], name: str) -> float | None:
    quality = row.get("quality")
    if not isinstance(quality, Mapping):
        return None
    return _finite(quality.get(name))


def _steady_frame_p95(row: Mapping[str, object]) -> float | None:
    performance = row.get("performance")
    if not isinstance(performance, Mapping):
        return None
    steady = performance.get("steady_state_ms")
    if not isinstance(steady, Mapping):
        return None
    frame = steady.get("frame_processing_ms")
    if not isinstance(frame, Mapping):
        return None
    return _finite(frame.get("p95"))


def _steady_p95(row: Mapping[str, object], name: str) -> float | None:
    performance = row.get("performance")
    if not isinstance(performance, Mapping):
        return None
    steady = performance.get("steady_state_ms")
    if not isinstance(steady, Mapping):
        return None
    metric = steady.get(name)
    if not isinstance(metric, Mapping):
        return None
    return _finite(metric.get("p95"))


def _watershed_decision(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    off = next(
        (
            row
            for row in rows
            if "mediapipe.edge_refine.off" in cast(Sequence[str], row.get("covers", []))
            and row.get("status") == "completed"
        ),
        None,
    )
    on = next(
        (
            row
            for row in rows
            if "mediapipe.edge_refine.on" in cast(Sequence[str], row.get("covers", []))
            and row.get("status") == "completed"
        ),
        None,
    )
    if off is None or on is None:
        return {
            "status": "not_decidable",
            "reason": "completed MediaPipe edge-refine off/on rows are required",
        }
    off_contour = _value(off, "compensated_contour_displacement_p95_px")
    on_contour = _value(on, "compensated_contour_displacement_p95_px")
    if off_contour is None or on_contour is None:
        outcome = "not_decidable"
    elif on_contour < off_contour:
        outcome = "improves"
    elif on_contour > off_contour:
        outcome = "worsens"
    else:
        outcome = "no_measured_change"
    model_backed = (
        off.get("evidence_kind") == "model-backed"
        and on.get("evidence_kind") == "model-backed"
    )
    return {
        "status": "decided" if model_backed else "proxy_only_real_decision_pending",
        "outcome": outcome,
        "off_contour_p95_px": off_contour,
        "on_contour_p95_px": on_contour,
        "same_source": bool(
            cast(dict[str, Any], off["same_source"])["pixels_and_order"]
            and cast(dict[str, Any], on["same_source"])["pixels_and_order"]
        ),
    }


def _factorial_analysis(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    combinations: dict[tuple[bool, float], Mapping[str, object]] = {}
    for row in rows:
        if (
            row.get("group") != "compositor_factorial"
            or row.get("status") != "completed"
        ):
            continue
        controls = cast(
            dict[str, Any],
            cast(dict[str, Any], row["variant_controls"])["compositing"],
        )
        combinations[
            (
                bool(controls["use_model_foreground"]),
                _round(float(controls["light_wrap"])),
            )
        ] = row
    plain = combinations.get((False, 0.0))
    results: list[dict[str, object]] = []
    for (foreground, wrap), row in sorted(combinations.items()):
        edge = _value(row, "edge_band_rgb_variation")
        leakage = _value(row, "opaque_backdrop_leakage_coefficient")
        composite_cost = _steady_p95(row, "composite_ms")
        results.append(
            {
                "id": row["id"],
                "use_model_foreground": foreground,
                "light_wrap": wrap,
                "edge_band_rgb_variation": edge,
                "opaque_backdrop_leakage_coefficient": leakage,
                "composite_ms_p95": composite_cost,
                "edge_delta_vs_plain": (
                    _round(edge - cast(float, _value(plain, "edge_band_rgb_variation")))
                    if edge is not None
                    and plain is not None
                    and _value(plain, "edge_band_rgb_variation") is not None
                    else None
                ),
                "composite_cost_delta_vs_plain_ms": (
                    _round(
                        composite_cost - cast(float, _steady_p95(plain, "composite_ms"))
                    )
                    if composite_cost is not None
                    and plain is not None
                    and _steady_p95(plain, "composite_ms") is not None
                    else None
                ),
            }
        )
    return {
        "complete": len(combinations) == 4,
        "combinations": results,
        "alpha_motion_held_by_recorded_mask": True,
        "quality_and_runtime_reported_separately": True,
    }


def _rejection_reasons(
    row: Mapping[str, object],
    baseline: Mapping[str, object],
    policy: AblationPolicy,
) -> tuple[list[dict[str, str]], list[str]]:
    reasons: list[dict[str, str]] = []
    improvements: list[str] = []
    if row.get("status") == "unavailable":
        return (
            [
                {
                    "category": "platform availability",
                    "detail": str(row["availability_reason"]),
                }
            ],
            improvements,
        )
    if row.get("kind") == "cadence_projection":
        return (
            [
                {
                    "category": "complexity",
                    "detail": (
                        "cadence projection is diagnostic and cannot qualify "
                        "a model or alpha policy"
                    ),
                }
            ],
            improvements,
        )
    opaque = _value(row, "opaque_core_alpha_p05")
    if opaque is not None and opaque < policy.opaque_alpha_p05:
        reasons.append(
            {
                "category": "detail loss",
                "detail": "opaque-core alpha gate failed",
            }
        )
    holes = _value(row, "foreground_hole_components")
    if holes is not None and holes > 0.0:
        reasons.append(
            {"category": "detail loss", "detail": "foreground holes were introduced"}
        )
    background = _value(row, "background_alpha_mean")
    if background is not None and background > policy.background_alpha_mean:
        reasons.append(
            {"category": "detail loss", "detail": "exterior halo gate failed"}
        )
    for metric, category in (
        ("compensated_contour_displacement_p95_px", "jitter"),
        ("motion_trail_area_ratio", "ghosting"),
        ("ground_truth_alpha_mse", "detail loss"),
        ("ground_truth_gradient_mae", "detail loss"),
    ):
        actual = _value(row, metric)
        reference = _value(baseline, metric)
        tolerance = (
            policy.trail_absolute_tolerance
            if metric == "motion_trail_area_ratio"
            else 1e-6
        )
        if (
            actual is not None
            and reference is not None
            and actual > reference * policy.relative_nonregression + tolerance
        ):
            reasons.append(
                {
                    "category": category,
                    "detail": f"{metric} regressed beyond the screening tolerance",
                }
            )
    frame_p95 = _steady_frame_p95(row)
    if (
        policy.performance_budget_ms is not None
        and frame_p95 is not None
        and frame_p95 > policy.performance_budget_ms
    ):
        reasons.append(
            {
                "category": "performance",
                "detail": "steady-state frame p95 exceeds the host screening budget",
            }
        )
    for metric, ratio, label in (
        (
            "compensated_contour_displacement_p95_px",
            policy.contour_improvement_ratio,
            "contour jitter",
        ),
        (
            "edge_band_rgb_variation",
            policy.edge_improvement_ratio,
            "edge-color motion",
        ),
    ):
        actual = _value(row, metric)
        reference = _value(baseline, metric)
        if (
            actual is not None
            and reference is not None
            and reference > 0.0
            and actual <= reference * ratio
        ):
            improvements.append(label)
    actual_leakage = _value(row, "opaque_backdrop_leakage_coefficient")
    baseline_leakage = _value(baseline, "opaque_backdrop_leakage_coefficient")
    if (
        actual_leakage is not None
        and baseline_leakage is not None
        and baseline_leakage - actual_leakage >= policy.leakage_absolute_improvement
    ):
        improvements.append("opaque backdrop leakage")
    actual_gt = _value(row, "ground_truth_alpha_mse")
    baseline_gt = _value(baseline, "ground_truth_alpha_mse")
    if (
        actual_gt is not None
        and baseline_gt is not None
        and baseline_gt > 0.0
        and actual_gt <= baseline_gt * 0.90
    ):
        improvements.append("ground-truth alpha error")
    baseline_frame = _steady_frame_p95(baseline)
    if (
        frame_p95 is not None
        and baseline_frame is not None
        and row.get("model_inference") == baseline.get("model_inference")
        and baseline_frame > 0.0
        and frame_p95 <= baseline_frame * policy.performance_improvement_ratio
    ):
        improvements.append("frame processing cost")
    if (
        not improvements
        and bool(row.get("experimental_complexity", False))
        and not reasons
    ):
        reasons.append(
            {
                "category": "complexity",
                "detail": "experimental policy adds complexity without material gain",
            }
        )
    if not improvements and not reasons:
        reasons.append(
            {
                "category": "complexity",
                "detail": "no material quality or performance improvement",
            }
        )
    return reasons, improvements


def _decisions(
    rows: list[dict[str, Any]],
    policy: AblationPolicy,
) -> dict[str, object]:
    baseline = rows[0]
    group_references: dict[str, Mapping[str, object]] = {}
    for row in rows[1:]:
        if bool(row.get("reference_in_group", False)):
            group_references[str(row["group"])] = row
    factorial_rows = [
        row
        for row in rows[1:]
        if row.get("group") == "compositor_factorial"
        and row.get("status") == "completed"
    ]
    if "compositor_factorial" not in group_references and factorial_rows:
        both = [
            row
            for row in factorial_rows
            if cast(dict[str, Any], row["variant_controls"])["compositing"].get(
                "use_model_foreground"
            )
            and float(
                cast(dict[str, Any], row["variant_controls"])["compositing"].get(
                    "light_wrap", 0.0
                )
            )
            > 0.0
        ]
        if both:
            group_references["compositor_factorial"] = both[0]
    eligible: dict[str, list[dict[str, Any]]] = {}
    rejected: list[dict[str, object]] = []
    for row in rows[1:]:
        reference = group_references.get(str(row["group"]), baseline)
        reasons, improvements = _rejection_reasons(row, reference, policy)
        row["screening"] = {
            "reference_row": reference["id"],
            "material_improvements": improvements,
            "rejection_reasons": reasons,
            "eligible_for_shortlist": not reasons,
        }
        if reasons:
            rejected.append(
                {
                    "id": row["id"],
                    "lane": row["lane"],
                    "reasons": reasons,
                }
            )
        else:
            eligible.setdefault(str(row["lane"]), []).append(row)
    shortlists: dict[str, list[str]] = {}
    for lane, candidates in eligible.items():
        candidates.sort(
            key=lambda row: (
                -len(row["screening"]["material_improvements"]),
                _steady_frame_p95(row)
                if _steady_frame_p95(row) is not None
                else math.inf,
                str(row["id"]),
            )
        )
        chosen = candidates[: policy.shortlist_limit_per_lane]
        shortlists[lane] = [str(row["id"]) for row in chosen]
        for row in candidates[policy.shortlist_limit_per_lane :]:
            reason = {
                "category": "complexity",
                "detail": "bounded shortlist limit reached by stronger candidates",
            }
            row["screening"]["eligible_for_shortlist"] = False
            row["screening"]["rejection_reasons"].append(reason)
            rejected.append({"id": row["id"], "lane": lane, "reasons": [reason]})
    rvm = shortlists.get("rvm", [])
    mediapipe = shortlists.get("mediapipe", [])
    compositor = shortlists.get("compositor", [])
    return {
        "policy": asdict(policy),
        "shortlists": shortlists,
        "matte_2_5_rvm_candidates": rvm,
        "matte_3_4_cost_candidates": compositor,
        "first_fix_by_lane": {
            "rvm_active_lane": (
                rvm[0] if rvm else "no screened RVM candidate; retain current policy"
            ),
            "mediapipe_fallback_lane": (
                mediapipe[0]
                if mediapipe
                else "no screened MediaPipe candidate; retain current policy"
            ),
        },
        "rejected_candidates": rejected,
        "one_host_screening_is_not_a_production_preset": True,
    }


def _factorial_coverage(
    variants: Sequence[Mapping[str, object]],
    baseline_wrap: float,
) -> dict[str, object]:
    expected = {
        (False, 0.0),
        (True, 0.0),
        (False, _round(baseline_wrap)),
        (True, _round(baseline_wrap)),
    }
    observed: set[tuple[bool, float]] = set()
    for row in variants:
        if row.get("group") != "compositor_factorial" or row.get("kind") != "frozen":
            continue
        controls = cast(dict[str, Any], row.get("compositing", {}))
        observed.add(
            (
                bool(controls["use_model_foreground"]),
                _round(float(controls["light_wrap"])),
            )
        )
    return {
        "expected": [
            {"use_model_foreground": foreground, "light_wrap": wrap}
            for foreground, wrap in sorted(expected)
        ],
        "observed": [
            {"use_model_foreground": foreground, "light_wrap": wrap}
            for foreground, wrap in sorted(observed)
        ],
        "complete": observed == expected,
    }


def run_ablation(
    source_bundle_root: Path | str,
    source_annotations_root: Path | str,
    plan_path: Path | str,
    output_root: Path | str,
    *,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> dict[str, Any]:
    """Execute a bounded ablation plan and write private canonical evidence."""

    if type(max_output_bytes) is not int or max_output_bytes < MIN_BUNDLE_BYTES:
        raise MatteQualityError(
            f"ablation max bytes must be at least {MIN_BUNDLE_BYTES}"
        )
    plan, plan_sha256 = load_plan(plan_path)
    policy = cast(AblationPolicy, plan.pop("_policy"))
    source_bundle = MatteReplayBundle(source_bundle_root)
    source_annotations = MatteQualityAnnotations(source_annotations_root, source_bundle)
    if not source_bundle.frames or not bool(
        source_bundle.manifest.get("matte_metrics_authoritative")
    ):
        raise MatteQualityError(
            "ablation requires a full metric-authoritative source bundle"
        )
    derived_count = sum(
        row["kind"] == "frozen" for row in cast(list[dict[str, Any]], plan["variants"])
    )
    estimated = (
        derived_count * (_artifact_volume(source_bundle) + 2 * 1024 * 1024)
        + 4 * 1024 * 1024
    )
    if estimated > max_output_bytes:
        raise MatteQualityError(
            "ablation plan exceeds the preflight output byte estimate"
        )
    output = Path(output_root)
    _private_directory(output, create=True)
    variants_dir = output / "variants"
    contacts_dir = output / "contact-sheets"
    _private_directory(variants_dir, create=True)
    _private_directory(contacts_dir, create=True)

    scope = cast(dict[str, Any], plan["scope"])
    baseline_row: dict[str, Any] = {
        "id": "baseline",
        "kind": "recorded",
        "lane": str(scope.get("baseline_lane", "rvm")),
        "group": "baseline",
        "covers": [],
        "timestamp_policy": "exact",
        "evidence_kind": (
            "model-backed"
            if bool(scope.get("model_inference_executed", False))
            else "generated-proxy"
        ),
    }
    metadata = EvaluationMetadata(
        hardware_label=str(scope.get("host_label", "")),
        backend=str(
            cast(
                dict[str, Any], source_bundle.frames[0].get("effective_controls", {})
            ).get("segmentation_backend", "")
        ),
        device=str(
            cast(
                dict[str, Any], source_bundle.frames[0].get("effective_controls", {})
            ).get("segmentation_device", "")
        ),
        configuration_label="baseline",
        notes="MATTE-0.3 one-host screening baseline",
    )
    rows: list[dict[str, Any]] = [
        _row_result(
            baseline_row,
            source_bundle,
            source_annotations,
            source=source_bundle,
            contact_dir=contacts_dir,
            metadata=metadata,
        )
    ]
    completed_axes: set[str] = set()
    attempted_axes: set[str] = set()
    for row in cast(list[dict[str, Any]], plan["variants"]):
        attempted_axes.update(cast(list[str], row["covers"]))
        kind = cast(VariantKind, row["kind"])
        if kind == "unavailable":
            result = {
                "id": row["id"],
                "kind": kind,
                "lane": row["lane"],
                "group": row["group"],
                "covers": row["covers"],
                "status": "unavailable",
                "availability_reason": row["availability_reason"],
                "fallback_substituted": False,
            }
        elif kind == "cadence_projection":
            result = {
                "id": row["id"],
                "kind": kind,
                "lane": row["lane"],
                "group": row["group"],
                "covers": row["covers"],
                "status": "completed",
                "same_source": {
                    "pixels_and_order": True,
                    "timestamps": row["schedule"]["type"] == "native",
                    "timestamp_policy": "scheduled-projection",
                },
                "cadence_projection": _scheduled_projection(source_bundle, row),
                "contact_sheet": {"alias_of": "baseline"},
                "model_inference": "not-run-projection-only",
                "experimental_complexity": False,
            }
            completed_axes.update(cast(list[str], row["covers"]))
        elif kind == "recorded":
            candidate_bundle = MatteReplayBundle(str(row["bundle"]))
            candidate_annotations = MatteQualityAnnotations(
                str(row["annotations"]), candidate_bundle
            )
            result = _row_result(
                row,
                candidate_bundle,
                candidate_annotations,
                source=source_bundle,
                contact_dir=contacts_dir,
                metadata=EvaluationMetadata(
                    hardware_label=str(scope.get("host_label", "")),
                    backend=str(row.get("backend_label", row["lane"])),
                    device=str(row.get("device_label", "")),
                    configuration_label=str(row["id"]),
                    notes="separately recorded same-source MATTE-0.3 row",
                ),
            )
            completed_axes.update(cast(list[str], row["covers"]))
        else:
            row_root = variants_dir / str(row["id"])
            _private_directory(row_root, create=True)
            remaining = max_output_bytes - _directory_bytes(output)
            candidate_bundle, candidate_annotations = _frozen_variant(
                source_bundle,
                source_annotations,
                row,
                row_root,
                max_bytes=remaining,
            )
            result = _row_result(
                row,
                candidate_bundle,
                candidate_annotations,
                source=source_bundle,
                contact_dir=contacts_dir,
                metadata=EvaluationMetadata(
                    hardware_label=str(scope.get("host_label", "")),
                    backend=f"{row['lane']}-frozen",
                    device="recorded",
                    configuration_label=str(row["id"]),
                    notes="model inference not rerun; recorded intermediates frozen",
                ),
            )
            completed_axes.update(cast(list[str], row["covers"]))
        rows.append(result)
        if _directory_bytes(output) > max_output_bytes:
            raise MatteQualityError("ablation output byte bound reached")

    _segmentation, baseline_compositing = _frame_controls(source_bundle)
    baseline_wrap_value = baseline_compositing.get("light_wrap", 0.0)
    if not isinstance(baseline_wrap_value, (int, float)) or isinstance(
        baseline_wrap_value, bool
    ):
        raise MatteQualityError("recorded baseline light wrap is invalid")
    required_axes = set(cast(list[str], plan["required_axes"]))
    decisions = _decisions(rows, policy)
    model_backed_rows = [
        row
        for row in rows
        if row.get("status") == "completed"
        and row.get("evidence_kind") == "model-backed"
    ]
    model_backed_cadence_axes = {
        axis
        for row in model_backed_rows
        if row.get("kind") == "recorded"
        for axis in cast(Sequence[str], row.get("covers", []))
        if axis.startswith("cadence.")
    }
    recurrent_cadence_required_axes = {
        axis
        for axis in required_axes
        if axis.startswith("cadence.") and "output" not in axis
    }
    cadence_rows = [row for row in rows if row.get("kind") == "cadence_projection"]
    repeat_rows = [
        row
        for row in cadence_rows
        if cast(dict[str, Any], row["cadence_projection"])["output_multiplier"] == 2
    ]
    watershed = _watershed_decision(rows)
    compositor_factorial = _factorial_analysis(rows)
    baseline_alpha_motion = _value(rows[0], "compensated_alpha_temporal_abs_diff")
    baseline_edge_motion = _value(rows[0], "edge_band_rgb_variation")
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "source": {
            "source_contract_sha256": _source_digest(source_bundle, timestamps=True),
            "source_pixel_order_sha256": _source_digest(
                source_bundle, timestamps=False
            ),
            "annotation_manifest_sha256": source_annotations.manifest_sha256,
            "plan_sha256": plan_sha256,
            "annotation_provenance": source_annotations.provenance,
        },
        "scope": {
            **scope,
            "one_host_screening_is_not_cross_device_qualification": True,
            "production_preset_selected": False,
            "reported_screenshots_used_as_ab": False,
        },
        "privacy": {
            "output_is_owner_only": True,
            "contains_identifiable_derived_images": True,
            "network_or_live_devices_opened_by_runner": False,
            "max_output_bytes": max_output_bytes,
            "bytes_written_before_report": _directory_bytes(output),
        },
        "coverage": {
            "required_axes": sorted(required_axes),
            "attempted_axes": sorted(attempted_axes),
            "completed_axes": sorted(completed_axes),
            "missing_attempted_axes": sorted(required_axes - attempted_axes),
            "uncompleted_required_axes": sorted(required_axes - completed_axes),
            "compositor_factorial": _factorial_coverage(
                cast(list[dict[str, Any]], plan["variants"]),
                float(baseline_wrap_value),
            ),
            "model_backed": {
                "completed_rows": [row["id"] for row in model_backed_rows],
                "rvm_rows": [
                    row["id"] for row in model_backed_rows if row["lane"] == "rvm"
                ],
                "mediapipe_rows": [
                    row["id"] for row in model_backed_rows if row["lane"] == "mediapipe"
                ],
                "recurrent_cadence_required_axes": sorted(
                    recurrent_cadence_required_axes
                ),
                "recurrent_cadence_completed_axes": sorted(
                    model_backed_cadence_axes & recurrent_cadence_required_axes
                ),
                "recurrent_cadence_uncompleted_axes": sorted(
                    recurrent_cadence_required_axes - model_backed_cadence_axes
                ),
                "recurrent_cadence_complete": not (
                    recurrent_cadence_required_axes - model_backed_cadence_axes
                ),
            },
        },
        "rows": rows,
        "decisions": decisions,
        "cadence_conclusion": {
            "repeat_projection_rows": [row["id"] for row in repeat_rows],
            "new_alpha_values_created": False,
            "statement": (
                "15-to-30 output repetition increases sends, not unique inputs "
                "or alpha/model invocations."
            ),
        },
        "required_decisions": {
            "watershed_mediapipe": watershed,
            "alpha_motion_vs_edge_color_motion": {
                "baseline_compensated_alpha_abs_diff": baseline_alpha_motion,
                "baseline_edge_band_rgb_variation": baseline_edge_motion,
                "units_are_not_combined_into_one_score": True,
                "per_variant_values": [
                    {
                        "id": row["id"],
                        "alpha_motion": cast(
                            dict[str, Any], row.get("attribution", {})
                        ).get("alpha_motion"),
                        "edge_color_motion": cast(
                            dict[str, Any], row.get("attribution", {})
                        ).get("edge_color_motion"),
                    }
                    for row in rows
                    if row.get("status") == "completed"
                    and row.get("kind") != "cadence_projection"
                ],
            },
            "compositor_factorial": compositor_factorial,
            "rvm_cross_device_candidates": decisions["matte_2_5_rvm_candidates"],
            "default_compositor_cost_candidates": decisions[
                "matte_3_4_cost_candidates"
            ],
            "separate_lane_first_fixes": decisions["first_fix_by_lane"],
        },
    }
    evidence = {
        "source": report["source"],
        "scope": report["scope"],
        "coverage": report["coverage"],
        "rows": rows,
        "decisions": decisions,
        "cadence_conclusion": report["cadence_conclusion"],
        "required_decisions": report["required_decisions"],
    }
    report["evidence_sha256"] = _sha256(_json_bytes(evidence))
    json_payload = _json_bytes(report)
    markdown_payload = report_markdown(report).encode("utf-8")
    if (
        _directory_bytes(output) + len(json_payload) + len(markdown_payload)
        > max_output_bytes
    ):
        raise MatteQualityError("ablation output byte bound reached")
    _atomic_private_write(output / "ablation.json", json_payload)
    _atomic_private_write(output / "ablation.md", markdown_payload)
    return report


def report_markdown(report: Mapping[str, object]) -> str:
    """Render a compact review companion; JSON remains authoritative."""

    scope = cast(dict[str, Any], report["scope"])
    coverage = cast(dict[str, Any], report["coverage"])
    decisions = cast(dict[str, Any], report["decisions"])
    rows = cast(list[dict[str, Any]], report["rows"])
    lines = [
        "# Matte ablation screening report",
        "",
        f"- Evidence SHA-256: `{report['evidence_sha256']}`",
        f"- Host: `{scope.get('host_label', '')}`",
        "- Classification: **one-host screening; not a production preset**",
        f"- Completed axes: `{coverage['completed_axes']}`",
        f"- Uncompleted required axes: `{coverage['uncompleted_required_axes']}`",
        "",
        "## Rows",
        "",
        "| Row | Lane | Kind | Status | Opaque p05 | Edge motion | Frame p95 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        quality = cast(dict[str, Any], row.get("quality", {}))
        lines.append(
            f"| `{row['id']}` | `{row['lane']}` | `{row['kind']}` | "
            f"`{row['status']}` | `{quality.get('opaque_core_alpha_p05')}` | "
            f"`{quality.get('edge_band_rgb_variation')}` | "
            f"`{_steady_frame_p95(row)}` |"
        )
    lines.extend(
        [
            "",
            "## Bounded shortlists",
            "",
            f"- MATTE-2.5 RVM: `{decisions['matte_2_5_rvm_candidates']}`",
            f"- MATTE-3.4 cost: `{decisions['matte_3_4_cost_candidates']}`",
            f"- First fixes: `{decisions['first_fix_by_lane']}`",
            "",
            "Rejected rows and their jitter, ghosting, detail-loss, performance, "
            "platform-availability, or complexity reasons are authoritative in "
            "`ablation.json`. Contact sheets contain private derived imagery.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser(*, prog: str = "custback matte-ablate") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a bounded same-source matte ablation screen",
    )
    parser.add_argument("bundle", help="full source replay bundle")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--plan", required=True, help="owner-only ablation plan JSON")
    parser.add_argument(
        "--output", required=True, help="new owner-only output directory"
    )
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=DEFAULT_MAX_OUTPUT_BYTES,
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback matte-ablate",
) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        report = run_ablation(
            args.bundle,
            args.annotations,
            args.plan,
            args.output,
            max_output_bytes=args.max_output_bytes,
        )
    except (
        OSError,
        json.JSONDecodeError,
        MatteDiagnosticsError,
        MatteQualityError,
    ) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2
    missing = report["coverage"]["uncompleted_required_axes"]
    print(
        f"screened {len(report['rows']) - 1} variant(s); "
        f"{len(missing)} required axis/axes incomplete"
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
