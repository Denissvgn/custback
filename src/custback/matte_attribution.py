"""Offline RVM alpha/compositor defect attribution for private replay bundles.

The evaluator holds the recorded source, backdrop, alpha tracks, clean
foreground, color transform, and compositor policy fixed while rendering a
small factorial of counterfactuals.  It never opens capture, a model, an
output, or the network.  Review images remain in a new owner-only directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import cv2
import numpy as np

from .compositor import composite
from .config import BlendSpace
from .matte_diagnostics import (
    DEFAULT_MAX_BYTES,
    MIN_BUNDLE_BYTES,
    MatteDiagnosticsError,
    MatteReplayBundle,
    _atomic_private_write,
    _controls,
    _json_bytes,
    _private_directory,
    _private_write,
    _recorded_transform,
)
from .matte_quality import (
    MatteQualityAnnotations,
    MatteQualityError,
    QualityNamedRegion,
    RegionKind,
    _bundle_manifest_digest,
    _component_count,
    _round,
)

REPORT_SCHEMA = "custback.matte-alpha-attribution"
REPORT_VERSION = 1
_REQUIRED_ANATOMY = ("torso", "shoulders", "head", "headphones", "hair")


@dataclass(frozen=True)
class AttributionThresholds:
    """Recorded decision thresholds for diagnostic classification.

    These thresholds classify evidence; they are not production alpha policy
    and are not release gates.
    """

    opaque_alpha_p05: float = 0.95
    opaque_fraction_below_0_95: float = 0.05
    background_alpha_mean: float = 0.01
    stage_alpha_delta: float = 0.01
    visual_delta: float = 0.01
    foreground_error: float = 0.03
    final_channel_delta: int = 1
    candidate_nonregression: float = 0.01


def _validate_thresholds(value: AttributionThresholds) -> None:
    for name, threshold in asdict(value).items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise MatteQualityError(f"attribution threshold {name} is invalid")
        if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
            raise MatteQualityError(f"attribution threshold {name} is invalid")


def _image_payload(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".png",
        np.ascontiguousarray(image),
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )
    if not success:
        raise MatteQualityError("could not encode attribution review image")
    return encoded.tobytes()


def _descriptor(path: str, payload: bytes, image: np.ndarray) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "encoding": "lossless-png",
        "shape": list(image.shape),
    }


class _PrivateImageWriter:
    def __init__(self, root: Path, max_bytes: int):
        if type(max_bytes) is not int or max_bytes < MIN_BUNDLE_BYTES:
            raise MatteQualityError(
                f"attribution max bytes must be at least {MIN_BUNDLE_BYTES}"
            )
        self.root = root
        self.max_bytes = max_bytes
        self.bytes_written = 0

    def write(
        self,
        directory: Path,
        relative: str,
        image: np.ndarray,
    ) -> dict[str, object]:
        payload = _image_payload(image)
        if self.bytes_written + len(payload) > self.max_bytes:
            raise MatteQualityError("attribution output byte bound reached")
        _private_write(directory / Path(relative).name, payload)
        self.bytes_written += len(payload)
        return _descriptor(relative, payload, image)


def _heatmap(value: np.ndarray, region: np.ndarray) -> np.ndarray:
    normalized = np.clip(value.astype(np.float32), 0.0, 1.0)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~region.astype(bool)] = 0
    return np.ascontiguousarray(colored)


def _rgb_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (
        np.mean(
            np.abs(first.astype(np.float32) - second.astype(np.float32)),
            axis=2,
            dtype=np.float32,
        )
        / 255.0
    )


def _leakage_map(
    source: np.ndarray,
    backdrop: np.ndarray,
    rendered: np.ndarray,
) -> np.ndarray:
    source_f = source.astype(np.float32)
    direction = backdrop.astype(np.float32) - source_f
    change = rendered.astype(np.float32) - source_f
    denominator = np.sum(direction * direction, axis=2, dtype=np.float32)
    numerator = np.sum(change * direction, axis=2, dtype=np.float32)
    return np.clip(
        np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-6,
        ),
        0.0,
        1.0,
    )


def _alpha_error_map(
    alpha: np.ndarray,
    kind: RegionKind,
    ground_truth: np.ndarray | None,
) -> np.ndarray:
    if kind == "opaque_core":
        return 1.0 - alpha
    if kind == "background":
        return alpha
    if ground_truth is not None:
        return np.abs(alpha - ground_truth)
    # Without ground truth, show the soft-alpha support without pretending it
    # is an error. The report labels ground-truth availability separately.
    return 4.0 * alpha * (1.0 - alpha)


def _alpha_metrics(
    alpha: np.ndarray,
    region: QualityNamedRegion,
    ground_truth: np.ndarray | None,
) -> dict[str, float | int | None]:
    selected = region.mask.astype(bool)
    values = alpha[selected].astype(np.float64)
    result: dict[str, float | int | None] = {
        "alpha_p05": _round(float(np.percentile(values, 5))),
        "alpha_p50": _round(float(np.percentile(values, 50))),
        "alpha_mean": _round(float(np.mean(values))),
        "alpha_mass": _round(float(np.sum(values))),
        "fraction_below_0_95": _round(float(np.mean(values < 0.95))),
        "fraction_below_0_90": _round(float(np.mean(values < 0.90))),
        "fraction_above_0_05": _round(float(np.mean(values > 0.05))),
        "uncertain_fraction": _round(float(np.mean((values > 0.05) & (values < 0.95)))),
        "mean_deficit": _round(float(np.mean(1.0 - values))),
        "hole_components": _component_count(selected & (alpha < 0.5)),
        "unexpected_foreground_components": _component_count(selected & (alpha >= 0.5)),
        "ground_truth_mae": None,
        "exterior_halo_width_p95_px": None,
    }
    if ground_truth is not None:
        result["ground_truth_mae"] = _round(
            float(
                np.mean(
                    np.abs(
                        alpha[selected].astype(np.float64)
                        - ground_truth[selected].astype(np.float64)
                    )
                )
            )
        )
        false_foreground = selected & (alpha > 0.05) & (ground_truth <= 0.05)
        if bool(np.any(false_foreground)):
            exterior = (ground_truth <= 0.05).astype(np.uint8)
            distance = cv2.distanceTransform(exterior, cv2.DIST_L2, cv2.DIST_MASK_5)
            result["exterior_halo_width_p95_px"] = _round(
                float(np.percentile(distance[false_foreground], 95))
            )
    return result


def _visual_metrics(
    source: np.ndarray,
    backdrop: np.ndarray,
    rendered: np.ndarray,
    region: np.ndarray,
) -> dict[str, float]:
    selected = region.astype(bool)
    source_pixels = source[selected].astype(np.float64)
    backdrop_pixels = backdrop[selected].astype(np.float64)
    rendered_pixels = rendered[selected].astype(np.float64)
    direction = backdrop_pixels - source_pixels
    change = rendered_pixels - source_pixels
    denominator = float(np.sum(direction * direction, dtype=np.float64))
    leakage = (
        float(np.sum(change * direction, dtype=np.float64)) / denominator
        if denominator > 0.0
        else 0.0
    )
    return {
        "source_mae": _round(
            float(np.mean(np.abs(rendered_pixels - source_pixels))) / 255.0
        ),
        "backdrop_leakage_coefficient": _round(leakage),
    }


def _region_mean_delta(
    first: np.ndarray,
    second: np.ndarray,
    region: np.ndarray,
) -> float:
    selected = region.astype(bool)
    return _round(float(np.mean(_rgb_delta(first, second)[selected])))


def _region_max_channel_delta(
    first: np.ndarray,
    second: np.ndarray,
    region: np.ndarray,
) -> int:
    selected = region.astype(bool)
    delta = np.abs(first[selected].astype(np.int16) - second[selected].astype(np.int16))
    return int(delta.max(initial=0))


def _foreground_error(
    foreground: np.ndarray | None,
    source: np.ndarray,
    ground_truth_foreground: np.ndarray | None,
    region: np.ndarray,
) -> tuple[float | None, str]:
    if foreground is None:
        return None, "unavailable"
    reference = (
        ground_truth_foreground if ground_truth_foreground is not None else source
    )
    return (
        _region_mean_delta(foreground, reference, region),
        (
            "ground_truth_foreground"
            if ground_truth_foreground is not None
            else "recorded_source_proxy"
        ),
    )


def _is_alpha_failure(
    metrics: Mapping[str, object],
    kind: RegionKind,
    thresholds: AttributionThresholds,
) -> bool:
    if kind == "opaque_core":
        return (
            float(cast(float, metrics["alpha_p05"])) < thresholds.opaque_alpha_p05
            or float(cast(float, metrics["fraction_below_0_95"]))
            > thresholds.opaque_fraction_below_0_95
            or int(cast(int, metrics["hole_components"])) > 0
        )
    if kind == "background":
        return (
            float(cast(float, metrics["alpha_mean"])) > thresholds.background_alpha_mean
        )
    gt_mae = metrics.get("ground_truth_mae")
    return isinstance(gt_mae, (int, float)) and float(gt_mae) > (
        thresholds.stage_alpha_delta
    )


def _refiner_worsened(
    raw: Mapping[str, object],
    refined: Mapping[str, object],
    kind: RegionKind,
    thresholds: AttributionThresholds,
) -> bool:
    if kind == "opaque_core":
        return float(cast(float, refined["mean_deficit"])) > float(
            cast(float, raw["mean_deficit"])
        ) + thresholds.stage_alpha_delta or int(
            cast(int, refined["hole_components"])
        ) > int(cast(int, raw["hole_components"]))
    if kind == "background":
        return (
            float(cast(float, refined["alpha_mean"]))
            > float(cast(float, raw["alpha_mean"])) + thresholds.stage_alpha_delta
        )
    raw_mae = raw.get("ground_truth_mae")
    refined_mae = refined.get("ground_truth_mae")
    return (
        isinstance(raw_mae, (int, float))
        and isinstance(refined_mae, (int, float))
        and float(refined_mae) > float(raw_mae) + thresholds.stage_alpha_delta
    )


def _stage_label(stages: Sequence[str]) -> str:
    unique = sorted(set(stages))
    if not unique:
        return "no_observed_failure"
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _named_regions(
    annotations: MatteQualityAnnotations,
    sequence: int,
    shape: tuple[int, int],
) -> dict[str, QualityNamedRegion]:
    regions = annotations.load_regions(sequence, shape=shape)
    if regions:
        return regions
    opaque = annotations.load_array(sequence, "opaque_core", shape=shape)
    background = annotations.load_array(sequence, "background", shape=shape)
    if opaque is not None and bool(np.any(opaque)):
        regions["opaque_core"] = QualityNamedRegion("opaque_core", opaque)
    if background is not None and bool(np.any(background)):
        regions["background"] = QualityNamedRegion("background", background)
    return regions


def _render_boundaries(
    frame: dict[str, Any],
    *,
    source: np.ndarray,
    raw_alpha: np.ndarray,
    refined_alpha: np.ndarray,
    foreground: np.ndarray | None,
    backdrop: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    segmentation, compositing = _controls(frame)
    raw_wrap = compositing.get("light_wrap", 0.0)
    if (
        isinstance(raw_wrap, bool)
        or not isinstance(raw_wrap, (int, float))
        or not math.isfinite(float(raw_wrap))
        or not 0.0 <= float(raw_wrap) <= 1.0
    ):
        raise MatteQualityError("recorded light-wrap control is invalid")
    blend_value = compositing.get("blend_space", "srgb_legacy")
    if blend_value not in ("srgb_legacy", "linear_srgb"):
        raise MatteQualityError("recorded blend-space control is invalid")
    blend_space = cast(BlendSpace, blend_value)
    use_foreground = bool(compositing.get("use_model_foreground", False))
    transform = _recorded_transform(frame)
    common = {"blend_space": blend_space, "color_transform": transform}
    refined_plain = composite(source, backdrop, refined_alpha, **common)
    views = {
        # The four acceptance boundaries.
        "raw_alpha": composite(source, backdrop, raw_alpha, **common),
        "post_refiner_alpha": refined_plain,
        "direct_full_frame_foreground": (
            composite(foreground, backdrop, raw_alpha, **common)
            if foreground is not None
            else composite(source, backdrop, raw_alpha, **common)
        ),
        "current_compositor": composite(
            source,
            backdrop,
            refined_alpha,
            light_wrap=float(raw_wrap),
            edge_foreground=foreground if use_foreground else None,
            **common,
        ),
        # Factorial controls used to distinguish foreground from wrap.
        "edge_foreground_only": composite(
            source,
            backdrop,
            refined_alpha,
            edge_foreground=foreground if use_foreground else None,
            **common,
        ),
        "light_wrap_only": composite(
            source,
            backdrop,
            refined_alpha,
            light_wrap=float(raw_wrap),
            **common,
        ),
    }
    return views, {
        "blend_space": blend_space,
        "light_wrap": _round(float(raw_wrap)),
        "use_model_foreground": use_foreground,
        "configured_rvm_downsample": segmentation.get("rvm_downsample"),
        "configured_mask_shift": segmentation.get("mask_shift"),
    }


def _candidate_report(
    frame_inputs: Sequence[dict[str, Any]],
    thresholds: AttributionThresholds,
) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}
    core_available = False
    core_pass = True
    halo_available = False
    halo_pass = True
    soft_available = False
    soft_pass = True
    gt_available = False
    gt_pass = True
    motion_available = len(frame_inputs) > 1
    candidate_motion: list[float] = []
    original_motion: list[float] = []
    candidate_trail: list[float] = []
    original_trail: list[float] = []
    candidate_halo_area: list[float] = []
    original_halo_area: list[float] = []
    candidate_halo_width: list[float] = []
    original_halo_width: list[float] = []
    previous_candidate: np.ndarray | None = None
    previous_raw: np.ndarray | None = None
    previous_gt: np.ndarray | None = None

    for item in frame_inputs:
        raw = cast(np.ndarray, item["raw_alpha"])
        candidate = np.ascontiguousarray((raw >= 0.5).astype(np.float32))
        gt = cast(np.ndarray | None, item["ground_truth"])
        regions = cast(dict[str, QualityNamedRegion], item["regions"])
        if gt is not None:
            gt_available = True
            original_error = float(np.mean(np.abs(raw - gt), dtype=np.float64))
            candidate_error = float(np.mean(np.abs(candidate - gt), dtype=np.float64))
            gt_pass = gt_pass and (
                candidate_error <= original_error + thresholds.candidate_nonregression
            )
        for region in regions.values():
            selected = region.mask.astype(bool)
            if region.kind == "opaque_core":
                core_available = True
                core_pass = core_pass and (
                    float(np.percentile(candidate[selected], 5))
                    >= thresholds.opaque_alpha_p05
                    and _component_count(selected & (candidate < 0.5)) == 0
                )
            elif region.kind == "background":
                halo_available = True
                candidate_false = selected & (candidate > 0.05)
                original_false = selected & (raw > 0.05)
                candidate_area = float(np.mean(candidate[selected] > 0.05))
                original_area = float(np.mean(raw[selected] > 0.05))
                candidate_halo_area.append(candidate_area)
                original_halo_area.append(original_area)
                halo_pass = halo_pass and (
                    float(np.mean(candidate[selected], dtype=np.float64))
                    <= thresholds.background_alpha_mean
                    and candidate_area
                    <= original_area + thresholds.candidate_nonregression
                )
                if gt is not None:
                    distance = cv2.distanceTransform(
                        (gt <= 0.05).astype(np.uint8),
                        cv2.DIST_L2,
                        cv2.DIST_MASK_5,
                    )
                    candidate_width = (
                        float(np.percentile(distance[candidate_false], 95))
                        if bool(np.any(candidate_false))
                        else 0.0
                    )
                    original_width = (
                        float(np.percentile(distance[original_false], 95))
                        if bool(np.any(original_false))
                        else 0.0
                    )
                    candidate_halo_width.append(candidate_width)
                    original_halo_width.append(original_width)
                    halo_pass = halo_pass and (
                        candidate_width
                        <= original_width + thresholds.candidate_nonregression
                    )
            elif region.kind == "soft_boundary":
                soft_available = True
                if gt is None:
                    soft_pass = False
                else:
                    original_error = float(
                        np.mean(np.abs(raw[selected] - gt[selected]), dtype=np.float64)
                    )
                    candidate_error = float(
                        np.mean(
                            np.abs(candidate[selected] - gt[selected]),
                            dtype=np.float64,
                        )
                    )
                    gt_uncertain = float(
                        np.mean(
                            (gt[selected] > 0.05) & (gt[selected] < 0.95),
                            dtype=np.float64,
                        )
                    )
                    candidate_uncertain = float(
                        np.mean(
                            (candidate[selected] > 0.05) & (candidate[selected] < 0.95),
                            dtype=np.float64,
                        )
                    )
                    soft_pass = soft_pass and (
                        candidate_error
                        <= original_error + thresholds.candidate_nonregression
                        and (
                            gt_uncertain <= 0.05
                            or candidate_uncertain >= gt_uncertain * 0.5
                        )
                    )
        if previous_candidate is not None and previous_raw is not None:
            affine = np.asarray(
                cast(Sequence[float], item["registration"]),
                dtype=np.float32,
            ).reshape(2, 3)
            size = (candidate.shape[1], candidate.shape[0])
            registered_candidate = cv2.warpAffine(
                previous_candidate,
                affine,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            registered_raw = cv2.warpAffine(
                previous_raw,
                affine,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            candidate_motion.append(
                float(
                    np.mean(
                        np.abs(candidate - registered_candidate),
                        dtype=np.float64,
                    )
                )
            )
            original_motion.append(
                float(np.mean(np.abs(raw - registered_raw), dtype=np.float64))
            )
            if gt is not None and previous_gt is not None:
                registered_previous_gt = cv2.warpAffine(
                    previous_gt,
                    affine,
                    size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                prior_only = (registered_previous_gt >= 0.5) & (gt < 0.5)
                current_area = max(1, int(np.count_nonzero(gt >= 0.5)))
                candidate_trail.append(
                    float(np.count_nonzero(prior_only & (candidate >= 0.5)))
                    / current_area
                )
                original_trail.append(
                    float(np.count_nonzero(prior_only & (raw >= 0.5))) / current_area
                )
        previous_candidate = candidate
        previous_raw = raw
        previous_gt = gt

    motion_pass = (
        bool(candidate_motion)
        and float(np.mean(candidate_motion))
        <= float(np.mean(original_motion)) + thresholds.candidate_nonregression
        and (
            not candidate_trail
            or float(np.mean(candidate_trail))
            <= float(np.mean(original_trail)) + thresholds.candidate_nonregression
        )
    )
    for name, available, passed in (
        ("opaque_core", core_available, core_pass),
        ("hair_and_soft_boundary", soft_available, soft_pass),
        ("exterior_halo", halo_available, halo_pass),
        ("ground_truth", gt_available, gt_pass),
        ("motion", motion_available and bool(candidate_motion), motion_pass),
    ):
        checks[name] = {
            "status": "pass"
            if available and passed
            else ("fail" if available else "not_evaluated")
        }
    checks["motion"].update(
        {
            "candidate_compensated_alpha_diff_mean": (
                _round(float(np.mean(candidate_motion))) if candidate_motion else None
            ),
            "raw_compensated_alpha_diff_mean": (
                _round(float(np.mean(original_motion))) if original_motion else None
            ),
            "candidate_motion_trail_mean": (
                _round(float(np.mean(candidate_trail))) if candidate_trail else None
            ),
            "raw_motion_trail_mean": (
                _round(float(np.mean(original_trail))) if original_trail else None
            ),
        }
    )
    checks["exterior_halo"].update(
        {
            "candidate_area_ratio_mean": (
                _round(float(np.mean(candidate_halo_area)))
                if candidate_halo_area
                else None
            ),
            "raw_area_ratio_mean": (
                _round(float(np.mean(original_halo_area)))
                if original_halo_area
                else None
            ),
            "candidate_width_p95_px_max": (
                _round(float(np.max(candidate_halo_width)))
                if candidate_halo_width
                else None
            ),
            "raw_width_p95_px_max": (
                _round(float(np.max(original_halo_width)))
                if original_halo_width
                else None
            ),
        }
    )
    accepted = all(check["status"] == "pass" for check in checks.values())
    rejected = [name for name, check in checks.items() if check["status"] != "pass"]
    return {
        "global_hard_threshold": {
            "candidate": "alpha = (raw_pha >= 0.5)",
            "status": "eligible_for_controlled_experiment" if accepted else "rejected",
            "production_default_changed": False,
            "checks": checks,
            "rejected_because": rejected,
            "policy": (
                "A global hard threshold is rejected unless opaque core, "
                "hair/soft-boundary, motion, exterior-halo, and ground-truth "
                "checks all pass on the same replay."
            ),
        },
        "opaque_core_restricted_calibration": {
            "status": "next_experiment",
            "production_default_changed": False,
            "constraints": [
                "operate only inside a high-confidence opaque-core support",
                "preserve recorded soft/hair alpha",
                "rerun motion, halo, and ground-truth gates",
            ],
        },
        "rvm_ratio_profile_or_qualified_model": {
            "status": "next_experiment",
            "production_default_changed": False,
            "controls": [
                "hold recorded source, backdrop, timestamps, and compositor fixed",
                "sweep the qualified downsample/profile set one variable at a time",
                "retain raw pha and fgr for every candidate",
            ],
        },
        "source_quality_lighting_cadence": {
            "status": "conditional_experiment",
            "production_default_changed": False,
            "controls": [
                "change only exposure, lighting, capture mode, or unique-input cadence",
                "do not treat repeated output frames as new model inputs",
                "compare against the same spatial and registered temporal gates",
            ],
        },
    }


def diagnose_bundle(
    bundle_root: Path | str,
    output_root: Path | str,
    *,
    annotations_root: Path | str,
    thresholds: AttributionThresholds = AttributionThresholds(),
    max_output_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Render and classify the four alpha/compositor attribution boundaries."""

    _validate_thresholds(thresholds)
    bundle = MatteReplayBundle(bundle_root)
    if not bundle.frames or not bool(
        bundle.manifest.get("matte_metrics_authoritative")
    ):
        raise MatteQualityError(
            "alpha attribution requires a full metric-authoritative replay bundle"
        )
    annotations = MatteQualityAnnotations(annotations_root, bundle)
    output = Path(output_root)
    writer = _PrivateImageWriter(output, max_output_bytes)
    _private_directory(output, create=True)
    frames_dir = output / "frames"
    _private_directory(frames_dir, create=True)

    per_frame: list[dict[str, Any]] = []
    candidate_inputs: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}
    observed_stage_counts: dict[str, int] = {}
    region_stage_counts: dict[str, dict[str, int]] = {}
    stable_core: dict[str, list[bool]] = {}
    observed_region_names: set[str] = set()
    availability = {
        "raw_alpha": True,
        "post_refiner_alpha": True,
        "clean_foreground": True,
        "backdrop": True,
        "final_composite": True,
    }

    for sequence, frame in enumerate(bundle.frames):
        source = bundle.load_array(frame, "raw_frame")
        raw_alpha = bundle.load_array(frame, "raw_mask").astype(np.float32, copy=False)
        refined_alpha = bundle.load_array(frame, "refined_mask").astype(
            np.float32, copy=False
        )
        backdrop = bundle.load_array(frame, "backdrop_frame")
        recorded_final = bundle.load_array(frame, "final_composite")
        artifacts = cast(dict[str, Any], frame["artifacts"])
        recorded_base = (
            bundle.load_array(frame, "base_composite")
            if "base_composite" in artifacts
            else recorded_final
        )
        foreground = (
            bundle.load_array(frame, "clean_foreground")
            if "clean_foreground" in artifacts
            else None
        )
        if foreground is None:
            availability["clean_foreground"] = False
        shape = raw_alpha.shape
        if (
            source.shape != (*shape, 3)
            or refined_alpha.shape != shape
            or backdrop.shape != source.shape
            or recorded_final.shape != source.shape
        ):
            raise MatteQualityError("attribution tracks do not align")
        regions = _named_regions(annotations, sequence, shape)
        if not regions:
            raise MatteQualityError(
                "alpha attribution requires at least one spatial annotation region"
            )
        observed_region_names.update(regions)
        ground_truth = annotations.load_array(
            sequence, "ground_truth_alpha", shape=shape
        )
        ground_truth_foreground = annotations.load_array(
            sequence, "ground_truth_foreground", shape=shape
        )
        views, controls = _render_boundaries(
            cast(dict[str, Any], frame),
            source=source,
            raw_alpha=np.ascontiguousarray(raw_alpha),
            refined_alpha=np.ascontiguousarray(refined_alpha),
            foreground=foreground,
            backdrop=backdrop,
        )
        frame_dir = frames_dir / f"{sequence:08d}"
        _private_directory(frame_dir, create=True)
        boundary_artifacts: dict[str, object] = {}
        for name, image in views.items():
            relative = f"frames/{sequence:08d}/{name}.png"
            boundary_artifacts[name] = writer.write(frame_dir, relative, image)

        current_delta = np.abs(
            views["current_compositor"].astype(np.int16)
            - recorded_base.astype(np.int16)
        )
        final_delta = np.abs(
            recorded_final.astype(np.int16) - recorded_base.astype(np.int16)
        )
        current_max_delta = int(current_delta.max(initial=0))
        post_base_max_delta = int(final_delta.max(initial=0))
        region_reports: dict[str, Any] = {}
        for name, region in sorted(regions.items()):
            raw_metrics = _alpha_metrics(raw_alpha, region, ground_truth)
            refined_metrics = _alpha_metrics(refined_alpha, region, ground_truth)
            foreground_error, foreground_reference = _foreground_error(
                foreground,
                source,
                ground_truth_foreground,
                region.mask,
            )
            deltas = {
                "raw_to_refiner_alpha_mean_abs": _round(
                    float(
                        np.mean(
                            np.abs(
                                refined_alpha[region.mask.astype(bool)]
                                - raw_alpha[region.mask.astype(bool)]
                            ),
                            dtype=np.float64,
                        )
                    )
                ),
                "full_foreground_vs_raw_source": _region_mean_delta(
                    views["direct_full_frame_foreground"],
                    views["raw_alpha"],
                    region.mask,
                ),
                "edge_foreground_vs_plain": _region_mean_delta(
                    views["edge_foreground_only"],
                    views["post_refiner_alpha"],
                    region.mask,
                ),
                "light_wrap_vs_plain": _region_mean_delta(
                    views["light_wrap_only"],
                    views["post_refiner_alpha"],
                    region.mask,
                ),
                "current_vs_recorded_base": _region_mean_delta(
                    views["current_compositor"],
                    recorded_base,
                    region.mask,
                ),
                "recorded_final_vs_base": _region_mean_delta(
                    recorded_final,
                    recorded_base,
                    region.mask,
                ),
                "current_vs_recorded_base_max_channel_delta": (
                    _region_max_channel_delta(
                        views["current_compositor"],
                        recorded_base,
                        region.mask,
                    )
                ),
                "recorded_final_vs_base_max_channel_delta": (
                    _region_max_channel_delta(
                        recorded_final,
                        recorded_base,
                        region.mask,
                    )
                ),
            }
            stages: list[str] = []
            raw_failure = _is_alpha_failure(raw_metrics, region.kind, thresholds)
            refined_failure = _is_alpha_failure(
                refined_metrics, region.kind, thresholds
            )
            if raw_failure:
                stages.append("raw_alpha")
            if refined_failure and (
                not raw_failure
                or _refiner_worsened(
                    raw_metrics, refined_metrics, region.kind, thresholds
                )
            ):
                stages.append("post_refiner_alpha")
            if (
                foreground_error is not None
                and (
                    foreground_reference == "ground_truth_foreground"
                    or region.kind == "opaque_core"
                )
                and foreground_error > thresholds.foreground_error
                and (
                    deltas["full_foreground_vs_raw_source"] > thresholds.visual_delta
                    or deltas["edge_foreground_vs_plain"] > thresholds.visual_delta
                )
            ):
                stages.append("clean_foreground")
            if deltas["light_wrap_vs_plain"] > thresholds.visual_delta:
                stages.append("light_wrap")
            if (
                deltas["current_vs_recorded_base_max_channel_delta"]
                > thresholds.final_channel_delta
                or deltas["recorded_final_vs_base_max_channel_delta"]
                > thresholds.final_channel_delta
            ):
                stages.append("final_blend")
            primary = _stage_label(stages)
            stage_counts[primary] = stage_counts.get(primary, 0) + 1
            for stage in sorted(set(stages)):
                observed_stage_counts[stage] = observed_stage_counts.get(stage, 0) + 1
            per_region_counts = region_stage_counts.setdefault(name, {})
            per_region_counts[primary] = per_region_counts.get(primary, 0) + 1
            if region.kind == "opaque_core":
                stable_core.setdefault(name, []).append(refined_failure)

            heatmaps: dict[str, object] = {}
            heatmap_values = {
                "raw_alpha_error": _alpha_error_map(
                    raw_alpha, region.kind, ground_truth
                ),
                "refined_alpha_error": _alpha_error_map(
                    refined_alpha, region.kind, ground_truth
                ),
                "refiner_absolute_delta": np.abs(refined_alpha - raw_alpha),
                "clean_foreground_error": (
                    _rgb_delta(
                        foreground,
                        (
                            ground_truth_foreground
                            if ground_truth_foreground is not None
                            else source
                        ),
                    )
                    if foreground is not None
                    else np.zeros(shape, dtype=np.float32)
                ),
                "edge_foreground_contribution": _rgb_delta(
                    views["edge_foreground_only"],
                    views["post_refiner_alpha"],
                ),
                "light_wrap_contribution": _rgb_delta(
                    views["light_wrap_only"],
                    views["post_refiner_alpha"],
                ),
                "final_backdrop_leakage": _leakage_map(
                    source, backdrop, recorded_final
                ),
            }
            for heatmap_name, values in heatmap_values.items():
                filename = f"{name}-{heatmap_name}.png"
                relative = f"frames/{sequence:08d}/{filename}"
                heatmaps[heatmap_name] = writer.write(
                    frame_dir,
                    relative,
                    _heatmap(values, region.mask),
                )
            region_reports[name] = {
                "kind": region.kind,
                "pixel_count": int(np.count_nonzero(region.mask)),
                "raw_alpha": raw_metrics,
                "post_refiner_alpha": refined_metrics,
                "final_composite": _visual_metrics(
                    source, backdrop, recorded_final, region.mask
                ),
                "clean_foreground_error": foreground_error,
                "clean_foreground_reference": foreground_reference,
                "counterfactual_deltas": deltas,
                "observed_stages": sorted(set(stages)),
                "primary_stage": primary,
                "heatmaps": heatmaps,
            }

        candidate_inputs.append(
            {
                "raw_alpha": raw_alpha,
                "ground_truth": ground_truth,
                "regions": regions,
                "registration": annotations.frames[sequence][
                    "registration_from_previous"
                ],
            }
        )
        per_frame.append(
            {
                "sequence": sequence,
                "capture_sequence": frame["capture_sequence"],
                "capture_monotonic_ns": frame["capture_monotonic_ns"],
                "effective_controls": frame.get("effective_controls", {}),
                "counterfactual_controls": controls,
                "reproduction": {
                    "current_vs_recorded_base_max_channel_delta": current_max_delta,
                    "recorded_final_vs_base_max_channel_delta": post_base_max_delta,
                    "tolerance": thresholds.final_channel_delta,
                },
                "boundary_artifacts": boundary_artifacts,
                "regions": region_reports,
            }
        )

    stable_failures = sorted(
        name
        for name, failures in stable_core.items()
        if len(per_frame) >= 2 and len(failures) == len(per_frame) and all(failures)
    )
    total_observations = sum(stage_counts.values())
    failing_primary_counts = {
        name: count
        for name, count in stage_counts.items()
        if name != "no_observed_failure"
    }
    dominant = (
        max(
            failing_primary_counts,
            key=lambda name: (failing_primary_counts[name], name),
        )
        if failing_primary_counts
        else "no_observed_failure"
    )
    overall_stage = _stage_label(tuple(observed_stage_counts))
    anatomy_missing = sorted(set(_REQUIRED_ANATOMY) - observed_region_names)
    per_frame_anatomy_missing = {
        str(frame["sequence"]): sorted(
            set(_REQUIRED_ANATOMY) - set(cast(dict[str, Any], frame["regions"]))
        )
        for frame in per_frame
    }
    coverage_complete = not any(per_frame_anatomy_missing.values()) and all(
        availability.values()
    )
    candidate_report = _candidate_report(candidate_inputs, thresholds)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "source": {
            "bundle_manifest_sha256": _bundle_manifest_digest(bundle),
            "annotation_manifest_sha256": annotations.manifest_sha256,
            "annotation_provenance": annotations.provenance,
        },
        "privacy": {
            "output_is_owner_only": True,
            "contains_identifiable_derived_images": True,
            "image_bytes_written": writer.bytes_written,
            "max_output_bytes": max_output_bytes,
            "bound_includes_images_json_and_markdown": True,
            "network_or_live_devices_used": False,
        },
        "method": {
            "four_required_boundaries": [
                "raw_alpha",
                "post_refiner_alpha",
                "direct_full_frame_foreground",
                "current_compositor",
            ],
            "factorial_controls": [
                "edge_foreground_only",
                "light_wrap_only",
            ],
            "thresholds": asdict(thresholds),
            "classification_is_diagnostic_not_release_policy": True,
            "soft_region_clean_foreground_requires_ground_truth": True,
        },
        "coverage": {
            "frame_count": len(per_frame),
            "region_names": sorted(observed_region_names),
            "required_anatomy": list(_REQUIRED_ANATOMY),
            "missing_required_anatomy": anatomy_missing,
            "per_frame_missing_required_anatomy": per_frame_anatomy_missing,
            "tracks": availability,
            "complete_for_matte_0_5": coverage_complete,
        },
        "aggregate": {
            "stage_counts": dict(sorted(stage_counts.items())),
            "observed_stage_counts": dict(sorted(observed_stage_counts.items())),
            "region_stage_counts": {
                name: dict(sorted(counts.items()))
                for name, counts in sorted(region_stage_counts.items())
            },
            "dominant_primary_stage": dominant,
            "overall_stage": overall_stage,
            "observation_count": total_observations,
            "stable_opaque_core_failure_regions": stable_failures,
        },
        "qualification": {
            "status": (
                "fail" if stable_failures or not coverage_complete else "review"
            ),
            "stable_opaque_core_failure_cannot_pass": True,
            "failed_for_stable_opaque_core": bool(stable_failures),
            "failed_for_incomplete_coverage": not coverage_complete,
        },
        "candidate_corrections": candidate_report,
        "conclusion": {
            "primary_stage": overall_stage,
            "next_experiment": (
                "Run an opaque-core-restricted calibration and RVM "
                "downsample/profile sweep on this exact replay; rerun every "
                "hair, motion, halo, and ground-truth gate."
                if "raw_alpha" in observed_stage_counts or stable_failures
                else (
                    "Ablate mask_shift on this exact replay."
                    if "post_refiner_alpha" in observed_stage_counts
                    else (
                        "Run the clean-foreground/light-wrap factorial on this "
                        "exact replay and retain the lower-error boundary."
                        if (
                            "clean_foreground" in observed_stage_counts
                            or "light_wrap" in observed_stage_counts
                        )
                        else "Inspect the recorded-base/final-blend discrepancy."
                    )
                )
            ),
            "rejected_alternatives": [
                "Do not make a global hard threshold the default unless every "
                "recorded candidate check passes.",
                "Do not infer temporal dancing or cadence causality from a "
                "single-frame spatial attribution.",
            ],
            "temporal_conclusion": (
                "not inferred from still-frame stage attribution; use the "
                "registered multi-frame MATTE-0.2 metrics"
            ),
        },
        "per_frame": per_frame,
    }
    deterministic = {
        "source": report["source"],
        "method": report["method"],
        "coverage": report["coverage"],
        "aggregate": report["aggregate"],
        "qualification": report["qualification"],
        "candidate_corrections": report["candidate_corrections"],
        "conclusion": report["conclusion"],
        "per_frame": per_frame,
    }
    report["evidence_sha256"] = hashlib.sha256(_json_bytes(deterministic)).hexdigest()
    report_payload = _json_bytes(report)
    markdown_payload = report_markdown(report).encode("utf-8")
    if (
        writer.bytes_written + len(report_payload) + len(markdown_payload)
        > max_output_bytes
    ):
        raise MatteQualityError("attribution output byte bound reached")
    _atomic_private_write(output / "attribution.json", report_payload)
    _atomic_private_write(output / "attribution.md", markdown_payload)
    return report


def report_markdown(report: Mapping[str, object]) -> str:
    """Render the review summary; canonical evidence remains JSON."""

    aggregate = cast(dict[str, Any], report["aggregate"])
    coverage = cast(dict[str, Any], report["coverage"])
    qualification = cast(dict[str, Any], report["qualification"])
    conclusion = cast(dict[str, Any], report["conclusion"])
    candidates = cast(dict[str, Any], report["candidate_corrections"])
    hard = cast(dict[str, Any], candidates["global_hard_threshold"])
    lines = [
        "# RVM alpha integrity attribution",
        "",
        f"- Evidence SHA-256: `{report['evidence_sha256']}`",
        f"- Primary stage: **{conclusion['primary_stage']}**",
        f"- Qualification: **{qualification['status']}**",
        f"- Stable opaque-core failures: "
        f"`{aggregate['stable_opaque_core_failure_regions']}`",
        f"- Region coverage: `{coverage['region_names']}`",
        f"- Missing anatomy regions: `{coverage['missing_required_anatomy']}`",
        f"- Global hard-threshold candidate: **{hard['status']}**",
        "",
        "## Stage counts",
        "",
        "| Stage | Region/frame observations |",
        "| --- | ---: |",
    ]
    for stage, count in cast(dict[str, int], aggregate["stage_counts"]).items():
        lines.append(f"| `{stage}` | {count} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(conclusion["next_experiment"]),
            "",
            str(conclusion["temporal_conclusion"]),
            "",
            "The JSON report and digest-bound lossless PNG artifacts are "
            "authoritative. These private outputs contain derived identifiable "
            "imagery and must remain owner-only.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser(
    *,
    prog: str = "custback matte-diagnose",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Attribute RVM alpha, foreground, wrap, and final-blend defects",
    )
    parser.add_argument("bundle", help="complete private replay bundle")
    parser.add_argument(
        "--annotations",
        required=True,
        help="private digest-bound trimap annotation bundle",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="new owner-only attribution report directory",
    )
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"hard output bound (default {DEFAULT_MAX_BYTES})",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback matte-diagnose",
) -> int:
    args = build_parser(prog=prog).parse_args(argv)
    try:
        report = diagnose_bundle(
            args.bundle,
            args.output,
            annotations_root=args.annotations,
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
    print(
        f"attributed {len(report['per_frame'])} frame(s); "
        f"primary stage {report['conclusion']['primary_stage']}; "
        f"qualification {report['qualification']['status']}"
    )
    return 1 if report["qualification"]["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
