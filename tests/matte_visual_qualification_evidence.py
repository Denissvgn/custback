#!/usr/bin/env python3
"""Generated MATTE-5.2 visual-qualification contract evidence.

The pixels in this module are deterministic MIT-licensed test fixtures.  They
exercise the evidence validator, not a camera, model runtime, virtual-camera
consumer, or human review.  Consequently the generated plan must always stay
``pending``/``not_decidable`` and can never authorize a preset or default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import matte_quality_evidence as quality_evidence
import numpy as np

from custback.compositor import composite
from custback.config import AccelerationConfig, CompositingConfig, SegmentationConfig
from custback.matte_diagnostics import (
    MatteCaptureMetadata,
    MatteDiagnosticRecorder,
    MatteFrameEvidence,
    MatteReplayBundle,
    _npy_bytes,
)
from custback.matte_policy import MatteBackendKind, resolve_matte_policy
from custback.matte_quality import (
    EvaluationMetadata,
    MatteQualityAnnotations,
    QualityFrameAnnotations,
    QualityNamedRegion,
    _bundle_manifest_digest,
    evaluate_bundle,
    write_quality_annotations,
)
from custback import matte_visual_qualification as qualification


_BACKGROUND = {
    "mode": "video",
    "fit_mode": "cover",
    "anchor_x": 0.5,
    "anchor_y": 0.5,
}

_GENERATED_CAPTURE_METHODS = {
    "in_memory": "generated-memory-tap",
    "highgui_pre_overlay": "generated-highgui-pre-overlay",
    "snapshot_jpeg": "generated-http-snapshot",
    "mjpeg_jpeg": "generated-mjpeg-part",
    "websocket_jpeg": "generated-output-websocket",
    "pyvirtualcam_loopback": "generated-pyvirtualcam-double",
    "windows_native_loopback": "generated-native-ring-double",
}

_QUALITY_GATES: tuple[dict[str, object], ...] = (
    {
        "id": "opaque-p05",
        "family": "opaque_core",
        "metric": "aggregate.metrics.opaque_core_alpha_p05.p05",
        "op": ">=",
        "value": 0.95,
    },
    {
        "id": "opaque-below",
        "family": "opaque_core",
        "metric": "aggregate.metrics.opaque_core_fraction_below_0_95.p95",
        "op": "<=",
        "value": 0.05,
    },
    {
        "id": "holes",
        "family": "opaque_core",
        "metric": "aggregate.metrics.foreground_hole_components.max",
        "op": "<=",
        "value": 0.0,
    },
    {
        "id": "background",
        "family": "background",
        "metric": "aggregate.metrics.background_alpha_mean.p95",
        "op": "<=",
        "value": 0.01,
    },
    {
        "id": "halo-area",
        "family": "halo",
        "metric": "aggregate.metrics.exterior_halo_area_ratio.p95",
        "op": "<=",
        "value": 0.05,
    },
    {
        "id": "halo-width",
        "family": "halo",
        "metric": "aggregate.metrics.exterior_halo_width_p95_px.p95",
        "op": "<=",
        "value": 8.0,
    },
    {
        "id": "alpha-mse",
        "family": "fine_detail",
        "metric": "aggregate.metrics.ground_truth_alpha_mse.p95",
        "op": "<=",
        "value": 0.01,
    },
    {
        "id": "gradient",
        "family": "fine_detail",
        "metric": "aggregate.metrics.ground_truth_gradient_mae.p95",
        "op": "<=",
        "value": 0.10,
    },
    {
        "id": "uncertain",
        "family": "fine_detail",
        "metric": "aggregate.metrics.uncertain_pixel_fraction.p50",
        "op": ">=",
        "value": 0.05,
    },
    {
        "id": "contour",
        "family": "temporal",
        "metric": "aggregate.metrics.contour_displacement_p95_px.p95",
        "op": "<=",
        "value": 1.5,
    },
    {
        "id": "temporal-alpha",
        "family": "temporal",
        "metric": "aggregate.metrics.compensated_alpha_temporal_abs_diff.p95",
        "op": "<=",
        "value": 0.10,
    },
    {
        "id": "trail",
        "family": "temporal",
        "metric": "aggregate.metrics.motion_trail_area_ratio.p95",
        "op": "<=",
        "value": 0.10,
    },
)


@dataclass(frozen=True)
class GeneratedQualification:
    """Paths making up one complete owner-only generated plan."""

    plan: Path
    baseline_bundle: Path
    baseline_annotations: Path
    candidate_bundle: Path
    candidate_annotations: Path
    boundary: Path
    review: Path
    boundary_artifacts: Mapping[str, Path]
    baseline_bundles: Mapping[str, Path]
    baseline_annotation_sets: Mapping[str, Path]
    candidate_bundles: Mapping[str, Path]
    candidate_annotation_sets: Mapping[str, Path]
    boundaries: Mapping[str, Path]
    reviews: Mapping[str, Path]
    all_boundary_artifacts: Mapping[str, Mapping[str, Path]]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)
    path.chmod(0o700)


def _write_private_json(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))
    path.chmod(0o600)


def _write_private_image(
    path: Path,
    pixels: np.ndarray,
    *,
    media_type: str,
) -> None:
    suffix = ".jpg" if media_type == "image/jpeg" else ".png"
    parameters = (
        [cv2.IMWRITE_JPEG_QUALITY, 95]
        if suffix == ".jpg"
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    success, encoded = cv2.imencode(suffix, pixels, parameters)
    if not success:
        raise AssertionError(f"could not encode generated {media_type} artifact")
    path.write_bytes(encoded.tobytes())
    path.chmod(0o600)


def _selected_configuration(
    *,
    canvas_shape: tuple[int, int] = (
        quality_evidence.HEIGHT,
        quality_evidence.WIDTH,
    ),
    dynamic_backdrop: bool = True,
) -> tuple[
    SegmentationConfig,
    CompositingConfig,
    dict[str, Any],
]:
    segmentation = SegmentationConfig.model_validate(
        {
            "backend": "mediapipe",
            "mask_blur": 0,
            "edge_refine": True,
            "mask_shift": 0,
            "temporal_smoothing": 0.0,
            "spatial_edge_refinement": {
                "mode": "stable_guided",
                "reference_short_edge_px": 720,
                "radius_at_reference_px": 8,
                "min_radius_px": 2,
                "max_radius_px": 12,
            },
            "boundary_stabilization": {
                "mode": "motion_aware",
                "time_constant_s": 0.1,
                "max_motion_px_per_s": 720.0,
            },
        }
    )
    compositing = CompositingConfig.model_validate(
        {
            "light_wrap": 0.25,
            "use_model_foreground": False,
            "blend_space": "linear_srgb",
            "light_wrap_stabilization": {
                "mode": "temporal_bounded",
                "time_constant_s": 0.12,
            },
        }
    )
    policy = resolve_matte_policy(
        segmentation,
        compositing,
        MatteBackendKind.CONFIDENCE_MASK_VIDEO,
        canvas_shape=canvas_shape,
        light_wrap_stabilization_eligible=dynamic_backdrop,
    ).to_dict()
    return segmentation, compositing, policy


def _baseline_configuration(
    *,
    canvas_shape: tuple[int, int] = (
        quality_evidence.HEIGHT,
        quality_evidence.WIDTH,
    ),
    dynamic_backdrop: bool = True,
) -> tuple[
    SegmentationConfig,
    CompositingConfig,
    dict[str, Any],
]:
    segmentation = SegmentationConfig.model_validate(
        {
            "backend": "mediapipe",
            "mask_blur": 0,
            "edge_refine": True,
            "mask_shift": 0,
            "temporal_smoothing": 0.0,
            "spatial_edge_refinement": {"mode": "legacy_watershed"},
            "boundary_stabilization": {"mode": "off"},
        }
    )
    compositing = CompositingConfig.model_validate(
        {
            "light_wrap": 0.25,
            "use_model_foreground": False,
            "blend_space": "linear_srgb",
            "light_wrap_stabilization": {"mode": "off"},
        }
    )
    policy = resolve_matte_policy(
        segmentation,
        compositing,
        MatteBackendKind.CONFIDENCE_MASK_VIDEO,
        canvas_shape=canvas_shape,
        light_wrap_stabilization_eligible=dynamic_backdrop,
    ).to_dict()
    return segmentation, compositing, policy


def _named_regions(alpha: np.ndarray) -> dict[str, QualityNamedRegion]:
    uncertain = ((alpha > 0.05) & (alpha < 0.95)).astype(np.uint8)
    opaque, background = quality_evidence._regions(alpha)
    # These are generated visibility regions, not semantic claims about a
    # person.  The plan remains pending regardless of their metric result.
    return {
        "hair": QualityNamedRegion("soft_boundary", uncertain),
        "headphones": QualityNamedRegion("opaque_core", opaque.astype(np.uint8)),
        "background": QualityNamedRegion(
            "background",
            background.astype(np.uint8),
        ),
    }


def _record_bundle(
    root: Path,
    *,
    name: str,
    candidate: bool,
    use_defective_mask: bool = False,
    canvas: tuple[int, int],
    background_mode: str,
    cadence: str,
) -> tuple[Path, Path, dict[str, Any]]:
    (
        ground_truth,
        defective,
        backdrops,
        registrations,
        _timestamps,
        _capture_sequences,
        _segment_kind,
    ) = quality_evidence._fixture_arrays("jitter_2px")
    width, height = canvas
    dynamic_backdrop = background_mode in {"video", "camera"}
    configuration_factory = (
        _selected_configuration if candidate else _baseline_configuration
    )
    segmentation, compositing, matte_policy = configuration_factory(
        canvas_shape=(height, width),
        dynamic_backdrop=dynamic_backdrop,
    )
    bundle_root = root / f"{name}-bundle"
    annotation_root = root / f"{name}-annotations"
    recorder = MatteDiagnosticRecorder(
        bundle_root,
        duration_s=10.0,
        max_bytes=64 * 1024 * 1024,
    )
    annotations: list[QualityFrameAnnotations] = []
    acceleration = AccelerationConfig().model_dump(mode="json")
    selected_indices = (0, 1, 2)
    start_ns = quality_evidence.START_NS
    if cadence == "fps_15":
        observed_timestamps = [start_ns + index * 66_666_667 for index in range(3)]
        observed_sequences = [1, 2, 3]
        capture_generations = [7, 7, 7]
        geometry_generations = [11, 11, 11]
    elif cadence == "fps_30":
        observed_timestamps = [start_ns + index * 33_333_333 for index in range(3)]
        observed_sequences = [1, 2, 3]
        capture_generations = [7, 7, 7]
        geometry_generations = [11, 11, 11]
    elif cadence == "fps_60":
        observed_timestamps = [start_ns + index * 16_666_667 for index in range(3)]
        observed_sequences = [1, 2, 3]
        capture_generations = [7, 7, 7]
        geometry_generations = [11, 11, 11]
    elif cadence == "discontinuous":
        observed_timestamps = [start_ns, start_ns + 21_000_000, start_ns + 91_000_000]
        observed_sequences = [1, 3, 4]
        capture_generations = [7, 7, 8]
        geometry_generations = [11, 11, 12]
    else:
        raise AssertionError(f"unsupported generated cadence {cadence}")
    background_controls = {**_BACKGROUND, "mode": background_mode}
    for sequence, (
        truth,
        flawed,
        backdrop,
        registration,
    ) in enumerate(
        zip(
            (ground_truth[index] for index in selected_indices),
            (defective[index] for index in selected_indices),
            (backdrops[index] for index in selected_indices),
            (registrations[index] for index in selected_indices),
        )
    ):
        timestamp = observed_timestamps[sequence]
        capture_sequence = observed_sequences[sequence]
        raw_small = quality_evidence._source(truth, sequence)
        raw = cv2.resize(raw_small, canvas, interpolation=cv2.INTER_LINEAR)
        truth = cv2.resize(truth, canvas, interpolation=cv2.INTER_LINEAR)
        flawed = cv2.resize(flawed, canvas, interpolation=cv2.INTER_LINEAR)
        if background_mode not in {"video", "camera"}:
            backdrop = backdrops[0]
        backdrop = cv2.resize(backdrop, canvas, interpolation=cv2.INTER_LINEAR)
        raw = np.ascontiguousarray(raw, dtype=np.uint8)
        truth = np.ascontiguousarray(truth, dtype=np.float32)
        flawed = np.ascontiguousarray(flawed, dtype=np.float32)
        backdrop = np.ascontiguousarray(backdrop, dtype=np.uint8)
        identity_ramp = (
            np.arange(8, dtype=np.uint8)[:, None] * 2
            + np.arange(8, dtype=np.uint8)[None, :] * 4
        )
        identity_base = 32 + sequence * 48
        backdrop[0:8, 0:8] = np.stack(
            (
                identity_base + identity_ramp,
                identity_base + identity_ramp + 8,
                identity_base + identity_ramp + 16,
            ),
            axis=2,
        ).astype(np.uint8)
        mask = truth.copy() if candidate and not use_defective_mask else flawed.copy()
        rendered = composite(
            raw,
            backdrop,
            mask,
            light_wrap=compositing.light_wrap,
            blend_space=compositing.blend_space,
        )
        evidence = MatteFrameEvidence(
            metadata=MatteCaptureMetadata(
                bundle_sequence=sequence,
                capture_sequence=capture_sequence,
                capture_monotonic_ns=timestamp,
                timestamp_source="capture-completion",
                capture_generation=capture_generations[sequence],
                geometry_generation=geometry_generations[sequence],
            ),
            raw_frame=raw,
            raw_mask=mask.copy(),
            refined_mask=mask.copy(),
            backdrop_frame=backdrop,
            base_composite=rendered,
            configured_controls={
                "segmentation": segmentation.model_dump(mode="json"),
                "acceleration": acceleration,
                "compositing": compositing.model_dump(mode="json"),
                "background": background_controls,
            },
            effective_controls={
                "segmentation_backend": "MediaPipeSegmenter",
                "segmentation_device": "cpu",
                "rvm_downsample_ratio": None,
                "mask_shift": 0,
                "light_wrap": compositing.light_wrap,
                "blend_space": compositing.blend_space,
                "matte_policy": matte_policy,
            },
            timings_ms={
                "backend_inference_ms": 1.0,
                "refinement_ms": 0.2,
                "background_ms": 0.1,
                "composite_ms": 0.2,
                "frame_processing_ms": 1.5,
                "output_send_ms": 0.1,
            },
            backdrop_identity={
                "provider": "generated-video",
                "presentation_index": sequence,
                "visual_generation": 3 + sequence,
            },
        )
        if not recorder.submit(evidence, rendered):
            raise AssertionError("generated matte evidence was unexpectedly rejected")
        recorder._queue.join()
        if not recorder.submit_output_event(
            sent_monotonic_ns=timestamp + 1_000_000,
            source_bundle_sequence=sequence,
            base_updated=True,
            exact_final_repeat=False,
        ):
            raise AssertionError(
                "generated output event was unexpectedly rejected: "
                f"{recorder._stop_reason}; {recorder.error}"
            )
        opaque, background_region = quality_evidence._regions(truth)
        annotations.append(
            QualityFrameAnnotations(
                segment="stationary",
                registration_from_previous=registration,
                opaque_core=opaque,
                background=background_region,
                ground_truth_alpha=truth,
                ground_truth_foreground=raw,
                regions=_named_regions(truth),
            )
        )
    recorder.close()
    write_quality_annotations(
        annotation_root,
        bundle_root,
        annotations,
        segments=(
            {
                "id": "stationary",
                "kind": "stationary",
                "start_sequence": 0,
                "end_sequence": len(annotations) - 1,
                "fixture": "generated-matte-5-2",
            },
        ),
        provenance={
            "kind": "generated",
            "license": "MIT",
            "generator": "tests/matte_visual_qualification_evidence.py",
            "contains_private_footage": False,
        },
        # Keep baseline/candidate annotation semantics identical. The
        # qualifier applies its own ratified gates; these embedded gates are
        # retained only as digest-bound source evidence.
        gates=_QUALITY_GATES,
    )
    return bundle_root, annotation_root, matte_policy


def _artifact_descriptor(path: Path, *, media_type: str) -> dict[str, object]:
    return {
        "filename": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "media_type": media_type,
    }


def _write_boundaries(
    root: Path,
    *,
    case_id: str,
    bundle_root: Path,
    native_applicable: bool,
) -> tuple[Path, dict[str, Path]]:
    bundle = MatteReplayBundle(bundle_root)
    sequence = min(2, len(bundle.frames) - 1)
    frame = bundle.frames[sequence]
    reference = bundle.load_array(frame, "final_composite")
    identity_region = np.ascontiguousarray(reference[0:8, 0:8], dtype=np.uint8)
    artifacts: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    for boundary_id in qualification.REQUIRED_BOUNDARIES:
        if boundary_id == "windows_native_loopback" and not native_applicable:
            artifacts[boundary_id] = {
                "status": "not_applicable",
                "artifact": None,
                "reason": "unsupported-canvas-or-cadence",
                "capture_method": "not-applicable",
                "platform": "generated",
            }
            continue
        media_type = (
            "image/jpeg"
            if boundary_id in {"snapshot_jpeg", "mjpeg_jpeg", "websocket_jpeg"}
            else "image/png"
        )
        suffix = ".jpg" if media_type == "image/jpeg" else ".png"
        path = root / f"{case_id}-boundary-{boundary_id}{suffix}"
        _write_private_image(path, reference, media_type=media_type)
        paths[boundary_id] = path
        artifacts[boundary_id] = {
            "status": "captured",
            "artifact": _artifact_descriptor(path, media_type=media_type),
            "reason": "",
            "capture_method": _GENERATED_CAPTURE_METHODS[boundary_id],
            "platform": "generated",
        }
    manifest_sha256 = _bundle_manifest_digest(bundle)
    final_descriptor = frame["artifacts"]["final_composite"]
    boundary = {
        "schema": qualification.BOUNDARY_SCHEMA,
        "version": qualification.BOUNDARY_VERSION,
        "case_id": case_id,
        "authority": "generated-fake",
        "source": {
            "bundle_manifest_sha256": manifest_sha256,
            "bundle_sequence": frame["sequence"],
            "capture_sequence": frame["capture_sequence"],
            "capture_generation": frame["capture_generation"],
            "geometry_generation": frame["geometry_generation"],
            "reference_final_composite_sha256": final_descriptor["sha256"],
            "identity_region": {
                "x": 0,
                "y": 0,
                "width": 8,
                "height": 8,
                "reference_region_sha256": _sha256_bytes(identity_region.tobytes()),
            },
        },
        "artifacts": artifacts,
    }
    boundary_path = root / f"{case_id}-boundaries.json"
    _write_private_json(boundary_path, boundary)
    return boundary_path, paths


_CASE_SPECS: tuple[dict[str, object], ...] = (
    {
        "id": "generated-small-video",
        "canvas": (quality_evidence.WIDTH, quality_evidence.HEIGHT),
        "background_mode": "video",
        "cadence": "fps_30",
        "coverage": {
            # Appearance/source tags exercise the strict vocabulary only.
            # Motion is bound to the actual stationary annotation segment.
            # Generated silhouettes cannot establish representative human
            # coverage, so provenance keeps the result pending regardless.
            "appearances": list(qualification.REQUIRED_COVERAGE["appearances"]),
            "motions": ["stationary"],
            "source_conditions": list(
                qualification.REQUIRED_COVERAGE["source_conditions"]
            ),
            "backgrounds": ["dynamic_video"],
            "cadences": ["fps_30"],
            # Deliberately truthful: a 96x72 generated array is none of the
            # release canvases. The aggregate report must list all three as
            # missing rather than granting generated evidence authority.
            "canvases": [],
        },
    },
)


def _review(
    *,
    plan_path: Path,
    boundary_path: Path,
    baseline_bundle: Path,
    baseline_annotations: Path,
    candidate_bundle: Path,
    candidate_annotations: Path,
    case_id: str,
) -> dict[str, object]:
    baseline_report = evaluate_bundle(
        baseline_bundle,
        annotations_root=baseline_annotations,
        metadata=EvaluationMetadata(
            backend="baseline",
            device="recorded",
            configuration_label=case_id,
        ),
    )
    candidate_report = evaluate_bundle(
        candidate_bundle,
        annotations_root=candidate_annotations,
        metadata=EvaluationMetadata(
            backend="MediaPipeSegmenter",
            device="cpu",
            configuration_label="selected-optional-candidate",
        ),
        baseline=baseline_report,
    )
    return {
        "schema": qualification.REVIEW_SCHEMA,
        "version": qualification.REVIEW_VERSION,
        "case_id": case_id,
        "provenance": {
            "kind": "generated",
            "reference": "MIT deterministic generated review fixture",
        },
        "bindings": {
            "plan_sha256": _sha256_file(plan_path),
            "baseline_bundle_manifest_sha256": _bundle_manifest_digest(
                MatteReplayBundle(baseline_bundle)
            ),
            "candidate_bundle_manifest_sha256": _bundle_manifest_digest(
                MatteReplayBundle(candidate_bundle)
            ),
            "baseline_annotation_manifest_sha256": MatteQualityAnnotations(
                baseline_annotations,
                MatteReplayBundle(baseline_bundle),
            ).manifest_sha256,
            "candidate_annotation_manifest_sha256": MatteQualityAnnotations(
                candidate_annotations,
                MatteReplayBundle(candidate_bundle),
            ).manifest_sha256,
            "baseline_quality_evidence_sha256": baseline_report["determinism"][
                "evidence_sha256"
            ],
            "candidate_quality_evidence_sha256": candidate_report["determinism"][
                "evidence_sha256"
            ],
            "boundary_evidence_sha256": _sha256_file(boundary_path),
        },
        "method": "side_by_side",
        "blinding": None,
        "reviewer": "generated-contract-reviewer",
        "reviewed_at": "2026-08-10T00:00:00Z",
        "concerns": {
            concern: "pass" for concern in qualification.REQUIRED_REVIEW_CONCERNS
        },
        "overall": "candidate_preferred",
        "notes": (
            "Generated schema exercise only; not a human or representative-video "
            "review."
        ),
    }


def generate_qualification(
    root: Path,
    *,
    provenance_kind: str = "generated",
    weak_candidate: bool = False,
) -> GeneratedQualification:
    """Create one complete generated plan under a new owner-only directory."""

    _private_directory(root)
    baseline_bundles: dict[str, Path] = {}
    baseline_annotations: dict[str, Path] = {}
    candidate_bundles: dict[str, Path] = {}
    candidate_annotations: dict[str, Path] = {}
    baseline_expectations: dict[str, dict[str, Any]] = {}
    candidate_policies: dict[str, dict[str, Any]] = {}
    boundaries: dict[str, Path] = {}
    all_boundary_artifacts: dict[str, Mapping[str, Path]] = {}
    reviews: dict[str, Path] = {}
    for raw_spec in _CASE_SPECS:
        case_id = str(raw_spec["id"])
        canvas = raw_spec["canvas"]
        assert isinstance(canvas, tuple)
        background_mode = str(raw_spec["background_mode"])
        cadence = str(raw_spec["cadence"])
        baseline_bundle, baseline_annotation, baseline_policy = _record_bundle(
            root,
            name=f"{case_id}-baseline",
            candidate=False,
            canvas=canvas,
            background_mode=background_mode,
            cadence=cadence,
        )
        candidate_bundle, candidate_annotation, candidate_policy = _record_bundle(
            root,
            name=f"{case_id}-candidate",
            candidate=True,
            use_defective_mask=weak_candidate,
            canvas=canvas,
            background_mode=background_mode,
            cadence=cadence,
        )
        baseline_bundles[case_id] = baseline_bundle
        baseline_annotations[case_id] = baseline_annotation
        candidate_bundles[case_id] = candidate_bundle
        candidate_annotations[case_id] = candidate_annotation
        baseline_segmentation, baseline_compositing, _ = _baseline_configuration(
            canvas_shape=(canvas[1], canvas[0]),
            dynamic_backdrop=background_mode in {"video", "camera"},
        )
        baseline_expectations[case_id] = {
            "segmentation": baseline_segmentation.model_dump(mode="json"),
            "compositing": baseline_compositing.model_dump(mode="json"),
            "effective": {
                "segmentation_backend": "MediaPipeSegmenter",
                "segmentation_device": "cpu",
                "rvm_downsample_ratio": None,
                "matte_policy": baseline_policy,
            },
        }
        candidate_policies[case_id] = candidate_policy
        # These are explicitly fake seam captures. They exercise the seven-way
        # boundary validator but never claim a physical Windows mode.
        native_applicable = True
        boundary, artifacts = _write_boundaries(
            root,
            case_id=case_id,
            bundle_root=candidate_bundle,
            native_applicable=native_applicable,
        )
        boundaries[case_id] = boundary
        all_boundary_artifacts[case_id] = artifacts
        reviews[case_id] = root / f"{case_id}-review.json"

    primary_case_id = "generated-small-video"
    segmentation, compositing, _policy = _selected_configuration()
    plan_path = root / "plan.json"
    plan = {
        "schema": qualification.PLAN_SCHEMA,
        "version": qualification.PLAN_VERSION,
        "provenance": {
            "qualification_id": "generated-matte-5-2-contract",
            "kind": provenance_kind,
            "license_or_consent_reference": (
                "MIT generated fixture"
                if provenance_kind == "generated"
                else "test-only consent reference"
            ),
            "contains_private_footage_in_repository": False,
        },
        "policy": {
            "quality_policy_id": "matte-0.2-ratified-v1",
            "maximum_full_frame_mae": 4.0,
            "maximum_edge_band_mae": 6.0,
        },
        "candidates": [
            {
                "id": "selected-optional-candidate",
                "segmentation": segmentation.model_dump(mode="json"),
                "compositing": compositing.model_dump(mode="json"),
            }
        ],
        "cases": [
            {
                "id": str(spec["id"]),
                "candidate_id": "selected-optional-candidate",
                "baseline": {
                    "bundle": str(baseline_bundles[str(spec["id"])]),
                    "annotations": str(baseline_annotations[str(spec["id"])]),
                },
                "candidate": {
                    "bundle": str(candidate_bundles[str(spec["id"])]),
                    "annotations": str(candidate_annotations[str(spec["id"])]),
                },
                "baseline_expected": baseline_expectations[str(spec["id"])],
                "expected_effective": {
                    "segmentation_backend": "MediaPipeSegmenter",
                    "segmentation_device": "cpu",
                    "rvm_downsample_ratio": None,
                    "matte_policy": candidate_policies[str(spec["id"])],
                },
                "coverage": spec["coverage"],
                "boundary_evidence": str(boundaries[str(spec["id"])]),
                "review": str(reviews[str(spec["id"])]),
            }
            for spec in _CASE_SPECS
        ],
    }
    _write_private_json(plan_path, plan)
    for case_id, review_path in reviews.items():
        _write_private_json(
            review_path,
            _review(
                plan_path=plan_path,
                boundary_path=boundaries[case_id],
                baseline_bundle=baseline_bundles[case_id],
                baseline_annotations=baseline_annotations[case_id],
                candidate_bundle=candidate_bundles[case_id],
                candidate_annotations=candidate_annotations[case_id],
                case_id=case_id,
            ),
        )
    return GeneratedQualification(
        plan=plan_path,
        baseline_bundle=baseline_bundles[primary_case_id],
        baseline_annotations=baseline_annotations[primary_case_id],
        candidate_bundle=candidate_bundles[primary_case_id],
        candidate_annotations=candidate_annotations[primary_case_id],
        boundary=boundaries[primary_case_id],
        review=reviews[primary_case_id],
        boundary_artifacts=all_boundary_artifacts[primary_case_id],
        baseline_bundles=baseline_bundles,
        baseline_annotation_sets=baseline_annotations,
        candidate_bundles=candidate_bundles,
        candidate_annotation_sets=candidate_annotations,
        boundaries=boundaries,
        reviews=reviews,
        all_boundary_artifacts=all_boundary_artifacts,
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read a fixture sidecar for a focused mutation test."""

    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise AssertionError("fixture JSON root is not an object")
    return value


def write_json(path: Path, value: object) -> None:
    """Rewrite one private fixture sidecar after a focused mutation."""

    _write_private_json(path, value)


def rebind_review(generated: GeneratedQualification) -> None:
    """Refresh review digests after an intentional plan/boundary mutation."""

    for case_id, review_path in generated.reviews.items():
        baseline_bundle = generated.baseline_bundles[case_id]
        baseline_annotations = generated.baseline_annotation_sets[case_id]
        candidate_bundle = generated.candidate_bundles[case_id]
        candidate_annotations = generated.candidate_annotation_sets[case_id]
        baseline_report = evaluate_bundle(
            baseline_bundle,
            annotations_root=baseline_annotations,
            metadata=EvaluationMetadata(
                backend="baseline",
                device="recorded",
                configuration_label=case_id,
            ),
        )
        candidate_report = evaluate_bundle(
            candidate_bundle,
            annotations_root=candidate_annotations,
            metadata=EvaluationMetadata(
                backend="MediaPipeSegmenter",
                device="cpu",
                configuration_label="selected-optional-candidate",
            ),
            baseline=baseline_report,
        )
        value = read_json(review_path)
        bindings = value["bindings"]
        bindings.update(
            {
                "plan_sha256": _sha256_file(generated.plan),
                "baseline_bundle_manifest_sha256": _bundle_manifest_digest(
                    MatteReplayBundle(baseline_bundle)
                ),
                "candidate_bundle_manifest_sha256": _bundle_manifest_digest(
                    MatteReplayBundle(candidate_bundle)
                ),
                "baseline_annotation_manifest_sha256": (
                    MatteQualityAnnotations(
                        baseline_annotations,
                        MatteReplayBundle(baseline_bundle),
                    ).manifest_sha256
                ),
                "candidate_annotation_manifest_sha256": (
                    MatteQualityAnnotations(
                        candidate_annotations,
                        MatteReplayBundle(candidate_bundle),
                    ).manifest_sha256
                ),
                "baseline_quality_evidence_sha256": baseline_report["determinism"][
                    "evidence_sha256"
                ],
                "candidate_quality_evidence_sha256": candidate_report["determinism"][
                    "evidence_sha256"
                ],
                "boundary_evidence_sha256": _sha256_file(generated.boundaries[case_id]),
            }
        )
        write_json(review_path, value)


def refresh_boundary_artifact(
    generated: GeneratedQualification,
    boundary_id: str,
) -> None:
    """Refresh a boundary descriptor after deliberately replacing its pixels."""

    value = read_json(generated.boundary)
    path = generated.boundary_artifacts[boundary_id]
    value["artifacts"][boundary_id]["artifact"] = _artifact_descriptor(
        path,
        media_type=(
            "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        ),
    )
    write_json(generated.boundary, value)
    rebind_review(generated)


def replace_boundary_pixels(
    generated: GeneratedQualification,
    boundary_id: str,
    pixels: np.ndarray,
) -> None:
    """Replace one boundary image and keep its declared digest truthful."""

    path = generated.boundary_artifacts[boundary_id]
    _write_private_image(
        path,
        np.ascontiguousarray(pixels, dtype=np.uint8),
        media_type=(
            "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        ),
    )
    refresh_boundary_artifact(generated, boundary_id)


def replace_candidate_final_composite(
    generated: GeneratedQualification,
    pixels: np.ndarray,
) -> None:
    """Replace the boundary-authoritative candidate composite and rebind it."""

    value = np.ascontiguousarray(pixels, dtype=np.uint8)
    boundary = read_json(generated.boundary)
    sequence = int(boundary["source"]["bundle_sequence"])
    manifest_path = generated.candidate_bundle / "manifest.json"
    manifest = read_json(manifest_path)
    descriptor = manifest["frames"][sequence]["artifacts"]["final_composite"]
    artifact_path = generated.candidate_bundle / descriptor["path"]
    payload = _npy_bytes(value)
    artifact_path.write_bytes(payload)
    artifact_path.chmod(0o600)
    descriptor.update(
        {
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    )
    write_json(manifest_path, manifest)
    rebind_annotations(generated.candidate_bundle, generated.candidate_annotations)

    identity = boundary["source"]["identity_region"]
    x = int(identity["x"])
    y = int(identity["y"])
    width = int(identity["width"])
    height = int(identity["height"])
    region = np.ascontiguousarray(
        value[y : y + height, x : x + width],
        dtype=np.uint8,
    )
    boundary["source"]["bundle_manifest_sha256"] = _bundle_manifest_digest(
        MatteReplayBundle(generated.candidate_bundle)
    )
    boundary["source"]["reference_final_composite_sha256"] = descriptor["sha256"]
    identity["reference_region_sha256"] = _sha256_bytes(region.tobytes())
    write_json(generated.boundary, boundary)
    rebind_review(generated)


def rebind_annotations(bundle_root: Path, annotation_root: Path) -> None:
    """Refresh the annotation source digest after a controlled manifest edit."""

    value = read_json(annotation_root / "annotations.json")
    value["source_bundle"]["manifest_sha256"] = _bundle_manifest_digest(
        MatteReplayBundle(bundle_root)
    )
    write_json(annotation_root / "annotations.json", value)


def rebind_candidate_bundle(generated: GeneratedQualification) -> None:
    """Refresh dependent digests after a candidate-manifest mutation."""

    rebind_annotations(generated.candidate_bundle, generated.candidate_annotations)
    boundary = read_json(generated.boundary)
    boundary["source"]["bundle_manifest_sha256"] = _bundle_manifest_digest(
        MatteReplayBundle(generated.candidate_bundle)
    )
    write_json(generated.boundary, boundary)
    rebind_review(generated)


if __name__ == "__main__":
    raise SystemExit(
        "This module only creates temporary generated test evidence; "
        "run tests/test_matte_visual_qualification.py instead."
    )
