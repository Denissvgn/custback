"""Deterministic, generated MATTE-2.5 qualification contract fixtures.

The default output is proxy evidence and can never select a profile.  Focused
tests may opt into a test-only ``model-backed`` attestation to exercise the
positive decision path; those synthetic arrays and timings are never packaged
as qualification results or represented as a real RVM benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import matte_quality_evidence as quality_evidence
import numpy as np
from custback.compositor import composite
from custback.config import (
    AccelerationConfig,
    AccelerationProvider,
    CompositingConfig,
    SegmentationConfig,
)
from custback.matte_ablation import (
    REPORT_SCHEMA as ABLATION_REPORT_SCHEMA,
    REPORT_VERSION as ABLATION_REPORT_VERSION,
)
from custback.matte_diagnostics import (
    MatteCaptureMetadata,
    MatteDiagnosticRecorder,
    MatteFrameEvidence,
    MatteReplayBundle,
)
from custback.matte_quality import (
    QualityFrameAnnotations,
    _bundle_manifest_digest,
    write_quality_annotations,
)
from custback.matte_policy import MatteBackendKind, resolve_matte_policy
from custback.matte_rvm_qualification import (
    PLAN_SCHEMA,
    PLAN_VERSION,
    RUN_SCHEMA,
    RUN_VERSION,
)
from custback.segmentation import RVM_MODEL


@dataclass(frozen=True)
class GeneratedQualification:
    plan: Path
    recorded_cell_ids: tuple[str, ...]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_private(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))
    path.chmod(0o600)


def _segments(
    frame_count: int,
    *,
    capture_stride: int = 1,
) -> tuple[list[dict[str, object]], list[str]]:
    kinds = ("stationary", "moving", "fast_motion", "occlusion")
    native_count = frame_count * capture_stride
    boundaries = [
        0,
        *[((index * native_count // 4) // 2) * 2 for index in range(1, 4)],
        native_count,
    ]
    segments: list[dict[str, object]] = []
    labels = [""] * frame_count
    for index, kind in enumerate(kinds):
        selected = [
            sequence
            for sequence in range(frame_count)
            if boundaries[index] <= capture_stride * sequence < boundaries[index + 1]
        ]
        assert selected
        start = selected[0]
        end = selected[-1]
        segment_id = f"coverage_{kind}"
        segments.append(
            {
                "id": segment_id,
                "kind": kind,
                "start_sequence": start,
                "end_sequence": end,
            }
        )
        for sequence in range(start, end + 1):
            labels[sequence] = segment_id
    return segments, labels


def _expanded_evidence(
    bundle_root: Path,
    annotation_root: Path,
    *,
    cadence: str,
    provider: str,
    send_delay_ms: float,
    slow_cadence: bool,
) -> None:
    """Expand a small physical fixture into a long, reference-reusing run."""

    target_count = 660 if cadence == "native30" else 330
    stride = 1 if cadence == "native30" else 2
    interval_ns = 33_333_333 * (20 if slow_cadence else 1)
    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    base_frames = manifest["frames"]
    first_timestamp = int(base_frames[0]["capture_monotonic_ns"])
    frames: list[dict[str, Any]] = []
    for sequence in range(target_count):
        frame = copy.deepcopy(base_frames[sequence % len(base_frames)])
        frame["sequence"] = sequence
        frame["capture_sequence"] = stride * sequence
        frame["capture_monotonic_ns"] = (
            first_timestamp + stride * sequence * interval_ns
        )
        frame["resource_samples"] = {
            "rss_bytes": 80_000_000 + sequence,
            **({} if provider == "cpu" else {"vram_bytes": 120_000_000 + sequence}),
        }
        frames.append(frame)
    manifest["frames"] = frames
    manifest["frame_count"] = target_count
    manifest["limits"]["duration_s"] = max(
        30.0,
        ((target_count - 1) * stride * interval_ns) / 1_000_000_000.0 + 1.0,
    )
    manifest["limits"]["max_bytes"] = 512 * 1024 * 1024
    manifest["artifact_bytes"] = sum(
        int(descriptor["bytes"])
        for frame in frames
        for descriptor in frame["artifacts"].values()
        if "alias_of" not in descriptor
    )
    manifest["output_timeline"] = {
        "version": 1,
        "timestamp_clock": "monotonic",
        "complete": True,
        "events": [
            {
                "sequence": sequence,
                "sent_monotonic_ns": frame["capture_monotonic_ns"]
                + int(round(send_delay_ms * 1_000_000.0)),
                "source_bundle_sequence": sequence,
                "base_composite_sequence": sequence,
                "base_updated": True,
                "exact_final_repeat": False,
                "post_base_final_output_provenance": None,
            }
            for sequence, frame in enumerate(frames)
        ],
    }
    _write_private(manifest_path, manifest)

    bundle = MatteReplayBundle(bundle_root)
    annotations_path = annotation_root / "annotations.json"
    annotation_manifest = json.loads(annotations_path.read_text(encoding="ascii"))
    base_annotations = annotation_manifest["frames"]
    segments, labels = _segments(target_count, capture_stride=stride)
    expanded_annotations: list[dict[str, Any]] = []
    for sequence in range(target_count):
        frame = copy.deepcopy(base_annotations[sequence % len(base_annotations)])
        frame["sequence"] = sequence
        frame["segment"] = labels[sequence]
        expanded_annotations.append(frame)
    annotation_manifest["source_bundle"]["manifest_sha256"] = _bundle_manifest_digest(
        bundle
    )
    annotation_manifest["segments"] = segments
    annotation_manifest["frames"] = expanded_annotations
    annotation_manifest["frame_count"] = target_count
    _write_private(annotations_path, annotation_manifest)


def _qualification_segmentation_sha256(segmentation: dict[str, Any]) -> str:
    projected = dict(segmentation)
    projected.pop("model_path", None)
    return _sha256(projected)


def _candidates() -> list[dict[str, Any]]:
    shortlist_segmentation = SegmentationConfig(
        backend="rvm",
        model_path="/private/rvm_resnet50_fp32.onnx",
        rvm_downsample=0.5,
        mask_blur=0,
        edge_refine=False,
        temporal_smoothing=0.0,
    ).model_dump(mode="json")
    auto_segmentation = SegmentationConfig(
        backend="rvm",
        rvm_downsample=0.0,
        mask_blur=0,
        edge_refine=False,
        temporal_smoothing=0.0,
    ).model_dump(mode="json")
    compatibility = SegmentationConfig().model_dump(mode="json")
    return [
        {
            "id": "shortlist",
            "role": "shortlist",
            "ablation_candidate_id": "rvm_shortlist",
            "model_id": "rvm_resnet50_fp32.onnx",
            "model_sha256": hashlib.sha256(
                b"qualified-test-rvm-alternative"
            ).hexdigest(),
            "model_bytes": 25_000_000,
            "segmentation": shortlist_segmentation,
        },
        {
            "id": "auto",
            "role": "auto",
            "ablation_candidate_id": "",
            "model_id": RVM_MODEL.filename,
            "model_sha256": RVM_MODEL.sha256,
            "model_bytes": RVM_MODEL.size,
            "segmentation": auto_segmentation,
        },
        {
            "id": "compat",
            "role": "compatibility_baseline",
            "ablation_candidate_id": "",
            "model_id": RVM_MODEL.filename,
            "model_sha256": RVM_MODEL.sha256,
            "model_bytes": RVM_MODEL.size,
            "segmentation": compatibility,
        },
    ]


def _gates(*, strict_opaque: bool) -> list[dict[str, object]]:
    paths = (
        ("opaque_p05", "opaque_core", "opaque_core_alpha_p05.p05", ">="),
        (
            "opaque_below",
            "opaque_core",
            "opaque_core_fraction_below_0_95.p95",
            "<=",
        ),
        ("holes", "opaque_core", "foreground_hole_components.max", "<="),
        ("background", "background", "background_alpha_mean.p95", "<="),
        ("halo_area", "halo", "exterior_halo_area_ratio.p95", "<="),
        ("halo_width", "halo", "exterior_halo_width_p95_px.p95", "<="),
        ("alpha_mse", "fine_detail", "ground_truth_alpha_mse.p95", "<="),
        (
            "gradient",
            "fine_detail",
            "ground_truth_gradient_mae.p95",
            "<=",
        ),
        (
            "uncertain",
            "fine_detail",
            "uncertain_pixel_fraction.p50",
            ">=",
        ),
        (
            "contour",
            "temporal",
            "contour_displacement_p95_px.p95",
            "<=",
        ),
        (
            "temporal_alpha",
            "temporal",
            "compensated_alpha_temporal_abs_diff.p95",
            "<=",
        ),
        ("trail", "temporal", "motion_trail_area_ratio.p95", "<="),
    )
    gates: list[dict[str, object]] = []
    ratified_thresholds = {
        "opaque_p05": 0.99 if strict_opaque else 0.95,
        "opaque_below": 0.05,
        "holes": 0.0,
        "background": 0.01,
        "halo_area": 0.05,
        "halo_width": 8.0,
        "alpha_mse": 0.01,
        "gradient": 0.10,
        "uncertain": 0.05,
        "contour": 1.5,
        "temporal_alpha": 0.10,
        "trail": 0.10,
    }
    for gate_id, family, suffix, op in paths:
        gates.append(
            {
                "id": gate_id,
                "family": family,
                "metric": f"aggregate.metrics.{suffix}",
                "op": op,
                "value": ratified_thresholds[gate_id],
            }
        )
    return gates


def _write_ablation(root: Path, candidate: dict[str, Any]) -> Path:
    configuration = {
        "qualification_segmentation_sha256": (
            _qualification_segmentation_sha256(candidate["segmentation"])
        ),
        "qualification_model_identity": candidate["model_id"],
        "qualification_model_sha256": candidate["model_sha256"],
        "qualification_model_bytes": candidate["model_bytes"],
    }
    report: dict[str, Any] = {
        "schema": ABLATION_REPORT_SCHEMA,
        "version": ABLATION_REPORT_VERSION,
        "source": {"kind": "generated-test"},
        "scope": {"model_inference_executed": True},
        "coverage": {"complete": True},
        "rows": [
            {
                "id": "rvm_shortlist",
                "lane": "rvm",
                "status": "completed",
                "evidence_kind": "model-backed",
                "configuration": configuration,
            }
        ],
        "decisions": {"matte_2_5_rvm_candidates": ["rvm_shortlist"]},
        "cadence_conclusion": {"complete": True},
        "required_decisions": {"complete": True},
    }
    evidence = {
        name: report[name]
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
    report["evidence_sha256"] = _sha256(evidence)
    path = root / "ablation.json"
    _write_private(path, report)
    return path


def _cell_id(
    candidate: str,
    hardware: str,
    provider: str,
    cadence: str,
    render_mode: str,
) -> str:
    cadence_code = "n" if cadence == "native30" else "d"
    mode_code = "r" if render_mode == "raw_model" else "q"
    return f"{candidate}_{hardware}_{provider}_{cadence_code}_{mode_code}"


def _hardware_inventory(hardware_id: str) -> tuple[str, str, dict[str, object]]:
    if hardware_id == "host1":
        return (
            "Linux CUDA test host",
            "linux",
            {
                "cpu_model": "Test CPU A",
                "accelerators": ["Test CUDA GPU"],
                "memory_bytes": 16_000_000_000,
                "os_name": "Test Linux",
                "os_version": "1.0",
                "architecture": "x86_64",
            },
        )
    return (
        "Windows DirectML test host",
        "windows",
        {
            "cpu_model": "Test CPU B",
            "accelerators": ["Test DirectML GPU"],
            "memory_bytes": 24_000_000_000,
            "os_name": "Test Windows",
            "os_version": "1.0",
            "architecture": "x86_64",
        },
    )


def _bundle_and_run(
    root: Path,
    *,
    cell_id: str,
    candidate: dict[str, Any],
    hardware_id: str,
    provider: str,
    cadence: str,
    render_mode: str,
    compositor: dict[str, Any],
    provenance_kind: str,
    fault: str | None,
    sustainable: bool,
    profile_semantics: bool,
    qualifying_quality: bool,
) -> tuple[Path, Path, Path]:
    (
        ground_truth,
        predicted,
        backdrops,
        registrations,
        timestamps,
        capture_sequences,
        _segment_kind,
    ) = quality_evidence._fixture_arrays("holes_and_halos")
    canonical_source_digests = [
        hashlib.sha256(
            np.ascontiguousarray(
                quality_evidence._source(alpha, source_index)
            ).tobytes()
        ).hexdigest()
        for source_index, alpha in enumerate(ground_truth)
    ]
    source_clip_sha256 = _sha256(canonical_source_digests)
    if fault == "slow_cadence":
        first_timestamp = timestamps[0]
        timestamps = [
            first_timestamp + 20 * (timestamp - first_timestamp)
            for timestamp in timestamps
        ]
    indices = list(range(6)) if cadence == "native30" else [0, 2, 4]
    bundle_root = root / f"{cell_id}-bundle"
    annotation_root = root / f"{cell_id}-annotations"
    run_path = root / f"{cell_id}-run.json"
    configured_compositor = dict(compositor)
    if render_mode == "raw_model":
        configured_compositor["use_model_foreground"] = False
        configured_compositor["light_wrap"] = 0.0
        configured_compositor = CompositingConfig.model_validate(
            configured_compositor
        ).model_dump(mode="json")
    segmentation = candidate["segmentation"]
    segmentation_config = SegmentationConfig.model_validate(segmentation)
    compositor_config = CompositingConfig.model_validate(configured_compositor)
    configured_background = {
        "mode": "color",
        "fit_mode": "cover",
        "anchor_x": 0.5,
        "anchor_y": 0.5,
    }
    configured_ratio = float(segmentation["rvm_downsample"])
    ratio = 1.0 if configured_ratio == 0.0 else configured_ratio
    acceleration = AccelerationConfig(
        mode="cpu" if provider == "cpu" else "gpu_required",
        provider=cast(
            AccelerationProvider,
            "auto" if provider == "cpu" else provider,
        ),
    ).model_dump(mode="json")
    annotations: list[QualityFrameAnnotations] = []
    recorder = MatteDiagnosticRecorder(
        bundle_root,
        duration_s=10.0,
        max_bytes=64 * 1024 * 1024,
    )
    for bundle_sequence, source_index in enumerate(indices):
        gt = ground_truth[source_index]
        mask = predicted[source_index]
        if profile_semantics:
            prediction_weight = {
                "compat": 0.50,
                "auto": 0.45,
                "shortlist": 0.40,
            }[candidate["id"]]
            mask = np.ascontiguousarray(
                prediction_weight * predicted[source_index]
                + (1.0 - prediction_weight) * gt,
                dtype=np.float32,
            )
            opaque, background = quality_evidence._regions(gt)
            mask[opaque] = 1.0
            # Performance/balanced deliberately trade protected soft-edge
            # support for lower service cost.  Keep annotated background
            # untouched so halo area/width remain measurable and ordinal.
            crisp_threshold = {
                "compat": 0.20,
                "auto": 0.10,
                "shortlist": 0.0,
            }[candidate["id"]]
            if crisp_threshold:
                crispable = ~background
                mask[crispable & (mask < crisp_threshold)] = 0.0
                mask[crispable & (mask > 1.0 - crisp_threshold)] = 1.0
        elif qualifying_quality:
            mask = np.ascontiguousarray(
                0.4 * predicted[source_index] + 0.6 * gt,
                dtype=np.float32,
            )
            opaque, _background = quality_evidence._regions(gt)
            mask[opaque] = 1.0
        backdrop = backdrops[source_index]
        raw = quality_evidence._source(gt, source_index)
        foreground = raw.copy()
        rendered = composite(
            raw,
            backdrop,
            mask,
            edge_foreground=(
                foreground if configured_compositor["use_model_foreground"] else None
            ),
            light_wrap=float(configured_compositor["light_wrap"]),
            blend_space=configured_compositor["blend_space"],
        )
        fallback = fault == "late_fallback" and bundle_sequence == len(indices) - 1
        frame_ratio = (
            0.75
            if fault == "ratio_drift" and bundle_sequence == len(indices) - 1
            else ratio
        )
        acceleration_state = {
            "applicable": True,
            "requested_mode": "cpu" if provider == "cpu" else "gpu_required",
            "requested_provider": "auto" if provider == "cpu" else provider,
            "device_id": 0,
            "state": "cpu_fallback" if provider == "cpu" else "gpu_active",
            "active_provider": provider,
            "fallback_active": fallback,
            "fallback_count": 1 if fallback else 0,
            "fallback_reason_code": "provider-fallback" if fallback else "",
        }
        service_ms = (
            70.0
            if fault == "timeline_overlap"
            else 20.0
            if fault == "budget"
            else (
                {"compat": 4.0, "auto": 5.0, "shortlist": 6.0}[candidate["id"]]
                if profile_semantics
                else 4.0
            )
        )
        timings = {
            "rvm_preprocess_ms": 0.5,
            "rvm_session_run_ms": (100.0 if fault == "contradictory_timing" else 2.0),
            "rvm_postprocess_ms": 0.3,
            "backend_inference_ms": 2.8,
            "refinement_ms": 0.1,
            "segmentation_ms": 2.9,
            "background_ms": 0.0,
            "color_correction_ms": 0.0,
            "composite_ms": 0.2,
            "frame_processing_ms": 3.2,
            "output_send_ms": 0.5,
            "frame_total_ms": service_ms,
        }
        telemetry = {
            "applicable": fault != "telemetry_not_applicable",
            "input_frame_shape": [quality_evidence.HEIGHT, quality_evidence.WIDTH],
            "output_alpha_shape": [quality_evidence.HEIGHT, quality_evidence.WIDTH],
            "output_foreground_shape": [
                quality_evidence.HEIGHT,
                quality_evidence.WIDTH,
                3,
            ],
            "configured_downsample_mode": (
                "auto" if configured_ratio == 0.0 else "explicit"
            ),
            "configured_downsample_ratio": configured_ratio,
            "resolved_downsample_ratio": frame_ratio,
            "preprocess_ms": timings["rvm_preprocess_ms"],
            "session_run_ms": timings["rvm_session_run_ms"],
            "postprocess_ms": timings["rvm_postprocess_ms"],
            "model_builtin": candidate["role"] != "shortlist",
            "model_identity": candidate["model_id"],
            "model_sha256": candidate["model_sha256"],
            "model_bytes": candidate["model_bytes"],
            "acceleration_state": acceleration_state["state"],
            "acceleration_active_provider": provider,
            "acceleration_fallback_active": fallback,
            "acceleration_fallback_count": 1 if fallback else 0,
        }
        evidence = MatteFrameEvidence(
            metadata=MatteCaptureMetadata(
                bundle_sequence=bundle_sequence,
                capture_sequence=capture_sequences[source_index],
                capture_monotonic_ns=timestamps[source_index],
                timestamp_source="capture-completion",
                capture_generation=0,
                geometry_generation=0,
            ),
            raw_frame=raw,
            raw_mask=mask.copy(),
            refined_mask=mask.copy(),
            clean_foreground=(
                None
                if fault == "missing_fgr" and bundle_sequence == len(indices) - 1
                else foreground
            ),
            backdrop_frame=backdrop,
            base_composite=rendered,
            configured_controls={
                "segmentation": segmentation,
                "acceleration": acceleration,
                "compositing": configured_compositor,
                "background": configured_background,
            },
            effective_controls={
                "segmentation_backend": "RVMSegmenter",
                "segmentation_device": provider,
                "produces_matte": True,
                "rvm_downsample_ratio": frame_ratio,
                "acceleration": acceleration_state,
                "rvm_telemetry": telemetry,
                "output_sink": {
                    "applicable": True,
                    "backend": "null",
                    "paces": False,
                },
                **(
                    lambda policy: {
                        "refiner": policy.effective_refiner_config(
                            segmentation_config
                        ).model_dump(mode="json"),
                        "edge_refinement_mode": (policy.effective.edge_refinement_mode),
                        "edge_refinement_radius_px": (
                            policy.effective.edge_refinement_radius_px
                        ),
                        "mask_shift": policy.effective.mask_shift,
                        "use_model_foreground": (policy.effective.use_model_foreground),
                        "light_wrap": policy.effective.light_wrap,
                        "blend_space": compositor_config.blend_space,
                        "matte_policy": policy.to_dict(),
                    }
                )(
                    resolve_matte_policy(
                        segmentation_config,
                        compositor_config,
                        MatteBackendKind.TRUE_ALPHA_RECURRENT,
                        resolved_rvm_ratio=frame_ratio,
                        canvas_shape=(
                            quality_evidence.HEIGHT,
                            quality_evidence.WIDTH,
                        ),
                        light_wrap_stabilization_eligible=False,
                    )
                ),
            },
            timings_ms=timings,
            resource_samples={
                "rss_bytes": 80_000_000 + bundle_sequence,
                **(
                    {}
                    if provider == "cpu"
                    else {"vram_bytes": 120_000_000 + bundle_sequence}
                ),
            },
            backdrop_identity={
                "provider": "generated",
                "logical_index": source_index,
            },
        )
        if render_mode == "qualified_compositor":
            if fault == "effective_foreground":
                evidence.effective_controls["use_model_foreground"] = False
            elif fault == "effective_wrap":
                evidence.effective_controls["light_wrap"] = 0.0
            elif fault == "effective_blend":
                evidence.effective_controls["blend_space"] = "linear_srgb"
            elif fault == "effective_refiner":
                tampered_refiner = dict(evidence.effective_controls["refiner"])
                tampered_refiner["mask_shift"] = 7
                evidence.effective_controls["refiner"] = tampered_refiner
        assert recorder.submit(evidence, rendered)
        recorder._queue.join()
        assert recorder.submit_output_event(
            sent_monotonic_ns=timestamps[source_index]
            + int(
                round(
                    (service_ms + 60_000.0 if fault == "queue_lie" else service_ms)
                    * 1_000_000.0
                )
            ),
            source_bundle_sequence=bundle_sequence,
            base_updated=True,
            exact_final_repeat=False,
        )
        opaque, background = quality_evidence._regions(gt)
        if bundle_sequence == len(indices) - 1 and (
            (fault == "annotation_drift" and render_mode == "qualified_compositor")
            or (fault == "decimated_annotation_drift" and cadence == "decimated15")
        ):
            opaque = opaque.copy()
            opaque.flat[0] = not bool(opaque.flat[0])
        annotations.append(
            QualityFrameAnnotations(
                segment="qualification",
                registration_from_previous=registrations[source_index],
                opaque_core=opaque,
                background=background,
                ground_truth_alpha=gt,
                ground_truth_foreground=raw,
            )
        )
    recorder.close()
    annotation_gates = _gates(strict_opaque=False)
    if fault == "annotation_gate":
        next(gate for gate in annotation_gates if gate["id"] == "alpha_mse")[
            "value"
        ] = 0.0
    write_quality_annotations(
        annotation_root,
        bundle_root,
        annotations,
        segments=(
            {
                "id": "qualification",
                "kind": "stationary",
                "start_sequence": 0,
                "end_sequence": len(indices) - 1,
            },
        ),
        provenance={
            "kind": provenance_kind,
            "license": "generated-test-license",
            "contains_private_footage": False,
        },
        gates=annotation_gates,
    )
    service_value = (
        70.0
        if fault == "timeline_overlap"
        else 20.0
        if fault == "budget"
        else (
            {"compat": 4.0, "auto": 5.0, "shortlist": 6.0}[candidate["id"]]
            if profile_semantics
            else 4.0
        )
    )
    if sustainable:
        _expanded_evidence(
            bundle_root,
            annotation_root,
            cadence=cadence,
            provider=provider,
            send_delay_ms=(
                service_value + 60_000.0 if fault == "queue_lie" else service_value
            ),
            slow_cadence=fault == "slow_cadence",
        )
    bundle = MatteReplayBundle(bundle_root)
    frame_count = len(bundle.frames)
    capture_span = (
        bundle.frames[-1]["capture_monotonic_ns"]
        - bundle.frames[0]["capture_monotonic_ns"]
    )
    fps = (frame_count - 1) * 1_000_000_000.0 / capture_span
    run_capture_fps = fps + 1.0 if fault == "cadence_lie" else fps
    hardware_label, hardware_platform, hardware_inventory = _hardware_inventory(
        hardware_id
    )
    execution_provider = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "directml": "DmlExecutionProvider",
    }[provider]
    source_frame_digests = [
        canonical_source_digests[indices[sequence % len(indices)]]
        for sequence in range(frame_count)
    ]
    if fault == "pre_resize_source_drift" and cadence == "decimated15":
        source_frame_digests[0] = hashlib.sha256(
            b"different-pre-resize-source"
        ).hexdigest()
    if fault == "native_source_family_drift" and render_mode == "qualified_compositor":
        source_frame_digests[0] = hashlib.sha256(
            b"different-cross-matrix-source"
        ).hexdigest()
    run = {
        "schema": RUN_SCHEMA,
        "version": RUN_VERSION,
        "bundle_manifest_sha256": _bundle_manifest_digest(bundle),
        "evidence_kind": (
            "generated-proxy" if provenance_kind == "generated" else "model-backed"
        ),
        "provenance": {
            "kind": provenance_kind,
            "reference": ("" if provenance_kind == "generated" else "test-consent"),
            "contains_private_pixels": False,
        },
        "source": {
            "clip_sha256": source_clip_sha256,
            "frame_sha256": source_frame_digests,
            "pixel_contract": "canonical-pre-resize-rgb8",
        },
        "hardware": {
            "id": hardware_id,
            "label": hardware_label,
            "platform": hardware_platform,
            "identity_sha256": _sha256(hardware_inventory),
            "identity_source": "qualification-hardware-inventory-v1",
            "inventory": hardware_inventory,
        },
        "model": {
            "id": candidate["model_id"],
            "sha256": candidate["model_sha256"],
            "bytes": candidate["model_bytes"],
            "license": "GPL-3.0",
            "license_reviewed": True,
            "packaging_supported": True,
            "download_integrity": True,
            "startup_succeeded": True,
        },
        "provider": {
            "requested": provider,
            "execution_proven": True,
            "runtime": {
                "onnxruntime_version": "1.20.0",
                "execution_provider": execution_provider,
                "provider_runtime_version": "1.20.0",
                "driver_version": (
                    "different-driver"
                    if fault == "environment_drift"
                    and render_mode == "qualified_compositor"
                    else ("not-applicable" if provider == "cpu" else "test-driver-1.0")
                ),
            },
        },
        "resource_evidence": {
            "sample_alignment": "one-per-unique-frame",
            "rss_source": "process-api",
            "rss_semantics": "process-current-resident-bytes",
            "sampling_point": "post-sink-submit",
            "vram_applicable": provider != "cpu",
            "vram_source": ("not-applicable" if provider == "cpu" else "provider-api"),
            "vram_semantics": (
                "not-applicable"
                if provider == "cpu"
                else "process-current-allocated-bytes"
            ),
        },
        "recurrence": {
            "input_selection": (
                "native-all"
                if cadence == "native30"
                else "every-other-native-preserved-timestamps"
            ),
            "model_invocation_count": frame_count,
            "fresh_temporal_state": True,
            "output_projection_used": False,
        },
        "warmup_frame_count": 30 if sustainable else 1,
        "startup_ms": [6.0],
        "memory_bytes": [80_000_000 + index for index in range(frame_count)],
        "vram_bytes": (
            None
            if provider == "cpu"
            else [120_000_000 + index for index in range(frame_count)]
        ),
        "service": {
            "new_frame_service_ms": [service_value] * frame_count,
            "queue_age_ms": [0.0] * frame_count,
            "duration_s": capture_span / 1_000_000_000.0,
            "capture_fps": run_capture_fps,
            "unique_composite_fps": fps,
            "output_fps": fps,
            "output_repeat_count": 0,
            "output_timeline_complete": True,
            "boundary": (
                "unique-dequeue-through-sink-submit-excluding-deliberate-pacing"
            ),
            "bundle_frame_total_alignment": "exact-non-pacing-sink",
        },
    }
    _write_private(run_path, run)
    return bundle_root, annotation_root, run_path


def generate_qualification(
    root: Path,
    *,
    provenance_kind: str = "generated",
    fault: str | None = None,
    strict_opaque: bool = False,
    sustainable: bool = False,
    record_all: bool = False,
    propose_profiles: bool = False,
) -> GeneratedQualification:
    candidates = _candidates()
    ablation = _write_ablation(root, candidates[0])
    compositor = CompositingConfig(
        light_wrap=0.1,
        use_model_foreground=True,
    ).model_dump(mode="json")
    hardware = (
        {"id": "host1", "providers": ["cpu", "cuda"], "canvas_ids": ["sd"]},
        {
            "id": "host2",
            "providers": ["cpu", "directml"],
            "canvas_ids": ["sd"],
        },
    )
    keys = [
        (
            candidate["id"],
            machine["id"],
            provider,
            cadence,
            render_mode,
        )
        for candidate in candidates
        for machine in hardware
        for provider in machine["providers"]
        for cadence in ("native30", "decimated15")
        for render_mode in ("raw_model", "qualified_compositor")
    ]
    ids = {key: _cell_id(*key) for key in keys}
    recorded_key_prefix = ("shortlist", "host1", "cpu")
    cells: list[dict[str, object]] = []
    recorded_ids: list[str] = []
    for key in keys:
        candidate_id, hardware_id, provider, cadence, render_mode = key
        cell_id = ids[key]
        native_id = (
            ""
            if cadence == "native30"
            else ids[
                (
                    candidate_id,
                    hardware_id,
                    provider,
                    "native30",
                    render_mode,
                )
            ]
        )
        pair_id = (
            ""
            if render_mode == "raw_model"
            else ids[
                (
                    candidate_id,
                    hardware_id,
                    provider,
                    cadence,
                    "raw_model",
                )
            ]
        )
        recorded = record_all or key[:3] == recorded_key_prefix
        bundle = annotations = run = ""
        if recorded:
            candidate = next(item for item in candidates if item["id"] == candidate_id)
            bundle_path, annotations_path, run_path = _bundle_and_run(
                root,
                cell_id=cell_id,
                candidate=candidate,
                hardware_id=hardware_id,
                provider=provider,
                cadence=cadence,
                render_mode=render_mode,
                compositor=compositor,
                provenance_kind=provenance_kind,
                fault=fault,
                sustainable=sustainable,
                profile_semantics=propose_profiles,
                qualifying_quality=not strict_opaque,
            )
            bundle = str(bundle_path)
            annotations = str(annotations_path)
            run = str(run_path)
            recorded_ids.append(cell_id)
        cells.append(
            {
                "id": cell_id,
                "candidate_id": candidate_id,
                "hardware_id": hardware_id,
                "provider": provider,
                "canvas_id": "sd",
                "cadence": cadence,
                "render_mode": render_mode,
                "status": "recorded" if recorded else "unavailable",
                "bundle": bundle,
                "annotations": annotations,
                "run": run,
                "availability_reason": "" if recorded else "not_collected",
                "native30_cell_id": native_id,
                "paired_raw_cell_id": pair_id,
            }
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "version": PLAN_VERSION,
        "ablation_report": str(ablation),
        "provenance": {
            "qualification_id": "generated_qualification",
            "kind": provenance_kind,
            "license_or_consent_reference": (
                "" if provenance_kind == "generated" else "test-consent"
            ),
            "contains_private_footage_in_repository": False,
        },
        "candidates": candidates,
        "hardware": list(hardware),
        "canvases": [
            {
                "id": "sd",
                "width": quality_evidence.WIDTH,
                "height": quality_evidence.HEIGHT,
            }
        ],
        "qualified_compositor": compositor,
        "policy": {
            "quality_policy_id": "matte-0.2-ratified-v1",
            "quality_gates": _gates(strict_opaque=strict_opaque),
            "native30_service_p95_ms": 10.0,
            "native30_min_unique_fps": 29.0,
            "decimated15_service_p95_ms": 10.0,
            "decimated15_min_unique_fps": 14.5,
            "min_warmup_frames": 30,
            "min_steady_frames": 300,
            "min_observation_s": 10.0,
            "max_queue_age_ms": 10.0,
            "max_queue_age_growth_ms": 1.0,
            "minimum_profile_hardware_count": 2,
        },
        "profile_proposals": (
            [
                {"name": "performance", "candidate_id": "compat"},
                {"name": "balanced", "candidate_id": "auto"},
                {"name": "quality", "candidate_id": "shortlist"},
            ]
            if propose_profiles
            else []
        ),
        "cells": cells,
    }
    plan_path = root / "qualification-plan.json"
    _write_private(plan_path, plan)
    return GeneratedQualification(plan_path, tuple(recorded_ids))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic proxy MATTE-2.5 matrix",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.output)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    generated = generate_qualification(root)
    print(generated.plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
