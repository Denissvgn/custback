#!/usr/bin/env python3
"""Generated, deterministic MATTE-0.3 screening evidence.

The generated rows exercise the matrix/report contract without claiming that
RVM, CUDA, or MediaPipe inference ran. Real backend rows use the same plan
schema with ``evidence_kind: model-backed`` and private recorded bundles.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

from custback.compositor import composite
from custback.matte_ablation import PLAN_SCHEMA, PLAN_VERSION, run_ablation
from custback.matte_diagnostics import (
    MatteCaptureMetadata,
    MatteDiagnosticRecorder,
    MatteFrameEvidence,
)
from custback.matte_quality import (
    QualityFrameAnnotations,
    QualityNamedRegion,
    write_quality_annotations,
)

HEIGHT = 54
WIDTH = 72
FRAME_COUNT = 6
START_NS = 20_000_000_000


def _scene() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, QualityNamedRegion],
]:
    hard = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(hard, (19, 27), (53, 50), 1, -1)
    cv2.ellipse(hard, (36, 18), (10, 13), 0, 0, 360, 1, -1)
    cv2.rectangle(hard, (23, 14), (26, 25), 1, -1)
    cv2.rectangle(hard, (46, 14), (49, 25), 1, -1)
    truth = cv2.GaussianBlur(hard.astype(np.float32), (5, 5), 0)
    original = np.full((HEIGHT, WIDTH, 3), (30, 50, 70), np.uint8)
    foreground = np.full_like(original, (70, 160, 225))
    source = np.rint(
        foreground.astype(np.float32) * truth[..., None]
        + original.astype(np.float32) * (1.0 - truth[..., None])
    ).astype(np.uint8)

    torso = np.zeros_like(hard)
    torso[34:46, 27:46] = 1
    shoulders = np.zeros_like(hard)
    shoulders[29:33, 23:50] = 1
    head = np.zeros_like(hard)
    head[13:21, 32:41] = 1
    headphones = np.zeros_like(hard)
    headphones[16:22, 25:27] = 1
    headphones[16:22, 45:47] = 1
    yy = np.indices(truth.shape)[0]
    hair = ((truth > 0.05) & (truth < 0.95) & (yy < 29)).astype(np.uint8)
    background = (truth <= 0.01).astype(np.uint8)
    regions = {
        "torso": QualityNamedRegion("opaque_core", torso),
        "shoulders": QualityNamedRegion("opaque_core", shoulders),
        "head": QualityNamedRegion("opaque_core", head),
        "headphones": QualityNamedRegion("opaque_core", headphones),
        "hair": QualityNamedRegion("soft_boundary", hair),
        "background": QualityNamedRegion("background", background),
    }
    return source, truth.astype(np.float32), foreground, regions


def _backdrop(sequence: int, *, dynamic: bool) -> np.ndarray:
    yy, xx = np.indices((HEIGHT, WIDTH))
    phase = sequence * 41 if dynamic else 0
    return np.stack(
        (
            (210 + xx + phase) % 256,
            (35 + yy * 2 + phase * 2) % 256,
            (20 + xx + yy + phase * 3) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def _prediction(
    truth: np.ndarray,
    sequence: int,
    profile: Literal["baseline", "stable", "jitter"],
) -> np.ndarray:
    if profile == "stable":
        return truth.copy()
    if profile == "jitter":
        return cv2.warpAffine(
            truth,
            np.asarray(
                [[1.0, 0.0, 2.0 if sequence % 2 else 0.0], [0.0, 1.0, 0.0]],
                np.float32,
            ),
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(np.float32)
    yy, xx = np.indices(truth.shape)
    noise = (((xx * 7 + yy * 11 + sequence * 5) % 9) - 4) / 300.0
    return np.clip(truth + noise.astype(np.float32), 0.0, 1.0)


def _controls(
    *,
    backend: str,
    ratio: float,
    light_wrap: float,
) -> dict[str, object]:
    return {
        "segmentation": {
            "backend": backend,
            "model_path": "",
            "delegate": "cpu",
            "rvm_downsample": ratio,
            "threshold": 0.5,
            "mask_blur": 0,
            "edge_refine": False,
            "mask_shift": 0,
            "temporal_smoothing": 0.0,
        },
        "acceleration": {"mode": "cpu", "provider": "auto", "device_id": 0},
        "compositing": {
            "light_wrap": light_wrap,
            "use_model_foreground": True,
            "blend_space": "srgb_legacy",
            "color_correction": {
                "mode": "off",
                "strength": 0.5,
                "exposure_limit_ev": 0.85,
                "white_balance_strength": 0.5,
                "adaptation_time_s": 0.8,
            },
        },
        "background": {
            "mode": "video",
            "fit_mode": "cover",
            "anchor_x": 0.5,
            "anchor_y": 0.5,
        },
    }


def generate_bundle(
    root: Path,
    name: str,
    *,
    profile: Literal["baseline", "stable", "jitter"],
    backend: str,
    ratio: float,
    dynamic_backdrop: bool = True,
    source_offset: int = 0,
) -> tuple[Path, Path]:
    source, truth, foreground, regions = _scene()
    if source_offset:
        source = np.clip(source.astype(np.int16) + source_offset, 0, 255).astype(
            np.uint8
        )
    bundle = root / f"{name}-bundle"
    annotations = root / f"{name}-annotations"
    controls = _controls(backend=backend, ratio=ratio, light_wrap=0.8)
    annotation_frames: list[QualityFrameAnnotations] = []
    with MatteDiagnosticRecorder(bundle, max_bytes=32 * 1024 * 1024) as recorder:
        for sequence in range(FRAME_COUNT):
            prediction = np.ascontiguousarray(
                _prediction(truth, sequence, profile), dtype=np.float32
            )
            backdrop = _backdrop(sequence, dynamic=dynamic_backdrop)
            rendered = composite(
                source,
                backdrop,
                prediction,
                light_wrap=0.8,
                edge_foreground=foreground,
            )
            inference_ms = (8.0 if backend == "rvm" else 12.0) + sequence * 0.1
            evidence = MatteFrameEvidence(
                metadata=MatteCaptureMetadata(
                    bundle_sequence=sequence,
                    capture_sequence=500 + sequence,
                    capture_monotonic_ns=START_NS + sequence * 33_333_333,
                    timestamp_source="capture-completion",
                    capture_generation=1,
                    geometry_generation=1,
                ),
                raw_frame=source,
                raw_mask=prediction,
                refined_mask=prediction.copy(),
                clean_foreground=foreground,
                backdrop_frame=backdrop,
                base_composite=rendered,
                configured_controls=controls,
                effective_controls={
                    "segmentation_backend": (
                        "RVMSegmenter" if backend == "rvm" else "MediaPipeSegmenter"
                    ),
                    "segmentation_device": "generated-proxy",
                    "produces_matte": backend == "rvm",
                    "rvm_downsample_ratio": ratio if backend == "rvm" else None,
                    "mask_shift": 0,
                    "use_model_foreground": True,
                    "light_wrap": 0.8,
                    "blend_space": "srgb_legacy",
                },
                timings_ms={
                    "backend_inference_ms": inference_ms,
                    "refinement_ms": 0.3,
                    "background_ms": 0.2,
                    "composite_ms": 0.5,
                    "frame_processing_ms": inference_ms + 1.0,
                },
                compositor_substages_ms={
                    "foreground_selection_ms": 0.1,
                    "light_wrap_ms": 0.25,
                    "alpha_blend_ms": 0.15,
                },
                backdrop_identity={
                    "provider": (
                        "generated-container-video"
                        if dynamic_backdrop
                        else "generated-constant"
                    ),
                    "logical_index": sequence if dynamic_backdrop else 0,
                    "pts_s": sequence / 24.0 if dynamic_backdrop else 0.0,
                },
            )
            assert recorder.submit(evidence, rendered)
            recorder._queue.join()
            assert recorder.submit_output_event(
                sent_monotonic_ns=START_NS + sequence * 33_333_333 + 1_000_000,
                source_bundle_sequence=sequence,
                base_updated=True,
                exact_final_repeat=False,
            )
            core = np.maximum.reduce(
                [
                    regions["torso"].mask,
                    regions["shoulders"].mask,
                    regions["head"].mask,
                    regions["headphones"].mask,
                ]
            )
            annotation_frames.append(
                QualityFrameAnnotations(
                    segment="stationary",
                    opaque_core=core,
                    background=regions["background"].mask,
                    ground_truth_alpha=truth,
                    ground_truth_foreground=foreground,
                    regions=regions,
                )
            )
    write_quality_annotations(
        annotations,
        bundle,
        annotation_frames,
        segments=[
            {
                "id": "stationary",
                "kind": "stationary",
                "start_sequence": 0,
                "end_sequence": FRAME_COUNT - 1,
            }
        ],
        provenance={
            "kind": "generated",
            "license": "MIT",
            "generator": "tests/matte_ablation_evidence.py",
            "contains_private_footage": False,
        },
    )
    return bundle, annotations


def generated_plan(
    root: Path,
    *,
    rvm_stable: tuple[Path, Path],
    mediapipe_off: tuple[Path, Path],
    mediapipe_on: tuple[Path, Path],
    constant_backdrop: tuple[Path, Path],
) -> Path:
    variants: list[dict[str, object]] = [
        {
            "id": "rvm_ratio_067_proxy",
            "kind": "recorded",
            "lane": "rvm",
            "group": "model_profile",
            "covers": ["rvm.ratio.0_67"],
            "bundle": str(rvm_stable[0]),
            "annotations": str(rvm_stable[1]),
            "timestamp_policy": "exact",
            "evidence_kind": "generated-proxy",
            "expected_backend": "RVMSegmenter",
            "expected_device": "generated-proxy",
            "expected_rvm_downsample_ratio": 0.67,
        },
        {
            "id": "rvm_cuda_unavailable",
            "kind": "unavailable",
            "lane": "rvm",
            "group": "backend",
            "covers": ["backend.rvm_cuda"],
            "availability_reason": "generated evidence host did not run CUDA",
            "evidence_kind": "unavailable",
        },
        {
            "id": "rvm_cpu_unavailable",
            "kind": "unavailable",
            "lane": "rvm",
            "group": "backend",
            "covers": ["backend.rvm_cpu"],
            "availability_reason": "generated evidence host did not run RVM CPU",
            "evidence_kind": "unavailable",
        },
        {
            "id": "mediapipe_watershed_off_proxy",
            "kind": "recorded",
            "lane": "mediapipe",
            "group": "backend",
            "covers": ["mediapipe.edge_refine.off"],
            "bundle": str(mediapipe_off[0]),
            "annotations": str(mediapipe_off[1]),
            "timestamp_policy": "exact",
            "evidence_kind": "generated-proxy",
            "reference_in_group": True,
            "expected_backend": "MediaPipeSegmenter",
            "expected_device": "generated-proxy",
        },
        {
            "id": "mediapipe_watershed_on_proxy",
            "kind": "recorded",
            "lane": "mediapipe",
            "group": "backend",
            "covers": ["mediapipe.edge_refine.on"],
            "bundle": str(mediapipe_on[0]),
            "annotations": str(mediapipe_on[1]),
            "timestamp_policy": "exact",
            "evidence_kind": "generated-proxy",
            "expected_backend": "MediaPipeSegmenter",
            "expected_device": "generated-proxy",
        },
        {
            "id": "constant_backdrop_proxy",
            "kind": "recorded",
            "lane": "compositor",
            "group": "backdrop",
            "covers": ["backdrop.constant"],
            "bundle": str(constant_backdrop[0]),
            "annotations": str(constant_backdrop[1]),
            "timestamp_policy": "exact",
            "evidence_kind": "generated-proxy",
            "expected_backend": "RVMSegmenter",
            "expected_device": "generated-proxy",
            "expected_rvm_downsample_ratio": 0.4,
        },
    ]
    for variant_id, foreground_enabled, wrap in (
        ("compositor_plain", False, 0.0),
        ("compositor_foreground_only", True, 0.0),
        ("compositor_wrap_only", False, 0.8),
        ("compositor_both", True, 0.8),
    ):
        variants.append(
            {
                "id": variant_id,
                "kind": "frozen",
                "lane": "compositor",
                "group": "compositor_factorial",
                "covers": [f"compositor.{variant_id.removeprefix('compositor_')}"],
                "compositing": {
                    "use_model_foreground": foreground_enabled,
                    "light_wrap": wrap,
                },
                "evidence_kind": "recorded-intermediate",
                "reference_in_group": variant_id == "compositor_both",
            }
        )
    variants.extend(
        [
            {
                "id": "blend_linear",
                "kind": "frozen",
                "lane": "compositor",
                "group": "blend_space",
                "covers": ["blend.linear_srgb"],
                "compositing": {"blend_space": "linear_srgb"},
            },
            {
                "id": "mask_shift_negative",
                "kind": "frozen",
                "lane": "rvm",
                "group": "one_variable",
                "covers": ["rvm.mask_shift.negative"],
                "postprocess": {"mask_shift": -1},
            },
            {
                "id": "mask_shift_positive",
                "kind": "frozen",
                "lane": "rvm",
                "group": "one_variable",
                "covers": ["rvm.mask_shift.positive"],
                "postprocess": {"mask_shift": 1},
            },
            {
                "id": "mediapipe_blur_wide_proxy",
                "kind": "frozen",
                "lane": "mediapipe",
                "group": "one_variable",
                "covers": ["mediapipe.mask_blur.wide"],
                "postprocess": {"mask_blur": 11},
                "evidence_kind": "generated-proxy",
                "experimental_complexity": True,
            },
        ]
    )
    for variant_id, schedule, multiplier in (
        ("cadence_15", {"type": "fixed", "fps": 15}, 1),
        ("cadence_30", {"type": "fixed", "fps": 30}, 1),
        ("cadence_60", {"type": "fixed", "fps": 60}, 1),
        ("cadence_decimate_30_to_15", {"type": "decimate", "factor": 2}, 1),
        (
            "cadence_irregular",
            {
                "type": "irregular",
                "deltas_ns": [
                    0,
                    20_000_000,
                    70_000_000,
                    100_000_000,
                    165_000_000,
                    210_000_000,
                ],
            },
            1,
        ),
        ("output_repeat_15_to_30", {"type": "fixed", "fps": 15}, 2),
    ):
        variants.append(
            {
                "id": variant_id,
                "kind": "cadence_projection",
                "lane": "cadence",
                "group": "cadence",
                "covers": [f"cadence.{variant_id.removeprefix('cadence_')}"],
                "schedule": schedule,
                "output_multiplier": multiplier,
                "evidence_kind": "projection",
            }
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "version": PLAN_VERSION,
        "scope": {
            "host_label": "generated CI evidence host",
            "baseline_lane": "rvm",
            "one_host_screening": True,
            "model_inference_executed": False,
            "reported_screenshots_used_as_ab": False,
        },
        "required_axes": [
            "backend.rvm_cuda",
            "backend.rvm_cpu",
            "rvm.ratio.0_67",
            "mediapipe.edge_refine.off",
            "mediapipe.edge_refine.on",
            "compositor.plain",
            "compositor.foreground_only",
            "compositor.wrap_only",
            "compositor.both",
            "blend.linear_srgb",
            "backdrop.constant",
            "cadence.15",
            "cadence.30",
            "cadence.60",
            "cadence.decimate_30_to_15",
            "cadence.irregular",
            "cadence.output_repeat_15_to_30",
        ],
        "decision_policy": {
            "shortlist_limit_per_lane": 2,
            "performance_budget_ms": 40.0,
        },
        "variants": variants,
    }
    path = root / "ablation-plan.json"
    path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n", "utf-8")
    os.chmod(path, 0o600)
    return path


def generate_screen(root: Path, output: Path) -> dict[str, object]:
    source = generate_bundle(
        root,
        "source",
        profile="baseline",
        backend="rvm",
        ratio=0.4,
    )
    stable = generate_bundle(
        root,
        "rvm-ratio-067",
        profile="stable",
        backend="rvm",
        ratio=0.67,
    )
    mediapipe_off = generate_bundle(
        root,
        "mediapipe-off",
        profile="jitter",
        backend="mediapipe",
        ratio=0.0,
    )
    mediapipe_on = generate_bundle(
        root,
        "mediapipe-on",
        profile="stable",
        backend="mediapipe",
        ratio=0.0,
    )
    constant = generate_bundle(
        root,
        "constant",
        profile="baseline",
        backend="rvm",
        ratio=0.4,
        dynamic_backdrop=False,
    )
    plan = generated_plan(
        root,
        rvm_stable=stable,
        mediapipe_off=mediapipe_off,
        mediapipe_on=mediapipe_on,
        constant_backdrop=constant,
    )
    return run_ablation(
        source[0],
        source[1],
        plan,
        output,
        max_output_bytes=512 * 1024 * 1024,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--work-dir",
        required=True,
        help="new or empty directory that retains generated bundles/contact sheets",
    )
    args = parser.parse_args(argv)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    report = generate_screen(work, work / "output")
    Path(args.json).write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        "utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
