#!/usr/bin/env python3
"""Generated, deterministic MATTE-0.2 evidence and baseline harness.

All pixels, mattes, annotations, and output timelines are generated in memory
under the repository's MIT license.  No private or third-party footage is
embedded in this module or in the checked-in reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import cv2
import numpy as np

from custback.compositor import composite
from custback.matte_diagnostics import (
    MatteCaptureMetadata,
    MatteDiagnosticRecorder,
    MatteFrameEvidence,
    MatteReplayBundle,
)
from custback.matte_quality import (
    EvaluationMetadata,
    QualityFrameAnnotations,
    evaluate_bundle,
    evaluate_gates,
    report_markdown,
    write_quality_annotations,
)

HEIGHT = 72
WIDTH = 96
START_NS = 10_000_000_000

FixtureName = Literal[
    "static_noisy_confidence",
    "jitter_1px",
    "jitter_2px",
    "translation_rotation",
    "fast_motion_occlusion",
    "fine_semitransparent_edges",
    "opaque_accessories",
    "underopaque_core",
    "holes_and_halos",
    "dynamic_backdrop_constant_alpha",
    "cadence_15fps",
    "cadence_30fps",
    "cadence_60fps",
    "cadence_irregular",
    "repeats_and_sequence_gaps",
    "frozen_motion",
]

FIXTURE_INVENTORY: tuple[str, ...] = (
    "static_noisy_confidence",
    "jitter_1px",
    "jitter_2px",
    "translation_rotation",
    "fast_motion_occlusion",
    "fine_semitransparent_edges",
    "opaque_accessories",
    "underopaque_core",
    "holes_and_halos",
    "dynamic_backdrop_constant_alpha",
    "cadence_15fps",
    "cadence_30fps",
    "cadence_60fps",
    "cadence_irregular",
    "repeats_and_sequence_gaps",
    "frozen_motion",
)

PROVISIONAL_GATES: tuple[dict[str, object], ...] = (
    {
        "id": "stationary-contour-p95",
        "metric": (
            "aggregate.segment_kinds.stationary.metrics.contour_displacement_p95_px.p95"
        ),
        "op": "<=",
        "value": 1.5,
        "notes": "Provisional MATTE-0.2 absolute stationary contour gate.",
    },
    {
        "id": "stationary-area-drift-p95",
        "metric": (
            "aggregate.segment_kinds.stationary.metrics."
            "stationary_subject_area_drift.p95"
        ),
        "op": "<=",
        "value": 0.01,
    },
    {
        "id": "fast-motion-trail-p95",
        "metric": (
            "aggregate.segment_kinds.fast_motion.metrics.motion_trail_area_ratio.p95"
        ),
        "op": "<=",
        "value": 0.10,
        "notes": "Absolute fixture guard; release comparison also uses <=1.10x.",
    },
    {
        "id": "previous-contour-dominance",
        "metric": "aggregate.motion.maximum_dominant_previous_contour_intervals",
        "op": "<=",
        "value": 1,
    },
    {
        "id": "opaque-core-p05",
        "metric": "aggregate.metrics.opaque_core_alpha_p05.p05",
        "op": ">=",
        "value": 0.95,
    },
    {
        "id": "opaque-core-below-095",
        "metric": "aggregate.metrics.opaque_core_fraction_below_0_95.p95",
        "op": "<=",
        "value": 0.05,
    },
    {
        "id": "background-alpha-mean",
        "metric": "aggregate.metrics.background_alpha_mean.p95",
        "op": "<=",
        "value": 0.01,
    },
    {
        "id": "ground-truth-alpha-mse",
        "metric": "aggregate.metrics.ground_truth_alpha_mse.p95",
        "op": "<=",
        "value": 0.01,
    },
)

RELATIVE_GATES: tuple[dict[str, object], ...] = (
    {
        "id": "stationary-contour-improvement",
        "metric": (
            "aggregate.segment_kinds.stationary.metrics.contour_displacement_p95_px.p95"
        ),
        "op": "<=",
        "baseline_multiplier": 0.60,
        "notes": "Candidate must reduce stationary p95 by at least 40%.",
    },
    {
        "id": "fast-motion-trail-nonregression",
        "metric": (
            "aggregate.segment_kinds.fast_motion.metrics.motion_trail_area_ratio.p95"
        ),
        "op": "<=",
        "baseline_multiplier": 1.10,
    },
    {
        "id": "dynamic-edge-color-improvement",
        "metric": (
            "aggregate.segments.dynamic_backdrop_constant_alpha.metrics."
            "edge_band_rgb_variation.p95"
        ),
        "op": "<=",
        "baseline_multiplier": 0.70,
        "notes": "Provisional 30% dynamic light-wrap shimmer reduction.",
    },
)


@dataclass(frozen=True)
class GeneratedFixture:
    name: str
    bundle: Path
    annotations: Path


def _base_alpha(*, center_x: int = 48, center_y: int = 37) -> np.ndarray:
    hard = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.ellipse(hard, (center_x, center_y - 14), (10, 13), 0, 0, 360, 255, -1)
    cv2.ellipse(hard, (center_x, center_y + 12), (22, 18), 0, 0, 360, 255, -1)
    cv2.rectangle(
        hard,
        (center_x - 18, center_y + 3),
        (center_x + 18, center_y + 25),
        255,
        -1,
    )
    # Headphones, glasses, and two fine hair/accessory structures.
    cv2.circle(hard, (center_x - 12, center_y - 14), 4, 255, 2)
    cv2.circle(hard, (center_x + 12, center_y - 14), 4, 255, 2)
    cv2.line(
        hard,
        (center_x - 8, center_y - 16),
        (center_x + 8, center_y - 16),
        255,
        1,
    )
    cv2.line(
        hard,
        (center_x - 7, center_y - 28),
        (center_x - 12, center_y - 36),
        255,
        1,
    )
    cv2.line(
        hard,
        (center_x + 7, center_y - 28),
        (center_x + 13, center_y - 35),
        255,
        1,
    )
    return cv2.GaussianBlur(hard.astype(np.float32) / 255.0, (5, 5), 0.9).astype(
        np.float32
    )


def _source(alpha: np.ndarray, sequence: int) -> np.ndarray:
    yy, xx = np.indices((HEIGHT, WIDTH))
    background = np.stack(
        (
            25 + (xx * 3 + sequence) % 40,
            30 + (yy * 2) % 45,
            35 + ((xx + yy) * 2) % 35,
        ),
        axis=2,
    ).astype(np.uint8)
    subject = np.empty_like(background)
    subject[..., 0] = np.where(yy > 40, 32, 175)
    subject[..., 1] = np.where(yy > 40, 45, 142)
    subject[..., 2] = np.where(yy > 40, 190, 118)
    weight = alpha[..., None]
    return np.rint(
        subject.astype(np.float32) * weight
        + background.astype(np.float32) * (1.0 - weight)
    ).astype(np.uint8)


def _backdrop(sequence: int, *, dynamic: bool) -> np.ndarray:
    yy, xx = np.indices((HEIGHT, WIDTH))
    phase = sequence * 37 if dynamic else 0
    return np.stack(
        (
            (45 + xx * 2 + phase) % 256,
            (80 + yy * 3 + phase * 2) % 256,
            (120 + xx + yy + phase * 3) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def _regions(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hard = alpha >= 0.99
    opaque = cv2.erode(
        hard.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    ).astype(bool)
    foreground = alpha > 0.01
    background = cv2.dilate(foreground.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
    return opaque, background


def _shift(alpha: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return cv2.warpAffine(
        alpha,
        np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(np.float32)


def _fixture_arrays(
    name: str,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[tuple[float, float, float, float, float, float]],
    list[int],
    list[int],
    str,
]:
    frame_count = 6
    ground_truth: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    backdrops: list[np.ndarray] = []
    registrations: list[tuple[float, float, float, float, float, float]] = []
    timestamps = [START_NS + index * 33_333_333 for index in range(frame_count)]
    capture_sequences = list(range(100, 100 + frame_count))
    segment_kind = "stationary"
    base = _base_alpha()

    for index in range(frame_count):
        gt = base.copy()
        prediction = gt.copy()
        registration = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        dynamic = False
        if name == "static_noisy_confidence":
            yy, xx = np.indices(gt.shape)
            noise = (((xx * 17 + yy * 11 + index * 13) % 23) - 11) / 400.0
            prediction = np.clip(gt + noise.astype(np.float32), 0.0, 1.0)
        elif name in ("jitter_1px", "jitter_2px"):
            distance = 1 if name == "jitter_1px" else 2
            prediction = _shift(gt, distance if index % 2 else 0, 0)
        elif name == "translation_rotation":
            angle = index * 2.0
            center = (WIDTH / 2.0, HEIGHT / 2.0)
            absolute = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32)
            absolute[:, 2] += (index * 1.5, index * 0.5)
            gt = cv2.warpAffine(base, absolute, (WIDTH, HEIGHT)).astype(np.float32)
            prediction = gt.copy()
            if index:
                previous = cv2.getRotationMatrix2D(center, (index - 1) * 2.0, 1.0)
                previous[:, 2] += ((index - 1) * 1.5, (index - 1) * 0.5)
                previous_3 = np.vstack([previous, [0.0, 0.0, 1.0]])
                absolute_3 = np.vstack([absolute, [0.0, 0.0, 1.0]])
                step = absolute_3 @ np.linalg.inv(previous_3)
                registration = cast(
                    tuple[float, float, float, float, float, float],
                    tuple(float(value) for value in step[:2].reshape(-1)),
                )
            segment_kind = "moving"
        elif name in ("fast_motion_occlusion", "frozen_motion"):
            gt = _shift(base, index * 7, 0)
            if index >= 3:
                gt[:, 65:] = 0.0
            prediction = base.copy() if name == "frozen_motion" else gt.copy()
            registration = (1.0, 0.0, 7.0, 0.0, 1.0, 0.0)
            segment_kind = "fast_motion"
        elif name == "fine_semitransparent_edges":
            prediction = gt.copy()
            prediction[(gt > 0.05) & (gt < 0.95)] *= 0.96
        elif name == "opaque_accessories":
            prediction = gt.copy()
        elif name == "underopaque_core":
            prediction = gt.copy()
            opaque, _background = _regions(gt)
            prediction[opaque] = 0.80
        elif name == "holes_and_halos":
            prediction = gt.copy()
            cv2.circle(prediction, (48, 48), 4, 0.0, -1)
            prediction = np.maximum(prediction, _shift(gt, 8, 0) * 0.25)
        elif name == "dynamic_backdrop_constant_alpha":
            prediction = gt.copy()
            dynamic = True
        elif name.startswith("cadence_"):
            prediction = gt.copy()
        elif name == "repeats_and_sequence_gaps":
            prediction = gt.copy()
            capture_sequences[index] = 100 + index * 2
        else:
            raise ValueError(f"unknown fixture {name}")
        ground_truth.append(np.ascontiguousarray(gt.astype(np.float32)))
        predicted.append(np.ascontiguousarray(prediction.astype(np.float32)))
        backdrops.append(_backdrop(index, dynamic=dynamic))
        registrations.append(registration)

    if name == "cadence_15fps":
        timestamps = [START_NS + index * 66_666_667 for index in range(frame_count)]
    elif name == "cadence_30fps":
        timestamps = [START_NS + index * 33_333_333 for index in range(frame_count)]
    elif name == "cadence_60fps":
        timestamps = [START_NS + index * 16_666_667 for index in range(frame_count)]
    elif name == "cadence_irregular":
        deltas = [0, 17_000_000, 51_000_000, 80_000_000, 145_000_000, 181_000_000]
        timestamps = [START_NS + delta for delta in deltas]
    return (
        ground_truth,
        predicted,
        backdrops,
        registrations,
        timestamps,
        capture_sequences,
        segment_kind,
    )


def generate_fixture(
    root: Path,
    name: str,
    *,
    gates: Sequence[dict[str, object]] = PROVISIONAL_GATES,
    repeat_each_output: int = 0,
    mask_profile: str = "fixture",
    use_ground_truth_matte: bool = False,
) -> GeneratedFixture:
    (
        ground_truth,
        predicted,
        backdrops,
        registrations,
        timestamps,
        capture_sequences,
        segment_kind,
    ) = _fixture_arrays(name)
    if use_ground_truth_matte:
        predicted = [alpha.copy() for alpha in ground_truth]
    bundle_path = root / f"{name}-{mask_profile}-bundle"
    annotation_path = root / f"{name}-{mask_profile}-annotations"
    recorder = MatteDiagnosticRecorder(
        bundle_path,
        duration_s=10.0,
        max_bytes=64 * 1024 * 1024,
    )
    annotations: list[QualityFrameAnnotations] = []
    last_output_timestamp = START_NS
    for sequence, (gt, mask, backdrop, timestamp, capture_sequence) in enumerate(
        zip(
            ground_truth,
            predicted,
            backdrops,
            timestamps,
            capture_sequences,
        )
    ):
        raw = _source(gt, sequence)
        clean_foreground = (
            np.clip(raw.astype(np.int16) + 20, 0, 255).astype(np.uint8)
            if name == "fine_semitransparent_edges"
            else raw.copy()
        )
        rendered = composite(
            raw,
            backdrop,
            mask,
            edge_foreground=clean_foreground,
        )
        evidence = MatteFrameEvidence(
            metadata=MatteCaptureMetadata(
                bundle_sequence=sequence,
                capture_sequence=capture_sequence,
                capture_monotonic_ns=timestamp,
                timestamp_source="capture-completion",
                capture_generation=0,
                geometry_generation=0,
            ),
            raw_frame=raw,
            raw_mask=mask.copy(),
            refined_mask=mask.copy(),
            clean_foreground=clean_foreground,
            backdrop_frame=backdrop,
            base_composite=rendered,
            configured_controls={
                "segmentation": {
                    "backend": mask_profile,
                    "threshold": 0.5,
                    "mask_blur": 0,
                    "mask_shift": 0,
                },
                "output": {"nominal_fps": 30},
            },
            effective_controls={
                "segmentation_backend": mask_profile,
                "segmentation_device": "cpu",
                "rvm_downsample_ratio": 0.4 if "rvm" in mask_profile else None,
                "mask_shift": 0,
                "light_wrap": 0.0,
                "blend_space": "srgb_legacy",
            },
            timings_ms={
                "backend_inference_ms": 8.0 + sequence * 0.1,
                "refinement_ms": 0.3,
                "background_ms": 0.1,
                "composite_ms": 0.2,
                "frame_processing_ms": 9.0 + sequence * 0.1,
                "output_send_ms": 0.05,
            },
            compositor_substages_ms={
                "foreground_selection_ms": 0.04,
                "light_wrap_ms": 0.0,
                "alpha_blend_ms": 0.16,
            },
            resource_samples={
                "allocation_bytes": 1_000_000 + sequence * 1024,
                "memory_bytes": 64_000_000 + sequence * 4096,
            },
            backdrop_identity={
                "provider": "generated",
                "logical_index": sequence,
            },
        )
        assert recorder.submit(evidence, rendered)
        recorder._queue.join()
        send_timestamp = max(timestamp + 1_000_000, last_output_timestamp + 1)
        assert recorder.submit_output_event(
            sent_monotonic_ns=send_timestamp,
            source_bundle_sequence=sequence,
            base_updated=True,
            exact_final_repeat=False,
        )
        last_output_timestamp = send_timestamp
        repeats = repeat_each_output
        if name == "repeats_and_sequence_gaps":
            repeats = 1
        for _repeat in range(repeats):
            last_output_timestamp += 8_000_000
            assert recorder.submit_output_event(
                sent_monotonic_ns=last_output_timestamp,
                source_bundle_sequence=sequence,
                base_updated=False,
                exact_final_repeat=True,
            )
        opaque, background = _regions(gt)
        annotations.append(
            QualityFrameAnnotations(
                segment=name,
                registration_from_previous=registrations[sequence],
                opaque_core=opaque,
                background=background,
                ground_truth_alpha=gt,
                ground_truth_foreground=raw,
            )
        )
    recorder.close()
    write_quality_annotations(
        annotation_path,
        bundle_path,
        annotations,
        segments=(
            {
                "id": name,
                "kind": segment_kind,
                "start_sequence": 0,
                "end_sequence": len(annotations) - 1,
                "fixture": name,
            },
        ),
        provenance={
            "kind": "generated",
            "license": "MIT",
            "generator": "tests/matte_quality_evidence.py",
            "contains_private_footage": False,
        },
        gates=gates,
    )
    return GeneratedFixture(name, bundle_path, annotation_path)


def _baseline_sequence(
    root: Path,
    profile: Literal["reported_mediapipe_cpu_proxy", "current_rvm_path_proxy"],
) -> tuple[dict[str, object], list[GeneratedFixture]]:
    fixture_names = (
        "jitter_2px",
        "fast_motion_occlusion",
        "opaque_accessories",
        "dynamic_backdrop_constant_alpha",
    )
    fixtures: list[GeneratedFixture] = []
    reports: list[dict[str, object]] = []
    for name in fixture_names:
        fixture = generate_fixture(
            root,
            name,
            gates=(),
            mask_profile=profile,
            use_ground_truth_matte=(
                profile == "current_rvm_path_proxy" and name == "jitter_2px"
            ),
        )
        report = evaluate_bundle(
            fixture.bundle,
            annotations_root=fixture.annotations,
            metadata=EvaluationMetadata(
                hardware_label="checked-in CI evidence host; see report",
                backend=profile,
                device="CPU",
                effective_detail=(
                    "reported model_selection=1-style policy"
                    if profile == "reported_mediapipe_cpu_proxy"
                    else "RVM ratio=0.4, raw pha, mask_shift=0"
                ),
                resampling="generated 96x72; OpenCV linear affine",
                configuration_label=profile,
                notes=(
                    "Deterministic policy proxy, not model inference. "
                    "Local model-backed qualification is digest-referenced separately."
                ),
            ),
        )
        fixtures.append(fixture)
        reports.append(report)
    combined = {
        "schema": "custback.matte-quality-baseline-suite",
        "version": 1,
        "profile": profile,
        "same_source_and_timestamp_contract": True,
        "reports": reports,
    }
    return combined, fixtures


def generate_baseline(root: Path) -> dict[str, object]:
    reported, reported_fixtures = _baseline_sequence(
        root,
        "reported_mediapipe_cpu_proxy",
    )
    current, current_fixtures = _baseline_sequence(
        root,
        "current_rvm_path_proxy",
    )
    comparisons: list[dict[str, object]] = []
    for baseline_fixture, candidate_fixture in zip(
        reported_fixtures,
        current_fixtures,
    ):
        baseline_bundle = MatteReplayBundle(baseline_fixture.bundle)
        candidate_bundle = MatteReplayBundle(candidate_fixture.bundle)
        baseline_source_contract = [
            {
                "capture_sequence": frame["capture_sequence"],
                "capture_monotonic_ns": frame["capture_monotonic_ns"],
                "raw_frame_sha256": frame["artifacts"]["raw_frame"]["sha256"],
            }
            for frame in baseline_bundle.frames
        ]
        candidate_source_contract = [
            {
                "capture_sequence": frame["capture_sequence"],
                "capture_monotonic_ns": frame["capture_monotonic_ns"],
                "raw_frame_sha256": frame["artifacts"]["raw_frame"]["sha256"],
            }
            for frame in candidate_bundle.frames
        ]
        if baseline_source_contract != candidate_source_contract:
            raise AssertionError("baseline profiles do not share source/timestamps")
        source_contract_sha256 = hashlib.sha256(
            json.dumps(
                baseline_source_contract,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        baseline_report = evaluate_bundle(
            baseline_fixture.bundle,
            annotations_root=baseline_fixture.annotations,
        )
        candidate_report = evaluate_bundle(
            candidate_fixture.bundle,
            annotations_root=candidate_fixture.annotations,
            baseline=baseline_report,
        )
        comparisons.append(
            {
                "fixture": baseline_fixture.name,
                "source_contract_sha256": source_contract_sha256,
                "baseline_evidence_sha256": baseline_report["determinism"][
                    "evidence_sha256"
                ],
                "candidate_evidence_sha256": candidate_report["determinism"][
                    "evidence_sha256"
                ],
                "relative_gates": evaluate_gates(
                    candidate_report,
                    RELATIVE_GATES,
                    baseline=baseline_report,
                ),
            }
        )
    return {
        "schema": "custback.matte-quality-baseline",
        "version": 1,
        "scope": "deterministic generated policy proxies",
        "reported_style_mediapipe_cpu": reported,
        "current_rvm_path": current,
        "comparisons": comparisons,
        "provisional_gates": [*PROVISIONAL_GATES, *RELATIVE_GATES],
        "limitations": {
            "model_inference_executed": False,
            "private_footage_committed": False,
            "reason": (
                "The repository contains neither a consented qualification clip nor "
                "installed MediaPipe/RVM runtimes and model artifacts. The local "
                "qualification manifest records their digests without committing pixels."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--work-dir")
    args = parser.parse_args(argv)
    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        report = generate_baseline(work_root)
    else:
        with tempfile.TemporaryDirectory(prefix="custback-matte-quality-") as temp:
            report = generate_baseline(Path(temp))
    Path(args.json).write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        current = cast(dict[str, Any], report["current_rvm_path"])
        first_report = cast(list[dict[str, object]], current["reports"])[0]
        Path(args.markdown).write_text(
            report_markdown(first_report),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
