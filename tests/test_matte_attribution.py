"""Generated stage-localization evidence for MATTE-0.5."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pytest

from custback.__main__ import main as custback_main
from custback.compositor import composite
from custback.matte_attribution import diagnose_bundle
from custback.matte_diagnostics import (
    MIN_BUNDLE_BYTES,
    MatteCaptureMetadata,
    MatteDiagnosticRecorder,
    MatteFrameEvidence,
)
from custback.matte_quality import (
    MatteQualityError,
    QualityFrameAnnotations,
    QualityNamedRegion,
    write_quality_annotations,
)

Defect = Literal["raw", "refiner", "foreground", "wrap", "final", "mixed"]
HEIGHT = 48
WIDTH = 64


def _scene() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, QualityNamedRegion],
]:
    hard = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(hard, (17, 24), (47, 43), 1, -1)
    cv2.ellipse(hard, (32, 17), (9, 11), 0, 0, 360, 1, -1)
    cv2.rectangle(hard, (20, 13), (23, 23), 1, -1)
    cv2.rectangle(hard, (41, 13), (44, 23), 1, -1)
    alpha = cv2.GaussianBlur(
        hard.astype(np.float32),
        (5, 5),
        0,
    ).astype(np.float32)

    original_background = np.full((HEIGHT, WIDTH, 3), (35, 55, 75), np.uint8)
    foreground = np.full_like(original_background, (75, 165, 225))
    source = np.rint(
        foreground.astype(np.float32) * alpha[..., None]
        + original_background.astype(np.float32) * (1.0 - alpha[..., None])
    ).astype(np.uint8)
    backdrop = np.full_like(source, (225, 45, 15))

    torso = np.zeros_like(hard)
    torso[30:40, 24:41] = 1
    shoulders = np.zeros_like(hard)
    shoulders[26:30, 20:45] = 1
    head = np.zeros_like(hard)
    head[12:20, 28:37] = 1
    headphones = np.zeros_like(hard)
    headphones[15:21, 22:24] = 1
    headphones[15:21, 40:42] = 1
    hair = ((alpha > 0.05) & (alpha < 0.95) & (np.indices(alpha.shape)[0] < 27)).astype(
        np.uint8
    )
    background = (alpha <= 0.01).astype(np.uint8)
    regions = {
        "torso": QualityNamedRegion("opaque_core", torso),
        "shoulders": QualityNamedRegion("opaque_core", shoulders),
        "head": QualityNamedRegion("opaque_core", head),
        "headphones": QualityNamedRegion("opaque_core", headphones),
        "hair": QualityNamedRegion("soft_boundary", hair),
        "background": QualityNamedRegion("background", background),
    }
    return source, alpha, foreground, backdrop, regions


def _controls(*, light_wrap: float, mask_shift: int) -> dict[str, object]:
    return {
        "segmentation": {
            "backend": "rvm",
            "model_path": "",
            "delegate": "cpu",
            "rvm_downsample": 0.4,
            "threshold": 0.5,
            "mask_blur": 0,
            "edge_refine": False,
            "mask_shift": mask_shift,
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
            "mode": "color",
            "fit_mode": "cover",
            "anchor_x": 0.5,
            "anchor_y": 0.5,
        },
    }


def _fixture(tmp_path: Path, defect: Defect) -> tuple[Path, Path]:
    source, truth, truth_foreground, backdrop, regions = _scene()
    bundle = tmp_path / f"{defect}-bundle"
    annotations = tmp_path / f"{defect}-annotations"
    light_wrap = 1.0 if defect in ("wrap", "mixed") else 0.0
    mask_shift = 2 if defect == "refiner" else 0
    controls = _controls(light_wrap=light_wrap, mask_shift=mask_shift)
    annotation_frames: list[QualityFrameAnnotations] = []

    with MatteDiagnosticRecorder(bundle, max_bytes=8_000_000) as recorder:
        for sequence in range(2):
            raw = truth.copy()
            refined = truth.copy()
            clean_foreground = truth_foreground.copy()
            if defect in ("raw", "mixed"):
                raw[30:40, 24:41] = 0.80
                refined[30:40, 24:41] = 0.80
            if defect == "refiner":
                refined[30:40, 24:41] = 0.75
            if defect == "foreground":
                hair = regions["hair"].mask.astype(bool)
                clean_foreground[hair] = backdrop[hair]
            base = composite(
                source,
                backdrop,
                np.ascontiguousarray(refined),
                light_wrap=light_wrap,
                edge_foreground=clean_foreground,
            )
            final = base.copy()
            if defect == "final":
                torso = regions["torso"].mask.astype(bool)
                final[torso] = backdrop[torso]
            evidence = MatteFrameEvidence(
                metadata=MatteCaptureMetadata(
                    bundle_sequence=sequence,
                    capture_sequence=sequence + 100,
                    capture_monotonic_ns=1_000_000_000 + sequence * 33_333_333,
                    timestamp_source="capture-completion",
                    capture_generation=1,
                    geometry_generation=1,
                ),
                raw_frame=source,
                raw_mask=np.ascontiguousarray(raw),
                refined_mask=np.ascontiguousarray(refined),
                clean_foreground=clean_foreground,
                backdrop_frame=backdrop,
                base_composite=base,
                configured_controls=controls,
                effective_controls={
                    "segmentation_backend": "RVMSegmenter",
                    "segmentation_device": "cpu",
                    "produces_matte": True,
                    "rvm_downsample_ratio": 0.4,
                    "mask_shift": mask_shift,
                    "use_model_foreground": True,
                    "light_wrap": light_wrap,
                    "blend_space": "srgb_legacy",
                },
            )
            assert recorder.submit(evidence, final)
            recorder._queue.join()
            annotation_frames.append(
                QualityFrameAnnotations(
                    segment="stationary",
                    opaque_core=np.maximum.reduce(
                        [
                            regions["torso"].mask,
                            regions["shoulders"].mask,
                            regions["head"].mask,
                            regions["headphones"].mask,
                        ]
                    ),
                    background=regions["background"].mask,
                    ground_truth_alpha=truth,
                    ground_truth_foreground=truth_foreground,
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
                "end_sequence": 1,
            }
        ],
        provenance={
            "kind": "generated",
            "license": "MIT",
            "consent_reference": "not-applicable-generated",
        },
    )
    return bundle, annotations


@pytest.mark.parametrize(
    ("defect", "region", "expected"),
    (
        ("raw", "torso", "raw_alpha"),
        ("refiner", "torso", "post_refiner_alpha"),
        ("foreground", "hair", "clean_foreground"),
        ("wrap", "hair", "light_wrap"),
        ("final", "torso", "final_blend"),
        ("mixed", "torso", "mixed"),
    ),
)
def test_generated_defect_is_located_at_its_producing_stage(
    tmp_path: Path,
    defect: Defect,
    region: str,
    expected: str,
):
    bundle, annotations = _fixture(tmp_path, defect)
    output = tmp_path / f"{defect}-attribution"
    report = diagnose_bundle(
        bundle,
        output,
        annotations_root=annotations,
    )

    assert report["per_frame"][0]["regions"][region]["primary_stage"] == expected
    assert report["conclusion"]["primary_stage"] == expected
    assert report["coverage"]["missing_required_anatomy"] == []
    assert report["per_frame"][0]["effective_controls"]["rvm_downsample_ratio"] == 0.4
    assert report["per_frame"][0]["effective_controls"]["mask_shift"] == (
        2 if defect == "refiner" else 0
    )
    counterfactual = report["per_frame"][0]["counterfactual_controls"]
    assert counterfactual["configured_rvm_downsample"] == 0.4
    assert counterfactual["configured_mask_shift"] == (2 if defect == "refiner" else 0)


def test_stable_opaque_core_failure_cannot_pass_and_hard_threshold_is_rejected(
    tmp_path: Path,
):
    bundle, annotations = _fixture(tmp_path, "raw")
    report = diagnose_bundle(
        bundle,
        tmp_path / "attribution",
        annotations_root=annotations,
    )

    assert report["qualification"] == {
        "status": "fail",
        "stable_opaque_core_failure_cannot_pass": True,
        "failed_for_stable_opaque_core": True,
        "failed_for_incomplete_coverage": False,
    }
    assert report["aggregate"]["stable_opaque_core_failure_regions"] == ["torso"]
    candidate = report["candidate_corrections"]["global_hard_threshold"]
    assert candidate["status"] == "rejected"
    assert candidate["production_default_changed"] is False
    assert candidate["checks"]["hair_and_soft_boundary"]["status"] == "fail"
    assert (
        "opaque-core-restricted calibration" in report["conclusion"]["next_experiment"]
    )
    assert "not inferred" in report["conclusion"]["temporal_conclusion"]


def test_four_boundaries_and_per_region_heatmaps_are_private_and_digest_bound(
    tmp_path: Path,
):
    bundle, annotations = _fixture(tmp_path, "foreground")
    output = tmp_path / "attribution"
    report = diagnose_bundle(bundle, output, annotations_root=annotations)
    frame = report["per_frame"][0]

    assert set(report["method"]["four_required_boundaries"]) == {
        "raw_alpha",
        "post_refiner_alpha",
        "direct_full_frame_foreground",
        "current_compositor",
    }
    for descriptor in frame["boundary_artifacts"].values():
        path = output / descriptor["path"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]
    for descriptor in frame["regions"]["hair"]["heatmaps"].values():
        path = output / descriptor["path"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    canonical = json.loads((output / "attribution.json").read_text("utf-8"))
    assert canonical["evidence_sha256"] == report["evidence_sha256"]
    repeated = diagnose_bundle(
        bundle,
        tmp_path / "attribution-repeat",
        annotations_root=annotations,
    )
    assert repeated["evidence_sha256"] == report["evidence_sha256"]


def test_matte_diagnose_cli_stays_offline_and_reports_review_success(tmp_path: Path):
    bundle, annotations = _fixture(tmp_path, "wrap")
    output = tmp_path / "attribution"
    assert (
        custback_main(
            [
                "matte-diagnose",
                str(bundle),
                "--annotations",
                str(annotations),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "attribution.md").exists()


def test_attribution_rejects_unboundedly_small_output_before_creation(
    tmp_path: Path,
):
    bundle, annotations = _fixture(tmp_path, "wrap")
    output = tmp_path / "attribution"
    with pytest.raises(MatteQualityError, match="at least"):
        diagnose_bundle(
            bundle,
            output,
            annotations_root=annotations,
            max_output_bytes=MIN_BUNDLE_BYTES - 1,
        )
    assert not output.exists()
