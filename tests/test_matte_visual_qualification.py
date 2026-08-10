"""Fail-closed MATTE-5.2 visual-qualification contract tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Callable

import cv2
import matte_visual_qualification_evidence as evidence
import numpy as np
import pytest

from custback import __main__ as core_main
from custback import matte_visual_qualification as visual_qualification
from custback.matte_quality import MatteQualityError
from custback.matte_visual_qualification import (
    REQUIRED_BOUNDARIES,
    main,
    qualify_plan,
    run_qualification,
)

_LOCAL_CAPTURE_METHODS = {
    "in_memory": "pipeline-memory-tap",
    "highgui_pre_overlay": "highgui-pre-overlay-tap",
    "snapshot_jpeg": "authenticated-http-snapshot",
    "mjpeg_jpeg": "authenticated-mjpeg-part",
    "websocket_jpeg": "authenticated-output-websocket",
    "pyvirtualcam_loopback": "pyvirtualcam-consumer-recording",
    "windows_native_loopback": "windows-native-consumer-recording",
}


@pytest.fixture
def generated(tmp_path: Path) -> evidence.GeneratedQualification:
    return evidence.generate_qualification(tmp_path / "private-evidence")


def _rewrite_plan(
    generated: evidence.GeneratedQualification,
    mutate: Callable[[dict[str, Any]], None],
    *,
    rebind: bool = True,
) -> None:
    plan = evidence.read_json(generated.plan)
    mutate(plan)
    evidence.write_json(generated.plan, plan)
    if rebind:
        evidence.rebind_review(generated)


def _assert_invalid_or_failed(generated: evidence.GeneratedQualification) -> None:
    """Assert a mutation cannot hide inside generated-authority pending status."""

    try:
        report = qualify_plan(generated.plan)
    except MatteQualityError:
        return
    assert report["status"] == "failed"
    cases = report.get("cases", [])
    assert not cases or any(case["outcome"] == "failed" for case in cases)


def _decode(path: Path) -> np.ndarray:
    pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert pixels is not None
    return pixels


def _optional_candidate(algorithm: str) -> dict[str, Any]:
    return {
        "id": "candidate",
        "segmentation": {
            "boundary_stabilization": {
                "mode": "motion_aware" if algorithm == "MATTE-2.1" else "off"
            },
            "edge_refine": algorithm == "MATTE-2.2",
            "spatial_edge_refinement": {
                "mode": (
                    "stable_guided" if algorithm == "MATTE-2.2" else "legacy_watershed"
                )
            },
        },
        "compositing": {
            "light_wrap": 0.25 if algorithm == "MATTE-2.4" else 0.0,
            "light_wrap_stabilization": {
                "mode": "temporal_bounded" if algorithm == "MATTE-2.4" else "off"
            },
        },
    }


def _relative_metric(
    metric_id: str,
    *,
    status: str = "pass",
    material: bool = False,
) -> dict[str, object]:
    return {
        "id": metric_id,
        "nonregression": status,
        "material_improvement": material,
    }


def _optional_case(
    algorithm: str,
    case_id: str,
    *,
    cadences: tuple[str, ...] = (),
    motions: tuple[str, ...] = (),
    canvases: tuple[str, ...] = (),
    backgrounds: tuple[str, ...] = (),
    algorithm_metrics: dict[str, object] | None = None,
    relative_metrics: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "candidate_id": "candidate",
        "coverage": {
            "cadences": list(cadences),
            "motions": list(motions),
            "canvases": list(canvases),
            "backgrounds": list(backgrounds),
        },
        "algorithm_contract": {
            "expected_effective": {
                "matte_policy": {
                    "effective": {
                        "boundary_stabilization_mode": (
                            "motion_aware" if algorithm == "MATTE-2.1" else "off"
                        ),
                        "edge_refine": algorithm == "MATTE-2.2",
                        "edge_refinement_mode": (
                            "stable_guided"
                            if algorithm == "MATTE-2.2"
                            else "legacy_watershed"
                        ),
                        "light_wrap": 0.25 if algorithm == "MATTE-2.4" else 0.0,
                        "light_wrap_stabilization_mode": (
                            "temporal_bounded" if algorithm == "MATTE-2.4" else "off"
                        ),
                    }
                }
            }
        },
        "algorithm_metrics": algorithm_metrics or {},
        "relative_metrics": relative_metrics or [],
    }


def _passing_temporal_metrics() -> dict[str, object]:
    return {
        "stationary_subject_area_drift_p95": 0.005,
        "maximum_dominant_previous_contour_intervals": 1.0,
        "baseline_stationary_contour_displacement_p95_px": 1.0,
        "candidate_stationary_contour_displacement_p95_px": 0.5,
    }


def test_checked_in_local_template_parses_and_covers_720p_temporal_skeleton(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "matte-visual-qualification-local-template.json"
    )
    private_root = tmp_path / "private-template"
    private_root.mkdir(mode=0o700)
    private_plan = private_root / "plan.json"
    shutil.copy2(source, private_plan)
    private_plan.chmod(0o600)

    plan, plan_sha256 = visual_qualification._load_plan(private_plan)

    assert plan["schema"] == visual_qualification.PLAN_SCHEMA
    assert len(plan["candidates"]) == 1
    assert len(plan["cases"]) == 7
    temporal_cadences = {
        cadence
        for case in plan["cases"]
        if "stationary" in case["coverage"]["motions"]
        and "1280x720" in case["coverage"]["canvases"]
        for cadence in case["coverage"]["cadences"]
    }
    assert {"fps_15", "fps_30", "fps_60"} <= temporal_cadences
    assert len(plan_sha256) == 64


def test_generated_full_contract_is_deterministic_private_path_free_and_pending(
    generated: evidence.GeneratedQualification,
    tmp_path: Path,
) -> None:
    first = qualify_plan(generated.plan)
    second = qualify_plan(generated.plan)

    assert first == second
    assert first["status"] == "pending"
    assert all(case["outcome"] == "pending" for case in first["cases"])
    assert first["provenance"]["kind"] == "generated"
    assert first["privacy"] == {
        "report_is_path_free": True,
        "report_contains_pixels": False,
        "network_camera_model_preview_or_sink_opened": False,
        "private_inputs_owner_only": True,
        "local_boundary_method_and_platform_attested": False,
        "physical_capture_origin_cryptographically_proven": False,
    }
    assert first["production"] == {
        "quality_preset_selected": False,
        "default_changed": False,
        "generated_evidence_can_qualify": False,
        "reactions_enabled": False,
    }
    assert (
        first["coverage"]["appearance_and_source_condition_authority"]
        == "generated-proxy"
    )
    assert first["review"]["authority"] == "generated-schema-proxy"
    assert first["review"]["all_cases_passed"] is False
    assert first["review"]["candidate_preferred_at_least_once"] is False
    assert first["review"]["proxy_all_cases_passed"] is True
    assert first["review"]["proxy_candidate_preferred_at_least_once"] is True
    serialized = json.dumps(first, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "private-evidence" not in serialized
    assert first["evidence_sha256"]

    for directory in (
        generated.plan.parent,
        generated.baseline_bundle,
        generated.baseline_annotations,
        generated.candidate_bundle,
        generated.candidate_annotations,
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    private_files = [
        generated.plan,
        generated.boundary,
        generated.review,
        *generated.boundary_artifacts.values(),
    ]
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in private_files)

    output = tmp_path / "qualification-output"
    written = run_qualification(generated.plan, output)
    assert written == first
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "qualification.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "qualification.md").stat().st_mode) == 0o600


def test_generated_provenance_cannot_be_promoted_by_perfect_pixels_or_review(
    generated: evidence.GeneratedQualification,
) -> None:
    report = qualify_plan(generated.plan)

    assert report["status"] == "pending"
    assert report["provenance"]["kind"] == "generated"
    assert report["production"]["generated_evidence_can_qualify"] is False


def test_generated_annotations_cannot_be_relabelled_as_consented_local(
    generated: evidence.GeneratedQualification,
) -> None:
    plan = evidence.read_json(generated.plan)
    plan["provenance"]["kind"] = "consented-local"
    plan["provenance"]["license_or_consent_reference"] = (
        "test consent reference that does not cover generated annotations"
    )
    evidence.write_json(generated.plan, plan)

    boundary = evidence.read_json(generated.boundary)
    boundary["authority"] = "local-observed"
    for boundary_id, entry in boundary["artifacts"].items():
        entry["capture_method"] = _LOCAL_CAPTURE_METHODS[boundary_id]
        entry["platform"] = "windows"
    evidence.write_json(generated.boundary, boundary)

    review = evidence.read_json(generated.review)
    review["provenance"] = {
        "kind": "consented-local",
        "reference": "test-only relabel attempt",
    }
    evidence.write_json(generated.review, review)
    evidence.rebind_review(generated)

    with pytest.raises(MatteQualityError):
        qualify_plan(generated.plan)


@pytest.mark.parametrize(
    "axis",
    ("appearances", "motions", "source_conditions"),
)
def test_omitted_coverage_axis_fails_closed(
    generated: evidence.GeneratedQualification,
    axis: str,
) -> None:
    _rewrite_plan(
        generated,
        lambda plan: plan["cases"][0]["coverage"].__setitem__(axis, []),
    )

    report = qualify_plan(generated.plan)
    assert report["status"] == "pending"
    assert report["coverage"]["proxy_observed"][axis] == []
    assert report["coverage"]["missing"][axis]


def test_missing_coverage_axis_is_invalid(
    generated: evidence.GeneratedQualification,
) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        del plan["cases"][0]["coverage"]["appearances"]

    _rewrite_plan(generated, mutate)
    _assert_invalid_or_failed(generated)


@pytest.mark.parametrize(
    "unsupported_motion",
    ("fast_turn", "hand_or_prop_crossing_face"),
)
def test_motion_claim_requires_matching_annotation_segment(
    generated: evidence.GeneratedQualification,
    unsupported_motion: str,
) -> None:
    _rewrite_plan(
        generated,
        lambda plan: plan["cases"][0]["coverage"].__setitem__(
            "motions",
            ["stationary", unsupported_motion],
        ),
    )

    _assert_invalid_or_failed(generated)


@pytest.mark.parametrize(
    ("axis", "false_claim"),
    (
        ("canvases", ["1280x720"]),
        ("cadences", ["fps_60"]),
        ("backgrounds", ["static_image"]),
    ),
)
def test_false_mechanical_coverage_claim_fails_closed(
    generated: evidence.GeneratedQualification,
    axis: str,
    false_claim: list[str],
) -> None:
    _rewrite_plan(
        generated,
        lambda plan: plan["cases"][0]["coverage"].__setitem__(axis, false_claim),
    )

    _assert_invalid_or_failed(generated)


@pytest.mark.parametrize(
    ("section", "path", "replacement"),
    (
        (
            "segmentation",
            ("boundary_stabilization", "mode"),
            "off",
        ),
        (
            "segmentation",
            ("spatial_edge_refinement", "mode"),
            "legacy_watershed",
        ),
        (
            "compositing",
            ("light_wrap_stabilization", "mode"),
            "off",
        ),
    ),
    ids=("motion-aware", "stable-guided", "temporal-light-wrap"),
)
def test_omitted_selected_optional_algorithm_fails_closed(
    generated: evidence.GeneratedQualification,
    section: str,
    path: tuple[str, str],
    replacement: str,
) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        target = plan["candidates"][0][section]
        target[path[0]][path[1]] = replacement

    _rewrite_plan(generated, mutate)
    _assert_invalid_or_failed(generated)


def test_matte_21_gate_passes_only_with_stationary_720p_at_every_cadence() -> None:
    cases = [
        _optional_case(
            "MATTE-2.1",
            f"temporal-{cadence}",
            cadences=(cadence,),
            motions=("stationary",),
            canvases=("1280x720",),
            algorithm_metrics=_passing_temporal_metrics(),
        )
        for cadence in ("fps_15", "fps_30", "fps_60")
    ]

    result = visual_qualification._optional_algorithm_gates(
        [_optional_candidate("MATTE-2.1")],
        cases,
    )[0]

    gate = result["algorithms"]["MATTE-2.1"]
    assert gate["status"] == "passed"
    assert [cell["cadence"] for cell in gate["cells"]] == [
        "fps_15",
        "fps_30",
        "fps_60",
    ]
    assert all(cell["canvas"] == "1280x720" for cell in gate["cells"])
    assert result["complete"] is True
    assert result["passed"] is True


def test_matte_21_gate_keeps_non_720p_cadence_pending() -> None:
    cases = [
        _optional_case(
            "MATTE-2.1",
            f"temporal-{cadence}",
            cadences=(cadence,),
            motions=("stationary",),
            canvases=(("1920x1080",) if cadence == "fps_60" else ("1280x720",)),
            algorithm_metrics=_passing_temporal_metrics(),
        )
        for cadence in ("fps_15", "fps_30", "fps_60")
    ]

    result = visual_qualification._optional_algorithm_gates(
        [_optional_candidate("MATTE-2.1")],
        cases,
    )[0]

    gate = result["algorithms"]["MATTE-2.1"]
    assert gate["status"] == "pending"
    fps_60 = next(cell for cell in gate["cells"] if cell["cadence"] == "fps_60")
    assert fps_60["case_ids"] == []
    assert fps_60["canvas"] == "1280x720"
    assert result["complete"] is False


def test_matte_21_gate_preserves_failure_before_later_incomplete_case() -> None:
    failed_metrics = {
        **_passing_temporal_metrics(),
        "candidate_stationary_contour_displacement_p95_px": 0.7,
    }
    incomplete_metrics = {
        **_passing_temporal_metrics(),
        "candidate_stationary_contour_displacement_p95_px": None,
    }
    cases = [
        _optional_case(
            "MATTE-2.1",
            f"temporal-{cadence}",
            cadences=(cadence,),
            motions=("stationary",),
            canvases=("1280x720",),
            algorithm_metrics=(
                failed_metrics if cadence == "fps_15" else _passing_temporal_metrics()
            ),
        )
        for cadence in ("fps_15", "fps_30", "fps_60")
    ]
    cases.append(
        _optional_case(
            "MATTE-2.1",
            "temporal-fps-15-incomplete",
            cadences=("fps_15",),
            motions=("stationary",),
            canvases=("1280x720",),
            algorithm_metrics=incomplete_metrics,
        )
    )

    result = visual_qualification._optional_algorithm_gates(
        [_optional_candidate("MATTE-2.1")],
        cases,
    )[0]

    gate = result["algorithms"]["MATTE-2.1"]
    assert gate["status"] == "failed"
    fps_15 = next(cell for cell in gate["cells"] if cell["cadence"] == "fps_15")
    assert fps_15["status"] == "failed"
    assert result["complete"] is True
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("scenario", "expected"),
    (("pass", "passed"), ("pending", "pending"), ("fail", "failed")),
)
def test_matte_22_gate_covers_each_canvas_with_spatial_improvement(
    scenario: str,
    expected: str,
) -> None:
    cases = [
        _optional_case(
            "MATTE-2.2",
            f"spatial-{canvas}",
            canvases=(canvas,),
            relative_metrics=[
                _relative_metric("ground_truth_mse", material=True),
                _relative_metric("ground_truth_gradient"),
            ],
        )
        for canvas in ("640x360", "1280x720", "1920x1080")
    ]
    if scenario == "pending":
        cases.pop()
    elif scenario == "fail":
        cases[-1]["relative_metrics"] = [
            _relative_metric("ground_truth_mse"),
            _relative_metric("ground_truth_gradient"),
        ]

    result = visual_qualification._optional_algorithm_gates(
        [_optional_candidate("MATTE-2.2")],
        cases,
    )[0]

    assert result["algorithms"]["MATTE-2.2"]["status"] == expected


@pytest.mark.parametrize(
    ("scenario", "expected"),
    (("pass", "passed"), ("pending", "pending"), ("fail", "failed")),
)
def test_matte_24_gate_requires_dynamic_and_live_shimmer_improvement(
    scenario: str,
    expected: str,
) -> None:
    alpha_metrics = (
        "opaque_core_deficit",
        "opaque_core_p05",
        "background_alpha",
        "halo_area",
    )
    cases = [
        _optional_case(
            "MATTE-2.4",
            f"wrap-{background}",
            backgrounds=(background,),
            relative_metrics=[
                _relative_metric("edge_shimmer", material=True),
                *(_relative_metric(metric_id) for metric_id in alpha_metrics),
            ],
        )
        for background in ("dynamic_video", "live_camera")
    ]
    if scenario == "pending":
        cases.pop()
    elif scenario == "fail":
        cases[-1]["relative_metrics"][0] = _relative_metric("edge_shimmer")

    result = visual_qualification._optional_algorithm_gates(
        [_optional_candidate("MATTE-2.4")],
        cases,
    )[0]

    assert result["algorithms"]["MATTE-2.4"]["status"] == expected


def test_unreferenced_declared_candidate_is_rejected(
    generated: evidence.GeneratedQualification,
) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        unused = dict(plan["candidates"][0])
        unused["id"] = "unreferenced-candidate"
        plan["candidates"].append(unused)

    _rewrite_plan(generated, mutate)
    _assert_invalid_or_failed(generated)


def test_missing_ground_truth_metric_evidence_stays_not_decidable(
    generated: evidence.GeneratedQualification,
) -> None:
    for annotation_root in (
        generated.baseline_annotations,
        generated.candidate_annotations,
    ):
        manifest_path = annotation_root / "annotations.json"
        annotations = evidence.read_json(manifest_path)
        for frame in annotations["frames"]:
            del frame["artifacts"]["ground_truth_alpha"]
        evidence.write_json(manifest_path, annotations)
    evidence.rebind_review(generated)

    report = qualify_plan(generated.plan)
    gates = {gate["id"]: gate for gate in report["quality"]["absolute_gates"]}
    assert report["status"] == "pending"
    assert report["quality"]["absolute_complete"] is False
    assert gates["ground_truth_mse"]["status"] == "not_evaluated"
    assert gates["ground_truth_gradient"]["status"] == "not_evaluated"


def test_weak_candidate_metric_evidence_is_failed(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path / "weak-candidate-evidence",
        weak_candidate=True,
    )

    report = qualify_plan(generated.plan)
    gates = {gate["id"]: gate for gate in report["quality"]["absolute_gates"]}
    assert report["status"] == "failed"
    assert gates["registered_contour"]["status"] == "fail"
    assert gates["ground_truth_mse"]["status"] == "fail"


def test_transport_policy_cannot_weaken_the_code_owned_gate(
    generated: evidence.GeneratedQualification,
) -> None:
    _rewrite_plan(
        generated,
        lambda plan: plan["policy"].__setitem__("maximum_full_frame_mae", 999.0),
    )

    _assert_invalid_or_failed(generated)


def test_baseline_and_candidate_source_mismatch_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    manifest_path = generated.baseline_bundle / "manifest.json"
    manifest = evidence.read_json(manifest_path)
    manifest["frames"][0]["capture_sequence"] += 1
    evidence.write_json(manifest_path, manifest)
    evidence.rebind_annotations(
        generated.baseline_bundle,
        generated.baseline_annotations,
    )

    _assert_invalid_or_failed(generated)


def test_manifest_digests_cannot_be_reused_across_cases(
    generated: evidence.GeneratedQualification,
) -> None:
    root = generated.plan.parent
    duplicate_paths = {
        "baseline_bundle": root / "duplicate-baseline-bundle",
        "baseline_annotations": root / "duplicate-baseline-annotations",
        "candidate_bundle": root / "duplicate-candidate-bundle",
        "candidate_annotations": root / "duplicate-candidate-annotations",
        "boundary": root / "duplicate-boundaries.json",
        "review": root / "duplicate-review.json",
    }
    shutil.copytree(generated.baseline_bundle, duplicate_paths["baseline_bundle"])
    shutil.copytree(
        generated.baseline_annotations,
        duplicate_paths["baseline_annotations"],
    )
    shutil.copytree(generated.candidate_bundle, duplicate_paths["candidate_bundle"])
    shutil.copytree(
        generated.candidate_annotations,
        duplicate_paths["candidate_annotations"],
    )
    shutil.copy2(generated.boundary, duplicate_paths["boundary"])
    shutil.copy2(generated.review, duplicate_paths["review"])

    plan = evidence.read_json(generated.plan)
    duplicate = json.loads(json.dumps(plan["cases"][0]))
    duplicate["id"] = "duplicate-manifest-case"
    duplicate["baseline"] = {
        "bundle": str(duplicate_paths["baseline_bundle"]),
        "annotations": str(duplicate_paths["baseline_annotations"]),
    }
    duplicate["candidate"] = {
        "bundle": str(duplicate_paths["candidate_bundle"]),
        "annotations": str(duplicate_paths["candidate_annotations"]),
    }
    duplicate["boundary_evidence"] = str(duplicate_paths["boundary"])
    duplicate["review"] = str(duplicate_paths["review"])
    plan["cases"].append(duplicate)
    evidence.write_json(generated.plan, plan)
    evidence.rebind_review(generated)

    with pytest.raises(MatteQualityError, match="content cannot be reused"):
        qualify_plan(generated.plan)


def test_local_cadence_requires_capture_completion_timestamps(
    generated: evidence.GeneratedQualification,
) -> None:
    reference = "test-only local cadence consent reference"
    plan = evidence.read_json(generated.plan)
    plan["provenance"]["kind"] = "consented-local"
    plan["provenance"]["license_or_consent_reference"] = reference
    evidence.write_json(generated.plan, plan)

    for bundle_root, annotation_root in (
        (generated.baseline_bundle, generated.baseline_annotations),
        (generated.candidate_bundle, generated.candidate_annotations),
    ):
        manifest_path = bundle_root / "manifest.json"
        manifest = evidence.read_json(manifest_path)
        for frame in manifest["frames"]:
            frame["timestamp_source"] = "unique-frame-dequeue"
        evidence.write_json(manifest_path, manifest)
        evidence.rebind_annotations(bundle_root, annotation_root)
        annotations_path = annotation_root / "annotations.json"
        annotations = evidence.read_json(annotations_path)
        annotations["provenance"]["kind"] = "consented-local"
        annotations["provenance"]["license"] = reference
        evidence.write_json(annotations_path, annotations)

    boundary = evidence.read_json(generated.boundary)
    boundary["authority"] = "local-observed"
    for boundary_id, entry in boundary["artifacts"].items():
        entry["capture_method"] = _LOCAL_CAPTURE_METHODS[boundary_id]
        entry["platform"] = "windows"
    evidence.write_json(generated.boundary, boundary)
    evidence.rebind_candidate_bundle(generated)

    review = evidence.read_json(generated.review)
    review["provenance"] = {"kind": "consented-local", "reference": reference}
    evidence.write_json(generated.review, review)
    evidence.rebind_review(generated)

    with pytest.raises(MatteQualityError, match="capture-completion timestamps"):
        qualify_plan(generated.plan)


def test_boundary_generation_mismatch_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    boundary = evidence.read_json(generated.boundary)
    boundary["source"]["capture_generation"] += 1
    evidence.write_json(generated.boundary, boundary)
    evidence.rebind_review(generated)

    _assert_invalid_or_failed(generated)


def test_boundary_identity_region_digest_mismatch_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    boundary = evidence.read_json(generated.boundary)
    boundary["source"]["identity_region"]["reference_region_sha256"] = "0" * 64
    evidence.write_json(generated.boundary, boundary)
    evidence.rebind_review(generated)

    _assert_invalid_or_failed(generated)


def test_identity_region_must_be_at_least_eight_pixels(
    generated: evidence.GeneratedQualification,
) -> None:
    boundary = evidence.read_json(generated.boundary)
    identity = boundary["source"]["identity_region"]
    reference = _decode(generated.boundary_artifacts["in_memory"])
    identity["width"] = 1
    identity["height"] = 1
    identity["reference_region_sha256"] = hashlib.sha256(
        np.ascontiguousarray(reference[0:1, 0:1]).tobytes()
    ).hexdigest()
    evidence.write_json(generated.boundary, boundary)
    evidence.rebind_review(generated)

    _assert_invalid_or_failed(generated)


def test_low_information_identity_region_is_rejected(
    generated: evidence.GeneratedQualification,
) -> None:
    reference = _decode(generated.boundary_artifacts["in_memory"])
    reference[0:8, 0:8] = reference[0, 0]
    evidence.replace_candidate_final_composite(generated, reference)

    with pytest.raises(MatteQualityError, match="identity region lacks contrast"):
        qualify_plan(generated.plan)


def test_candidate_configuration_mismatch_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        plan["candidates"][0]["segmentation"]["mask_shift"] = -1

    _rewrite_plan(generated, mutate)
    _assert_invalid_or_failed(generated)


def test_baseline_configuration_sabotage_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    manifest_path = generated.baseline_bundle / "manifest.json"
    manifest = evidence.read_json(manifest_path)
    for frame in manifest["frames"]:
        frame["configured_controls"]["segmentation"]["mask_shift"] = -1
    evidence.write_json(manifest_path, manifest)
    evidence.rebind_annotations(
        generated.baseline_bundle,
        generated.baseline_annotations,
    )

    _assert_invalid_or_failed(generated)


def test_post_base_reaction_provenance_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    manifest_path = generated.candidate_bundle / "manifest.json"
    manifest = evidence.read_json(manifest_path)
    event = manifest["output_timeline"]["events"][0]
    event["post_base_final_output_provenance"] = {
        "schema": "custback.matte-post-base-output-provenance",
        "version": 1,
        "stage": "test-reaction",
        "metrics": {"active": True},
    }
    evidence.write_json(manifest_path, manifest)
    evidence.rebind_candidate_bundle(generated)

    _assert_invalid_or_failed(generated)


def test_enabled_reaction_control_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    manifest_path = generated.candidate_bundle / "manifest.json"
    manifest = evidence.read_json(manifest_path)
    manifest["frames"][0]["configured_controls"]["reactions"] = {"enabled": True}
    evidence.write_json(manifest_path, manifest)
    evidence.rebind_candidate_bundle(generated)

    _assert_invalid_or_failed(generated)


def test_missing_boundary_artifact_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    generated.boundary_artifacts["pyvirtualcam_loopback"].unlink()

    _assert_invalid_or_failed(generated)


def test_resized_boundary_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    original = _decode(generated.boundary_artifacts["windows_native_loopback"])
    resized = cv2.resize(
        original,
        (original.shape[1] - 4, original.shape[0] - 2),
        interpolation=cv2.INTER_AREA,
    )
    evidence.replace_boundary_pixels(
        generated,
        "windows_native_loopback",
        resized,
    )

    _assert_invalid_or_failed(generated)


def test_shifted_subject_edge_fails_even_when_artifact_digest_is_valid(
    generated: evidence.GeneratedQualification,
) -> None:
    original = _decode(generated.boundary_artifacts["pyvirtualcam_loopback"])
    shifted = cv2.warpAffine(
        original,
        np.asarray([[1.0, 0.0, 4.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (original.shape[1], original.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    evidence.replace_boundary_pixels(
        generated,
        "pyvirtualcam_loopback",
        shifted,
    )

    _assert_invalid_or_failed(generated)


def test_stale_identity_region_fails_below_global_transport_threshold(
    generated: evidence.GeneratedQualification,
) -> None:
    boundary = evidence.read_json(generated.boundary)
    identity = boundary["source"]["identity_region"]
    transported = _decode(generated.boundary_artifacts["in_memory"])
    x = identity["x"]
    y = identity["y"]
    width = identity["width"]
    height = identity["height"]
    region = transported[y : y + height, x : x + width].astype(np.int16)
    transported[y : y + height, x : x + width] = np.clip(
        region + 32,
        0,
        255,
    ).astype(np.uint8)
    evidence.replace_boundary_pixels(
        generated,
        "snapshot_jpeg",
        transported,
    )

    report = qualify_plan(generated.plan)
    comparison = report["cases"][0]["boundaries"]["boundaries"]["snapshot_jpeg"][
        "comparison"
    ]
    assert comparison["full_frame_mae"] <= report["policy"]["maximum_full_frame_mae"]
    assert comparison["edge_band_mae"] <= report["policy"]["maximum_edge_band_mae"]
    assert (
        comparison["identity_region_mae"] > report["policy"]["maximum_full_frame_mae"]
    )
    assert report["status"] == "failed"


def test_boundary_digest_mismatch_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    boundary = evidence.read_json(generated.boundary)
    boundary["artifacts"]["highgui_pre_overlay"]["artifact"]["sha256"] = "0" * 64
    evidence.write_json(generated.boundary, boundary)
    evidence.rebind_review(generated)

    _assert_invalid_or_failed(generated)


def test_boundary_path_reuse_cannot_substitute_for_missing_capture(
    generated: evidence.GeneratedQualification,
) -> None:
    boundary = evidence.read_json(generated.boundary)
    boundary["artifacts"]["windows_native_loopback"]["artifact"] = dict(
        boundary["artifacts"]["pyvirtualcam_loopback"]["artifact"]
    )
    evidence.write_json(generated.boundary, boundary)
    evidence.rebind_review(generated)

    _assert_invalid_or_failed(generated)


@pytest.mark.parametrize(
    "boundary_id",
    ("pyvirtualcam_loopback", "windows_native_loopback"),
)
def test_absent_physical_loopback_descriptor_fails_closed(
    generated: evidence.GeneratedQualification,
    boundary_id: str,
) -> None:
    boundary = evidence.read_json(generated.boundary)
    del boundary["artifacts"][boundary_id]
    evidence.write_json(generated.boundary, boundary)
    evidence.rebind_review(generated)

    _assert_invalid_or_failed(generated)


def test_review_concern_failure_rejects_candidate(
    generated: evidence.GeneratedQualification,
) -> None:
    review = evidence.read_json(generated.review)
    review["concerns"]["hair_retention"] = "fail"
    review["overall"] = "candidate_worse"
    evidence.write_json(generated.review, review)

    _assert_invalid_or_failed(generated)


def test_review_binding_digest_mismatch_fails_closed(
    generated: evidence.GeneratedQualification,
) -> None:
    review = evidence.read_json(generated.review)
    review["bindings"]["boundary_evidence_sha256"] = "0" * 64
    evidence.write_json(generated.review, review)

    _assert_invalid_or_failed(generated)


@pytest.mark.parametrize(
    "binding",
    (
        "baseline_bundle_manifest_sha256",
        "baseline_annotation_manifest_sha256",
        "candidate_annotation_manifest_sha256",
        "baseline_quality_evidence_sha256",
        "candidate_quality_evidence_sha256",
    ),
)
def test_expanded_review_binding_substitution_is_rejected(
    generated: evidence.GeneratedQualification,
    binding: str,
) -> None:
    review = evidence.read_json(generated.review)
    review["bindings"][binding] = "0" * 64
    evidence.write_json(generated.review, review)

    with pytest.raises(MatteQualityError, match=rf"review {binding} differs"):
        qualify_plan(generated.plan)


def test_private_footage_repository_claim_is_rejected(
    generated: evidence.GeneratedQualification,
) -> None:
    _rewrite_plan(
        generated,
        lambda plan: plan["provenance"].__setitem__(
            "contains_private_footage_in_repository",
            True,
        ),
    )

    with pytest.raises(MatteQualityError):
        qualify_plan(generated.plan)


def test_symlinked_plan_or_evidence_is_rejected(
    generated: evidence.GeneratedQualification,
    tmp_path: Path,
) -> None:
    plan_link = tmp_path / "plan-link.json"
    plan_link.symlink_to(generated.plan)
    with pytest.raises(MatteQualityError):
        qualify_plan(plan_link)

    artifact = generated.boundary_artifacts["in_memory"]
    replacement = artifact.with_name("replacement-in-memory.png")
    artifact.rename(replacement)
    artifact.symlink_to(replacement.name)
    with pytest.raises(MatteQualityError):
        qualify_plan(generated.plan)


def test_existing_output_directory_is_never_overwritten(
    generated: evidence.GeneratedQualification,
    tmp_path: Path,
) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="ascii")

    with pytest.raises((FileExistsError, MatteQualityError)):
        run_qualification(generated.plan, output)
    assert marker.read_text(encoding="ascii") == "keep"


def test_cli_returns_one_for_valid_generated_pending_evidence(
    generated: evidence.GeneratedQualification,
    tmp_path: Path,
) -> None:
    output = tmp_path / "cli-output"

    assert main([str(generated.plan), "--output", str(output)]) == 1
    assert (output / "qualification.json").is_file()


def test_core_cli_dispatches_matte_visual_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(
        visual_qualification,
        "main",
        lambda argv, *, prog: calls.append((argv, prog)) or 7,
    )

    result = core_main.main(
        ["matte-visual-qualify", "private-plan.json", "--output", "new-report"]
    )

    assert result == 7
    assert calls == [
        (
            ["private-plan.json", "--output", "new-report"],
            "custback matte-visual-qualify",
        )
    ]


def test_cli_returns_two_for_invalid_evidence(
    generated: evidence.GeneratedQualification,
    tmp_path: Path,
) -> None:
    generated.boundary_artifacts["in_memory"].unlink()

    assert (
        main(
            [
                str(generated.plan),
                "--output",
                str(tmp_path / "invalid-output"),
            ]
        )
        == 2
    )


def test_boundary_inventory_is_exact_and_all_artifacts_are_distinct_files(
    generated: evidence.GeneratedQualification,
) -> None:
    boundary = evidence.read_json(generated.boundary)

    assert set(boundary["artifacts"]) == set(REQUIRED_BOUNDARIES)
    filenames = [
        item["artifact"]["filename"]
        for item in boundary["artifacts"].values()
        if item["status"] == "captured"
    ]
    assert len(filenames) == len(set(filenames)) == 7
    assert all((generated.boundary.parent / name).is_file() for name in filenames)
    assert os.path.commonpath(
        [str(path.resolve()) for path in generated.boundary_artifacts.values()]
    ) == str(generated.boundary.parent.resolve())
