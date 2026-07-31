"""Numeric acceptance tests for the MATTE-0.2 evidence evaluator."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import matte_quality_evidence as evidence

from custback.__main__ import main as custback_main
from custback.matte_quality import (
    MatteQualityAnnotations,
    MatteQualityError,
    evaluate_bundle,
)
from custback.matte_diagnostics import MatteReplayBundle


def _gate(report: dict[str, object], gate_id: str) -> dict[str, object]:
    gates = report["gates"]
    assert isinstance(gates, list)
    return next(gate for gate in gates if gate["id"] == gate_id)


def test_frozen_mask_fails_motion_lag_despite_zero_uncompensated_flicker(tmp_path):
    fixture = evidence.generate_fixture(tmp_path, "frozen_motion")
    report = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    metrics = report["aggregate"]["metrics"]
    assert metrics["refined_alpha_temporal_abs_diff"]["max"] == 0.0
    assert metrics["compensated_alpha_temporal_abs_diff"]["p95"] > 0.1
    assert metrics["motion_trail_area_ratio"]["p95"] > 0.1
    assert _gate(report, "fast-motion-trail-p95")["status"] == "fail"
    assert _gate(report, "previous-contour-dominance")["status"] == "fail"


def test_two_pixel_stationary_jitter_fails_signed_distance_contour_gate(tmp_path):
    fixture = evidence.generate_fixture(tmp_path, "jitter_2px")
    report = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    stationary = report["aggregate"]["segment_kinds"]["stationary"]["metrics"]
    assert stationary["contour_displacement_p95_px"]["p95"] == pytest.approx(2.0)
    assert _gate(report, "stationary-contour-p95")["status"] == "fail"


def test_registration_compensates_known_translation_and_rotation(tmp_path):
    fixture = evidence.generate_fixture(tmp_path, "translation_rotation")
    report = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    metrics = report["aggregate"]["metrics"]
    assert metrics["refined_alpha_temporal_abs_diff"]["p95"] > 0.03
    assert metrics["compensated_alpha_temporal_abs_diff"]["p95"] < 0.003
    assert metrics["contour_displacement_p95_px"]["p95"] < 0.5


def test_dynamic_backdrop_constant_alpha_changes_only_edge_color_family(tmp_path):
    fixture = evidence.generate_fixture(
        tmp_path,
        "dynamic_backdrop_constant_alpha",
    )
    report = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    metrics = report["aggregate"]["metrics"]
    for name in (
        "raw_alpha_temporal_abs_diff",
        "refined_alpha_temporal_abs_diff",
        "compensated_alpha_temporal_abs_diff",
        "contour_displacement_p95_px",
        "stationary_subject_area_drift",
    ):
        assert metrics[name]["max"] == 0.0
    assert metrics["edge_band_rgb_variation"]["p95"] > 0.20


def test_temporally_stable_underopaque_core_fails_spatial_gate_only(tmp_path):
    fixture = evidence.generate_fixture(tmp_path, "underopaque_core")
    report = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    metrics = report["aggregate"]["metrics"]
    assert metrics["contour_displacement_p95_px"]["max"] == 0.0
    assert metrics["refined_alpha_temporal_abs_diff"]["max"] == 0.0
    assert metrics["opaque_core_alpha_p05"]["p05"] == pytest.approx(0.8)
    assert metrics["opaque_backdrop_leakage_coefficient"]["p95"] > 0.19
    assert _gate(report, "stationary-contour-p95")["status"] == "pass"
    assert _gate(report, "opaque-core-p05")["status"] == "fail"


def test_hole_halo_and_fine_edge_fixtures_exercise_spatial_metrics(tmp_path):
    defects = evidence.generate_fixture(tmp_path, "holes_and_halos")
    defect_report = evaluate_bundle(
        defects.bundle,
        annotations_root=defects.annotations,
    )
    defect_metrics = defect_report["aggregate"]["metrics"]
    assert defect_metrics["foreground_hole_components"]["p95"] >= 1
    assert defect_metrics["exterior_halo_area_ratio"]["p95"] > 0
    assert defect_metrics["exterior_halo_width_p95_px"]["p95"] > 0

    fine = evidence.generate_fixture(tmp_path, "fine_semitransparent_edges")
    fine_report = evaluate_bundle(
        fine.bundle,
        annotations_root=fine.annotations,
    )
    fine_metrics = fine_report["aggregate"]["metrics"]
    assert fine_metrics["uncertain_pixel_fraction"]["p95"] > 0.09
    assert fine_metrics["ground_truth_gradient_mae"]["p95"] > 0
    assert fine_metrics["clean_foreground_rgb_error"]["p95"] > 0.07


@pytest.mark.parametrize(
    ("fixture_name", "expected_fps"),
    (
        ("cadence_15fps", 15.0),
        ("cadence_30fps", 30.0),
        ("cadence_60fps", 60.0),
    ),
)
def test_generated_cadence_fixtures_use_recorded_monotonic_time(
    tmp_path,
    fixture_name,
    expected_fps,
):
    fixture = evidence.generate_fixture(tmp_path, fixture_name)
    report = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    assert report["aggregate"]["cadence"]["unique_input_fps"] == pytest.approx(
        expected_fps,
        abs=1e-5,
    )


def test_irregular_cadence_uses_span_not_assumed_nominal_fps(tmp_path):
    fixture = evidence.generate_fixture(tmp_path, "cadence_irregular")
    report = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    cadence = report["aggregate"]["cadence"]
    assert cadence["unique_input_fps"] == pytest.approx(27.62430939)
    assert cadence["unique_input_fps"] != 30.0


def test_output_timeline_reports_repeats_updates_send_fps_and_sequence_gaps(
    tmp_path,
):
    fixture = evidence.generate_fixture(tmp_path, "repeats_and_sequence_gaps")
    first = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    second = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    assert first["aggregate"] == second["aggregate"]
    assert first["per_frame"] == second["per_frame"]
    assert (
        first["determinism"]["evidence_sha256"]
        == second["determinism"]["evidence_sha256"]
    )
    cadence = first["aggregate"]["cadence"]
    assert cadence["unique_input_count"] == 6
    assert cadence["output_send_count"] == 12
    assert cadence["exact_final_output_repeat_count"] == 6
    assert cadence["base_composite_update_count"] == 6
    assert cadence["base_composite_reuse_count"] == 6
    assert cadence["capture_sequence_gap_count"] == 5
    assert cadence["capture_missing_input_count"] == 5
    assert cadence["output_send_fps"] > cadence["unique_input_fps"]
    performance = first["aggregate"]["performance"]
    assert performance["timings_ms"]["backend_inference_ms"]["p95"] == 8.475
    assert performance["runtime_allocation_bytes"]["available"] is True
    assert performance["runtime_memory_bytes"]["p95"] == 64_019_456.0


def test_post_base_extension_cannot_overwrite_base_metric_namespace(tmp_path):
    fixture = evidence.generate_fixture(tmp_path, "cadence_30fps", gates=())
    manifest_path = fixture.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_timeline"]["events"][0]["post_base_final_output_provenance"] = {
        "schema": "custback.matte-post-base-output-provenance",
        "version": 1,
        "stage": "test-reaction",
        "metrics": {
            "unique_input_fps": -999,
            "base_composite_update_count": -999,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o600)

    report = evaluate_bundle(fixture.bundle)
    cadence = report["aggregate"]["cadence"]
    assert cadence["unique_input_fps"] == pytest.approx(30.0, abs=1e-5)
    assert cadence["base_composite_update_count"] == 6
    extension = report["extensions"]["post_base"]["events"][0]["provenance"]
    assert extension["metrics"]["unique_input_fps"] == -999


def test_annotations_are_digest_bound_private_and_path_safe(tmp_path):
    fixture = evidence.generate_fixture(tmp_path, "opaque_accessories")
    assert stat.S_IMODE(fixture.annotations.stat().st_mode) == 0o700
    annotations = MatteQualityAnnotations(
        fixture.annotations,
        MatteReplayBundle(fixture.bundle),
    )
    assert annotations.provenance["contains_private_footage"] is False

    manifest_path = fixture.annotations / "annotations.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frames"][0]["artifacts"]["opaque_core"]["path"] = "../outside.npy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    with pytest.raises(MatteQualityError, match="escapes"):
        evaluate_bundle(
            fixture.bundle,
            annotations_root=fixture.annotations,
        )


def test_cli_writes_owner_only_json_and_markdown_and_returns_gate_status(tmp_path):
    fixture = evidence.generate_fixture(
        tmp_path,
        "dynamic_backdrop_constant_alpha",
    )
    report_path = tmp_path / "quality.json"
    markdown_path = tmp_path / "quality.md"
    assert (
        custback_main(
            [
                "matte-evaluate",
                str(fixture.bundle),
                "--annotations",
                str(fixture.annotations),
                "--json",
                str(report_path),
                "--markdown",
                str(markdown_path),
                "--hardware-label",
                "generated-test-host",
                "--backend",
                "generated",
                "--device",
                "CPU",
                "--effective-detail",
                "96x72",
                "--resampling",
                "OpenCV linear",
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["environment"]["hardware"]["label"] == "generated-test-host"
    assert report["environment"]["backend"] == "generated"
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(markdown_path.stat().st_mode) == 0o600
    assert "# Matte quality report" in markdown_path.read_text(encoding="utf-8")


def test_generated_fixture_inventory_covers_matte_0_2_cases():
    assert set(evidence.FIXTURE_INVENTORY) == {
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
    }


def test_generated_baseline_is_deterministic_and_keeps_relative_gates_explicit(
    tmp_path,
):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = evidence.generate_baseline(first_root)
    second = evidence.generate_baseline(second_root)
    assert first == second
    comparison_rows = cast(list[dict[str, Any]], first["comparisons"])
    comparisons = {comparison["fixture"]: comparison for comparison in comparison_rows}
    assert all(
        len(comparison["source_contract_sha256"]) == 64
        for comparison in comparisons.values()
    )
    jitter_gates = cast(
        list[dict[str, Any]],
        comparisons["jitter_2px"]["relative_gates"],
    )
    assert (
        next(
            gate
            for gate in jitter_gates
            if gate["id"] == "stationary-contour-improvement"
        )["status"]
        == "pass"
    )
    dynamic_gates = cast(
        list[dict[str, Any]],
        comparisons["dynamic_backdrop_constant_alpha"]["relative_gates"],
    )
    assert (
        next(
            gate
            for gate in dynamic_gates
            if gate["id"] == "dynamic-edge-color-improvement"
        )["status"]
        == "fail"
    )


def test_quality_evaluation_rejects_annotation_manifest_from_other_bundle(tmp_path):
    first = evidence.generate_fixture(tmp_path, "cadence_15fps")
    second = evidence.generate_fixture(tmp_path, "cadence_60fps")
    with pytest.raises(MatteQualityError, match="do not match"):
        evaluate_bundle(
            second.bundle,
            annotations_root=first.annotations,
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX chmod semantics")
def test_public_report_contains_no_bundle_or_annotation_paths(tmp_path):
    fixture = evidence.generate_fixture(tmp_path, "opaque_accessories")
    report = evaluate_bundle(
        fixture.bundle,
        annotations_root=fixture.annotations,
    )
    serialized = json.dumps(report)
    assert str(fixture.bundle) not in serialized
    assert str(fixture.annotations) not in serialized
    assert not any("raw_frame" in key for key in report["source"])
