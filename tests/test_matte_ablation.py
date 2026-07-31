"""Acceptance tests for the bounded MATTE-0.3 ablation screen."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import matte_ablation_evidence as evidence

from custback.__main__ import main as custback_main
from custback.matte_ablation import (
    PLAN_SCHEMA,
    PLAN_VERSION,
    MatteQualityError,
    load_plan,
    run_ablation,
)


def _write_plan(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", "utf-8")
    os.chmod(path, 0o600)
    return path


def _base_plan(variants: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": PLAN_SCHEMA,
        "version": PLAN_VERSION,
        "scope": {
            "host_label": "generated-test",
            "baseline_lane": "rvm",
            "one_host_screening": True,
            "model_inference_executed": False,
            "reported_screenshots_used_as_ab": False,
        },
        "required_axes": [
            axis
            for variant in variants
            for axis in cast(list[str], variant.get("covers", []))
        ],
        "variants": variants,
    }


@pytest.fixture(scope="module")
def generated_screen(tmp_path_factory):
    root = tmp_path_factory.mktemp("matte-ablation")
    output = root / "output"
    report = evidence.generate_screen(root, output)
    return root, output, report


def _row(report: dict[str, Any], row_id: str) -> dict[str, Any]:
    return next(row for row in report["rows"] if row["id"] == row_id)


def test_matrix_preserves_source_identity_and_records_unavailable_without_fallback(
    generated_screen,
):
    _root, _output, report = generated_screen
    assert len(report["rows"]) == 21
    for row in report["rows"]:
        if row["status"] == "completed" and row["kind"] not in ("cadence_projection",):
            assert row["same_source"]["pixels_and_order"] is True
            assert row["same_source"]["timestamps"] is True
    cuda = _row(report, "rvm_cuda_unavailable")
    assert cuda["status"] == "unavailable"
    assert cuda["fallback_substituted"] is False
    assert report["coverage"]["uncompleted_required_axes"] == [
        "backend.rvm_cpu",
        "backend.rvm_cuda",
    ]
    assert report["scope"]["reported_screenshots_used_as_ab"] is False
    assert report["scope"]["production_preset_selected"] is False
    model_backed = report["coverage"]["model_backed"]
    assert model_backed["completed_rows"] == []
    assert model_backed["recurrent_cadence_complete"] is False
    assert model_backed["recurrent_cadence_uncompleted_axes"] == [
        "cadence.15",
        "cadence.30",
        "cadence.60",
        "cadence.decimate_30_to_15",
        "cadence.irregular",
    ]


def test_factorial_separates_alpha_edge_color_and_compositor_cost(generated_screen):
    _root, _output, report = generated_screen
    coverage = report["coverage"]["compositor_factorial"]
    decision = report["required_decisions"]["compositor_factorial"]
    assert coverage["complete"] is True
    assert decision["complete"] is True
    assert len(decision["combinations"]) == 4
    assert decision["alpha_motion_held_by_recorded_mask"] is True
    assert decision["quality_and_runtime_reported_separately"] is True
    for row_id in (
        "compositor_plain",
        "compositor_foreground_only",
        "compositor_wrap_only",
        "compositor_both",
    ):
        row = _row(report, row_id)
        assert row["track_integrity"]["raw_pha_retained"] is True
        assert row["track_integrity"]["clean_foreground_retained"] is True
        assert row["performance"]["warm_up_first_frame_ms"]["composite_ms"] >= 0
        assert (
            row["performance"]["steady_state_ms"]["composite_ms"]["count"]
            == evidence.FRAME_COUNT - 1
        )
    separate = report["required_decisions"]["alpha_motion_vs_edge_color_motion"]
    assert separate["units_are_not_combined_into_one_score"] is True
    assert len(separate["per_variant_values"]) >= 10


def test_generated_watershed_direction_and_shortlists_remain_proxy_only(
    generated_screen,
):
    _root, _output, report = generated_screen
    watershed = report["required_decisions"]["watershed_mediapipe"]
    assert watershed["status"] == "proxy_only_real_decision_pending"
    assert watershed["outcome"] == "improves"
    assert watershed["on_contour_p95_px"] < watershed["off_contour_p95_px"]

    decisions = report["decisions"]
    assert decisions["matte_2_5_rvm_candidates"] == ["rvm_ratio_067_proxy"]
    assert decisions["first_fix_by_lane"] == {
        "rvm_active_lane": "rvm_ratio_067_proxy",
        "mediapipe_fallback_lane": "mediapipe_watershed_on_proxy",
    }
    assert decisions["one_host_screening_is_not_a_production_preset"] is True
    assert all(
        len(rows) <= decisions["policy"]["shortlist_limit_per_lane"]
        for rows in decisions["shortlists"].values()
    )
    assert _row(report, "rvm_ratio_067_proxy")["evidence_kind"] == "generated-proxy"
    rvm_configuration = _row(report, "rvm_ratio_067_proxy")["configuration"]
    assert rvm_configuration["configured_rvm_downsample"] == 0.67
    assert rvm_configuration["effective_rvm_downsample_ratio"] == 0.67
    categories = {
        reason["category"]
        for rejected in decisions["rejected_candidates"]
        for reason in rejected["reasons"]
    }
    assert {"detail loss", "platform availability", "complexity"} <= categories


def test_repeat_cadence_increases_sends_without_new_alpha(generated_screen):
    _root, _output, report = generated_screen
    projection = _row(report, "output_repeat_15_to_30")["cadence_projection"]
    assert projection["unique_input_fps"] == pytest.approx(15.0)
    assert projection["output_send_fps_projection"] == pytest.approx(30.0)
    assert projection["model_invocation_count"] == evidence.FRAME_COUNT
    assert projection["output_send_count"] == evidence.FRAME_COUNT * 2
    assert projection["exact_repeat_count"] == evidence.FRAME_COUNT
    assert projection["introduces_new_alpha_values"] is False
    assert report["cadence_conclusion"]["new_alpha_values_created"] is False


def test_contact_sheets_and_report_are_private_digest_bound_and_path_free(
    generated_screen,
):
    root, output, report = generated_screen
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for row in report["rows"]:
        contact = row.get("contact_sheet")
        if not isinstance(contact, dict) or "path" not in contact:
            continue
        path = output / contact["path"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert hashlib.sha256(path.read_bytes()).hexdigest() == contact["sha256"]
    canonical = json.loads((output / "ablation.json").read_text("utf-8"))
    assert canonical["evidence_sha256"] == report["evidence_sha256"]
    serialized = json.dumps(report, sort_keys=True)
    assert str(root) not in serialized
    assert stat.S_IMODE((output / "ablation.json").stat().st_mode) == 0o600


def test_plan_rejects_screenshot_ab_and_implicit_generic_rvm_policy(tmp_path):
    screenshot = _base_plan(
        [
            {
                "id": "plain",
                "kind": "frozen",
                "lane": "compositor",
                "group": "one_variable",
                "covers": ["compositor.plain"],
                "compositing": {"light_wrap": 0.0},
            }
        ]
    )
    screenshot["scope"]["reported_screenshots_used_as_ab"] = True
    with pytest.raises(MatteQualityError, match="screenshots"):
        load_plan(_write_plan(tmp_path / "screenshot.json", screenshot))

    rvm = _base_plan(
        [
            {
                "id": "generic_ema",
                "kind": "frozen",
                "lane": "rvm",
                "group": "one_variable",
                "covers": ["rvm.generic_ema"],
                "postprocess": {"temporal_smoothing": 0.5},
            }
        ]
    )
    with pytest.raises(MatteQualityError, match="explicit RVM policy"):
        load_plan(_write_plan(tmp_path / "rvm.json", rvm))


def test_recorded_variant_with_different_source_pixels_is_rejected(
    generated_screen,
    tmp_path,
):
    root, _output, _report = generated_screen
    wrong = evidence.generate_bundle(
        tmp_path,
        "wrong-source",
        profile="stable",
        backend="rvm",
        ratio=0.67,
        source_offset=1,
    )
    variants = [
        {
            "id": "wrong_source",
            "kind": "recorded",
            "lane": "rvm",
            "group": "model_profile",
            "covers": ["rvm.wrong_source"],
            "bundle": str(wrong[0]),
            "annotations": str(wrong[1]),
            "timestamp_policy": "exact",
            "evidence_kind": "generated-proxy",
            "expected_backend": "RVMSegmenter",
            "expected_device": "generated-proxy",
            "expected_rvm_downsample_ratio": 0.67,
        }
    ]
    plan = _write_plan(tmp_path / "plan.json", _base_plan(variants))
    with pytest.raises(MatteQualityError, match="source identity"):
        run_ablation(
            root / "source-bundle",
            root / "source-annotations",
            plan,
            tmp_path / "output",
            max_output_bytes=64 * 1024 * 1024,
        )


def test_recorded_variant_cannot_hide_backend_fallback(generated_screen, tmp_path):
    root, _output, _report = generated_screen
    variants = [
        {
            "id": "mislabeled_backend",
            "kind": "recorded",
            "lane": "mediapipe",
            "group": "backend",
            "covers": ["backend.mediapipe_cpu"],
            "bundle": str(root / "rvm-ratio-067-bundle"),
            "annotations": str(root / "rvm-ratio-067-annotations"),
            "timestamp_policy": "exact",
            "evidence_kind": "generated-proxy",
            "expected_backend": "MediaPipeSegmenter",
            "expected_device": "generated-proxy",
        }
    ]
    plan = _write_plan(tmp_path / "plan.json", _base_plan(variants))
    with pytest.raises(MatteQualityError, match="effective backend"):
        run_ablation(
            root / "source-bundle",
            root / "source-annotations",
            plan,
            tmp_path / "output",
            max_output_bytes=64 * 1024 * 1024,
        )


def test_cli_completes_a_bounded_projection_plan(generated_screen, tmp_path):
    root, _output, _report = generated_screen
    variants = [
        {
            "id": "native_projection",
            "kind": "cadence_projection",
            "lane": "cadence",
            "group": "cadence",
            "covers": ["cadence.native"],
            "schedule": {"type": "native"},
            "output_multiplier": 1,
            "evidence_kind": "projection",
        }
    ]
    plan = _write_plan(tmp_path / "plan.json", _base_plan(variants))
    output = tmp_path / "output"
    assert (
        custback_main(
            [
                "matte-ablate",
                str(root / "source-bundle"),
                "--annotations",
                str(root / "source-annotations"),
                "--plan",
                str(plan),
                "--output",
                str(output),
                "--max-output-bytes",
                str(64 * 1024 * 1024),
            ]
        )
        == 0
    )
    assert (output / "ablation.md").exists()


def test_preflight_bound_rejects_matrix_before_output_creation(
    generated_screen,
    tmp_path,
):
    root, _output, _report = generated_screen
    # Reuse the comprehensive owner-only generated plan.
    plan = root / "ablation-plan.json"
    output = tmp_path / "too-small"
    with pytest.raises(MatteQualityError, match="preflight"):
        run_ablation(
            root / "source-bundle",
            root / "source-annotations",
            plan,
            output,
            max_output_bytes=64 * 1024,
        )
    assert not output.exists()
