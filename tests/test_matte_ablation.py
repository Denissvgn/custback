"""Acceptance tests for the bounded MATTE-0.3 ablation screen."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import matte_ablation_evidence as evidence

import custback.matte_ablation as matte_ablation
from custback.__main__ import main as custback_main
from custback.matte_ablation import (
    PLAN_SCHEMA,
    PLAN_VERSION,
    MatteQualityError,
    load_plan,
    run_ablation,
)
from custback.matte_diagnostics import MatteReplayBundle
from custback.matte_quality import MatteQualityAnnotations
from custback.segmentation import SegmentationFrameContext, TemporalResetReason


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
            configured_spatial = row["configuration"][
                "configured_spatial_edge_refinement"
            ]
            effective_spatial = row["configuration"][
                "effective_spatial_edge_refinement"
            ]
            assert set(configured_spatial) == {
                "mode",
                "reference_short_edge_px",
                "radius_at_reference_px",
                "min_radius_px",
                "max_radius_px",
            }
            assert set(effective_spatial) == set(configured_spatial)
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
    for row_id, paired_id in (
        ("compositor_wrap_only", "compositor_plain"),
        ("compositor_both", "compositor_foreground_only"),
    ):
        row = _row(report, row_id)
        metric = row["quality"]["light_wrap_attributable_rgb_variation"]
        pair = row["attribution"]["light_wrap_pair"]
        assert metric is None
        assert pair["status"] == "not_decidable"
        assert pair["paired_no_wrap_id"] == paired_id
        assert pair["summary"]["p95"] is None
        assert pair["segment_summaries"]["stationary"]["count"] == 0
        assert pair["empty_segments"] == ["stationary"]
        assert len(pair["identity_contract_sha256"]) == 64
        assert all(pair["identity_proof"].values())
        assert all(
            {"sequence", "capture_sequence", "segment", "value"} <= set(frame)
            for frame in pair["per_frame"]
        )
    factorial = {
        combination["id"]: combination for combination in decision["combinations"]
    }
    assert factorial["compositor_both"]["light_wrap_attributable_rgb_variation"] is None
    dynamic_wrap = report["required_decisions"]["dynamic_light_wrap"]
    assert dynamic_wrap["status"] == "not_decidable"
    assert dynamic_wrap["computed_pair_count"] == 0
    assert len(dynamic_wrap["paired_rows"]) == 2
    assert dynamic_wrap["production_default_selected"] is False
    separate = report["required_decisions"]["alpha_motion_vs_edge_color_motion"]
    assert separate["units_are_not_combined_into_one_score"] is True
    assert len(separate["per_variant_values"]) >= 10


def test_paired_light_wrap_metric_is_computed_from_fixed_alpha_composites(
    tmp_path,
):
    bundle, annotations = evidence.generate_bundle(
        tmp_path,
        "fixed-alpha-dynamic-wrap",
        profile="stable",
        backend="rvm",
        ratio=0.4,
        dynamic_backdrop=True,
    )
    plan = _base_plan(
        [
            {
                "id": "wrapped",
                "kind": "frozen",
                "lane": "compositor",
                "group": "one_variable",
                "covers": ["compositor.wrap.on"],
                "compositing": {"light_wrap": 0.8},
                "paired_no_wrap_id": "plain",
            },
            {
                "id": "plain",
                "kind": "frozen",
                "lane": "compositor",
                "group": "one_variable",
                "covers": ["compositor.wrap.off"],
                "compositing": {"light_wrap": 0.0},
            },
            {
                "id": "mismatched_plain",
                "kind": "frozen",
                "lane": "compositor",
                "group": "backdrop",
                "covers": ["compositor.wrap.invalid-control"],
                "compositing": {
                    "light_wrap": 0.0,
                    "use_model_foreground": False,
                },
            },
        ]
    )
    plan_path = _write_plan(tmp_path / "fixed-alpha-plan.json", plan)

    report = run_ablation(
        bundle,
        annotations,
        plan_path,
        tmp_path / "fixed-alpha-output",
        max_output_bytes=256 * 1024 * 1024,
    )

    wrapped = _row(report, "wrapped")
    plain = _row(report, "plain")
    pair = wrapped["attribution"]["light_wrap_pair"]
    metric = wrapped["quality"]["light_wrap_attributable_rgb_variation"]
    assert pair["status"] == "computed"
    assert pair["paired_no_wrap_id"] == "plain"
    assert pair["summary"]["count"] == evidence.FRAME_COUNT - 1
    assert pair["segment_summaries"]["stationary"]["count"] == (
        evidence.FRAME_COUNT - 1
    )
    assert metric == pair["summary"]["p95"]
    assert metric > 0.0
    aggregate_subtraction = abs(
        wrapped["quality"]["edge_band_rgb_variation"]
        - plain["quality"]["edge_band_rgb_variation"]
    )
    assert metric != pytest.approx(aggregate_subtraction)
    decision = report["required_decisions"]["dynamic_light_wrap"]
    assert decision["status"] == "not_decidable"
    assert decision["computed_pair_count"] == 1
    assert decision["production_default_selected"] is False

    output = tmp_path / "fixed-alpha-output" / "variants"
    wrapped_bundle = MatteReplayBundle(output / "wrapped" / "bundle")
    wrapped_annotations = MatteQualityAnnotations(
        output / "wrapped" / "annotations",
        wrapped_bundle,
    )
    mismatched_bundle = MatteReplayBundle(output / "mismatched_plain" / "bundle")
    mismatched_annotations = MatteQualityAnnotations(
        output / "mismatched_plain" / "annotations",
        mismatched_bundle,
    )
    mismatch = matte_ablation._light_wrap_pair_result(
        wrapped_bundle,
        wrapped_annotations,
        mismatched_bundle,
        mismatched_annotations,
    )
    assert mismatch["status"] == "not_decidable"
    summary = mismatch["summary"]
    identity_proof = mismatch["identity_proof"]
    assert isinstance(summary, dict)
    assert isinstance(identity_proof, dict)
    assert summary["p95"] is None
    assert identity_proof["model_foreground"] is False


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
    assert len(rvm_configuration["qualification_segmentation_sha256"]) == 64
    categories = {
        reason["category"]
        for rejected in decisions["rejected_candidates"]
        for reason in rejected["reasons"]
    }
    assert {"detail loss", "platform availability", "complexity"} <= categories


def test_shortlist_segmentation_digest_ignores_only_private_model_path():
    first = {
        "backend": "rvm",
        "model_path": "/private/first.onnx",
        "rvm_downsample": 0.5,
        "mask_shift": 0.0,
    }
    second = {**first, "model_path": "/another/private/location.onnx"}
    changed = {**second, "rvm_downsample": 0.75}

    first_digest = matte_ablation._qualification_segmentation_sha256(first)
    assert matte_ablation._qualification_segmentation_sha256(second) == first_digest
    assert matte_ablation._qualification_segmentation_sha256(changed) != first_digest


def test_shortlist_model_contract_requires_path_safe_all_frame_identity():
    telemetry = {
        "applicable": True,
        "model_identity": "rvm_mobilenetv3_fp32.onnx",
        "model_sha256": "a" * 64,
        "model_bytes": 1_234,
    }

    class Bundle:
        frames = [
            {"effective_controls": {"rvm_telemetry": dict(telemetry)}},
            {"effective_controls": {"rvm_telemetry": dict(telemetry)}},
        ]

    assert matte_ablation._qualification_model_contract(cast(Any, Bundle())) == {
        "qualification_model_identity": "rvm_mobilenetv3_fp32.onnx",
        "qualification_model_sha256": "a" * 64,
        "qualification_model_bytes": 1_234,
    }

    Bundle.frames[1]["effective_controls"]["rvm_telemetry"]["model_sha256"] = "b" * 64
    assert matte_ablation._qualification_model_contract(cast(Any, Bundle())) == {
        "qualification_model_identity": None,
        "qualification_model_sha256": None,
        "qualification_model_bytes": None,
    }
    Bundle.frames[0]["effective_controls"]["rvm_telemetry"]["model_identity"] = (
        "/private/rvm.onnx"
    )
    assert matte_ablation._qualification_model_contract(cast(Any, Bundle())) == {
        "qualification_model_identity": None,
        "qualification_model_sha256": None,
        "qualification_model_bytes": None,
    }


def test_temporal_light_wrap_cannot_shortlist_without_computed_pair() -> None:
    baseline: dict[str, object] = {
        "status": "completed",
        "kind": "frozen",
        "quality": {},
        "performance": {},
    }
    candidate: dict[str, object] = {
        "status": "completed",
        "kind": "frozen",
        "configuration": {"light_wrap_stabilization": {"mode": "temporal_bounded"}},
        "quality": {"light_wrap_attributable_rgb_variation": 0.01},
        "performance": {},
    }

    reasons, _improvements = matte_ablation._rejection_reasons(
        candidate,
        baseline,
        matte_ablation.AblationPolicy(),
    )

    assert {
        "category": "evidence completeness",
        "detail": (
            "temporal light-wrap candidate lacks a computed same-frame "
            "no-wrap attribution pair"
        ),
    } in reasons


def test_generated_plan_does_not_fabricate_spatial_refinement_evidence(
    generated_screen,
):
    _root, _output, report = generated_screen
    decision = report["required_decisions"]["spatial_edge_refinement_mediapipe"]

    assert decision == {
        "status": "not_decidable",
        "reason": (
            "completed same-source MediaPipe legacy_watershed and "
            "stable_guided rows are required"
        ),
        "required_axes": [
            "mediapipe.spatial_edge_refinement.legacy_watershed",
            "mediapipe.spatial_edge_refinement.stable_guided",
        ],
    }


def test_spatial_refinement_decision_distinguishes_proxy_and_model_backed():
    def decision_row(
        mode: str,
        contour_p95: float,
        *,
        evidence_kind: str,
    ) -> dict[str, object]:
        return {
            "status": "completed",
            "covers": [f"mediapipe.spatial_edge_refinement.{mode}"],
            "same_source": {
                "pixels_and_order": True,
                "timestamps": True,
            },
            "quality": {
                "compensated_contour_displacement_p95_px": contour_p95,
            },
            "configuration": {
                "edge_refine": True,
                "effective_edge_refinement_mode": mode,
                "effective_edge_refinement_radius_px": 8,
            },
            "evidence_kind": evidence_kind,
        }

    proxy_rows = [
        decision_row(
            "legacy_watershed",
            1.2,
            evidence_kind="generated-proxy",
        ),
        decision_row("stable_guided", 0.8, evidence_kind="generated-proxy"),
    ]
    proxy = matte_ablation._spatial_edge_refinement_decision(proxy_rows)
    assert proxy["status"] == "proxy_only_real_decision_pending"
    assert proxy["outcome"] == "improves"
    assert proxy["model_backed"] is False
    assert proxy["same_source"] is True

    model_rows = [{**row, "evidence_kind": "model-backed"} for row in proxy_rows]
    model_backed = matte_ablation._spatial_edge_refinement_decision(model_rows)
    assert model_backed["status"] == "decided"
    assert model_backed["outcome"] == "improves"
    assert model_backed["model_backed"] is True

    mislabeled = [
        proxy_rows[0],
        {
            **proxy_rows[1],
            "configuration": {
                **cast(dict[str, object], proxy_rows[1]["configuration"]),
                "effective_edge_refinement_mode": "off",
                "effective_edge_refinement_radius_px": 0,
            },
        },
    ]
    assert (
        matte_ablation._spatial_edge_refinement_decision(mislabeled)["status"]
        == "not_decidable"
    )


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


def test_frozen_refiner_replays_recorded_temporal_context(
    generated_screen,
    tmp_path,
    monkeypatch,
):
    root, _output, _report = generated_screen
    source_bundle = MatteReplayBundle(root / "source-bundle")
    source_annotations = MatteQualityAnnotations(
        root / "source-annotations",
        source_bundle,
    )
    events: list[tuple[object, ...]] = []

    class SpyRefiner:
        def __init__(self, _cfg):
            pass

        def reset_temporal_state(self, reason, timestamp_ns):
            events.append(("reset", reason, timestamp_ns))

        def refine(self, mask, _frame, *, context=None):
            events.append(("refine", context))
            return mask

    monkeypatch.setattr(matte_ablation, "MaskRefiner", SpyRefiner)
    row_root = tmp_path / "row"
    row_root.mkdir(mode=0o700)
    matte_ablation._frozen_variant(
        source_bundle,
        source_annotations,
        {
            "lane": "mediapipe",
            "postprocess": {"temporal_smoothing": 0.5},
            "compositing": {},
        },
        row_root,
        max_bytes=64 * 1024 * 1024,
    )

    assert events[0] == (
        "reset",
        TemporalResetReason.INITIAL,
        int(source_bundle.frames[0]["capture_monotonic_ns"]),
    )
    contexts = [event[1] for event in events if event[0] == "refine"]
    assert contexts == [
        SegmentationFrameContext(
            sequence=int(frame["capture_sequence"]),
            timestamp_ns=int(frame["capture_monotonic_ns"]),
            generation=int(frame["capture_generation"]),
            geometry_generation=int(frame["geometry_generation"]),
            shape=source_bundle.load_array(frame, "raw_frame").shape[:2],
        )
        for frame in source_bundle.frames
    ]


def test_frozen_rvm_generic_controls_require_explicit_policy_and_record_evidence(
    generated_screen,
    tmp_path,
):
    root, _output, _report = generated_screen
    source_bundle = MatteReplayBundle(root / "source-bundle")
    source_annotations = MatteQualityAnnotations(
        root / "source-annotations",
        source_bundle,
    )
    postprocess = {
        "mask_blur": 7,
        "edge_refine": True,
        "temporal_smoothing": 0.65,
    }

    def replay(name: str, *, explicit: bool) -> MatteReplayBundle:
        row_root = tmp_path / name
        row_root.mkdir(mode=0o700)
        row: dict[str, object] = {
            "lane": "rvm",
            "postprocess": postprocess,
            "compositing": {},
        }
        if explicit:
            row["explicit_rvm_policy"] = True
        bundle, _annotations = matte_ablation._frozen_variant(
            source_bundle,
            source_annotations,
            row,
            row_root,
            max_bytes=64 * 1024 * 1024,
        )
        return bundle

    production = replay("production-rvm", explicit=False)
    experimental = replay("explicit-rvm", explicit=True)
    production_effective = cast(
        dict[str, Any],
        production.frames[0]["effective_controls"],
    )
    experimental_effective = cast(
        dict[str, Any],
        experimental.frames[0]["effective_controls"],
    )
    production_policy = cast(
        dict[str, Any],
        production_effective["matte_policy"],
    )
    experimental_policy = cast(
        dict[str, Any],
        experimental_effective["matte_policy"],
    )

    assert production_policy["configured"]["mask_blur"] == 7
    assert production_policy["configured"]["edge_refine"] is True
    assert production_policy["configured"]["temporal_smoothing"] == 0.65
    assert production_policy["experimental_rvm_generic"] is False
    assert production_policy["effective"]["mask_blur"] == 0
    assert production_policy["effective"]["edge_refine"] is False
    assert production_policy["effective"]["temporal_smoothing"] == 0.0
    assert {
        production_policy["controls"][name]["state"]
        for name in ("mask_blur", "edge_refine", "temporal_smoothing")
    } == {"bypassed"}
    assert production_effective["refiner"]["mask_blur"] == 0
    assert production_effective["refiner"]["edge_refine"] is False
    assert production_effective["refiner"]["temporal_smoothing"] == 0.0

    assert experimental_policy["configured"] == production_policy["configured"]
    assert experimental_policy["experimental_rvm_generic"] is True
    assert experimental_policy["effective"]["mask_blur"] == 7
    assert experimental_policy["effective"]["edge_refine"] is True
    assert experimental_policy["effective"]["temporal_smoothing"] == 0.65
    assert {
        experimental_policy["controls"][name]["state"]
        for name in ("mask_blur", "edge_refine", "temporal_smoothing")
    } == {"effective"}
    assert experimental_effective["refiner"]["mask_blur"] == 7
    assert experimental_effective["refiner"]["edge_refine"] is True
    assert experimental_effective["refiner"]["temporal_smoothing"] == 0.65

    explicit_changed_alpha = False
    for source_frame, production_frame, experimental_frame in zip(
        source_bundle.frames,
        production.frames,
        experimental.frames,
        strict=True,
    ):
        assert (
            production_frame["effective_controls"]["matte_policy"] == production_policy
        )
        assert (
            experimental_frame["effective_controls"]["matte_policy"]
            == experimental_policy
        )
        raw_alpha = source_bundle.load_array(source_frame, "raw_mask")
        np.testing.assert_array_equal(
            production.load_array(production_frame, "refined_mask"),
            raw_alpha,
        )
        explicit_changed_alpha |= not np.array_equal(
            experimental.load_array(experimental_frame, "refined_mask"),
            raw_alpha,
        )
    assert explicit_changed_alpha is True


def test_explicit_rvm_rows_do_not_activate_unmentioned_or_unreplayed_generics(
    generated_screen,
    tmp_path,
):
    root, _output, _report = generated_screen
    source_bundle = MatteReplayBundle(root / "source-bundle")
    source_annotations = MatteQualityAnnotations(
        root / "source-annotations",
        source_bundle,
    )

    def replay(
        name: str,
        *,
        postprocess: dict[str, object],
        compositing: dict[str, object],
    ) -> dict[str, Any]:
        row_root = tmp_path / name
        row_root.mkdir(mode=0o700)
        bundle, _annotations = matte_ablation._frozen_variant(
            source_bundle,
            source_annotations,
            {
                "lane": "rvm",
                "explicit_rvm_policy": True,
                "postprocess": postprocess,
                "compositing": compositing,
            },
            row_root,
            max_bytes=64 * 1024 * 1024,
        )
        return cast(
            dict[str, Any],
            bundle.frames[0]["effective_controls"]["matte_policy"],
        )

    blur_only = replay(
        "blur-only",
        postprocess={"mask_blur": 3},
        compositing={},
    )
    assert blur_only["experimental_rvm_generic"] is True
    assert blur_only["effective"]["mask_blur"] == 3
    assert blur_only["effective"]["edge_refine"] is False
    assert blur_only["effective"]["temporal_smoothing"] == 0.0
    assert blur_only["controls"]["mask_blur"]["state"] == "effective"
    assert blur_only["controls"]["edge_refine"]["state"] == "bypassed"
    assert blur_only["controls"]["temporal_smoothing"]["state"] == "bypassed"

    compositor_only = replay(
        "compositor-only",
        postprocess={},
        compositing={"light_wrap": 0.0},
    )
    assert compositor_only["experimental_rvm_generic"] is False
    assert compositor_only["effective"]["mask_blur"] == 0
    assert compositor_only["effective"]["edge_refine"] is False
    assert compositor_only["effective"]["temporal_smoothing"] == 0.0


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

    rvm_motion = _base_plan(
        [
            {
                "id": "generic_motion",
                "kind": "frozen",
                "lane": "rvm",
                "group": "one_variable",
                "covers": ["rvm.generic_motion"],
                "postprocess": {"boundary_stabilization": {"mode": "motion_aware"}},
            }
        ]
    )
    with pytest.raises(MatteQualityError, match="explicit RVM policy"):
        load_plan(_write_plan(tmp_path / "rvm-motion.json", rvm_motion))

    rvm_spatial = _base_plan(
        [
            {
                "id": "generic_spatial",
                "kind": "frozen",
                "lane": "rvm",
                "group": "spatial_policy",
                "covers": ["rvm.generic_spatial"],
                "postprocess": {
                    "edge_refine": True,
                    "spatial_edge_refinement": {"mode": "stable_guided"},
                },
            }
        ]
    )
    with pytest.raises(MatteQualityError, match="explicit RVM policy"):
        load_plan(_write_plan(tmp_path / "rvm-spatial.json", rvm_spatial))


@pytest.mark.parametrize(
    "override",
    [
        {"mode": "unstable"},
        {"mode": None},
        {"unknown": 1},
        {"reference_short_edge_px": "720"},
        {"radius_at_reference_px": 33},
        {"min_radius_px": 9, "radius_at_reference_px": 8},
    ],
)
def test_plan_rejects_invalid_spatial_edge_refinement_overrides(
    tmp_path,
    override,
):
    plan = _base_plan(
        [
            {
                "id": "invalid_spatial",
                "kind": "frozen",
                "lane": "mediapipe",
                "group": "one_variable",
                "covers": ["mediapipe.spatial.invalid"],
                "postprocess": {"spatial_edge_refinement": override},
            }
        ]
    )

    with pytest.raises(MatteQualityError, match="frozen variant overrides"):
        load_plan(_write_plan(tmp_path / "invalid-spatial.json", plan))


def test_plan_accepts_partial_light_wrap_stabilization_override(tmp_path):
    plan = _base_plan(
        [
            {
                "id": "stable_wrap",
                "kind": "frozen",
                "lane": "compositor",
                "group": "one_variable",
                "covers": ["compositor.light_wrap_stabilization"],
                "compositing": {
                    "light_wrap_stabilization": {
                        "mode": "temporal_bounded",
                    }
                },
            }
        ]
    )

    loaded, _digest = load_plan(_write_plan(tmp_path / "stable-wrap.json", plan))

    assert loaded["variants"][0]["compositing"] == {
        "light_wrap_stabilization": {
            "mode": "temporal_bounded",
        }
    }


def test_plan_accepts_forward_and_baseline_light_wrap_pair_references(tmp_path):
    plan = _base_plan(
        [
            {
                "id": "wrapped",
                "kind": "frozen",
                "lane": "compositor",
                "group": "one_variable",
                "covers": ["compositor.wrap.paired"],
                "compositing": {"light_wrap": 0.8},
                "paired_no_wrap_id": "plain",
            },
            {
                "id": "plain",
                "kind": "frozen",
                "lane": "compositor",
                "group": "one_variable",
                "covers": ["compositor.wrap.off"],
                "compositing": {"light_wrap": 0.0},
            },
            {
                "id": "recorded_pair",
                "kind": "recorded",
                "lane": "compositor",
                "group": "backdrop",
                "covers": ["compositor.wrap.recorded"],
                "bundle": "/private/bundle",
                "annotations": "/private/annotations",
                "expected_backend": "MediaPipeSegmenter",
                "expected_device": "cpu",
                "evidence_kind": "generated-proxy",
                "paired_no_wrap_id": "baseline",
            },
        ]
    )

    loaded, _digest = load_plan(_write_plan(tmp_path / "pairs.json", plan))

    assert loaded["variants"][0]["paired_no_wrap_id"] == "plain"
    assert loaded["variants"][2]["paired_no_wrap_id"] == "baseline"


@pytest.mark.parametrize(
    ("paired_id", "target_lane"),
    [
        ("wrapped", "compositor"),
        ("missing", "compositor"),
        ("alpha_row", "mediapipe"),
    ],
)
def test_plan_rejects_invalid_light_wrap_pair_targets(
    tmp_path,
    paired_id,
    target_lane,
):
    plan = _base_plan(
        [
            {
                "id": "wrapped",
                "kind": "frozen",
                "lane": "compositor",
                "group": "one_variable",
                "covers": ["compositor.wrap.paired"],
                "compositing": {"light_wrap": 0.8},
                "paired_no_wrap_id": paired_id,
            },
            {
                "id": "alpha_row",
                "kind": "frozen",
                "lane": target_lane,
                "group": "one_variable",
                "covers": ["mediapipe.wrap.invalid"],
                "compositing": {"light_wrap": 0.0},
            },
        ]
    )

    with pytest.raises(MatteQualityError, match="pair"):
        load_plan(_write_plan(tmp_path / f"pair-{paired_id}.json", plan))


@pytest.mark.parametrize(
    "override",
    [
        None,
        [],
        {},
        {"mode": "unstable"},
        {"mode": None},
        {"time_constant_s": True},
        {"time_constant_s": "0.12"},
        {"time_constant_s": 0.009},
        {"time_constant_s": 1.001},
        {"unknown": 1},
    ],
)
def test_plan_rejects_invalid_light_wrap_stabilization_overrides(
    tmp_path,
    override,
):
    plan = _base_plan(
        [
            {
                "id": "invalid_wrap",
                "kind": "frozen",
                "lane": "compositor",
                "group": "one_variable",
                "covers": ["compositor.light_wrap_stabilization.invalid"],
                "compositing": {
                    "light_wrap_stabilization": override,
                },
            }
        ]
    )

    with pytest.raises(MatteQualityError, match="frozen variant overrides"):
        load_plan(_write_plan(tmp_path / "invalid-wrap.json", plan))


def test_frozen_light_wrap_context_uses_recorded_timeline_and_capture_fallback():
    recorded = matte_ablation._frozen_light_wrap_context(
        {
            "capture_sequence": 71,
            "capture_monotonic_ns": 9_000_000_000,
            "backdrop_identity": {
                "provider": "VideoBackdrop",
                "visual_generation": 3,
                "logical_index": 13,
                "timeline_s": 0.375,
                "discontinuity_revision": 4,
            },
        }
    )
    fallback = matte_ablation._frozen_light_wrap_context(
        {
            "capture_sequence": 72,
            "capture_monotonic_ns": 9_033_333_333,
            "backdrop_identity": {
                "provider": "recorded-proxy",
                "visual_generation": 3,
            },
        }
    )
    camera = matte_ablation._frozen_light_wrap_context(
        {
            "capture_sequence": 73,
            "capture_monotonic_ns": 9_066_666_666,
            "backdrop_identity": {
                "provider": "CameraBackdrop",
                "visual_generation": 4,
                "generation": 9,
                "capture_monotonic_ns": 8_500_000_000,
            },
        }
    )

    assert recorded.frame_id == 13
    assert recorded.timestamp_ns == 375_000_000
    assert recorded.discontinuity_revision == 4
    assert recorded.source_token[-1] == 3
    assert fallback.frame_id == 72
    assert fallback.timestamp_ns == 9_033_333_333
    assert fallback.discontinuity_revision == 0
    assert fallback.source_token == (
        "frozen-recorded-backdrop",
        "recorded-proxy",
        3,
    )
    assert camera.frame_id == 9
    assert camera.timestamp_ns == 8_500_000_000
    assert camera.source_token[-1] == 4


def test_frozen_light_wrap_context_requires_backdrop_generation_proof():
    with pytest.raises(MatteQualityError, match="generation identity"):
        matte_ablation._frozen_light_wrap_context(
            {
                "capture_sequence": 72,
                "capture_monotonic_ns": 9_033_333_333,
                "backdrop_identity": {
                    "provider": "legacy-recording",
                },
            }
        )


def test_frozen_light_wrap_stabilization_is_deep_merged_and_deterministic(
    generated_screen,
    tmp_path,
):
    root, _output, _report = generated_screen
    source_bundle = MatteReplayBundle(root / "source-bundle")
    source_annotations = MatteQualityAnnotations(
        root / "source-annotations",
        source_bundle,
    )

    def replay(name: str) -> MatteReplayBundle:
        row_root = tmp_path / name
        row_root.mkdir(mode=0o700)
        bundle, _annotations = matte_ablation._frozen_variant(
            source_bundle,
            source_annotations,
            {
                "lane": "compositor",
                "postprocess": {},
                "compositing": {
                    "light_wrap_stabilization": {
                        "mode": "temporal_bounded",
                    }
                },
            },
            row_root,
            max_bytes=64 * 1024 * 1024,
        )
        return bundle

    first = replay("stable-wrap-a")
    second = replay("stable-wrap-b")

    configured = first.frames[0]["configured_controls"]["compositing"][
        "light_wrap_stabilization"
    ]
    assert configured == {
        "mode": "temporal_bounded",
        "time_constant_s": 0.12,
    }

    first_hashes: list[str] = []
    second_hashes: list[str] = []
    for first_frame, second_frame in zip(
        first.frames,
        second.frames,
        strict=True,
    ):
        first_hashes.append(
            hashlib.sha256(
                first.load_array(first_frame, "base_composite").tobytes()
            ).hexdigest()
        )
        second_hashes.append(
            hashlib.sha256(
                second.load_array(second_frame, "base_composite").tobytes()
            ).hexdigest()
        )
        assert (
            first_frame["effective_controls"]["light_wrap_stabilization"]
            == second_frame["effective_controls"]["light_wrap_stabilization"]
        )
    assert first_hashes == second_hashes

    first_snapshot = first.frames[0]["effective_controls"]["light_wrap_stabilization"]
    last_snapshot = first.frames[-1]["effective_controls"]["light_wrap_stabilization"]
    assert first_snapshot["configured_mode"] == "temporal_bounded"
    assert first_snapshot["effective_mode"] == "temporal_bounded"
    assert first_snapshot["time_constant_s"] == 0.12
    assert first_snapshot["updates"] == 1
    assert first_snapshot["last_reset_reason"] == "initial"
    assert first_snapshot["retained_bytes"] > 0
    assert last_snapshot["updates"] == len(source_bundle.frames)
    assert last_snapshot["last_dt_s"] == pytest.approx(0.033333333)


def test_frozen_spatial_policy_deep_merges_and_reports_effective_controls(
    generated_screen,
    tmp_path,
):
    root, _output, _report = generated_screen
    variants = [
        {
            "id": "stable_spatial",
            "kind": "frozen",
            "lane": "mediapipe",
            "group": "spatial_policy",
            "covers": [
                "mediapipe.spatial_edge_refinement.stable_guided",
            ],
            "postprocess": {
                "edge_refine": True,
                "spatial_edge_refinement": {
                    "mode": "stable_guided",
                    "min_radius_px": 3,
                },
            },
        }
    ]
    plan = _write_plan(tmp_path / "spatial-plan.json", _base_plan(variants))

    report = run_ablation(
        root / "source-bundle",
        root / "source-annotations",
        plan,
        tmp_path / "spatial-output",
        max_output_bytes=64 * 1024 * 1024,
    )

    configuration = _row(report, "stable_spatial")["configuration"]
    expected = {
        "mode": "stable_guided",
        "reference_short_edge_px": 720,
        "radius_at_reference_px": 8,
        "min_radius_px": 3,
        "max_radius_px": 12,
    }
    assert configuration["configured_spatial_edge_refinement"] == expected
    assert configuration["effective_spatial_edge_refinement"] == expected
    assert configuration["effective_edge_refinement_mode"] == "stable_guided"
    assert configuration["effective_edge_refinement_radius_px"] == 3


def test_frozen_motion_policy_reports_the_exercised_effective_refiner(
    generated_screen,
    tmp_path,
):
    root, _output, _report = generated_screen
    source_bundle = MatteReplayBundle(root / "source-bundle")
    source_annotations = MatteQualityAnnotations(
        root / "source-annotations",
        source_bundle,
    )
    row_root = tmp_path / "motion-row"
    row_root.mkdir(mode=0o700)

    bundle, _annotations = matte_ablation._frozen_variant(
        source_bundle,
        source_annotations,
        {
            "lane": "mediapipe",
            "postprocess": {
                "boundary_stabilization": {
                    "mode": "motion_aware",
                    "time_constant_s": 0.08,
                    "max_motion_px_per_s": 480.0,
                }
            },
            "compositing": {},
        },
        row_root,
        max_bytes=64 * 1024 * 1024,
    )

    effective = bundle.frames[0]["effective_controls"]["refiner"]
    assert effective["temporal_smoothing"] == 0.0
    assert effective["boundary_stabilization"] == {
        "mode": "motion_aware",
        "time_constant_s": 0.08,
        "max_motion_px_per_s": 480.0,
    }


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


def test_recorded_variant_cannot_hide_late_device_fallback(
    generated_screen,
    tmp_path,
):
    root, _output, _report = generated_screen
    late_fallback = evidence.generate_bundle(
        tmp_path,
        "late-fallback",
        profile="stable",
        backend="rvm",
        ratio=0.67,
        device_sequence=(
            "generated-proxy",
            "generated-proxy",
            "cpu",
            "cpu",
            "cpu",
            "cpu",
        ),
    )
    variants = [
        {
            "id": "late_fallback",
            "kind": "recorded",
            "lane": "rvm",
            "group": "backend",
            "covers": ["backend.rvm_generated"],
            "bundle": str(late_fallback[0]),
            "annotations": str(late_fallback[1]),
            "timestamp_policy": "exact",
            "evidence_kind": "generated-proxy",
            "expected_backend": "RVMSegmenter",
            "expected_device": "generated-proxy",
            "expected_rvm_downsample_ratio": 0.67,
        }
    ]
    plan = _write_plan(tmp_path / "late-fallback-plan.json", _base_plan(variants))

    with pytest.raises(MatteQualityError, match="effective device"):
        run_ablation(
            root / "source-bundle",
            root / "source-annotations",
            plan,
            tmp_path / "late-fallback-output",
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
