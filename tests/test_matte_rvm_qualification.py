"""Fail-closed MATTE-2.5 matrix qualification tests."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, cast

import pytest

import matte_rvm_qualification_evidence as evidence
from custback.matte_diagnostics import MatteReplayBundle
from custback.matte_quality import MatteQualityAnnotations, MatteQualityError
from custback.matte_rvm_qualification import (
    _GATE_SPECS,
    _LoadedCell,
    _profile_semantic_reasons,
    _quality_cache_key,
    qualify_plan,
    run_qualification,
)


def _recorded(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report["rows"]
    assert isinstance(rows, list)
    return [row for row in rows if row["evidence_kind"] != "unavailable"]


def _semantic_cell(
    candidate_id: str,
    *,
    service_p95_ms: float,
    quality_rank: float,
) -> _LoadedCell:
    quality_gates = [
        {
            "metric": metric,
            "actual": quality_rank if operation == ">=" else 4.0 - quality_rank,
        }
        for metric, (operation, _minimum, _maximum) in _GATE_SPECS.items()
    ]
    return _LoadedCell(
        plan={
            "candidate_id": candidate_id,
            "hardware_id": "host",
            "provider": "cpu",
            "canvas_id": "sd",
            "cadence": "native30",
            "render_mode": "raw_model",
        },
        result={
            "outcome": "qualified",
            "performance": {
                "new_frame_service_steady_state_ms": {
                    "p95": service_p95_ms,
                }
            },
            "quality_gates": quality_gates,
        },
        bundle=cast(Any, object()),
    )


def test_generated_matrix_is_deterministic_private_and_never_selects_profile(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(tmp_path)

    first = qualify_plan(generated.plan)
    second = qualify_plan(generated.plan)

    assert first == second
    assert first["profiles"]["status"] == "not_proposed"
    assert first["profiles"]["definitions"] == []
    assert first["production"] == {
        "default_changed": False,
        "high_detail_global_default_selected": False,
        "generated_proxy_can_select_profile": False,
    }
    assert {row["outcome"] for row in _recorded(first)} == {"not_decidable"}
    assert all(
        row["effective"]["active_provider"] == "cpu"
        and row["effective"]["effective_rvm_downsample_ratio"] == 0.5
        for row in _recorded(first)
    )
    assert "rvm_resnet50_fp32.onnx" in json.dumps(first)
    assert str(tmp_path) not in json.dumps(first)
    contract = first["qualification_contract"]
    assert contract["quality_policy_id"] == "matte-0.2-ratified-v1"
    assert {candidate["id"] for candidate in contract["candidates"]} == {
        "shortlist",
        "auto",
        "compat",
    }
    assert all(
        "model_path" not in candidate["segmentation"]
        for candidate in contract["candidates"]
    )
    assert contract["qualified_compositor_sha256"]

    output = tmp_path / "report"
    written = run_qualification(generated.plan, output)
    assert written == first
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "qualification.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "qualification.md").stat().st_mode) == 0o600


def test_model_backed_rows_qualify_but_unavailable_matrix_blocks_candidate(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    assert {row["outcome"] for row in _recorded(report)} == {"qualified"}
    assert all(
        row["performance"]["budget_status"] == "within_budget"
        and row["performance"]["new_frame_service_steady_state_ms"]["p95"] == 4.0
        and row["provider"]["active"] == "cpu"
        for row in _recorded(report)
    )
    decision = next(
        item
        for item in report["candidate_decisions"]
        if item["candidate_id"] == "shortlist"
    )
    assert decision["outcome"] == "not_decidable"
    assert decision["declared_cell_count"] == decision["required_cell_count"]
    assert report["coverage"]["complete"] is False


@pytest.mark.timeout(60)
def test_complete_cross_device_matrix_publishes_self_describing_profiles(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        sustainable=True,
        record_all=True,
        propose_profiles=True,
    )

    report = qualify_plan(generated.plan)

    assert report["coverage"]["complete"] is True
    assert {row["outcome"] for row in report["rows"]} == {"qualified"}
    assert {row["provider"]["active"] for row in report["rows"]} == {
        "cpu",
        "cuda",
        "directml",
    }
    assert {item["outcome"] for item in report["candidate_decisions"]} == {"qualified"}
    profiles = report["profiles"]
    assert profiles["status"] == "qualified"
    assert {item["name"] for item in profiles["definitions"]} == {
        "performance",
        "balanced",
        "quality",
    }
    for definition in profiles["definitions"]:
        assert "model_path" not in definition["segmentation"]
        assert definition["model"]["sha256"]
        assert definition["qualified_compositor_sha256"]
        assert len(definition["qualification_scope"]) >= 2
        assert {
            scope["provider"]["requested"]
            for scope in definition["qualification_scope"]
        } >= {"cpu"}
        assert all(
            scope["canvas"]["id"] == "sd" for scope in definition["qualification_scope"]
        )
    quality = next(
        item for item in profiles["definitions"] if item["name"] == "quality"
    )
    assert quality["model"]["id"] == "rvm_resnet50_fp32.onnx"
    auto_rows = [row for row in report["rows"] if row["key"][0] == "auto"]
    assert auto_rows
    assert all(
        row["effective"]["effective_rvm_downsample_ratio"] == 1.0 for row in auto_rows
    )


def test_profile_labels_cannot_be_swapped_over_measured_semantics() -> None:
    loaded = {
        "performance": _semantic_cell(
            "candidate_performance",
            service_p95_ms=4.0,
            quality_rank=1.0,
        ),
        "balanced": _semantic_cell(
            "candidate_balanced",
            service_p95_ms=5.0,
            quality_rank=2.0,
        ),
        "quality": _semantic_cell(
            "candidate_quality",
            service_p95_ms=6.0,
            quality_rank=3.0,
        ),
    }
    correct = [
        {"name": "performance", "candidate_id": "candidate_performance"},
        {"name": "balanced", "candidate_id": "candidate_balanced"},
        {"name": "quality", "candidate_id": "candidate_quality"},
    ]
    swapped = [
        {"name": "performance", "candidate_id": "candidate_quality"},
        {"name": "balanced", "candidate_id": "candidate_balanced"},
        {"name": "quality", "candidate_id": "candidate_performance"},
    ]

    assert _profile_semantic_reasons(correct, loaded) == []
    assert _profile_semantic_reasons(swapped, loaded) == [
        "profile latency ordering is not stable across every scope",
        "profile quality ordering is not stable across every scope",
    ]


def test_opaque_core_gate_rejects_temporally_stable_candidate(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        strict_opaque=True,
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    for row in _recorded(report):
        assert row["outcome"] == "rejected"
        opaque = next(
            gate for gate in row["quality_gates"] if gate["id"] == "opaque_p05"
        )
        contour = next(gate for gate in row["quality_gates"] if gate["id"] == "contour")
        assert opaque["status"] == "fail"
        assert contour["status"] == "pass"


@pytest.mark.parametrize(
    ("fault", "outcome", "category", "budget_status"),
    (
        (
            "telemetry_not_applicable",
            "not_decidable",
            "evidence completeness",
            "within_budget",
        ),
        ("late_fallback", "rejected", "provider fallback", "within_budget"),
        ("ratio_drift", "rejected", "configuration", "within_budget"),
        ("missing_fgr", "not_decidable", "evidence completeness", "within_budget"),
        ("budget", "rejected", "performance", "degraded"),
        ("cadence_lie", "not_decidable", "evidence completeness", "within_budget"),
    ),
)
def test_per_frame_contracts_and_steady_budget_fail_closed(
    tmp_path: Path,
    fault: str,
    outcome: str,
    category: str,
    budget_status: str,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault=fault,
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    rows = _recorded(report)
    assert all(row["outcome"] == outcome for row in rows)
    assert all(
        any(reason["category"] == category for reason in row["reasons"]) for row in rows
    )
    assert all(row["performance"]["budget_status"] == budget_status for row in rows)


def test_shortlist_binding_and_unavailable_reason_are_path_safe(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(tmp_path)
    plan = json.loads(generated.plan.read_text(encoding="ascii"))
    plan["candidates"][0]["segmentation"]["model_path"] = "/different/local/model.onnx"
    generated.plan.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    generated.plan.chmod(0o600)
    report = qualify_plan(generated.plan)
    assert _recorded(report)

    plan["candidates"][0]["segmentation"]["rvm_downsample"] = 0.6
    generated.plan.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    generated.plan.chmod(0o600)
    with pytest.raises(MatteQualityError, match="segmentation is not bound"):
        qualify_plan(generated.plan)

    plan["candidates"][0]["segmentation"]["rvm_downsample"] = 0.5
    unavailable = next(
        cell for cell in plan["cells"] if cell["status"] == "unavailable"
    )
    unavailable["availability_reason"] = "/private/device/error text"
    generated.plan.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    generated.plan.chmod(0o600)
    with pytest.raises(MatteQualityError, match="availability reason code"):
        qualify_plan(generated.plan)


def test_run_samples_must_align_to_every_unique_frame(tmp_path: Path) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
    )
    plan = json.loads(generated.plan.read_text(encoding="ascii"))
    recorded = next(cell for cell in plan["cells"] if cell["status"] == "recorded")
    run_path = Path(recorded["run"])
    run = json.loads(run_path.read_text(encoding="ascii"))
    run["memory_bytes"].pop()
    run_path.write_text(
        json.dumps(run, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    run_path.chmod(0o600)

    report = qualify_plan(generated.plan)

    row = next(row for row in _recorded(report) if row["id"] == recorded["id"])
    assert row["outcome"] == "not_decidable"
    assert any(
        reason["detail"] == "RSS samples do not cover every unique frame"
        for reason in row["reasons"]
    )


def test_caller_cannot_redefine_native30_or_decimated15_as_slow_cadence(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(tmp_path)
    plan = json.loads(generated.plan.read_text(encoding="ascii"))
    plan["policy"]["native30_min_unique_fps"] = 0.1
    plan["policy"]["decimated15_min_unique_fps"] = 0.1
    generated.plan.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    generated.plan.chmod(0o600)

    with pytest.raises(MatteQualityError, match="ratified cadence floors"):
        qualify_plan(generated.plan)


def test_gate_operation_is_bound_to_each_metric_direction(tmp_path: Path) -> None:
    generated = evidence.generate_qualification(tmp_path)
    plan = json.loads(generated.plan.read_text(encoding="ascii"))
    opaque = next(
        gate for gate in plan["policy"]["quality_gates"] if gate["id"] == "opaque_p05"
    )
    opaque["op"] = "<="
    generated.plan.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    generated.plan.chmod(0o600)

    with pytest.raises(MatteQualityError, match="metric direction"):
        qualify_plan(generated.plan)


def test_quality_gate_threshold_cannot_weaken_ratified_policy(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(tmp_path)
    plan = json.loads(generated.plan.read_text(encoding="ascii"))
    opaque = next(
        gate for gate in plan["policy"]["quality_gates"] if gate["id"] == "opaque_p05"
    )
    opaque["value"] = 0.94
    generated.plan.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    generated.plan.chmod(0o600)

    with pytest.raises(MatteQualityError, match="weaker than the ratified"):
        qualify_plan(generated.plan)


def test_caller_cannot_reduce_sustainability_to_a_trivial_sample(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(tmp_path)
    plan = json.loads(generated.plan.read_text(encoding="ascii"))
    plan["policy"]["min_warmup_frames"] = 1
    plan["policy"]["min_steady_frames"] = 1
    plan["policy"]["min_observation_s"] = 0.001
    generated.plan.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    generated.plan.chmod(0o600)

    with pytest.raises(MatteQualityError, match="ratified floors"):
        qualify_plan(generated.plan)


def test_static_short_clip_stays_not_decidable(tmp_path: Path) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
    )

    report = qualify_plan(generated.plan)

    assert {row["outcome"] for row in _recorded(report)} == {"not_decidable"}
    assert all(
        any(
            "stationary, moving, fast-motion, and occlusion" in reason["detail"]
            for reason in row["reasons"]
        )
        for row in _recorded(report)
    )


@pytest.mark.parametrize(
    "fault",
    (
        "effective_foreground",
        "effective_wrap",
        "effective_blend",
        "effective_refiner",
    ),
)
def test_qualified_compositor_effective_policy_must_match_plan(
    tmp_path: Path,
    fault: str,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault=fault,
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    compositor_rows = [
        row for row in _recorded(report) if row["key"][5] == "qualified_compositor"
    ]
    assert compositor_rows
    assert all(row["outcome"] == "rejected" for row in compositor_rows)
    assert all(
        any(
            reason["category"] == "attribution"
            and "effective matte/compositor policy" in reason["detail"]
            for reason in row["reasons"]
        )
        for row in compositor_rows
    )


def test_annotation_drift_invalidates_the_complete_source_family(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="annotation_drift",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    rows = _recorded(report)
    assert rows
    assert all(row["outcome"] == "rejected" for row in rows)
    assert all(
        any(reason["category"] == "annotation identity" for reason in row["reasons"])
        for row in rows
    )


def test_digest_bound_annotation_release_gate_must_pass(tmp_path: Path) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="annotation_gate",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    rows = _recorded(report)
    assert rows
    assert all(row["outcome"] == "rejected" for row in rows)
    assert all(
        any(gate["status"] == "fail" for gate in row["annotation_gates"])
        and any(reason["category"] == "annotation gate" for reason in row["reasons"])
        for row in rows
    )


def test_decimated_annotations_must_project_from_native_evidence(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="decimated_annotation_drift",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    decimated = [row for row in _recorded(report) if row["key"][4] == "decimated15"]
    assert decimated
    assert all(row["outcome"] == "rejected" for row in decimated)
    assert all(
        any(
            reason["detail"]
            == (
                "decimated15 annotations are not the exact every-other "
                "native30 projection"
            )
            for reason in row["reasons"]
        )
        for row in decimated
    )


def test_decimated_pre_resize_source_digest_must_project_from_native(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="pre_resize_source_drift",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    decimated = [row for row in _recorded(report) if row["key"][4] == "decimated15"]
    assert decimated
    assert all(row["outcome"] == "rejected" for row in decimated)
    assert all(
        any(
            reason["detail"]
            == (
                "decimated15 pre-resize source is not the exact every-other "
                "native30 projection"
            )
            for reason in row["reasons"]
        )
        for row in decimated
    )


def test_native_pre_resize_source_identity_is_shared_across_matrix(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="native_source_family_drift",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    rows = _recorded(report)
    assert rows
    assert all(row["outcome"] == "rejected" for row in rows)
    assert all(
        any(
            reason["detail"] == "native30 pre-resize source differs across the matrix"
            for reason in row["reasons"]
        )
        for row in rows
    )


def test_contradictory_stage_and_full_frame_timings_fail_closed(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="contradictory_timing",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    rows = _recorded(report)
    assert all(row["outcome"] == "not_decidable" for row in rows)
    assert all(
        any(
            reason["detail"]
            == "per-frame timing components exceed an enclosing boundary"
            for reason in row["reasons"]
        )
        for row in rows
    )


def test_overlapping_full_frame_service_cannot_claim_serialized_throughput(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="timeline_overlap",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    rows = _recorded(report)
    assert all(row["outcome"] == "not_decidable" for row in rows)
    assert all(
        any(
            reason["detail"]
            == ("frame service or queue age contradicts the serialized send timeline")
            for reason in row["reasons"]
        )
        for row in rows
    )


def test_queue_age_must_equal_capture_to_dequeue_timeline(tmp_path: Path) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="queue_lie",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    rows = _recorded(report)
    assert all(row["outcome"] == "not_decidable" for row in rows)
    assert all(
        any(
            reason["detail"]
            == ("frame service or queue age contradicts the serialized send timeline")
            for reason in row["reasons"]
        )
        for row in rows
    )


def test_quality_cache_identity_includes_the_output_timeline(tmp_path: Path) -> None:
    normal = evidence.generate_qualification(tmp_path / "normal")
    shifted = evidence.generate_qualification(
        tmp_path / "shifted",
        fault="queue_lie",
    )

    def first_evidence(
        plan_path: Path,
    ) -> tuple[MatteReplayBundle, MatteQualityAnnotations]:
        plan = json.loads(plan_path.read_text(encoding="ascii"))
        cell = next(cell for cell in plan["cells"] if cell["status"] == "recorded")
        bundle = MatteReplayBundle(cell["bundle"])
        return bundle, MatteQualityAnnotations(cell["annotations"], bundle)

    normal_bundle, normal_annotations = first_evidence(normal.plan)
    shifted_bundle, shifted_annotations = first_evidence(shifted.plan)

    assert _quality_cache_key(normal_bundle, normal_annotations) != _quality_cache_key(
        shifted_bundle,
        shifted_annotations,
    )


def test_slow_source_timestamps_cannot_qualify_nominal_cadence_lanes(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="slow_cadence",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    rows = _recorded(report)
    assert rows
    assert all(row["outcome"] == "rejected" for row in rows)
    assert all(
        any(
            reason["category"] == "performance"
            and "source cadence is outside" in reason["detail"]
            for reason in row["reasons"]
        )
        for row in rows
    )


def test_hardware_inventory_identity_must_be_stable_across_cells(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
    )
    plan = json.loads(generated.plan.read_text(encoding="ascii"))
    recorded = [cell for cell in plan["cells"] if cell["status"] == "recorded"]
    run_path = Path(recorded[-1]["run"])
    run = json.loads(run_path.read_text(encoding="ascii"))
    run["hardware"]["label"] = "contradictory host label"
    run_path.write_text(
        json.dumps(run, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    run_path.chmod(0o600)

    report = qualify_plan(generated.plan)

    assert all(row["outcome"] == "not_decidable" for row in _recorded(report))
    assert all(
        any(
            reason["detail"] == "hardware inventory identity is inconsistent"
            for reason in row["reasons"]
        )
        for row in _recorded(report)
    )


def test_provider_runtime_environment_must_be_stable_across_cells(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(
        tmp_path,
        provenance_kind="consented-local",
        fault="environment_drift",
        sustainable=True,
    )

    report = qualify_plan(generated.plan)

    rows = _recorded(report)
    assert rows
    assert all(row["outcome"] == "not_decidable" for row in rows)
    assert all(
        any(
            reason["detail"] == "hardware/provider environment identity is inconsistent"
            for reason in row["reasons"]
        )
        for row in rows
    )


def test_hardware_and_provider_runtime_attestations_are_digest_bound(
    tmp_path: Path,
) -> None:
    generated = evidence.generate_qualification(tmp_path)
    plan = json.loads(generated.plan.read_text(encoding="ascii"))
    recorded = next(cell for cell in plan["cells"] if cell["status"] == "recorded")
    run_path = Path(recorded["run"])
    original = json.loads(run_path.read_text(encoding="ascii"))

    tampered_inventory = json.loads(json.dumps(original))
    tampered_inventory["hardware"]["inventory"]["cpu_model"] = "Different Test CPU"
    run_path.write_text(
        json.dumps(tampered_inventory, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    run_path.chmod(0o600)
    with pytest.raises(MatteQualityError, match="does not bind its inventory"):
        qualify_plan(generated.plan)

    tampered_runtime = json.loads(json.dumps(original))
    tampered_runtime["provider"]["runtime"]["execution_provider"] = (
        "CUDAExecutionProvider"
    )
    run_path.write_text(
        json.dumps(tampered_runtime, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    run_path.chmod(0o600)
    with pytest.raises(MatteQualityError, match="runtime contradicts its request"):
        qualify_plan(generated.plan)
