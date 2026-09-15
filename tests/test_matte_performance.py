"""Deterministic MATTE-3.4 performance evidence and decision tests."""

from __future__ import annotations

import copy
import itertools
import json
import stat
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import custback.__main__ as core_main
import custback.matte_performance as performance
from custback.compositor import (
    COMPOSITOR_P95_SUB_BUDGET_MS,
    COMPOSITOR_SUBSTAGE_NAMES,
)
from custback.matte_performance import (
    COMPOSITOR_VARIANTS,
    FULL_PATH_SCHEMA,
    FULL_PATH_VERSION,
    ROW_SCHEMA,
    ROW_VERSION,
    CompositorFrameSample,
    MattePerformanceError,
    build_performance_report,
    logical_arrival_sweep,
    nearest_rank_summary,
    profile_fixed_720p_compositor,
    profile_fixed_720p_compositor_samples,
    report_markdown,
    validate_performance_report,
)

SOURCE_SHA256 = "a" * 64
LINEAGE_SHA256 = "e" * 64


def _row(
    variant: performance.CompositorVariant,
    *,
    total_ms: float = 20.0,
    measured: int = 300,
    warmup: int = 30,
    evidence_kind: str = "fixed-replay-measured",
    max_delta: int | None = None,
    foreground_endpoint_exact: bool = True,
    background_endpoint_exact: bool = True,
    source_frame_count: int | None = None,
) -> dict[str, Any]:
    delta = (
        0
        if max_delta is None and variant.blend_space == "srgb_legacy"
        else 1
        if max_delta is None
        else max_delta
    )
    timings = {name: [0.0] * measured for name in ("total", *COMPOSITOR_SUBSTAGE_NAMES)}
    timings["total"] = [total_ms] * measured
    timings["input_mask_validation"] = [0.25] * measured
    timings["final_blend_conversion"] = [0.75] * measured
    return {
        "schema": ROW_SCHEMA,
        "version": ROW_VERSION,
        "id": variant.id,
        "source_sha256": SOURCE_SHA256,
        "source_scope": {
            "kind": "explicit-frame-sequence",
            "source_frame_count": (
                warmup + measured if source_frame_count is None else source_frame_count
            ),
            "profiled_frame_count": warmup + measured,
            "unique_capture_sequence": True,
            "privacy_guarded_bundle": False,
            "bundle_manifest_sha256": None,
            "measured_frame_lineage_sha256": LINEAGE_SHA256,
            "execution_pacing": "unpaced-tight-loop",
        },
        "evidence_kind": evidence_kind,
        "blend_space": variant.blend_space,
        "use_model_foreground": variant.use_model_foreground,
        "light_wrap": variant.light_wrap,
        "warmup_frame_count": warmup,
        "measured_frame_count": measured,
        "timing_samples_ms": timings,
        "allocation_samples_bytes": {
            "known_transient_allocation_bytes": [2_764_800] * measured,
            "retained_workspace_bytes": [18_000_000] * measured,
        },
        "memory_bandwidth": {
            "available": False,
            "bytes_per_second": None,
            "counter_source": None,
            "source_sha256": None,
            "hardware_identity_sha256": None,
            "provider_environment_sha256": None,
            "measurement_run_sha256": None,
        },
        "equivalence": {
            "reference_contract": (
                "frozen-srgb-legacy-v1"
                if variant.blend_space == "srgb_legacy"
                else "linear-srgb-reference-v1"
            ),
            "max_channel_delta": delta,
            "foreground_endpoint_exact": foreground_endpoint_exact,
            "background_endpoint_exact": background_endpoint_exact,
            "output_shape": [720, 1280, 3],
            "output_dtype": "uint8",
            "output_c_contiguous": True,
            "deterministic_repeat_exact": True,
        },
    }


def _rows(**overrides: Any) -> list[dict[str, Any]]:
    return [_row(variant, **overrides) for variant in COMPOSITOR_VARIANTS]


def _full_path(
    *,
    service_ms: float = 30.0,
    evidence_kind: str = "model-backed",
    measured: int = 300,
    warmup: int = 30,
    source_sha256: str = SOURCE_SHA256,
    output_repeat_count: int = 0,
    no_unread_repeat_count: int = 0,
    measured_frame_lineage_sha256: str = LINEAGE_SHA256,
    frame_processing_ms: float = 20.0,
    compositor_ms: float = 10.0,
    non_pacing_cycle_ms: float | None = None,
) -> dict[str, Any]:
    deadline_misses = (
        measured if frame_processing_ms > performance.COMPLETE_SERVICE_BUDGET_MS else 0
    )
    serialized_ms = service_ms + 7.0
    cycle_ms = service_ms + 1.0 if non_pacing_cycle_ms is None else non_pacing_cycle_ms
    service_samples = [service_ms] * measured
    cycle_samples = [cycle_ms] * measured
    serialized_deadline_misses = (
        measured
        if serialized_ms
        > performance.COMPLETE_SERVICE_BUDGET_MS
        + performance.SERIALIZED_DEADLINE_GRACE_MS
        else 0
    )
    return {
        "schema": FULL_PATH_SCHEMA,
        "version": FULL_PATH_VERSION,
        "evidence_kind": evidence_kind,
        "source_sha256": source_sha256,
        "hardware_id": "cuda_host",
        "hardware_identity_sha256": "b" * 64,
        "provider_environment_sha256": "c" * 64,
        "measurement_run_sha256": "9" * 64,
        "rvm_model_sha256": performance.RVM_MODEL.sha256,
        "bundle_manifest_sha256": None,
        "measured_frame_lineage_sha256": measured_frame_lineage_sha256,
        "matrix_row_id": "srgb_legacy_both",
        "effective_policy": {
            "resolved_rvm_downsample_ratio": 0.4,
            "raw_alpha_mode": "native_soft_alpha",
            "halo_mode": "mask_shift_only",
            "mask_shift_px": 0,
            "boundary_stabilization_mode": "off",
            "use_model_foreground": True,
            "light_wrap": 0.25,
            "light_wrap_stabilization_mode": "off",
            "blend_space": "srgb_legacy",
            "color_correction_mode": "off",
        },
        "background_provider_scope": "resident-recorded-frame-copy",
        "sink_id": "pyvirtualcam",
        "sink_identity_sha256": "d" * 64,
        "sink_submission_copy_in_service_boundary": True,
        "canvas": {"width": 1280, "height": 720, "nominal_fps": 30},
        "backend": "rvm",
        "provider": "cuda",
        "fallback_count": 0,
        "boundary": ("unique-dequeue-through-sink-submit-excluding-deliberate-pacing"),
        "warmup_frame_count": warmup,
        "service_samples_ms": service_samples,
        "non_pacing_cycle_samples_ms": cycle_samples,
        "frame_processing_samples_ms": [frame_processing_ms] * measured,
        "serialized_new_frame_samples_ms": [serialized_ms] * measured,
        "rvm_preprocess_samples_ms": [2.0] * measured,
        "rvm_inference_samples_ms": [5.0] * measured,
        "rvm_postprocess_samples_ms": [2.0] * measured,
        "background_selection_samples_ms": [1.0] * measured,
        "compositor_samples_ms": [compositor_ms] * measured,
        "post_composite_validation_samples_ms": [1.0] * measured,
        "sink_submission_samples_ms": [1.0] * measured,
        "pacing_wait_samples_ms": [7.0] * measured,
        "schedule_lateness_samples_ms": (
            performance._logical_schedule_lateness_samples(  # noqa: SLF001
                service_samples,
                cycle_samples,
            )
        ),
        "unique_composite_count": measured,
        "model_invocation_count": measured,
        "output_send_count": measured + output_repeat_count,
        "output_repeat_count": output_repeat_count,
        "no_unread_repeat_count": no_unread_repeat_count,
        "capture_slot_overwrite_count": 0,
        "capture_sequence_gap_count": 0,
        "capture_missing_input_count": 0,
        "processing_deadline_miss_count": deadline_misses,
        "serialized_new_frame_deadline_miss_count": (serialized_deadline_misses),
        "sink_recovery_count": 0,
    }


def test_nearest_rank_summary_includes_p50_p95_and_p99() -> None:
    summary = nearest_rank_summary(list(range(1, 101)))

    assert summary == {
        "count": 100,
        "mean": 50.5,
        "p50": 50.0,
        "p95": 95.0,
        "p99": 99.0,
        "max": 100.0,
    }


def test_required_matrix_is_exactly_four_features_in_both_spaces() -> None:
    assert [
        (row.blend_space, row.use_model_foreground, row.light_wrap)
        for row in COMPOSITOR_VARIANTS
    ] == [
        ("srgb_legacy", False, 0.0),
        ("srgb_legacy", True, 0.0),
        ("srgb_legacy", False, 0.25),
        ("srgb_legacy", True, 0.25),
        ("linear_srgb", False, 0.0),
        ("linear_srgb", True, 0.0),
        ("linear_srgb", False, 0.25),
        ("linear_srgb", True, 0.25),
    ]
    assert COMPOSITOR_P95_SUB_BUDGET_MS == 22.0


def test_complete_model_backed_matrix_and_service_qualify_deterministically() -> None:
    first = build_performance_report(_rows(), full_path_evidence=_full_path())
    second = build_performance_report(_rows(), full_path_evidence=_full_path())

    assert first == second
    assert validate_performance_report(first) == first
    assert first["decision"]["outcome"] == "qualified"
    assert first["matrix"]["outcome"] == "qualified"
    assert first["matrix"]["signed_worst_compositor_headroom_ms"] == 2.0
    assert first["full_path"]["signed_service_headroom_ms"] == 3.333334
    assert first["full_path"]["sustainable_unique_frame_profile_fps"] == 30
    assert first["decision"]["output_repeats_credited_as_unique_work"] is False
    assert first["decision"]["reaction_ready"] is False
    assert [row["arrival_fps"] for row in first["full_path"]["arrival_sweep"]] == [
        30,
        27,
        24,
        20,
        15,
    ]


def test_missing_or_proxy_full_path_fails_closed() -> None:
    missing = build_performance_report(_rows())
    proxy = build_performance_report(
        _rows(),
        full_path_evidence=_full_path(evidence_kind="generated-proxy"),
    )

    assert missing["decision"]["outcome"] == "not_decidable"
    assert missing["full_path"]["status"] == "missing"
    assert proxy["decision"]["outcome"] == "not_decidable"
    assert proxy["full_path"]["checks"]["model_backed"] is False
    assert proxy["full_path"]["sustainable_unique_frame_profile_fps"] is None
    assert proxy["decision"]["declared_sustainable_unique_frame_profile_fps"] is None


def test_full_path_requires_builtin_model_and_distinct_identity_digests() -> None:
    wrong_model = _full_path()
    wrong_model["rvm_model_sha256"] = "d" * 64
    same_identity = _full_path()
    same_identity["provider_environment_sha256"] = same_identity[
        "hardware_identity_sha256"
    ]

    model_report = build_performance_report(
        _rows(),
        full_path_evidence=wrong_model,
    )
    identity_report = build_performance_report(
        _rows(),
        full_path_evidence=same_identity,
    )

    assert model_report["full_path"]["outcome"] == "rejected"
    assert (
        model_report["full_path"]["checks"]["builtin_rvm_cuda_without_fallback"]
        is False
    )
    assert identity_report["full_path"]["outcome"] == "rejected"
    assert (
        identity_report["full_path"]["checks"][
            "opaque_hardware_provider_and_sink_digests_present"
        ]
        is False
    )


def test_full_path_binds_balanced_policy_matrix_row_and_sink_boundary() -> None:
    plain_row = _full_path()
    plain_row["matrix_row_id"] = "srgb_legacy_plain"
    wrong_ratio = _full_path()
    wrong_ratio["effective_policy"]["resolved_rvm_downsample_ratio"] = 0.5
    incomplete_sink = _full_path()
    incomplete_sink["sink_submission_copy_in_service_boundary"] = False

    plain_report = build_performance_report(
        _rows(),
        full_path_evidence=plain_row,
    )
    sink_report = build_performance_report(
        _rows(),
        full_path_evidence=incomplete_sink,
    )
    ratio_report = build_performance_report(
        _rows(),
        full_path_evidence=wrong_ratio,
    )

    assert plain_report["full_path"]["outcome"] == "rejected"
    assert (
        plain_report["full_path"]["checks"]["balanced_effective_policy_and_sink"]
        is False
    )
    assert sink_report["full_path"]["outcome"] == "rejected"
    assert (
        sink_report["full_path"]["checks"]["balanced_effective_policy_and_sink"]
        is False
    )
    assert ratio_report["full_path"]["outcome"] == "rejected"
    assert (
        ratio_report["full_path"]["checks"]["balanced_effective_policy_and_sink"]
        is False
    )


@pytest.mark.parametrize("sink_id", ("pyvirtualcam", "native"))
def test_full_path_accepts_only_named_production_sink_ids(sink_id: str) -> None:
    evidence = _full_path()
    evidence["sink_id"] = sink_id

    report = build_performance_report(
        _rows(),
        full_path_evidence=evidence,
    )

    assert report["full_path"]["outcome"] == "qualified"
    assert report["full_path"]["evidence"]["sink_id"] == sink_id


@pytest.mark.parametrize(
    "sink_id",
    (None, "null", "auto", "virtual_camera", "unknown"),
)
def test_full_path_rejects_null_or_unknown_sink_id(sink_id: object) -> None:
    evidence = _full_path()
    evidence["sink_id"] = sink_id

    with pytest.raises(MattePerformanceError, match="sink id"):
        build_performance_report(
            _rows(),
            full_path_evidence=evidence,
        )


def test_generated_or_short_compositor_samples_cannot_qualify() -> None:
    report = build_performance_report(
        _rows(
            measured=3,
            warmup=1,
            evidence_kind="generated-proxy",
        )
    )

    assert report["matrix"]["outcome"] == "not_decidable"
    assert {row["outcome"] for row in report["matrix"]["rows"]} == {"not_decidable"}
    assert all(
        row["checks"]["sample_floor"] is False
        and row["checks"]["fixed_replay_measured"] is False
        for row in report["matrix"]["rows"]
    )


def test_compositor_subbudget_has_signed_negative_headroom_and_rejects() -> None:
    report = build_performance_report(
        _rows(total_ms=22.5),
        full_path_evidence=_full_path(),
    )

    assert report["matrix"]["outcome"] == "rejected"
    assert report["decision"]["outcome"] == "rejected"
    assert report["matrix"]["signed_worst_compositor_headroom_ms"] == -0.5
    assert {
        row["signed_compositor_headroom_ms"] for row in report["matrix"]["rows"]
    } == {-0.5}


@pytest.mark.parametrize(
    ("blend_space", "delta", "accepted"),
    (
        ("srgb_legacy", 0, True),
        ("srgb_legacy", 1, False),
        ("linear_srgb", 1, True),
        ("linear_srgb", 2, False),
    ),
)
def test_output_equivalence_uses_space_owned_tolerance(
    blend_space: str,
    delta: int,
    accepted: bool,
) -> None:
    rows = _rows()
    target = next(row for row in rows if row["blend_space"] == blend_space)
    target["equivalence"]["max_channel_delta"] = delta

    report = build_performance_report(rows)
    result = next(row for row in report["matrix"]["rows"] if row["id"] == target["id"])

    assert result["checks"]["output_equivalence"] is accepted
    assert (result["outcome"] != "rejected") is accepted


def test_alpha_endpoint_failure_rejects_even_with_fast_proxy_samples() -> None:
    rows = _rows()
    rows[0]["equivalence"]["foreground_endpoint_exact"] = False

    report = build_performance_report(rows)

    assert report["matrix"]["rows"][0]["outcome"] == "rejected"
    assert report["decision"]["outcome"] == "rejected"


def test_arrival_sweep_classifies_explicit_tier_not_reciprocal_of_p95() -> None:
    sweep = logical_arrival_sweep([40.0] * 300)

    assert [row["arrival_fps"] for row in sweep] == [30, 27, 24, 20, 15]
    assert (
        next(row for row in sweep if row["arrival_fps"] == 30)["unique_composite_fps"]
        == 25.0
    )
    # The declared profile is one of the explicit arrival tiers. It is 24,
    # not the tempting but unsupported ``1000 / p95 == 25`` label.
    sustainable = next(
        row["arrival_fps"] for row in sweep if row["sustained_without_sequence_gaps"]
    )
    assert sustainable == 24


def test_27_unique_fps_alternative_can_qualify_above_service_p95_budget() -> None:
    evidence = _full_path()
    # Sixteen ordered 37 ms outliers put nearest-rank p95 above the service
    # budget. The following 30 ms samples drain the newest-frame queue back to
    # its starting age while sustaining well above 27 unique composites/s.
    service_samples = [37.0] * 16 + [30.0] * 284
    cycle_samples = [38.0] * 16 + [31.0] * 284
    evidence["service_samples_ms"] = service_samples
    evidence["non_pacing_cycle_samples_ms"] = cycle_samples
    evidence["serialized_new_frame_samples_ms"] = [
        sample + 7.0 for sample in service_samples
    ]
    evidence["schedule_lateness_samples_ms"] = (
        performance._logical_schedule_lateness_samples(  # noqa: SLF001
            service_samples,
            cycle_samples,
        )
    )
    evidence["serialized_new_frame_deadline_miss_count"] = 300
    report = build_performance_report(
        _rows(),
        full_path_evidence=evidence,
    )

    assert report["full_path"]["checks"]["service_p95_budget"] is False
    assert report["full_path"]["checks"]["arrival_30_unique_fps_alternative"] is True
    assert report["full_path"]["outcome"] == "qualified"
    assert report["full_path"]["sustainable_unique_frame_profile_fps"] == 27
    assert report["full_path"]["signed_service_headroom_ms"] == -3.666666


def test_arrival_sweep_does_not_call_a_growing_queue_sustainable() -> None:
    sweep = logical_arrival_sweep([33.4] * 300)
    arrival_30 = next(row for row in sweep if row["arrival_fps"] == 30)
    arrival_27 = next(row for row in sweep if row["arrival_fps"] == 27)

    assert arrival_30["logical_capture_sequence_gap_count"] == 0
    assert cast(float, arrival_30["ending_minus_starting_queue_age_ms"]) > 0.0
    assert arrival_30["queue_age_not_growing"] is False
    assert arrival_30["sustained_without_sequence_gaps"] is False
    assert arrival_27["sustained_without_sequence_gaps"] is True


def test_arrival_sweep_separates_gap_events_from_missing_inputs() -> None:
    arrival_30 = next(
        row
        for row in logical_arrival_sweep([100.0, 100.0, 100.0])
        if row["arrival_fps"] == 30
    )

    assert arrival_30["logical_capture_sequence_gap_count"] == 2
    assert arrival_30["logical_capture_missing_input_count"] == 4


def test_lower_rate_full_path_is_rejected_and_declared_honestly() -> None:
    report = build_performance_report(
        _rows(),
        full_path_evidence=_full_path(service_ms=40.0),
    )

    assert report["full_path"]["outcome"] == "rejected"
    assert report["decision"]["outcome"] == "rejected"
    assert report["full_path"]["sustainable_unique_frame_profile_fps"] == 24
    assert report["full_path"]["checks"]["service_p95_budget"] is False
    assert report["full_path"]["checks"]["arrival_30_unique_fps_alternative"] is False


def test_service_p95_cannot_hide_an_over_budget_non_pacing_cycle() -> None:
    report = build_performance_report(
        _rows(),
        full_path_evidence=_full_path(non_pacing_cycle_ms=40.0),
    )

    assert report["full_path"]["checks"]["service_p95_budget"] is True
    assert (
        report["full_path"]["checks"]["non_pacing_cycle_sustains_minimum_tier"] is False
    )
    assert report["full_path"]["sustainable_unique_frame_profile_fps"] == 24
    assert report["full_path"]["outcome"] == "rejected"


def test_full_path_compositor_p95_must_meet_budget_on_qualified_hardware() -> None:
    evidence = _full_path(
        service_ms=32.0,
        frame_processing_ms=28.0,
        compositor_ms=23.0,
    )
    for name in (
        "rvm_preprocess_samples_ms",
        "rvm_inference_samples_ms",
        "rvm_postprocess_samples_ms",
    ):
        evidence[name] = [1.0] * 300

    report = build_performance_report(_rows(), full_path_evidence=evidence)

    assert report["matrix"]["outcome"] == "qualified"
    assert report["full_path"]["checks"]["full_path_compositor_p95_sub_budget"] is False
    assert report["full_path"]["signed_compositor_headroom_ms"] == -1.0
    assert report["full_path"]["outcome"] == "rejected"


def test_unqualified_matrix_timing_still_allows_honest_lower_rate_classification() -> (
    None
):
    rows = _rows()
    selected = next(row for row in rows if row["id"] == "srgb_legacy_both")
    selected["timing_samples_ms"]["total"] = [23.0] * 300

    report = build_performance_report(
        rows,
        full_path_evidence=_full_path(service_ms=40.0),
    )

    assert report["matrix"]["outcome"] == "rejected"
    assert (
        report["full_path"]["checks"]["same_fixed_source_and_authoritative_matrix_row"]
        is True
    )
    assert (
        report["full_path"]["checks"]["same_fixed_source_and_qualified_matrix_row"]
        is False
    )
    assert report["full_path"]["sustainable_unique_frame_profile_fps"] == 24


def test_full_path_rejects_mismatched_matrix_span_and_cycled_source_claim() -> None:
    wrong_span = build_performance_report(
        _rows(warmup=40, measured=400),
        full_path_evidence=_full_path(warmup=30, measured=410),
    )
    cycled_source = build_performance_report(
        _rows(source_frame_count=2),
        full_path_evidence=_full_path(),
    )

    assert (
        wrong_span["full_path"]["checks"]["exact_matrix_warmup_and_measured_span"]
        is False
    )
    assert wrong_span["full_path"]["outcome"] == "rejected"
    assert (
        cycled_source["full_path"]["checks"][
            "distinct_source_frame_per_full_path_invocation"
        ]
        is False
    )
    assert cycled_source["full_path"]["sustainable_unique_frame_profile_fps"] is None
    assert cycled_source["full_path"]["outcome"] == "rejected"


def test_full_path_rejects_tampered_cycle_lateness_and_background_scope() -> None:
    bad_lateness = _full_path()
    bad_lateness["schedule_lateness_samples_ms"][0] = 0.5
    with pytest.raises(MattePerformanceError, match="schedule lateness"):
        build_performance_report(_rows(), full_path_evidence=bad_lateness)

    bad_cycle = _full_path()
    bad_cycle["serialized_new_frame_samples_ms"] = [40.0] * 300
    bad_cycle["pacing_wait_samples_ms"] = [5.0] * 300
    bad_cycle["serialized_new_frame_deadline_miss_count"] = 300
    with pytest.raises(MattePerformanceError, match="timing boundaries"):
        build_performance_report(_rows(), full_path_evidence=bad_cycle)

    bad_background = _full_path()
    bad_background["background_provider_scope"] = "video-decoder"
    with pytest.raises(MattePerformanceError, match="background provider scope"):
        build_performance_report(_rows(), full_path_evidence=bad_background)


def test_raw_samples_counts_nesting_and_repeat_events_stay_separate() -> None:
    missing_stage = _rows()
    missing_stage[0]["timing_samples_ms"].pop("edge_band")
    with pytest.raises(MattePerformanceError, match="strict version-1 schema"):
        build_performance_report(missing_stage)

    contradictory = _rows()
    contradictory[0]["timing_samples_ms"]["edge_band"] = [21.0] * 300
    with pytest.raises(MattePerformanceError, match="substages exceed"):
        build_performance_report(contradictory)

    repeated = build_performance_report(
        _rows(),
        full_path_evidence=_full_path(
            output_repeat_count=2,
            no_unread_repeat_count=1,
        ),
    )
    evidence = repeated["full_path"]["evidence"]
    assert evidence["unique_composite_count"] == 300
    assert evidence["output_send_count"] == 302
    assert evidence["output_repeat_count"] == 2
    assert evidence["no_unread_repeat_count"] == 1
    assert repeated["decision"]["output_repeats_credited_as_unique_work"] is False
    assert repeated["full_path"]["outcome"] == "qualified"

    contradictory_repeats = _full_path(
        output_repeat_count=1,
        no_unread_repeat_count=2,
    )
    with pytest.raises(MattePerformanceError, match="repeat counters"):
        build_performance_report(_rows(), full_path_evidence=contradictory_repeats)


def test_full_path_requires_exact_measured_lineage_and_narrow_processing() -> None:
    wrong_lineage = build_performance_report(
        _rows(),
        full_path_evidence=_full_path(
            measured_frame_lineage_sha256="f" * 64,
        ),
    )
    assert wrong_lineage["full_path"]["outcome"] == "rejected"
    assert (
        wrong_lineage["full_path"]["checks"][
            "same_fixed_source_and_qualified_matrix_row"
        ]
        is False
    )
    assert wrong_lineage["full_path"]["sustainable_unique_frame_profile_fps"] is None

    invalid_processing = _full_path()
    invalid_processing["frame_processing_samples_ms"] = [18.0] * 300
    with pytest.raises(MattePerformanceError, match="timing boundaries"):
        build_performance_report(_rows(), full_path_evidence=invalid_processing)

    invalid_serialized = _full_path()
    invalid_serialized["serialized_new_frame_samples_ms"] = [30.0] * 300
    invalid_serialized["serialized_new_frame_deadline_miss_count"] = 0
    with pytest.raises(MattePerformanceError, match="timing boundaries"):
        build_performance_report(_rows(), full_path_evidence=invalid_serialized)


def test_report_validator_detects_edited_summary_or_digest() -> None:
    report = build_performance_report(_rows(), full_path_evidence=_full_path())
    edited = copy.deepcopy(report)
    edited["matrix"]["rows"][0]["timing_summary_ms"]["total"]["p95"] = 0.0

    with pytest.raises(MattePerformanceError, match="does not match"):
        validate_performance_report(edited)

    edited = copy.deepcopy(report)
    edited["evidence_sha256"] = "0" * 64
    with pytest.raises(MattePerformanceError, match="does not match"):
        validate_performance_report(edited)


def test_fixed_profiler_rejects_non_720p_and_runs_unpaced_proxy_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_frame = np.zeros((72, 128, 3), np.uint8)
    small_mask = np.ones((72, 128), np.float32)
    with pytest.raises(MattePerformanceError, match="1280x720"):
        profile_fixed_720p_compositor(
            small_frame,
            small_frame,
            small_mask,
            small_frame,
            measured_frame_count=1,
        )
    valid_frame = np.zeros((720, 1280, 3), np.uint8)
    valid_mask = np.ones((720, 1280), np.float32)
    valid_mask[0] = 0.0
    with pytest.raises(MattePerformanceError, match="cannot qualify"):
        profile_fixed_720p_compositor(
            valid_frame,
            valid_frame,
            valid_mask,
            valid_frame,
            warmup_frame_count=0,
            measured_frame_count=1,
            evidence_kind="fixed-replay-measured",
        )

    foreground = np.full((720, 1280, 3), 120, np.uint8)
    backdrop = np.full((720, 1280, 3), 20, np.uint8)
    clean = np.full((720, 1280, 3), 140, np.uint8)
    # RVM sigmoid output need not contain exact endpoints. The profiler must
    # keep this measured mask unchanged and use a derived endpoint-proof copy.
    mask = np.full((720, 1280), 0.5, np.float32)
    production_calls = 0
    reference_calls = 0
    production_modes: dict[str, list[bool]] = {
        variant.id: [] for variant in COMPOSITOR_VARIANTS
    }

    def fake_production(
        sample: CompositorFrameSample,
        variant: performance.CompositorVariant,
        *,
        diagnostics: dict[str, float] | None = None,
        **_kwargs: object,
    ) -> np.ndarray:
        nonlocal production_calls
        production_calls += 1
        production_modes[variant.id].append(diagnostics is not None)
        if diagnostics is not None:
            diagnostics.update({name: 0.01 for name in COMPOSITOR_SUBSTAGE_NAMES})
        output = sample.foreground_bgr.copy()
        output[sample.mask == 0.0] = sample.backdrop_bgr[sample.mask == 0.0]
        return output

    def fake_reference(
        sample: CompositorFrameSample,
        _variant: performance.CompositorVariant,
        **_kwargs: object,
    ) -> np.ndarray:
        nonlocal reference_calls
        reference_calls += 1
        output = sample.foreground_bgr.copy()
        output[sample.mask == 0.0] = sample.backdrop_bgr[sample.mask == 0.0]
        return output

    ticks = itertools.count(start=0, step=1_000_000)
    monkeypatch.setattr(performance, "_production_composite", fake_production)
    monkeypatch.setattr(performance, "_reference_composite", fake_reference)
    monkeypatch.setattr(
        performance,
        "_prepare_profile_wrap",
        lambda *_args, **_kwargs: None,
    )
    report = profile_fixed_720p_compositor(
        foreground,
        backdrop,
        mask,
        clean,
        warmup_frame_count=1,
        measured_frame_count=2,
        evidence_kind="generated-proxy",
        clock_ns=lambda: next(ticks),
    )

    assert production_calls == len(COMPOSITOR_VARIANTS) * 9
    assert reference_calls == len(COMPOSITOR_VARIANTS) * 4
    assert all(modes == [False] * 7 + [True] * 2 for modes in production_modes.values())
    assert report["scope"] == {
        "width": 1280,
        "height": 720,
        "nominal_fps": 30,
        "fixed_replay": True,
        "reactions_enabled": False,
        "source": {
            "kind": "single-frame",
            "source_frame_count": 1,
            "profiled_frame_count": 3,
            "unique_capture_sequence": False,
            "privacy_guarded_bundle": False,
            "bundle_manifest_sha256": None,
            "measured_frame_lineage_sha256": report["scope"]["source"][
                "measured_frame_lineage_sha256"
            ],
            "execution_pacing": "unpaced-tight-loop",
        },
        "source_sha256": report["matrix"]["source_sha256"],
    }
    assert report["matrix"]["completed_row_count"] == 8
    assert report["matrix"]["outcome"] == "not_decidable"
    assert report["decision"]["outcome"] == "not_decidable"
    assert all(
        row["timing_summary_ms"]["total"]["p95"] == 1.0
        for row in report["matrix"]["rows"]
    )
    assert all(
        row["memory_bandwidth"]
        == {
            "available": False,
            "bytes_per_second": None,
            "counter_source": None,
            "source_sha256": None,
            "hardware_identity_sha256": None,
            "provider_environment_sha256": None,
            "measurement_run_sha256": None,
        }
        and row["allocation_measurement_scope"]
        == "known-lower-bound-not-native-memory-bandwidth"
        for row in report["matrix"]["rows"]
    )


def test_ordered_two_frame_sequence_may_cycle_but_short_run_does_not_qualify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = np.full((720, 1280, 3), 80, np.uint8)
    second = np.full((720, 1280, 3), 90, np.uint8)
    backdrop = np.full((720, 1280, 3), 10, np.uint8)
    clean = np.full((720, 1280, 3), 100, np.uint8)
    mask = np.ones((720, 1280), np.float32)
    mask[0] = 0.0
    samples = [
        CompositorFrameSample(first, backdrop, mask, clean, 4, 1_000_000),
        CompositorFrameSample(second, backdrop, mask, clean, 5, 34_333_333),
    ]

    def render(
        sample: CompositorFrameSample,
        _variant: performance.CompositorVariant,
        *,
        diagnostics: dict[str, float] | None = None,
        **_kwargs: object,
    ) -> np.ndarray:
        if diagnostics is not None:
            diagnostics.update({name: 0.01 for name in COMPOSITOR_SUBSTAGE_NAMES})
        output = sample.foreground_bgr.copy()
        output[sample.mask == 0.0] = sample.backdrop_bgr[sample.mask == 0.0]
        return output

    monkeypatch.setattr(performance, "_production_composite", render)
    monkeypatch.setattr(performance, "_reference_composite", render)
    monkeypatch.setattr(
        performance,
        "_prepare_profile_wrap",
        lambda *_args, **_kwargs: None,
    )
    ticks = itertools.count(start=0, step=1_000_000)
    report = profile_fixed_720p_compositor_samples(
        samples,
        warmup_frame_count=1,
        measured_frame_count=2,
        clock_ns=lambda: next(ticks),
    )

    assert report["scope"]["source"]["source_frame_count"] == 2
    assert report["scope"]["source"]["profiled_frame_count"] == 3
    assert all(
        row["checks"]["ordered_multi_frame_source_sequence"] is True
        for row in report["matrix"]["rows"]
    )
    assert report["matrix"]["outcome"] == "not_decidable"


def test_production_linear_lane_and_wrap_preparation_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreground = np.full((2, 2, 3), 80, np.uint8)
    backdrop = np.full((2, 2, 3), 10, np.uint8)
    clean = np.full((2, 2, 3), 100, np.uint8)
    mask = np.asarray([[0.0, 0.5], [1.0, 0.5]], np.float32)
    sample = CompositorFrameSample(foreground, backdrop, mask, clean, 0, 0)
    variant = next(row for row in COMPOSITOR_VARIANTS if row.id == "linear_srgb_both")
    linear_inputs = (
        np.zeros_like(foreground, dtype=np.float32),
        np.zeros_like(backdrop, dtype=np.float32),
        np.zeros_like(clean, dtype=np.float32),
    )
    accelerated_calls = 0
    wrap_calls = 0

    def accelerated(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal accelerated_calls
        accelerated_calls += 1
        return foreground.copy()

    sentinel = object()

    def prepare(*_args: object, **kwargs: object) -> object:
        nonlocal wrap_calls
        wrap_calls += 1
        assert isinstance(kwargs["stabilizer"], performance.LightWrapStabilizer)
        assert isinstance(kwargs["context"], performance.LightWrapFrameContext)
        assert kwargs["backdrop_linear_bgr"] is linear_inputs[1]
        return sentinel

    monkeypatch.setattr(
        performance,
        "_composite_linear_bgr_prevalidated",
        accelerated,
    )
    monkeypatch.setattr(performance, "prepare_light_wrap", prepare)
    prepared = performance._prepare_profile_wrap(
        sample,
        variant,
        ordinal=7,
        source_sha256=SOURCE_SHA256,
        stabilizer=performance.LightWrapStabilizer(0.12),
        backdrop_linear_bgr=linear_inputs[1],
        diagnostics={},
    )
    output = performance._production_composite(
        sample,
        variant,
        linear_inputs=linear_inputs,
        prepared_light_wrap=prepared,
        workspace=None,
        diagnostics={},
    )

    assert wrap_calls == 1
    assert accelerated_calls == 1
    assert np.array_equal(output, foreground)


def test_balanced_matrix_keeps_default_light_wrap_stabilization_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreground = np.full((720, 1280, 3), 80, np.uint8)
    backdrop = np.full((720, 1280, 3), 10, np.uint8)
    clean = np.full((720, 1280, 3), 100, np.uint8)
    mask = np.full((720, 1280), 0.5, np.float32)

    def forbidden_stabilized_prepare(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("balanced compatibility profile must remain stateless")

    monkeypatch.setattr(
        performance,
        "prepare_light_wrap",
        forbidden_stabilized_prepare,
    )

    report = profile_fixed_720p_compositor(
        foreground,
        backdrop,
        mask,
        clean,
        warmup_frame_count=0,
        measured_frame_count=1,
    )

    assert report["decision"]["outcome"] == "not_decidable"
    assert all(
        row["timing_samples_ms"]["light_wrap_temporal_filter"] == [0.0]
        for row in report["matrix"]["rows"]
    )


def test_native_memory_bandwidth_is_never_inferred_from_allocations() -> None:
    rows = _rows()
    rows[0]["memory_bandwidth"] = {
        "available": True,
        "bytes_per_second": 123_000_000.0,
        "counter_source": "native_counter",
        "source_sha256": SOURCE_SHA256,
        "hardware_identity_sha256": "b" * 64,
        "provider_environment_sha256": "c" * 64,
        "measurement_run_sha256": "9" * 64,
    }
    report = build_performance_report(rows)

    assert report["matrix"]["rows"][0]["memory_bandwidth"] == {
        "available": True,
        "bytes_per_second": 123_000_000.0,
        "counter_source": "native_counter",
        "source_sha256": SOURCE_SHA256,
        "hardware_identity_sha256": "b" * 64,
        "provider_environment_sha256": "c" * 64,
        "measurement_run_sha256": "9" * 64,
    }
    assert report["matrix"]["rows"][1]["memory_bandwidth"]["available"] is False


def test_native_memory_bandwidth_is_bound_to_source_and_full_path_run() -> None:
    rows = _rows()
    measured_bandwidth = {
        "available": True,
        "bytes_per_second": 123_000_000.0,
        "counter_source": "native_counter",
        "source_sha256": SOURCE_SHA256,
        "hardware_identity_sha256": "b" * 64,
        "provider_environment_sha256": "c" * 64,
        "measurement_run_sha256": "9" * 64,
    }
    for row in rows:
        row["memory_bandwidth"] = measured_bandwidth.copy()

    report = build_performance_report(
        rows,
        full_path_evidence=_full_path(),
    )
    assert report["decision"]["outcome"] == "qualified"

    wrong_run = _full_path()
    wrong_run["measurement_run_sha256"] = "8" * 64
    with pytest.raises(MattePerformanceError, match="full-path measurement run"):
        build_performance_report(rows, full_path_evidence=wrong_run)

    wrong_source = _rows()
    wrong_source[0]["memory_bandwidth"] = {
        **measured_bandwidth,
        "source_sha256": "f" * 64,
    }
    with pytest.raises(MattePerformanceError, match="matrix source"):
        build_performance_report(wrong_source)


def test_markdown_keeps_headroom_and_repeat_truth_explicit() -> None:
    report = build_performance_report(_rows(), full_path_evidence=_full_path())
    text = report_markdown(report)

    assert "signed 22 ms headroom" in text
    assert "Full-path signed 33.333334 ms headroom" in text
    assert "Full-path signed 22 ms compositor headroom" in text
    assert "Output repeats credited as unique work: `false`" in text
    assert "30/27/24/20/15 FPS" in text
    assert "not `1000 / p95`" in text


def test_checked_in_full_path_template_matches_the_strict_schema() -> None:
    template_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "qualification"
        / "matte-performance-local-template.json"
    )
    template = cast(
        dict[str, Any],
        json.loads(template_path.read_text(encoding="utf-8")),
    )
    template.update(
        {
            "source_sha256": SOURCE_SHA256,
            "bundle_manifest_sha256": None,
            "measured_frame_lineage_sha256": LINEAGE_SHA256,
        }
    )

    report = build_performance_report(
        _rows(measured=3, warmup=1, evidence_kind="generated-proxy"),
        full_path_evidence=template,
    )

    assert report["full_path"]["status"] == "evaluated"
    assert report["full_path"]["outcome"] == "not_decidable"


def test_qualification_identity_binds_exact_cuda_device_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = performance._ReplaySource(  # noqa: SLF001
        kind="privacy-replay-bundle",
        frame_count=1,
        source_sha256=SOURCE_SHA256,
        bundle_manifest_sha256="f" * 64,
        lineage=((1, 1, 1, 1),),
        load=lambda _index: pytest.fail("source pixels must not load"),
        load_model_frame=lambda _index: pytest.fail("source pixels must not load"),
    )
    ort = types.SimpleNamespace(
        __version__="1.22.0",
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        get_device=lambda: "GPU",
        get_build_info=lambda: "build exact",
    )
    monkeypatch.setattr(
        performance.importlib,
        "import_module",
        lambda name: ort if name == "onnxruntime" else pytest.fail(name),
    )
    segmenter = types.SimpleNamespace(
        rvm_telemetry_snapshot=lambda: types.SimpleNamespace(
            acceleration_active_provider="cuda"
        )
    )
    output = types.SimpleNamespace(
        width=1280,
        height=720,
        fps=30,
        cam=types.SimpleNamespace(device="test sink"),
    )
    cuda_identity: dict[str, object] = {
        "identity_source": "custback-cuda-driver-runtime-nvml-v1",
        "ordinal": 0,
        "uuid": "GPU-00112233-4455-6677-8899-aabbccddeeff",
        "uuid_kind": "gpu",
        "pci_bus_id": "00000000:65:00.0",
        "cuda_name": "NVIDIA Test GPU",
        "nvml_name": "NVIDIA Test GPU",
        "cuda_total_memory_bytes": 24 * 1024**3,
        "nvml_total_memory_bytes": 24 * 1024**3,
        "compute_capability_major": 8,
        "compute_capability_minor": 9,
        "cuda_driver_version": 12040,
        "cuda_runtime_version": 12030,
        "nvidia_driver_version": "595.84",
        "nvml_version": "13.595.84",
    }
    monkeypatch.setattr(
        performance,
        "collect_cuda_device_identity",
        lambda _device_id: dict(cuda_identity),
    )

    first = performance._qualification_identity(  # noqa: SLF001
        segmenter=segmenter,
        output=output,
        source=source,
        hardware_id="cuda_lab",
        device_id=0,
        sink_backend="pyvirtualcam",
        run_started_ns=123,
    )
    cuda_identity["uuid"] = "GPU-ffeeddcc-bbaa-9988-7766-554433221100"
    changed_device = performance._qualification_identity(  # noqa: SLF001
        segmenter=segmenter,
        output=output,
        source=source,
        hardware_id="cuda_lab",
        device_id=0,
        sink_backend="pyvirtualcam",
        run_started_ns=123,
    )

    assert first[0] != changed_device[0]
    assert first[1] == changed_device[1]
    assert first[2] == changed_device[2]
    assert first[3] != changed_device[3]
    assert all(performance._DIGEST.fullmatch(value) for value in first)  # noqa: SLF001


def test_qualification_identity_redacts_cuda_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ort = types.SimpleNamespace(
        __version__="1.22.0",
        get_available_providers=lambda: ["CUDAExecutionProvider"],
        get_device=lambda: "GPU",
        get_build_info=lambda: "build exact",
    )
    monkeypatch.setattr(performance.importlib, "import_module", lambda _name: ort)
    monkeypatch.setattr(
        performance,
        "collect_cuda_device_identity",
        lambda _device_id: (_ for _ in ()).throw(
            RuntimeError("/private/driver/path and host details")
        ),
    )

    with pytest.raises(
        MattePerformanceError,
        match=r"^exact CUDA hardware identity is unavailable for full-path evidence$",
    ):
        performance._qualification_identity(  # noqa: SLF001
            segmenter=object(),
            output=object(),
            source=cast(Any, object()),
            hardware_id="cuda_lab",
            device_id=0,
            sink_backend="pyvirtualcam",
            run_started_ns=123,
        )


@pytest.mark.parametrize(
    ("negotiated_width", "negotiated_height", "negotiated_fps"),
    (
        (1279, 720, 30),
        (1280, 719, 30),
        (1280, 720, 29),
    ),
)
def test_open_qualification_sink_rejects_negotiated_mode_mismatch_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    negotiated_width: int,
    negotiated_height: int,
    negotiated_fps: int,
) -> None:
    import custback.vcam as vcam

    class FakeOutput:
        fallback_active = False

        def __init__(self) -> None:
            self.width = negotiated_width
            self.height = negotiated_height
            self.fps = negotiated_fps
            self.closed = False

        def close(self) -> None:
            self.closed = True

    output = FakeOutput()

    def open_output(config: object, width: int, height: int) -> FakeOutput:
        assert getattr(config, "backend") == "native"
        assert getattr(config, "fps") == 30
        assert (width, height) == (1280, 720)
        return output

    monkeypatch.setattr(vcam, "open_output", open_output)

    with pytest.raises(MattePerformanceError, match="exact 1280x720@30"):
        performance._open_qualification_sink("native")  # noqa: SLF001

    assert output.closed is True


def test_model_backed_collector_runs_distinct_frames_unpaced_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from custback.matte_policy import MatteBackendKind
    from custback.segmentation import (
        RVMTelemetry,
        SegmentationFrameContext,
        Segmenter,
    )
    from custback.vcam import OutputSendTiming

    foreground = np.full((720, 1280, 3), 96, np.uint8)
    backdrop = np.full((720, 1280, 3), 24, np.uint8)
    mask = np.full((720, 1280), 0.5, np.float32)
    clean = np.full((720, 1280, 3), 112, np.uint8)
    sequences = (10, 11, 15)
    samples = tuple(
        CompositorFrameSample(
            foreground,
            backdrop,
            mask,
            clean,
            sequences[index],
            1_000_000_000 + index * 33_333_333,
        )
        for index in range(3)
    )
    source = performance._ReplaySource(  # noqa: SLF001
        kind="privacy-replay-bundle",
        frame_count=3,
        source_sha256=SOURCE_SHA256,
        bundle_manifest_sha256="f" * 64,
        lineage=tuple(
            (
                sample.capture_sequence,
                sample.capture_timestamp_ns,
                1,
                1,
            )
            for sample in samples
        ),
        load=lambda index: samples[index],
        load_model_frame=lambda index: performance._ModelReplayFrame(  # noqa: SLF001
            foreground_bgr=samples[index].foreground_bgr,
            backdrop_bgr=samples[index].backdrop_bgr,
            capture_sequence=samples[index].capture_sequence,
            capture_timestamp_ns=samples[index].capture_timestamp_ns,
            capture_generation=1,
            geometry_generation=1,
        ),
    )

    class FakeRVM(Segmenter):
        produces_matte = True
        matte_backend_kind = MatteBackendKind.TRUE_ALPHA_RECURRENT
        last_downsample_ratio = 0.4

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.closed = False
            self.telemetry = RVMTelemetry(
                input_frame_shape=(720, 1280),
                output_alpha_shape=(720, 1280),
                output_foreground_shape=(720, 1280, 3),
                configured_downsample_mode="auto",
                configured_downsample_ratio=0.0,
                resolved_downsample_ratio=0.4,
                preprocess_ms=0.1,
                session_run_ms=0.2,
                postprocess_ms=0.1,
                model_builtin=True,
                model_identity="builtin-rvm",
                model_sha256=performance.RVM_MODEL.sha256,
                model_bytes=performance.RVM_MODEL.size,
                acceleration_state="gpu-required",
                acceleration_active_provider="cuda",
                acceleration_fallback_active=False,
                acceleration_fallback_count=0,
            )

        def segment(
            self,
            frame_bgr: np.ndarray,
            *,
            context: SegmentationFrameContext | None = None,
        ) -> np.ndarray:
            self._accept_frame_context(context, frame_bgr.shape[:2])
            self.calls += 1
            self.last_foreground = frame_bgr.copy()
            return np.full(frame_bgr.shape[:2], 0.5, np.float32)

        def rvm_telemetry_snapshot(self) -> RVMTelemetry:
            return self.telemetry

        def close(self) -> None:
            self.closed = True

    class FakeOutput:
        paces = False

        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def send_with_timing(self, frame: np.ndarray) -> OutputSendTiming:
            assert frame.shape == (720, 1280, 3)
            started_ns = performance.time.monotonic_ns()
            self.calls += 1
            completed_ns = max(started_ns, performance.time.monotonic_ns())
            return OutputSendTiming(
                submitted_at_ns=completed_ns,
                completed_at_ns=completed_ns,
                submission_ms=(completed_ns - started_ns) / 1_000_000.0,
                pacing_wait_ms=0.0,
            )

        def close(self) -> None:
            self.closed = True

    segmenter = FakeRVM()
    output = FakeOutput()
    monkeypatch.setattr(performance, "_private_bundle_source", lambda _root: source)
    monkeypatch.setattr(
        performance,
        "_open_balanced_rvm_segmenter",
        lambda _device: segmenter,
    )
    monkeypatch.setattr(
        performance,
        "_open_qualification_sink",
        lambda _backend: output,
    )
    monkeypatch.setattr(
        performance,
        "_qualification_identity",
        lambda **_kwargs: ("b" * 64, "c" * 64, "d" * 64, "9" * 64),
    )

    evidence = performance.collect_private_720p_full_path_evidence(
        "private-bundle",
        warmup_frame_count=1,
        measured_frame_count=2,
        hardware_id="cuda_host",
        sink_backend="native",
    )

    assert evidence["schema"] == FULL_PATH_SCHEMA
    assert evidence["evidence_kind"] == "model-backed"
    assert evidence["source_sha256"] == SOURCE_SHA256
    assert evidence["bundle_manifest_sha256"] == "f" * 64
    assert evidence["matrix_row_id"] == "srgb_legacy_both"
    assert evidence["effective_policy"] == performance._BALANCED_EFFECTIVE_POLICY  # noqa: SLF001
    assert evidence["warmup_frame_count"] == 1
    assert evidence["unique_composite_count"] == 2
    assert evidence["model_invocation_count"] == 2
    assert evidence["output_send_count"] == 2
    assert evidence["background_provider_scope"] == "resident-recorded-frame-copy"
    assert evidence["capture_sequence_gap_count"] == 1
    assert evidence["capture_missing_input_count"] == 3
    assert all(len(evidence[name]) == 2 for name in performance.FULL_PATH_SAMPLE_NAMES)
    assert evidence["pacing_wait_samples_ms"] == [0.0, 0.0]
    assert segmenter.calls == 3
    assert output.calls == 3
    assert segmenter.closed is True
    assert output.closed is True


def test_model_backed_collector_closes_resources_when_sink_open_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = performance._ReplaySource(  # noqa: SLF001
        kind="privacy-replay-bundle",
        frame_count=1,
        source_sha256=SOURCE_SHA256,
        bundle_manifest_sha256="f" * 64,
        lineage=((1, 1, 1, 1),),
        load=lambda _index: pytest.fail("source frame must not load"),
        load_model_frame=lambda _index: pytest.fail("source model frame must not load"),
    )

    class FakeSegmenter:
        closed = False

        def close(self) -> None:
            self.closed = True

    segmenter = FakeSegmenter()
    monkeypatch.setattr(performance, "_private_bundle_source", lambda _root: source)
    monkeypatch.setattr(
        performance,
        "_open_balanced_rvm_segmenter",
        lambda _device: segmenter,
    )
    monkeypatch.setattr(
        performance,
        "_open_qualification_sink",
        lambda _backend: (_ for _ in ()).throw(RuntimeError("sink unavailable")),
    )
    with pytest.raises(RuntimeError, match="sink unavailable"):
        performance.collect_private_720p_full_path_evidence(
            "private-bundle",
            warmup_frame_count=0,
            measured_frame_count=1,
        )

    assert segmenter.closed is True


def test_model_backed_collector_rejects_resident_replay_over_memory_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_pair_bytes = performance.FIXED_HEIGHT * performance.FIXED_WIDTH * 3 * 2
    measured = performance.MAX_FULL_PATH_PRELOAD_BYTES // frame_pair_bytes + 1
    source = performance._ReplaySource(  # noqa: SLF001
        kind="privacy-replay-bundle",
        frame_count=measured,
        source_sha256=SOURCE_SHA256,
        bundle_manifest_sha256="f" * 64,
        lineage=tuple((index, index + 1, 1, 1) for index in range(measured)),
        load=lambda _index: pytest.fail("matrix source frame must not load"),
        load_model_frame=lambda _index: pytest.fail(
            "oversized resident frame must not load"
        ),
    )
    monkeypatch.setattr(performance, "_private_bundle_source", lambda _root: source)
    monkeypatch.setattr(
        performance,
        "_open_balanced_rvm_segmenter",
        lambda _device: pytest.fail("segmenter must not open"),
    )

    with pytest.raises(MattePerformanceError, match="fixed memory bound"):
        performance.collect_private_720p_full_path_evidence(
            "private-bundle",
            warmup_frame_count=0,
            measured_frame_count=measured,
        )


def test_core_cli_dispatches_matte_performance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(
        performance,
        "main",
        lambda argv, *, prog: calls.append((argv, prog)) or 7,
    )

    result = core_main.main(
        ["matte-performance", "private-bundle", "--output", "new-report"]
    )

    assert result == 7
    assert calls == [
        (
            ["private-bundle", "--output", "new-report"],
            "custback matte-performance",
        )
    ]


@pytest.mark.parametrize(
    ("outcome", "expected_exit"),
    (("qualified", 0), ("not_decidable", 1), ("rejected", 1)),
)
def test_performance_cli_exit_code_tracks_fail_closed_decision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
    expected_exit: int,
) -> None:
    calls: list[tuple[object, ...]] = []

    def run(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((*args, kwargs))
        return {
            "decision": {"outcome": outcome},
            "matrix": {"completed_row_count": 8},
        }

    monkeypatch.setattr(performance, "run_private_720p_profile", run)

    result = performance.main(
        [
            "private-bundle",
            "--output",
            "new-report",
            "--warmup-frames",
            "31",
            "--measured-frames",
            "301",
            "--full-path-evidence",
            "full-path.json",
        ]
    )

    assert result == expected_exit
    assert calls == [
        (
            "private-bundle",
            "new-report",
            {
                "warmup_frame_count": 31,
                "measured_frame_count": 301,
                "full_path_evidence_path": "full-path.json",
                "native_memory_bandwidth_path": None,
                "collect_full_path": False,
                "hardware_id": "local_cuda",
                "cuda_device_id": 0,
                "sink_backend": "pyvirtualcam",
            },
        )
    ]
    assert f"outcome {outcome}" in capsys.readouterr().out


def test_performance_cli_dispatches_model_backed_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def run(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((*args, kwargs))
        return {
            "decision": {"outcome": "qualified"},
            "matrix": {"completed_row_count": 8},
        }

    monkeypatch.setattr(performance, "run_private_720p_profile", run)

    result = performance.main(
        [
            "private-bundle",
            "--output",
            "new-report",
            "--collect-full-path",
            "--hardware-id",
            "cuda_lab",
            "--cuda-device-id",
            "2",
            "--sink-backend",
            "native",
        ]
    )

    assert result == 0
    assert calls == [
        (
            "private-bundle",
            "new-report",
            {
                "warmup_frame_count": 30,
                "measured_frame_count": 300,
                "full_path_evidence_path": None,
                "native_memory_bandwidth_path": None,
                "collect_full_path": True,
                "hardware_id": "cuda_lab",
                "cuda_device_id": 2,
                "sink_backend": "native",
            },
        )
    ]


def test_performance_cli_redacts_runtime_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        performance,
        "run_private_720p_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("/private/provider/path and device serial")
        ),
    )

    result = performance.main(["private-bundle", "--output", "new-report"])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == (
        "custback matte-performance: full-path performance collection failed\n"
    )


def test_performance_output_cannot_overlap_private_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(mode=0o700)

    with pytest.raises(MattePerformanceError, match="must not overlap"):
        performance.run_private_720p_profile(
            bundle,
            bundle / "report",
            warmup_frame_count=1,
            measured_frame_count=1,
        )


def test_performance_output_publishes_one_complete_private_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(mode=0o700)
    output = tmp_path / "report"
    expected = {"decision": {"outcome": "not_decidable"}}
    monkeypatch.setattr(
        performance,
        "profile_private_720p_replay_bundle",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(performance, "report_markdown", lambda _report: "# report\n")

    result = performance.run_private_720p_profile(
        bundle,
        output,
        warmup_frame_count=1,
        measured_frame_count=1,
    )

    assert result == expected
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "performance.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "performance.md").stat().st_mode) == 0o600
    assert (output / "performance.json").read_text(encoding="ascii").endswith("\n")
    assert (output / "performance.md").read_text(encoding="utf-8") == "# report\n"


def test_private_profile_joins_same_run_model_backed_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir(mode=0o700)
    output = tmp_path / "report"
    evidence = _full_path(measured=2, warmup=1)
    collector_calls: list[tuple[object, ...]] = []
    profile_calls: list[dict[str, object]] = []

    def collect(*args: object, **kwargs: object) -> dict[str, Any]:
        collector_calls.append((*args, kwargs))
        return evidence

    def profile(*_args: object, **kwargs: object) -> dict[str, object]:
        profile_calls.append(kwargs)
        return {"decision": {"outcome": "not_decidable"}}

    monkeypatch.setattr(
        performance,
        "collect_private_720p_full_path_evidence",
        collect,
    )
    monkeypatch.setattr(performance, "profile_private_720p_replay_bundle", profile)
    monkeypatch.setattr(performance, "report_markdown", lambda _report: "# report\n")

    performance.run_private_720p_profile(
        bundle,
        output,
        warmup_frame_count=1,
        measured_frame_count=2,
        collect_full_path=True,
        hardware_id="cuda_lab",
        cuda_device_id=2,
        sink_backend="native",
    )

    assert collector_calls == [
        (
            bundle,
            {
                "warmup_frame_count": 1,
                "measured_frame_count": 2,
                "hardware_id": "cuda_lab",
                "cuda_device_id": 2,
                "sink_backend": "native",
            },
        )
    ]
    assert profile_calls[0]["full_path_evidence"] is evidence


def test_performance_output_failure_leaves_no_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import custback.matte_diagnostics as diagnostics

    bundle = tmp_path / "bundle"
    bundle.mkdir(mode=0o700)
    output = tmp_path / "report"
    monkeypatch.setattr(
        performance,
        "profile_private_720p_replay_bundle",
        lambda *_args, **_kwargs: {"decision": {"outcome": "not_decidable"}},
    )
    monkeypatch.setattr(performance, "report_markdown", lambda _report: "# report\n")
    original_write = diagnostics._atomic_private_write
    writes = 0

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected report publication failure")
        original_write(path, payload)

    monkeypatch.setattr(diagnostics, "_atomic_private_write", fail_second_write)

    with pytest.raises(OSError, match="injected"):
        performance.run_private_720p_profile(
            bundle,
            output,
            warmup_frame_count=1,
            measured_frame_count=1,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".custback-matte-performance-*"))
