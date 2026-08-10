#!/usr/bin/env python3
"""Generated MATTE-5.3 plan fixtures with no qualification authority.

The objects built here exercise the strict platform/profile matrix parser.  No
camera, model, provider, output device, or physical consumer is opened.  Every
matrix cell therefore starts as explicitly unavailable and can never authorize
a platform, profile, preset, or default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from custback import matte_performance as performance
from custback import matte_platform_qualification as qualification


@dataclass(frozen=True)
class GeneratedPlatformQualification:
    """Paths making up one owner-only generated matrix fixture."""

    root: Path
    plan: Path
    prerequisites: dict[str, Path]
    runs: dict[str, Path]
    captures: dict[str, Path]


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: object) -> str:
    """Return the canonical digest used by the platform evidence contract."""

    return sha256_bytes(json_bytes(value))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    path.write_bytes(json_bytes(value))
    path.chmod(0o600)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    assert isinstance(value, dict)
    return value


def _digest_report(schema: str, status: str = "pending") -> dict[str, object]:
    report: dict[str, object] = {
        "schema": schema,
        "version": 1,
        "status": status,
        "authority": "generated-schema-proxy",
    }
    report["evidence_sha256"] = sha256_bytes(json_bytes(report))
    return report


def _prerequisite_descriptor(root: Path, path: Path) -> dict[str, str]:
    report = read_json(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "evidence_sha256": str(report["evidence_sha256"]),
    }


def artifact_descriptor(root: Path, path: Path) -> dict[str, str]:
    """Describe one deterministic JSON artifact relative to its private root."""

    report = read_json(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "evidence_sha256": str(report["evidence_sha256"]),
    }


def file_descriptor(root: Path, path: Path) -> dict[str, str]:
    """Describe an official capture report, which has no internal digest."""

    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def sign_report(report: dict[str, object]) -> dict[str, object]:
    """Attach the canonical content digest used by qualification reports."""

    report = dict(report)
    report.pop("evidence_sha256", None)
    report["evidence_sha256"] = sha256_bytes(json_bytes(report))
    return report


def _cell(
    cell_id: str,
    route_id: str,
    profile_id: str,
    provider: str,
    dependency_profile: str,
) -> dict[str, object]:
    return {
        "id": cell_id,
        "route_id": route_id,
        "profile_id": profile_id,
        "provider": provider,
        "canvas": {"width": 1280, "height": 720},
        "target_fps": qualification.TARGET_FPS,
        "dependency_profile": dependency_profile,
        "state": "unavailable",
        "reason": "generated-schema-proxy",
        "run": None,
        "capture_report": None,
    }


def generated_cells() -> list[dict[str, object]]:
    """Return the smallest matrix covering every required v1 dimension."""

    return [
        _cell(
            "linux-rvm-cuda",
            "linux-v4l2-pyvirtualcam",
            "rvm_matting",
            "cuda",
            "standard",
        ),
        _cell(
            "macos-rvm-cpu",
            "macos-obsvcam-pyvirtualcam",
            "rvm_matting",
            "cpu",
            "standard",
        ),
        _cell(
            "windows-msmf-rvm-directml",
            "windows-msmf-pyvirtualcam",
            "rvm_matting",
            "directml",
            "standard",
        ),
        _cell(
            "windows-dshow-mediapipe",
            "windows-dshow-pyvirtualcam",
            "mediapipe_segmentation",
            "cpu",
            "standard",
        ),
        _cell(
            "windows-native-heuristic",
            "windows-native",
            "heuristic_segmentation",
            "cpu",
            "standard",
        ),
        _cell(
            "linux-heuristic-without-mediapipe",
            "linux-v4l2-pyvirtualcam",
            "heuristic_segmentation",
            "cpu",
            "without_mediapipe",
        ),
        _cell(
            "macos-passthrough-without-gpu",
            "macos-obsvcam-pyvirtualcam",
            "none_passthrough",
            "cpu",
            "without_gpu_provider",
        ),
    ]


def generated_profiles(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for profile_id, contract in qualification.PROFILE_CONTRACTS.items():
        quality_claim = contract["quality_claim"] is True
        visual_contract = generated_visual_contract(profile_id)
        segmentation = visual_contract["segmentation"]
        configured_segmentation = dict(segmentation)
        configured_segmentation.pop("model_path_present", None)
        result.append(
            {
                "id": profile_id,
                "backend": contract["backend"],
                "segmenter": contract["segmenter"],
                "backend_kind": contract["backend_kind"],
                "quality_claim": contract["quality_claim"],
                "visual_candidate_id": (
                    f"{profile_id.replace('_', '-')}-candidate"
                    if quality_claim
                    else None
                ),
                "visual_algorithm_contract_sha256": (
                    canonical_digest(visual_contract) if quality_claim else None
                ),
                "configured_policy_sha256": (
                    canonical_digest(configured_segmentation)
                    if quality_claim
                    else sha256_bytes(f"{profile_id}:configured".encode())
                ),
                "effective_policy_sha256": sha256_bytes(
                    f"{profile_id}:effective".encode()
                ),
                "limits": [
                    str(cell["id"])
                    for cell in cells
                    if cell["profile_id"] == profile_id
                ],
            }
        )
    return result


def generated_visual_contract(profile_id: str) -> dict[str, dict[str, object]]:
    """Return the path-free visual policy bound by one quality profile."""

    if profile_id == "rvm_matting":
        segmentation: dict[str, object] = {
            "backend": "rvm",
            "model_path_present": False,
            "rvm_downsample": 0.0,
            "mask_shift": 0,
            "boundary_stabilization": {"mode": "off"},
        }
        compositing: dict[str, object] = {
            "use_model_foreground": True,
            "light_wrap": 0.25,
            "light_wrap_stabilization": {"mode": "off"},
            "blend_space": "srgb_legacy",
            "color_correction": {"mode": "off"},
        }
    elif profile_id == "mediapipe_segmentation":
        segmentation = {"backend": "mediapipe"}
        compositing = {}
    else:
        segmentation = {
            "backend": qualification.PROFILE_CONTRACTS[profile_id]["backend"]
        }
        compositing = {}
    return {"segmentation": segmentation, "compositing": compositing}


def generated_visual_prerequisite(
    profiles: list[dict[str, object]],
) -> dict[str, object]:
    """Return a digest-bound, pending MATTE-5.2 prerequisite proxy."""

    return sign_report(
        {
            "schema": qualification.VISUAL_REPORT_SCHEMA,
            "version": qualification.VISUAL_REPORT_VERSION,
            "status": "pending",
            "production": {
                "quality_preset_selected": False,
                "default_changed": False,
                "generated_evidence_can_qualify": False,
                "reactions_enabled": False,
            },
            "algorithm_manifest": [
                {
                    "id": profile["visual_candidate_id"],
                    "contract_sha256": profile["visual_algorithm_contract_sha256"],
                    **generated_visual_contract(str(profile["id"])),
                }
                for profile in profiles
                if profile["quality_claim"] is True
            ],
            "authority": "generated-schema-proxy",
        }
    )


def generated_fixed_replay_lineage(
    *, measured: int = qualification.MIN_MEASURED_FRAMES
) -> list[dict[str, int]]:
    """Return the measured identity lineage shared with MATTE-3.4."""

    return [
        {
            "capture_sequence": index,
            "capture_timestamp_ns": round(index * 1_000_000_000 / 30),
            "capture_generation": 1,
            "geometry_generation": 1,
        }
        for index in range(measured)
    ]


def generated_lineage_digest(
    *,
    warmup: int = qualification.MIN_WARMUP_FRAMES,
    measured: int = qualification.MIN_MEASURED_FRAMES,
) -> str:
    """Bind the generated fixed-replay source to exact measured identities."""

    return canonical_digest(
        {
            "schema": "custback.matte-performance-measured-lineage",
            "version": 1,
            "source_sha256": sha256_bytes(b"generated-performance-source"),
            "warmup_frame_count": warmup,
            "measured_frame_count": measured,
            "frames": generated_fixed_replay_lineage(measured=measured),
        }
    )


def _generated_performance_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in performance.COMPOSITOR_VARIANTS:
        measured = 300
        warmup = 30
        timings = {
            name: [0.0] * measured
            for name in ("total", *qualification.COMPOSITOR_SUBSTAGE_NAMES)
        }
        timings["total"] = [20.0] * measured
        timings["input_mask_validation"] = [0.25] * measured
        timings["final_blend_conversion"] = [0.75] * measured
        rows.append(
            {
                "schema": performance.ROW_SCHEMA,
                "version": performance.ROW_VERSION,
                "id": variant.id,
                "source_sha256": sha256_bytes(b"generated-performance-source"),
                "source_scope": {
                    "kind": "explicit-frame-sequence",
                    "source_frame_count": warmup + measured,
                    "profiled_frame_count": warmup + measured,
                    "unique_capture_sequence": True,
                    "privacy_guarded_bundle": False,
                    "bundle_manifest_sha256": None,
                    "measured_frame_lineage_sha256": generated_lineage_digest(
                        warmup=warmup, measured=measured
                    ),
                    "execution_pacing": "unpaced-tight-loop",
                },
                "evidence_kind": "generated-proxy",
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
                    "max_channel_delta": (
                        0 if variant.blend_space == "srgb_legacy" else 1
                    ),
                    "foreground_endpoint_exact": True,
                    "background_endpoint_exact": True,
                    "output_shape": [720, 1280, 3],
                    "output_dtype": "uint8",
                    "output_c_contiguous": True,
                    "deterministic_repeat_exact": True,
                },
            }
        )
    return rows


def generated_performance_prerequisite() -> dict[str, object]:
    """Build a validator-authoritative but non-decidable MATTE-3.4 report."""

    return performance.build_performance_report(_generated_performance_rows())


def generated_performance_with_full_path(
    effective_policy: dict[str, object],
) -> dict[str, object]:
    """Build valid generated-proxy full-path evidence for join mutations."""

    measured = 300
    service_ms = 30.0
    cycle_ms = 31.0
    serialized_ms = 37.0
    service_samples = [service_ms] * measured
    cycle_samples = [cycle_ms] * measured
    full_path: dict[str, object] = {
        "schema": performance.FULL_PATH_SCHEMA,
        "version": performance.FULL_PATH_VERSION,
        "evidence_kind": "generated-proxy",
        "source_sha256": sha256_bytes(b"generated-performance-source"),
        "hardware_id": "generated-cuda-host",
        "hardware_identity_sha256": sha256_bytes(b"generated-performance-hardware"),
        "provider_environment_sha256": sha256_bytes(b"generated-performance-provider"),
        "measurement_run_sha256": sha256_bytes(b"generated-performance-run"),
        "rvm_model_sha256": performance.RVM_MODEL.sha256,
        "bundle_manifest_sha256": None,
        "measured_frame_lineage_sha256": generated_lineage_digest(
            warmup=30, measured=measured
        ),
        "matrix_row_id": "srgb_legacy_both",
        "effective_policy": effective_policy,
        "background_provider_scope": "resident-recorded-frame-copy",
        "sink_id": "pyvirtualcam",
        "sink_identity_sha256": sha256_bytes(b"generated-performance-sink"),
        "sink_submission_copy_in_service_boundary": True,
        "canvas": {"width": 1280, "height": 720, "nominal_fps": 30},
        "backend": "rvm",
        "provider": "cuda",
        "fallback_count": 0,
        "boundary": "unique-dequeue-through-sink-submit-excluding-deliberate-pacing",
        "warmup_frame_count": 30,
        "service_samples_ms": service_samples,
        "non_pacing_cycle_samples_ms": cycle_samples,
        "frame_processing_samples_ms": [20.0] * measured,
        "serialized_new_frame_samples_ms": [serialized_ms] * measured,
        "rvm_preprocess_samples_ms": [2.0] * measured,
        "rvm_inference_samples_ms": [5.0] * measured,
        "rvm_postprocess_samples_ms": [2.0] * measured,
        "background_selection_samples_ms": [1.0] * measured,
        "compositor_samples_ms": [10.0] * measured,
        "post_composite_validation_samples_ms": [1.0] * measured,
        "sink_submission_samples_ms": [1.0] * measured,
        "pacing_wait_samples_ms": [7.0] * measured,
        "schedule_lateness_samples_ms": performance._logical_schedule_lateness_samples(  # noqa: SLF001
            service_samples, cycle_samples
        ),
        "unique_composite_count": measured,
        "model_invocation_count": measured,
        "output_send_count": measured,
        "output_repeat_count": 0,
        "no_unread_repeat_count": 0,
        "capture_slot_overwrite_count": 0,
        "capture_sequence_gap_count": 0,
        "capture_missing_input_count": 0,
        "processing_deadline_miss_count": 0,
        "serialized_new_frame_deadline_miss_count": measured,
        "sink_recovery_count": 0,
    }
    return performance.build_performance_report(
        _generated_performance_rows(), full_path_evidence=full_path
    )


def generated_balanced_effective_policy() -> dict[str, object]:
    """Return the strict MATTE-3.4 balanced policy shape."""

    return {
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
    }


def generated_rvm_prerequisite() -> dict[str, object]:
    """Return a deterministic MATTE-2.5 report with no proposed profile."""

    deterministic: dict[str, object] = {
        "source": {"kind": "generated-schema-proxy"},
        "provenance": {"kind": "generated"},
        "privacy": {"contains_pixels": False},
        "qualification_contract": {"authority": "none"},
        "coverage": {"complete": False},
        "rows": [],
        "candidate_decisions": [],
        "profiles": {
            "status": "not_proposed",
            "definitions": [],
            "reasons": ["generated-schema-proxy"],
            "stable_cross_device_meaning_required": True,
        },
        "production": {
            "default_changed": False,
            "high_detail_global_default_selected": False,
            "generated_proxy_can_select_profile": False,
        },
    }
    return {
        "schema": qualification.RVM_REPORT_SCHEMA,
        "version": qualification.RVM_REPORT_VERSION,
        **deterministic,
        "evidence_sha256": sha256_bytes(json_bytes(deterministic)),
    }


def sign_rvm_report(report: dict[str, Any]) -> dict[str, Any]:
    """Refresh the MATTE-2.5 deterministic-subset digest after a mutation."""

    deterministic_keys = (
        "source",
        "provenance",
        "privacy",
        "qualification_contract",
        "coverage",
        "rows",
        "candidate_decisions",
        "profiles",
        "production",
    )
    report = dict(report)
    report["evidence_sha256"] = sha256_bytes(
        json_bytes({key: report[key] for key in deterministic_keys})
    )
    return report


def generated_run_samples(
    *,
    provider: str = "cpu",
    rvm: bool = True,
    measured_frames: int = qualification.MIN_MEASURED_FRAMES,
    target_fps: int = qualification.TARGET_FPS,
) -> list[dict[str, object]]:
    """Return deterministic raw per-output samples for the run parser.

    These are schema/derivation fixtures, not benchmark observations.  They
    intentionally carry generated proxy timings and therefore cannot authorize
    any physical platform even when every numerical gate is satisfied.
    """

    frame_ms = 1000.0 / target_fps
    substages = {name: 0.25 for name in qualification.COMPOSITOR_SUBSTAGE_NAMES}
    result: list[dict[str, object]] = []
    for index in range(measured_frames):
        capture_completed_ms = index * frame_ms
        pacing_wait_ms = frame_ms - 20.0
        processing_started_ms = capture_completed_ms + pacing_wait_ms
        sink_completed_ms = capture_completed_ms + frame_ms
        result.append(
            {
                "index": index,
                "capture_sequence": index,
                "source_capture_sequence": index,
                "source_capture_timestamp_ns": round(
                    index * 1_000_000_000 / target_fps
                ),
                "source_capture_generation": 1,
                "source_geometry_generation": 1,
                "segmentation_sequence": index,
                "composite_sequence": index,
                "output_sequence": index,
                "capture_generation": 1,
                "capture_completed_ms": round(capture_completed_ms, 6),
                "processing_started_ms": round(processing_started_ms, 6),
                "sink_completed_ms": round(sink_completed_ms, 6),
                "complete_service_ms": 20.0,
                "serialized_cycle_ms": round(frame_ms, 6),
                "frame_processing_ms": 14.0,
                "model_preprocess_ms": 1.0 if rvm else None,
                "model_inference_ms": 2.0 if rvm else None,
                "model_postprocess_ms": 1.0 if rvm else None,
                "segmentation_ms": 4.0,
                "refinement_ms": 1.0,
                "background_ms": 1.0,
                "compositor_ms": 5.0,
                "compositor_substages_ms": dict(substages),
                "sink_prepare_ms": 1.0,
                "sink_submit_ms": 1.0,
                "sink_copy_ms": 1.0,
                "pacing_wait_ms": round(pacing_wait_ms, 6),
                "schedule_lateness_ms": 0.0,
                "queue_age_ms": round(pacing_wait_ms, 6),
                "end_to_end_age_ms": round(frame_ms, 6),
                "process_cpu_percent": 25.0,
                "rss_bytes": 256 * 1024 * 1024,
                "gpu_utilization_percent": 50.0 if provider != "cpu" else None,
                "vram_bytes": 1024 * 1024 * 1024 if provider != "cpu" else None,
                "capture_drop_count": 0,
                "capture_slot_overwrite_count": 0,
                "deadline_miss": False,
                "sink_recovery_count": 0,
                "no_unread_repeat": False,
            }
        )
    return result


def generated_run_sample_sets(
    *, provider: str = "cpu", rvm: bool = True
) -> dict[str, list[dict[str, object]]]:
    """Return independently materialized physical-capture and replay traces."""

    result = {
        source: generated_run_samples(provider=provider, rvm=rvm)
        for source in ("physical_capture", "fixed_replay")
    }
    for row in result["physical_capture"]:
        row["process_cpu_percent"] = 24.0
    return result


def derived_run_counters(
    samples: list[dict[str, object]],
) -> dict[str, int]:
    """Compute the submitted counters directly from raw fixture rows."""

    def integer(row: dict[str, object], name: str) -> int:
        value = row[name]
        assert type(value) is int
        return value

    capture_sequences = [integer(row, "capture_sequence") for row in samples]
    return {
        "measured_frames": len(samples),
        "unique_capture_frames": len(set(capture_sequences)),
        "unique_segmentations": len(
            {integer(row, "segmentation_sequence") for row in samples}
        ),
        "unique_composites": len(
            {integer(row, "composite_sequence") for row in samples}
        ),
        "sink_submissions": len(samples),
        "capture_gaps": sum(
            max(0, current - previous - 1)
            for previous, current in zip(capture_sequences, capture_sequences[1:])
        ),
        "capture_drops": sum(integer(row, "capture_drop_count") for row in samples),
        "capture_slot_overwrites": sum(
            integer(row, "capture_slot_overwrite_count") for row in samples
        ),
        "processing_deadline_misses": sum(
            1 for row in samples if row["deadline_miss"] is True
        ),
        "sink_recoveries": sum(integer(row, "sink_recovery_count") for row in samples),
        "no_unread_repeats": sum(
            1 for row in samples if row["no_unread_repeat"] is True
        ),
    }


def generated_resource_samples(
    *, provider: str = "cpu", sample_count: int = qualification.MIN_RESOURCE_SAMPLES
) -> list[dict[str, object]]:
    """Return a stable deterministic 30-minute resource observation trace."""

    assert sample_count >= 2
    return [
        {
            "offset_s": round(
                qualification.MIN_SOAK_SECONDS * index / (sample_count - 1), 6
            ),
            "process_cpu_percent": 25.0,
            "rss_bytes": 256 * 1024 * 1024,
            "gpu_percent": 50.0 if provider != "cpu" else None,
            "vram_bytes": 1024 * 1024 * 1024 if provider != "cpu" else None,
        }
        for index in range(sample_count)
    ]


def generated_lifecycle(*, provider: str = "cpu") -> dict[str, object]:
    """Return explicit stable/restart/hot-patch/shutdown observations."""

    snapshots: list[dict[str, object]] = []
    marker_by_offset = {
        0.0: "sustained_start",
        600.0: "pre_restart",
        660.0: "post_restart",
        1200.0: "pre_hot_patch",
        1260.0: "post_hot_patch",
        qualification.MIN_SOAK_SECONDS: "shutdown",
    }
    for index in range(qualification.MIN_RESOURCE_SAMPLES):
        offset = round(
            qualification.MIN_SOAK_SECONDS
            * index
            / (qualification.MIN_RESOURCE_SAMPLES - 1),
            6,
        )
        generation = 1 if offset <= 600.0 else 2 if offset <= 1200.0 else 3
        successful = round(offset * 30.0)
        snapshots.append(
            {
                "kind": marker_by_offset.get(offset, "heartbeat"),
                "offset_s": offset,
                "generation": generation,
                "active_provider": provider,
                "unique_capture_frames": successful,
                "unique_segmentations": successful,
                "unique_composites": successful,
                "sink_submissions": successful,
                "capture_gaps": 0,
                "capture_drops": 0,
                "capture_slot_overwrites": 0,
                "processing_deadline_misses": 0,
                "sink_recoveries": 0,
            }
        )
    return {
        "soak_duration_s": qualification.MIN_SOAK_SECONDS,
        "events": [
            {"kind": "sustained", "offset_s": 0.0, "generation": 1},
            {"kind": "restart", "offset_s": 630.0, "generation": 2},
            {"kind": "hot_patch", "offset_s": 1230.0, "generation": 3},
            {
                "kind": "shutdown",
                "offset_s": qualification.MIN_SOAK_SECONDS,
                "generation": 3,
            },
        ],
        "counter_snapshots": snapshots,
        "restart": {
            "reset_visible": True,
            "fallback_occurred": provider != "cpu",
            "fallback_visible": provider != "cpu",
            "first_output_fresh": True,
            "stale_state_flash": False,
            "provider_before": provider,
            "provider_during_fallback": "cpu" if provider != "cpu" else None,
            "provider_after_recovery": provider,
            "recovery_generation": 2,
            "first_output_generation": 2,
        },
        "hot_patch": {
            "transactional": True,
            "reset_visible": True,
            "first_output_fresh": True,
            "stale_state_flash": False,
            "first_output_generation": 3,
        },
        "shutdown": {
            "duration_ms": 100.0,
            "workers_alive": 0,
            "resources_open": 0,
            "close_error": None,
        },
        "history": {
            "bounded": True,
            "previous_frame_slots": 1,
            "previous_mask_slots": 1,
            "work_buffer_slots": 4,
        },
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def nearest(percentile: float) -> float:
        index = max(1, int((percentile * len(ordered) + 0.999999999).__floor__()))
        return ordered[min(index - 1, len(ordered) - 1)]

    return {
        "count": len(values),
        "p50": nearest(0.50),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
        "max": max(values),
    }


def generated_capture_report(
    cell: dict[str, object], *, fixture_id: str
) -> dict[str, object]:
    """Return content-free capture-only evidence with no physical authority."""

    sample_count = 150
    offsets = [
        round(index * (1000.0 / qualification.TARGET_FPS), 6)
        for index in range(sample_count)
    ]
    trace = [
        {
            "source_sequence_offset": index,
            "generation_offset": 0,
            "geometry_generation_offset": 0,
            "completion_offset_ms": offset,
            "read_ms": 2.0,
            "pre_normalization_ms": 0.25,
            "normalization_ms": 1.0,
            "publish_ms": 0.25,
            "total_ms": 3.5,
            "reader_cpu_ms": 0.1,
        }
        for index, offset in enumerate(offsets)
    ]
    intervals = [current - previous for previous, current in zip(offsets, offsets[1:])]
    completion_span = offsets[-1]
    wall_fps = (sample_count - 1) * 1000.0 / completion_span
    timing = {
        "sample_count": sample_count,
        "source_success_count": sample_count,
        "observer_missing_sample_count": 0,
        "generation_count": 1,
        "geometry_generation_count": 1,
        "measurement_window": {
            "duration_seconds": 5.0,
            "leading_gap_ms": 0.0,
            "trailing_gap_ms": 5000.0 - completion_span,
            "completion_span_ms": completion_span,
            "availability_fps": 30.0,
        },
        "active_capture_fps": wall_fps,
        "wall_completion_fps": wall_fps,
        "maximum_completion_gap_ms": max(intervals),
        "trace": trace,
        "interval_ms": _distribution(intervals),
        "cross_generation_outage_ms": _distribution([]),
        "read_ms": _distribution([2.0] * (sample_count - 1)),
        "pre_normalization_ms": _distribution([0.25] * (sample_count - 1)),
        "normalization_ms": _distribution([1.0] * (sample_count - 1)),
        "publish_ms": _distribution([0.25] * (sample_count - 1)),
        "total_ms": _distribution([3.5] * (sample_count - 1)),
        "reader_cpu_ms": _distribution([0.1] * (sample_count - 1)),
        "correlation": {
            "interval_vs_read": None,
            "interval_vs_previous_normalization": None,
        },
    }
    canvas = cell["canvas"]
    assert isinstance(canvas, dict)
    route = qualification.ROUTE_CONTRACTS[str(cell["route_id"])]
    return {
        "schema": qualification.CAPTURE_REPORT_SCHEMA,
        "version": qualification.CAPTURE_REPORT_VERSION,
        "privacy": {
            "contains_pixels": False,
            "contains_frame_hashes": False,
            "contains_wall_clock_timestamps": False,
            "contains_device_path_or_index": False,
            "contains_only_opaque_identity_bindings": True,
            "contains_credentials": False,
            "timing_trace_uses_relative_monotonic_offsets": True,
        },
        "capture_only_contract": {
            "production_capture_reader": True,
            "camera_acquisition": True,
            "canonical_normalization": True,
            "segmentation": False,
            "backdrop": False,
            "compositor": False,
            "preview": False,
            "api": False,
            "output_sink": False,
            "camera_control_policy": "preserve",
            "camera_control_writes": False,
        },
        "requested": {
            "width": canvas["width"],
            "height": canvas["height"],
            "canvas_width": canvas["width"],
            "canvas_height": canvas["height"],
            "fps": cell["target_fps"],
            "pixel_format": "auto",
            "mode_mismatch": "warn",
        },
        "negotiated": {
            "backend": route["capture_backend"],
            "pixel_format": "MJPG",
            "width": canvas["width"],
            "height": canvas["height"],
            "fps_reported": cell["target_fps"],
            "delivered_width": canvas["width"],
            "delivered_height": canvas["height"],
            "oriented_width": canvas["width"],
            "oriented_height": canvas["height"],
            "normalized_width": canvas["width"],
            "normalized_height": canvas["height"],
        },
        "camera_controls": {
            "policy": "preserve",
            "backend_family": "generated",
            "qualification": "unqualified",
            "writes_performed": False,
            "generation": 1,
            "properties": {},
        },
        "measurement": {
            "warmup_seconds_requested": 2.0,
            "measurement_seconds_requested": 5.0,
            "measurement_seconds_actual": 5.0,
            "successful_reads": sample_count,
            "harness_deliveries": sample_count,
            "latest_slot_overwrites": 0,
            "read_failures": 0,
            "restarts": 0,
            "geometry_transitions": 0,
            "warmup_read_failures": 0,
            "warmup_restarts": 0,
            "warmup_geometry_transitions": 0,
            "stalled_at_end": False,
            "capture_error": None,
            "close_error": None,
            "process_cpu_ms": 25.0,
            "process_cpu_percent_of_one_core": 0.5,
        },
        "timing": timing,
        "pacing": {
            "capture": {
                "measured": True,
                "active_source_fps": wall_fps,
                "wall_completion_fps": wall_fps,
                "target_fps": cell["target_fps"],
            },
            "processed_frames": {
                "measured": False,
                "reason": "not-run-by-capture-only-harness",
                "unique_fps": None,
                "frame_processing_ms": None,
            },
            "output": {
                "measured_by_capture_only_harness": False,
                "reason": "no-output-sink-was-opened",
            },
        },
        "native_comparison": {
            "provided": False,
            "compatible": None,
            "compatibility_reasons": ["native-comparison-not-provided"],
            "summary": None,
        },
        "full_runtime_comparison": {
            "provided": False,
            "compatible": None,
            "compatibility_reasons": ["full-runtime-evidence-not-provided"],
            "capture_pacing": None,
            "processed_frame_pacing": {
                "measured": False,
                "reason": "not-run-by-capture-only-harness",
                "unique_fps": None,
                "frame_processing_ms": None,
            },
        },
        "diagnosis": {
            "code": "target-sustained",
            "confidence": "high",
            "actionable": True,
            "target_sustained": True,
            "target_threshold_fps": 27.0,
            "capture_only_fps": wall_fps,
            "availability_fps": 30.0,
            "measurement_window_complete": True,
            "measurement_window_sustained": True,
            "measurement_boundary_tolerance_ms": 100.0,
            "minimum_successful_reads": 135,
            "segmentation_executed": False,
            "segmentation_causal_for_capture_only_result": False,
            "runtime_comparison": "not-compared",
            "actions": [
                "compare-a-matched-full-processing-run-with-capture-pacing-kept-separate"
            ],
            "non_claims": [
                "opencv-read-time-does-not-separate-device-transfer-and-decode",
                "auto-exposure-observation-does-not-prove-low-light-causality",
                "capture-recovery-does-not-prove-processed-frame-budget-recovery",
            ],
        },
        "qualification": {
            "backlog_acceptance_mode": "1280x720@30",
            "minimum_unique_fps": 27.0,
            "exact_acceptance_mode_requested": True,
            "exact_mode_verified": True,
            "hardware_verified": False,
            "acceptance_satisfied": False,
            "outcome": "hardware-evidence-required",
            "note": (
                "Synthetic or CI timing validates the harness but cannot qualify "
                "physical-camera cadence."
            ),
        },
        "condition": {
            "id": f"generated-{fixture_id}",
            "hardware_verified": False,
            "source_kind": "generated-non-authoritative",
            "physical_source_eligible": False,
            "device_identity_sha256": None,
            "hardware_identity_sha256": None,
            "device_identity_bound": False,
            "hardware_identity_bound": False,
        },
    }


def generated_run_report(
    *,
    plan: dict[str, object],
    cell: dict[str, object],
    profile: dict[str, object],
    capture_file_sha256: str,
) -> dict[str, object]:
    """Return dual-source generated run evidence bound to one exact cell."""

    provider = str(cell["provider"])
    dependency = str(cell["dependency_profile"])
    profile_id = str(cell["profile_id"])
    sample_sets = generated_run_sample_sets(
        provider=provider, rvm=profile_id == "rvm_matting"
    )
    mediapipe_installed = dependency != "without_mediapipe"
    gpu_provider_installed = provider != "cpu"
    installed_distributions = ["custback"]
    if mediapipe_installed:
        installed_distributions.append("mediapipe")
    installed_distributions.sort(key=str.casefold)
    available_providers = [qualification.PROVIDER_RUNTIME_NAMES["cpu"]]
    if gpu_provider_installed:
        available_providers.append(qualification.PROVIDER_RUNTIME_NAMES[provider])
    available_providers.sort()
    measured_lineage = generated_fixed_replay_lineage()
    if dependency == "standard":
        requested_profile = profile_id
        fallback_reason = None
        fallback_visible = False
    elif dependency == "without_mediapipe":
        requested_profile = "mediapipe_segmentation"
        fallback_reason = "mediapipe-unavailable"
        fallback_visible = True
    else:
        requested_profile = "rvm_matting"
        fallback_reason = "gpu-provider-unavailable"
        fallback_visible = True
    candidate = plan["candidate"]
    provenance = plan["provenance"]
    canvas = cell["canvas"]
    assert isinstance(candidate, dict)
    assert isinstance(provenance, dict)
    assert isinstance(canvas, dict)
    route = qualification.ROUTE_CONTRACTS[str(cell["route_id"])]
    return sign_report(
        {
            "schema": qualification.RUN_SCHEMA,
            "version": qualification.RUN_VERSION,
            "provenance": {
                "kind": "generated",
                "run_id": str(cell["id"]),
                "build_sha256": candidate["build_sha256"],
                "contains_pixels": False,
                "contains_paths": False,
            },
            "candidate": {
                "revision": candidate["revision"],
                "profile_id": profile_id,
                "configured_policy_sha256": profile["configured_policy_sha256"],
                "effective_policy_sha256": profile["effective_policy_sha256"],
                "visual_candidate_id": profile["visual_candidate_id"],
                "visual_algorithm_contract_sha256": profile[
                    "visual_algorithm_contract_sha256"
                ],
            },
            "model": (
                {
                    "id": performance.RVM_MODEL.filename,
                    "sha256": performance.RVM_MODEL.sha256,
                    "bytes": performance.RVM_MODEL.size,
                }
                if profile_id == "rvm_matting"
                else None
            ),
            "platform": {"route_id": cell["route_id"], **route},
            "hardware": {
                "identity_sha256": sha256_bytes(b"generated-hardware"),
                "logical_cpu_count": 8,
                "gpu_identity_sha256": (
                    sha256_bytes(f"generated-{provider}-gpu".encode())
                    if provider != "cpu"
                    else None
                ),
            },
            "runtime": {
                "python": "generated-python",
                "package_set_sha256": sha256_bytes(b"generated-package-set"),
                "provider_environment_sha256": sha256_bytes(
                    f"generated-{provider}-environment".encode()
                ),
            },
            "provider": {
                "name": provider,
                "device": f"generated-{provider}",
                "device_id": 0 if provider != "cpu" else None,
                "execution_verified": True,
            },
            "sink": {
                "backend": route["sink_backend"],
                "consumer": route["consumer"],
                "consumer_recording_verified": False,
                "no_unread_repeat_semantics": True,
            },
            "selection": {
                "requested_profile_id": requested_profile,
                "selected_profile_id": profile_id,
                "fallback_reason": fallback_reason,
                "fallback_visible": fallback_visible,
                "slow_profile_suppressed": dependency != "standard",
            },
            "dependencies": {
                "profile": dependency,
                "mediapipe_installed": mediapipe_installed,
                "gpu_provider_installed": gpu_provider_installed,
                "installed_distribution_sha256": canonical_digest(
                    installed_distributions
                ),
                "installed_distributions": installed_distributions,
                "available_providers": available_providers,
                "mediapipe_probe": ("available" if mediapipe_installed else "missing"),
                "gpu_provider_probe": (
                    "available" if gpu_provider_installed else "missing"
                ),
            },
            "physical_authority": {
                "capture_origin_attested": provenance[
                    "physical_capture_origin_attested"
                ],
                "consumer_origin_attested": provenance[
                    "physical_consumer_origin_attested"
                ],
                "same_host_attested": provenance["same_host_attested"],
                "cryptographically_proven": False,
            },
            "capture_binding": {
                "capture_file_sha256": capture_file_sha256,
                "device_identity_sha256": None,
                "hardware_identity_sha256": None,
                "same_host_attested": provenance["same_host_attested"],
                "cryptographic_same_host_proof": False,
            },
            "scope": {
                "width": canvas["width"],
                "height": canvas["height"],
                "target_fps": cell["target_fps"],
                "warmup_frames_by_source": {
                    "physical_capture": qualification.MIN_WARMUP_FRAMES,
                    "fixed_replay": qualification.MIN_WARMUP_FRAMES,
                },
                "measured_seconds_by_source": {
                    "physical_capture": qualification.MIN_MEASURED_SECONDS,
                    "fixed_replay": qualification.MIN_MEASURED_SECONDS,
                },
                "reactions_enabled": False,
                "post_base_event_count": 0,
            },
            "sources": {
                "physical_capture": {
                    "kind": "physical-live-capture",
                    "acquisition": "paced-production-reader",
                    "capture_report_sha256": capture_file_sha256,
                    "trace_sha256": canonical_digest(sample_sets["physical_capture"]),
                },
                "fixed_replay": {
                    "kind": "immutable-fixed-replay",
                    "acquisition": "target-paced-replay",
                    "source_sha256": sha256_bytes(b"generated-performance-source"),
                    "bundle_manifest_sha256": None,
                    "measured_frame_lineage_sha256": generated_lineage_digest(),
                    "warmup_frame_count": qualification.MIN_WARMUP_FRAMES,
                    "measured_frame_count": qualification.MIN_MEASURED_FRAMES,
                    "measured_lineage": measured_lineage,
                    "trace_sha256": canonical_digest(sample_sets["fixed_replay"]),
                },
            },
            "samples": sample_sets,
            "counters": {
                source: derived_run_counters(samples)
                for source, samples in sample_sets.items()
            },
            "resource_samples": generated_resource_samples(provider=provider),
            "lifecycle": generated_lifecycle(provider=provider),
        }
    )


def rebind_capture_window(
    report: dict[str, Any],
    *,
    completion_span_ms: float,
    leading_gap_ms: float,
    trailing_gap_ms: float,
) -> None:
    """Recompute all raw-derived capture-window fields after a timing mutation."""

    timing = report["timing"]
    assert isinstance(timing, dict)
    trace = timing["trace"]
    assert isinstance(trace, list)
    offsets = [
        round(completion_span_ms * index / (len(trace) - 1), 6)
        for index in range(len(trace))
    ]
    for row, offset in zip(trace, offsets):
        assert isinstance(row, dict)
        row["completion_offset_ms"] = offset
    intervals = [current - previous for previous, current in zip(offsets, offsets[1:])]
    wall_fps = (len(trace) - 1) * 1000.0 / offsets[-1]
    timing["interval_ms"] = _distribution(intervals)
    timing["active_capture_fps"] = wall_fps
    timing["wall_completion_fps"] = wall_fps
    timing["maximum_completion_gap_ms"] = max(intervals)
    timing["measurement_window"] = {
        "duration_seconds": (leading_gap_ms + completion_span_ms + trailing_gap_ms)
        / 1000.0,
        "leading_gap_ms": leading_gap_ms,
        "trailing_gap_ms": trailing_gap_ms,
        "completion_span_ms": offsets[-1],
        "availability_fps": len(trace)
        / ((leading_gap_ms + completion_span_ms + trailing_gap_ms) / 1000.0),
    }
    pacing = report["pacing"]
    assert isinstance(pacing, dict)
    capture_pacing = pacing["capture"]
    assert isinstance(capture_pacing, dict)
    capture_pacing["active_source_fps"] = wall_fps
    capture_pacing["wall_completion_fps"] = wall_fps
    diagnosis = report["diagnosis"]
    assert isinstance(diagnosis, dict)
    availability_fps = len(trace) / (
        (leading_gap_ms + completion_span_ms + trailing_gap_ms) / 1000.0
    )
    diagnosis["availability_fps"] = availability_fps
    diagnosis["capture_only_fps"] = min(availability_fps, wall_fps)
    measurement = report["measurement"]
    assert isinstance(measurement, dict)
    measurement["measurement_seconds_actual"] = (
        leading_gap_ms + completion_span_ms + trailing_gap_ms
    ) / 1000.0


def generate_qualification(
    root: Path, *, recorded: bool = False
) -> GeneratedPlatformQualification:
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    root.chmod(0o700)

    prerequisite_paths = {
        "visual": root / "visual.json",
        "performance": root / "performance.json",
        "rvm": root / "rvm.json",
    }
    prerequisite_schemas = {
        "visual": qualification.VISUAL_REPORT_SCHEMA,
        "performance": qualification.PERFORMANCE_REPORT_SCHEMA,
        "rvm": qualification.RVM_REPORT_SCHEMA,
    }
    for name, path in prerequisite_paths.items():
        write_json(path, _digest_report(prerequisite_schemas[name]))

    cells = generated_cells()
    generated_profile_rows = generated_profiles(cells)
    if recorded:
        write_json(
            prerequisite_paths["visual"],
            generated_visual_prerequisite(generated_profile_rows),
        )
        write_json(
            prerequisite_paths["performance"], generated_performance_prerequisite()
        )
        write_json(prerequisite_paths["rvm"], generated_rvm_prerequisite())
    plan = {
        "schema": qualification.PLAN_SCHEMA,
        "version": qualification.PLAN_VERSION,
        "qualification": {"id": "generated-matte-platform-matrix"},
        "provenance": {
            "kind": "generated",
            "license_or_consent_sha256": None,
            "physical_capture_origin_attested": False,
            "physical_consumer_origin_attested": False,
            "same_host_attested": False,
            "physical_origin_cryptographically_proven": False,
        },
        "reactions": {"enabled": False, "post_base_event_count": 0},
        "candidate": {
            "revision": "generated-candidate",
            "build_sha256": sha256_bytes(b"generated-candidate-build"),
            "defaults_changed": False,
        },
        "prerequisites": {
            name: _prerequisite_descriptor(root, path)
            for name, path in prerequisite_paths.items()
        },
        "routes": [
            {"id": route_id, **contract}
            for route_id, contract in qualification.ROUTE_CONTRACTS.items()
        ],
        "profiles": generated_profile_rows,
        "cells": cells,
    }
    run_paths: dict[str, Path] = {}
    capture_paths: dict[str, Path] = {}
    if recorded:
        profiles = {str(profile["id"]): profile for profile in plan["profiles"]}
        for cell in cells:
            cell_id = str(cell["id"])
            capture_path = root / f"capture-{cell_id}.json"
            write_json(
                capture_path,
                generated_capture_report(cell, fixture_id=cell_id),
            )
            capture_descriptor = file_descriptor(root, capture_path)
            run_path = root / f"run-{cell_id}.json"
            write_json(
                run_path,
                generated_run_report(
                    plan=plan,
                    cell=cell,
                    profile=profiles[str(cell["profile_id"])],
                    capture_file_sha256=capture_descriptor["sha256"],
                ),
            )
            cell.update(
                {
                    "state": "recorded",
                    "reason": None,
                    "run": artifact_descriptor(root, run_path),
                    "capture_report": capture_descriptor,
                }
            )
            run_paths[cell_id] = run_path
            capture_paths[cell_id] = capture_path
    plan_path = root / "plan.json"
    write_json(plan_path, plan)
    return GeneratedPlatformQualification(
        root=root,
        plan=plan_path,
        prerequisites=prerequisite_paths,
        runs=run_paths,
        captures=capture_paths,
    )
