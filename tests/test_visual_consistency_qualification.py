from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

_QUALIFICATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "release"
    / "visual_consistency_qualification.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "custback_visual_consistency_qualification_test", _QUALIFICATION_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
qualification = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qualification
_SPEC.loader.exec_module(qualification)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return qualification.load_manifest()


@pytest.fixture(scope="module")
def deterministic_report(
    manifest: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("visual-qualification")
    report = qualification.build_report(
        manifest,
        contact_sheet_path=root / "contact-sheet.png",
        soak_frames=64,
    )
    qualification.validate_report(report, manifest, claim="deterministic")
    return report


def _timing_profile(total_ms: float, frames: int) -> dict[str, list[float]]:
    values = {
        "geometry_ms": 2.0,
        "linear_conversion_ms": 1.0,
        "color_analysis_ms": 1.0,
        "composite_ms": total_ms - 4.0,
        "output_send_ms": 0.0,
        "frame_total_ms": total_ms,
    }
    return {name: [value] * frames for name, value in values.items()}


def _timing_stats(samples: dict[str, list[float]]) -> dict[str, Any]:
    return {
        name: qualification._percentiles(values) for name, values in samples.items()
    }


def _ewma_snapshot() -> dict[str, Any]:
    values = {
        "frames_out": 12,
        "fps": 60.0,
        "fps_attainment_pct": 100.0,
        "output_repeated_frames": 0,
        "processing_deadline_misses": 0,
        "segmentation_ms": 1.0,
        "background_ms": 1.0,
        "color_correction_ms": 1.0,
        "composite_ms": 1.0,
        "output_send_ms": 0.1,
        "frame_processing_ms": 4.1,
    }
    return {
        "source": "FrameHub.stats_dict production EWMA fields",
        "frames": 12,
        "baseline": dict(values),
        "candidate": dict(values),
    }


def _passing_calibrated(manifest: dict[str, Any]) -> dict[str, Any]:
    frames = 300
    tiers = []
    for definition in manifest["performance_tiers"]:
        baseline = _timing_profile(10.0, frames)
        candidate = _timing_profile(11.0, frames)
        samples = {"baseline": baseline, "candidate": candidate}
        tiers.append(
            {
                "id": definition["id"],
                "status": "pass",
                "width": definition["width"],
                "height": definition["height"],
                "target_fps": definition["fps"],
                "measured_frames": frames,
                "warmup_frames": 5,
                "baseline": _timing_stats(baseline),
                "candidate": _timing_stats(candidate),
                "samples": samples,
                "samples_sha256": hashlib.sha256(
                    qualification._canonical_json(samples)
                ).hexdigest(),
                "added_p95_ms": 1.0,
                "relative_end_to_end_overhead_percent": 10.0,
                "baseline_deadline_miss_percent": 0.0,
                "candidate_deadline_miss_percent": 0.0,
                "deadline_miss_increase_percentage_points": 0.0,
                "effective_fps": 90.909,
                "fps_attainment_percent": 100.0,
                "peak_tracemalloc_bytes": 1,
                "rss_growth_bytes": 0,
                "production_ewma": _ewma_snapshot(),
                "checks": {
                    "relative_overhead": True,
                    "added_p95": True,
                    "deadline_miss_delta": True,
                    "fps_attainment": True,
                },
            }
        )
    return {
        "status": "pass",
        "authority": "pinned-reference-runner",
        "runner": {
            "id": "pinned-linux-visual-01",
            "pinned": True,
            "cpu": "reviewed-cpu",
            "os": "reviewed-linux-image",
            "os_family": "linux",
        },
        "tiers": tiers,
    }


def test_deterministic_contract_records_real_operation_and_memory_bounds(
    manifest: dict[str, Any], deterministic_report: dict[str, Any]
) -> None:
    deterministic = deterministic_report["deterministic"]
    repeated = deterministic["contracts"]["repeated-output-work-skip"]["metrics"]
    assert repeated == {
        "source_frames": 1,
        "output_frames": 6,
        "repeated_output_frames": 5,
        "color_analysis_calls": 1,
        "linear_decode_calls": 2,
        "composite_calls": 1,
    }
    soak = deterministic["soak"]
    assert soak["privacy_history_storage_bytes"] == 16 * 1024 * 1024
    assert soak["maximum_canvas_frame_bytes"] == 1920 * 1080 * 3
    assert soak["maximum_color_analysis_cache_bytes"] == 192**2 * 3 * 4
    assert soak["fixed_session_state_semantic_bytes"] == 23_440_384
    assert soak["fixed_session_state_semantic_bytes"] <= 24 * 1024 * 1024
    assert soak["tracemalloc_current_growth_bytes"] <= 5 * 1024 * 1024
    assert set(deterministic_report["source"]["files"]) == set(
        qualification.SOURCE_FILES
    )
    assert "config/default.yaml" in qualification.SOURCE_FILES
    assert "src/custback/default.yaml" in qualification.SOURCE_FILES
    qualification.validate_report(deterministic_report, manifest, claim="deterministic")


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (
            (
                "deterministic",
                "contracts",
                "repeated-output-work-skip",
                "metrics",
                "composite_calls",
            ),
            2,
        ),
        (
            ("deterministic", "soak", "privacy_history_storage_bytes"),
            4096,
        ),
        (
            ("deterministic", "soak", "maximum_color_analysis_cache_bytes"),
            4096,
        ),
    ),
)
def test_deterministic_validator_rejects_semantic_tampering(
    manifest: dict[str, Any],
    deterministic_report: dict[str, Any],
    path: tuple[str, ...],
    value: object,
) -> None:
    tampered = copy.deepcopy(deterministic_report)
    target = tampered
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(qualification.QualificationError):
        qualification.validate_report(tampered, manifest, claim="deterministic")


def test_calibrated_validator_derives_metrics_from_raw_samples(
    manifest: dict[str, Any],
) -> None:
    calibrated = _passing_calibrated(manifest)
    qualification._validate_calibrated_release(calibrated, manifest)

    tampered = copy.deepcopy(calibrated)
    tampered["tiers"][0]["effective_fps"] = 999.0
    with pytest.raises(
        qualification.QualificationError, match="derived metrics are inconsistent"
    ):
        qualification._validate_calibrated_release(tampered, manifest)

    missing_rss = copy.deepcopy(calibrated)
    missing_rss["tiers"][0]["rss_growth_bytes"] = None
    with pytest.raises(qualification.QualificationError, match="RSS"):
        qualification._validate_calibrated_release(missing_rss, manifest)


def test_calibrated_cli_fails_closed_unless_observation_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = {"calibrated": {"status": "fail"}}
    monkeypatch.setattr(
        qualification,
        "load_manifest",
        lambda _path: {"budgets": {"soak_frames": 10_000}},
    )
    monkeypatch.setattr(qualification, "build_report", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(qualification, "_write_exclusive", lambda *_args: None)
    arguments = [
        "run",
        "--output",
        str(tmp_path / "report.json"),
        "--contact-sheet",
        str(tmp_path / "contact.png"),
        "--calibrated",
    ]
    assert qualification.main(arguments) == 1
    assert qualification.main([*arguments, "--observation-only"]) == 0


def test_calibrated_candidate_uses_production_linear_bgr_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width, height, fps = 256, 144, 30
    geometry_source, foreground, backdrop, mask = qualification._tier_arrays(
        width, height
    )
    plan = qualification.plan_transform(
        (geometry_source.shape[1], geometry_source.shape[0]),
        (width, height),
        fit="cover",
    )
    output = qualification.NullOutput(width, height, fps)
    calls: list[str] = []

    real_decode = qualification.color_mod._bgr_u8_to_linear_bgr_prevalidated
    real_estimate = (
        qualification.color_mod._estimate_color_transform_linear_bgr_prevalidated
    )
    real_composite = qualification.compositor_mod._composite_linear_bgr_prevalidated
    backdrop_analysis_linear_bgr = (
        qualification.color_mod._linear_bgr_analysis_raster_prevalidated(
            real_decode(backdrop)
        )
    )

    def decode(frame: np.ndarray) -> np.ndarray:
        calls.append("decode")
        return real_decode(frame)

    def estimate(
        foreground_linear_bgr: np.ndarray,
        backdrop_linear_bgr: np.ndarray,
        analysis_mask: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        calls.append("estimate")
        assert kwargs["resize_executor"] is resize_executor
        assert kwargs["backdrop_analysis_linear_bgr"] is backdrop_analysis_linear_bgr
        return real_estimate(
            foreground_linear_bgr,
            backdrop_linear_bgr,
            analysis_mask,
            **kwargs,
        )

    def composite(*args: Any, **kwargs: Any) -> np.ndarray:
        calls.append("composite")
        return real_composite(*args, **kwargs)

    def reject_legacy_path(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("calibrated candidate used the public RGB path")

    monkeypatch.setattr(
        qualification.color_mod,
        "_bgr_u8_to_linear_bgr_prevalidated",
        decode,
    )
    monkeypatch.setattr(
        qualification.color_mod,
        "_estimate_color_transform_linear_bgr_prevalidated",
        estimate,
    )
    monkeypatch.setattr(
        qualification.compositor_mod,
        "_composite_linear_bgr_prevalidated",
        composite,
    )
    monkeypatch.setattr(
        qualification.color_mod,
        "bgr_u8_to_linear_rgb",
        reject_legacy_path,
    )
    monkeypatch.setattr(
        qualification,
        "estimate_color_transform_linear",
        reject_legacy_path,
    )
    monkeypatch.setattr(
        qualification,
        "composite_linear_predecoded",
        reject_legacy_path,
    )

    with qualification.ThreadPoolExecutor(max_workers=3) as resize_executor:
        timings = qualification._one_candidate_frame(
            geometry_source,
            foreground,
            backdrop,
            mask,
            plan,
            output,
            resize_executor=resize_executor,
            backdrop_analysis_linear_bgr=backdrop_analysis_linear_bgr,
        )

    assert calls == ["decode", "decode", "estimate", "composite"]
    assert max(backdrop_analysis_linear_bgr.shape[:2]) == 192
    assert set(timings) == set(qualification.TIMING_STAGES)
    assert all(value >= 0.0 for value in timings.values())


def test_production_ewma_closes_analysis_workers_and_output_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_executor = qualification.pipeline_mod.ThreadPoolExecutor
    real_output = qualification._BoundedNullOutput
    executors = []
    outputs = []

    class ExecutorSpy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.delegate = real_executor(*args, **kwargs)
            self.shutdown_calls: list[tuple[bool, bool]] = []
            self.submit_calls = 0
            executors.append(self)

        def submit(self, *args: Any, **kwargs: Any) -> Any:
            self.submit_calls += 1
            return self.delegate.submit(*args, **kwargs)

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            self.shutdown_calls.append((wait, cancel_futures))
            self.delegate.shutdown(wait=wait, cancel_futures=cancel_futures)

    class OutputSpy(real_output):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            outputs.append(self)

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(qualification.pipeline_mod, "ThreadPoolExecutor", ExecutorSpy)
    monkeypatch.setattr(qualification, "_BoundedNullOutput", OutputSpy)
    width, height, fps = 256, 144, 30
    geometry_source, foreground, _backdrop, mask = qualification._tier_arrays(
        width, height
    )

    snapshot = qualification._production_ewma_run(
        width=width,
        height=height,
        fps=fps,
        frames=2,
        foreground=foreground,
        geometry_source=geometry_source,
        mask=mask,
        candidate=True,
    )

    assert snapshot["frames_out"] == 2
    assert len(executors) == 1
    executor = executors[0]
    assert executor.submit_calls > 0
    assert executor.shutdown_calls == [(True, True)]
    with pytest.raises(RuntimeError):
        executor.submit(lambda: None)
    assert len(outputs) == 1
    assert outputs[0].close_calls == 1


@pytest.mark.parametrize("fail_memory_probe", (False, True))
def test_calibrated_tier_reuses_and_always_closes_bounded_executor(
    manifest: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    fail_memory_probe: bool,
) -> None:
    instances = []

    class ExecutorSpy:
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            self.max_workers = max_workers
            self.thread_name_prefix = thread_name_prefix
            self.shutdown_calls: list[tuple[bool, bool]] = []
            instances.append(self)

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            self.shutdown_calls.append((wait, cancel_futures))

    timings = {
        "geometry_ms": 0.1,
        "linear_conversion_ms": 0.1,
        "color_analysis_ms": 0.1,
        "composite_ms": 0.1,
        "output_send_ms": 0.1,
        "frame_total_ms": 0.5,
    }
    candidate_executors = []
    candidate_backdrop_analyses = []
    analysis_builds = []
    real_analysis_builder = (
        qualification.color_mod._linear_bgr_analysis_raster_prevalidated
    )

    def build_analysis(value: np.ndarray) -> np.ndarray:
        result = real_analysis_builder(value)
        analysis_builds.append(result)
        return result

    def candidate_frame(
        *_args: Any,
        resize_executor: object,
        backdrop_analysis_linear_bgr: np.ndarray,
    ) -> dict[str, float]:
        candidate_executors.append(resize_executor)
        candidate_backdrop_analyses.append(backdrop_analysis_linear_bgr)
        if fail_memory_probe and len(candidate_executors) == 3:
            raise RuntimeError("memory probe failed")
        return dict(timings)

    monkeypatch.setattr(qualification, "ThreadPoolExecutor", ExecutorSpy)
    monkeypatch.setattr(
        qualification.color_mod,
        "_linear_bgr_analysis_raster_prevalidated",
        build_analysis,
    )
    monkeypatch.setattr(
        qualification,
        "_one_baseline_frame",
        lambda *_args, **_kwargs: dict(timings),
    )
    monkeypatch.setattr(qualification, "_one_candidate_frame", candidate_frame)
    monkeypatch.setattr(
        qualification, "_production_ewma_snapshot", lambda **_kwargs: {}
    )
    monkeypatch.setattr(qualification, "_process_rss_bytes", lambda: 0)

    tier = manifest["performance_tiers"][0]
    if fail_memory_probe:
        with pytest.raises(RuntimeError, match="memory probe failed"):
            qualification._benchmark_tier(
                tier,
                manifest,
                measured_frames=1,
                warmup_frames=1,
            )
    else:
        result = qualification._benchmark_tier(
            tier,
            manifest,
            measured_frames=1,
            warmup_frames=1,
        )
        assert result["measured_frames"] == 1

    assert len(instances) == 1
    executor = instances[0]
    assert executor.max_workers == 3
    assert executor.thread_name_prefix == "custback-qualification-color-analysis"
    assert candidate_executors == [executor, executor, executor]
    assert len(analysis_builds) == 1
    cached_analysis = analysis_builds[0]
    assert max(cached_analysis.shape[:2]) == 192
    assert len(candidate_backdrop_analyses) == 3
    assert all(value is cached_analysis for value in candidate_backdrop_analyses)
    assert executor.shutdown_calls == [(True, True)]
    assert not qualification.tracemalloc.is_tracing()


def _write_physical_artifact(
    root: Path, filename: str, media_type: str, seed: int
) -> dict[str, Any]:
    path = root / filename
    if media_type == "image/png":
        image = np.full((3, 4, 3), seed, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
    else:
        path.write_text(json.dumps({"seed": seed}), encoding="utf-8")
    payload = path.read_bytes()
    return {
        "filename": filename,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "media_type": media_type,
    }


def _passing_physical(manifest: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    webcams = [
        {
            "id": "webcam-a",
            "manufacturer": "Vendor A",
            "model": "Model A",
            "auto_white_balance": "enabled",
            "auto_exposure": "enabled",
        },
        {
            "id": "webcam-b",
            "manufacturer": "Vendor B",
            "model": "Model B",
            "auto_white_balance": "disabled",
            "auto_exposure": "disabled",
        },
    ]
    consumers = [
        {"id": "linux-app", "name": "Meet A", "version": "1", "os": "linux"},
        {"id": "mac-app", "name": "Meet B", "version": "2", "os": "macos"},
        {"id": "win-app", "name": "Meet C", "version": "3", "os": "windows"},
    ]
    consumer_by_os = {consumer["os"]: consumer["id"] for consumer in consumers}
    measurements = {
        "crop_edge_error_pixels": 0,
        "orientation_error_degrees": 0,
        "transport_mean_absolute_error": 0.5,
        "luminance_gap_reduction_percent": 25.0,
        "neutral_error_reduction_percent": 15.0,
        "skin_hue_drift_degrees": 1.0,
        "skin_chroma_drift_percent": 2.0,
        "clothing_hue_drift_degrees": 1.0,
        "clothing_chroma_drift_percent": 2.0,
        "steady_state_ev_delta_p95": 0.01,
        "steady_state_wb_log2_delta_p95": 0.01,
        "no_temporal_oscillation": True,
        "preview_api_equal": True,
        "all_consumers_match": True,
    }
    rows = []
    seed = 1
    for definition in manifest["physical_matrix"]:
        artifacts = {}
        for artifact_id, media_type in manifest["physical_requirements"][
            "required_artifacts"
        ].items():
            filename = f"{definition['id']}-{artifact_id}"
            filename += ".png" if media_type == "image/png" else ".json"
            artifacts[artifact_id] = _write_physical_artifact(
                evidence_root, filename, media_type, seed
            )
            seed += 1
        rows.append(
            {
                **definition,
                "status": "pass",
                "webcam_ids": ["webcam-a", "webcam-b"],
                "consumer_ids": [consumer_by_os[definition["os"]]],
                "artifacts": artifacts,
                "measurements": dict(measurements),
                "notes": "measured by the physical qualification procedure",
            }
        )
    return {
        "status": "pass",
        "webcams": webcams,
        "consumers": consumers,
        "rows": rows,
    }


def test_physical_matrix_binds_backend_consumers_and_unique_artifacts(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    physical = _passing_physical(manifest, tmp_path)
    qualification._validate_physical_release(
        physical,
        manifest,
        evidence_root=tmp_path,
        seen_artifacts=set(),
        seen_digests=set(),
    )
    physical["rows"][0]["transport"] = "unreviewed-transport"
    with pytest.raises(qualification.QualificationError, match="manifest"):
        qualification._validate_physical_release(
            physical,
            manifest,
            evidence_root=tmp_path,
            seen_artifacts=set(),
            seen_digests=set(),
        )


def test_trusted_git_bridge_is_all_or_none_and_exact(
    manifest: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "1" * 40
    tree = "2" * 40
    monkeypatch.setenv("CUSTBACK_RELEASE_GIT_ROOT", str(tmp_path))
    with pytest.raises(qualification.QualificationError, match="supplied together"):
        qualification._release_git_context(commit, tree)

    monkeypatch.setenv("CUSTBACK_RELEASE_SOURCE_COMMIT", "3" * 40)
    monkeypatch.setenv("CUSTBACK_RELEASE_SOURCE_TREE", "4" * 40)

    def git_value(*arguments: str, cwd: Path) -> str:
        del cwd
        mapping: dict[tuple[str, ...], str] = {
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("rev-parse", "HEAD"): "3" * 40,
            ("rev-parse", "HEAD^{tree}"): "4" * 40,
            ("rev-parse", f"{commit}^{{tree}}"): tree,
        }
        return mapping[arguments]

    monkeypatch.setattr(qualification, "_git_value", git_value)
    monkeypatch.setattr(qualification, "_git_status", lambda **_kwargs: "")
    monkeypatch.setattr(
        qualification, "_git_is_ancestor", lambda *_args, **_kwargs: True
    )
    root, bridged, trusted_head = qualification._release_git_context(commit, tree)
    assert root == tmp_path
    assert bridged is True
    assert trusted_head == "3" * 40


def test_release_validation_keeps_historical_source_binding(
    manifest: dict[str, Any],
    deterministic_report: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = copy.deepcopy(deterministic_report)
    report["source"].update({"commit": "1" * 40, "tree": "2" * 40, "clean": True})
    report["candidate"] = {
        "status": "bound",
        "manifest_filename": "candidate.json",
        "manifest_sha256": "3" * 64,
    }
    report["release_qualified"] = True
    soak = {
        "frames": 10_000,
        "tracemalloc_current_growth_bytes": 0,
        "rss_growth_bytes": 0,
    }
    deterministic = {
        "soak": soak,
        "contact_sheet": {
            "filename": "contact.png",
            "sha256": "4" * 64,
            "bytes": 1,
        },
    }
    monkeypatch.setattr(qualification, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        qualification, "_validate_deterministic", lambda *_args: deterministic
    )
    monkeypatch.setattr(
        qualification,
        "_release_git_context",
        lambda *_args: (tmp_path, False, "5" * 40),
    )
    monkeypatch.setattr(
        qualification, "_git_is_ancestor", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(qualification, "_git_value", lambda *_args, **_kwargs: "2" * 40)
    monkeypatch.setattr(qualification, "_git_status", lambda **_kwargs: "")
    monkeypatch.setattr(
        qualification,
        "_git_blob_digest",
        lambda _commit, filename, **_kwargs: report["source"]["files"][filename],
    )
    monkeypatch.setattr(
        qualification,
        "_safe_evidence_file",
        lambda *_args, **_kwargs: tmp_path / "evidence",
    )
    monkeypatch.setattr(
        qualification, "_validate_candidate_manifest", lambda *_args: None
    )
    monkeypatch.setattr(
        qualification, "_validate_artifact_content", lambda *_args: None
    )
    monkeypatch.setattr(
        qualification, "_validate_calibrated_release", lambda *_args: None
    )
    monkeypatch.setattr(
        qualification, "_validate_physical_release", lambda *_args, **_kwargs: None
    )

    def reject_staged_source_read(_path: Path) -> str:
        raise AssertionError("release validation read staged source as historical")

    monkeypatch.setattr(qualification, "_sha256_file", reject_staged_source_read)
    qualification.validate_report(
        report,
        manifest,
        claim="release",
        evidence_root=tmp_path,
        expected_commit="1" * 40,
    )


def test_minimal_cli_requires_only_the_standard_library(tmp_path: Path) -> None:
    output = tmp_path / "template.json"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(_QUALIFICATION_PATH),
            "template",
            "--output",
            str(output),
        ],
        cwd=qualification.ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["physical"]["status"] == (
        "pending"
    )

    invalid_report = tmp_path / "invalid-report.json"
    invalid_report.write_text("{}\n", encoding="utf-8")
    validation = subprocess.run(
        [
            sys.executable,
            "-S",
            str(_QUALIFICATION_PATH),
            "validate",
            "--report",
            str(invalid_report),
            "--claim",
            "deterministic",
        ],
        cwd=qualification.ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
        timeout=30,
    )
    assert validation.returncode == 1
    assert "[custback visual qualification]" in validation.stderr
    assert "ModuleNotFoundError" not in validation.stderr
