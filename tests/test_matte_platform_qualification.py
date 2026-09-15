"""Fail-closed MATTE-5.3 performance/platform qualification tests."""

from __future__ import annotations

import copy
import shutil
import stat
from pathlib import Path
from typing import Any, Callable

import matte_platform_qualification_evidence as evidence
import pytest

import custback.__main__ as core_main
from custback import matte_platform_qualification as qualification


LOCAL_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "qualification"
    / "matte-platform-qualification-local-template.json"
)


@pytest.fixture
def generated(tmp_path: Path) -> evidence.GeneratedPlatformQualification:
    return evidence.generate_qualification(tmp_path / "private-platform-evidence")


@pytest.fixture
def recorded(tmp_path: Path) -> evidence.GeneratedPlatformQualification:
    return evidence.generate_qualification(
        tmp_path / "private-recorded-platform-evidence", recorded=True
    )


def _rewrite_plan(
    generated: evidence.GeneratedPlatformQualification,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    plan = evidence.read_json(generated.plan)
    mutate(plan)
    evidence.write_json(generated.plan, plan)


def _rewrite_run(
    generated: evidence.GeneratedPlatformQualification,
    cell_id: str,
    mutate: Callable[[dict[str, Any]], object],
    *,
    rebind_sources: bool = True,
) -> None:
    run_path = generated.runs[cell_id]
    run = evidence.read_json(run_path)
    mutate(run)
    if rebind_sources:
        samples = run.get("samples")
        sources = run.get("sources")
        if isinstance(samples, dict) and isinstance(sources, dict):
            for source_name, rows in samples.items():
                contract = sources.get(source_name)
                if isinstance(contract, dict):
                    contract["trace_sha256"] = evidence.canonical_digest(rows)
    run = evidence.sign_report(run)
    evidence.write_json(run_path, run)
    plan = evidence.read_json(generated.plan)
    descriptor = _cell(plan, cell_id)["run"]
    assert isinstance(descriptor, dict)
    descriptor.update(evidence.artifact_descriptor(generated.root, run_path))
    evidence.write_json(generated.plan, plan)


def _rewrite_capture(
    generated: evidence.GeneratedPlatformQualification,
    cell_id: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    capture_path = generated.captures[cell_id]
    capture = evidence.read_json(capture_path)
    mutate(capture)
    evidence.write_json(capture_path, capture)
    plan = evidence.read_json(generated.plan)
    descriptor = _cell(plan, cell_id)["capture_report"]
    assert isinstance(descriptor, dict)
    descriptor.update(evidence.file_descriptor(generated.root, capture_path))
    evidence.write_json(generated.plan, plan)


def _replace_prerequisite(
    generated: evidence.GeneratedPlatformQualification,
    name: str,
    report: dict[str, object],
) -> None:
    path = generated.prerequisites[name]
    evidence.write_json(path, report)
    plan = evidence.read_json(generated.plan)
    descriptor = plan["prerequisites"][name]
    assert isinstance(descriptor, dict)
    descriptor.update(evidence.artifact_descriptor(generated.root, path))
    evidence.write_json(generated.plan, plan)


def _mutate_source_compositor(run: dict[str, Any]) -> None:
    for row in run["samples"]["fixed_replay"]:
        capture_ms = row["capture_completed_ms"]
        assert isinstance(capture_ms, float)
        frame_ms = 1000.0 / 30.0
        processing_ms = capture_ms + frame_ms - 32.0
        row.update(
            {
                "compositor_ms": 23.0,
                "frame_processing_ms": 29.0,
                "complete_service_ms": 32.0,
                "processing_started_ms": processing_ms,
                "sink_completed_ms": capture_ms + frame_ms,
                "queue_age_ms": frame_ms - 32.0,
                "end_to_end_age_ms": frame_ms,
                "serialized_cycle_ms": frame_ms,
                "pacing_wait_ms": frame_ms - 32.0,
            }
        )


def _mutate_source_service(run: dict[str, Any]) -> None:
    samples = run["samples"]["fixed_replay"]
    for index, row in enumerate(samples):
        capture_ms = index * 34.0
        processing_ms = capture_ms + 1.0
        sink_ms = processing_ms + 34.0
        row.update(
            {
                "capture_completed_ms": capture_ms,
                "processing_started_ms": processing_ms,
                "sink_completed_ms": sink_ms,
                "complete_service_ms": 34.0,
                "end_to_end_age_ms": 35.0,
                "serialized_cycle_ms": 34.0,
                "pacing_wait_ms": 0.0,
                "schedule_lateness_ms": max(
                    0.0, sink_ms - 35.0 - index * (1000.0 / 30.0)
                ),
                "deadline_miss": True,
            }
        )
    run["scope"]["measured_seconds_by_source"]["fixed_replay"] = (
        (len(samples) - 1) * 34.0 + 35.0
    ) / 1000.0
    run["counters"]["fixed_replay"]["processing_deadline_misses"] = len(samples)


def _mutate_source_e2e_drift(run: dict[str, Any]) -> None:
    for index, row in enumerate(run["samples"]["fixed_replay"][-100:], start=200):
        queue_age = 1.0 + (index - 199) * 0.8
        row["capture_completed_ms"] = row["processing_started_ms"] - queue_age
        row["queue_age_ms"] = queue_age
        row["end_to_end_age_ms"] = queue_age + row["complete_service_ms"]


def _mutate_source_output_rate(run: dict[str, Any]) -> None:
    duration = 11.2
    samples = run["samples"]["fixed_replay"]
    frame_ms = 1000.0 / 30.0
    interval_ms = (duration * 1000.0 - frame_ms) / (len(samples) - 1)
    target_interval_ms = 1000.0 / 30.0
    for index, row in enumerate(samples):
        capture_ms = round(index * interval_ms, 6)
        row["capture_completed_ms"] = capture_ms
        row["processing_started_ms"] = capture_ms + frame_ms - 20.0
        row["sink_completed_ms"] = capture_ms + frame_ms
        row["queue_age_ms"] = frame_ms - 20.0
        row["end_to_end_age_ms"] = frame_ms
        if index > 0:
            row["serialized_cycle_ms"] = interval_ms
            row["pacing_wait_ms"] = interval_ms - 20.0
        row["schedule_lateness_ms"] = max(
            0.0,
            row["sink_completed_ms"] - frame_ms - index * target_interval_ms,
        )
    run["scope"]["measured_seconds_by_source"]["fixed_replay"] = duration


def _mutate_resource_maximum_gap(run: dict[str, Any]) -> None:
    samples = run["resource_samples"]
    soak_seconds = float(qualification.MIN_SOAK_SECONDS)
    rejected_gap = 2.0 * soak_seconds / (len(samples) - 1) + 1.0
    clustered_span = soak_seconds - rejected_gap
    offsets = [
        round(clustered_span * index / (len(samples) - 2), 6)
        for index in range(len(samples) - 1)
    ] + [soak_seconds]
    for row, offset in zip(samples, offsets):
        row["offset_s"] = offset


def _mutate_physical_stall(run: dict[str, Any], *, start: int, stop: int) -> None:
    samples = run["samples"]["physical_capture"]
    frozen = samples[start - 1]
    skipped = stop - start
    for index, row in enumerate(samples[start:], start=start):
        if index < stop:
            sequence = start - 1
            row["capture_completed_ms"] = frozen["capture_completed_ms"]
            row["source_capture_timestamp_ns"] = frozen["source_capture_timestamp_ns"]
        else:
            sequence = index - skipped
        row["capture_sequence"] = sequence
        row["source_capture_sequence"] = sequence
        row["segmentation_sequence"] = sequence
        row["composite_sequence"] = sequence
        row["queue_age_ms"] = row["processing_started_ms"] - row["capture_completed_ms"]
        row["end_to_end_age_ms"] = (
            row["sink_completed_ms"] - row["capture_completed_ms"]
        )
    for previous, current in zip(samples, samples[1:]):
        current["no_unread_repeat"] = (
            current["composite_sequence"] == previous["composite_sequence"]
        )
    run["counters"]["physical_capture"] = evidence.derived_run_counters(samples)


def _profile(plan: dict[str, Any], profile_id: str) -> dict[str, Any]:
    return next(profile for profile in plan["profiles"] if profile["id"] == profile_id)


def _cell(plan: dict[str, Any], cell_id: str) -> dict[str, Any]:
    return next(cell for cell in plan["cells"] if cell["id"] == cell_id)


def test_generated_plan_is_private_deterministic_and_covers_every_required_lane(
    generated: evidence.GeneratedPlatformQualification,
) -> None:
    first, path, first_sha = qualification._load_plan(generated.plan)
    second, second_path, second_sha = qualification._load_plan(generated.plan)

    assert first == second
    assert path == second_path == generated.plan
    assert first_sha == second_sha == evidence.sha256_file(generated.plan)
    assert first["provenance"] == {
        "kind": "generated",
        "license_or_consent_sha256": None,
        "physical_capture_origin_attested": False,
        "physical_consumer_origin_attested": False,
        "same_host_attested": False,
        "physical_origin_cryptographically_proven": False,
    }
    assert {cell["route_id"] for cell in first["cells"]} == set(
        qualification.ROUTE_CONTRACTS
    )
    assert {cell["profile_id"] for cell in first["cells"]} == set(
        qualification.PROFILE_CONTRACTS
    )
    assert {cell["provider"] for cell in first["cells"]} == set(qualification.PROVIDERS)
    assert {cell["dependency_profile"] for cell in first["cells"]} == set(
        qualification.DEPENDENCY_PROFILES
    )
    assert all(cell["state"] == "unavailable" for cell in first["cells"])
    assert stat.S_IMODE(generated.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(generated.plan.stat().st_mode) == 0o600
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in generated.prerequisites.values()
    )


def test_checked_in_local_template_remains_load_plan_compatible(
    tmp_path: Path,
) -> None:
    private_root = tmp_path / "private-template"
    private_root.mkdir(mode=0o700)
    copied = private_root / "plan.json"
    shutil.copyfile(LOCAL_TEMPLATE, copied)
    copied.chmod(0o600)

    loaded, loaded_path, file_sha256 = qualification._load_plan(copied)

    assert loaded_path == copied
    assert file_sha256 == evidence.sha256_file(copied)
    assert loaded["schema"] == qualification.PLAN_SCHEMA
    assert {cell["route_id"] for cell in loaded["cells"]} == set(
        qualification.ROUTE_CONTRACTS
    )
    assert {cell["profile_id"] for cell in loaded["cells"]} == set(
        qualification.PROFILE_CONTRACTS
    )
    assert {cell["provider"] for cell in loaded["cells"]} == set(
        qualification.PROVIDERS
    )
    assert {cell["dependency_profile"] for cell in loaded["cells"]} == set(
        qualification.DEPENDENCY_PROFILES
    )


def test_generated_recorded_matrix_exercises_every_lane_but_only_reports_pending(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    report = qualification.qualify_plan(recorded.plan)

    assert report["status"] == "pending"
    assert report["provenance"] == {
        "qualification_id": "generated-matte-platform-matrix",
        "kind": "generated",
        "physical_origin_authority": "owner-attested",
        "physical_origin_cryptographically_proven": False,
        "candidate_build_authority": "owner-attested-exact-contract-join",
        "candidate_build_cryptographically_bound_to_prerequisites": False,
    }
    assert report["coverage"]["complete"] is False
    assert report["coverage"]["qualified_cells"] == []
    assert report["coverage"]["unavailable_cells"] == []
    assert set(report["coverage"]["pending_cells"]) == set(recorded.runs)
    assert all(cell["outcome"] == "pending" for cell in report["cells"])
    assert all(profile["outcome"] == "pending" for profile in report["profiles"])
    for cell in report["cells"]:
        assert cell["run"]["scope"]["measured_seconds_by_source"] == {
            "physical_capture": qualification.MIN_MEASURED_SECONDS,
            "fixed_replay": qualification.MIN_MEASURED_SECONDS,
        }
        assert set(cell["run"]["measurements"]) == {
            "physical_capture",
            "fixed_replay",
        }
        assert {
            gate["id"]
            for gate in cell["gates"]
            if gate["applicable"] is True and gate["passed"] is False
        } == {
            "capture-only-prerequisite",
            "physical-capture-consumer-same-host-attestation",
        }
    encoded = evidence.json_bytes(report)
    assert str(recorded.root).encode() not in encoded
    assert report["production"] == {
        "defaults_changed": False,
        "preset_catalog_changed": False,
        "generated_evidence_can_qualify": False,
        "reactions_enabled": False,
        "react_budget_qualified_here": False,
    }


def test_missing_profile_route_and_route_cpu_lanes_are_disclosed_fail_closed(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    report = qualification.qualify_plan(recorded.plan)

    assert report["coverage"]["missing_cpu_routes"] == ["windows-msmf-pyvirtualcam"]
    assert "rvm_matting@windows-native" in report["coverage"]["missing_profile_routes"]
    assert (
        "none_passthrough@linux-v4l2-pyvirtualcam"
        in report["coverage"]["missing_profile_routes"]
    )
    missing_by_profile = {
        profile["id"]: profile["missing_routes"] for profile in report["profiles"]
    }
    assert all(missing_by_profile.values())
    assert all(profile["outcome"] == "pending" for profile in report["profiles"])
    assert report["coverage"]["complete"] is False
    assert report["status"] == "pending"


def test_local_provenance_without_physical_attestations_stays_pending(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    plan = evidence.read_json(recorded.plan)
    plan["provenance"].update(
        {
            "kind": "consented-local",
            "license_or_consent_sha256": "a" * 64,
        }
    )
    evidence.write_json(recorded.plan, plan)
    for cell_id in recorded.runs:
        _rewrite_run(
            recorded,
            cell_id,
            lambda run: run["provenance"].update({"kind": "consented-local"}),
        )

    report = qualification.qualify_plan(recorded.plan)

    assert report["status"] == "pending"
    assert report["coverage"]["qualified_cells"] == []
    assert all(cell["outcome"] == "pending" for cell in report["cells"])
    assert (
        "local physical capture/consumer/same-host authority is incomplete"
        in report["reasons"]
    )


def test_generated_recorded_report_is_deterministic_and_owner_private(
    tmp_path: Path,
) -> None:
    first = evidence.generate_qualification(tmp_path / "first", recorded=True)
    second = evidence.generate_qualification(tmp_path / "second", recorded=True)

    first_report = qualification.qualify_plan(first.plan)
    second_report = qualification.qualify_plan(second.plan)

    assert first_report == second_report
    output = tmp_path / "report"
    written = qualification.run_qualification(first.plan, output)
    assert written == first_report
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "qualification.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "qualification.md").stat().st_mode) == 0o600
    assert evidence.read_json(output / "qualification.json") == first_report


def test_report_redacts_raw_distribution_inventory(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    private_distribution = "private-wheel-build-abc"

    def add_private_inventory(run: dict[str, Any]) -> None:
        dependencies = run["dependencies"]
        dependencies["installed_distributions"].append(private_distribution)
        dependencies["installed_distributions"].sort(key=str.casefold)
        dependencies["installed_distribution_sha256"] = evidence.canonical_digest(
            dependencies["installed_distributions"]
        )

    _rewrite_run(recorded, "linux-rvm-cuda", add_private_inventory)

    report = qualification.qualify_plan(recorded.plan)
    run_summary = _cell(report, "linux-rvm-cuda")["run"]

    assert "installed_distributions" not in run_summary["dependencies"]
    assert private_distribution.encode() not in evidence.json_bytes(report)


def test_output_and_private_input_paths_fail_closed(
    recorded: evidence.GeneratedPlatformQualification,
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing-output"
    existing.mkdir(mode=0o700)
    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="output directory already exists",
    ):
        qualification.run_qualification(recorded.plan, existing)

    symlink_output = tmp_path / "symlink-output"
    symlink_output.symlink_to(existing, target_is_directory=True)
    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="output directory already exists",
    ):
        qualification.run_qualification(recorded.plan, symlink_output)

    plan_symlink = tmp_path / "plan-link.json"
    plan_symlink.symlink_to(recorded.plan)
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="regular files"
    ):
        qualification.qualify_plan(plan_symlink)

    recorded.plan.chmod(0o644)
    with pytest.raises(PermissionError, match="not private to its owner"):
        qualification.qualify_plan(recorded.plan)


def test_intermediate_evidence_directory_symlink_is_rejected(
    generated: evidence.GeneratedPlatformQualification,
) -> None:
    real_directory = generated.root / "real-evidence"
    real_directory.mkdir(mode=0o700)
    copied_visual = real_directory / "visual.json"
    shutil.copy2(generated.prerequisites["visual"], copied_visual)
    copied_visual.chmod(0o600)
    linked_directory = generated.root / "linked-evidence"
    linked_directory.symlink_to(real_directory, target_is_directory=True)

    _rewrite_plan(
        generated,
        lambda plan: plan["prerequisites"]["visual"].update(
            {"path": "linked-evidence/visual.json"}
        ),
    )

    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="must not traverse a symlink",
    ):
        qualification._load_plan(generated.plan)


def test_platform_cli_returns_pending_and_invalid_exit_codes_without_path_leaks(
    recorded: evidence.GeneratedPlatformQualification,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = qualification.main(
        [str(recorded.plan), "--output", str(tmp_path / "pending-report")]
    )
    captured = capsys.readouterr()
    assert result == 1
    assert "status pending" in captured.out
    assert str(recorded.root) not in captured.out + captured.err

    missing = tmp_path / "private-secret-plan-name.json"
    result = qualification.main(
        [str(missing), "--output", str(tmp_path / "invalid-report")]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert "bundle artifact is missing" in captured.err
    assert "private-secret-plan-name" not in captured.err


def test_core_cli_dispatches_platform_qualification_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str] | None, str]] = []

    def platform_main(argv: list[str] | None = None, *, prog: str) -> int:
        calls.append((argv, prog))
        return 23

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("normal runtime must not initialize")

    monkeypatch.setattr(qualification, "main", platform_main)
    monkeypatch.setattr(core_main, "Pipeline", forbidden)

    result = core_main.main(
        ["matte-platform-qualify", "private-plan", "--output", "private-report"]
    )

    assert result == 23
    assert calls == [
        (
            ["private-plan", "--output", "private-report"],
            "custback matte-platform-qualify",
        )
    ]


def test_unavailable_required_row_remains_visible_and_cannot_qualify(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    plan = evidence.read_json(recorded.plan)
    cell = _cell(plan, "windows-native-heuristic")
    cell.update(
        {
            "state": "unavailable",
            "reason": "physical-route-unavailable",
            "run": None,
            "capture_report": None,
        }
    )
    evidence.write_json(recorded.plan, plan)

    report = qualification.qualify_plan(recorded.plan)

    assert report["status"] == "pending"
    assert report["coverage"]["unavailable_cells"] == ["windows-native-heuristic"]
    row = next(
        cell for cell in report["cells"] if cell["id"] == "windows-native-heuristic"
    )
    assert row["outcome"] == "unavailable"
    assert row["gates"] == []


def test_prerequisite_file_and_internal_digests_fail_closed(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    visual_path = recorded.prerequisites["visual"]
    visual_path.write_bytes(visual_path.read_bytes() + b" ")
    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="private evidence digest does not match",
    ):
        qualification.qualify_plan(recorded.plan)

    recorded = evidence.generate_qualification(
        recorded.root.parent / "internal-digest", recorded=True
    )
    visual_path = recorded.prerequisites["visual"]
    visual = evidence.read_json(visual_path)
    visual["authority"] = "edited-without-resigning"
    evidence.write_json(visual_path, visual)
    plan = evidence.read_json(recorded.plan)
    plan["prerequisites"]["visual"]["sha256"] = evidence.sha256_file(visual_path)
    evidence.write_json(recorded.plan, plan)
    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="intact MATTE-5.2 report",
    ):
        qualification.qualify_plan(recorded.plan)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report.update({"schema": "not-matte-5.2"}),
            "intact MATTE-5.2 report",
        ),
        (
            lambda report: report.update({"status": "qualified-ish"}),
            "visual prerequisite status is invalid",
        ),
        (
            lambda report: report["algorithm_manifest"][0].update(
                {"contract_sha256": "f" * 64}
            ),
            "visual prerequisite does not bind",
        ),
        (
            lambda report: report["algorithm_manifest"][0].update(
                {"id": "wrong-candidate"}
            ),
            "visual prerequisite does not bind",
        ),
    ],
)
def test_visual_prerequisite_schema_status_candidate_and_algorithm_are_exact(
    recorded: evidence.GeneratedPlatformQualification,
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    visual = evidence.read_json(recorded.prerequisites["visual"])
    mutate(visual)
    _replace_prerequisite(recorded, "visual", evidence.sign_report(visual))

    with pytest.raises(qualification.MattePlatformQualificationError, match=message):
        qualification.qualify_plan(recorded.plan)


def test_performance_effective_policy_must_match_declared_rvm_tier(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    performance_report = evidence.generated_performance_with_full_path(
        evidence.generated_balanced_effective_policy()
    )
    _replace_prerequisite(recorded, "performance", performance_report)

    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="performance policy does not match the RVM tier",
    ):
        qualification.qualify_plan(recorded.plan)


def test_qualified_rvm_prerequisite_must_bind_the_same_visual_candidate(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    rvm = evidence.read_json(recorded.prerequisites["rvm"])
    rvm["profiles"].update(
        {
            "status": "qualified",
            "definitions": [{"candidate_id": "wrong-rvm-candidate"}],
        }
    )
    _replace_prerequisite(recorded, "rvm", evidence.sign_rvm_report(rvm))

    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="qualified RVM prerequisite does not bind",
    ):
        qualification.qualify_plan(recorded.plan)


def test_rvm_scope_compatibility_is_narrow_and_device_ordinal_is_exact(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    plan, _plan_path, _plan_sha256 = qualification._load_plan(recorded.plan)
    _prerequisites, _statuses, base_authorizations = qualification._load_prerequisites(
        plan
    )
    profiles = {profile["id"]: profile for profile in plan["profiles"]}

    linux_gpu_cell = _cell(plan, "linux-rvm-cuda")
    capture, _capture_outcome, _capture_passed = qualification._capture_report(
        linux_gpu_cell["capture_report"], linux_gpu_cell
    )
    linux_cpu_cell = copy.deepcopy(linux_gpu_cell)
    linux_cpu_cell.update({"id": "linux-rvm-cpu-scope", "provider": "cpu"})
    profile = profiles["rvm_matting"]
    capture_file_sha256 = capture["file_sha256"]
    assert isinstance(capture_file_sha256, str)
    cpu_run = evidence.generated_run_report(
        plan=plan,
        cell=linux_cpu_cell,
        profile=profile,
        capture_file_sha256=capture_file_sha256,
    )
    cpu_run_path = recorded.root / "run-linux-rvm-cpu-scope.json"
    evidence.write_json(cpu_run_path, cpu_run)
    cpu_descriptor = {
        "path": cpu_run_path,
        "sha256": evidence.sha256_file(cpu_run_path),
        "evidence_sha256": cpu_run["evidence_sha256"],
    }

    def scope_for(
        run: dict[str, Any], cell: dict[str, Any], *, device_id: int
    ) -> dict[str, Any]:
        return {
            "hardware": {
                "identity_sha256": run["hardware"]["identity_sha256"],
                "platform": "linux-x86_64",
            },
            "provider": {
                "requested": run["provider"]["name"],
                "active": run["provider"]["name"],
                "execution_proven": True,
                "fallback_observed": False,
                "device_id": device_id,
                "environment_sha256": run["runtime"]["provider_environment_sha256"],
            },
            "canvas": copy.deepcopy(cell["canvas"]),
            "cadences": ["native30"],
            "render_modes": ["qualified_compositor"],
        }

    def load(
        descriptor: dict[str, Any],
        cell: dict[str, Any],
        run: dict[str, Any],
        scope: dict[str, Any],
    ) -> dict[str, Any]:
        authorizations = dict(base_authorizations)
        authorizations["rvm_definition"] = {
            "model": copy.deepcopy(run["model"]),
            "qualification_scope": [scope],
        }
        return qualification._load_run(
            descriptor,
            plan=plan,
            cell=cell,
            profile=profile,
            capture=capture,
            authorizations=authorizations,
        )

    cpu_scope = scope_for(cpu_run, linux_cpu_cell, device_id=0)
    loaded_cpu = load(cpu_descriptor, linux_cpu_cell, cpu_run, cpu_scope)
    assert loaded_cpu["provider"]["device_id"] is None

    mismatch_mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda scope: scope["hardware"].update({"identity_sha256": "f" * 64}),
        lambda scope: scope["provider"].update({"environment_sha256": "e" * 64}),
        lambda scope: scope["canvas"].update({"width": 1920}),
        lambda scope: scope.update({"cadences": ["native29"]}),
        lambda scope: scope.update({"render_modes": ["preview"]}),
    )
    for mutate in mismatch_mutations:
        mismatched_scope = copy.deepcopy(cpu_scope)
        mutate(mismatched_scope)
        with pytest.raises(
            qualification.MattePlatformQualificationError,
            match="outside the selected RVM definition qualification scope",
        ):
            load(cpu_descriptor, linux_cpu_cell, cpu_run, mismatched_scope)

    gpu_run = evidence.read_json(recorded.runs["linux-rvm-cuda"])
    gpu_scope = scope_for(gpu_run, linux_gpu_cell, device_id=1)
    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="outside the selected RVM definition qualification scope",
    ):
        load(linux_gpu_cell["run"], linux_gpu_cell, gpu_run, gpu_scope)


def test_prerequisite_and_cell_artifacts_cannot_be_reused(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    plan = evidence.read_json(recorded.plan)
    plan["prerequisites"]["rvm"] = copy.deepcopy(plan["prerequisites"]["visual"])
    evidence.write_json(recorded.plan, plan)
    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="artifacts must be distinct",
    ):
        qualification._load_plan(recorded.plan)

    recorded = evidence.generate_qualification(
        recorded.root.parent / "duplicate-cell-artifact", recorded=True
    )
    plan = evidence.read_json(recorded.plan)
    plan["cells"][1]["run"] = copy.deepcopy(plan["cells"][0]["run"])
    evidence.write_json(recorded.plan, plan)
    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="distinct run artifacts",
    ):
        qualification._load_plan(recorded.plan)


def test_capture_report_recomputes_summary_instead_of_trusting_submitted_fps(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    cell_id = "linux-rvm-cuda"
    _rewrite_capture(
        recorded,
        cell_id,
        lambda capture: capture["timing"].update({"active_capture_fps": 999.0}),
    )
    plan, _, _ = qualification._load_plan(recorded.plan)
    cell = _cell(plan, cell_id)

    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="active capture FPS was not derived",
    ):
        qualification._capture_report(cell["capture_report"], cell)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda capture: capture.update({"version": True}),
            "capture prerequisite schema is invalid",
        ),
        (
            lambda capture: capture["privacy"].update({"contains_pixels": True}),
            "capture prerequisite is not private",
        ),
    ],
)
def test_capture_schema_and_privacy_cannot_use_bool_or_content_claims(
    recorded: evidence.GeneratedPlatformQualification,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    cell_id = "linux-rvm-cuda"
    _rewrite_capture(recorded, cell_id, mutate)

    with pytest.raises(qualification.MattePlatformQualificationError, match=message):
        qualification.qualify_plan(recorded.plan)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda run: run["samples"].update(
                {"physical_capture": copy.deepcopy(run["samples"]["fixed_replay"])}
            ),
            "measurements must be independent",
        ),
        (
            lambda run: run["sources"]["fixed_replay"]["measured_lineage"][3].update(
                {"capture_timestamp_ns": 1}
            ),
            "lineage does not reproduce MATTE-3.4 authority",
        ),
        (
            lambda run: run["dependencies"]["installed_distributions"].append(
                "rogue-package"
            ),
            "dependency digest does not bind",
        ),
    ],
)
def test_run_sources_lineage_and_dependency_inventory_are_digest_bound(
    recorded: evidence.GeneratedPlatformQualification,
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    _rewrite_run(recorded, "linux-rvm-cuda", mutate)

    with pytest.raises(qualification.MattePlatformQualificationError, match=message):
        qualification.qualify_plan(recorded.plan)


@pytest.mark.parametrize(
    ("official_outcome", "expected_outcome", "expected_passed"),
    [
        ("hardware-evidence-required", "pending", False),
        ("hardware-actionable-limitation", "pending", False),
        ("hardware-target-sustained", "qualified", True),
    ],
)
def test_capture_uses_official_disposition_and_only_target_sustained_can_pass(
    recorded: evidence.GeneratedPlatformQualification,
    official_outcome: str,
    expected_outcome: str,
    expected_passed: bool,
) -> None:
    cell_id = "linux-rvm-cuda"

    def set_disposition(capture: dict[str, Any]) -> None:
        condition = capture["condition"]
        assert isinstance(condition, dict)
        condition.update(
            {
                "hardware_verified": True,
                "source_kind": "integer-camera-index",
                "physical_source_eligible": True,
                "device_identity_sha256": "1" * 64,
                "hardware_identity_sha256": "2" * 64,
                "device_identity_bound": True,
                "hardware_identity_bound": True,
            }
        )
        diagnosis = capture["diagnosis"]
        assert isinstance(diagnosis, dict)
        diagnosis.update(
            {
                "code": (
                    "target-sustained"
                    if official_outcome == "hardware-target-sustained"
                    else "normalization-budget-limited"
                    if official_outcome == "hardware-actionable-limitation"
                    else "unresolved-capture-scheduling-or-backend-limit"
                ),
                "target_sustained": official_outcome == "hardware-target-sustained",
                "actionable": official_outcome
                in (
                    "hardware-target-sustained",
                    "hardware-actionable-limitation",
                ),
            }
        )
        disposition = capture["qualification"]
        assert isinstance(disposition, dict)
        disposition.update(
            {
                "hardware_verified": True,
                "acceptance_satisfied": official_outcome
                != "hardware-evidence-required",
                "outcome": official_outcome,
            }
        )

    _rewrite_capture(recorded, cell_id, set_disposition)
    plan, _, _ = qualification._load_plan(recorded.plan)
    cell = _cell(plan, cell_id)

    summary, outcome, passed = qualification._capture_report(
        cell["capture_report"], cell
    )

    assert summary["matte31_outcome"] == official_outcome
    assert outcome == expected_outcome
    assert passed is expected_passed


@pytest.mark.parametrize(
    ("leading_gap_ms", "trailing_gap_ms"),
    [(70.0, 0.0), (0.0, 70.0)],
    ids=("leading-burst-starvation", "trailing-starvation"),
)
def test_capture_window_boundary_starvation_fails_even_with_healthy_active_rate(
    recorded: evidence.GeneratedPlatformQualification,
    leading_gap_ms: float,
    trailing_gap_ms: float,
) -> None:
    cell_id = "linux-rvm-cuda"

    def starve_boundary(capture: dict[str, Any]) -> None:
        evidence.rebind_capture_window(
            capture,
            completion_span_ms=4930.0,
            leading_gap_ms=leading_gap_ms,
            trailing_gap_ms=trailing_gap_ms,
        )
        condition = capture["condition"]
        assert isinstance(condition, dict)
        condition.update(
            {
                "hardware_verified": True,
                "physical_source_eligible": True,
                "device_identity_sha256": "1" * 64,
                "hardware_identity_sha256": "2" * 64,
                "device_identity_bound": True,
                "hardware_identity_bound": True,
                "source_kind": "integer-camera-index",
            }
        )
        capture["qualification"].update(
            {
                "hardware_verified": True,
                "acceptance_satisfied": False,
                "outcome": "hardware-evidence-required",
            }
        )

    _rewrite_capture(recorded, cell_id, starve_boundary)
    plan, _, _ = qualification._load_plan(recorded.plan)
    cell = _cell(plan, cell_id)

    summary, outcome, passed = qualification._capture_report(
        cell["capture_report"], cell
    )

    assert summary["availability_fps"] == 30.0
    assert outcome == "failed"
    assert passed is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda run: run.update({"version": True}),
            "platform run evidence schema is invalid",
        ),
        (
            lambda run: run["candidate"].update({"effective_policy_sha256": "f" * 64}),
            "visual/configured/effective tier",
        ),
        (
            lambda run: run["platform"].update({"consumer": "fake-consumer"}),
            "platform contract",
        ),
        (
            lambda run: run["capture_binding"].update(
                {"capture_file_sha256": "f" * 64}
            ),
            "bound to its capture-only report",
        ),
        (
            lambda run: run["samples"].pop("fixed_replay"),
            "run samples must contain exactly",
        ),
        (
            lambda run: run["counters"]["fixed_replay"].update(
                {"unique_composites": 1}
            ),
            "fixed_replay counters were not derived",
        ),
        (
            lambda run: run["scope"].update({"post_base_event_count": True}),
            "scope disagrees or enables reactions",
        ),
    ],
)
def test_recorded_run_rejects_candidate_route_capture_and_dual_source_lies(
    recorded: evidence.GeneratedPlatformQualification,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    cell_id = "linux-rvm-cuda"
    _rewrite_run(recorded, cell_id, mutate)

    with pytest.raises(qualification.MattePlatformQualificationError, match=message):
        qualification.qualify_plan(recorded.plan)


@pytest.mark.parametrize(
    "cell_id",
    ["linux-heuristic-without-mediapipe", "macos-passthrough-without-gpu"],
)
def test_missing_dependency_fallback_must_be_visible_exact_and_suppress_slow_tier(
    recorded: evidence.GeneratedPlatformQualification,
    cell_id: str,
) -> None:
    _rewrite_run(
        recorded,
        cell_id,
        lambda run: run["selection"].update({"slow_profile_suppressed": False}),
    )

    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="fallback is not visible and exact",
    ):
        qualification.qualify_plan(recorded.plan)


@pytest.mark.parametrize(
    ("mutate", "failed_gate"),
    [
        (
            lambda run: [
                row.update({"rss_bytes": row["rss_bytes"] + 65 * 1024 * 1024})
                for row in run["resource_samples"][-2:]
            ],
            "rss-drift",
        ),
        (
            lambda run: run["resource_samples"][-1].update({"offset_s": 1799.0}),
            "resource-soak-duration",
        ),
        (
            lambda run: run["lifecycle"]["restart"].update({"stale_state_flash": True}),
            "restart-reset-and-fresh-output",
        ),
        (
            lambda run: run["lifecycle"]["history"].update({"previous_mask_slots": 3}),
            "bounded-temporal-history",
        ),
        (
            lambda run: [
                row.update({"serialized_cycle_ms": 33.34})
                for row in run["samples"]["fixed_replay"]
            ],
            "fixed_replay-serialized-cycle-p95",
        ),
        (
            lambda run: run["resource_samples"][-3].update(
                {"rss_bytes": 256 * 1024 * 1024 + 257 * 1024 * 1024}
            ),
            "rss-observed-span",
        ),
        (
            lambda run: [
                row.update({"gpu_percent": 0.0}) for row in run["resource_samples"]
            ],
            "accelerator-activity-observed",
        ),
        (_mutate_source_compositor, "fixed_replay-compositor-p95-720p"),
        (
            lambda run: [
                row.update({"refinement_ms": 6.0, "frame_processing_ms": 16.0})
                for row in run["samples"]["fixed_replay"]
            ],
            "fixed_replay-refinement-p95-720p",
        ),
        (_mutate_source_e2e_drift, "fixed_replay-end-to-end-age-drift"),
        (_mutate_source_output_rate, "fixed_replay-output-rate"),
    ],
)
def test_measured_resource_and_lifecycle_regressions_fail_the_cell(
    recorded: evidence.GeneratedPlatformQualification,
    mutate: Callable[[dict[str, Any]], object],
    failed_gate: str,
) -> None:
    cell_id = "linux-rvm-cuda"
    _rewrite_run(recorded, cell_id, mutate)

    report = qualification.qualify_plan(recorded.plan)

    assert report["status"] == "failed"
    cell = _cell(report, cell_id)
    assert cell["outcome"] == "failed"
    gate = next(gate for gate in cell["gates"] if gate["id"] == failed_gate)
    assert gate["passed"] is False


def test_short_physical_trace_fails_cleanly_instead_of_crashing(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    def shorten(run: dict[str, Any]) -> None:
        rows = run["samples"]["physical_capture"][:1]
        run["samples"]["physical_capture"] = rows
        run["counters"]["physical_capture"] = evidence.derived_run_counters(rows)
        run["scope"]["measured_seconds_by_source"]["physical_capture"] = (
            rows[0]["sink_completed_ms"] / 1000.0
        )

    _rewrite_run(recorded, "linux-rvm-cuda", shorten)

    report = qualification.qualify_plan(recorded.plan)

    assert report["status"] == "failed"
    gate = next(
        gate
        for gate in _cell(report, "linux-rvm-cuda")["gates"]
        if gate["id"] == "physical_capture-sample-count"
    )
    assert gate["passed"] is False


@pytest.mark.parametrize(
    ("start", "stop"),
    [(120, 150), (270, 300)],
    ids=("mid-window-stall", "terminal-stall"),
)
def test_physical_unique_frame_stalls_cannot_hide_behind_output_submissions(
    recorded: evidence.GeneratedPlatformQualification,
    start: int,
    stop: int,
) -> None:
    _rewrite_run(
        recorded,
        "linux-rvm-cuda",
        lambda run: _mutate_physical_stall(run, start=start, stop=stop),
    )

    report = qualification.qualify_plan(recorded.plan)

    cell = _cell(report, "linux-rvm-cuda")
    assert cell["outcome"] == "failed"
    unique_gap = next(
        gate
        for gate in cell["gates"]
        if gate["id"] == "physical_capture-unique-capture-maximum-gap"
    )
    assert unique_gap["passed"] is False


def test_serialized_sample_overlap_is_rejected_before_percentile_gates() -> None:
    samples = evidence.generated_run_samples()
    samples[1].update(
        {
            "capture_completed_ms": 32.0,
            "processing_started_ms": 33.2,
            "sink_completed_ms": 53.2,
            "queue_age_ms": 1.2,
            "end_to_end_age_ms": 21.2,
        }
    )

    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="serialized run samples overlap",
    ):
        qualification._load_samples(
            samples,
            source_name="fixed_replay",
            provider="cpu",
            logical_cpu_count=8,
            measured_seconds=10.0,
            target_fps=30,
            rvm=True,
        )


def test_generated_raw_run_samples_derive_rates_percentiles_and_counters() -> None:
    samples = evidence.generated_run_samples(provider="cuda", rvm=True)

    normalized, derived = qualification._load_samples(
        samples,
        source_name="fixed_replay",
        provider="cuda",
        logical_cpu_count=8,
        measured_seconds=10.0,
        target_fps=30,
        rvm=True,
    )

    assert normalized == samples
    assert derived["counters"] == {
        "measured_frames": 300,
        "unique_capture_frames": 300,
        "unique_segmentations": 300,
        "unique_composites": 300,
        "sink_submissions": 300,
        "capture_gaps": 0,
        "capture_drops": 0,
        "capture_slot_overwrites": 0,
        "processing_deadline_misses": 0,
        "sink_recoveries": 0,
        "no_unread_repeats": 0,
    }
    assert derived["rates"] == {
        "capture_fps": 30.0,
        "segmentation_fps": 30.0,
        "unique_composite_fps": 30.0,
        "output_fps": 30.0,
    }
    assert derived["timings"]["complete_service_ms"] == {
        "count": 300,
        "p50": 20.0,
        "p95": 20.0,
        "p99": 20.0,
        "max": 20.0,
    }
    assert derived["timings"]["inter_frame_jitter_ms"]["p99"] == pytest.approx(0.000001)
    assert derived["e2e_age_p95_drift_ms"] == 0.0


def test_generated_run_sources_are_independent_and_each_counter_derived() -> None:
    sample_sets = evidence.generated_run_sample_sets(provider="cpu", rvm=False)

    assert set(sample_sets) == {"physical_capture", "fixed_replay"}
    assert sample_sets["physical_capture"] is not sample_sets["fixed_replay"]
    sample_sets["physical_capture"][0]["rss_bytes"] = 128 * 1024 * 1024
    assert sample_sets["fixed_replay"][0]["rss_bytes"] == 256 * 1024 * 1024
    for source_name, samples in sample_sets.items():
        _, derived = qualification._load_samples(
            samples,
            source_name=source_name,
            provider="cpu",
            logical_cpu_count=8,
            measured_seconds=10.0,
            target_fps=30,
            rvm=False,
        )
        assert evidence.derived_run_counters(samples) == derived["counters"]


def test_generated_run_source_lineage_and_inventory_bindings_are_exact(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    for run_path in recorded.runs.values():
        run = evidence.read_json(run_path)
        samples = run["samples"]
        sources = run["sources"]
        assert sources["physical_capture"]["trace_sha256"] == (
            evidence.canonical_digest(samples["physical_capture"])
        )
        assert sources["fixed_replay"]["trace_sha256"] == (
            evidence.canonical_digest(samples["fixed_replay"])
        )
        assert (
            sources["physical_capture"]["trace_sha256"]
            != sources["fixed_replay"]["trace_sha256"]
        )
        fixed = sources["fixed_replay"]
        lineage_authority = {
            "schema": "custback.matte-performance-measured-lineage",
            "version": 1,
            "source_sha256": fixed["source_sha256"],
            "warmup_frame_count": fixed["warmup_frame_count"],
            "measured_frame_count": fixed["measured_frame_count"],
            "frames": fixed["measured_lineage"],
        }
        assert fixed["measured_frame_lineage_sha256"] == (
            evidence.canonical_digest(lineage_authority)
        )
        dependencies = run["dependencies"]
        inventory = dependencies["installed_distributions"]
        assert inventory == sorted(inventory, key=str.casefold)
        assert dependencies["installed_distribution_sha256"] == (
            evidence.canonical_digest(inventory)
        )
        selected_provider = qualification.PROVIDER_RUNTIME_NAMES[
            run["provider"]["name"]
        ]
        assert selected_provider in dependencies["available_providers"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda rows: rows[4].update({"index": 5}),
            "indexes must be contiguous",
        ),
        (
            lambda rows: rows[4].update({"complete_service_ms": 19.0}),
            "complete service was not derived",
        ),
        (
            lambda rows: rows[4].update({"deadline_miss": True}),
            "deadline flags were not derived",
        ),
        (
            lambda rows: rows[4].update({"composite_sequence": 8}),
            "composite sequence has an unexplained gap",
        ),
        (
            lambda rows: rows[4].update({"no_unread_repeat": True}),
            "repeat flags do not match",
        ),
        (
            lambda rows: rows[4].update({"pacing_wait_ms": 14.0}),
            "timing scopes are arithmetically inconsistent",
        ),
        (
            lambda rows: rows[4].update({"capture_sequence": 3}),
            "capture identity and completion time disagree",
        ),
        (
            lambda rows: rows[4].update({"frame_processing_ms": 10.0}),
            "timing scopes are arithmetically inconsistent",
        ),
        (
            lambda rows: rows[4].update({"model_inference_ms": 5.0}),
            "timing scopes are arithmetically inconsistent",
        ),
    ],
)
def test_raw_run_samples_reject_submitted_derived_or_identity_lies(
    mutate: Callable[[list[dict[str, object]]], None],
    message: str,
) -> None:
    samples = evidence.generated_run_samples()
    mutate(samples)

    with pytest.raises(qualification.MattePlatformQualificationError, match=message):
        qualification._load_samples(
            samples,
            source_name="fixed_replay",
            provider="cpu",
            logical_cpu_count=8,
            measured_seconds=10.0,
            target_fps=30,
            rvm=True,
        )


def test_declared_measurement_window_cannot_inflate_sample_rates() -> None:
    samples = evidence.generated_run_samples()

    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="span the declared measurement window",
    ):
        qualification._load_samples(
            samples,
            source_name="fixed_replay",
            provider="cpu",
            logical_cpu_count=8,
            measured_seconds=1.0,
            target_fps=30,
            rvm=True,
        )


def test_raw_run_samples_require_provider_specific_resource_observations() -> None:
    cpu_samples = evidence.generated_run_samples(provider="cpu")
    cpu_samples[0]["gpu_utilization_percent"] = 10.0
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="CPU samples"
    ):
        qualification._load_samples(
            cpu_samples,
            source_name="fixed_replay",
            provider="cpu",
            logical_cpu_count=8,
            measured_seconds=10.0,
            target_fps=30,
            rvm=True,
        )

    gpu_samples = evidence.generated_run_samples(provider="cuda")
    gpu_samples[0]["vram_bytes"] = None
    with pytest.raises(qualification.MattePlatformQualificationError, match="VRAM"):
        qualification._load_samples(
            gpu_samples,
            source_name="fixed_replay",
            provider="cuda",
            logical_cpu_count=8,
            measured_seconds=10.0,
            target_fps=30,
            rvm=True,
        )


@pytest.mark.parametrize("provider", ["cpu", "cuda", "directml"])
def test_generated_resource_samples_derive_stable_provider_specific_trends(
    provider: str,
) -> None:
    raw = evidence.generated_resource_samples(provider=provider)

    normalized, derived = qualification._load_resource_samples(
        raw, provider=provider, logical_cpu_count=8
    )

    assert normalized == raw
    assert derived["sample_count"] == qualification.MIN_RESOURCE_SAMPLES
    assert derived["span_s"] == qualification.MIN_SOAK_SECONDS
    rss = derived["rss_bytes"]
    assert isinstance(rss, dict)
    assert rss["signed_drift_bytes"] == 0.0
    if provider == "cpu":
        assert derived["gpu_percent"] is None
        assert derived["vram_bytes"] is None
    else:
        gpu = derived["gpu_percent"]
        vram = derived["vram_bytes"]
        assert isinstance(gpu, dict)
        assert isinstance(vram, dict)
        assert gpu["p95"] == 50.0
        assert vram["signed_drift_bytes"] == 0.0


def test_resource_samples_reject_time_and_provider_contradictions() -> None:
    raw = evidence.generated_resource_samples()
    raw[0]["offset_s"] = 1.0
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="zero origin"
    ):
        qualification._load_resource_samples(raw, provider="cpu", logical_cpu_count=8)

    raw = evidence.generated_resource_samples()
    raw[2]["offset_s"] = raw[1]["offset_s"]
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="offsets must increase"
    ):
        qualification._load_resource_samples(raw, provider="cpu", logical_cpu_count=8)

    raw = evidence.generated_resource_samples()
    raw[0]["gpu_percent"] = 1.0
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="GPU fields inapplicable"
    ):
        qualification._load_resource_samples(raw, provider="cpu", logical_cpu_count=8)


def test_resource_samples_reject_sparse_or_bursty_observation_cadence() -> None:
    run: dict[str, Any] = {"resource_samples": evidence.generated_resource_samples()}
    _mutate_resource_maximum_gap(run)

    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="bounded uniform cadence",
    ):
        qualification._load_resource_samples(
            run["resource_samples"], provider="cpu", logical_cpu_count=8
        )


def test_generated_lifecycle_records_exact_stable_restart_patch_shutdown_order() -> (
    None
):
    raw = evidence.generated_lifecycle()

    lifecycle = qualification._load_lifecycle(raw, provider="cpu")

    assert lifecycle["counter_snapshots"] == raw["counter_snapshots"]
    assert lifecycle["soak_rates"] == {
        "unique_capture_frames": 30.0,
        "unique_segmentations": 30.0,
        "unique_composites": 30.0,
        "sink_submissions": 30.0,
    }
    assert set(lifecycle["soak_event_counts"].values()) == {0}
    assert [event["kind"] for event in lifecycle["events"]] == [
        "sustained",
        "restart",
        "hot_patch",
        "shutdown",
    ]
    assert [event["generation"] for event in lifecycle["events"]] == [1, 2, 3, 3]


def test_lifecycle_heartbeat_must_sustain_every_observed_interval(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    def starve_heartbeat(run: dict[str, Any]) -> None:
        snapshots = run["lifecycle"]["counter_snapshots"]
        previous = snapshots[4]
        starved = snapshots[5]
        for name in (
            "unique_capture_frames",
            "unique_segmentations",
            "unique_composites",
            "sink_submissions",
        ):
            starved[name] = previous[name]

    _rewrite_run(recorded, "linux-rvm-cuda", starve_heartbeat)

    report = qualification.qualify_plan(recorded.plan)
    lifecycle = _cell(report, "linux-rvm-cuda")["run"]["lifecycle"]
    assert lifecycle["minimum_interval_rates"]["sink_submissions"] == 0.0
    gate = next(
        gate
        for gate in _cell(report, "linux-rvm-cuda")["gates"]
        if gate["id"] == "sustained-soak-throughput"
    )
    assert gate["passed"] is False


def test_gpu_lifecycle_requires_exact_visible_fallback_and_recovery(
    recorded: evidence.GeneratedPlatformQualification,
) -> None:
    _rewrite_run(
        recorded,
        "linux-rvm-cuda",
        lambda run: run["lifecycle"]["restart"].update(
            {"fallback_occurred": False, "fallback_visible": False}
        ),
    )

    with pytest.raises(
        qualification.MattePlatformQualificationError,
        match="runtime provider restart/fallback recovery is not exact",
    ):
        qualification.qualify_plan(recorded.plan)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda lifecycle: lifecycle["events"].reverse(),
            "record sustained, restart, hot patch, and shutdown in order",
        ),
        (
            lambda lifecycle: lifecycle["events"][1].update({"generation": 1}),
            "generation transitions",
        ),
        (
            lambda lifecycle: lifecycle["restart"].update({"fallback_occurred": True}),
            "fallback visibility",
        ),
        (
            lambda lifecycle: lifecycle["hot_patch"].update({"transactional": 1}),
            "must be boolean",
        ),
    ],
)
def test_lifecycle_rejects_order_generation_fallback_and_type_lies(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    lifecycle = evidence.generated_lifecycle()
    mutate(lifecycle)

    with pytest.raises(qualification.MattePlatformQualificationError, match=message):
        qualification._load_lifecycle(lifecycle, provider="cpu")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: plan["routes"].pop(),
        lambda plan: plan["routes"].reverse(),
        lambda plan: plan["routes"][0].update({"consumer": "fake-loopback"}),
        lambda plan: plan["profiles"].pop(),
        lambda plan: plan["profiles"][0].update({"quality_claim": False}),
        lambda plan: plan["cells"].pop(),
        lambda plan: plan["cells"].append(copy.deepcopy(plan["cells"][0])),
    ],
)
def test_catalog_and_matrix_shape_are_exact_and_non_vacuous(
    generated: evidence.GeneratedPlatformQualification,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    _rewrite_plan(generated, mutate)
    with pytest.raises(qualification.MattePlatformQualificationError):
        qualification._load_plan(generated.plan)


def test_profile_limits_must_cover_each_cell_once_under_the_owning_profile(
    generated: evidence.GeneratedPlatformQualification,
) -> None:
    def duplicate(plan: dict[str, Any]) -> None:
        rvm = _profile(plan, "rvm_matting")
        rvm["limits"].append(rvm["limits"][0])

    _rewrite_plan(generated, duplicate)
    with pytest.raises(qualification.MattePlatformQualificationError, match="limits"):
        qualification._load_plan(generated.plan)

    generated = evidence.generate_qualification(
        generated.root.parent / "private-platform-evidence-wrong-owner"
    )

    def wrong_owner(plan: dict[str, Any]) -> None:
        heuristic = _profile(plan, "heuristic_segmentation")
        passthrough = _profile(plan, "none_passthrough")
        moved = heuristic["limits"].pop()
        passthrough["limits"].append(moved)

    _rewrite_plan(generated, wrong_owner)
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="wrong profile"
    ):
        qualification._load_plan(generated.plan)


@pytest.mark.parametrize(
    ("cell_id", "changes", "message"),
    [
        ("macos-rvm-cpu", {"provider": "directml"}, "DirectML"),
        ("macos-rvm-cpu", {"provider": "cuda"}, "CUDA"),
        ("windows-dshow-mediapipe", {"provider": "directml"}, "non-RVM"),
        (
            "windows-dshow-mediapipe",
            {"dependency_profile": "without_mediapipe"},
            "without-MediaPipe",
        ),
        (
            "linux-rvm-cuda",
            {"dependency_profile": "without_gpu_provider"},
            "missing-GPU-provider",
        ),
    ],
)
def test_platform_provider_and_dependency_contradictions_are_rejected(
    generated: evidence.GeneratedPlatformQualification,
    cell_id: str,
    changes: dict[str, object],
    message: str,
) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        _cell(plan, cell_id).update(changes)

    _rewrite_plan(generated, mutate)
    with pytest.raises(qualification.MattePlatformQualificationError, match=message):
        qualification._load_plan(generated.plan)


def test_windows_native_cannot_claim_an_unadvertised_canvas(
    generated: evidence.GeneratedPlatformQualification,
) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        _cell(plan, "windows-native-heuristic")["canvas"] = {
            "width": 640,
            "height": 360,
        }

    _rewrite_plan(generated, mutate)
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="native mode"
    ):
        qualification._load_plan(generated.plan)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: plan.update({"version": True}),
        lambda plan: plan["reactions"].update({"enabled": True}),
        lambda plan: plan["reactions"].update({"post_base_event_count": True}),
        lambda plan: plan["reactions"].update({"post_base_event_count": 1}),
        lambda plan: plan["candidate"].update({"defaults_changed": True}),
        lambda plan: plan["provenance"].update(
            {"physical_origin_cryptographically_proven": True}
        ),
        lambda plan: plan["provenance"].update(
            {"physical_capture_origin_attested": True}
        ),
    ],
)
def test_generated_evidence_cannot_claim_reactions_defaults_or_physical_authority(
    generated: evidence.GeneratedPlatformQualification,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    _rewrite_plan(generated, mutate)
    with pytest.raises(qualification.MattePlatformQualificationError):
        qualification._load_plan(generated.plan)


def test_recorded_and_unavailable_cells_have_disjoint_artifact_contracts(
    generated: evidence.GeneratedPlatformQualification,
) -> None:
    def unavailable_with_artifact(plan: dict[str, Any]) -> None:
        cell = plan["cells"][0]
        cell["run"] = copy.deepcopy(plan["prerequisites"]["visual"])

    _rewrite_plan(generated, unavailable_with_artifact)
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="must not name"
    ):
        qualification._load_plan(generated.plan)

    generated = evidence.generate_qualification(
        generated.root.parent / "private-platform-evidence-recorded"
    )

    def recorded_with_reason(plan: dict[str, Any]) -> None:
        cell = plan["cells"][0]
        cell["state"] = "recorded"
        cell["run"] = copy.deepcopy(plan["prerequisites"]["visual"])
        cell["capture_report"] = copy.deepcopy(plan["prerequisites"]["visual"])

    _rewrite_plan(generated, recorded_with_reason)
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="cannot have"
    ):
        qualification._load_plan(generated.plan)


def test_prerequisite_and_cell_descriptors_reject_unsafe_paths(
    generated: evidence.GeneratedPlatformQualification,
) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        plan["prerequisites"]["visual"]["path"] = "../visual.json"

    _rewrite_plan(generated, mutate)
    with pytest.raises(
        qualification.MattePlatformQualificationError, match="relative path"
    ):
        qualification._load_plan(generated.plan)
