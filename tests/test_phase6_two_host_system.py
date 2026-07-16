"""Lightweight contracts for the Docker-backed two-host release harness."""

from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import stat
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
HARNESS_PATH = ROOT / "scripts" / "release" / "two-host-system-test.py"
DOCKERFILE = ROOT / "scripts" / "release" / "two-host" / "Dockerfile"
PROBE = ROOT / "scripts" / "release" / "two-host" / "probe.py"


def _load_harness():
    name = "custback_phase6_two_host_system"
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


def test_manifest_and_implementation_have_the_exact_tls_scenario_set(harness):
    manifest = json.loads(harness.DEFAULT_MANIFEST.read_text())

    assert tuple(manifest["tls_scenarios"]) == harness.TLS_SCENARIOS
    assert tuple(harness.SCENARIO_HANDLERS) == harness.TLS_SCENARIOS
    assert harness.load_tls_scenarios() == harness.TLS_SCENARIOS
    assert set(harness.SCENARIO_OPTIONS) <= set(harness.TLS_SCENARIOS)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "reordered"])
def test_scenario_manifest_drift_fails_before_docker(tmp_path, harness, mutation):
    manifest = json.loads(harness.DEFAULT_MANIFEST.read_text())
    scenarios = manifest["tls_scenarios"]
    if mutation == "missing":
        scenarios.pop()
    elif mutation == "extra":
        scenarios.append("plaintext-downgrade")
    elif mutation == "duplicate":
        scenarios.append(scenarios[0])
    else:
        scenarios[0], scenarios[1] = scenarios[1], scenarios[0]
    changed = tmp_path / "required-gates.json"
    changed.write_text(json.dumps(manifest))

    with pytest.raises(harness.HarnessError):
        harness.load_tls_scenarios(changed)


def test_each_scenario_handler_contains_its_required_system_operations(harness):
    required = {
        "nominal-rendering": ("_nominal",),
        "renderer-outage": ("stop_renderer", "wait_privacy_slate"),
        "stale-renderer-output": ("pause_renderer", "wait_privacy_slate"),
        "wrong-renderer-token": ("ws_auth", "wait_privacy_slate"),
        "wrong-control-token": (
            "avatar_auth_failed",
            "establish_browser_session",
            "assert_browser_session",
        ),
        "renderer-untrusted-ca": ("_renderer_certificate_failure",),
        "renderer-expired-certificate": ("_renderer_certificate_failure",),
        "renderer-wrong-host-certificate": ("_renderer_certificate_failure",),
        "control-untrusted-ca": ("_control_certificate_failure",),
        "control-expired-certificate": ("_control_certificate_failure",),
        "control-wrong-host-certificate": ("_control_certificate_failure",),
        "renderer-firewall-path-removed": ("block", "wait_privacy_slate"),
        "control-firewall-path-removed": ("block", "avatar_unreachable"),
        "renderer-reconnect": ("unblock", "reconnects"),
        "credential-and-ca-rotation": (
            "rotate_renderer_listener_token",
            "rotate_renderer_client_token",
            "rotate_control_listener_token",
            "rotate_control_client_token",
            "stage_meeting_ca",
            "stage_renderer_ca",
            "old-renderer-token",
        ),
        "privacy-slate-startup": ("renderer=False", "wait_privacy_slate"),
        "privacy-slate-frame-plane-failure": ("block", "wait_privacy_slate"),
        "control-error-mapping": ("avatar_auth_failed", "avatar_unreachable"),
        "failed-tls-handshake-no-payload": ("assert_tls_handshake_only",),
        "clean-host-repeat-a": ("_nominal",),
        "clean-host-repeat-b": ("_nominal",),
    }
    assert set(required) == set(harness.SCENARIO_HANDLERS)
    for scenario_id, tokens in required.items():
        source = inspect.getsource(harness.SCENARIO_HANDLERS[scenario_id])
        for token in tokens:
            assert token in source, f"{scenario_id} is missing {token}"


class _FakeDocker:
    def __init__(self, harness):
        self.harness = harness
        self.commands = []
        self.cleanup_commands = []

    def run(self, *arguments, **_options):
        self.commands.append(tuple(arguments))
        return self.harness.CommandResult(tuple(arguments), 0, "", "")

    def best_effort(self, *arguments):
        self.cleanup_commands.append(tuple(arguments))


def _stack_without_docker(tmp_path, harness):
    docker = _FakeDocker(harness)
    context = types.SimpleNamespace(
        docker=docker,
        image="custback-phase6:test",
        run_id="contract-run",
        command_timeout=30.0,
        scenario_timeout=10.0,
    )
    stack = object.__new__(harness.ScenarioStack)
    stack.context = context
    stack.scenario_id = "nominal-rendering"
    stack.options = harness.ScenarioOptions()
    stack.network = "isolated-phase6-network"
    stack.meeting_container = "meeting-container"
    stack.renderer_container = "renderer-container"
    stack.tracker = harness.ResourceTracker(docker)
    stack.material = types.SimpleNamespace(
        root=tmp_path,
        meeting=tmp_path / "meeting-secrets",
        renderer=tmp_path / "renderer-secrets",
    )
    stack.material.meeting.mkdir()
    stack.material.renderer.mkdir()
    stack.meeting_address = ""
    stack.renderer_address = ""
    stack.started_meeting = False
    stack.started_renderer = False
    stack._blocked = set()
    stack._wait_tcp = lambda *_args: None
    stack._wait_core_ready = lambda: None
    stack._wait_avatar_ready = lambda: None
    stack._refresh_addresses = lambda: None
    return stack, docker


def test_container_commands_use_isolated_network_separate_secrets_and_packaged_cli(
    tmp_path, harness
):
    stack, docker = _stack_without_docker(tmp_path, harness)

    stack.start_meeting()
    stack.start_renderer()

    meeting, renderer = docker.commands
    assert "--network" in meeting and "isolated-phase6-network" in meeting
    assert "--network" in renderer and "isolated-phase6-network" in renderer
    assert "host" not in meeting
    assert "--read-only" in meeting and "--read-only" in renderer
    assert "NET_ADMIN" in meeting and "NET_ADMIN" in renderer
    assert any(
        str(stack.material.meeting) in item and item.endswith(",readonly")
        for item in meeting
    )
    assert any(
        str(stack.material.renderer) in item and item.endswith(",readonly")
        for item in renderer
    )
    assert stack.material.meeting != stack.material.renderer
    assert meeting[-4:] == (
        "custback",
        "-c",
        "/run/custback/config.yaml",
        "--no-file-log",
    )
    assert renderer[-4:] == (
        "custback",
        "avatar",
        "-c",
        "/run/custback/avatar.yaml",
    )


def test_firewall_commands_remove_each_direction_independently(tmp_path, harness):
    stack, docker = _stack_without_docker(tmp_path, harness)

    stack.block("renderer", harness.MEETING_PORT)
    stack.block("meeting", harness.RENDERER_PORT)
    stack.unblock("renderer", harness.MEETING_PORT)

    assert docker.commands[0] == (
        "exec",
        "renderer-container",
        "iptables",
        "-I",
        "OUTPUT",
        "1",
        "-p",
        "tcp",
        "--dport",
        "8710",
        "-j",
        "REJECT",
    )
    assert docker.commands[1][1] == "meeting-container"
    assert "8711" in docker.commands[1]
    assert docker.commands[2][3:5] == ("-D", "OUTPUT")


def test_resource_cleanup_is_reverse_ordered_and_idempotent(harness):
    docker = _FakeDocker(harness)
    tracker = harness.ResourceTracker(docker)
    tracker.expect_container("meeting")
    tracker.expect_container("renderer")
    tracker.expect_network("isolated")
    tracker.expect_image("candidate")

    tracker.cleanup()
    tracker.cleanup()

    assert docker.cleanup_commands == [
        ("rm", "-f", "renderer"),
        ("rm", "-f", "meeting"),
        ("network", "rm", "isolated"),
        ("image", "rm", "-f", "candidate"),
    ]


def test_harness_teardown_fails_closed_on_labeled_resource_remnants(harness):
    docker = _FakeDocker(harness)
    subject = object.__new__(harness.TwoHostHarness)
    subject.docker = docker
    subject.run_id = "cleanup-contract"

    subject._assert_run_resources_removed()
    assert [command[:2] for command in docker.commands] == [
        ("ps", "-aq"),
        ("network", "ls"),
        ("image", "ls"),
    ]

    def leave_container(*arguments, **_options):
        output = "remaining-container\n" if arguments[:2] == ("ps", "-aq") else ""
        return harness.CommandResult(tuple(arguments), 0, output, "")

    docker.run = leave_container
    with pytest.raises(harness.HarnessError, match="labeled containers"):
        subject._assert_run_resources_removed()


def test_log_collection_failure_cannot_skip_scenario_cleanup(tmp_path, harness):
    stack, docker = _stack_without_docker(tmp_path, harness)
    stack.tracker.expect_container("meeting-container")

    def fail_logs(*_arguments, **_options):
        raise harness.HarnessError("log timeout")

    docker.run = fail_logs
    stack.__exit__(None, None, None)

    assert docker.cleanup_commands == [("rm", "-f", "meeting-container")]


def _tls_record(content_type: int, payload: bytes) -> bytes:
    return bytes((content_type, 3, 3)) + len(payload).to_bytes(2, "big") + payload


def test_failed_tls_capture_rejects_application_data_tokens_and_truncation(harness):
    handshake = _tls_record(22, b"client-hello")
    harness.assert_tls_handshake_only(handshake, (b"renderer-secret",))

    with pytest.raises(harness.HarnessError, match="application data"):
        harness.assert_tls_handshake_only(
            handshake + _tls_record(23, b"camera-frame"), ()
        )
    with pytest.raises(harness.HarnessError, match="credential or payload"):
        harness.assert_tls_handshake_only(
            _tls_record(22, b"renderer-secret"), (b"renderer-secret",)
        )
    with pytest.raises(harness.HarnessError, match="truncated"):
        harness.assert_tls_handshake_only(handshake[:-1], ())


def test_image_build_installs_only_the_supplied_wheel_and_in_network_probe():
    dockerfile = DOCKERFILE.read_text()
    probe = PROBE.read_text()

    assert "ARG CUSTBACK_WHEEL" in dockerfile
    assert "pip install /tmp/${CUSTBACK_WHEEL}" in dockerfile
    assert "COPY probe.py /opt/custback-two-host-probe.py" in dockerfile
    assert "COPY src" not in dockerfile
    assert "capture-once" in probe
    assert "output-frame" in probe
    assert "record-output" in probe
    assert "browser-session-create" in probe
    assert "browser-session-check" in probe
    assert "wait-tcp" in probe


def test_status_waits_record_consecutive_preview_frames(harness):
    nominal = inspect.getsource(harness.ScenarioStack.wait_nominal)
    fallback = inspect.getsource(harness.ScenarioStack.wait_privacy_slate)

    assert 'record_output("rendered")' in nominal
    assert 'record_output("slate")' in fallback


def test_wheel_input_must_be_a_real_non_symlink_archive(tmp_path, harness):
    wheel = tmp_path / "custback-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("custback/__init__.py", "")
    resolved, digest = harness.validate_wheel(wheel)
    assert resolved == wheel.resolve()
    assert len(digest) == 64

    linked = tmp_path / "custback-1.2.4-py3-none-any.whl"
    linked.symlink_to(wheel)
    with pytest.raises(harness.HarnessError, match="non-symlink"):
        harness.validate_wheel(linked)


def test_workflow_artifact_directory_selects_exactly_one_wheel(tmp_path, harness):
    wheel = tmp_path / "custback-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("custback/__init__.py", "")
    assert harness.wheel_from_artifacts(tmp_path) == wheel

    second = tmp_path / "custback-1.2.4-py3-none-any.whl"
    with zipfile.ZipFile(second, "w") as archive:
        archive.writestr("custback/__init__.py", "")
    with pytest.raises(harness.HarnessError, match="exactly one"):
        harness.wheel_from_artifacts(tmp_path)


def test_workflow_cli_and_clean_host_selection_contract(tmp_path, harness):
    wheel = tmp_path / "custback-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("custback/__init__.py", "")

    selected = harness.TwoHostHarness(
        wheel,
        scenario_ids=("renderer-outage",),
        clean_host_evidence=False,
    )
    assert selected.scenarios == ("renderer-outage",)

    clean = harness.TwoHostHarness(
        wheel,
        scenario_ids=("clean-host-repeat-a",),
        clean_host_evidence=True,
    )
    assert clean.scenarios == ("clean-host-repeat-a",)
    assert clean.clean_host_evidence is True

    with pytest.raises(harness.HarnessError, match="clean-host"):
        harness.TwoHostHarness(
            wheel,
            scenario_ids=("nominal-rendering",),
            clean_host_evidence=True,
        )
    parser = harness.build_parser()
    actions = {option for action in parser._actions for option in action.option_strings}
    assert {"--artifacts", "--scenario", "--clean-host-evidence", "--result"} <= actions


@pytest.mark.skipif(shutil.which("openssl") is None, reason="OpenSSL is unavailable")
def test_ephemeral_material_has_independent_verified_cas_and_private_modes(
    tmp_path, harness
):
    material = harness.MaterialFactory(harness.CommandRunner()).create(
        tmp_path,
        "nominal-rendering",
        harness.ScenarioOptions(),
    )

    assert material.meeting != material.renderer
    assert (material.meeting / "server-ca.pem").read_bytes() != (
        material.renderer / "server-ca.pem"
    ).read_bytes()
    for directory in (material.root, material.meeting, material.renderer):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for directory in (material.meeting, material.renderer):
        for path in directory.iterdir():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600

    for directory, hostname in (
        (material.meeting, harness.MEETING_NAME),
        (material.renderer, harness.RENDERER_NAME),
    ):
        verified = subprocess.run(
            (
                "openssl",
                "verify",
                "-CAfile",
                str(directory / "server-ca.pem"),
                "-verify_hostname",
                hostname,
                str(directory / "server.crt"),
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert verified.returncode == 0, verified.stderr
