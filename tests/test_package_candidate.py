"""Candidate identity and incomplete-evidence refusal contracts."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "package_candidate", ROOT / "scripts/release/package-candidate.py"
)
candidate_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate_tools)


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    directory = tmp_path / "candidate"
    directory.mkdir()
    env = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": candidate_tools.REPOSITORY,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_WORKFLOW_REF": f"{candidate_tools.WORKFLOW}@refs/heads/main",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_RUN_ID": "1234",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    artifacts = []
    for name, filename in [
        ("python-wheel", "custback-0.4.0-py3-none-any.whl"),
        ("python-sdist", "custback-0.4.0.tar.gz"),
        ("npm-tarball", "custback-0.4.0.tgz"),
    ]:
        path = directory / filename
        path.write_bytes(name.encode())
        artifacts.append(
            {
                "id": name,
                "filename": filename,
                "sha256": candidate_tools.digest(path),
                "size": path.stat().st_size,
            }
        )
    manifest = json.loads((ROOT / "scripts/release/required-gates.json").read_text())
    value = {
        "schema_version": 1,
        "authorization": "release",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "source": {"version": "0.4.0", "commit": env["GITHUB_SHA"], "tree": "b" * 40},
        "provenance": {
            "provider": "github-actions",
            "repository": env["GITHUB_REPOSITORY"],
            "commit": env["GITHUB_SHA"],
            "run_id": env["GITHUB_RUN_ID"],
            "run_attempt": 1,
            "workflow_ref": env["GITHUB_WORKFLOW_REF"],
        },
        "artifacts": artifacts,
    }
    candidate_tools.write_json(directory / "candidate-manifest.json", value)
    return directory, value, env


def test_candidate_requires_exact_files_and_provenance(candidate):
    directory, value, env = candidate
    assert candidate_tools.verify(directory, env) == value
    artifact = directory / value["artifacts"][0]["filename"]
    artifact.write_bytes(b"substituted file")
    with pytest.raises(ValueError, match="digest"):
        candidate_tools.verify(directory, env)


@pytest.mark.parametrize(
    "key,value",
    [
        ("GITHUB_ACTIONS", "false"),
        ("GITHUB_REPOSITORY", "other/custback"),
        ("GITHUB_EVENT_NAME", "pull_request"),
        ("GITHUB_REF", "refs/heads/feature"),
        ("GITHUB_SHA", "c" * 40),
        ("GITHUB_RUN_ID", "4321"),
        ("GITHUB_RUN_ATTEMPT", "2"),
        ("GITHUB_WORKFLOW_REF", "wrong/workflow"),
    ],
)
def test_candidate_rejects_different_execution(candidate, key, value):
    directory, _, env = candidate
    with pytest.raises(ValueError):
        candidate_tools.verify(directory, {**env, key: value})


@pytest.mark.parametrize(
    "mutation",
    ["extra-file", "duplicate-id", "path-escape", "manifest", "diagnostic", "version"],
)
def test_candidate_rejects_wrong_payload(candidate, mutation):
    directory, value, env = candidate
    if mutation == "extra-file":
        (directory / "secret.txt").write_text("unexpected fixture")
    elif mutation == "duplicate-id":
        value["artifacts"][1] = copy.deepcopy(value["artifacts"][0])
    elif mutation == "path-escape":
        value["artifacts"][0]["filename"] = "../outside.whl"
    elif mutation == "manifest":
        value["manifest_sha256"] = "f" * 64
    elif mutation == "diagnostic":
        value["authorization"] = "diagnostic"
    else:
        value["source"]["version"] = "0.5.0"
    (directory / "candidate-manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        candidate_tools.verify(directory, env)


def host_reports(candidate, tmp_path):
    directory, value, _ = candidate
    reports = tmp_path / "reports"
    for host in candidate_tools.HOSTS:
        candidate_tools.write_json(
            reports / f"{host}.json",
            {
                **candidate_tools.binding(directory, value),
                "runner_os": host,
                "runner_environment": "github-hosted",
                "python": "3.12.12",
                "result": "success",
                "passed_artifacts": [
                    row["id"]
                    for row in value["artifacts"]
                    if host != "Windows" or row["id"] != "npm-tarball"
                ],
            },
        )
    ci = tmp_path / "ci.json"
    candidate_tools.write_json(
        ci,
        [
            {
                "conclusion": "success",
                "headSha": value["source"]["commit"],
                "headBranch": "main",
                "event": "push",
            }
        ],
    )
    return reports, ci


def test_qualification_records_bounded_scope_and_exact_checksums(candidate, tmp_path):
    directory, value, _ = candidate
    reports, ci = host_reports(candidate, tmp_path)
    output = tmp_path / "qualification/qualification.json"
    candidate_tools.qualify(directory, reports, ci, output)
    qualification = json.loads(output.read_text())
    assert qualification["profile"] == "core-packages-v1"
    assert qualification["deferred"] == candidate_tools.DEFERRED
    assert qualification["publication"].startswith("manual")
    assert qualification["artifacts"] == value["artifacts"]
    assert len((output.parent / "SHA256SUMS").read_text().splitlines()) == 3


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-host",
        "wrong-host",
        "failed",
        "missing-artifact",
        "wrong-candidate",
        "source-ci",
        "python-version",
    ],
)
def test_qualification_rejects_incomplete_or_unrelated_evidence(
    candidate, tmp_path, mutation
):
    directory, _, _ = candidate
    reports, ci = host_reports(candidate, tmp_path)
    filename = reports / "Windows.json"
    report = json.loads(filename.read_text())
    if mutation == "missing-host":
        filename.unlink()
    elif mutation == "source-ci":
        ci.write_text("[]")
    else:
        field, value = {
            "wrong-host": ("runner_os", "Linux"),
            "failed": ("result", "failure"),
            "missing-artifact": ("passed_artifacts", ["python-wheel"]),
            "wrong-candidate": ("candidate_sha256", "f" * 64),
            "python-version": ("python", "3.11.15"),
        }[mutation]
        report[field] = value
        filename.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        candidate_tools.qualify(
            directory, reports, ci, tmp_path / "out/qualification.json"
        )


def test_candidate_workflow_is_manual_and_cannot_publish():
    source = (ROOT / ".github/workflows/package-candidate.yml").read_text()
    assert "on:\n  workflow_dispatch:\n" in source
    assert "  push:" not in source
    for forbidden in (
        "twine upload",
        "npm publish",
        "gh release create",
        "secrets.",
        "contents: write",
    ):
        assert forbidden not in source
    assert "os: [ubuntu-latest, windows-2022, macos-latest]" in source
    assert "needs: [build, artifact-smoke]" in source
    assert "needs.artifact-smoke.result == 'success'" in source
    assert "--signer-digest" in source and "--source-digest" in source
    assert "--deny-self-hosted-runners" in source


def test_avatar_template_uses_committed_bytes_despite_checkout_line_endings(tmp_path):
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args])

    git("init", "--quiet")
    (tmp_path / "config").mkdir()
    template = tmp_path / "config/avatar.yaml"
    original = b"driver:\n  backend: idle\n"
    template.write_bytes(original)
    git("-c", "core.autocrlf=false", "add", "config/avatar.yaml")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    commit = git("rev-parse", "HEAD").decode().strip()
    template.write_bytes(original.replace(b"\n", b"\r\n"))
    assert template.read_bytes() != original
    assert candidate_tools.committed_avatar_template(commit, tmp_path) == original
    with pytest.raises(ValueError, match="Invalid template source commit"):
        candidate_tools.committed_avatar_template("main:other-file", tmp_path)
