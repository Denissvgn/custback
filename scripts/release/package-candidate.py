"""Verify and smoke-test immutable core package candidates without publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "core-packages-v1"
REPOSITORY = "Denissvgn/custback"
WORKFLOW = f"{REPOSITORY}/.github/workflows/package-candidate.yml"
HOSTS = {"Linux": "ubuntu-latest", "Windows": "windows-2022", "macOS": "macos-latest"}
DEFERRED = [
    "full Python/Node and dependency-bound matrices",
    "optional vision, Audio2Face, and CUDA hardware qualification",
    "stress and performance thresholds",
    "migration and two-host TLS qualification",
    "physical camera and consumer compatibility",
    "signed Windows installers and native camera integration",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    require(path.is_file() and not path.is_symlink(), f"Not a regular file: {path}")
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def read_json(path):
    require(
        path.is_file() and not path.is_symlink(), f"Not a regular JSON file: {path}"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def verify(directory, env=None):
    env = os.environ if env is None else env
    require(env.get("GITHUB_ACTIONS") == "true", "Hosted GitHub context required")
    require(env.get("GITHUB_REPOSITORY") == REPOSITORY, "Wrong repository")
    require(
        env.get("GITHUB_EVENT_NAME") == "workflow_dispatch", "Manual dispatch required"
    )
    require(env.get("GITHUB_REF") == "refs/heads/main", "Candidate must come from main")
    require(
        env.get("GITHUB_WORKFLOW_REF") == f"{WORKFLOW}@refs/heads/main",
        "Wrong workflow",
    )
    candidate = read_json(directory / "candidate-manifest.json")
    require(candidate.get("schema_version") == 1, "Unsupported candidate schema")
    require(
        candidate.get("authorization") == "release", "Diagnostic build is ineligible"
    )
    source = candidate["source"]
    require(
        re.fullmatch(r"0\.4\.[0-9]+", source["version"]), "Profile is limited to 0.4.x"
    )
    require(re.fullmatch(r"[0-9a-f]{40}", source["commit"]), "Invalid source commit")
    require(source["commit"] == env.get("GITHUB_SHA"), "Wrong candidate commit")
    require(re.fullmatch(r"[0-9a-f]{40}", source["tree"]), "Invalid source tree")
    provenance = candidate["provenance"]
    expected = {
        "provider": "github-actions",
        "repository": REPOSITORY,
        "commit": source["commit"],
        "run_id": env.get("GITHUB_RUN_ID"),
        "run_attempt": int(env.get("GITHUB_RUN_ATTEMPT", "0")),
        "workflow_ref": env.get("GITHUB_WORKFLOW_REF"),
    }
    require(provenance == expected, "Wrong candidate workflow/run/attempt provenance")
    require(str(provenance["run_id"]).isdigit(), "Invalid run id")
    require(provenance["run_attempt"] > 0, "Invalid run attempt")
    manifest = read_json(ROOT / "scripts/release/required-gates.json")
    # Match the builder's canonical JavaScript JSON serialization.
    manifest_bytes = json.dumps(
        manifest, separators=(",", ":"), ensure_ascii=False, sort_keys=True
    ).encode()
    require(
        candidate["manifest_id"] == manifest["manifest_id"], "Wrong artifact manifest"
    )
    require(
        candidate["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(),
        "Artifact manifest changed",
    )
    names = {
        "python-wheel": f"custback-{source['version']}-py3-none-any.whl",
        "python-sdist": f"custback-{source['version']}.tar.gz",
        "npm-tarball": f"custback-{source['version']}.tgz",
    }
    require(len(candidate["artifacts"]) == 3, "Exactly three artifacts required")
    require(
        {row["id"] for row in candidate["artifacts"]} == set(names),
        "Wrong artifact IDs",
    )
    require(
        {p.name for p in directory.iterdir()}
        == {*names.values(), "candidate-manifest.json"},
        "Unexpected or missing candidate files",
    )
    for artifact in candidate["artifacts"]:
        require(
            artifact["filename"] == names[artifact["id"]], "Unsafe artifact filename"
        )
        filename = directory / artifact["filename"]
        require(digest(filename) == artifact["sha256"], "Artifact digest mismatch")
        require(filename.stat().st_size == artifact["size"], "Artifact size mismatch")
    return candidate


def binding(directory, candidate):
    return {
        "profile": PROFILE,
        "candidate_sha256": digest(directory / "candidate-manifest.json"),
        "source": candidate["source"],
        "provenance": candidate["provenance"],
        "artifacts": candidate["artifacts"],
    }


def command(args, cwd, env=None):
    subprocess.run(
        [str(arg) for arg in args], cwd=cwd, env=env, check=True, timeout=900
    )


def committed_avatar_template(commit, root=ROOT):
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Invalid template source commit")
    return subprocess.check_output(
        ["git", "-C", str(root), "show", f"{commit}:config/avatar.yaml"], timeout=30
    )


def smoke(directory, report):
    candidate = verify(directory)
    runner_os = os.environ["RUNNER_OS"]
    require(runner_os in HOSTS, "Unsupported runner OS")
    require(sys.version_info[:2] == (3, 12), "Smoke profile requires Python 3.12")
    require(
        platform.system()
        == {"Linux": "Linux", "Windows": "Windows", "macOS": "Darwin"}[runner_os],
        "Runner OS mismatch",
    )
    require(
        os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted",
        "Hosted runner required",
    )
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="custback-installed-") as temporary:
        scratch = Path(temporary)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        for artifact in candidate["artifacts"]:
            if artifact["id"] == "npm-tarball":
                if runner_os == "Windows":
                    continue
                prefix = scratch / "npm-prefix"
                npm_env = dict(
                    env,
                    CUSTBACK_EXTRAS="",
                    CUSTBACK_SKIP_INSTALL="0",
                    CUSTBACK_VENV=str(scratch / "npm-runtime"),
                )
                command(
                    [
                        "npm",
                        "install",
                        "--global",
                        directory / artifact["filename"],
                        "--prefix",
                        prefix,
                        "--no-audit",
                        "--no-fund",
                    ],
                    scratch,
                    npm_env,
                )
                launcher = prefix / "bin/custback"
                for args in (["--help"], ["doctor"], ["avatar", "--smoke"]):
                    command([launcher, *args], scratch, npm_env)
            else:
                venv = scratch / artifact["id"]
                command([sys.executable, "-m", "venv", venv], scratch, env)
                bins = venv / ("Scripts" if runner_os == "Windows" else "bin")
                python = bins / ("python.exe" if runner_os == "Windows" else "python")
                launcher = bins / (
                    "custback.exe" if runner_os == "Windows" else "custback"
                )
                command(
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        directory / artifact["filename"],
                    ],
                    scratch,
                    env,
                )
                command([python, "-m", "pip", "check"], scratch, env)
                probe = (
                    "import importlib.metadata as m,custback,pathlib; assert m.version('custback') == "
                    + repr(candidate["source"]["version"])
                    + "; assert pathlib.Path(custback.__file__).resolve().is_relative_to("
                    + repr(str(venv.resolve()))
                    + ")"
                )
                command([python, "-I", "-c", probe], scratch, env)
                command([launcher, "--help"], scratch, env)
                command([launcher, "avatar", "--smoke"], scratch, env)
                exported = scratch / f"{artifact['id']}-avatar.yaml"
                command(
                    [launcher, "avatar", "config", "export", exported], scratch, env
                )
                require(
                    exported.read_bytes()
                    == committed_avatar_template(candidate["source"]["commit"]),
                    "Installed avatar configuration differs",
                )
                command(
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "pytest>=8,<10",
                        "pytest-timeout>=2.3,<3",
                    ],
                    scratch,
                    env,
                )
                command(
                    [
                        python,
                        "-I",
                        "-m",
                        "pytest",
                        "-q",
                        "--import-mode=importlib",
                        ROOT / "tests/test_config.py",
                        ROOT / "tests/test_config_merge.py",
                        ROOT / "tests/test_platform_seam.py",
                    ],
                    scratch,
                    env,
                )
            outcomes.append(artifact["id"])
    # Detect changes to bytes during installation before recording success.
    require(verify(directory) == candidate, "Candidate changed during smoke")
    write_json(
        report,
        {
            **binding(directory, candidate),
            "runner_os": runner_os,
            "runner_environment": "github-hosted",
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "node": subprocess.check_output(["node", "--version"], text=True).strip(),
            "passed_artifacts": outcomes,
            "result": "success",
        },
    )


def qualify(directory, reports, source_ci, output):
    candidate = verify(directory)
    expected_binding = binding(directory, candidate)
    ci_runs = read_json(source_ci)
    require(
        any(
            row.get("conclusion") == "success"
            and row.get("headSha") == candidate["source"]["commit"]
            and row.get("headBranch") == "main"
            and row.get("event") == "push"
            for row in ci_runs
        ),
        "No successful source CI on this main commit",
    )
    require(
        {p.name for p in reports.iterdir()} == {f"{os_name}.json" for os_name in HOSTS},
        "Missing or extra host reports",
    )
    results = []
    for host in HOSTS:
        report = read_json(reports / f"{host}.json")
        require(
            all(report.get(key) == value for key, value in expected_binding.items()),
            "Wrong report candidate binding",
        )
        require(
            report.get("runner_os") == host
            and report.get("runner_environment") == "github-hosted",
            "Wrong report host",
        )
        require(report.get("result") == "success", "Failed host report")
        require(
            re.fullmatch(r"3\.12\.\d+", report.get("python", "")),
            "Wrong Python smoke version",
        )
        expected = [
            row["id"]
            for row in candidate["artifacts"]
            if host != "Windows" or row["id"] != "npm-tarball"
        ]
        require(
            report.get("passed_artifacts") == expected,
            "Missing artifact installation checks",
        )
        results.append(report)
    write_json(
        output,
        {
            "schema_version": 1,
            **expected_binding,
            "result": "success",
            "scope": "Python wheel and sdist on Ubuntu, Windows filesystem API (Windows Server 2022 runner), and macOS; npm launcher on Ubuntu and macOS; Python 3.12 core dependencies",
            "runner_images": HOSTS,
            "reports": results,
            "source_ci_runs": ci_runs,
            "deferred": DEFERRED,
            "publication": "manual; this workflow publishes nothing",
        },
    )
    (output.parent / "SHA256SUMS").write_text(
        "".join(
            f"{row['sha256']}  {row['filename']}\n" for row in candidate["artifacts"]
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["verify", "smoke", "qualify"])
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--source-ci", type=Path)
    args = parser.parse_args()
    if args.operation == "verify":
        verify(args.candidate)
    elif args.operation == "smoke":
        require(args.report is not None, "--report required")
        smoke(args.candidate.resolve(), args.report)
    else:
        require(
            all([args.reports, args.source_ci, args.report]),
            "--reports, --source-ci, and --report required",
        )
        qualify(args.candidate, args.reports, args.source_ci, args.report)


if __name__ == "__main__":
    main()
