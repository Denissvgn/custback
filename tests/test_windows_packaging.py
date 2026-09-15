"""Static verification of the frozen-Windows packaging artifacts (WIN-5.1).

These artifacts are *built* on the ``windows-latest`` CI job, but their source
is byte-static Python/PowerShell that can be checked on any platform: the spec
must be syntactically valid, must stay ``onedir``, must not bundle
licensing-gated model weights, must carry the dynamically dispatched
``custback`` imports (platform-security backend, segmentation delegates), and
must not hard-code the version that the release gate pins elsewhere.

The test locates the packaging tree relative to the source checkout and skips
when it is absent (e.g. running from an installed wheel/sdist that ships only
``src`` and ``tests``), so it never breaks the release-artifact smoke.
"""

from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

SPEC_DIR = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "pyinstaller"
PROJECT_ROOT = SPEC_DIR.parents[2]
PYVIRTUALCAM_REQUIREMENT = (
    "pyvirtualcam>=0.11,<1; sys_platform != 'win32' or "
    "(platform_machine != 'ARM64' and platform_machine != 'arm64')"
)

pytestmark = pytest.mark.skipif(
    not (SPEC_DIR / "custback.spec").exists(),
    reason="frozen-build packaging tree is not present in this layout",
)


def _spec_text() -> str:
    return (SPEC_DIR / "custback.spec").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "relpath",
    [
        "custback.spec",
        "entry_custback.py",
        "entry_custback_avatar.py",
        "hooks/hook-custback.py",
        "rthooks/pyi_rth_custback.py",
    ],
)
def test_frozen_sources_compile(relpath: str) -> None:
    path = SPEC_DIR / relpath
    assert path.exists(), f"missing frozen-build source: {relpath}"
    # doraise makes a syntax error a test failure rather than a silent skip.
    py_compile.compile(str(path), doraise=True)


def test_spec_is_onedir_not_onefile() -> None:
    text = _spec_text()
    # onedir assembles a COLLECT target and keeps binaries out of the EXE
    # (exclude_binaries=True). A onefile build folds a.binaries into EXE() and
    # has no COLLECT; guard against an accidental regression to that mode.
    assert "COLLECT(" in text, "onedir build must assemble a COLLECT target"
    assert "exclude_binaries=True" in text


def test_spec_excludes_licensing_gated_model_weights() -> None:
    text = _spec_text()
    # Weights are downloaded on first run (WIN-0.4, D6); the spec must not
    # collect *.onnx / *.tflite as data.
    assert '"**/*.onnx"' in text and '"**/*.tflite"' in text
    assert 'excludes=["**/*.onnx", "**/*.tflite"' in text
    # And no weight filename may be hard-referenced as a bundled data file.
    assert "rvm_mobilenetv3" not in text
    assert "selfie_segmenter" not in text


def test_spec_carries_dynamic_custback_imports() -> None:
    text = _spec_text()
    for needed in (
        "custback._platform.windows",
        "pywintypes",
        "win32security",
        "win32file",
        "ntsecuritycon",
    ):
        assert needed in text, f"spec is missing hidden import {needed!r}"


def test_spec_carries_both_annotated_configuration_templates() -> None:
    text = _spec_text()
    assert '"**/*.yaml", "**/system-profile-catalog.json"' in text
    assert (PROJECT_ROOT / "src" / "custback" / "default.yaml").read_bytes() == (
        PROJECT_ROOT / "config" / "default.yaml"
    ).read_bytes()
    assert (
        PROJECT_ROOT / "src" / "custback" / "avatar" / "avatar.yaml"
    ).read_bytes() == (PROJECT_ROOT / "config" / "avatar.yaml").read_bytes()
    catalog = PROJECT_ROOT / "src" / "custback" / "system-profile-catalog.json"
    assert json.loads(catalog.read_text(encoding="utf-8"))["schema"] == (
        "custback.system-profile-catalog"
    )
    hook = (
        PROJECT_ROOT
        / "packaging"
        / "windows"
        / "pyinstaller"
        / "hooks"
        / "hook-custback.py"
    ).read_text(encoding="utf-8")
    assert '"**/system-profile-catalog.json"' in hook


def test_spec_carries_pyav_submodules_and_private_ffmpeg_bundle() -> None:
    text = _spec_text()
    assert '"av"' in text
    assert 'collect_submodules("av")' in text
    assert "collect_dynamic_libs(_pkg)" in text
    assert 'get_package_paths("av")' in text
    assert '"av.libs"' in text
    assert '_datas += [(_source, "av.libs")' in text
    assert "PyAV native bundle is empty" in text


def test_windows_arm64_omits_pyvirtualcam_dependency_and_freeze_inputs() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"{PYVIRTUALCAM_REQUIREMENT}"' in pyproject

    marker = Requirement(PYVIRTUALCAM_REQUIREMENT).marker
    assert marker is not None
    assert not marker.evaluate(
        environment={"sys_platform": "win32", "platform_machine": "ARM64"}
    )
    assert not marker.evaluate(
        environment={"sys_platform": "win32", "platform_machine": "arm64"}
    )
    assert marker.evaluate(
        environment={"sys_platform": "win32", "platform_machine": "AMD64"}
    )
    assert marker.evaluate(
        environment={"sys_platform": "linux", "platform_machine": "aarch64"}
    )

    text = _spec_text()
    assert 'sys.platform == "win32"' in text
    assert 'platform.machine().upper() == "ARM64"' in text
    assert '_native_packages.append("pyvirtualcam")' in text
    assert '_hiddenimports.append("pyvirtualcam")' in text
    assert '_excludes.append("pyvirtualcam")' in text
    assert "if not _WINDOWS_ARM64:" in text


def test_spec_does_not_hardcode_pinned_version() -> None:
    text = _spec_text()
    # The version must come from importlib.metadata / custback.__version__, not
    # a literal that would silently drift from the release-gate-pinned value.
    assert 'version("custback")' in text
    assert '"0.4.0"' not in text and "'0.4.0'" not in text


def test_engine_runs_as_console_for_supervision() -> None:
    assert "console=True" in _spec_text()


def test_runtime_hook_marks_frozen_and_guards_win32() -> None:
    rthook = (SPEC_DIR / "rthooks" / "pyi_rth_custback.py").read_text(encoding="utf-8")
    assert "CUSTBACK_FROZEN" in rthook
    assert 'sys.platform == "win32"' in rthook


def test_build_smoke_scrubs_toolchain_environment() -> None:
    build = (SPEC_DIR / "build.ps1").read_text(encoding="utf-8")
    # The clean-VM acceptance requires proving no Python/PYTHONPATH/CUDA leak.
    for scrubbed in ("PYTHONPATH", "PYTHONHOME", "CUDA_PATH"):
        assert scrubbed in build
    assert "--synthetic" in build and "--no-vcam" in build and "--no-api" in build
    assert "--frozen-video-color-smoke" in build


def test_frozen_entry_video_color_probe_is_offline_and_meaningful() -> None:
    entry = (SPEC_DIR / "entry_custback.py").read_text(encoding="utf-8")
    assert "VideoFrame.from_ndarray" in entry
    assert "normalize_video_frame" in entry
    assert "bt709/limited/bt709/srgb" in entry
    assert "srgb-full-bgr" in entry
    assert "av.open" not in entry

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(SPEC_DIR / "entry_custback.py"),
            "--frozen-video-color-smoke",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_installed_artifact_smoke_also_normalizes_tagged_video_in_memory() -> None:
    release = (PROJECT_ROOT / "scripts" / "release" / "verify-release.js").read_text(
        encoding="utf-8"
    )
    assert "import av," in release
    assert "custback.video_decoder" in release
    assert "VideoFrame.from_ndarray" in release
    assert "normalize_video_frame(video_frame" in release
    assert "bt709/limited/bt709/srgb" in release


def test_manual_ci_runs_bounded_windows_frozen_engine_smoke() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "full-ci.yml").read_text(
        encoding="utf-8"
    )
    marker = "\n  windows-frozen-engine:\n"
    assert marker in workflow
    job = workflow.split(marker, 1)[1].split("\n  node:\n", 1)[0]
    assert "runs-on: windows-2022" in job
    assert "timeout-minutes: 30" in job
    assert 'python-version: "3.12"' in job
    assert "cache: pip" in job
    assert "./packaging/windows/pyinstaller/build.ps1" in job
    assert "-Arch x64" in job
    assert '-Extras "rvm,mediapipe,windows"' in job
    assert "-AvatarProfile vision" in job


# -- Phase 6 -----------------------------------------------------------------
def _build_script() -> str:
    return (SPEC_DIR / "build.ps1").read_text(encoding="utf-8")


def _csharp_block_after(text: str, marker: str) -> str:
    """Return the contents of the braced C# block following *marker*."""
    marker_index = text.index(marker)
    block_start = text.index("{", marker_index + len(marker))
    depth = 0
    for index in range(block_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[block_start + 1 : index]
    raise AssertionError(f"unterminated C# block after {marker!r}")


def test_spec_freezes_avatar_second_executable() -> None:
    # WIN-6.4: one onedir payload, two console executables; the shell expects
    # engine\custback-avatar.exe (WIN-5.5) beside the engine.
    text = _spec_text()
    assert "entry_custback_avatar.py" in text
    assert 'name="custback-avatar"' in text
    assert text.count("exclude_binaries=True") == 2
    assert text.count("COLLECT(") == 1  # shared payload, not two artifacts


def test_spec_avatar_profiles_are_mutually_exclusive() -> None:
    # Each payload carries one independently supported avatar driver profile.
    text = _spec_text()
    assert "CUSTBACK_AVATAR_PROFILE" in text
    assert '_excludes += ["mediapipe"]' in text
    assert '_excludes += ["custback.avatar.audio2face"]' in text
    build = _build_script()
    assert "-AvatarProfile" in build or "AvatarProfile" in build
    assert "separate supported driver profiles" in build


def test_build_script_smokes_frozen_avatar() -> None:
    assert "--smoke" in _build_script()


def test_build_script_guards_acceleration_and_arch_profiles() -> None:
    build = _build_script()
    # WIN-6.2: CUDA and DirectML wheels cannot share a freeze venv.
    assert "mutually exclusive" in build
    # WIN-6.3: no cross-freeze, no CUDA/mediapipe on ARM64.
    assert "arm64" in build
    assert "platform.machine()" in build
    for guard in (
        "'gpu' (CUDA) extra is not available on Windows ARM64",
        "no Windows ARM64 wheel",
    ):
        assert guard in build, f"missing ARM64 guard: {guard}"


def test_shell_supervises_avatar_with_real_cli_contract() -> None:
    shell_dir = SPEC_DIR.parents[0] / "shell"
    engine = (shell_dir / "Engine.cs").read_text(encoding="utf-8")
    start_async = _csharp_block_after(
        engine, "internal async Task StartAsync(CancellationToken cancellationToken)"
    )
    start_avatar = _csharp_block_after(engine, "private void StartAvatar()")

    # The avatar CLI has no "serve" subcommand; supervision must pass the
    # real flags: engine WS source, renderer token path, avatar API token path
    # (paths, never secrets, on the command line — WIN-5.3 discipline).
    assert '"serve"' not in engine
    assert "8711" not in engine
    assert "bool superviseAvatar = _options.SuperviseAvatar;" in start_async
    assert (
        "AvatarPort = superviseAvatar ? ReserveLoopbackPort(Port) : 0;" in start_async
    )
    for flag in ("--enable-api", "--api-host", "--api-plaintext"):
        assert start_async.count(f'"{flag}"') == 1

    supervised_marker = "if (superviseAvatar)"
    assert start_async.count(supervised_marker) == 2
    proxy_block = _csharp_block_after(start_async, supervised_marker)
    for flag in ("--avatar-url", "--avatar-token-file", "--avatar-plaintext"):
        assert proxy_block.count(f'"{flag}"') == 1
        assert start_async.count(f'"{flag}"') == 1
    assert '$"http://127.0.0.1:{AvatarPort}"' in proxy_block
    assert "_options.AvatarTokenFile" in proxy_block

    ready_index = start_async.index("await WaitForReadyAsync")
    launch_guard_index = start_async.index(supervised_marker, ready_index)
    launch_block = _csharp_block_after(
        start_async[launch_guard_index:], supervised_marker
    )
    assert "StartAvatar();" in launch_block

    for flag in (
        "--source",
        "--source-token-file",
        "--source-plaintext",
        "--enable-api",
        "--api-host",
        "--api-token-file",
        "--api-port",
        "--api-plaintext",
    ):
        assert start_avatar.count(f'"{flag}"') == 1
    assert '"--source", $"ws://127.0.0.1:{Port}"' in start_avatar
    assert '"--api-port", AvatarPort.ToString()' in start_avatar

    engine_config_block = _csharp_block_after(
        start_async, "if (File.Exists(_options.EngineConfigFile))"
    )
    avatar_config_block = _csharp_block_after(
        start_avatar, "if (File.Exists(_options.AvatarConfigFile))"
    )
    assert (
        'engineArguments.AddRange(new[] { "--config", _options.EngineConfigFile });'
        in engine_config_block
    )
    assert (
        'avatarArguments.AddRange(new[] { "--config", _options.AvatarConfigFile });'
        in avatar_config_block
    )
    assert engine.count('"--config"') == 2

    assert "--driver" not in engine
    assert "MaxAvatarRestarts" in engine
    csproj = (shell_dir / "Custback.Shell.csproj").read_text(encoding="utf-8")
    assert "win-arm64" in csproj  # WIN-6.3 publish RID


def test_installer_is_arch_parameterized_and_avatar_gated() -> None:
    installer_dir = SPEC_DIR.parents[0] / "installer"
    build = (installer_dir / "build.ps1").read_text(encoding="utf-8")
    # WIN-6.3: MSI/bundle arch and the matching VC++ redistributable.
    assert '"x64", "arm64"' in build
    assert "-arch $Arch" in build
    bundle = (installer_dir / "Bundle.wxs").read_text(encoding="utf-8")
    assert "VC_redist.$(var.Arch).exe" in bundle
    # WIN-6.4/WIN-5.5: the avatar ships only on request; default payload
    # drops custback-avatar.exe before harvesting.
    assert "IncludeAvatar" in build
    assert "custback-avatar.exe" in build
