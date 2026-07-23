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

import py_compile
from pathlib import Path

import pytest

SPEC_DIR = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "pyinstaller"

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


# -- Phase 6 -----------------------------------------------------------------
def _build_script() -> str:
    return (SPEC_DIR / "build.ps1").read_text(encoding="utf-8")


def test_spec_freezes_avatar_second_executable() -> None:
    # WIN-6.4: one onedir payload, two console executables; the shell expects
    # engine\custback-avatar.exe (WIN-5.5) beside the engine.
    text = _spec_text()
    assert "entry_custback_avatar.py" in text
    assert 'name="custback-avatar"' in text
    assert text.count("exclude_binaries=True") == 2
    assert text.count("COLLECT(") == 1  # shared payload, not two artifacts


def test_spec_avatar_profiles_are_mutually_exclusive() -> None:
    # The mediapipe (vision) and audio2face driver stacks conflict on
    # protobuf; the spec must carry exactly one per payload (WIN-6.4).
    text = _spec_text()
    assert "CUSTBACK_AVATAR_PROFILE" in text
    assert '_excludes += ["mediapipe"]' in text
    assert '_excludes += ["custback.avatar.audio2face"]' in text
    build = _build_script()
    assert "-AvatarProfile" in build or "AvatarProfile" in build
    assert "protobuf conflict" in build


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
    # The avatar CLI has no "serve" subcommand; supervision must pass the
    # real flags: engine WS source, renderer token path, avatar API token path
    # (paths, never secrets, on the command line — WIN-5.3 discipline).
    assert '"serve"' not in engine
    for flag in ("--source", "--source-token-file", "--api-token-file"):
        assert flag in engine, f"avatar supervision is missing {flag}"
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
