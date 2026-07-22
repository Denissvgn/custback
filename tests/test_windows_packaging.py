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
