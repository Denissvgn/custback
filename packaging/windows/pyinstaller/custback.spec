# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir spec for the frozen Windows custback engine (WIN-5.1).

Produces a self-contained one-directory build of the ``custback`` engine that
runs on a clean Windows 11 x64 machine with no Python, Node, pip, or CUDA
toolkit on ``PATH``.  The C# WebView2/tray shell (WIN-5.2) supervises the
``custback.exe`` produced here; the signed per-user installer (WIN-5.6) wraps
the whole ``dist/custback/`` directory.

Design decisions (see WINDOWS_DECISIONS.md / WINDOWS_IMPLEMENTATION_PLAN.md):

* ``onedir``, never ``onefile``.  A one-file build unpacks native DLLs (OpenCV,
  ONNX Runtime, MediaPipe, pyvirtualcam) into a fresh temp directory on every
  launch, which is slower, defeats CUDA DLL discovery (the provider looks beside
  the loaded ``onnxruntime`` module), and reliably trips SmartScreen/antivirus.
  D5 commits to onedir.

* RVM / MediaPipe model weights are **not** bundled.  They are GPL-3 / separately
  licensed (WIN-0.4, D6) and are fetched on first run with checksum verification
  by :mod:`custback.segmentation`.  Bundling is gated on legal sign-off; until it
  clears, the frozen app downloads to ``%LOCALAPPDATA%`` on first use exactly as
  the source build does.  ``excludes``/``datas`` below therefore ship no
  ``*.onnx`` / ``*.tflite`` weights.

* CUDA / cuDNN provider DLLs are discovered at runtime by
  :func:`custback.acceleration.preload_acceleration_dlls` from ``CUDA_PATH`` /
  ``PATH``; the installer (WIN-5.6) places the exact pinned onnxruntime-gpu /
  CUDA components.  Whatever ``onnxruntime`` ships inside its own wheel (the CPU
  provider plus the CUDA provider shim) is collected here so a CPU-only clean VM
  still runs.

This file is executed by PyInstaller (``pyinstaller custback.spec``); the names
``Analysis``, ``PYZ``, ``EXE``, ``COLLECT`` and ``SPECPATH`` are injected by the
tool and are intentionally undefined to a plain interpreter.
"""

import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# --------------------------------------------------------------------------- #
# Resolve the version once so the executable's file-version resource matches
# the single source of truth (pyproject / __init__).  Never hard-code it here:
# the release gate pins the version across every manifest and drift would be a
# silent packaging lie.
# --------------------------------------------------------------------------- #
try:
    from importlib.metadata import version as _dist_version

    CUSTBACK_VERSION = _dist_version("custback")
except Exception:  # pragma: no cover - source tree without an installed dist
    import custback

    CUSTBACK_VERSION = custback.__version__

# --------------------------------------------------------------------------- #
# Native packages PyInstaller cannot fully trace by static analysis.  Each of
# these carries dynamically loaded DLLs and/or data resources.
# --------------------------------------------------------------------------- #
_binaries = []
_datas = []
_hiddenimports = []

for _pkg in ("cv2", "onnxruntime", "mediapipe", "pyvirtualcam", "numpy"):
    try:
        _binaries += collect_dynamic_libs(_pkg)
    except Exception:  # a package absent from this profile is not fatal
        pass

# MediaPipe ships its graph configs and the selfie-segmenter task graph as data;
# without these the delegate import at custback.segmentation succeeds but the
# graph fails to build.  Model *weights* are excluded (licensing/first-run).
for _pkg in ("mediapipe", "cv2"):
    try:
        _datas += collect_data_files(
            _pkg, excludes=["**/*.onnx", "**/*.tflite", "**/*.pb.bin"]
        )
    except Exception:
        pass

# The avatar service config template is loaded via
# importlib.resources.files("custback.avatar")/"avatar.yaml"; a frozen build must
# carry it as a real data file next to the package.
_datas += collect_data_files("custback", includes=["**/*.yaml"])

# uvicorn/websockets select their loop/protocol implementations by string import
# at runtime; FastAPI + pydantic pull optional submodules the same way.
for _pkg in ("uvicorn", "websockets", "anyio", "pydantic"):
    try:
        _hiddenimports += collect_submodules(_pkg)
    except Exception:
        pass

# The RVM/MediaPipe delegates and the Windows filesystem-security backend are
# reached through importlib.import_module / platform dispatch, so static analysis
# does not see them.  pywin32 is a hard Windows dependency of
# custback._platform.windows (LockFileEx / owner-only DACLs / reparse rejection).
_hiddenimports += [
    "custback._platform.windows",
    "onnxruntime",
    "mediapipe",
    "cv2",
    "pyvirtualcam",
    # pywin32 modules imported by custback._platform.windows
    "ntsecuritycon",
    "pywintypes",
    "win32api",
    "win32con",
    "win32file",
    "win32security",
    "winerror",
]

# Trim large, GUI-only, or test-only trees that would otherwise inflate the
# artifact and drag in unwanted native libraries.  OpenCV HighGUI is off by
# default on Windows (D8: WebView/MJPEG preview), so no Qt/Tk is needed.
_excludes = [
    "tkinter",
    "matplotlib",
    "pytest",
    "IPython",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "custback.avatar.audio2face",  # WIN-6.4 second-service parity, not this build
]

block_cipher = None

a = Analysis(
    ["entry_custback.py"],
    pathex=[],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=_hiddenimports,
    hookspath=[os.path.join(SPECPATH, "hooks")],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPECPATH, "rthooks", "pyi_rth_custback.py")],
    excludes=_excludes,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)


def _version_resource():
    """Build a Windows VS_VERSION_INFO resource matching CUSTBACK_VERSION.

    Returns ``None`` on any failure so a non-Windows dev machine can still parse
    this spec; the real build always runs on windows-latest where the win32
    version helpers are present.
    """

    try:
        parts = [int(p) for p in CUSTBACK_VERSION.split(".")[:3]]
    except ValueError:
        return None
    while len(parts) < 4:
        parts.append(0)
    filevers = tuple(parts[:4])
    try:
        from PyInstaller.utils.win32.versioninfo import (
            FixedFileInfo,
            StringFileInfo,
            StringStruct,
            StringTable,
            VarFileInfo,
            VarStruct,
            VSVersionInfo,
        )
    except Exception:
        return None
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=filevers, prodvers=filevers),
        kids=[
            StringFileInfo(
                [
                    StringTable(
                        "040904B0",
                        [
                            StringStruct("CompanyName", "Bramen"),
                            StringStruct("FileDescription", "Custback engine"),
                            StringStruct("FileVersion", CUSTBACK_VERSION),
                            StringStruct("InternalName", "custback"),
                            StringStruct("OriginalFilename", "custback.exe"),
                            StringStruct("ProductName", "Custback"),
                            StringStruct("ProductVersion", CUSTBACK_VERSION),
                        ],
                    )
                ]
            ),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
        ],
    )


_icon = os.path.join(SPECPATH, "custback.ico")
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="custback",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=True: the engine is a headless service.  The shell (WIN-5.2)
    # launches it with CREATE_NO_WINDOW so no console flashes, while still
    # capturing stdout/stderr for supervision and doctor output.
    console=True,
    disable_windowed_traceback=False,
    version=_version_resource(),
    icon=_icon if os.path.exists(_icon) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="custback",
)
