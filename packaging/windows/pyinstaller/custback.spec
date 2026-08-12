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
import platform
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    get_package_paths,
)

# --------------------------------------------------------------------------- #
# Avatar driver profile (WIN-6.4).  The `mediapipe` (vision driver) and
# `audio2face` (gRPC driver) extras cannot share one environment — NVIDIA's
# protocol wheels require protobuf>=5.29 while mediapipe requires protobuf<5 —
# so one frozen payload carries exactly one driver stack.  `vision` is the
# shipped default (D7 / driver `auto`); `audio2face` is the opt-in second
# flavor built from a venv holding the audio2face extra instead of mediapipe.
# --------------------------------------------------------------------------- #
AVATAR_PROFILE = os.environ.get("CUSTBACK_AVATAR_PROFILE", "vision").strip().lower()
if AVATAR_PROFILE not in ("vision", "audio2face"):
    raise SystemExit(
        f"CUSTBACK_AVATAR_PROFILE must be 'vision' or 'audio2face', "
        f"not {AVATAR_PROFILE!r}"
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

# pyvirtualcam publishes no Windows ARM64 wheel.  The matching PEP 508 marker
# in pyproject.toml keeps it out of that environment; mirror the same target
# predicate here so PyInstaller neither collects nor force-imports an absent
# package.  build.ps1 verifies that the running interpreter matches -Arch.
_WINDOWS_ARM64 = sys.platform == "win32" and platform.machine().upper() == "ARM64"
_native_packages = ["av", "cv2", "onnxruntime", "mediapipe", "numpy"]
if not _WINDOWS_ARM64:
    _native_packages.append("pyvirtualcam")

for _pkg in _native_packages:
    try:
        _binaries += collect_dynamic_libs(_pkg)
    except Exception:  # a package absent from this profile is not fatal
        pass

# PyAV is a core dependency and its Cython extension modules reach parts of the
# package and standard library dynamically.  Keep this explicit instead of
# depending solely on the version of pyinstaller-hooks-contrib installed beside
# PyInstaller.
try:
    _hiddenimports += collect_submodules("av")
except Exception as exc:
    raise SystemExit("the frozen build requires an importable PyAV package") from exc
_hiddenimports += ["dataclasses", "fractions", "uuid"]

# Since PyAV 9.1.1, Windows wheels place their private FFmpeg DLLs in the
# sibling ``site-packages/av.libs`` directory.  ``collect_dynamic_libs("av")``
# cannot see outside the package directory.  Preserve that exact sibling
# layout as data: treating the DLLs as ordinary binaries makes PyInstaller
# analyze and duplicate them at the onedir root, while PyAV's wheel loader
# expects ``av.libs``.
if sys.platform == "win32":
    try:
        _av_package_base, _av_package_dir = get_package_paths("av")
    except Exception as exc:
        raise SystemExit("cannot locate PyAV for the frozen build") from exc
    _av_lib_dir = os.path.join(_av_package_base, "av.libs")
    if not os.path.isdir(_av_lib_dir):
        raise SystemExit(f"PyAV wheel is missing its native bundle: {_av_lib_dir}")
    _av_lib_files = [
        os.path.join(_av_lib_dir, _name)
        for _name in sorted(os.listdir(_av_lib_dir))
        if os.path.isfile(os.path.join(_av_lib_dir, _name))
    ]
    if not _av_lib_files:
        raise SystemExit(f"PyAV native bundle is empty: {_av_lib_dir}")
    _datas += [(_source, "av.libs") for _source in _av_lib_files]

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

# Config templates and the immutable system-profile catalog are loaded through
# importlib.resources; a frozen build must carry them as real package data.
_datas += collect_data_files(
    "custback", includes=["**/*.yaml", "**/system-profile-catalog.json"]
)

# uvicorn/websockets select their loop/protocol implementations by string import
# at runtime; FastAPI + pydantic pull optional submodules the same way.  The
# MediaPipe tasks tree (avatar vision driver, WIN-6.4) is likewise reached via
# `mediapipe.tasks.python` at driver-start time, not import time.
for _pkg in ("uvicorn", "websockets", "anyio", "pydantic", "mediapipe"):
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
    # pywin32 modules imported by custback._platform.windows
    "ntsecuritycon",
    "pywintypes",
    "win32api",
    "win32con",
    "win32file",
    "win32security",
    "winerror",
]
if not _WINDOWS_ARM64:
    _hiddenimports.append("pyvirtualcam")

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
]
if _WINDOWS_ARM64:
    # Keep static analysis of the lazy import in custback.vcam aligned with
    # dependency resolution: this backend is unavailable in an ARM64 payload.
    _excludes.append("pyvirtualcam")

# WIN-6.4: exactly one avatar driver stack per payload (see AVATAR_PROFILE).
if AVATAR_PROFILE == "audio2face":
    _excludes += ["mediapipe"]
    _hiddenimports += [
        "custback.avatar.audio2face",
        "grpc",
        "sounddevice",
        "nvidia_ace.audio_pb2",
        "nvidia_audio2face_3d.audio2face_pb2_grpc",
        "nvidia_audio2face_3d.messages_pb2",
    ]
    for _pkg in ("grpc", "sounddevice"):
        try:
            _binaries += collect_dynamic_libs(_pkg)
        except Exception:
            pass
else:
    # The vision profile ships without the gRPC driver; `driver: auto` and an
    # explicit `audio2face` config degrade with a clear error, never a crash.
    _excludes += ["custback.avatar.audio2face"]

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

# WIN-6.4: the avatar service is a second executable in the *same* onedir
# payload — the shell expects engine\custback-avatar.exe (WIN-5.5) and both
# processes share one dependency set, so freezing them together keeps the
# artifact small and the DLL story single-sourced.
a_avatar = Analysis(
    ["entry_custback_avatar.py"],
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
pyz_avatar = PYZ(a_avatar.pure, a_avatar.zipped_data, cipher=block_cipher)


def _version_resource(description, original_filename):
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
                            StringStruct("FileDescription", description),
                            StringStruct("FileVersion", CUSTBACK_VERSION),
                            StringStruct(
                                "InternalName",
                                os.path.splitext(original_filename)[0],
                            ),
                            StringStruct("OriginalFilename", original_filename),
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
    version=_version_resource("Custback engine", "custback.exe"),
    icon=_icon if os.path.exists(_icon) else None,
)

# Same supervision contract as the engine: console=True + CREATE_NO_WINDOW in
# the shell keeps the avatar's stdout/stderr in the supervision log (WIN-5.5).
exe_avatar = EXE(
    pyz_avatar,
    a_avatar.scripts,
    [],
    exclude_binaries=True,
    name="custback-avatar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    version=_version_resource("Custback avatar service", "custback-avatar.exe"),
    icon=_icon if os.path.exists(_icon) else None,
)

coll = COLLECT(
    exe,
    exe_avatar,
    a.binaries,
    a.zipfiles,
    a.datas,
    a_avatar.binaries,
    a_avatar.zipfiles,
    a_avatar.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="custback",
)
