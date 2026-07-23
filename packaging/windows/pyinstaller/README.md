# Frozen Windows engine (PyInstaller onedir) — WIN-5.1 / WIN-6.4

This directory freezes the `custback` engine **and the avatar service** into
one self-contained one-directory Windows build. The C# WebView2/tray shell
(`../shell/`, WIN-5.2) supervises the resulting `custback.exe` (and
`custback-avatar.exe` when installed, WIN-5.5/WIN-6.4); the signed per-user
installer (`../installer/`, WIN-5.6) wraps the whole `dist/custback/` tree.

## Layout

| File | Purpose |
| --- | --- |
| `custback.spec` | The onedir PyInstaller spec. Two console executables (engine + avatar) over one shared payload; collects OpenCV/ONNX Runtime/MediaPipe/pyvirtualcam natives, `custback` package data (`avatar.yaml`), and pywin32; excludes model weights and GUI toolkits. Avatar driver stack chosen by `CUSTBACK_AVATAR_PROFILE` (vision default; audio2face swaps stacks — protobuf conflict makes them one-per-payload). |
| `entry_custback.py` | Frozen entry script → `custback.__main__:main` with `multiprocessing.freeze_support()`. |
| `entry_custback_avatar.py` | Frozen entry script → `custback.avatar.__main__:main` (WIN-6.4). |
| `hooks/hook-custback.py` | Analysis hook: hidden imports for `custback._platform.*` and segmentation delegates; packaged YAML data. |
| `rthooks/pyi_rth_custback.py` | Runtime hook: makes bundled native DLLs discoverable and sets `CUSTBACK_FROZEN=1`. |
| `build.ps1` | Build + clean-environment smoke driver for `windows-latest` (engine synthetic run + avatar `--smoke`). |

## Build

```powershell
# CPU profile (default; vision avatar driver)
pwsh packaging/windows/pyinstaller/build.ps1

# CUDA profile on a GPU runner
pwsh packaging/windows/pyinstaller/build.ps1 -Extras "gpu,mediapipe,windows"

# DirectML profile (WIN-6.2; mutually exclusive with the CUDA extra)
pwsh packaging/windows/pyinstaller/build.ps1 -Extras "directml,mediapipe,windows"

# ARM64 (WIN-6.3; requires ARM64 Python — no cross-freeze)
pwsh packaging/windows/pyinstaller/build.ps1 -Arch arm64

# Audio2Face avatar flavor (WIN-6.4; swaps the conflicting driver stack)
pwsh packaging/windows/pyinstaller/build.ps1 -AvatarProfile audio2face
```

Output: `dist/custback/custback.exe` + `dist/custback/custback-avatar.exe`
plus their shared onedir payload.

## What is intentionally *not* bundled

* **Model weights** (`rvm_mobilenetv3_fp32.onnx`, `selfie_segmenter.tflite`).
  GPL-3 / separately licensed (WIN-0.4, D6). `custback.segmentation` downloads
  them on first run to `%LOCALAPPDATA%` with SHA-256 verification. Bundling is
  gated on legal sign-off.
* **CUDA / cuDNN runtime DLLs** beyond what the `onnxruntime` wheel ships. The
  installer (WIN-5.6) places the exact pinned components; at runtime
  `custback.acceleration.preload_acceleration_dlls` adds them to the DLL search
  path (non-fatal — a clean CPU-only VM still runs and reports CPU).
* **Both avatar driver stacks at once.** One payload carries either the
  vision (MediaPipe) or the audio2face (gRPC) driver — their protobuf
  requirements conflict (see the `audio2face` extra in `pyproject.toml`), so
  `-AvatarProfile` selects exactly one and `driver: auto` degrades cleanly in
  the other flavor. The installer additionally ships `custback-avatar.exe`
  only with `-IncludeAvatar` (D7: core first).

## Why `onedir`, `console=True`

* `onedir` keeps native DLLs on disk beside the exe so CUDA provider discovery
  and antivirus behave; `onefile` re-extracts to temp on every launch.
* `console=True` keeps stdout/stderr available to the supervising shell for
  doctor/diagnostics; the shell launches the process with `CREATE_NO_WINDOW`
  so no console window appears to the user.

## Status

`IMPL*` — spec and hooks are complete and reviewable; the freeze itself and the
clean-VM artifact smoke run on the `windows-latest` CI job (WIN-1.8 / WIN-5.8),
which is where WIN-5.1 flips to `DONE`. The spec/hooks are byte-static Python
and are syntax-checked in the Linux suite (`tests/test_windows_packaging.py`).
