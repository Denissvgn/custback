# Frozen Windows engine (PyInstaller onedir) — WIN-5.1

This directory freezes the `custback` engine into a self-contained
one-directory Windows build. The C# WebView2/tray shell (`../shell/`, WIN-5.2)
supervises the resulting `custback.exe`; the signed per-user installer
(`../installer/`, WIN-5.6) wraps the whole `dist/custback/` tree.

## Layout

| File | Purpose |
| --- | --- |
| `custback.spec` | The onedir PyInstaller spec. Collects OpenCV/ONNX Runtime/MediaPipe/pyvirtualcam natives, `custback` package data (`avatar.yaml`), and pywin32; excludes model weights and GUI toolkits. |
| `entry_custback.py` | Frozen entry script → `custback.__main__:main` with `multiprocessing.freeze_support()`. |
| `hooks/hook-custback.py` | Analysis hook: hidden imports for `custback._platform.*` and segmentation delegates; packaged YAML data. |
| `rthooks/pyi_rth_custback.py` | Runtime hook: makes bundled native DLLs discoverable and sets `CUSTBACK_FROZEN=1`. |
| `build.ps1` | Build + clean-environment smoke driver for `windows-latest`. |

## Build

```powershell
# CPU profile (default)
pwsh packaging/windows/pyinstaller/build.ps1

# CUDA profile on a GPU runner
pwsh packaging/windows/pyinstaller/build.ps1 -Extras "gpu,mediapipe,windows"
```

Output: `dist/custback/custback.exe` plus its onedir payload.

## What is intentionally *not* bundled

* **Model weights** (`rvm_mobilenetv3_fp32.onnx`, `selfie_segmenter.tflite`).
  GPL-3 / separately licensed (WIN-0.4, D6). `custback.segmentation` downloads
  them on first run to `%LOCALAPPDATA%` with SHA-256 verification. Bundling is
  gated on legal sign-off.
* **CUDA / cuDNN runtime DLLs** beyond what the `onnxruntime` wheel ships. The
  installer (WIN-5.6) places the exact pinned components; at runtime
  `custback.acceleration.preload_acceleration_dlls` adds them to the DLL search
  path (non-fatal — a clean CPU-only VM still runs and reports CPU).
* **The avatar/Audio2Face service.** Core background replacement ships first
  (D7); avatar Windows parity is WIN-6.4.

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
