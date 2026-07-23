# Custback native Media Foundation virtual camera (WIN-6.1)

A Windows 11 virtual camera that presents the engine's processed frames to
meeting apps **without OBS**: a C++/WinRT custom media source, activated by the
Windows Frame Server, fed by the engine over a shared-memory frame ring.

This directory is complete, reviewable source. Like the rest of
`packaging/windows/`, it is *built* on `windows-latest`; the byte-static parts
(protocol constants, CLSID consistency, project structure) are verified on any
platform by `tests/test_windows_vcam.py`.

## Architecture

```
custback.exe (engine)                    Frame Server service         meeting app
  vcam_native.NativeVirtualCameraOutput    CustbackVCam.dll             Teams/Zoom/...
  └─ writes BGRX frames into  ──────────►  Activator → MediaSource      │
     Local\CustbackVCamFrame0 (seqlock)    └─ MediaStream reads the     │
                                              newest complete frame  ──►│ RGB32 samples
Custback.Shell.exe
  └─ MFCreateVirtualCamera(CLSID, session lifetime, current user)
     owns camera lifecycle: Start on boot, Stop/Shutdown/Remove on quit
```

- **Frame transport** — `FrameRing.h` mirrors `src/custback/vcam_native.py`
  (the single source of truth): one named page-file section, 64-byte header +
  one BGRX frame, seqlock latest-frame-wins, no cross-process events. The
  camera keeps serving samples (placeholder) when the engine is idle, which is
  what consumers expect from a hardware camera.
- **Lifecycle** — the shell registers the camera with **session lifetime** and
  **current-user access**: the device exists only while the shell runs, so a
  crash leaves nothing behind and uninstall only has to remove the per-user COM
  registration (`Package.wxs`, WIN-5.7). No system-wide state, no driver.
- **Media source** — `Activator` (IMFActivate) is the COM class the Frame
  Server CoCreates in its service process; `MediaSource`/`MediaStream`
  implement the software-camera contract (IMFMediaSourceEx, IMFMediaStream2,
  IKsControl) with one always-selected RGB32 stream at 1280×720\@30 and
  1920×1080\@30.
- **Loop prevention** — the friendly name "Custback Camera" matches the
  existing `custback` marker in `camera_devices.py`, so enumeration never
  offers the output camera back as an input (WIN-3.4).

## Identity

The activator CLSID `{7A4C1B2E-9D35-4E6A-8B1F-52C84D9A6E01}` must match, byte
for byte, in four places (cross-checked by `tests/test_windows_vcam.py`):

| Where | Constant |
| --- | --- |
| `Activator.h` | `kActivatorClsid` / `kActivatorClsidString` |
| `src/custback/vcam_native.py` | `VCAM_CLSID` |
| `packaging/windows/shell/VirtualCameraSession.cs` | `ActivatorClsid` |
| `packaging/windows/installer/Package.wxs` | vcam registration + uninstall cleanup |

## Build

```powershell
pwsh packaging/windows/vcam/build.ps1              # x64
pwsh packaging/windows/vcam/build.ps1 -Arch arm64  # WIN-6.3
```

Requires VS 2022 build tools and Windows 11 SDK 10.0.22000+ (the first SDK
carrying `MFCreateVirtualCamera`). The produced `CustbackVCam.dll` is placed by
the installer next to the shell and registered per-user (HKCU only, no
elevation, matching the per-user install of D5).

## Gate (feature-matrix guardrail)

`output.backend: native` is explicit opt-in; `auto` keeps the OBS path until
this camera passes its own clean-machine gate in the validation matrix
(visible + stable in Teams, Zoom, Meet, and a browser test page on a machine
with no OBS installed). Until then the feature stays **Planned (WIN-6.1)** in
`WINDOWS_FEATURE_MATRIX.md` and is not advertised.
