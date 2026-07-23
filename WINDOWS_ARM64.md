# Windows ARM64 support profile (WIN-6.3)

Status: **machinery landed — Planned, not Supported, until hardware evidence
passes** · Phase 6 · Companion to `WINDOWS_IMPLEMENTATION_PLAN.md`,
`WINDOWS_FEATURE_MATRIX.md`, and `WINDOWS_DECISIONS.md` (D1 keeps Win11 x64 the
baseline; ARM64 is gated on a concrete customer requirement).

This document is the durable record of the three things WIN-6.3 owes:
the **dependency** profile, the **build** path, and the **evidence** an ARM64
"Supported" claim requires. Everything below builds with the same scripts as
x64 — architecture is a parameter, not a fork.

> Wheel-availability rows marked **(verify)** must be re-confirmed against the
> exact pinned versions at build time; PyPI ARM64 coverage moves quickly.

## 1. Dependency profile

| Component | win_arm64 status (verify) | ARM64 profile decision |
| --- | --- | --- |
| numpy, pillow, pydantic, fastapi, uvicorn, pyyaml, websockets, httpx, python-multipart | wheels available | unchanged |
| opencv-contrib-python | arm64 wheels ship from 4.11 | pin within the existing `>=4.8,<6` range |
| onnxruntime (`rvm` extra) | arm64 wheels available | **CPU RVM is the baseline** |
| onnxruntime-gpu (`gpu` extra) | no CUDA on Windows-on-ARM | **refused** by `pyinstaller/build.ps1 -Arch arm64` |
| onnxruntime-directml (`directml` extra) | arm64 wheels available | opt-in; advertising still gated on WIN-6.2 evidence *from ARM64 hardware* |
| mediapipe (`mediapipe` extra) | **no win_arm64 wheel** | **refused**; segmentation falls back per the existing `backend: auto` ladder |
| pywin32 (`windows` extra) | arm64 wheels available (306+) | unchanged — the Phase-2 security backend is mandatory |
| pyvirtualcam | **no win_arm64 wheel; needs local native build** | **excluded** by the core PEP 508 marker and frozen-build spec; the native MF camera **will be the ARM64 output path once WIN-6.1 evidence lands**; until then `auto` produces API-only output (`NullOutput`) and `output.backend: native` is the manual gate-build opt-in |
| grpcio / nvidia-ace (`audio2face` extra) | grpcio arm64 wheels exist; NVIDIA wheels x86_64-only | **out of ARM64 scope** (WIN-6.4 ships the vision profile) |

Consequences encoded in the build scripts:

- `packaging/windows/pyinstaller/build.ps1 -Arch arm64` defaults the extras to
  `rvm,windows` and hard-fails on `gpu` or `mediapipe`.
- The core dependency marker excludes `pyvirtualcam` only when
  `sys_platform == 'win32'` and `platform_machine` is either common casing of
  ARM64; the PyInstaller spec mirrors that predicate for native collection,
  hidden imports, and analysis exclusions.
- The intended Windows `auto` ladder is pyvirtualcam → native → null, but its
  native rung remains disabled by `_AUTO_NATIVE_ENABLED = False` until WIN-6.1
  evidence lands. Consequently an ARM64 build defaults to API-only
  `NullOutput` today; `output.backend: native` is the manual gate-build opt-in.
  The `pyvirtualcam` import stays lazy inside `PyVirtualCamOutput` for other
  platforms and Windows x64.

## 2. Build path (all arch-parameterized, no forks)

```powershell
# engine (on an ARM64 runner/machine with ARM64 Python — no cross-freeze)
pwsh packaging/windows/pyinstaller/build.ps1 -Arch arm64

# shell
dotnet publish packaging/windows/shell -r win-arm64 -c Release

# native virtual camera
pwsh packaging/windows/vcam/build.ps1 -Arch arm64

# installer (chains redist\VC_redist.arm64.exe)
pwsh packaging/windows/installer/build.ps1 -Version <ver> `
    -EngineDir ... -ShellDir ... -Arch arm64 -VCamDll ...
```

Guards that make a wrong build impossible rather than merely discouraged:

- the engine build verifies `platform.machine()` matches `-Arch` (PyInstaller
  cannot cross-freeze),
- the WiX MSI/bundle are built with `-arch arm64` and pick the matching VC++
  redistributable via `$(var.Arch)`,
- `CustbackVCam.vcxproj` declares first-class `ARM64` configurations.

## 3. Evidence required before "Supported"

ARM64 inherits nothing (feature-matrix rule): each row below needs its own
pass on ARM64 hardware, produced through the same gate machinery as x64 and
attached to the WIN-01 release slot when WIN-1.8/WIN-5.8 wire the Windows
evidence source. GitHub's `windows-11-arm` runners cover the automated half;
the camera/consumer matrix needs physical hardware.

| Gate | Content |
| --- | --- |
| arm64-core | import + synthetic + unit suite on ARM64 Python 3.12 |
| arm64-storage-ntfs | the Phase-2 NTFS security/crash suite |
| arm64-camera | physical camera → native MF camera visible in Teams/Zoom/Meet + browser |
| arm64-acceleration | CPU RVM attainment at 720p; DirectML only with WIN-6.2-on-ARM64 evidence |
| arm64-installer-clean-vm | ARM64 bundle install/upgrade/uninstall on a clean ARM64 VM |

Until every row passes, `WINDOWS_FEATURE_MATRIX.md` keeps Win11 ARM64 at
**Planned (WIN-6.3)** and no ARM64 claim ships in user-facing text (the
advertising guardrail).
