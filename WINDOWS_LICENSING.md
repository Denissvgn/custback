# Windows licensing and compliance review

Status: **planning gate — NOT a legal conclusion** · Phase 0 (WIN-0.4)
Companion to `WINDOWS_DECISIONS.md` and `WINDOWS_FEATURE_MATRIX.md`.

> **Disclaimer.** This document is an engineering planning artifact, not legal
> advice. License identifications below are the commonly understood terms of
> each component and are marked **(verify)** where they must be confirmed
> against the exact version and build actually shipped. A qualified
> legal/compliance review must clear the **Bundle** column before any Windows
> installer that includes these components is distributed. Keeping a model or
> dependency as a first-run download does not by itself settle licensing.

Project license: **MIT** (`pyproject.toml`, `LICENSE`, "Copyright (c) 2026
Bramen"). The tension this review manages is distributing an MIT-licensed
product that, when **frozen into a single installer**, would also distribute
copyleft (GPL) and proprietary components as one combined artifact.

The distinction that matters throughout: today the npm/pip flows have the **user
install** third-party packages themselves (closer to mere aggregation / a
user-invoked install). A PyInstaller `onedir` installer instead **ships those
components inside the distributed artifact** — that is a distribution of the
combined work, which is where copyleft obligations most plausibly attach.

Legend for the two decision columns:
- **CI use** — may the component be used in build/test/CI? (almost always yes)
- **Bundle** — may it be included in the distributed Windows installer artifact?
  **HOLD** = must be cleared by legal before bundling.

---

## Core runtime dependencies (from `pyproject.toml`)

| Component | License (verify) | Usage | CI use | Bundle |
| --- | --- | --- | --- | --- |
| numpy | BSD-3-Clause | import | Yes | Permissive — OK, keep notice |
| opencv-contrib-python | Apache-2.0 (OpenCV) **+ bundled FFmpeg (verify build)** | import + native libs | Yes | **HOLD**: confirm the wheel's FFmpeg is LGPL-built, not GPL; keep third-party notices |
| pillow | HPND (MIT-CMU style) | import | Yes | Permissive — OK |
| pydantic | MIT | import | Yes | OK |
| pyvirtualcam | **GPL-2.0** | import at runtime (virtual-camera output) | Yes | **HOLD — central issue**: bundling GPL-2.0 into the installer likely imposes GPL-2.0 on the combined artifact. See "Central tensions". |
| fastapi | MIT | import | Yes | OK |
| uvicorn | BSD-3-Clause | import | Yes | OK |
| pyyaml | MIT | import | Yes | OK |
| websockets | BSD-3-Clause | import | Yes | OK |
| python-multipart | Apache-2.0 | import | Yes | OK, keep notice |
| httpx | BSD-3-Clause | import | Yes | OK |

## Optional extras

| Component | License (verify) | Extra | CI use | Bundle |
| --- | --- | --- | --- | --- |
| mediapipe | Apache-2.0 (library) | `mediapipe` | Yes | Library OK; **model terms (verify)** — see models table |
| onnxruntime | MIT | `rvm` | Yes | OK |
| onnxruntime-gpu | MIT (wheel) **+ CUDA/cuDNN runtime** | `gpu` | Yes | Wheel OK; **HOLD** on CUDA/cuDNN redistribution (see below) |
| grpcio | Apache-2.0 | `audio2face` | Yes | OK (avatar is post-MVP, WIN-6.4) |
| nvidia-ace, nvidia-audio2face-3d | **NVIDIA proprietary EULA (verify)** | `audio2face` | Yes | **HOLD** — redistribution terms; post-MVP |
| protobuf | BSD-3-Clause | `audio2face` | Yes | OK |
| sounddevice (+ PortAudio) | MIT (both) | `audio2face` | Yes | OK; post-MVP |

## Models (downloaded at first use today)

| Model | License (verify) | Bundle |
| --- | --- | --- |
| Robust Video Matting weights | **GPL-3.0** (upstream RVM release) | **HOLD — central issue**: bundling weights distributes GPL-3.0 material. First-run download avoids bundling but does not settle downstream terms. |
| MediaPipe selfie-segmentation model | **Google model terms (verify)** | **HOLD**: confirm redistribution allowance before bundling |

## Windows-new components (Phases 5–6)

| Component | License (verify) | How used | Bundle |
| --- | --- | --- | --- |
| PyInstaller | GPL-2.0 **with bootloader exception** | build tool; bootloader ships in the frozen app | OK: the exception explicitly permits packaging an app of any license; PyInstaller itself is not "distributed" beyond the exception-covered bootloader (verify current exception text) |
| WebView2 Evergreen runtime | Microsoft proprietary, redistributable per MS terms | detect/install at runtime | Prefer **detect-and-install** over bundling; follow WebView2 distribution terms |
| Visual C++ Redistributable | Microsoft proprietary redistributable | prerequisite | Redistributable under MS terms — follow them |
| OBS Virtual Camera / OBS Studio | **GPL-2.0** | external prerequisite, detected — **not** bundled | Do **not** redistribute OBS components; detect and guide install only |
| Installer toolchain — Inno Setup | permissive (modified BSD-style, verify) | build tool | OK; the tool is not shipped in output |
| Installer toolchain — WiX | MS-RL / MS-PL (verify) | build tool | OK; tool not shipped in output |
| CUDA Toolkit / cuDNN runtime DLLs | **NVIDIA proprietary EULA + CUDA redistributable terms** | bundled runtime DLLs for the CUDA profile | **HOLD**: only the specific DLLs on NVIDIA's redistributable list, under the CUDA Supplement terms; keep the required NVIDIA notices |

## Central tensions (must be resolved before a Windows bundle ships)

1. **pyvirtualcam (GPL-2.0) inside the frozen installer.** This is the single
   largest issue. Options, to be evaluated by legal + engineering:
   - Ship the OBS virtual-camera output through a **separate process / optional
     component** the user installs, preserving the aggregation boundary; or
   - Replace `pyvirtualcam` with a differently licensed virtual-camera path
     (e.g. a native Media Foundation source, WIN-6.1); or
   - Adopt a GPL-compatible distribution policy for the combined Windows
     artifact. Note the model weights issue below interacts with this choice.
2. **RVM weights (GPL-3.0).** Keep as first-run download for the MVP (decision
   D6) and do not bundle until cleared. If bundling is ever desired, the whole
   distributed artifact's compatibility with GPL-3.0 must be assessed.
3. **CUDA/cuDNN redistribution.** Bundle only the DLLs NVIDIA lists as
   redistributable, under the CUDA EULA Supplement, with required notices. Pin
   one qualified version set (also a technical requirement, WIN-4/WIN-5).
4. **OpenCV/FFmpeg build.** Confirm the shipped `opencv-contrib-python` wheel's
   FFmpeg is not GPL-built; otherwise the FFmpeg license flows into the bundle.

## Decisions required before Phase 5 bundling

- [ ] Legal ruling on pyvirtualcam in a frozen artifact (drives virtual-camera
      architecture, interacts with WIN-3.7 / WIN-6.1).
- [ ] Confirm first-run RVM download is the MVP model policy; no weight bundling.
- [ ] Confirm CUDA/cuDNN redistributable DLL set + notices for the CUDA profile.
- [ ] Verify OpenCV wheel FFmpeg licensing.
- [ ] Verify MediaPipe model redistribution terms (only if bundling).
- [ ] Assemble a third-party notices file for the installer (all Apache-2.0 /
      BSD / MIT / HPND notices + any NVIDIA/MS required notices).

## Relationship to the release gate

The Windows release-gate slot (`WINDOWS_DECISIONS.md`, Part B) should require an
SBOM + third-party-notices artifact for the Windows installer, mirroring the
existing `LICENSE-01` blocker discipline (`remediation-blockers.json`), so a
Windows publish cannot proceed without the cleared notices. This is wired in
WIN-5.8, not Phase 0.
