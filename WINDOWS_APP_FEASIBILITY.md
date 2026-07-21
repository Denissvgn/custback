# Windows desktop feasibility and GPU/CPU runtime plan

Assessment date: 2026-07-17

Scope: planning analysis only; no Windows implementation is included here.

## Executive conclusion

Custback can become a Windows desktop application without rewriting its camera,
segmentation, compositing, API, or browser UI. The core design is already a good
fit: a local Python frame engine, an authenticated loopback API, a self-contained
web UI, and adapter boundaries around capture, segmentation, and virtual-camera
output.

This is nevertheless a medium-to-large port, not a packaging toggle. The main
obstacle is not image processing or ONNX Runtime. It is the project's deliberate
use of POSIX filesystem security and transaction primitives (`fcntl`, `flock`,
UID/mode checks, directory `fsync`, and Linux/macOS no-replace rename calls).
Those protect tokens and uploaded files, so Windows support must provide real
NTFS/Win32 equivalents rather than skip the checks.

Recommended first product shape:

- Windows 11 x64, CPython 3.12.
- Existing Python engine packaged as a one-directory payload.
- Small C# desktop/tray shell using WebView2 and the existing web UI.
- OBS Virtual Camera plus the existing `pyvirtualcam` adapter for the first
  release; a native Windows virtual camera is a later, separate component.
- Acceleration policy `auto` by default: prove GPU execution, otherwise start
  the same RVM model on CPU; also offer an explicit CPU mode.
- NVIDIA CUDA first because that is the project's existing accelerated path.
  Run an early DirectML/Windows ML spike before deciding whether the default
  Windows build should cover AMD and Intel GPUs too.
- Avatar and Audio2Face parity after the core camera application is stable.

The most useful next step is a bounded Windows proof, not a wholesale desktop
rewrite. Its success gates should be: native Windows import, synthetic CPU run,
real camera capture, OBS virtual-camera output, actual RVM GPU proof with CPU
fallback, and execution from a clean packaged artifact.

## What can be reused

| Area | Reuse | Evidence and implication |
| --- | --- | --- |
| Frame pipeline | High | `Pipeline._open_resources` already composes capture, segmenter, refiner, backdrop, and output adapters (`src/custback/pipeline.py:864`). The main loop is platform-neutral NumPy/OpenCV work (`src/custback/pipeline.py:1827`). |
| Camera capture | Medium/high | The real source uses generic `cv2.VideoCapture` (`src/custback/capture.py:232`). V4L2 negotiation is conditional, so Windows can use OpenCV's Media Foundation or DirectShow backend without changing the pipeline. |
| Segmentation/compositing | High | RVM uses ONNX Runtime and NumPy; MediaPipe and heuristic fallbacks are also isolated (`src/custback/segmentation.py:464`, `src/custback/segmentation.py:515`, `src/custback/segmentation.py:794`). |
| Virtual-camera output | High at the interface, medium in deployment | `VideoOutput`, `NullOutput`, and `PyVirtualCamOutput` are already adapters (`src/custback/vcam.py:21`). Windows needs a supported camera backend and setup flow, not a pipeline rewrite. |
| API and UI | High | FastAPI is local by default and the UI is one embedded HTML/CSS/JS document (`src/custback/api/webui.py:1`). It can be loaded unchanged inside WebView2. |
| Runtime configuration | High | Strict Pydantic/YAML configuration and transactional hot reconfiguration can remain behind the desktop shell (`src/custback/config.py:500`, `src/custback/config.py:636`). |
| Status/diagnostics | Medium/high | Backend, device, timing, FPS, and fallback fields already reach `/status` (`src/custback/pipeline.py:1284`, `src/custback/hub.py:326`). GPU fallback needs richer semantics, not a new telemetry system. |
| Optional avatar service | Medium | It is already a separate local/remote service. Packaging and some storage/drivers are additional Windows work, so it should not block the first camera release. |

The resulting process relationship should remain simple:

```text
Windows tray/WebView2 shell
        |
        | supervises and authenticates
        v
localhost FastAPI + existing web UI
        |
        v
OpenCV camera -> RVM segmenter -> compositor -> pyvirtualcam/OBS -> meeting app
                     GPU first
                     CPU fallback
```

The live application should run in the interactive user's session, not as a
Windows service. Physical cameras, virtual cameras, WebView2, tray behavior, and
privacy prompts are all user-session concerns.

## Current Windows blockers

### P0: the Python process cannot currently start on native Windows

`src/custback/segmentation.py:23` imports `fcntl` unconditionally. Its model
cache lock uses `flock` and `os.fchmod` (`src/custback/segmentation.py:121`).
Importing the module therefore fails before selecting even the heuristic or
CPU backend. The avatar driver imports the same model-acquisition code.

This should become a platform locking abstraction. A Windows implementation
can use `LockFileEx`, a named mutex, or a well-reviewed cross-platform library,
but it must retain bounded waiting, exclusive writers, checksum verification,
and atomic publication.

### P0: upload and credential security is POSIX-specific

The durable ownership ledger intentionally requires POSIX `flock`
(`src/custback/storage_tx.py:272`) and only implements atomic no-replace rename
with Linux `renameat2` or macOS `renamex_np` (`src/custback/storage_tx.py:57`).
It is used by background uploads and both avatar stores. Token provisioning,
logging, and store hardening also depend on `0600`/`0700`, UID ownership,
`os.fchmod`, and no-follow descriptor checks, for example:

- `src/custback/api/security.py:97`
- `src/custback/diagnostics.py:175`
- `src/custback/api/server.py:524`
- `src/custback/avatar/store.py:142`

Windows needs an equivalent contract based on the current user's SID, private
DACLs, reparse-point rejection, Win32 file identity, share modes, file locks,
and atomic create/move behavior. Silently accepting inherited ACLs or omitting
the transaction ledger would weaken existing security and crash recovery.

This is the largest and highest-risk part of the port.

### P0: current npm distribution explicitly excludes Windows

- `package.json:40` lists only `linux` and `darwin`.
- `packaging/npm/install.js:506` rejects `win32`.
- `packaging/npm/custback.js:118` only knows the Linux/macOS setup scripts.
- Venv paths are hard-coded as `bin/python`, `bin/custback`, and
  `bin/custback-avatar` (`packaging/npm/install.js:281`,
  `packaging/npm/custback.js:29`). Windows venvs use `Scripts` and `.exe`/`.cmd`
  launchers.
- Atomic environment promotion depends on directory symlinks
  (`packaging/npm/managed-venv.js:333`), which needs a junction/symlink and
  privilege redesign on Windows.

Porting this npm lifecycle is possible, but it is poor consumer-desktop UX: it
would still require Node, Python, networked pip, and platform setup. The Windows
app should initially use its own frozen runtime and installer. npm-on-Windows
can remain a later developer distribution if there is demand.

### P1: paths use Unix/XDG conventions

Defaults currently point to `~/.config`, `~/.local/share`, `~/.cache`, and
`~/.local/state`, including:

- models: `src/custback/segmentation.py:91`
- backgrounds: `src/custback/backgrounds.py:49`
- API tokens: `src/custback/api/security.py:25`
- logs: `src/custback/diagnostics.py:267`
- avatar assets: `src/custback/avatar/config.py:320`

Introduce one platform-path module and route every default through it. On
Windows, use Known Folders such as `%LOCALAPPDATA%\Custback` and
`%APPDATA%\Custback`; do not spread Windows conditionals through storage,
security, and model code. Define migration behavior before changing existing
users' paths.

### P1: Windows camera behavior is unqualified

Generic OpenCV capture should work, but a product needs more than camera index
`0`:

- enumerate friendly device names and stable identifiers;
- select and test Media Foundation versus DirectShow explicitly;
- handle Windows camera privacy denial with a useful message;
- avoid offering the Custback/OBS output camera as the input camera;
- test negotiated resolution/FPS reporting on Windows backends;
- handle index reorder, unplug/replug, suspend, resume, and meeting-app
  contention.

The current bounded reader and recovery controller are valuable foundations,
but Windows hardware tests are still required.

### P1: virtual-camera setup has no Windows path

The current implementation already calls `pyvirtualcam.Camera`, which supports
OBS Virtual Camera on Windows. The missing work is detection, first-run setup,
device selection, health reporting, and clean-machine validation. Existing
error/help text only mentions Linux and macOS (`src/custback/vcam.py:89`).

### P1: no Windows CI or release evidence exists

Core CI is Ubuntu-based and package smoke covers Ubuntu/macOS only
(`.github/workflows/ci.yml:43`, `.github/workflows/ci.yml:208`). CUDA release
qualification is Linux-only (`.github/workflows/release.yml:275`,
`.github/workflows/release.yml:786`). Windows needs its own software and
hardware gates.

The repository's production release is also deliberately blocked by open
`REL-01` (`REMEDIATION_PLAN.md:3`,
`scripts/release/remediation-blockers.json:170`). Add Windows release evidence
after extending that exact-commit gate; do not create a bypass around it.

## GPU-first, CPU-fallback assessment

### What exists today

The runtime is close to the desired ordering once the GPU extra is installed:

- automatic backend quality order is RVM -> MediaPipe -> heuristic
  (`src/custback/segmentation.py:794`);
- RVM orders `CUDAExecutionProvider`, `CoreMLExecutionProvider`, then
  `CPUExecutionProvider` (`src/custback/segmentation.py:535`);
- the installer has a strong CUDA probe that profiles a real ONNX node rather
  than trusting provider registration (`src/custback/gpu_probe.py:1`);
- status already exposes `segmentation_backend` and `segmentation_device`.

However, current defaults do not meet the requested behavior:

- a normal npm install implicitly tries MediaPipe, not RVM/GPU
  (`packaging/npm/install.js:110`);
- MediaPipe's delegate defaults to CPU (`config/default.yaml:34`);
- explicitly selecting the `gpu` extra rejects installation unless CUDA
  inference succeeds (`packaging/npm/install.js:360`), rather than keeping a
  healthy CPU RVM path;
- `RVMSegmenter.segment()` has no application-owned recovery around
  `session.run` (`src/custback/segmentation.py:557`); a later GPU/DLL/OOM failure
  can terminate the pipeline;
- device status is inferred from provider order, not updated after an internal
  provider fallback;
- the CUDA probe proves a tiny `Add` graph, not the actual RVM graph;
- there is no way to force RVM CPU while an accelerator provider exists;
- "GPU" currently accelerates segmentation inference only. Compositing, blur,
  most transforms, and local avatar drawing remain CPU OpenCV/NumPy work.

### Recommended runtime contract

Add explicit acceleration configuration separate from the MediaPipe delegate:

```yaml
acceleration:
  mode: auto          # auto | cpu | gpu_required
  provider: auto      # auto | cuda | directml (when implemented)
  device_id: 0
```

Semantics:

- `auto` is the default. Prefer the selected GPU provider, but a GPU problem is
  non-fatal and latches the process onto CPU RVM.
- `cpu` constructs only `CPUExecutionProvider`; it is deterministic and useful
  for troubleshooting, battery use, and tests.
- `gpu_required` preserves a strict diagnostic/deployment contract: startup
  fails if actual accelerator execution cannot be proved. This is not the
  consumer default.

Application-owned state should be observable:

```text
STARTING
   -> GPU_PROBING -> GPU_ACTIVE
                  -> CPU_FALLBACK
   -> CPU_REQUESTED -> CPU_ACTIVE

GPU_ACTIVE -- init/run/DLL/OOM failure --> CPU_FALLBACK (latched until restart)
```

Implementation behavior:

1. Load the verified RVM model before opening camera/output resources.
2. Probe the preferred provider non-fatally. On Windows/CUDA, preload the
   required DLLs when supported by the pinned ONNX Runtime version.
3. Warm up the actual RVM graph on a synthetic frame. The existing small probe
   remains useful for diagnostics, but it is not sufficient product evidence.
4. If provider initialization or warm-up fails, create a CPU-only RVM session.
5. If an active GPU session later fails, create a CPU-only session, clear RVM
   recurrent state and `last_foreground`, retry the current frame once, and stay
   on CPU. Do not retry GPU on every frame.
6. Publish requested policy, active provider, fallback state/reason/count, and
   transition time through `/status`, doctor, logs, and the desktop UI.
7. Keep fallback reasons bounded and credential/path-free, as current
   diagnostics require.

The CPU fallback preserves function and the RVM output contract, but it may not
preserve 30 FPS on every machine. Measure and display FPS attainment. If a
lower-quality real-time CPU profile is desired (MediaPipe or a lower RVM
internal resolution), make that a named user choice rather than a silent
algorithm change.

### GPU vendor scope

There are three viable product positions:

| Position | Provider | Benefit | Cost/risk |
| --- | --- | --- | --- |
| Lowest-risk first release | CUDA -> CPU | Reuses current code, probe, docs, and NVIDIA positioning. A single GPU-capable ORT build also contains the CPU provider. | AMD/Intel machines run CPU-only; CUDA/cuDNN DLL packaging is large and must be pinned/qualified. |
| Universal Windows build | DirectML -> CPU | Covers recent DirectX 12 GPUs from NVIDIA, AMD, and Intel through one Windows provider. | Current code does not select `DmlExecutionProvider`; it requires DML session options and real RVM compatibility/performance tests. DirectML is supported but in sustained engineering. |
| Long-term Windows-native acceleration | Windows ML -> provider -> CPU | Windows can manage hardware-specific GPU/NPU providers and a universal CPU fallback. | Newer deployment model, framework/runtime dependency, Python integration and frozen-app behavior need a dedicated spike. |

Recommendation: implement and release-qualify CUDA -> CPU first if the existing
NVIDIA scope is acceptable. In the same project phase, benchmark RVM on
DirectML and prototype Windows ML packaging. If generic "GPU" support is a hard
requirement, that spike becomes a release gate and the product must not be
described as GPU-capable on all Windows hardware until it passes.

ONNX Runtime CPU, CUDA, and DirectML Python distributions expose the same
`onnxruntime` module and generally cannot be treated as independent packages in
one ordinary environment. Prefer separate, explicitly tested product profiles
or the newer plugin/provider-management model; do not install conflicting ORT
wheels together.

## Virtual camera choices

| Choice | Windows scope | Recommendation |
| --- | --- | --- |
| OBS Virtual Camera through `pyvirtualcam` | Windows 10/11 where a compatible OBS installation is present | Use for the first release. Detect it, guide setup, and test its single-camera-instance limitation. Do not silently redistribute partial OBS components. |
| Unity Capture through `pyvirtualcam` | Windows with the third-party virtual camera installed | Possible secondary compatibility option, but adds another installer/driver and maintenance surface. |
| Native Media Foundation virtual camera | Windows build 22000+ | Best eventual Windows 11 experience, but it requires a native custom media source, registration/lifecycle code, packaging, and uninstall cleanup. Treat as its own C++/WinRT project. |
| Custom legacy DirectShow/driver path | Broad legacy scope | Avoid unless Windows 10 support and no external prerequisite are mandatory; signing and long-term driver maintenance are disproportionate. |

For the MVP, the installer should detect OBS Virtual Camera and offer a clear,
consent-based setup path. Normal app execution should remain unprivileged.

## Recommended desktop and packaging architecture

### Desktop shell

A small C# WebView2/tray launcher is the best balance for a polished Windows
app. It should:

- enforce a single instance with a named mutex;
- start and supervise the packaged Python engine;
- select/reserve the loopback port and wait for explicit readiness;
- load the existing UI in WebView2;
- establish the current HttpOnly browser session without putting the long-lived
  token in a URL, command line, web storage, or log;
- show startup/doctor failures before the API is available;
- keep processing when the window is hidden to the tray;
- request graceful shutdown and wait for camera/output cleanup;
- handle engine crashes, suspend/resume, and optional launch-at-login;
- supervise the second avatar process only when that feature is installed.

The API should remain bound to numeric loopback. App mode needs a private
shutdown/lifecycle channel and a safe one-time WebView session bootstrap. The
native shell should not weaken the existing Host, Origin, cookie, and bearer
boundaries just because both processes are local.

Electron is feasible but adds another Chromium/Node runtime on top of Python,
OpenCV, and ONNX Runtime. Tauri is lighter but introduces Rust and still needs a
sidecar. A full WinUI/C# rewrite of image processing would discard working code
without solving the virtual-camera or ML deployment problems.

### Python payload

Use a PyInstaller `onedir` artifact first, built on Windows with CPython 3.12.
It is easier to inspect and debug native DLL collection than `onefile`, avoids
extracting a large ML payload on every launch, and gives CUDA/ORT clearer DLL
locations. PyInstaller builds are OS-specific, so Windows artifacts must be
built and tested on Windows.

The build will need an explicit spec/hooks for:

- dynamic optional imports in segmentation;
- ONNX Runtime provider DLLs;
- OpenCV, MediaPipe, Pillow, and `pyvirtualcam` native components;
- `custback.avatar/avatar.yaml` loaded through package resources;
- pinned model files if licensing permits bundling;
- version metadata, icon, and third-party notices.

Test the unpacked artifact with `PATH`, `PYTHONPATH`, Python, Node, build tools,
and CUDA toolkit assumptions removed. Consider a Windows dependency profile
using `opencv-contrib-python-headless` because WebView2/MJPEG replaces
`cv2.imshow`; retain the full OpenCV wheel only if the native HighGUI preview is
a supported Windows feature.

Package the payload and shell in a signed per-user EXE/MSI installer (for
example WiX or Inno Setup). A later MSIX/Store path is possible, but should not
block the first clean-machine installer. Include or detect:

- WebView2 Evergreen runtime;
- compatible Visual C++ runtime;
- OBS Virtual Camera prerequisite for the MVP;
- exact ONNX Runtime/CUDA/cuDNN components for the chosen profile;
- start-menu shortcut, uninstaller, upgrade/rollback behavior, version info,
  signatures, and timestamping.

Do not delete user backgrounds, rigs, configuration, or logs on ordinary
uninstall without an explicit user choice. If a future native virtual camera is
registered, uninstall must remove that registration.

## Delivery sequence and exit gates

These sizes are relative engineering scope, not schedule estimates.

| Phase | Scope | Size | Exit gate |
| --- | --- | --- | --- |
| 0. Baseline | Start from the qualified release baseline and define Windows support/version/vendor decisions. | S | Existing exact-commit release gates remain intact; Windows work has an explicit feature matrix. |
| 1. Portable core | Platform paths, model lock, conditional platform code, Windows dependency resolution. | M | `python -m custback --synthetic --no-vcam --no-api` and CPU RVM run on `windows-latest`. |
| 2. Windows security/storage | SID/DACL privacy, reparse-point policy, lock/no-replace transaction primitives, token/log/store tests. | L/XL | Synthetic run with the full local API works; upload, restart recovery, concurrency, and token privacy tests pass on NTFS. |
| 3. Camera/output MVP | Friendly input enumeration, Media Foundation/DirectShow qualification, OBS detection/setup, pyvirtualcam output. | L | Physical camera -> Custback -> OBS Virtual Camera is visible and stable in Teams, Zoom, Chrome/Meet, and a browser test page. |
| 4. Acceleration policy | Actual-RVM warm-up, CUDA -> CPU fallback, CPU force mode, truthful status/doctor, DirectML/Windows ML spike. | L | Clean CPU-only machine starts; NVIDIA machine proves RVM GPU nodes; injected startup and mid-run GPU failures retry on CPU without pipeline exit. |
| 5. Desktop product | PyInstaller onedir, WebView2/tray shell, session bootstrap, installer, signing, upgrade/uninstall. | L | Clean VM with no Python/Node/build tools installs, runs, upgrades, and uninstalls; no secret appears in process arguments or URLs. |
| 6. Extended Windows support | Native Win11 virtual camera, generic GPU provider, ARM64, avatar/A2F as separately approved work. | XL | Each feature has clean-machine, hardware, security, packaging, and release evidence rather than inheriting the core claim. |

Phase 1 can use `--no-api` to isolate import/compute portability, but Phase 2 is
not optional for a real desktop build because the UI and uploads depend on the
API stores.

## Required validation matrix

### Automated Windows CI

- Python 3.12 x64 core, unit, and package tests.
- Import and synthetic/null-output smoke from source and frozen artifact.
- CPU RVM inference with the real pinned model.
- Model-lock contention, timeout, interrupted download, and checksum tests.
- NTFS ACL, reparse point, file identity, no-replace publication, crash
  recovery, quota, and concurrent upload tests.
- Loopback authentication/session, WebSocket, MJPEG, and shutdown tests.
- Installer build, signature verification, SBOM/notices, and artifact-content
  checks.
- Upgrade from the previous Windows artifact and uninstall/data-retention tests.

### Hardware/manual release gates

- Windows 11 x64 with no compatible GPU: CPU fallback starts and is reported.
- Windows 11 x64 with supported NVIDIA GPU: actual RVM inference is profiled on
  CUDA, not merely provider registration.
- Advertised CUDA provider with missing/incompatible DLLs: non-fatal CPU start.
- GPU failure after several frames, including an out-of-memory-style failure:
  the current frame retries once on CPU, recurrent state resets, and fallback
  remains latched.
- AMD and Intel devices if a DirectML/Windows ML claim is made.
- Multiple physical cameras, privacy denied, unplug/replug, index reorder,
  suspend/resume, and capture-device contention.
- OBS missing, installed, busy, and already used by another process.
- Enumeration and sustained calls in Teams, Zoom, Chrome/Meet, and at least one
  additional DirectShow/Media Foundation consumer.
- 720p and 1080p measurement of capture FPS, segmentation time, output FPS,
  memory, GPU memory, CPU use, startup time, and fallback performance.
- Online and offline/blocked-network first run according to the declared
  installer contract.

## Product and compliance risks

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Windows security semantics | Replacing `chmod`/`flock` naively can expose tokens or break crash-safe ownership. | Treat NTFS primitives as a first-class platform implementation with adversarial tests and review. |
| GPU package size/DLL compatibility | `onnxruntime-gpu` plus CUDA/cuDNN can make a large installer and fail from version mismatch. | Pin one qualified version set, use one-directory layout, preload/probe explicitly, and test a clean machine without a toolkit. |
| CPU fallback performance | Functional fallback may miss the configured frame rate. | Measure, expose attainment, and provide explicit quality/performance presets. |
| Virtual camera dependency | OBS is external and provides a single camera instance. | Detect and explain it; keep native Win11 output as a planned independent component. |
| Device identity | OpenCV indices are unstable and can include the output camera. | Add friendly enumeration/stable mapping and loop prevention. |
| Frozen native modules | Dynamic imports and provider DLLs are easy to omit. | Explicit PyInstaller spec plus unpacked/clean-VM artifact tests. |
| New Windows ML surface | It is attractive for broad hardware but newer and framework-dependent in Python. | Prototype and benchmark before committing the main product to it. |
| Code signing/reputation | Unsigned camera software and installers create trust and SmartScreen friction. | Sign and timestamp every executable/installer; include signatures in release gates. |
| Licensing | The project is MIT, while upstream `pyvirtualcam` is GPL-2.0 and the RVM repository/model release is GPL-3.0. Bundling them may create source/notice/distribution obligations. | Perform a real legal/compliance review before freezing or bundling dependencies/models. If the desired distribution is incompatible, replace the dependency/model or choose a compatible distribution policy. Also audit OBS, OpenCV/FFmpeg, CUDA/cuDNN, MediaPipe model, installer, and WebView2 terms. |
| Offline model behavior | Models are currently downloaded at first use; failure can select a lower-quality backend. | Bundle only after licensing approval, or make the download/quality state explicit in first-run UI and cache it with existing hash checks. |

The licensing item is a planning gate, not a legal conclusion. Keeping a model
as a first-run download should not be assumed to settle licensing questions.

## Decisions needed before implementation planning

Recommended defaults are shown first.

1. **Minimum OS/architecture:** Windows 11 x64 first; add Windows 10 or ARM64
   only for a concrete user requirement.
2. **Meaning of GPU:** NVIDIA CUDA for the lowest-risk first release, with an
   early DirectML/Windows ML go/no-go spike for AMD/Intel support.
3. **Virtual camera:** OBS prerequisite first; native Media Foundation camera
   later.
4. **Desktop UX:** WebView2/tray shell around the existing local web app.
5. **Distribution:** signed per-user EXE/MSI plus frozen one-directory Python
   payload; npm remains a developer path, not the Windows consumer installer.
6. **Offline behavior:** prefer a fully deterministic bundle only after model
   and dependency licensing is cleared; otherwise make first-run downloads
   explicit.
7. **Feature parity:** core background replacement first; avatar and
   Audio2Face in a later Windows milestone.
8. **Native preview:** browser/WebView preview only by default; keep HighGUI
   only if it provides a defined support benefit.

## Go/no-go recommendation

Proceed with a Windows technical spike if these constraints are acceptable:

- a native Windows storage/security implementation is in scope;
- an OBS prerequisite is acceptable for the first virtual-camera release;
- initial GPU support may mean NVIDIA, with CPU on other hardware;
- the desktop artifact will be built and hardware-tested on Windows;
- third-party/model licensing is reviewed before distribution.

Do not proceed as a packaging-only effort, do not use Docker/WSL for the host
camera path, and do not begin with a full C#/C++ rewrite. If the spike cannot
preserve security invariants or cannot produce a stable clean-machine camera
path, stop and reconsider the product boundary before investing in the shell.

## External references checked

These links support platform facts in this assessment and should be rechecked
when implementation starts:

- [pyvirtualcam supported virtual cameras](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras)
- [ONNX Runtime execution-provider ordering](https://onnxruntime.ai/docs/execution-providers/)
- [ONNX Runtime CUDA requirements and DLL preloading](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime DirectML provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [Windows ML overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)
- [Windows ML Python deployment](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app)
- [Windows Media Foundation virtual camera API](https://learn.microsoft.com/en-us/windows/win32/api/mfvirtualcamera/nf-mfvirtualcamera-mfcreatevirtualcamera)
- [WebView2 runtime distribution](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution)
- [PyInstaller operating modes and OS-specific builds](https://www.pyinstaller.org/en/stable/operating-mode.html)
- [MSIX signing overview](https://learn.microsoft.com/en-us/windows/msix/package/signing-package-overview)
- [Robust Video Matting repository and license](https://github.com/PeterL1n/RobustVideoMatting)
- [pyvirtualcam repository and license](https://github.com/letmaik/pyvirtualcam)
