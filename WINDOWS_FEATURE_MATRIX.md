# Windows feature matrix

Status: **proposed — pending product sign-off** · Phase 0 (WIN-0.3)
Companion to `WINDOWS_DECISIONS.md`, `WINDOWS_APP_FEASIBILITY.md`, and
`WINDOWS_IMPLEMENTATION_PLAN.md`.

Purpose: state explicitly what the Windows build supports, so that no capability
is implied by inheritance from the Linux/macOS claim. Each **Supported** cell
must map to a phase exit gate that produces clean-machine / hardware / security /
packaging evidence for it. A capability with no such gate is **Not supported**,
regardless of whether the underlying code path appears to run.

Legend:
- **Supported** — has a passing phase exit gate; may be advertised.
- **Planned (WIN-x)** — scoped to a later task; must not be advertised yet.
- **Not planned** — explicitly out of scope for the current product boundary.
- **N/A** — not applicable to that column.

Baseline product target (from `WINDOWS_DECISIONS.md`): **Windows 11 x64**,
**CUDA→CPU**, **OBS Virtual Camera prerequisite**, **WebView2/tray shell**,
**signed per-user installer**.

---

## 1. Platform / architecture

| Capability | Win11 x64 | Win10 x64 | Win11 ARM64 |
| --- | --- | --- | --- |
| Native Python import + synthetic run | Planned (WIN-1) → gate WIN-1.8 | Not planned | Not planned |
| Full local API + NTFS storage | Planned (WIN-2) | Not planned | Not planned |
| Camera → OBS virtual camera | Planned (WIN-3) | Not planned | Not planned |
| Desktop installer | Planned (WIN-5) | Not planned | Not planned |
| Any support at all | Yes (target) | Not planned (revisit D1) | Planned (WIN-6.3) |

Windows 10 and ARM64 are gated behind a concrete customer requirement (decision
D1). ARM64 has an explicit later task (WIN-6.3); Windows 10 reopens the capture
and virtual-camera matrix and has no task yet.

## 2. Segmentation backend × acceleration

| Backend / provider | CPU | NVIDIA CUDA | AMD/Intel (DirectML) | NPU (Windows ML) |
| --- | --- | --- | --- | --- |
| RVM (matting) | Planned (WIN-1/WIN-4) → gate WIN-1.8, WIN-4 | Planned (WIN-4) → hardware gate | Spike only (WIN-4.7); Planned (WIN-6.2) if go | Not planned (spike WIN-4.7) |
| MediaPipe | Planned (WIN-1) | Not planned | Not planned | Not planned |
| Heuristic / none | Planned (WIN-1) | N/A | N/A | N/A |

Acceleration truth rules (from `WINDOWS_APP_FEASIBILITY.md`, formalized in
WIN-4):
- `auto` prefers the selected GPU provider but latches to CPU RVM on any GPU
  problem (non-fatal).
- A machine with no compatible GPU runs CPU RVM and reports it.
- "GPU" accelerates **segmentation inference only**; compositing, blur, most
  transforms, and local avatar drawing remain CPU OpenCV/NumPy work.
- No "all Windows GPUs" claim until WIN-6.2 passes a real DirectML/Windows ML
  release gate.

## 3. Virtual-camera output

| Path | Status | Gate |
| --- | --- | --- |
| OBS Virtual Camera via `pyvirtualcam` | Planned (WIN-3.7) | Camera → Teams/Zoom/Meet/browser stable |
| Unity Capture via `pyvirtualcam` | Not planned (secondary compat option) | — |
| Native Media Foundation virtual camera | Planned (WIN-6.1) | Own C++/WinRT clean-machine gate |
| Legacy DirectShow driver | Not planned | — |

## 4. Camera input

| Capability | Status | Gate |
| --- | --- | --- |
| Generic capture (index) | Planned (WIN-1) — works but insufficient for product | — |
| Friendly enumeration + stable IDs | Planned (WIN-3.1) | — |
| Explicit MSMF vs DSHOW selection | Planned (WIN-3.2) | — |
| Privacy-denial handling | Planned (WIN-3.3) | — |
| Output-camera loop prevention | Planned (WIN-3.4) | — |
| Replug / suspend / contention resilience | Planned (WIN-3.6) | Multi-camera hardware gate |

## 5. Security / storage (NTFS)

| Capability | POSIX today | Windows |
| --- | --- | --- |
| Ownership verification | `effective_uid()` via seam | Planned (WIN-2.6, SID-based); **fail-closed now** (WIN-1.4) |
| Private mode (0600/0700) | `set_private_mode()` via seam | Planned (WIN-2.3, DACL); **fail-closed now** |
| No-follow / reparse rejection | `nofollow_flag()` via seam | Planned (WIN-2.4); **fail-closed now** |
| Exclusive lock | `lock_exclusive()` via seam | Planned (WIN-2.1, LockFileEx); **fail-closed now** |
| Atomic no-replace rename | `renameat2`/`renamex_np` | Planned (WIN-2.2); raises `ENOTSUP` now |
| File identity | `st_dev`/`st_ino` | Planned (WIN-2.5, Win32 file id) |
| Directory durability | dir `fsync` | Planned (WIN-2.7) |

"Fail-closed now" reflects the Phase 1 seam already merged: the primitive raises
`PlatformSecurityUnsupported` on Windows rather than silently passing (CC-1).
Token, upload, model-lock, and secure-log paths therefore refuse to run on
Windows until their WIN-2.x implementations land — which is the intended interim
state, not a regression.

## 6. Application surfaces

| Capability | Status | Gate |
| --- | --- | --- |
| Embedded web UI in WebView2 | Planned (WIN-5.2) | — |
| Secure WebView session bootstrap (no token in URL/args/log) | Planned (WIN-5.3) | Clean-VM: no secret in process args/URLs |
| Tray lifecycle / suspend / crash recovery | Planned (WIN-5.4) | — |
| Avatar second process supervision | Planned (WIN-5.5), feature-gated | — |
| Native HighGUI preview | Not planned (decision D8) | — |

## 7. Distribution

| Capability | Status | Gate |
| --- | --- | --- |
| Signed per-user EXE/MSI installer | Planned (WIN-5.6) | Clean VM install/run/upgrade/uninstall |
| WebView2 Evergreen + VC++ runtime handling | Planned (WIN-5.6) | — |
| OBS prerequisite detection/setup | Planned (WIN-3.7 / WIN-5.6) | — |
| Upgrade + uninstall data-retention | Planned (WIN-5.7) | Data-retention test |
| npm install on Windows | Not planned (developer path only, decision D5) | — |
| MSIX / Store | Not planned (additive, post-MVP) | — |

## 8. Advertising guardrail

Until a row's gate passes, marketing/readme/status text must not claim it for
Windows. In particular: do not describe the Windows build as GPU-accelerated on
non-NVIDIA hardware, as having a driver-free virtual camera, or as supporting
Windows 10/ARM64, before the corresponding WIN-4.7/6.2, WIN-6.1, or WIN-6.3
gates pass.
