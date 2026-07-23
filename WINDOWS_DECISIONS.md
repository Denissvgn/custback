# Windows support decisions and release-gate slot

Status: **proposed — pending product sign-off** · Phase 0 (WIN-0.1, WIN-0.2)
Companion to `WINDOWS_APP_FEASIBILITY.md` and `WINDOWS_IMPLEMENTATION_PLAN.md`.

This file is the durable decision record for the Windows port. It exists so that
Phases 1–6 build against a fixed target rather than re-litigating scope. Every
decision below shows the recommended default first, the rationale, and the
concrete trigger that would justify revisiting it.

Placement note: like `WINDOWS_APP_FEASIBILITY.md` and
`WINDOWS_IMPLEMENTATION_PLAN.md`, this is an internal planning document kept at
the repository root. Root-level `*.md` (other than `LICENSE` and
`REMEDIATION_PLAN.md`) are deliberately **not** shipped to the npm package — the
reviewed npm payload in `scripts/release/verify-release.js` lists exactly one
docs file (`docs/remote-deployment.md`). Adding these under `docs/` would ship
internal planning to end users and break the exact-match payload gate, so they
stay at root.

---

## Part A — Support-matrix decisions (WIN-0.1)

Each decision has a status of **Proposed** until a product owner ratifies it
here (change to **Ratified <date>** with initials). Ratification is the Phase 0
exit condition, not agreement by silence.

### D1 — Minimum OS and architecture
- **Decision (proposed):** Windows 11 x64 only for the first release.
- **Rationale:** Windows 11 gives the Media Foundation virtual-camera API
  (build 22000+) as a later native option, modern WebView2, and a single test
  surface. Windows 10 and ARM64 multiply capture/driver/hardware test cost for
  no committed user yet.
- **Revisit trigger:** a concrete customer requirement for Windows 10 or ARM64.
  ARM64 is scoped separately as WIN-6.3; Windows 10 would reopen the capture and
  virtual-camera test matrix.

### D2 — Meaning of "GPU"
- **Decision (proposed):** NVIDIA CUDA → CPU for the first release. Run an early
  DirectML/Windows ML go/no-go spike (WIN-4.7) before promising AMD/Intel.
- **Rationale:** CUDA is the project's only existing accelerated path
  (`src/custback/segmentation.py` orders `CUDAExecutionProvider` first; the
  installer already has a real CUDA probe in `src/custback/gpu_probe.py`). A
  single GPU-capable ONNX Runtime build also carries the CPU provider.
- **Revisit trigger:** WIN-4.7 shows RVM runs acceptably on `DmlExecutionProvider`
  **and** generic-GPU support becomes a hard requirement — then WIN-6.2 becomes a
  release gate and the product may not be described as GPU-capable on all Windows
  hardware until it passes.

### D3 — Virtual camera
- **Decision (proposed):** OBS Virtual Camera prerequisite (via the existing
  `pyvirtualcam` adapter) for the MVP; native Media Foundation camera later
  (WIN-6.1).
- **Rationale:** `src/custback/vcam.py` already calls `pyvirtualcam.Camera`,
  which supports OBS Virtual Camera on Windows. The missing work is detection,
  consent-based setup, and health reporting — not a pipeline rewrite. A native
  camera is a separate C++/WinRT project with signing and lifecycle cost.
- **Revisit trigger:** "no external prerequisite" becomes mandatory, or the OBS
  single-camera-instance limitation blocks a required workflow.

### D4 — Desktop UX
- **Decision (proposed):** A small C# WebView2 + tray shell around the existing
  local web app.
- **Rationale:** The UI is one embedded HTML/CSS/JS document
  (`src/custback/api/webui.py`) that loads unchanged in WebView2. Electron adds a
  second Chromium/Node runtime on top of Python/OpenCV/ONNX Runtime; Tauri adds
  Rust and still needs a sidecar; a WinUI rewrite discards working code without
  solving the virtual-camera or ML-deployment problems.
- **Revisit trigger:** a requirement WebView2 cannot meet (e.g. an OS version
  without the Evergreen runtime and no ability to bundle it).

### D5 — Distribution
- **Decision (proposed):** A signed per-user EXE/MSI installer wrapping a frozen
  one-directory Python payload (PyInstaller `onedir`). npm remains a developer
  path, not the Windows consumer installer.
- **Rationale:** The npm lifecycle explicitly rejects `win32`
  (`packaging/npm/install.js`) and its venv/symlink assumptions are POSIX-shaped.
  A frozen runtime removes the Node/Python/networked-pip prerequisites that make
  npm poor consumer-desktop UX. See Part B for how this distribution attaches to
  the release gate.
- **Revisit trigger:** MSIX/Store distribution becomes a requirement (additive,
  should not block the first clean-machine installer).

### D6 — Offline / model behavior
- **Decision (proposed):** First-run model download with explicit first-run UI
  and existing hash verification. Prefer a fully deterministic bundle only after
  model licensing is cleared (see `WINDOWS_LICENSING.md`).
- **Rationale:** Models are currently fetched at first use with checksum
  verification (`src/custback/segmentation.py`). Bundling the RVM weights is a
  licensing question (GPL-3), not just a packaging one.
- **Revisit trigger:** WIN-0.4 clears model bundling, or an air-gapped
  deployment requirement lands.

### D7 — Feature parity ordering
- **Decision (proposed):** Core background replacement first. Avatar and
  Audio2Face are a later Windows milestone (WIN-6.4).
- **Rationale:** The avatar service is already a separate process with its own
  storage/drivers; packaging it is additional Windows work that should not block
  the first camera release.
- **Revisit trigger:** avatar is the primary reason a customer wants Windows.

### D8 — Native preview
- **Decision (proposed):** Browser/WebView preview only. Keep OpenCV HighGUI off
  by default on Windows.
- **Rationale:** `src/custback/preview.py` uses `cv2.namedWindow`; WebView2/MJPEG
  replaces it, which also enables the `opencv-contrib-python-headless` dependency
  option (WIN-1.6).
- **Revisit trigger:** a defined support benefit for a native HighGUI window.

### D9 — Packaged avatar control-plane wiring
- **Decision (2026-07-23):** when the shell supervises an installed avatar, it
  reserves a second ephemeral loopback port that is distinct from the engine
  port. It passes `--avatar-url http://127.0.0.1:<avatar-port>` and
  `--avatar-token-file <avatar-api-token>` to the engine, and passes
  `--api-port <avatar-port>` to the avatar service. These flags are absent when
  the avatar is not installed. Command lines carry paths, never token values.
- **Rationale:** the shell is the only component that already knows whether the
  avatar is installed, both process ports, and the packaged token path. Explicit
  CLI wiring avoids mutating a user configuration or coupling the engine to the
  avatar's source-install default port. Reserving a distinct ephemeral port also
  removes fixed-port `8711` collisions. The engine freezes the destination and
  token path at startup but reads the token value per request, so its existing
  engine-before-avatar startup order remains valid.
- **Revisit trigger:** avatar supervision moves out of the shell, or a supported
  deployment needs an externally managed/non-loopback avatar control plane.

---

## Part B — Windows release-gate slot design (WIN-0.2)

**Goal:** define exactly where Windows release evidence attaches to the existing
exact-commit gate, so that when `REL-01` eventually closes and the
Linux/macOS release is unblocked, a *Windows* release still requires
*Windows-specific* evidence — without ever creating a bypass around `REL-01`
(CC-3).

### What the current gate actually enforces (verified)

`scripts/release/verify-release.js` and `scripts/release/remediation-blockers.json`
implement a hash-pinned contract:

- `remediation-blockers.json` uses `schema_version: 2`, carries
  `release_blocked`, and lists blockers each with `id`, `phase`,
  `status` (`open` | `resolved`), `title`, and a `regression` descriptor.
- `remediationRegistry()` validates the shape and enforces
  `release_blocked === (openBlockers > 0)`.
- `remediationContractDigest()` hashes the whole contract (phase, release_blocked,
  and every blocker's id/phase/status/title/regression) with SHA-256, and
  `verifyBlockerRegressionCoverage()` fails unless it equals the pinned
  `REVIEWED_REMEDIATION_CONTRACT_SHA256`
  (`d60119dc48db2a8327348b788da1bef6218f4bf5211dadc74178fdd505042cda` at time of
  writing). **Any** edit to the registry — even a title — invalidates the digest
  until this constant is updated in the same reviewed commit.
- Regression semantics per status:
  - **resolved / pytest:** the node passes normally (no skip/xfail/xpass).
  - **resolved / node:** exactly one non-TODO/non-SKIP `ok` TAP line.
  - **open / pytest:** the node produces a strict `xfailed` (never `xpassed`); a
    `guard` is *forbidden* for pytest.
  - **open / node:** `regression.test` must produce a `# TODO` TAP result **and**
    `regression.guard` must be a second, currently-passing test that proves the
    known failure. `REL-01` is exactly this pattern
    (`packaging/npm/test/release-check.test.js`).
- `required-gates.json` separately lists `required_gates`,
  `workflow.required_job_ids`, `aggregate_job_id`, and `publish_job_id`. The
  release workflow structure is checked against this manifest.

### Why the contract is NOT mutated in Phase 0

Adding a Windows blocker now would require either a fabricated regression test
with no subject, or leaving the gate broken:

- An **open Node** Windows blocker needs a `guard` that proves a *current* known
  failure of the Windows publish machinery. That machinery does not exist until
  Phase 5 (WIN-5.1/5.2/5.6). There is nothing real to prove a failure against.
- Mutating `remediation-blockers.json` also requires recomputing and committing
  `REVIEWED_REMEDIATION_CONTRACT_SHA256`. That hash update is meant to accompany
  reviewed evidence, not a placeholder.

Therefore Phase 0 **specifies** the slot; **WIN-5.8 applies it** once the
Windows evidence machinery exists. This keeps the existing exact-commit gate
intact (the Phase 0 exit condition) and honors CC-3 (never bypass `REL-01`).

### The slot specification (to be applied in WIN-5.8)

1. **New open blocker** appended to `remediation-blockers.json` (do not modify
   `REL-01`):

   ```json
   {
     "id": "WIN-01", "phase": 6, "status": "open",
     "title": "Windows release can publish without Windows clean-machine, camera, acceleration, and installer evidence",
     "regression": {
       "runner": "node",
       "file": "packaging/npm/test/windows-release-check.test.js",
       "test": "WIN-01: Windows production publish requires exact Windows release evidence",
       "guard": "WIN-01: open registry still blocks the installed Windows publish machinery"
     }
   }
   ```

   Notes: the id must match `^[A-Z][A-Z0-9]+-\d{2}$` (so `WIN-01`, not
   `WIN-REL-01`). Keeping it `open` keeps `release_blocked: true` and does not
   disturb `REL-01`'s own lifecycle. When added, recompute
   `REVIEWED_REMEDIATION_CONTRACT_SHA256` in the same commit.

2. **New regression test** `packaging/npm/test/windows-release-check.test.js`,
   modeled on `release-check.test.js`: the `test` asserts (as a TODO until the
   machinery lands) that the publish job refuses to run without the Windows
   evidence artifacts; the `guard` proves that today the installed publish
   machinery does not yet require them. Register the file in
   `REVIEWED_NPM_PAYLOAD` (the exact list in `verify-release.js`).

3. **required-gates.json additions** (Windows evidence, gated behind the Windows
   build being requested — must not force Windows evidence on a Linux-only
   release):
   - `required_gates`: add `windows-core`, `windows-storage-ntfs`,
     `windows-camera-obs`, `windows-acceleration`, `windows-installer-clean-vm`.
   - `matrices`: add a `windows` python matrix row (`windows-latest`, 3.12) and a
     `windows` platform row (`windows-x64-hardware`) for the CUDA/CPU hardware
     gate.
   - `artifacts`: add `windows-installer`
     (`^Custback-Setup-[0-9]+\.[0-9]+\.[0-9]+.*\.exe$` or `.msi`).
   - `workflow.required_job_ids`: add `windows-core`, `windows-artifact-build`,
     `windows-artifact-validation`, `windows-installer`.

4. **Scoping rule:** the Windows gates are required **only** when a Windows
   artifact is part of the release manifest. A Linux/macOS-only release must not
   be blocked by absent Windows evidence, and — symmetrically — a Windows release
   must not be publishable without it. WIN-5.8 encodes this as a manifest-driven
   conditional, verified by the new regression test.

### Applied state (WIN-5.8)

The slot's **blocker half is now live**: `WIN-01` (open, phase 6) is appended to
`remediation-blockers.json` exactly as specified in item 1, and
`REVIEWED_REMEDIATION_CONTRACT_SHA256` was recomputed in the same edit
(`c64a8907…`). The regression `packaging/npm/test/windows-release-check.test.js`
(item 2) is registered in `REVIEWED_NPM_PAYLOAD`; its `# TODO` acceptance
enumerates the item-3 gates/jobs/artifact and the item-4 scoping rule, and its
guard proves today's gap. `REL-01` is untouched and release stays blocked
(CC-3).

The slot's **manifest/workflow half (items 3–4) is intentionally still deferred**
— now to WIN-1.8, not Phase 0. `required-gates.json` is an *exact-match* evidence
contract (`phase6-evidence.js` `exactKeys`) with no `windows-latest` evidence
source until WIN-1.8, and `release.yml`'s aggregate gate requires its `needs` to
equal `workflow.required_job_ids` exactly. Adding the Windows gates/jobs before a
real Windows evidence source exists would either break every release (including
Linux-only, violating item 4) or fabricate machinery with no subject — the same
reason Phase 0 declined to mutate the registry. When WIN-1.8 lands the
`windows-latest` job and the conditional evidence plumbing, flipping the
`WIN-01` guard from passing to failing is the signal to add the gates and move
the blocker toward resolved.

### Phase 0 verification (what is checked now)

Because Phase 0 does not touch the registry, the check is that the pinned
contract still holds after the Phase 1 and Phase 0 changes: recomputing
`remediationContractDigest()` over the current `remediation-blockers.json` must
still equal `REVIEWED_REMEDIATION_CONTRACT_SHA256`. This is exercised in the
Phase 0 completion notes and should be part of any future `verify-release.js`
dry-run.
