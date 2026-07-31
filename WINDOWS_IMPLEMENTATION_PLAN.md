# Windows desktop application — implementation plan and task decomposition

Status: proposed · Owner: TBD · Companion to `WINDOWS_APP_FEASIBILITY.md`

This document turns the feasibility assessment into an actionable, trackable
work breakdown. The feasibility doc answers *whether* and *what shape*; this doc
answers *which tasks, in what order, touching which files, gated by what*.

It is grounded in a code review of the current tree (branch `master`). Every
task references real files. File/line anchors were accurate at plan authoring
time; re-verify before starting a task.

---

## How to read this plan

**Task ID:** `WIN-<phase>.<n>` (e.g. `WIN-1.2`). Stable; do not renumber.

**Size legend** (relative engineering scope, not schedule):
`S` small · `M` medium · `L` large · `XL` extra-large.

**Status legend:** `TODO` · `IN-PROGRESS` · `BLOCKED` · `DONE`.

**Each task lists:** goal, files touched, acceptance criteria, dependencies.

A task is **Done** only when its acceptance criteria pass *and* the cross-cutting
rules below are satisfied (release-gate manifests updated, no security
regression, tests green on the relevant CI OS).

---

## Cross-cutting rules (apply to every task)

These are not a phase; they are invariants that gate every merge.

- **CC-1 — No silent security regression.** Any code path that enforces
  ownership, private mode, or no-follow/no-reparse on POSIX must, on an
  unimplemented platform, **fail closed** (raise), never silently pass. See the
  review finding under WIN-1.4. This rule overrides convenience.
- **CC-2 — Release-gate manifest discipline.** `scripts/release/verify-release.js`
  pins an *exact* source-file manifest (starts at line 28). Any new module
  (e.g. `_platform/*.py`) or renamed/removed file MUST update, in the same
  commit: `verify-release.js`, `MANIFEST.in`, `pyproject.toml` packaging, and the
  npm `verify-release` path. A new file that is not in the manifest fails the
  release gate. See memory note "custback release gates".
- **CC-3 — REL-01 stays intact.** Production release is deliberately blocked by
  open `REL-01` (`REMEDIATION_PLAN.md`, `scripts/release/remediation-blockers.json`).
  Windows evidence is *added to* that exact-commit gate (WIN-0.2 / WIN-5.8).
  Never create a bypass around it.
- **CC-4 — One platform seam, not sprinkled conditionals.** All OS-specific
  filesystem/security/path behavior lives behind the `custback._platform`
  package (WIN-1.1 / WIN-1.5). Storage, security, model, and avatar code call the
  seam; they do not contain `sys.platform` branches.
- **CC-5 — Licensing is a distribution gate.** No bundling of GPL model/weights
  or GPL virtual-camera components into a distributed artifact until WIN-0.4
  clears it. Code and CI may use them; the *installer* may not ship them
  unresolved.

---

## Task index

| ID | Title | Phase | Size | Deps | Status |
| --- | --- | --- | --- | --- | --- |
| WIN-0.1 | Support-matrix decision record | 0 | S | — | DONE (decisions proposed, awaiting sign-off) |
| WIN-0.2 | Extend REL-01 gate for Windows evidence | 0 | S | WIN-0.1 | DONE (slot specified; mutation deferred to WIN-5.8) |
| WIN-0.3 | Windows feature matrix doc | 0 | S | WIN-0.1 | DONE |
| WIN-0.4 | Licensing / compliance review | 0 | M | — | DONE (review underway; bundle decisions open) |
| WIN-1.1 | `custback._platform` fs seam (POSIX impl + fail-closed Win stubs) | 1 | M | CC-1 | DONE |
| WIN-1.2 | Remove unconditional `fcntl` import crash | 1 | S | WIN-1.1 | DONE |
| WIN-1.3 | Route all `os.fchmod` through the seam | 1 | M | WIN-1.1 | DONE |
| WIN-1.4 | Make silent security no-ops fail-closed | 1 | M | WIN-1.1 | DONE |
| WIN-1.5 | `custback._platform.paths` Known-Folders routing | 1 | M | WIN-1.1 | TODO |
| WIN-1.6 | Windows dependency profile | 1 | M | — | TODO |
| WIN-1.7 | Update packaging manifests for new modules | 1 | S | WIN-1.1 | DONE |
| WIN-1.8 | Windows CI job (import + synthetic + CPU RVM) | 1 | M | WIN-1.2..1.6 | TODO |
| WIN-2.1 | Win32 file locking (bounded, exclusive) | 2 | L | WIN-1.1 | IMPL* |
| WIN-2.2 | NTFS atomic no-replace create/rename | 2 | L | WIN-1.1 | IMPL* |
| WIN-2.3 | Private DACL create (replace chmod 0600/0700) | 2 | L | WIN-1.1 | IMPL* |
| WIN-2.4 | Reparse-point/symlink rejection on open | 2 | M | WIN-1.1 | IMPL* |
| WIN-2.5 | Win32 file identity (replace st_ino/st_dev) | 2 | M | WIN-1.1 | IMPL* |
| WIN-2.6 | SID-based ownership verification | 2 | M | WIN-2.3 | IMPL* |
| WIN-2.7 | Directory durability equivalent | 2 | S | WIN-1.1 | IMPL* |
| WIN-2.8 | NTFS security/crash-recovery test suite | 2 | L | WIN-2.1..2.7 | IMPL* |

`IMPL*` = implemented behind the `custback._platform` seam with the pywin32
Win32/NTFS backend; POSIX is byte-for-byte and the full suite is green on Linux;
the NTFS-specific execution/adversarial assertions run under the still-TODO
`windows-latest` CI job (WIN-1.8), which is where these flip to `DONE`.
| WIN-3.1 | Friendly camera enumeration + stable IDs | 3 | M | WIN-1.8 | IMPL* |
| WIN-3.2 | Explicit MSMF/DSHOW backend selection | 3 | M | WIN-3.1 | IMPL* |
| WIN-3.3 | Camera privacy-denial handling | 3 | S | WIN-3.2 | IMPL* |
| WIN-3.4 | Output-camera loop prevention | 3 | S | WIN-3.1 | IMPL* |
| WIN-3.5 | Resolution/FPS negotiation verification | 3 | S | WIN-3.2 | IMPL* |
| WIN-3.6 | Device lifecycle (replug/suspend/contention) | 3 | M | WIN-3.2 | IMPL* (existing recovery covers replug/suspend/contention; reorder mitigated by stable IDs at selection) |
| WIN-3.7 | OBS Virtual Camera detect + setup | 3 | M | WIN-1.8 | IMPL* |
| WIN-4.1 | `acceleration` config model + migration | 4 | M | — | DONE |
| WIN-4.2 | Real-RVM warm-up probe | 4 | M | WIN-4.1 | IMPL* |
| WIN-4.3 | GPU state machine + non-fatal probe + DLL preload | 4 | L | WIN-4.2 | IMPL* |
| WIN-4.4 | Inference-time recovery around `session.run` | 4 | M | WIN-4.3 | IMPL* |
| WIN-4.5 | Truthful acceleration status/doctor/UI | 4 | M | WIN-4.3 | IMPL* |
| WIN-4.6 | CPU-force and gpu_required modes | 4 | S | WIN-4.1 | IMPL* |
| WIN-4.7 | DirectML / Windows ML go/no-go spike | 4 | M | WIN-4.2 | DONE (planning gate) |

`IMPL*` here = implemented behind the acceleration seam
(`custback/acceleration.py`); the policy, latched state machine, real-RVM
proof, provider selection, inference-time recovery, and truthful status are
green on Linux with a fake ORT. The provider-*execution* assertions (a real
CUDA/DirectML node proven via the ORT profile, and the injected mid-run GPU/OOM
failure) are the Phase-4/5 hardware gates in the validation matrix, which is
where these flip to `DONE`.
| WIN-5.1 | PyInstaller onedir spec + hooks | 5 | L | WIN-2.8, WIN-3.7, WIN-4.4 | IMPL* |
| WIN-5.2 | C# WebView2/tray shell | 5 | L | WIN-5.1 | IMPL* |
| WIN-5.3 | Secure WebView session bootstrap | 5 | M | WIN-5.2 | IMPL* (engine contract DONE on Linux) |
| WIN-5.4 | Tray lifecycle / suspend / crash handling | 5 | M | WIN-5.2 | IMPL* |
| WIN-5.5 | Supervise avatar second process | 5 | S | WIN-5.2 | IMPL* |
| WIN-5.6 | Signed per-user EXE/MSI installer | 5 | L | WIN-5.1 | IMPL* |
| WIN-5.7 | Uninstall data-retention policy | 5 | S | WIN-5.6 | IMPL* |
| WIN-5.8 | Windows release-evidence pipeline | 5 | M | WIN-5.6, WIN-0.2 | DEFERRED (release blocker temporarily removed; job wiring pends WIN-1.8) |
| WIN-6.1 | Native Win11 Media Foundation virtual camera | 6 | XL | WIN-5.8 | IMPL* |
| WIN-6.2 | Generic GPU provider (DirectML/Windows ML) | 6 | XL | WIN-4.7 | IMPL* (gate machinery; AMD/Intel hardware evidence pending) |
| WIN-6.3 | ARM64 support | 6 | L | WIN-5.8 | IMPL* (build/dependency machinery; ARM64 hardware evidence pending) |
| WIN-6.4 | Avatar + Audio2Face Windows parity | 6 | XL | WIN-5.8 | IMPL* |

**Critical path:** WIN-1.1 → WIN-1.4 → (WIN-2.1..2.7) → WIN-2.8 → WIN-5.1 →
WIN-5.6 → WIN-5.8. Phase 2 is the long pole; Phases 3 and 4 parallelize against
it once Phase 1 lands.

---

## Phase 0 — Baseline and decisions (size S–M)

**Goal:** lock the product scope so later phases have a fixed target; extend the
release gate to *admit* Windows evidence without weakening it.

**Exit gate:** existing exact-commit release gates remain intact; a written
Windows feature matrix and support decision exist; licensing review is underway.

### WIN-0.1 — Support-matrix decision record · S
- **Goal:** commit the answers to the feasibility doc's 8 open decisions
  (min OS = Win11 x64; GPU = NVIDIA CUDA first; vcam = OBS prereq first;
  UX = WebView2/tray; distribution = signed per-user EXE/MSI; offline =
  first-run download unless licensing clears bundling; parity = core camera
  first; preview = WebView only).
- **Files:** `WINDOWS_DECISIONS.md` (Part A), at repo root — not `docs/`,
  because arbitrary root `*.md` are not shipped to npm whereas `docs/*.md` are
  exact-matched in the reviewed npm payload (`verify-release.js`). Matches the
  existing `WINDOWS_APP_FEASIBILITY.md` convention.
- **Accept:** each decision has a chosen default + rationale + revisit trigger.
  **Status: done** — D1–D8 recorded as *proposed*; product ratification is the
  remaining step (flip each to *Ratified* in `WINDOWS_DECISIONS.md`).

### WIN-0.2 — Extend REL-01 gate for Windows evidence · S
- **Goal:** define *where* Windows artifacts/evidence attach to the existing
  exact-commit gate, without a bypass (CC-3).
- **Files:** spec in `WINDOWS_DECISIONS.md` (Part B). The actual mutation of
  `scripts/release/remediation-blockers.json`,
  `scripts/release/required-gates.json`, `scripts/release/verify-release.js`,
  and `scripts/release/assemble-evidence.js` is **deferred to WIN-5.8**.
- **Why deferred:** the registry is hash-pinned
  (`REVIEWED_REMEDIATION_CONTRACT_SHA256`) and an *open Node blocker* needs a
  `guard` proving a **current** failure of the Windows publish machinery — which
  does not exist until Phase 5. Adding it now would fabricate a test with no
  subject or leave the gate broken. Part B specifies the exact `WIN-01` blocker
  entry, regression-test shape, `required-gates.json` additions, and the
  manifest-scoped rule so a Linux-only release is not blocked by absent Windows
  evidence and a Windows release cannot publish without it.
- **Accept:** **done** — slot fully specified; the existing contract is proven
  intact (recomputed `remediationContractDigest` still equals the pinned hash;
  only `REL-01` open; `release_blocked` true).

### WIN-0.3 — Windows feature matrix doc · S
- **Goal:** enumerate features × {Win11 x64, ARM64, CUDA, DirectML, CPU} with
  supported/planned/not-planned, so no claim is implied by inheritance.
- **Files:** `WINDOWS_FEATURE_MATRIX.md` at repo root.
- **Accept:** matrix reviewed; each "supported" cell maps to a phase exit gate.
  **Status: done** — every cell is Supported/Planned(WIN-x)/Not-planned/N.A. with
  an advertising guardrail.

### WIN-0.4 — Licensing / compliance review · M
- **Goal:** resolve CC-5. Audit: RVM model/weights (GPL-3), `pyvirtualcam`
  (GPL-2), OBS components, OpenCV/FFmpeg, CUDA/cuDNN redistribution, MediaPipe
  model, WebView2 runtime, installer toolchain.
- **Files:** `WINDOWS_LICENSING.md` at repo root; `LICENSE`/notices in WIN-5.6
  if bundling.
- **Accept:** a written go/no-go per component for (a) using in CI, (b) bundling
  in the distributed installer. **Status: done as a planning gate** — component
  CI-use/Bundle table plus a "decisions required before Phase 5 bundling"
  checklist; the HOLD items need legal sign-off before WIN-5.1 bundles them.

---

## Phase 1 — Portable core (size M)

**Goal:** the Python process imports and runs on native Windows with CPU RVM;
security-sensitive operations fail closed rather than silently passing.

**Exit gate:** on `windows-latest`, `import custback` succeeds,
`python -m custback --synthetic --no-vcam --no-api` runs a synthetic pipeline,
CPU RVM inference runs against the real pinned model, and every ownership /
private-mode / no-follow operation either works correctly or raises a clear
"platform unsupported" error (never a silent pass). Flags already exist:
`--synthetic`, `--no-api`, `--no-vcam` in `src/custback/__main__.py`.

### WIN-1.1 — `custback._platform` filesystem seam · M
- **Goal:** introduce a single abstraction with a POSIX backend (current
  behavior, byte-for-byte) and a Windows backend that is fail-closed stubs until
  Phase 2. Surface: `lock_exclusive(fd, blocking, timeout)`,
  `unlock(fd)`, `rename_noreplace(src, dst)`, `open_private(path, mode)`,
  `set_private_mode(fd, mode)`, `open_nofollow(path, flags)`,
  `fsync_dir(path)`, `verify_owner(stat_or_handle)`, `is_reparse(path)`.
- **Files:** new `src/custback/_platform/__init__.py`,
  `src/custback/_platform/posix.py`, `src/custback/_platform/windows.py`,
  `src/custback/_platform/base.py`. Define `PlatformSecurityUnsupported` error.
- **Accept:** POSIX behavior unchanged (existing tests green on Linux/macOS);
  Windows backend importable and every security method raises
  `PlatformSecurityUnsupported` (satisfies CC-1). Unit tests for the seam.
- **Deps:** CC-1, CC-2 (manifest), CC-4.

### WIN-1.2 — Remove the unconditional `fcntl` import crash · S
- **Goal:** the P0 import failure. `src/custback/segmentation.py:23`
  imports `fcntl` at module top; `src/custback/storage_tx.py` likewise.
  Route the model-cache lock (`segmentation.py:~121-142`) and the transaction
  lock (`storage_tx.py:~290`) through WIN-1.1.
- **Files:** `src/custback/segmentation.py`, `src/custback/storage_tx.py`.
- **Accept:** `import custback.segmentation` and `import custback.storage_tx`
  succeed on Windows; Linux locking semantics unchanged.
- **Deps:** WIN-1.1.

### WIN-1.3 — Route all `os.fchmod` through the seam · M
- **Goal:** `os.fchmod` does not exist on Windows → `AttributeError`. Nine call
  sites across eight files must go through `set_private_mode`.
- **Files (all confirmed):** `src/custback/api/security.py:170`,
  `src/custback/api/server.py:588,694,720`,
  `src/custback/avatar/__main__.py:172`,
  `src/custback/avatar/store.py:183,235`,
  `src/custback/diagnostics.py:217,233`,
  `src/custback/migration.py:324,895`, `src/custback/segmentation.py:126`,
  `src/custback/storage_tx.py:113,267,288`.
- **Accept:** no direct `os.fchmod` remains in `src/custback` (grep clean);
  POSIX mode bits unchanged; Windows path defers to DACL work (WIN-2.3) or
  fail-closed stub.
- **Deps:** WIN-1.1.

### WIN-1.4 — Make silent security no-ops fail-closed · M  *(review finding — do first)*
- **Goal:** today the ownership/no-follow guards are wired to *pass trivially*
  on Windows, which is worse than crashing. Fix so they fail closed.
  - `src/custback/api/server.py:556` — `effective_uid = getattr(os, "geteuid",
    lambda: before.st_uid)()` makes the `st_uid != effective_uid` check always
    false on Windows (`st_uid` is `0` there too). Same shape at `server.py:711`.
  - `src/custback/avatar/store.py:123` — `getuid is None or ...` returns `True`.
  - `getattr(os, "O_NOFOLLOW", 0)` degrades to `0` everywhere, silently dropping
    symlink/reparse protection (`security.py`, `server.py`, `store.py`,
    `storage_tx.py`, `diagnostics.py`, `avatar_proxy.py`).
- **Approach:** ownership and no-follow enforcement route through
  `verify_owner` / `open_nofollow` (WIN-1.1). On the Windows stub they raise
  until WIN-2.4/WIN-2.6 land. No security-relevant guard may evaluate to a
  silent pass on an unimplemented platform.
- **Files:** the six files above + the seam.
- **Accept:** a test asserts that, with the Windows backend forced, each guard
  raises rather than returning success; Linux behavior unchanged.
- **Deps:** WIN-1.1. **Satisfies CC-1.**

### WIN-1.5 — `custback._platform.paths` Known-Folders routing · M
- **Goal:** replace hard-coded XDG/POSIX defaults with a path module. On
  Windows use `%LOCALAPPDATA%\Custback` (cache/state/models/logs) and
  `%APPDATA%\Custback` (config/tokens); keep XDG on POSIX. Define migration
  behavior for existing users before changing paths.
- **Files (confirmed sites):** `src/custback/config.py:376,377,445`,
  `src/custback/api/security.py:26,28`,
  `src/custback/segmentation.py:91` (`DEFAULT_MODEL_DIR`),
  `src/custback/backgrounds.py:49`, `src/custback/diagnostics.py:267,280-287`,
  `src/custback/avatar/config.py:320,321,386,84`,
  and YAML defaults `config/default.yaml`, `config/avatar.yaml`,
  `src/custback/avatar/avatar.yaml:9,75,76,99`.
- **Accept:** on Windows, defaults resolve under Known Folders; on POSIX, byte-
  identical to today; migration path documented and tested.
- **Deps:** WIN-1.1.

### WIN-1.6 — Windows dependency profile · M
- **Goal:** a resolvable Windows dependency set. Evaluate
  `opencv-contrib-python-headless` (WebView/MJPEG replaces HighGUI —
  `src/custback/preview.py:293` uses `cv2.namedWindow`), confirm `pyvirtualcam`
  Windows wheel, pin ORT profile (CPU or CUDA — never both, see
  `install.js:98`).
- **Files:** `pyproject.toml` (extras/markers), packaging notes.
- **Accept:** `pip install` of the Windows profile succeeds in a clean Win venv;
  `import cv2, pyvirtualcam, onnxruntime, custback` succeeds.

### WIN-1.7 — Update packaging manifests for new modules · S
- **Goal:** satisfy CC-2 for the files added in WIN-1.1/1.5.
- **Files:** `scripts/release/verify-release.js` (manifest, ~line 28+),
  `MANIFEST.in`, `pyproject.toml` package data, npm verify path.
- **Accept:** `node scripts/release/verify-release.js` passes with the new files
  present and fails if one is removed.

### WIN-1.8 — Windows CI job · M
- **Goal:** first automated Windows evidence. Current matrix is
  `[ubuntu-latest, macos-latest]` (`.github/workflows/ci.yml:214`).
- **Files:** `.github/workflows/ci.yml`.
- **Accept:** a `windows-latest` job runs: unit tests, `import custback`,
  `python -m custback --synthetic --no-vcam --no-api`, and CPU RVM inference on
  the real pinned model. Green.
- **Deps:** WIN-1.2..1.6.

---

## Phase 2 — Windows security and storage (size L/XL) — *the long pole*

**Goal:** replace every fail-closed Windows stub from Phase 1 with a real
NTFS/Win32 implementation that preserves the current security and crash-recovery
contract. This is the highest-risk phase; treat NTFS primitives as first-class
with adversarial tests.

**Exit gate:** synthetic run with the full local API works on Windows; upload,
restart recovery, concurrency, and token-privacy tests pass on NTFS; no security
guard is a silent pass.

**Implementation notes (landed behind the seam):**
- **Bindings:** `pywin32` (`win32file`/`win32security`/`win32api`), declared as
  the `custback[windows]` optional extra (WIN-1.6 owns finalizing the profile;
  the frozen installer bundles it per D5). `windows.py` is imported only under
  `sys.platform == "win32"`, so POSIX hosts never load pywin32.
- **Seam surface widened** past the Phase-1 five to: `open_nofollow`,
  `fsync_dir`, `rename_noreplace`, `hardlink`, `owner_matches`,
  `stat_owner_matches`, `is_private_to_owner`, `file_identity`, `is_reparse`,
  `chmod_private`, `listdir_secure` (plus the original lock/`set_private_mode`).
  Call sites no longer compose POSIX flags inline (CC-4); `nofollow_flag()` and
  `effective_uid()` were removed so no site can silently degrade.
- **Ownership (WIN-2.6):** `owner_matches(fd)` is the *authoritative* post-open
  SID check and now runs in every flow that opens a security-sensitive inode
  (several gained an explicit post-open check they lacked). `stat_owner_matches`
  is the *advisory* pre-open pre-filter — real `st_uid` on POSIX, `True` on
  Windows where a `stat_result` carries no owner. Token privacy uses
  `is_private_to_owner`, which enumerates the DACL on Windows.
- **WIN-2.5:** CPython already sources `st_dev`/`st_ino` from the volume serial +
  128-bit `FILE_ID_INFO` on Windows (3.12+), so the ledger's identity checks are
  correct as-is; `file_identity` makes this explicit and testable.
- **WIN-2.7:** `FlushFileBuffers` on a read-only directory handle falls back to
  the NTFS-journal-ordered no-op (documented), matching the plan's alternative.
- **Validation:** POSIX byte-for-byte (full suite green on Linux);
  `tests/test_platform_seam.py` asserts the security contract against the seam so
  it runs on both backends; two junction/DACL tests are Windows-only (`skipif`).

### WIN-2.1 — Win32 file locking · L
- Bounded-wait, exclusive-writer locking via `LockFileEx`/named mutex behind
  `lock_exclusive`. Preserve the model-cache and transaction-ledger contract
  (bounded waiting, exclusive writers). Replaces `fcntl.flock`
  (`segmentation.py`, `storage_tx.py:290`).
- **Accept:** contention/timeout tests match POSIX semantics; interrupted holder
  releases correctly.

### WIN-2.2 — NTFS atomic no-replace create/rename · L
- Implement `rename_noreplace` on Windows (`CreateFile CREATE_NEW`, or
  `SetFileInformationByHandle` with `FileRenameInfoEx` and
  replace-if-exists=FALSE, or `MoveFileEx` without `REPLACE_EXISTING`). Today
  `src/custback/storage_tx.py:57` only supports Linux `renameat2` / macOS
  `renamex_np` and otherwise raises `ENOTSUP`.
- **Accept:** publishing over an existing unmarked destination fails atomically;
  used by background uploads and both avatar stores.

### WIN-2.3 — Private DACL creation · L
- Replace `chmod 0600/0700` semantics (WIN-1.3) with explicit owner-only DACLs
  for the current user SID, no inherited ACEs, on token/log/store files and
  directories.
- **Files:** seam + `api/security.py`, `api/server.py`, `avatar/store.py`,
  `diagnostics.py`, `migration.py`, `storage_tx.py`.
- **Accept:** created files/dirs grant access only to the current SID; inherited
  ACEs are stripped; adversarial test confirms another user is denied.

### WIN-2.4 — Reparse-point / symlink rejection · M
- Replace `O_NOFOLLOW` semantics: reject junctions/symlinks/reparse points on
  open (`FILE_FLAG_OPEN_REPARSE_POINT` inspection or explicit reparse check).
  Unblocks the guards WIN-1.4 stubbed.
- **Accept:** opening through a junction to a sensitive target is refused.

### WIN-2.5 — Win32 file identity · M
- Replace `st_ino`/`st_dev` inode identity used by the transaction ledger
  (`storage_tx.py` `_inode`) with volume serial + file index
  (`GetFileInformationByHandle` / `FILE_ID_INFO`).
- **Accept:** identity is stable across the transaction; crash-recovery matching
  works on NTFS.

### WIN-2.6 — SID-based ownership verification · M
- Replace `st_uid`/`geteuid` checks with owner-SID comparison behind
  `verify_owner`. Unblocks WIN-1.4 ownership guards on Windows.
- **Accept:** a file owned by another SID fails verification; current-user files
  pass. **Deps:** WIN-2.3.

### WIN-2.7 — Directory durability equivalent · S
- Provide a Windows `fsync_dir` equivalent (directory-handle
  `FlushFileBuffers` where supported, or documented ordering guarantee).
  Current `sync_directory` opens a dir fd + `os.fsync`, which fails on Windows.
- **Accept:** namespace changes are durably ordered or the limitation is
  explicitly documented and tested.

### WIN-2.8 — NTFS security / crash-recovery test suite · L
- **Accept:** tests pass on NTFS for: private-DACL enforcement, reparse-point
  rejection, no-replace publication, crash/restart recovery, concurrent upload,
  quota, and token privacy. These become the WIN-1.8 job's Phase-2 additions.
- **Deps:** WIN-2.1..2.7.

---

## Phase 3 — Camera and output MVP (size L)

**Goal:** a real Windows camera reaches a meeting app through Custback and OBS
Virtual Camera. Parallelizable with Phase 2 once Phase 1 lands.

**Exit gate:** physical camera → Custback → OBS Virtual Camera is visible and
stable in Teams, Zoom, Chrome/Meet, and a browser test page.

**Implementation notes (landed behind a camera seam):**
- **Camera seam:** all OS-specific *camera* behavior (enumeration, backend
  order, output-loop detection, privacy/OBS guidance) lives in the new
  `src/custback/camera_devices.py`, so `capture.py`/`vcam.py` contain no camera
  `sys.platform` branch — the CC-4 discipline applied to capture. Added to the
  release manifest (`verify-release.js`) per CC-2.
- **Enumeration (WIN-3.1/3.4):** Linux reads sysfs friendly names + `/dev/v4l/by-id`
  stable IDs deterministically; other platforms probe OpenCV indices with an
  optional Media Foundation/DirectShow name provider. Virtual *output* cameras
  (OBS/Custback/v4l2loopback) are filtered from inputs by name. Surfaced through
  `custback --list-cameras`.
- **Backend selection (WIN-3.2):** `capture.py` opens with an explicit
  `apiPreference` in Windows MSMF→DSHOW order, falling through on open failure.
  Off Windows the candidate list is empty, so the open call is the historical
  single-arg `cv2.VideoCapture(device)` — **POSIX capture is byte-for-byte
  unchanged**.
- **Privacy denial (WIN-3.3):** a total open failure on Windows appends a
  credential-free hint naming Settings > Privacy & security > Camera.
- **Negotiation (WIN-3.5):** the existing post-first-frame verification is
  backend-agnostic and now reports the explicit MSMF/DSHOW backend.
- **Lifecycle (WIN-3.6):** the existing bounded reader/recovery controller
  already covers unplug/replug, suspend/resume, and meeting-app contention;
  stable IDs address reorder at *selection* time (int-index reopen keeps its
  documented reorder caveat).
- **OBS output (WIN-3.7):** `vcam.py` gives platform-specific OBS setup guidance,
  distinguishes the single-instance "in use" case from "not installed" in the
  fallback reason, and never redistributes OBS components (CC-5).
- **Validation:** POSIX byte-for-byte (full suite green on Linux);
  `tests/test_camera_devices.py` and `tests/test_vcam.py` assert the contract on
  both backends, and `tests/test_capture.py` forces the Windows platform to
  exercise MSMF→DSHOW selection and the privacy hint. The MSMF/DSHOW *execution*
  and real Media Foundation enumeration/OBS assertions run under the still-TODO
  `windows-latest` CI job (WIN-1.8), which is where these flip to `DONE`.

### WIN-3.1 — Friendly enumeration + stable IDs · M
- Enumerate device friendly names and stable identifiers (Media Foundation
  enumeration) instead of bare index `0`. Extend `src/custback/capture.py`
  (currently generic `cv2.VideoCapture` with V4L2-conditional negotiation).

### WIN-3.2 — Explicit MSMF/DSHOW selection · M
- Select and test `cv2.CAP_MSMF` vs `cv2.CAP_DSHOW` explicitly, analogous to the
  existing V4L2 negotiation (`capture.py:~233-256`). **Deps:** WIN-3.1.

### WIN-3.3 — Camera privacy-denial handling · S
- Detect Windows camera privacy denial and surface an actionable message.
  **Deps:** WIN-3.2.

### WIN-3.4 — Output-camera loop prevention · S
- Do not offer the Custback/OBS output camera as an input. **Deps:** WIN-3.1.

### WIN-3.5 — Resolution/FPS negotiation verification · S
- Verify negotiated resolution/FPS reporting on MSMF/DSHOW. **Deps:** WIN-3.2.

### WIN-3.6 — Device lifecycle · M
- Handle index reorder, unplug/replug, suspend/resume, and meeting-app
  contention; extend the existing bounded reader + recovery controller in
  `capture.py`. **Deps:** WIN-3.2.

### WIN-3.7 — OBS Virtual Camera detect + setup · M
- Detect OBS Virtual Camera, provide consent-based first-run setup, health
  reporting, single-instance-limit handling. Update `src/custback/vcam.py:89`
  help text (currently Linux/macOS only). Do not silently redistribute OBS
  components (CC-5).

---

## Phase 4 — Acceleration policy (size L)

**Goal:** GPU-first, CPU-fallback with truthful status. Parallelizable with
Phases 2–3.

**Exit gate:** clean CPU-only machine starts and reports CPU; NVIDIA machine
proves RVM GPU nodes (not mere provider registration); injected startup *and*
mid-run GPU failures retry once on CPU and latch, without pipeline exit.

**Implementation notes (landed behind the acceleration seam):**
- **Acceleration seam:** all provider-selection, proof, DLL-preload, and latched
  lifecycle logic lives in the new `src/custback/acceleration.py`, so
  `segmentation.py` no longer composes ORT provider lists inline and no code
  path branches on `sys.platform` for acceleration (CC-4). Added to the release
  manifest (`verify-release.js`) per CC-2. It imports nothing GPU-specific at
  load; the segmenter passes in the already-imported `onnxruntime`.
- **Config (WIN-4.1):** `AccelerationConfig {mode: auto|cpu|gpu_required,
  provider: auto|cuda|directml, device_id}` is a strict Pydantic model, separate
  from `segmentation.delegate` (MediaPipe-only). It is purely additive with safe
  defaults, so `extra="forbid"` still loads a pre-Phase-4 config (no legacy
  `camera_device`-style rewrite is needed; a backward-compat test asserts a
  section-less config gains the default policy and an explicit one round-trips).
  Hot-reconfig re-stages the segmenter when `acceleration` *or* `segmentation`
  changes (`_segmenter_key` in `pipeline.py`).
- **Real-RVM proof (WIN-4.2):** `prove_rvm_provider` runs one synthetic frame
  through the *actual* RVM graph in a short-lived profiling session and confirms
  a node executed on the candidate provider — registration-without-execution and
  init fallback both fail the proof. `gpu_probe.py`'s tiny `Add` graph is kept
  for diagnostics only.
- **State machine (WIN-4.3):** `AccelerationState` latches
  `STARTING → GPU_PROBING → GPU_ACTIVE | CPU_FALLBACK`; a proven provider is
  warmed up before the first real frame; `preload_acceleration_dlls` adds CUDA
  DLL directories on Windows before the first session (non-fatal). CPU_FALLBACK
  is the single on-CPU terminal state; `fallback_active` distinguishes an
  intended CPU landing (`mode: cpu`, or `auto` with no GPU registered) from a
  degraded one, and `fallback_count` counts each distinct degradation once.
- **Inference recovery (WIN-4.4):** `RVMSegmenter.segment` wraps `session.run`;
  a GPU/DLL/OOM failure while `on_gpu` rebuilds a CPU-only session, clears the
  recurrent state and `last_foreground`, retries the current frame once, and
  stays latched on CPU. A CPU-side failure is re-raised, never retried (it would
  loop). GPU is never retried per frame.
- **Status (WIN-4.5):** the latched status (requested mode/provider, *actual*
  post-fallback active provider, state, fallback flag/reason/count, transition
  age) is read each frame into `hub` stats and surfaced in `/status`, the web-UI
  System-status panel, and the config-change audit whitelist (`diagnostics.py`).
  The active provider reflects reality, never the requested order.
- **Modes (WIN-4.6):** `mode: cpu` builds only `CPUExecutionProvider`;
  `mode: gpu_required` raises `GpuRequiredError` at startup unless real GPU
  execution is proven, and that error is never swallowed by `backend: auto`
  segmenter fallback.
- **DirectML (WIN-4.7):** the `DmlExecutionProvider` path is wired through
  provider selection and proof; `WINDOWS_ACCELERATION_SPIKE.md` records the
  benchmark method and go/no-go criteria that gate the WIN-6.2 "all Windows GPUs"
  claim. Bundling/benchmark on hardware remains open.
- **Validation:** POSIX byte-for-byte (RVM CPU path unchanged; full suite green
  on Linux). `tests/test_acceleration.py` and the acceleration cases in
  `tests/test_segmentation_rvm.py` assert the contract with a fake ORT that
  writes an ORT-shaped profile; the real CUDA/DirectML execution and injected
  mid-run failure run under the Phase-4/5 hardware gates.

### WIN-4.1 — `acceleration` config model + migration · M
- Add `acceleration: {mode: auto|cpu|gpu_required, provider: auto|cuda|directml,
  device_id}` as a strict Pydantic model. Note: config is `extra="forbid"`
  (`config.py:138`), so this needs a schema addition, a config-version bump,
  a migration (`src/custback/migration.py`), and hot-reconfig validation — not a
  YAML-only change. Keep it separate from the existing `segmentation.delegate`
  (`config/default.yaml:39`, MediaPipe-only).
- **Files:** `src/custback/config.py`, `config/default.yaml`,
  `src/custback/migration.py`, migration fixtures under `tests/fixtures/migration`.

### WIN-4.2 — Real-RVM warm-up probe · M
- Warm up the *actual* RVM graph on a synthetic frame. The existing CUDA probe
  (`src/custback/gpu_probe.py`) proves only a tiny `Add` graph — keep it for
  diagnostics but it is not product evidence. **Deps:** WIN-4.1.

### WIN-4.3 — GPU state machine + non-fatal probe + DLL preload · L
- Implement `STARTING → GPU_PROBING → GPU_ACTIVE | CPU_FALLBACK` (latched).
  Load the verified model before opening camera/output. Probe the preferred
  provider non-fatally; on Windows/CUDA preload required DLLs for the pinned ORT
  version. Provider order today is CUDA/CoreML/CPU (`segmentation.py:538`).
  **Deps:** WIN-4.2.

### WIN-4.4 — Inference-time recovery around `session.run` · M
- `RVMSegmenter.segment()` has no recovery around `self._session.run(...)`
  (`src/custback/segmentation.py:572`). On a GPU/DLL/OOM failure: build a
  CPU-only session, clear recurrent state and `last_foreground`, retry the
  current frame once, stay on CPU. Do not retry GPU per frame. **Deps:** WIN-4.3.

### WIN-4.5 — Truthful acceleration status/doctor/UI · M
- Publish requested policy, active provider, fallback state/reason/count, and
  transition time through `/status`, doctor, logs, and the web UI. Device status
  must reflect *actual* post-fallback provider, not inferred provider order.
  Reasons bounded and credential/path-free. **Files:** `pipeline.py`, `hub.py`,
  `diagnostics.py`, `api/webui.py`. **Deps:** WIN-4.3.

### WIN-4.6 — CPU-force and gpu_required modes · S
- `mode: cpu` constructs only `CPUExecutionProvider` (deterministic, testable);
  `mode: gpu_required` fails startup if accelerator execution cannot be proved.
  **Deps:** WIN-4.1.

### WIN-4.7 — DirectML / Windows ML go/no-go spike · M
- Benchmark RVM on `DmlExecutionProvider`; prototype Windows ML packaging.
  Output: a go/no-go decision that gates any "all Windows GPUs" claim (feeds
  WIN-6.2). **Deps:** WIN-4.2.

---

## Phase 5 — Desktop product (size L)

**Goal:** a signed, self-contained installer produces a clean-machine desktop
app. Requires Phases 2–4 complete (the UI and uploads depend on the Phase-2 API
stores).

**Exit gate:** a clean VM with no Python/Node/build tools installs, runs,
upgrades, and uninstalls; no secret appears in process arguments or URLs.

**Implementation notes (landed as reviewable source + an engine contract):**
- **Packaging tree:** the Windows-built artifacts live under `packaging/windows/`
  (`pyinstaller/`, `shell/`, `installer/`). That path is matched by no npm
  `files` glob, so the exact npm-payload gate (CC-2) is unaffected; only the two
  files that *are* payload-matched — `tests/test_windows_packaging.py` and
  `packaging/npm/test/windows-release-check.test.js` — were added to the
  reviewed manifests in `verify-release.js`.
- **Engine side is real and Linux-verified.** WIN-5.3's contract that the shell
  depends on — the `/auth/session` bearer→cookie bootstrap and the new
  bearer-only `POST /lifecycle/shutdown` private lifecycle channel
  (`api/server.py`, wired from `__main__.py`) — is implemented and asserted on
  Linux (`tests/test_api.py`): a WebView session (HttpOnly cookie only) is
  rejected (403), the bearer succeeds (202), and an unwired process reports 503.
- **Secret hygiene (WIN-5.3).** The bearer never appears in a URL, command line,
  web storage, or log: the engine writes it to a private-DACL token file
  (WIN-2.3); the shell passes only the *path*, exchanges the token in a POST
  body over loopback, and injects only the opaque session cookie into WebView2.
- **`IMPL*` semantics here:** the PyInstaller spec/hooks (WIN-5.1), the C#
  shell (WIN-5.2/5.4/5.5), and the WiX installer/retention (WIN-5.6/5.7) are
  complete, reviewable source whose *byte-static* parts are checked on Linux
  (`tests/test_windows_packaging.py` syntax-checks the spec/hooks and asserts
  no bundled weights, onedir, pywin32 hidden imports, no hard-coded version).
  The freeze, the .NET/WebView2 build, the MSI/bundle build, and the clean-VM
  install/upgrade/uninstall runs happen on the still-TODO `windows-latest` job
  (WIN-1.8), which is where these flip to `DONE`.
- **WIN-5.8 release slot is temporarily deferred.** `WIN-01` is not registered
  as a release blocker. `packaging/npm/test/windows-release-check.test.js`
  retains a non-blocking TODO enumerating the target Windows evidence gates.
  Mutating the consumed `required-gates.json` arrays and `release.yml` jobs
  remains deferred to WIN-1.8, where Windows evidence can be added with the
  manifest-scoped rule that preserves Linux/macOS-only releases.

### WIN-5.1 — PyInstaller onedir spec + hooks · L
- `onedir` (not `onefile`) spec with explicit hooks for: dynamic segmentation
  imports, ORT provider DLLs, OpenCV/MediaPipe/Pillow/`pyvirtualcam` natives,
  `custback/avatar/avatar.yaml` package resource, pinned model files (only if
  WIN-0.4 permits), version metadata/icon/third-party notices. Build on Windows.
- **Accept:** unpacked artifact runs with `PATH`/`PYTHONPATH`/Python/Node/CUDA
  toolkit assumptions removed. **Deps:** WIN-2.8, WIN-3.7, WIN-4.4.

### WIN-5.2 — C# WebView2/tray shell · L
- Single-instance named mutex; supervise the packaged engine; reserve loopback
  port and wait for explicit readiness; load the existing UI in WebView2; show
  startup/doctor failures before the API is up. Do not weaken existing Host/
  Origin/cookie/bearer boundaries because both processes are local. **Deps:** WIN-5.1.

### WIN-5.3 — Secure WebView session bootstrap · M
- Establish the HttpOnly browser session without putting the long-lived token in
  a URL, command line, web storage, or log; add a private lifecycle/shutdown
  channel. **Deps:** WIN-5.2.

### WIN-5.4 — Tray lifecycle · M
- Keep processing when hidden to tray; graceful shutdown waits for camera/output
  cleanup; handle engine crashes, suspend/resume, optional launch-at-login.
  **Deps:** WIN-5.2.

### WIN-5.5 — Supervise avatar second process · S
- Start/supervise the avatar service process only when that feature is
  installed. Per D9, reserve a second ephemeral loopback port distinct from the
  engine port, pass its URL and the avatar token path to the engine proxy, and
  pass the same port to the avatar service. **Deps:** WIN-5.2.

### WIN-5.6 — Signed per-user EXE/MSI installer · L
- WiX or Inno Setup, per-user. Include/detect: WebView2 Evergreen runtime,
  compatible VC++ runtime, OBS Virtual Camera prerequisite, exact
  ORT/CUDA/cuDNN components for the chosen profile, start-menu shortcut,
  uninstaller, upgrade/rollback, version info. Sign and timestamp every
  executable/installer (SmartScreen). **Deps:** WIN-5.1.

### WIN-5.7 — Uninstall data-retention policy · S
- Do not delete user backgrounds, rigs, config, or logs on ordinary uninstall
  without explicit choice; if a native vcam is later registered, uninstall must
  remove that registration. **Deps:** WIN-5.6.

### WIN-5.8 — Windows release-evidence pipeline · M
- Wire the Windows installer build, signature verification, SBOM/notices,
  artifact-content checks, upgrade-from-previous, and uninstall/data-retention
  tests into the release gate slot defined in WIN-0.2 (CC-3). **Deps:** WIN-5.6,
  WIN-0.2.

---

## Phase 6 — Extended Windows support (size XL, separately approved)

**Goal:** features that must earn their own clean-machine/hardware/security/
packaging/release evidence rather than inheriting the core claim.

- **WIN-6.1 — Native Win11 Media Foundation virtual camera** (`XL`): its own
  C++/WinRT project — custom media source, registration/lifecycle, packaging,
  uninstall cleanup. **Deps:** WIN-5.8.
- **WIN-6.2 — Generic GPU provider** (`XL`): DirectML or Windows ML as a release
  gate; product may not be described as GPU-capable on all Windows hardware
  until it passes. **Deps:** WIN-4.7.
- **WIN-6.3 — ARM64 support** (`L`): dependency, build, and hardware evidence.
  **Deps:** WIN-5.8.
- **WIN-6.4 — Avatar + Audio2Face Windows parity** (`XL`): packaging, storage,
  and driver work for the second service on Windows. **Deps:** WIN-5.8.

**Implementation notes (landed as reviewable source; `IMPL*` = the
Windows-execution/hardware halves pend the WIN-1.8 job and the validation
matrix, same convention as Phases 2–5):**

- **WIN-6.1 (native virtual camera).** `packaging/windows/vcam/` is the
  C++/WinRT project: `Activator` (IMFActivate, CLSID
  `{7A4C1B2E-9D35-4E6A-8B1F-52C84D9A6E01}`) → `MediaSource`/`MediaStream`
  (IMFMediaSourceEx/IMFMediaStream2/IKsControl, one always-selected RGB32
  stream at 720p/1080p\@30). Frames arrive over a named-section seqlock ring —
  `src/custback/vcam_native.py` is the layout's source of truth,
  `FrameRing.h` mirrors it, and the reader holds the last frame or a
  placeholder when the engine idles (a camera must never stall its consumer).
  The engine side is `output.backend: native` → `NativeVirtualCameraOutput`
  (explicit opt-in; the Windows `auto` ladder is already pyvirtualcam → native
  → null, but `_AUTO_NATIVE_ENABLED = False` disables the native rung until the
  WIN-6.1 clean-machine gate passes; POSIX fails closed). Lifecycle: after
  engine readiness the shell reads the authenticated status and calls
  `VirtualCameraSession.TryStart` only when the active backend is native.
  A DLL-bearing build with any other backend starts no placeholder-only camera
  and logs the mismatch. `VirtualCameraSession.cs` calls
  `MFCreateVirtualCamera` with **session** lifetime + current-user access, so
  nothing persists past the process; Windows status reports the named-section
  ring as present/absent (unsupported off Windows). The MSI owns the per-user
  COM registration
  (`IncludeNativeVCam` define, staged by `installer/build.ps1 -VCamDll`) and
  its removal — the WIN-5.7 placeholder CLSID is now the real one. Loop
  prevention holds ("Custback Camera" matches the existing `custback`
  marker). `tests/test_windows_vcam.py` pins the protocol constants, CLSID
  consistency across all four files, fail-closed POSIX behavior, and the
  packaging wiring.
- **WIN-6.2 (generic GPU gate).** The claim is now gated by machinery, not
  prose: the `directml` extra (`onnxruntime-directml`, mutually exclusive
  with `gpu` — enforced at freeze time and at gate time) plus
  `scripts/release/windows-acceleration-gate.py` — `run` produces per-machine
  schema-2 evidence (real-RVM proof via `prove_rvm_provider`, worst per-frame
  mean alpha drift vs CPU as `alpha_delta_mean_worst`, 720p/1080p timing,
  adapter identity, wheel-conflict state); `run --strict` fails a local
  NO-GO, while CI must use `check` as the deterministic validator (both-vendor
  AMD+Intel coverage, drift ≤ 0.005 worst per-frame mean / 0.02 max, 720p ≥
  target FPS, no co-installation) that becomes
  the `windows-acceleration` gate validator when WIN-1.8/WIN-5.8 wire the
  Windows evidence source. Criteria are pinned by
  `tests/test_windows_acceleration_gate.py`; DirectML stays unadvertised
  until real AMD+Intel evidence passes.
- **WIN-6.3 (ARM64).** Architecture is a parameter, not a fork:
  `pyinstaller/build.ps1 -Arch arm64` (verifies `platform.machine()`, trims
  extras to the ARM64 profile, refuses CUDA/mediapipe),
  `installer/build.ps1 -Arch` → WiX `-arch` + `VC_redist.$(var.Arch).exe`,
  `CustbackVCam.vcxproj` ARM64 configurations, shell `RuntimeIdentifiers`
  `win-x64;win-arm64`. `WINDOWS_ARM64.md` records the dependency profile:
  a PEP 508 marker and the PyInstaller spec omit wheel-less `pyvirtualcam`;
  native MF will replace that path only after WIN-6.1 evidence. Today `auto`
  reaches API-only `NullOutput` and explicit `native` is the manual gate-build
  opt-in. It also records the build commands and the arm64-* evidence rows
  required before any Supported claim.
- **WIN-6.4 (avatar parity).** `custback.spec` now freezes **two console
  executables into one onedir payload** (`custback.exe` +
  `custback-avatar.exe`, shared dependency set), with the avatar driver stack
  selected per payload via `CUSTBACK_AVATAR_PROFILE` (vision/MediaPipe
  default per D7; audio2face flavor swaps stacks — the protobuf conflict
  makes them one-per-payload, mirrored by `build.ps1 -AvatarProfile` guards).
  Per D9, the shell reserves an ephemeral avatar API port distinct from the
  engine port and, only for an installed avatar, starts the engine with
  `--avatar-url http://127.0.0.1:<avatar-port> --avatar-token-file
  <avatar-api-token>`. The proxy freezes that destination and token path but
  reads the service-minted token value per request, preserving the existing
  engine-before-avatar order. The shell's supervised avatar contract is the
  *real* CLI (the WIN-5.5 `serve` argument was a latent bug): started only
  after engine readiness with `--source ws://127.0.0.1:<engine-port>
  --source-token-file <renderer-token> --api-port <avatar-port>
  --api-token-file <avatar-api-token>` (paths, never secrets), with a bounded
  3-restart budget. Driver selection is intentionally left to the avatar
  config file; the shell does not pass `--driver`. Quit gives the avatar a
  short grace then a tree kill. Source-install engine/avatar token defaults
  share the platform config-directory resolver; the packaged product also
  passes the same explicit token path to both processes. Storage stays on the
  Phase-2 seam. The installer ships the avatar only with
  `-IncludeAvatar` (dropped from the staged payload otherwise), and
  `build.ps1` smoke-runs `custback-avatar.exe --smoke` in the scrubbed
  environment.

---

## Validation matrix (from feasibility doc, mapped to phases)

**Automated Windows CI** (grows across phases — WIN-1.8 seeds it):
Python 3.12 x64 core/unit/package; import + synthetic/null smoke from source and
frozen artifact; CPU RVM with real pinned model; model-lock contention/timeout/
interrupted-download/checksum; NTFS ACL/reparse/identity/no-replace/crash-
recovery/quota/concurrent-upload; loopback auth/session/WebSocket/MJPEG/shutdown;
installer build/signature/SBOM/notices/content; upgrade + uninstall/data-
retention.

**Hardware / manual release gates** (Phase 4/5 exit): no-GPU CPU fallback
reported; NVIDIA RVM profiled on CUDA; advertised-CUDA-but-missing-DLL non-fatal
start; mid-run GPU/OOM failure retries once on CPU and latches; AMD/Intel if a
DirectML claim is made; multi-camera/privacy-denied/replug/reorder/suspend/
contention; OBS missing/installed/busy/in-use; Teams/Zoom/Meet + one more MF/DS
consumer; 720p+1080p perf measurement; online and offline/blocked first run.

---

## Risk register (mapped to tasks)

| Risk | Mitigation task(s) |
| --- | --- |
| Silent security no-op on Windows (review finding) | WIN-1.4 (fail-closed), then WIN-2.3/2.4/2.6 |
| Naive replacement of chmod/flock exposes tokens | WIN-2.1..2.8 with adversarial tests |
| GPU package size / DLL mismatch | WIN-1.6 pin, WIN-4.3 preload/probe, WIN-5.1 onedir |
| CPU fallback misses frame rate | WIN-4.5 attainment reporting, WIN-4.6 presets |
| OBS external / single instance | WIN-3.7 detect+explain; WIN-6.1 native later |
| Unstable OpenCV device indices incl. output cam | WIN-3.1 stable IDs, WIN-3.4 loop prevention |
| Frozen native modules omitted | WIN-5.1 explicit spec + clean-VM artifact test |
| Newer Windows ML surface | WIN-4.7 spike before committing |
| Unsigned camera software / SmartScreen | WIN-5.6 sign + timestamp everything |
| GPL model/vcam licensing vs MIT project | WIN-0.4 gate (CC-5) before bundling |
| Offline model behavior | WIN-0.4 bundling decision; WIN-4.5 first-run state |
| New file breaks release gate | CC-2 discipline, WIN-1.7 |

---

## Go/no-go checkpoints

- **After Phase 1:** if native import + synthetic + CPU RVM cannot pass on
  `windows-latest`, stop and reconsider — the rest depends on it.
- **After Phase 2:** if security invariants cannot be preserved on NTFS, stop
  before investing in the shell (feasibility doc's explicit stop condition).
- **After Phase 3:** if a stable clean-machine camera path is not achievable,
  reconsider the product boundary before Phase 5.
