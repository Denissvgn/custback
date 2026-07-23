# Phase 6 review — mitigation plan

Status: **IN PROGRESS (MIT-A and MIT-D complete; MIT-B and MIT-C1/C3/C4
implemented, Windows gates pending; MIT-C2 contingent)** · Source: code review
of commit `5ee298e` ("feat: add Windows virtual camera support and type checks")
· Companion to `WINDOWS_IMPLEMENTATION_PLAN.md`, `WINDOWS_FEATURE_MATRIX.md`,
`WINDOWS_ARM64.md`, `WINDOWS_ACCELERATION_SPIKE.md`.

The Phase 6 review found four defects, three hardware-pending design risks,
and three consistency gaps. This document decomposes them into executable
tasks. Conventions match the implementation plan: every task has an ID,
size (S/M/L), dependencies, and an acceptance criterion that is a test or a
gate, not prose. Status flips to `DONE` here and nowhere else.

Task groups:

- **MIT-A** — correctness fixes (small, self-contained, land first)
- **MIT-B** — packaged-product integration wiring (implemented; decision D9,
  Windows acceptance gates pending)
- **MIT-C** — hardware-gate hardening (protects the WIN-6.1/6.3 gates)
- **MIT-D** — tooling and consistency (complete)

## Task index

| ID | Task | Size | Deps | Status |
| --- | --- | --- | --- | --- |
| MIT-A1 | Frame ring starts inactive | S | — | DONE |
| MIT-A2 | Shell stops forcing `--driver auto` | S | — | DONE |
| MIT-B1 | Decide avatar control-plane wiring (design) | S | — | DONE |
| MIT-B2 | Wire engine `/avatar/*` proxy in the packaged product | M | MIT-B1 | IMPL* (local two-process pass; WIN-1.8 row pending) |
| MIT-B3 | Align avatar token path across engine and shell | S | MIT-B1 | DONE |
| MIT-B4 | Native camera / output backend coherence | M | — | IMPL* (clean-machine rows pending) |
| MIT-B5 | Reconcile ARM64 output path with the `auto` guardrail | S | MIT-B4 | DONE |
| MIT-C1 | Section-namespace probe on the WIN-6.1 gate checklist | S | — | IMPL* (probe/checklist landed; clean-machine run pending) |
| MIT-C2 | Global-section fallback with explicit DACL (contingent) | M | MIT-C1 fails | CONTINGENT (not activated) |
| MIT-C3 | Seqlock memory-ordering hardening (C++ reader) | S | — | IMPL* (source/static gate landed; Windows build pending) |
| MIT-C4 | IMFVirtualCamera projection assertions on hardware | S | WIN-1.8 | IMPL* (self-check/checklist landed; Windows run pending) |
| MIT-D1 | Wire pyright into CI | S | — | DONE |
| MIT-D2 | Acceleration-gate exit-code and naming cleanup | S | — | DONE |
| MIT-D3 | `.hallmark/` provenance decision | S | — | DONE |

Recommended landing order: **A1, A2, D2, D3 → B1 → B2, B3, B4 → B5, D1 →
C1–C4 ride the existing WIN-1.8 / clean-machine gate wiring.**

---

## Group A — correctness fixes

### MIT-A1 — Frame ring starts inactive (S)

**Defect.** `FrameRingWriter.__init__` publishes the initial header with
`flags=FLAG_ACTIVE` while its own comment says "inactive"
(`src/custback/vcam_native.py`). A reader attaching between ring creation and
the first `publish()` accepts the zero-filled payload and serves a solid
black frame instead of the placeholder — the exact state the placeholder
design exists to prevent.

**Change.**
1. In `__init__`, publish the initial header with `flags=0`; the first
   `publish()` already raises `FLAG_ACTIVE`.
2. Keep the comment; it becomes true.

**Tests.**
- Extend `tests/test_windows_vcam.py`: after constructing a writer and before
  any `publish()`, `read_latest_frame(buffer)` must return `None`
  (reader shows placeholder). After the first `publish()`, it returns the
  frame. After `close()`, `None` again.

**Acceptance.** New test passes; no other ring test changes behavior. The C++
mirror needs no change (the reader already keys off `FLAG_ACTIVE`).

### MIT-A2 — Shell stops forcing `--driver auto` (S)

**Defect.** `Engine.StartAvatar` (`packaging/windows/shell/Engine.cs`) passes
`--driver auto`. `config_from_args` applies CLI args after the config file
and `"auto"` is truthy, so a user-configured `driver.backend: audio2face` is
silently clobbered. The audio2face payload flavor (WIN-6.4) can never engage
its driver under shell supervision.

**Change.**
1. Remove `"--driver", "auto"` from the supervision argument list. The avatar
   config's own default (`backend: auto`) provides identical behavior when
   nothing is configured, and a configured backend now wins.
2. Do **not** add a shell-side driver setting; driver choice belongs to the
   avatar config file (one source of truth).

**Tests.**
- Update `tests/test_windows_packaging.py::test_shell_supervises_avatar_with_real_cli_contract`:
  assert `--driver` is **absent** from `Engine.cs` (flip of the current
  assertion), keep `--source`, `--source-token-file`, `--api-token-file`.
- Add a CLI-precedence regression test in `tests/test_avatar_config.py` (or
  nearest): `config_from_args` with `args.driver=None` must preserve a
  config-file `audio2face` backend; with an explicit `--driver` it must
  override. (Pins the semantics the shell now relies on.)

**Acceptance.** Both tests pass; `WINDOWS_IMPLEMENTATION_PLAN.md` WIN-6.4
note updated (the supervision contract no longer lists `--driver`).

---

## Group B — packaged-product integration wiring

**Status: IMPL* (2026-07-23).** D9 records the packaged control-plane and port
decision; B2–B5 implement and pin the resulting wiring and guardrails. B2's
local process acceptance passes, while its reusable WIN-1.8 row and B4's
packaged clean-machine rows remain evidence-gated.

The engine, shell, and avatar service each behave correctly alone; the gaps
are in what nobody passes between them. B1 is the one genuine design
decision; B2–B5 are mechanical once it is made.

### MIT-B1 — Decide avatar control-plane wiring (S, design note)

**Gap.** In an `-IncludeAvatar` install the avatar *frame* path works (the
service connects to the engine's renderer WebSocket on its own), but the
*control* path is dead: the engine's `/avatar/*` proxy is disabled
(`avatar.url` defaults empty) and nothing configures it, so the web UI's
avatar tab cannot reach the avatar service.

**Options.**
- **(a) Shell passes engine CLI flags** (`--avatar-url`, `--avatar-token-file`)
  when it supervises an avatar — new engine CLI surface, explicit, mirrors the
  existing "paths, not secrets" token flags. **Recommended**: it keeps the
  wiring in the one component that already knows both port and token path,
  needs no config-file mutation, and works with a read-only config.
- (b) Shell writes/merges an engine config drop-in setting `avatar.url` +
  `avatar.token_file` — no new CLI surface, but the shell starts mutating
  engine config, and stale drop-ins outlive an uninstalled avatar.
- (c) Windows-aware engine defaults (probe `%APPDATA%\Custback\...` and
  `127.0.0.1:8711`) — zero wiring, but implicit cross-process coupling and a
  probe on every start; rejected by the same reasoning that made WIN-5.3
  token paths explicit.

**Deliverable.** A dated decision entry in `WINDOWS_DECISIONS.md` (D-series)
choosing one option, with the port-collision story (avatar API port is
currently the fixed default 8711 while the engine port is shell-reserved —
decide whether the shell should reserve the avatar port the same way).

**Resolution.** D9 (2026-07-23) chooses option (a): the shell passes explicit
engine proxy flags and reserves a second ephemeral loopback port, distinct
from the engine port, which it also passes to the avatar as `--api-port`.

**Acceptance.** Decision recorded; B2/B3 reference D9.

### MIT-B2 — Wire the engine `/avatar/*` proxy in the packaged product (M)

**Change (per D9).**
1. Engine CLI: add `--avatar-url` and `--avatar-token-file` to
   `src/custback/__main__.py`, mapped onto `avatar.url` /
   `avatar.token_file` with the same "assemble one candidate, validate
   atomically" pattern the existing overrides use.
2. Shell: in `Engine.StartAsync`, when `SuperviseAvatar`, extend the engine
   argument list with the avatar API URL (`http://127.0.0.1:<avatar-port>`)
   and `_options.AvatarTokenFile`. Reserve the avatar port beside, and
   distinctly from, the engine port and pass `--api-port` to the avatar too.
3. Ordering: the proxy reads the token lazily per request (the avatar service
   mints the token on its own startup), so engine-before-avatar startup order
   stays valid — verify, don't assume, in the test below.

**Tests.**
- Engine CLI mapping test (config override reaches
  `RuntimeConfig.avatar.url` / `.token_file`).
- `tests/test_avatar_proxy.py`: proxy configured with a token file that does
  not exist yet at app startup but exists at request time must succeed
  (pins the lazy-read ordering the shell relies on).
- `tests/test_windows_packaging.py`: `Engine.cs` must pass `--avatar-url` and
  `--avatar-token-file` iff it supervises the avatar; assert both flags
  appear in the supervised argument list.

**Acceptance.** With a locally simulated two-process run (engine + avatar,
loopback), `GET /avatar/status` through the engine returns the avatar
service's state. Add this as a scenario to the WIN-1.8 supervised-run job
when it lands (`WINDOWS_FEATURE_MATRIX.md` §6 row gains the proxy check).

**Evidence (2026-07-23).**
`test_real_engine_and_avatar_processes_proxy_status_after_token_mint` launches
the actual engine and avatar CLI processes on distinct reserved loopback ports,
starts the engine before the avatar token exists, and passes only after an
authenticated `GET /avatar/status` through the engine returns the avatar
service state. The reusable Windows WIN-1.8 shell/hardware row remains planned
as recorded in the feature matrix.

### MIT-B3 — Align the avatar token path across engine and shell (S)

**Gap.** Engine default `avatar.token_file` is
`~/.config/custback/avatar-api-token`; the shell provisions
`%APPDATA%\Custback\avatar-api-token`. Two different directories on Windows.

**Change.** With D9 the explicit flag removes the mismatch for the
packaged product; this task is the source-install cleanup: route the engine
default through the platform seam's config-dir helper (same mechanism the
Phase-2 seam uses) so `avatar.token_file` and the avatar service's own
`api.token_file` default resolve to the *same* per-platform directory.

**Tests.** A seam-level test asserting both defaults resolve to one path per
platform (parametrized POSIX/Windows via the existing seam test fixtures in
`tests/test_platform_seam.py`).

**Acceptance.** Defaults agree on every platform; docs in `config/default.yaml`
comment updated.

### MIT-B4 — Native camera / output backend coherence (M)

**Gap.** The shell starts "Custback Camera" whenever `CustbackVCam.dll` is
present, but default `output.backend: auto` never writes the ring — consumers
see a permanent placeholder camera. Nothing configures `native` in packaged
builds and no doc records the manual step.

**Change.** Keep both guardrails (DLL-gated shell start; `auto` never selects
native until the clean-machine gate passes) but make the mismatch impossible
to ship silently:
1. Shell: before `TryStart`, ask the engine which output backend is active
   (the status/diagnostics API already reports truthful `output` state — use
   it). If the backend is not `native`, **do not start the camera**; log
   `"native vcam DLL installed but engine output.backend is '<x>'; native
   camera not started"`. This turns the placeholder camera into an explicit,
   diagnosable condition.
2. Installer/docs: `packaging/windows/vcam/README.md` and the installer
   README gain the one-line operator step (`output.backend: native`) for
   gate-testing builds, until WIN-6.1 evidence lets `auto` participate.
3. Engine diagnostics: `--doctor`/status output on Windows reports
   `native ring: <section present/absent>` so a tester can see both halves.

**Tests.**
- `tests/test_windows_vcam.py::test_shell_owns_camera_lifecycle` extended:
  `VirtualCameraSession`/`TrayApplicationContext` must reference the engine
  backend check (assert on the source, same style as the existing wiring
  tests).
- Diagnostics test for the ring-presence line (POSIX: reports "unsupported",
  fail-closed wording).

**Acceptance.** A packaged build with the DLL but default config starts **no**
native camera and logs why; a `backend: native` config starts it. Both become
rows in the WIN-6.1 clean-machine gate checklist.

### MIT-B5 — Reconcile the ARM64 output path with the `auto` guardrail (S)

**Gap.** `WINDOWS_ARM64.md` names the native camera "the ARM64 output path",
but `auto` degrades to `NullOutput` on ARM64 (no pyvirtualcam wheel) — an
ARM64 build ships with no working camera by default. The two guardrails
contradict on ARM64.

**Change (docs + one scoped code rule).**
1. Encode the intended end-state in `vcam.py`: once WIN-6.1 evidence passes,
   `auto` on Windows may try `native` **after** pyvirtualcam fails (ladder:
   pyvirtualcam → native → null). Implement the ladder now but keep the
   native rung **feature-flagged off** by a single module-level constant that
   the WIN-6.1 gate flip will change (`_AUTO_NATIVE_ENABLED = False`), so the
   flip is a one-line, test-pinned change instead of a rewrite.
2. `WINDOWS_ARM64.md` §1: replace "is the ARM64 output path" with the honest
   present tense — "will be, once WIN-6.1 evidence lands; until then ARM64
   defaults to API-only output (`NullOutput`) and `backend: native` is the
   manual opt-in".
3. `tests/test_windows_vcam.py::test_auto_backend_does_not_select_native`
   stays as the guardrail pin and gains its counterpart: a test asserting the
   ladder order *when the flag is flipped* (parametrize on the constant).

**Acceptance.** Docs and code agree; flipping one constant (plus its test
expectation) is the entire WIN-6.1-pass follow-up.

---

## Group C — hardware-gate hardening

These protect the gates from failing late or passing wrongly. None can be
closed from this machine; each lands as machinery + checklist rows.

**Status: IMPL* (2026-07-23).** C1, C3, and C4 machinery and byte-static tests
have landed; their Windows build and hardware rows remain pending. C2 stays
dormant unless the C1 trace proves that the Frame Server cannot open the
session-local section.

### MIT-C1 — Section-namespace probe on the WIN-6.1 gate checklist (S)

**Risk.** The ring uses `Local\CustbackVCamFrame0` and the docstring claims
session-local security suffices. The media source runs inside the Frame
Server *service* process; if that host is session-0 (Local Service), a
`Local\` section from the user session is invisible to it and the transport
silently never connects (camera = permanent placeholder — indistinguishable
from MIT-B4's symptom, which is why B4's diagnostics land first).

**Change.**
1. Add a probe mode to the media source for the gate build only: an exported
   diagnostic (or ETW/`OutputDebugString` trace on `Open()` failure) that
   records *why* the ring didn't open (`OpenFileMappingW` error code).
2. WIN-6.1 clean-machine gate checklist gains an explicit first row:
   "engine publishing + native camera shows live frames (not placeholder) in
   the Camera app" — run **before** any Teams/Zoom rows, since it isolates
   the transport from consumer quirks.
3. Soften the `vcam_native.py` docstring: state the namespace assumption as
   an assumption pending gate evidence, referencing this task.

**Acceptance.** Gate checklist updated in `packaging/windows/vcam/README.md`;
docstring no longer overclaims.

**Evidence (2026-07-23).** `build.ps1 -GateDiagnostics` enables a
gate-payload-only `OutputDebugStringW` trace containing the failed
`OpenFileMappingW` name and Win32 error code. The clean-machine checklist now
starts with engine-published live frames in Windows Camera, before any
consumer-specific rows, and the Python module calls the `Local\` visibility a
pending assumption. Execution remains Windows/hardware pending.

### MIT-C2 — Global-section fallback with explicit DACL (M, contingent on C1 failing)

Only if the C1 probe shows the service cannot open the `Local\` section:
1. Switch to `Global\CustbackVCamFrame0` created via `CreateFileMappingW`
   with an explicit security descriptor: read access for the Frame Server
   service SID (or `NT AUTHORITY\LOCAL SERVICE`), full control for the
   owner, nothing else. This leaves `mmap.tagname` (no DACL support) —
   the writer moves to a small pywin32 code path behind the existing
   platform seam, POSIX behavior unchanged (constructor still accepts an
   injected buffer for tests).
2. Mirror the name change in `FrameRing.h`; the drift test in
   `tests/test_windows_vcam.py` already forces the two to move together.
3. Security note in the module docstring: exactly which SID can read, why
   write stays owner-only.

**Acceptance.** Drift tests pass; C1's gate row goes green on hardware.

### MIT-C3 — Seqlock memory-ordering hardening (S)

**Risk.** `FrameRingReader::CopyLatest` re-checks `seq` after the payload
`memcpy` with plain loads — no acquire barrier. On ARM64's weaker memory
model the re-check can be satisfied by a stale load and admit a torn frame.
(Python writer side is sequenced by the interpreter; the risk is the C++
reader.)

**Change.**
1. In `FrameRing.h`, read `seq` (both the initial header snapshot's re-read
   and the post-copy re-check) through `std::atomic_ref<const uint32_t>`
   with `memory_order_acquire`, and insert `std::atomic_thread_fence(acquire)`
   between the payload copy and the seq re-check. Zero cost on x64, correct
   on ARM64.
2. Writer side note: document in `vcam_native.py` that CPython's byte-level
   buffer stores provide the release ordering the protocol needs on the
   writer, and that a future non-CPython writer must add explicit fences.

**Tests.** Byte-static: the drift test gains an assertion that `FrameRing.h`
contains `atomic_ref`/`atomic_thread_fence` (same source-pinning style as the
existing packaging-wiring tests). Behavioral proof is hardware-bound.

**Acceptance.** Compiles in the vcxproj (C++20 already required by
`atomic_ref` — confirm `LanguageStandard` in `CustbackVCam.vcxproj`, bump to
`stdcpp20` if needed); drift test extended.

**Evidence (2026-07-23).** The reader validates the header and post-payload
sequence values with acquire `atomic_ref` loads and places an acquire fence
between payload copy and the final load. The project now selects C++20, the
writer-side ordering requirement is documented, and byte-static tests pin all
three conditions. Windows x64/ARM64 compilation and behavioral proof remain
gate-bound.

### MIT-C4 — IMFVirtualCamera projection assertions on hardware (S)

**Risk.** The C# COM projection (IID `1C08A864-…`, 30-slot IMFAttributes
vtable order, enum values) is unverifiable off-Windows; a wrong slot corrupts
at call time, not build time.

**Change.** Add a shell self-check (debug/gate builds): after
`MFCreateVirtualCamera`, call one cheap late-vtable method
(`GetItemByIndex`-region method such as `GetCount`) and log the HRESULT —
a vtable misalignment fails here loudly instead of inside `Start`. Add a row
to the WIN-1.8 shell smoke.

**Acceptance.** Self-check present and referenced by the gate checklist.

**Evidence (2026-07-23).** Debug builds and Release builds published with
`-p:CustbackGateBuild=true` call the preserved-signature
`IMFAttributes::GetCount` slot immediately after `MFCreateVirtualCamera`, log
its HRESULT and attribute count, and throw on failure before `Start`. The
WIN-1.8 shell-smoke instructions and ordered WIN-6.1 checklist require the
successful log line; execution remains Windows/hardware pending.

---

## Group D — tooling and consistency

### MIT-D1 — Wire pyright into CI (S)

**Gap.** `pyrightconfig.json` landed (commit message says "type checks") but
nothing runs pyright: not `ci.yml`, not the release gates, not the venv.

**Change.**
1. Add a `pyright` job to `.github/workflows/ci.yml` (pin the pyright
   version; `basic` mode per the existing config).
2. Decide gate status: if it is release-blocking, add it to
   `required_gates`/`required_job_ids` in
   `scripts/release/required-gates.json` **and** the reviewed manifest in
   `scripts/release/verify-release.js` (memory rule: gate files are pinned);
   if advisory-only, say so in a comment in `pyrightconfig.json` so the
   commit message's claim has a truthful referent.
3. Run it once locally and fix or explicitly suppress any existing findings
   so the job starts green.

**Acceptance.** CI job green; gates manifest consistent with the decision.

**Implementation (2026-07-23).** Pyright 1.1.411 is exact-pinned in the
development environment and runs in a dedicated Python 3.10 push/PR CI job
against the configured `.venv`. It is enforcing in that workflow but advisory
to the artifact-bound Phase 6 publication gate, as recorded in
`pyrightconfig.json`; the release required-gate manifest therefore remains
unchanged. The release source verifier pins the dependency and CI contract.
The initial full include completed with zero errors and warnings after narrow
cross-platform typing fixes.

### MIT-D2 — Acceleration-gate exit-code and naming cleanup (S)

**Gaps.** (1) `windows-acceleration-gate.py run` exits 0 even on a local
NO-GO — correct for evidence collection, but nothing stops future CI wiring
from trusting it. (2) `correctness.alpha_delta_mean` stores the *max* of
per-frame means; the name misleads evidence readers.

**Change.**
1. Add `--strict` to `run` (exit 1 on local NO-GO) and a docstring line
   stating that CI must gate on `check`, never on `run`'s default exit code.
2. Rename the evidence field to `alpha_delta_mean_worst` (and `…_max` stays)
   **and bump `SCHEMA_VERSION` to 2** — the checker rejects mismatched
   schemas by design, so the rename is safe only with the bump. Update
   `EVIDENCE_KEYS`-adjacent docs and `tests/test_windows_acceleration_gate.py`
   fixtures.

**Tests.** Extend the gate tests: `--strict` semantics; schema-2 evidence
accepted, schema-1 rejected with the existing "wrong schema_version" reason.

**Acceptance.** Tests pass; no evidence files exist yet, so the bump has no
migration cost — which is exactly why this lands now, before WIN-1.8 wiring.

**Implementation (2026-07-23).** The harness now emits and accepts schema 2
with `correctness.alpha_delta_mean_worst`, rejects schema 1 before nested
evaluation, and validates the exact nested correctness keys. `run --strict`
returns 1 for a local NO-GO; default `run` remains a collection command and
prints a warning directing CI to `check`. The 13-test focused gate suite,
Ruff lint/format checks, and bytecode compilation pass.

### MIT-D3 — `.hallmark/` provenance decision (S)

**Gap.** `.hallmark/` (web-UI design-tool provenance) is committed at the
repo root; it is unclear whether that was deliberate.

**Change.** Decide and make the tree match: either keep it (add one line to
`README.md`'s repo-layout section saying what it is), or drop it
(`git rm -r --cached .hallmark` + `.gitignore` entry). Keeping it is
harmless; the requirement is that the decision be visible, not implicit.

**Acceptance.** Either documented or ignored; no silently-tracked tool state.

**Implementation.** Removed `.hallmark/` from the Git index while preserving
the local tool state, added the directory to `.gitignore`, and removed its
former repository-layout entry from the README.

---

## Sequencing and gate impact

```
A1 ─┐
A2 ─┤  (independent, land immediately)
D2 ─┤
D3 ─┘
B1 ──► B2 ──┐
       B3 ──┤──► WIN-1.8 supervised-run scenario (proxy check)
B4 ─────────┤──► WIN-6.1 clean-machine gate rows (live-frame first, C1 probe)
       B5 ──┘
D1 ──► required-gates.json (if blocking)
C1 ──► C2 (only if probe fails)   C3, C4 ──► gate checklists
```

Nothing here moves a Phase 6 status past `IMPL*`: the WIN-6.x rows in
`WINDOWS_IMPLEMENTATION_PLAN.md` still pend WIN-1.8 and hardware evidence.
This plan removes the defects that would have made that evidence misleading
(placeholder-camera false negatives, dead avatar control plane) before any
hardware time is spent.
