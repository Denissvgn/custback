# Phase 0 visual-consistency implementation review

- Status: Accepted; VIS-0.1 through VIS-0.4 complete
- Review date: 2026-07-30
- Scope: contract, deterministic baseline, algorithm selection, strict
  configuration, migration, public policy state, and lifecycle plumbing
- Backlog: `AUTO_COLOR_CORRECTION_AND_SCALING_BACKLOG.md`

## Outcome

Phase 0 is closed with no unresolved Phase 0 acceptance gap or contradictory
decision. It establishes the contract and validated control plane needed for
later pixel-path work. It deliberately does **not** claim that production
camera fitting, ICC conversion, linear-light compositing, or foreground
harmonization is implemented; those remain ordered Phase 1 and Phase 2 tasks.

The compatibility landing state is:

| Policy | Schema-v1/default value | Qualified target after later gates |
| --- | --- | --- |
| Main-camera fit | `stretch` | `cover` |
| Backdrop fit | `cover`, center anchor | unchanged |
| Blend space | `srgb_legacy` | `linear_srgb` |
| Color correction | `off` | `auto` |
| Canvas | paired `output.width`/`height`, otherwise camera request | unchanged |

No target default may flip without the later golden, performance, platform,
privacy, rollback, and schema-migration work in VIS-4.3.

## Requirement-by-requirement closure

### VIS-0.1 — frame/color contract

The accepted ADR at `docs/adr/0001-visual-consistency-contract.md`:

- separates acquisition, delivered, oriented, and canonical-canvas sizes;
- assigns one representation, color assumption, and geometry owner to every
  core source and sink;
- freezes orientation → rotation → viewer-horizontal mirror → fit order;
- defines `cover`, `contain`, and explicit legacy `stretch`, anchor
  coordinates, odd-pixel rounding, padding validity, and direction-aware
  interpolation;
- includes worked 4:3→16:9, 16:9→4:3, portrait→landscape, odd-size, and
  anchor-0/1 examples;
- defines external full-range sRGB-assumed `uint8 BGR` and bounded internal
  linear-sRGB `float32 RGB` boundaries;
- records raw, remote, exact-size, privacy-slate, and Windows native-output
  invariants;
- prohibits histogram-only range inference and remote-frame auto-fitting;
- defines hard/soft temporal resets and transactional state ownership;
- classifies restart-only and hot policy fields;
- makes `blend_space` a temporary compatibility switch; and
- selects explicit persisted schema versioning rather than globally
  reinterpreting absent fields.

The Phase 0 safe highlight fallback is now explicit hard clipping. A different
roll-off remains a measured VIS-2.1 decision rather than an undefined
“ratified policy.”

### VIS-0.2 — fixtures, metrics, and baseline

The non-production generator at `tests/visual_consistency_evidence.py`, its
acceptance suite, and the evidence report provide:

- asymmetric square, 4:3, 16:9, portrait, ultrawide, and odd-size grids with
  labels, non-centered face/circle outline, and exact EXIF 1–8 transforms;
- warm/cool neutrals, real ±1 EV estimator cases, clipping, shadows, skin-like
  and saturated-clothing patches, neutral/saturated targets, static noise,
  drift, hard cut, missing frame, and reconnect sequences;
- generated, valid, byte-stable sRGB, Display-P3, Adobe-RGB, and synthetic
  CMYK ICC-tagged assets with hashes and CC0 provenance;
- numeric geometry, crop, luminance-gap, neutral-axis, hue/chroma,
  jitter/settling, and soft-edge metrics;
- a production `FrameHub.stats_dict` EWMA timing snapshot plus isolated
  microbenchmarks; and
- a generated contact sheet whose pixels are parity-tested but never used as
  a subjective acceptance oracle.

The current production discrepancies are captured exactly:

- 4:3→16:9 direct stretch has `33.333%` distortion; and
- encoded 50% blending returns `100`, while the linear-light reference is
  `146`.

The unavailable rotation-metadata video writer/prober is the tooling exception
explicitly permitted by VIS-0.2. EXIF coverage is complete, and decoder-level
video orientation remains VIS-1.2 work.

### VIS-0.3 — bounded algorithm selection

The evidence compares exposure-only, bounded exposure plus diagonal WB, and
deliberately aggressive moment transfer. The selected default-strength
candidate:

- reduces luminance gap by `47.375%`;
- reduces neutral-axis error by `23.789%`;
- keeps skin drift to `2.155°` hue / `10.684%` normalized chroma; and
- keeps clothing drift to `1.123°` hue / `8.851%` normalized chroma.

Aggressive transfer is rejected because it reaches `73.738°` hue and
`97.299%` normalized-chroma drift.

The final estimator policy uses a 192-pixel analysis long edge, 0.90 mask
threshold plus 5×5 erosion, local 19×19 annulus, 96-sample minimum, separate
exposure and WB confidence thresholds of 0.45, ±0.85 EV default clamp,
0.86–1.16 WB gains, and independent 0.50 exposure/WB strengths. WB confidence
explicitly includes neutral-sample availability; valid exposure can therefore
fall back to exposure-only without misreporting WB confidence.

The preservation gates qualify the selected default strength pair, matching
the backlog acceptance criterion. Advanced strengths remain bounded
interpolation controls and are not represented as having the same measured
default-setting drift. `exposure_limit_ev` defaults to 0.85 and has the
experiment-approved schema safety ceiling of 1.0 EV.

`adaptation_time_s: 0.8` is ratified as the initial schema default, not
misrepresented as an instantaneous-estimator measurement. VIS-2.4 must validate
its runtime effect before `auto` is eligible to become the default; changing it
requires an explicit ADR/config amendment.

### VIS-0.4 — strict configuration and lifecycle

The production configuration now provides:

- strict fit, right-angle rotation, anchor, frame-dimension, blend-space,
  correction-mode, strength, exposure-limit, and adaptation-time types;
- paired optional output dimensions and the sole
  `resolved_output_size(config)` resolver;
- schema-v1 compatibility defaults for camera/background geometry,
  `srgb_legacy`, correction `off`, and the selected bounded constants;
- schema-aware materialization before model defaults, so explicit or
  versionless v1 documents cannot inherit a future schema's target defaults;
- explicit migration that writes schema version and effective visual policy
  for both versionless and incomplete explicit-v1 documents;
- restart-only camera/canvas/schema policy and hot transactional backdrop
  geometry, blend-space, and correction policy;
- separate backdrop provider identity and visual-state keys, so fit/anchor
  changes advance visual generation without reopening or rewinding a video;
- rollback of resources, effective config, version, and visual generation on
  failed activation;
- public/OpenAPI geometry, blend, correction, and schema values without new
  backdrop source/device disclosure; and
- startup diagnostics that emit the effective schema and visual policy through
  the sanitized allow-list.

The default YAML, merge-patch null/reset behavior, API 409 shape, migration
goldens/provenance, and release manifests are covered by tests.

## Review findings closed

The closing audit found and resolved the following discrepancies:

| Finding | Resolution |
| --- | --- |
| Injecting `schema_version: 1` alone would not protect v1 absent-field behavior after a future default flip | Added schema-specific pre-validation materialization and a non-mutation regression |
| Explicit v1 migration could leave behavior-affecting fields absent | Migration now materializes incomplete explicit-v1 as well as versionless documents and is idempotent |
| WB confidence omitted neutral availability | Split exposure/WB confidence and added neutral availability to the WB score and artifacts |
| Evidence substituted microbenchmarks for existing runtime stats | Added a real synthetic/null production-pipeline EWMA snapshot through `FrameHub.stats_dict` |
| Diagnostics helper was tested but startup emitted no schema/visual policy | Added a sanitized startup visual-policy record and lifecycle test |
| The evidence helper was reviewed but absent from Python sdist selection | Added it explicitly to `MANIFEST.in`; rebuilt wheel/sdist successfully |
| Evidence report retained completed allow-list and constants items as caveats | Replaced them with closed reconciliation records and updated hashes |
| “Ratified highlight policy” had no decision | Defined hard clip as the Phase 0 fallback and left measured roll-off to VIS-2.1 |
| ADR wording blurred the 0.85 default clamp and 1.0 schema ceiling | Documented both roles explicitly |
| Contact-sheet labels depended on Pillow's version-specific default font | Replaced them with the embedded deterministic bitmap font and regenerated the reviewed PNG |
| Rejected-patch tests proved config/version atomicity but not the resource-preparation boundary directly | Added a live-pipeline regression that makes any preparation call fail and proves invalid/restart-only visual patches never reach it |
| Test rollback fixture produced a Pyright argument error | Added an explicit test-only type cast |

## Verification record

- Full Python suite: `1004 passed, 3 skipped`. The restricted sandbox denied
  five loopback socket tests; the exact five passed when rerun with loopback
  permission.
- Node/package suite: `141 passed`, `2` expected TODOs, `0 failed`. The TODOs
  are the pre-existing fail-closed `REL-01` and `WIN-01` publication blockers.
- Ruff lint and format: clean.
- Pyright: `0 errors, 0 warnings`.
- `git diff --check`: clean.
- Deterministic evidence and committed artifacts regenerate byte-for-byte.
  The internal evidence fingerprint is
  `1871e8ea3c00304fd86a62d240831cb8548f99de27c53732ba20c6c7be73a5f8`;
  JSON SHA-256 is
  `e6ccca5dc8d29b61f60cba5576263864b64823ad3a6f12241c8430a733695c07`;
  PNG SHA-256 is
  `fbcc4b9c713e393e5a08ce531b8c89d4f1b4da5311412ec5e87ad134a7fc7f85`.
- The minimum-dependency lane (Python 3.10, NumPy 1.24.0, OpenCV 4.8.1.78,
  Pillow 10.0.0) produced equal JSON, equal contact-sheet pixels, and the same
  JSON fingerprint and PNG hash.
- A local no-isolation wheel/sdist build completed, with the evidence helper in
  the sdist. The npm release-check unit and exact reviewed-payload checks pass.

Ordinary publication remains intentionally blocked by the repository's
pre-existing `REL-01` and `WIN-01` remediation entries. Those are release
program blockers, not Phase 0 visual-consistency defects.

## Later-phase boundaries

The following are deliberately deferred, with owners already present in the
backlog:

- VIS-1.1–VIS-1.6 implement and integrate shared geometry, source orientation,
  canonical canvas enforcement, and native Windows constraints.
- VIS-2.1–VIS-2.5 implement color/profile primitives, linear compositing, the
  selected estimator, temporal behavior, and transactional pixel integration.
- VIS-3.1–VIS-3.4 add runtime telemetry/UI and qualified metadata/hardware
  behavior.
- VIS-4.1–VIS-4.3 provide end-to-end privacy/temporal regressions,
  cross-platform live-camera qualification, and reversible default rollout.

These are implementation dependencies, not caveats in the completed Phase 0
contract, evidence, algorithm decision, or configuration surface.
