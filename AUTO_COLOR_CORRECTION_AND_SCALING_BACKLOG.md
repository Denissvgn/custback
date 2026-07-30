# Automatic color correction and consistent scaling — implementation backlog

Status: **accepted; Phases 0–3 complete; Phase 4 engineering complete,
release qualification open** · Owner: **TBD** · Written: **2026-07-30**
Code baseline: `master` at `f5753484b182`

This document converts a repository review into an implementation-ordered
backlog for two related output-quality problems:

1. camera and backdrop frames currently follow different scaling rules; and
2. foreground and backdrop pixels are composited without an explicit color
   contract or automatic harmonization.

The plan is intentionally implementation-specific. Each task records why it
exists, where it fits, likely files, dependencies, and executable acceptance
criteria. Line anchors were accurate at authoring time; re-check them before
starting a task.

The scope is the core/stage-1 camera pipeline. The avatar renderer consumes the
core's normalized raw-frame stream and therefore needs geometry regression
coverage, but avatar-specific color grading is not part of this backlog.

Phase 0 closed on 2026-07-30. Its normative decisions and evidence are in
`docs/adr/0001-visual-consistency-contract.md`,
`docs/visual-consistency-phase0-evidence.md`, and
`docs/visual-consistency-phase0-implementation-review.md`. Phase 1 closed on
2026-07-30; its implementation and verification record is in
`docs/visual-consistency-phase1-implementation-review.md`. The schema-v1
camera default remains `stretch`. Phase 2 closed on 2026-07-30; its
implementation, independent audit, performance observation, and verification
record are in `docs/visual-consistency-phase2-implementation-review.md`.
Phase 3 closed on 2026-07-30 after an independent gap/overclaim audit, full
host and minimum-dependency suites, and installed-artifact verification. Its
closure record is `docs/visual-consistency-phase3-implementation-review.md`;
the source-color and hardware-control decisions are in
`docs/visual-consistency-phase3-video-color-qualification.md` and
`docs/camera-control-characterization.md`.
Phase 4 engineering completed on 2026-07-30 after integrated regression,
performance, evidence-integrity, packaging, and rollout audits. Its
implementation record is
`docs/visual-consistency-phase4-implementation-review.md`. Deterministic and
local calibrated qualification pass, but Phase 4 is not release-closed:
pinned-runner evidence and the real Linux/macOS/Windows camera/consumer matrix
remain mandatory, and no target default has been activated.
Schema-v1 also retains `srgb_legacy` blending and correction `off`; pinned
platform qualification and the default-changing rollout remain explicitly
gated by Phase 4.

---

## Executive result

The project has a strong exact-shape pipeline once a frame reaches
segmentation, but the shape is obtained inconsistently:

- a mismatched main-camera frame is resized directly to the requested width and
  height, independently scaling X and Y (`src/custback/capture.py:526-533`);
- image, video, live-camera, and blur backdrops use aspect-fill plus a centered
  crop (`src/custback/backgrounds.py:66-81`, `205-208`, `758-790`,
  `855-861`, `916-919`).

A 640×480 camera delivered into the default 1280×720 canvas is therefore
stretched from 4:3 to 16:9, while a 4:3 backdrop is proportionally enlarged and
cropped. The subject and scene visibly obey different geometry.

The color path has a parallel inconsistency:

- sources are treated only as untagged `uint8 BGR`;
- still-image ICC profiles are verified by Pillow but ignored when OpenCV
  decodes the pixels;
- camera/video range, transfer, and matrix provenance are not retained;
- there is no exposure, white-balance, or foreground/background color matching;
- soft-edge blending and light wrap happen directly in gamma-encoded 0–255
  values.

The recommended solution is not one large compositor patch. It is two bounded
components with explicit contracts:

1. **Geometry normalization** creates one canonical, aspect-preserving canvas
   before segmentation and raw-frame publication.
2. **Color harmonization** estimates a conservative, mask-aware foreground
   exposure/illuminant transform after segmentation and after the backdrop is
   fitted, then applies it consistently to the camera foreground and RVM clean
   foreground during linear-light compositing.

This order preserves segmentation quality, mask alignment, the remote privacy
boundary, and the existing latest-frame capture architecture.

---

## Repository evidence

| Finding | Evidence | Consequence |
| --- | --- | --- |
| Acquisition size and output size are implicitly the same setting | `CameraConfig.width/height` are at `src/custback/config.py:173-182`; `OutputConfig` has no dimensions at `377-386`; output opens with camera dimensions at `src/custback/pipeline.py:917-943` | Device negotiation, processing canvas, API frame shape, and vcam size cannot be reasoned about separately |
| Camera mismatch handling distorts aspect ratio | Negotiation/readback is at `src/custback/capture.py:274-295`, `333-471`; direct resize is at `526-533` | A 4:3 fallback camera mode is stretched to 16:9 |
| Backdrops preserve aspect ratio but always center-crop | `_fit` is at `src/custback/backgrounds.py:66-81` | Background geometry differs from foreground geometry, and there is no focal-point control |
| Resize quality is implicit | Camera resize and `_fit` omit interpolation; OpenCV therefore uses its default | Downscaling does not deliberately use area resampling; behavior is not frozen by tests |
| Orientation semantics are incomplete | Main capture only mirrors horizontally after resize at `src/custback/capture.py:526-533`; image verification and OpenCV decode are split at `src/custback/backgrounds.py:148-202` | EXIF/container rotation and mirror order are undefined and can vary by backend |
| Capture validates too little | The reader accepts an ndarray with `ndim >= 2` at `src/custback/capture.py:501-513`, despite the module's BGR contract at `1-5` | Gray, BGRA, float, or malformed frames can fail later and unpredictably |
| Live backdrop cameras have a weaker contract than the main camera | `CameraBackdrop` opens without width/FPS/format negotiation at `src/custback/backgrounds.py:824-853` and only fits each returned frame at `855-861` | Two similar cameras can be decoded, timed, and framed differently |
| No color correction exists | `CameraConfig` has no color policy at `src/custback/config.py:173-182`; repository search finds no WB/exposure/gamma/harmonization stage | Camera auto-WB/exposure and backdrop color remain unrelated |
| Still-image profiles are ignored | Pillow verifies/decompresses at `src/custback/backgrounds.py:148-172`; pixels are separately loaded with `cv2.imread` at `193-202` | Tagged P3/Adobe-RGB/CMYK assets can have incorrect saturation or hue |
| Video color metadata is opaque | `VideoBackdrop` reads only dimensions/FPS/count at `src/custback/backgrounds.py:225-292` and accepts any bounded 3-channel `uint8` frame at `306-331` | BT.601/709, range, and transfer behavior depends on the OpenCV backend and is not observable |
| The compositor blends encoded values | `src/custback/compositor.py:55-69` casts 0–255 BGR to float, blends, then casts back | Soft edges and light wrap are photometrically wrong and can appear dark |
| A gamma-space result is frozen in tests | `tests/test_processing.py:137-140` expects 50% of encoded 200 over black to be 100 | Moving to linear light requires an intentional compatibility change and golden updates |
| Segmentation assumes display-referred input | MediaPipe labels converted input as sRGB at `src/custback/segmentation.py:511-515`; RVM consumes divided RGB at `654-660` | Harmonization should initially run after segmentation, not before it |
| The integration point already exists | Mask and backdrop acquisition are at `src/custback/pipeline.py:1792-1817`; edge foreground and compositing are at `1818-1830` | Color estimation can be inserted without changing capture or model inputs |
| Raw/remote privacy boundaries are explicit | Raw publish is at `src/custback/pipeline.py:1895-1900`; remote candidates bypass local compositing at `1923-1948`; wrong-size remote output is rejected | Corrected pixels must not replace raw publication or silently resize remote output |
| Hot activation protects live state | `_Resources`/`_Activation` are at `src/custback/pipeline.py:316-347`; trial activation promises not to touch working state at `1045-1055` | A temporal harmonizer must participate in staged activation or be cloned/reset transactionally |
| Geometry telemetry is incomplete | Capture status exists at `src/custback/hub.py:22-84` and API schema at `src/custback/api/server.py:324-377` | Operators cannot see “640×480 source → cover crop → 1280×720 output” |
| Windows native output has another mismatch | MF advertises a small fixed mode set in `packaging/windows/vcam/MediaSource.h:23-34`; `ComposeFrame` overlap-copies rather than scales at `packaging/windows/vcam/MediaStream.cpp:275-306` | Smaller input becomes black-bordered and larger input is cropped, despite a valid Python ring |
| Tests mostly prove shape, not geometry | `tests/test_processing.py:192-200` uses a uniform square; `tests/test_capture.py:388-405` asserts shape/logs only | Distortion, crop coordinates, orientation, and interpolation regressions can pass |

---

## Target architecture

```text
OpenCV camera frame
        │
        ▼
validate BGR ── resolve rotation/mirror ── geometry normalize
        │                                  to canonical canvas
        ├──────────────────────────────► publish raw / remote-avatar input
        │
        ▼
segment original normalized frame ──► mask + optional RVM clean foreground
        │
        │                         backdrop decode
        │                    (EXIF/ICC normalization)
        │                              │
        │                              ▼
        │                     geometry normalize
        │                     to canonical canvas
        │                              │
        └──────────────────────────────┤
                                       ▼
                          color-transform estimator
                          (small mask-aware samples)
                                       │
                                       ▼
                     temporally filtered, bounded transform
                                       │
                                       ▼
               linear-light edge cleanup / light wrap / alpha composite
                                       │
                                       ▼
                       encode contiguous uint8 BGR once
                                       │
                                       ▼
                 output validation / privacy guard / all output sinks
```

### Boundary contract

The recommended contract to ratify in VIS-0.1 is:

- **External frame representation:** non-empty, contiguous `HxWx3 uint8 BGR`.
- **External color assumption:** full-range display-referred sRGB/BT.709
  primaries unless a source is explicitly color-managed. This is an assumption
  for OpenCV camera/video frames, not a claim that every backend exposes
  trustworthy metadata.
- **Internal photometric representation:** finite linear-sRGB `float32` in a
  documented bounded range.
- **Canonical geometry:** every capture, mask, fitted backdrop, preview/API
  frame, and vcam frame uses one resolved output width and height.
- **Untrusted remote output:** must already equal the canonical geometry; it is
  rejected, never auto-fitted.

### Recommended configuration shape

Names may be adjusted in VIS-0.1, but the semantics should remain explicit.
The values below are the **qualified steady-state target**, not the defaults
for the first compatibility-preserving implementation:

```yaml
camera:
  width: 1280             # requested acquisition mode
  height: 720
  fit_mode: cover         # target; initial model default is stretch
  anchor_x: 0.5           # 0 = left, 1 = right
  anchor_y: 0.5           # 0 = top, 1 = bottom
  rotation: 0             # 0 | 90 | 180 | 270
  mirror: false

background:
  fit_mode: cover
  anchor_x: 0.5
  anchor_y: 0.5

output:
  width: null             # both omitted => camera width/height
  height: null

compositing:
  blend_space: linear_srgb  # target; initial model default is srgb_legacy
  color_correction:
    mode: auto              # target; initial model default is off
    strength: 0.5
    exposure_limit_ev: 0.85
    white_balance_strength: 0.5
    adaptation_time_s: 0.8
```

The first implementation release uses `stretch`, `srgb_legacy`, and `off` when
the new fields are absent. VIS-0.1 selected explicit persisted schema
versioning: versionless documents are permanently schema v1, migrations
materialize behavior-affecting fields, and later default flips require a
schema bump. The project does not infer installation age from field absence.

Only `mode` and `strength` need prominent UI controls. The remaining bounded
fields are advanced YAML/API settings unless user research shows a need.

---

## Required behavior by mode

| Mode | Camera geometry | Backdrop geometry | Automatic foreground correction | Notes |
| --- | --- | --- | --- | --- |
| `passthrough` | Yes | N/A | No; byte identity after geometry normalization | Preserve the existing “original room” semantic |
| `blur` | Yes | Same normalized camera frame | No | Foreground and blurred scene share one illuminant |
| `color` | Yes | Exact canvas | Identity by default; optional exposure-only for neutral colors | Never derive WB from a saturated solid |
| `image` | Yes | Configured fit/anchor | Exposure + restrained WB when confident | ICC-normalize before matching |
| `video` | Yes | Configured fit/anchor | Exposure + restrained WB when confident | Handle cuts and opaque decoder color metadata |
| `camera` | Yes | Configured fit/anchor | Exposure + restrained WB when confident | Two independent auto-control loops require slow, bounded adaptation |
| `remote` | Yes for raw input | Renderer-owned | No in the core | Preserve exact-size validation and privacy-slate behavior |

---

## Product and engineering decisions ratified in Phase 0

The Phase 0 ADR records the compatibility landing state, target state,
migration mechanism, and rationale for each decision below.

1. **Separate acquisition and canvas dimensions.** Keep
   `camera.width/height` as the requested device mode. Add optional paired
   `output.width/height`; omission resolves to the camera request for backward
   configuration compatibility.
2. **Target proportional camera fitting to `cover`.** It matches current
   backdrop behavior and prevents distortion. Land `stretch` as the initial
   compatibility default; later make `cover` the default only through the
   ratified migration/default-flip policy. `stretch` remains an explicit legacy
   option. This changes framing for users whose camera silently falls back to
   another aspect ratio and therefore needs release notes.
3. **Keep current backdrop default framing.** `cover` with center anchor is the
   current effective behavior. Add anchors so portrait/square assets need not
   always lose their center-independent subject.
4. **Use deterministic orientation.** Apply metadata/manual rotation, then
   horizontal mirror in final viewer coordinates, then fit. Never guess webcam
   rotation from dimensions.
5. **Use software correction, not continuous hardware control.** OpenCV
   WB/exposure semantics differ among V4L2, MSMF, and DSHOW. Hardware control is
   a separately qualified follow-up.
6. **Correct after segmentation.** This preserves MediaPipe/RVM input
   distribution and ensures the fitted backdrop drives the estimate.
7. **Keep raw and remote frames uncorrected.** Geometry normalization affects
   the canonical raw frame; color harmonization affects only local composite
   rendering.
8. **Prefer conservative exposure + illuminant adaptation.** Do not use full
   LAB mean/std or histogram transfer as the default; those approaches can
   recolor faces and clothing to resemble arbitrary scene content.
9. **Make temporal state generation-owned.** Reset or replace it on backdrop,
   geometry, correction-policy, frame-size, or camera-generation changes. A
   failed hot activation must leave the old estimator state untouched.
10. **Treat linear-light compositing as an intentional visual migration.**
    Land `srgb_legacy` as the initial compatibility default, preserve the switch
    during qualification, then make the final default decision from golden and
    user-visible results using the same explicit migration policy.

---

## Cross-cutting rules

These rules gate every task below.

- **VC-1 — Preserve the privacy boundary.** Raw publication, raw fingerprints,
  remote renderer input/output, wrong-size rejection, and the input-independent
  privacy slate must not be color-corrected or helpfully resized.
- **VC-2 — Preserve frame ownership and latest-only behavior.** Do not retain
  full frame history. Geometry plans and color parameters may be cached;
  unbounded pixel buffers may not.
- **VC-3 — No live-state mutation in candidate trials.** A hot PATCH that fails
  construction/trial/commit must not advance video playback, temporal color
  state, mask state, or the effective config version.
- **VC-4 — One geometry implementation.** Capture, image, video, live backdrop,
  blur, preview/API, and supported outputs must not carry private variations of
  fit math.
- **VC-5 — Direction-aware resampling.** Use no operation for exact size,
  `INTER_AREA` for downscale, and `INTER_LINEAR` for upscale unless a measured,
  documented replacement is selected.
- **VC-6 — No color inference from histogram range alone.** A 16–235-looking
  histogram is not proof of limited-range video. Use trustworthy metadata,
  a qualified backend contract, or an explicit override.
- **VC-7 — Correction failure is locally fail-soft.** An instantaneous
  low-confidence estimate does not update parameters: with no prior reliable
  estimate the transform is identity; otherwise VIS-2.4 briefly freezes the
  last bounded transform and then decays it toward identity. An estimator
  exception resets/decays according to the ratified error policy and emits a
  bounded diagnostic. Neither case may emit malformed pixels or terminate an
  otherwise valid local camera loop.
- **VC-8 — Update release allow-lists with new code/tests.**
  `scripts/release/verify-release.js` has exact reviewed Python module, test,
  dependency, and npm-payload lists. Any new `geometry.py`, `color.py`, test, or
  shipped document must update those lists and packaging evidence in the same
  change.
- **VC-9 — Measure before changing defaults.** Each visual-default flip is a
  separate, reversible commit after golden, performance, platform, and privacy
  gates pass.
- **VC-10 — Keep the MVP dependency-neutral.** NumPy, OpenCV, and Pillow already
  provide the required geometry, linear-color, and ICC primitives. A new video
  decoder or color-science dependency belongs to a separately qualified
  follow-up with packaging/licensing evidence.

---

## Task index and implementation order

Size is relative engineering scope, not calendar time.

| ID | Task | Phase | Priority | Size | Dependencies | Status |
| --- | --- | --- | --- | --- | --- | --- |
| VIS-0.1 | Ratify frame/color boundaries and experiment rules | 0 | P0 | M | — | DONE |
| VIS-0.2 | Build visual fixtures, metrics, and current-baseline report | 0 | P0 | M | VIS-0.1 | DONE |
| VIS-0.3 | Select the bounded harmonization algorithm | 0 | P0 | M | VIS-0.2 | DONE |
| VIS-0.4 | Add strict geometry/color configuration models | 0 | P0 | M | VIS-0.1, VIS-0.3 | DONE |
| VIS-1.1 | Implement the shared geometry planner/transform | 1 | P0 | L | VIS-0.1 | DONE |
| VIS-1.2 | Normalize orientation and source-frame validity | 1 | P0 | M | VIS-1.1 | DONE |
| VIS-1.3 | Integrate geometry into main camera capture | 1 | P0 | L | VIS-0.4, VIS-1.1, VIS-1.2 | DONE |
| VIS-1.4 | Integrate geometry into every backdrop provider | 1 | P0 | L | VIS-0.4, VIS-1.1, VIS-1.2 | DONE |
| VIS-1.5 | Enforce one canonical canvas across pipeline/API/outputs | 1 | P0 | L | VIS-1.3, VIS-1.4 | DONE |
| VIS-1.6 | Repair or constrain Windows native output scaling | 1 | P0 | L | VIS-1.5 | DONE |
| VIS-2.1 | Implement tested sRGB/linear and image-profile primitives | 2 | P0 | L | VIS-0.1, VIS-0.2 | DONE |
| VIS-2.2 | Move compositor and light wrap to linear light | 2 | P0 | L | VIS-2.1 | DONE |
| VIS-2.3 | Implement the mask-aware color estimator | 2 | P0 | L | VIS-0.3, VIS-2.1 | DONE |
| VIS-2.4 | Add temporal adaptation, confidence, and reset behavior | 2 | P0 | L | VIS-2.3 | DONE |
| VIS-2.5 | Integrate harmonization as transactional pipeline state | 2 | P0 | L | VIS-1.5, VIS-2.2, VIS-2.4 | DONE |
| VIS-3.1 | Add geometry/color telemetry and transition logging | 3 | P1 | M | VIS-1.5, VIS-2.5 | DONE |
| VIS-3.2 | Add Web UI controls and operator diagnostics | 3 | P1 | M | VIS-0.4, VIS-3.1 | DONE |
| VIS-3.3 | Normalize video/output color metadata where provable | 3 | P2 | L | VIS-2.1 | DONE |
| VIS-3.4 | Characterize optional camera hardware controls | 3 | P2 | M | VIS-3.1 | DONE |
| VIS-4.1 | Add end-to-end, privacy, and temporal regressions | 4 | P0 | L | VIS-1.6, VIS-2.5 | DONE |
| VIS-4.2 | Run performance and cross-platform visual qualification | 4 | P0 | L | VIS-3.1, VIS-4.1 | IMPLEMENTED; PINNED/PHYSICAL EVIDENCE PENDING |
| VIS-4.3 | Document, package, and stage the default rollout | 4 | P0 | M | VIS-3.2, VIS-4.2 | DONE; STAGED/DEFAULTS GATED |

**Critical path:** VIS-0.1 → VIS-0.2 → VIS-0.3 → VIS-0.4 →
VIS-1.1/1.2 → VIS-1.3/1.4 → VIS-1.5 → VIS-1.6, in parallel with
VIS-2.1/2.3 → VIS-2.2/2.4 → VIS-2.5; both branches join at VIS-4.1 →
VIS-4.2 → VIS-4.3.

After VIS-0.4, camera and backdrop geometry work can proceed in parallel.
Color primitives/linear compositing can proceed beside geometry integration,
but pipeline harmonization must wait until the canonical canvas is stable.
VIS-3.3 and VIS-3.4 are post-MVP P2 follow-ups; normally pull them only after
VIS-4.2, unless their investigation is required to make VIS-4.2 pass.

---

## Phase 0 — Contract, baseline, and configuration

### VIS-0.1 — Ratify frame/color boundaries and experiment rules · M

**Goal:** remove implicit assumptions before code establishes another
incompatible convention.

**Context:** camera dimensions currently mean acquisition request, processing
shape, remote frame shape, and output shape at once. Color is only called BGR,
with no range/transfer/profile semantics. Defaults for `cover`, linear blending,
and `auto` correction have user-visible consequences.

**Work:**

- Create a short ADR or contract document covering:
  - acquisition dimensions versus resolved canvas dimensions;
  - `cover`/`contain`/`stretch` math and anchor coordinates;
  - orientation → mirror → fit order;
  - down/upscale interpolation;
  - external BGR and internal linear-sRGB assumptions;
  - provisional mode eligibility, correction bounds, and identity cases to be
    finalized by VIS-0.3;
  - temporal reset and scene-cut rules;
  - raw/remote exact-size and color invariants;
  - Windows native supported-mode behavior;
  - legacy-config and default-rollout decisions.
- Include worked geometry examples for 4:3→16:9, 16:9→4:3,
  portrait→landscape, odd dimensions, and anchors at 0/1.
- Record which fields are restart-only versus hot.
- Decide whether `blend_space` remains a temporary compatibility switch or a
  permanent advanced control.
- Choose an explicit default-migration mechanism: either later absent-field
  defaults change globally with pinning guidance, or persisted config/schema
  versioning distinguishes migrated installs. Record the initial compatible
  defaults as `stretch`, `srgb_legacy`, and correction `off`.

**Likely files:** new root-level contract/ADR, this backlog if decisions change.

**Acceptance:**

- Every source and sink has one representation, color assumption, and geometry
  owner.
- Boundary/privacy/geometry ownership decisions have a selected rule,
  rationale, migration impact, and revisit trigger.
- Algorithm-dependent clamps, eligibility, and target defaults are clearly
  labeled provisional, have an experiment/gate in VIS-0.3, and cannot become
  model defaults before that task amends the ADR.
- The ADR explicitly prohibits histogram-only full/limited-range detection and
  remote-frame auto-fitting.

**Dependencies:** none.

### VIS-0.2 — Build visual fixtures, metrics, and current-baseline report · M

**Goal:** make “more consistent” measurable before changing pixels.

**Context:** existing processing tests use uniform arrays and mostly assert
shape. Those cases cannot reveal geometric distortion, dark alpha edges, skin
recoloring, temporal pumping, or source-profile errors.

**Work:**

- Add deterministic geometry fixtures:
  - labeled corners and asymmetric coordinate grid;
  - circle/face-outline target;
  - square, 4:3, 16:9, portrait, ultrawide, and odd-sized frames;
  - EXIF orientations 1–8 and a rotation-metadata video where tooling permits.
- Add deterministic color fixtures:
  - neutral patches under warm/cool casts;
  - ±1 EV exposure pairs;
  - clipped highlights and deep shadows;
  - skin-like and saturated clothing patches;
  - neutral and saturated solid backdrops;
  - tagged sRGB/P3/Adobe-RGB/CMYK images;
  - static-noise, slow drift, hard scene cut, and reconnect sequences.
- Prefer generated fixtures. If binary fixtures are required, record generator,
  license/provenance, dimensions, and SHA-256; add them to release manifests.
- Implement metrics for:
  - retained aspect ratio/crop coordinates;
  - foreground/backdrop log-luminance gap;
  - neutral-axis chroma error;
  - foreground hue/chroma drift;
  - per-frame EV/gain jitter and scene-cut settling;
  - edge luminance around soft alpha.
- Produce a baseline artifact/contact sheet using current code and record
  processing timings from existing stats.

**Likely files:** new `tests/test_visual_consistency.py` or focused geometry/color
test modules, reviewed fixtures, a non-production comparison script if needed,
`scripts/release/verify-release.js`.

**Acceptance:**

- The current 4:3 camera → 16:9 stretch fails a distortion assertion.
- The current encoded 50% blend is captured as 100 and the intended
  linear-light reference as approximately 146.
- Re-running the suite produces identical numeric results within documented
  platform tolerances.
- No test depends only on a human saying that two frames “look better.”

**Dependencies:** VIS-0.1.

### VIS-0.3 — Select the bounded harmonization algorithm · M

**Goal:** choose a safe estimator from evidence rather than embedding an
aggressive color-transfer recipe in production.

**Context:** full histogram or LAB mean/std matching can make skin and clothing
take on arbitrary backdrop colors. Live backdrop cameras also run their own
auto-WB/exposure loops, so an instant per-frame correction can oscillate.

**Work:**

- Compare at least:
  1. exposure-only matching;
  2. bounded log-luminance exposure plus low-strength diagonal
     white-balance/von-Kries adaptation in linear RGB;
  3. a deliberately aggressive moment-transfer baseline to demonstrate why it
     is or is not acceptable.
- Analyze a downscaled copy only (provisional 160–256 pixels on the long edge).
- Compare source sampling from an eroded high-confidence foreground core.
- Compare target sampling from the fitted backdrop near/behind the subject,
  with a robust global-backdrop fallback.
- Exclude clipped, near-black, and high-saturation samples from WB estimation.
- Define confidence from usable sample count, mask coverage, clipping, and
  neutral-sample availability.
- Ratify provisional clamps:
  - exposure no wider than approximately ±0.75–1.0 EV;
  - channel gains approximately 0.85–1.18;
  - default strength approximately 0.4–0.7;
  - a low-confidence instantaneous estimate is identity/no-update; the
    stateful policy later freezes and decays as specified in VIS-2.4.
- Define skin/clothing preservation thresholds and solid-color behavior.
- Amend the VIS-0.1 ADR with the selected estimator, final clamps,
  mode/low-confidence semantics, and target-default recommendation.

**Likely files:** experimental test/helper code only; final design recorded in
the VIS-0.1 contract.

**Acceptance:**

- The chosen method materially reduces the synthetic luminance/neutral-cast
  error without violating the ratified foreground hue/chroma bound.
- Saturated backdrops, tiny/all-zero/all-one masks, clipping, and insufficient
  neutral pixels select identity or exposure-only behavior.
- The selected estimator can be implemented with bounded per-frame time and
  constant memory.
- The amended ADR contains no provisional algorithm decision required by
  VIS-0.4.

**Dependencies:** VIS-0.2.

### VIS-0.4 — Add strict geometry/color configuration models · M

**Goal:** establish validated settings and restart semantics before integration.

**Context:** configuration is strict Pydantic, RFC 7396 patches are
transactional, and all `camera.*`/`output.*` changes are currently restart-only
at `src/custback/pipeline.py:426-437`. Public state serializes through an
allow-list at `src/custback/api/server.py:205-215`, `285-307`.

**Work:**

- Add reusable strict geometry types/literals and a nested color-correction
  model.
- Add paired optional `output.width/height`; reject only-one-set.
- Add `resolved_output_size(config)` as the only canvas-resolution helper.
- Validate right-angle rotation, anchors `[0,1]`, finite correction numbers,
  mode enums, time constants, strengths, and exposure/WB clamps.
- Keep camera/output canvas and camera transform fields restart-only.
- Make backdrop fit/anchor and color-correction fields hot-patchable.
- Update `_backdrop_key` carefully:
  - asset/source identity should still determine provider replacement;
  - pure fit/anchor changes should not reopen/reset a video;
  - a separate visual-state key should reset fitted caches and harmonizer state.
- Update `config/default.yaml`, config round trips, defaults/invalid-values
  tests, merge-patch null/reset tests, public API models, sanitized diagnostic
  string allow-list, and migration/export fixtures as required.
- Add the new background fields explicitly to `PublicBackgroundConfig` at
  `src/custback/api/server.py:285-297`; unlike the embedded camera/compositing
  models, this is an explicit allow-list and would otherwise silently omit
  them.
- Do not add MVP CLI flags. Comparable quality controls already use YAML/API/UI,
  and the current CLI intentionally exposes a smaller operational surface.
- Land the model defaults as `camera.fit_mode: stretch`,
  `compositing.blend_space: srgb_legacy`, and correction `mode: off`. Do not
  flip any of them in this task; VIS-4.3 owns qualified, reversible flips or
  schema migration.

**Likely files:** `src/custback/config.py`, `src/custback/pipeline.py`,
`src/custback/api/server.py`, `src/custback/diagnostics.py`,
`src/custback/__main__.py`, `config/default.yaml`, `tests/test_config.py`,
`tests/test_pipeline.py`, `tests/test_api.py`, migration fixtures/tests.

**Acceptance:**

- Old YAML without new fields loads as `stretch`/`srgb_legacy`/`off` unless the
  ADR selected and implemented an explicit versioned migration.
- Invalid combinations fail before opening a camera/file/output.
- PATCH of a restart-only field returns the existing `409 restart_required`
  shape without changing version/resources.
- Hot correction/background-geometry PATCH is atomic and versioned.
- Public config exposes safe policy values but no new source path/device data.

**Dependencies:** VIS-0.1, VIS-0.3.

---

## Phase 1 — Canonical geometry

### VIS-1.1 — Implement the shared geometry planner/transform · L

**Goal:** replace private resize/crop behavior with one tested, source-agnostic
geometry implementation.

**Context:** `_fit` currently resizes then center-crops, while capture stretches.
Both rely on implicit interpolation and neither exposes the applied crop/scale.

**Work:**

- Add a focused module such as `src/custback/geometry.py`.
- Define immutable `GeometrySpec` and `TransformPlan` values.
- Separate pure planning from pixel work:
  `plan_transform(source_size, target_size, rotation, mirror, fit, anchors)`.
- Support:
  - `cover`: uniform scale and anchored crop;
  - `contain`: uniform scale and deterministic opaque padding;
  - `stretch`: independent axes only when explicitly requested;
  - rotations 0/90/180/270 and documented horizontal mirror;
  - exact no-op and contiguous-output handling.
- Calculate cover sizes/crops with safe rounding (`ceil` plus clamped bounds)
  so a slice is never one pixel short.
- Select `INTER_AREA` for true downscale and `INTER_LINEAR` for true upscale.
  Define mixed-axis behavior for `stretch`.
- Return transform telemetry: source/oriented/target size, scales, crop rectangle,
  pad rectangle, fit, rotation, and mirror.
- Cache plans by immutable geometry key; do not cache live frames in this module.
- Keep the old `_fit` temporarily as a compatibility wrapper if needed, then
  delete it after all call sites migrate.

**Likely files:** new `src/custback/geometry.py`,
`tests/test_geometry.py`, `scripts/release/verify-release.js`.

**Acceptance:**

- Labeled-corner/grid tests prove exact crop/pad/anchor/orientation behavior.
- A circle remains circular under `cover` and `contain`.
- Odd sizes and anchors 0/1 produce exact target dimensions.
- Invalid dtype, channel count, empty dimensions, NaN settings, and unsupported
  rotation fail with deterministic errors.
- Tests spy on interpolation selection without relying on brittle exact
  OpenCV pixel values.
- Result is `HxWx3 uint8`, finite by construction, and C-contiguous.

**Dependencies:** VIS-0.1.

### VIS-1.2 — Normalize orientation and source-frame validity · M

**Goal:** ensure planning receives a deterministic oriented BGR source.

**Context:** Pillow validates still assets but OpenCV independently decodes them;
webcam/video rotation is implicit, and capture accepts malformed arrays.
Mirrored EXIF orientations need more than a simple 90-degree rotation.

**Work:**

- Define one explicit source order:
  validate/decode → metadata/manual orientation → mirror in viewer coordinates
  → geometry fit.
- Static images:
  - apply every EXIF orientation, including mirrored forms;
  - validate dimensions/pixel limits against the authoritative decoded image;
  - avoid “Pillow validates one orientation, OpenCV produces another.”
- Video:
  - discover backend rotation metadata/auto-rotation support;
  - disable implicit auto-rotation where possible and apply it exactly once;
  - record a deterministic manual-only or unsupported fallback when metadata
    and backend auto-rotation behavior cannot be proven; do not infer rotation
    from pixels or dimensions.
- Cameras:
  - support explicit right-angle rotation;
  - do not infer portrait rotation from aspect ratio.
- Add one shared strict frame validator for positive `HxWx3 uint8`.
- Normalize contiguous layout after transpose/flip without unnecessary copies.

**Likely files:** `src/custback/geometry.py`,
`src/custback/backgrounds.py`, `src/custback/capture.py`,
`tests/test_geometry.py`, `tests/test_processing.py`,
`tests/test_capture.py`.

**Acceptance:**

- EXIF orientations 1–8 and labeled manual rotations produce the documented
  corner positions.
- Rotation then horizontal mirror is frozen by a non-symmetric test.
- Qualified backends and mocks prove metadata rotation is applied exactly once,
  including ignored-control cases that remain observable.
- Backends whose metadata/auto-rotation behavior is opaque are reported as
  ambiguous and follow the documented manual-only/unsupported policy; the
  product does not claim automatic orientation correctness for them.
- Gray, BGRA, float, zero-sized, or malformed camera/background frames are
  rejected at their source boundary.

**Dependencies:** VIS-1.1.

### VIS-1.3 — Integrate geometry into main camera capture · L

**Goal:** normalize the camera into the canonical canvas before segmentation.

**Context:** this is the lowest-risk location because masks and RVM clean
foreground are then generated at the final geometry and never need separate
resampling.

**Work:**

- Resolve the output canvas once at capture/resource construction.
- Replace `cv2.resize(frame, requested_size)` at
  `src/custback/capture.py:526-533` with the shared transform.
- Keep device negotiation based on `camera.width/height`, not output canvas.
- Preserve mode-mismatch `warn|error`, MJPEG retry, recovery, latest-only slot,
  mirror semantics, and measured capture FPS.
- Track property-reported, delivered, oriented, and normalized dimensions
  separately.
- Re-plan and emit one transition if delivered size changes during a generation
  or after reconnect; never reuse a stale transform.
- Define whether a mid-generation size change is recoverable or fatal under
  `mode_mismatch: error`.
- Ensure transform work occurs once, in the capture reader, before slot publish.

**Likely files:** `src/custback/capture.py`, `src/custback/pipeline.py`,
`tests/test_capture.py`.

**Acceptance:**

- A 640×480 circle normalized to 1280×720 is not stretched and has the
  contract-defined crop.
- 1920×1080→1280×720 uses area downsampling; true upscaling uses linear.
- Property/delivered disagreement, dynamic size change, reconnect, portrait
  rotation, and mirror order are covered.
- Existing nonblocking read, dropped-frame, recovery-timeout, and shutdown
  tests remain green.
- The capture slot always contains the resolved canonical shape and valid BGR.

**Dependencies:** VIS-0.4, VIS-1.1, VIS-1.2.

### VIS-1.4 — Integrate geometry into every backdrop provider · L

**Goal:** make the displayed scene use the same fit semantics and canvas
contract without breaking provider clocks/caches.

**Context:** static, video, camera, and blur providers currently call `_fit`.
Video maintains wall-clock phase and multiple caches, so a geometry-only change
must not reconstruct the provider or reset playback.

**Work:**

- Route `ImageBackdrop`, `VideoBackdrop`, `CameraBackdrop`, and `BlurBackdrop`
  through the shared planner.
- Keep `ColorBackdrop` as exact canvas allocation.
- Key fitted-pixel caches by raw frame identity/size, resolved orientation,
  target size, fit, anchors, and pad policy.
- Key streaming plan caches similarly, but retain only the latest raw/fitted
  frames already required by the provider.
- Allow background fit/anchor changes to invalidate fit caches without reopening
  a video or camera and without resetting video timing/counters.
- Validate live-camera/video decoded frames before geometry work.
- Decide how a video resolution change is handled; preserve the existing
  quarantine/last-good-frame safety policy for oversized/invalid frames.
- Ensure blur does not accidentally resize an already canonical source twice.
- Trial a new geometry policy on detached/synthetic data so hot activation does
  not advance the working video or block on the working live backdrop.

**Likely files:** `src/custback/backgrounds.py`,
`src/custback/pipeline.py`, `tests/test_processing.py`,
`tests/test_pipeline.py`.

**Acceptance:**

- Actual crop coordinates—not only shape—are asserted for square, 4:3,
  portrait, ultrawide, and anchor-edge cases.
- Image/video/live-camera providers produce equivalent geometry from equivalent
  source pixels.
- A hot anchor/fit PATCH changes the next committed frame without resetting
  video phase or mutating the provider on a failed trial.
- Existing video timing, skip/reuse/seek, image bomb-limit, and camera-close
  tests remain green.

**Dependencies:** VIS-0.4, VIS-1.1, VIS-1.2.

### VIS-1.5 — Enforce one canonical canvas across pipeline/API/outputs · L

**Goal:** make the resolved output canvas an explicit invariant rather than
“whatever shape capture happened to return.”

**Context:** current output validation only requires processed output to equal
the current source frame at `src/custback/pipeline.py:1430-1442`. Remote decoder
and vcam construction derive expected dimensions from camera configuration.

**Work:**

- Store resolved canvas dimensions in `_Resources`.
- Open output with resolved output dimensions, not acquisition request.
- Assert capture/preflight/steady-state frames equal the canvas before raw
  publish or segmentation.
- Request/generate backdrops at the same canvas and keep mask validation exact.
- Update raw WebSocket/JPEG metadata and avatar source expectations to the
  resolved canvas.
- Preserve exact-size remote-renderer validation and the wrong-size privacy
  slate; never run remote output through geometry.
- Preserve repeated-frame behavior: repeats use the last already normalized,
  guarded output and do not re-transform.
- Add validation at every real output backend so `NullOutput` cannot hide a
  malformed frame contract.
- Report effective output dimensions/FPS from the output resource.

**Likely files:** `src/custback/pipeline.py`, `src/custback/hub.py`,
`src/custback/api/server.py`, `src/custback/vcam.py`,
`src/custback/vcam_native.py`, avatar source protocol/tests,
`tests/test_pipeline.py`, `tests/test_api.py`, `tests/test_vcam.py`.

**Acceptance:**

- Passthrough, all local modes, remote startup/fallback/success, preflight,
  repeat, preview, MJPEG, raw WebSocket, pyvirtualcam, and native writer share
  the exact resolved dimensions.
- A non-default acquisition/canvas pair works end to end.
- Wrong-size remote output still produces the privacy slate and the existing
  fallback reason.
- Raw fingerprints remain computed from the same normalized frame sent to the
  renderer.
- Backend tests verify constructor dimensions and every sent frame contract.

**Dependencies:** VIS-1.3, VIS-1.4.

### VIS-1.6 — Repair or constrain Windows native output scaling · L

**Goal:** prevent the native Media Foundation path from silently cropping or
letterboxing by overlap copy.

**Context:** Python can create a ring for broad configured dimensions, while
the media source advertises a small fixed mode set. `ComposeFrame` does not
scale when ring and consumer modes differ.

**Work:**

- Decide and document one of:
  1. advertise/propagate the resolved canvas plus supported consumer modes and
     implement real scaling; or
  2. explicitly reject unsupported canvas/FPS combinations until real scaling
     exists.
- If scaling is implemented:
  - use the same documented fit policy and direction-aware interpolation;
  - cache converted output by ring frame counter and negotiated media type so
    repeated MF samples do not rescale unchanged pixels;
  - version the ring metadata if fit/rotation policy must cross the process
    boundary;
  - propagate or truthfully constrain effective FPS.
- Add color metadata (primaries/transfer/nominal range) only after VIS-3.3
  proves the correct Media Foundation attributes.
- Until this task lands, fail fast for a mismatched explicit native
  configuration instead of producing a cropped/postage-stamp frame.

**Likely files:** `src/custback/vcam.py`, `src/custback/vcam_native.py`,
`packaging/windows/vcam/FrameRing.h`,
`MediaSource.cpp/.h`, `MediaStream.cpp/.h`,
`tests/test_windows_vcam.py`, Windows system-test evidence.

**Acceptance:**

- 640×480→720p, 1080p→720p, opposite aspect ratios, odd sizes, stride/BGRX,
  consumer renegotiation, and non-30-FPS policy are tested.
- No source/consumer mismatch is handled by undocumented overlap copy.
- Windows Camera plus the supported Teams/Zoom/browser matrix shows the same
  geometry as preview/API, or unsupported combinations fail clearly.

**Dependencies:** VIS-1.5.

---

## Phase 2 — Color normalization and harmonization

### VIS-2.1 — Implement tested sRGB/linear and image-profile primitives · L

**Goal:** create a small, explicit color foundation before estimating or
blending illumination.

**Context:** current code treats encoded BGR values as linear floats. Static
image profiles are ignored, and repeated encode/decode stages would waste time
and magnify rounding error.

**Work:**

- Add a focused module such as `src/custback/color.py`.
- Implement or wrap tested:
  - sRGB EOTF/OETF using LUTs where beneficial;
  - BGR↔linear-RGB conversions with unmistakable channel naming;
  - linear luminance and log-luminance;
  - robust masked median/quantile/statistic helpers;
  - bounded diagonal exposure/WB transform;
  - highlight roll-off/clipping and final rounded `uint8` conversion.
- Guarantee finite, clipped, contiguous output for adversarial values.
- Make still-image decode color-managed:
  - use the existing secure validation/pixel-limit flow;
  - apply EXIF orientation from VIS-1.2;
  - convert valid embedded ICC profiles to sRGB via Pillow `ImageCms`;
  - document the sRGB assumption for untagged assets;
  - define deterministic rejection or safe fallback for malformed profiles;
  - convert to BGR only at the provider boundary.
- Reuse the same orientation/profile normalization for background thumbnails
  so the library preview does not disagree with the selected full-size asset.
- Avoid decoding an untrusted path through a second library after validating a
  different file state; preserve existing storage/security invariants.

**Likely files:** new `src/custback/color.py`,
`src/custback/backgrounds.py`, `tests/test_color.py`,
`tests/test_processing.py`, release allow-lists.

**Acceptance:**

- Known sRGB transfer vectors match reference values; encode/decode round trip
  differs by at most one code value.
- Transfer functions are monotonic and preserve endpoints.
- Synthetic exposure and channel-gain transforms recover known values within
  ratified tolerance.
- Tagged P3/Adobe-RGB/CMYK and untagged sRGB fixtures produce deterministic
  sRGB BGR; malformed profiles cannot bypass size/validation rules.
- No NaN, Inf, negative-index wrap, or `uint8` modulo behavior is possible.

**Dependencies:** VIS-0.1, VIS-0.2.

### VIS-2.2 — Move compositor and light wrap to linear light · L

**Goal:** remove dark/incorrect soft edges and establish the photometric space
in which the foreground transform is applied.

**Context:** `composite()` currently blends encoded values and truncates on
`astype(np.uint8)`. A naïve separate color-correction function could decode and
encode the same full frame multiple times.

**Work:**

- Refactor the compositor so foreground, backdrop, and optional RVM
  `edge_foreground` are decoded once per frame.
- Apply clean-foreground replacement, optional foreground color transform,
  light wrap, and alpha blending in linear light.
- Encode/round once at the end.
- Preserve bit-exact foreground/background endpoints when mask is exactly 1/0.
- Decide and test whether `BlurBackdrop` also performs its Gaussian operation in
  linear light; do not accidentally change blur mode while assessing
  foreground/background matching.
- Retain a temporary `srgb_legacy` path if required by VIS-0.1 rollout, with
  byte-identity tests and no hidden automatic selection.
- Keep public shape/type errors as strict as today.

**Likely files:** `src/custback/compositor.py`,
`src/custback/backgrounds.py`, `tests/test_processing.py`.

**Acceptance:**

- A 50% blend of encoded 200 over black is approximately 146 in
  `linear_srgb`, not 100.
- Exact 0/1 masks remain exact; soft-edge golden cases lose the dark fringe.
- Light wrap affects only the edge band and has bounded energy.
- A malformed mask/shape/edge foreground still fails deterministically.
- The legacy path, if retained, remains byte-identical until its planned
  removal/default flip.

**Dependencies:** VIS-2.1.

### VIS-2.3 — Implement the mask-aware color estimator · L

**Goal:** estimate a conservative foreground transform from the currently
displayed camera/backdrop pair.

**Context:** statistics must use the final fitted backdrop, not the uncropped
asset, and must avoid segmentation edges, clipped pixels, and saturated scene
content. The estimator should return parameters and confidence, not mutate
frames or temporal state.

**Work:**

- Define an immutable `ColorEstimate`/`ColorTransform`.
- Downsample frame, fitted backdrop, and mask to the bounded analysis size with
  area interpolation.
- Build:
  - an eroded/high-confidence foreground-core sample;
  - a local backdrop sample near/under the subject;
  - a robust global-backdrop fallback.
- Estimate log-luminance exposure difference.
- Estimate diagonal WB/illuminant gains only when both samples have enough
  low-chroma, non-clipped pixels.
- Return confidence and a bounded reason enum:
  `ok`, `insufficient-mask`, `insufficient-neutral`, `clipped`,
  `mode-excluded`, `solid-saturated`, `invalid`.
- Apply VIS-0.3 clamps and strength interpolation toward identity.
- Use percentage/area thresholds so output-resolution changes do not retune the
  behavior.
- Do not infer full/limited range from pixel values.

**Likely files:** `src/custback/color.py`, `tests/test_color.py`.

**Acceptance:**

- Known synthetic exposure/tint values are recovered within ratified tolerance.
- Outliers and 10–20% mask contamination do not dominate estimates.
- Tiny/extreme/invalid masks and saturated solids return bounded identity or
  exposure-only estimates with the expected reason.
- Skin-like and saturated clothing patches remain within the selected
  hue/chroma bounds at default strength.
- Work and memory are bounded by analysis resolution, not session duration.

**Dependencies:** VIS-0.3, VIS-2.1.

### VIS-2.4 — Add temporal adaptation, confidence, and reset behavior · L

**Goal:** prevent correction flicker, pumping, stale transforms, and feedback
against camera auto-controls.

**Context:** mask temporal smoothing exists, but color parameters have no state.
The pipeline repeats the last output when no new capture arrives at
`src/custback/pipeline.py:1970-1975`; repeated frames must not advance color
time.

**Work:**

- Add a stateful `ColorHarmonizer` around the pure estimator.
- Smooth parameters by elapsed time:
  `alpha = 1 - exp(-dt / tau)`, so behavior is equivalent at 15/30/60 FPS.
- Add deadbands, per-second slew limits, warm-up/fast-acquisition, and a slow
  steady-state time constant.
- Freeze briefly on low confidence, then decay toward identity after a bounded
  timeout.
- Detect meaningful camera/backdrop scene cuts from analysis statistics and
  reset/fast-acquire without a one-frame unbounded flash.
- Reset on:
  - frame-size/geometry change;
  - backdrop asset/source/effective mode/fit change;
  - correction-config change;
  - camera generation/restart;
  - pipeline startup.
- Expose capture generation or safely consume existing restart count; do not
  infer reconnect from wall-clock gaps alone.
- State contains parameters/statistics only, not full historical frames.

**Likely files:** `src/custback/color.py`, `src/custback/capture.py`,
`tests/test_color.py`.

**Acceptance:**

- Equivalent sequences at 15/30/60 FPS converge to equivalent transforms.
- A static noisy sequence stays below the ratified EV/gain jitter limits.
- A hard scene cut settles within the ratified window without an out-of-clamp
  frame.
- Low-confidence periods freeze then converge smoothly to identity.
- Camera reconnect and background switch cannot reuse stale correction.
- Calling repeat-output paths does not advance the harmonizer.

**Dependencies:** VIS-2.3.

### VIS-2.5 — Integrate harmonization as transactional pipeline state · L

**Goal:** add automatic correction to local compositing without weakening
activation, privacy, or failure behavior.

**Context:** `_trial_activation` deliberately avoids existing segmenter,
refiner, and backdrop state. A mutable harmonizer cannot live as an untracked
field on `Pipeline` or a failed PATCH could change subsequent frames.

**Work:**

- Add a harmonizer generation to `_Resources`/`_Activation`, or an equally
  explicit transactional owner.
- Create/clone/reset staged harmonizer state when color policy or the effective
  visual backdrop/geometry generation changes.
- Trial candidates only with staged/synthetic state; never update the live EMA.
- In `_local_composite`:
  1. segment/refine the original normalized camera frame;
  2. obtain the final fitted backdrop;
  3. estimate/update correction for eligible modes;
  4. pass one transform into the linear compositor;
  5. apply that transform identically to the camera foreground and RVM
     `last_foreground`.
- Bypass/identity:
  - passthrough;
  - blur;
  - saturated solid color under the selected policy;
  - remote frames/privacy slate;
  - disabled correction;
  - low confidence with no previously reliable transform.
- When confidence drops after a reliable estimate, use exactly the
  freeze-then-stale-decay behavior from VIS-2.4; do not replace it with an
  immediate identity bypass. Estimator exceptions follow the separately
  ratified reset/decay policy.
- Keep `hub.publish_raw(frame)`, raw fingerprints, and remote candidates
  byte-identical to the normalized, uncorrected camera frame.
- Convert estimator exceptions through the ratified reset/decay policy plus a
  transition diagnostic. Still fail if output validation itself is violated.
- Add `color_correction_ms` to stage timing without merging it into
  segmentation/background time.

**Likely files:** `src/custback/pipeline.py`,
`src/custback/color.py`, `src/custback/compositor.py`,
`tests/test_pipeline.py`.

**Acceptance:**

- Image/video/live-camera local modes correct only foreground pixels under mask.
- RVM clean foreground receives the exact same transform as the camera
  foreground.
- Blur, passthrough, remote output, privacy slate, and raw WebSocket frames
  satisfy their identity contracts.
- Failed/timed-out/conflicting hot activation leaves old config, provider, and
  temporal state unchanged.
- No correction error can publish a malformed frame or leak raw data in remote
  mode.

**Dependencies:** VIS-1.5, VIS-2.2, VIS-2.4.

---

## Phase 3 — Observability, controls, and deeper source color support

**Closure note (2026-07-30):** VIS-3.1–VIS-3.4 are accepted within their
provable software boundary. The live-consumer qualification bullet in
VIS-3.3 and representative-device qualification bullet in VIS-3.4 are
formally retained by VIS-4.2, which already owns cross-platform visual and
hardware qualification. Phase 3 does not claim pyvirtualcam/OBS/conferencing
consumer equivalence or qualified V4L2/MSMF/DSHOW writes. VIS-3.3 delivers a
metadata-aware local decoder, explicit output declarations where provable,
and honest unsupported cases; VIS-3.4 delivers the accepted side-effect-free
`preserve`/`unqualified` capability report. See the Phase-3 implementation
review for the exact verification ledger and deferrals.

### VIS-3.1 — Add geometry/color telemetry and transition logging · M

**Goal:** let operators and tests distinguish source problems, applied
transforms, bypasses, and fallbacks.

**Context:** existing status exposes negotiated capture size and stage timing
but not normalized geometry or correction state. Logs already use transition
semantics for fallbacks and safe summaries.

**Work:**

- Extend capture/output status with:
  - delivered/oriented source width/height;
  - output width/height and effective output FPS;
  - camera/background fit, rotation, mirror, scale, crop, and pad;
  - capture generation.
- Extend correction status with:
  - active/effective mode and reason;
  - confidence;
  - exposure EV and bounded WB gains;
  - warm-up/stale state;
  - `color_correction_ms`;
  - declared input color assumption.
- Preserve existing `capture_width/height` meaning or version/migrate it
  explicitly; do not silently relabel negotiated size as output size.
- Update identity and per-frame stats atomically with the matching output.
- Log only first application and transitions: source shape/reconnect, fit plan,
  correction active/bypass/reason, scene cut, confidence loss/recovery.
- Keep logs credential-free and omit asset/device paths.
- Add correction/transform totals to the existing shutdown summary.
- Add a schema-parity regression: every key from `FrameHub.stats_dict()` must
  be represented by the status response/OpenAPI model. This also closes the
  current drift where acceleration keys are serialized by
  `src/custback/hub.py:385-397` but are absent from
  `_StatusResponse` at `src/custback/api/server.py:324-377`.

**Likely files:** `src/custback/capture.py`, `src/custback/hub.py`,
`src/custback/pipeline.py`, `src/custback/api/server.py`,
`src/custback/preview.py`, `src/custback/diagnostics.py`,
status/preview/API tests.

**Acceptance:**

- Status can express: “640×480 delivered → cover/crop → 1280×720, correction
  +0.35 EV, WB active, confidence 0.82.”
- Disabled, warming, low-confidence, mode-excluded, stale-decay, and active
  states are distinguishable.
- Logs are transition-only and contain no local source identifiers.
- API/OpenAPI response models match hub output exactly.

**Dependencies:** VIS-1.5, VIS-2.5.

### VIS-3.2 — Add Web UI controls and operator diagnostics · M

**Goal:** expose useful controls without turning the quality panel into a color
science console.

**Context:** the existing quality panel already patches segmentation and
compositing live at `src/custback/api/webui.py:483-508`, `1689-1749`. System
diagnostics enumerate status values at `511-533`, `1751+`.

**Work:**

- Add prominent:
  - automatic color correction toggle;
  - correction strength;
  - camera/background fit selector where product-approved;
  - simple focal-point controls or click-to-position for backdrop anchors.
- Put exposure limit, WB strength, adaptation time, blend-space compatibility,
  rotation, and explicit canvas dimensions under advanced/restart-marked UI or
  YAML-only based on VIS-0.1.
- Mark camera/output geometry changes as restart-required before submission.
- Show a concise live diagnostic summary and full values in System status.
- On PATCH failure, restore controls from effective config just as current
  quality controls do.
- Support keyboard, screen reader labels, 320–768px layouts, and current design
  tokens.

**Likely files:** `src/custback/api/webui.py`, `tests/test_webui.py`,
`tests/test_api.py`.

**Acceptance:**

- Controls generate minimal merge patches and correctly recover after 409/422/
  503 responses.
- UI never presents a low-confidence frozen/stale transform as a fresh active
  estimate, and never claims correction when status says disabled/bypassed.
- Restart-only fields are visibly marked.
- Existing accessibility/responsive Web UI tests pass and new controls are
  covered.

**Dependencies:** VIS-0.4, VIS-3.1.

### VIS-3.3 — Normalize video/output color metadata where provable · L

**Goal:** reduce backend-dependent source interpretation that software
harmonization cannot reliably diagnose.

**Context:** OpenCV does not expose enough trustworthy color metadata in the
current code path. Guessing range/matrix from a histogram is unsafe.

**Work:**

- Qualify the actual OpenCV/FFmpeg backends on supported OSes for:
  range, matrix, primaries, transfer, and rotation behavior.
- If the contract cannot be proven, spike a metadata-aware decoder such as
  PyAV/FFmpeg with bounded dependency, packaging, license, security, timing,
  seek, VFR, and hardware-support analysis.
- Preserve the current `VideoBackdrop` clock/skip/reuse/seek behavior when
  changing decode implementation.
- Add explicit source overrides only for reproducible operator-owned assets and
  only after automatic metadata resolution; never histogram-guess.
- Qualify pyvirtualcam consumer interpretation on Linux/macOS/Windows.
- Set Windows MF RGB32 primaries/transfer/nominal-range attributes only from a
  proven output contract.

**Likely files:** `src/custback/backgrounds.py`, optional new decoder module,
`pyproject.toml`, npm install/release dependency allow-lists,
Windows `MediaSource.cpp`, processing/packaging/system tests.

**Acceptance:**

- Tagged limited/full BT.601/709 fixtures produce the same reference sRGB
  result on each advertised backend, or unsupported cases are explicitly
  documented/rejected.
- Playback phase, looping, VFR timing, and bounded decode behavior remain green.
- No automatic decision is based only on observed pixel range.

**Dependencies:** VIS-2.1. This is a post-MVP follow-up unless qualification
shows it is required to meet VIS-4.2.

### VIS-3.4 — Characterize optional camera hardware controls · M

**Goal:** decide whether locking or exposing camera WB/exposure would improve
stability beyond software correction.

**Context:** `CAP_PROP_AUTO_WB`, WB temperature, auto exposure, exposure, gain,
gamma, and units/accepted values vary significantly by V4L2/MSMF/DSHOW. A
generic continuous writer can fight the device and make output worse.

**Work:**

- Read and report supported properties/values without changing them.
- Qualify V4L2, MSMF, and DSHOW independently on representative devices.
- If justified, design explicit policies:
  `preserve`, `lock_after_warmup`, and/or `manual`.
- Verify every write by readback and make unsupported/rejected settings visible.
- Never drive hardware settings continuously from the software harmonizer.
- Reset software temporal state after any confirmed hardware transition.

**Likely files:** `src/custback/capture.py`, config/status/UI, platform/hardware
tests.

**Acceptance:**

- A capability report is truthful and side-effect free.
- Any implemented setting is backend-qualified, read-back verified, opt-in,
  restart-safe, and cannot form a fast feedback loop with the harmonizer.

**Dependencies:** VIS-3.1. This is normally scheduled after VIS-4.2 and is not
an MVP dependency.

---

## Phase 4 — Qualification and rollout

**Implementation status (2026-07-31):** VIS-4.1 is complete. The VIS-4.2
harness, deterministic gate, calibrated production-path benchmark, memory
soak, physical-evidence schema, and fail-closed validator are implemented.
The 10,000-frame deterministic contract and a 300-frame-per-tier local
observation pass, but only a clean pinned-runner report plus the complete
physical platform matrix may close release qualification. VIS-4.3
documentation, migration, package allowlists, workflow coverage, four-stage
ledger, rollback instructions, and non-self-referential Git evidence sequence
are staged. Schema-v1 remains `stretch` + `srgb_legacy` + correction `off`.
The diagnostic report and contact sheet are retained at
`docs/visual-consistency-phase4-local-observation.json` and
`docs/visual-consistency-phase4-local-observation-contact-sheet.png`; they are
explicitly non-authorizing because this worktree is dirty and unpinned.
See `docs/visual-consistency-phase4-implementation-review.md` for the
requirement map, exact results, closing-audit resolutions, and ordered external
handoff.

The closing audit additionally made the repeated-frame contract executable:
pixel-identical successful reads remain distinct temporal inputs, while only a
missing read repeats the last guarded output without analysis or temporal
advance. Real-pixel clipped/saturated/low-confidence/noisy scenarios,
correction-off blur/color/image identity, and remote near-raw privacy fallback
through the actual ASGI MJPEG route are also covered.

### VIS-4.1 — Add end-to-end, privacy, and temporal regressions · L

**Goal:** prove the integrated feature, not only its helper functions.

**Context:** geometry, temporal color state, hot activation, raw publication,
remote rendering, and output sinks meet only inside the running pipeline.
Helper-level tests cannot prove that the same frame generation reaches every
boundary or that a correction/reset change preserves the existing privacy
fallbacks.

**Work:**

- Geometry matrix:
  - 4:3 webcam plus square/portrait/16:9 image/video/live backdrop into 16:9;
  - cover/contain/stretch, anchor edges, rotation/mirror, odd sizes;
  - reconnect and mid-stream size change;
  - 720p and 1080p canvas.
- Color matrix:
  - warm/cool and ±EV pairs;
  - clipping, saturation, low confidence, slow drift, noise, hard cuts;
  - ICC-tagged images;
  - RVM edge foreground consistency;
  - blur/color/passthrough mode-specific identity.
- Lifecycle matrix:
  - startup preflight;
  - successful/failed/timed-out/conflicting PATCH;
  - background upload/staged activation;
  - repeated capture frames;
  - provider replacement and shutdown.
- Privacy/API matrix:
  - raw frame/JPEG remains uncorrected;
  - remote exact/wrong/invalid size;
  - current/delayed raw-echo privacy checks;
  - privacy slate identity;
  - preview/MJPEG/pyvirtualcam/native sink equality.
- Add property/fuzz-style cases for valid dimensions/anchors/masks without
  adding a heavyweight dependency unless justified.
- Add rapid enable/disable, background-switch, and reconnect coverage to the
  existing stress suite so state generations cannot cross under repeated hot
  changes.

**Likely files:** all focused tests plus phase-6/two-host system tests and release
manifests.

**Acceptance:**

- Every target invariant and mode-table row has an executable regression.
- The former default 4:3→16:9 scenario passes proportional-geometry assertions
  when configured with target `cover`; explicit legacy `stretch` remains
  distorted by design and retains its negative regression.
- Disabled/legacy paths meet their byte-identity contracts.
- Remote privacy behavior is unchanged or stricter; no test updates weaken raw
  echo or wrong-size expectations.
- The complete existing Python/Node quality gates remain green.

**Likely files:** `tests/test_pipeline.py`, `tests/test_api.py`,
`tests/test_streaming.py`, `tests/test_processing.py`,
`tests/test_capture.py`, `tests/test_vcam.py`,
`tests/test_windows_vcam.py`, `tests/test_phase6_stress.py`,
`tests/test_phase6_two_host_system.py`, focused new test modules and fixtures,
release test allow-lists.

**Dependencies:** VIS-1.6, VIS-2.5. VIS-1.6 may satisfy this dependency by
implementing scaling or by explicitly constraining unsupported native
canvas/FPS combinations; silent mismatch rendering is never an option.

### VIS-4.2 — Run performance and cross-platform visual qualification · L

**Goal:** ensure consistency improvements do not make real-time output unstable.

**Context:** the design adds direction-aware resampling, full-frame
sRGB/linear conversions, and a per-frame estimator around code that already
operates against output deadlines. OpenCV capture/decode behavior and virtual
camera consumers also vary by OS/backend, so unit results on one machine are
not sufficient release evidence.

**Work:**

- Benchmark 720p/30, 720p/60, 1080p/30 on CPU for:
  geometry, color analysis, linear conversion/composite, output send, total
  frame time, FPS attainment, deadline misses, and memory.
- Use current EWMA/stats as baseline, plus p50/p95 from a reproducible harness.
- Verify:
  - no duplicate full-frame resize on the common path;
  - estimator work is capped by analysis resolution;
  - no per-session frame-history growth;
  - repeated source/output frames skip unnecessary analysis/conversion where
    safe;
  - live video/camera backdrop decode remains paced.
- Run visual/contact-sheet and live-camera qualification on Linux V4L2,
  macOS/OBS, Windows MSMF/DSHOW + pyvirtualcam, and explicit native Windows if
  supported.
- Include at least two webcams with different default auto-WB/exposure behavior
  and multiple meeting-app consumers.
- Ratify an overhead budget. Provisional target: correction/scaling consumes no
  more than 20% relative end-to-end overhead, with combined added p95 no greater
  than 5 ms at 720p and 8 ms at 1080p on the selected reference CPU. Over 300
  measured frames it should add less than one percentage point of deadline
  misses. Calibrated timing gates belong on a pinned runner; shared CI should
  assert deterministic operation counts and contracts rather than flaky wall
  time.
- Run a 10,000-frame RSS/tracemalloc soak with no unbounded history and a
  provisional retained-growth limit of 5 MiB.

**Acceptance:**

- The agreed p95/FPS/memory budgets pass on each supported tier.
- Geometry and color metrics improve versus VIS-0.2 without violating
  foreground-preservation thresholds.
- Preview, API, and each supported virtual-camera consumer show the same crop,
  orientation, and color within the documented transport tolerance.
- Any unsupported native/video color path is gated or documented, not silently
  advertised.

**Likely files:** reproducible benchmark/qualification harness under `scripts/`,
`scripts/release/required-gates.json`,
`scripts/release/assemble-evidence.js`,
`scripts/release/phase6-evidence.js`,
`scripts/release/two-host-system-test.py`, `.github/workflows/ci.yml`,
`.github/workflows/release.yml`, performance/system-test documentation and
evidence manifests.

**Dependencies:** VIS-3.1, VIS-4.1.

### VIS-4.3 — Document, package, and stage the default rollout · M

**Goal:** land the feature without an irreversible visual surprise or release
manifest failure.

**Context:** this repository verifies exact module, test, dependency, and
package payloads, while each proposed target default changes visible framing or
pixels. Shipping code without manifest changes will fail release gates;
shipping all default flips together would make regressions difficult to
attribute or roll back.

**Work:**

- Update:
  - `README.md` run/config/quality/status sections;
  - annotated `config/default.yaml`;
  - API/OpenAPI examples;
  - migration/upgrade notes;
  - troubleshooting for crop, bars, low confidence, camera auto-controls, and
    tagged/untagged media.
- Update exact release module/test/dependency/payload allow-lists, manifests,
  npm package checks, and migration fixtures.
- Ensure geometry/color tests run in every supported Python/NumPy/OpenCV CI
  compatibility profile and optional segmenter profile, not only the default
  Linux unit-test job.
- Roll out in reversible steps:
  1. land helpers and telemetry with current behavior;
  2. enable proportional camera `cover`, retain explicit `stretch`;
  3. enable linear compositing after golden approval;
  4. enable conservative `auto` correction after platform/performance approval;
  5. remove/deprecate temporary legacy switches only in a later documented
     release.
- Emit a one-time upgrade note when a negotiated aspect mismatch will change
  from stretch to cover.
- State mode exclusions and the external-frame color assumption clearly.

**Acceptance:**

- A developer can configure every supported policy from documentation without
  reading source.
- Existing configs have a deterministic migration/default result.
- Clean wheel, sdist, npm, and platform artifact checks include all new modules,
  tests, defaults, and dependencies.
- Each default flip is a separate commit with rollback instructions and the
  VIS-4.2 evidence attached.

**Likely files:** `README.md`, `config/default.yaml`, `docs/`,
`pyproject.toml`, `package.json`, `scripts/release/verify-release.js`,
`scripts/release/required-gates.json`, release/CI workflows, migration fixtures
and release notes.

**Dependencies:** VIS-3.2, VIS-4.2.

---

## Validation matrix

| Scenario | Geometry assertion | Color assertion | Required sinks |
| --- | --- | --- | --- |
| 4:3 camera → 16:9 canvas | Circle remains round; crop equals anchor plan | Passthrough unchanged after geometry | Preview, MJPEG, pyvirtualcam, native |
| Portrait tagged image background | Orientation applied once; intended focal point retained | ICC reference within tolerance | Preview, snapshot, thumbnail |
| 16:9 video with scene cut | No duplicate resize; clock/phase preserved | Correction fast-acquires then stabilizes within clamps | Preview, MJPEG |
| Live second camera | Same fit math as static/video | Slow bounded adaptation; no oscillation | Preview, pyvirtualcam |
| Blur mode | One camera normalization; no second fit | Correction identity | All local sinks |
| Saturated solid color | Exact canvas | WB identity; optional bounded exposure only | All local sinks |
| Capture reconnect/size change | New transform plan/generation | Old correction discarded | Preview, status, output |
| Hot background anchor change | Next committed frame uses new crop; video phase retained | Harmonizer reset/fast-acquire | API + preview |
| Failed hot correction PATCH | Old geometry/provider/state untouched | Old transform/EMA untouched | API version + output |
| Remote valid output | Exact canonical shape required | Core does not recolor | Remote WS, all output sinks |
| Remote wrong-size/raw echo | No auto-fit | Fixed privacy slate | Preview, MJPEG, vcam |
| Repeated output frame | No repeated geometry work | No temporal-state advance | Stats + output |
| 1080p CPU | Exact shape and one planned resize/source | Meets p95/FPS budget | Null + real output |

---

## Risk register

| Risk | Impact | Mitigation / owning tasks |
| --- | --- | --- |
| `cover` fixes distortion but crops a user's head | High UX | Anchors, preview, migration note, explicit `contain/stretch`; VIS-0.1, 1.1, 3.2, 4.3 |
| Output canvas decoupling breaks remote/avatar shape assumptions | High correctness/privacy | One resolver and exact-size regressions; VIS-0.4, 1.5, 4.1 |
| EXIF/container metadata is applied twice | Medium visual | Explicit decode/orientation contract and labeled fixtures; VIS-1.2 |
| Geometry-only PATCH resets video playback | Medium UX | Separate provider identity and fitted-cache/visual-generation keys; VIS-0.4, 1.4 |
| Aggressive color matching recolors skin/clothes | High visual | Algorithm bake-off, confidence, clamps, strength, identity fallback; VIS-0.3, 2.3 |
| Camera auto-controls and software correction oscillate | High visual | Time-based slow adaptation; no continuous hardware writes; VIS-2.4, 3.4 |
| Temporal state survives a scene/source/reconnect | High visual | Generation-owned state and explicit resets; VIS-2.4, 2.5 |
| Candidate trial mutates live EMA/provider | High lifecycle | Stage/clone state and detached trials; VIS-1.4, 2.5, 4.1 |
| Linear blending causes broad snapshot changes | Medium compatibility | Legacy switch, golden comparison, staged default; VIS-0.1, 2.2, 4.3 |
| ICC conversion weakens image security/limits | High security | Preserve authoritative validated decode and bomb checks; VIS-2.1 |
| Video range/matrix is guessed incorrectly | High color | No histogram guessing; qualify metadata-aware decoder; VIS-3.3 |
| Added conversions miss real-time deadlines | High reliability | LUTs/downscaled analysis, one encode, timing telemetry/gates; VIS-2.1, 3.1, 4.2 |
| Windows native path silently crops/pads | High platform | Scale correctly or reject unsupported formats; VIS-1.6 |
| New modules/tests are omitted from exact release payload | High release | VC-8 and same-change manifest updates; all implementation tasks, VIS-4.3 |

---

## Definition of done

This initiative is complete only when all of the following are true:

- A negotiated 4:3 camera feeding a 16:9 canvas is proportionally fitted, with
  deterministic documented crop/pad and no circle/face distortion.
- All local backdrop providers use the same fit/orientation/interpolation
  implementation and exact canonical canvas.
- Preview, API, pyvirtualcam, and every advertised native output agree on
  dimensions, orientation, crop, and effective FPS.
- Automatic correction is conservative, mask-aware, temporally stable,
  confidence-gated, mode-aware, and applies equally to camera and RVM clean
  foreground.
- Soft-edge alpha and light wrap use the ratified photometric space without
  dark-fringe regression.
- Tagged still images are normalized to the declared output color space;
  untagged camera/video assumptions are explicit and truthful.
- Passthrough/blur/remote/privacy modes retain their identity and privacy
  contracts.
- Failed/timed-out/conflicting hot changes cannot mutate working geometry,
  provider playback, color state, config, or version.
- Status and logs explain the effective source→canvas transform and correction
  state without revealing paths, devices, or credentials.
- Geometry/color additions meet the agreed p95, FPS-attainment, memory, and
  deadline-miss budgets at 720p and 1080p on supported platforms.
- Focused unit, integrated pipeline, API/UI, privacy, packaging, and
  cross-platform system gates pass from clean artifacts.
- Default changes and legacy escape hatches are documented and landed in
  separately reversible commits.
