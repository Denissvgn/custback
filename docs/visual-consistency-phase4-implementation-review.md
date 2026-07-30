# Phase 4 qualification and rollout implementation review

- Status: Engineering complete; release qualification and default activation
  remain gated
- Date: 2026-07-31
- Scope: `AUTO_COLOR_CORRECTION_AND_SCALING_BACKLOG.md`, VIS-4.1 through
  VIS-4.3
- Contract: `docs/adr/0001-visual-consistency-contract.md`
- Qualification manifest:
  `scripts/release/visual-qualification-manifest.json`
- Qualification runbook:
  `docs/visual-consistency-phase4-qualification-runbook.md`
- Rollout authority: `scripts/release/visual-policy-rollout.json`

This is the Phase-4 implementation and closing-audit record. It distinguishes
completed engineering from evidence that can only be collected on a clean,
pinned release candidate and real camera/output platforms.

## Outcome and release decision

The Phase-4 implementation is complete:

| Task | Engineering status | Release status |
| --- | --- | --- |
| VIS-4.1 integrated regressions | Complete | Passed locally and configured in every required CI profile |
| VIS-4.2 qualification machinery | Complete | Deterministic and local calibrated checks pass; pinned and physical evidence pending |
| VIS-4.3 documentation, packaging, and rollout staging | Complete | Staged; target defaults remain inactive |

The release decision is **no default activation yet**. The checked-in
schema-v1 defaults remain:

```yaml
camera:
  fit_mode: stretch
compositing:
  blend_space: srgb_legacy
  color_correction:
    mode: "off"
```

`config/default.yaml` and `src/custback/default.yaml` are byte-identical.
The `camera-cover`, `linear-compositing`, and `automatic-correction` rollout
stages remain `pending`, with no evidence, change commit, or approval entered.

The remaining work is external qualification, not an unimplemented code path:

1. run the calibrated contract on the selected clean, pinned Linux reference
   runner and bind the report to the exact candidate;
2. complete all five physical platform/backend rows with two distinct webcams,
   different auto-control profiles, and at least two meeting consumers;
3. attach and hash the required contact sheet, preview capture, API snapshot,
   and consumer capture for each row; then
4. activate one rollout stage at a time through the separate default-change
   and later evidence-approval commits described below.

## VIS-4.1 — integrated visual, privacy, and temporal regressions

`tests/test_visual_consistency_e2e.py` drives the production `Pipeline`,
geometry, temporal harmonization, compositing, publication, and
virtual-camera boundaries with controlled capture, provider, segmenter, and
refiner doubles. Separate rows cross a real `OpenCVCapture` reconnect and the
actual `ImageBackdrop`; focused provider, lifecycle, privacy, and stress suites
cover the remaining production boundaries.

| Required boundary | Executable evidence |
| --- | --- |
| 4:3 camera and square/portrait/16:9 providers into canonical output | Four representative integrated rows collectively cover image/video/live labels, cover/contain/stretch, anchor edges, rotation/mirror, odd sizes, 720p, and 1080p; focused provider-equivalence and seeded geometry-property tests supply the broader cross-product |
| Production reconnect and delivered-size change | `OpenCVCapture` is driven from 32x24 to 16x32 through `Pipeline`, pyvirtualcam, and the native ring; the new generation and geometry are asserted |
| Warm/cool and exposure pairs | Real color-pair sequences cross the complete frame lane while the raw publication remains byte-unchanged |
| Clipping, saturation, low confidence, and sensor noise | Real-pixel integrated rows exercise the production estimator/harmonizer and assert identity or exposure-only fallback, bounded EV/WB, stable noisy output, raw/background endpoint integrity, and public status |
| ICC-tagged portrait media | Actual `ImageBackdrop` rows decode an ICC-tagged image before fitted background use; EXIF orientations 1–8 are asserted separately at the authoritative decode/geometry boundary |
| Slow drift and hard scene cuts | Controlled backdrop sequences assert bounded steady adaptation, one-frame cut hold, one source read per output, and bounded post-cut acquisition; focused provider tests cover playback-phase retention |
| Foreground preservation and RVM edge behavior | Focused color/compositor tests assert mask locality, exact endpoints, and one transform for the camera and clean edge foreground |
| Disabled local modes | Blur, color, and image rows prove correction-off output remains byte-identical at the full-foreground endpoint and that the estimator is never called |
| Raw and output separation | Raw frame/JPEG identity is asserted while auto correction affects only eligible local output |
| One live frame at every API boundary | A frozen active-auto frame is exercised through the real raw/output WebSockets, snapshot route, and streaming MJPEG response |
| Remote privacy behavior | Current, delayed, near, wrong-sized, and malformed raw echoes remain rejected; a near-raw echo crosses the real renderer WebSocket and reaches the actual ASGI MJPEG/snapshot routes only as the fixed privacy slate. Pipeline preview publication, pyvirtualcam, and native output retain the same exact pre-transport pixels |
| Hot lifecycle behavior | Successful, rejected, conflicting, timed-out, and cancelled configuration paths preserve generation and harmonizer ownership |
| Repeated-output behavior | A missing new capture repeats the last guarded output without advancing temporal state or repeating analysis work; distinct non-null captures remain real temporal inputs |
| Rapid visual changes | The Phase-6 stress suite performs 100 enable/disable and provider changes plus explicit synthetic capture-generation resets without cross-generation state; focused tests cover anchor-only activation and real capture reconnect |

Gaps found during the closing audit were added explicitly:

- the reconnect test now crosses the real `OpenCVCapture` boundary and both
  virtual-camera transports in one test, rather than relying on isolated
  capture and sink tests;
- the active-auto API test now proves that one production frame is consistent
  across both WebSockets, snapshot, and the actual ASGI MJPEG stream;
- a real renderer-WebSocket near-echo now proves that the privacy slate crosses
  the actual ASGI MJPEG and snapshot routes within transport tolerance;
- successful pixel-identical capture reads are distinguished from synthesized
  repeats: the former remain temporal inputs, while only a missing read reuses
  guarded output without analysis or state advance; and
- clipped, saturated, low-confidence, noisy, and correction-off mode rows now
  cross the production pipeline instead of relying only on helper-level
  evidence.

The focused Phase-4 production/evidence suite finished with 363 passing tests.
The live API module finished with 64 passing tests. The final complete Python
suite finished with 1,490 passed and 3 skipped.

## VIS-4.2 — qualification and performance

### Reproducible authority model

`scripts/release/visual_consistency_qualification.py` has three deliberately
separate authorities:

| Authority | What it may prove | What it may not prove |
| --- | --- | --- |
| `shared-ci-deterministic` | Visual metrics, foreground preservation, operation counts, pacing, bounded work/state, and the 10,000-frame soak | Wall-clock performance or real-device interoperability |
| `local-observation` | Diagnostic p50/p95 samples on the current host | Pinned-runner or release approval |
| `pinned-reference-runner` plus complete physical evidence | Reviewed timing budgets and release qualification for an exact clean candidate | A later source tree or an unrecorded platform |

The validator rejects unknown fields, missing samples, derived-metric
inconsistency, altered digests, insufficient frame counts, absent RSS,
unhashed/substituted artifacts, incomplete physical rows, source mismatch,
non-ancestor evidence, and dirty or unpinned release claims.

Shared CI and release workflows run the deterministic gate. Calibrated timing
is intentionally excluded from shared CI to avoid turning host contention into
a flaky release signal.

### Deterministic result

The final-tree local run used the manifest defaults: 10,000 measured soak
frames after a 512-frame warmup. It passed every deterministic contract:

- Phase-0 visual improvement;
- foreground preservation;
- one geometry-helper resize per source on the measured common path;
- 192-pixel analysis cap;
- repeated-output work skip;
- video and live-camera source pacing; and
- bounded session history/state.

The deterministic evidence digest was:

```text
bae8f45a437e66b92ff2db12ce763605410797321b228e83d7d722fbd3f42621
```

Retained-state accounting is explicit:

| Component | Conservative bytes |
| --- | ---: |
| Fixed raw-replay Bloom storage | 16,777,216 |
| One maximum 1920x1080 BGR frame | 6,220,800 |
| Maximum 192x192x3 float32 analysis cache | 442,368 |
| Total semantic fixed state | 23,440,384 |
| Reviewed limit | 25,165,824 |
| Headroom | 1,725,440 |

The targeted retained-state soak retained 29,306 traced bytes and 4,096 RSS
bytes across 10,000 post-warmup observations, both below the 5 MiB growth
limit. It exercises the named long-lived structures rather than claiming
10,000 complete `Pipeline._loop` renders: the geometry-plan cache remained at
its exact 512-entry bound, the harmonizer snapshot retained only scalars, the
replay storage remained exactly 16 MiB, and only one latest frame was
retained.

The audit rejected reducing the replay Bloom storage to satisfy a smaller
headline number. At five MiB its projected saturation would make false replay
revocations operationally significant. The implementation instead preserves
the reviewed 16 MiB privacy structure and accounts for it honestly within a
separate 24 MiB fixed-session bound.

### Production-path performance remediation

The initial qualifier exposed a real CPU blocker: the public NumPy linear-RGB
reference compositor was too slow at 720p and 1080p. Production now retains
the public reference APIs while using an internal, equivalent OpenCV-native
linear-BGR lane:

- uint8 BGR is EOTF-decoded through a compiled lookup table without a
  full-frame channel-reversal copy;
- foreground, mask, and backdrop analysis rasters use bounded linear-light
  `INTER_AREA` work, with independent resizes joined through a
  resource-owned three-worker pool;
- immutable `ImageBackdrop` analysis retains one generation-scoped raster no
  larger than 192 pixels on its long edge;
- transforms preserve public RGB semantics by applying gains in B, G, R order
  to the internal buffer;
- alpha blending, RVM edge replacement, and light wrap use compiled OpenCV
  operations; and
- the compositor consumes its newly owned output-linear buffer for in-place
  LBGR-to-Lab-to-encoded-BGR conversion, avoiding two full-frame float
  temporaries.

The accelerated lane is compared against the public linear-RGB reference for
all 256 code values, randomized inputs, transforms, RVM edge replacement, and
light wrap. Exact alpha endpoints remain byte-identical; all other encoded
pixels differ by at most one code value. Public APIs remain non-consuming.

The executor is closed on normal resource shutdown, qualification success and
failure, and resource-construction failure. Static-image cache identity is
invalidated by generation/provider/geometry change and cleared on close.
Dynamic video/camera backdrops are not assigned the immutable-image cache.

The final audit also forced OpenCV transform and encode failures. Once
structural contracts have passed, a photometric `cv2.error` becomes
`ColorError` and triggers exactly one identity render. Malformed frames,
masks, and shapes remain strict `ValueError` failures and are not swallowed.

### Final local calibrated observation

The retained final-tree report was generated at `2026-07-30T21:30:24Z` with
5 warmup and 300 measured frames per tier. It is a dirty-tree, unpinned
`local-observation` from Python 3.14/OpenCV 5/NumPy 2 in a container using a
non-reviewed power policy, so `release_qualified` is correctly `false`.

| Tier | Legacy p95 | Candidate p95 | Added p95 | Relative overhead | Deadline misses, legacy → candidate | FPS attainment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 720p30 | 14.4371 ms | 14.4985 ms | +0.0614 ms | +0.425% | 0.0000% → 0.0000% | 100% |
| 720p60 | 14.5744 ms | 14.4750 ms | -0.0994 ms | -0.682% | 0.3333% → 0.3333% | 100% |
| 1080p30 | 37.0897 ms | 29.6537 ms | -7.4360 ms | -20.049% | 61.3333% → 1.0000% | 100% |

The variable 1080p baseline miss rate is another reason this host result is
diagnostic only. It is preserved rather than normalized away; the pinned
runner must repeat the contract under its reviewed CPU, governor, affinity,
cooling, image, and background-workload controls.

Every unchanged local check passed:

- relative end-to-end overhead at or below 20%;
- added p95 at or below 5/5/8 ms;
- deadline-miss increase at or below one percentage point; and
- FPS attainment at or above 99%.

The raw report is retained as
`docs/visual-consistency-phase4-local-observation.json`; its SHA-256 is
`11c2c9f6deb34e76c1c7b0ff83efb54b1d7eace35c95cb7c4ad20d24a96232c5`.
The generated contact sheet is retained as
`docs/visual-consistency-phase4-local-observation-contact-sheet.png`;
the generated 1920x180 contact sheet SHA-256 was
`f529a9a6bd5189b8bb3dd05aff00a5ccade13f790460b4560c189184ba3dd8c7`.
These hashes record the closing observation but are not entered into the
rollout ledger or presented as release evidence. A deterministic validation
of the report passed. A release validation failed closed, as required, because
the implementation worktree is not a clean candidate.

### Physical matrix still required

The current environment is Linux-only and has no `/dev/video*` device. It
cannot truthfully fill any real-camera/consumer row.

| Required row | Status |
| --- | --- |
| Linux V4L2 capture → pyvirtualcam/v4l2loopback | Pending |
| macOS AVFoundation capture → pyvirtualcam/OBS Virtual Camera | Pending |
| Windows MSMF capture → pyvirtualcam/OBS Virtual Camera | Pending |
| Windows DSHOW capture → pyvirtualcam/OBS Virtual Camera | Pending |
| Windows MSMF capture → native Media Foundation virtual camera | Pending |

The physical contract also requires two distinct webcams with different
auto-exposure/white-balance behavior, two distinct meeting consumers, exact
crop/orientation checks, transport tolerance, luminance/neutral improvement,
foreground hue/chroma preservation, and steady-state EV/WB stability.
`docs/visual-consistency-phase4-qualification-template.json` and the runbook
provide the fail-closed collection format.

## VIS-4.3 — documentation, packaging, and staged rollout

The implementation updates the README, annotated defaults, Web/API examples,
migration fixtures, troubleshooting, packaging manifests, release allowlists,
and CI/release workflows. The documented controls cover geometry policies,
anchors, blend space, correction strength and limits, adaptation, exclusions,
camera-control assumptions, and tagged/untagged source behavior.

Schema-v1 migration remains deterministic. Existing files that omit new keys
receive the compatibility policy. The one-time aspect-mismatch note explains
the future stretch-to-cover change without logging a camera path.

Both OpenCV compatibility profiles execute the integrated visual,
qualification, color, processing, and pipeline suites:

- OpenCV 4.8.1 with the NumPy-1-compatible profile; and
- OpenCV 5.x with the NumPy-2-compatible profile.

MediaPipe and RVM optional profiles also retain the visual E2E and
qualification coverage. Workflow regression tests enforce those lists so a
future filter cannot silently remove the gate.

### Non-self-referential rollout sequence

A report cannot be stored in the same commit that it claims to qualify.
The staged verifier therefore requires this sequence for each target default:

1. create one isolated default-change commit `C`;
2. run qualification externally against clean commit `C`;
3. create a later approval/ledger commit containing the exact report and
   setting the stage active; and
4. build the candidate from that later clean approval commit.

The validator binds historical source blobs at `C`, requires `C` to be an
ancestor of the approval commit, and binds current evidence/ledger artifacts
to the approval HEAD. The candidate builder passes a trusted original Git
root, HEAD, and tree into the staged npm prepack so a gitless staging copy
cannot substitute its own provenance. Standalone gitless approval fails
closed.

Stages are deliberately reversible:

1. compatibility (`stretch`, `srgb_legacy`, correction `off`);
2. proportional camera `cover`;
3. linear compositing; then
4. conservative automatic correction.

Each target stage has explicit pins and operator rollback instructions. No
stage removes the legacy switches; deprecation belongs to a later release.

## Closing audit and resolved discrepancies

| Finding | Resolution |
| --- | --- |
| Capture reconnect and sink tests were separate | Added one production `OpenCVCapture` → `Pipeline` → both-vcam integrated regression |
| API equality did not exercise every live transport together | Added raw/output WebSocket, snapshot, and actual ASGI MJPEG coverage over one active-auto frame |
| Privacy-slate MJPEG evidence used the encoder directly | Added a real raw-WebSocket near-echo that is rejected and reaches the actual ASGI MJPEG/snapshot routes only as the slate |
| Pixel-identical frames were incidental, not a repeated-capture contract | Added explicit successful-identical-read versus missing-read cases with segmentation, estimator, harmonizer, input, output, and repeat counts |
| Edge-color and disabled-mode integration relied on helper tests | Added clipped, saturated, low-confidence, noisy, and correction-off blur/color/image rows through `Pipeline._loop` |
| The first review overstated several integrated test scopes | Distinguished controlled doubles from production providers, representative rows from cross-products, pipeline preview from HighGUI, and synthetic generation reset from real reconnect |
| Qualifier timed the old public NumPy path | Calibrated samples now call the shipped private linear-BGR decode, estimator, cache, and compositor |
| Direct EWMA probe leaked the new worker pool | Qualification resources close in `finally`; a spy proves worker shutdown and one output close |
| A full-frame cache looked attractive for speed | Rejected: a 1080p float32 cache would push semantic fixed state to roughly 48 MiB |
| Bounded analysis cache was initially omitted from memory evidence | Added exact 442,368-byte conservative accounting, validator recomputation, digest binding, mutation test, and runbook calculation |
| Three analysis workers could escape on construction failure | ExitStack now shuts down only the executor while existing callbacks close each backend exactly once |
| OpenCV acceleration could alter pixels | Added exhaustive code-ramp/random equivalence, correct gain-order, exact endpoint, RVM, and wrap tests; non-endpoints stay within one code value |
| A post-validation OpenCV error bypassed identity fallback | Translate only validated photometric `cv2.error` to `ColorError`; preserve strict structural errors and prove one retry |
| A default flip and its evidence could form a self-reference | Split default commit from later evidence-approval commit and validate historical/current Git blobs separately |
| Staged npm prepack lost Git context | Added trusted clean-root/HEAD/tree bridge with byte binding to the original source |
| OpenCV/optional CI profiles could omit integrated tests | Added exact workflow guards and unfiltered E2E/qualification coverage |

The final independent audit found no remaining locally actionable,
release-significant production correctness, privacy, portability,
performance, or evidence-integrity defect. It did identify test-scope
overstatements, which this review narrows explicitly; clean-runner timing and
real-device/platform evidence remain external release gates.

## Final verification

| Command or gate | Result |
| --- | --- |
| Focused color/processing/pipeline/E2E/qualification/stress suite | 363 passed |
| Live API module, including remote privacy through ASGI MJPEG | 64 passed |
| Full Python suite outside the loopback-restricted sandbox | 1,490 passed, 3 skipped |
| Ruff lint | Passed |
| Ruff format check | Passed; 102 files already formatted |
| Pyright | 0 errors, 0 warnings |
| Workflow YAML parse | Passed |
| Node/npm suite outside the nested-process sandbox | 147 passed, 2 expected TODO release blockers |
| Installed wheel/sdist/npm artifact smoke | Passed for 0.4.0 (diagnostic only) |
| `git diff --check` | Passed |
| Deterministic 10,000-frame report validation | Passed |
| Local 300-frame-per-tier calibrated checks | All tiers passed |
| Release claim for the dirty local report | Rejected, as required |

The two Node TODOs are the repository's existing release-publication blockers
(`REL-01` and `WIN-01`), not silent Phase-4 passes. The Phase-4 rollout also
remains independently blocked by its pinned and physical qualification
requirements.

The installed-artifact smoke used the documented
`CUSTBACK_RELEASE_ALLOW_TMPFS=1` diagnostic override because this environment's
only external writable temporary root is memory-backed. The smoke remains
non-authorizing; the full release path still rejects an unapproved tmpfs
workspace.

## Ordered handoff

The next developer/release-owner actions are:

1. prepare a clean candidate without changing compatibility defaults;
2. run the pinned Linux calibration and archive its raw report/contact sheet;
3. execute the physical template on every required OS/backend/transport using
   the required cameras and consumers;
4. validate the combined report with `--claim release`;
5. create only the `camera-cover` default commit;
6. qualify that exact commit externally;
7. land its later ledger-approval commit and build from that clean HEAD; then
8. repeat the same isolated sequence for linear compositing and, finally,
   automatic correction.

Until those steps are complete, explicit opt-in configurations may use the
implemented policies, but the repository must continue to ship the schema-v1
compatibility defaults.
