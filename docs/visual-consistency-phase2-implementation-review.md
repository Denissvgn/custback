# Phase 2 color normalization and harmonization implementation review

- Status: Complete; VIS-2.1 through VIS-2.5 accepted
- Date: 2026-07-30
- Scope: `AUTO_COLOR_CORRECTION_AND_SCALING_BACKLOG.md`, VIS-2.1 through
  VIS-2.5
- Contract: `docs/adr/0001-visual-consistency-contract.md`

The VIS-2 backlog rows are closed by this review. This acceptance does not
authorize the Phase-4 default rollout.

## Outcome

Phase 2 adds an explicit color-managed still-image boundary, bounded
linear-sRGB primitives, a linear-light compositor, the selected mask-aware
estimator, an elapsed-time harmonizer, and generation-owned transactional
pipeline integration.

An independent closing audit mapped every Phase-2 acceptance criterion and the
cross-cutting privacy, ownership, geometry, compatibility, packaging, and
failure contracts to executable evidence. It found three temporal/invalidation
gaps; all three were fixed and regression-tested before acceptance:

1. A hard reset now consumes the first reliable estimate as the new seed
   without applying a one-frame correction flash.
2. Sparse low-confidence updates decay only for time after the 0.5-second
   freeze boundary.
3. Live backdrop source-size and fitted-geometry changes participate in the
   reset token through the provider's immutable current `TransformPlan`.

The compatibility landing state is unchanged:

| Policy | Schema-v1/default | Implemented opt-in behavior |
| --- | --- | --- |
| Blend space | `srgb_legacy` | `linear_srgb` |
| Color correction | `off` | `auto` |
| Eligible correction modes | None while correction is off | `image`, `video`, and live `camera` |
| Identity/bypass modes | `passthrough`, `blur`, `color`, and `remote` | Unchanged |

`auto` is supported with either blend space. It never silently selects
`linear_srgb`. Raw publication, segmentation input, remote output, and the
privacy slate remain uncorrected.

## Requirement review

### VIS-2.1 — tested sRGB/linear and image-profile primitives

Implemented in `src/custback/color.py`,
`src/custback/backgrounds.py`, and `src/custback/avatar/store.py`:

- `srgb_eotf` and `srgb_oetf` implement the reference piecewise transfer
  functions. A 256-entry EOTF table accelerates external `uint8` conversion.
- `bgr_u8_to_linear_rgb` and `linear_rgb_to_bgr_u8` make channel order,
  finite-value validation, hard clipping, round-to-nearest, ownership, dtype,
  and contiguity explicit.
- Linear luminance, masked median/quantile helpers, immutable
  `ColorTransform`, independent exposure/WB strength interpolation, and
  bounded transform application use finite linear RGB.
- The Phase-0 highlight decision remains hard clipping: transform
  intermediates retain bounded `[0, 4]` headroom and final encoding clips to
  `[0, 1]`. No unmeasured highlight roll-off was introduced.
- `decode_image_to_srgb_bgr` owns one file descriptor across verification and
  decode, verifies container identity and configured pixel limits, applies
  EXIF `1..8` once, converts valid embedded ICC profiles through ImageCms to
  sRGB, assumes sRGB only when no profile is present, and returns contiguous
  BGR.
- Malformed, non-byte, and present-but-empty ICC profiles are rejected.
  Pillow decompression-bomb signals map to the configured pixel-limit failure.
- `ImageBackdrop`, both upload stores, and background thumbnails use the
  shared decoder. The previous Pillow-validation/OpenCV-decode split is gone.
  Thumbnail and full-size rendering now use the same profile and orientation
  policy.
- The thumbnail APIs pass the upload store's configured image-pixel ceiling
  instead of relying on a divergent implicit limit.

Primary executable evidence:

- `tests/test_color.py` covers sRGB reference vectors, all-code-value
  round trips, endpoint/monotonic behavior, adversarial finite/range
  contracts, luminance/statistics, transform bounds, tagged sRGB,
  Display-P3, Adobe-RGB, CMYK, untagged sRGB, Phase-0 profile fixtures,
  malformed/empty ICC data, EXIF orientation, one-descriptor ownership, format
  mismatch, and pixel bombs.
- `tests/test_background_geometry.py` proves `ImageBackdrop` uses the shared
  decoder and rejects malformed ICC data without a second OpenCV decode.
- `tests/test_avatar_store.py` proves upload validation uses only the shared
  decoder, malformed ICC data is rejected, configured limits reach
  thumbnails, and a color-managed oriented thumbnail matches the full-size
  backdrop within JPEG tolerance.
- Updated API tests replace assumptions about the retired private Pillow /
  OpenCV dual-decoder path with the shared decoder boundary.

### VIS-2.2 — linear-light compositor and light wrap

Implemented in `src/custback/compositor.py` and
`src/custback/backgrounds.py`:

- `composite` has explicit `srgb_legacy` and `linear_srgb` paths with the
  existing legacy mode as the default.
- `linear_srgb` decodes foreground, fitted backdrop, and optional RVM clean
  foreground to linear RGB; applies the same `ColorTransform` to both
  foreground representations; performs clean-foreground edge replacement,
  light wrap, and alpha blending in linear light; then encodes once.
- `composite_linear_predecoded` accepts estimator-owned decoded arrays so an
  `auto` frame does not repeat full-frame EOTF work.
- Exact mask-0 backdrop endpoints remain byte-identical. Mask-1 foreground
  endpoints remain byte-identical for an identity transform. Returned frames
  are owned and contiguous.
- Shape, dtype, contiguity, finite mask/range, light-wrap, transform, and
  paired BGR/linear edge-foreground contracts fail deterministically.
- `srgb_legacy` freezes the historical encoded blend and light-wrap
  arithmetic. An active correction is applied to foreground/RVM pixels in
  linear RGB and encoded once before that legacy blend. Identity correction
  remains byte-identical.
- `BlurBackdrop` deliberately keeps its Gaussian and normalized masked
  Gaussian in encoded BGR. Blur is correction-excluded; changing its blur
  space is outside Phase 2.

Primary executable evidence:

- `tests/test_processing.py` covers the approximately-146 linear half blend,
  exact endpoints, ownership, the frozen legacy reference, legacy correction
  order, one-decode/one-encode behavior, shared camera/RVM transform,
  predecoded entry points, edge-only light wrap, and malformed contracts.
- `tests/test_background_geometry.py` freezes the encoded-space blur result and
  preserves the canonical no-second-fit blur contract.

### VIS-2.3 — mask-aware color estimator

Implemented in `src/custback/color.py` with valid-content metadata supplied by
`src/custback/capture.py` and `src/custback/backgrounds.py`:

- `ColorEstimate`, `ColorTransform`, `ColorSceneSignature`, `ColorReason`, and
  `ColorBehavior` are immutable scalar/result contracts.
- BGR inputs are EOTF-decoded before the bounded 192-pixel-long-edge analysis
  resize. The predecoded entry point produces equivalent estimates.
- The estimator builds the accepted 0.90-confidence, 5x5-eroded source core
  and 19x19-dilated local target annulus, with valid-content global exposure
  fallback.
- Source and target sampling is intersected with half-open capture/backdrop
  `content_rect` values. Confidence denominators use valid content, so
  `contain` black bars and narrow valid rasters do not dilute or bias the
  estimate.
- The implementation enforces the accepted 96-sample, 0.02 near-black, 0.98
  near-clip, 0.30 neutral-saturation, 0.45 confidence, 0.86–1.16 gain, and
  configured exposure/strength bounds.
- Exposure and WB confidence remain separate. A reliable exposure can return
  exposure-only when no qualified local neutral relationship exists.
- Result precedence is deterministic:
  `invalid` input before mode exclusion; excluded valid modes; insufficient
  source/coverage/confidence; invalid target; clipped/black sample failure;
  solid-saturated or insufficient-neutral exposure-only; then `ok`.
- Histogram-looking studio-range pixels are never reinterpreted as limited
  range. The estimator is pure, uses bounded workspace, and retains no frame
  history.

Primary executable evidence:

- `tests/test_color.py` covers Phase-0 selected-math and hue/chroma gates,
  symmetric exposure recovery, external/predecoded parity, the analysis cap,
  contain/narrow valid-content behavior, exact reason precedence, excluded
  modes without analysis, 20% core contamination, resolution equivalence,
  and the no-histogram-range-inference rule.
- `tests/test_background_geometry.py` covers matching-plan
  `content_rect`, stale-plan rejection, contain padding exclusion, and
  full-canvas color/blur providers.

### VIS-2.4 — temporal adaptation, confidence, and resets

Implemented by `ColorHarmonizer` in `src/custback/color.py`:

- Filtering uses monotonic elapsed time and
  `alpha = 1 - exp(-dt / tau)`, followed by parameter deadbands and
  per-second slew limits.
- The ratified constants are:
  0.010 EV / 0.005 log2 deadbands; 0.50 EV/s / 0.20 log2/s steady slew;
  1.50 EV/s / 0.40 log2/s fast slew; 1.0 s fast acquisition;
  `clamp(tau/4, 0.05, 0.20) s` fast tau; 0.5 s low-confidence freeze;
  1.5 s stale-decay tau; 5.0 s exact clear; 0.75 EV / 0.20 log2 scene-cut
  thresholds; and live-camera steady tau/slew multipliers of 2.0/0.5.
- Startup, reset, and a capture-generation change begin at identity. The first
  reliable frame seeds scalar statistics without flashing its transform.
- A scene-cut frame holds the prior bounded transform and discards the cut
  estimate; subsequent post-cut samples fast-acquire.
- Low confidence or an estimator exception uses identity when there is no
  reliable history, otherwise freezes then decays monotonically to exact
  identity.
- Same-timestamp updates are idempotent, backwards time is rejected, and a
  five-second gap clears stale state instead of taking a giant EMA step.
- `clone`, `snapshot`, and live state contain only bounded scalars/signatures;
  no full frame or growing history is retained.

Primary executable evidence:

- `tests/test_color.py` covers equivalent 15/30/60-FPS convergence, static
  jitter gates, scene-cut hold/settling/clamps, freeze/decay/exact clear,
  dense/sparse low-confidence equivalence across the freeze boundary,
  no-history errors, first-estimate seeding after generation reset, long-gap
  resets, idempotence, clone independence, slower live-camera steady response,
  10,000-update memory bounds, and bounded non-retained analysis workspace.

### VIS-2.5 — transactional pipeline integration

Implemented in `src/custback/pipeline.py`:

- `_Resources` owns the live `ColorHarmonizer` and reset token.
  `_Activation` owns a fresh staged harmonizer whenever `_color_state_key`
  changes.
- The invalidation key includes camera/canvas geometry, backdrop visual/source
  identity and geometry, blend space, correction policy, and segmentation
  identity. The frame-time token adds capture generation, delivered-geometry
  generation, valid foreground content, provider identity, the fitted
  backdrop's immutable transform plan/source size, visual generation, and
  canvas.
- Candidate trial uses a cloned staged harmonizer and detached/synthetic
  visual inputs with live tracking disabled. It cannot advance the live EMA,
  video position, mask state, backdrop cache, or reset token.
- Commit swaps configuration, version, resources, visual generation, provider
  policy, harmonizer, and token at one frame boundary. An install exception
  restores the complete prior tuple. Conflict, cancellation, timeout, and
  failed trial discard staged state.
- Local processing segments/refines the original uncorrected canonical frame,
  obtains the final fitted backdrop and valid rectangles, decodes each color
  input once, estimates/updates once, and passes one bounded transform to the
  selected compositor.
- The same transform is applied to camera foreground and RVM
  `last_foreground`. Only pixels selected through the mask are rendered as
  corrected foreground.
- Correction preparation is bypassed for correction `off` and for
  `passthrough`, `blur`, `color`, and `remote`. Remote/privacy processing
  bypasses the estimator before any color preparation.
- Raw hub publication and fingerprints retain the normalized uncorrected
  frame. A repeated output reuses the last guarded frame and never advances
  the harmonizer.
- Estimator exceptions follow `on_error` and render a bounded identity or
  freeze/decay result. A photometric application `ColorError` retries the
  frame with identity. Structural compositor errors and invalid final output
  remain strict.
- `color_correction_ms` is a separate deterministic stage timing bucket and is
  zero for ineligible frames. Detailed public estimator state remains
  VIS-3.1.

Primary executable evidence in `tests/test_pipeline.py`:

- `test_color_correction_eligibility_is_explicit_and_bypasses_to_identity`;
- `test_active_correction_preserves_raw_hub_bytes_and_is_mask_local`;
- `test_auto_legacy_reuses_estimator_decodes_for_foreground_and_rvm_edge`;
- `test_rvm_edge_foreground_receives_the_same_color_transform`;
- `test_auto_correction_leaves_remote_candidate_or_privacy_slate_untouched`;
- `test_estimator_exception_is_fail_soft_with_identity_render`;
- `test_color_application_error_retries_only_with_identity`;
- `test_malformed_compositor_or_output_remains_strict`;
- `test_repeat_output_does_not_advance_harmonizer`;
- `test_live_reset_consumes_first_reliable_estimate_with_production_harmonizer`;
- `test_live_backdrop_geometry_token_change_hard_resets_harmonizer`;
- `test_color_correction_has_a_separate_deterministic_timing_bucket`;
- `test_successful_color_state_commit_installs_pristine_harmonizer`;
- `test_unrelated_hot_commit_preserves_harmonizer_object_and_snapshot`;
- `test_rejected_activation_paths_do_not_mutate_live_harmonizer`;
- `test_preparation_timeout_does_not_mutate_live_harmonizer`;
- `test_queued_ack_timeout_and_late_cancel_ack_preserve_live_harmonizer`; and
- the existing failed-install, hot-geometry, conflict, timeout, raw,
  remote-replay, privacy-slate, and exact-canvas regressions.

## Resolved Phase 2 decisions

1. Still-image ICC conversion is fail-closed. A present malformed or empty
   profile is not treated as an untagged sRGB image.
2. Still validation and decode share one Pillow-owned descriptor; OpenCV is not
   a second still-image decoder. Full-size and thumbnail consumers share the
   same normalization operation.
3. Runtime BGR is decoded before analysis resizing. Eligible `auto` frames
   share those decoded arrays between estimator and compositor.
4. Estimator reasons follow the fixed precedence recorded in the ADR. Expected
   low confidence is a normal temporal input.
5. Synthetic contain padding is excluded through exact valid-content
   rectangles. Missing or stale fitted-provider metadata fails rather than
   guessing.
6. `auto + srgb_legacy` is supported: foreground correction is linear, while
   alpha/light-wrap remain the frozen encoded legacy operations. `auto` does
   not imply a blend-space change.
7. Blur Gaussian operations remain encoded-space compatibility behavior.
8. Temporal constants, cut-frame hold, fast acquisition, freeze/decay, exact
   stale clear, live-camera slowdown, and source-generation reset are now
   ratified implementation behavior.
9. Harmonizer state is a transactional resource. Candidate trial uses cloned
   state; relevant commits install pristine state; unrelated commits preserve
   it; rejected or rolled-back changes cannot mutate it.

## Privacy and failure review

- Segmentation consumes the uncorrected canonical camera frame.
- Raw publication occurs before local correction and remains the authority for
  raw fingerprints and remote input.
- Remote candidates, remote output, and the input-independent privacy slate
  bypass correction. No estimator fallback may substitute raw or local
  composite pixels into remote output.
- A low-confidence estimate is not a processing failure; it selects identity,
  exposure-only, freeze, or decay according to state.
- Unexpected estimator failures are bounded and transition-logged. Photometric
  application failures can retry identity, but malformed shapes, masks,
  compositor results, and final output remain contract errors.
- Harmonizer state cannot cross a capture generation, relevant geometry/source
  generation, visual-policy generation, or correction-policy change.

## Explicit Phase 3 and Phase 4 deferrals

| Backlog owner | Deliberately not claimed by Phase 2 |
| --- | --- |
| VIS-3.1 | Full public phase/reason/confidence/transform/reset counters and transition telemetry. Phase 2 exposes only the separate timing bucket, internal snapshots, and bounded fallback transitions. |
| VIS-3.2 | Web UI correction controls, operator explanations, and presentation diagnostics. The strict API/config fields already exist, but UI qualification does not. |
| VIS-3.3 | Trustworthy video/container/output color metadata, conversion matrices, nominal-range signaling, and Windows Media Foundation color attributes. OpenCV video/camera inputs retain the documented sRGB/BT.709 full-range assumption. |
| VIS-3.4 | Read-back-verified camera exposure/WB lock or manual-control characterization. Phase 2 performs no continuous hardware-control writes. |
| VIS-4.1 | The expanded black-box end-to-end visual, temporal, privacy, and transport matrix. Phase 2 focused integration tests do not replace that release-level matrix. |
| VIS-4.2 | Calibrated p95 CPU/memory budgets, long-run drift, real camera/backend behavior, and cross-platform visual qualification. No local observation below is a rollout gate. |
| VIS-4.3 | Schema bump, target-default flip, release notes, operator migration, staged rollout, and rollback drill. Schema v1 remains `stretch` / `srgb_legacy` / correction `off`. |

## Performance observation

VIS-4.2 owns calibrated performance acceptance. The following is a non-gating
local observation, not a budget or cross-platform qualification:

| Scenario | Resolution | p50 | p95 | Peak/retained memory |
| --- | --- | ---: | ---: | ---: |
| `linear_srgb`, correction off | 1280×720 | 77.989 ms | 80.636 ms | 104.594 MiB / 0.250 KiB |
| `linear_srgb`, correction auto | 1280×720 | 84.514 ms | 87.455 ms | 115.146 MiB / 0.648 KiB |
| `srgb_legacy`, correction auto | 1280×720 | 72.746 ms | 74.216 ms | 90.567 MiB / 0.602 KiB |
| Bounded predecoded estimator only | 1280×720 input | 7.564 ms | 8.053 ms | 2.638 MiB / 0.371 KiB |

The observation used Ubuntu 26.04 / Linux 7.0.0-28-generic on a 12th Gen
Intel Core i7-12700H, Python 3.14.4, NumPy 2.5.1, OpenCV 5.0.0, and Pillow
12.3.0. OpenCV used one thread. Inputs were deterministic 1280×720 ramps with
a soft elliptical mask; every scenario had 10 warm-up and 60 measured
iterations. Latency used `time.perf_counter`; memory is a one-call
`tracemalloc` peak and retained allocation after collection. The auto
measurement covers decode, the predecoded estimator, and the predecoded
compositor. The estimator row covers only its bounded predecoded entry point.

## Verification record

All results below were collected from the accepted tree on 2026-07-30:

- Host full suite: `.venv/bin/pytest -q` completed with **1370 passed,
  3 skipped** in 19.25 seconds. API lifecycle tests were rerun with loopback
  access outside the restricted sandbox. The host used Python 3.14.4, NumPy
  2.5.1, OpenCV 5.0.0, Pillow 12.3.0, Pydantic 2.13.4, and pytest 9.1.1.
- Focused Phase-2 suites completed with **477 passed** in 10.47 seconds:
  `tests/test_color.py`, `tests/test_processing.py`,
  `tests/test_background_geometry.py`, `tests/test_capture_geometry.py`,
  `tests/test_avatar_store.py`, `tests/test_pipeline.py`,
  `tests/test_api.py`, and `tests/test_avatar_api.py`.
- The exact minimum-dependency lane ran in `python:3.10-slim` with Python
  3.10.20, NumPy 1.24.0, OpenCV 4.8.1, Pillow 10.0.0, Pydantic 2.7.0, and
  pytest 8.0.0: **1369 passed, 4 skipped** in 34.30 seconds. Its only output
  beyond test results was two known Pydantic protected-namespace warnings.
- Full Ruff lint and formatting checks passed. Full Pyright completed with
  **0 errors**. `git diff --check` passed.
- Both workflow YAML files parsed successfully. `npm test` completed with
  **141 passed, 0 failed, 2 intentional TODO** release-remediation cases.
- `npm run release:check -- --quick` reached only the two pre-existing
  authorizing release blockers, REL-01 and WIN-01. Phase 2 does not claim or
  bypass those blockers.
- The final non-authorizing installed-package smoke passed. It built exactly
  one wheel and one sdist from a clean staged tree; checked exact manifests,
  metadata, entry points, dependencies, licenses, and reviewed allowlists;
  confirmed `custback/color.py` in installed artifacts; installed and imported
  both artifacts in isolated environments; ran `pip check`; and exercised the
  installed CLI/avatar smoke paths.
- This Linux run did not execute the Windows compile/native-output job or use a
  live Windows camera/backend. The CI job definition and its static/source
  regressions include the Phase-2 files. Live platform, backend-color, camera,
  long-run, and calibrated performance qualification remain VIS-4.2.

No Phase-3 or Phase-4 result is implied by this acceptance. Compatibility
defaults remain schema-v1 `stretch`, `srgb_legacy`, and correction `off`.
