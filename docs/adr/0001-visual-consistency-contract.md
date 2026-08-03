# ADR 0001: Visual consistency frame, geometry, and color contract

- Status: Accepted
- Date: 2026-07-30
- Last amended: 2026-07-30 (Phase 3 implementation contract)
- Scope: core/stage-1 camera pipeline
- Backlog: `AUTO_COLOR_CORRECTION_AND_SCALING_BACKLOG.md`, VIS-0.1
- Phase 2 review: `docs/visual-consistency-phase2-implementation-review.md`
- Phase 3 review:
  `docs/visual-consistency-phase3-implementation-review.md`

## Context

The main camera and backdrops currently reach the processing pipeline by
different geometric paths. A main-camera mode mismatch is resized independently
in X and Y, while image, video, live-camera, and blur backdrops preserve aspect
ratio and center-crop. Camera acquisition dimensions also implicitly determine
the processing canvas, raw/remote frame dimensions, and output dimensions.

Before this program, color had similarly implicit semantics. OpenCV sources
were treated as untagged `uint8 BGR`, still-image profiles were not applied,
and the compositor blended encoded values. The remote/avatar path has an
existing fail-closed privacy boundary that must not be weakened while those
contracts are made explicit.

This ADR fixes the boundary and ownership rules required by VIS-0.1. VIS-0.3
has selected the estimator, bounds, confidence gates, and mode eligibility from
the reproducible evidence in
`docs/visual-consistency-phase0-evidence.md`. The initial implementation
defaults remain compatibility-preserving.

The Phase 2 amendment records the implemented image-decode, linear-light,
estimator, temporal, and transactional behavior. It does not authorize the
schema/default rollout reserved for VIS-4.3.

The Phase 3 amendment records frame-aligned public telemetry, operator-control
lifecycle, metadata-aware local-video normalization, truthful output signaling,
and the preserve-only camera-control decision. Generated fixtures and
read-only capability reports do not establish live camera or consumer
qualification. Final integration/package evidence is tracked in the Phase 3
review, and target-default rollout remains reserved for Phase 4.

Normative words such as **must**, **must not**, and **may** describe the target
implementation contract, not necessarily the behavior of the code at the date
of this ADR.

## Decision summary

| Concern | Initial compatible behavior | Qualified target behavior |
| --- | --- | --- |
| Camera acquisition | `camera.width` and `camera.height` request a device mode | Unchanged |
| Canonical canvas | Optional paired `output.width`/`height`; when absent, resolve to the camera request | Unchanged |
| Camera fit | `stretch` | `cover`, after geometry qualification and versioned default migration |
| Backdrop fit | `cover`, center anchor | `cover`, center anchor |
| Blend space | `srgb_legacy` | `linear_srgb`, after golden/performance qualification and versioned default migration |
| Foreground correction | `off` | `auto`, only after later integration, visual, performance, privacy, and rollback gates pass |
| Migration | A persisted document without a schema version has legacy-v1 semantics | New defaults are introduced only by a new explicit schema version |

The target column is not permission to flip a default. Each flip is a separate,
reversible release change after its named gates pass.

## Canonical terms and invariants

- An **acquisition size** is the width and height requested from a main camera.
  It does not define what the device will actually deliver.
- A **delivered size** is the size of the decoded camera frame read from the
  backend.
- An **oriented size** is the size after metadata orientation and configured
  right-angle rotation.
- The **canonical canvas** is the single resolved processing/output width and
  height. It has square pixels and positive dimensions.
- `camera.width` and `camera.height` remain acquisition requests.
- New `output.width` and `output.height` fields are paired: both are set or
  both are absent. When absent, the canonical canvas resolves to
  `(camera.width, camera.height)`.
- A single `resolved_output_size(config)` helper owns that resolution rule.
- The main camera must be normalized to the canonical canvas before raw
  publication or segmentation. Every mask, fitted backdrop, local composite,
  privacy slate, remote result, API frame, and output frame then has exactly
  that canvas.
- Geometry is performed once at a source boundary. Downstream stages must
  validate exact size rather than silently fit a second time.
- Logical external frames are non-empty, C-contiguous `H x W x 3` NumPy arrays
  with dtype `uint8` and channel order B, G, R. Alpha is never implicit in this
  representation.
- Masks are separate, finite, C-contiguous `H x W float32` arrays in `[0, 1]`.

## Source, processing, and sink ownership

This table assigns one logical representation, color assumption, and geometry
owner to every core source and sink. JPEG and BGRX are transports around the
logical external-frame contract, not additional working color spaces.

| Boundary | Logical representation and color assumption | Geometry owner |
| --- | --- | --- |
| Main OpenCV camera, including synthetic capture | Decoded `uint8 BGR`; full-range display-referred sRGB/BT.709 primaries are assumed because backend color metadata is not trustworthy or retained | Capture validates, applies configured orientation/mirror/fit, and publishes exactly the canonical canvas |
| Static image backdrop | Pillow-decoded pixels; embedded valid ICC profile is converted to sRGB, untagged input is assumed sRGB, then converted once to `uint8 BGR` | Image provider applies EXIF orientation exactly once, then the shared backdrop geometry plan |
| Video backdrop | PyAV retains per-frame/codec declarations. Tagged BT.601/BT.709 matrix, limited/full range, BT.709/BT.470BG/SMPTE170M primaries, and sRGB/BT.709 SDR transfer are normalized to full-range display-referred sRGB `uint8 BGR`. Unsupported tagged HDR/wide-gamut declarations are rejected. Missing fields use the observable schema-v1 legacy BT.709/full/BT.709/sRGB assumption; explicit overrides are local-file-only and visible in status. | Video provider resolves metadata/override/legacy color and qualified pure right-angle display-matrix rotation, rejects mirrored/non-right-angle matrices, then applies the shared backdrop geometry plan |
| Live-camera backdrop | Decoded `uint8 BGR` under the same untagged OpenCV assumption as the main camera | Camera-backdrop provider and the shared backdrop geometry plan |
| Solid-color backdrop | Exact-canvas `uint8 BGR`; configured code values are sRGB code values | Color provider allocates the canonical canvas; no fit |
| Blur backdrop | The already normalized main-camera frame and its mask | No second source fit; blur provider returns the same canonical canvas |
| Segmentation input | Uncorrected canonical `uint8 BGR`, converted to the model's expected display-referred RGB form | No geometry ownership; shape must already equal the canvas |
| Segmentation output / RVM clean foreground | Mask as defined above; optional clean foreground is canonical `uint8 BGR` under the input color assumption | Segmenter must return exact-canvas results; it must not fit |
| Local compositor | External BGR inputs; `linear_srgb` mode decodes to bounded internal linear RGB, while `srgb_legacy` is a temporary encoded-value compatibility path | No fit; exact-shape validation only |
| Raw `FrameHub` slot and raw fingerprint input | Pixel-identical canonical, uncorrected BGR values from capture | Capture owns geometry; hub stores latest only |
| Raw renderer WebSocket / avatar input | JPEG serialization of the canonical uncorrected raw frame; JPEG loss means transport bytes and decoded pixels are not claimed to be byte-identical | Core supplies canonical dimensions; renderer must preserve those output dimensions |
| Remote/avatar return | JPEG decoded to canonical `uint8 BGR`, assumed full-range display-referred sRGB | Renderer owns content; core validates exact size and **must never fit it** |
| Privacy slate | Fixed, input-independent, opaque canonical `uint8 BGR` code values | Core allocates exact canvas; it is not corrected or fitted |
| Processed `FrameHub` slot, snapshot, MJPEG, and output WebSocket | Canonical output BGR; HTTP/WebSocket JPEG is a lossy serialization of it | Pipeline owns final canvas; encoders must not resize |
| Local HighGUI preview | A copy of the processed frame; UI overlays may alter only the displayed copy | No fit; window scaling is presentation-only and not pipeline geometry |
| Null and pyvirtualcam outputs | Canonical `uint8 BGR`, full-range display-referred sRGB. pyvirtualcam is explicitly opened with `PixelFormat.BGR`; its platform consumer exposes no portable primaries/transfer/range signaling API, so application interpretation remains a Phase-4 live qualification boundary. | Validate exact canvas on every send; no fit |
| Windows shared ring | Canonical BGR converted mechanically to opaque BGRX | Python writer validates exact ring dimensions; no fit |
| Windows Media Foundation consumer | Opaque RGB32/BGRX sample with the ring's mechanically copied sRGB pixels; the media type declares BT.709 primaries, sRGB transfer, and nominal 0..255 range | Native media source may expose only a canvas/FPS combination it can deliver exactly; overlap-copy is not a valid fit |

Camera full-range sRGB and missing video metadata remain explicit,
observable interoperability assumptions, not claims about every driver or
codec. Histogram shape alone must never change color interpretation: in
particular, values mostly between 16 and 235 are **not** proof of limited
range. Video resolution is deterministic: an explicitly configured local-file
override owns its selected field; every `auto` field uses a trusted frame tag,
then a codec-context tag, then the documented per-field legacy assumption. No
pixel statistic participates.

## Geometry contract

### Operation order

Every source uses this order:

1. securely decode and validate positive `H x W x 3 uint8` pixels;
2. apply source metadata orientation exactly once, including mirrored EXIF
   orientations where applicable;
3. apply configured clockwise rotation (`0`, `90`, `180`, or `270` degrees);
4. apply configured horizontal mirror in the resulting viewer coordinate
   system;
5. fit to the canonical canvas.

Rotation is never inferred from portrait dimensions. Mirroring after rotation
means "horizontal" always refers to the displayed viewer X axis. Applying the
fit last makes anchors use those same displayed X/Y axes.

### Fit and anchor math

Let the oriented source be `Ws x Hs`, the target be `Wt x Ht`, and
`ax, ay` be finite anchors in `[0, 1]`. Rectangles below are half-open.
Anchor `0` preserves/places content at the left or top; anchor `1` preserves or
places it at the right or bottom.

For `cover`:

1. `s = max(Wt / Ws, Ht / Hs)`.
2. `Rw = max(Wt, ceil(Ws * s))` and
   `Rh = max(Ht, ceil(Hs * s))`.
3. Resize to `Rw x Rh`.
4. Let `Ex = Rw - Wt`, `Ey = Rh - Ht`.
5. Crop origin is
   `Cx = floor(ax * Ex)`, `Cy = floor(ay * Ey)`, clamped to the valid
   integer range.
6. Return crop `[Cx, Cx + Wt) x [Cy, Cy + Ht)`.

The `ceil` rule ensures the crop cannot be a pixel short. Implementations must
avoid floating-point near-integer errors, by rational/integer arithmetic or an
equivalent clamped calculation.

For `contain`:

1. `s = min(Wt / Ws, Ht / Hs)`.
2. `Rw = max(1, min(Wt, floor(Ws * s)))` and
   `Rh = max(1, min(Ht, floor(Hs * s)))`.
3. Resize to `Rw x Rh`.
4. Let `Px = Wt - Rw`, `Py = Ht - Rh`.
5. Place the resized content at
   `L = floor(ax * Px)`, `T = floor(ay * Py)`.
6. Fill the remaining target pixels with opaque sRGB black `(0, 0, 0)` BGR.

The transform plan must identify valid content separately from synthetic
padding so color statistics do not treat black bars as scene illumination.
A configurable pad color is deliberately excluded from the first contract;
revisit it only if a real presentation requirement cannot be met by `cover`.

For `stretch`, resize directly to `Wt x Ht` with independent
`Sx = Wt / Ws` and `Sy = Ht / Hs`. Anchors have no effect. `stretch` exists for
legacy compatibility; new behavior must never select it implicitly.

Exact oriented size is a pixel-preserving no-op except when a copy is required
to establish C-contiguous ownership. Proportional downscales use
`INTER_AREA`; proportional upscales use `INTER_LINEAR`. For a mixed-axis
`stretch`, resize the shrinking axis first with `INTER_AREA`, then resize the
growing axis with `INTER_LINEAR`; an unchanged axis is not resampled. This
freezes direction-aware behavior without asking one OpenCV interpolation mode
to serve opposite directions.

### Worked examples

Coordinates are expressed after orientation/mirror. "Center" means
`ax = ay = 0.5`.

| Source to target | Plan | Center result | Anchor-edge result |
| --- | --- | --- | --- |
| 640x480 (4:3) to 1280x720 (16:9), `cover` | `s=2`; raster 1280x960; vertical excess 240 | Crop starts at `(0,120)` and returns 1280x720 | `ay=0` starts at row 0; `ay=1` starts at row 240 |
| 640x480 to 1280x720, `contain` | `s=1.5`; raster 960x720; horizontal padding 320 | 160 black columns on the left and right | `ax=0` puts all 320 columns on the right; `ax=1` puts them on the left |
| 640x480 to 1280x720, `stretch` | `Sx=2`, `Sy=1.5` | Direct 1280x720 resize; a circle becomes horizontally elongated | Anchors have no effect |
| 1920x1080 (16:9) to 640x480 (4:3), `cover` | `s=4/9`; raster 854x480; horizontal excess 214 | Crop starts at `(107,0)` | `ax=0` starts at column 0; `ax=1` starts at column 214 |
| 1920x1080 to 640x480, `contain` | `s=1/3`; raster 640x360; vertical padding 120 | 60 black rows at top and bottom | `ay=0` puts all 120 rows at bottom; `ay=1` puts them at top |
| 720x1280 portrait to 1280x720, `cover` | `s=16/9`; raster 1280x2276; vertical excess 1556 | Crop starts at `(0,778)` | `ay=0` starts at row 0; `ay=1` starts at row 1556 |
| 720x1280 portrait, then 90-degree clockwise rotation, to 1280x720 | Oriented size is exactly 1280x720 | No resize or crop; a configured mirror acts horizontally on this landscape result | Anchors have no effect |
| 641x479 to odd target 1281x721, `cover` | `s=1281/641`; raster 1281x958; vertical excess 237 | Crop starts at row 118, leaving 119 cropped rows below | `ay=0` starts at row 0; `ay=1` starts at row 237 |
| 641x479 to 1281x721, `contain` | `s=721/479`; raster 964x721; horizontal padding 317 | 158 columns left and 159 right | `ax=0` places all 317 right; `ax=1` places all 317 left |

## Color contract

### External and internal representations

- External BGR code values are full-range (`0..255`), display-referred sRGB
  with D65 white and BT.709/sRGB primaries.
- Static images with a valid embedded ICC profile are converted to sRGB before
  entering the BGR boundary. Untagged images use the declared sRGB assumption.
  A malformed embedded profile is rejected deterministically rather than
  silently decoded under a different profile.
- The internal photometric representation is finite, C-contiguous
  `float32 RGB` in linear sRGB. EOTF-decoded inputs are in `[0, 1]`.
  Photometric intermediate values are hard-clamped to `[0, 4]`. The Phase 0
  safe encoding fallback is a hard clip to `[0, 1]`, followed by the sRGB OETF,
  round-to-nearest, and external `uint8 BGR` conversion. Phase 2 retains that
  fallback. Replacing it requires a separately measured and ratified highlight
  roll-off/clipping policy.
- NaN, infinity, negative values, integer wraparound, and implicit BGR/RGB
  channel swaps are contract violations.

Segmentation continues to consume the uncorrected display-referred canonical
camera frame. Harmonization is estimated only after the segmentation mask and
final fitted backdrop exist. It modifies the local foreground for rendering,
not the model input or backdrop. The same foreground transform must be applied
to the camera foreground and any RVM clean/edge foreground.

Harmonization is a software-rendering operation. The core must not continuously
drive camera exposure, gain, gamma, or white-balance controls in response to
the estimator; backend meanings differ and a second automatic control loop can
oscillate against the camera firmware. Reading capabilities or offering a
qualified explicit lock/manual policy is separate follow-up work.

`compositing.blend_space` is a **temporary compatibility switch**, not a
permanent creative control. `srgb_legacy` preserves the encoded-value
compositor while qualification occurs. `linear_srgb` decodes foreground,
backdrop, light-wrap input, and model foreground, performs photometric
operations and alpha compositing in linear light, then encodes once.

The switch must remain accepted through the first stable release after
`linear_srgb` becomes the default. It may then be removed only through an
announced schema migration that converts or rejects explicit legacy pins.
Revisit permanent support if real downstream workflows demonstrate a need for
encoded-space byte compatibility that cannot be served by pinning an older
schema/release.

### Accepted harmonization policy

VIS-0.3 compared exposure-only matching, bounded exposure plus diagonal WB,
and aggressive per-channel moment transfer. The bounded candidate was the only
one that materially reduced both luminance and neutral-axis error while
staying inside the protected skin/clothing hue and chroma gates. The evidence
report and deterministic harness are the numeric authority for this decision.

The accepted estimator contract is:

- estimate robust log-luminance exposure plus diagonal white-balance /
  von-Kries-style gains in linear RGB;
- analyze a downscaled copy with a `192 px` long edge;
- take source samples from mask confidence `>= 0.90`, eroded once with a
  `5x5` kernel at analysis resolution;
- take target samples from a `19x19` dilated annulus outside the subject, with
  a valid-content global backdrop fallback for exposure only;
- require at least `96` usable source and target samples;
- exclude non-finite samples, luminance `Y <= 0.02`, and samples with any
  channel `>= 0.98`; WB additionally requires saturation `<= 0.30`;
- require exposure confidence `>= 0.45`, combining bounded sample count,
  usable-sample ratio, and mask coverage; require WB confidence `>= 0.45` after
  additionally weighting local neutral-sample availability;
- clamp raw exposure to the configured limit, which defaults to
  `-0.85..+0.85 EV` and has a schema safety ceiling of `1.0 EV`, then
  interpolate it independently with default `strength: 0.50`;
- clamp raw per-channel WB gains to `0.86..1.16`, then interpolate their
  logarithms independently with default `white_balance_strength: 0.50`;
  WB strength is not multiplied by exposure strength;
- return identity/no-update when confidence or source samples are
  insufficient; use exposure-only when exposure is valid but a local neutral
  WB relationship is not;
- preserve skin-like patches within `5 degrees` hue and `12%` normalized
  chroma drift, and saturated clothing within `8 degrees` hue and `15%`
  normalized chroma drift on the Phase-0 fixtures.

`adaptation_time_s: 0.8` is the ratified initial bounded configuration value
for the temporal state machine. It was not measured by the instantaneous
VIS-0.3 estimator; VIS-2.4 validates its elapsed-time runtime behavior at
15/30/60 FPS. A different value requires an explicit ADR/config amendment,
and `auto` remains ineligible to become the default until the later Phase-4
visual, performance, privacy, platform, and rollback gates pass.

Full histogram transfer, LAB mean/standard-deviation transfer, and any method
whose primary effect is to reproduce the backdrop's color distribution are
rejected as the production default. Histogram-only full/limited-range
detection is prohibited.

Accepted mode eligibility is:

| Background mode | Correction behavior |
| --- | --- |
| `passthrough` | Identity; output remains pixel-identical to canonical raw BGR |
| `blur` | Identity; foreground and backdrop derive from the same capture |
| `remote` | Identity in the core; remote output is never corrected |
| `color` | Identity in the initial implementation; a solid color is not evidence of scene illumination |
| `image` | Exposure plus restrained WB when confidence is sufficient |
| `video` | Exposure plus restrained WB when confidence is sufficient; scene-cut rules apply |
| `camera` | Slow, bounded exposure plus restrained WB when confidence is sufficient |

For otherwise eligible image/video/camera modes, a saturated target or
all-one mask may select exposure-only; all-zero/tiny masks, a clipped
foreground core, insufficient samples, and exposure confidence below `0.45`
select identity/no-update. WB confidence below `0.45` selects exposure-only
when exposure confidence remains valid. Estimator work is bounded by the fixed
analysis raster and retains no frame history.

### Phase 2 implementation ratification

#### Decode, conversion, and reuse ownership

The implemented still-image boundary uses Pillow and LittleCMS through one
shared `decode_image_to_srgb_bgr` operation:

- the caller supplies the expected container format and decoded-pixel ceiling;
- one owned file descriptor is used for container verification and pixel
  decode, with file identity checked before and after those operations;
- Pillow decompression-bomb errors and warnings are normalized to the caller's
  configured pixel-limit error;
- EXIF orientation `1..8` is applied exactly once;
- a present ICC profile, including a present-but-empty value, must parse and
  convert successfully to sRGB; malformed profiles are rejected;
- an untagged image is explicitly assumed sRGB; and
- the returned boundary value is finite, owned, C-contiguous `uint8 BGR`.

Image backdrop rendering, upload validation, and background thumbnail
rendering call this same operation. None validates with Pillow and then
reopens the path through OpenCV. Full-size rendering and thumbnails therefore
share the same orientation/profile policy, while each request retains its own
configured pixel ceiling.

For runtime photometric work, external BGR is EOTF-decoded before any analysis
downscale. Area resampling therefore averages linear-light samples, not sRGB
code values. In an eligible `auto` frame the pipeline decodes the camera,
fitted backdrop, and optional RVM foreground once, shares those linear arrays
with the estimator and compositor, and encodes transformed output at the
appropriate compositor boundary. The estimator's predecoded entry point must
remain equivalent to its external-BGR entry point.

#### Estimator outcome precedence and valid content

`ColorReason` is a deterministic outcome, not a severity ranking. The first
applicable rule wins in this order:

1. Invalid frame, mask, mode, policy, or content-rectangle contracts return
   `invalid`, even when the named background mode would otherwise be excluded.
2. Valid `passthrough`, `blur`, `color`, and `remote` inputs return
   `mode-excluded` without decoding or analysis.
3. In an eligible mode, insufficient eroded foreground samples, valid-content
   coverage, or exposure confidence returns `insufficient-mask`.
4. An insufficient target region after local/global selection returns
   `invalid`.
5. Too few usable non-black/non-clipped samples returns `clipped`.
6. A reliable exposure estimate without qualified local neutral samples
   returns exposure-only: `solid-saturated` when the target satisfies the
   solid/saturated test, otherwise `insufficient-neutral`.
7. A reliable exposure and WB estimate returns `ok`.

Expected low-confidence results are state-machine inputs, not exceptions.
Unexpected estimator errors are normalized to the same `invalid`
identity/freeze-and-decay policy.

Both capture geometry and fitted backdrop geometry expose a half-open valid
`content_rect` for the exact frame returned. Estimator source/target masks are
intersected with these rectangles, and coverage is measured against valid
content rather than the full canvas. This excludes synthetic `contain` padding,
including cases where valid content occupies only a narrow fraction of the
canvas. Intrinsically exact-canvas color and blur providers report the full
canvas. A fitted provider with no matching current transform plan fails rather
than guessing that padding is valid.

#### Compatibility behavior

`color_correction.mode: auto` and `compositing.blend_space: srgb_legacy` are a
supported compatibility combination. `auto` never silently changes the blend
space. The bounded foreground and optional RVM transforms are still applied in
linear RGB and re-encoded once; historical alpha blending and light wrap then
remain in encoded BGR. With correction off or an identity transform, the
legacy path remains byte-identical. `linear_srgb` instead performs transform,
RVM edge replacement, light wrap, and alpha compositing in linear light before
one final encode.

`BlurBackdrop` deliberately retains its Gaussian or normalized masked Gaussian
in encoded BGR. Blur mode is correction-excluded and derives both sides from
the same capture; changing its blur space would create an unrelated visual
change and break the compatibility baseline. A linear-light blur requires
separate evidence and an explicit contract amendment.

## Temporal state and scene cuts

Temporal state is owned by the active pipeline resource/config generation. It
contains bounded scalar parameters and bounded analysis signatures only; it
must not retain full-frame history.

The following are hard reset boundaries. The replacement state begins at
identity and no parameter from the old state may be reused:

- main-camera capture generation change or reconnect;
- delivered-size change that creates a new geometry plan;
- canonical canvas, camera rotation, mirror, fit, or anchor change;
- backdrop mode, asset/source identity, provider generation, fit, or anchor
  change;
- correction mode, bounds, strength, adaptation policy, or blend-space change;
- explicit correction disable/enable and pipeline restart.

A same-provider video/backdrop scene cut is a soft reset, not a cross-fade of
pre- and post-cut statistics. The cut frame uses the previous already-bounded
applied transform; its estimate is discarded. Subsequent frames enter a
bounded fast-acquisition state using only post-cut samples, then return to the
normal time constant. If confidence remains low, the state follows the
freeze/decay rule below. This avoids applying an arbitrary cut-frame estimate
and avoids an unconditional one-frame jump to identity.

The implemented temporal constants are:

| Control | Ratified value |
| --- | ---: |
| Configured normal adaptation time | `0.8 s` default; schema range `0.05..10 s` |
| Exposure / WB-log2 deadband | `0.010 EV` / `0.005` |
| Steady exposure / WB-log2 slew | `0.50 EV/s` / `0.20/s` |
| Fast exposure / WB-log2 slew | `1.50 EV/s` / `0.40/s` |
| Fast-acquisition duration | `1.0 s` |
| Fast time constant | `clamp(adaptation_time_s / 4, 0.05 s, 0.20 s)` |
| Low-confidence freeze | `0.5 s` |
| Stale decay time constant | `1.5 s` |
| Exact-identity stale clear / long-gap reset | `5.0 s` |
| Scene-cut luminance threshold | `0.75 EV` in source or target signature |
| Scene-cut chroma threshold | `0.20` maximum absolute log2-chroma delta |
| Live-camera steady multiplier | time constant `x2`; slew limits `x0.5` |

Normal and fast filtering use
`alpha = 1 - exp(-dt / tau)` followed by the applicable per-second slew limit.
The first reliable frame after startup or a hard reset seeds scalar
statistics but leaves the applied transform at identity. A scene-cut frame
similarly holds the previous transform and seeds the post-cut signature; only
subsequent frames fast-acquire. Live-camera slowing applies only in steady
state, not during bounded fast acquisition.

With no previously reliable estimate, low confidence means identity. With a
previous estimate, low confidence temporarily freezes the last bounded
transform for `0.5 s`, then decays it monotonically toward identity with the
`1.5 s` time constant, and clears it to exact identity at `5.0 s`.
When updates are sparse, only the elapsed portion after the freeze boundary is
integrated as decay time; a capture/repeat gap cannot retroactively turn the
frozen interval into decay.
An estimator exception follows the same identity/freeze-and-decay policy and
emits a bounded transition diagnostic; it must not crash an otherwise valid
local camera loop or produce malformed pixels.
Time filtering uses monotonic elapsed time, not a fixed per-frame coefficient.
A repeated output frame, failed source read, rejected candidate activation, or
unsent trial frame must not advance temporal state. A long timing gap that
reaches `5.0 s` clears stale state rather than taking one large EMA step.
An equal timestamp is idempotent and a backwards timestamp is rejected.

Candidate construction and trial must use fresh or cloned candidate state.
A failed hot activation leaves the live estimator, video position, backdrop
caches, mask state, effective config, and config version unchanged. Commit
swaps config and temporal state at one frame boundary.

The implementation makes this ownership explicit:

- `_Resources` owns one segmenter/refiner pair together with its exact
  segmentation/acceleration policy and generation;
- a segmentation-policy `_Activation` owns a fresh pair, trials it against a
  detached copy of the boundary capture, then resets the pair before commit;
- trial masks and recurrence are never promoted. Commit installs only the
  scrubbed pair and requests a configuration reset, after which the normal
  frame lane processes that exact capture identity as the first authoritative
  input. In passthrough/remote mode the installed pair remains clean with the
  reset pending until matte processing next runs;
- background-presentation-only changes retain the live pair, ownership object,
  generation, and temporal history;
- `_Resources` owns one harmonizer and one reset token;
- `_Activation` constructs a fresh harmonizer whenever the correction policy,
  blend space, backdrop visual identity/geometry, camera/canvas geometry, or
  segmentation model key changes;
- trial uses a clone of staged state with live-state tracking disabled;
- commit swaps the harmonizer with config, version, provider policy, and visual
  generation; any install exception restores the prior harmonizer and token;
- at runtime, capture generation, delivered-geometry generation,
  foreground-content rectangle, provider identity, the fitted backdrop's
  immutable transform plan/source size, visual generation, or canvas changes
  reset to identity while consuming the next reliable estimate as the new
  scalar seed; and
- the normal repeated-output path reuses the last guarded output and never
  calls estimator or harmonizer update.

## Observability and operator-control contract

Public identity belongs to the output frame, not merely to the most recently
accepted configuration. `FrameHub` must install a frame's public stats and
publish/notify that frame through one locked boundary. A hot configuration
commit may be acknowledged before its first output is rendered, but status
continues to describe the prior output until the first matching frame is
published.

Geometry names retain distinct meanings:

- `capture_width`/`capture_height` remain the backend-reported/negotiated
  camera mode;
- delivered, oriented, and normalized camera dimensions have separate fields;
- canonical output dimensions and configured/effective output FPS have
  separate fields; and
- camera/backdrop plans report fit, rotation, mirror, scale, crop, padding,
  generation, and transition counts without a source identifier.

Correction status separates configured mode, effective transform, and temporal
state. Disabled, mode-excluded, warming, low-confidence hold, stale decay,
scene cut, and active states must remain distinguishable.
`color_correction_active` means a non-identity bounded transform is applied to
this output; an `active` temporal state with a reliable identity estimate
retains `color_correction_active: false`. Status also exposes
reason, confidence, exposure EV, bounded WB gains, WB-active, timing, frame
totals, scene cuts, transition counts, and the declared camera input
assumption. The status response and OpenAPI model must have exact key parity
with the hub serialization.

Geometry and correction logs are first-observation/transition diagnostics,
not per-frame logs. They may contain dimensions, plans, bounded transform
values, state, reason, and confidence, but must not contain device or asset
paths. Shutdown emits aggregate geometry/correction counts.

The Web UI follows the same lifecycle contract:

- automatic correction, strength, background fit, camera fit, and backdrop
  focal anchors are the primary controls;
- exposure limit, WB strength, adaptation time, blend-space compatibility,
  rotation, and paired output dimensions are advanced;
- camera/output geometry is visibly restart-required;
- every request uses the smallest valid merge patch, with paired output
  dimensions sent together because the schema makes them inseparable;
- a `409`, `422`, or `503` response causes controls to be restored from the
  effective configuration before the error is surfaced; and
- the concise summary never labels a held, stale, warming, excluded, bypassed,
  or identity transform as fresh active correction.

Local-video status reports decoder backend, resolved input/output color,
metadata/legacy/override status, and exact assumed/overridden fields. Camera
control status is a generation-bound, path-free read-only report. Its accepted
policy is `preserve`; V4L2, MSMF, DSHOW, and other OpenCV backends remain
unqualified for writes, and no harmonizer-driven hardware feedback loop is
permitted.

## Raw, remote, and privacy invariants

Let `R` be the validated, oriented, mirrored, fitted canonical camera frame.

1. `R` is published to the raw hub before local rendering and is the source for
   raw fingerprints. Color harmonization must not modify `R`.
2. Raw JPEG transport serializes `R`; compression differences are expected,
   but no corrected/composited frame may be substituted.
3. In local passthrough mode the output pixels are `R` exactly.
4. The remote renderer receives `R` through the existing authenticated raw
   channel and owns its rendered content. The returned decoded frame must be
   `H x W x 3 uint8 BGR` at the exact canonical canvas.
5. A wrong-size, malformed, stale, unauthenticated, raw-echo, or replay-like
   remote frame is rejected. It is never resized, cropped, padded, rotated,
   mirrored, color-corrected, or used as estimator input.
6. Remote failure emits the fixed input-independent privacy slate. The core
   must not fall back to a local camera-derived composite, raw frame, stale
   remote frame, or previous-session frame.
7. The privacy slate and raw fingerprint path remain in external BGR code
   values and bypass harmonization and linear compositing.
8. Preview, API, null output, pyvirtualcam, and native output consume the same
   already-guarded final frame. A sink may serialize or mechanically add an
   opaque byte, but it must not apply private geometry or color correction.

These invariants take precedence over visual recovery. A helpful fit or color
adjustment at the remote boundary is a privacy/contract defect.

## Windows native output

Until a real, qualified native scaler exists, the native backend follows the
constraint option from VIS-1.6:

- supported canonical canvas/FPS combinations are exactly those truthfully
  supported and advertised by the media source; initially they are
  1280x720@30 and 1920x1080@30;
- explicit `output.backend: native` with any other canvas or output FPS fails
  startup clearly;
- auto-output selection may skip native and continue to the documented
  fallback only when the combination is unsupported;
- a consumer may negotiate only the exact media type matching the active ring;
- a ring/consumer mismatch must fail or show an input-independent placeholder;
  center overlap-copy, silent crop, postage-stamp padding, and implicit stretch
  are prohibited.

The Python ring representation is opaque BGRX with square pixels. The writer
validates canonical full-range display-referred sRGB BGR and only appends an
opaque X byte. The Media Foundation RGB32 type therefore truthfully declares
BT.709 primaries, the sRGB transfer function, and nominal range 0..255.
Revisit native scaling only if supported application negotiation requires
multiple consumer sizes from one canonical ring and Windows qualification can
prove the same geometry as preview/API.

## Configuration lifecycle and default migration

### Restart-only and hot fields

New fields use these lifecycle rules:

| Field group | Lifecycle | Reason |
| --- | --- | --- |
| `camera.fit_mode`, `camera.anchor_x`, `camera.anchor_y`, `camera.rotation`, existing `camera.mirror` | Restart-only | They change the frame published by the capture slot and therefore raw, segmentation, remote, and temporal-state geometry |
| `output.width`, `output.height` | Restart-only | They change the canonical canvas and every opened output/remote protocol dimension |
| Existing `camera.*` and `output.*` fields | Restart-only | Preserve current acquisition/output resource ownership |
| `background.fit_mode`, `background.anchor_x`, `background.anchor_y` | Hot and transactional | They alter displayed geometry but not provider identity; commit invalidates fit cache and temporal state without reopening/resetting video |
| `compositing.blend_space` | Hot and transactional | A staged trial can validate the path; commit replaces temporal state at a frame boundary |
| `compositing.color_correction.*` | Hot and transactional | Policy changes stage fresh temporal state and become effective atomically |
| Persisted `schema_version` | Load/export/migration only; not hot-patchable | It defines how absent fields are interpreted and cannot change the meaning of a live document mid-generation |

Pure backdrop fit/anchor changes must not reopen a live camera, seek/restart a
video, or advance a provider during trial. Asset/source identity retains its
existing provider-replacement behavior. All hot changes follow
construct/validate, detached trial, compare-and-swap commit, and deferred old
resource cleanup. Failure at any point preserves the prior config version and
live state.

### Selected migration mechanism

Persisted configuration/schema versioning is selected. A global change to the
meaning of absent fields is rejected.

1. VIS-0.4 introduces an explicit top-level persisted schema version. The
   initial visual-contract schema is version 1.
2. A persisted mapping with no schema field is defined as legacy-v1 syntax,
   regardless of when or where the file was created. This is syntax semantics,
   not an attempt to infer installation age.
3. Version 1 resolves absent new fields to:
   `camera.fit_mode: stretch`,
   `compositing.blend_space: srgb_legacy`, and
   `compositing.color_correction.mode: off`.
4. The distributed default/config export writes the explicit current schema
   version and effective visual fields. An in-memory no-file startup uses the
   current new-install schema.
5. A later default flip creates a new schema version. The new distributed
   config may receive qualified target defaults, while a v1 or versionless
   persisted config retains legacy behavior.
6. Migration is explicit and atomic. It writes the destination schema version
   and materializes behavior-affecting fields; it never guesses whether an
   omitted field was intentional. Rollback instructions pin the old explicit
   values.
7. API snapshots and diagnostics report effective values and schema version,
   not ambiguous absence.

The qualified target profile is `camera.fit_mode: cover`,
`compositing.blend_space: linear_srgb`, and
`compositing.color_correction.mode: auto`. No target value becomes a default
before its golden, performance, platform, privacy, and rollback gates pass and
a separate default migration change is reviewed.

## Decision register

| Decision | Selected rule and rationale | Migration impact | Revisit trigger |
| --- | --- | --- | --- |
| Acquisition vs canvas | Separate acquisition request from optional paired canonical canvas. Device negotiation and output shape are different facts. | Old configs resolve canvas to camera request and retain their shape. | A future multi-output pipeline genuinely needs multiple simultaneous canvases. |
| Camera geometry | Shared `cover`/`contain`/explicit `stretch`; target camera default `cover`. Proportional fit prevents subject distortion and matches backdrop semantics. | First release stays `stretch`; later `cover` can alter framing and needs versioned migration/release notes. | Visual fixtures disprove `cover` as the safest general camera framing, or face-aware framing is qualified. |
| Anchors and rounding | Viewer-coordinate anchors with floor placement and safe cover-ceil/contain-floor raster math. This makes odd pixels and edge anchors deterministic. | Current center crop can move by at most the explicitly tested rounding difference. | Cross-platform OpenCV behavior cannot implement the plan consistently. |
| Orientation | Metadata/manual orientation, then viewer-horizontal mirror, then fit. This eliminates backend/order ambiguity. | Existing mirrored portrait setups may frame differently and must pin explicit rotation/mirror. | A backend exposes trustworthy orientation that cannot be normalized under this order. |
| Resampling | No-op exact size, area downscale, linear upscale, separable mixed stretch. Quality is direction-aware and testable. | Downscaled legacy-stretch pixels may change even while framing remains legacy. | Benchmarks show an alternative is materially better within the performance budget. |
| External/internal color | External full-range sRGB BGR; internal bounded linear-sRGB float RGB for photometric work. One explicit conversion boundary prevents gamma-space ambiguity. | Linear output intentionally changes soft-edge pixels; legacy blend switch stages it. | A wide-gamut/HDR end-to-end sink is introduced with trustworthy metadata. |
| Still-image decode | One Pillow/ImageCms operation owns format/pixel validation, file-identity checks, EXIF orientation, ICC-to-sRGB conversion, and BGR publication. Malformed or empty present profiles are rejected; untagged assets assume sRGB. | Full-size and thumbnail pixels now agree on profile/orientation policy; invalid assets may be rejected more strictly. | A sandboxed metadata-aware decoder can prove equivalent security and color behavior, or HDR/wide-gamut output becomes end-to-end. |
| Local-video color | Use a bounded local-file-only PyAV/FFmpeg adapter with libavformat restricted to the `file` protocol. Resolve each field by explicit override, frame tag, codec tag, then observable schema-v1 legacy assumption; never infer from pixel range. Normalize the supported BT.601/709 SDR subset to full-range sRGB BGR and reject unsupported tagged color. | PyAV/FFmpeg becomes a core dependency; untagged schema-v1 assets keep deterministic legacy interpretation and operators can override reproducible local assets. Trusted manifests may still reference a graph of readable local files; this is not single-inode isolation. | A safer metadata-aware backend proves the same timing/security/color contract, or an end-to-end HDR/wide-gamut contract is introduced. |
| Analysis conversion | EOTF-decode before analysis resize and share each frame's decoded arrays with estimation/compositing. This avoids gamma-weighted statistics and duplicate full-frame conversions. | `auto` incurs bounded analysis and one conversion set only in eligible modes. | Profiling shows a different representation materially improves the Phase-4 budget without changing pixels/statistics. |
| Estimator outcomes | Validate before mode exclusion; then apply the fixed mask/target/clipping/exposure/neutral precedence and intersect all sampling with valid content. Low confidence is bounded state input, not exceptional control flow. | Contain padding and malformed inputs can no longer influence a plausible-looking estimate. | New evidence requires another recoverable reason or changes a confidence gate. |
| Correction placement | Estimate after segmentation/fitted backdrop; transform only local foreground and matching RVM foreground. This preserves model input distribution. | Raw/model inputs stay stable; local composites change only when enabled. | Model qualification proves pre-inference normalization improves quality without regressions. |
| Blur color space | Keep Gaussian and normalized masked Gaussian in encoded BGR; blur is correction-excluded and a linear conversion would be an unrelated compatibility change. | Existing blur output remains stable while local image/video/camera compositing can use linear light. | Dedicated visual/performance evidence justifies a versioned linear-blur change. |
| Frame-aligned telemetry | Publish output identity/stats with the matching frame; preserve negotiated/delivered/oriented/normalized/output meanings; expose distinct correction phases and exact hub/API/OpenAPI parity. Transition logs are path-free and aggregate totals close at shutdown. | Status gains fields, but existing capture dimensions are not relabelled and raw pixels are unchanged. | A versioned event/metrics interface replaces the latest-value status model while preserving frame identity. |
| Operator controls | Primary controls cover auto/strength/fit/anchors; advanced controls cover bounded color and restart-only canvas/orientation policy. Use minimal patches and restore effective values after rejected/conflicting/unavailable updates. | The UI exposes existing opt-in behavior without changing defaults or bypassing server lifecycle authority. | Usability evidence supports a different grouping or a safe transactional live lifecycle for a restart-only field. |
| Camera controls | Read a bounded property set once per capture generation and report ambiguous/unavailable values truthfully. Accept only `preserve`; do not write controls or couple hardware to the software harmonizer. | Existing camera firmware behavior remains untouched and status explicitly says backend writes are unqualified. | Separate representative-device V4L2/MSMF/DSHOW adapters prove capability ranges, opt-in restart safety, write/readback, and harmonizer reset. |
| Raw/remote privacy | Canonical but uncorrected raw; exact-size uncorrected remote output; reject rather than fit. This preserves fail-closed privacy evidence. | Remote renderer must adopt output canvas dimensions when they differ from acquisition. | A versioned renderer protocol explicitly negotiates dimensions without weakening exact-size validation. |
| Temporal ownership | Generation-owned constant-memory state, ratified time/deadband/slew/cut/freeze/decay constants, hard resets on identity/geometry changes, soft reset on cuts, and no trial/repeat advancement. This prevents stale correction and rollback mutation. | Hot visual changes start a pristine correction state and may briefly warm up. | Experiments prove safe state transfer across a specific boundary and add transactional tests. |
| Native Windows output | Constrain to exact advertised canvas/FPS until real scaling exists. Silent overlap-copy produces output that disagrees with every other sink. | Unsupported explicit native configs fail instead of cropping/padding. | Windows consumers require multi-size negotiation and a qualified native scaler is available. |
| Output color signaling | Windows RGB32 truthfully declares BT.709 primaries, sRGB transfer, and nominal 0..255 range. Pyvirtualcam remains validated BGR with no portable metadata-signaling claim. | Native Windows consumers receive explicit attributes; other virtual-camera interpretation is unchanged and remains a live qualification boundary. | A portable pyvirtualcam signaling API is available or representative consumers require a separately qualified conversion. |
| Default migration | Version persisted semantics; never globally reinterpret absent fields. Existing behavior remains reproducible. | New target defaults require a schema bump/export/migration path. | The project removes persisted configuration entirely or adopts an equivalent explicit migration ledger. |
| Blend-space control | Temporary compatibility switch. `auto` works with either explicit space: correction is linear, while `srgb_legacy` retains encoded blend/wrap and identity byte compatibility. Linear light remains the target invariant. | Keep legacy accepted through at least one stable release after the flip, then deprecate via schema migration. | Demonstrated downstream workflows require permanent encoded-space compatibility. |

## Consequences and remaining qualification boundaries

The contract creates one geometry owner per source, one canvas for all core
sinks, and an explicit color/privacy boundary. It also introduces visible
changes that must land behind compatibility defaults and measured migrations.

Phase 1 implemented the canonical geometry and exact native-output constraint.
Phase 2 implemented shared color-managed still decode, explicit linear/legacy
compositors, the bounded estimator, elapsed-time harmonizer, and transactional
pipeline integration.
Phase 3 implements frame-aligned geometry/color status, transition and
shutdown diagnostics, accessible operator controls, deterministic
metadata-aware local-video normalization, truthful Windows RGB32 attributes,
and preserve-only read-only camera-control reporting.

Phase 3 also adds a native codec dependency. PyAV metadata is BSD-3-Clause,
but shipped wheels can bundle FFmpeg libraries and codecs with their own
licenses and obligations. Every candidate artifact therefore needs an exact
transitive binary/license inventory; the top-level PyAV license is not
sufficient release evidence. Restricting libavformat to the `file` protocol
prevents new network/non-file protocol authority, but a selected manifest can
still reference other readable local files and native parser/codec security
risk remains.

The remaining boundaries are Phase-4 qualification and rollout work:

- schema v1 deliberately retains `camera.fit_mode: stretch`,
  `compositing.blend_space: srgb_legacy`, and
  `compositing.color_correction.mode: off`;
- live application interpretation of pyvirtualcam output on representative
  Linux/macOS/Windows consumers remains VIS-4.2;
- V4L2/MSMF/DSHOW are classified and observed separately but remain
  unqualified for control writes; representative-device capability,
  write/readback, and any future manual/lock policy remain VIS-4.2 follow-up;
- black-box end-to-end regression expansion and calibrated cross-platform
  visual/performance qualification remain VIS-4.1/VIS-4.2; and
- any target-default flip, schema migration, release staging, and rollback
  instructions remain VIS-4.3.

These are not silent implementation claims. In particular, correction remains
default-off and `linear_srgb` remains opt-in until the Phase-4 gates pass.
Generated tagged fixtures prove deterministic normalization math; they do not
prove live hardware, every codec/wheel build, native consumer interpretation,
or artifact-specific licensing compliance.
