# Stable, resolution-aware matte edge refinement

Status: **MATTE-2.2 implemented as an opt-in candidate; schema-version-1
watershed remains the compatibility/default policy**

This document records the spatial-refinement contract and its qualification
boundary. The checked-in MATTE-0.3 result does not select a production default:
its MediaPipe “watershed on” row is a generated stable-alpha proxy, not an
execution of the production watershed. The digest-only local reference remains
`awaiting-consented-model-backed-evidence` with the MediaPipe watershed
decision `not-decided`.

## Configuration and compatibility

The historical boolean remains authoritative, and a nested policy selects the
implementation used when it is enabled:

```yaml
segmentation:
  edge_refine: true
  spatial_edge_refinement:
    mode: legacy_watershed       # legacy_watershed | stable_guided
    reference_short_edge_px: 720
    radius_at_reference_px: 8
    min_radius_px: 2
    max_radius_px: 12
```

The semantics are:

- `edge_refine: false` bypasses spatial edge refinement regardless of the
  nested mode;
- `edge_refine: true` with `legacy_watershed` preserves the schema-version-1
  marker-watershed path;
- an older persisted configuration or version-1 replay bundle with no nested
  policy resolves to the complete values shown above; and
- `stable_guided` is an explicit qualification candidate, not a production
  preset or an instruction to change an installation.

The four radius fields make the scale decision explicit and observable. The
resolved effective radius is:

```text
clip(
  round(radius_at_reference_px * short_edge_px / reference_short_edge_px),
  min_radius_px,
  max_radius_px,
)
```

`effective_edge_refinement_mode` reports `off` when `edge_refine` is false or
the active backend is RVM; otherwise it reports the configured implementation.
`effective_edge_refinement_radius_px` is zero when effective mode is off and
otherwise reports the resolved/clamped canonical-canvas radius. The existing
`effective_edge_refine` boolean remains available for compatibility.

RVM continues to bypass generic spatial refinement. Its recurrent true-alpha
output is not hard-refined merely because the configured boolean is true.
MediaPipe and the coarse heuristic lane may exercise the configured policy;
configured and effective values must remain distinct in diagnostics and
status.

Changing the policy is a segmentation-resource change. Hot activation stages
and trials candidate-owned segmenter/refiner state, commits a new segmentation
generation, and resets the matte timeline before the first authoritative
input. Candidate failure or rollback preserves the live policy and temporal
state. Presentation-only background changes do not rebuild the refiner.

`mask_shift` and `mask_blur` retain their historical pixel-valued meanings.
The stable candidate does not silently scale or reinterpret either control.
Qualification isolates the spatial candidate with those controls held fixed.

## Why watershed remains only as compatibility behavior

The existing watershed starts from a hard `alpha >= 0.5` mask, erodes
foreground and background by a fixed eight pixels, and assigns the unknown
band to foreground/background basins. It writes the watershed separator as
`0.5`; the later configured blur recreates a soft-looking transition.

That path remains useful as an exact rollback and can improve a displaced
synthetic step. It is not a suitable basis for the stable candidate:

- eight pixels represents different relative search distances at 360p, 720p,
  and 1080p;
- any non-uniform guide can start watershed, so camera noise or MJPEG blocks
  can select a different basin each frame;
- basin labels replace genuine confidence/soft-alpha structure with mostly
  binary values; and
- a single global contrast check cannot distinguish a strong unique subject
  edge from several weak or ambiguous nearby edges.

The current thin-component protection and uniform-guide no-op remain part of
the legacy contract. They are not evidence that watershed is temporally stable
on model-backed clips.

## Stable guided-alpha candidate

The candidate is bounded, deterministic, and soft-alpha preserving.

1. Validate and normalize current alpha to finite contiguous
   `float32 [0,1]`. All-zero/all-one mattes, invalid guide geometry, unavailable
   OpenCV operations, or an empty active band return that exact current alpha.
2. Resolve a search radius from the canonical canvas short edge. With the
   schema-version-1/default policy values:

   ```text
   radius_px = clip(round(8 * short_edge_px / 720), 2, 12)
   ```

   This yields `4`, `8`, and `12` pixels for `640×360`, `1280×720`, and
   `1920×1080`. The explicit two-pixel floor bounds small canvases; the
   twelve-pixel ceiling prevents large canvases from expanding work or chasing
   distant image structure.
3. Build an active region from the bounded neighborhood of the current `0.5`
   contour; existing soft values outside it remain untouched. Protect exact
   foreground/background endpoints outside that region. Local foreground and
   background density identifies hair-like pixels and narrow slits inside the
   band and keeps those pixels authoritative instead of letting refinement
   erase them.
4. Convert only the required current-frame guide region to a bounded grayscale
   analysis representation and denoise it before computing local statistics.
   Denoising suppresses isolated camera noise and compression-block residuals;
   it is not a temporal image accumulator.
5. Form a guided-alpha candidate with bounded box/local-statistic operations.
   Candidate influence is admitted only where all applicable checks agree:

   - local guide contrast/variance exceeds the noise floor;
   - a coherent guide gradient exists near the matte boundary;
   - guide/alpha covariance supports moving the alpha transition toward that
     gradient;
   - the candidate does not reverse foreground/background ordering; and
   - the best support is sufficiently separated from competing nearby
     gradients to be unambiguous.

   A weak, flat, block-dominated, texture-only, or tied response has zero
   confidence.
6. Blend the candidate into the original alpha by that bounded confidence.
   Do not threshold the result and do not write a special `0.5` contour. Exact
   endpoints, protected thin components, and pixels outside the active region
   remain byte-for-byte equal to current alpha. A topology/finite/range check
   rejects an unsafe candidate.
7. Return a finite contiguous `float32 [0,1]` alpha. Any native exception,
   malformed intermediate, overlarge active region, failed contrast or
   ambiguity check, or topology failure falls back to the exact current alpha,
   not a partial candidate and never the previous frame.

The guided calculation is continuous in local statistics rather than a
winner-takes-all watershed basin. Alternating nearly equal gradients therefore
become an ambiguity no-op instead of alternating contours.

## Stage order and temporal interaction

The qualified new-policy order is:

```text
current backend alpha
  -> stable spatial guided-alpha refinement
  -> intentional mask_shift
  -> configured mask_blur
  -> optional motion-aware temporal stabilization
  -> compositor
```

Spatial refinement uses only the current source/matte. It does not create a
second temporal owner or advance on output repeats. If
`boundary_stabilization.mode: motion_aware` is explicitly selected, that later
stage registers the already spatially refined prior alpha to the current
source, applies capture-time `dt`, and rejects unreliable correspondence.
This ordering lets motion-aware stabilization damp residual selected-support
variation without allowing an old contour to direct current spatial search.

The complete legacy path retains its historical order and pixels:
`legacy_watershed`, shift, blur, and the separately configured compatibility
EMA. No default or replay is silently moved onto the new order.

## Runtime and privacy boundaries

Work is restricted to a radius-bounded contour/uncertain region, with an
explicit active-area limit and fixed-size OpenCV kernels/statistic buffers.
There is no unbounded component list, frame history, or per-output-repeat work.
Implementations must use vectorized/core OpenCV operations rather than
source-resolution Python pixel loops or an optional `opencv-contrib`
dependency. Spatial-refinement time is included in `refinement_ms` and the
public aggregate `segmentation_ms`.

A 2026-08-03 development-container smoke used 31 warmed calls over a synthetic
vertical displaced edge. It observed about `3.7/4.3 ms` median/p95 at 360p,
`15.0/15.6 ms` at 720p, and `25.1/25.7 ms` at 1080p. The candidate retains no
cross-frame state. These figures isolate the spatial stage on one host; they do
not include model inference, compositing, camera transfer, or qualification
clips and therefore cannot authorize a preset or the 30 FPS production gate.

The guide crop, alpha, and component map are camera/silhouette-derived
sensitive data. They are frame-local, are not retained as a new temporal
history, and never appear in normal logs, public status, or `FrameHub`
history. Existing opt-in replay bundles remain owner-only and bounded; policy,
resolved radius, scalar timings, and fallback state are safe to record, while
guide pixels and intermediate maps are not added as ordinary telemetry.

An edge-refinement failure cannot weaken remote privacy behavior. The only
runtime fallback is the already validated current matte, after which the
existing compositor/privacy gates remain authoritative.

## Qualification and acceptance gates

Deterministic generated tests must cover:

- straight and curved contours displaced on either side of a known source
  edge, with lower spatial/ground-truth error after refinement;
- the same geometrically scaled scene at `640×360`, `1280×720`, and
  `1920×1080`, resolving radii `4/8/12` and producing comparable normalized
  improvement;
- temporally alternating nearby gradients, including a stronger true edge and
  ambiguous/tied distractors;
- uniform and low-contrast guides, seeded camera noise, and deterministic
  JPEG/MJPEG-style `8×8` block artifacts;
- one/few-pixel foreground and background hair-like components, holes, and
  semi-transparent edges;
- all-zero/all-one/tiny-subject mattes, malformed inputs, OpenCV failure, and
  overlarge active regions; and
- default/schema-v1 exactness, RVM bypass, hot activation/rollback, reset,
  output-repeat, and remote-privacy behavior.

Generated mechanics are necessary but cannot select a default. A same-source
consented/licensed, model-backed qualification must show:

- stationary compensated contour displacement p95 does not increase versus
  edge refinement off and the legacy comparison, while the release candidate
  remains within `1.5 px` and `0.60 ×` its baseline where that ratified gate
  applies;
- displaced-edge spatial error improves without increasing stationary subject
  area drift beyond `1%`;
- qualified thin components retain connectivity and alpha mass, and
  soft-edge width/uncertain-pixel fraction do not collapse into a binary
  cutout;
- opaque-core p05, background alpha, holes/components, exterior halo
  mass/width, backdrop leakage, fine-detail SAD/gradient error, and motion-trail
  metrics do not regress;
- behavior is consistent across 360p/720p/1080p, MediaPipe/heuristic lanes,
  capture cadence, noise/compression conditions, and the qualified temporal
  policy combination; and
- p95 refinement and allocation volume fit the MATTE-3.4/MATTE-5.3 stage
  budget, with the serialized 30 FPS path at or below `33.3 ms` p95 or
  sustaining at least 27 unique composites/s without growing latency.

RVM, CPU/GPU, restart/discontinuity, downstream-consumer, and long-run memory
coverage remain mandatory for any advertised preset. Output repeats do not
count as unique performance.

## Rollout and rollback

Until reviewed model-backed and platform evidence exists:

- both default YAML files remain `legacy_watershed`;
- WebUI presets and operator mitigations do not select `stable_guided`;
- generated proxy improvement is labeled directional only; and
- the stable candidate is available solely for private same-source
  qualification.

The one-patch rollback for an explicitly trialed candidate is:

```json
{"segmentation":{"spatial_edge_refinement":{"mode":"legacy_watershed"}}}
```

An operator should restore the exact saved `GET /config` value rather than
assuming the compatibility example if their installation was already
customized. Rollback rebuilds the segmenter/refiner generation and resets
temporal state at the next authoritative input.
