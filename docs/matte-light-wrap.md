# Dynamic light-wrap stabilization

Status: **MATTE-2.4 experimental implementation and qualification contract**

Light wrap mixes a blurred backdrop sample into the foreground inside the
existing soft band `4 * alpha * (1 - alpha)`. That can help a static composite,
but a moving video or camera backdrop can also change edge color on every
unique backdrop frame while the subject and alpha are unchanged.

The `temporal_bounded` policy stabilizes only that blurred wrap sample. It does
not modify alpha, the model foreground, or the ordinary backdrop contribution
through alpha.

## Evidence boundary and compatibility default

The compatibility configuration remains:

```yaml
compositing:
  light_wrap: 0.25
  light_wrap_stabilization:
    mode: "off"
    time_constant_s: 0.12
```

`mode: off` preserves the historical stateless wrap pixels exactly.
`time_constant_s` is inert until `mode: temporal_bounded` is selected. The
stateful mode is eligible only for a positive `light_wrap` value with a video
or camera backdrop. It is an explicit qualification candidate, not a
production preset or a video-specific implicit default.

The checked-in MATTE-0.3 generated screen does not authorize changing that
default. Its compositor factorial is useful plumbing evidence, but it does not
hold alpha fixed across the required experiment, cover both narrow correct and
broad under-opaque edges against fixed and moving backdrops, or produce a
non-null edge-variation result for all four combinations. It is also
generated-proxy evidence rather than a consented model-backed run.

Deterministic tests may establish arithmetic, elapsed-time behavior, reset
semantics, bounded memory, transactional activation, and exact bypass. They
cannot establish that the candidate looks better on representative subjects or
that its cost fits supported hosts. Production enablement requires the complete
same-source matrix below, ratified gates, model-backed evidence where model
foreground is claimed, and the normal performance/platform qualification.

## Algorithm

For each unique eligible backdrop sample:

1. Work in the compositor's active color space. The legacy path uses encoded
   BGR values with scale 255; `linear_srgb` uses decoded linear-light BGR with
   scale 1.
2. Resize the backdrop to one eighth of each canvas dimension, with a
   four-pixel minimum, then apply the existing 9×9 Gaussian blur. Only this
   reduced `float32` raster enters temporal state.
3. Identify the sample with the backdrop provider's frame ID, actual
   presentation timestamp, and discontinuity revision. A repeated identity
   reuses the previous filtered sample instead of treating the output-loop
   iteration as new media time.
4. For a normal monotonic update, compute the current-sample weight from
   elapsed time:

   `weight = 1 - exp(-dt / time_constant_s)`

5. Decompose the proposed RGB change into Rec. 709 luminance and chroma.
   Bound normalized luminance movement to 1.5 per second and chroma movement
   to 2.0 per second before installing the update.
6. Upsample the filtered raster to the canvas and substitute it only for the
   current blurred wrap sample. The compositor still applies model-foreground
   substitution first, then light wrap, then ordinary alpha blending.

The existing `4 * alpha * (1 - alpha)` band and configured scalar strength are
unchanged. Restricting wrap farther inside the foreground, choosing a lower
video strength, or adding a separate local-motion disable gate remain ablation
candidates; this implementation must not be cited as evidence for them.

When the first sample or a reset uses the current backdrop unchanged, the
historical blur/upsample path is retained so enabling the candidate does not
introduce an unrelated spatial-filter difference.

## Time, reset, and transactional ownership

Video timing comes from the provider's monotonic presentation timeline, not
raw container PTS that can restart at a loop. Camera backdrops use their actual
capture-completion monotonic timestamp. Reusing a video frame does not advance
the filter, while 15, 30, and 60 FPS samples with the same elapsed media time
follow the same continuous-time response.

History is discarded and the current sample is installed immediately on:

- backdrop/provider or presentation-geometry replacement;
- a video seek, loop discontinuity, or discontinuity revision;
- canvas shape or compositing working-space change;
- non-monotonic time or a gap greater than 750 ms; and
- a scene cut whose normalized luminance score is at least 0.32 or chroma
  score is at least 0.45. Each score is the larger of global-channel
  displacement and mean per-pixel displacement over the already blurred
  low-resolution sample, so a spatial cut with unchanged global means still
  resets.

Installing the current sample prevents old-scene color from trailing around
the subject after a reset. The thresholds and per-second bounds are candidate
constants, not generally qualified defaults.

Light-wrap history has its own generation and never owns or resets segmenter,
RVM, refiner, or color-harmonizer state. A hot configuration candidate gets a
detached stabilizer for its trial render, and that trial history is discarded.
The clean staged state and its new generation are installed only after the
render validates and the configuration commits; a failed or superseded
activation leaves the live history unchanged. Each live frame is likewise
prepared against a clone and promoted only after successful output validation.

`light_wrap: 0` is stronger than a visual approximation: it performs no wrap
blur or preparation, creates no active wrap state, and advances no wrap
history. Its output is byte-identical to the matched no-wrap compositor with
all other controls held fixed.

## Color-space contract

The temporal operation stays in the same space as the light-wrap arithmetic:

- `srgb_legacy` filters and bounds the historical encoded-value BGR sample;
- `linear_srgb` decodes the backdrop first, filters and bounds linear-light
  values, applies wrap and alpha blending in linear light, then encodes once.

A blend-space change starts a fresh light-wrap generation. State is never
reinterpreted across encoded and linear scales. Exact alpha-zero and alpha-one
endpoint guarantees remain compositor responsibilities, and alpha itself is
never an input to the temporal filter.

## Privacy and retained state

The filter retains two reduced `float32` BGR rasters: the filtered sample and
the previous raw blurred sample. At a 1280×720 canvas that is 345,600 retained
bytes in total. The full-canvas prepared wrap raster is frame-local and is not
retained by the stabilizer. Closing or replacing a generation releases both
reduced rasters.

Although reduced and blurred, these rasters are still derived from a backdrop
and must be treated as sensitive process memory. They are not written to
normal logs, `/status`, or ordinary diagnostic history. Opt-in replay evidence
already contains the exact backdrop and final pixels and therefore remains
owner-only. Its additive light-wrap snapshot is content-free: configured and
effective mode, generation, update/repeat/reset/scene-cut counts, last reset
reason and `dt`, and retained-byte count. Asset paths and intermediate wrap
rasters do not belong in that metadata.

## Performance contract

The state bound is two one-eighth-scale three-channel float rasters, independent
of clip duration. Work occurs only for a new eligible composite and consists
of reduced resize/blur, elapsed-time filtering and bounds, and one full-canvas
upsample. Linear mode may additionally need a backdrop decode when no shared
linear buffer is available. There is no background worker and no unbounded
queue.

These structural bounds are not a performance qualification. Report
first-frame and steady-state compositor p50/p95, wrap-preparation substage
timing, allocations, peak memory, and output deadline misses at supported
resolutions. Compare against the same source, blend space, model-foreground
setting, and scalar wrap strength. A candidate that improves color variation
but consumes unrecovered frame budget must remain disabled.

## Required ablation and gates

Use fixed source, fixed foreground, fixed raw/refined alpha, and identical
backdrop frames and timestamps. Run all four
`use_model_foreground × light_wrap` controls against:

- a correct narrow soft edge and a deliberately broad under-opaque band;
- a fixed backdrop and a moving backdrop; and
- legacy stateless wrap and every stateful candidate being evaluated.

Evaluate a lower scalar, temporal-only filtering, luminance/chroma bounds,
wrap farther inside the foreground, excessive motion/luminance gating, and the
combined candidate as distinct rows. Run shortlisted rows in both blend
spaces, at fixed 15/30/60 FPS and irregular cadence, with repeated backdrop
identities, a scene cut, a seek/loop discontinuity, and hot activation
success/failure.

Keep the existing final-composite `edge_band_rgb_variation`, but do not use it
alone: moving backdrop color legitimately changes through fractional alpha.
For every wrap-on row, use `paired_no_wrap_id` to identify a matched same-frame
wrap-off control and compute the required paired
`light_wrap_attributable_rgb_variation`. Let
`W[t] = (C_base_wrap_on[t] - C_base_wrap_off[t]) / 255`; the metric is the mean
absolute RGB change between `W[t]` and its registered predecessor in the
held-alpha soft-edge band. The pair must share every input, timestamp, alpha
digest, configured stabilization policy, model-foreground decision, blend
space, and color transform. Only the scalar strength changes to exact zero in
the control. Subtracting two aggregate metrics or using post-base output is not
equivalent and is not accepted.

A decision fails closed when a required pair, scenario, segment, alpha
identity proof, or metric is missing or null. A candidate must:

- pass the ratified paired and final edge-variation gates without changing any
  alpha digest or alpha metric;
- stay within the ratified static-appearance delta from historical wrap;
- show no old-scene wrap contribution after a cut, seek, or reset;
- preserve byte-exact no-wrap output and an unchanged state snapshot at
  `light_wrap: 0`; and
- pass the supported-host timing and memory gates.

Generated fixtures validate this protocol but cannot select a production
default. Record the final decision only from reviewed, consented evidence.

## Rollback

Both controls are hot and use the normal transactional configuration path.
Save the current values before changing them.

- Set `compositing.light_wrap_stabilization.mode` to `off` to discard state and
  restore historical stateless light wrap at the current scalar strength.
- Set `compositing.light_wrap` to `0` when an operator needs exact wrap bypass.
  This also prevents temporal-state creation or advancement.
- Restore the saved scalar and leave stabilization `off` to return to the
  compatibility configuration (`light_wrap: 0.25` for an otherwise default
  installation).

A rejected patch does not require cleanup because it cannot replace the live
generation. Confirm the committed configuration and effective scalar after a
successful patch, then follow the
[immediate matte operator guide](matte-operator-mitigations.md) for observation
and rollback recording.
