# Motion-aware matte boundary stabilization

Status: **MATTE-2.1 implemented as an experimental, default-off policy; no
production preset is selected**

This document records the boundary-stabilization design and its qualification
boundary. The checked-in MATTE-0.3 evidence is generated-proxy evidence, and
the local model-backed reference remains
`awaiting-consented-model-backed-evidence`. Nothing here authorizes a default
change or an operator mitigation.

## Configuration and compatibility

The new policy is an explicit nested segmentation control:

```yaml
segmentation:
  temporal_smoothing: 0.35
  boundary_stabilization:
    mode: "off"               # off | motion_aware
    time_constant_s: 0.10
    max_motion_px_per_s: 720.0
```

These numeric values are inert implementation defaults while `mode` is `off`;
they are not a qualified strength or preset.

| Field | Accepted values | Default | Meaning |
| --- | --- | ---: | --- |
| `mode` | `off`, `motion_aware` | `off` | Selects the new stage; it does not rename the legacy EMA. |
| `time_constant_s` | finite `0.01`–`0.5` | `0.1` | Elapsed-time EMA acquisition constant. |
| `max_motion_px_per_s` | finite `1.0`–`30720.0` | `720.0` | Maximum trusted motion in full-resolution source pixels per second. |

`temporal_smoothing` retains its historical meaning and range. With boundary
stabilization off, non-matting backends continue to use the existing
frame-count-based, locally change-gated EMA exactly as before. RVM continues to
neutralize that generic EMA because it already owns recurrent temporal state.
Old persisted configurations and version-1 replay bundles omit the nested
control and therefore select `off` without changing pixels.

Selecting `motion_aware` is an explicit policy change:

- the legacy `temporal_smoothing` EMA is bypassed rather than converted,
  reinterpreted, or stacked with the new filter;
- `time_constant_s` controls elapsed-time acquisition;
- `max_motion_px_per_s` is the full-resolution source-pixel velocity above
  which prior alpha is not trusted; flow measured on the analysis guide is
  scaled back to source coordinates; and
- RVM may use the policy only when it is explicitly selected. Ordinary RVM
  operation and non-explicit RVM ablation rows remain off.

A policy change follows the existing segmentation activation contract. It
stages and trials candidate-owned state, commits a new segmentation generation,
and resets before the first authoritative input. Failure or rollback cannot
advance the live refiner. Presentation-only background changes do not reset
the stabilizer.

The independent `spatial_edge_refinement` policy runs before this temporal
stage. Selecting `stable_guided` does not implicitly enable motion-aware
stabilization, and selecting `motion_aware` does not change the schema-version-1
watershed default. The spatial candidate and its evidence boundary are
documented in [the MATTE-2.2 design record](matte-spatial-refinement.md).

`GET /config` is the configured-intent authority. Effective status and matte
replay controls report the selected mode and parameters separately from
`effective_temporal_smoothing`; that compatibility field continues to mean
only the legacy EMA. No time constant is disguised as an equivalent legacy
smoothing value.

## Selected algorithm

The selected experimental path combines bounded motion correspondence with
confidence-gated boundary-only temporal blending.

1. The configured spatial policy (`legacy_watershed` or the explicit
   `stable_guided` candidate), intentional `mask_shift`, and configured blur
   run in that order. The stabilizer receives the resulting current `float32`
   alpha, the canonical source frame, and its exact capture
   sequence/timestamp context.
2. The source is converted to a grayscale analysis guide with preserved aspect
   ratio and a maximum long edge of 320 pixels. State is constant-memory:
   previous stabilized alpha, the previous bounded guide, timestamp, and
   bounded hold metadata.
3. OpenCV Dense Inverse Search (DIS) estimates both previous-to-current and
   current-to-previous flow on the bounded guide, where forward/backward and
   photometric confidence are computed. Only points in a dilated alpha-boundary
   region sample that confidence and the full-resolution previous alpha.
   Full-frame source-resolution optical flow is never computed.
4. Previous alpha is registered to the current source position. Confidence
   combines bounded motion, bidirectional consistency, and availability of a
   useful uncertain boundary. Bad correspondence, excessive motion,
   occlusion/disocclusion, a scene discontinuity, or an unavailable flow path
   selects current alpha immediately.
5. For accepted correspondence, elapsed capture time `dt` determines the EMA
   acquisition weight:

   ```text
   current_weight = 1 - exp(-dt / time_constant_s)
   ```

   Confidence reduces the amount of prior alpha retained; it never increases
   it. An internal maximum-hold bound prevents a weak old contour from
   accumulating indefinitely.
6. Blending is restricted to the narrow, dilated boundary band. Exact
   foreground/background endpoints and the current opaque core outside that
   band pass through unchanged. The result is clipped and returned as finite,
   contiguous `float32 [0,1]`.

The first input, missing frame context, missing OpenCV/DIS support, invalid or
non-positive elapsed time, incompatible state shape, flow failure, or failed
confidence check all use current alpha. Existing capture-generation, geometry,
non-monotonic-time, long-gap, backend-recovery, and segmentation-policy resets
discard every guide, alpha, and hold value before processing the boundary
frame. Output repeats never call or advance the stabilizer.

## Design-spike comparison

The required alternatives were evaluated as bounded components, not as four
production modes:

| Candidate | Useful property | Limitation | Decision |
| --- | --- | --- | --- |
| Previous-alpha warping with low-resolution optical flow | Follows translation and rotation instead of blending two spatially different contours | Flow alone can confidently preserve the wrong edge during occlusion or low-texture motion; full-resolution dense flow would exceed the intended cost | Selected as bounded 320-long-edge bidirectional DIS, with sparse boundary evaluation |
| Confidence-gated blending in an uncertain boundary band | Preserves opaque/background endpoints and rejects unreliable correspondence | Without registration it still averages old and new edge positions | Selected as the gate and spatial scope around registered alpha |
| Asymmetric attack/release or alpha hysteresis | Cheap and can suppress small stationary confidence oscillation | Has no source correspondence, can bias grow/shrink behavior, and can leave a motion trail or harden genuine soft alpha | Not selected for the runtime path |
| Registered temporal median or robust multi-frame filter | Rejects isolated outliers and is simpler to reason about than unconstrained long-history smoothing | Requires more mask history, adds at least one-frame decision lag, and can quantize or erase fine/semi-transparent structure | Not selected; current-alpha fallback is safer when flow confidence fails |

The combination matters: DIS supplies correspondence, while the boundary and
confidence rules prevent correspondence from becoming permission to retain
stale alpha. Hysteresis and median remain useful offline comparators, not
silent fallbacks.

## Runtime and privacy boundaries

The flow guide has a fixed 320-pixel long-edge ceiling, boundary confidence
uses a bounded sparse sample, and retained history does not grow with session
length. OpenCV operations remain on the serialized unique-input refinement
lane. Their cost is included in public `segmentation_ms` and, in a full
privacy-aware replay bundle, the explicit `refinement_ms` stage. Output sends
and repeated outputs are not credited as stabilizer work or unique matte
performance.

A 2026-08-03 implementation smoke on the development container (OpenCV 5.0,
synthetic textured 1280x720 input, 320x180 guide, 4 warm-up plus 21 measured
unique inputs) observed about `7.1 ms` median / `8.4 ms` p95 and `3.85 MiB` of
retained alpha/guide/hold state. This is a bounded-path observation, not a
platform qualification: it does not include model inference, compositor cost,
camera transfer, or evidence from the reported host, and therefore cannot
select the policy for production.

The previous guide and alpha are camera-derived and silhouette-derived
sensitive state even though the guide is grayscale and downscaled. They remain
in memory under the active refiner generation only, are released on reset and
teardown, and are never written to normal logs or public status. No flow field,
guide image, confidence sample, or warped mask is added to ordinary
observability. Existing opt-in replay bundles remain owner-only, bounded, and
privacy-sensitive; MATTE-2.1 adds no unbounded diagnostic track.

Failure is conservative for quality and lifecycle safety: the current
validated matte wins. Candidate activation uses a detached source frame and
candidate-owned temporal state, then scrubs that state before commit.

## Evidence and release gates

Generated fixtures can establish deterministic mechanics, including:

- comparable results for equal elapsed time at 15, 30, and 60 unique FPS and
  irregular timestamps;
- stationary one/two-pixel jitter reduction;
- registered translation and rotation without a persistent double edge;
- current-alpha fallback during fast motion and occlusion/disocclusion;
- preservation of fine, soft, all-zero, all-one, and tiny-subject masks;
- finite contiguous output and reset behavior; and
- bounded runtime and stable state allocation.

They cannot establish real RVM/MediaPipe quality, CUDA/CPU headroom, camera
noise behavior, or a production strength. Production selection still requires
the same-source, consented/licensed, model-backed MATTE-0.1/0.3 qualification
and the ratified MATTE-0.2 gates:

- stationary compensated contour displacement p95 at most `1.5 px` and at
  most `0.60 ×` its comparison baseline;
- stationary subject-area drift at most `1%`;
- fast-motion trail p95 no more than `1.10 ×` the unstabilized comparison and
  no previous contour visibly dominant for more than one unique-input
  interval;
- opaque-core alpha p05 at least `0.95`, with no more than `5%` of the core
  below `0.95`;
- annotated-background mean alpha at most `0.01`, with no regression in
  exterior halo mass/width, holes, or unexpected components;
- generated ground-truth alpha MSE p95 at most `0.01`, with fine-detail SAD and
  gradient metrics reviewed alongside it; and
- no regression in backdrop leakage, clean-foreground edge color, or the
  separately measured dynamic-light-wrap metric.

MATTE-3.4 and MATTE-5.3 must additionally show that the serialized new-frame
path meets its platform frame budget—at 30 FPS, `33.3 ms` p95 or at least 27
unique composites per second without growing latency—and that the added
refinement fits inside its ratified share. Qualification must cover CPU/GPU,
15/30/60 FPS, irregular/gapped input, restart/reset, hot activation, and
downstream consumers.

Until those records exist, `boundary_stabilization.mode` remains `off` in both
default YAML files, `spatial_edge_refinement.mode` remains
`legacy_watershed`, no WebUI quality preset selects either experimental path,
and operator guidance must describe them only as qualification controls with
one-patch rollback to `off` and `legacy_watershed`, respectively.
