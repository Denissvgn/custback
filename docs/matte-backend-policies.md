# Backend-specific matte policies

Status: **MATTE-2.3 implemented as a typed runtime/evidence contract; schema
version 1 and its quality defaults remain unchanged**

The same persisted quality controls do not apply equally to every segmentation
backend. RVM returns recurrent soft alpha and a clean-foreground prediction,
MediaPipe returns a video confidence mask, the heuristic produces a coarse
binary mask, and the null/passthrough path has no matte to refine.

`custback.matte_policy` is the single authority that separates configured
intent from effective runtime behavior. It replaces inference from requested
`backend: auto`, class names, or a partially modified refiner configuration.
This task does not select a new production quality profile and does not create
a second public status schema.

## Typed snapshot contract

`resolve_matte_policy(...)` consumes:

- the configured `SegmentationConfig` and `CompositingConfig`;
- the kind declared by the backend that was actually constructed;
- the runtime-resolved RVM ratio, when a successful RVM input has supplied it;
- the canonical `(height, width)` when a spatial radius can be resolved;
- whether local compositing is passthrough; and
- the private `experimental_rvm_generic` ablation flag; and
- whether the active backdrop supplies an eligible dynamic presentation
  timeline for temporal light-wrap stabilization.

It returns an immutable, JSON-compatible `MattePolicySnapshot` containing:

- `selected_backend_kind`: the constructed backend capability;
- `backend_kind`: the effective kind, which becomes `null_passthrough` while
  passthrough neutralizes the matte path;
- `passthrough` and `experimental_rvm_generic`;
- `configured`: the exact relevant persisted intent;
- `effective`: the values and alpha semantics the active path consumes; and
- `controls`: one `MatteControl` per operator-facing semantic.

The four `MatteBackendKind` values are:

| Value | Runtime meaning |
| --- | --- |
| `true_alpha_recurrent` | RVM native recurrent soft alpha and optional clean foreground |
| `confidence_mask_video` | MediaPipe video confidence mask |
| `binary_coarse` | Heuristic thresholded binary mask |
| `null_passthrough` | No meaningful matte/refinement path |

Each `MatteControl` stores `configured`, `effective`, `state`, and a stable,
content-free `reason`. `MatteControlState` has these exact meanings:

| State | Meaning |
| --- | --- |
| `effective` | The control or backend semantic is selected for this path. Its runtime value is the value in `effective`. An automatic RVM ratio may still be `null` until the first successful inference. |
| `bypassed` | The control is supported but configured to its neutral/off value, is superseded by the selected temporal owner, or is deliberately neutralized for compatibility with this backend. |
| `inapplicable` | The selected path lacks the model output or algorithm to which the control refers. Changing the configured value cannot affect that path. |

An `effective` state describes policy participation, not a promise that every
frame changes pixels. For example, enabled light wrap is still an identity on
exact alpha endpoints. A configured-off supported control is `bypassed`, not
`inapplicable`, so a later backend-aware UI can distinguish “turn it on” from
“this backend cannot use it.”

Configured objects are never mutated. `MattePolicySnapshot.effective_refiner_config`
mechanically creates the detached configuration used by `MaskRefiner`.
Configured values remain available even when their effective values are
neutral, which makes rollback and evidence review unambiguous.

## Backend matrix

In the table, “effective when enabled” means a non-neutral configured value is
`effective` and its neutral/off value is `bypassed`.

| Semantic | RVM true-alpha recurrent | MediaPipe confidence-mask video | Heuristic binary/coarse | Null or passthrough |
| --- | --- | --- | --- | --- |
| Raw alpha/mask | `native_soft_alpha`, preserved without threshold | `confidence_soft_mask`, preserved without threshold | `thresholded_binary_mask` | `opaque_passthrough`; no matte |
| RVM ratio | Effective; runtime value is unresolved until successful inference | Inapplicable | Inapplicable | Inapplicable |
| `threshold` | Inapplicable | Inapplicable | Effective score cutoff | Inapplicable |
| `mask_blur` | Bypassed by the normal RVM policy | Effective when enabled | Effective when enabled | Inapplicable |
| Spatial edge refinement | Bypassed, with effective mode `off` and radius `0` | Effective when enabled | Effective when enabled | Inapplicable |
| `mask_shift` | Effective when nonzero; the compatibility halo control | Effective when nonzero | Effective when nonzero | Inapplicable |
| Legacy `temporal_smoothing` | Bypassed by RVM recurrence | Effective when enabled and no motion-aware policy owns time | Effective when enabled and no motion-aware policy owns time | Inapplicable |
| Motion-aware stabilization | Effective only when explicitly selected | Effective only when explicitly selected | Effective only when explicitly selected | Inapplicable |
| Model foreground | Effective when enabled | Inapplicable; no clean-foreground output | Inapplicable; no clean-foreground output | Inapplicable |
| Light wrap | Effective when enabled | Effective when enabled | Effective when enabled | Inapplicable |
| Temporal light-wrap stabilization | Effective when explicitly selected and the backdrop is video/camera; bypassed for static/no-timeline backdrops | Same | Same | Inapplicable |
| Opaque-core/halo policy | Native model alpha, no calibration; `mask_shift` only | Confidence mask, no calibration; generic postprocess | Heuristic threshold; generic postprocess | None |

### RVM safeguards

The compatibility/default RVM policy is deliberately model-only:

```text
native RVM pha
  -> optional configured mask_shift
  -> optional explicitly selected motion-aware stabilization
  -> compositor
```

Generic Gaussian blur, legacy or stable-guided spatial refinement, and the
legacy frame-count EMA are bypassed. `threshold` is inapplicable: changing it
must not hard-threshold, calibrate, or otherwise alter RVM hair and
semi-transparent alpha. There is no hidden opaque-core restoration. Until
consented/model-backed MATTE-0.5 and MATTE-2.5 evidence selects a correction,
the effective modes remain `native_soft_alpha`,
`model_alpha_no_calibration`, and `mask_shift_only`.

`boundary_stabilization.mode: motion_aware` remains an explicit, default-off
candidate. When selected it is the sole refiner temporal owner; the generic
EMA remains zero. Otherwise residual temporal behavior is `model_only`, using
RVM's own recurrence.

The `experimental_rvm_generic` input exists only for frozen, owner-only
ablation rows that explicitly declare an experimental RVM policy. It can
exercise generic blur/edge/EMA candidates over recorded inputs, but it is not
a persisted configuration field, is never enabled by the production pipeline,
and cannot authorize a default. The ablation runner first neutralizes generic
RVM controls omitted by that row, so a one-variable candidate cannot silently
activate unrelated configured compatibility defaults; compositor-only rows do
not set the experimental flag or claim a refiner they never replayed.

For `rvm_downsample: 0`, configured intent remains zero (“auto”). The effective
ratio is `null` with reason `awaiting-first-rvm-inference` until a successful
input resolves it, then contains the finite runtime ratio. An explicit
configured ratio is likewise not reported as exercised before successful
inference.

### MediaPipe confidence masks

MediaPipe confidence output stays soft and is not passed through
`segmentation.threshold`. Its configured generic spatial refinement, blur,
mask shift, and one temporal policy remain applicable. The spatial policy is
the schema-version-1 `legacy_watershed` by default; `stable_guided` is a
separate opt-in candidate described in
[the spatial-refinement record](matte-spatial-refinement.md).

When motion-aware stabilization is selected, it replaces the legacy EMA rather
than stacking with it. MediaPipe has no RVM ratio or clean-foreground
prediction, so those controls are inapplicable.

### Heuristic cutoff compatibility

The heuristic is the only backend that uses `segmentation.threshold`. Its
schema-version-1 implementation compares the heuristic score with:

```text
effective cutoff = configured threshold * 0.8
```

The snapshot therefore reports the persisted threshold under `configured` and
the multiplied cutoff under `effective`, with state `effective` even when the
configured value is zero. This records the algorithm actually exercised
without silently changing its compatibility behavior. The resulting binary
mask may then use the same generic spatial, shift, blur, and temporal policy
family as MediaPipe.

### Null and passthrough

The null backend and local passthrough have no matte to improve. Every matte
and soft-edge compositor control is `inapplicable` and resolves to a neutral
effective value. In passthrough mode the snapshot retains
`selected_backend_kind` so diagnostics can distinguish the installed backend
from the currently neutral presentation path; only effective `backend_kind`
becomes `null_passthrough`.

This is an applicability decision, not a privacy fallback. Remote raw-echo and
privacy-slate gates remain later, independent authorities.

## Temporal and spatial ordering

For MediaPipe and heuristic paths, the effective order remains:

```text
current backend mask
  -> selected spatial edge refinement
  -> mask_shift
  -> mask_blur
  -> exactly one temporal policy
  -> compositor
```

“Exactly one” means legacy EMA when it is configured above zero and
motion-aware mode is off, motion-aware stabilization when explicitly selected,
or neither when both are off. Selecting motion-aware mode exposes the
configured legacy value but marks it `bypassed` with reason
`replaced-by-motion-aware`.

RVM uses the model-only order above. Output repeats consume the already
rendered safe base and do not rerun the refiner or segmenter or advance matte
temporal state. Status may refresh the content-free policy projection on an
output tick; that read-only resolution is not matte work.

## Configuration, replay, and status ownership

`GET /config` remains the configured-intent authority and retains the existing
schema-version-1 paths. No `matte_policy` field is added to persisted config,
the PATCH schema, or either default YAML.

The active pipeline resolves a fresh snapshot from its actual constructed
backend, current configuration, canonical canvas, passthrough state, and latest
successful RVM ratio. It also reports temporal light-wrap mode as bypassed when
the current backdrop has no dynamic timeline, even if the candidate is
configured. Existing flat `GET /status` fields are compatibility projections
of that snapshot. MATTE-4.1 owns any future public nested policy,
selection/fallback, OpenAPI, ready-log, or overlay schema; MATTE-2.3 does not
publish a competing representation.

A full owner-only replay bundle stores the JSON-safe snapshot beneath
`effective_controls.matte_policy`. Its separate `configured_controls` retains
the persisted segmentation and compositor values. The snapshot contains only
bounded enums, scalars, booleans, and reasons—never pixels, masks, model paths,
raw exceptions, or frame timestamps. Older version-1 bundles without the
nested snapshot remain valid and continue to use their recorded compatibility
controls.

## Transactional activation

Backend and segmentation-policy changes continue to use the existing staged
activation contract:

1. Construct the candidate backend.
2. Resolve policy from that actual backend capability and build a
   candidate-owned refiner.
3. Reset and trial only candidate-owned temporal state.
4. Atomically commit backend, refiner, configuration, version, policy inputs,
   and the new segmentation generation.
5. Reset before the next authoritative unique input.

A failed candidate or timeout closes candidate resources and preserves the
live backend, snapshot inputs, config/version, recurrent/refiner state, and
generation. A successful backend switch cannot blend across policy
generations. Compositor-only changes update configured/effective policy
projection at the same config boundary without rebuilding the segmenter or
resetting matte state.

Changing to or from passthrough changes the effective policy projection without
pretending that the constructed backend itself changed.

## Qualification and rollback

This contract makes existing behavior explicit; it does not establish that the
current values are visually optimal. Production selection still requires
same-source consented/licensed model-backed evidence covering:

- opaque-core alpha, holes, exterior halo mass/width, and backdrop leakage;
- hair, semi-transparent boundaries, thin components, and ground-truth error;
- stationary contour displacement, area drift, fast-motion trails, and
  cadence/reset behavior;
- RVM ratio/model/device and MediaPipe fallback behavior;
- model-foreground and light-wrap interactions;
- CPU/GPU and 15/30/60 FPS performance, restart, fallback, and long-run
  resource behavior; and
- remote privacy and every downstream output boundary.

RVM ratio/model/provider candidates use the separate
[MATTE-2.5 profile qualification contract](matte-rvm-profiles.md). Its
hardware/canvas/native-versus-decimated/raw-versus-qualified matrix must pass
before a named profile can be proposed; the current generated screen cannot
create one.

The normal RVM path remains model-only because the checked-in generated proxy
evidence cannot select an alpha calibration, generic postprocess, residual
stabilizer, profile, or default. An explicit motion-aware trial rolls back with:

```json
{"segmentation":{"boundary_stabilization":{"mode":"off"}}}
```

For any broader experiment, save authenticated `GET /config` before the trial
and restore its exact `segmentation` and `compositing` values. Do not assume the
repository defaults describe an installation with persisted overrides.
`experimental_rvm_generic` needs no live rollback because it has no config/API
surface and exists only inside the bounded ablation runner.

Any future policy/default change requires the MATTE-5.4 schema, migration,
documentation, qualification, and one-patch rollback review. Versionless and
schema-version-1 persisted configurations continue to retain their existing
values and behavior.
