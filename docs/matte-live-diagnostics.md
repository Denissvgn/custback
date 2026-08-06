# Local live matte diagnostics

Status: **MATTE-4.3 native-preview diagnostic contract**

Custback's normal native preview shows the frame published to the configured
output. When `--preview` is enabled, `d` and `D` explicitly switch that same
local HighGUI window into a private matte-diagnostic sink. The diagnostic sink
is an inspection aid: it does not change the production frame, mask, policy,
or temporal state, and it is not a qualification result by itself.

These views contain raw camera pixels, masks, and identifiable silhouettes.
They exist only in process memory and only in the native preview. Custback does
not add them to `FrameHub`, the virtual-camera sink, the browser preview,
MJPEG/snapshot/WebSocket routes, or any other API endpoint.

## Starting and leaving diagnostic mode

Start an ordinary local preview:

```console
custback --mode image --image ./office.jpg --preview
```

The window opens on the production output. The diagnostic controls are:

| Key | Action |
| --- | --- |
| `d` | move forward: output → raw camera → … → instability → output |
| `D` | move backward through the same cycle |
| `h` | show the complete preview help |
| `q` / `ESC` | close the preview and stop custback |

Selecting a view activates evidence capture for subsequent unique camera
inputs. The first frame can therefore briefly say that it is waiting. A view
whose inputs are unavailable shows a neutral slate and an explicit reason; for
example, MediaPipe does not provide RVM clean foreground. Falling through
either end of the cycle returns to production output and immediately clears
the retained diagnostic sample and temporal history. Closing the preview or
tearing down the pipeline does the same.

The capital `D` binding is intentional: it is the previous-view action, not a
second enable switch. From output, `D` selects the last diagnostic view.

## View catalogue

The cycle order and the claims made by each view are fixed:

| View | What is displayed | Interpretation and limits |
| --- | --- | --- |
| Raw camera | A copy of the canonical source frame used by segmentation | Exact source pixels for that unique input. |
| Raw model alpha | The backend alpha mapped linearly from `[0, 1]` to grayscale | Exact validated backend alpha. For non-RVM backends it retains that backend's confidence/binary semantics; the backend-aware policy shown in the overlay is authoritative. |
| Refined alpha | The post-refiner/stabilizer alpha mapped to grayscale | Exact alpha passed to the compositor, including effective spatial, shift, and temporal policy. |
| RVM clean foreground | A copy of the backend clean-foreground prediction | Exact when the selected backend supplied it. It is unavailable rather than synthesized for other backends. |
| Exact backdrop | A copy of the backdrop frame paired with this composite | Exact normalized backdrop used for that base frame. Matte-less modes report it unavailable. |
| Alpha over source | Refined alpha's color scale mixed over the source | A visibility aid made from exact inputs, not a new alpha estimate or calibrated error map. |
| Uncertain boundary | `0.05 < alpha < 0.95`, weighted by `4 × alpha × (1 - alpha)` | Shows the strict soft-alpha band. It is not a semantic hair/person classifier. |
| Inferred opaque-core deficit | `1 - alpha` inside an eroded `alpha >= 0.5` region | A morphology proxy. The inferred core is not annotated ground truth and cannot qualify opaque anatomy. |
| Inferred foreground holes | `1 - alpha` in enclosed components of the below-0.5 region | A topology proxy. Legitimate enclosed background can look like a hole. |
| Inferred exterior halo | Alpha above `0.05` in the border-connected below-0.5 region | A topology proxy. Enclosed below-0.5 holes are excluded, but without a known-background annotation the remaining support still cannot establish a true semantic exterior. |
| Model foreground only | A held-source/alpha/backdrop counterfactual with model foreground enabled and light wrap disabled | This is a complete composite, not an isolated RGB delta. It is available only when a clean foreground exists. Compare it with the other held-input views or use offline attribution for a quantified contribution. |
| Light wrap only | A held-source/alpha/backdrop counterfactual with source foreground and the effective wrap strength | This is also a complete counterfactual composite, not a wrap-only raster. When production consumed a stabilized `PreparedLightWrap`, the live view consumes the exact same immutable sample. |
| Final composite contribution | Per-pixel magnitude of the difference between the actual pre-reaction base and a plain alpha composite using the same source, backdrop, alpha, applied color transform, and blend space | Locates the combined model-foreground/light-wrap contribution. It does not assign that contribution to one feature and is not a counterfactual final-output render. |
| Flow-compensated instability | Registered refined-alpha residual in red and registered downstream-contribution residual in cyan | “Flow” here is a compact view name: the implementation uses confidence-gated global phase-correlation translation, not dense optical flow or ground-truth correspondence. Invalid/non-overlapping borders are black. |

The three morphology views deliberately say **inferred** in the window. They
are fast local search tools. Authoritative opaque-core, hole, and halo
classification requires the private replay annotations described in
[RVM alpha integrity and opaque-core attribution](matte-alpha-attribution.md).

The two “only” compositor views are controlled full composites: upstream
source, refined alpha, backdrop, blend space, and color transform remain held.
Their names describe which optional edge-color feature is enabled, not that
all other pixels have been subtracted away. `Final composite contribution` is
the corresponding magnitude visualization, but combines all differences from
the plain-alpha baseline.

## Overlay and telemetry ownership

The diagnostic overlay is drawn on a copy of the rendered view. It never
writes labels into a production frame, source frame, alpha, clean foreground,
or backdrop. Its first line always identifies the sink as local,
pre-reaction, and never sent to output.

The overlay reuses existing telemetry contracts instead of publishing another
status schema:

| Overlay group | Source and meaning |
| --- | --- |
| `INPUT` | Exact capture sequence plus sequence/time deltas derived from adjacent submitted unique inputs and their frame-aligned monotonic capture timestamps. |
| `TEMP ALPHA` | Local, on-demand raw/refined alpha temporal metrics. These values are not added to `GET /status`. |
| `TEMP EDGE` | Registered change in the actual downstream contribution on alpha-stable uncertain-edge support, plus an explicit availability state. |
| `REGISTRATION` | Phase-correlation state, signed displacement, response, and valid-overlap fraction for the current pair. |
| `EFFECTIVE` | The versioned `GET /status.matte_policy` snapshot: resolved RVM ratio, mask shift, model-foreground state, light-wrap strength, and blend space. Configured-versus-effective states and reasons keep the semantics defined by the [backend-policy contract](matte-backend-policies.md). |
| `RESET` | Existing `matte_reset_count` and `matte_last_reset_reason`; `segmentation_generation` and `config_version` also bound temporal pairing. |
| `FRAME SEG ms` | Backend inference, applicable RVM preprocess/session/postprocess, refinement, and segmentation-total timings carried by the existing private evidence seam. |
| `FRAME COMPOSITOR ms` | Backdrop, color correction, compositor preparation, blend, total, and output-validation timings for this frame. |
| `PUBLIC EWMA ms` | Existing versioned `timing_ms` values, including `segmentation.total`, `compositor.total`, and `pipeline.new_frame_service`; their boundaries remain those in the [cadence observability contract](cadence-observability.md). |
| `COMPOSITOR SUBSTAGES ms` | Every entry in the existing fixed compositor substage map, split over two lines. Zero means measured but inapplicable work; `n/a` means the frame did not carry that instrumentation. |

The local temporal fields have these precise meanings:

- `raw` and `refined` are mean absolute alpha changes against the preceding
  unique input;
- their parenthesized `registered` values compare against the preceding alpha
  after source-derived global translation, averaging only the valid overlap;
- `downstream contribution RGB` first subtracts a matching plain-alpha
  composite from each frame's exact pre-reaction base, registers the previous
  signed contribution raster, and measures its RGB change only where the
  current refined alpha is uncertain, registered alpha residual is at most
  `0.01`, and both frames overlap. This cancels ordinary source/backdrop blend
  motion while retaining model-foreground and stateless or stabilized
  light-wrap motion;
- `edge_colour_state` distinguishes a measured value from warming, reset,
  missing contribution evidence, registration failure, or no stable edge
  support; and
- history is `warming` without a pair, `ready` for a compatible increasing
  sequence, or `reset` across capture/geometry/segmentation generations,
  non-increasing time/sequence, shape changes, a config-version transition,
  or a matte reset-count change.

Registration is accepted only for finite estimates with a phase-correlation
response of at least `0.10` and at least 50% geometric overlap on each axis.
Otherwise the registered metrics and instability image are explicitly
unavailable rather than presenting identity registration as evidence. The
signed displacement, response, state, and actual valid-overlap fraction are
retained with the private local frame telemetry. They describe this global
translation approximation, not a public flow field. All metric values are
observations; they do not redefine cadence, timing, backend policy, or
qualification thresholds.

## Privacy, reaction, and output isolation

The implementation has a deliberately narrow ownership boundary:

1. No local diagnostic view is selected by default. Merely opening
   `--preview` continues to read the normal output hub.
2. On `d`/`D`, the pipeline copies the already-produced frame-aligned evidence
   into a depth-one private queue. Superseded diagnostic work is dropped
   instead of delaying production.
3. A daemon worker renders only the selected view. The preview draws its HUD
   on another copy.
4. The normal output is published before diagnostic submission. The monitor
   has no output-sink or `FrameHub` reference, so it cannot publish a
   diagnostic frame to either path.
5. Returning to output invalidates in-flight work and clears pending, current,
   previous, and idle-worker references. The window immediately repaints the
   last production frame even if no newer output arrives. Final close also
   waits up to five seconds for an in-flight renderer and warns if it cannot
   acknowledge shutdown in that bound.

There is no unauthenticated diagnostic image route—and, by design, no
authenticated diagnostic image route either. The WebUI continues to show only
normal output or its separately authenticated avatar source. A local
diagnostic frame is not written to the opt-in replay directory unless the
separate `--matte-diagnostics-dir` recorder was explicitly requested; live
preview selection itself never persists pixels.

Matte evidence is captured at the pre-reaction base-composite boundary.
Consequently every alpha, contribution, proxy, and instability view remains
uncovered and unmodified by output-only reactions. Cycling back to the normal
output is the explicit way to inspect the final published frame, which may
include a separately declared post-base reaction. Such effects cannot alter
the retained raw, refined, contribution, or heatmap evidence.

When no diagnostic view and no private recorder is active, the pipeline's
evidence request is false: it does not copy masks or full-frame diagnostic
tracks, collect compositor-substage maps for this feature, enqueue work, or
start the rendering worker. The normal preview therefore keeps the ordinary
production path rather than paying a hidden full-frame diagnostic cost.

## Locating a Run-B-style defect

Use the live views to choose the first suspect boundary, not to pronounce a
pass:

1. Compare raw and refined alpha. Leakage already present in raw alpha points
   upstream at backend/model/profile behavior. Leakage introduced or enlarged
   only in refined alpha points at effective refiner, shift, or temporal
   policy.
2. Inspect RVM clean foreground. If alpha is sound but edge color is wrong
   there, investigate model-foreground prediction or its use.
3. Compare the held-input model-foreground and light-wrap counterfactuals,
   then inspect final-composite contribution. A clean matte whose symptom
   appears only downstream points at foreground replacement, wrap, color
   transform, or final base blending.
4. Inspect instability and the `TEMP ALPHA`, `TEMP EDGE`, and `REGISTRATION`
   lines. Red with elevated registered alpha metrics indicates matte motion.
   Cyan with low registered alpha change and a `ready` downstream-contribution
   state indicates model-foreground or light-wrap edge-color motion. A
   low-confidence registration is unavailable evidence, not a zero.
5. Record a short consented, owner-only full bundle for the same source,
   backdrop, RVM profile, and camera mode, then run the annotation-backed
   classifier:

```console
custback --matte-diagnostics-dir ./private-run-b \
  --matte-diagnostics-duration 20

custback matte-diagnose ./private-run-b \
  --annotations ./private-run-b-annotations \
  --output ./private-run-b-attribution
```

The live result guides where to look; `matte-diagnose` supplies digest-bound
named regions, fixed-scale heatmaps, held-input factorial controls, and
raw-alpha → post-refiner → direct RVM foreground → current-compositor
classification. Keep the source bundle, annotations, and generated images
private. See [Matte replay bundles](matte-replay-bundle.md) for recording,
permissions, limits, replay variants, and artifact validation.

Schema-v1 replay bundles retain the exact backdrop and actual base composite,
but not the full-canvas experimental `PreparedLightWrap` sample. The live
light-wrap view is therefore exact for a stabilized run, while an offline
factorial rerender of that feature may not be pixel-identical. The saved base
still locates the combined downstream contribution. For pixel-exact offline
wrap isolation, qualify with stateless wrap or retain the live comparison
until a future replay schema explicitly carries the prepared sample.
