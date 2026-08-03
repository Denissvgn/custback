# Matte replay bundles

Matte replay recording is an opt-in local diagnostic feature. A bundle contains
identifiable camera pixels and masks that can reveal a recognizable silhouette.
Treat the whole directory as sensitive: do not attach it to an issue, place it
in a shared folder, or commit it as a test fixture without the subject's
explicit consent.

Recording is disabled by default and is not represented in `/status`, normal
logs, or `FrameHub` history. Enable it only with an explicit new output
directory:

```console
custback --matte-diagnostics-dir ./private-matte-run \
  --matte-diagnostics-duration 20 \
  --matte-diagnostics-max-bytes 536870912
```

The directory must not already exist. Custback creates bundle directories as
owner-only (`0700` on POSIX) and files as owner-only (`0600` on POSIX). The
writer stops at the duration or byte limit. A one-frame asynchronous queue
keeps disk stalls out of the live send path; writer backpressure stops evidence
capture instead of delaying future output. Only unique camera inputs are
recorded in the frame track—output repeats are never synthesized into model
inputs. MATTE-0.2 adds a scalar-only `output_timeline` after successful sends.
It references the unique source/base sequence and records update/reuse and
exact-repeat provenance without duplicating pixels.
The byte limit must be at least 65,536 bytes so the terminal manifest can
always be committed; the default is 536,870,912 bytes.

Use `--matte-diagnostics-mode composite-only` when only downstream output
comparison is needed. Such a bundle is explicitly marked
`matte_metrics_authoritative: false`; it cannot be used for alpha metrics or
frozen-intermediate attribution.

## Version 1 layout

The root `manifest.json` has schema `custback.matte-replay`, version `1`, a
terminal `complete` state, recording limits, a frame list, and the
`post_base_final_output_provenance` extension point. Its version-1
`output_timeline` contains monotonic send timestamps, source/base sequence,
`base_updated`, `exact_final_repeat`, and an optional typed post-base
provenance object. Post-base metrics remain nested under that object and cannot
overwrite camera, matte, base-update, or send metric namespaces. While recording,
`manifest.partial.json` is atomically replaced after each committed frame. A
partial manifest without a complete manifest means the recording was
interrupted and replay rejects it.

Each `frames/NNNNNNNN/` directory contains lossless NumPy arrays:

- canonical raw BGR camera frame;
- backend `float32` raw alpha;
- post-refiner `float32` alpha;
- optional backend clean foreground;
- exact BGR backdrop frame;
- pre-extension base composite; and
- final output composite.

When base and final pixels are identical, the manifest represents the base
track as an explicit alias of the final artifact instead of duplicating bytes.
Every artifact descriptor freezes its relative path, byte count, SHA-256
digest, dtype, and shape. Replay disables NumPy pickle loading and rejects
absolute paths, traversal, symlinks, non-private files, malformed manifests,
and digest/shape/type mismatches.

Frame entries retain the bundle sequence, capture sequence, monotonic
timestamp, timestamp source, capture and geometry generations, configured and
effective segmentation/compositor controls, effective color transform,
backdrop frame identity/timestamp where available, and stage timings. Current
camera capture can provide capture-completion time; injected/test sources that
do not provide it are honestly labeled `unique-frame-dequeue`. MediaPipe frames
may also contain strict, content-free `segmentation_diagnostics`: input/model/
returned mask dimensions, resize method, effective timestamp delta, and
same-millisecond quantization adjustment. The segmenter-local epoch and
absolute effective model timestamp are deliberately omitted. Existing
version-1 bundles without this optional object remain valid.

Successful RVM frames additionally carry path-free
`effective_controls.rvm_telemetry`: input/output shapes; configured
auto-versus-explicit detail; the resolved ratio; separate preprocessing,
ONNX-session, and output-validation timings; model identity, SHA-256, and byte
count; and the provider/fallback snapshot associated with that result. The
same stage costs are copied to `timings_ms` as `rvm_preprocess_ms`,
`rvm_session_run_ms`, and `rvm_postprocess_ms` so they can be aggregated
without interpreting controls. Frame-derived values remain null until an RVM
result is fully validated, and a reset clears them. Non-RVM frames record the
telemetry object as not applicable.

Full recordings also store the typed backend-policy snapshot beneath
`effective_controls.matte_policy`. It keeps the actual selected/effective
backend kinds and, for each matte control, its configured value, effective
value, `effective`/`bypassed`/`inapplicable` state, and content-free reason.
The separate `configured_controls` object remains the persisted-intent
authority. Older version-1 bundles without the additive snapshot remain valid;
frozen replay never invents a policy snapshot that was not recorded. See
[the backend-policy contract](matte-backend-policies.md).

Spatial-refinement controls include the configured
`spatial_edge_refinement` policy and, where refinement is effective, its
resolved canonical-canvas radius. Schema-version-1 bundles recorded before
that nested control existed resolve to `legacy_watershed`, preserving model
rerun behavior. `stable_guided` remains an explicit qualification candidate;
recording it does not make it a production preset. The policy/radius/timing
scalars are safe manifest metadata, but denoised guides, covariance maps, and
intermediate candidate alpha are not added as ordinary diagnostic tracks. See
[the spatial-refinement contract](matte-spatial-refinement.md).

Eligible dynamic-backdrop frames may also carry the additive
`effective_controls.light_wrap_stabilization` snapshot: configured/effective
mode, time constant, state generation, update/repeat/reset/scene-cut counts,
last reset reason and elapsed `dt`, and retained-byte count. Video
`backdrop_identity` includes the path-free presentation timeline,
discontinuity revision, and visual generation needed to distinguish reuse,
seek/loop, and provider replacement; camera identity uses its frame generation
and capture-completion monotonic time. Older version-1 bundles without these
optional fields remain valid, but they cannot exercise a stateful frozen
light-wrap candidate without the generation proof. Intermediate filtered wrap
rasters are not a normal artifact track; the exact backdrop and composites
already make the opt-in bundle identifiable. See the
[dynamic light-wrap contract](matte-light-wrap.md).

When available, `resource_samples.rss_bytes` records process resident memory
after the complete frame has passed sink submission. An optional
`vram_bytes` sample uses the same additive version-1 extension. Missing
resource instrumentation is preserved as unavailable by the evaluator; it is
never converted to zero.

## Offline replay

Replay never opens live capture, the API, preview, or a virtual-camera
consumer. Its output directory must also be new and explicit.

```console
# Freeze model/refiner/backdrop inputs and reproduce the recorded compositor.
custback matte-replay ./private-matte-run \
  --output-dir ./private-replay

# Attribute a compositor feature with every upstream array fixed.
custback matte-replay ./private-matte-run \
  --output-dir ./private-no-wrap \
  --light-wrap 0 --model-foreground off

# Rerun the recorded segmentation/refinement selection over raw inputs.
custback matte-replay ./private-matte-run \
  --output-dir ./private-model-rerun --mode rerun

# Extract a final-composite-only bundle for downstream comparison.
custback matte-replay ./private-output-only \
  --output-dir ./private-extracted --mode reference
```

Frozen replay defaults to refined alpha and the recorded light-wrap,
model-foreground, blend-space, and effective color transform. `--mask-stage
raw`, `--light-wrap`, `--model-foreground`, and `--blend-space` are attribution
variants; they never mutate source artifacts. `replay.json` retains input order
and timestamp metadata and reports byte equality plus maximum per-channel
difference against each recorded final composite. An unchanged frozen run also
records `baseline_reproduction_passed` against the tolerance below before any
attribution variant is considered.

MATTE-2.4 qualification uses the recorded backdrop presentation timeline and
discontinuity identity as algorithm inputs. `--realtime` may pace replay for
observation, but wall-clock scheduling must not alter pixels or temporal state.
The `--light-wrap 0` variant is the same-frame control for a recorded wrap-on
composite; paired reports must prove that source/backdrop artifacts,
timestamps, alpha digests, model-foreground selection, blend space, and color
transform are otherwise identical.

Model-rerun mode reconstructs the recorded spatial policy before intentional
`mask_shift`, configured blur, and any optional motion-aware temporal
stabilization. Frozen mode instead consumes the recorded post-refiner alpha;
use the owner-only ablation workflow to compare legacy and stable-guided
postprocess candidates over fixed raw alpha/source timestamps.

When segmentation diagnostics are available, replay records the original
snapshot and, for model-rerun mode, the freshly observed snapshot separately so
timestamp and resampling behavior can be compared without exposing an absolute
model clock.

With color correction off, unchanged frozen replay is expected to be byte
exact. When a recorded correction path used shared predecoded intermediates,
the public offline compositor follows the same color contract; a maximum
one-code-value channel delta is the documented attribution tolerance.

## Offline quality evaluation

Full metric-authoritative bundles can be joined to a private, digest-bound
annotation directory and evaluated without starting a model or device:

```console
custback matte-evaluate ./private-matte-run \
  --annotations ./private-matte-annotations \
  --json ./private-matte-quality.json \
  --markdown ./private-matte-quality.md \
  --backend rvm --device CUDA \
  --effective-detail "ratio=0.4; mask_shift=0" \
  --resampling "canonical 1280x720"
```

Reports contain artifact/annotation digests and metrics, never bundle paths or
pixel arrays. JSON and Markdown outputs are owner-only. Annotation arrays
remain privacy-sensitive because core/background masks can reveal a
silhouette; keep them beside the private replay rather than in the repository.
Metric and annotation definitions are frozen in
[`matte-quality-metrics.md`](matte-quality-metrics.md).

## RVM alpha/compositor attribution

`matte-diagnose` consumes the same full replay plus digest-bound named trimap
annotations. It writes lossless review views, per-region heatmaps, canonical
JSON, and a Markdown summary to one new owner-only directory:

```console
custback matte-diagnose ./private-matte-run \
  --annotations ./private-matte-annotations \
  --output ./private-rvm-attribution \
  --max-output-bytes 536870912
```

The four required views preserve the recorded color/blend contract and compare
raw `pha`, post-refiner alpha, direct full-frame RVM foreground with raw `pha`,
and the current edge-foreground/light-wrap compositor. Foreground-only and
wrap-only controls isolate the latter two effects. The report classifies each
region as raw alpha, post-refiner alpha, clean foreground, light wrap, final
blend, mixed, or no observed failure. It fails qualification when the same
annotated opaque core remains deficient across at least two unique inputs.

All views and heatmaps are derived identifiable imagery. They have the same
privacy handling as the replay, are bounded by `--max-output-bytes`, and must
not be committed or shared without consent. See
[`matte-alpha-attribution.md`](matte-alpha-attribution.md) for the protocol and
interpretation contract.

## Bounded ablation matrix

`custback matte-ablate` consumes the replay, annotations, and an owner-only
plan. It executes frozen postprocess/compositor rows, projects repeat cadence,
and joins separately recorded backend/model rows only after source-identity
validation. It never substitutes an available fallback for a requested
unavailable backend.

See [`matte-ablation.md`](matte-ablation.md) for plan row types, matrix
coverage, privacy bounds, contact sheets, performance summaries, and the
one-host shortlist contract.
