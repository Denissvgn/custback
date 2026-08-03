# Immediate matte operator mitigations

Status: **MATTE-0.4 reversible diagnostic guidance**

This guide helps an operator isolate a visible matte defect while formal
qualification continues. It does not change defaults or publish a production
preset. The checked-in MATTE-0.3 screen used generated pixels and no real model
inference: that generated proxy does not qualify `0.67`, a nonzero mask shift,
MediaPipe watershed, or any other setting as a universal fix. Only a reviewed,
model-backed local ablation report may supply a candidate RVM ratio, mask shift,
or MediaPipe edge-refine direction.

Run one experiment at a time on the same scene. Restore the baseline before the
next experiment. A still screenshot cannot establish temporal improvement.

Before promoting any observed improvement from “diagnostic” to “mitigation,”
review the private `ablation.json` and require:

- `schema: custback.matte-ablation-report`;
- the candidate row to be completed, same-source, and supported by the relevant
  quality/performance decision;
- `evidence_kind: model-backed` for backend, model, ratio, or recurrent-cadence
  claims; and
- the candidate ID to be retained by the report’s bounded lane shortlist.

The two compositor toggles below are safe attribution experiments because they
make no directional quality claim. The full factorial report decides whether
either observed direction is worth retaining.

## Establish the rollback and truth sources

Use the management bearer only on the existing authenticated API. Do not put it
in a URL or shell command argument.

```bash
export CUSTBACK_API_TOKEN="$(custback --show-api-token)"
auth_header() {
  printf 'header = "Authorization: Bearer %s"\n' "$CUSTBACK_API_TOKEN"
}
umask 077
MITIGATION_AUTH="$(mktemp /tmp/custback-matte-auth.XXXXXX)"
auth_header > "$MITIGATION_AUTH"
chmod 600 "$MITIGATION_AUTH"
api_get() {
  curl --fail --silent --show-error --config "$MITIGATION_AUTH" \
    "http://127.0.0.1:8710/$1"
}
api_patch() {
  curl --fail --silent --show-error --config "$MITIGATION_AUTH" \
    -X PATCH -H 'content-type: application/merge-patch+json' \
    --data-binary @- http://127.0.0.1:8710/config
}
```

Set a new private path, preserve its mode, and capture the active configuration
before changing anything:

```bash
MITIGATION_ROLLBACK="$(mktemp /tmp/custback-matte-rollback.XXXXXX)"
api_get config > "$MITIGATION_ROLLBACK"
chmod 600 "$MITIGATION_ROLLBACK"
cleanup_mitigation_files() {
  rm -f -- "$MITIGATION_AUTH" "$MITIGATION_ROLLBACK"
}
trap cleanup_mitigation_files EXIT
```

That saved `GET /config` response—not a repository default—is the exact
rollback authority. Compatibility defaults are shown only as recognizable
reference values:

| Field | Compatibility default | Lifecycle |
| --- | ---: | --- |
| `compositing.light_wrap` | `0.25` | hot compositor change |
| `compositing.use_model_foreground` | `true` | hot compositor change |
| `segmentation.rvm_downsample` | `0.0` (auto) | rebuild/reset |
| `segmentation.mask_shift` | `0` | rebuild/reset |
| `segmentation.edge_refine` | `true` | rebuild/reset |
| `segmentation.spatial_edge_refinement.mode` | `legacy_watershed` | rebuild/reset |
| `segmentation.boundary_stabilization.mode` | `off` | rebuild/reset |
| `segmentation.delegate` | `cpu` | rebuild/reset |
| `output.fps` | `30` | process restart |

`GET /config` reports configured intent. `GET /status` reports active backend
and effective behavior. After activation, inspect both:

```bash
api_get config | jq '{
  segmentation: .segmentation,
  compositing: .compositing,
  output_fps: .output.fps
}'
api_get status | jq '{
  config_version,
  segmentation_backend,
  segmentation_device,
  segmentation_generation,
  segmentation_produces_matte,
  effective_rvm_downsample_ratio,
  effective_mask_blur,
  effective_edge_refine,
  effective_edge_refinement_mode,
  effective_edge_refinement_radius_px,
  effective_mask_shift,
  effective_temporal_smoothing,
  effective_boundary_stabilization_mode,
  effective_boundary_stabilization_time_constant_s,
  effective_boundary_stabilization_max_motion_px_per_s,
  effective_use_model_foreground,
  effective_light_wrap,
  capture_fps_reported,
  capture_target_fps,
  capture_fps,
  capture_read_ms,
  segmentation_ms,
  composite_ms,
  frame_processing_ms,
  processing_deadline_misses,
  output_target_fps,
  output_effective_fps,
  output_repeated_frames
}'
```

Wait for several fresh frames after every activation. A segmentation change
increments `segmentation_generation`; it rebuilds the segmenter and refiner and
resets RVM recurrence, the MediaPipe task, and refiner temporal state. A
compositor-only change does not rebuild the segmenter, so that generation must
stay unchanged.

Trust the exact `segmentation_backend` and `segmentation_device`, not the
requested `backend: auto` value. An automatic RVM-to-MediaPipe selection is
visible in those exact fields even when a generic fallback flag is not set.
Compare measured `capture_fps` with `capture_target_fps`; the driver-reported
`capture_fps_reported` is not proof of delivered cadence.

The motion-aware boundary stabilizer is a MATTE-2.1 qualification control, not
an immediate mitigation. Leave
`segmentation.boundary_stabilization.mode: off` unless the exact backend,
scene, cadence, and host have a reviewed model-backed quality and performance
report. Its rollback is the saved configuration above (or explicitly `off`);
do not translate `temporal_smoothing` into a time constant.

Likewise, `spatial_edge_refinement.mode: stable_guided` is a MATTE-2.2
qualification control, not an immediate mitigation. The checked-in
“watershed on” generated row contains pre-generated stable proxy alpha; it did
not execute either production spatial algorithm and cannot select a default.
Leave the schema-version-1 `legacy_watershed` policy in place unless a reviewed
same-source model-backed report explicitly selects the candidate. See the
[spatial-refinement contract](matte-spatial-refinement.md).

## Install or rebuild the RVM backend

For an npm installation, inspect the existing intent and retain every extra the
deployment still needs. Rebuilding with an explicit list replaces the extras
intent:

```bash
custback extras --json
custback rebuild --extras gpu
custback doctor
```

`gpu` is the qualified NVIDIA/CUDA RVM dependency profile; use `rvm` for the
CPU-only RVM profile. Include other compatible required extras in the explicit
list rather than silently dropping them. For a source/pip installation:

```bash
pip install -e '.[gpu]'
# CPU-only alternative:
pip install -e '.[rvm]'
```

Restart custback, then require `RVMSegmenter` and the intended device in
`GET /status`. Provider registration alone is insufficient; the runtime proves
real RVM graph execution before reporting CUDA. If the requested accelerator
is not effective, stop the quality comparison and diagnose installation or
provider availability. For npm rollback, rebuild with the exact requested
extras list captured by the initial `custback extras --json` (an explicitly
empty list clears extras), restart, and confirm the restored list plus
`custback doctor`. Environment rollback is intentionally separate from the
live configuration rollback file.

## One-variable compositor attribution

Run these only for a local composite mode such as image, video, color, blur, or
camera. They are irrelevant to remote/avatar rendering.

### Light-wrap isolation

**Apply:** Keep model foreground at its saved value and test `light_wrap: 0`
alone.

```bash
printf '%s\n' '{"compositing":{"light_wrap":0}}' | api_patch
```

**Confirm:** `GET /config` shows `light_wrap` as zero,
`effective_light_wrap` is zero in `GET /status`, and
`segmentation_generation` is unchanged. Compare edge-color motion separately
from alpha motion.

**Rollback:** Restore the exact saved value (normally `0.25`).

```bash
jq '{compositing:{light_wrap:.compositing.light_wrap}}' \
  "$MITIGATION_ROLLBACK" | api_patch
```

**Resource effect:** This is hot at a frame boundary and does not rebuild the
segmenter or reset temporal matte state.

### Model-foreground isolation

Begin only after light wrap has been restored.

**Apply:** Keep light wrap at its saved value and test
`use_model_foreground: false` alone.

```bash
printf '%s\n' \
  '{"compositing":{"use_model_foreground":false}}' | api_patch
```

**Confirm:** `GET /config` shows the configured value,
`effective_use_model_foreground` is false, and `segmentation_generation` is
unchanged. If the active backend is not matte-producing, the effective field
is already false and this experiment is not applicable.

**Rollback:** Restore the exact saved value (normally `true`).

```bash
jq '{compositing:{
  use_model_foreground:.compositing.use_model_foreground
}}' "$MITIGATION_ROLLBACK" | api_patch
```

**Resource effect:** This is hot at a frame boundary and does not rebuild the
segmenter or refiner.

Do not disable both features in the first comparison. The MATTE-0.3 full
factorial exists precisely because their visual and performance effects
interact.

## Evidence-gated segmentation experiments

### Reviewed RVM ratio or mask shift

Skip this experiment unless the local MATTE-0.3 reference is reviewed and the
selected row is model-backed, same-source, and present in the report's bounded
RVM shortlist. That shortlist is only admission to the
[MATTE-2.5 qualification matrix](matte-rvm-profiles.md), not a portable named
profile or default. Generated-proxy rows are insufficient.

**Apply:** Patch exactly one reviewed value, never both together:

```bash
printf '%s\n' \
  '{"segmentation":{"rvm_downsample":VALUE_FROM_REVIEWED_REPORT}}' | api_patch
# Or, in a separate restored run:
printf '%s\n' \
  '{"segmentation":{"mask_shift":VALUE_FROM_REVIEWED_REPORT}}' | api_patch
```

Substitute a JSON number from the reviewed report before execution. The
repository does not currently publish such a qualified number.

**Confirm:** Require `RVMSegmenter`; compare configured
`segmentation.rvm_downsample` with `effective_rvm_downsample_ratio`, or compare
configured `segmentation.mask_shift` with `effective_mask_shift`. Confirm that
`segmentation_generation` advanced and discard warm-up frames before judging
the steady result.

**Rollback:** Restore the saved fields exactly. Compatibility rollback examples
are `rvm_downsample: 0.0` (auto) and `mask_shift: 0`, but persisted values win.

```bash
jq '{segmentation:{
  rvm_downsample:.segmentation.rvm_downsample
}}' "$MITIGATION_ROLLBACK" | api_patch
jq '{segmentation:{
  mask_shift:.segmentation.mask_shift
}}' "$MITIGATION_ROLLBACK" | api_patch
```

**Resource effect:** Applying either value rebuilds the segmenter and refiner
and resets recurrent/temporal state. Rollback rebuilds and resets them again.

RVM's effective refiner intentionally reports blur zero,
`effective_edge_refine` false, temporal smoothing zero, and the configured mask
shift. Do not try to repair RVM with generic blur, edge refinement, or EMA.
The configured/effective/applicability distinctions for every backend are in
[the backend-policy contract](matte-backend-policies.md).

### MediaPipe GPU delegate

Install the `mediapipe` extra first. Test this only on a platform where
MediaPipe exposes its GPU delegate.

**Apply:** Pin MediaPipe for the comparison and request its GPU delegate:

```bash
printf '%s\n' \
  '{"segmentation":{"backend":"mediapipe","delegate":"gpu"}}' | api_patch
```

**Confirm:** Require `segmentation_backend` to equal `MediaPipeSegmenter` and
`segmentation_device` to equal `gpu`. MediaPipe may warn and fall back to CPU;
a configured `delegate: gpu` with effective device `cpu` is not a GPU result.
Confirm that `segmentation_generation` advanced.

**Rollback:** Restore both saved values (normally `backend: auto` and
`delegate: cpu`) in one transaction:

```bash
jq '{segmentation:{
  backend:.segmentation.backend,
  delegate:.segmentation.delegate
}}' "$MITIGATION_ROLLBACK" | api_patch
```

**Resource effect:** Apply and rollback each rebuild the segmenter and refiner
and reset their temporal state.

### MediaPipe edge refinement

Proceed only while `segmentation_backend` is exactly `MediaPipeSegmenter`.
RVM always neutralizes this generic control. A real model-backed MATTE-0.3
off/on result must select the direction; the checked-in generated result is not
enough. This diagnostic toggles the `edge_refine` gate around the already
configured spatial mode; it does not change algorithms. Schema-version-1
configurations use `legacy_watershed`.

**Apply:** Toggle only `edge_refine` from its saved value. For a saved `true`
baseline, the independent off test is:

```bash
printf '%s\n' '{"segmentation":{"edge_refine":false}}' | api_patch
```

If the saved baseline is false, test true instead. Do not alter blur or
smoothing in the same run.

**Confirm:** Require `MediaPipeSegmenter`, inspect `effective_edge_refine`,
`effective_edge_refinement_mode`, and
`effective_edge_refinement_radius_px`, and confirm
`segmentation_generation` advanced. Effective mode/radius must be `off`/`0`
while the boolean is off. Discard post-reset warm-up frames.

**Rollback:** Restore the exact saved value (normally `edge_refine: true`).

```bash
jq '{segmentation:{
  edge_refine:.segmentation.edge_refine
}}' "$MITIGATION_ROLLBACK" | api_patch
```

**Resource effect:** Apply and rollback each rebuild the segmenter and refiner
and reset MediaPipe/refiner temporal state.

Do not patch `spatial_edge_refinement.mode` as part of this on/off experiment.
Private stable-guided qualification must start from the saved configuration,
change only that nested mode, retain the resolved radius and refinement p95,
and roll back in one patch to the exact saved mode (normally
`legacy_watershed`).

`segmentation.threshold` affects only `HeuristicSegmenter`. Increasing it does
not change RVM or MediaPipe output and is not a mitigation for either backend.

## Cadence diagnosis

### Lighting and capture versus processing

`camera_controls` in status is read-only observation. This procedure never
writes camera controls. Compare two warm status intervals under the same mode:

- If measured `capture_fps` is below target in both the normal composite and a
  local passthrough run, while read interval grows, investigate lighting,
  exposure behavior, device mode, USB bandwidth, or driver/backend pacing.
- If capture stays near target but `frame_processing_ms`,
  `segmentation_ms`, or `composite_ms` grows beyond the frame budget while
  deadline misses and capture-slot drops rise, the full processing path is the
  constraint.

Try a stable, brighter, diffuse scene and observe whether cadence changes.
Do not automatically force exposure/gain values: support varies by camera and
bad writes can persist outside custback.

For a lightweight local comparison, temporarily PATCH
`background.mode: passthrough`, then restore the saved background section.
Steady passthrough frames skip segmentation, backdrop, and compositing, but the
normal sink/API remain active. A stricter capture-only run requires stopping
the normal process and launching a copied owner-only config with the same
camera mode, local passthrough, null output, API disabled, and preview disabled.
Passthrough exposes raw local camera pixels; never use it as a remote fallback.

In that copy, change only:

```yaml
background:
  mode: passthrough
output:
  backend: "null"
  preview: false
api:
  enabled: false
```

Then run `custback --config /private/path/capture-only.yaml` without `--fps`.
Compare its bounded startup/shutdown capture health with the full run, discard
the copy after review, and restart the unchanged normal configuration.

### Output-rate diagnosis

**Apply:** After measuring a sustained capture rate, edit only `output.fps` in
an owner-only copied YAML configuration, restart, and compare repeat/send
cadence. Do not use the `--fps` shortcut: it changes camera and output targets
together. A live PATCH correctly returns `409 restart_required`.

**Confirm:** After restart, verify configured output FPS and
`output_target_fps`, then compare `output_effective_fps` and
`output_repeated_frames`. A lower target changes send/repeat cadence; it does
not create alpha or unique camera/model observations.

**Rollback:** Restore the exact prior `output.fps` from the saved configuration
(compatibility default `output.fps: 30`) and restart again.

**Resource effect:** This is restart-only and reconstructs process resources.
It is a diagnosis, not a matte-quality fix.

## Privacy and completion rules

- Do not run local compositor experiments by changing a remote/avatar
  `background.mode`. Renderer loss must continue to produce the fixed
  `privacy-slate`; never select a camera-derived local composite as remote
  fallback.
- Do not weaken API bind, TLS, origin, bearer, renderer-token, or remote replay
  protections.
- Optional matte bundles contain identifiable pixels and silhouettes. Keep
  their existing explicit opt-in, owner-only, duration/size-bounded contract.
- A candidate is an immediate mitigation only when the same-source,
  model-backed ablation report supports it, quality gates pass, performance is
  acceptable, and the operator can reproduce apply, confirm, and rollback.
- Output repetition or a lower output target does not create alpha, remove an
  alpha defect, or increase unique matte cadence.
