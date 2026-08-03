# RVM profile qualification

MATTE-2.5 is an evidence and decision contract. It does not add
`performance`, `balanced`, or `quality` to persisted configuration, and it
does not change the schema-1 `rvm_downsample: 0.0` automatic default.
MATTE-4.2 owns any future user-facing preset and MATTE-5.4 owns a future
default or migration.

The formal offline command is:

```console
custback matte-rvm-qualify \
  --plan ./private-rvm-qualification-plan.json \
  --output ./private-rvm-qualification
```

The plan, replay bundles, annotations, run attestations, and output directory
are private local evidence. They can identify the recorded person even though
the final JSON report contains only digests, scalar measurements, and
content-free runtime identity. Never upload those inputs without the subject's
consent.

## Start from the packaged templates

The checked-in
[`matte-rvm-qualification-local-template.json`](matte-rvm-qualification-local-template.json)
is a content-free template pack. Its top-level wrapper is deliberately not a
qualifier input. Copy the nested `templates.plan` object to one owner-only plan
file, and copy either run object to a separate owner-only sidecar for each
recorded cell. For example, with `jq`:

```console
umask 077
jq '.templates.plan' \
  docs/matte-rvm-qualification-local-template.json \
  > ./private-rvm-qualification-plan.json
jq '.templates.run_cpu_native_raw' \
  docs/matte-rvm-qualification-local-template.json \
  > ./private-native-cpu-raw-run.json
```

The template plan contains one aligned CPU raw/compositor native/decimated
family and one CUDA raw cell to demonstrate the references. It is intentionally
not the complete Cartesian matrix. Duplicate those records for every declared
candidate, hardware target, provider, canvas, cadence, and render mode.

Every value beginning with `REPLACE_`, every all-zero or illustrative digest,
and every timing/resource sample must be replaced with measured, reviewed
local evidence. Review local service budgets and any strengthened gate
thresholds while retaining the ratified policy ID and never weakening its
fixed limits. Recompute
`hardware.identity_sha256` after changing its inventory, and replace the
illustrative canonical pre-resize source digests. The short sample arrays in
the run objects illustrate the strict schema only; a qualifying run needs at
least 30 warm-up frames followed by 300 steady-state frames over at least 10
seconds. The CUDA run is also the DirectML shape: change
`provider.requested` to `directml`, change
`provider.runtime.execution_provider` to `DmlExecutionProvider`, supply the
matching runtime/driver and hardware identity, and retain the accelerator VRAM
contract. A copied template is not qualification evidence merely because it
parses.

The plan/run envelopes and the exact subobjects listed below require every
listed key and reject unlisted keys. Segmentation and compositing use the
current strict configuration models: defaulted fields may be omitted, but they
are normalized before comparison and extra fields are rejected. Plan/run files
must be regular, owner-only files, not symlinks. IDs match
`[a-z][a-z0-9_-]{0,63}`; digests are lowercase 64-character SHA-256 values.
Relative evidence paths are resolved from the command's working directory, not
from the plan file's directory.

## Plan contract: `custback.rvm-qualification-plan` version 1

The plan has exactly these root keys:

| Key | Version-1 value |
| --- | --- |
| `schema` | `custback.rvm-qualification-plan` |
| `version` | integer `1` |
| `ablation_report` | non-empty path to the private, digest-valid MATTE-0.3 JSON report |
| `provenance` | qualification provenance object |
| `candidates` | 1–8 candidate objects |
| `hardware` | 1–16 hardware coverage objects |
| `canvases` | 1–8 canvas objects |
| `qualified_compositor` | one strict current `CompositingConfig` object |
| `policy` | quality, service, cadence, sample, and coverage gates |
| `profile_proposals` | empty, or the complete three-name proposal set |
| `cells` | 1–4096 matrix cell objects |

`provenance` has exactly `qualification_id`, `kind`,
`license_or_consent_reference`, and
`contains_private_footage_in_repository`. `kind` is `consented-local`,
`licensed-local`, or `generated`. Consent/license provenance requires a
non-empty reference. The repository-footage flag must be `false`; `generated`
evidence remains diagnostic proxy evidence and cannot select a profile.

Each candidate has exactly:

| Key | Constraint |
| --- | --- |
| `id` | unique safe ID |
| `role` | `shortlist`, `auto`, or `compatibility_baseline` |
| `ablation_candidate_id` | safe MATTE-0.3 ID for `shortlist`; empty for reserved roles |
| `model_id` | bounded model filename/identity |
| `model_sha256` | model bytes digest |
| `model_bytes` | positive integer |
| `segmentation` | strict current segmentation object |

There must be exactly one `auto`, exactly one `compatibility_baseline`, and at
least one `shortlist`. A shortlist row must be a completed, model-backed RVM
candidate admitted by MATTE-0.3, and its normalized segmentation and model
identity must match the digest-bound ablation row. `model_path` is deliberately
excluded from that segmentation digest so identical model bytes may live at
different local paths. The two reserved roles must use the current built-in
RVM model. The compatibility role must equal current `SegmentationConfig`
defaults exactly; the auto role must set `rvm_downsample` to `0.0`.

The normalized `segmentation` keys are `backend`, `model_path`, `delegate`,
`rvm_downsample`, `threshold`, `mask_blur`, `edge_refine`, `mask_shift`,
`temporal_smoothing`, `boundary_stabilization`, and
`spatial_edge_refinement`. The two nested objects have exactly:

- `boundary_stabilization`: `mode` (`off` or `motion_aware`),
  `time_constant_s`, and `max_motion_px_per_s`;
- `spatial_edge_refinement`: `mode` (`legacy_watershed` or `stable_guided`),
  `reference_short_edge_px`, `radius_at_reference_px`, `min_radius_px`, and
  `max_radius_px`.

Qualification candidates select only `backend: auto` or `backend: rvm` and
must otherwise pass the current strict configuration validation. The template
spells out every normalized field to make future drift visible.

Each canvas has exactly `id`, positive integer `width`, and positive integer
`height`. Each hardware object has exactly `id`, `providers`, and `canvas_ids`.
Provider values are only `cpu`, `cuda`, and `directml`; each hardware target
lists unique providers including `cpu`, and the union across the plan must be
exactly all three supported providers. Canvas references are unique and must
exist.

`qualified_compositor` normalizes to exactly `light_wrap`,
`use_model_foreground`, `blend_space`, `light_wrap_stabilization`, and
`color_correction`. `blend_space` is `srgb_legacy` or `linear_srgb`;
`light_wrap_stabilization` has `mode` (`off` or `temporal_bounded`) and
`time_constant_s`; `color_correction` has `mode` (`off` or `auto`), `strength`,
`exposure_limit_ev`, `white_balance_strength`, and `adaptation_time_s`.
Raw-model cells are evaluated with the same policy except
`use_model_foreground: false` and `light_wrap: 0.0`.

The policy has exactly:

- `quality_policy_id`, exactly `matte-0.2-ratified-v1`;
- `quality_gates`;
- `native30_service_p95_ms` and `native30_min_unique_fps`;
- `decimated15_service_p95_ms` and `decimated15_min_unique_fps`;
- `min_warmup_frames`, `min_steady_frames`, and `min_observation_s`;
- `max_queue_age_ms`;
- `max_queue_age_growth_ms`; and
- `minimum_profile_hardware_count`, which must be at least 2.

Service budgets, minimum FPS, and observation duration are positive finite
numbers. Both queue-age limits are finite and non-negative, and
`max_queue_age_ms` cannot exceed the ratified `33.333334` ms ceiling.
Warm-up, steady-frame, and hardware minima are positive integers. Version 1
will not load a policy weaker than 30 warm-up frames, 300 steady-state frames,
a 10-second observation, `native30_min_unique_fps >= 29.0`, or
`decimated15_min_unique_fps >= 14.5`. Service budgets and queue-age growth can
be locally tightened; their template values are not performance claims.

Each quality gate has exactly `id`, `family`, `metric`, `op`, and `value`;
`id` is unique and safe and `value` is finite. The operation is metric-owned,
not caller-selected: reversing it makes the plan malformed. Ratio/alpha/MSE
thresholds shown with a `[0,1]` domain must remain in that domain; the other
thresholds are non-negative. The policy must contain every following
family/metric pair exactly once. The last column is the weakest accepted
ratified limit: callers may strengthen it but cannot weaken it.

| Family | Required metric path | Operation | Domain | Weakest accepted limit |
| --- | --- | --- | --- | --- |
| `opaque_core` | `aggregate.metrics.opaque_core_alpha_p05.p05` | `>=` | `[0,1]` | `0.95` |
| `opaque_core` | `aggregate.metrics.opaque_core_fraction_below_0_95.p95` | `<=` | `[0,1]` | `0.05` |
| `opaque_core` | `aggregate.metrics.foreground_hole_components.max` | `<=` | `>= 0` | `0.0` |
| `background` | `aggregate.metrics.background_alpha_mean.p95` | `<=` | `[0,1]` | `0.01` |
| `halo` | `aggregate.metrics.exterior_halo_area_ratio.p95` | `<=` | `[0,1]` | `0.05` |
| `halo` | `aggregate.metrics.exterior_halo_width_p95_px.p95` | `<=` | `>= 0` | `8.0` |
| `fine_detail` | `aggregate.metrics.ground_truth_alpha_mse.p95` | `<=` | `[0,1]` | `0.01` |
| `fine_detail` | `aggregate.metrics.ground_truth_gradient_mae.p95` | `<=` | `>= 0` | `0.10` |
| `fine_detail` | `aggregate.metrics.uncertain_pixel_fraction.p50` | `>=` | `[0,1]` | `0.05` |
| `temporal` | `aggregate.metrics.contour_displacement_p95_px.p95` | `<=` | `>= 0` | `1.5` |
| `temporal` | `aggregate.metrics.compensated_alpha_temporal_abs_diff.p95` | `<=` | `[0,1]` | `0.10` |
| `temporal` | `aggregate.metrics.motion_trail_area_ratio.p95` | `<=` | `[0,1]` | `0.10` |

`profile_proposals` may be empty. If non-empty, it contains exactly one object
for each name `performance`, `balanced`, and `quality`; every object has only
`name` and `candidate_id`. The three candidates and their
model/segmentation-digest meanings must be distinct. Names are not trusted as
labels: the qualifier proves their latency and quality ordering in every
hardware/provider/canvas/cadence/render scope. A proposal is published only
after its candidate qualifies on at least `minimum_profile_hardware_count`
distinct verified hardware identities and across its complete matrix.

Each cell has exactly:

| Key | Constraint |
| --- | --- |
| `id` | unique safe ID |
| `candidate_id` | declared candidate |
| `hardware_id` | declared hardware |
| `provider` | `cpu`, `cuda`, or `directml`, assigned to that hardware |
| `canvas_id` | canvas assigned to that hardware |
| `cadence` | `native30` or `decimated15` |
| `render_mode` | `raw_model` or `qualified_compositor` |
| `status` | `recorded` or `unavailable` |
| `bundle`, `annotations`, `run` | three non-empty private paths for `recorded`; all empty for `unavailable` |
| `availability_reason` | empty for `recorded`; non-empty safe reason code for `unavailable` |
| `native30_cell_id` | empty for native; matching native cell ID for decimated |
| `paired_raw_cell_id` | empty for raw; matching raw cell ID for compositor |

The required cell set is the exact Cartesian product of every candidate and
each hardware target's declared providers/canvases with both cadences and both
render modes. Duplicate or extra cell combinations are malformed. Missing
combinations remain visible in the report and prevent completion. A decimated
parent must otherwise match candidate, hardware, provider, canvas, and render
mode. A compositor pair must otherwise match candidate, hardware, provider,
canvas, and cadence.

## Run contract: `custback.rvm-qualification-run` version 1

Every recorded cell points to its own content-free run sidecar. The run has
exactly these root keys:

| Key | Version-1 value |
| --- | --- |
| `schema` | `custback.rvm-qualification-run` |
| `version` | integer `1` |
| `bundle_manifest_sha256` | digest of that cell's replay manifest |
| `evidence_kind` | `model-backed` or `generated-proxy` |
| `provenance` | content-free provenance object |
| `source` | canonical pre-resize clip/frame identity |
| `hardware` | stable inventory identity |
| `model` | artifact, license, packaging, integrity, and startup contract |
| `provider` | requested provider, runtime/driver identity, and execution proof |
| `resource_evidence` | sample sources, units/semantics, alignment, and applicability |
| `recurrence` | fresh-state native/decimated execution proof |
| `warmup_frame_count` | non-negative integer prefix excluded from steady-state gates |
| `startup_ms` | non-empty finite non-negative cold-start samples |
| `memory_bytes` | finite non-negative RSS sample per unique frame |
| `vram_bytes` | `null` for CPU; finite non-negative sample per unique accelerator frame |
| `service` | full-path timing, queue, cadence, output timeline, and boundary proof |

Run provenance has exactly `kind`, `reference`, and
`contains_private_pixels`. Its kind uses the plan provenance enum, its reference
is non-empty for consented/licensed evidence, and `contains_private_pixels`
must be `false`. Plan, run, and annotation provenance kinds must agree.

`source` has exactly `clip_sha256`, `frame_sha256`, and `pixel_contract`.
`clip_sha256` is a lowercase SHA-256 digest; `frame_sha256` is a non-empty
ordered list of at most 100,000 lowercase SHA-256 digests, one per unique
replay frame; and `pixel_contract` is exactly
`canonical-pre-resize-rgb8`. This identity is computed before canvas resize so
it can bind different canvas targets to the same captured pixels without
retaining them in the content-free run sidecar.

Hardware has exactly `id`, `label`, `platform`, `identity_sha256`,
`identity_source`, and `inventory`. The ID matches the cell.
`identity_source` is exactly `qualification-hardware-inventory-v1`.
`inventory` has exactly:

- `cpu_model`: a path-free label;
- `accelerators`: a unique list of at most eight path-free labels;
- `memory_bytes`: a positive integer; and
- `os_name`, `os_version`, and `architecture`: path-free labels.

`identity_sha256` is the SHA-256 of the canonical JSON bytes for that exact
inventory object (sorted keys, compact separators, ASCII escaping, and a final
newline). The strict run loader rejects an inventory/digest mismatch. Label,
platform, digest, source, and therefore inventory identity remain identical
for that hardware ID in every run, and one digest cannot represent two
hardware IDs. `label` is a 1–128 character path-free label using letters,
digits, spaces, dots, underscores, or hyphens; `platform` is a lowercase safe
ID. Keep usernames, hostnames, serial numbers, and local paths out of these
report-facing fields.

Model has exactly `id`, `sha256`, `bytes`, `license`, `license_reviewed`,
`packaging_supported`, `download_integrity`, and `startup_succeeded`. Identity
and size match the candidate. The four flags must be booleans and all must be
`true` to qualify; `license` must be non-empty. For a custom `model_path`,
`model_id` is its path-free basename. Per-frame RVM telemetry exposes that
same basename—not the private local path—and must match the candidate/run
model ID, digest, byte count, and `model_builtin: false`.

Provider has exactly `requested`, `execution_proven`, and `runtime`.
`requested` is `cpu`, `cuda`, or `directml` and matches the cell; execution
must be proven to qualify. `runtime` has exactly `onnxruntime_version`,
`execution_provider`, `provider_runtime_version`, and `driver_version`, all
path-free labels. The execution-provider identity is exact:
`CPUExecutionProvider` for CPU, `CUDAExecutionProvider` for CUDA, and
`DmlExecutionProvider` for DirectML. The bundle independently proves the
provider on every frame:

| Cell provider | Per-frame acceleration request | Required state and active provider |
| --- | --- | --- |
| `cpu` | `requested_mode: cpu`, `requested_provider: auto` | `state: cpu_fallback`, `active_provider: cpu` |
| `cuda` | `requested_mode: gpu_required`, `requested_provider: cuda` | `state: gpu_active`, `active_provider: cuda` |
| `directml` | `requested_mode: gpu_required`, `requested_provider: directml` | `state: gpu_active`, `active_provider: directml` |

Every acceleration snapshot is applicable, has a non-negative integer
`device_id`, and remains byte-for-byte stable. `fallback_active` is `false`,
`fallback_count` is `0`, and `fallback_reason_code` is empty. The per-frame RVM
telemetry must agree on provider state, model identity, canvas shapes,
configured/resolved ratio, and RVM phase timings. A late fallback is therefore
not hidden by a successful startup attestation.

For every `(hardware_id, provider)` pair, the hardware identity digest, full
provider-runtime object, effective accelerator `device_id`, and complete
`resource_evidence` object must remain identical across candidates, canvases,
cadences, and render modes. The device ID for an accelerator must index the
declared `hardware.inventory.accelerators` list. The report binds this stable
environment as `provider.environment_sha256`; drift makes all affected rows
not decidable.

`resource_evidence` has exactly:

| Key | Required value |
| --- | --- |
| `sample_alignment` | `one-per-unique-frame` |
| `rss_source` | `process-api` or `external-sampler` |
| `rss_semantics` | `process-current-resident-bytes` |
| `sampling_point` | `post-sink-submit` |
| `vram_applicable` | `false` for CPU; `true` for CUDA/DirectML |
| `vram_source` | `not-applicable` for CPU; `provider-api` or `external-sampler` for accelerators |
| `vram_semantics` | `not-applicable` for CPU; `process-current-allocated-bytes` for accelerators |

`memory_bytes`, service samples, and queue samples contain exactly one value per
unique replay frame. RSS and VRAM values are non-negative integer byte counts.
Accelerator `vram_bytes` has the same exact count; CPU uses JSON `null`.
Optional RSS/VRAM samples embedded in the bundle, when present, must cover all
frames and exactly match the sidecar arrays.

`recurrence` has exactly `input_selection`, `model_invocation_count`,
`fresh_temporal_state`, and `output_projection_used`. Native cells use
`native-all`; decimated cells use
`every-other-native-preserved-timestamps`. Invocation count equals the replay
frame count, each run begins with fresh temporal state, and the latter two
flags are exactly `true` and `false`, respectively. Decimated bundle identity
is also checked against every second frame of its native parent.

`service` has exactly `new_frame_service_ms`, `queue_age_ms`, `duration_s`,
`capture_fps`, `unique_composite_fps`, `output_fps`, `output_repeat_count`,
`output_timeline_complete`, `boundary`, and
`bundle_frame_total_alignment`. All scalar/sample numbers are finite and
non-negative; sample arrays are non-empty and bounded to 100,000 entries.
Duration, all three cadence values, repeat count, and output completeness must
agree with the digest-bound replay timeline. In addition, the evidence source
`capture_fps` must be within the ratified cadence range: inclusive `29.0–31.0`
for `native30`, or inclusive `14.5–15.5` for `decimated15`. Output repeats
never raise `unique_composite_fps`.

The boundary string is exactly:

```text
unique-dequeue-through-sink-submit-excluding-deliberate-pacing
```

`bundle_frame_total_alignment` is either `exact-non-pacing-sink` or
`separate-pacing-boundary`. Only `exact-non-pacing-sink` is authoritative for
qualification: every service sample must equal that replay frame's
`frame_total_ms` within 0.000001 ms. Each frame must also contain a stable
`output_sink` snapshot with exactly `applicable`, `backend`, and `paces`;
`applicable` is `true`, `backend` is `null`, `pyvirtualcam`, `native`, or
`unknown`, and `paces` is `false`. A separate pacing boundary is retained as
evidence but makes the cell not decidable.

Every warm-up and steady frame must carry all twelve finite non-negative timing
fields:

- `rvm_preprocess_ms`, `rvm_session_run_ms`, and `rvm_postprocess_ms`;
- `backend_inference_ms`, `refinement_ms`, and `segmentation_ms`;
- `background_ms`, `color_correction_ms`, and `composite_ms`; and
- `frame_processing_ms`, `output_send_ms`, and `frame_total_ms`.

The qualifier also rejects contradictory nesting:
RVM preprocess + session + postprocess cannot exceed backend inference;
backend inference + refinement cannot exceed segmentation; segmentation +
background + color correction + composite cannot exceed frame processing; and
frame processing + output send cannot exceed frame total. The three RVM
telemetry phase values must exactly match their per-frame timing values.

`warmup_frame_count` must be at least 30 and leave at least 300 steady-state
frames; the steady-state service p95, not the warm-up distribution, is compared
with the cadence budget. Observation duration must be at least 10 seconds.
Unique-composite FPS and final-minus-initial queue age are independently
gated. The replay output timeline must contain exactly one non-repeat base
update for each frame. For each update, dequeue/service start is derived as
`sent_monotonic_ns - round(frame_total_ms × 1,000,000)`, and its queue sample
must equal `(derived_start_ns - capture_monotonic_ns) / 1,000,000` within
0.000001 ms. Derived start cannot precede capture or the previous frame's send,
and send cannot precede capture. This proves capture-to-dequeue queue age and
non-overlapping serialized service rather than trusting a caller-authored
array. The maximum sample is gated by `max_queue_age_ms`; final-minus-initial
growth is gated independently by `max_queue_age_growth_ms`.

## CLI outputs and exit status

The output path must not already exist. On a completed evaluation the command
creates an owner-only directory containing canonical `qualification.json` and
a compact `qualification.md`; this happens for both passing and evidence-based
failing decisions.

Every JSON report, including a failing or not-decidable report, contains
`qualification_contract`. It binds the full normalized policy (including
`quality_policy_id` and all gates), the path-free candidate catalog and model
identities, the exact qualified compositor and digest, canvases, and declared
hardware scope. The report evidence digest covers this contract. A qualified
profile definition adds its exact environment/canvas/cadence/render scope, but
the qualification contract is present even when no definition is published.

| Exit | Meaning |
| --- | --- |
| `0` | coverage is complete and every row is `qualified`; profile status is either `qualified` or `not_proposed` |
| `1` | a row is rejected, unavailable, or not decidable; coverage is incomplete; or proposed profiles are not decidable |
| `2` | CLI usage, unsafe/malformed input, unsupported schema, digest/binding error, or output/read/write failure |

An empty `profile_proposals` list intentionally produces
`profiles.status: not_proposed`; it does not turn an otherwise complete,
all-qualified matrix into a failure. Generated/proxy evidence produces
not-decidable rows and therefore exits 1.

## Why qualification is separate from screening

`custback matte-ablate` is a bounded same-source screen. Its RVM shortlist
narrows the experiment, but a one-host or generated-proxy row cannot establish
a portable real-time profile. The qualifier therefore joins three independent
evidence boundaries:

1. a digest-verified MATTE-0.3 report, used only to authorize model-backed
   shortlist IDs;
2. the direct replay bundle and digest-bound annotations for every formal
   matrix cell, used to recompute alpha/detail/cadence metrics and prove pixel
   identity; and
3. a bundle-bound, content-free run attestation for facts that replay version
   1 cannot infer, including hardware, cold startup, complete service time,
   latency growth, RSS/VRAM, provider proof, and fallback history.

Quality report JSON alone is insufficient because it cannot prove the
native-to-decimated source mapping or the raw-model/qualified-compositor pair.
An ablation label alone is insufficient because `model-backed` is declared by
the local plan and the ablation is explicitly a screening result.

## Required matrix

Every proposed candidate must cover the exact Cartesian product:

```text
candidate
  × declared hardware and actual provider
  × target canvas
  × native 30 FPS and true 30→15 recurrent decimation
  × raw-model attribution and qualified compositor
```

Candidates are limited to:

- model-backed RVM rows in the MATTE-0.3 bounded shortlist;
- automatic detail (`rvm_downsample: 0`); and
- the current compatibility baseline.

Frozen intermediates, cadence projections, unavailable rows, and generated
proxies are useful diagnostics but cannot fill a formal cell. A missing or
unsupported provider remains explicit; CPU fallback never substitutes for a
CUDA or DirectML cell.

The project-supported RVM provider set is CPU, CUDA, and DirectML. A named
cross-device profile remains unqualified until its declared platform scope
covers CPU and every applicable accelerator, at every target canvas. A report
may still retain useful hardware-scoped candidate results without claiming a
portable profile.

## Identity proofs

The qualifier fails closed unless all of these contracts hold:

- Native-30 cells for a canvas use the same raw pixels, order, capture
  sequences, timestamps, capture generation, and geometry generation across
  candidates, hardware, providers, and render modes. Capture lineage without
  the canvas-dependent pixel digest is also identical across canvas targets.
- A decimated-15 cell is exactly every second native input with the original
  sequence, timestamp, and generation values retained. Merely rewriting
  timestamps or projecting cadence does not rerun RVM recurrence and does not
  qualify. Its run source keeps the same pre-resize clip digest and pixel
  contract and uses exactly every second native `frame_sha256`.
- The raw and user-visible cells have identical raw input, raw `pha`,
  post-refiner alpha, RVM `fgr`, and backdrop artifacts. Their segmentation
  policy, resolved ratio, model artifact, provider, and fallback state are
  identical.
- Annotation identity is shared within each canvas/cadence source family,
  including provenance, segment declarations, data-defined gates, frame
  sequence/segment assignment, registration transforms, annotation array
  bytes/digest/dtype/shape, and named-region identity. Local paths are excluded
  from that comparison. A decimated annotation manifest must preserve the
  native provenance and gate list and contain the exact every-other frame
  projection of segment kind, registration, artifacts, regions, and their
  descriptors.
- Annotation segment kinds are exactly `stationary`, `moving`, `fast_motion`,
  and `occlusion`. Each kind must cover opaque core, background, ground-truth
  alpha, and ground-truth foreground evidence.
- Every frame reports `RVMSegmenter`, one stable actual provider, no
  accelerator fallback, one model SHA-256/byte identity, and one resolved
  detail ratio.

Local paths and raw exception text are not copied into the report. Provider
failures use a stable reason code.

## Effective policy proof

Configured intent is not accepted as proof of the rendered policy. Every frame
must carry a stable local-background identity with exactly `mode`, `fit_mode`,
`anchor_x`, and `anchor_y`; the background mode is limited to `blur`, `image`,
`video`, `color`, or `camera`. The recorded segmentation and compositing
configuration must normalize to the candidate and the render-mode compositor
exactly.

For each frame the qualifier independently resolves the
`TRUE_ALPHA_RECURRENT` backend policy from the candidate, expected compositor,
actual canvas, and resolved RVM ratio. Video/camera background is the only
light-wrap stabilization-eligible case. Qualification requires both the full
`matte_policy` object and these flattened effective facts to match that
resolution:

- `produces_matte`;
- `edge_refinement_mode` and `edge_refinement_radius_px`;
- `mask_shift`;
- `use_model_foreground`;
- `light_wrap`; and
- compositor `blend_space`.

The effective refiner must also equal the resolved refiner configuration after
excluding its local `model_path`. Raw-model cells derive their expected policy
from the declared compositor with model-foreground substitution disabled and
light wrap exactly zero; user-visible cells use the exact declared qualified
compositor. A tampered refiner, foreground source, blend space, light wrap, or
full policy snapshot rejects attribution even when configured intent still
looks correct.

## Automatic and explicit detail

Configured and effective detail remain distinct:

- `rvm_downsample: 0` is configured automatic intent;
- the effective ratio is `null` until a successful RVM result is validated;
- automatic detail resolves to
  `min(1, max(0.125, 512 / max(canvas_height, canvas_width)))`;
- an explicit value resolves to that exact configured value; and
- a reset clears the last successful value until the next successful frame.

Qualification checks the value on every frame, not just frame zero. The report
may show the nominal internal long edge (`canvas_long_edge × ratio`), but does
not call that an exact tensor shape unless the model runtime provides such a
fact.

## Quality gates

There is no composite quality score. Each required annotated family is an
independent fail-closed gate:

- opaque-core alpha p05, fraction below 0.95, and foreground holes;
- annotated-background alpha mean plus exterior halo area and width;
- ground-truth alpha MSE, gradient error, and protected fine-detail coverage;
- registered contour and alpha motion; and
- fast-motion trail behavior.

All data-defined annotation gates must be present and pass. `not_evaluated`,
missing, or null is not a pass. In particular, excellent temporal contour
stability cannot compensate for an under-opaque shirt, missing headphones, an
exterior halo, or failed fine detail. A missing gate list makes evidence
incomplete, a `fail` gate rejects the row, and any status other than `pass`
makes the gate evidence not decidable.

Raw-model cells establish alpha attribution. Qualified-compositor cells add
the user-visible cost without changing the alpha decision.

## Performance and resource evidence

Compatibility `backend_inference_ms` remains the historical combined
segmentation bucket. Formal evidence additionally records separate RVM:

- preprocessing;
- ONNX session execution; and
- output validation/conversion postprocessing.

Warm-up and steady-state p50/p95 are reported separately for those stages,
refinement, and the complete new-frame path. A replay's first recorded frame
is not automatically called a cold start: the run attestation must prove a
fresh session and record startup/first-inference measurements.

The complete path starts when a unique captured frame is dequeued for
processing and ends after privacy guard, final validation, and sink send.
Qualification uses that distribution, not a live EWMA and not the narrower
pre-guard `frame_processing_ms` bucket.

Every run also reports:

- process RSS;
- accelerator VRAM when applicable, otherwise explicit non-applicability;
- sustained capture, unique composite, and output cadence;
- deadline and latency growth; and
- the exact hardware inventory identity; and
- requested/active provider, runtime/driver identity, and fallback history.

A row whose p95 misses its declared frame budget is rejected for that target
and marked degraded. It is never silently advertised as real-time at a lower
rate. Output repeats do not count as unique composites.

## Decisions and defaults

Each cell is `qualified`, `rejected`, `not_decidable`, or `unavailable`.
Budget state is reported separately as `within_budget`, `degraded`, or
`not_measured`.

- A measured quality, model, provider, or performance gate failure rejects the
  cell.
- Missing metrics, incomplete timing/resource evidence, or a missing identity
  proof makes the cell not decidable.
- An unavailable provider is retained as unavailable and is never replaced.

A candidate is qualified only when every required cell is qualified. Proposed
`performance`, `balanced`, and `quality` names are accepted only as explicit
plan proposals that independently satisfy the complete cross-device matrix.
The qualifier never chooses names using a hidden weighted score.

For every hardware/provider/canvas/cadence/render scope, their steady-state
full-frame service p95 must satisfy:

```text
performance <= balanced <= quality
performance < quality
```

For every required quality metric, `balanced` must lie between `performance`
and `quality` in the metric's declared direction, `quality` must be non-worse
than both, and at least one metric must be strictly better for `quality` than
`performance` in every scope. Swapping labels therefore fails even when all
three candidates independently pass their absolute gates.

When profile status is `qualified`, each published definition is
self-describing. It includes the exact model ID/digest/size; normalized
path-free segmentation and its digest; the qualified compositor and its
digest; semantic-ordering contract; and each qualified hardware inventory,
provider/runtime identity, canvas dimensions, cadence set, and render-mode
set. Consumers must not apply a definition outside that explicit
qualification scope.

Checked-in deterministic evidence is generated proxy evidence, and the
copyable model-backed objects are inert templates rather than measurements.
No checked-in artifact therefore selects a named profile:

```text
recommended_profiles = {}
production_profile_selected = false
global_default_changed = false
high_detail_global_default = false
```

The template pack also retains a content-free `local_review` index for recording
local artifact digests and human review. Only separately copied and completed
objects below `templates` have the strict plan/run shape described above.

## Rollback

This task does not add a live profile setting. Existing ratio experiments use
the transactional segmentation rebuild and roll back by restoring the exact
previous configuration:

```json
{"segmentation":{"rvm_downsample":0.0}}
```

That JSON is only the schema-1 compatibility example. An operator must restore
the value saved from authenticated `GET /config`; persisted installations may
have a different reviewed value. Rollback rebuilds RVM and clears recurrent
state before the next authoritative input.
