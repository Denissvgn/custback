# Backend, postprocess, compositor, and cadence ablation

Status: **MATTE-0.3 implementation and generated screening protocol**

`custback matte-ablate` runs a bounded, same-source exploratory screen over a
private MATTE-0.1 replay. Its canonical output schema is
`custback.matte-ablation-report`, version 1. The result is a one-host shortlist,
not a cross-device qualification or production preset.

## Evidence classes

The plan distinguishes four row kinds:

| Kind | What it may establish |
| --- | --- |
| `frozen` | Reapply mask postprocess or compositor policy to recorded raw `pha`, RVM `fgr`, source, and backdrop without rerunning a model. |
| `recorded` | Join a separately captured backend/model/device/cadence bundle after proving raw-source frame order and, normally, timestamps are identical. |
| `cadence_projection` | Project unique-input/output-send cadence and exact repeats. It never claims recurrent model quality or creates a new alpha value. |
| `unavailable` | Record a missing provider/model/platform with a reason. No fallback row is substituted. |

Every row declares `evidence_kind`: `model-backed`, `recorded-intermediate`,
`generated-proxy`, `projection`, or `unavailable`. Generic edge refinement,
blur, or EMA is rejected on an RVM lane unless the row explicitly opts into an
experimental RVM policy. Ordinary RVM frozen rows may vary `mask_shift`; ratio,
model, backend, device, and recurrent-cadence claims require a separately
recorded row.

Frozen rows use the same typed backend-policy resolver as the live pipeline.
Normal RVM rows therefore preserve native soft alpha and bypass generic
blur/edge/EMA. Only an already-guarded `explicit_rvm_policy: true` row supplies
the private `experimental_rvm_generic` resolver input. Each derived bundle
records the resulting configured/effective/state/reason snapshot under
`effective_controls.matte_policy`; this evidence-only flag is not a persisted
configuration or live API control. Generic RVM fields omitted by an explicit
row start from production-neutral values, and a compositor-only row cannot
claim that inherited blur/edge/EMA ran. See
[the backend-policy contract](matte-backend-policies.md).

Schema-version-1 MediaPipe refinement is
`spatial_edge_refinement.mode: legacy_watershed`. The opt-in
`stable_guided` mode is a distinct frozen-postprocess candidate; it must not be
relabeled as the historical on/off row. Its resolved canonical-canvas search
radius, configured/effective mode, soft-edge metrics, and fallback behavior
belong in the row evidence. See the
[spatial-refinement design record](matte-spatial-refinement.md).
Spatial-policy rows use the `spatial_policy` group and explicitly set both the
authoritative `edge_refine: true` switch and the nested policy; a nested mode
alone is rejected so an ablation cannot silently record a candidate it never
executed.

The owner-only plan has schema `custback.matte-ablation-plan`, version 1. It
records the required axes, one-host/model-inference scope, bounded shortlist
policy, and variants. Paths to private recorded rows are read from the plan but
never copied into the report. Every recorded row must declare
`expected_backend` and `expected_device`; an RVM row must also declare its
resolved `expected_rvm_downsample_ratio`. The runner compares those values with
the bundle's effective controls and fails the row instead of silently accepting
a fallback result.

```console
cp ./my-matte-ablation-plan.json ./private-plan.json
chmod 600 ./private-plan.json

custback matte-ablate ./private-run-b \
  --annotations ./private-run-b-annotations \
  --plan ./private-plan.json \
  --output ./private-run-b-ablation \
  --max-output-bytes 2147483648
```

The output directory must be new. It is owner-only and contains derived
variant bundles, digest-bound annotations, representative lossless contact
sheets, `ablation.json`, and `ablation.md`. These files contain identifiable
derived imagery and have the same privacy requirements as the source replay.
The runner preflights estimated variant volume and enforces the final byte
bound across variants, contact sheets, and reports.

## Matrix protocol

Use one native source sequence and enumerate these axes. A missing or
unlicensed row remains explicitly unavailable:

- active RVM accelerator, MediaPipe CPU fallback, and one bounded RVM CPU spot
  row;
- RVM auto/current plus the screened `0.4`, `0.5`, `0.67`, and `1.0` ratios;
- current MobileNet and only licensed, integrity-pinned model candidates;
- MediaPipe edge refine off, schema-version-1 `legacy_watershed`, and opt-in
  `stable_guided`; mask blur zero/current/wider; temporal smoothing
  zero/current/candidate;
- RVM `mask_shift` zero, both signs around zero, and current effective value;
- plain, foreground-only, wrap-only, and foreground+wrap full factorial on both
  a correct narrow soft edge and a broad under-opaque band;
- fixed and moving backdrops with legacy stateless wrap, the default-off
  `temporal_bounded` candidate, and separately identified lower-strength,
  temporal-only, bound, inner-edge, and excessive-change-gate candidates;
- legacy and linear-sRGB blend space for every shortlisted wrap candidate;
- native fixed 15/30/60, 30→15 temporal decimation, irregular/gapped cadence,
  and model-backed cadence rows where recurrent sensitivity is being claimed;
- output cadence equal to unique input and 2× exact-repeat output.

The plan’s `required_axes` is compared with both attempted and completed axes.
Unavailable rows count as attempted but not completed. A report with
uncompleted required axes exits nonzero while retaining its useful bounded
screening evidence. Machine-readable `coverage.model_backed` separately lists
the recurrent cadence axes established by recorded model inference. Fixed,
decimated, and irregular projections therefore exercise scheduling math without
claiming recurrent-model quality.

For every completed image row, the report retains:

- raw `pha`, post-refiner alpha, clean foreground where available, and final
  composite;
- source-pixel/timestamp identity proof and all artifact/report digests;
- MATTE-0.2 opaque-core, holes, halo, leakage, contour, trail, fine-detail,
  edge-color, and cadence metrics;
- first-frame warm-up and later steady-state timing summaries, including
  available compositor substages; and
- an owner-only representative contact sheet.

Alpha motion and edge-color motion remain separate quantities. The compositor
factorial reports edge variation, opaque leakage, and compositor p95 cost for
each combination relative to the plain control. A 15→30 repeat projection
records twice the sends but the original model invocation count and alpha
digest set.

Dynamic-wrap rows additionally declare `paired_no_wrap_id`, pointing to an
image-producing compositor row (or `baseline`) that is the same-frame wrap-off
control. Forward references are allowed; missing, self, cadence, unavailable,
or non-compositor targets are rejected. The pair must have identical
source/backdrop artifacts and timestamps, alpha digests, configured
stabilization policy, model-foreground decision, blend space, and color
transform; only the scalar `light_wrap` value differs. The report keeps the
existing final `edge_band_rgb_variation` and the paired
`light_wrap_attributable_rgb_variation`, computed from temporal movement of
`base_wrap_on - base_wrap_off` in the held-alpha soft-edge band. It must not
use post-base output or infer the paired value by subtracting aggregates.

Each paired row retains the aggregate p95 under `quality`, full and
per-segment summaries, per-frame content-free scalars, proof flags, and an
identity-contract digest under `attribution.light_wrap_pair`. Missing pairs,
null metrics, changed alpha, or an all-null required segment make the pair
explicitly `not_decidable`; zero is never substituted. Absent narrow/broad and
fixed/moving coverage likewise prevents a MATTE-2.4 decision. Scene-cut/seek
rows, static-appearance deltas, exact `light_wrap: 0` bypass, and comparable
compositor cost remain separate gates. See the
[dynamic light-wrap contract](matte-light-wrap.md).

`required_decisions.dynamic_light_wrap` therefore remains `not_decidable`
unless the external review has all of those rows and separate gates. It lists
the declared pair statuses and computed count but never promotes one valid
pair—or generated proxy evidence—into a production default.

Spatial candidates are additionally compared on geometrically identical
360p/720p/1080p fixtures, alternating nearby gradients, low/uniform contrast,
compression-like blocks, camera noise, and thin/soft components. A lower
single-frame displaced-edge error cannot shortlist a row that increases
stationary contour p95, collapses soft alpha, changes protected topology, or
exceeds the refinement budget.

## Decisions and rejection reasons

The report applies serialized screening policy rather than hidden production
constants. Opaque-core, hole, halo, registered jitter, motion-trail,
ground-truth detail, and optional host frame-budget checks fail a row. Material
improvement is assessed separately for contour jitter, edge-color motion,
opaque leakage, ground-truth error, and comparable-basis processing cost.

The checked-in MediaPipe “watershed on” generated row is a directional proxy:
its refined alpha was generated as a stable control and the production
watershed did not run. It therefore cannot select watershed, `stable_guided`,
or a default. Only same-source model-backed rows that actually execute the
named spatial policy can resolve that decision.

Shortlists are bounded per lane and are routed separately to MATTE-2.5 (RVM
profile qualification) and MATTE-3.4 (processing/compositor cost). Every
rejected row names one or more of the backlog categories: jitter, ghosting,
detail loss, performance, platform availability, or complexity. The first RVM
and MediaPipe fallback experiments are reported separately. Even a successful
one-host row remains a qualification candidate, never a production preset.
MATTE-2.5 consumes that shortlist only as an admission boundary and then
requires a new model-backed hardware/canvas/cadence/compositor matrix; see the
[RVM profile qualification contract](matte-rvm-profiles.md).
Only reviewed model-backed candidates may be carried into the reversible
[operator mitigation procedure](matte-operator-mitigations.md); generated
screening rows cannot authorize an operator value.

## Generated acceptance screen

`tests/matte_ablation_evidence.py` builds only MIT-licensed synthetic pixels.
It exercises 20 variants plus the baseline: same-source generated RVM and
MediaPipe proxies, the full compositor factorial, blend space, mask shift,
postprocess, constant/video backdrop, all required cadence projections, and
explicit unavailable CUDA/RVM-CPU rows. It deliberately marks
`model_inference_executed: false`; its directional watershed result and RVM
ratio shortlist remain proxy-only. In particular, the stable MediaPipe proxy
is not evidence that `_watershed_edge_snap` or `stable_guided` executed.

```console
PYTHONPATH=src .venv/bin/pytest -q tests/test_matte_ablation.py
PYTHONPATH=src .venv/bin/python tests/matte_ablation_evidence.py \
  --work-dir /tmp/custback-matte-ablation-work \
  --json /tmp/custback-matte-ablation-generated.json
```

No private Run-B pixels or fabricated model measurements are checked in. Use
[`matte-ablation-local-reference-template.json`](matte-ablation-local-reference-template.json)
to retain only digests and reviewed decisions from the first consented,
model-backed local run.
