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
- MediaPipe edge refine off/on; mask blur zero/current/wider; temporal
  smoothing zero/current/candidate;
- RVM `mask_shift` zero, both signs around zero, and current effective value;
- plain, foreground-only, wrap-only, and foreground+wrap full factorial;
- legacy and linear-sRGB blend space;
- constant backdrop and the same container-timed 24 FPS video;
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

## Decisions and rejection reasons

The report applies serialized screening policy rather than hidden production
constants. Opaque-core, hole, halo, registered jitter, motion-trail,
ground-truth detail, and optional host frame-budget checks fail a row. Material
improvement is assessed separately for contour jitter, edge-color motion,
opaque leakage, ground-truth error, and comparable-basis processing cost.

Shortlists are bounded per lane and are routed separately to MATTE-2.5 (RVM
profile qualification) and MATTE-3.4 (processing/compositor cost). Every
rejected row names one or more of the backlog categories: jitter, ghosting,
detail loss, performance, platform availability, or complexity. The first RVM
and MediaPipe fallback experiments are reported separately. Even a successful
one-host row remains a qualification candidate, never a production preset.

## Generated acceptance screen

`tests/matte_ablation_evidence.py` builds only MIT-licensed synthetic pixels.
It exercises 20 variants plus the baseline: same-source generated RVM and
MediaPipe proxies, the full compositor factorial, blend space, mask shift,
postprocess, constant/video backdrop, all required cadence projections, and
explicit unavailable CUDA/RVM-CPU rows. It deliberately marks
`model_inference_executed: false`; its directional watershed result and RVM
ratio shortlist remain proxy-only.

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
