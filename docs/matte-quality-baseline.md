# MATTE-0.2 baseline report

Status: **deterministic generated baseline complete; local model-backed evidence
requires a consented/licensed clip**

This is the review record for the evaluator, generated fixtures, and
provisional gates. It compares a reported-style MediaPipe CPU policy proxy with
the current RVM policy proxy over identical generated raw sources and monotonic
timestamps. A proxy is a controlled matte input, not model inference. The
repository has no consented user footage, installed MediaPipe/RVM runtime, or
model artifacts, so this report does not pretend to contain a real backend
benchmark.

That distinction is intentional: it gives CI a deterministic oracle while the
local qualification workflow binds real evidence by digest without committing
private pixels.

## Reproduction and environment

```console
PYTHONPATH=src .venv/bin/pytest -q tests/test_matte_quality.py
PYTHONPATH=src .venv/bin/python tests/matte_quality_evidence.py \
  --json /tmp/custback-matte-quality-baseline.json
```

Two consecutive runs on the review host were byte-identical. The generated
suite JSON SHA-256 was
`2bda474ef73bb97597d9a411ca7b5f8640c11c2f6aa29ef7b8193080692b2286`.
Metrics are rounded to eight decimals with `1e-6` cross-platform tolerance.

| Recorded field | Value |
| --- | --- |
| Host | Linux `7.0.0-28-generic`, x86_64, 20 logical CPUs |
| Python | `3.14.4` |
| NumPy | `2.5.1` |
| OpenCV | `5.0.0` |
| Device | CPU |
| Source/detail | Generated `96×72`, same pixels and timestamps per comparison |
| Resampling | OpenCV linear affine; signed-distance mask at native fixture size |
| Reported-style configuration | MediaPipe CPU policy proxy, model-selection-1-style full-frame person confidence |
| Current-path configuration | RVM policy proxy, ratio `0.4`, raw `pha`, `mask_shift=0`, model foreground available |
| Compositor | `srgb_legacy`, light wrap `0.0` unless the isolated dynamic fixture says otherwise |
| Cadence | Six unique inputs at `30.0000003 FPS`; six sends, no repeats in comparison clips |

Timing, allocation, and memory samples in generated bundles are deterministic
fixture values for schema coverage, not backend benchmarks. Real replay
reports mark allocation/memory unavailable unless instrumentation supplied
those samples; artifact volume and evaluator array working set always remain
separately named.

## Controlled comparison

| Fixture / p95 metric | Reported-style MediaPipe CPU proxy | Current RVM path proxy | Interpretation |
| --- | ---: | ---: | --- |
| Stationary 2 px alternating jitter: signed-distance contour | `2.00000000 px` | `0.00000000 px` | The `1.5 px` gate rejects the defect; the controlled current proxy exceeds the required 40% reduction. |
| Same fixture: compensated alpha absolute difference | `0.04060044` | `0.00000000` | Registration does not erase stationary estimator jitter. |
| Same fixture: ground-truth alpha MSE | `0.01687053` | `0.00000000` | The `0.01` generated-fixture tolerance rejects the coarse displacement. |
| Fast motion/occlusion: motion trail area | `0.00000000` | `0.00000000` | Neither accurate proxy freezes or trails; the dedicated frozen fixture fails at `3.08709014` p95 and remains dominant for five intervals. |
| Opaque accessories: core alpha p05 | `1.00000000` | `1.00000000` | Both clean controls pass. The stable `0.80` fixture fails core p05 and below-0.95 gates while contour jitter remains zero. |
| Opaque accessories: background alpha mean | `0.00000000` | `0.00000000` | Clean controls do not create exterior mass. Hole/halo fixtures cover topology and signed-distance width separately. |
| Dynamic backdrop, fixed alpha: contour / compensated alpha | `0 / 0` | `0 / 0` | Backdrop animation is not mislabeled as alpha instability. |
| Dynamic backdrop, fixed alpha: edge-band RGB variation | `0.23724316` | `0.23724316` | Changing the matte backend alone cannot fix light-wrap/color shimmer. This remains a separate compositor-stage comparison. |

The first comparison evidence digests are
`dce38f8120b15ba4c1bf2a1654947ad19ed9d2269a070457e7784b849b916cc9`
(reported proxy) and
`dd7efd3cfaaf4571295f7bf467f56287c3a34ce92d59e65d66f4b400b648a7ff`
(current proxy).

## Ratified starting gates

These remain qualification-report data rather than constants in runtime code:

- stationary compensated contour p95 `<= 1.5 px` and candidate
  `<= 0.60 ×` its comparison baseline;
- stationary summed-alpha area drift p95 `<= 1%`;
- fast-motion trail p95 `<= 1.10 ×` the unstabilized comparison and no
  dominant previous contour for more than one unique-input interval;
- annotated opaque-core alpha p05 `>= 0.95`, with no more than `5%` of core
  below `0.95`;
- annotated-background mean alpha `<= 0.01`; halo width, topology, and mass
  additionally use fixture/local-clip bounds;
- generated ground-truth alpha MSE p95 `<= 0.01`, with fine-detail SAD and
  gradient metrics reviewed alongside it; and
- a light-wrap/shimmer candidate must reduce fixed-alpha dynamic-backdrop
  edge-band RGB variation p95 by at least `30%` without failing static
  appearance gates.

The `30%` threshold is ratified as the Phase-0 comparison criterion, not
claimed as achieved by the current path. The controlled baseline shows why it
must remain independent of alpha stability.

## Local qualification boundary

Private footage is deliberately not committed. A local operator must record a
licensed or explicitly consented clip with MATTE-0.1, create private
core/background/ground-truth annotations where available, and retain:

- non-identifying qualification ID and consent/license reference;
- replay manifest SHA-256 and annotation manifest SHA-256;
- backend/device, model artifact digest/version, effective RVM ratio or
  MediaPipe detail, resampling, cadence, and full effective configuration;
- generated report evidence SHA-256; and
- report review status.

The digest-only reference format is in
[`matte-quality-local-qualification-template.json`](matte-quality-local-qualification-template.json).
Its checked-in state is intentionally `awaiting-local-evidence`; replacing it
with made-up footage, consent, or model results would violate the privacy and
evidence contract.
