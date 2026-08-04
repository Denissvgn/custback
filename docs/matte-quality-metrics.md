# Matte quality metrics and annotation contract

Status: **MATTE-0.2 metric contract, version 1**

`custback matte-evaluate` is an offline consumer of complete
`custback.matte-replay` version-1 bundles. It never opens capture, segmentation,
network, preview, or virtual-output resources. The canonical report schema is
`custback.matte-quality-report`, version `1`.

Metric values are rounded to eight decimal places. Deterministic generated
evidence is compared with an absolute cross-platform tolerance of `1e-6`;
recorded timing and resource samples are host observations and are not expected
to reproduce across machines.

## Annotation bundle

Private annotations use schema `custback.matte-quality-annotations`, version
`1`. `annotations.json` binds to the exact replay `manifest.json` SHA-256.
Every lossless NPY array is also bound by relative path, size, SHA-256, dtype,
and shape, with pickle loading disabled.

Each unique input belongs to a named segment of kind `stationary`, `moving`,
`fast_motion`, or `occlusion` and can provide:

- an affine transform mapping the previous source coordinates to the current
  source coordinates;
- a boolean opaque-core mask;
- a boolean known-background mask;
- optional float32 ground-truth alpha; and
- optional uint8 BGR ground-truth foreground.

For stage attribution, a frame can also provide optional named regions. Region
names are lowercase identifiers, each region has a boolean mask, and its trimap
kind is `opaque_core`, `background`, or `soft_boundary`. The MATTE-0.5 Run-B
protocol uses `torso`, `shoulders`, `head`, `headphones`, and `hair` (plus a
known-background region). This is a backward-compatible version-1 extension:
older annotation manifests without `regions` remain valid and retain their
original digest.

Annotations are as sensitive as replay silhouettes. Directories and files are
owner-only, paths cannot escape the annotation root, and an annotation set
cannot be applied to another replay after either manifest changes.

The annotation manifest may contain release gates. Gates are report data—not
production constants—and use an aggregate metric path, `<=` or `>=`, and
either an absolute value or a baseline multiplier.

## Numeric definitions

| Family | Report metric | Definition |
| --- | --- | --- |
| Cadence | `unique_input_fps` | `(unique inputs - 1) / elapsed capture-monotonic time`. Capture sequence gaps and implied missing-input count are reported separately. |
| Cadence | `output_send_fps` | `(send events - 1) / elapsed send-monotonic time`. Null when the older bundle has no scalar output timeline. |
| Cadence | offline `output_repeat_ratio` | Consecutive equal final-artifact SHA-256 transitions divided by all output transitions. This is the retained-artifact implementation of exact final-output equality. |
| Cadence | base update/reuse FPS | Update or reuse events after the first send divided by the output observation interval. Counts are also retained. |
| Temporal alpha | raw/refined absolute difference | Mean absolute alpha difference against the preceding unique input without registration. |
| Compensated alpha | compensated absolute difference | Mean absolute difference against the previous refined alpha warped by the annotated affine transform. Without annotations, deterministic source-luma phase correlation estimates translation. |
| Contour | displacement p50/p95 | Absolute difference between current and registered-previous signed-distance fields, sampled over the union of their 3×3 contour bands. |
| Area | stationary subject drift | Absolute change in summed refined alpha from the first frame of the stationary segment, divided by that reference area. |
| Soft edge | width | Count of pixels satisfying `0.05 < alpha < 0.95` divided by the `alpha >= 0.5` exterior contour perimeter. |
| Soft edge | uncertain fraction | Fraction of the complete frame satisfying `0.05 < alpha < 0.95`. |
| Opaque core | p05/p50 and deficit | Alpha quantiles, mean `1-alpha`, and fractions below `0.95` and `0.90` inside the annotated core. |
| Background | alpha/halo | Alpha sum and mean, fraction above `0.05`, and p95 distance of those false-foreground pixels into ground-truth exterior. |
| Topology | holes/components | 8-connected `alpha < 0.5` components in opaque core and `alpha >= 0.5` components in annotated background. |
| Leakage | coefficient | Least-squares coefficient of `(composite-source)` along `(backdrop-source)` in opaque core; correlation and normalized composite/source MAE are companion metrics. |
| RVM foreground | RGB error | Mean absolute clean-foreground versus annotated foreground RGB error where ground-truth alpha is between `0.05` and `0.95`, normalized by 255. Null when either track is absent. |
| Motion | trail area | Pixels belonging to the previous ground-truth contour but not the current contour that remain `alpha >= 0.5`, divided by current ground-truth foreground area. Consecutive intervals above `1%` form the dominance-run metric. |
| Edge color | RGB variation | Mean normalized RGB change from registered previous final composite where current alpha is uncertain and differs from registered alpha by no more than `0.01`. |
| Light-wrap attribution | `light_wrap_attributable_rgb_variation` | Mean temporal RGB change of the paired base-compositor contribution `W[t] = (C_wrap_on[t] - C_wrap_off[t]) / 255` in the held-alpha soft-edge band. Each same-frame pair differs only in light-wrap strength; configured stabilization is identical and the preceding contribution is registered by the annotation transform. |
| Ground truth | SAD/MAE/MSE/gradient | Sum/mean absolute alpha error, mean squared alpha error, and mean magnitude of the Sobel-gradient error. |
| Performance | timing p50/p95 | Every recorded segmentation, RVM preprocessing/session/postprocessing, refinement, background, compositor, compositor-substage, send, and full-frame timing is summarized independently. `frame_total_ms` ends after sink submission; the narrower compatibility `frame_processing_ms` remains separate. |
| Resources | allocation/memory | Instrumented `allocation_bytes`, `memory_bytes`, process `rss_bytes`, and accelerator `vram_bytes` samples are summarized independently when present. Null with `available=false` is mandatory when the recorder did not carry a sample. Diagnostic artifact volume and evaluator loaded-array bytes are separate, always-available measurements and are not mislabeled as runtime allocation. |

The paired light-wrap metric isolates temporal movement introduced by wrap
from legitimate moving-backdrop color visible through fractional alpha. It is
computed from per-frame base composites, not by subtracting two aggregate
`edge_band_rgb_variation` values or using a post-base output. Wrap-on and
wrap-off rows must have identical source/backdrop artifacts and timestamps,
raw/refined alpha digests, configured stabilization, model-foreground
selection, blend space, and color transform. The metric is null—and a
light-wrap qualification decision fails closed—when a required pair or
identity proof is absent or any required segment has no held-alpha interval.
It supplements rather than replaces final edge variation, static-appearance
comparison, scene-cut smear, and alpha invariance checks. See the
[dynamic light-wrap qualification contract](matte-light-wrap.md).

Per-frame metrics retain sequence, segment, registration method/matrix, and
capture timestamp. Aggregate summaries include count, mean, p05, p50, p95,
minimum, and maximum. Empty optional families contain null summaries rather
than invented zeros.

## Post-base extension isolation

The output timeline's optional provenance object must have schema
`custback.matte-post-base-output-provenance`, version `1`, a stage name, and a
metrics object. The evaluator copies it only into
`extensions.post_base.events[].provenance`. It never merges extension keys into
`aggregate.cadence`, alpha, base-update, or send metrics. A later reaction
stage can therefore add its own measurements without relabeling the
camera/matte/base contract.

Normal-run status uses the same isolation rule for its typed
`extensions.post_base` namespace. Its `exact_final_output_repeat_*` fields
compare each successful final frame's bytes with the immediately preceding
successful final frame without exposing a digest. The offline
`output_repeat_ratio` obtains the same equality signal from retained artifact
digests. Both may observe equality between two successful unique captures and
remain distinct from the synthesized/no-unread `base_composite_reuse_*`
provenance clock. See the
[visual cadence observability contract](cadence-observability.md).

## Generated evidence

`tests/matte_quality_evidence.py` generates all inputs in memory under the MIT
license:

- static confidence noise and alternating one-/two-pixel jitter;
- known translation and rotation;
- fast motion, occlusion, and deliberately frozen alpha;
- fine structures, semitransparent edges, light/dark clothing, shoulders,
  headphones, glasses, and opaque accessory cores;
- under-opaque core, holes, and exterior halo;
- constant alpha with a dynamic backdrop;
- 15/30/60 FPS and irregular monotonic timestamps; and
- output repeats and capture-sequence gaps.

The acceptance tests prove that frozen alpha fails motion lag despite zero
uncompensated flicker, two-pixel jitter fails the stationary contour gate,
dynamic backdrop changes the edge-color family while alpha metrics remain
zero, and stable under-opacity fails the core gate while passing jitter gates.
Generated paired wrap controls can validate
`light_wrap_attributable_rgb_variation` arithmetic and fail-closed coverage,
but they cannot ratify a production light-wrap policy or default.

Reproduce:

```console
PYTHONPATH=src .venv/bin/pytest -q tests/test_matte_quality.py
PYTHONPATH=src .venv/bin/python tests/matte_quality_evidence.py \
  --json /tmp/custback-matte-quality-baseline.json
```
