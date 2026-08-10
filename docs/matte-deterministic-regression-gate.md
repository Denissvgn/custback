# Deterministic matte regression gate

Status: **MATTE-5.1 deterministic test contract**

MATTE-5.1 is the fast regression gate for temporal matte behavior and the
quality defects already represented by generated fixtures. It is an
incremental umbrella: a newly selected matte algorithm or mask-producing path
must add its focused tests to this gate when it lands.

## Focused gate

From an editable development environment, run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_mediapipe_segmenter.py \
  tests/test_segmentation_timeline.py \
  tests/test_segmentation_rvm.py \
  tests/test_matte_stabilizer.py \
  tests/test_spatial_edge_refinement.py \
  tests/test_light_wrap.py \
  tests/test_processing.py \
  tests/test_matte_policy.py \
  tests/test_matte_quality.py \
  tests/test_matte_attribution.py \
  tests/test_pipeline.py
```

This command is a review shortcut, not a separate source of truth. The normal
CI jobs run the complete `python -m pytest -q` suite across supported Python
and dependency profiles, and the optional-backend jobs repeat it with
MediaPipe and RVM extras installed. Ruff, Pyright, Node, artifact, and
packaging gates remain independent and must also stay green.

## Coverage map

The mapping intentionally names modules and behaviors rather than individual
test functions. Renaming a test does not change the contract; deleting a
behavior does.

| Required behavior | Deterministic coverage |
| --- | --- |
| Capture-time conversion at 15/30/60 FPS, same-millisecond quantization, irregular gaps, and non-monotonic input | `test_mediapipe_segmenter.py` exercises MediaPipe's relative video clock and monotonic bump; `test_segmentation_timeline.py` owns discontinuity classification. |
| Reset on capture/geometry generation, provider fallback, long gap, and config commit; no reset on ordinary jitter or repeated output | `test_segmentation_timeline.py`, `test_segmentation_rvm.py`, and `test_pipeline.py` verify the paired segmenter/refiner boundary and that an output repeat performs no new matte work. |
| MediaPipe mask validation and interpolation | `test_mediapipe_segmenter.py` covers missing, malformed, non-finite, and out-of-range masks plus explicit upsample/downsample interpolation, clamping, dtype, and contiguity. |
| RVM recurrent state and evidence attribution | `test_segmentation_rvm.py` covers same-size restart/reset, recurrent feedback, fake provider recovery, native raw alpha versus post-`mask_shift` alpha, aligned frame identity, resolved downsample ratio, and effective policy evidence. |
| Opaque-core deficit, holes, exterior halo, backdrop leakage, fine edges, and stage attribution | `test_matte_quality.py` and `test_matte_attribution.py` use generated annotated fixtures to make each defect observable and to locate raw-alpha, post-refiner, clean-foreground, light-wrap, mixed, and final-blend failures. |
| Motion-aware temporal stabilization | `test_matte_stabilizer.py` covers stationary jitter at 15/30/60 FPS, irregular cadence, known translation and rotation, fast or untrusted motion, occlusion and newly revealed foreground, connected hair-like soft detail, zero/one/tiny masks, invalid values, bounded flow work, and every reset reason. |
| Stable, resolution-aware spatial refinement | `test_spatial_edge_refinement.py` covers alternating/ambiguous gradients, scale-normalized radii and output, low contrast/noise fallback, soft ramps, endpoint masks, thin connected features, malformed inputs, bounded work, and deterministic repeat calls. |
| Fixed-alpha dynamic light wrap and the model-foreground/wrap factorial | `test_light_wrap.py` covers model foreground on/off, wrap on/off, both blend spaces, exact stateless bypass, elapsed-time stabilization, resets, fixed-alpha preservation, and dynamic edge-variation reduction. |
| Optimized compositor equivalence, exact alpha endpoints, and bounded workspace lifetime | `test_processing.py` compares optimized and reference pixels for every model-foreground/wrap combination, verifies exact backdrop/foreground endpoints, independent outputs, allocation accounting, recovery, idempotent close, and rejected use after close. |
| Backend-specific effective policy | `test_matte_policy.py` covers every backend kind, configured-versus-effective values, RVM soft-alpha ownership and ratio reporting, bypass/inapplicability reasons, passthrough neutralization, and backend-switch rollback. |
| Transactional activation, rollback, and close behavior | `test_segmentation_rvm.py`, `test_segmentation_timeline.py`, `test_light_wrap.py`, `test_processing.py`, and `test_pipeline.py` cover trial isolation, exact-boundary commit, failed install/timeout rollback, generation-owned state, and exactly-once or idempotent release as appropriate. |
| Privacy fallback for mask-producing paths | `test_mediapipe_segmenter.py` and `test_segmentation_rvm.py` reject invalid backend outputs; `test_pipeline.py` revalidates raw and post-refinement masks, rejects unsafe privacy masks and raw echoes, and proves the virtual-camera and preview sinks receive an input-independent slate on failure. |

## Determinism and privacy contract

The focused gate uses fixed generated arrays, fixed random seeds, synthetic
timestamps, fake MediaPipe/RVM runtimes, and controlled motion estimates. It
must not open a camera, contact the network, require a GPU, download a model,
or read private footage. Optional packages may select additional import paths,
but their tests still use fake inference/provider seams.

Generated images, alpha masks, annotations, and defect fixtures are the only
pixel evidence admitted to this gate. A mask can identify a person in real
use, so no private replay bundle belongs in the repository or CI artifacts.
Production boundaries validate backend and refined masks before composition;
an invalid privacy-sensitive path fails closed to the existing
input-independent slate.

## Defect sensitivity

Tests first prove that their synthetic stimulus contains the old defect, then
assert the corrected or diagnostic result. Representative strict checks
include:

- a raw alternating contour measures exactly two pixels of jitter, while the
  stabilized contour p95 remains at or below `1.2 px` and area-drift p95 at or
  below `0.01`;
- a frozen matte produces compensated temporal error and motion-trail ratios
  above `0.1`, and both lag gates must fail despite zero uncompensated flicker;
- known translation/rotation begins above `0.03` uncompensated temporal error,
  then registers below `0.003` with contour displacement below `0.5 px`;
- an opaque-core fixture with alpha p05 `0.8` and backdrop leakage above
  `0.19` must fail the opaque-core gate even though its contour is stable;
- temporal light-wrap stabilization must retain the exact alpha and reduce
  wrap-induced edge variation to at most `70%` of the stateless path.

The assertions in the test modules are authoritative. A threshold change must
retain the paired defective-stimulus assertion so weakening a gate cannot turn
the old simulation into a false pass.

## Optional algorithm status

The gate includes the optional algorithms that are present in the codebase,
without promoting them to defaults:

| Task | Tested control | Checked-in default |
| --- | --- | --- |
| MATTE-2.1 | `boundary_stabilization.mode: motion_aware` | `off`; the historical compatibility EMA remains selected. |
| MATTE-2.2 | `spatial_edge_refinement.mode: stable_guided` | `legacy_watershed`; `stable_guided` remains an explicit qualification candidate. |
| MATTE-2.4 | `light_wrap_stabilization.mode: temporal_bounded` | `off`; positive light wrap continues to use the stateless compatibility path. |

Passing deterministic tests authorizes none of these controls as a quality
preset or automatic backend choice.

## Boundary to visual and hardware qualification

MATTE-5.1 proves arithmetic, state transitions, strict generated-defect
sensitivity, fallback safety, and resource bounds. It does not prove hair or
accessory quality on representative people, real camera cadence, real
MediaPipe/RVM model output, GPU/provider performance, or parity through every
preview and virtual-camera boundary.

Those claims belong to MATTE-5.2's consented/licensed, owner-controlled visual
qualification and MATTE-5.3's platform/performance qualification. Private
replay, RVM profile qualification, and compositor/service measurement remain
the separate workflows documented in
[Matte replay bundles](matte-replay-bundle.md),
[RVM profile qualification](matte-rvm-profiles.md), and
[Matte performance qualification](matte-performance.md). Heavy visual or
hardware runs stay outside the fast unit gate and their evidence must not be
relabelled as CI-generated proof.
