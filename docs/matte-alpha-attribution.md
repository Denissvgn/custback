# RVM alpha integrity and opaque-core attribution

Status: **MATTE-0.5 implementation and generated qualification protocol**

The canonical `custback.matte-alpha-attribution` version-1 report determines
where a spatial opacity, hole, halo, or backdrop-leakage symptom first appears.
It does not change production alpha, rerun RVM, or draw a temporal conclusion
from a still image.

## Evidence protocol

Record a consented 10–20 second Run-B-style full bundle with the same source,
backdrop, RVM profile, and camera mode that showed the defect. Include
stationary torso/shoulder/head/headphone/hair coverage, head motion, quick
lateral motion, and hair/ear detail. Verify that every frame carries:

- raw source, raw RVM `pha`, post-`mask_shift` alpha, RVM `fgr`, exact backdrop,
  base composite, and final composite;
- configured `segmentation.rvm_downsample` and `segmentation.mask_shift`, plus
  effective `rvm_downsample_ratio` and `mask_shift`; and
- digest-bound named `torso`, `shoulders`, `head`, `headphones`, `hair`, and
  known-background masks. Opaque anatomy uses `opaque_core`, hair uses
  `soft_boundary`, and exterior support uses `background`.

The replay and annotations remain private. Run:

```console
custback matte-diagnose ./private-run-b \
  --annotations ./private-run-b-annotations \
  --output ./private-run-b-attribution
```

The output directory contains `attribution.json`, `attribution.md`, and
lossless PNGs under `frames/`. Each frame has these required boundaries:

| Boundary | Held-fixed comparison |
| --- | --- |
| `raw_alpha` | source with raw `pha`, recorded backdrop/transform/blend space |
| `post_refiner_alpha` | source with post-refiner alpha |
| `direct_full_frame_foreground` | full RVM `fgr` with raw `pha` |
| `current_compositor` | source plus recorded edge-only `fgr` and light wrap |

`edge_foreground_only` and `light_wrap_only` are additional factorial controls.
They distinguish foreground contamination from wrap without changing source,
alpha, backdrop, or frame identity.

## Classification contract

Each named region reports raw/refined alpha quantiles, mean deficit,
below-opaque fractions, hole/component counts, halo mass/width where
ground-truth exterior exists, final backdrop-leakage coefficient, clean
foreground error, and counterfactual RGB deltas. Fixed-scale heatmaps cover
raw/refined deficit, refiner delta, clean-foreground error, foreground and wrap
contributions, and final backdrop leakage.

The classifier records every evidenced stage and uses `mixed` when more than
one contributes. Thresholds are serialized in the report and are diagnostic
decision aids, not hidden production constants or release gates. A deficient
opaque region in every frame of a sequence of at least two unique inputs makes
qualification fail even if its contour is perfectly stable.

Clean-foreground/source difference is retained as a visibility proxy, but it
does not classify a soft/hair foreground defect without annotated
ground-truth foreground: legitimate decontamination is expected to differ from
the captured source at those boundaries. Opaque-core source comparison remains
usable because the original backdrop should not contribute there.

When raw alpha fails, the report proposes an opaque-core-restricted calibration
and RVM ratio/profile sweep on the exact replay. A simulated global
`raw_pha >= 0.5` candidate is rejected unless the same evidence passes all five
families: opaque core, hair/soft boundary, exterior halo, ground truth, and
motion non-regression. Missing evidence is not treated as a pass. No candidate
changes the production default.

## Generated acceptance evidence

`tests/test_matte_attribution.py` creates only MIT-licensed synthetic pixels and
two-frame sequences. It injects defects independently at raw alpha,
post-refiner alpha, RVM clean foreground, light wrap, and final blend, plus a
mixed case. The tests require the report to locate the producing stage on the
same frames, reject stable opaque-core failure, reject a hard threshold that
destroys genuine soft alpha, verify all four views and every heatmap digest,
and enforce owner-only output.

```console
PYTHONPATH=src .venv/bin/pytest -q tests/test_matte_attribution.py
```

No private or licensed Run-B footage is checked into this repository. The
generated suite qualifies the implementation, while the first consented local
Run-B report remains an operator evidence step; its manifest/report digests,
not its pixels or paths, may be referenced in
[`matte-alpha-attribution-local-template.json`](matte-alpha-attribution-local-template.json).
