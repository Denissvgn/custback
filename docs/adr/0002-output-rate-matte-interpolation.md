# ADR 0002: Retain repeats instead of output-rate matte interpolation

- Status: Accepted
- Date: 2026-08-04
- Decision: Reject output-rate matte interpolation for the current runtime
- Scope: local composite output cadence
- Backlog: `VIDEO_MATTE_QUALITY_BACKLOG.md`, MATTE-3.3

## Context

A reported 1280x720 run consumed 541 unique frames, sent 1,105 output frames,
and repeated the last output 564 times. Its rolling capture rate was about
15 FPS while output remained near 30 FPS. Repeating a previously guarded
output is deterministic, adds no alpha estimate, and does not pretend that a
missing camera observation is a new temporal input. The visible result can
still hold for two sends and then jump.

MATTE-3.3 asks whether synthesizing output-rate RGB/alpha frames would improve
that presentation enough to justify its latency, artifact, resource, and
privacy costs. It also requires comparison with advancing only the backdrop
and with matching output FPS to sustained input FPS.

The prerequisite evidence is incomplete:

- MATTE-3.1 provides a bounded capture-only diagnostic, but no checked-in
  artifact qualifies a physical camera or establishes that 15 FPS is
  unavoidable after capture recovery.
- MATTE-2.5 provides a fail-closed profile qualification workflow, but its
  checked-in generated proxy selects no production RVM profile.
- MATTE-3.4 has not yet established recovered 720p unique-frame/compositor
  headroom. The reported run used 26.4 ms for segmentation, 32.7 ms for
  compositing, 61.3 ms for frame processing, and another 2.9 ms for output
  send. Those rolling/EWMA observations establish a budget risk, not a
  portable percentile or an interpolation cost.

The existing replay bundle is also not an interpolation evaluator. Pixel
artifacts are stored once per unique capture/base frame. An output-timeline
event references that stored source and can prove an exact repeat, but it
cannot contain a synthesized midpoint composite, midpoint alpha or backdrop,
output-time occlusion truth, presentation latency, or per-output synthesis
cost. Existing fast-motion, occlusion, edge, and dynamic-backdrop fixtures
define useful metric families; applying their unique-frame values to pixels
that were never recorded would be false evidence.

## Decision

Output-rate matte interpolation is rejected for the current runtime. Exact
repeat remains the only automatic output-rate conversion policy. This task
adds no interpolator, frame-history buffer, configuration field, API or UI
control, automatic FPS negotiation, or synthetic-frame telemetry.

An operator may still set `output.fps` to a measured sustained input rate and
restart where the selected sink supports that rate. That is a reversible
cadence diagnosis, not an automatic policy and not a matte-quality fix. It
reduces output sends and repeats; it does not add unique camera/model
observations or smooth motion. In particular, the Windows native 720p/1080p
virtual-camera modes currently advertise only 30 FPS, so 15 FPS matching is not
a portable runtime policy.

The decision is fail-closed and revisitable. It does not claim that motion
compensation can never help. It records that no candidate currently shows a
clear benefit inside the latency, performance, quality, and privacy budgets
required by MATTE-3.3.

## Alternatives considered

| Strategy | Current outcome | Latency and cadence | Occlusion and edge behavior | CPU/GPU cost | Privacy and temporal state |
| --- | --- | --- | --- | --- | --- |
| Repeat the last guarded output | **Retain as default** | No lookahead or synthesis delay; preserves the configured send cadence while visual updates remain at unique-input cadence | Adds no optical-flow or invented-alpha artifact; visibly holds and then jumps | No second segmentation, refinement, backdrop, or composite pass | Reuses a final guarded frame; never advances capture, model, refiner, backdrop, or base temporal state |
| Low-latency motion-compensated composite interpolation | **Reject: not qualified** | A true midpoint between 15 FPS observations needs the next observation. That endpoint arrives one 66.7 ms input interval after the previous endpoint and at least one 30 FPS output slot (about 33.3 ms) after the midpoint's intended presentation time. Causal extrapolation avoids that wait by predicting unknown motion instead | Warps subject, alpha, and backdrop together; disocclusion has no source pixels and thin detail can split, trail, or halo. No output-tick ground truth has measured the effect | Requires flow/warp and final-image construction on send opportunities; no recovered 720p headroom or cross-device p95 exists | Requires retained image state and new provenance. It cannot advance observation-owned state, and a synthesized result would require a final privacy guard |
| Advance the backdrop while holding the subject/matte | **Reject: not qualified** | Can be causal, but subject motion still updates at 15 FPS and therefore does not meet the stated smoothing goal | Creates subject/backdrop cadence mismatch. Moving edge color or light wrap can shimmer around a held alpha, and later disocclusion still cannot be inferred | Requires extra backdrop work and recomposition on output ticks; cost is unmeasured and competes with the already constrained compositor | Would need a separately owned backdrop-output clock. It must not masquerade as a new capture, matte, or base observation |
| Match output FPS to sustained unique-input FPS | **Keep as manual diagnosis; reject as automatic default** | No synthesized frames or added processing latency; sends honestly at the slower visual cadence | Avoids duplicate sends but does not improve alpha, occlusion, edge motion, or unique-frame smoothness | Reduces send work; dynamic sink renegotiation and stable-rate hysteresis are not implemented or qualified | Existing restart-only `output.fps` preserves privacy and observation semantics; no automatic control is added |

The 66.7 ms and 33.3 ms values are nominal cadence intervals, not measured
host latency. A future causal predictor and a future-frame midpoint
interpolator must be reported as different candidates rather than hiding
prediction or lookahead behind one “low latency” label.

## Existing runtime invariants retained

The current pipeline processes a successful capture once. Segmentation,
refinement, backdrop selection, and safe-base construction occur only for that
unique input. When no unread capture is available, the pipeline reuses the
last guarded output and does not call or advance those stages.

Cadence status keeps these meanings separate:

- capture sequence and capture gaps describe camera observations;
- segmentation and base-update clocks describe unique-frame work;
- base reuse describes a send opportunity with no new safe base;
- exact final-output repeat describes byte equality with the preceding sent
  frame; and
- output send describes successful publication.

A successful pixel-identical capture is still a unique observation and can
advance segmentation/base clocks while also counting as an exact final repeat.
An unread-input repeat cannot. This decision does not overload any of those
clocks with a synthetic-frame meaning.

## Privacy audit

The local and remote paths do not have interchangeable image authority. Remote
mode may publish only an accepted renderer result or an input-independent
privacy slate; it cannot fall back to a camera-derived local composite.

A future local synthesizer must not be inserted after the final remote guard
or share camera/matte history with remote mode. Even two individually guarded
endpoints do not make an interpolated warp or blend safe: the new pixels can
become close to a current, delayed, JPEG-reencoded, or prior-session raw frame.
Every synthesized result would therefore need the final raw-echo guard again
before both preview and output publication.

Until a separate privacy qualification proves otherwise, synthesis must be
inapplicable in remote mode. Any future retained history must be local-session
owned, bounded, and cleared on:

- capture sequence discontinuity or capture-generation change;
- geometry-generation or canonical-canvas change;
- segmenter/refiner generation or backend change;
- background-mode or local/remote-mode change;
- renderer session change, failure, or shutdown; and
- timestamp regression or an elapsed-time gap beyond the candidate's bound.

Future privacy tests must include synthesized equality or near-equality with
current, delayed, JPEG-reencoded, and prior-session raw frames. They must also
prove that malformed/wrong-size renderer output, guard-capacity exhaustion,
and renderer loss still produce only the fixed privacy slate.

## Evidence required to revisit

Reconsideration requires all of the following, on the same reviewed source and
configuration where a comparison is claimed:

1. MATTE-3.1 physical-camera evidence that establishes the sustained input
   premise after capture-path recovery, or an explicit lower-rate camera
   profile.
2. MATTE-3.4 fixed-replay service evidence that establishes residual 720p
   headroom after the 30 FPS unique-frame/compositor budget is recovered.
3. A model-backed MATTE-2.5 profile for every backend/provider/canvas scope in
   which interpolation would be selectable.
4. A new owner-only, bounded, digest-bound output-timeline evidence format.
   It must store each candidate's actual 30 Hz output artifact and provenance;
   replay version 1 must not be relabeled or stretched to imply this.
5. Paired native-30 ground truth and the same source decimated to 15 FPS, with
   held-out odd output frames. Required scenes include soft translating
   boundaries, thin detail, midpoint occlusion/disocclusion, fast motion, and a
   dynamic high-contrast backdrop.
6. Same-host comparisons of all four strategies. They must measure complete
   presentation latency (including lookahead and queue age), output-tick
   occlusion error, contour/trail/edge-alpha and edge-RGB quality, synthesis
   p50/p95, full service p50/p95, CPU utilization, GPU utilization, RSS, and
   VRAM where applicable.
7. A remote privacy review and adversarial raw-echo suite covering the final
   synthesized pixels and retained history.

Generated fixtures can validate the harness and rejection paths. They cannot
select a production policy. Missing cells, proxy-only evidence, a quality
trade with no clear improvement over repeat, a complete-service budget
failure, or any privacy failure keeps the outcome rejected.

## Contract if a future ADR accepts interpolation

Acceptance would require an isolated post-base stage and an explicit,
versioned provenance contract:

- `interpolated_output_count` and `interpolated_output_fps` count synthesized
  successful sends separately from base reuse, exact repeats, and total sends;
- every synthetic output identifies the two bounded base sequences used, its
  interpolation fraction, candidate kind, and state generation without
  exposing wall-clock time or pixel-derived identifiers;
- an interpolated output does not increment capture sequence, segmentation
  update, model invocation, refiner update, or base-composite update counts;
- an interpolated output is never fed back as a capture/model observation and
  never advances RVM, MediaPipe, boundary-stabilizer, spatial-refiner,
  harmonizer, or light-wrap temporal state;
- state reset and privacy-guard events are observable through bounded,
  content-free reason codes; and
- overload, evidence failure, or privacy uncertainty rolls back immediately to
  exact repeat without changing the current safe base.

Backdrop-only advancement, if ever accepted independently, would require a
separate backdrop-presentation counter. It must not use the interpolation
counter or advance capture/matte/base clocks.

## Consequences

The runtime remains simple, deterministic, reversible, and privacy preserving.
It also retains visible 15-to-30 FPS hold/jump motion when unique input is only
15 FPS. The project addresses that first through capture recovery and
unique-frame/compositor budget work rather than masking it with predicted
pixels.

No operator migration or rollback action is needed. Exact repeat is both the
unchanged default and the fallback required by any future candidate.

## References

- `VIDEO_MATTE_QUALITY_BACKLOG.md`, observations and MATTE-3.3
- `docs/capture-cadence-diagnostics.md`
- `docs/cadence-observability.md`
- `docs/matte-ablation.md`
- `docs/matte-quality-metrics.md`
- `docs/matte-rvm-profiles.md`
- `docs/matte-operator-mitigations.md`
