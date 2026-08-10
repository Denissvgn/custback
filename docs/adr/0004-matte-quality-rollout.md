# ADR 0004: Matte quality compatibility, qualification, and rollout contract

- Status: Accepted
- Date: 2026-08-10
- Scope: MATTE-5.4
- Supersedes: no existing configuration semantics

## Context

Custback now has explicit capture timestamps and generations, backend-aware
matte policies, experimental spatial and temporal refiners, content-free local
diagnostics, deterministic regressions, visual qualification, and platform
qualification. Availability is not authority to make any of those candidates
a default.

The checked-in MATTE-2.5, MATTE-5.2, and MATTE-5.3 fixtures exercise schemas
and failure handling. They do not contain consented model-backed, physical
camera, consumer-sink, sustained-platform, or rollback approval. A generated
fixture therefore cannot authorize a named quality preset, an RVM install
profile, or a default change.

Persisted segmentation values are hot-patchable and historically meaningful.
In particular, `segmentation.temporal_smoothing` is a frame-count compatibility
EMA on MediaPipe and heuristic paths, while ordinary RVM bypasses it in favor
of the model's recurrent state. Reinterpreting that scalar as an elapsed-time
constant would silently change old output. Existing users can likewise depend
on the legacy watershed, mask softness, mask shift, stateless light wrap, and
RVM generic-postprocess bypass.

This ADR makes rollout and rollback semantics explicit without claiming that
unavailable physical qualification has passed.

## Decision

### Compatibility and schema

Schema version 1 and versionless persisted documents retain their existing
meaning. Loading a document does not rewrite it, and an omitted field is
materialized under schema-1 compatibility rules.

The chosen compatibility strategy is to preserve existing fields and add new
algorithms under explicit modes:

- `segmentation.temporal_smoothing` keeps its historical meaning and range;
- `segmentation.boundary_stabilization.mode: off` preserves that path, while
  `motion_aware` explicitly selects the elapsed-time candidate;
- `segmentation.spatial_edge_refinement.mode: legacy_watershed` preserves the
  historical spatial path, while `stable_guided` explicitly selects the new
  candidate; and
- `compositing.light_wrap_stabilization.mode: off` preserves stateless light
  wrap, while `temporal_bounded` explicitly selects the dynamic-backdrop
  candidate.

No migration converts a legacy smoothing scalar to a time constant. No preset
name is persisted as an alias for concrete values. A future schema may choose
new-install defaults, but it must continue to interpret versionless and
schema-1 omissions with this compatibility contract.

### Timestamp and reset ownership

Every authoritative unique frame carries capture sequence, monotonic capture
timestamp, capture generation, and geometry generation. The segmenter/refiner
timeline consumes that context once. Output repeats reuse the already
committed base and do not advance recurrent, spatial, light-wrap, or other
matte state.

The matte timeline resets at startup; backend or matte-policy activation;
capture, geometry, source, or model generation change; capture restart; and
timestamp discontinuity. A successful transactional activation increments the
segmentation generation and resets candidate-owned state before its first
authoritative input. A failed activation keeps the old configuration,
generation, backend, pixels, and temporal state. State from one generation is
never blended into another.

### Backend-specific effective policy

Configured intent and effective behavior remain separate:

| Active path | Effective compatibility policy |
| --- | --- |
| RVM | Native recurrent soft alpha; generic threshold, blur, spatial refinement, and legacy EMA bypassed; optional mask shift, explicitly selected motion-aware stabilization, model foreground, and light wrap remain independently applicable. |
| MediaPipe | Soft confidence mask; threshold inapplicable; configured generic spatial refinement, mask shift, blur, and exactly one temporal owner apply. |
| Heuristic | Historical `threshold × 0.8` cutoff; configured generic spatial refinement, mask shift, blur, and exactly one temporal owner apply. |
| Null or passthrough | No matte; matte/refiner controls are inapplicable and resolve to neutral values. |

`GET /config` is the configured-intent authority. The versioned
`GET /status.matte_policy` and `segmentation_selection` objects are the
effective runtime authority. UI and operator decisions must use the selected
backend, provider, fallback, control states, and matching `config_version`, not
infer behavior from `segmentation.backend: auto`.

### Light-wrap temporal policy

Schema-1 light wrap remains stateless. The temporal-bounded candidate owns
only dynamic video/camera backdrop samples, advances on their presentation
timeline, and resets at the documented discontinuities. It is bypassed for a
static backdrop or a path without an eligible timeline. It is independently
selectable and independently reversible; enabling motion-aware matte
stabilization never enables temporal light wrap.

### Quality tiers and default disposition

The backend quality tiers describe capability, not universal quality:

- RVM is the true-alpha **matting** tier;
- MediaPipe is the confidence-mask **segmentation** tier;
- the heuristic is an explicit coarse **heuristic** fallback; and
- null/passthrough makes no matte-quality claim.

The current new-install policy remains the schema-1 compatibility policy.
The npm installer may attempt MediaPipe and `backend: auto` may select the best
installed backend, but that behavior is not a portable named quality preset.
RVM remains an explicit optional install profile. Performance, Balanced, and
Quality presets remain unavailable while the checked-in evidence disposition
is pending.

A later release may promote only an exact candidate whose MATTE-5.2 visual and
MATTE-5.3 platform reports are qualified for every platform/canvas/provider it
advertises. Unqualified routes must keep a qualified lower tier or report an
explicit unavailable/fallback state; they must not silently receive RVM or a
higher-detail profile.

### Performance, privacy, and evidence gates

Default promotion requires the exact release candidate to link and pass:

1. MATTE-0.2 baseline and MATTE-0.3 same-source ablation;
2. MATTE-2.5 RVM profile evidence when RVM is proposed;
3. MATTE-3.1 physical capture and MATTE-3.4 complete-path/compositor evidence;
4. MATTE-5.1 deterministic regressions;
5. MATTE-5.2 representative visual, route-parity, and human review;
6. MATTE-5.3 physical route, dependency, performance, resource, fallback,
   restart/hot-patch, and shutdown qualification;
7. a privacy review proving that public rollout artifacts contain no frames,
   masks, paths, device labels, package inventories, or raw failures; and
8. migration and rollback results for old, partial, current, and future-stage
   configurations plus the supported package/install profiles.

Digests and owner attestations join the exact private evidence. They do not
turn generated data into physical evidence or prove that independently
produced v1 reports used the same executable. Release review retains the
private build and consent/license provenance for that limitation.

### Rollout, telemetry, and rollback

Promotion is staged: internal observation, explicit opt-in, bounded canary,
then a new-install default only after the preceding stage meets its stop/go
criteria. Persisted old configurations are never silently rewritten by an
install or ordinary startup.

Canary telemetry is content-free. It may contain bounded enums, counters,
rates, durations, generations, control states/reasons, and opaque digests. It
must not contain pixels, masks, file/model paths, device display strings,
package names/versions, raw exceptions, or per-frame timestamps.

Every promoted policy has a documented single merge-patch rollback to the
schema-1 compatibility values. Rollback is transactional, keeps the recognized
schema version, does not delete or redownload configuration or model caches,
and verifies the next matching `config_version`, generation/reset boundary,
selected backend/provider, and fresh output. Package-profile rollback uses the
installer's retained generation rather than deleting a managed environment.
The exact operator procedure is in
[`docs/matte-quality-rollout.md`](../matte-quality-rollout.md).

The legacy compatibility path remains selectable for at least one stable
release after any default promotion. Removing it requires a later ADR and a
separate migration, deprecation, and rollback decision.

### Reaction separation

All matte rollout evidence and the default decision are pre-reaction and have
reactions disabled. Reaction configuration, defaults, migration, headroom,
telemetry, and rollback belong to the `REACT` lane. A reaction cannot improve
or replace a matte gate, and residual matte headroom does not authorize one.

## Consequences

- Existing persisted configurations keep deterministic behavior.
- New algorithms are explicit opt-ins until qualified; no scalar is silently
  reinterpreted.
- The product can expose truthful backend-aware controls and diagnostics while
  withholding named presets and default claims.
- A future promotion must be a reviewable, evidence-bound release change, and
  rollback is a normal configuration transaction rather than destructive
  cache or config removal.
- The current release remains pending physical qualification. This ADR records
  the decision boundary; it is not evidence that the boundary was crossed.

## Related records

- [Backend-specific matte policies](../matte-backend-policies.md)
- [Motion-aware boundary stabilization](../matte-boundary-stabilization.md)
- [Stable spatial refinement](../matte-spatial-refinement.md)
- [Temporal light wrap](../matte-light-wrap.md)
- [End-to-end visual qualification](../matte-visual-qualification.md)
- [Performance and platform qualification](../matte-platform-qualification.md)
- [Deterministic regression gate](../matte-deterministic-regression-gate.md)
