# Matte quality rollout, migration, and rollback

Status: **compatibility default hold; Experimental profiles are not qualified**

This is the MATTE-5.4 operator and release-owner guide. It defines how a
separately qualified matte policy would become a reversible rollout; it does
not create that qualification. At this revision the checked-in visual/platform
evidence is generated and pending, so schema-1 compatibility remains the only
default policy. Performance, Balanced, Quality, and Motion stable are exposed
only as explicitly acknowledged non-qualified concrete profiles with
`quality_claim: false`. A local one-host screen may advance an exact row from
`experimental` to `locally_screened`, but portable qualification remains
blocked pending the complete second-hardware matrix.

Reactions are outside this procedure. Keep them disabled while collecting,
canarying, or deciding matte behavior. A reaction rollout uses the separate
`REACT` evidence, migration, telemetry, and rollback lane.

The normative decision is
[`ADR 0004`](adr/0004-matte-quality-rollout.md). Backend applicability remains
defined by [the matte-policy contract](matte-backend-policies.md).
The machine-readable release authority is
[`scripts/release/matte-policy-rollout.json`](../scripts/release/matte-policy-rollout.json).
Release validation rejects drift between its canonical patch/digest, the
schema-1 defaults, the code-owned rollback helper, status disposition, and the
packaged system-profile catalog plus every concrete profile-patch digest.

The patch digest is SHA-256 over UTF-8 canonical JSON, not over the pretty
printed ledger bytes. Canonicalization preserves array order, recursively
sorts object keys, emits no insignificant whitespace, and uses JSON/ECMAScript
primitive and number serialization. Consequently integral floats normalize
(`0.0` becomes `0`). The current `matte-legacy-v1` canonical digest is
`27638e419a0dcf5955d52e2eb4ead2dafbdca7f2bbe0535108aa7c56c1f2f60d`.

This binding is an executable release check, not merely a CI convention. The
release verifier recomputes the canonical ledger digest and distributed YAML
values, then runs a dependency-free Python AST evaluator against the reviewed
package source to compare the returned `legacy_matte_policy_patch()` and exact
empty path-free rollout-status contracts. Local release checks, prepack, and
the release workflow therefore share the same fail-closed semantics.

## Current release decision

| Concern | Current disposition |
| --- | --- |
| Config schema | Version 1; versionless persisted mappings keep schema-1 semantics |
| New-install matte policy | Compatibility values from the two byte-identical default YAML files |
| Named profiles | Selectable only as explicitly acknowledged `experimental` or `locally_screened` entries; no portable quality claim |
| Default backend | `auto`; this is selection intent, not a promise of RVM or a quality tier |
| RVM package profile | Optional explicit `rvm` or `gpu` extra; never added by a default rollout on an unqualified route |
| Spatial candidate | `stable_guided` available only by explicit selection; default remains `legacy_watershed` |
| Temporal candidate | `motion_aware` available only by explicit selection; default remains `off` |
| Temporal light wrap | `temporal_bounded` available only by explicit selection; default remains `off` |
| Promotion authority | Held pending consented/licensed model-backed visual and physical MATTE-5.3 evidence |
| Reaction state | Disabled and separately governed |

An installed optional backend can change what `backend: auto` selects. Always
inspect the effective `segmentation_selection`, provider, and `matte_policy`;
the configured word `auto` alone is not a runtime result.

The checked-in rollout ledger is intentionally not a partially completed
approval. It binds the immutable Experimental catalog and patch hashes while
its `promotion.status` remains `pending`, the candidate/commit fields are null,
and its seven baseline, ablation, visual, performance, platform, privacy, and
migration evidence slots contain no report or digest. Generated fixtures may
test those joins but cannot fill a physical promotion slot.

## Persisted-config behavior

Startup does not migrate or rewrite a user file. A versionless mapping and an
explicit schema-1 mapping have the same matte defaults. The explicit
`custback migrate --config PATH` operation remains atomic and keeps its private
backup; it materializes compatibility values rather than promoting a
candidate.

The schema decision is additive:

- the historical `temporal_smoothing` scalar is not reinterpreted;
- motion-aware stabilization has its own explicit mode and time parameters;
- stable guided spatial refinement has its own explicit mode;
- temporal light wrap has its own explicit mode; and
- a preset expands to concrete fields and is never persisted as a drifting
  alias.

An old or partial config therefore gets documented schema-1 behavior. A
future new-install default must use a newer, explicit rollout decision without
changing how versionless/schema-1 omissions load.

Run the migration checks before a release:

```console
PYTHONPATH=src pytest -q tests/test_config.py tests/test_phase6_migration.py
npm test
```

These tests do not replace a package-upgrade drill on every supported install
profile. The release record must also retain results for a versionless file,
an explicit schema-1 file, a partial matte mapping, the current defaults, an
unsafe file that requires operator action, an interrupted migration, and a
second idempotent migration.

## Release-evidence chain

A release owner records immutable digests for every applicable row below and
binds them to the exact candidate being promoted. Narrative links are useful
for review but do not replace the machine-readable reports or the corresponding
slot in `scripts/release/matte-policy-rollout.json`.

| Gate | Contract and required result |
| --- | --- |
| Baseline | [MATTE-0.2 baseline](matte-quality-baseline.md), with the selected corpus/source contract and no model-backed claim from generated proxy data |
| Attribution and ablation | [RVM attribution](matte-alpha-attribution.md) plus [same-source ablation](matte-ablation.md), selecting the exact policy rather than a favorable isolated metric |
| Backend profile | [MATTE-2.5 RVM qualification](matte-rvm-profiles.md) when RVM is proposed; model, ratio, provider, canvas, and cadence scope must match |
| Capture | [MATTE-3.1 capture diagnosis](capture-cadence-diagnostics.md), using eligible physical evidence for each advertised route |
| Performance | [MATTE-3.4 performance](matte-performance.md), including complete non-pacing service and compositor sub-budget rather than repeated output FPS |
| Deterministic regression | [MATTE-5.1 gate](matte-deterministic-regression-gate.md), run on the exact candidate |
| Visual | [MATTE-5.2 qualification](matte-visual-qualification.md), qualified with representative cases, route parity, and human review |
| Platform | [MATTE-5.3 qualification](matte-platform-qualification.md), qualified for every advertised route/provider/canvas and required missing-dependency lane |
| Privacy | Replay consent/license authority, generated-versus-physical origin, content/path flags, owner-only storage, and a public report audit under [the replay privacy contract](matte-replay-bundle.md) |
| Migration/package | Old/partial/current config matrix, pip/npm install profiles, persisted extras intent, failed rebuild rollback, and the one-patch policy rollback below |

The promotion decision is fail-closed if any report is pending, failed,
rejected, not decidable, mismatched, generated-only, or outside the proposed
platform scope. One qualified CUDA host does not authorize CPU, DirectML,
macOS, another canvas, or another virtual-camera route. An unavailable high
tier remains visible; it is not silently relabelled as a lower qualified tier.

Private evidence can include identifiable frames, masks, recordings, device
inventory, and consent/license records. Keep it in a new owner-only directory
outside the repository. Public rollout artifacts may retain only bounded
enums, counters, rates, durations, policy values/reasons, and opaque digests.
They must contain no pixels, masks, paths, device labels, package names or
versions, raw provider errors, credentials, or per-frame timestamps.

## Before a canary

Freeze these items before collecting the go/no-go baseline:

1. The exact source/build identity and distributed default files.
2. The reviewed rollout ledger, concrete candidate merge patch, and its
   configured/effective policy and file digests.
3. The advertised platform, route, canvas, FPS, backend, provider, model, and
   dependency matrix.
4. The predeclared cohort, observation window, expansion steps, and stop
   owner. Do not tune those thresholds after seeing results.
5. A copy of the current configured intent, its `X-Config-Version`, the
   content-free status snapshot, and the exact rollback patch ID.
6. Reactions disabled and every unlisted post-base effect absent.

Confirm the runtime before applying a candidate:

```console
custback doctor
custback extras --json

export CUSTBACK_API_TOKEN="$(custback --show-api-token)"
auth_header() { printf 'header = "Authorization: Bearer %s"\n' "$CUSTBACK_API_TOKEN"; }
auth_header | curl --config - http://127.0.0.1:8710/config
auth_header | curl --config - http://127.0.0.1:8710/status
```

Do not copy those responses to shared telemetry without applying the public
field allowlist below. `/config` may contain operator-owned local resource
references even though credential paths are redacted from the API model.

## Canary order and stop/go rules

Only exact qualified cells may enter a production canary. Use this order:

1. **Compatibility observation.** Record a status baseline without changing
   policy. The runtime must report the compatibility rollout stage and the
   legacy patch must match configured intent.
2. **Owner-controlled opt-in.** Apply the concrete candidate to one retained,
   qualified host per exact route. Complete the MATTE-5.3 30-minute soak and
   restart/hot-patch/shutdown sequence before broader use.
3. **Bounded platform canary.** Expand within each exact qualified cell only.
   Use the predeclared cohort and window; never infer another platform from a
   passing one. Compare counter deltas, not process-lifetime totals from hosts
   with different uptime.
4. **New-install default.** Promote only after every advertised cell and
   migration/package profile passes, release evidence is attached to the exact
   candidate, and a rollback drill succeeds. Existing schema-1 files remain
   unchanged.
5. **Compatibility retention.** Keep the legacy patch and code path for at
   least one stable release after promotion.

Stop expansion and apply rollback when any of these occurs:

- the rollout decision, configured patch, effective policy, build, model,
  route, provider, canvas, or config version does not match the approved cell;
- an unexpected backend/provider/dependency fallback or tier suppression is
  reported;
- cadence mismatch activates, unique capture/segmentation/composite or sink
  rate falls below the qualified limit, deadline/gap/drop/overwrite ratios
  exceed the qualified limit, or a sink recovery occurs;
- restart/hot patch does not advance/reset the expected generation, a stale
  frame crosses a generation, or shutdown exceeds the qualified bound;
- RSS/VRAM drift or span exceeds the qualified soak bound;
- visual review finds a regression in opaque core, holes, halo/leakage, hair,
  thin components, motion trails, light-wrap shimmer, or consumer output;
- public telemetry or an artifact exposes a private field; or
- an operator cannot complete the one-patch rollback exactly as rehearsed.

A stopped canary is evidence, not permission to silently reduce the advertised
FPS or substitute another backend. Record the failed cell and return it to the
qualification workflow.

## Sanitized rollout telemetry

Use `GET /status` locally and aggregate only an allowlisted projection. The
runtime `matte_rollout` object is the rollout authority; it reports a bounded
schema/version, stage/decision, whether a qualified default is active, the
preset catalog's evidence status, the legacy rollback-patch ID, and aggregate
apply/success/failure/rollback counters. The WebUI displays that decision
separately from the configured backend.

The following existing status fields are also safe and useful:

| Purpose | Fields |
| --- | --- |
| Binding | `config_version`, `segmentation_generation`, `capture_generation`, `segmentation_selection` bounded enums/reasons, and `matte_policy` bounded configured/effective/control states |
| Reset health | `matte_reset_count`, `matte_last_reset_reason`, `capture_sequence_gap_count`, `capture_missing_input_count` |
| Unique work | `segmentation_update_count/fps`, `base_composite_update_count/fps`, `output_send_count/fps`, and exact repeat/reuse counts/ratios |
| Timing health | p50/p95 cadence fields, `processing_deadline_misses`, `serialized_new_frame_deadline_misses`, `cadence_mismatch_active`, and bounded stage durations |
| Fallback/lifecycle | bounded segmentation/acceleration/output fallback state and counters, `output_sink_recovery_events`, capture restarts/read failures/stall, and the rollout aggregate counters |

Store interval deltas and the cohort's opaque ID, never frames or a stable
user/device identifier. Do not export the complete `/config`, model path,
native ring path, raw fallback exception, private evidence path, package
inventory, device display name, API token, or frame-level timestamps. Cap
reason strings to the code-owned bounded values already used by public status.

The rollout counters are observational. They do not prove pixel quality,
physical origin, successful consumer recording, or absence of an unreported
process crash; the qualification artifacts and canary owner remain separate
authorities.

## One-patch rollback

`custback.config.legacy_matte_policy_patch()` is the code-owned schema-1
rollback authority. It returns a new detached merge patch on every call and
intentionally omits `schema_version`, model/cache paths, output, background,
API, and avatar settings. In JSON form the patch is:

```json
{
  "segmentation": {
    "backend": "auto",
    "delegate": "cpu",
    "rvm_downsample": 0.0,
    "threshold": 0.5,
    "mask_blur": 7,
    "edge_refine": true,
    "spatial_edge_refinement": {
      "mode": "legacy_watershed",
      "reference_short_edge_px": 720,
      "radius_at_reference_px": 8,
      "min_radius_px": 2,
      "max_radius_px": 12
    },
    "mask_shift": 0,
    "temporal_smoothing": 0.35,
    "boundary_stabilization": {
      "mode": "off",
      "time_constant_s": 0.1,
      "max_motion_px_per_s": 720.0
    }
  },
  "acceleration": {
    "mode": "auto",
    "provider": "auto",
    "device_id": 0
  },
  "compositing": {
    "light_wrap": 0.25,
    "use_model_foreground": true,
    "blend_space": "srgb_legacy",
    "light_wrap_stabilization": {
      "mode": "off",
      "time_constant_s": 0.12
    },
    "color_correction": {
      "mode": "off",
      "strength": 0.5,
      "exposure_limit_ev": 0.85,
      "white_balance_strength": 0.5,
      "adaptation_time_s": 0.8
    }
  }
}
```

Apply this object once through the authenticated transactional
`PATCH /config` API. Use `application/merge-patch+json`, not a sequence of
individual changes. A successful activation commits one new config version;
segmentation policy/backend changes rebuild and reset the matte timeline at
one generation boundary. A `409`, `422`, `503`, or activation timeout applies
nothing and keeps the previous live generation.

After success, fetch both endpoints again and verify:

- `/config` contains the complete patch and returns the new
  `X-Config-Version`;
- `/status.config_version` catches up to that same version;
- `matte_rollout` identifies the legacy rollback patch and records one
  successful rollback;
- `segmentation_selection` truthfully shows the backend/provider actually
  selected by `auto`, including any fallback;
- `matte_policy` reports schema-1 effective controls for that selected path;
- the segmentation generation and reset count advance together; and
- the first authoritative output is fresh, with no cross-generation flash.

The rollback does not lower the recognized schema, delete the user config,
remove an installed optional backend, or delete/redownload the model cache.
Because `auto` considers installed backends, the selected runtime tier after
rollback may differ from a host that never installed RVM; that fact must remain
visible in status. If package intent also needs to return to an earlier set of
extras, first complete the policy rollback, then run a separately reviewed
`custback rebuild --extras LIST`. Rebuild stages a candidate environment and
must leave the current runtime usable on failure. It still does not require
deleting the config or model cache.

Persist the rollback values to the operator-owned YAML only after the live
rollback succeeds and the result has been reviewed. Do not delete the YAML or
cache as a substitute for an explicit policy decision.

## Release checklist

- [ ] `config/default.yaml` and `src/custback/default.yaml` are byte-identical.
- [ ] The machine rollout authority reports compatibility hold, unless an
      exact physical evidence set and promotion change were separately
      reviewed.
- [ ] Old, versionless, partial, and schema-1 configs retain documented
      values; migration is atomic and idempotent.
- [ ] Every advertised profile/route/provider/canvas has qualified visual and
      platform evidence from the exact candidate.
- [ ] Reactions and unlisted post-base effects are disabled.
- [ ] Public telemetry/report artifacts pass the privacy allowlist.
- [ ] pip/npm artifacts contain this ADR, this runbook, the qualification
      runbooks/templates, and the rollout authority.
- [ ] The concrete candidate applies transactionally and status matches its
      config version and effective backend policy.
- [ ] The exact one-patch rollback is rehearsed without config/cache deletion.
- [ ] The legacy path remains supported for at least one stable release after
      any promotion.
