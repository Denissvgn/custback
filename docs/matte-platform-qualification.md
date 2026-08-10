# MATTE-5.3 performance and platform qualification

Status: **owner-operated evidence join; generated evidence is pending**

MATTE-5.3 decides whether one already reviewed matte candidate meets its
declared performance, platform, fallback, sink, and lifecycle limits. The
qualifier is offline: it opens no camera, model, preview, API, network service,
consumer, or virtual-camera sink. Operators first collect the physical and
fixed-replay evidence on each declared host, then join those immutable files.

This workflow does not select a preset, change a default, or qualify reactions.
The plan requires `defaults_changed: false`; the plan and every run require
reactions disabled and `post_base_event_count: 0`. Residual headroom is not
permission to enable an effect. Active effects remain a separate `REACT-5.2`
qualification.

## Run the offline verifier

The checked-in
[`matte-platform-qualification-local-template.json`](matte-platform-qualification-local-template.json)
is a content-free shape skeleton, not evidence or a complete qualification
matrix. It is intentionally marked `generated`; all example cells are
`unavailable`, all digests are placeholders, and the physical attestations are
false. Copy it outside the checkout before editing it, then add every lane
reported in `coverage.missing_required_lanes` rather than merely replacing the
example rows:

```console
PRIVATE_EVIDENCE_ROOT=/absolute/path/outside/repository/matte-platform
install -d -m 700 "$PRIVATE_EVIDENCE_ROOT"
install -m 600 docs/matte-platform-qualification-local-template.json \
  "$PRIVATE_EVIDENCE_ROOT/plan.json"

# Put prerequisite reports and per-cell evidence below this private root,
# replace every placeholder digest and value, then run the offline join.
custback matte-platform-qualify \
  "$PRIVATE_EVIDENCE_ROOT/plan.json" \
  --output "$PRIVATE_EVIDENCE_ROOT/qualification-output"
```

The output directory must be new and owner-only. It contains the authoritative
`qualification.json` and compact `qualification.md` review companion. Exit
status `0` is reserved for a fully qualified result, `1` means a valid result
is still pending or has failed, and `2` means the plan/evidence/privacy
contract or execution is invalid. A generated fixture, missing row,
unavailable provider, short run, or unreplaced template value is never a
platform approval.

All paths in the plan are safe relative paths below the plan directory. The
plan, prerequisite reports, capture reports, and run sidecars must be
owner-only regular files rather than symlinks. Prerequisite and run descriptors
carry the exact file SHA-256 plus the document's canonical `evidence_sha256`.
The official capture-report v1 schema has no internal evidence digest, so its
cell descriptor is exactly `path` plus `sha256`. Do not invent a capture
`evidence_sha256` or copy a favorable summary into a new wrapper. Run files are
unique per cell. One capture-only report may be reused only by cells with the
same route, canvas, target FPS, and therefore the same capture scope; the
qualifier rejects cross-scope reuse.

## Evidence authority and prerequisites

Physical origin is an evidence-owner attestation, not a cryptographic proof.
For a real local run, use `consented-local` or `licensed-local` provenance,
retain the consent/license record privately, and attest the physical camera,
consumer recording, and same-host observation. The plan and run sidecars must
still keep `physical_origin_cryptographically_proven` / `cryptographically_proven`
false. Matching digests and pixels cannot prove that a camera or meeting-app
capture came from a physical device.

Every plan binds three independent, content-free prerequisite reports:

| Plan entry | Required report | What remains independent |
| --- | --- | --- |
| `prerequisites.visual` | MATTE-5.2 `custback.matte-visual-qualification-report` | candidate algorithm identity and representative visual outcome |
| `prerequisites.performance` | MATTE-3.4 `custback.matte-performance-report` | fixed 720p compositor and complete RVM/CUDA service budget |
| `prerequisites.rvm` | MATTE-2.5 `custback.matte-rvm-qualification-report` | RVM provider/canvas/cadence profile meaning |

The join checks the visual candidate, exact segmentation/compositing contract,
configured and effective policy digests, platform/canvas-specific visual sink
evidence, prerequisite canonical digests, reaction-free scope, and
default-neutral disposition. RVM cells additionally match the qualified model,
hardware, provider environment/device ordinal, canvas, cadence, and compositor
scope from MATTE-2.5. The fixed replay binds the exact MATTE-3.4 source, bundle,
warm-up, and measured frame lineage. A pending, not-decidable, not-proposed,
rejected, or candidate-mismatched prerequisite cannot be hidden by a fast
platform cell. Capture-only and fixed-replay evidence are separate; neither can
mask the other.

The plan's candidate build digest is an owner-attested join key. The three
upstream v1 report schemas do not expose a shared binary/source-build digest,
so MATTE-5.3 does not represent that key as cryptographic proof that every
report used the same executable. Exact algorithm, model, policy, source, and
evidence joins remain machine-checked; retain the private build provenance for
release review.

## Exact reviewed route catalog

Version 1 accepts these five route contracts, in this exact order and with no
backend relabelling:

| Route ID | Platform | Capture backend | Sink | Physical consumer |
| --- | --- | --- | --- | --- |
| `linux-v4l2-pyvirtualcam` | Linux | `V4L2` | `pyvirtualcam` | `v4l2loopback` |
| `macos-obsvcam-pyvirtualcam` | macOS (`darwin`) | `AVFOUNDATION` | `pyvirtualcam` | `obs-virtual-camera` |
| `windows-msmf-pyvirtualcam` | Windows | `MSMF` | `pyvirtualcam` | `obs-virtual-camera` |
| `windows-dshow-pyvirtualcam` | Windows | `DSHOW` | `pyvirtualcam` | `obs-virtual-camera` |
| `windows-native` | Windows | `MSMF` | `native` | `media-foundation-virtual-camera` |

Do not substitute another capture API, fake sink, lower canvas, or projected
trace for an unavailable route. Windows native cells use only the advertised
1280x720@30 or 1920x1080@30 modes. Every route retains at least one `standard`
dependency cell; unsupported hardware stays visible as `unavailable` with a
stable reason code.

## Backend tiers, providers, and limits

The profile catalog is code-owned:

| Profile ID | Effective segmenter | Meaning |
| --- | --- | --- |
| `rvm_matting` | `RVMSegmenter` | true-matte tier; binds one MATTE-5.2 candidate contract |
| `mediapipe_segmentation` | `MediaPipeSegmenter` | segmentation tier; binds one MATTE-5.2 candidate contract |
| `heuristic_segmentation` | `HeuristicSegmenter` | explicit non-quality fallback tier |
| `none_passthrough` | `NullSegmenter` | explicit non-quality passthrough tier |

Each profile lists the exact cell IDs that define its supported limits. Every
cell declares one reviewed route, provider (`cpu`, `cuda`, or `directml`),
canvas (640x360, 1280x720, or 1920x1080), and package profile (`standard`,
`without_mediapipe`, or `without_gpu_provider`). Version 1 qualifies a balanced
30 FPS target, and every backend tier must expose at least one 1280x720@30
limit. CUDA cannot be claimed on macOS, DirectML is Windows-only, and non-RVM
tiers execute on CPU.

In addition to a profile/route row and a standard CPU row for every reviewed
route, the report keeps these code-owned lanes visible: RVM/CUDA on Linux and
Windows, RVM/DirectML on Windows, and both missing-dependency behaviors on each
Linux, macOS, and Windows platform family. A row may truthfully be
`unavailable`, but omission remains pending and is listed under
`coverage.missing_required_lanes`.

An unavailable MediaPipe package must visibly select the declared fallback
from a `mediapipe_segmentation` request with reason
`mediapipe-unavailable`. A missing GPU provider must visibly execute its
declared CPU fallback from an `rvm_matting` request with reason
`gpu-provider-unavailable`. A `standard` cell cannot silently fall back. The
run records the requested and selected profiles, a bounded private package and
provider inventory plus its public digest, actual provider/device ordinal,
fallback visibility, and whether an unsustainable slow profile was suppressed.
The report omits package names/versions and device display strings. Exact
applicability is structural: a `standard`
non-fallback run requires `slow_profile_suppressed: false`; either
missing-dependency fallback requires it to be `true` together with the exact
visible reason above.

## Capture-only evidence

For every recorded cell, first run the production capture-only harness on the
same host, physical device, backend, canvas, and 30 FPS request:

```console
custback capture-diagnose \
  --config "$PRIVATE_EVIDENCE_ROOT/private-runtime.yaml" \
  --width 1280 --height 720 --fps 30 \
  --condition-id adequate_diffuse_light \
  --hardware-verified \
  --device-identity-sha256 "$DEVICE_IDENTITY_SHA256" \
  --hardware-identity-sha256 "$HARDWARE_IDENTITY_SHA256" \
  --output "$PRIVATE_EVIDENCE_ROOT/capture-linux-rvm"
```

Use the backend and a native-supported canvas for the actual cell. Follow the
[capture cadence diagnostic contract](capture-cadence-diagnostics.md),
including its physical-source eligibility and opaque device/host identity
requirements. The MATTE-5.3 join recomputes rate from the bounded trace and
checks the exact source sequence, generation, geometry generation, timing
summaries, failure/restart counters, and negotiated mode. Qualification needs
at least five measured seconds, at least 90% of target capture cadence, one
stable capture and geometry generation, no observer gaps, and no read,
restart, geometry, stall, or close failure. Capture-diagnostics v1 grants the
authoritative `hardware-target-sustained` disposition only for exact
1280x720@30 evidence. A 360p or 1080p cell is accepted structurally but remains
pending until its capture authority is expanded; in particular, native 1080p
support is not itself a MATTE-5.3 qualification.

The capture harness excludes segmentation, backdrop, compositor, preview, API,
and sink work. Do not relabel a full-pipeline observation as capture-only.

## Physical-capture and fixed-replay runs

Each recorded cell also references one
`custback.matte-platform-run-evidence` version-1 sidecar. It is a
content-free measurement record produced by an owner-controlled local/native
collector. Version 1 deliberately ships a validator/decision authority, not a
hardware collector, and Custback does not record or synthesize this sidecar.
Operators must use a reviewed local collector that emits the exact documented
schema; hand-authored claims have no more authority than their retained private
logs and attestations. The
sidecar binds the candidate build and policy, exact route,
hardware/runtime/provider identities, sink consumer, package set,
physical-authority attestations, and the exact capture-report file and
device/host identities.

One sidecar contains two independent sample arrays under `samples` and two
independently declared warm-up/duration scopes:

- `physical_capture` measures the complete live source-to-consumer path;
- `fixed_replay` target-paces an immutable replay while measuring repeatable
  processing capacity, so live camera cadence cannot hide a slow model,
  refiner, compositor, or sink. Its source/bundle/lineage contract is inherited
  from the separate unpaced MATTE-3.4 authority; its MATTE-5.3 presentation
  clock is intentionally target-paced.

Each source needs at least 30 warm-up frames, 300 measured submissions, and 10
measured seconds to qualify. Samples retain monotonic capture, segmentation,
composite, and output identities. Output repeats remain output submissions but
are never credited as unique segmentation or composite work. Per-source
counters are recomputed from the samples; hand-edited totals fail validation.
Local work IDs are zero-based, while fixed-replay samples separately retain the
upstream source sequence, timestamp, capture generation, and geometry
generation used to recompute the MATTE-3.4 lineage digest.

The complete non-pacing service clock starts at processing and ends after sink
submission/copy. For each output, `serialized_cycle_ms` is the previous-to-
current sink-completion interval and `pacing_wait_ms` is the non-overlapping
wait from the previous sink completion to current processing start; the first
row uses the measurement origin. Service plus pacing therefore equals the
cycle. These values, schedule lateness, queue age, and end-to-end age are
recomputed from clocks; impossible or overlapping timing evidence is rejected.
The sidecar records and the report
derives p50/p95/p99 for:

- model preprocessing, inference, and postprocessing where RVM applies;
- total segmentation, refinement, background selection, compositor, and every
  named compositor substage;
- the existing narrower frame-processing boundary and complete serialized
  cycle;
- sink preparation, submission, and copy;
- deliberate pacing, schedule lateness, queue age, and end-to-end age;
- output interval and inter-frame jitter; and
- process CPU/RSS plus GPU utilization/VRAM when an accelerator applies.

Timestamp arithmetic independently derives complete service, queue age, and
end-to-end age. Capture gaps/drops, latest-slot overwrites, processing deadline
misses, sink recoveries, and no-unread repeats stay separate; the qualifier
does not turn correlation into a causal claim.

## Performance and sustained-resource gates

For both physical capture and fixed replay, a balanced 30 FPS cell requires:

- at least 27 unique composites/s and output at least 90% of target;
- capture and unique segmentation rates of at least 27/s independently;
- p95 complete non-pacing service through sink copy at or below `33.333334 ms`;
- p95 complete serialized cycle at or below `33.333334 ms`;
- p95 1280x720 compositor time at or below the ratified `22 ms` sub-budget;
- p95 1280x720 refinement time at or below `5 ms`;
- bounded end-to-end age (p95 and maximum at most two target frames, and
  tail-versus-head p95 drift at most `33.333334 ms`);
- output maximum gap at most 1.5 target frames, p95 jitter at most half a frame,
  p95 lateness at most one frame, and unique-capture/composite maximum gaps at
  most two frames, including leading and trailing measurement boundaries;
- capture gaps, drops, latest-slot overwrites, and processing deadline misses
  each at most 5%; capture-only and full-run capture rates agree within 10%;
- no sink recovery in the steady window and at most a 5% processing-deadline
  miss ratio; output-repeat identity remains explicit and uncredited; and
- a verified physical consumer recording for local authoritative evidence.

Resource evidence is a separate ordered `resource_samples` series. Record at
least 31 approximately uniform samples spanning a 30-minute (`1800 s`) soak,
with no observation gap above 60 seconds. It reports process CPU
and RSS, plus GPU/VRAM only for an actual CUDA or DirectML provider.
Qualification limits signed RSS and VRAM drift to 64 MiB and each observed
max-minus-min span to 256 MiB. Head/tail drift is computed from elapsed-time
quarters rather than row density. An accelerator row must show at least 1%
median sampled GPU activity both in the soak and measured frames; a CPU row
must mark GPU and VRAM inapplicable instead of filling them with zero.

## Restart, hot patch, fallback, and shutdown

Every run records exactly four ordered lifecycle events: `sustained`,
`restart`, `hot_patch`, and `shutdown`. Restart and hot patch each advance the
recorded generation by exactly one; shutdown retains the final generation.
At least 31 counter snapshots, including bounded-gap heartbeats and explicit
pre/post markers, bind continuous unique capture/segmentation/composite/sink
work and event counters to the active provider and generation. Retain the
private operator log that ties those content-free offsets to the physical run.

The lifecycle gate requires:

- restart/reset visibility, a fresh first output, no stale-state flash, and a
  visible provider fallback exactly when one occurred;
- for CUDA/DirectML rows, an actually exercised and visibly reported runtime
  provider fallback during restart (separate from package-selection fallback);
- a transactional hot patch, visible reset, fresh first output, and no
  cross-generation flash;
- aggregate and every heartbeat interval at or above 27 unique capture,
  segmentation, composite, and sink submissions per second, with per-interval
  gap/drop/overwrite/deadline ratios at most 5% and zero sink recovery;
- bounded temporal history (at most two previous-frame slots, two previous-mask
  slots, and 32 work-buffer slots); and
- shutdown within `5000 ms`, with zero workers/resources left open and no
  close error.

A favorable steady-state average cannot waive a restart, fallback, hot-patch,
memory, or shutdown failure. Missing-dependency lanes suppress the unavailable
high-quality request visibly. A selected standard tier that cannot sustain its
declared limit fails its measured cell; it cannot silently qualify at a slower
rate.

## Collection checklist

- [ ] Copy the template into a new owner-only private root, replace its
      placeholders, and add every reported missing profile/route/provider/
      dependency lane; the checked-in rows are only a shape skeleton.
- [ ] Freeze the candidate build, configured/effective policies, and exact
      advertised profile limits without changing defaults.
- [ ] Disable reactions and every unlisted post-base stage.
- [ ] Bind intact MATTE-5.2 visual, MATTE-3.4 performance, and MATTE-2.5 RVM
      reports from the same candidate contract.
- [ ] Record capture-only evidence independently on every declared route.
- [ ] Record both physical-capture and fixed-replay sample sets per cell.
- [ ] Verify CPU/CUDA/DirectML execution and both missing-dependency fallback
      profiles where declared; never replace an unavailable provider.
- [ ] Retain physical consumer recordings and attest their origin and same-host
      relationship without claiming cryptographic proof.
- [ ] Run the sustained soak, restart, transactional hot patch, and bounded
      shutdown lifecycle in order.
- [ ] Run `matte-platform-qualify` and retain pending/failed reports as evidence;
      do not advertise or roll out a preset unless the separate
      [MATTE-5.4 authority](matte-quality-rollout.md) advances from its current
      compatibility hold.

## Privacy and public artifacts

Camera footage, replay frames, masks, consumer recordings, host/package
inventory, device display strings, and raw provider errors remain private.
Keep them outside the repository and do not upload them to CI. The private plan
contains bounded relative paths to its evidence. The public qualification
report retains no paths, pixels, distribution names/versions, or device labels—
only bounded IDs, opaque digests, policy values, derived numeric summaries/
counters, decision reasons, and owner attestations. Run provenance must state
`contains_pixels: false` and `contains_paths: false`.

Generated CI evidence may exercise schema, digest, arithmetic, and failure
sensitivity, but it has no physical, performance, provider, consumer, preset,
or default authority. The checked-in template must remain generated and
unavailable.

Related contracts:

- [End-to-end matte visual qualification](matte-visual-qualification.md)
- [MATTE-3.4 performance evidence](matte-performance.md)
- [RVM profile qualification](matte-rvm-profiles.md)
- [Capture cadence diagnosis](capture-cadence-diagnostics.md)
- [Deterministic matte regression gate](matte-deterministic-regression-gate.md)
