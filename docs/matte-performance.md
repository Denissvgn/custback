# MATTE-3.4 720p performance evidence

MATTE-3.4 provides a deterministic evidence and decision contract for the
balanced 1280x720@30 compositor and RVM/CUDA service path. It is a local,
owner-operated diagnostic. It does not upload source pixels, write extracted
frames into the report, or make a generated benchmark authoritative.

The Python entry points are:

- `profile_private_720p_replay_bundle(...)` for a validated, privacy-aware
  full-capture replay bundle;
- `profile_fixed_720p_compositor_samples(...)` for an explicitly supplied
  ordered sequence of `CompositorFrameSample` values;
- `profile_fixed_720p_compositor(...)` for a single-frame generated proxy;
- `collect_private_720p_full_path_evidence(...)` for the exact built-in
  balanced RVM/CUDA path and an explicit real output sink;
- `build_performance_report(...)`, `validate_performance_report(...)`, and
  `report_markdown(...)` for evidence collection and review.

The CLI integration uses the `matte-performance` command. Bundle extraction is
source-bound and fail-closed: the bundle must remain owner-only, complete, and
contain concrete `raw_frame`, `backdrop_frame`, `refined_mask`, and
`clean_foreground` tracks. The content-free report records a source SHA-256,
source kind, source-frame count, profiled-frame count, and—when applicable—the
bundle manifest SHA-256. It never records the bundle path.

```console
# Compositor matrix only. The report remains not_decidable without full-path
# model-backed evidence and exits 1 after writing the reviewable result.
custback matte-performance PRIVATE_BUNDLE --output NEW_REPORT_DIR

# Join a source-matched full-service sidecar.
custback matte-performance PRIVATE_BUNDLE \
  --full-path-evidence OWNER_ONLY_FULL_PATH.json \
  --output NEW_JOINED_REPORT_DIR

# Collect and join the full path in the same run. This requires a working
# built-in RVM model, CUDA provider, and explicit non-fallback output sink.
custback matte-performance PRIVATE_BUNDLE \
  --collect-full-path \
  --hardware-id cuda_lab \
  --cuda-device-id 0 \
  --sink-backend pyvirtualcam \
  --output NEW_QUALIFICATION_REPORT_DIR
```

The output directory must be new, owner-only, and outside the input bundle. It
contains `performance.json` (authoritative raw samples and decision) and
`performance.md` (review summary). Exit `0` means the complete matrix and
full-path decision qualified; exit `1` means `rejected` or `not_decidable`;
exit `2` means malformed/unsafe input or an execution error. `--warmup-frames`
and `--measured-frames` support short diagnostic runs, but values below 30/300
cannot qualify. Profiling and validation finish before output publication; the
two files appear together through one no-replace directory rename, so a failed
run does not leave an empty or partial report directory.

`--collect-full-path` is the preferred producer for local qualification. It
requires at least 330 distinct replay frames at the default 30 warm-up/300
measured settings; it never cycles frames through RVM or credits output
repeats. It opens only the requested `pyvirtualcam` or `native` sink, rejects
null/fallback output, verifies that the opened sink retained exact
1280x720@30 mode, and runs the production sink validation/submission seam
without its deliberate pacing sleep. The report records pacing as zero and
derives schedule lateness from each sink-submission service sample positioned
on the cumulative complete non-pacing cycle, not from replay-file reads.

Before warm-up begins, the collector owner-checks and loads only the required
raw and backdrop tracks into a bounded resident sequence. The default 330-frame
run retains 1,824,768,000 bytes (about 1.70 GiB); collection fails before model
work when the required sequence would exceed the fixed 2 GiB cap or cannot be
allocated. There is no source-artifact I/O between model invocations. Replay
acquisition is therefore outside both the timed service boundary and its
logical lateness clock, while disk gaps cannot cool or pace the sustained
model run.

For each resident frame, the collector uses production `Pipeline`,
`FrameHub`, and resource-generation ownership. The service clock begins before
capture validation, sequence/reset classification, raw latest-slot ownership,
segmentation, backdrop policy, compositor work, final privacy guard/output
validation, and sink submission/copy. It ends when the sink reports
submission. A second non-pacing cycle clock continues through the production
cadence accounting, timing/status projection, and atomic output/status
publication tail. The collector clears both private raw and output latest
slots and closes every resource even on failure.

The full-path replay has the explicit background-provider scope
`resident-recorded-frame-copy`: selecting the recorded backdrop includes one
full-frame provider-owned copy and reports it as
`background_selection_samples_ms`. This qualifies only the fixed recorded
backdrop replay profile. It does not measure or qualify video decoding,
dynamic camera backgrounds, or their frame-selection policy.

The command does not open live capture, preview, or a network service and does
not persist frame pixels, but it does submit composites from the sensitive
replay to the selected local virtual-camera device. Operators should stop other
consumers or select an isolated test sink when that local disclosure is not
intended.

[`matte-performance-local-template.json`](matte-performance-local-template.json)
is a three-sample generated-proxy shape template for the full-path sidecar. Its
zero/placeholder digests and short arrays must be replaced from the same
private replay and hardware run; changing `evidence_kind` without real
model-backed measurements is invalid. It remains useful for a separate native
collector; the integrated `--collect-full-path` producer fills these fields
directly. The report's first matrix-only run provides the required source and
bundle-manifest digests.

## Fixed matrix

Every run has exactly eight cells:

| Blend contract | Plain | Model foreground | Light wrap | Both |
| --- | --- | --- | --- | --- |
| `srgb_legacy` | yes | yes | yes | yes |
| `linear_srgb` | yes | yes | yes | yes |

The light-wrap cells run the shipped compatibility policy:
`light_wrap_stabilization.mode: off`. The production compositor owns the
stateless blur and interpolation work, and temporal-filter samples are
therefore zero. An explicitly stabilized candidate is not mislabeled as this
balanced path. The linear cells run the accelerated production linear-BGR
lane. Their full-frame EOTF input conversion is inside the measured total
unless a future evidence version proves exact upstream reuse. Source artifact
I/O and replay-contract validation remain outside the measured compositor
boundary.

The loop is intentionally unpaced. There is no wait or sleep between calls, so
capture or output pacing cannot hide compute capacity. An ordered source
sequence with at least two distinct capture frames may cycle deterministically
through 30 warm-up and 300 measured calls. This keeps a default bounded replay
practical; the report records source and profiled counts separately. The
full-path gate, unlike the compositor matrix, still requires 300 genuinely
unique model invocations, composites, and sends.

Before any cell starts, the matrix loads the exact resident source set needed
by the profiled span. The default 330-frame four-track set (raw, backdrop,
clean foreground, and float32 mask) is 3,953,664,000 bytes, about 3.68 GiB; a
fixed 5 GiB cap rejects larger resident sets before timing. Reference,
deterministic-repeat, and alpha-endpoint renders complete before that cell's
warm-up. Each measured cell is therefore a consecutive tight loop with no
artifact reads or correctness renders between timed calls.

The source scope contains a SHA-256 of the exact ordered measured capture
sequence, timestamp, capture-generation, and geometry-generation lineage,
including deterministic cycling when present. The full-path sidecar must match
that digest and warm-up/measured span; sharing only the same bundle manifest is
insufficient.

A qualified compositor row requires:

- fixed 1280x720 BGR inputs and finite float32 alpha in `[0, 1]`;
- at least 30 warm-up and 300 measured calls;
- an ordered multi-frame explicit sequence or validated private replay bundle;
- `fixed-replay-measured` evidence rather than `generated-proxy`;
- output-equivalence and deterministic-output checks;
- compositor p95 at or below the ratified 22 ms sub-budget.

The single-frame API always has `generated-proxy` scope and cannot qualify.
Short runs and generated arrays are useful for CI schema tests only. CI has no
strict wall-clock performance assertion.

## Timing and allocation evidence

Rows retain bounded raw samples and derive p50, p95, and p99 using nearest-rank
percentiles. Hand-edited summaries fail report validation. The compositor
diagnostic stages are:

1. input/mask validation;
2. edge-band calculation;
3. model-foreground replacement;
4. backdrop blur/resize;
5. light-wrap temporal filtering;
6. light-wrap interpolation;
7. final blend/conversion;
8. internal output validation.

The difference between the stage sum and total is allowed and visible. In
particular, it contains linear EOTF conversion and small orchestration costs.

`known_transient_allocation_bytes` includes known output/workspace allocation,
prepared-wrap pixels, and linear decode arrays created inside the boundary.
`retained_workspace_bytes` includes reusable legacy buffers and retained
light-wrap state. Both are lower bounds: hidden OpenCV/NumPy temporaries and
allocator traffic are not inferred.

`memory_bandwidth` is a separate field. When unavailable, the value, counter
source, source digest, hardware/provider digests, and measurement-run digest
are all null. A supplied native counter must bind every available row to the
matrix source and one hardware/provider/run identity; joined full-path
evidence must match those identities. Output byte counts are never presented
as memory bandwidth.

`--memory-bandwidth-evidence` accepts one owner-only JSON object keyed by the
exact eight matrix row IDs. Each value has this shape:

```json
{
  "available": true,
  "bytes_per_second": 123000000.0,
  "counter_source": "native_counter",
  "source_sha256": "<matrix source digest>",
  "hardware_identity_sha256": "<hardware digest>",
  "provider_environment_sha256": "<provider digest>",
  "measurement_run_sha256": "<same-run digest>"
}
```

The integrated collector does not synthesize those hardware counters. If they
cannot be captured and bound to the exact same run, omit this option and the
report truthfully records bandwidth as unavailable.

Workspaces are owned by one matrix cell, used serially, and closed in a
`finally` block. Light-wrap state is likewise cell-local and bounded. No state
survives the report call.

## Output equivalence

The measured production path is compared outside timed work with a separate
frozen reference:

- `srgb_legacy` must be byte exact;
- `linear_srgb` permits at most one code value per channel;
- shape must be 1280x720x3, dtype `uint8`, and storage C-contiguous;
- a repeated render from cloned temporal state must be byte deterministic.

Real model alpha may contain no exact zero or one. The measured source mask is
therefore left untouched. Outside timed work, the profiler derives a mask copy
with explicit zero and one pixels and proves non-vacuously that background and
foreground endpoints remain byte exact for every matrix cell.

## Full-path sidecar

The compositor matrix alone cannot qualify the advertised RVM/CUDA profile. A
version-1 `custback.matte-full-path-performance-evidence` sidecar must contain
at least 30 warm-up and 300 measured unique-frame samples for:

- complete service from unique-frame dequeue through sink submission/copy;
- the complete non-pacing cycle through synchronous cadence/status and atomic
  output publication;
- the existing narrower `frame_processing_ms` boundary;
- the complete serialized new-frame loop separately;
- RVM preprocessing, ONNX inference, and postprocessing separately;
- resident recorded-backdrop selection/copy;
- compositor work;
- post-composite guard/output validation;
- sink preparation/submission/copy;
- deliberate output pacing;
- schedule lateness.

The complete service boundary excludes deliberate pacing wait. Its component
stages may not exceed the enclosing service sample. The sidecar also binds the
evidence to the matrix source SHA-256, bundle manifest SHA-256 when present,
ordered measured-lineage SHA-256, built-in RVM model SHA-256, and opaque
hardware, provider-environment, sink, and measurement-run SHA-256 values.
Qualification requires the built-in RVM model, CUDA provider, no fallback, and
distinct hardware/provider/sink identities.

The matrix and full path must use the exact same warm-up/measured span, and the
source must contain at least one distinct recorded frame for every full-path
model invocation. A sidecar cannot turn a short, cyclic matrix source into
unique-frame evidence merely by matching a manifest or total profiled count.

The hardware/provider digests are fail-closed bindings, not operator labels.
Before any resident replay pixels are loaded, the integrated collector obtains
the selected CUDA ordinal's driver UUID, PCI bus ID, device names and memory,
compute capability, CUDA driver/runtime versions, NVIDIA driver version, and
NVML version through CUDA Driver, CUDA Runtime, and NVML APIs. Those exact
facts, the operator tier label, and the ONNX Runtime environment are hashed
into the report; raw device facts, hostnames, and paths are not published.
Collection stops if the three APIs cannot identify one matching device.

The sidecar must describe the actual balanced compatibility policy: resolved
RVM ratio `0.4` at 720p, native soft alpha, `mask_shift: 0`, model-only
temporal behavior, model foreground enabled, `light_wrap: 0.25`, temporal
light-wrap stabilization off, legacy sRGB blending, and color correction off.
It must use the `srgb_legacy_both` matrix row and attest that sink
submission/copy is inside the service boundary. Plain-feature, linear-light,
experimental-stabilization, or unidentified-sink evidence cannot qualify the
balanced lane.

These event classes remain separate:

- output repeats and no-unread repeats;
- capture-slot overwrites, capture sequence-gap events, and the number of
  missing sequence values;
- processing deadline misses;
- sink recovery;
- deliberate pacing wait;
- schedule lateness.

The validator correlates none of them into an unsupported causal claim.
Repeats are never credited as unique model or compositor work. They are
allowed in a trace only when output sends equal unique composites plus repeats
and no-unread repeats are a subset of all repeats. Processing-only deadline
misses derive from `frame_processing_ms`; serialized new-frame misses use the
separate serialized samples and production grace threshold. Complete service
plus deliberate pacing must fit inside that serialized sample; overlapping or
impossibly nested timing evidence is rejected.

## Decisions and headroom

The compositor sub-budget is p95 `<= 22 ms`. Every row and the matrix report
signed headroom as `22 - p95`; a negative result remains negative. The
model-backed full-path run must independently meet the same compositor p95
gate on the qualified hardware, so a fast matrix collected on another host
cannot hide an over-budget compositor in the complete path.

Complete non-pacing service qualifies when either:

- p95 is `<= 33.333334 ms`; or
- the explicit 30-FPS logical-arrival simulation sustains at least 27 unique
  composites/s with bounded queue age.

In both cases, the complete non-pacing cycle must sustain an explicit tier of
at least 27 FPS. A fast sink-submission boundary cannot hide a slow
post-submission status/publication tail.

The ordered complete-cycle samples are swept at logical arrivals of
30/27/24/20/15 FPS. The declared sustainable lower-rate profile is the highest
explicit tier without logical sequence gaps. It is never computed as
`1000 / p95`. A tier is declared only when model-backed, source/lineage,
balanced-policy, sink, hardware, sample-floor, and no-recovery prerequisites
are authoritative. Generated, short, mismatched, or otherwise incomplete
evidence leaves the declared tier null even when its diagnostic simulation
looks fast.

The sweep checks both logical sequence gaps and ending-versus-starting queue
age. Gap count is the number of discontinuity events; missing-input count is
the number of skipped logical sequence values. A finite trace that has not yet
skipped an arrival but is accumulating queue age is not labeled sustainable at
that tier.

Report outcomes are:

- `qualified`: all eight authoritative matrix rows and model-backed full-path
  evidence pass;
- `rejected`: authoritative evidence fails equivalence, identity, backend, or
  performance gates;
- `not_decidable`: evidence is missing, generated, single-frame, or below the
  sample floor.

Reactions are disabled throughout MATTE-3.4. `decision.reaction_ready` is always
false. Signed residual headroom is evidence for a later reaction qualification,
not proof that any optional reaction stage already fits.
