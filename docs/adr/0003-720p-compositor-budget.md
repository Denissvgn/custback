# ADR 0003: Recover the 720p base-compositor budget without claiming an unqualified frame rate

- Status: Accepted
- Date: 2026-08-04
- Scope: MATTE-3.4

## Decision

The production `srgb_legacy` compositor uses compiled OpenCV arithmetic and a
serial, generation-owned workspace. The optimized lane must remain byte-exact
with the frozen allocation-heavy implementation. Each returned output owns
independent storage; the workspace never retains source pixels after its
resource generation closes.

The base compositor has a `22.0 ms` p95 sub-budget at 1280x720. This is a
compositor-only screen, not a claim that the complete service path meets 30
FPS. It reserves `11.333334 ms` of the 30 FPS presentation interval for RVM
pre/postprocessing and inference, backdrop work, post-composite privacy
guard/validation, and sink preparation/submission.

The complete balanced RVM/CUDA path qualifies only when fixed 1280x720 replay
evidence, with no input pacing, proves either:

1. p95 service time from unique-frame dequeue through privacy guard,
   validation, and sink submission/copy is at most `33.333334 ms`, excluding
   deliberate pacing; or
2. an explicit logical-arrival sweep proves at least 27 unique composites/s
   without queue-age growth.

Both branches also require the measured complete non-pacing cycle—including
the synchronous cadence/status and atomic output-publication tail after sink
submission—to sustain an explicit tier of at least 27 FPS. The balanced
full-path compositor itself must independently remain at or below the
`22.0 ms` p95 sub-budget on that qualified hardware.

Output FPS, repeated output ticks, an EWMA, or `1000 / p95` cannot substitute
for the second test. Deliberate sink/application pacing and schedule lateness
remain separate evidence. A finite arrival trace with no skip yet still fails
the sustainable-tier label when its ending queue age exceeds its starting age.

No checked-in artifact currently qualifies the balanced RVM/CUDA path on a
hardware tier. The default classification therefore remains `not_decidable`;
no named performance profile or feature default changes. A failed or
incomplete run must not silently become a lower-rate or 30 FPS claim.

## Accepted recovery work

The optimization order is evidence constrained:

1. Use compiled OpenCV multiply/add/merge operations while preserving the
   historical float32 operation order.
2. Reuse bounded scalar, three-channel, and reduced-resolution work buffers
   for one resource generation. Allocate a fresh final `uint8` frame on every
   call so published frames cannot alias later work.
3. Avoid the duplicate downscale-and-blur pass when preparing stabilized light
   wrap. The first native blurred raster is reused only when the temporal
   filter returns the current sample unchanged; filtered history is otherwise
   upscaled.
4. Prepare the RVM input tensor without redundant full-frame RGB/float
   temporaries, while retaining the synchronous ONNX input ownership
   boundary.

These changes preserve exact alpha endpoints and the `srgb_legacy` byte
contract. The `linear_srgb` contract remains tolerance based away from
endpoints and byte-exact at alpha zero and one.

## Deferred alternatives

- Uncertain-band ROI processing is deferred. Proving equivalence at ROI
  boundaries and accounting for gather/scatter cost needs separate evidence.
- Backdrop/frame caching is allowed only with exact content/generation
  identity. No general temporal cache is introduced.
- Model-foreground and light-wrap defaults are unchanged. Their visual value
  and cost are reported as four distinct feature cells.
- Pipeline overlap is deferred. It would add recurrent-ordering, latency,
  shutdown, and privacy ownership that local reductions do not yet justify.
- Memory-bandwidth counters are reported unavailable when no native counter is
  present; allocation lower bounds are not relabeled as bandwidth.

## Measurement contract

Every report contains all four feature combinations (`plain`,
`model_foreground`, `light_wrap`, and `both`) for both `srgb_legacy` and
`linear_srgb`. A qualifying cell has at least 30 unreported warm-up frames and
300 measured frames. Raw samples are retained, and p50/p95/p99 use the
nearest-rank rule. Light-wrap cells use the shipped stateless compatibility
policy; temporal stabilization is not silently enabled for the benchmark.

Compositor evidence separates input/mask validation, edge-band work, model
foreground replacement, backdrop blur/resize, temporal wrap filtering, wrap
interpolation, final blend/conversion, and internal output validation. Pipeline
evidence separately records post-composite and final privacy-guard validation,
sink submission/copy, deliberate pacing, schedule lateness, and complete
new-frame service. It also records the complete non-pacing cycle through
cadence/status projection and output publication. The established narrower
frame-processing boundary and serialized new-frame loop remain separate
samples and deadline counters.

Full-path evidence binds the exact measured capture lineage, balanced
`srgb_legacy_both` effective controls, RVM ratio, sink identity, and one opaque
hardware/provider/measurement-run identity. Output repeats may be present but
remain separate from the one-model-invocation-per-unique-composite count.
The integrated collector fails closed unless CUDA Driver, CUDA Runtime, and
NVML agree on one exact selected device identity. It hashes UUID, PCI bus ID,
device names/memory, compute capability, CUDA driver/runtime versions, NVIDIA
driver, and NVML version with the operator tier; it does not publish those raw
facts.

All required raw/backdrop pairs are owner-checked and loaded before warm-up.
The default 330-frame full-path sequence retains 1,824,768,000 bytes (about
1.70 GiB) and a fixed 2 GiB cap rejects a larger run before model execution.
The timed service run performs no replay-artifact I/O. Schedule lateness places
each sink-submission service sample on the cumulative complete-cycle clock.
Each resident envelope retains its recorded capture and geometry generations.
The collector exercises production `Pipeline`, `FrameHub`, resource,
compositor, privacy-guard, validation, sink-submission, cadence/status, and
atomic output-publication seams, then explicitly clears raw and output
ownership before closing the generation.

The full path uses only a named `resident-recorded-frame-copy` background
provider. Its full-frame copy is timed separately; video decoding and dynamic
background selection are outside this fixed-replay qualification. The matrix
preloads raw, backdrop, clean-foreground, and float32-mask tracks before any
cell. A default 330-frame resident matrix is 3,953,664,000 bytes (about
3.68 GiB), with a fixed 5 GiB cap. Reference, repeat, and alpha-endpoint
correctness renders complete before each cell's warm-up, leaving consecutive
timed calls free of artifact I/O and correctness work.

Known application allocations are distinguished from process RSS, accelerator
VRAM, and unavailable native memory-bandwidth counters. Workspaces expose
content-free byte counts and close idempotently on rebuild or shutdown.
Activation trials use a temporary workspace so rollback cannot publish or
retain trial-owned compositor state.

Residual p95 compute headroom is:

```text
33.333334 ms - complete_service_p95_ms
```

The value is signed. A non-positive value fails the p95-service branch but the
explicit at-least-27-unique/s bounded-queue alternative may still qualify. It
never supplies reaction headroom. A positive value does not make the path
reaction-ready: later reaction qualification must measure its own
zero/one/maximum profiles inside that observed residual at the post-base seam,
with reactions disabled for this decision.

## Consequences

The production legacy lane substantially reduces full-frame temporary
allocation and exposes its cost rather than hiding it inside a single
compositor EWMA. Operators receive an honest `not_decidable`, qualified, or
explicit lower-rate result tied to source and hardware evidence. Near-30 output
ticks containing repeats are never described as 30 FPS processing.

The privacy-aware replay bundle remains opt-in and private. Matrix-only
profiling never opens capture, a network service, preview, or a sink. Explicit
full-path collection opens only the requested real local virtual-camera sink
after verifying exact 1280x720@30 mode and submits replay composites through it
without pacing; it never opens live capture, preview, or a network service.
Neither mode copies identifiable pixels into its JSON report.
