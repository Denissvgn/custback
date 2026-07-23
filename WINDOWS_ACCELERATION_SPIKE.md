# WIN-4.7 — DirectML / Windows ML acceleration go/no-go spike

Status: planning gate (code hook + WIN-6.2 gate harness landed; benchmark +
decision pending hardware)
· Companion to `WINDOWS_IMPLEMENTATION_PLAN.md` Phase 4 · Feeds WIN-6.2

This is the go/no-go spike required by WIN-4.7. Its job is not to ship a generic
GPU story; it is to decide, with evidence, whether Robust Video Matting (RVM) on
`DmlExecutionProvider` (DirectML) — or a Windows ML packaging of the same — is
good enough to justify a "runs on all Windows GPUs" claim (WIN-6.2). Until this
passes, the product is described as **NVIDIA/CUDA-accelerated with CPU fallback**
only, per the WIN-0.3 advertising guardrail.

## What already landed (the code hook)

The acceleration seam added in Phase 4 is provider-generic, so the DirectML path
is *wired but unproven*:

- `AccelerationConfig.provider` accepts `directml`
  (`src/custback/config.py`). `mode: cpu` still forbids a pinned GPU provider.
- `acceleration.resolve_provider_candidates()` maps `directml` →
  `DmlExecutionProvider` and, under `provider: auto`, tries CUDA before DirectML
  before CoreML (`_AUTO_GPU_ORDER`).
- `acceleration.prove_rvm_provider()` proves **real RVM node execution** on the
  chosen provider via ORT profiling — the same standard CUDA must meet. A
  DirectML provider that registers but cannot execute RVM is treated as absent
  and latches to CPU (in `auto`) or fails startup (in `gpu_required`).
- `AccelerationConfig.device_id` is threaded into the DirectML provider options
  for multi-adapter machines.

Because proof and fallback are provider-agnostic, the spike does **not** need new
runtime plumbing — only the `onnxruntime-directml` wheel, hardware, and a
benchmark harness. That keeps the go/no-go an evidence question, not an
engineering-risk question.

## Decision this spike must return

A written go/no-go for each of:

1. **DirectML RVM correctness** — does `prove_rvm_provider` return `proven=True`
   for `DmlExecutionProvider` on representative AMD and Intel adapters, and does
   the matte output match the CPU reference within tolerance?
2. **DirectML RVM performance** — 720p and 1080p median/95p frame time vs. the
   CPU baseline and the CUDA reference on comparable-tier hardware. Threshold:
   must sustain the configured output FPS at 720p to count as "GPU-capable".
3. **Windows ML packaging** — is a Windows ML (`Microsoft.AI.MachineLearning` /
   ORT-in-WinML) packaging viable for the frozen installer without shipping a
   second ORT stack, and what is the size/redistribution cost?
4. **Wheel/runtime conflict** — `onnxruntime-directml` vs `onnxruntime-gpu`
   (CUDA) cannot both be installed (mirrors the CUDA/CPU either-or in
   `install.js`). Confirm the installer profile selection story before any
   bundling.

## Gate harness (WIN-6.2, landed)

The method below is now executable, not prose:
`scripts/release/windows-acceleration-gate.py` implements both halves —

- `run` (Windows hardware): proves DirectML RVM execution via
  `custback.acceleration.prove_rvm_provider`, measures alpha drift against the
  CPU reference and 720p/1080p frame times, records adapter identity and the
  wheel-conflict state, and writes one evidence JSON per machine.
- `check` (any OS, stdlib-only): validates the evidence against the exact
  schema and the go/no-go criteria (both-vendor coverage, drift tolerance
  mean ≤ 0.005 / max ≤ 0.02, 720p ≥ target FPS, no CUDA/DirectML
  co-installation). Exit 0 = go. `tests/test_windows_acceleration_gate.py`
  pins these criteria so they cannot drift silently.

The `directml` pip extra (`onnxruntime-directml`) exists for the harness and
the opt-in profile; `packaging/windows/pyinstaller/build.ps1` refuses to
freeze `gpu` and `directml` together (question 4's either-or, enforced).
This checker becomes the `windows-acceleration` gate validator once WIN-1.8 /
WIN-5.8 wire a Windows evidence source; until evidence passes, the feature
matrix keeps DirectML unadvertised.

## Benchmark method (reproducible)

- Env: clean Win11 x64 venv; install `onnxruntime-directml`; pin the same
  `rvm_mobilenetv3_fp32.onnx` the product uses.
- Correctness: run `custback.acceleration.prove_rvm_provider(ort, model,
  ProviderCandidate("DmlExecutionProvider", {"device_id": N}))`; then compare
  `RVMSegmenter` output on a fixed synthetic sequence against a
  `mode: cpu` run (per-pixel alpha delta histogram).
- Performance: drive `python -m custback --synthetic` with
  `acceleration: {mode: auto, provider: directml}` at 1280×720 and 1920×1080;
  read `acceleration_active_provider`, `fps_attainment_pct`, and
  `frame_processing_ms` from `/status`.
- Record per adapter: {vendor, model, driver}, proven?, FPS, attainment,
  fallback_count.

## Go / No-go criteria

- **Go (feeds WIN-6.2):** DirectML proven on ≥1 AMD and ≥1 Intel adapter, matte
  within tolerance, and 720p attainment ≥ target on mid-tier hardware, with a
  viable installer profile that does not collide with the CUDA wheel.
- **No-go:** any of correctness failure, sub-target 720p, or an unresolved
  packaging/redistribution blocker. The CUDA-first claim stands; DirectML remains
  behind `provider: directml` as an unadvertised, best-effort opt-in that still
  falls back to CPU truthfully.

## Out of scope

Shipping/bundling DirectML in the installer (that is WIN-5.1/WIN-5.6 gated by
WIN-0.4 licensing and by this decision) and the "all Windows GPUs" advertising
claim (WIN-6.2, which this spike gates).
