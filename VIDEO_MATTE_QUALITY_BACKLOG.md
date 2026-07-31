# Video matte stability, output quality, and reaction-effects backlog

**Status:** proposed; evidence revised after the 2026-07-31 RVM/CUDA run;
reaction-effects lane added

**Primary symptoms:** the subject silhouette, especially the face/head boundary,
appears to "dance" between video frames; the RVM/CUDA sample also shows a broad
halo and apparent backdrop leakage through normally opaque clothing/accessories

**Scope:** local background replacement (`blur`, `image`, `video`, `color`, and
camera backdrops), shared capture/segmentation/output infrastructure, and
optional explicitly triggered final-output reaction effects

**Task prefixes:** `MATTE` for matte/output-quality work; `REACT` for reaction
effects

## 1. Purpose

This document turns the reported video-quality problem into an implementation
backlog. It preserves the observations, code-path analysis, hypotheses,
measurement plan, dependencies, acceptance criteria, and rollout constraints
needed for a developer to work on individual tasks without reconstructing the
investigation.

Section 11 adds a separate reaction-effects delivery lane. Reactions are
adjacent output functionality rather than a workaround for matte defects: they
must not obscure alpha evidence, inflate reported camera/matte update rates, or
block declaring the original matte problem fixed. Production enablement waits
for the base output-loop budget and privacy contracts to pass independently.

The attached screenshots are useful but are not, by themselves, proof of
temporal behavior. A still image can show a coarse/haloed edge, while "dancing"
requires a frame sequence to measure. The backlog therefore treats the proposed
causes as ranked hypotheses and makes a reproducible video baseline the first
gate.

## 2. Evidence from the observed runs

### Run A — original MediaPipe/CPU screenshot

The first preview overlay reports:

- capture is approximately `15.0/30 fps`;
- output is approximately `29.7/30 fps`;
- segmentation is `mediapipe/cpu`;
- the camera reports `1280x720 MJPG` and `30.0 fps`;
- the pipeline warns `CAPTURE BELOW TARGET`;
- the camera frame is already `1280x720`, with the schema-v1 `stretch` policy;
- the video backdrop is `24.0 fps`, container-timed, with about `38%` skipped.

This run established that the output sink repeats already-rendered frames to
meet its send cadence while only about 15 distinct segmentation/composite
updates occur per second. It also motivated the MediaPipe timing, watershed,
fallback-reporting, and true-matting availability work below.

### Run B — RVM/CUDA after the provider-session fix

The later run proves that the preferred backend and accelerator are active:

```text
RVM acceleration active on CUDAExecutionProvider
using rvm matting backend on cuda
segmenter=RVMSegmenter/cuda
```

At shutdown after 40.2 seconds, the summary reports the following.
`frames_in` counts pipeline-consumed unique frames (including startup/preflight),
not every camera-delivered frame. Other frame counts are run totals;
`capture_fps`/output FPS use rolling timestamp windows, and
`capture_read_ms`/stage times are EWMAs. They must not be recomputed by dividing
totals by uptime or treated as p95:

| Observation | Value | Consequence |
| --- | ---: | --- |
| Pipeline-consumed unique frames / output sends | `541 / 1105` | One new output was produced per pipeline-consumed frame |
| Repeated output sends | `564` (`51.0%`) | About half of displayed/sent frames contain no new camera or matte observation |
| Capture rate / read time | `15.0 fps / 66.6 ms` | Rolling capture timestamps and EWMA `cap.read()` wall time both indicate about 15 Hz despite a negotiated 30 FPS mode |
| Output rate | `27.9 fps` rolling | The sink is near its target by repeating, not by producing 30 unique composites |
| Segmentation | `26.4 ms` | The combined RVM stage (GPU inference plus CPU conversion/validation/refiner work) fits inside 33.3 ms, but leaves little frame budget |
| Background | `1.9 ms` | Video decoding/background selection is not the primary compute bottleneck |
| Composite | `32.7 ms` | CPU compositing alone consumes essentially the entire 30 FPS frame budget |
| Frame processing | `61.3 ms` | This processing-only EWMA corresponds to about 16.3 unique frames/s if representative; it excludes later guard/send work |
| Output send | `2.9 ms` | Virtual-camera send is not the dominant cost |
| Video skips | `334` | `334 / (541 + 334) = 38.2%`, matching phase retention when a 24 FPS backdrop is sampled at about 15 unique updates/s |
| Failures / restarts | `0 / 0` | The sample is a steady generation, not a restart/recovery artifact |
| Capture overwrites / deadline misses | not in shutdown line | The log cannot show how many camera frames were overwritten before pipeline consumption or how often processing missed its deadline |

The screenshot is spatial evidence, not proof of temporal "dancing." It appears
to show:

- a broad soft/dark-colored fringe around the skull, headphones, shoulders,
  and shirt boundary;
- recognizable backdrop content bleeding through apparently opaque shirt/torso
  regions;
- uncertain or incomplete accessory coverage around the headphones; and
- jagged/patchy shoulder opacity while the center of the face is comparatively
  opaque.

That is a wider defect than contour jitter. It suggests an alpha-coverage or
opaque-core confidence problem that temporal smoothing alone cannot fix.
However, raw RVM alpha (`pha`), RVM clean foreground (`fgr`), and the exact
compositing controls were not captured, so the screenshot cannot yet locate
the defect between model alpha and edge-color processing.

The ready/overlay output does not report the effective `rvm_downsample`,
`mask_shift`, `use_model_foreground`, or `light_wrap`. Repository defaults imply
an auto RVM ratio of `512 / 1280 = 0.4`, `mask_shift: 0`,
`use_model_foreground: true`, `light_wrap: 0.25`, and `srgb_legacy`, but
`CONFIG v0` is the runtime config-store revision/generation rather than proof
that no persisted override exists. Treat those values as hypotheses until
diagnostics record configured and effective settings.

### Combined conclusions and limits

1. The prior ONNX Runtime provider conflict is fixed for this machine.
   RVM/CUDA availability is no longer a candidate cause of Run B.
2. Preferred-backend activation did not, by itself, eliminate visible spatial
   artifacts. RVM quality, RVM policy, and the compositor must now be measured
   directly.
3. The two screenshots are not a controlled MediaPipe-versus-RVM comparison:
   pose, clothing, headphones, and frames differ. Only a same-source replay can
   support a relative backend-quality conclusion.
4. Run B has two separately measured cadence constraints: a roughly 15 FPS
   capture path and a roughly 16 FPS serialized processing path. They may share
   resource contention, so capture-only and fixed-replay tests must establish
   causality. The measured run cannot reach 30 unique outputs by repairing only
   one while the other remains unchanged.
5. `1105 - 564 = 541`: every pipeline-consumed unique frame generated one new
   output and all remaining sends were repeats. This equality does not prove
   that every camera-delivered frame was consumed; capture-slot overwrite/drop
   counters are absent from the shutdown line. Repeats do not invent alpha
   noise, but make updates persist and then jump at roughly 15 Hz.
6. Matching `1280x720` input/canvas/output, one stable geometry generation,
   zero restarts, and disabled color correction make geometry conversion,
   stale restart state, and color correction unlikely causes of Run B.
7. RVM bypasses watershed, generic blur, and generic temporal EMA. Those cannot
   explain Run B, although they remain relevant to the MediaPipe fallback lane.
8. The local preview already contains the defect, so meeting-app compression is
   not the primary source. The virtual-camera/consumer boundary still needs a
   final isolation test.

## 3. Current pipeline and relevant behavior

The steady-state path is:

```text
OpenCV capture worker
  -> newest unread canonical BGR frame
  -> Segmenter.segment(frame)
  -> MaskRefiner.refine(mask, frame)
  -> backdrop.frame(...)
  -> compositor(frame, backdrop, mask, optional RVM foreground)
  -> processed hub / preview / virtual camera
  -> repeat last guarded output when no new capture frame is available
```

Important implementation facts:

| Area | Current behavior | Quality implication |
| --- | --- | --- |
| Capture slot | `OpenCVCapture.read()` returns the newest unread pixel array, but not its capture timestamp or sequence metadata | Segmentation and temporal filters cannot use true frame intervals or directly detect sequence gaps |
| Capture cadence | The reader is independent and records measured FPS/drop counts | A measured 15 FPS can come from device exposure/mode, backend behavior, or system contention; it is not automatically a segmentation timing result |
| Output cadence | `Pipeline._loop()` processes each new capture once and repeats `last_output` on empty reads | Both observed runs show about 15 unique mattes/s under a near-30 FPS send cadence; this makes each contour change persist for roughly two sends |
| MediaPipe time | `MediaPipeSegmenter.segment()` increments `_ts_ms` by a fixed 33 ms | The VIDEO-mode model is told that a 15 FPS stream is 30 FPS, and capture gaps/restarts are hidden |
| MediaPipe resize | A returned mask with the wrong shape is resized without an explicit interpolation policy | The intended soft-alpha resampling contract and exact dimensions are not observable or frozen by tests |
| RVM time/state | RVM feeds recurrent state forward and resets it on a size change or GPU recovery | A camera restart/discontinuity at the same dimensions can reuse state from an unrelated temporal sequence |
| RVM alpha/detail | RVM returns `pha` as the alpha without calibration; auto detail uses about `512 / long_edge` (`0.4` at 1280px) | Opaque-core underconfidence, exterior halo mass, and detail/cadence sensitivity are not currently measured |
| Mask refiner | Watershed edge snapping, mask shift, Gaussian feathering, then adaptive EMA | Watershed is independently solved per frame; the EMA reduces its hold at pixels whose alpha changes, including moving/jittering contours |
| RVM postprocess | RVM disables watershed, blur, and temporal EMA, retaining only mask shift | This protects fine matte detail but leaves no configurable safety net for residual RVM shimmer and makes one config field have backend-dependent effects |
| RVM foreground | The compositor can replace the source color with RVM `fgr` in `4 * alpha * (1 - alpha)`; repository default is enabled | The clean-foreground prediction can improve spill at a good edge or make a broad uncertain region look washed/contaminated; it does not repair low alpha |
| Light wrap | A blurred copy of the current backdrop is mixed into the same soft edge band; repository default strength is `0.25` | Motion or brightness changes in a video backdrop can change edge color even when alpha position is stable, creating perceived shimmer |
| Legacy compositor | `srgb_legacy` creates several full-frame float32 arrays for model-foreground replacement, wrap, and final blend | Run B measured `32.7 ms` at 720p, so compositing is both a quality interaction and a P0 throughput problem |
| Backend fallback | `auto` tries RVM, then MediaPipe, then heuristic; only heuristic is reported as a segmentation fallback | Operators are not warned that the preferred matting tier was unavailable when MediaPipe is selected |
| Quality controls | The UI exposes threshold, RVM detail, blur, shift, smoothing, edge refine, and light wrap | `threshold` currently affects only the heuristic backend, and several controls are silently neutralized for RVM; operators can believe a change affected the active path when it did not |
| Timing telemetry | Status contains deadline/drop fields, but shutdown/ready evidence omits some counters and effective RVM controls | Average output FPS can hide whether repeats came from capture starvation, input overwrite, or processing deadline misses |

Relevant code:

- `src/custback/capture.py`: `CaptureHealth`, `_reader_loop`,
  `_normalize_delivered_frame`, and `read`;
- `src/custback/segmentation.py`: `MediaPipeSegmenter`, `RVMSegmenter`,
  `MaskRefiner`, `refiner_for`, and `create_segmenter`;
- `src/custback/pipeline.py`: `_segment_and_refine_mask`,
  `_local_composite`, `_loop`, and `_identity_stats`;
- `src/custback/compositor.py`: edge-foreground replacement and light wrap;
- `src/custback/preview.py`: status/warning overlay;
- `src/custback/api/webui.py`: quality controls;
- `tests/test_processing.py`, `tests/test_segmentation_rvm.py`,
  `tests/test_capture.py`, and `tests/test_pipeline.py`.

## 4. Ranked hypotheses

The tasks must keep these causes separable. A single visual symptom can contain
more than one of them. Confidence below refers to contribution to the observed
run, not certainty about the lowest-level root cause.

### H1 — Effective alpha/opaque coverage is deficient across subject regions

**Confidence: high for the final-composite artifact; producing stage unknown.**

Recognizable backdrop texture appears through broad shirt/shoulder regions in
Run B. In the current blend equation, sharp backdrop content can contribute
there only when the effective alpha is below one. Raw RVM `pha` was not
captured, so raw-model underconfidence is a leading sub-hypothesis rather than
a conclusion. RVM's refiner disables blur, watershed, and temporal EMA, leaving
candidate sources such as raw `pha`, the model/profile/source conditions,
effective `mask_shift`, and low-cadence recurrent observations.

Attribution signature: if raw-model underconfidence is correct, raw `pha` has
low opaque-core percentiles, holes, or excessive uncertain/exterior mass in
annotated shirt/headphone regions. If raw `pha` passes but post-shift/final
alpha fails, `MATTE-0.5` assigns the defect to that later boundary instead.

### H2 — Frame-processing work separately constrains unique output near 16 FPS

**Confidence: high; directly measured.**

Run B spends `61.3 ms` in the measured frame-processing scope, versus a
`33.3 ms` full-loop budget at 30 FPS. `segmentation_ms=26.4`,
`background_ms=1.9`, and `composite_ms=32.7` nearly account for that value.
The timer ends before the later guard/output validation and serial
`output.send` (`2.9 ms` rolling), so `1 / 61.3 ms ~= 16.3 FPS` is a
processing-only reciprocal, not a complete-loop capacity measurement. If it
remains representative under a fixed replay, the current serialized path
cannot make 30 unique composites.

The `srgb_legacy` compositor is CPU/NumPy work. A local 1280x720 diagnostic spot
benchmark during this analysis reproduced the runtime shape: about `16.0 ms`
for plain alpha blend, `24.0 ms` with wrap only, `25.2 ms` with model foreground
only, and `33.2 ms` with both; linear-light plus both was about `83.2 ms`.
These host observations are diagnostic evidence, not portable release gates.

Expected signature: a fixed 30 FPS replay reproduces a service rate near 16
unique outputs/s; disabling inferred-default edge features reduces composite
time toward the plain-blend result. `MATTE-3.4` must measure the distribution
rather than treating the EWMA reciprocal as a hard maximum.

### H3 — The capture health window shows about 15 FPS

**Confidence: high for the symptom; cause unknown.**

The driver negotiates 30 FPS MJPG, but `capture_read_ms=66.6` maps directly to
15.0 FPS in the shutdown health window. Zero read failures, zero restarts, and
stable generation/geometry make recovery churn an unlikely explanation for
that window, but do not prove a constant rate across all 40.2 seconds. Possible
causes include low-light auto-exposure, device/driver pacing, mode/backend
behavior, decode cost, USB bandwidth, or resource contention.

Expected signature: capture-only also reads at about 66.6 ms, or the rate
changes with lighting/device controls, native capture, format, or removal of
the processing workload.

### H4 — Capture/processing cadence mismatch amplifies hold/jump edge motion

**Confidence: high as an amplifier, low as the sole alpha cause.**

Exactly `564 / 1105 = 51.0%` of Run B output sends repeat the previous result.
Repeats do not invent new contours, but they expose each matte/backdrop state
for roughly two sends and then jump to the next. Sampling a 24 FPS backdrop at
about 15 unique updates also causes the observed `38.2%` phase-retention skips.

Expected signature: output repeats occur on send opportunities where
`read()` finds no unread capture, while edge/backdrop updates occur only for
pipeline-consumed sequences. Capture sequence gaps are a distinct signal that
the newest-frame slot overwrote one or more frames; repeats and gaps must not be
equated.

### H5 — Model foreground and light wrap amplify color contamination in a bad alpha band

**Confidence: medium-high as a secondary visual effect.**

If repository defaults are effective, both RVM clean-foreground substitution
and `light_wrap: 0.25` run inside `4 * alpha * (1 - alpha)`. They do not change
alpha or repair core transparency, but can make an already broad uncertain
region darker, washed, or backdrop-colored. At `alpha=0.5`, the band weight is
one: the final approximate mix becomes 37.5% RVM foreground, 12.5% blurred
backdrop, and 50% sharp backdrop.

Expected signature: raw alpha metrics stay constant while the edge-band RGB
artifact changes across the four
`use_model_foreground × light_wrap` combinations. A moving video backdrop
changes RGB metrics even with fixed alpha.

### H6 — RVM auto detail/model/source conditions are insufficient for this subject

**Confidence: medium-high.**

Auto detail is approximately `0.4` at 1280x720. Higher ratios may improve
clothing/accessory coverage and fringe localization but cost inference time and
memory. The MobileNet model, MJPG source quality, low-light/noisy imagery,
headphones, clothing contrast, and feeding recurrent state at 15 Hz may also
matter. None can be selected from a still.

Expected signature: the same raw frames produce materially different
opaque-core/halo metrics across `0.4/0.5/0.67/1.0`, model variants, or temporal
decimation while the compositor is held constant.

### H7 — Temporal state lacks real timestamps, resets, and an evidence-backed RVM policy

**Confidence: medium for general dancing; low as an explanation for Run B's
steady-state spatial failure.**

RVM receives only unique frames and recurrent state, not capture time. The
general refiner has no RVM stabilization because the model is assumed to be
temporally consistent. Same-size discontinuities do not reset state. Run B had
no restart, so reset bugs do not explain that still, but 15 Hz recurrence and
residual temporal shimmer remain unqualified.

Expected signature: raw alpha flicker changes when identical frames are replayed
at 15 versus 30 Hz, or after a synthetic generation gap/restart.

### H8 — MediaPipe timing and watershed remain fallback-specific risks

**Confidence: high for Run A investigation; inapplicable to Run B.**

MediaPipe advances a fixed 33 ms per processed frame and its generic policy can
run per-frame watershed plus adaptive EMA. These are still credible causes of
Run A contour instability and must be corrected because `auto` can select this
backend. They must not block the RVM/CUDA alpha, compositor, and cadence work.

Expected signature: same-source MediaPipe raw/refined comparisons identify
timing- or watershed-dependent contour motion that is absent from raw RVM.

### H9 — Downstream scaling/encoding adds a second artifact

**Confidence: low for the preview symptom, still worth isolating.**

The local preview already contains the defect, so the virtual-camera consumer
is unlikely to be the primary cause. It can still add ringing, chroma-related
edge errors, or another resize.

Expected signature: a processed-hub frame is stable while the virtual-camera
consumer recording is not.

## 5. Target quality contract

The exact numeric thresholds are provisional until `MATTE-0.2` records the
baseline, but implementation and review should use this contract:

- Static and slow-moving boundaries must not visibly oscillate.
- Annotated opaque foreground cores such as face, torso, clothing, and solid
  accessories must remain opaque: backdrop structure must not be recognizable
  through them.
- Exterior alpha mass, fringe/halo width, foreground holes, and false
  foreground components must stay within fixture-specific bounds.
- Real subject motion must not leave a persistent previous-frame silhouette.
- Soft hair and semi-transparent boundaries must remain soft; stabilization
  must not reduce the matte to a binary cutout.
- Temporal behavior must be based on elapsed capture time, not assumed FPS.
- A repeated output frame must not advance segmentation, matte temporal state,
  backdrop playback, or quality counters that represent unique input work.
- Camera restart, geometry change, source-generation change, and a qualified
  long timestamp gap must reset incompatible temporal state at the exact next
  input boundary.
- Backend-specific behavior must be explicit and observable. A UI control must
  not appear active when the selected backend ignores it.
- The pipeline must retain the canonical `float32 HxW [0,1]` alpha contract and
  canonical BGR frame contract.
- Local quality improvements must not weaken remote-mode privacy gates or cause
  a raw camera-derived fallback to be published.
- Debug evidence containing a person's pixels or silhouette must be opt-in,
  local, bounded, excluded from normal logs/status, and never committed as a
  default test fixture.
- A profile advertised for 30 unique FPS must complete its steady-state
  serialized new-frame path—from unique-frame dequeue through guard/validation
  and completed sink send—within `33.3 ms` at p95 on qualified hardware, or
  demonstrate at least 27 unique composites/s without growing latency. Output
  repeats do not satisfy this gate.
- Quality defaults must be changed only after CPU/GPU, 15/30/60 FPS, motion,
  restart, and downstream-consumer qualification.

## 6. Delivery plan

| ID | Task | Phase | Priority | Size | Depends on |
| --- | --- | ---: | ---: | ---: | --- |
| MATTE-0.1 | Add a privacy-aware raw/mask/composite replay bundle | 0 | P0 | M | — |
| MATTE-0.2 | Define metrics, fixtures, and baseline report | 0 | P0 | L | MATTE-0.1 |
| MATTE-0.5 | Diagnose RVM alpha integrity and opaque-core leakage | 0 | P0 | M | MATTE-0.1, MATTE-0.2 |
| MATTE-0.3 | Run the backend/postprocess/cadence ablation matrix | 0 | P0 | M | MATTE-0.2, MATTE-0.5 |
| MATTE-0.4 | Publish immediate operator mitigations | 0 | P0 | S | MATTE-0.3 |
| MATTE-1.1 | Carry capture sequence and timestamp with each frame | 1 | P0 | L | — |
| MATTE-1.2 | Add timestamp/reset semantics to the segmenter contract | 1 | P0 | L | MATTE-1.1 |
| MATTE-1.3 | Correct MediaPipe timing and soft-mask resampling | 1 | P1 | M | MATTE-1.2 |
| MATTE-1.4 | Reset RVM/refiner state on discontinuities | 1 | P0 | M | MATTE-1.2 |
| MATTE-1.5 | Preserve transactional hot-activation behavior | 1 | P0 | M | MATTE-1.2, MATTE-1.4 |
| MATTE-2.1 | Implement elapsed-time, motion-aware boundary stabilization | 2 | P0 if selected | XL | MATTE-0.3, MATTE-0.5, MATTE-1.2, MATTE-1.4, MATTE-1.5 |
| MATTE-2.2 | Make spatial edge refinement stable and resolution-aware | 2 | P1 | L | MATTE-0.3 |
| MATTE-2.3 | Introduce explicit backend-specific matte policies | 2 | P0 | M | MATTE-0.3, MATTE-0.5 |
| MATTE-2.4 | Bound dynamic light-wrap edge shimmer | 2 | P1 | M | MATTE-0.3 |
| MATTE-2.5 | Qualify RVM alpha/detail/performance profiles | 2 | P0 | L | MATTE-0.3, MATTE-0.5 |
| MATTE-3.1 | Diagnose and recover the 15 FPS capture path | 3 | P0 | L | — |
| MATTE-3.2 | Expose unique-frame cadence and mismatch health | 3 | P0 | M | MATTE-1.1 |
| MATTE-3.3 | Decide whether output-rate matte interpolation is warranted | 3 | P2 | M | MATTE-2.3, MATTE-2.5, MATTE-3.1, MATTE-3.4 |
| MATTE-3.4 | Profile and recover the 720p unique-frame/compositor budget | 3 | P0 | L | MATTE-0.3 |
| MATTE-4.1 | Report backend quality tier and selection/fallback reasons | 4 | P0 | M | — |
| MATTE-4.2 | Add honest quality presets and backend-aware controls | 4 | P1 | L | MATTE-2.3, MATTE-2.5, MATTE-4.1 |
| MATTE-4.3 | Add local matte diagnostic views and telemetry | 4 | P0 | L | MATTE-0.5, MATTE-1.1, MATTE-2.3, MATTE-3.2, MATTE-4.1 |
| MATTE-5.1 | Add deterministic temporal/unit regression tests | 5 | P0 | L | incremental with each selected implementation task |
| MATTE-5.2 | Add end-to-end visual qualification | 5 | P0 | L | MATTE-2.5, MATTE-3.2, MATTE-4.3, MATTE-5.1, plus selected MATTE-2.x changes |
| MATTE-5.3 | Qualify performance and platform behavior | 5 | P0 | L | MATTE-3.1, MATTE-3.4, MATTE-5.2 |
| MATTE-5.4 | Roll out defaults, migration, documentation, and rollback | 5 | P1 | M | MATTE-4.2, MATTE-5.3 |

For the current RVM/CUDA lane, the minimum useful investigation/decision slice
is `MATTE-0.1`, `0.2`, `0.5`, `0.3`, `2.3`, `2.5`, `3.1`, `3.2`, `3.4`, `4.1`,
`4.3`, and its prerequisite `1.1`. This list is not a release dependency
closure: a release candidate must add the full dependencies of every selected
change and all applicable Phase-5 gates. Timestamp/reset work remains required
before a stabilization/default rollout; MediaPipe-specific `MATTE-1.3` may
proceed in parallel and must not block raw-RVM diagnosis or compositor
recovery. Do not wait for optional output-rate interpolation to deliver the
core fix.

`MATTE-2.1` is conditional: it becomes P0 only if `MATTE-0.3`/`MATTE-0.5`
demonstrate residual temporal alpha instability that the selected model/policy
does not meet. It is not mandatory for a qualified model-only RVM policy.

The `REACT` delivery table and cross-dependencies are in Section 11. Reaction
implementation may proceed behind a disabled flag, but production enablement
is gated on `MATTE-3.4`/`MATTE-5.3`; it cannot consume unrecovered output
headroom or replace a matte acceptance gate.

---

## 7. Detailed tasks

### MATTE-0.1 — Add a privacy-aware raw/mask/composite replay bundle · M

**Goal:** make the reported artifact reproducible without a live camera or
virtual-camera consumer.

**Context:** Current tests use generated arrays and fake model outputs. They
prove shape, value range, and simple smoothing behavior, but they do not
reproduce a real face/head contour over time. The screenshot cannot reveal raw
mask behavior, and saving only the final composite makes alpha instability
impossible to distinguish from light wrap or backdrop motion.

**Deliverables:**

- Add an opt-in local diagnostic recorder that writes a bounded session bundle:
  - canonical raw camera frames;
  - raw backend masks;
  - refined masks;
  - backend clean-foreground output when available (RVM `fgr`);
  - the exact backdrop frame used for each unique composite;
  - final composites;
  - monotonic capture timestamp and sequence;
  - capture generation/geometry generation;
  - configured and effective matte/compositor controls, including resolved RVM
    ratio, `mask_shift`, model foreground, light wrap, and blend space;
  - a versioned extension point for optional post-base/final-output provenance;
    `REACT-0.2`, not this task, owns reaction preset digests and trigger/cancel
    timestamps when reaction evidence is requested;
  - per-stage and available compositor-substage timing;
  - backdrop frame identity/timestamp where applicable.
- Use a versioned, documented manifest and lossless masks (`.npy`/`.npz` or
  lossless 16-bit representation). Do not use JPEG for metric-authoritative
  alpha data.
- Add an offline replay command/module that bypasses live capture and feeds the
  recorded frames and timestamps through the selected segmentation/refinement/
  compositing path.
- Add a frozen-intermediate attribution mode that injects recorded raw/post-
  refiner alpha, RVM `fgr`, and exact backdrop without rerunning the model. It
  must allow compositor stages/features to be swapped while every upstream
  array remains fixed.
- Support a final-composite-only capture mode for downstream comparisons, but
  mark it insufficient for matte metrics.
- Bound recording by duration and size; use owner-only permissions and an
  explicit output directory.

**Implementation notes:**

- Keep diagnostic persistence outside normal `FrameHub` history. The hub is a
  latest-frame transport and must not become an unbounded recorder.
- Preserve exact unique input sequence. Do not synthesize the output repeats
  into the model-input track.
- Matte-authoritative recording has no optional post-base effects. A downstream
  evidence extension such as `REACT-0.2` must retain the guarded pre-effect
  base and final output as separate tracks and must never derive matte metrics
  from final-effect pixels.
- A mask can reveal a recognizable silhouette. Treat it as sensitive even when
  RGB pixels are absent.
- The first field clip should include 10–20 seconds each of:
  stationary pose, slow head turn, quick lateral motion, hand near the face,
  hair/ear detail, and a capture stall/restart if reproducible.

**Likely files:**

- new `src/custback/matte_diagnostics.py` or a similarly isolated module;
- `src/custback/pipeline.py`;
- CLI wiring in `src/custback/__main__.py`;
- tests in a new `tests/test_matte_diagnostics.py`;
- documentation under `docs/`.

**Acceptance criteria:**

- A recorded bundle replays the same unique input order and timing metadata.
- Raw/refined masks and optional RVM foreground remain numerically identical
  after round-trip storage.
- Frozen-intermediate replay reproduces the recorded reference composite within
  the documented byte-exact or color-contract tolerance before an attribution
  variant is applied.
- Recording stops at configured duration/size without affecting the live
  output contract.
- Recording is off by default and no raw pixels/masks appear in logs or public
  `/status`.
- Tests cover interrupted writes, malformed manifests, path handling, and
  owner-only permissions on supported platforms.

### MATTE-0.2 — Define metrics, fixtures, and baseline report · L

**Goal:** convert the reported border dance, opacity leakage, and halo into
metrics that distinguish jitter, ghosting, spatial/coverage error, color
shimmer, and cadence.

**Context:** A lower frame-to-frame mask difference is not automatically
better—freezing a mask scores well while being unusable during motion. Metrics
must compensate for source motion and evaluate both stationary and moving
segments.

**Deliverables:**

- Build an offline evaluator that produces per-frame and aggregate:
  - unique-input FPS and output repeat ratio;
  - base-composite update/reuse FPS, exact final-output repeats, and send FPS,
    with a typed extension seam for later post-base stages;
  - raw/refined alpha temporal absolute difference;
  - optical-flow- or registration-compensated alpha difference;
  - contour displacement using signed-distance fields, including p50/p95;
  - subject area drift during stationary segments;
  - soft-edge width and uncertain-pixel fraction (`0.05 < alpha < 0.95`);
  - annotated opaque-core alpha p05/p50, mean `1 - alpha`, and fraction below
    `0.95`/`0.90`;
  - annotated background alpha mass, exterior halo area/width by signed
    distance, holes, and unexpected connected components;
  - composite backdrop-leakage/correlation inside annotated opaque regions;
  - RVM clean-foreground RGB error in regions where it replaces source color;
  - motion trail/lag area behind the current contour;
  - edge-band RGB variation with alpha held/compensated;
  - optional alpha SAD/MSE/gradient error against ground-truth mattes;
  - segmentation/refinement/compositor-substage/full-frame p50/p95 time,
    allocation volume, and memory.
- Create generated deterministic clips for:
  - a static shape with noisy confidence;
  - one- and two-pixel alternating contour jitter;
  - translation/rotation with known ground truth;
  - fast motion and occlusion;
  - fine structures and semi-transparent edges;
  - opaque light/dark clothing, shoulders, headphones, glasses, and other solid
    accessories with annotated core/background trimaps;
  - deliberately under-opaque cores, foreground holes, and exterior halos;
  - a dynamic backdrop with constant alpha to isolate light wrap;
  - 15/30/60 FPS and irregular timestamps;
  - repeated output frames and sequence gaps.
- Add one consented/licensed local qualification clip set. Do not commit
  private user footage; CI can use generated/licensed fixtures while local
  evidence is referenced by manifest digest.
- Produce `docs/matte-quality-baseline.md` (or JSON + review Markdown) comparing
  the reported-style MediaPipe CPU configuration with the current RVM path on
  the same replayed source/timestamps.

**Provisional gates to ratify with the baseline:**

- On a stationary 720p boundary, p95 compensated contour displacement should
  be no more than `1.5 px` and at least `40%` lower than baseline.
- Stabilization must not increase fast-motion trail area by more than `10%`
  relative to the unstabilized backend, and no previous contour should remain
  visibly dominant for more than one unique input interval.
- Static subject-area drift should be no more than `1%` of the foreground area.
- On annotated opaque cores, ratify a p05 alpha and below-`0.95` area gate that
  rejects the Run-B-style visible backdrop leakage without classifying genuine
  hair/soft boundaries as core.
- Exterior halo mass/width and false-foreground alpha inside annotated
  background must not regress beyond fixture-specific bounds.
- Fine-detail/ground-truth alpha error must not regress beyond the agreed
  fixture tolerance.
- Dynamic light wrap must be evaluated separately from alpha stability. On the
  fixed-alpha/dynamic-backdrop fixture, a candidate presented as a shimmer fix
  should reduce p95 edge-band RGB variation by at least `30%` versus the current
  default without failing static-appearance gates; `MATTE-0.2` must ratify or
  explicitly amend this provisional threshold before implementation tuning.

These are starting release criteria, not constants to bury in production code.
If the baseline shows they are unrealistic, amend them in the report before
implementation tuning.

**Likely files:**

- new `tests/matte_quality_evidence.py` or `scripts/matte_quality.py`;
- generated fixture helpers under `tests/fixtures/`;
- new qualification report under `docs/`;
- CI/release manifest changes only after runtime is bounded.

**Acceptance criteria:**

- A frozen input/config produces deterministic metrics within documented
  platform tolerance.
- A deliberately frozen mask fails motion-lag gates despite low flicker.
- A deliberately jittered stationary mask fails contour gates.
- A constant mask with dynamic light wrap changes only the edge-color metric.
- A deliberately under-opaque but temporally stable matte fails opaque-core
  gates even though it passes jitter gates.
- Optional post-base metric namespaces cannot overwrite or relabel the
  camera/matte/base-update metrics; concrete reaction fixtures belong to
  `REACT-0.2`.
- The baseline report records hardware, dependency versions, backend/device,
  effective detail/resampling settings, input cadence, and config.

### MATTE-0.5 — Diagnose RVM alpha integrity and opaque-core leakage · M

**Goal:** locate the Run-B opacity/halo defect before adding temporal filters or
changing production alpha semantics.

**Context:** RVM returns raw `pha`; the RVM refiner normally applies only
`mask_shift`. The final screenshot shows what appears to be sharp backdrop
content through opaque clothing, but no raw alpha or clean foreground was
recorded. Model foreground and light wrap cannot increase alpha, although they
can amplify color contamination wherever alpha is already uncertain.

**Deliverables:**

- Record/replay at least one consented Run-B-style sequence with:
  - raw source, raw `pha`, post-`mask_shift` alpha, RVM `fgr`, exact backdrop,
    and final composite;
  - foreground-core/background/soft-boundary trimap annotations for torso,
    shoulders, head, headphones, and hair;
  - configured/effective RVM ratio and `mask_shift`.
- Produce per-region heatmaps and metrics from `MATTE-0.2`, including
  opaque-core deficit, foreground holes, exterior halo mass/width, and backdrop
  leakage in the final composite.
- Compare four attribution boundaries while holding source/alpha/backdrop fixed:
  1. raw `pha`;
  2. post-refiner alpha;
  3. direct full-frame RVM `fgr * pha + backdrop * (1 - pha)`;
  4. current edge-only RVM foreground plus light-wrap compositor.
- Determine whether the primary spatial failure is already present in `pha`,
  introduced by effective `mask_shift`, or only made more visible by `fgr`/
  light wrap. Record mixed cases rather than forcing one global label.
- If raw alpha fails, evaluate candidate classes without committing a default:
  - RVM ratio/profile or qualified model change;
  - source-quality/lighting/cadence improvement;
  - topology/confidence-aware opaque-core restoration or alpha calibration
    restricted away from genuine soft structures.
- Explicitly reject global hard thresholding as a generic RVM fix unless it
  passes hair, motion, halo, and ground-truth gates.

**Likely files:**

- diagnostic recorder/evaluator introduced by `MATTE-0.1`/`MATTE-0.2`;
- `src/custback/segmentation.py` for effective alpha-stage observability;
- `src/custback/pipeline.py`/`src/custback/compositor.py` for attribution views;
- focused generated trimap/alpha fixtures and a local qualification report.

**Acceptance criteria:**

- The report locates each observed defect at raw-alpha, refined-alpha,
  clean-foreground, wrap, or final-blend boundaries using the same frames.
- Opaque clothing/accessory failure cannot pass merely because its contour is
  temporally stable.
- Any proposed alpha correction preserves annotated hair/soft regions and does
  not introduce holes, cutout edges, or motion trails.
- No temporal "dancing" conclusion is drawn from a still image.
- The output is actionable: it selects the next profile/model/compositor or
  alpha-policy experiment and records why alternatives were rejected.

### MATTE-0.3 — Run the backend/postprocess/cadence ablation matrix · M

**Goal:** identify which changes materially reduce the reported defect before
replacing algorithms or changing defaults.

**Context:** The likely causes interact. For example, disabling light wrap can
make an edge look calmer without changing alpha; disabling watershed can help
one camera and harm a coarse mask on another. The experiment must retain one
variable at a time. This is a bounded exploratory screen on the Run-B
host/replay that shortlists candidates; it is not cross-platform profile
qualification.

**Matrix:**

- backend/device available on the evidence host:
  - active RVM/CUDA;
  - MediaPipe CPU for the fallback comparison;
  - one RVM CPU spot baseline if it completes within the experiment bound;
- RVM downsample: auto and a small qualified set such as `0.4`, `0.5`, `0.67`,
  and `1.0`, subject to memory/performance;
- RVM model: current MobileNet and any separately licensed/packaged candidate
  justified by `MATTE-0.5`;
- edge refine: on/off;
- mask blur: `0`, current default, and one wider candidate;
- temporal smoothing: `0`, current default, and candidate time-based policies;
- mask shift: `0`, both signs around zero, and current effective value;
- a full factorial for `use_model_foreground` off/on and `light_wrap` zero/
  current effective value, including plain, foreground-only, wrap-only, and
  both;
- blend space: current `srgb_legacy` and `linear_srgb`, with quality and runtime
  reported separately rather than assuming either is free;
- backdrop: constant image and the same container-timed 24 FPS video;
- capture cadence: the same native sequence at fixed 15/30/60, temporal
  decimation from 30 to 15, and irregular/gapped replay;
- output cadence: equal to unique input and 2x repeated output.

For every RVM variant, retain raw `pha`, post-refiner alpha, `fgr`, final
composite, warm-up versus steady-state measurements, and full stage timings.
Generic edge-refine/blur/EMA variants are MediaPipe-specific unless an explicit
RVM policy candidate enables them.

**Required decisions:**

- Confirm whether watershed improves or worsens MediaPipe temporal contour
  error on real clips.
- Quantify how much of the visible symptom is alpha motion versus edge-color
  motion.
- Locate opaque-core leakage and exterior halo at raw alpha, model foreground,
  light wrap, blend-space, and final-composite boundaries.
- Identify which RVM candidates are worth formal cross-device qualification.
- Attribute the default-feature compositor cost and determine which feature
  combinations are worth their measured quality/performance cost.
- Decide separate first fixes for the active RVM/CUDA lane and MediaPipe
  fallback lane.
- Record the effect of the 15-to-30 repeat cadence without claiming it creates
  new alpha values.

**Acceptance criteria:**

- Results use the same source frame sequence/timestamps across variants.
- Results do not treat the two reported screenshots as an A/B backend test.
- A report includes metrics, representative contact sheets/short clips,
  performance, and a bounded shortlist for `MATTE-2.5`/`MATTE-3.4`; it does
  not label one-host screening as a production preset.
- The recommendation names rejected candidates and the reason (jitter,
  ghosting, detail loss, performance, platform availability, or complexity).

### MATTE-0.4 — Publish immediate operator mitigations · S

**Goal:** give users a reversible quality improvement while engineering work is
in progress.

**Context:** Run B proves that simply enabling RVM/CUDA is not sufficient.
Reversible diagnostics can isolate RVM alpha, edge-color features, capture
cadence, and processing cost, but values must come from `MATTE-0.3`; do not
present guesses as universal fixes.

**Deliverables:**

- Add a troubleshooting section covering:
  - how to verify `segmentation_backend`, device, and measured capture FPS;
  - how to install/rebuild with the qualified RVM extra;
  - how to test MediaPipe GPU where supported;
  - how to test `light_wrap: 0` and `use_model_foreground: false`
    independently to distinguish raw alpha from edge-color processing;
  - how to test the qualified RVM ratio and `mask_shift`, including which
    changes rebuild/reset RVM and the exact rollback value;
  - how to test MediaPipe-only `edge_refine` safely when fallback is active;
  - how lighting/auto-exposure can reduce webcam cadence, without automatically
    writing camera controls;
  - how to separate capture-only under-rate from full-processing under-rate;
  - how to match output FPS to a known sustained capture rate for diagnosis,
    without presenting repeats or a lower output target as a quality fix.
- Include rollback values and warn which config changes rebuild resources.

**Acceptance criteria:**

- Every recommendation is supported by the ablation report.
- Each apply/rollback sequence is tested, states whether it rebuilds temporal
  resources, and tells the operator how to confirm the configured and effective
  value after activation.
- The guide does not imply that increasing generic `threshold` affects
  MediaPipe/RVM if the code still does not use it.
- No recommendation weakens remote privacy behavior.

### MATTE-1.1 — Carry capture sequence and timestamp with each frame · L

**Goal:** give every temporal consumer the identity and elapsed time of the
specific pixel array it processes.

**Context:** `CaptureHealth` intentionally describes the last frame returned by
`read`, but `read()` returns only an ndarray. Sampling health in a separate call
can race a newer capture-slot update and has no per-frame timestamp field. A
temporal algorithm must not infer time from output loop rate.

**Deliverables:**

- Introduce an immutable capture envelope, for example:

  ```python
  @dataclass(frozen=True)
  class CapturedFrame:
      pixels: np.ndarray
      sequence: int
      captured_at_ns: int
      generation: int
      geometry_generation: int
      content_rect: tuple[int, int, int, int]
  ```

  The final type/name may differ, but metadata and pixels must be atomically
  associated.
- Update real and synthetic capture sources, fakes, preflight, activation
  trials, and the steady-state loop.
- Define timestamp origin as monotonic process time at successful capture read
  completion unless a trustworthy device timestamp contract is later added.
- Preserve latest-frame semantics and `None` for no unread frame.
- Count source sequence gaps caused by overwritten capture slots.
- Keep raw `FrameHub` publication as pixels unless the API explicitly needs
  metadata; do not leak internal timing objects into existing wire contracts.

**Implementation notes:**

- A pixel-identical successful capture is still a new unique input with a new
  sequence/timestamp.
- An output repeat is not a new captured frame.
- Use integer nanoseconds internally to avoid float ordering errors.
- Capture generation and geometry generation must refer to the same frame as
  the pixels, not to a newer slot.

**Likely files:**

- `src/custback/capture.py`;
- `src/custback/pipeline.py`;
- synthetic/fake capture implementations;
- `tests/test_capture.py`, `tests/test_pipeline.py`,
  `tests/test_canonical_canvas.py`.

**Acceptance criteria:**

- Sequence is strictly increasing within a capture instance.
- Timestamp is monotonic within a generation and atomically matches pixels.
- Slot overwrites create an observable sequence gap/drop but never reorder
  frames.
- Preflight-to-loop handoff does not process the same captured sequence twice
  unless it is explicitly the already-sent startup frame.
- Existing public frame/output contracts remain compatible.

### MATTE-1.2 — Add timestamp/reset semantics to the segmenter contract · L

**Goal:** make temporal behavior a first-class, backend-independent contract.

**Context:** The current `segment(frame)` API cannot express real elapsed time
or discontinuity. RVM has hidden recurrent state, MediaPipe has a hidden
timestamp counter, and `MaskRefiner` has hidden previous-alpha state.

**Deliverables:**

- Extend segmentation input with capture timestamp/sequence, either through a
  typed input object or explicit keyword arguments.
- Add a documented `reset(reason, timestamp)`/`reset_temporal_state` contract to
  `Segmenter` and `MaskRefiner`.
- Define reset reasons at minimum:
  - initial/startup;
  - capture generation change;
  - geometry generation/shape change;
  - non-monotonic timestamp;
  - timestamp gap above the qualified limit;
  - backend/provider recovery;
  - committed segmentation config/resource generation change.
- Define whether a reset happens before processing the boundary frame (target:
  yes) and expose the last reset reason/count.
- Define sequence-gap semantics explicitly: a newest-slot overwrite is observed
  and counted, but a gap alone does not reset state when timestamp delta and
  generation remain within policy; generation change, non-monotonic time, or a
  qualified long elapsed gap does reset before the boundary frame.
- Keep stateless segmenters as no-op implementations.

**Acceptance criteria:**

- Segmenters never receive a non-monotonic effective timestamp.
- The exact first frame after a discontinuity starts from clean temporal state.
- Repeat output does not call `segment`, `refine`, or `reset`.
- An ordinary sequence gap inside the elapsed-time limit is recorded without a
  reset; a gap accompanied by a qualified timestamp/generation discontinuity
  resets exactly once.
- Tests cover pixel-identical frames, sequence gaps, long gaps, camera restart
  at the same resolution, resize, and provider fallback.

### MATTE-1.3 — Correct MediaPipe timing and soft-mask resampling · M

**Goal:** make MediaPipe VIDEO-mode inference represent actual capture time and
freeze its alpha-resize behavior.

**Context:** `MediaPipeSegmenter` currently adds exactly 33 ms for every
processed frame. In the reported run, processed frames arrive at roughly 67 ms
intervals. Capture stalls are also compressed into a single 33 ms step.

**Deliverables:**

- Convert monotonic capture nanoseconds to strictly increasing integer
  milliseconds relative to a segmenter-local epoch.
- Handle two frames that quantize to the same millisecond by advancing the
  latter minimally while retaining observability of the adjustment.
- Recreate/reset the MediaPipe task if its API cannot safely continue after a
  timestamp discontinuity.
- Validate confidence-mask count, dtype, finiteness, and dimensions before use.
- If resize is necessary, specify soft-mask interpolation explicitly (initial
  candidate: linear for upsampling, area for downsampling) and re-clamp to
  `[0,1]`; ratify with `MATTE-0.3`.
- Record input/output mask dimensions and effective timestamp delta in local
  diagnostic telemetry.

**Likely files:**

- `src/custback/segmentation.py`;
- `tests/test_processing.py`;
- a MediaPipe fake in a focused test module.

**Acceptance criteria:**

- Replaying 15/30/60 FPS or irregular timestamps yields corresponding
  MediaPipe timestamp deltas.
- MediaPipe never receives duplicate/decreasing timestamps.
- A long gap or generation change follows the reset contract.
- Resize tests use a nonuniform soft-alpha pattern and freeze interpolation,
  dtype, range, and contiguity.

### MATTE-1.4 — Reset RVM/refiner state on discontinuities · M

**Goal:** prevent stale temporal state from crossing unrelated capture
sequences.

**Context:** RVM resets recurrent tensors on resolution change and GPU recovery,
but not when a camera restarts at the same shape. `MaskRefiner` resets only when
the object is replaced or explicitly called. Run B had zero restarts and one
stable generation, so this task is a correctness prerequisite for temporal
rollout rather than an explanation of that sample's steady spatial defect.

**Deliverables:**

- Reset all four RVM recurrent tensors, `last_foreground`, and any associated
  size/time state on the reasons defined in `MATTE-1.2`.
- Reset `MaskRefiner._prev` at the same exact frame boundary.
- Add a qualified timestamp-gap reset threshold or policy. It should be long
  enough not to reset during ordinary jitter but short enough not to blend
  across a stall.
- Expose reset count and last reason without exposing frame content.

**Acceptance criteria:**

- A same-resolution camera generation change feeds zero recurrent state to RVM.
- The first refined mask after reset does not blend with the prior generation.
- Provider fallback performs one clean retry and records one reset reason.
- Ordinary irregular 15 FPS cadence does not continuously reset state.

### MATTE-1.5 — Preserve transactional hot-activation behavior · M

**Goal:** ensure new temporal state cannot be advanced, leaked, or partially
committed during config trials.

**Context:** Pipeline resource changes are staged and trialed before commit.
Temporal state must remain generation-owned. A failed candidate trial must not
alter the live RVM/refiner state; a successful segmentation-policy change must
not accidentally reuse state created under different settings.

**Deliverables:**

- Include segmenter/refiner policy and temporal-state ownership in staged
  `_Resources`/activation.
- Trial a candidate only with candidate-owned state.
- On commit, either promote the trial state with the exact trial input identity
  or deliberately reset and process the next unique input; document the choice.
- On rollback/failure, close candidate resources and preserve live state.
- Ensure changes to unrelated background presentation do not reset matte state
  unless the segmentation input geometry/source changes.

**Acceptance criteria:**

- Failed hot patches leave live recurrent/refiner state byte-for-byte or
  behaviorally unchanged.
- Successful segmentation changes do not blend across policy generations.
- Background-only hot patches do not needlessly recreate the segmenter.
- Tests cover timeout, candidate failure, commit, rollback, and deferred close.

### MATTE-2.1 — Implement elapsed-time, motion-aware boundary stabilization · XL

**Goal:** reduce stationary/slow contour jitter without freezing real motion or
destroying soft alpha.

**Context:** The existing filter is an alpha-domain EMA:

```text
keep = smoothing * clamp(1 - 4 * blurred_abs_difference, 0, 1)
output = keep * previous + (1 - keep) * current
```

It is frame-count-based and loses hold where alpha changes. It has no
correspondence between a previous moving edge and its current position.
The RVM lane may be developed once the common timestamp/reset contract is
available; enabling the same policy for MediaPipe additionally requires
`MATTE-1.3`.

**Required design spike:**

Evaluate at least:

1. previous-alpha warping using bounded low-resolution optical flow from the
   previous/current camera frames;
2. confidence-gated temporal blending in only a narrow uncertain boundary band;
3. asymmetric attack/release or alpha hysteresis for small stationary changes;
4. a simpler registered temporal median/robust filter as a fallback.

Do not add a full-frame heavyweight optical-flow dependency without showing
that the bounded OpenCV implementation meets the performance gate.

**Target algorithm properties:**

- Convert a user-facing time constant to per-frame weight using real `dt`, for
  example `1 - exp(-dt/tau)`, rather than assuming 30 FPS.
- Align previous alpha to current source motion before blending.
- Use source/mask confidence to reject bad correspondence and fall back to the
  current matte during occlusion, scene discontinuity, or fast motion.
- Restrict stabilization work to a dilated boundary region/downscaled guide
  where possible.
- Preserve exact or near-exact alpha endpoints outside the boundary band.
- Prevent long trails with a bounded maximum hold and reset rules.
- Return finite contiguous `float32 [0,1]`.

**Configuration:**

- Prefer a policy/preset plus a small number of meaningful parameters
  (`mode`, time constant, maximum motion) over exposing every internal
  threshold.
- Preserve the old `temporal_smoothing` field through a compatibility mapping
  or schema migration; do not silently reinterpret persisted values without a
  versioned decision.

**Likely files:**

- `src/custback/segmentation.py` or new `src/custback/matte.py`;
- `src/custback/config.py` and both default YAML files;
- `src/custback/pipeline.py`;
- focused temporal tests and qualification harness.

**Acceptance criteria:**

- Meets ratified stationary contour and area-drift gates at 15/30/60 FPS.
- Equivalent elapsed time produces comparable smoothing across frame rates.
- Translation/rotation fixtures follow motion without a persistent double edge.
- Occlusion and fast-motion fixtures fall back without catastrophic mask
  tearing.
- Runtime is bounded and included in `segmentation_ms` or a new explicit
  `matte_refinement_ms`.
- RVM and MediaPipe can select different strengths without hidden behavior.
- A lower jitter score cannot pass if opaque-core alpha deficit, exterior halo,
  or backdrop leakage regresses; the stabilizer must not freeze a bad RVM matte.

### MATTE-2.2 — Make spatial edge refinement stable and resolution-aware · L

**Goal:** retain useful boundary snapping without letting per-frame image noise
select a different contour.

**Context:** `_watershed_edge_snap` uses a fixed eight-pixel uncertainty band
and writes the watershed boundary as `0.5` before Gaussian blur. It can improve
a displaced synthetic step edge, but existing tests are single-frame and do not
measure temporal stability, fine structures, or resolution scaling.

**Deliverables:**

- Use `MATTE-0.3` to decide whether watershed remains, is gated to specific
  backends, or is replaced by a guided/joint-bilateral edge refinement.
- Make the search radius resolution-aware with an explicit min/max, referenced
  to the canonical canvas or model-mask scale.
- Gate snapping on local gradient/contrast and mask uncertainty; do not chase a
  weak or ambiguous edge.
- Stabilize selected edge support over time or apply spatial refinement after
  motion-aware alignment in the qualified order.
- Preserve thin foreground/background components and soft matte values.
- Define deterministic behavior on MJPEG block noise and uniform guides.

**Acceptance criteria:**

- Does not increase stationary p95 contour displacement on qualification clips.
- Improves spatial error on displaced-edge fixtures.
- Does not delete qualified fine structures or turn the entire soft band into a
  binary contour.
- Behavior scales consistently between 640x360, 1280x720, and 1920x1080.
- Tests cover temporally alternating nearby gradients, compression-like blocks,
  low contrast, thin hair-like components, and camera noise.

### MATTE-2.3 — Introduce explicit backend-specific matte policies · M

**Goal:** make postprocessing intentional and observable for each backend.

**Context:** `refiner_for` silently turns off blur, edge refine, and temporal
smoothing for RVM. MediaPipe uses all generic defaults. The same persisted
configuration therefore does not mean the same runtime behavior.

**Deliverables:**

- Own one typed runtime effective-policy resolver/snapshot. This task defines
  applicability and configured-versus-effective semantics; it does not create
  a competing public status schema.
- Define effective policy for:
  - true-alpha recurrent mattes (RVM);
  - confidence-mask video segmentation (MediaPipe);
  - binary/coarse fallback masks (heuristic);
  - null/passthrough.
- Store configured and effective values separately in that internal snapshot.
  `MATTE-4.1` owns public status/OpenAPI/ready/overlay transport of the snapshot.
- For RVM, explicitly define the policy for raw alpha, `mask_shift`,
  opaque-core/halo handling, model foreground, light wrap, and any residual
  stabilization. Do not hide alpha calibration inside a generic threshold.
- Decide from evidence whether RVM needs a very light residual temporal policy
  or should remain model-only.
- Remove or relabel controls that do not apply to an active backend.
- Make `threshold` either useful for a specifically documented alpha remap/
  hysteresis policy or explicitly heuristic-only. Do not hard-threshold RVM
  hair mattes as a shortcut.

**Acceptance criteria:**

- Unit/integration tests prove the effective-policy snapshot for every backend
  and config combination; public visibility is the `MATTE-4.1` acceptance gate.
- Status reports the resolved RVM ratio and whether blur, edge refine, generic
  smoothing, threshold, mask shift, model foreground, and light wrap are
  effective, bypassed, or inapplicable.
- UI/API documentation identifies backend applicability.
- No quality control is silently ignored without an effective-state indication.
- Backend switch transactionally selects a fresh compatible policy/state.

### MATTE-2.4 — Bound dynamic light-wrap edge shimmer · M

**Goal:** prevent a moving video backdrop from making the subject edge flicker
or pulse independently of the matte.

**Context:** Current light wrap uses the current blurred backdrop inside
`4 * alpha * (1 - alpha)`. With a video backdrop, edge color can change on each
unique composite even when the subject and alpha are static. In Run B the same
band can first receive RVM clean foreground; wrap must be evaluated both alone
and in interaction with that substitution and broad low-alpha regions.

**Deliverables:**

- Measure current edge-band RGB variance with a fixed alpha/foreground and
  dynamic backdrop.
- Re-run the four `use_model_foreground × light_wrap` combinations on:
  - a correct narrow soft edge;
  - a deliberately broad under-opaque band;
  - fixed and moving backdrops.
- Evaluate:
  - a lower video-mode default;
  - temporal low-pass of only the wrap sample using actual backdrop time;
  - luminance/chroma bounds per update;
  - restricting wrap farther inside the foreground edge;
  - disabling wrap when backdrop motion/luminance change is excessive.
- Keep light-wrap temporal state generation-owned and reset it on backdrop
  change/seek/scene cut.
- Preserve linear-light behavior when `blend_space: linear_srgb`.

**Acceptance criteria:**

- Dynamic-backdrop edge-color variation meets the ratified gate without
  changing alpha metrics.
- Static backdrop appearance is not materially regressed.
- Video scene cuts do not smear the old scene color around the subject.
- `light_wrap: 0` remains an exact bypass with no temporal state advancement.

### MATTE-2.5 — Qualify RVM alpha/detail/performance profiles · L

**Goal:** make the preferred true-alpha backend a practical option at a
sustainable cadence without accepting opaque-core leakage or an excessive halo.

**Context:** At 1280x720, auto RVM uses a long-edge internal resolution near
512 pixels (`rvm_downsample ~= 0.4`). Increasing the ratio may improve fine
edges/coverage but costs inference time/memory. Run B proves CUDA is active and
measures `26.4 ms` for the combined segmentation stage, yet visible defects
remain and the narrower frame-processing EWMA is `61.3 ms` before later
guard/send work. RVM availability is therefore solved for this host; alpha
quality and end-to-end sustainability are not.

**Deliverables:**

- Take only the ratios/models/policies shortlisted by `MATTE-0.3`, plus auto
  and the current compatibility baseline, into the formal matrix.
- Benchmark that shortlist on RVM CPU and every supported accelerator at target
  canvas sizes.
- Qualify the current MobileNet model and only shortlisted alternative RVM
  model variants whose license, packaging, download integrity, startup, and
  performance contracts can be supported.
- Run the same native 30 FPS sequence at native cadence and temporally decimated
  15 FPS to measure recurrent-cadence sensitivity.
- Report:
  - opaque-core, halo, fine-detail, and temporal quality metrics;
  - warm-up and steady-state inference/preprocess/postprocess p50/p95;
  - full-frame p50/p95;
  - memory/VRAM;
  - sustained capture and output cadence;
  - provider fallback behavior.
- Report results with model foreground/light wrap disabled for raw-model
  attribution and with the qualified compositor policy for user-visible cost.
- Define `performance`, `balanced`, and `quality` RVM profiles only if the data
  shows stable cross-device meaning.
- Expose the effective resolved downsample ratio, not only configured `0=auto`.
- Do not make an unsustainable high detail ratio the global default.

**Acceptance criteria:**

- Each recommended profile has explicit hardware/canvas qualification.
- Auto resolution is observable and covered by tests.
- A profile fails if annotated opaque-core/background/halo gates fail even when
  its temporal contour score is good.
- A profile that misses its declared frame budget is rejected or clearly
  degraded; it is not advertised as real-time.

### MATTE-3.1 — Diagnose and recover the 15 FPS capture path · L

**Goal:** restore the requested unique-frame cadence where the camera and system
can actually supply it, and give a precise reason when they cannot.

**Context:** The screenshot shows a negotiated/reported 30 FPS MJPG mode but a
measured 15 FPS stream. Run B makes the observation stronger:
`capture_read_ms=66.6`, zero read failures, zero restarts, and stable generation
describe a recent health window near 15 Hz without recovery churn. The log does
not prove the full-run distribution. Repeating output masks the sink contract
but cannot restore motion or matte updates. Full processing has a separately
measured 61.3 ms constraint handled by `MATTE-3.4`; do not conflate the two.

**Diagnostic matrix:**

- true capture-only: a direct `cap.read()`/normalization harness, or
  `background.mode: passthrough` + null sink + preview/API diagnostics off,
  verified not to run segmentation, backdrop, or compositor work;
- MediaPipe CPU/GPU;
- RVM CPU/GPU;
- local preview off/on;
- virtual camera off/on;
- 640x360, 1280x720, and one supported higher mode;
- MJPG versus backend default where supported;
- adequate lighting versus the reported environment;
- camera auto-exposure versus a read-only observed control report;
- same device through a native capture tool for comparison.

The direct harness is sufficient for this task's required timestamp evidence;
production envelope/status correlation from `MATTE-1.1`/`MATTE-3.2` is
corroborating evidence and may land in parallel rather than a hard dependency.

**Deliverables:**

- Correlate capture timestamps with `read_ms`, CPU utilization, per-stage
  timing, drops, USB/backend logs, and camera controls.
- Compare native capture-tool timestamps with custback capture-only timestamps
  before attributing the 66.6 ms interval to model/compositor load.
- Distinguish:
  - driver-reported FPS mismatch;
  - low-light exposure throttling;
  - device/USB bandwidth;
  - decode/normalization cost;
  - CPU contention/starvation;
  - output/backend blocking.
- If contention is proven, evaluate process/thread priority/affinity only with
  cross-platform evidence; first prefer reducing work or qualified acceleration.
- If camera controls are needed, follow the existing preserve-only camera
  control contract. A qualified explicit lock/manual policy is separate from
  continuous automatic writes.
- Improve mode selection or actionable diagnostics for known backend cases.

**Acceptance criteria:**

- Capture-only sustains at least 27 unique reads/s for a hardware-verified
  1280x720@30 mode, or reports a specific actionable device/exposure/backend
  limitation with corroborating evidence.
- Capture-only measurements show whether segmentation is causal.
- The report states capture pacing and processed-frame pacing independently;
  fixing one is not recorded as fixing the other.
- No fix creates an uncontrolled feedback loop with camera auto-exposure/
  white balance.
- Regression tests retain bounded capture worker shutdown/recovery behavior.

### MATTE-3.2 — Expose unique-frame cadence and mismatch health · M

**Goal:** make it obvious that output FPS and visual-update FPS are different.

**Context:** The current overlay shows input and output FPS and a general
`CAPTURE BELOW TARGET` warning, but it does not state repeat ratio,
segmentation-update FPS, or the temporal impact of an input/output mismatch.
Optional post-base stages can later introduce another clock: final pixels may
change at output cadence while the safe base and matte remain reused. This task
owns base/reuse/send truth and a typed extension seam; each optional stage owns
its own counters and tests.

**Deliverables:**

- Add stable status fields for:
  - capture sequence and sequence gaps;
  - unique base-composite and segmentation FPS;
  - no-unread-frame safe-base reuse FPS/ratio;
  - exact final-output repeat FPS/ratio;
  - output send FPS;
  - last unique-frame age;
  - actual capture timestamp delta p50/p95;
  - output inter-send delta/jitter p50/p95 and new-composite inter-arrival
    p50/p95;
  - processing deadline misses and capture-slot overwritten/dropped frames in
    ready/shutdown evidence, not only live status;
  - output-sink recovery/pacing events separately from application repeats;
  - matte reset count/last reason;
- Consume the backend/effective-policy snapshot owned by
  `MATTE-2.3`/`MATTE-4.1` when correlating cadence; do not redefine its fields
  or applicability semantics here.
- Define a stable timing-field schema into which `MATTE-2.5`/`MATTE-3.4` can
  publish RVM and compositor substages without this task owning those profilers.
- Define a namespaced post-base provenance extension so `REACT-4.1` can add
  reaction-update/reaction-only counters without changing the meaning of
  capture, segmentation, base-composite, reuse, or send fields.
- Document each timer boundary. In particular, distinguish the current
  processing-only scope from complete serialized new-frame loop time including
  post-composite guard/validation, sink submission/copy, deliberate pacing
  wait, and schedule lateness.
- Add a warning such as `VISUAL UPDATES 15 FPS; OUTPUT REPEATS TO 30 FPS`.
- Preserve API/OpenAPI type consistency and sanitized/path-free status.

**Acceptance criteria:**

- Run B would clearly show 541 unique visual updates, 1105 output sends, 564
  no-unread safe-base reuses (`51.0%`), plus separate capture sequence-gap/
  overwrite, processing-deadline, and sink-pacing counters.
- A downstream namespace cannot mutate or relabel base/matte rates and reuse
  counters; active-reaction counter behavior is tested by `REACT-4.1`.
- Successful pixel-identical captures count as unique inputs; synthesized
  repeats do not.
- Status is bounded and does not expose raw timestamps that identify recording
  wall time or contain frame-derived data.

### MATTE-3.3 — Decide whether output-rate matte interpolation is warranted · M

**Goal:** determine whether a 15 FPS camera can produce acceptably smooth 30 FPS
output after capture and matte fixes.

**Context:** The current repeat policy is safe and deterministic. Interpolating
new RGB/alpha frames would add latency, optical-flow artifacts, and privacy
complexity. It should not be the first response to a camera that ought to
deliver 30 FPS.

**Decision work:**

- Compare:
  - repeats;
  - low-latency motion-compensated composite interpolation;
  - backdrop-only advancement while holding the subject;
  - matching output FPS to sustained input FPS.
- Measure latency, occlusion artifacts, edge quality, and CPU/GPU cost.
- Audit remote privacy and raw-frame echo guards before any synthesized local
  frame could share implementation with remote mode.

**Acceptance criteria:**

- Produce an ADR-style accept/reject decision.
- Default outcome should remain repeat unless qualification shows a clear
  benefit inside the latency/performance/privacy budgets.
- If accepted, interpolated frames are explicitly counted and never advance
  capture/model temporal state as if they were observations.

### MATTE-3.4 — Profile and recover the 720p unique-frame/compositor budget · L

**Goal:** make the advertised balanced 1280x720@30 RVM/CUDA path capable of at
least 27 unique composites/s, or honestly classify it as a lower-rate profile.

**Context:** Capture and processing are separately observed under target and
may share contention. Run B's `61.3 ms` frame-processing EWMA corresponds to
about 16.3 unique FPS before later guard/send work; `32.7 ms` compositing alone
consumes nearly the complete 30 FPS budget and is slower than the `26.4 ms`
combined CUDA RVM stage. Background selection is only `1.9 ms`, and serial
output send is another `2.9 ms`. The current legacy compositor performs
multiple full-frame float32 casts/allocations for edge-band calculation, RVM
foreground substitution, backdrop blur/wrap, and final alpha blend.

**Deliverables:**

- Reproduce Run B with a fixed 1280x720 30 FPS replay so capture pacing cannot
  hide processing capacity.
- Add bounded diagnostic substage timing/allocation measurements for:
  - input/mask validation;
  - edge-band calculation;
  - RVM foreground replacement;
  - backdrop downscale/blur/upscale;
  - light-wrap interpolation;
  - final alpha blend/conversion;
  - compositor-internal and post-composite guard/output validation;
  - sink preparation/submission/copy;
  - deliberate output pacing wait/sleep;
  - output schedule lateness/deadline misses.
- Benchmark plain blend, model-foreground only, light-wrap only, and both under
  `srgb_legacy`; separately benchmark `linear_srgb`. Record p50/p95/p99, memory
  bandwidth/allocation volume, and output equivalence.
- Profile RVM conversion/validation around ONNX inference so GPU time is not
  conflated with CPU pre/postprocessing.
- Evaluate, in evidence-backed order:
  - compiled/fused OpenCV operations instead of chained Python/NumPy
    temporaries;
  - reusable generation-owned work buffers with bounded lifetime;
  - uncertain-band ROI work when it preserves exact behavior;
  - reuse/caching only when backdrop/frame identity makes it semantically exact;
  - a qualified feature-default change only if its visual value does not
    justify its cost;
  - pipeline overlap only if local reductions are insufficient and recurrent
    ordering, latency, shutdown, and privacy remain bounded.
- Preserve `srgb_legacy` compatibility and exact alpha endpoints. A faster path
  must pass byte-exact tests where current contracts require them and
  tolerance-based tests only where the documented color contract permits.
- Preserve an explicit timing/ownership seam after the privacy-guarded safe
  base and before final output-tick rendering. `MATTE-3.4` qualifies this seam
  with reactions disabled; `REACT-5.2` later measures zero/one/maximum active
  reaction profiles without hiding base-compositor cost.
- Report the p95 compute headroom remaining before the target presentation
  interval. A base path that merely reaches `33.3 ms` has no budget for an
  optional reaction stage and must not be described as reaction-ready.
- Report no-unread output repeats, capture-slot overwrites/sequence gaps,
  processing deadline misses, and sink recovery/pacing as separate event
  classes. Correlate them without claiming that an overwrite or deadline miss
  directly caused an application repeat.

**Likely files:**

- `src/custback/compositor.py`;
- `src/custback/pipeline.py`;
- `src/custback/hub.py` and shutdown/status logging;
- `tests/test_processing.py`, `tests/test_pipeline.py`, and a non-blocking local
  performance evidence harness.

**Acceptance criteria:**

- On qualified RVM/CUDA hardware with a fixed 30 FPS replay, the balanced
  1280x720 compute/service path is at or below `33.3 ms` p95 from unique-frame
  dequeue through guard, validation, and sink submission/copy—excluding
  intentional output pacing sleep—or sustains at least 27 unique composites/s
  without growing queue age. Report that scope, the deliberate pacing wait and
  schedule lateness separately, and the existing narrower
  `frame_processing_ms`.
- The optimized compositor has a ratified sub-budget and cannot consume the
  entire 33.3 ms frame allowance.
- The evidence reports residual p95 headroom; reaction qualification must use
  the measured residual rather than assuming the full provisional reaction
  budget fits.
- All four feature combinations and both blend-space contracts pass visual,
  endpoint, dtype/range/contiguity, and deterministic regression tests.
- Work buffers/state are bounded and close cleanly on rebuild, failure,
  rollback, and shutdown.
- If the gate is not achievable on a hardware tier, status/presets declare its
  sustainable unique-frame rate; near-30 output repeats are not called 30 FPS
  processing.

### MATTE-4.1 — Report backend quality tier and selection/fallback reasons · M

**Goal:** tell operators when `auto` selected a lower-quality backend and how to
obtain the preferred one.

**Context:** `create_segmenter` logs why RVM is unavailable, but status marks a
fallback only when the final backend is heuristic. Run A selected MediaPipe
without clearly preserving the RVM-unavailable quality downgrade. Run B
correctly logs RVM/CUDA activation, but neither ready nor overlay reports the
effective RVM ratio and postprocess/compositor policy needed to reproduce
quality.

**Deliverables:**

- Own the public backend-selection/effective-policy status schema and its
  OpenAPI, ready-log, shutdown, and preview representation. Consume the policy
  snapshot from `MATTE-2.3` when available.
- Return/store structured selection attempts:
  - requested backend;
  - selected backend;
  - quality tier (`matting`, `segmentation`, `heuristic`, `none`);
  - availability/preparation result for each preferred candidate;
  - sanitized fallback reason/category;
  - active device/provider;
  - configured/effective backend-specific quality controls.
- Distinguish an expected explicit `backend: mediapipe` choice from `auto`
  falling from RVM to MediaPipe.
- Add status/overlay/doctor guidance such as `RVM unavailable: runtime not
  installed` without exposing model paths or raw exception details.
- Review installer defaults/profiles. If RVM is not installed by default on a
  platform, make the quality tradeoff explicit rather than calling MediaPipe
  the unqualified best result.

**Likely files:**

- `src/custback/segmentation.py`;
- pipeline resource/status models;
- `src/custback/diagnostics.py`, `src/custback/preview.py`;
- installer/package docs and tests where selection policy changes.

**Acceptance criteria:**

- Run A would state that `auto` selected MediaPipe CPU because RVM was
  unavailable, with an actionable sanitized reason.
- Run B would state successful RVM/CUDA selection with no fallback warning and
  would report the resolved ratio plus effective RVM/compositor policy.
- Explicit MediaPipe does not produce a misleading fallback warning.
- `gpu_required` still fails rather than degrading.
- Public status/OpenAPI and preview tests cover every quality tier.

### MATTE-4.2 — Add honest quality presets and backend-aware controls · L

**Goal:** make good configurations discoverable without requiring users to
understand every mask operator.

**Context:** The quality panel currently exposes low-level fields. Some do
nothing for RVM, `threshold` is heuristic-only, and aggressive combinations can
trade jitter for ghosting/halos.

**Deliverables:**

- Add evidence-backed `performance`, `balanced`, and `quality` presets if
  `MATTE-0.3`/`MATTE-2.5` support them.
- Show active backend quality tier, device, unique update FPS, and effective
  policy near the controls.
- Disable/hide with explanation:
  - heuristic-only threshold for MediaPipe/RVM;
  - RVM-inapplicable edge refine/blur if still bypassed;
  - MediaPipe delegate for backends that do not use it;
  - model foreground when the backend does not provide one.
- Label controls in visual terms and show whether a change rebuilds the
  segmenter/resets temporal state.
- Show both configured and effective values when a backend overrides a generic
  field; never use `CONFIG v0` as a substitute for runtime settings.
- Keep advanced fields available for diagnosis, but avoid presenting every
  internal algorithm constant.
- Make preset expansion a concrete config patch whose effective values are
  visible and versionable.

**Acceptance criteria:**

- Changing a visible enabled control changes the active effective policy.
- Preset switch is transactional and rolls back on activation failure.
- Accessibility, mobile layout, config merge/reset semantics, and WebUI tests
  remain covered.

### MATTE-4.3 — Add local matte diagnostic views and telemetry · L

**Goal:** let a developer/operator tell where instability enters the pipeline.

**Context:** The normal preview displays only the final composite. Diagnosing
raw model noise versus postprocess noise currently requires code changes.

**Deliverables:**

- Consume the policy snapshot/public schema from `MATTE-2.3`/`MATTE-4.1` and
  cadence/timing fields from `MATTE-3.2`; do not redefine their semantics or
  add duplicate public fields.
- Add local-only preview modes for:
  - raw camera;
  - raw model alpha;
  - refined/stabilized alpha;
  - RVM clean foreground;
  - exact backdrop frame;
  - alpha over source;
  - uncertain boundary band;
  - opaque-core deficit, foreground-hole, and exterior-halo heatmaps;
  - model-foreground-only, light-wrap-only, and final-composite contribution
    views;
  - frame-to-frame/flow-compensated instability heatmap.
- Show unique input sequence/delta, raw/refined temporal metrics, reset reason,
  effective RVM/compositor controls, and stage/substage timings.
- Never send these views to the configured virtual-camera output unless an
  explicit diagnostic sink is selected.
- Do not expose mask images through unauthenticated/public endpoints.
- Keep overlay drawing on copies, consistent with current preview ownership.
- Keep matte diagnostic evidence pre-reaction by default. A separate explicit
  final-output view may show reactions, but an effect must never cover or alter
  raw-alpha, mask, contribution, or heatmap evidence.

**Acceptance criteria:**

- A replay can locate Run-B-style leakage in raw/refined alpha versus RVM
  foreground, wrap, or final blend, and can separate alpha motion from edge
  color motion.
- Enabling diagnostics does not mutate the production frame or mask.
- Normal mode pays no material full-frame diagnostic cost.
- Tests assert no diagnostic image reaches normal output/hub publications.

### MATTE-5.1 — Add deterministic temporal unit/regression tests · L

**Goal:** prevent quality fixes from regressing across refactors.

**Delivery rule:** This is an incremental umbrella gate, not a reason to defer
tests to one late PR. Every selected implementation task lands focused tests
with its code; final completion covers all tasks selected for the release.
Tests for `MATTE-2.1`/`2.2` are required only if those optional algorithms ship.

**Required test groups:**

- timestamp conversion at 15/30/60 FPS, same-ms quantization, irregular gaps,
  and non-monotonic input;
- reset on generation, geometry, provider fallback, long gap, and config
  commit;
- no reset on ordinary cadence jitter or output repeats;
- MediaPipe mask validation and interpolation;
- RVM recurrent state on same-size restart;
- RVM raw/post-shift alpha attribution and effective ratio/policy reporting;
- opaque-core deficit, hole, exterior-halo, and backdrop-leakage metric
  fixtures;
- temporal stabilization:
  - stationary jitter reduction;
  - known translation/rotation;
  - fast motion;
  - occlusion/disocclusion;
  - fine/soft edges;
  - all-zero/all-one/tiny subject masks;
  - NaN/out-of-range rejection;
- spatial refinement with alternating gradients and resolution scaling;
- dynamic light wrap with fixed alpha and all four model-foreground/wrap
  combinations;
- compositor optimized/reference equivalence, exact endpoints, and bounded
  work-buffer lifecycle;
- backend-specific effective policy;
- activation commit/rollback and close behavior;
- privacy fallback behavior with every new mask path.

**Acceptance criteria:**

- Tests use deterministic generated arrays/fake runtimes and do not require a
  camera, network, GPU, or private footage.
- Metric thresholds are strict enough that the old defect simulation fails.
- Runtime fits existing CI budgets or heavy visual tests are separated into a
  documented qualification gate.
- Existing Python/Node quality and packaging tests remain green.

### MATTE-5.2 — Add end-to-end visual qualification · L

**Goal:** prove that metric improvements look better on representative video
and survive every local output boundary.

**Delivery rule:** The qualification manifest lists the exact selected
algorithms. If `MATTE-2.1`, `2.2`, or `2.4` ships, this task depends on and
qualifies it; a model-only policy does not pretend that an unselected
stabilizer/refiner was tested.

**Qualification set:**

- subject appearances: bald/short hair, long/fine hair, glasses, facial hair,
  headphones/solid accessories, dark/light opaque clothing, skin-tone and
  lighting diversity;
- motions: stationary, speech micro-motion, slow turn, fast turn, hand/prop
  crossing the face, entering/leaving frame;
- source conditions: bright, dim, compression noise, low contrast, clutter;
- backgrounds: static image, dynamic video, blur, solid color, live camera;
- cadences: 15/30/60, irregular, drops, restart;
- canvases: at least 640x360, 1280x720, and 1920x1080 where supported.

**Boundaries to compare from the same processed generation:**

- in-memory composite;
- local HighGUI preview before overlay;
- processed snapshot/MJPEG/WebSocket decode;
- pyvirtualcam loopback recording;
- Windows native virtual camera recording where applicable.

**Acceptance criteria:**

- Meets ratified metric gates and a blinded or side-by-side human review.
- No downstream boundary introduces an unexplained extra resize or contour
  defect.
- Results include backend/device/config/cadence and representative clips or
  contact sheets.
- Matte-quality metrics and human review run with reactions disabled. A
  separate deterministic reaction pass may prove final-sink parity but cannot
  improve or mask a matte score.
- Review explicitly checks halo, edge shimmer, ghost trail, cutout sharpness,
  hair retention, opaque-core backdrop leakage, accessory coverage, and motion
  cadence.

### MATTE-5.3 — Qualify performance and platform behavior · L

**Goal:** deliver the quality improvement without missing real-time budgets or
breaking supported platforms.

**Matrix:**

- Linux V4L2 + pyvirtualcam;
- macOS/OBS virtual camera on supported hardware;
- Windows pyvirtualcam/native MF profiles;
- CPU-only;
- NVIDIA CUDA where supported;
- qualified Windows DirectML profiles;
- Python/package profiles that lack MediaPipe or a GPU provider.

**Measurements:**

- capture-only FPS/read interval/drops and full-processing FPS separately;
- unique segmentation/composite FPS;
- output FPS/no-unread repeats plus separate capture-gap, processing-deadline,
  and sink-recovery counters;
- model pre/inference/post, refinement, compositor-substage, existing
  frame-processing scope, complete new-frame compute/service path,
  sink preparation/submission/copy, deliberate pacing wait, and schedule
  lateness p50/p95/p99;
- processing deadline misses, capture-slot overwrites, and inter-frame jitter;
- end-to-end age/latency;
- CPU utilization and memory;
- GPU utilization/VRAM where available;
- sustained run, restart, hot patch, and shutdown resource behavior.

**Provisional performance gates:**

- A balanced 30 FPS profile sustains at least 27 unique composites/s on its
  qualified hardware/canvas with p95 new-frame compute/service (through sink
  submission/copy but excluding intentional pacing sleep) at or below
  `33.3 ms`, unless a separately ratified scheduling model proves the same
  bounded result.
- Output sink remains at least 90% of target without unbounded latency growth.
- Capture-only and fixed-replay processing each meet their own declared target;
  one cannot mask failure of the other.
- The compositor stays within the sub-budget ratified by `MATTE-3.4`; output
  repeats are not credited as unique performance.
- New temporal/spatial refinement stays within its ratified stage budget and
  allocates bounded history (normally previous frame/mask plus bounded work
  buffers).
- A slower high-quality profile is not silently selected on hardware that
  cannot sustain it.

**Acceptance criteria:**

- Every advertised preset/backend tier lists qualified platforms and limits.
- Long-run memory is stable and native workers close within existing bounded
  shutdown rules.
- Provider fallback/reset is visible and does not produce a stale-state flash.
- This base-platform gate is authoritative with reactions disabled.
  `REACT-5.2` consumes its budget and separately qualifies active effects; it
  cannot relabel effect-only pixel changes as unique matte performance.

### MATTE-5.4 — Roll out defaults, migration, documentation, and rollback · M

**Goal:** safely make the qualified behavior available and, later, default.

**Context:** Segmentation configuration is persisted and hot-patchable. Existing
users may rely on current mask softness, light wrap, or RVM bypass semantics.
Changing the meaning of `temporal_smoothing` or backend defaults without a
schema decision would create silent visual changes.

**Deliverables:**

- Amend the visual-consistency ADR or add a matte-quality ADR covering:
  - timestamp/reset contract;
  - selected temporal/spatial algorithm;
  - backend-specific effective policies;
  - light-wrap temporal policy;
  - quality-tier/default decision;
  - performance and privacy gates.
- Decide schema compatibility:
  - preserve current fields and add a new stabilization mode; or
  - migrate legacy values to an explicit legacy policy.
- Keep old behavior selectable for at least one stable release if a default
  changes.
- Add sanitized rollout telemetry/counters, canary guidance, and a one-patch
  rollback to legacy policy.
- Update both `config/default.yaml` and `src/custback/default.yaml` together,
  README quality/troubleshooting/status sections, WebUI help, diagnostics, and
  packaging/install profiles.
- Do not flip to RVM or a higher detail profile on platforms that fail
  dependency/performance qualification.
- Keep reaction config/default/migration work in the `REACT` lane. Reaction
  delivery is not a prerequisite for declaring the matte defect fixed.

**Acceptance criteria:**

- Old persisted configs have documented, tested behavior.
- New installs select only a qualified sustainable profile.
- Rollback does not require deleting user config or model caches.
- Release evidence links baseline, ablation, visual, performance, platform,
  privacy, and migration results.

## 8. Suggested implementation sequence

1. Record the exact RVM/CUDA path and establish temporal, opaque-core, halo,
   leakage, and timing metrics (`MATTE-0.1`, `0.2`, `0.5`).
2. Run the same-source RVM/compositor/cadence factorial (`MATTE-0.3`); publish
   only evidence-backed, reversible diagnostics (`MATTE-0.4`).
3. Land the capture envelope and core cadence/gap/deadline telemetry
   (`MATTE-1.1`, `MATTE-3.2`) early so later diagnostics share stable event
   definitions.
4. In parallel, isolate the camera's 66.6 ms read cadence (`MATTE-3.1`),
   recover the 61.3 ms processing/32.7 ms compositor budget (`MATTE-3.4`), and
   add raw-alpha/foreground/contribution views (`MATTE-4.3`).
5. Select an RVM alpha/detail profile and explicit backend policy from evidence
   (`MATTE-2.3`, `MATTE-2.5`). Do not use temporal smoothing to conceal
   opaque-core failure.
6. Complete segmenter timestamp/reset and activation contracts
   (`MATTE-1.2`–`1.5`); MediaPipe timing remains a parallel fallback-specific
   correction.
7. Branch from the ablation evidence: implement motion-aware stabilization
   (`MATTE-2.1`) only for residual temporal alpha instability; pursue
   MediaPipe spatial refinement (`MATTE-2.2`) or light-wrap RGB stabilization
   (`MATTE-2.4`) independently when their own metrics fail. Add ordering
   dependencies only if the selected algorithms actually share temporal state.
8. Improve selection/presets/backend-aware controls (`MATTE-4.1`, `4.2`) and
   complete deterministic, visual, performance, platform, migration, and
   rollback gates before changing defaults.
9. Develop the disabled reaction engine in parallel where safe, but do not
   enable or ship it as a qualified feature until `MATTE-3.4` and `MATTE-5.3`
   establish output-loop headroom; then follow the Section-11 `REACT` sequence.

## 9. Pull-request boundaries

Keep changes reviewable. A reasonable split is:

1. replay bundle, raw RVM `pha`/`fgr`, and metrics only;
2. RVM opaque-core/halo attribution report and ablation evidence;
3. compositor substage telemetry and optimized compatible path;
4. capture-only cadence diagnosis/fix;
5. capture envelope and pipeline timing plumbing;
6. segmenter timestamp/reset contract;
7. MediaPipe timing/resampling;
8. RVM/refiner discontinuity reset and activation lifecycle;
9. explicit RVM/backend policy and profile selection;
10. new stabilizer behind an off/experimental mode, if evidence requires it;
11. spatial refinement and light-wrap changes;
12. status/selection metadata and local diagnostic views;
13. WebUI presets/backend-aware controls;
14. qualification evidence and default/migration change.

Do not combine an algorithm introduction, default flip, installer-profile
change, and migration in one unreviewable patch.

Reaction work follows the separate PR boundaries in Section 11. Do not hide a
matte/compositor fix inside a reaction-engine PR or use effects in base-quality
evidence.

## 10. Definition of done

The reported problem is considered resolved only when:

- a representative clip reproduces the old defect and passes the ratified new
  temporal, opaque-core, halo, and spatial gates;
- raw RVM alpha, model foreground, light wrap, and final blend attribution
  demonstrates that opaque clothing/accessories do not leak recognizable
  backdrop structure and genuine soft hair remains soft;
- MediaPipe uses actual capture time and all temporal backends reset correctly;
- the chosen temporal policy reduces stationary/slow boundary jitter without
  visible motion trails or lost fine detail; an added stabilizer is required
  only if the evidence selects one over a qualified model-only RVM policy;
- dynamic video light wrap cannot masquerade as alpha instability;
- the target device either sustains the requested unique capture FPS or reports
  a specific limitation and an actionable configuration;
- a qualified 30 FPS profile also sustains its unique processing target inside
  the ratified new-frame/compositor budget; a near-30 send rate made from 51%
  repeats does not qualify;
- status distinguishes capture/unique visual updates/output repeats and reports
  no-unread repeats separately from capture gaps/deadlines/sink events, selected
  quality tier/fallback reason, resolved RVM ratio, and configured/effective
  matte/compositor controls;
- in-memory preview and qualified virtual-camera consumer recordings agree;
- remote privacy, canonical geometry/color contracts, hot activation, bounded
  recovery, and shutdown guarantees remain intact;
- CPU/GPU/platform, visual, performance, migration, and rollback evidence is
  complete;
- documentation gives users a supported quality path instead of requiring
  trial-and-error tuning.

Reactions are disabled for all evidence above. Their separate definition of
done is in Section 11; an incomplete reaction lane does not keep the original
matte problem open.

## 11. Reaction and additive video-effects extension

### 11.1 Product outcome and scope

A **reaction preset** is a validated, persistent visual definition: asset,
duration, blend mode, placement, opacity/intensity, fade envelope, z-order, and
retrigger policy. A **reaction instance** is a transient one-shot event created
by a trigger. Instances are output state, not configuration: they are never
persisted across restart and a trigger does not increment the config version.

The initial user-visible scope is:

- authenticated/manual one-shot reactions;
- front-of-frame, output-space placement only;
- transparent still images with a bounded fade envelope;
- short RGB video effects with black as the identity color under explicit
  `add` or `screen` blending;
- a small generated or checksum-pinned built-in pack so the engine can ship
  before accepting arbitrary custom media;
- bounded simultaneous instances, deterministic z-order, restart/ignore/stack
  policies, cooldown, and a global stop/disable;
- the same final pixels at the hub, Web preview/streams, HighGUI preview, and
  virtual camera.

The initial scope explicitly excludes:

- audio playback or audio tracks in reaction assets;
- implicit black/luma/chroma keying for `normal` blend;
- true-alpha animated codecs until cross-platform decode preserves alpha;
- behind-subject, subject-only, or face/body-attached placement;
- automatic gesture, expression, voice, MIDI, Stream Deck, or chat triggers;
- operating-system global hotkeys.

Behind-subject/subject-only layers and automatic/tracked triggers remain
separate P2 follow-ups. They require a qualified alpha/tracking contract and
must not complicate the privacy-simple MVP.

Viewer safety is part of the product contract:

- reactions have a master disable, stop-all, opacity/intensity cap, and
  reduced-motion presentation;
- built-ins carry provenance, license, digest, and reviewed flash/intensity
  classification;
- custom clips are marked unqualified for photosensitivity unless a later
  analyzer/review says otherwise, and never auto-trigger by default;
- no asset may produce sound through custback.

### 11.2 Render architecture and ownership

Reactions are not part of segmentation, matte refinement, backdrop playback,
foreground color correction, or `_local_composite`. They operate on a
privacy-checked base at final output cadence:

```text
unique camera frame (when available)
  -> segmentation / backdrop / safe base composite
  -> remote raw-echo and privacy guard
  -> immutable SafeBaseFrame

every output tick, including camera-base reuse
  -> accept bounded reaction commands
  -> select reaction frames from monotonic elapsed time / source PTS
  -> render active reactions over fresh SafeBaseFrame pixels
  -> final geometry/dtype validation
  -> hub + preview/streams + virtual camera
```

Required invariants:

1. Split the current `last_output` concept into an immutable `SafeBaseFrame`
   envelope and the most recently published final frame. The envelope carries
   owned/read-only pixels, privacy-fallback state and sanitized reason, base
   sequence, monotonic presentation time, and capture/config/canvas
   generations. The engine must not infer privacy state from BGR pixels.
2. Never feed a previously reacted frame back into the reaction compositor;
   additive/screen effects would otherwise accumulate on camera-repeat ticks.
   Initialize each output tick from the current envelope's untouched pixels.
3. Advance reactions from monotonic output presentation time, not capture frame
   number. A 30 FPS effect can therefore animate across a base that updates at
   15 FPS without advancing RVM, matte, backdrop, or base-composite state.
4. Drain a dedicated bounded reaction command queue on every output tick. Do
   not use `PATCH /config`, and do not reuse the current mutation queue if it is
   serviced only when a unique camera frame is processed.
5. Reserve a priority control path for stop-all and master disable. It must be
   observed on output ticks during capture stalls and cannot be rejected behind
   a full trigger queue; a bounded acknowledgement timeout fails closed rather
   than claiming the stop was applied.
6. Run the remote raw-echo/privacy guard before reactions. The reaction engine
   receives only the guarded BGR base and never receives the raw camera frame.
   An overlay cannot make a raw-like remote candidate pass the guard.
7. If the envelope marks a privacy fallback slate, publish it byte-identically
   and suppress reaction pixels. The reaction clock continues and instances
   expire normally while suppressed, so recovery cannot restart stale effects.
8. Decode, normalize, resize-plan, and prefetch off the frame lane. A decoder
   underrun, bad effect frame, or render failure fails soft to the already-safe
   base and increments bounded telemetry; it never blocks or crashes output.
9. Treat every final frame handed to `FrameHub`, asynchronous JPEG/stream
   encoding, preview, or the virtual-camera sink as immutable for the entire
   consumer ownership window. Never recycle or mutate a reaction work buffer
   that an asynchronous consumer may still retain.
10. With no active reaction, the stage is a byte-identical bypass with no
   full-frame copy/allocation attributable to effects.
11. Effects are screen-space and do not mirror with the camera, so text and
   symbols remain readable.

The likely implementation seam is in `Pipeline._loop()` after
`_guard_remote_output()` and before output send/hub publication. New
resource-owned components should live in `src/custback/reactions.py` and,
if custom imports are enabled, `src/custback/reaction_assets.py`.

### 11.3 Render, timing, asset, and control contracts

#### Canonical render contract

For normalized accumulated destination RGB `D`, straight source RGB `S`, asset
coverage `alpha`, and the time-dependent scalar
`g(t) = clamp(preset_opacity * master_opacity * intensity_cap *
trigger_override * fade(t), 0, 1)`, define effective coverage
`a = alpha * g(t)`. The straight-alpha reference equations are:

```text
normal: S * a + D * (1 - a)
add:    clamp(D + S * a, 0, 1)
screen: 1 - (1 - D) * (1 - S * a)
```

The canonical cached representation is premultiplied source
`P = S * alpha`. Its implementation equations are therefore:

```text
normal: P * g(t) + D * (1 - alpha * g(t))
add:    clamp(D + P * g(t), 0, 1)
screen: 1 - (1 - D) * (1 - P * g(t))
```

- Start `D` from a copy-on-write view of the untouched `SafeBaseFrame` and
  composite each stable-ordered layer sequentially over the accumulated result.
  "Fresh base" means fresh per output tick, not that simultaneous layers ignore
  earlier layers in the same tick.
- `add` and `screen` make black RGB an identity without inventing an alpha key.
- RGB assets without an alpha channel use `alpha = 1` only for explicit
  `add`/`screen` or full-frame opaque presets.
- Anchored `normal` media requires explicit alpha. Opaque `normal` video is
  allowed only for an explicitly full-frame preset.
- Source color is normalized to the existing full-range sRGB contract; alpha is
  linear coverage and is premultiplied once in the internal representation.
- Rendering follows the selected `compositing.blend_space`, with frozen
  references for `srgb_legacy` and `linear_srgb`. Unsupported combinations are
  rejected rather than silently approximated.
- Placement uses output-normalized anchors/width or explicit full-canvas
  contain/cover, preserves aspect ratio, clips to the canvas, and touches only
  the computed ROI.
- Stable layer order is `(z_index, accepted_trigger_sequence)`.
- Fade, master opacity, viewer intensity cap, and any permitted bounded trigger
  override enter only through `g(t)`; they cannot extend the asset duration or
  be applied twice.

#### Scheduler and event contract

- The frame lane assigns a monotonic accepted sequence and effective start time
  at the next output boundary.
- Source frames are selected from normalized PTS. Late output ticks skip effect
  frames to retain phase; playback never slows to drain a backlog.
- Presets define one of `restart`, `ignore_if_active`, or bounded `stack`.
- Active and queued counts, per-preset cooldown, global admission rate, and
  decoded-cache bytes are bounded configuration.
- Stop-all/master-disable use a reserved priority control path. When ordinary
  trigger capacity is full, pending triggers may be rejected/evicted according
  to the frozen policy, but an emergency stop cannot wait behind them.
- An admitted instance owns an immutable preset/asset digest snapshot. A later
  catalog edit cannot change it mid-animation.
- Canvas/output generation change cancels or re-plans instances according to
  one versioned policy; stale-size pixels are never rendered.
- Idempotency keys are retained in a bounded TTL/LRU structure so retries do
  not duplicate a reaction.

#### Durable versus ephemeral state

An illustrative compatibility-disabled configuration is:

```yaml
reactions:
  enabled: false
  master_opacity: 1.0
  max_active: 2
  command_queue_limit: 16
  slots: []
```

Exact fields and limits are ratified by `REACT-0.1`. Config owns only durable
enable/limits/slot policy. Catalog assets live in a private reaction namespace.
Active instances and trigger/cancel events never appear in persisted config and
never advance `X-Config-Version`.

The authenticated event API target is:

- `GET /reactions` — sanitized catalog IDs/readiness/metadata/thumbnails;
- `POST /reactions/{id}/trigger` — admit and return `202` plus instance ID;
- `GET /reaction-instances` — bounded active/suppressed instance status;
- `DELETE /reaction-instances/{id}` — cancel one;
- `DELETE /reaction-instances` — stop all;
- separate import/delete routes only after the custom-import gate.

Trigger requests carry only a catalog ID and optional bounded overrides/
idempotency token—never a path or URL. Capacity/cooldown/rate rejection is
`429`, not-ready/in-use conflict is `409`, malformed data is `422`, unsupported
media is `415`, and a stopped pipeline is `503`.

#### Asset/security contract

- Start with generated or bundled assets whose license, provenance, digest, and
  safety classification are known.
- A custom reaction store is separate from `/backgrounds`; do not expose its
  absolute directory, original filenames, or raw decoder errors.
- Reuse the upload store's owner/no-follow checks, private modes, streaming byte
  reservations, hidden staging, atomic promotion, quota ledger, cancellation
  cleanup, and active-reference deletion protection.
- Do not reuse its current video validation unchanged: reaction imports also
  bound duration, FPS, frame count, total decoded pixels/bytes, decode wall
  time, color/alpha format, timestamps, and per-frame resolution.
- Reject network/playlist/nested FFmpeg protocols, subtitles, attachments, and
  audio streams. Assets are operator-owned local bytes only.
- Decode/prefetch is bounded and off-lane. The output lane performs no file I/O,
  codec call, unbounded allocation, or wait for an effect frame.
- Public thumbnails are separately rendered/sanitized; raw asset-serving is not
  required.

### 11.4 Reaction delivery plan

| ID | Task | Phase | Priority | Size | Depends on |
| --- | --- | ---: | ---: | ---: | --- |
| REACT-0.1 | Freeze reaction UX/render/time/privacy/threat contracts | 0 | P0 | M | — |
| REACT-0.2 | Add deterministic reaction fixtures, replay metrics, and budgets | 0 | P0 | M | REACT-0.1, MATTE-0.1, MATTE-0.2 |
| REACT-1.1 | Add typed one-shot media normalization and immutable manifests | 1 | P0 | L | REACT-0.1 |
| REACT-1.2 | Add the private catalog and qualified built-in reaction pack | 1 | P0 | M | REACT-1.1 |
| REACT-1.3 | Add hardened custom reaction import/deletion lifecycle | 1 | P1 | L | REACT-1.1, REACT-1.2, REACT-5.2 |
| REACT-2.1 | Add bounded trigger instances and monotonic output scheduler | 2 | P0 | M | REACT-0.1 |
| REACT-2.2 | Add premultiplied ROI `normal`/`add`/`screen` compositor | 2 | P0 | L | REACT-0.2, REACT-1.1 |
| REACT-2.3 | Integrate guarded-base, repeat-safe final output rendering | 2 | P0 | L | REACT-1.2, REACT-2.1, REACT-2.2, MATTE-3.2 |
| REACT-3.1 | Add authenticated catalog/trigger/cancel/status API | 3 | P0 | M | REACT-1.2, REACT-2.1, REACT-2.3 |
| REACT-3.2 | Add WebUI tray and focused-preview reaction controls | 3 | P1 | M | REACT-3.1 |
| REACT-4.1 | Add reaction observability and overload/degrade policy | 4 | P0 | M | REACT-2.3, REACT-3.1, MATTE-3.2, MATTE-4.1 |
| REACT-5.1 | Add deterministic, API, security, privacy, and lifecycle tests | 5 | P0 | L | incremental with each selected REACT task |
| REACT-5.2 | Qualify visual/color/performance/platform/sink parity | 5 | P0 | L | REACT-0.2, REACT-2.3, REACT-4.1, REACT-5.1, MATTE-3.4, MATTE-5.3 |
| REACT-5.3 | Roll out schema/defaults/assets/docs and rollback | 5 | P1 | M | REACT-3.2, REACT-5.2 |
| REACT-6.1 | Add matte-aware behind-subject and subject-only layers | 6 | P2 | L | REACT-2.3, REACT-5.2, MATTE-2.3, MATTE-2.5, MATTE-5.2 |
| REACT-6.2 | Investigate tracked/automatic/external trigger producers | 6 | P2 | XL | REACT-2.1, REACT-5.2, separate tracking/trigger ADR |

Core implementation can land behind `reactions.enabled: false` while matte work
continues. A production-enabled reaction profile must wait for `MATTE-3.4` and
`MATTE-5.3` to prove base-loop headroom. The first usable/qualified slice is
`REACT-0.1`, `0.2`, `1.1`, `1.2`, `2.1`–`2.3`, `3.1`, `3.2`, `4.1`, `5.1`,
`5.2`, and `5.3`. Custom imports and all Phase-6 work are not MVP blockers.

### 11.5 Detailed reaction tasks

#### REACT-0.1 — Freeze reaction UX/render/time/privacy/threat contracts · M

**Goal:** remove product and security ambiguity before introducing an
output-clock state machine.

**Deliverables:**

- Add an ADR/spec covering:
  - preset versus instance ownership;
  - front-layer MVP and later matte-aware layers;
  - formulas, color space, alpha representation, placement, z-order, fades;
  - output-clock/PTS behavior, late skips, retrigger/cancel/end semantics;
  - command admission, limits, idempotency, config generation behavior;
  - privacy-guard ordering and byte-identical fallback-slate suppression;
  - viewer-safety/reduced-motion/intensity policy;
  - built-in versus custom-asset trust boundaries.
- Define bounded default/cap ranges but do not choose a production-enabled
  default before qualification.
- Threat-model raw-echo masking, asset parser abuse, decompression/resource
  exhaustion, trigger floods, deletion races, stale generation state, and
  high-cardinality/path leakage.
- Specify that gesture/audio/tracking systems, if later added, are event
  producers and cannot bypass the same admission/privacy contracts.

**Acceptance criteria:**

- Every MVP mode has an exact formula, timebase, placement, failure, and
  accessibility outcome.
- No reaction path can run before or influence the remote privacy decision.
- A trigger is clearly ephemeral and config changes remain transactional.
- Unsupported media/layer/trigger combinations have explicit rejection
  behavior rather than fallback guesses.

#### REACT-0.2 — Add deterministic reaction fixtures, replay metrics, and budgets · M

**Goal:** make reaction correctness and cost measurable before integration.

**Deliverables:**

- Generate deterministic fixtures for:
  - transparent edges and premultiplication fringes;
  - black-background `add`/`screen`;
  - `normal` alpha endpoints and opacity/fade ramps;
  - placement clipping, aspect fit, z-order, simultaneous/retriggered effects;
  - 15 FPS base reuse under 30/60 FPS output clocks;
  - irregular PTS, late ticks, underrun, cancellation, and generation change;
  - privacy-slate suppression and near-raw remote candidates.
- Extend offline replay with a guarded-base plus reaction event trace containing
  bounded preset IDs, asset digests, accepted sequence, start/cancel time, seed,
  and output time. Asset bytes remain opt-in for copyright/privacy.
- Report:
  - per-mode pixel/color error versus reference;
  - changed pixels outside planned ROI;
  - trigger-to-first-visible latency;
  - reaction render p50/p95/p99;
  - effect source frames rendered/skipped/underrun;
  - base-reuse, reaction-only, exact-repeat, and send FPS;
  - allocation/cache bytes and output inter-send jitter.
- Ratify provisional budgets. Starting targets for the evidence host are
  `<=2 ms` p95 for one bounded 720p ROI and `<=5 ms` p95 at the admitted active
  cap. These are ceilings, not proof of headroom: base compute plus
  admitted-cap reaction render plus final validation/sink submission and a
  ratified scheduling margin must fit the target presentation interval without
  reducing base-update throughput. Exclude deliberate pacing sleep, but report
  it and schedule lateness separately. Amend targets in evidence rather than
  burying them in code.

**Acceptance criteria:**

- A fixed clock/event trace produces deterministic frames and instance state.
- Reapplying onto a previous reacted output fails a no-accumulation fixture.
- Encoded/linear reference differences are explicit and tested.
- A reaction-only output change cannot increase camera/matte/base-composite FPS.
- The harness distinguishes disabled, one-layer, and admitted-cap cost.

#### REACT-1.1 — Add typed one-shot media normalization and immutable manifests · L

**Goal:** turn trusted built-in or imported media into bounded frame-lane-safe
reaction assets.

**Deliverables:**

- Define a versioned immutable manifest with:
  - opaque asset ID/content digest/schema;
  - media kind, dimensions, duration, normalized FPS/PTS;
  - RGB/color and alpha contract;
  - allowed blend/placement modes;
  - decoded/cache bounds;
  - provenance/license and flash/intensity classification.
- Implement one-shot decoding/normalization:
  - transparent PNG/static RGBA for `normal`;
  - bounded RGB video for black-identity `add`/`screen`;
  - generated/procedural assets where practical;
  - no loop, audio, network source, or implicit keying.
- Normalize orientation, SDR color, irregular timestamps, and alpha coverage.
- Predecode or prefetch into a bounded resource-owned cache/ring off the frame
  lane. Choose by decoded-byte budget, not unbounded frame count.
- Validate every returned frame's dtype, dimensions, finiteness/range, alpha,
  and immutable manifest identity.
- On underrun or fatal decode error, return an explicit unavailable result so
  output can use the safe base.

**Likely files:**

- new `src/custback/reactions.py`;
- optional new `src/custback/reaction_assets.py`;
- selected helpers from `src/custback/video_decoder.py` without weakening its
  background contract;
- focused `tests/test_reactions.py`.

**Acceptance criteria:**

- All decode/file I/O is off the output lane.
- Media duration/FPS/PTS and exact end behavior are deterministic under a fake
  clock.
- Missing alpha is rejected for anchored `normal` instead of becoming an opaque
  rectangle.
- Cache/decoder memory, handles, threads, cancellation, and close are bounded.
- True-alpha animated formats remain disabled until their alpha survives
  supported-platform qualification.

#### REACT-1.2 — Add the private catalog and qualified built-in reaction pack · M

**Goal:** provide usable, safe reaction IDs without exposing paths or requiring
custom parser attack surface for the first release.

**Deliverables:**

- Add a private catalog that maps sanitized stable IDs to immutable manifests.
- Add a small generated or properly licensed/checksum-pinned built-in set
  covering at least:
  - one transparent/fading `normal` reaction;
  - one black-background additive/screen reaction;
  - one low-motion/reduced-intensity alternative.
- Package non-procedural assets reproducibly and verify digest/license metadata
  at build/startup.
- Generate bounded sanitized thumbnails; never serve the source asset.
- Define refcount/readiness/quarantine states and prevent deletion/replacement
  while an immutable instance owns an asset.
- Expose labels/metadata without original filenames, absolute directories, or
  raw decoder details.

**Likely files:**

- `src/custback/reaction_assets.py`;
- package-data entries in `pyproject.toml`/`MANIFEST.in` where needed;
- API public models only after `REACT-3.1`;
- catalog/packaging tests.

**Acceptance criteria:**

- Clean install/wheel/npm-managed environments resolve identical digests.
- Built-ins have documented license/provenance and viewer-safety review.
- Catalog corruption/missing asset fails readiness without breaking base video.
- Catalog listing contains no local path or high-cardinality exception data.

#### REACT-1.3 — Add hardened custom reaction import/deletion lifecycle · L

**Goal:** let operators add effects without turning a media parser or filesystem
path into an unbounded live-output authority.

**Context:** This is deliberately post-MVP/P1. Existing background upload
hardening is reusable, but its video validator does not bound reaction duration,
FPS, frame count, total decoded pixels, decode wall time, or alpha semantics.

**Deliverables:**

- Add a separate owner-private reaction namespace, quota ledger, hidden staging,
  atomic promotion, startup cleanup, and generated IDs.
- Accept bounded local upload bytes only. Reject URLs, playlists, nested
  protocols, symlinks, devices, subtitles, attachments, and audio.
- Enforce byte/storage/file, dimension, duration, FPS, frame-count,
  total-decoded-pixel/byte, decode-time, codec/container, color/alpha, and
  per-frame consistency limits before promotion.
- Normalize accepted media to the canonical manifest/asset form off-lane.
- Serialize upload/delete/trigger races with catalog generations/refcounts.
  Active/preloaded deletion returns `409` unless an explicit cancel-and-delete
  transaction is later designed.
- Never expose the configured directory or original upload name.

**Acceptance criteria:**

- Truncation, malformed metadata, resolution changes, extreme PTS, zip/path
  traversal equivalents, symlink races, quota races, cancellation, and crash
  recovery are tested.
- A failed import leaves no visible partial asset and returns quota ownership.
- A custom asset cannot cause file/network I/O on the frame lane.
- Custom assets remain visibly marked unqualified for photosensitivity unless
  separately reviewed.

#### REACT-2.1 — Add bounded trigger instances and monotonic output scheduler · M

**Goal:** make reaction admission and playback deterministic during normal
capture, repeats, stalls, and trigger bursts.

**Deliverables:**

- Add immutable `ReactionDefinition`/instance/receipt types and a controller
  with:
  - bounded command queue and active set;
  - a separately reserved stop-all/master-disable control slot or atomic
    priority channel;
  - monotonic accepted sequence/effective start;
  - idempotency TTL/LRU;
  - global/per-preset token bucket/cooldown;
  - `restart`, `ignore_if_active`, and bounded `stack`;
  - cancel-one/stop-all;
  - deterministic z-order and seeded randomness.
- Drain commands at every output boundary even when no unread camera frame
  exists.
- Apply/acknowledge priority stop/disable at an output boundary during capture
  stalls and trigger floods. If the output loop cannot acknowledge within the
  bounded API deadline, report failure rather than a false stopped state.
- Select frames from actual elapsed output time and normalized PTS; skip late
  source frames and expire exactly at the defined end.
- Keep active state non-persistent and generation-owned. Define canvas-change,
  config-disable, shutdown, and privacy-suppression behavior.
- Return admission only after capacity/readiness checks; never claim an effect
  started merely because an HTTP request arrived.

**Acceptance criteria:**

- Fake-clock tests cover boundary start, 15/30/60 output, long stall, late skip,
  exact end, retrigger policies, cancel, idempotency, queue overflow, and
  deterministic ordering, including stop-all/master-disable with an ordinary
  queue already full.
- Output/base reuse does not pause or advance matte/backdrop state.
- No command/instance/idempotency structure grows without a configured bound.
- Privacy suppression advances/expires time without rendering on the slate.

#### REACT-2.2 — Add premultiplied ROI reaction compositor · L

**Goal:** render high-quality effects without another unconditional full-frame
float pipeline.

**Deliverables:**

- Implement validated ROI transforms and `normal`, `add`, and `screen`
  reference/optimized paths.
- Use canonical premultiplied alpha internally and preserve transparent-edge
  color without dark/bright fringes.
- Support selected `srgb_legacy` and `linear_srgb` semantics with explicit
  reference tolerances and exact identity endpoints.
- Clip placement and compute only the touched ROI; reuse bounded work buffers
  where lifecycle-safe.
- Initialize from the same untouched safe base once per output tick, then
  composite multiple layers sequentially in stable order over the accumulated
  destination.
- Make zero opacity, empty/outside ROI, black add/screen, and no-active cases
  exact identities.
- Never mutate the cached base, asset frame, alpha, or other shared buffer.

**Likely files:**

- `src/custback/reactions.py`;
- selected color helpers from `src/custback/compositor.py`;
- generated reference/benchmark tests.

**Acceptance criteria:**

- Formula, clipping, endpoint, dtype/range/contiguity, z-order, and both
  blend-space tests pass.
- Repeated calls with the same base/event/time are deterministic.
- A prior reacted frame cannot be accepted as an implicit base.
- Outside-ROI pixels remain byte-identical.
- Disabled/no-active path performs no material full-frame work.

#### REACT-2.3 — Integrate guarded-base, repeat-safe final output rendering · L

**Goal:** put reaction pixels in every final sink while preserving base,
privacy, and timing truth.

**Deliverables:**

- Add resource-owned reaction controller/engine lifecycle to pipeline startup,
  hot config, rollback, generation change, and shutdown.
- Replace the pixel-only `last_safe_base` idea with the immutable
  `SafeBaseFrame` envelope defined in Section 11.2, separate from last
  published final output.
- Apply reactions after `_guard_remote_output()` and before final
  validation/send/hub publication on every output tick.
- Ensure trigger commands are serviced during base reuse/capture stall.
- Suppress reaction pixels on any privacy fallback and keep its slate
  byte-identical.
- Preserve one final frame identity for hub/snapshot/MJPEG/WebSocket/preview/
  virtual-camera consumers.
- Define copy/lease/refcount ownership so published NumPy buffers remain
  immutable until every synchronous/asynchronous consumer is done; pooled
  work buffers cannot be recycled while retained by `FrameHub` or an encoder.
- On decode/render/overload failure, publish the safe base and continue; effects
  are the first optional work shed under deadline pressure.
- Update provenance counters so effect-only pixel changes do not alter camera,
  segmentation, backdrop, or base-composite counters.

**Likely files:**

- `src/custback/pipeline.py`;
- `src/custback/reactions.py`;
- `src/custback/hub.py`;
- lifecycle/pipeline/privacy/sink-parity tests.

**Acceptance criteria:**

- A 15 FPS base under 30 FPS output shows time-correct effect animation without
  cumulative brightness.
- A current/delayed near-raw remote candidate plus an opaque effect still yields
  the unchanged privacy slate.
- Repeated fallback envelopes remain byte-identical without pixel-based slate
  detection, and a deliberately slow asynchronous consumer never observes a
  later tick mutating its published frame.
- Reaction frames never reach raw hub, segmentation, color analysis, remote
  renderer input, or matte diagnostics.
- All final sinks observe the same generation/frame within their documented
  encoding tolerance.
- Engine failure does not stop, delay indefinitely, or replace the safe base.

#### REACT-3.1 — Add authenticated catalog/trigger/cancel/status API · M

**Goal:** expose a low-latency ephemeral command plane without weakening the
existing authenticated configuration/storage boundary.

**Deliverables:**

- Add strict bounded public models and routes described in Section 11.3.
- Accept a trigger only by catalog ID; never accept a filesystem path/URL.
- Support bounded `Idempotency-Key` semantics and return a stable receipt with
  admitted instance ID/state/effective sequence.
- Serialize trigger/cancel/catalog races through the reaction controller and
  asset refcounts.
- Route stop-all/master-disable through the reserved priority control path;
  ordinary trigger backpressure cannot prevent admission. Return success only
  after output-boundary acknowledgement, with a bounded timeout/`503` if the
  frame loop cannot apply it.
- Add endpoint-specific request/rate/backpressure limits and stable errors:
  `404`, `409`, `413`, `415`, `422`, `429`, and `503`.
- Keep trigger/cancel out of `PATCH /config` and preserve config version.
- Return bounded sanitized status; no original filename/path, raw decoder
  exception, unbounded event history, or source asset bytes.

**Likely files:**

- `src/custback/api/server.py`;
- public API/OpenAPI models and tests;
- coordinator capabilities in `src/custback/pipeline.py`;
- security/race/idempotency tests.

**Acceptance criteria:**

- `202` means the pipeline admitted the event, not merely that ASGI parsed it.
- Auth/origin/session/no-store/nosniff contracts cover every route.
- Trigger flood cannot grow work or starve config/lifecycle control.
- Stop-all/master-disable succeeds during capture stalls and a full ordinary
  trigger queue, or returns the documented bounded failure without claiming
  effects stopped.
- Duplicate idempotent requests create one instance.
- Concurrent stop/delete/trigger outcomes are deterministic and documented.

#### REACT-3.2 — Add WebUI tray and focused-preview reaction controls · M

**Goal:** make qualified reactions immediately usable and their real state
obvious.

**Deliverables:**

- Add an accessible reaction tray near the preview with sanitized thumbnail,
  label, readiness/safety classification, trigger button, active progress, and
  stop-all.
- Show accepted, active, suppressed, completed, rejected, and failed states;
  do not optimistically claim playback before API admission.
- Respect reduced-motion and master disable/intensity settings.
- Add focused-preview shortcuts only through a reaction palette/mode so they do
  not collide with existing `0`–`5`, `n/p`, `[ ]`, `h`, `q`, or `ESC`
  semantics. Ignore key auto-repeat and text-input focus.
- Document that these are window-focused controls, not global hotkeys.
- Keep asset/config editing separate from the instantaneous trigger control.

**Likely files:**

- `src/custback/api/webui.py`;
- `src/custback/preview.py`;
- API/UI/accessibility/keyboard tests.

**Acceptance criteria:**

- Keyboard-only and screen-reader labeling are covered.
- UI/native preview agree with admitted/active/suppressed state.
- Privacy fallback explains why effects are suppressed.
- Key repeat or held shortcuts cannot bypass cooldown/admission limits.

#### REACT-4.1 — Add reaction observability and overload/degrade policy · M

**Goal:** expose effect health without corrupting base-video truth or leaking
asset identity/path data.

**Deliverables:**

- Add stable bounded status/overlay/ready/shutdown fields for:
  - enabled/readiness and catalog count;
  - active/queued/suppressed counts;
  - triggers admitted/rejected/idempotent/cancelled/completed;
  - reaction-only output frames and exact final repeats;
  - effect frames rendered/skipped/underrun;
  - render/decode/cache failures and quarantines;
  - privacy suppressions and overload bypasses;
  - cache bytes/hits and configured caps;
  - `reaction_render_ms` p50/p95 or EWMA with documented scope;
  - trigger-to-first-visible latency and output inter-send jitter.
- Preserve `MATTE-3.2` ownership of base/matte/reuse/send semantics; consume
  those fields instead of redefining them. This task owns
  reaction-update/reaction-only counters and tests through the namespaced
  extension seam.
- Consume separately measured sink submission/copy time, deliberate pacing
  wait, and schedule lateness; do not attribute pyvirtualcam pacing sleep to
  reaction compute.
- Define overload order: skip/omit optional effect work, preserve safe base,
  keep monotonic effect time, record one transition log, and recover without a
  stale replay.
- Use stable preset/event categories or bounded IDs; never log paths, original
  names, hashes at unbounded cardinality, or raw exception text.

**Acceptance criteria:**

- A 15 FPS base plus 30 FPS effect reports both rates truthfully.
- No-active and privacy-suppressed states are distinguishable.
- Overload/corrupt media keeps base output healthy and reports a sanitized
  reason.
- Public status and OpenAPI remain bounded/type-consistent.

#### REACT-5.1 — Add deterministic, API, security, privacy, and lifecycle tests · L

**Delivery rule:** This is an incremental umbrella. Each REACT implementation
PR lands focused tests; final completion covers every selected feature.

**Required groups:**

- formulas/color/alpha/ROI/z-order/fade/golden frames;
- output-clock PTS, repeat/no-accumulation, skip, end, retrigger, idempotency;
- queue/cooldown/rate/capacity/overload bounds;
- priority stop-all/master-disable during full trigger queues and capture stalls;
- decode/cache/underrun/fatal-error/resource close;
- startup/config activation/rollback/canvas change/shutdown;
- API auth/origin/schema/error/concurrency;
- catalog ownership/symlink/TOCTOU/quota/import/delete races where applicable;
- remote near-raw guard ordering and byte-identical fallback slate;
- base/matte/reaction provenance counters;
- hub/preview/stream/virtual-camera identity;
- immutable published-buffer ownership with a deliberately slow retained
  consumer and pooled-buffer reuse pressure;
- viewer-safety metadata and reduced-motion behavior.

**Acceptance criteria:**

- Deterministic core tests need no camera, GPU, network, private footage, or
  copyrighted asset.
- Trigger-flood/malformed-media tests remain bounded and cannot stall CI.
- Every failure path returns/publishes the guarded base.
- Existing matte/privacy/config/lifecycle suites remain green.

#### REACT-5.2 — Qualify visual/color/performance/platform/sink parity · L

**Goal:** prove reactions are safe and smooth on top of an already-qualified
base without consuming its truth or budget.

**Matrix:**

- Linux/macOS/Windows and supported output backends;
- CPU-only and qualified accelerators;
- 720p/1080p and qualified higher canvases;
- 15/30/60 base/output combinations and irregular stalls;
- zero, one, and admitted-cap reactions;
- transparent still, procedural, and additive/screen video;
- static/video/camera/blur/passthrough backgrounds;
- healthy remote output and every privacy fallback;
- trigger burst, retrigger, cancel, corrupt/underrun, config rebuild, shutdown.

**Gates:**

- Effects-disabled output remains byte-identical and has negligible branch/
  allocation overhead.
- The base profile first passes `MATTE-3.4`/`MATTE-5.3`; reaction-only visual
  updates cannot satisfy that gate.
- A base pass at the target boundary is not sufficient headroom. For one-effect
  and admitted-cap profiles, measured base compute + reaction render + final
  validation/sink submission + scheduling margin fits the target interval
  without lowering base-update throughput. Deliberate pacing sleep and schedule
  lateness are reported separately.
- Those profiles also meet the ratified `reaction_render_ms`, output-jitter,
  trigger-latency, CPU/GPU, and memory/handle budgets from `REACT-0.2`.
- Late/overload policy preserves phase and safe base without unbounded latency.
- Transparent edges, black identity, clipping, fades, intensity, and both blend
  spaces pass visual/color review.
- In-memory final output, Web streams, preview, and virtual-camera recordings
  agree within documented encoding tolerance.
- Long burst/retrigger soak has stable memory, cache, queues, threads, and
  handles.

**Acceptance criteria:**

- Each advertised active-reaction cap lists qualified hardware/canvas limits.
- A failing tier stays disabled or exposes a smaller qualified cap.
- Privacy fallback is byte-identical in every active-effect state.
- Viewer-safety/reduced-motion review and built-in asset provenance are signed
  off.

#### REACT-5.3 — Roll out schema/defaults/assets/docs and rollback · M

**Goal:** ship the qualified lane without changing existing output by default.

**Deliverables:**

- Add a versioned config/schema decision for durable reaction enable/limits/
  slots; active instances remain ephemeral.
- Keep reactions disabled by default through the initial compatible rollout.
- Update both default YAML copies, config migration/merge/reset, diagnostics,
  ready/shutdown docs, WebUI help, README, packaging, and built-in asset
  manifests together.
- Add canary guidance, per-tier capacity, viewer-safety labels, custom-import
  trust warning, and one-patch rollback/disable.
- Preserve old configs without requiring catalog/cache deletion.
- Never bundle unlicensed media or enable an unqualified asset/cap/platform.

**Acceptance criteria:**

- Old persisted config yields the exact no-reaction path.
- Global disable/rollback immediately stops admission and follows documented
  active-instance cancellation while retaining recoverable catalog data.
- Release evidence links contract, threat model, fixtures, privacy, visual,
  performance, platform, accessibility, migration, and rollback results.

#### REACT-6.1 — Add matte-aware behind-subject and subject-only layers · L

**Goal:** support effects that pass behind or are clipped to the person only
after matte quality is trustworthy.

**Constraints:**

- Atomically pair the exact displayed safe base with its alpha/generation.
- `behind_subject` multiplies effect alpha by `1 - subject_alpha`;
  `subject_only` multiplies it by `subject_alpha`.
- Preserve soft edges; no nearest-neighbor mask resampling, hard threshold, or
  stale mask on base reuse/generation change.
- Suppress/reject when alpha is missing or backend/profile is unqualified.
  Never silently become a front overlay.
- Disable matte-aware layers in remote mode unless a new privacy design proves
  the renderer/base/mask authority.
- Requalify halo, edge shimmer, performance, and sink parity.

**Acceptance criteria:**

- Depends on passing RVM alpha/halo and end-to-end visual gates.
- No stale or wrong-generation alpha can reach a rendered layer.
- Missing/unqualified alpha has explicit UI/API/status behavior.

#### REACT-6.2 — Investigate tracked/automatic/external trigger producers · XL

**Goal:** decide whether gesture/expression, face/body attachment, voice/MIDI,
Stream Deck, or chat integrations justify a separate subsystem.

**Decision requirements:**

- Produce a separate ADR per producer class covering authority, authentication,
  local-only inference, landmark/audio persistence/egress, false triggers,
  cooldown/debounce, latency, accessibility, and platform support.
- Feed only the same bounded `ReactionController` command contract; no producer
  can bypass admission, privacy, safety, or rate limits.
- Face/body-attached placement requires an atomically paired tracking frame/
  generation and must not reuse avatar blendshape state without an explicit
  compatibility contract.
- Automatic triggers are opt-in and disabled by default.

**Acceptance criteria:**

- This task produces accept/reject ADRs and bounded spikes, not an implicit MVP
  commitment.

### 11.6 Reaction implementation sequence

1. Freeze the UX/render/time/privacy/threat contract and evidence budgets
   (`REACT-0.1`, `0.2`).
2. Build the typed normalizer/manifest and small qualified built-in catalog
   (`REACT-1.1`, `1.2`).
3. Implement the bounded scheduler and ROI renderer independently with fake
   clocks/frames (`REACT-2.1`, `2.2`).
4. Integrate the safe-base/final-output split behind the disabled flag
   (`REACT-2.3`), preserving MATTE provenance.
5. Add the authenticated command/status plane, observability, then UI controls
   (`REACT-3.1`, `4.1`, `3.2`).
6. Complete incremental tests and qualify only after base-loop headroom passes
   (`REACT-5.1`, `5.2`, `MATTE-3.4`, `MATTE-5.3`).
7. Roll out disabled-by-default schema/assets/docs (`REACT-5.3`).
8. Add custom imports only after the built-in engine is qualified
   (`REACT-1.3`).
9. Treat matte-aware layers and automatic/tracked triggers as independent
   evidence-gated follow-ups (`REACT-6.1`, `6.2`).

### 11.7 Reaction pull-request boundaries

A reviewable split is:

1. ADR/threat model/formulas/generated fixtures;
2. typed manifest and deterministic built-in normalizer;
3. private catalog/built-in pack and packaging;
4. scheduler/controller with fake clock only;
5. ROI renderer/reference equivalence only;
6. guarded-base pipeline integration behind disabled config;
7. API/admission/idempotency/rate limiting;
8. status/telemetry/degrade policy;
9. WebUI/preview controls;
10. qualification evidence and disabled-default rollout;
11. hardened custom import lifecycle;
12. one PR/ADR per optional Phase-6 capability.

Do not combine custom media parsing, core render integration, an enabled
default, and automatic triggers in one patch.

### 11.8 Reaction definition of done

The initial reaction lane is done only when:

- effects render from a fresh guarded safe base at output time without
  accumulation or capture-cadence stutter;
- safe-base privacy/generation provenance is explicit, and every published
  frame remains immutable for the full consumer ownership window;
- remote guard ordering is proven and every privacy fallback slate remains
  byte-identical;
- no reaction changes segmentation, matte, backdrop, color correction, raw
  history, or base-quality metrics;
- built-in assets/formats/blend math/color/alpha/placement/time semantics are
  versioned, deterministic, licensed, and viewer-safety reviewed;
- trigger/cancel/idempotency/cooldown/capacity/rate/overload behavior is bounded
  and truthful;
- stop-all/master-disable remains admissible during capture stalls and trigger
  saturation and is not acknowledged before the output loop applies it;
- decode/render failures always publish the safe base and close resources;
- base reuse, reaction-only frames, exact final repeats, and send FPS remain
  separately observable;
- disabled, one-effect, and admitted-cap profiles pass complete-loop,
  visual/color, platform, sink-parity, accessibility, privacy, and soak gates;
- the same final generation reaches hub, preview/streams, and virtual camera;
- reactions remain compatibility-disabled until a qualified rollout and can be
  stopped/disabled without deleting user data;
- custom imports, matte-aware layers, and automatic/tracked triggers remain off
  unless their separate tasks pass.
