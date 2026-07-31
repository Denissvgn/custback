# Video matte stability and output quality backlog

**Status:** proposed

**Primary symptom:** the subject silhouette, especially the face/head boundary,
appears to "dance" between video frames

**Scope:** local background replacement (`blur`, `image`, `video`, `color`, and
camera backdrops), with shared capture/segmentation/output infrastructure

**Task prefix:** `MATTE`

## 1. Purpose

This document turns the reported video-quality problem into an implementation
backlog. It preserves the observations, code-path analysis, hypotheses,
measurement plan, dependencies, acceptance criteria, and rollout constraints
needed for a developer to work on individual tasks without reconstructing the
investigation.

The attached screenshot is useful but is not, by itself, proof of temporal
behavior. A still image can show a coarse/haloed edge, while "dancing" requires
a frame sequence to measure. The backlog therefore treats the proposed causes
as ranked hypotheses and makes a reproducible video baseline the first gate.

## 2. Evidence from the reported run

The preview overlay in the screenshot reports:

- capture is approximately `15.0/30 fps`;
- output is approximately `29.7/30 fps`;
- segmentation is `mediapipe/cpu`;
- the camera reports `1280x720 MJPG` and `30.0 fps`;
- the pipeline warns `CAPTURE BELOW TARGET`;
- the camera frame is already `1280x720`, with the schema-v1 `stretch` policy;
- the video backdrop is `24.0 fps`, container-timed, with about `38%` skipped.

These observations matter:

1. The output sink is meeting its 30 FPS contract by repeating already-rendered
   frames when no new camera frame exists. Only about 15 distinct camera,
   segmentation, backdrop, and composite updates occur per second.
2. The `38%` backdrop skip figure is consistent with rendering a 24 FPS video
   backdrop at roughly 15 distinct composite updates per second. The output
   frame repeats do not advance the backdrop.
3. The active segmenter is MediaPipe, not RVM. The code and documentation
   describe RVM as the preferred true-alpha, recurrent video-matting backend.
4. The preview is already showing the processed application frame. The primary
   defect is therefore likely upstream of meeting-app compression, although the
   virtual-camera/consumer path still needs an isolation test.
5. Matching input and output dimensions make an extra full-frame resize an
   unlikely primary cause in this run. The legacy `stretch` policy should still
   be corrected through the existing visual-consistency rollout, but it does not
   explain a time-varying contour when both sides are 1280x720.

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
| Output cadence | `Pipeline._loop()` processes each new capture once and repeats `last_output` on empty reads | 30 FPS output contains only ~15 unique mattes in the reported run; this makes every contour change persist for two output frames |
| MediaPipe time | `MediaPipeSegmenter.segment()` increments `_ts_ms` by a fixed 33 ms | The VIDEO-mode model is told that a 15 FPS stream is 30 FPS, and capture gaps/restarts are hidden |
| MediaPipe resize | A returned mask with the wrong shape is resized without an explicit interpolation policy | The intended soft-alpha resampling contract and exact dimensions are not observable or frozen by tests |
| RVM time/state | RVM feeds recurrent state forward and resets it on a size change or GPU recovery | A camera restart/discontinuity at the same dimensions can reuse state from an unrelated temporal sequence |
| Mask refiner | Watershed edge snapping, mask shift, Gaussian feathering, then adaptive EMA | Watershed is independently solved per frame; the EMA reduces its hold at pixels whose alpha changes, including moving/jittering contours |
| RVM postprocess | RVM disables watershed, blur, and temporal EMA, retaining only mask shift | This protects fine matte detail but leaves no configurable safety net for residual RVM shimmer and makes one config field have backend-dependent effects |
| Light wrap | A blurred copy of the current backdrop is mixed into the soft edge band; default strength is `0.25` | Motion or brightness changes in a video backdrop can change edge color even when alpha position is stable, creating perceived shimmer |
| Backend fallback | `auto` tries RVM, then MediaPipe, then heuristic; only heuristic is reported as a segmentation fallback | Operators are not warned that the preferred matting tier was unavailable when MediaPipe is selected |
| Quality controls | The UI exposes threshold, RVM detail, blur, shift, smoothing, edge refine, and light wrap | `threshold` currently affects only the heuristic backend, and several controls are silently neutralized for RVM; operators can believe a change affected the active path when it did not |

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
more than one of them.

### H1 — MediaPipe contour instability is exposed by the current postprocessor

**Confidence: high.**

The reported run uses the coarser MediaPipe selfie-segmentation backend.
Per-frame confidence changes are passed through a spatial watershed operation.
The current temporal EMA intentionally approaches zero hold as local mask
difference grows. That is useful for large motion, but a one-to-several-pixel
contour displacement is exactly where stronger correspondence-aware temporal
support is needed.

Expected signature: the raw/refined alpha contour changes on stationary or
slow-moving head/shoulder regions while the source pixels remain aligned.

### H2 — Low unique-frame cadence makes each contour update more visible

**Confidence: high as an amplifier, low as the sole cause.**

The pipeline emits near 30 FPS but creates only about 15 new composites per
second. Repeating is the correct safe behavior for a missing capture frame, and
it does not invent new matte noise. It does, however, turn a small 15 Hz
boundary change into a more noticeable hold/jump pattern. The 24 FPS background
also updates at the unique-camera-frame cadence.

Expected signature: every second output frame is pixel-identical or nearly so;
contour changes line up with each new capture sequence.

### H3 — Incorrect temporal timestamps and missing discontinuity resets

**Confidence: medium-high.**

MediaPipe advances a fixed 33 ms per processed frame even at 15 FPS. Neither
MediaPipe nor the general refiner receives capture time. RVM state is
resolution-bound but not explicitly capture-generation-bound.

Expected signature: behavior changes after stalls/restarts or when replaying the
same frames with different cadence; stabilization strength varies with FPS.

### H4 — Per-frame watershed edge snapping moves the boundary

**Confidence: medium.**

Marker watershed is useful for a coarse binary mask but is solved independently
on every camera frame. Sensor noise, compression blocks, autofocus, exposure,
and weak foreground/background contrast can change the selected image edge.
Its fixed eight-pixel band is also resolution-dependent in physical terms.

Expected signature: the raw MediaPipe mask is steadier than the refined mask, or
the refined contour jumps toward different nearby gradients.

### H5 — Video-dependent light wrap creates edge-color shimmer

**Confidence: medium as a secondary artifact.**

At default settings, the current video backdrop is blurred and mixed into the
subject's soft edge. Backdrop motion therefore changes edge color. Users may
describe that as a dancing border even when the alpha contour itself is stable.

Expected signature: alpha boundary metrics remain stable, but edge-band RGB
changes fall substantially with `compositing.light_wrap: 0`.

### H6 — The camera is really producing 15 FPS

**Confidence: high; cause unknown.**

The driver reports 30 FPS, while the capture worker measures roughly 15 FPS.
Possible causes include low-light auto-exposure, a backend/format mismatch,
USB bandwidth, decoding cost, or CPU scheduling pressure. The capture worker is
independent, so diagnosis must use device/system evidence instead of assuming
the segmenter is responsible.

Expected signature: raw capture alone also runs at 15 FPS, or capture FPS
recovers when inference is disabled / lighting or the device mode changes.

### H7 — Downstream scaling/encoding adds a second artifact

**Confidence: low for the preview symptom, still worth isolating.**

The local preview already contains the reported defect, so the virtual-camera
consumer is unlikely to be the primary cause. It can still add ringing,
chroma-related edge errors, or another resize.

Expected signature: a processed-hub frame is stable while the virtual-camera
consumer recording is not.

## 5. Target quality contract

The exact numeric thresholds are provisional until `MATTE-0.2` records the
baseline, but implementation and review should use this contract:

- Static and slow-moving boundaries must not visibly oscillate.
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
- Quality defaults must be changed only after CPU/GPU, 15/30/60 FPS, motion,
  restart, and downstream-consumer qualification.

## 6. Delivery plan

| ID | Task | Phase | Priority | Size | Depends on |
| --- | --- | ---: | ---: | ---: | --- |
| MATTE-0.1 | Add a privacy-aware raw/mask/composite replay bundle | 0 | P0 | M | — |
| MATTE-0.2 | Define metrics, fixtures, and baseline report | 0 | P0 | L | MATTE-0.1 |
| MATTE-0.3 | Run the backend/postprocess/cadence ablation matrix | 0 | P0 | M | MATTE-0.2 |
| MATTE-0.4 | Publish immediate operator mitigations | 0 | P0 | S | MATTE-0.3 |
| MATTE-1.1 | Carry capture sequence and timestamp with each frame | 1 | P0 | L | MATTE-0.2 |
| MATTE-1.2 | Add timestamp/reset semantics to the segmenter contract | 1 | P0 | L | MATTE-1.1 |
| MATTE-1.3 | Correct MediaPipe timing and soft-mask resampling | 1 | P0 | M | MATTE-1.2 |
| MATTE-1.4 | Reset RVM/refiner state on discontinuities | 1 | P0 | M | MATTE-1.2 |
| MATTE-1.5 | Preserve transactional hot-activation behavior | 1 | P0 | M | MATTE-1.2, MATTE-1.4 |
| MATTE-2.1 | Implement elapsed-time, motion-aware boundary stabilization | 2 | P0 | XL | MATTE-0.3, MATTE-1.3, MATTE-1.4, MATTE-1.5 |
| MATTE-2.2 | Make spatial edge refinement stable and resolution-aware | 2 | P1 | L | MATTE-0.3, MATTE-2.1 |
| MATTE-2.3 | Introduce explicit backend-specific matte policies | 2 | P1 | M | MATTE-2.1, MATTE-2.2 |
| MATTE-2.4 | Bound dynamic light-wrap edge shimmer | 2 | P1 | M | MATTE-0.3 |
| MATTE-2.5 | Qualify RVM detail/performance profiles | 2 | P1 | M | MATTE-0.2 |
| MATTE-3.1 | Diagnose and recover the 15 FPS capture path | 3 | P0 | L | MATTE-0.1 |
| MATTE-3.2 | Expose unique-frame cadence and mismatch health | 3 | P1 | M | MATTE-1.1, MATTE-3.1 |
| MATTE-3.3 | Decide whether output-rate matte interpolation is warranted | 3 | P2 | M | MATTE-2.1, MATTE-3.1 |
| MATTE-4.1 | Report backend quality tier and selection/fallback reasons | 4 | P0 | M | MATTE-0.3 |
| MATTE-4.2 | Add honest quality presets and backend-aware controls | 4 | P1 | L | MATTE-2.3, MATTE-4.1 |
| MATTE-4.3 | Add local matte diagnostic views and telemetry | 4 | P1 | L | MATTE-1.1, MATTE-2.1 |
| MATTE-5.1 | Add deterministic temporal unit/regression tests | 5 | P0 | L | MATTE-2.3, MATTE-2.4, MATTE-3.2, MATTE-4.1 |
| MATTE-5.2 | Add end-to-end visual qualification | 5 | P0 | L | MATTE-2.5, MATTE-3.2, MATTE-4.3, MATTE-5.1 |
| MATTE-5.3 | Qualify performance and platform behavior | 5 | P0 | L | MATTE-3.1, MATTE-5.2 |
| MATTE-5.4 | Roll out defaults, migration, documentation, and rollback | 5 | P1 | M | MATTE-4.2, MATTE-5.3 |

The minimum useful release slice is `MATTE-0.1` through `MATTE-1.5`,
`MATTE-2.1`, `MATTE-3.1`, `MATTE-4.1`, and the applicable Phase-5 gates.
Do not wait for optional output-rate interpolation to deliver the core fix.

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
  - final composites;
  - monotonic capture timestamp and sequence;
  - capture generation/geometry generation;
  - effective config and active backend/device;
  - per-stage timing;
  - backdrop frame identity/timestamp where applicable.
- Use a versioned, documented manifest and lossless masks (`.npy`/`.npz` or
  lossless 16-bit representation). Do not use JPEG for metric-authoritative
  alpha data.
- Add an offline replay command/module that bypasses live capture and feeds the
  recorded frames and timestamps through the selected segmentation/refinement/
  compositing path.
- Support a final-composite-only capture mode for downstream comparisons, but
  mark it insufficient for matte metrics.
- Bound recording by duration and size; use owner-only permissions and an
  explicit output directory.

**Implementation notes:**

- Keep diagnostic persistence outside normal `FrameHub` history. The hub is a
  latest-frame transport and must not become an unbounded recorder.
- Preserve exact unique input sequence. Do not synthesize the output repeats
  into the model-input track.
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
- Raw/refined masks remain numerically identical after round-trip storage.
- Recording stops at configured duration/size without affecting the live
  output contract.
- Recording is off by default and no raw pixels/masks appear in logs or public
  `/status`.
- Tests cover interrupted writes, malformed manifests, path handling, and
  owner-only permissions on supported platforms.

### MATTE-0.2 — Define metrics, fixtures, and baseline report · L

**Goal:** convert "dancing border" into metrics that distinguish jitter,
ghosting, spatial error, color shimmer, and cadence.

**Context:** A lower frame-to-frame mask difference is not automatically
better—freezing a mask scores well while being unusable during motion. Metrics
must compensate for source motion and evaluate both stationary and moving
segments.

**Deliverables:**

- Build an offline evaluator that produces per-frame and aggregate:
  - unique-input FPS and output repeat ratio;
  - raw/refined alpha temporal absolute difference;
  - optical-flow- or registration-compensated alpha difference;
  - contour displacement using signed-distance fields, including p50/p95;
  - subject area drift during stationary segments;
  - soft-edge width and uncertain-pixel fraction (`0.05 < alpha < 0.95`);
  - motion trail/lag area behind the current contour;
  - edge-band RGB variation with alpha held/compensated;
  - optional alpha SAD/MSE/gradient error against ground-truth mattes;
  - segmentation/refinement/composite p50/p95 time and memory.
- Create generated deterministic clips for:
  - a static shape with noisy confidence;
  - one- and two-pixel alternating contour jitter;
  - translation/rotation with known ground truth;
  - fast motion and occlusion;
  - fine structures and semi-transparent edges;
  - a dynamic backdrop with constant alpha to isolate light wrap;
  - 15/30/60 FPS and irregular timestamps;
  - repeated output frames and sequence gaps.
- Add one consented/licensed local qualification clip set. Do not commit
  private user footage; CI can use generated/licensed fixtures while local
  evidence is referenced by manifest digest.
- Produce `docs/matte-quality-baseline.md` (or JSON + review Markdown) comparing
  the reported-style MediaPipe CPU configuration with the current RVM path.

**Provisional gates to ratify with the baseline:**

- On a stationary 720p boundary, p95 compensated contour displacement should
  be no more than `1.5 px` and at least `40%` lower than baseline.
- Stabilization must not increase fast-motion trail area by more than `10%`
  relative to the unstabilized backend, and no previous contour should remain
  visibly dominant for more than one unique input interval.
- Static subject-area drift should be no more than `1%` of the foreground area.
- Fine-detail/ground-truth alpha error must not regress beyond the agreed
  fixture tolerance.
- Dynamic light wrap must be evaluated separately from alpha stability.

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
- The baseline report records hardware, dependency versions, backend/device,
  effective detail/resampling settings, input cadence, and config.

### MATTE-0.3 — Run the backend/postprocess/cadence ablation matrix · M

**Goal:** identify which changes materially reduce the reported defect before
replacing algorithms or changing defaults.

**Context:** The likely causes interact. For example, disabling light wrap can
make an edge look calmer without changing alpha; disabling watershed can help
one camera and harm a coarse mask on another. The experiment must retain one
variable at a time.

**Matrix:**

- backend/device:
  - MediaPipe CPU;
  - MediaPipe GPU delegate where supported;
  - RVM CPU;
  - RVM with each qualified accelerator;
- RVM downsample: auto and a small qualified set such as `0.4`, `0.5`, `0.67`,
  subject to memory/performance;
- edge refine: on/off;
- mask blur: `0`, current default, and one wider candidate;
- temporal smoothing: `0`, current default, and candidate time-based policies;
- mask shift: `0`, `-1`, and `-2`;
- light wrap: `0` and current default;
- capture cadence: fixed 15/30/60 and irregular/gapped replay;
- output cadence: equal to unique input and 2x repeated output.

**Required decisions:**

- Confirm whether watershed improves or worsens MediaPipe temporal contour
  error on real clips.
- Quantify how much of the visible symptom is alpha motion versus edge-color
  motion.
- Confirm whether RVM at a sustainable profile meets the visual target on the
  target hardware.
- Decide whether the first production fix should tune MediaPipe, make RVM more
  available, or both.
- Record the effect of the 15-to-30 repeat cadence without claiming it creates
  new alpha values.

**Acceptance criteria:**

- Results use the same source frame sequence/timestamps across variants.
- A report includes metrics, representative contact sheets/short clips,
  performance, and a recommended production candidate.
- The recommendation names rejected candidates and the reason (jitter,
  ghosting, detail loss, performance, platform availability, or complexity).

### MATTE-0.4 — Publish immediate operator mitigations · S

**Goal:** give users a reversible quality improvement while engineering work is
in progress.

**Context:** The screenshot already exposes actionable runtime facts. An
operator can test RVM, improve capture cadence, disable light wrap to isolate
color shimmer, or disable edge refine if the ablation proves it unstable.
Values must come from `MATTE-0.3`; do not present guesses as universal fixes.

**Deliverables:**

- Add a troubleshooting section covering:
  - how to verify `segmentation_backend`, device, and measured capture FPS;
  - how to install/rebuild with the qualified RVM extra;
  - how to test MediaPipe GPU where supported;
  - how to distinguish alpha jitter from light wrap by setting light wrap to
    zero temporarily;
  - how to test `edge_refine` and `mask_shift` safely;
  - how lighting/auto-exposure can reduce webcam cadence, without automatically
    writing camera controls;
  - how to match output FPS to a known sustained capture rate for diagnosis.
- Include rollback values and warn which config changes rebuild resources.

**Acceptance criteria:**

- Every recommendation is supported by the ablation report.
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
- Keep stateless segmenters as no-op implementations.

**Acceptance criteria:**

- Segmenters never receive a non-monotonic effective timestamp.
- The exact first frame after a discontinuity starts from clean temporal state.
- Repeat output does not call `segment`, `refine`, or `reset`.
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
the object is replaced or explicitly called.

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

- Define effective policy for:
  - true-alpha recurrent mattes (RVM);
  - confidence-mask video segmentation (MediaPipe);
  - binary/coarse fallback masks (heuristic);
  - null/passthrough.
- Store/report configured and effective values separately.
- Decide from evidence whether RVM needs a very light residual temporal policy
  or should remain model-only.
- Remove or relabel controls that do not apply to an active backend.
- Make `threshold` either useful for a specifically documented alpha remap/
  hysteresis policy or explicitly heuristic-only. Do not hard-threshold RVM
  hair mattes as a shortcut.

**Acceptance criteria:**

- Effective policy is visible in status/diagnostics.
- UI/API documentation identifies backend applicability.
- No quality control is silently ignored without an effective-state indication.
- Backend switch transactionally selects a fresh compatible policy/state.

### MATTE-2.4 — Bound dynamic light-wrap edge shimmer · M

**Goal:** prevent a moving video backdrop from making the subject edge flicker
or pulse independently of the matte.

**Context:** Current light wrap uses the current blurred backdrop inside
`4 * alpha * (1 - alpha)`. With a video backdrop, edge color can change on each
unique composite even when the subject and alpha are static.

**Deliverables:**

- Measure current edge-band RGB variance with a fixed alpha/foreground and
  dynamic backdrop.
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

### MATTE-2.5 — Qualify RVM detail/performance profiles · M

**Goal:** make the preferred true-alpha backend a practical option at a
sustainable cadence.

**Context:** At 1280x720, auto RVM uses a long-edge internal resolution near
512 pixels (`rvm_downsample ~= 0.4`). Increasing the ratio may improve fine
edges but costs inference time/memory. The reported environment instead chose
MediaPipe CPU, likely because the RVM runtime was not installed or prepared.

**Deliverables:**

- Benchmark RVM CPU and every supported accelerator at target canvas sizes and
  downsample ratios.
- Report:
  - alpha/temporal quality metrics;
  - inference p50/p95;
  - full-frame p50/p95;
  - memory/VRAM;
  - sustained capture and output cadence;
  - provider fallback behavior.
- Define `performance`, `balanced`, and `quality` RVM profiles only if the data
  shows stable cross-device meaning.
- Expose the effective resolved downsample ratio, not only configured `0=auto`.
- Do not make an unsustainable high detail ratio the global default.

**Acceptance criteria:**

- Each recommended profile has explicit hardware/canvas qualification.
- Auto resolution is observable and covered by tests.
- A profile that misses its declared frame budget is rejected or clearly
  degraded; it is not advertised as real-time.

### MATTE-3.1 — Diagnose and recover the 15 FPS capture path · L

**Goal:** restore the requested unique-frame cadence where the camera and system
can actually supply it, and give a precise reason when they cannot.

**Context:** The screenshot shows a negotiated/reported 30 FPS MJPG mode but a
measured 15 FPS stream. Repeating output masks the sink contract but cannot
restore motion or matte updates.

**Diagnostic matrix:**

- capture-only (`segmentation: none`, null output);
- MediaPipe CPU/GPU;
- RVM CPU/GPU;
- local preview off/on;
- virtual camera off/on;
- 640x360, 1280x720, and one supported higher mode;
- MJPG versus backend default where supported;
- adequate lighting versus the reported environment;
- camera auto-exposure versus a read-only observed control report;
- same device through a native capture tool for comparison.

**Deliverables:**

- Correlate capture timestamps with `read_ms`, CPU utilization, per-stage
  timing, drops, USB/backend logs, and camera controls.
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

- The target setup sustains at least 90% of requested capture FPS when hardware
  supports it, or reports a specific actionable limitation.
- Capture-only measurements show whether segmentation is causal.
- No fix creates an uncontrolled feedback loop with camera auto-exposure/
  white balance.
- Regression tests retain bounded capture worker shutdown/recovery behavior.

### MATTE-3.2 — Expose unique-frame cadence and mismatch health · M

**Goal:** make it obvious that output FPS and visual-update FPS are different.

**Context:** The current overlay shows input and output FPS and a general
`CAPTURE BELOW TARGET` warning, but it does not state repeat ratio,
segmentation-update FPS, or the temporal impact of an input/output mismatch.

**Deliverables:**

- Add stable status fields for:
  - capture sequence and sequence gaps;
  - unique composite/segmentation FPS;
  - output repeat FPS/ratio;
  - last unique-frame age;
  - actual capture timestamp delta p50/p95;
  - matte reset count/last reason;
  - effective matte policy;
  - configured/effective RVM ratio;
  - configured versus effective MediaPipe timing source.
- Split `segmentation_ms` from `matte_refinement_ms` if needed.
- Add a warning such as `VISUAL UPDATES 15 FPS; OUTPUT REPEATS TO 30 FPS`.
- Preserve API/OpenAPI type consistency and sanitized/path-free status.

**Acceptance criteria:**

- The reported run shape would clearly show ~15 unique visual updates, ~30
  output sends, and ~50% repeats.
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

### MATTE-4.1 — Report backend quality tier and selection/fallback reasons · M

**Goal:** tell operators when `auto` selected a lower-quality backend and how to
obtain the preferred one.

**Context:** `create_segmenter` logs why RVM is unavailable, but status marks a
fallback only when the final backend is heuristic. In the screenshot,
MediaPipe is working as designed, yet it is also a degradation from the
preferred RVM tier and no warning explains that.

**Deliverables:**

- Return/store structured selection attempts:
  - requested backend;
  - selected backend;
  - quality tier (`matting`, `segmentation`, `heuristic`, `none`);
  - availability/preparation result for each preferred candidate;
  - sanitized fallback reason/category;
  - active device/provider.
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

- The reported configuration would state that `auto` selected MediaPipe CPU
  because RVM was unavailable, with an actionable sanitized reason.
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

- Add local-only preview modes for:
  - raw camera;
  - raw model alpha;
  - refined/stabilized alpha;
  - alpha over source;
  - uncertain boundary band;
  - frame-to-frame/flow-compensated instability heatmap.
- Show unique input sequence/delta, raw/refined temporal metrics, reset reason,
  and stage timings.
- Never send these views to the configured virtual-camera output unless an
  explicit diagnostic sink is selected.
- Do not expose mask images through unauthenticated/public endpoints.
- Keep overlay drawing on copies, consistent with current preview ownership.

**Acceptance criteria:**

- A replay can visually isolate H1/H4 (mask motion) from H5 (edge color).
- Enabling diagnostics does not mutate the production frame or mask.
- Normal mode pays no material full-frame diagnostic cost.
- Tests assert no diagnostic image reaches normal output/hub publications.

### MATTE-5.1 — Add deterministic temporal unit/regression tests · L

**Goal:** prevent quality fixes from regressing across refactors.

**Required test groups:**

- timestamp conversion at 15/30/60 FPS, same-ms quantization, irregular gaps,
  and non-monotonic input;
- reset on generation, geometry, provider fallback, long gap, and config
  commit;
- no reset on ordinary cadence jitter or output repeats;
- MediaPipe mask validation and interpolation;
- RVM recurrent state on same-size restart;
- temporal stabilization:
  - stationary jitter reduction;
  - known translation/rotation;
  - fast motion;
  - occlusion/disocclusion;
  - fine/soft edges;
  - all-zero/all-one/tiny subject masks;
  - NaN/out-of-range rejection;
- spatial refinement with alternating gradients and resolution scaling;
- dynamic light wrap with fixed alpha;
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

**Qualification set:**

- subject appearances: bald/short hair, long/fine hair, glasses, facial hair,
  dark/light clothing, skin-tone and lighting diversity;
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
- Review explicitly checks halo, edge shimmer, ghost trail, cutout sharpness,
  hair retention, and motion cadence.

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

- capture FPS and drops;
- unique segmentation/composite FPS;
- output FPS/repeats;
- stage p50/p95/p99;
- end-to-end age/latency;
- CPU utilization and memory;
- GPU utilization/VRAM where available;
- sustained run, restart, hot patch, and shutdown resource behavior.

**Provisional performance gates:**

- Balanced profile sustains at least 90% of declared unique-input target on its
  qualified hardware/canvas.
- Output sink remains at least 90% of target without unbounded latency growth.
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

**Acceptance criteria:**

- Old persisted configs have documented, tested behavior.
- New installs select only a qualified sustainable profile.
- Rollback does not require deleting user config or model caches.
- Release evidence links baseline, ablation, visual, performance, platform,
  privacy, and migration results.

## 8. Suggested implementation sequence

1. Record the reported setup and establish metrics (`MATTE-0.1`–`0.3`).
2. Publish only evidence-backed configuration mitigations (`MATTE-0.4`).
3. Land frame identity/time and reset contracts (`MATTE-1.1`–`1.5`).
4. Correct MediaPipe timing before comparing new temporal algorithms.
5. Implement and qualify motion-aware stabilization (`MATTE-2.1`), then decide
   the spatial-refinement and light-wrap changes from isolated metrics.
6. Diagnose capture cadence in parallel with algorithm work (`MATTE-3.1`).
   The target is more unique source frames, not merely a 30 FPS send counter.
7. Improve selection/status/UI honesty (`MATTE-4.1`–`4.3`).
8. Complete deterministic, visual, performance, platform, and migration gates
   before changing defaults.

## 9. Pull-request boundaries

Keep changes reviewable. A reasonable split is:

1. replay bundle and metrics only;
2. capture envelope and pipeline plumbing;
3. segmenter timestamp/reset contract;
4. MediaPipe timing/resampling;
5. RVM/refiner discontinuity reset and activation lifecycle;
6. new stabilizer behind an off/experimental mode;
7. spatial refinement and light-wrap changes;
8. status/selection metadata;
9. WebUI presets/backend-aware controls;
10. qualification evidence and default/migration change.

Do not combine an algorithm introduction, default flip, installer-profile
change, and migration in one unreviewable patch.

## 10. Definition of done

The reported problem is considered resolved only when:

- a representative clip reproduces the old defect and passes the ratified new
  temporal/spatial gates;
- MediaPipe uses actual capture time and all temporal backends reset correctly;
- the chosen stabilizer reduces stationary/slow boundary jitter without visible
  motion trails or lost fine detail;
- dynamic video light wrap cannot masquerade as alpha instability;
- the target device either sustains the requested unique capture FPS or reports
  a specific limitation and an actionable configuration;
- status distinguishes capture/unique visual updates/output repeats and reports
  the selected quality tier/fallback reason;
- in-memory preview and qualified virtual-camera consumer recordings agree;
- remote privacy, canonical geometry/color contracts, hot activation, bounded
  recovery, and shutdown guarantees remain intact;
- CPU/GPU/platform, visual, performance, migration, and rollback evidence is
  complete;
- documentation gives users a supported quality path instead of requiring
  trial-and-error tuning.
