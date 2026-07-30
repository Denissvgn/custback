# VIS-4.2 visual-consistency qualification runbook

## Authority and current disposition

This procedure is the release evidence path for automatic background color
correction and geometry scaling. It deliberately separates three kinds of
evidence:

1. `shared-ci-deterministic` proves reproducible production contracts,
   operation counts, Phase-0 visual comparisons, and bounded retained state.
2. `pinned-reference-runner` proves the reviewed 720p/1080p timing budgets.
3. Physical-camera evidence proves real capture backends, virtual-camera
   transports, preview/API equivalence, and meeting-app consumption.

All three are required for `release_qualified=true`. CI's deterministic job is
necessary but is not permission to enable a policy by default. At the time this
runbook was added, no authoritative pinned-runner and physical matrix report
had been supplied, so the release disposition remains **pending / no-go**.
Local diagnostic timings are observations only, especially while production
performance work is changing the candidate path.

The reviewed contract is
[`scripts/release/visual-qualification-manifest.json`](../scripts/release/visual-qualification-manifest.json).
The validator is
[`scripts/release/visual_consistency_qualification.py`](../scripts/release/visual_consistency_qualification.py).
Do not weaken or replace the default manifest for a release claim.

## What the harness measures

The deterministic portion uses production geometry, color estimation,
compositing, pipeline loop, hub statistics, video pacing, and camera pacing
seams. It checks:

- Phase-0 geometry improvement and color improvement/preservation thresholds;
- byte-exact foreground and background alpha endpoints;
- at most one common-path geometry resize at 720p and 1080p;
- the 192-pixel maximum color-analysis long edge;
- one analysis and one composite for six identical output frames;
- decoder/camera pacing operation counts;
- a 10,000-frame retained-state soak.

The calibrated portion records raw per-frame samples and p50/p95 for:

- geometry;
- linear conversion;
- color analysis;
- compositing;
- output send;
- total frame time.

It also records effective FPS, FPS attainment, baseline/candidate deadline
misses, RSS growth, tracemalloc peak, and the existing production
`FrameHub.stats_dict` EWMA fields. The validator recomputes percentiles, FPS,
attainment, deadline misses, overhead, and every budget decision from the
hashed raw samples.

The synthetic deterministic and calibrated fixtures do not substitute for
capture hardware or consumer applications. The physical matrix closes that
boundary.

## Reviewed budgets

| Requirement | Limit |
| --- | ---: |
| Relative total-frame p95 overhead | 20% |
| Added p95, 720p30 and 720p60 | 5 ms |
| Added p95, 1080p30 | 8 ms |
| Deadline-miss increase | 1 percentage point |
| FPS attainment | 99% |
| Measured frames per tier | 300 minimum |
| Post-warmup RSS/tracemalloc growth in 10k soak | 5 MiB |
| Fixed semantic session state | 24 MiB (23,440,384 bytes accounted) |
| Color-analysis long edge | 192 px |

The fixed-state allowance is separate from post-warmup growth. It covers the
production 16 MiB fail-closed privacy replay index, one maximum 1080p BGR
canvas (6,220,800 bytes), and one conservative 192×192, three-channel float32
color-analysis cache (442,368 bytes). The exact accounted total is 23,440,384
bytes, leaving 1,725,440 bytes of bounded headroom below 24 MiB. It must not be
hidden by using a smaller qualification-only replay capacity or omitting the
immutable-image analysis cache. Growth after warmup remains limited to 5 MiB.

## 1. Deterministic evidence

Use a clean checkout and Python 3.12 with the development dependencies:

```bash
python -m pip install -e '.[dev]'
python -m pip check
mkdir -p build/visual-qualification
python scripts/release/visual_consistency_qualification.py run \
  --output build/visual-qualification/deterministic.json \
  --contact-sheet build/visual-qualification/contact-sheet.png
python scripts/release/visual_consistency_qualification.py validate \
  --report build/visual-qualification/deterministic.json \
  --claim deterministic
```

Do not pass a reduced `--soak-frames` value for evidence. That option exists
only for focused developer tests. The release validator independently requires
at least the manifest's 10,000 frames.

CI runs the same 10,000-frame command, revalidates the emitted report, and
uploads both report and contact sheet as
`visual-qualification-deterministic`.

## 2. Pinned Linux performance evidence

The reviewed reference runner is Linux because authoritative RSS collection is
required. Pin the runner identity and CPU description in the command. Keep its
hardware, OS image, Python/dependency lock, CPU governor/affinity, cooling, and
background workload stable between baseline and candidate runs. Record any
runner-image revision in the runner ID.

```bash
python scripts/release/visual_consistency_qualification.py run \
  --output build/visual-qualification/calibrated.json \
  --contact-sheet build/visual-qualification/calibrated-contact-sheet.png \
  --calibrated \
  --measured-frames 300 \
  --warmup-frames 5 \
  --pinned-runner \
  --runner-id visual-linux-x64-01-image-REVISION \
  --reference-cpu 'EXACT CPU MODEL / REVIEWED GOVERNOR'
```

The command writes the report and exits nonzero if any calibrated tier fails.
That nonzero result is the gate result. Never wrap it in `|| true`.

`--observation-only` is an explicit diagnostic escape hatch: it retains a
failing report but exits zero. An observation-only result has no qualification
authority and must never be used by a CI or release gate.

Inspect each tier before proceeding:

- exactly 300 or more raw samples exist for baseline and candidate;
- stage samples sum to total-frame samples;
- RSS is non-null;
- all four budget booleans are true;
- `status` is `pass`;
- report authority is `pinned-reference-runner`.

Do not copy favorable summaries into another report. Preserve the raw samples,
their digest, the runner identity, and the original report as one evidence
unit.

## 3. Physical camera and consumer matrix

Generate the pending schema:

```bash
python scripts/release/visual_consistency_qualification.py template \
  --output build/visual-qualification/physical-template.json
```

Use at least two genuinely distinct webcam manufacturer/model pairs. Their
recorded auto-white-balance/auto-exposure states must form at least two
different profiles. Use at least two distinct meeting application names, and
record exact versions. Every declared webcam and consumer must be referenced by
at least one matrix row; a consumer's OS must equal its row OS.

The required rows are exact:

| Row | Capture | Output | Consumer transport |
| --- | --- | --- | --- |
| `linux-v4l2-pyvirtualcam` | Linux V4L2 | pyvirtualcam | v4l2loopback |
| `macos-obsvcam-pyvirtualcam` | macOS AVFoundation | pyvirtualcam | OBS Virtual Camera |
| `windows-msmf-pyvirtualcam` | Windows MSMF | pyvirtualcam | OBS Virtual Camera |
| `windows-dshow-pyvirtualcam` | Windows DSHOW | pyvirtualcam | OBS Virtual Camera |
| `windows-native` | Windows MSMF | native | Media Foundation virtual camera |

Do not relabel an untested backend. If a required backend or transport is
unavailable, leave the row pending or mark it failed; release qualification
must fail closed. The native Windows row is required while that supported
output is distributed.

For each row, collect four unique artifacts. Paths and file bytes must not be
reused across rows.

| Artifact ID | Media type | Required content |
| --- | --- | --- |
| `contact_sheet` | `image/png` | baseline/candidate fixture and ROI comparison |
| `preview_capture` | `image/png` | application preview output |
| `api_snapshot` | `application/json` | API state/stats and exact test metadata |
| `consumer_capture` | `image/png` | frame captured by the named meeting app |

Each artifact entry contains `filename`, lowercase SHA-256, exact byte count,
and `media_type`. Files must be regular, non-symlink files under the evidence
root. PNGs must decode with nonzero dimensions; JSON must contain a non-empty
object.

### Physical measurement method

Use the same source fixture, frame index, canvas size, mask, and ROIs for
baseline and candidate.

- `crop_edge_error_pixels`: place four edge/corner fiducials and report the
  largest absolute expected-versus-observed edge displacement.
- `orientation_error_degrees`: compare the labeled top/left fiducials; mirrored
  or rotated output is a failure. The limit is zero degrees.
- `transport_mean_absolute_error`: decode the consumer capture and compute the
  mean absolute BGR channel difference from the expected virtual-camera frame.
- `luminance_gap_reduction_percent`:
  `100 * (baseline_gap - candidate_gap) / max(baseline_gap, epsilon)`.
- `neutral_error_reduction_percent`: use the same reduction formula on the
  neutral-axis error from the fixed neutral ROI.
- Hue drift: circular hue distance in degrees on the fixed skin/clothing ROIs.
- Chroma drift: absolute normalized chroma change as a percentage on those
  ROIs.
- `steady_state_ev_delta_p95`: p95 absolute frame-to-frame EV-estimate delta
  after the documented warmup.
- `steady_state_wb_log2_delta_p95`: p95 absolute frame-to-frame log2 WB-gain
  delta after warmup.

Capture enough steady-state frames to expose oscillation; record the frame
count and warmup in `api_snapshot`. Set `no_temporal_oscillation` only when the
sequence has no sustained alternating/cycling correction. Set
`preview_api_equal` only after the preview and API frame/state agree for the
same generation. Set `all_consumers_match` only after every consumer named in
the row stays inside the crop/orientation/MAE limits.

The physical thresholds are:

- crop error at most 1 px;
- orientation error 0 degrees;
- transport MAE at most 3;
- luminance gap reduction at least 20%;
- neutral error reduction at least 10%;
- skin hue/chroma drift at most 5 degrees / 12%;
- clothing hue/chroma drift at most 8 degrees / 15%;
- steady-state EV/WB delta p95 at most 0.05 / 0.02.

## 4. Bind the release candidate

Keep all evidence immutable after hashing. In the calibrated report:

1. Replace the pending `physical` object with the completed physical object.
2. Set `candidate.status` to `bound`.
3. Set `candidate.manifest_filename` to a safe relative candidate-manifest
   path under the evidence root.
4. Set `candidate.manifest_sha256` to that file's SHA-256.
5. Set `release_qualified` to `true` only after calibrated and physical
   sections both pass.

The candidate manifest must be a non-empty JSON object with:

- `source.commit` equal to the qualification report's source commit;
- a non-empty `artifacts` array;
- for every artifact: non-empty `id` and `filename`, lowercase SHA-256, and
  positive `size`.

Then validate:

```bash
python scripts/release/visual_consistency_qualification.py validate \
  --report build/visual-qualification/calibrated.json \
  --claim release \
  --evidence-root build/visual-qualification \
  --expected-commit EXACT_QUALIFIED_COMMIT
```

Release validation rejects:

- a custom or modified requirements manifest;
- a dirty source checkout;
- a source commit that is not in the trusted repository history;
- source-file digests that do not match the report's historical commit;
- a candidate manifest bound to another commit;
- missing, duplicate, symlinked, malformed, or digest-mismatched evidence;
- summarized timings that differ from raw samples;
- unavailable authoritative RSS;
- incomplete physical rows or threshold failures;
- a short or growing soak;
- any `release_qualified` value other than literal `true`.

## Git-archive/prepack bridge

A staged `git archive` has no `.git` directory and cannot prove its own
history. The internal authorizing prepack path may point the validator at a
trusted checkout with these all-or-none variables:

```text
CUSTBACK_RELEASE_GIT_ROOT
CUSTBACK_RELEASE_SOURCE_COMMIT
CUSTBACK_RELEASE_SOURCE_TREE
```

The root must be an absolute, real, non-symlink repository top-level directory.
It must be clean, and its actual `HEAD` and tree must exactly match the supplied
commit/tree. The qualification's historical commit must be an ancestor and its
tree/source blobs must exist there. The staged tree remains the source of the
report and evidence files. Every nested candidate/contact-sheet/physical
artifact opened under this bridge must be inside the staged source root and
byte-match the same path at trusted current release `HEAD`.

These variables are an internal build bridge, not a way to accept a
self-asserted commit from an archive.

## Handoff checklist

- [ ] Default reviewed manifest is unchanged and its Phase-0 baseline hash
      matches.
- [ ] Deterministic 10,000-frame report and contact sheet pass independent
      validation.
- [ ] Pinned Linux run includes 300+ frames per tier and non-null RSS.
- [ ] All 720p30, 720p60, and 1080p30 budget checks pass.
- [ ] Two distinct webcams with differing auto-control profiles are used.
- [ ] Two or more distinct meeting applications are used.
- [ ] Every exact OS/backend/transport row passes with unique hashed artifacts.
- [ ] Candidate manifest and all release artifacts are bound to the qualified
      source commit.
- [ ] Strict release validation exits zero.
- [ ] Rollout approval references the immutable report/artifact digests.
