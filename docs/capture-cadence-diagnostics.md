# Capture cadence diagnosis

- Status: MATTE-3.1 capture-only harness implemented; physical-camera outcome
  unqualified
- Date: 2026-08-04
- Backlog owner: MATTE-3.1

## Current decision

`custback capture-diagnose` measures the production OpenCV camera reader and
canonical normalization path without constructing the normal video pipeline.
It does not create a segmenter, backdrop, compositor, preview, API, or output
sink. This makes a slow capture-only result independent of model and
compositor execution.

The checked-in implementation and tests qualify the harness, schema, timing
math, privacy boundary, and bounded shutdown behavior. They do **not** qualify
any physical camera, host, USB path, OpenCV backend, or 1280x720@30 mode. Until
an operator supplies real local hardware evidence, the project-level
MATTE-3.1 hardware outcome remains `hardware-evidence-required`.

An exit status of zero means that the measured capture-only run reached its
requested cadence. It is not, by itself, a hardware qualification claim.
Hardware acceptance is recorded separately under `qualification` in
`capture.json`.

## Run the capture-only harness

Stop the normal Custback process first. Most cameras allow only one active
reader, and a competing process would invalidate the comparison.

Use a new output directory:

```console
umask 077
custback capture-diagnose \
  --config ./private-runtime.yaml \
  --condition-id reported_light \
  --warmup-seconds 2 \
  --duration-seconds 10 \
  --output ./private-capture-reported-light
```

The command validates the runtime configuration but activates only the camera
and canonical-canvas values needed for acquisition and normalization.
Camera-only overrides are available without changing output cadence:

```console
custback capture-diagnose \
  --config ./private-runtime.yaml \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --pixel-format mjpeg \
  --condition-id adequate_diffuse_light \
  --output ./private-capture-adequate-light
```

Unlike the normal top-level `--fps` option, the diagnostic command's `--fps`
changes only the requested camera mode. The output directory must not already
exist. It is created owner-only and contains:

- `capture.json`, the authoritative versioned report; and
- `capture.md`, a compact review summary.

The default warm-up is 2 seconds. The measurement must be 5–60 seconds and
defaults to 10 seconds. Warm-up samples are excluded from the report.

## What is and is not measured

The reader records one bounded, pixel-free timing sample for every successful
publication:

| Boundary | Meaning |
| --- | --- |
| `read_ms` | OpenCV `VideoCapture.read()` wall time, including device pacing, transfer, and backend decode |
| `pre_normalization_ms` | validation plus first-frame mode/control observation before canonical normalization |
| `normalization_ms` | the production rotation, mirror, fit, crop/pad, and resize path |
| `publish_ms` | publication into the latest-frame slot |
| `total_ms` | the complete reader interval from before `read()` through publication |
| `reader_cpu_ms` | current reader-thread CPU time where the Python runtime exposes it |

The serialized trace uses relative monotonic completion offsets. It contains no
wall-clock time, pixels, frame hashes, camera index, device path, URL, or
credentials. Process CPU is reported as a percentage of one logical core; it
is not a system-wide utilization measurement.

OpenCV does not expose separate USB-transfer and compressed-frame decode
timings at the `read()` boundary. A long `read_ms` can therefore establish that
normalization is not the dominant cost, but cannot distinguish exposure
pacing, sensor/driver cadence, USB bandwidth, and backend decode by itself.
Likewise, an observed auto-exposure value does not prove that lighting caused
the cadence.

The report keeps three pacing domains separate:

- `pacing.capture` comes from capture-only read-completion timestamps;
- `pacing.processed_frames` is explicitly unmeasured unless a matched strict
  two-snapshot runtime evidence file is supplied; and
- `pacing.output` is always unmeasured because the harness opens no sink.

Output repeats never count as unique camera reads or processed frames.

## Report contract

`capture.json` has schema `custback.capture-diagnostic-report`, version 1. Its
root objects are:

| Object | Contents |
| --- | --- |
| `privacy` | assertions about omitted pixels, identities, credentials, and wall time |
| `capture_only_contract` | the stages opened and deliberately absent |
| `condition` | bounded comparison ID and local identity/verification flags |
| `requested` | camera request plus canonical canvas size |
| `negotiated` | backend, FourCC, reported mode, delivered size, and normalized size |
| `camera_controls` | generation-bound preserve-only observations |
| `measurement` | duration, reads, deliveries, overwrites, failures, restarts, stalls, and CPU |
| `timing` | full-run summaries, correlations, and relative per-frame trace |
| `pacing` | separately scoped capture, processing, and output cadence |
| `native_comparison` | digest-bound native-tool summary or an explicit absence |
| `full_runtime_comparison` | selected matched runtime fields or an explicit absence |
| `diagnosis` | evidence-bounded code, confidence, actions, and non-claims |
| `qualification` | the separate MATTE-3.1 hardware acceptance result |

Capture cadence includes both successful read-completion intervals and
full-window availability (`successful reads / measurement seconds`). The
acceptance gate requires both rates to reach the threshold; leading/trailing
boundary tolerances only detect starvation and never reduce the required
unique-read count. Cross-generation outages remain visible and are not folded
into an apparently healthy active-generation rate. Restarts, failures, stalls,
or geometry-generation changes make the run unstable for acceptance.

The diagnostic codes have deliberately narrow meanings:

| Code | Supported conclusion |
| --- | --- |
| `target-sustained` | this capture-only run met 90% of its requested FPS with matching geometry and no instability |
| `driver-reported-mode-mismatch` | negotiated/reported mode or delivered dimensions did not match the request |
| `opencv-capture-path-limited` | matched native evidence met the target while the Custback/OpenCV path did not |
| `shared-device-environment-or-backend-limit` | both matched capture-only paths were under target; the exact device, exposure, USB, or backend cause remains unresolved |
| `normalization-budget-limited` | canonical normalization consumed a material share of the frame budget |
| `capture-read-path-paced` | OpenCV read time tracks the under-rate result; native evidence is still needed for attribution |
| `unstable-capture` | failures, restarts, stalls, generation changes, or shutdown errors invalidate steady-rate attribution |
| `insufficient-capture-samples` | the run did not produce enough timestamp evidence |
| `capture-window-starvation` | samples exist, but leading or trailing gaps leave the bounded measurement window incomplete |
| `unresolved-capture-scheduling-or-backend-limit` | capture was under target without enough evidence for a narrower code |

If capture-only is under rate, segmentation was absent and therefore cannot be
the cause of that capture-only result. If capture-only meets target while a
full run does not, the report establishes only that some excluded workload or
contention matters; it does not single out segmentation without separate
stage evidence.

## MATTE-3.1 acceptance

The backlog's exact acceptance mode is a hardware-verified physical
1280x720@30 camera. A sustained result requires at least 27 unique reads/s,
matching negotiated and delivered dimensions, at least five seconds of
measurement, and no instability.

Do not pass `--hardware-verified` for CI, an assumed mode, or a driver property
that has not been checked on the physical device. The command rejects
synthetic capture outright. The flag is an explicit local attestation, not
automatic discovery.

Exact hardware acceptance requires all of the following:

- the explicit `--hardware-verified` local attestation;
- non-empty lowercase 64-character `--device-identity-sha256` and
  `--hardware-identity-sha256` bindings;
- a numeric camera index, including a numeric string, or a recognized local
  Linux camera path matching `/dev/videoN` or
  `/dev/v4l/by-id/SAFE_DEVICE_NAME`;
- the exact requested 1280x720@30 mode; and
- either the sustained target result or the evidence requirements for a
  specific actionable limitation.

Media files, URLs, RTSP/HTTP streams, and unrecognized device strings are
rejected before capture because they are not physical-camera evidence. Without
every acceptance prerequisite, the report remains:

```json
{
  "acceptance_satisfied": false,
  "outcome": "hardware-evidence-required"
}
```

For an actionable limitation, a negotiated mode mismatch can be corroborated
by the production backend's own readback. Other device/environment/backend
limitations require compatible same-device native evidence before satisfying
the alternative acceptance branch. A generic
`shared-device-environment-or-backend-limit` result remains non-qualifying:
both paths being slow corroborates the symptom, but does not isolate the
specific actionable limitation required for acceptance.

No physical-camera evidence is committed to this repository. Local reports
must be reviewed on the target hardware before anyone records MATTE-3.1 as
recovered or as an actionable hardware limitation.

## Native capture comparison

The command can join a separately recorded native-tool trace:

```console
custback capture-diagnose \
  --config ./private-runtime.yaml \
  --condition-id adequate_diffuse_light \
  --hardware-verified \
  --device-identity-sha256 DEVICE_DIGEST \
  --hardware-identity-sha256 HOST_DIGEST \
  --native-evidence ./private-native-capture.json \
  --output ./private-capture-with-native-comparison
```

Custback never launches the external tool. Record the native run separately
with output-rate conversion, repetition, and camera-control writes disabled,
then translate its host read-completion timestamps into an owner-only JSON
sidecar. Copy
[`capture-native-evidence-local-template.json`](capture-native-evidence-local-template.json)
and replace every `REPLACE_` value and illustrative timestamp.
The checked-in file is deliberately not evidence: its placeholder digests and
offset strings fail the strict parser until they are replaced with the
complete measured trace.

The accepted sidecar has schema `custback.native-capture-evidence`, version 1,
with exactly these root fields:

| Field | Requirement |
| --- | --- |
| `schema`, `version` | exact schema name and integer version 1 |
| `device_identity_sha256` | lowercase 64-character digest for the same physical device |
| `hardware_identity_sha256` | lowercase 64-character digest for the same host inventory |
| `condition_id` | safe ID matching the Custback run |
| `hardware_verified` | exactly `true` |
| `tool` | exact `name`, `version`, and `capture_api` strings |
| `requested` | exact width, height, FPS, and pixel-format request |
| `delivered` | width, height, and observed pixel format |
| `timestamp_kind` | exactly `host-read-completion` |
| `one_uninterrupted_run` | exactly `true` |
| `output_rate_conversion` | exactly `false` |
| `camera_control_writes` | exactly `false` |
| `completion_offsets_ms` | 2–16384 finite, strictly increasing offsets beginning at `0.0` and spanning at least five seconds |
| `failures` | non-negative native read-failure count |

The native tool's own aggregate FPS is not trusted. Custback recomputes cadence
and interval summaries from the offsets. A trace with a completion gap larger
than the bounded three-frame/100 ms tolerance cannot corroborate sustained
cadence even when its aggregate rate reaches the target. Device presentation
timestamps are not interchangeable with host read-completion timestamps and
are not accepted by version 1. Delivered pixel format is an exact bounded
FourCC such as `MJPG`, `YUYV`, or `NV12`; `unknown` remains reportable but
cannot make a native comparison compatible. A nonzero `failures` value remains
visible but makes the native comparison incompatible with corroboration. It
must be zero for compatible native evidence. The final report retains the
evidence-file digest, the opaque identity bindings, and a bounded summary,
never the input path or the underlying device/host identity text.

Native and Custback capture cannot usually open the camera simultaneously.
Run them consecutively under the same declared condition. Compatibility
requires matching device and hardware digests, condition ID, request,
including the pixel-format policy, delivered dimensions, and compatible pixel
format. A different capture API is expected and retained as evidence. Missing
or mismatched native evidence stays explicitly `not provided` or
`incompatible`; it is never promoted into corroboration.

The identity digests are opaque local bindings. Generate them from a reviewed,
stable local device identity and host inventory using one documented local
canonicalization, and use the same digests for both runs. A bare camera index
is not stable enough to identify a physical device. Do not put the source
device path, serial number, or inventory text into the report.

## Optional matched full-runtime evidence

`--runtime-evidence OWNER_ONLY_RUNTIME.json` accepts a strict two-snapshot
sidecar from a matched normal run. It does not accept one loose `GET /status`
object. Copy
[`capture-runtime-evidence-local-template.json`](capture-runtime-evidence-local-template.json),
retain it as an owner-only regular file, and replace every `REPLACE_` value
with two observations from the same uninterrupted full-runtime window. The
checked-in placeholders deliberately fail the strict parser and are not
evidence.

The sidecar has schema `custback.capture-runtime-evidence`, version 1, and
exactly these root fields:

| Field | Requirement |
| --- | --- |
| `schema`, `version` | exact schema name and integer version 1 |
| `device_identity_sha256` | same opaque device binding used by the capture-only run |
| `hardware_identity_sha256` | same opaque host binding used by the capture-only run |
| `condition_id` | same safe comparison condition ID |
| `hardware_verified` | exactly `true` |
| `one_uninterrupted_run` | exactly `true` |
| `requested` | camera width, height, FPS, pixel-format policy, and canonical canvas width/height |
| `negotiated` | bounded backend/FourCC plus property width/height, reported FPS, and actual delivered width/height |
| `start`, `end` | exact uptime and cumulative-counter snapshots |
| `timings_ms` | exact nullable timing values for capture read, segmentation, background, color correction, composite, output send, and full-frame processing |

Each snapshot contains exactly:

- the same safe `run_id`;
- `uptime_s`;
- `frames_in` and `frames_out`;
- `capture_frames_read` and `capture_dropped_frames`;
- `capture_read_failures` and `capture_restarts`; and
- `processing_deadline_misses`.

The end uptime must be at least five seconds after the start, and every
cumulative counter must be nondecreasing. Cadence is always derived from the
bounded window:

```text
window_s       = end.uptime_s - start.uptime_s
capture_fps    = Δcapture_frames_read / window_s
processed_fps  = Δframes_in / window_s
output_fps     = Δframes_out / window_s
```

Custback never computes these comparison rates from `frames_in / uptime_s`,
`frames_out / uptime_s`, or another lifetime counter divided by total process
uptime. The start snapshot excludes earlier warm-up and unrelated operating
history. A nonzero read-failure or capture-restart delta keeps the counters
visible but makes the runtime evidence incompatible with causal comparison;
slot overwrites remain valid contention evidence. The final capture report
retains the sidecar digest, window duration, counter deltas, derived rates, and
supplied nullable stage timings.

Compatibility requires matching opaque device and hardware digests, condition
ID, the requested dimensions/FPS/pixel-format policy/canonical canvas, and
negotiated property dimensions/backend/pixel format/reported FPS plus actual
delivered dimensions. An incompatible sidecar stays visible with reason codes
but cannot support a causal comparison.

This optional evidence can distinguish a capture-only recovery from a matched
full-run regression, but it cannot replace the per-stage profiling owned by
MATTE-3.4 or the unique-frame status contract owned by MATTE-3.2.

## Diagnostic matrix

Use one change per run and preserve the comparison IDs and local identity
bindings:

1. Run capture-only at 640x360, 1280x720, and one higher device-supported mode.
2. Compare `--pixel-format mjpeg` with `--pixel-format backend` where both are
   supported.
3. Repeat the same mode under the reported lighting and stable adequate
   diffuse lighting.
4. Keep camera controls in the read-only `preserve` policy and compare their
   generation-bound observations.
5. Record the same device and condition through a native capture tool.
6. Compare capture-only results with MediaPipe CPU/GPU and RVM CPU/GPU full
   runs, preview off/on, and virtual camera off/on, without relabeling the
   full-run cadence as capture-only.

Priority or affinity changes are not a first response. Evaluate them only
after matched evidence establishes CPU contention and cross-platform behavior.
Prefer a supported camera mode, qualified acceleration, or reduced processing
work first.

## Camera-control and privacy rules

The harness uses the existing preserve-only camera-control contract. It reads
a bounded set of properties once per successful capture generation and never
writes exposure, gain, gamma, or white balance. Mode-negotiation writes for
FourCC, dimensions, and FPS remain the production behavior.

Never turn a lighting experiment into an automatic exposure feedback loop.
Any future explicit manual/lock policy requires its own backend capability,
range, readback, persistence, and representative-device qualification.

Although the report contains no pixels, device path, or credentials, keep it
owner-only because timing, backend, mode, controls, and hardware evidence can
still fingerprint the local environment. Raw native logs and camera footage
remain outside this report and must not be committed.

## Executable evidence

The MATTE-3.1 tests cover:

- capture-only resource isolation;
- strict CLI and owner-only output behavior;
- full-run timestamp math, warm-up exclusion, gaps, and generation changes;
- 15 FPS, target-sustained, mode-mismatch, normalization, and native-comparison
  classifications;
- strict native evidence parsing and identity/mode compatibility;
- strict two-snapshot runtime evidence and counter-delta cadence;
- distinct capture, processing, and output pacing;
- control-write prohibition and privacy-safe error/report content; and
- bounded reader recovery and shutdown regressions.

Those deterministic tests establish software behavior only. The first real
1280x720@30 hardware result must remain local evidence and must state its
camera, host, lighting, backend, format, and native comparison scope through
opaque bindings and reviewed records.
