# Phase 3 observability, controls, and source-color implementation review

- Status: Complete; VIS-3.1 through VIS-3.4 accepted after closing audit
- Date: 2026-07-30
- Scope: `AUTO_COLOR_CORRECTION_AND_SCALING_BACKLOG.md`, VIS-3.1 through
  VIS-3.4
- Contract: `docs/adr/0001-visual-consistency-contract.md`
- Video qualification:
  `docs/visual-consistency-phase3-video-color-qualification.md`
- Camera-control decision: `docs/camera-control-characterization.md`

This document is the Phase-3 closure record. It records the implemented
behavior, the exact final verification evidence, the independent closing
audit, and the work formally retained by Phase 4. The audit found no
functional blocker in the stated VIS-3.1 through VIS-3.4 acceptance boundary.

## Outcome

Phase 3 makes the geometry and color decisions from Phases 1 and 2 observable
and operable without changing their compatibility defaults:

| Concern | Implemented Phase-3 behavior | Compatibility boundary |
| --- | --- | --- |
| Frame/status identity | A processed frame and its matching public stats are published through one locked hub boundary | Existing raw and remote publication contracts are unchanged |
| Geometry telemetry | Negotiated, delivered, oriented, normalized, output, fit-plan, generation, and effective-FPS facts are distinct | `capture_width`/`capture_height` retain their negotiated-device-property meaning |
| Correction telemetry | Configured mode, effective mode, state, reason, confidence, EV, WB gains, timing, counters, and input assumption are public | Correction remains default-off |
| Operator controls | Common color and fit controls are prominent; technical and restart-only controls are advanced and labelled | Restart-only changes are not silently made live |
| Local video decode | PyAV/FFmpeg metadata is resolved deterministically and supported SDR inputs are normalized to full-range sRGB BGR | Missing metadata uses an explicit observable schema-v1 assumption; unsupported tagged color is rejected |
| Output color signaling | Windows RGB32 declares the proven BT.709/sRGB/full-range contract | pyvirtualcam has no portable color-metadata signaling API; consumer interpretation is not claimed |
| Camera hardware controls | A generation-bound read-only capability observation is reported | The only policy is `preserve`; no control is written |

Schema-v1 and distributed defaults remain
`camera.fit_mode: stretch`, `compositing.blend_space: srgb_legacy`, and
`compositing.color_correction.mode: off`. Phase 3 does not authorize the
Phase-4 target-default migration.

## Requirement review

### VIS-3.1 — geometry/color telemetry and transition logging

Implemented in `src/custback/capture.py`, `src/custback/hub.py`,
`src/custback/pipeline.py`, `src/custback/api/server.py`,
`src/custback/preview.py`, and `src/custback/__main__.py`:

- `CaptureHealth` separates backend-reported/negotiated size from delivered,
  oriented, and normalized dimensions. Capture generation and geometry
  generation/transitions remain explicit.
- Public status reports the canonical output dimensions, configured output
  FPS, effective output FPS, camera and backdrop fit/rotation/mirror, scale,
  half-open crop rectangle, padding on all four sides, and transform
  transition totals.
- The legacy `capture_width` and `capture_height` fields still mean the
  backend-reported/negotiated camera mode. New
  `capture_delivered_*`, `capture_oriented_*`, and
  `capture_normalized_*` fields prevent an output canvas from being silently
  relabelled as capture negotiation.
- Correction status separates configured mode, transform application, and
  temporal state:
  `color_correction_mode`, `color_correction_effective_mode`,
  `color_correction_active`, `color_correction_state`,
  `color_correction_reason`, confidence, exposure EV, three bounded WB gains,
  WB-active, warming, stale, applied/bypassed counts, scene cuts, transitions,
  and the `color_correction_ms` stage timing.
- The camera boundary assumption is named
  `display-referred-srgb-bt709-full-range`. This is an interoperability
  assumption for metadata-opaque OpenCV capture, not a claim about every
  camera or driver.
- Video-provider status adds decoder backend, resolved input and output color,
  resolution status, and exact assumed/overridden field lists. Mutable nested
  status values, including the camera-control report and lists, are copied at
  the hub boundary.
- `FrameHub.publish_output(frame, stats=...)` installs stats while holding the
  stats lock and publishes/notifies the matching output before releasing it.
  Hot-commit identity is not promoted to public status until the first frame
  from that identity is ready.
- Camera transforms, backdrop transforms, and correction state are logged on
  first observation and transition only. The messages contain dimensions,
  plans, state/reason/confidence, and bounded transform values, but no device
  or asset path.
- The local preview HUD renders the delivered-to-output geometry and current
  correction state/EV/confidence. Low confidence and stale decay receive
  distinct warnings.
- The shutdown summary includes capture generation, camera/backdrop geometry
  transitions, applied and bypassed correction-frame totals, scene cuts, and
  correction transitions.
- `_StatusResponse`, `FrameHub.stats_dict()`, and the generated OpenAPI
  properties have an exact key-parity regression. `native_ring` remains the
  one API-computed addition.

Operator-state meanings are intentionally non-overlapping:

| State | Meaning |
| --- | --- |
| `disabled` | Effective configuration selects `off` |
| `mode-excluded` | The current mode has an explicit identity/bypass policy |
| `warming` | Auto is eligible but no fresh applied estimate is ready |
| `low-confidence` | Confidence is insufficient; an old bounded transform may be held or identity may be used |
| `stale-decay` | A held transform is decaying toward exact identity |
| `scene-cut` | The previous scene estimate was invalidated and fast reacquisition is beginning |
| `active` | A reliable state is available; `active` is still false when its effective transform is identity |

Primary executable evidence in `tests/test_observability.py` covers:

- output/stats atomic publication and hot-commit identity promotion;
- the complete “640x480 delivered -> cover/crop -> 1280x720, +0.35 EV,
  WB active, confidence 0.82” status example;
- legacy capture-dimension meaning and defensive copying of nested status;
- every correction state above;
- transition-only, path-free geometry and color logs;
- exact hub/Pydantic/OpenAPI key parity;
- preview geometry/color/warning text; and
- shutdown transform/correction totals.

`tests/test_capture_geometry.py`, `tests/test_pipeline.py`,
`tests/test_api.py`, and `tests/test_preview.py` provide the surrounding
capture-generation, runtime-loop, response, and presentation coverage.

Review boundary: the hub guarantees frame/status ordering at its publication
boundary. This does not imply that independent hardware counters were sampled
at one physical instant; the pipeline assembles a bounded per-frame snapshot
from the current generation before publication.

### VIS-3.2 — Web controls and operator diagnostics

Implemented in the self-contained Web UI at
`src/custback/api/webui.py`:

- The main quality panel prominently exposes automatic color correction,
  correction strength, background fit, camera fit, and horizontal/vertical
  backdrop focal anchors.
- The advanced panel exposes exposure limit, white-balance strength,
  adaptation time, and the legacy/linear blend-space compatibility switch.
  Camera rotation and paired output canvas dimensions are grouped and visibly
  marked restart-required.
- Each control emits a leaf-level merge patch. Output dimensions are patched
  together because the schema requires both or neither; this is the smallest
  valid patch for that invariant.
- On a `409`, `422`, or `503` PATCH failure, the UI fetches `/config` and
  renders the effective configuration before surfacing the error. If that GET
  cannot complete, it still renders the last known effective configuration.
- The concise color summary calls a transform active only when the temporal
  state is `active`, `color_correction_active` is true, and the effective mode
  is not `off`, `bypass`, or `identity`. Disabled, excluded, warming,
  scene-cut, low-confidence hold, and stale decay use distinct wording.
- The System view retains the complete status object and adds readable
  formatting for source-color arrays and the nested, generation-bound
  camera-control report instead of rendering them as generic JavaScript
  objects.
- Every new input has an associated label. The live summary uses polite
  status semantics, focus and reduced-motion behavior are preserved, and the
  quality layout has explicit narrow/mobile and wider breakpoints spanning the
  required 320–768 px range using existing design tokens.

Primary executable evidence in `tests/test_webui.py` covers:

- scripted-element and JavaScript parse integrity;
- labels, live-region semantics, breakpoints, and restart markings;
- exact minimal patch shapes for every new control;
- static and Node-executed recovery from `409`, `422`, and `503`;
- static and Node-executed correction-summary semantics;
- readable video-color and camera-control diagnostics; and
- crop, pad, and stretch geometry-summary distinctions.

Review boundary: restart-labelled controls submit to the same API so the
server remains the lifecycle authority. A rejection restores the effective
value; the UI neither restarts the process nor claims that a restart-only
change was applied. Phase 3 also does not add camera-hardware write controls.

### VIS-3.3 — metadata-aware video and output color

Implemented in `src/custback/video_decoder.py`,
`src/custback/backgrounds.py`, `src/custback/config.py`, the Windows Media
Foundation source, dependency/release metadata, and CI/release workflows.
The detailed decoder contract and performance observation are in
`docs/visual-consistency-phase3-video-color-qualification.md`.

The production local-video path uses a PyAV-backed, cv2-compatible adapter so
the established `VideoBackdrop` scheduler remains the owner of monotonic
playback phase, look-ahead, reuse, bounded skipping, large-gap seeking, EOF
length correction, loop phase, and last-good-frame retention. The decoder
adapter owns sequential decode, frame/codec metadata, explicit conversion,
timestamps, and bounded seek scanning.

Color resolution is deterministic per field:

1. a configured override for an operator-owned local file;
2. a supported declaration on the decoded frame;
3. a supported declaration on the codec context; then
4. the schema-v1 legacy assumption for a genuinely unspecified field.

The legacy tuple is `bt709/full/bt709/srgb`. Its assumed fields and every
override are public status. Pixel values, histograms, and apparent 16–235
occupancy never participate.

The accepted SDR subset is deliberately narrow:

| Field | Accepted inputs | Output action |
| --- | --- | --- |
| Matrix | BT.601 or BT.709 declarations | FFmpeg conversion receives the resolved source matrix explicitly |
| Range | limited/MPEG or full/JPEG | FFmpeg conversion receives the resolved source range and emits full RGB |
| Primaries | BT.709, BT.470BG, or SMPTE170M | Non-sRGB primaries are converted in linear light to sRGB primaries |
| Transfer | sRGB or BT.709/SMPTE170M SDR | BT.709 is EOTF-decoded and re-encoded through sRGB; sRGB is preserved |

Tagged PQ, HLG, BT.2020, and other unsupported HDR/wide-gamut declarations are
rejected with the offending field/value. A later-frame tag change is resolved
again and cannot be swallowed as EOF or silently assigned the legacy tuple.

The decoder boundary is local-file-only:

- protocol-shaped and UNC inputs are rejected before filesystem or AV I/O;
- the production path requires `Path.is_file()` before `av.open`;
- libavformat receives `protocol_whitelist=file`, so a trusted local
  container/manifest may resolve a local-file graph but cannot enter nested
  HTTP/HTTPS/concat/data/crypto protocols;
- stream metadata and every raw decoded frame are checked against configured
  dimension and supported-color bounds before grab/seek discard;
- only the first video stream is selected;
- hardware decoding is not requested; and
- a seek scans at most 300 decoded frames after the keyframe seek.

This prevents PyAV from expanding Custback's configured background-source
authority to FFmpeg network/non-file protocols. It does not provide
single-inode isolation: an operator-selected manifest remains trusted to name
other readable local files. Native exception causes are suppressed so those
top-level or nested paths cannot enter application tracebacks. This does not
remove the ordinary native codec/parser attack surface. Dependency updates,
vulnerability review, and artifact provenance remain release responsibilities.

The dependency markers are:

```text
av>=17,<18; python_version < '3.11'
av>=18,<19; python_version >= '3.11'
```

PyAV package metadata declares BSD-3-Clause. That metadata describes PyAV; it
does not by itself discharge the obligations of the FFmpeg libraries and
codecs bundled into a particular wheel or copied into a release artifact. The
locally observed PyAV 18 Linux wheel reports bundled FFmpeg libraries under
LGPL version 3 or later and exposes an `av.libs` bundle from a build with x264
and x265 enabled. That observation is neither a guarantee for every
platform/version wheel nor legal clearance for a shipped artifact. Release
owners must inventory the exact resolved wheel and codec build for each
artifact, preserve applicable notices and license texts, and complete
artifact-specific source/relinking or other compliance obligations as
required. Codec availability and licensing are properties of that reviewed
artifact, not inferred from the top-level PyAV license.

The Windows frozen payload now collects PyAV extensions and preserves the
wheel's sibling `av.libs` layout. Its build runs an offline in-memory tagged
normalization probe in the scrubbed frozen environment. That implementation
does not authorize publication: every frozen artifact still requires the
artifact-specific FFmpeg/codec inventory, notices, source/relinking analysis,
and other applicable license obligations described above.

For output, every software sink receives validated full-range sRGB BGR. The
Windows native ring mechanically adds an opaque byte and its RGB32 Media
Foundation type declares BT.709 primaries, sRGB transfer, and nominal
0..255 range. No YUV matrix is asserted for RGB32. Pyvirtualcam is opened as
BGR but exposes no portable primaries/transfer/range signaling control;
application interpretation on OBS/v4l2loopback and representative
Linux/macOS/Windows consumers remains unqualified.

Primary executable evidence in `tests/test_video_color.py` covers:

- generated lossless tagged BT.601/BT.709 x limited/full fixtures converging
  to one reference sRGB raster within three 8-bit code values;
- transfer and primary conversion as pixel operations, not metadata retags;
- exact frame/codec/override/legacy precedence and public telemetry;
- no histogram-based range inference;
- local-only override, protocol/UNC rejection before top-level I/O, and nested
  HTTP-manifest refusal before TCP/TLS;
- startup and later-frame unsupported-tag rejection;
- unsupported-color/oversized raw-frame rejection before grab or seek discard;
- decoder-error quarantine rather than EOF reinterpretation;
- later-frame dimension-bound failure;
- displayed-frame/color-status coupling despite look-ahead;
- real VFR timing, frame reuse, EOF loop behavior, and backend status;
- real PyAV MOV display-matrix rotation through `VideoBackdrop`, with mirrored
  matrices rejected; and
- the exact 300-frame seek-scan bound against a longer synthetic iterator.

Inherited backdrop tests retain malformed-frame, generic-backend orientation,
dynamic-size, skip/seek failure, cache invalidation, and last-good scheduler
coverage. They also prove orientation/FPS/invalid/decode logs and formatted
source-open tracebacks do not expose a private asset path/token.
Windows source tests assert the three RGB32 attributes. The configured CI
matrix adds the video-color suite to Python/OpenCV compatibility lanes, pins
PyAV 17.1.0 in the Python-3.10 minimum-dependency lane, and defines a
Windows-2022 native build plus video-color test job. A separate bounded
`windows-frozen-engine` job builds the x64 CPU/vision onedir payload and runs
the scrubbed tagged-video, engine, and avatar smokes. These workflow
definitions are coverage commitments; they are not evidence that the final
candidate has passed until CI reports green.

### VIS-3.4 — camera hardware-control characterization

Implemented in `src/custback/capture.py` with the accepted decision recorded
in `docs/camera-control-characterization.md`:

- After the first valid frame establishes a capture generation, Custback reads
  the available OpenCV properties for automatic WB, WB temperature,
  automatic exposure, exposure, gain, and gamma exactly once.
- The immutable report names the independently classified backend family
  (`v4l2`, `msmf`, `dshow`, or `other`), generation, policy,
  qualification, whether writes occurred, and per-property observation.
- A finite non-zero read is `reported`; zero is `indeterminate-zero` because
  OpenCV also uses it as an unsupported sentinel; an absent, exceptional, or
  non-finite read is `unavailable`.
- The report contains no device identifier or path. Reconnect replaces it with
  the new generation's report. Synthetic capture is explicitly
  `not-applicable`.
- The only accepted policy is `preserve`, every real backend remains
  `unqualified`, and `writes_performed` is false. There is no
  `lock_after_warmup` or `manual` schema/UI surface and no call to set a
  camera-control property.

`tests/test_capture_geometry.py` proves exact read counts, zero/non-finite
semantics, no control writes, independent V4L2/MSMF/DSHOW classification,
generation replacement, and the synthetic report.
`tests/test_observability.py` proves the nested report reaches public status
without sharing mutable state, and `tests/test_webui.py` proves it is rendered
readably.

The representative-device qualification requested as exploratory VIS-3.4
work is not claimed. Backend families are independently characterized as
unqualified at the generic OpenCV boundary, based on their different native
capability contracts, and the resulting decision is to expose no write
policy. This satisfies the side-effect-free report acceptance criterion but
does not qualify a device value, range, unit, accepted write, or readback.

A future hardware policy is a new decision: it requires a backend-specific
capability/range adapter, opt-in restart-safe configuration, representative
devices for that exact backend, verified write/readback semantics, visible
rejection, and a software harmonizer reset after each confirmed transition.
It must never become a per-frame feedback loop.

## Cross-cutting review

### Defaults, migration, and privacy

- Phase 3 adds fields and controls but does not reinterpret a schema-v1
  omission or flip a visual default.
- Local source-color overrides are explicit values in the background config;
  `auto` continues deterministic metadata/legacy resolution.
- Raw publication and fingerprinting remain normalized but uncorrected camera
  BGR. Remote frames and the privacy slate remain correction-excluded and
  exact-canvas.
- Telemetry and transition logs contain no asset/device path. Existing
  authenticated configuration APIs may still return their separately defined
  public configuration; Phase-3 status does not duplicate source identifiers.

### Platform and performance limits

The only Phase-3 performance number is a non-gating local observation,
documented with its environment in the video qualification:

| Linux 1280x720 FFV1/YUV444 path | p50/frame | p95/frame |
| --- | ---: | ---: |
| OpenCV opaque BGR decode | 1.068 ms | 2.850 ms |
| PyAV decode plus explicit matrix/range normalization | 1.970 ms | 2.142 ms |

The separately measured process high-water deltas were 71,468 KiB for OpenCV
and 27,964 KiB for PyAV. They include decoder buffers and allocator retention,
are not per-frame allocations, and must not be used as a general memory claim.
The sample is not a calibrated cross-platform benchmark and does not cover
long playback, every codec, HDR rejection load, hardware decode, or consumer
delivery.

No Phase-3 unit or generated-fixture result constitutes:

- a live V4L2/MSMF/DSHOW camera-control qualification;
- a real OBS, v4l2loopback, conferencing-application, or pyvirtualcam color
  interpretation qualification;
- a real Windows virtual-camera consumer negotiation result;
- a calibrated Linux/macOS/Windows p95, RSS, thermal, or long-run playback
  result; or
- release/legal clearance for every transitive binary dependency.

## Verification ledger

All local gates below ran against the final reviewed tree on 2026-07-30.
External CI rows remain coverage commitments rather than fabricated results.

| Gate | Required scope | Review state |
| --- | --- | --- |
| VIS-3.1 focused | Observability, API/lifecycle, preview, pipeline, capture, and capture geometry in the combined Phase-3 integration run | **PASS; combined run 552 passed in 9.78 s** |
| VIS-3.2 focused | Web UI including Node semantic harnesses | **PASS; included in the 552-test run** |
| VIS-3.3 focused | Video color, processing/background geometry, Windows color attributes, and frozen packaging | **PASS; included in 552; source subset 176 passed; final PyAV-17 video/background subset 55 passed** |
| VIS-3.4 focused | Camera-control capture, observability, API, and Web rendering regressions | **PASS; included in the 552-test run** |
| Static policy | Ruff check/format, Pyright, workflow YAML, Python spec/entry, release JavaScript, and diff checks | **PASS; Ruff 95 files, Pyright 0 errors/warnings, 2 workflows parsed** |
| Host full suite | Python 3.14.4, PyAV 18.0.0, NumPy 2.5.1, OpenCV 5.0.0 | **PASS; 1,433 passed, 3 skipped in 20.30 s; `pip check` clean** |
| Minimum dependencies | Python 3.10.20, PyAV 17.1.0, NumPy 1.24.0, OpenCV 4.8.1.78 and reviewed floors | **PASS; 1,428 passed, 8 environment-only skips, 2 known Pydantic warnings in 36.22 s; `pip check` clean** |
| Newest dependencies | Host environment above, including PyAV 18 | **PASS; same complete 1,433-test host run** |
| Installed artifacts | Wheel/sdist build, exact archive inspection, isolated installs/imports, tagged pixel/contract probe, extras, and npm payload/install smoke | **PASS; `verifyPackageSmoke()` returned 0.4.0** |
| npm regressions | Complete Node test set | **PASS; 141 passed, 2 expected TODO blocker tests** |
| CI platforms | Python 3.10–3.14, OpenCV bounds, macOS artifacts, Windows native/video, and Windows frozen engine | **CONFIGURED; external run evidence still required for a release candidate** |
| Closing audit | Independent exact VIS-3.1–VIS-3.4 mapping and overclaim/gap review | **PASS; no functional Phase-3 blocker found** |

The combined Phase-3 command covered:

```text
tests/test_observability.py tests/test_capture.py
tests/test_capture_geometry.py tests/test_pipeline.py tests/test_api.py
tests/test_api_lifecycle.py tests/test_preview.py tests/test_webui.py
tests/test_video_color.py tests/test_processing.py
tests/test_background_geometry.py tests/test_windows_packaging.py
tests/test_windows_vcam.py
```

The npm TODO cases are deliberate executable assertions that the unrelated
pre-existing `REL-01` and `WIN-01` production-publication gates remain open;
they do not fail the Node process and were not reclassified as Phase-3
successes. `verifyPackageSmoke()` is non-authorizing by design and proves the
installable artifacts without bypassing those publication gates.

## Phase-4 deferrals

The following work remains intentionally outside Phase 3:

- VIS-4.1 black-box geometry/color/lifecycle/privacy matrices across the
  running API, renderer, and every sink;
- VIS-4.2 calibrated visual, latency, CPU/RSS, thermal, long-run, platform,
  real-camera, and real-consumer qualification, including pyvirtualcam
  interpretation, representative V4L2/MSMF/DSHOW devices, and any future
  camera-control adapter;
- VIS-4.3 target-default schema migration, release staging, rollback
  instructions, and the go/no-go decision for `cover`, `linear_srgb`, and
  `auto`; and
- any HDR/wide-gamut working/output space, hardware video decode claim, native
  multi-size scaler, or camera-hardware write policy.

Until those gates pass, production-safe guidance is to retain the explicit
schema-v1 defaults, treat video legacy assumptions and overrides as
observable operator decisions, keep camera controls in `preserve`, and avoid
claiming cross-platform consumer color equivalence from generated fixtures
alone.

The VIS-3.3 live-consumer work bullet and VIS-3.4 representative-device work
bullet are formally retained by VIS-4.2. Phase 3 accepts only what is provable
without that hardware: the metadata-aware software contract and Windows
source declaration for VIS-3.3, and a truthful side-effect-free
`preserve`/`unqualified` capability report for VIS-3.4. A future write policy
or cross-platform consumer-equivalence claim requires new Phase-4 evidence.
