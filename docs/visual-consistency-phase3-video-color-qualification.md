# VIS-3.3 video and output color qualification

Status: implemented; live consumer application qualification remains a
VIS-4.2 boundary.

## Production contract

Local video backdrops use the bounded PyAV dependency and its bundled FFmpeg
libraries. The decoder retains declarations on each decoded frame and falls
back to the stream codec context. Resolution is per field:

1. an explicitly configured local-file override, if present;
2. frame metadata;
3. codec-context metadata; then
4. the schema-v1 legacy assumption for a genuinely unspecified field.

The legacy tuple is `bt709/full/bt709/srgb`. Its use is visible as
`background_video_color_status: legacy-assumption` with exact
`background_video_color_assumed_fields`. Overrides are visible as
`operator-override` and `background_video_color_overridden_fields`.
Histogram or pixel-range observations are not inputs to this decision.

Supported declarations are deliberately narrow:

| Field | Accepted metadata | Normalized meaning |
| --- | --- | --- |
| YUV matrix | FFmpeg `BT709`; `BT470BG`/`SMPTE170M` matrix values | `bt709`; `bt601` |
| range | `MPEG`; `JPEG` | limited; full |
| primaries | `BT709`; `BT470BG`; `SMPTE170M` | source D65 linear RGB is converted to linear sRGB |
| transfer | `BT709`/`SMPTE170M`; `IEC61966-2-1` | BT.709 is decoded and re-encoded as sRGB; sRGB is preserved |

FFmpeg `libswscale` is supplied the resolved source matrix and range
explicitly and produces full-range RGB. It does not perform the transfer or
primary conversion. Custback therefore performs those conversions explicitly
before publishing contiguous `uint8 BGR` under the existing
`srgb-full-bgr` contract.

Tagged BT.2020, PQ, HLG, and other unsupported HDR/wide-gamut declarations
fail candidate construction with the offending field/value. They are not
treated as EOF, retried, or silently assigned the legacy tuple. A color-tag
change on a later frame is resolved again and likewise fails explicitly if it
leaves the supported set.

## Authority, security, and bounds

- URL/FFmpeg-protocol and UNC/network-share forms are rejected before any
  filesystem probe or `av.open`. Every remaining production metadata-aware
  source must satisfy `Path.is_file()`, with or without overrides.
- `av.open` receives `protocol_whitelist=file`. This is a network/protocol
  boundary, not single-inode isolation: a trusted local container or manifest
  may still name a graph of other local files readable by the service.
  Nested HTTP/HTTPS, concat, data, crypto, and other non-file protocols are
  refused by libavformat before a TCP/TLS transport is opened. Native open and
  decode exception chains are suppressed at the application boundary so
  local or nested asset paths cannot enter logs.
- Stream dimensions are checked before the first decoded frame. Dimensions
  and color declarations are checked on every raw decoded frame before it can
  be selected, skipped by `grab`, or discarded by a post-seek scan. A
  dimension increase beyond configured upload bounds fails rather than
  allocating a fitted cache. Pixel conversion and displayed color telemetry
  remain coupled only to the selected/retrieved frame.
- Only the first video stream is selected. Hardware decoding is not requested,
  so GPU-specific color paths are not advertised.
- Random access seeks backward to a keyframe, then scans at most 300 decoded
  frames for the requested timestamp/index. Failed seeks retain the
  `VideoBackdrop` last-good frame.
- `VideoBackdrop` still owns its monotonic epoch, one-frame look-ahead,
  timestamp validation, VFR deadlines, reuse counters, at-most-eight
  sequential skips, large-gap seek, EOF length correction, and loop phase.
  The PyAV adapter implements only sequential decode, grab/retrieve, bounded
  seek, timestamps, and color conversion behind that scheduler.
- Decoded color metadata travels with the pending frame. Prefetch may update
  the adapter's latest decode, but public status changes only when
  `VideoBackdrop` promotes the matching pixels to the displayed frame.
- Pure right-angle PyAV `DISPLAYMATRIX` rotation is qualified. PyAV's
  counter-clockwise value is translated once to custback's clockwise geometry
  convention. Mirrored or non-right-angle display matrices are rejected
  explicitly rather than silently losing orientation. The decoder keeps
  FFmpeg auto-rotation disabled and `VideoBackdrop` applies the qualified
  rotation exactly once.
- Backdrop orientation, FPS fallback, malformed/decode warnings, and raised
  source-open errors contain no asset identifier. Tests render the complete
  exception traceback to prove the suppressed native cause cannot reintroduce
  a private path/token.

## Dependency and license posture

The reviewed dependency markers are:

```text
av>=17,<18; python_version < '3.11'
av>=18,<19; python_version >= '3.11'
```

They retain the project's Python 3.10–3.14 support range. PyAV package
metadata declares BSD-3-Clause. Official wheels are available for the
supported Linux, macOS, and Windows Python/platform matrix and bundle FFmpeg,
avoiding an unbounded host executable/subprocess dependency.

The observed Linux PyAV 18.0.0 wheel contains FFmpeg 8.1.2:
`libavutil 60.26.102`, `libavcodec 62.28.102`,
`libavformat 62.12.102`, and `libswscale 9.5.102`. Its runtime
`av._core.library_meta` reports `LGPL version 3 or later` for every bundled
FFmpeg library. The CI minimum lane pins PyAV 17.1.0 on Python 3.10; the
normal Python matrix resolves PyAV 18 on Python 3.11–3.14. Linux and macOS
artifact suites run the full generated-fixture tests, and the Windows native
job installs the dependency and runs the same video-color suite.

Codec availability remains that of the reviewed wheel. The production path
does not shell out to `ffmpeg`/`ffprobe`, and no system FFmpeg package is
required.

BSD-3-Clause covers PyAV itself, not every binary in a PyAV wheel. The
observed manylinux wheel has an `av.libs` directory and reports a build that
enables transitive codec libraries including x264 and x265. The custback
wheel and sdist reference PyAV as a dependency and do not copy those binaries.
An npm managed environment or the implemented Windows frozen payload can,
however, redistribute the resolved PyAV/FFmpeg/codec binaries. The frozen
PyInstaller spec preserves the wheel's `av.libs` layout and its build runs an
offline tagged-normalization smoke; neither action is legal clearance. Every
redistributed artifact must carry an artifact-specific
license/source/notice compliance review before it is authorized for
publication; PyAV's BSD metadata alone is not sufficient evidence for that
redistribution.

## Output declarations

- Null and pyvirtualcam sinks receive validated full-range sRGB BGR.
  Pyvirtualcam is opened with `PixelFormat.BGR`, but its API has no portable
  primaries/transfer/range attribute. Actual OBS/v4l2loopback application
  interpretation is therefore explicitly not claimed from unit tests and must
  be sampled in VIS-4.2 on representative Linux/macOS/Windows consumers.
- The native Windows ring conversion is mechanical BGR to BGRX. Its RGB32
  Media Foundation type now declares `MFVideoPrimaries_BT709`,
  `MFVideoTransFunc_sRGB`, and `MFNominalRange_0_255`. No YUV matrix is
  declared for RGB32.

## Executable evidence

`tests/test_video_color.py` creates lossless tagged FFV1/YUV444 fixtures at
test time. Its BT.601/BT.709 × limited/full matrix resolves to the same
reference sRGB raster within three 8-bit code values. It also proves:

- exact tag/codec/override/legacy precedence and telemetry;
- displayed-frame/color-status atomicity despite one-frame prefetch;
- explicit override, differing frame tag, codec fallback, and legacy
  precedence on a per-field basis;
- PQ rejection during decoder construction;
- later-frame unsupported metadata and decoder exceptions remain explicit;
- unsupported color and oversized frames fail before both `grab` skipping and
  post-seek discard, while displayed color telemetry remains unchanged;
- URL, FFmpeg protocol, and UNC forms are rejected before top-level I/O, and a
  local HLS manifest with an HTTP segment is refused by the nested protocol
  whitelist without entering TCP/TLS;
- BT.709 transfer conversion is a pixel operation, not retagging;
- BT.470BG primary conversion is performed in linear light;
- no histogram-only range inference;
- VFR timing, cached-frame reuse, actual PyAV loop progress, and backend status
  through the real `VideoBackdrop`;
- a real MOV/PNG display-matrix fixture through PyAV and `VideoBackdrop`,
  including counter-clockwise-to-clockwise translation and explicit mirrored
  matrix rejection; and
- an exact 300-frame post-seek decode bound against an iterator with more than
  300 available frames.

Inherited `tests/test_processing.py` and `tests/test_background_geometry.py`
continue to prove clock/skip/reuse/seek failure, EOF correction, malformed
frames, generic-backend orientation policy, dynamic dimensions, fitted-cache
invalidation, last-good retention, and path-free video diagnostics at the
scheduler boundary.

## Non-gating local performance observation

Environment: Ubuntu/Linux 7.0.0-28 x86_64, Intel Core i7-12700H,
Python 3.14.4, NumPy 2.5.1, OpenCV 5.0.0 (20 reported threads), PyAV 18.0.0,
FFmpeg 8.1.2.

Fixture: deterministic 1280×720, 72-frame lossless FFV1/YUV444,
BT.709 matrix, limited range, BT.709 primaries, sRGB transfer; 9,142,161
bytes. Each path used 10 warm-up frames and 62 sequential measured frames with
`time.perf_counter`.

| Path | p50/frame | p95/frame |
| --- | ---: | ---: |
| OpenCV opaque BGR decode | 1.068 ms | 2.850 ms |
| PyAV decode + explicit matrix/range normalization | 1.970 ms | 2.142 ms |

Separate-process Linux `ru_maxrss` high-water deltas while decoding all 72
frames were 71,468 KiB for OpenCV and 27,964 KiB for PyAV. These deltas include
native decoder buffers and allocator retention, are not steady-state
per-frame allocations, and are not a release gate. Calibrated cross-platform
p95/RSS and long-run playback remain VIS-4.2.
