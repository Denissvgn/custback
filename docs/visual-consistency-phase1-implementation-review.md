# Phase 1 canonical geometry implementation review

- Status: Phase 1 complete; implementation and local/release verification passed
- Date: 2026-07-30
- Scope: `AUTO_COLOR_CORRECTION_AND_SCALING_BACKLOG.md`, VIS-1.1 through
  VIS-1.6
- Contract: `docs/adr/0001-visual-consistency-contract.md`

## Outcome

Phase 1 replaces the camera/backdrop resize split with one immutable, cached
geometry plan and makes the resolved output canvas a checked invariant from
capture through every publication and output boundary.

The schema-v1 compatibility default remains `camera.fit_mode: stretch`.
Operators can select proportional `cover` or `contain` now; changing the
default remains the separately qualified and versioned rollout owned by
VIS-4.3. No Phase-2 color behavior or default was enabled as part of this
phase.

## Requirement review

| Task | Implemented contract | Primary executable evidence |
| --- | --- | --- |
| VIS-1.1 | `geometry.py` separates a bounded, immutable plan cache from pixel execution; implements integer-safe cover/contain/stretch, anchor edges, right-angle rotation, viewer-horizontal mirror, direction-aware interpolation, black containment padding, exact no-op behavior, telemetry, strict BGR validation, and contiguous output. The module caches plans only, never frames. | `tests/test_geometry.py` covers every ADR worked example, 300 seeded invariant cases, cache bounds/identity, labeled pixels, circles, interpolation spies, malformed settings/frames, and output ownership. |
| VIS-1.2 | Pillow is the authoritative still-image decoder and applies EXIF 1–8 exactly once before BGR conversion. Video providers disable and read back OpenCV auto-rotation before applying valid metadata; opaque, ignored, or invalid controls are visibly `manual-only-*` and never cause inferred rotation. Main/live/video source boundaries use the shared strict validator. | `tests/test_background_geometry.py` uses real lossless EXIF-tagged files, an OpenCV decode tripwire, qualified/ambiguous video-control mocks, malformed-source matrices, and orientation parity. `tests/test_capture_geometry.py` freezes manual rotation-then-mirror order and rejects malformed camera arrays before publication. |
| VIS-1.3 | Live and synthetic capture receive the resolved canvas while device negotiation remains on `camera.width/height`. The reader validates and transforms each delivered native frame once before the latest-only slot. Health keeps property-reported, delivered, oriented, and normalized sizes separately. A delivered-size transition replans in `warn` mode and is fatal in `error` mode; reconnect generations cannot reuse stale local state. | `tests/test_capture_geometry.py` covers acquisition/canvas separation, 4:3 cover without circle distortion, interpolation, dimension telemetry, property disagreement, size changes, reconnect, portrait rotation/mirror, malformed inputs, and one transform per delivery. The inherited negotiation/recovery/drop/shutdown suite remains green. |
| VIS-1.4 | Image, video, and live-camera providers call the shared geometry executor. Fitted caches include source generation/identity and size, resolved orientation, target, fit, anchors, and black pad policy. A hot geometry commit invalidates fitted pixels without reopening a provider or resetting video clocks/counters. Blur validates and returns its already canonical source without a second fit; color allocates the canvas directly. Candidate geometry is exercised only on bounded detached pixels. | `tests/test_background_geometry.py` proves cross-provider pixel parity, exact crop/anchor behavior, cache invalidation, video resolution changes/last-good handling, and blur ownership. Pipeline lifecycle regressions prove provider identity and visual generation remain separate. |
| VIS-1.5 | `_Resources` resolves and owns one canvas. Capture and output constructors receive it; preflight and steady-state capture, mask, model foreground, backdrop, raw/output hub publication, privacy slate, remote return, repeat, and every sink validate it exactly. The API renderer receiver uses `resolved_output_size`; snapshot, MJPEG, and WebSocket metadata expose the effective dimensions. Status separates acquisition/delivery/orientation facts from effective output width/height/FPS. Remote output is rejected, never fitted. | `tests/test_canonical_canvas.py`, `tests/test_pipeline.py`, `tests/test_api.py`, `tests/test_output_geometry.py`, and the inherited privacy/streaming/avatar tests cover non-default acquisition/canvas pairs, local/remote modes, preflight, repeats, fingerprints, wrong-size privacy fallback, API transports, and backend contracts. |
| VIS-1.6 | The Windows native route follows the ADR’s constrained option rather than adding an independent scaler. Explicit native output accepts exactly 1280x720@30 or 1920x1080@30 and rejects other canvas/FPS combinations before component/ring access. Auto skips unsupported modes. The writer precomputes BGRX bytes and publishes a valid inactive ring header at construction; Media Foundation waits boundedly for a concurrent write, reads the header during initialization, and advertises only the matching exact type, so a consumer cannot choose the other size. A present invalid/unsupported header fails activation instead of silently defaulting to 720p. Stable exact frames replace a last-good cache only after seqlock and post-copy active-flag verification; a transient read reuses that exact frame, while inactive/invalid/mismatched state clears it and emits the input-independent placeholder. The overlap-copy path is removed. Both readers validate BGRX, exact stride, dimensions, and bounds. Python/native writers strictly validate canonical contiguous BGR before opaque BGRX conversion. The project targets Windows SDK 22621, which is installed on the pinned runner, while retaining build 22000 as its runtime floor. | `tests/test_output_geometry.py` cross-checks Python/C++ modes, ring-aware single-type advertisement and fail-closed initialization, writer critical-section order, transient/last-good behavior, pre-open rejection, auto fallback, frame/ring contracts, truncated/corrupt FOURCC/stride rejection, BGRX bytes/stride/counters, consumer type refresh, post-copy deactivation races, and the absence of overlap-copy/scaling code. `windows-native-vcam` compiles the x64 C++ project and checks its COM exports on the pinned `windows-2022` CI runner. |

## Resolved contract choices

The implementation closes the Phase-1 ambiguities as follows:

1. Source order is metadata orientation, configured clockwise rotation,
   viewer-horizontal mirror, then fit. Background video has metadata
   orientation only because schema v1 defines no manual background rotation.
2. `crop_rect` is half-open in the resized source raster. `content_rect` is
   half-open in target coordinates. `padding` records left/top/right/bottom
   opaque-black extents.
3. A main-camera delivered-size change is recoverable and replanned under
   `mode_mismatch: warn`; it terminates that capture under
   `mode_mismatch: error`.
4. Blur owns no geometry: its input is already canonical and a mismatch is a
   contract error, not an invitation to resample again.
5. A video backend is metadata-qualified only when a valid right-angle value
   exists and disabling auto-rotation succeeds with matching readback.
   Everything else is reported as ambiguous/manual-only; portrait dimensions
   never imply rotation.
6. Native output is constrained, not scaled. Ring/consumer mismatch is an
   input-independent placeholder, never overlap content.

## Lifecycle and privacy review

- Camera transform fields and canvas dimensions remain restart-only.
- Backdrop fit/anchor updates are planned on detached synthetic pixels. A
  failed, timed-out, or conflicting trial never calls the working video/live
  provider or changes its fitted cache.
- A successful commit updates the reused provider policy at the same frame
  boundary as config/version publication, invalidates only fitted pixels, and
  increments the visual generation.
- Raw publication and its replay fingerprint consume the same normalized,
  uncorrected BGR frame.
- Remote/avatar output must already match the canonical canvas. Wrong-size,
  malformed, raw-echo, and delayed-echo candidates retain the existing
  input-independent privacy-slate behavior.
- Repeated output reuses the last normalized, guarded frame and performs no
  capture or backdrop geometry work.

## Performance observation

Phase 4 owns calibrated p95 and cross-platform budgets. As a non-gating Phase-1
observation, 100 warmed transforms on the current x86_64 development container
(Python 3.14.4, NumPy 2.5.1, OpenCV 5.0.0) measured:

| Transform | p50 | p95 |
| --- | ---: | ---: |
| 640x480 -> 1280x720 `cover` | 0.224 ms | 0.232 ms |
| 1920x1080 -> 1280x720 `cover` | 0.898 ms | 1.737 ms |
| Exact 1280x720 no-op | 0.004 ms | 0.004 ms |

Deterministic tests, rather than these machine-specific timings, gate Phase 1:
one transform per delivered camera frame, none on repeated pipeline output,
one fitted backdrop result per raw/cache key, no second blur fit, a 512-entry
plan bound, and constant latest-frame provider storage.

## Evidence boundaries

Two platform observations remain deliberately unclaimed:

- This repository has mocks for qualified and opaque OpenCV video-rotation
  behavior but no portable real metadata-rotated container/backend fixture.
  An opaque real backend therefore follows the tested manual-only policy; live
  backend qualification remains VIS-3.3/VIS-4.2 work.
- The native camera remains a planned, auto-disabled Windows feature. Source
  and static contracts are covered locally, and a pinned `windows-2022` CI job
  is defined to compile the project and inspect its COM exports. This local
  review cannot execute that Windows runner. The live Windows
  Camera/Teams/Zoom/browser matrix remains the existing WIN-6.1/VIS-4.2
  qualification gate. Unsupported modes already fail clearly, and a negotiated
  ring mismatch exposes no source pixels.

These are bounded qualification limits, not silent correctness claims or open
Phase-1 implementation decisions.

## Verification record

The closing verification was run on the final Phase-1 source tree:

- Full Python suite: `.venv/bin/pytest -q` completed with **1,206 passed,
  3 skipped** on Python 3.14.4, NumPy 2.5.1, and OpenCV 5.0.0.
- Minimum compatibility lane: the eight Phase-1 suites completed with
  **372 passed** in the official `python:3.10-slim` image using Python
  3.10.20, NumPy 1.24.0, and OpenCV 4.8.1.
- Native/output/Windows packaging gate:
  `test_windows_vcam.py`, `test_output_geometry.py`, and
  `test_windows_packaging.py` completed with **89 passed**. This includes the
  deterministic in-copy deactivation regression and the SDK target contract.
- Node release suite: `npm test` completed with **141 passed, 2 intentional
  TODO** checks for the repository's already-open REL-01 and WIN-01 production
  publication gates. The exact npm payload verifier passed.
- Static gates: Ruff lint passed; Ruff formatting reported all **94 files**
  formatted; Pyright reported **0 errors, 0 warnings**; workflow YAML parsed;
  the focused Node release/workflow policy tests completed **3/3**; and
  `git diff --check` passed.
- Package gates: a fresh source copy built both
  `custback-0.4.0-py3-none-any.whl` and `custback-0.4.0.tar.gz`. The wheel
  contains every Phase-1 runtime module, the sdist contains the five new
  geometry/canvas suites, and a wheel-only import/640x480-to-1280x720 transform
  smoke test passed.

The Windows C++ job is defined against the pinned `windows-2022` image and
targets its installed SDK 10.0.22621.0, but it was not executed from this Linux
workspace. Live Windows consumer qualification and real-backend rotation
qualification remain explicitly bounded by WIN-6.1/VIS-4.2 and VIS-3.3/VIS-4.2
respectively, as recorded above.
