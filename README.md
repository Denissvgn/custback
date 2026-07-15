# custback

Local virtual camera with background replacement for meeting apps and
browsers, on **Ubuntu (26.04 LTS)** and **macOS**.

```
real camera ──► segmentation ──► compositor ──► virtual camera ──► Zoom/Meet/Teams/browser
                (person mask)      │  ▲
                                   ▼  │
                              HTTP / WebSocket API
                     (preview, control, avatar frame forwarding)
```

* **Backgrounds**: static image, **live backdrops** (looping video file, a
  second camera, or a network stream URL), blur, solid color, passthrough.
* **Rendering quality**: true alpha matting (RVM), edge-aware mask
  refinement, light wrap, color-spill removal, person-free background blur —
  see [Rendering quality & GPU](#rendering-quality--gpu-acceleration).
* **NVIDIA GPU acceleration** (optional): matting runs on CUDA 12 when
  `onnxruntime-gpu` completes verified CUDA inference; use the `rvm` extra for an
  explicitly CPU-only npm installation.
* **API**: control everything at runtime, preview in a browser, and forward
  frames to an external service — the integration point for the upcoming
  avatar-replacement stage.
* **Local-first**: everything runs on your machine; the API binds to
  `127.0.0.1` by default.

## Install

### Via npm (recommended)

```bash
npm install -g custback
# Local release artifact: TARBALL=$(npm pack) && npm install -g "./$TARBALL"
custback setup              # one-time OS virtual-camera setup (v4l2loopback / OBS)
custback doctor             # verify the installation
custback --mode blur --preview
```

The npm package is a self-contained launcher: on install it finds a suitable
Python (>= 3.10 and < 3.15, preferring versions with MediaPipe wheels), creates a private
venv inside the package, and installs the bundled Python app into it. Wrapper
subcommands:

| Command | Purpose |
| --- | --- |
| `custback setup` | OS-level virtual camera setup (runs the right script for your platform) |
| `custback doctor` | validate the managed venv, versions, dependencies, and selected extras; setup gaps are warnings |
| `custback rebuild` | build and validate a new venv generation, then switch to it atomically |
| anything else | passed through to the app (`custback --help`) |

Environment overrides: `CUSTBACK_VENV=/dedicated/path` relocates the private
venv. The target must be absent or already owned by custback; rebuild refuses
unsafe, unrelated, and unmarked directories and never recursively deletes the
configured path. `CUSTBACK_SKIP_INSTALL=1` skips the Python bootstrap at install
time. `CUSTBACK_EXTRAS=gpu` (or `rvm`) requires that matting backend; explicitly
requested extras are never silently discarded. MediaPipe is attempted by
default and may fall back to the core heuristic backend when it was not
explicitly requested. See
[Rendering quality & GPU](#rendering-quality--gpu-acceleration).
Each installer subprocess is bounded to 15 minutes by default
(`CUSTBACK_INSTALL_TIMEOUT_MS` accepts a positive millisecond override); doctor
probes use a 60-second bound (`CUSTBACK_DOCTOR_TIMEOUT_MS`).

On the first 0.3.x rebuild, a legacy npm-managed venv is adopted only when its
old custback stamp and launcher layout validate. It is retained as the rollback
generation until the newly built environment passes imports, dependency, and
CLI probes; unrelated custom venvs must be moved aside explicitly.

### Manual (pip)

#### Ubuntu

```bash
./scripts/install_linux.sh      # installs v4l2loopback, creates "custback Camera" (/dev/video10)
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mediapipe,dev]'   # add ,gpu (NVIDIA) or ,rvm for matting — see below
```

`exclusive_caps=1` is preconfigured so Chrome/Electron apps (Zoom, Teams,
Slack) list the device.

#### macOS

```bash
./scripts/install_macos.sh      # installs OBS, which provides the virtual camera extension
# open OBS once, click "Start Virtual Camera" to register the extension, then quit OBS
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mediapipe,dev]'
```

The device appears as **OBS Virtual Camera** in meeting apps.

> MediaPipe wheels lag new Python releases; if `pip install -e '.[mediapipe,dev]'`
> fails to resolve mediapipe, install without the extra (`pip install -e '.[dev]'`)
> — custback then uses a low-quality fallback segmenter and logs a warning.
> Use a Python version with mediapipe wheels (3.11/3.12) for production quality.

## Run

```bash
custback --mode blur                        # blurred real background
custback --image ~/walls/office.jpg         # static backdrop
custback --video ~/walls/beach_loop.mp4     # live backdrop (loops)
custback --bg-camera 2                      # live backdrop from a second camera
custback --bg-camera rtsp://cam/stream     # ... or a network stream
custback --mode color                       # green-screen style solid color
custback -c config/default.yaml             # everything from YAML
custback --synthetic --no-vcam              # hardware-free demo (test pattern)
custback --mode blur --preview              # verify on screen (q/ESC quits)
custback --camera-pixel-format backend       # opt out of V4L2 MJPEG negotiation
custback --camera-mode-mismatch error        # fail instead of warning on mismatch
```

On Linux/V4L2, the default `camera.pixel_format: auto` requests MJPEG before
the dimensions and rate. This avoids the common silent 720p YUYV fallback to
10 FPS on cameras that support 720p30 only in MJPEG. Custback reads the
negotiated mode back after the first frame and reports both input and output
rates separately. Short capture stalls keep the last safe processed frame on
the virtual camera while the device is reopened; the default
`camera.recovery_timeout_s: 10` then fails clearly instead of freezing forever.

`--preview` opens a window showing exactly what the virtual camera sends,
with the active mode and fps overlaid — the quickest way to verify
functionality before joining a meeting. (Headless alternative: the MJPEG
preview at <http://127.0.0.1:8710/>.)

The window is interactive — a hint bar at the bottom is always visible, and
`h` expands it into full help:

| Key | Action |
| --- | --- |
| `0`–`5` | switch mode: passthrough / blur / color / image / video / camera |
| `n` / `p` | next / previous background file (image+video mode) or color preset (color mode) |
| `[` / `]` | decrease / increase blur strength |
| `h` | toggle the help overlay |
| `q` / `ESC` | quit (stops the whole app) |

Background files for `n`/`p` cycling are read from
`~/.local/share/custback/backgrounds` — the same directory `POST
/background/image` and `POST /background/video` upload into, so files added
through the API show up in the preview's cycling immediately. Pressing `3`
(image) or `4` (video) with no files there yet shows an on-screen hint
instead of silently doing nothing.

Open <http://127.0.0.1:8710/> and enter the local API token for a live
preview. The authenticated local API index at
<http://127.0.0.1:8710/docs> links to the OpenAPI document.

Every run also writes a private rotating diagnostic log to
`$XDG_STATE_HOME/custback/custback.log` (or
`~/.local/state/custback/custback.log`): 5 MiB plus three backups, all mode
`0600`. Use `--log-file PATH` to override it or `--no-file-log` to keep only
stderr. Logs and `GET /status` share a short run ID; accepted live controls,
fallback transitions, capture recovery, readiness, and the shutdown summary
are recorded with credential-free summaries and without paths, URLs, or API
tokens.

## Rendering quality & GPU acceleration

The person/background boundary is where composites live or die, so several
stages work on it (all tunable live via `PATCH /config`, defaults on):

* **Segmentation backends** — `segmentation.backend: auto` picks the best
  installed one: **rvm** ([Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting),
  true alpha matting with hair-level edges, temporal consistency and a clean
  foreground prediction) → **mediapipe** (selfie segmentation) → heuristic
  fallback. Built-in models live in `~/.cache/custback/models`; their pinned
  size and SHA-256 are verified on every use, and downloads are locked,
  bounded, and published atomically.
* **Edge-aware refinement** (`segmentation.edge_refine`) — bounded marker
  watershed may move the contour only inside an eight-pixel uncertainty band,
  so it follows nearby hair and shoulder edges without disturbing the mask
  elsewhere. RVM mattes skip this binary-mask operation.
* **Halo control** (`segmentation.mask_shift`) — grow/shrink the mask by N
  pixels; `-1`/`-2` removes leftover background fringes.
* **Adaptive temporal smoothing** (`segmentation.temporal_smoothing`) —
  static regions are damped against flicker while moving edges track
  immediately (no ghost trails).
* **Light wrap** (`compositing.light_wrap`) — backdrop light bleeds subtly
  into the person's edge band, the classic compositing trick that makes the
  subject sit *in* the scene rather than on top of it.
* **Color-spill removal** (`compositing.use_model_foreground`, rvm only) —
  edge pixels contaminated by your real room's colors are replaced with the
  model's clean-foreground prediction.
* **Person-free blur** — in blur mode the person is excluded from the
  background blur (normalized masked convolution), so they leave no smeared
  ghost around their own silhouette. The blur also runs at reduced
  resolution: same look, roughly 10x cheaper at 720p.

### Using the NVIDIA GPU

The rvm backend runs on CUDA 12 when `onnxruntime-gpu` is installed (needs an
NVIDIA driver plus CUDA 12 runtime and cuDNN 9; ~10x faster than CPU matting).
The `gpu` extra deliberately stays below ONNX Runtime 1.27 because its Python
3.14 Linux wheel requires CUDA 13. `nvidia-smi` reporting CUDA 13.x means the
driver can support that generation; it does not install the CUDA 13 runtime.
The CUDA compiler/toolkit is not needed when the CUDA 12 runtime libraries are
already present.

```bash
# npm install
CUSTBACK_EXTRAS=gpu custback rebuild     # or set it on npm install -g custback

# pip install
pip install -e '.[gpu,dev]'              # CPU-only matting instead: .[rvm,dev]
```

No configuration is needed at runtime: `backend: auto` prefers rvm, and rvm
prefers CUDA. The npm `gpu` extra runs a real profiled ONNX inference and
validates that its node executed on CUDA, rather than trusting provider
registration alone; choose `rvm` for CPU inference. Check what's active with
`custback doctor`, `GET /status` (`segmentation_backend` /
`segmentation_device`), or the preview overlay (e.g. `blur 30 fps rvm/cuda`).

MediaPipe users can try its GPU delegate with `segmentation.delegate: gpu`
(falls back to CPU with a warning if the delegate can't initialize).

## Runtime API

### Authentication and network boundary

Every data, media, upload, documentation, and WebSocket route requires a
Bearer token. Resolution order is `CUSTBACK_API_TOKEN`, then
`~/.config/custback/api-token` (or `api.token_file`). On first start custback
generates 32 random bytes and writes the token file with mode `0600`. Use the
explicit command below when you need to enter it in the browser login shell:

```bash
custback --show-api-token
```

The browser exchanges that token at `POST /auth/session` for a bounded
server-side `HttpOnly`, `SameSite=Strict` cookie; `DELETE /auth/session` signs
out. Tokens are never returned by `/config`, logged normally, or accepted in
URLs. Native clients must send `Authorization: Bearer ...`. This shell helper
feeds curl's header through standard input, keeping the secret out of curl's
command-line arguments:

```bash
export CUSTBACK_API_TOKEN="$(custback --show-api-token)"
auth_header() { printf 'header = "Authorization: Bearer %s"\n' "$CUSTBACK_API_TOKEN"; }
auth_header | curl --config - http://127.0.0.1:8710/status
```

Loopback is the safe default. A non-loopback bind is refused unless
`api.allow_non_loopback` is true, `api.allowed_origins` contains explicit
exact HTTPS origins, the token is strong, and both TLS certificate/key paths
are configured. `Host` and browser `Origin` are checked exactly; wildcard and
`null` origins are rejected.

| Endpoint | Purpose |
| --- | --- |
| `GET /status` | run ID; input/output FPS; capture/drop/repeat/skip counters; stage timings; active and fallback backends |
| `GET /config` / `PATCH /config` | read / partially update config live |
| `POST /background/image` | upload static backdrop and switch to it |
| `POST /background/video` | upload live (video) backdrop and switch to it |
| `GET /backgrounds` | list uploaded backdrops |
| `DELETE /backgrounds/{id}` | delete an inactive uploaded backdrop |
| `GET /video/mjpeg` | processed output as MJPEG stream |
| `GET /video/snapshot.jpg` | single processed frame |
| `WS /ws/frames?stream=raw\|output` | binary JPEG frame forwarding (see below) |

Examples:

```bash
auth_header | curl --config - -X PATCH http://127.0.0.1:8710/config \
     -H 'content-type: application/json' \
     -d '{"background": {"mode": "blur", "blur_strength": 51}}'
auth_header | curl --config - -X POST http://127.0.0.1:8710/background/image \
     -F file=@office.jpg
auth_header | curl --config - -X POST http://127.0.0.1:8710/background/video \
     -F file=@beach_loop.mp4
```

`GET /config` always describes effective behavior and carries
`X-Config-Version`. PATCHes are serialized and transactional. Background,
segmentation, compositing, remote timeout, and remote fallback fields can
activate live; camera, output, API bind/security, and upload-limit changes
return `409 restart_required`. A mixed hot/restart PATCH applies nothing.
Invalid content returns `422`, activation unavailability returns `503`, and a
no-op preserves the version.

Uploads are streamed in 1 MiB chunks to mode-`0600` UUID files, inspected
before activation, and committed only after backdrop preflight. Defaults are
20 MiB/16 MP per image, 256 MiB/4K per video, and 2 GiB/100 files in aggregate.
Oversize, unsupported, invalid, quota, and active-delete failures use
`413`, `415`, `422`, `507`, and `409` respectively.

## Avatar stage (stage 2) integration

Full avatar replacement (user tracking + facial-expression matching) plugs in
through the WebSocket frame API — no pipeline changes needed:

1. The avatar service connects to `WS /ws/frames?stream=raw` and receives
   every raw camera frame as binary JPEG.
2. It renders the avatar and sends frames back on the same socket.
3. With `background.mode = "remote"`, returned frames become the virtual
   camera output. If the service stalls longer than `api.remote_timeout_ms`,
   custback uses `background.remote_fallback_mode` (blur by default). Missing,
   stale, malformed, wrong-sized, prior-session, and disconnected output can
   never reveal the raw capture. If local segmentation/compositing also fails,
   custback fails closed to a full-frame blur.

A runnable reference client is in
[`examples/avatar_client.py`](examples/avatar_client.py):

```bash
custback --mode remote &
python examples/avatar_client.py
```

Video backdrops advance from monotonic media time rather than one frame per
pipeline call: early calls reuse, late calls skip, large jumps seek, and loops
retain wall-clock phase. Container timing is preferred when reliable;
validated source FPS is otherwise used, with a documented nominal 30 FPS for
invalid or variable-rate metadata. A decode failure retains the last good
frame.

## Migrating to 0.3.0

This release intentionally breaks the old unauthenticated control plane:

* Fetch or provision the token, add Bearer authentication to every native
  HTTP/WebSocket client, and use the root login shell for browsers. Tokens in
  query strings are no longer supported.
* Treat PATCH as a transaction. Restart-only or mixed patches now return
  `409 restart_required`; clients should read `X-Config-Version` and retry
  conflicts from a fresh `GET /config`. Python embedders must likewise route
  live changes through `Pipeline.apply_config_patch`; direct
  `RuntimeConfig.update()` and config-only `RuntimeConfig.commit()` calls are
  rejected.
* Upload callers must handle `201` metadata plus the committed config version,
  the immutable size/quota limits, and UUID asset IDs. Active assets must be
  switched away before deletion.
* Remote mode no longer falls through to raw camera frames. Configure a local
  `background.remote_fallback_mode`; the default is privacy-safe blur.
* Configuration is strict: unknown fields, coercible strings, out-of-range
  values, and unsafe network settings are startup/API errors. Even positive
  blur/feather kernels are still normalized upward to the next odd value.

## Layout

| Module | Responsibility |
| --- | --- |
| `capture.py` | camera sources (OpenCV, synthetic test pattern) |
| `segmentation.py` | person mask: RVM matting (CUDA/CPU) / MediaPipe / heuristic fallback; bounded marker-watershed refinement + adaptive temporal smoothing |
| `backgrounds.py` | backdrop providers: image, video loop, second camera/stream, person-free blur, color |
| `compositor.py` | alpha blending of person over backdrop; light wrap + color-spill removal |
| `diagnostics.py` | secure rotating logs, run correlation, and safe config audit records |
| `gpu_probe.py` | real CUDA inference/profile capability probe for installer and doctor |
| `vcam.py` | virtual camera output (pyvirtualcam → v4l2loopback / OBS extension) |
| `hub.py` | thread-safe frame exchange between pipeline and API |
| `preview.py` | interactive on-screen verification window (main thread; mode/file/blur controls, q/ESC quits) |
| `pipeline.py` | main loop; transactional frame-boundary reconfiguration |
| `api/server.py` | authenticated FastAPI control, uploads, MJPEG, and WebSockets |

## Tests

The whole pipeline is testable without a camera, virtual camera, or mediapipe:

```bash
pytest
npm test
npm run release:check   # build, inspect, install, and smoke-test Python/npm artifacts
```

Release-gate subprocesses have a 15-minute default bound; set
`CUSTBACK_RELEASE_TIMEOUT_MS` to a positive millisecond value for slower build
hosts.
