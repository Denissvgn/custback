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
* **Avatar stage**: the bundled `custback-avatar` service replaces you with
  an animated avatar — expression tracking (MediaPipe) or NVIDIA
  Audio2Face-3D lip sync, selectable visible parts, scale/position, and its
  own backdrop — locally or from another GPU host. See
  [Avatar stage (stage 2)](#avatar-stage-stage-2).
* **API**: control everything at runtime, preview in a browser, and forward
  frames to an external service — the integration point the avatar stage
  builds on.
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
| `custback avatar …` | run the bundled stage-2 avatar service (`custback-avatar` in the managed venv) |
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

Raw-frame renderers use a separate least-privilege credential. Custback
provisions `api.renderer_token_file` (default
`~/.config/custback/renderer-token`); renderer clients load it or
`CUSTBACK_RENDERER_TOKEN` without creating a missing file. This credential is
accepted only on `WS /ws/frames?stream=raw` and cannot authenticate REST,
browser sessions, or the output stream. Inspect/provision it explicitly with
`custback --show-renderer-token`.

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
| `GET /backgrounds` | list uploaded backdrops (with the store directory) |
| `GET /backgrounds/{id}/thumbnail.jpg` | downscaled preview of an uploaded backdrop |
| `DELETE /backgrounds/{id}` | delete an inactive uploaded backdrop |
| `GET /video/mjpeg` | processed output as MJPEG stream |
| `GET /video/snapshot.jpg` | single processed frame |
| `GET /` | the web control UI (see below) |
| `/avatar/{path}` | authenticated reverse proxy to the avatar control API (see below) |
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
return `409 restart_required`. Avatar proxy URL, token file, CA bundle, and
mTLS identity are likewise startup-only. A mixed hot/restart PATCH applies nothing.
Invalid content returns `422`, activation unavailability returns `503`, and a
no-op preserves the version.

Public config responses and their OpenAPI models omit management/renderer
token paths, proxy credential/trust paths, and private-key paths.

Uploads are streamed in 1 MiB chunks to mode-`0600` UUID files, inspected
before activation, and committed only after backdrop preflight. Defaults are
20 MiB/16 MP per image, 256 MiB/4K per video, and 2 GiB/100 files in aggregate.
Oversize, unsupported, invalid, quota, and active-delete failures use
`413`, `415`, `422`, `507`, and `409` respectively.

### The web control UI

`GET /` (after the browser login shell) serves a single-page control UI for
the whole system: a live preview of the vcam output (or the raw avatar
render), background selection with thumbnail galleries, uploads for both
stores, Meet-style avatar tiles, the animation-mode picker (follow my
movements / follow my voice only / idle presence), and sliders for framing,
size, position, and smoothing. Everything hot-patchable applies live to the
preview; controls that would need a restart surface the `409` reason instead
of failing silently.

The page talks to this origin only. Avatar controls go through the
`/avatar/{path}` reverse proxy. Configure the operator-owned `avatar:` URL,
existing control-token file, and optional CA/mTLS files before starting
custback; the browser cannot edit outbound destinations or credential paths.
Custback resolves one immutable target and injects the avatar Bearer token
server-side. The proxy disables redirects and environment proxies, forwards
only exact documented method/path pairs, and maps missing/unreachable/bad-auth
services to `503 avatar_unconfigured`, `502 avatar_unreachable`, or
`502 avatar_auth_failed` without expiring the browser session.

The header's **Avatar** toggle is plain config: enabling patches
`background.mode: remote` (remembering the previous local mode in
`background.remote_fallback_mode`); disabling restores that fallback mode.
While remote mode is active, renderer stalls always show the fixed privacy
slate; the remembered local mode is used only when Avatar is disabled.

## Avatar stage (stage 2)

Avatar replacement plugs in through the WebSocket frame API — no pipeline
changes needed:

1. An avatar service connects to `WS /ws/frames?stream=raw` and receives
   every raw camera frame as binary JPEG.
2. It renders the avatar and sends frames back on the same socket.
3. With `background.mode = "remote"`, returned frames become the virtual
   camera output. If the service stalls longer than `api.remote_timeout_ms`,
   custback emits one fixed, opaque, camera-independent privacy slate. Missing,
   stale, malformed, wrong-sized, invalid-mask, raw/near-raw, delayed-raw,
   prior-session, and disconnected output all fail closed to that same slate.
   The final gate protects startup probes, repeats, preview publication, and
   the virtual-camera sink.

### The built-in avatar service: `custback-avatar`

The package ships that stage-2 service. It tracks you in the forwarded
camera frames, animates an avatar (52 ARKit blendshape channels + head
pose), composites it over its own selected background at exactly the camera
frame size, and returns the frames — a person-like presence in the meeting
without your pixels ever leaving the machine that runs custback-avatar.

```bash
custback --mode remote &                # custback shows what the avatar service returns
custback-avatar                         # connects to ws://127.0.0.1:8710, renders the avatar
custback-avatar -c config/avatar.yaml   # everything from YAML (see the annotated example)
custback-avatar --avatar robin --style realistic --framing bust
                                        # a different presenter, soft-shaded,
                                        # head-and-chest framed for meeting tiles
custback-avatar --parts head,eyes,brows,nose,mouth,hair --scale 0.6 \
    --bg-image ~/walls/office.jpg       # floating head over an office backdrop
```

**Animation drivers** (`driver.backend`):

| Driver | Input | Notes |
| --- | --- | --- |
| `vision` | raw camera frames | MediaPipe Face Landmarker: blendshapes + head pose follow your real expressions. Needs the `mediapipe` extra; the pinned `face_landmarker.task` model is verified and cached like the segmentation models. |
| `audio2face` | microphone / WAV audio | Streams audio to an [NVIDIA Audio2Face-3D](https://huggingface.co/nvidia/Audio2Face-3D-v3.0) endpoint and applies the returned ARKit blendshapes — lip sync from speech, no camera tracking. Use `grpc://127.0.0.1:port` only on numeric loopback or verified `grpcs://host:port` remotely, with optional private CA/mTLS files. Needs the `audio2face` extra and a running Audio2Face-3D service. |
| `idle` | none | Deterministic synthetic presence: gentle head sway, periodic blinks. |
| `auto` (default) | — | `vision` when mediapipe is installed, otherwise `idle` with a warning. |

**Appearance** is controlled live: `appearance.avatar` picks the builtin
presenter (`casey`, `robin`, `alex`, `nova` — different skin/hair/wardrobe),
`appearance.style` the treatment (`cartoon`, `realistic` for natural
proportions with soft shading, `sketch` for a pencil drawing), and
`appearance.framing` how much stays in shot: `bust` (default) keeps the
head **and chest** in frame like a webcam — the right look for Meet/Zoom
tiles — while `full` shows the waist-up shot and `closeup` fills the frame
with the face. `appearance.parts` selects the visible layers
(`torso, head, mouth, nose, eyes, brows, hair`), `appearance.scale` sets
the framed avatar height relative to the frame (0.1–3),
`offset_x`/`offset_y` move it, and `background.mode` picks the scene behind
it (`color`, `image`, `video`, or `blur` of the real room). Point
`appearance.rig` at a directory of PNG layers (`head.png`, `torso.png`, …,
optional `eyes_closed.png`/`mouth_open.png` variants and a `rig.yaml` for
pivot/sway/framing tuning) for a custom look; PNG rigs render as authored
(`sketch` still applies, `avatar`/`realistic` are builtin-only).

**Control API**: the service has its own Bearer-authenticated control plane
on `127.0.0.1:8711` with its own token (`custback-avatar --show-api-token`,
env `CUSTBACK_AVATAR_API_TOKEN`): `GET /status`; `GET /avatars` for the
selectable avatars/styles/framings/parts plus installed rigs and the
animation `modes` (each mapped to a `driver.backend` with availability for
this host); `GET`/`PATCH /config`; `GET /video/snapshot.jpg`;
`GET /video/mjpeg`; and tile thumbnails at
`GET /avatars/{name}/thumbnail.jpg` / `GET /rigs/{name}/thumbnail.jpg` /
`GET /backgrounds/{name}/thumbnail.jpg`. Appearance, background, driver, and
render fields patch live; `source`, `storage`, and `api` changes return
`409 restart_required`. The same non-loopback rules as custback's API apply
(explicit opt-in + TLS + exact origins).

**Uploads live on the avatar host** (they are its inputs): `POST
/rigs?name=<slug>` installs a zipped PNG-layer rig — members are
allow-listed by name, capped in count and size, and validated by actually
loading the rig before the name becomes visible; `GET /rigs` lists installs
and `DELETE /rigs/{name}` refuses the active one. `POST
/backgrounds/image|video?name=<file>` (raw body) stores scene media for
`background.image_path`/`video_path`, listed and deleted via
`GET`/`DELETE /backgrounds…`. The `storage:` config section sets both
directories and all quotas, including compressed plus extracted rig staging,
aggregate rig count/bytes, and per-layer/total decoded pixels. Managed
directories/files are kept at exact `0700`/`0600` modes. Audit or repair a
pre-existing store before startup with:

```bash
custback-avatar -c config/avatar.yaml --check-storage-permissions
custback-avatar -c config/avatar.yaml --fix-storage-permissions
```

Through custback's `/avatar/*` proxy the web UI reaches all of this with the
browser session alone — uploads land on whichever machine renders the avatar.

### Running the avatar service on another host

Rendering and tracking can move off the meeting machine — e.g. custback on
a laptop and the avatar service next to a bigger GPU:

```bash
# meeting machine: TLS-protected non-loopback custback API
custback --mode remote --api-host 0.0.0.0 --allow-non-loopback-api \
    --api-tls-cert cert.pem --api-tls-key key.pem     # + api.allowed_origins in YAML

# GPU host: connect back over verified WSS with the renderer-only token.
# Configure source.tls_ca_file in config/avatar.yaml for a private CA.
CUSTBACK_RENDERER_TOKEN="$(cat renderer-token)" \
custback-avatar -c config/avatar.yaml \
  --source wss://laptop.example:8710 --driver vision
```

GPU guidance:

* **Local NVIDIA GeForce RTX 3060** — use the `vision` driver (MediaPipe,
  CPU-friendly) for the avatar and the existing `gpu` extra for RVM matting.
  Audio2Face-3D officially supports GeForce RTX 3080 and up (plus
  data-center GPUs), so the RTX 3060 is *not* a supported A2F host.
* **Remote NVIDIA GB10 (DGX Spark)** — run `custback-avatar` on the GB10
  host (aarch64 Linux; the core service needs only OpenCV/NumPy wheels) and
  point `driver.audio2face.url` at an Audio2Face-3D NIM on the same box.
  Check NVIDIA's NIM support matrix for the GB10/Blackwell container before
  planning on it; the `vision` driver is the portable alternative.

The wire protocol is unchanged, so a custom stage-2 renderer still works: a
minimal reference client lives in
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
* Remote mode no longer uses camera-derived local compositing as a fallback.
  Every renderer failure or raw echo produces a fixed input-independent slate;
  `background.remote_fallback_mode` is retained only as the local mode restored
  when remote/avatar mode is disabled.
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
| `avatar/` | stage-2 avatar service (`custback-avatar`): drivers (`drivers.py`, `audio2face.py`), rigs (`rig.py`), composition (`renderer.py`), WS client loop (`service.py`), control API (`api.py`) |

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
