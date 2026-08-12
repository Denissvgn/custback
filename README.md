# custback

Local virtual camera with background replacement for meeting apps and
browsers, on **Ubuntu / Debian** and **macOS**.

```
real camera ──► segmentation ──► compositor ──► virtual camera ──► Zoom/Meet/Teams/browser
                (person mask)      │  ▲
                                   ▼  │
                              HTTP / WebSocket API
                     (preview, control, avatar frame forwarding)
```

* **Backgrounds**: static image, **live backdrops** (looping video file or an
  operator-approved local camera), blur, solid color, passthrough.
* **Rendering quality**: true alpha matting (RVM), edge-aware mask
  refinement, light wrap, color-spill removal, person-free background blur —
  see [Rendering quality & GPU](#rendering-quality--gpu-acceleration).
* **Visual consistency**: one canonical output canvas, proportional fit modes,
  profile-aware local media, linear-light compositing, and bounded foreground
  color correction with frame-aligned diagnostics and reversible rollout.
* **NVIDIA GPU acceleration** (optional): matting runs on CUDA 12 when
  `onnxruntime-gpu` completes verified CUDA inference; use the `rvm` extra for an
  explicitly CPU-only npm installation.
* **Avatar stage**: the bundled `custback avatar` service replaces you with
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
# Local release artifact: TARBALL=$(npm pack --silent) && npm install -g "./$TARBALL"
custback setup              # one-time OS virtual-camera setup (v4l2loopback / OBS)
custback doctor             # verify the installation
custback --mode blur --preview
```

The npm package is a self-contained launcher: on install it finds a suitable
Python (>= 3.10 and < 3.15, preferring versions with MediaPipe wheels), creates
a private venv at a deterministic path scoped to the npm prefix (outside the
replaceable package directory), and installs the bundled Python app into it.
Wrapper subcommands:

| Command | Purpose |
| --- | --- |
| `custback setup` | OS-level virtual camera setup (runs the right script for your platform) |
| `custback doctor` | validate the managed venv, versions, dependencies, and selected extras; setup gaps are warnings |
| `custback extras` | show persisted requested extras, installed extras, and available choices (`--json` is supported) |
| `custback rebuild [--extras LIST]` | build and validate a new venv generation, then switch to it atomically; an explicit empty list clears extras |
| `custback purge [--dry-run \| --yes]` | preview or explicitly remove the ownership-validated managed Python runtime before npm uninstall |
| `custback avatar …` | run the bundled stage-2 avatar service; the npm `custback-avatar` binary is a compatibility alias |
| `custback avatar config export [PATH]` | print the bundled annotated avatar YAML, or create `PATH` without overwriting it |
| `custback avatar --smoke` | initialize and tear down the installed idle renderer without camera or network access |
| `custback-npm-migrate --prefix PREFIX` | pre-upgrade bridge for a package-local npm venv created by custback 0.3 |
| anything else | passed through to the app (`custback --help`) |

Environment overrides: `CUSTBACK_VENV=/dedicated/path` relocates the private
venv. The target must be absent or already owned by custback; rebuild refuses
unsafe, unrelated, and unmarked directories and never recursively deletes the
configured path. `CUSTBACK_SKIP_INSTALL=1` skips the Python bootstrap at install
time. `custback rebuild --extras gpu` (or `rvm`) requires that matting backend;
`audio2face` installs the remote Audio2Face protocol. `CUSTBACK_EXTRAS` provides
the same comma-separated selection. When the variable and flag are absent the
last successful intent is preserved; an explicitly empty value clears it.
Requested extras are never silently discarded. MediaPipe is attempted by
default and may fall back to the core heuristic backend when it was not
explicitly requested; `audio2face` and `mediapipe` cannot be combined because
their published protobuf constraints conflict. See
[Rendering quality & GPU](#rendering-quality--gpu-acceleration).
Each installer subprocess is bounded to 15 minutes by default
(`CUSTBACK_INSTALL_TIMEOUT_MS` accepts a positive millisecond override); doctor
probes use a 60-second bound (`CUSTBACK_DOCTOR_TIMEOUT_MS`).

The default npm compatibility profile attempts to install MediaPipe
confidence-mask segmentation, but it does not install the RVM/ONNX Runtime
**matting** tier. When present, MediaPipe is labelled the **segmentation**
capability tier; that installed-capability label is not an evidence-qualified
named preset or sustainability claim, and the
[MATTE-5.4 authority](docs/matte-quality-rollout.md) remains on compatibility
hold. `segmentation.backend: auto` can only prefer RVM when that optional
runtime is present. Choose one RVM profile explicitly for true-alpha edges:

```bash
custback rebuild --extras rvm  # CPU RVM
custback rebuild --extras gpu  # NVIDIA/CUDA RVM
custback doctor
```

The two RVM profiles are alternatives, not a combined extras list. An explicit
`--extras` list replaces the stored intent; use `custback extras --json` first
and retain any other compatible extras the installation still needs.
Installation never upgrades an unqualified platform to RVM or a higher-detail
policy automatically.

An ordinary npm replacement removes the package-local venv used by custback
0.3 before the new package's lifecycle script can inspect it. Run the candidate
package's bridge **before** that first upgrade; it validates the old ownership
stamp, moves the venv to the prefix-scoped target, rewrites its absolute launcher
paths, and persists the explicit extras intent. For a global install:

```bash
CANDIDATE=custback@0.4.0  # or an absolute path to the candidate .tgz
PREFIX=$(npm prefix --global)
CUSTBACK_SKIP_INSTALL=1 npx --yes --package "$CANDIDATE" \
  custback-npm-migrate --prefix "$PREFIX"
npm install --global "$CANDIDATE"
```

The command is restart-safe at every durable boundary and refuses symlinked,
unowned, cross-filesystem, or colliding targets. Do not run the final `npm
install` if it reports an error. Upgrades after this bridge preserve the active
and rollback generations plus selected extras automatically because they live
outside the replaceable package directory.

### Removing an npm installation

Ordinary `npm uninstall -g custback` removes the npm package and its command
shims, but intentionally retains Custback's prefix-scoped Python runtime so a
later reinstall can reuse a healthy environment. To remove that runtime too,
run the explicit purge **before** npm uninstall:

```bash
custback purge --dry-run  # `custback purge` is also a preview
custback purge --yes
npm uninstall -g custback
```

Purge removes only the selected, ownership-validated runtime link/direct venv,
its validated Python generations, and its persisted extras intent. It rejects
foreign, corrupt, or symlinked metadata and a concurrently held Custback lock
rather than recursively deleting it. The empty ownership-marked generations
directory is retained to avoid racing a future rebuild; it contains no Python
library files.
Purge does not remove the npm package itself, a linked source checkout,
OS-level virtual-camera setup, or user configuration, media, models, and
caches.

For a custom runtime, supply the same target explicitly; purge never searches
for other `CUSTBACK_VENV` locations:

```bash
CUSTBACK_VENV=/dedicated/path custback purge --yes
```

### Manual (pip)

#### Ubuntu / Debian

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
> — custback then uses the heuristic quality tier and logs a warning.
> Use a Python version with MediaPipe wheels (3.11/3.12) for the confidence-mask
> segmentation tier, or install `rvm`/`gpu` for the true-alpha matting tier.

## Run

```bash
custback --mode blur                        # blurred real background
custback --image ~/walls/office.jpg         # static backdrop
custback --video ~/walls/beach_loop.mp4     # live backdrop (loops)
custback --bg-camera 2                      # live backdrop from a second camera
custback --mode color                       # green-screen style solid color
custback -c config/default.yaml             # everything from YAML
custback --synthetic --no-vcam              # hardware-free demo (test pattern)
custback --mode blur --preview              # verify on screen (q/ESC quits)
custback --camera-pixel-format backend       # opt out of V4L2 MJPEG negotiation
custback --camera-mode-mismatch error        # fail instead of warning on mismatch
```

Geometry, canvas, blend-space, and color-correction policy are configured in
YAML (or through the authenticated hot API where supported). Start with the
annotated `config/default.yaml`; the complete policy, migration, mode
eligibility, troubleshooting, and rollback guide is
[Visual-consistency configuration and rollout](docs/visual-consistency-rollout.md).
Camera geometry and output-canvas fields require a restart; backdrop geometry
and compositing fields activate transactionally at a frame boundary.

Live-camera sources are startup authority. Remote URI schemes are rejected
because OpenCV cannot enforce verified TLS, redirect, proxy, and address-class
policy. Operators can define immutable local sources under `backdrop_targets`
in YAML and set `background.camera_target` to a public ID; the hot API may
select another configured ID but cannot supply a path, URL, or backend option.

On Linux/V4L2, the default `camera.pixel_format: auto` requests MJPEG before
the dimensions and rate. This avoids the common silent 720p YUYV fallback to
10 FPS on cameras that support 720p30 only in MJPEG. Custback reads the
negotiated mode back after the first frame and reports both input and output
rates separately. Short capture stalls keep the last safe processed frame on
the virtual camera while the device is reopened; the default
`camera.recovery_timeout_s: 10` then fails clearly instead of freezing forever.

`--preview` opens on exactly what the virtual camera sends, with the active
mode and fps overlaid — the quickest way to verify functionality before
joining a meeting. Its explicit `d` / `D` cycle switches only that native
window to private, pre-reaction matte views; those frames never enter the
virtual camera, frame hub, browser preview, or API. (Headless output-preview
alternative: the MJPEG preview at <http://127.0.0.1:8710/>.)

The window is interactive — a hint bar at the bottom is always visible, and
`h` expands it into full help:

| Key | Action |
| --- | --- |
| `0`–`5` | switch mode: passthrough / blur / color / image / video / camera |
| `n` / `p` | next / previous background file (image+video mode) or color preset (color mode) |
| `[` / `]` | decrease / increase blur strength |
| `d` / `D` | next / previous local matte diagnostic view; cycles back to production output |
| `h` | toggle the help overlay |
| `q` / `ESC` | quit (stops the whole app) |

Background files for `n`/`p` cycling are read from
`~/.local/share/custback/backgrounds` — the same directory `POST
/background/image` and `POST /background/video` upload into, so files added
through the API show up in the preview's cycling immediately. Pressing `3`
(image) or `4` (video) with no files there yet shows an on-screen hint
instead of silently doing nothing.

Matte views include raw/refined alpha, RVM foreground, the exact backdrop,
boundary/defect proxies, compositor counterfactuals, and registered temporal
instability. They are on-demand local inspection tools, not qualification
evidence; morphology and motion caveats, telemetry meanings, privacy
boundaries, and the Run-B attribution workflow are documented in
[Local live matte diagnostics](docs/matte-live-diagnostics.md).

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

To separate camera acquisition from model, compositor, API, preview, and sink
load, stop the normal process and run
`custback capture-diagnose --output NEW_DIR`. It exercises only the production
camera reader and canonical
normalization path, writes no pixels, and reports capture pacing separately
from optional matched full-runtime pacing. A full-runtime comparison uses a
strict owner-only two-snapshot sidecar passed through `--runtime-evidence`;
capture, processing, and output rates are computed from counter deltas over
that bounded window, never from lifetime counters divided by total uptime.
Exact hardware acceptance additionally requires both opaque identity digests,
an explicit hardware attestation, and a numeric or recognized local
camera-device source. File and URL streams are rejected. Deterministic tests
do not qualify physical hardware; without reviewed local evidence the report
remains `hardware-evidence-required`. See the
[capture cadence diagnostic contract](docs/capture-cadence-diagnostics.md).
To compare the exact camera/canvas modes in the server-owned Experimental
catalog, use `custback system-profile-probe --profile NAME
--accept-experimental --output NEW_DIR`. It opens and closes requested modes
sequentially, writes one owner-only pixel-free matrix, and never changes
preferences or infers full-path qualification. See
[Experimental system profiles](docs/system-profiles.md).

During a normal run, capture, segmentation, safe-base updates, base reuse,
exact final-output repeats, and output sends are separate status clocks.
The preview and control dashboard warn when near-target output is being made
from slower visual updates. Counts, rolling rates, timestamp/send jitter
summaries, deadline and pacing event classes, timing boundaries, and the typed
post-base extension are defined by the
[visual cadence observability contract](docs/cadence-observability.md).
The depth-one publisher sends at the target cadence by adopting the newest
guarded safe base or byte-identically repeating the last eligible final frame;
a scheduler reuse never reruns or advances segmentation, refinement,
compositor, or other temporal image state. A newly processed pixel-identical
base can also increment the exact-repeat clock, so reuse and byte equality stay
separate. `GET /status.runtime_performance` schema v2 reports transport and
configured-unique targets independently. Output attainment and sink deadlines
use the transport target; unique/base attainment and processing deadlines use
the unique target. An explicit 15-capture/30-transport configuration can be
healthy `intentional-repeat`, while an observed 15/30 result from a 30/30
request remains an `unexpected-shortfall`. Its config-version-bound
`recommended_mitigation` is advisory only: the dashboard applies it only after
a user click against a fresh config/status pair and rejects stale advice. It
never changes defaults,
qualifies a preset, or advances the `compatibility_hold` matte rollout. The
[operator guide](docs/matte-operator-mitigations.md#runtime-performance-recommendation)
defines the review, disable, confirmation, and rollback procedure.

Identifiable matte evidence is separate and off by default. An explicit
`--matte-diagnostics-dir NEW_DIR` records a duration/size-bounded private
raw/mask/backdrop/composite bundle; `--matte-diagnostics-mode composite-only`
records downstream output without claiming matte authority. Replay and
single-variant compositor attribution run offline with `custback matte-replay`;
the four-boundary RVM diagnosis is `custback matte-diagnose`, and bounded
same-source screening is `custback matte-ablate`. Cross-device RVM
alpha/detail/performance qualification is the separate, fail-closed
`custback matte-rvm-qualify` workflow; it does not create a runtime preset or
change the automatic default. See
[docs/matte-replay-bundle.md](docs/matte-replay-bundle.md) for the privacy,
format, and command contract, and
[docs/matte-rvm-profiles.md](docs/matte-rvm-profiles.md) for the formal
qualification matrix. The separate native-preview-only workflow is described
in [docs/matte-live-diagnostics.md](docs/matte-live-diagnostics.md); enabling
it does not persist a bundle.
End-to-end visual and sink qualification then joins those private reports with
same-generation HighGUI/API/virtual-camera captures and human review:
`custback matte-visual-qualify PRIVATE_PLAN --output NEW_DIR`. Start from
the content-free
[local template](docs/matte-visual-qualification-local-template.json) and
follow the [MATTE-5.2 runbook](docs/matte-visual-qualification.md). Generated
or fake route evidence always remains pending; qualification still needs
consented/licensed representative clips, owner-attested physical preview and
loopback captures, live-camera/60 FPS/1080p coverage, and a completed human
review. Matching pixels and declared capture methods do not cryptographically
prove physical-device origin; retain the private capture record.
The 720p compositor/service budget is measured separately with
`custback matte-performance PRIVATE_BUNDLE --output NEW_DIR`. It runs the
eight-cell compositor matrix without capture or output pacing and remains
`not_decidable` until a source-matched, model-backed RVM/CUDA full-service
sidecar is supplied. On qualified local hardware,
`--collect-full-path --hardware-id TIER --sink-backend pyvirtualcam` produces
and joins that evidence through the real unpaced sink-submission seam; it
requires 330 distinct replay frames at default settings and never credits
repeats. Its fixed-replay background profile measures a resident recorded-frame
copy, and the cadence sweep includes the synchronous status/publication tail
after submission. The ratified 22 ms compositor sub-budget is enforced again
on the full hardware path; signed service headroom, the explicit
30/27/24/20/15 FPS arrival sweep, and lower-rate classification rules are in
[docs/matte-performance.md](docs/matte-performance.md) and
[ADR 0003](docs/adr/0003-720p-compositor-budget.md).
Cross-platform performance, sink, fallback, and lifecycle evidence is joined
without changing a preset or default by
`custback matte-platform-qualify PRIVATE_PLAN --output NEW_DIR`. Start from
the content-free
[local MATTE-5.3 template](docs/matte-platform-qualification-local-template.json)
and follow the
[platform qualification runbook](docs/matte-platform-qualification.md).
The checked-in JSON is a shape skeleton rather than the full code-owned matrix,
and v1 validates sidecars from an owner-controlled collector rather than
opening hardware itself. Generated evidence remains pending: qualification
requires owner-attested physical route observations, independent capture-only
and fixed-replay runs, sustained and restart/hot-patch/shutdown observations,
and the exact reviewed platform routes with reactions disabled.
MATTE-5.4 therefore keeps the schema-1 matte policy and default on an explicit
compatibility hold. The separately versioned server catalog exposes only
acknowledged `experimental` or `locally_screened`, `quality_claim: false`
concrete profiles; one-host screening cannot promote them to portable
qualified presets. The
[matte rollout, migration, and rollback guide](docs/matte-quality-rollout.md)
links the baseline, ablation, visual, performance, platform, privacy, and
migration gates; defines sanitized canary counters; and provides the exact
single transactional rollback patch. That rollback preserves the user config,
recognized schema, optional packages, and model cache. A future promotion is a
separate evidence-bound release decision, not an inference from installed RVM
or generated qualification fixtures. Reactions remain a separate `REACT` lane.
For reversible, backend-aware troubleshooting while qualification is pending,
use the
[immediate matte operator guide](docs/matte-operator-mitigations.md).
Output-rate matte interpolation remains rejected and exact repeat remains the
automatic cadence policy; the evidence limits, four-strategy comparison,
privacy audit, and reconsideration gates are recorded in
[ADR 0002](docs/adr/0002-output-rate-matte-interpolation.md).

## Rendering quality & GPU acceleration

The person/background boundary is where composites live or die, so several
stages work on it. The compatibility defaults and which fields are hot are
listed below; do not assume every available quality policy is enabled:

* **Segmentation backends** — `segmentation.backend: auto` picks the best
  installed one: **rvm** ([Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting),
  true alpha matting with hair-level edges, temporal consistency and a clean
  foreground prediction) → **mediapipe** (selfie segmentation) → heuristic
  fallback. Built-in models live in `~/.cache/custback/models`; their pinned
  size and SHA-256 are verified on every use, and downloads are locked,
  bounded, and published atomically. These are the **matting**,
  **segmentation**, and **heuristic** quality tiers respectively. The default
  npm profile attempts MediaPipe; RVM is optional and requires
  `custback rebuild --extras rvm` (CPU) or
  `custback rebuild --extras gpu` (NVIDIA/CUDA).
* **Edge-aware refinement** (`segmentation.edge_refine` and
  `segmentation.spatial_edge_refinement`) — schema 1 retains the historical
  bounded marker watershed. An opt-in `stable_guided` candidate uses a
  resolution-scaled, contrast/confidence-gated guided-alpha path that preserves
  soft values and falls back to the exact current matte when support is weak
  or ambiguous. It is available for private replay qualification, not selected
  as a production preset; see
  [the spatial design and evidence boundary](docs/matte-spatial-refinement.md).
  RVM mattes skip both generic spatial policies.
* **Halo control** (`segmentation.mask_shift`) — grow/shrink the mask by N
  pixels; `-1`/`-2` removes leftover background fringes.
* **Compatibility temporal smoothing** (`segmentation.temporal_smoothing`) —
  the historical frame-count EMA remains unchanged for existing
  configurations. It is locally change-gated, but it has no source-motion
  correspondence and is not claimed to eliminate every trail.
* **Motion-aware boundary stabilization**
  (`segmentation.boundary_stabilization`) — an experimental, default-off
  policy registers prior alpha with a bounded low-resolution source guide,
  applies real capture `dt`, and blends only a confidence-approved contour
  band. It is available for replay qualification, not selected as a production
  preset; see [the design and evidence boundary](docs/matte-boundary-stabilization.md).
* **Light wrap** (`compositing.light_wrap`) — backdrop light bleeds subtly
  into the person's edge band. The optional
  `compositing.light_wrap_stabilization` policy bounds changes from video or
  camera backdrops using actual backdrop time; it is experimental and
  default-off, while the historical stateless pixels remain the compatibility
  path. See the
  [light-wrap design and evidence boundary](docs/matte-light-wrap.md).
* **Color-spill removal** (`compositing.use_model_foreground`, rvm only) —
  edge pixels contaminated by your real room's colors are replaced with the
  model's clean-foreground prediction.
* **Backend-specific controls** — `segmentation.threshold` affects only the
  heuristic backend, where its compatibility score cutoff is
  `threshold × 0.8`. RVM preserves native soft alpha without threshold or
  opaque-core calibration; it neutralizes generic blur, legacy/guided spatial
  refinement, and the legacy temporal EMA while retaining `mask_shift`. An
  explicitly selected motion-aware policy remains separate and active.
  Controls are resolved as effective, bypassed, or inapplicable for the actual
  selected backend. `GET /config` reports configured intent, while existing
  `GET /status` fields report effective compatibility values; see the
  [backend-policy contract](docs/matte-backend-policies.md).
* **Person-free blur** — in blur mode the person is excluded from the
  background blur (normalized masked convolution), so they leave no smeared
  ghost around their own silhouette. The blur also runs at reduced
  resolution: same look, roughly 10x cheaper at 720p.
* **Canonical geometry** — camera and backdrop frames use the same
  `cover` / `contain` / explicit `stretch` planner, with orientation before
  viewer-horizontal mirror and anchor-controlled crop or padding. The
  schema-1 camera default remains legacy `stretch`; backdrop fit defaults to
  `cover`.
* **Linear compositing** — `compositing.blend_space: linear_srgb` performs
  correction, model-foreground replacement, light wrap, and alpha blending in
  linear light. Schema 1 keeps `srgb_legacy` for byte-compatible rollout.
* **Bounded foreground harmonization** —
  `compositing.color_correction.mode: auto` estimates restrained exposure and
  white-balance changes for image, video, and live-camera backdrops. It is
  excluded for passthrough, blur, solid color, and remote output; low
  confidence safely holds/decays or uses identity. Schema 1 keeps it `off`.

### Matte-quality rollout status

The active matte release stage is `compatibility_hold`: schema version 1,
`backend: auto`, legacy watershed/EMA behavior for applicable non-RVM paths,
native recurrent RVM alpha with generic postprocessing bypassed, stateless
light wrap, and no evidence-qualified named preset. The checked-in MATTE-5.2
and MATTE-5.3 fixtures are generated and pending; they do not authorize RVM, a
higher-detail profile, or any candidate algorithm as a new-install default.
The executable authority is
`scripts/release/matte-policy-rollout.json`: it pins the code-owned legacy
patch and digest, a non-qualified concrete preset catalog, seven pending promotion
evidence slots, non-destructive rollback, and reaction exclusion. Release
checks reject ledger/default/helper/status/catalog drift by recomputing the
recursively key-sorted JSON/ECMAScript digest and running a dependency-free
Python AST check of the helper/status contracts. Integral floats canonicalize
as integers; the current patch digest is
`27638e419a0dcf5955d52e2eb4ead2dafbdca7f2bbe0535108aa7c56c1f2f60d`.

`GET /status.matte_rollout` reports the bounded rollout stage and decision,
whether a qualified default is active, the preset catalog's evidence status,
the code-owned legacy rollback-patch ID, and aggregate
apply/success/failure/rollback counters. Read it together with
`segmentation_selection`, `matte_policy`, and the matching `config_version`:
installed capabilities and configured `auto` are not proof of the
backend/provider actually producing the frame. The WebUI presents the same
distinction. It enables an acknowledged non-qualified row only when its exact
model, provider, canvas, CLI-lock, and sink requirements are available;
portable qualification remains pending.

Old, versionless, partial, and schema-1 files retain compatibility semantics;
ordinary loading does not rewrite them. The one-patch rollback does not delete
configuration or model caches and leaves unrelated output/background/API/
avatar settings untouched. See the
[MATTE-5.4 guide](docs/matte-quality-rollout.md) and
[ADR 0004](docs/adr/0004-matte-quality-rollout.md) for the evidence chain,
canary stop/go rules, sanitized telemetry allowlist, migration behavior, and
exact rollback patch. Reactions are explicitly excluded.

### Visual-policy rollout status

The active release stage is `compatibility`:

| Policy | Current schema-1 default | Opt-in target |
| --- | --- | --- |
| Main-camera fit | `stretch` | `cover` |
| Composite space | `srgb_legacy` | `linear_srgb` |
| Foreground correction | `off` | `auto` |

The target values are implemented but are not default claims. Each flip is a
separate future commit and schema stage after VIS-4.2 calibrated fixtures,
physical cameras, consumer sinks, platform performance, privacy, and rollback
evidence approve it. The executable stage ledger is
`scripts/release/visual-policy-rollout.json`; release checks reject ledger,
template, evidence, and commit drift. Existing versionless and schema-1 files
always retain `stretch` / `srgb_legacy` / `off`.

Use `GET /status` to diagnose the exact output frame: it reports delivered,
oriented, normalized, and canvas dimensions; crop/pad/scale plans; capture and
output rates; color-correction state/reason/confidence/EV/WB/timing/counters;
the external color assumption; video tag/override/assumption status; and
read-only camera auto-control observations. The preview HUD summarizes the
same geometry and correction state. See the
[rollout guide](docs/visual-consistency-rollout.md#troubleshooting) for crop,
bars, low confidence, camera auto-controls, and tagged/untagged media.

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
| `GET /status` | run ID; capture/segmentation/base/reuse/exact-final-repeat/send cadence; strict runtime-performance v2 dual-target health and advisory mitigation; gaps, deadlines, pacing and jitter; stage timings; versioned backend selection/effective matte policy and matte rollout/default/rollback decision |
| `GET /config` / `PATCH /config` | read / partially update config live |
| `GET /profiles` | server-owned Experimental catalog, bounded availability, active/desired matches, revisions, CLI locks, and restart-pending fields |
| `POST /profiles/apply` / `POST /profiles/reset` | revision-bound staging/reset of concrete managed profile values; never restarts the process |
| `POST /background/image` | upload static backdrop and switch to it |
| `POST /background/video` | upload live (video) backdrop and switch to it |
| `GET /backgrounds` | list uploaded backdrops (with the store directory) |
| `GET /backgrounds/{id}/thumbnail.jpg` | downscaled preview of an uploaded backdrop |
| `DELETE /backgrounds/{id}` | delete an inactive uploaded backdrop |
| `GET /video/mjpeg` | processed output as MJPEG stream |
| `GET /video/snapshot.jpg` | single processed frame |
| `GET /` | the web control UI (see below) |
| `/avatar/{path}` | authenticated reverse proxy to the avatar control API (see below) |
| `WS /ws/frames?stream=raw\|output` | read-only management JPEG previews; the renderer token upgrades only `raw` to the exclusive protocol-v1 renderer lease described below |

Examples:

```bash
auth_header | curl --config - -X PATCH http://127.0.0.1:8710/config \
     -H 'content-type: application/json' \
     -d '{"background": {"mode": "blur", "blur_strength": 51}}'
auth_header | curl --config - -X PATCH http://127.0.0.1:8710/config \
     -H 'content-type: application/merge-patch+json' \
     -d '{"compositing": {"blend_space": "linear_srgb"}}'
auth_header | curl --config - -X PATCH http://127.0.0.1:8710/config \
     -H 'content-type: application/merge-patch+json' \
     -d '{"compositing": {"color_correction": {"mode": "auto", "strength": 0.5}}}'
auth_header | curl --config - -X POST http://127.0.0.1:8710/background/image \
     -F file=@office.jpg
auth_header | curl --config - -X POST http://127.0.0.1:8710/background/video \
     -F file=@beach_loop.mp4
```

`GET /config` describes configured intent and carries `X-Config-Version`;
`GET /status` reports active backend/device, backend-resolved matte controls,
strict path-free `runtime_performance` v2 health, and the fail-closed
`matte_rollout` decision. Runtime performance advice is never applied
automatically, and neither status object claims pixel quality or physical
evidence.
Profile selections are different from raw patches: the server expands a
reviewed quality/framing ID into concrete values, requires explicit
Experimental acknowledgment, and saves the complete restart-bound row in an
owner-only managed overlay without rewriting operator YAML. CLI overrides win.
The UI shows a restart banner but never receives lifecycle-shutdown authority.
See [Experimental system profiles](docs/system-profiles.md).
PATCHes are serialized and transactional. Background,
segmentation, compositing, remote timeout, and remote fallback fields can
activate live; camera, output, API bind/security, and upload-limit changes
return `409 restart_required`. Avatar proxy URL, token file, CA bundle, and
mTLS identity are likewise startup-only. A mixed hot/restart PATCH applies nothing.
Invalid content returns `422`, activation unavailability returns `503`, and a
no-op preserves the version.
Clients that need compare-and-swap semantics may send the optional canonical
integer `X-Expected-Config-Version` request header; ordinary headerless
`PATCH /config` remains backward compatible. A stale conditional patch fails
with `409 config_conflict` without changing live resources.

If a persisted image or video cannot be opened during startup, the service and
control API remain available without changing that configured intent. Output
stays on the fixed camera-independent slate and status reports
`background_fallback_active=true` with
`background_fallback_reason=asset-unavailable`; a successfully preflighted
live background patch clears the fallback. Custback never substitutes
passthrough or blur for the missing asset.

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

The Quality panel labels configured intent separately from the effective
backend, provider, fallback, and matte policy for the current config version.
Its rollout row says when defaults are held pending physical qualification;
named presets remain disabled until their concrete patches are evidence
qualified. Advanced controls remain useful for private diagnosis, but their
availability is not a recommendation. Use the MATTE-5.4 one-patch rollback
instead of deleting a config or model cache.

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

1. An avatar service connects to `WS /ws/frames?stream=raw` and receives each
   JPEG in the strict binary remote-frame protocol-v1 envelope, including its
   exact raw epoch. A renderer-token connection is the single exclusive writer
   lease: a replacement renderer revokes the prior session and fences output
   to the privacy slate. A management-token connection to the same route is a
   read-only ordinary-JPEG preview and never acquires renderer authority.
2. It renders the avatar and returns a `rendered-output` envelope carrying the
   same raw epoch on the same socket. Bare JPEG responses are rejected, and a
   delayed response is discarded rather than relabeled as current.
3. With `background.mode = "remote"`, returned frames become the virtual
   camera output. If the service stalls longer than `api.remote_timeout_ms`,
   custback emits one fixed, opaque, camera-independent privacy slate. Missing,
   stale, malformed, wrong-sized, invalid-mask, raw/near-raw, delayed-raw,
   prior-session, and disconnected output all fail closed to that same slate.
   The final gate protects startup probes, repeats, preview publication, and
   the virtual-camera sink.

### The built-in avatar service: `custback avatar`

The package ships that stage-2 service. It tracks you in the forwarded
camera frames, animates an avatar (52 ARKit blendshape channels + head
pose), composites it over its own selected background at exactly the camera
frame size, and returns the frames — a person-like presence in the meeting
without your pixels ever leaving the machine that runs `custback avatar`.
The npm `custback-avatar` binary remains available as a compatibility alias;
new scripts and examples should use the canonical subcommand.

```bash
custback --mode remote &                       # custback shows what the avatar service returns
custback avatar                                # connects to ws://127.0.0.1:8710, renders the avatar
custback avatar config export ./avatar.yaml    # export the bundled annotated template
custback avatar --smoke                        # hardware-free installed-package check
custback avatar -c ./avatar.yaml               # run everything from YAML
custback avatar --avatar robin --style realistic --framing bust
                                        # a different presenter, soft-shaded,
                                        # head-and-chest framed for meeting tiles
custback avatar --parts head,eyes,brows,nose,mouth,hair --scale 0.6 \
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
on `127.0.0.1:8711` with its own token (`custback avatar --show-api-token`,
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
custback avatar -c ./avatar.yaml --check-storage-permissions
custback avatar -c ./avatar.yaml --fix-storage-permissions
```

Through custback's `/avatar/*` proxy the web UI reaches all of this with the
browser session alone — uploads land on whichever machine renders the avatar.

### Running the avatar service on another host

Rendering and tracking can move off the meeting machine. The production
runbook is [Two-host avatar deployment](docs/remote-deployment.md). It defines
the distinct renderer-scoped token over WSS and avatar-control token over
HTTPS, both CA and certificate paths, firewall direction, safe rotation, and
fail-closed outage behavior. Remote plaintext is not a recovery option.

GPU guidance:

* **Local NVIDIA GeForce RTX 3060** — use the `vision` driver (MediaPipe,
  CPU-friendly) for the avatar and the existing `gpu` extra for RVM matting.
  Audio2Face-3D officially supports GeForce RTX 3080 and up (plus
  data-center GPUs), so the RTX 3060 is *not* a supported A2F host.
* **Remote NVIDIA GB10 (DGX Spark)** — run `custback avatar` on the GB10
  host (aarch64 Linux; the core service needs only OpenCV/NumPy wheels) and
  point `driver.audio2face.url` at an Audio2Face-3D NIM on the same box.
  Check NVIDIA's NIM support matrix for the GB10/Blackwell container before
  planning on it; the `vision` driver is the portable alternative.

The remote-frame wire contract changed in 0.4.0: custom stage-2 renderers must
upgrade to the mandatory protocol-v1 envelope and echo each positive raw epoch
exactly. Upgrade meeting-host and renderer binaries together; legacy bare-JPEG
responses fail closed. A minimal reference client lives in
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

## Migrating from 0.3 to 0.4

Run the installed migrator before starting 0.4 with an existing configuration:

```bash
custback migrate --config ~/.config/custback/config.yaml \
  --target-id legacy-camera
custback migrate --audit-storage
# After reviewing the audit:
custback migrate --repair-storage
```

A numeric or absolute-local legacy `background.camera_device` becomes an
operator-owned `backdrop_targets.legacy-camera` entry selected by
`background.camera_target`. The command retains a private byte-exact backup,
uses a digest-bound durable journal and atomic same-directory replacement, and
can be rerun after interruption. URI, remote, relative, conflicting, symlinked,
or otherwise ambiguous sources exit with status 4 and require explicit operator
action; they are never converted into hot network authority.

Versionless and explicit schema-1 configurations deterministically retain the
visual compatibility profile: camera `stretch`, `srgb_legacy` compositing,
correction `off`, and an output canvas derived from the camera request when
`output.width` / `height` are absent. The migrator materializes those values;
ordinary loading does not rewrite the file. When the negotiated camera aspect
differs from the canvas, custback emits one upgrade note per capture lifetime
explaining that a future staged `cover` default would crop rather than
distort, and how to pin or preview the policy. Later default schemas and their
rollbacks are defined in the
[visual-consistency rollout guide](docs/visual-consistency-rollout.md#deterministic-upgrade-behavior).

The same migration materializes the schema-1 matte compatibility policy:
`backend: auto`, legacy watershed and frame-count EMA where applicable,
motion-aware stabilization off, temporal light-wrap stabilization off, and
the existing backend-specific RVM bypass semantics. It never converts
`temporal_smoothing` into a time constant or promotes a named preset. The
complete field set and non-destructive rollback are in the
[matte rollout guide](docs/matte-quality-rollout.md#one-patch-rollback).

Storage audit is no-follow and non-mutating. Repair first rejects symlinks,
special files, foreign ownership, overlapping roots, and inode changes, then
sets safe managed directories to `0700` and regular assets to `0600`. Durable
upload/cleanup ownership ledgers live beside the payload roots and are not
changed or dropped. Use `--core-store`, `--avatar-rigs-store`, and
`--avatar-backgrounds-store` to audit non-default locations.

The npm 0.3 package-local runtime needs the separate pre-upgrade bridge shown in
[npm installation](#via-npm-recommended); npm removes that directory before a new
postinstall can recover it.

## Legacy 0.3 client changes

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
| `segmentation.py` | person mask: RVM matting (CUDA/CPU) / MediaPipe / heuristic fallback; spatial refinement and timeline-owned temporal processing |
| `matte_policy.py` | typed configured-versus-effective backend policy and applicability resolver |
| `backgrounds.py` | backdrop providers: image, video loop, approved local camera target, person-free blur, color |
| `compositor.py` | alpha blending of person over backdrop; light wrap + color-spill removal |
| `light_wrap.py` | generation-owned elapsed-time stabilization for dynamic-backdrop wrap samples |
| `diagnostics.py` | secure rotating logs, run correlation, and safe config audit records |
| `cadence.py` | bounded unique-base/reuse/exact-repeat/send cadence and interval health |
| `output_scheduler.py` | depth-one target-paced publication of guarded final frames with exact-repeat and privacy provenance |
| `remote_protocol.py` | strict bounded remote-renderer protocol-v1 envelope with exact raw-epoch provenance |
| `runtime_performance.py` | bounded path-free output/unique-attainment health, stage summaries, epochs, and advisory mitigations |
| `capture_diagnostics.py` | bounded, pixel-free capture-only cadence measurement and native/runtime evidence comparison |
| `matte_diagnostics.py` | opt-in private bounded matte recorder and offline frozen/model replay |
| `matte_live_diagnostics.py` | on-demand, native-preview-only matte views and frame-paired temporal telemetry |
| `matte_quality.py` / `matte_attribution.py` / `matte_ablation.py` | digest-bound metrics, four-boundary RVM attribution, and bounded same-source screening |
| `matte_visual_qualification.py` | fail-closed MATTE-5.2 taxonomy, route-parity, artifact, and human-review qualification |
| `matte_platform_qualification.py` | fail-closed MATTE-5.3 platform, performance, sink, fallback, and lifecycle evidence join |
| `matte_rollout.py` | fail-closed MATTE-5.4 default disposition plus path-free canary and rollback telemetry |
| `gpu_probe.py` | real CUDA inference/profile capability probe for installer and doctor |
| `vcam.py` | virtual camera output (pyvirtualcam → v4l2loopback / OBS extension) |
| `hub.py` | thread-safe frame exchange between pipeline and API |
| `preview.py` | interactive on-screen output verification and explicit local matte-diagnostic sink |
| `pipeline.py` | main loop; transactional frame-boundary reconfiguration |
| `api/server.py` | authenticated FastAPI control, uploads, MJPEG, and WebSockets |
| `avatar/` | stage-2 avatar service (`custback avatar`): drivers (`drivers.py`, `audio2face.py`), rigs (`rig.py`), composition (`renderer.py`), WS client loop (`service.py`), control API (`api.py`) |

## Tests

The whole pipeline is testable without a camera, virtual camera, or mediapipe:

```bash
pytest
npm test
npm run release:check -- --quick
```

The focused temporal/matte subset, its generated-fixture contract, and the
boundary between fast CI regression coverage and private visual/hardware
qualification are documented in the
[deterministic matte regression gate](docs/matte-deterministic-regression-gate.md).
`tests/test_output_scheduler.py`, `tests/test_runtime_performance.py`,
`tests/test_runtime_performance_status.py`, and
`tests/test_remote_protocol.py`, `tests/test_background_video_lifetime.py`, plus
`tests/test_background_asset_fallback.py` deterministically cover absolute
target pacing and exact repeats, independent output/unique health and
hysteresis, strict bounded status/OpenAPI projection, truthful WebUI rows, and
fail-closed recovery from unavailable persisted image/video assets.
They do not qualify a physical sink, camera, backend, or sustainable host
profile.
The separate owner-only end-to-end workflow and its intentionally pending
checked-in template are documented in the
[matte visual qualification runbook](docs/matte-visual-qualification.md).
The independent capture, fixed-replay, platform-route, and lifecycle evidence
required for MATTE-5.3 is documented in the
[matte platform qualification runbook](docs/matte-platform-qualification.md).
The held default decision, old-config semantics, canary telemetry, complete
release-evidence chain, and non-destructive rollback drill required for
MATTE-5.4 are documented in the
[matte rollout guide](docs/matte-quality-rollout.md).

Geometry and color contracts run in every supported Python version, minimum
and newest dependency profiles, the OpenCV/NumPy compatibility matrix, and
the optional MediaPipe/RVM profiles. Artifact smoke installs the clean wheel,
sdist, and npm tarball; Windows jobs build the exact native camera and frozen
engine and exercise tagged-video normalization. Workflow definitions are
coverage commitments until their exact external runs are attached as
evidence—local green tests are not represented as physical-device approval.

No remediation blocker currently prevents `prepack` or `release:check` from
proceeding to their normal qualification checks. Windows production evidence is
deferred and is not required by the current release manifest.

The full artifact gate creates and installs several isolated Python environments.
Point it at a pre-existing disk-backed directory so those environments do not
consume tmpfs/RAM; each environment is removed as soon as its profile finishes:

```bash
mkdir -p "$HOME/.cache/custback-release"
CUSTBACK_RELEASE_TMPDIR="$HOME/.cache/custback-release" npm run release:check
```

The full gate rejects Linux tmpfs/ramfs storage by default. Override that guard
only when the host has measured headroom with
`CUSTBACK_RELEASE_ALLOW_TMPFS=1`. Native dependency builds default to two
parallel jobs; set `CUSTBACK_RELEASE_BUILD_JOBS` from 1 through 32 only when the
host has measured headroom. Release-gate subprocesses have a 15-minute default
bound; set `CUSTBACK_RELEASE_TIMEOUT_MS` to a positive millisecond value for
slower build hosts. A force-killed gate cannot run its cleanup handler and may
leave a `custback-release-*` directory in the configured base; after confirming
that no gate is running, inspect and remove that exact abandoned directory
before retrying.

### Phase 6 release qualification

The reviewed gate inventory lives in
`scripts/release/required-gates.json`. The production
`.github/workflows/release.yml` workflow builds the wheel, sdist, and npm
tarball once from one clean commit, records their SHA-256 digests, and passes
those files unchanged to every runtime, optional-backend, migration, stress,
clean-tree, and two-host job. Migration and isolated-host jobs emit strict
artifact-bound reports; evidence assembly rejects a missing, duplicated,
wrong-runtime, wrong-host-class, or substituted report.

No `custback` 0.3 artifact is available from the npm or PyPI registries. The
migration gate therefore rebuilds explicitly unpublished reference artifacts
once from reviewed commit `f01baadfa3b1e2a1ef19eceda315eedf06fbe883` and labels
them as source reconstructions. It does not represent them as previously
published bytes.

`REL-01` is resolved: the production workflow and reviewed gate manifest enforce
the exact Phase 6 evidence contract. Windows production evidence remains
deferred and is temporarily outside the blocker registry and required-gate
manifest. The aggregate gate and publish job reverify the exact GitHub run
context, commit, artifact digests, report bindings, candidate attestations, and
the attestation on the evidence document itself; the publish job uploads those
qualified files without rebuilding them.
