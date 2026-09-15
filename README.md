# Custback

Turn your webcam into a local virtual camera with background blur, replacement,
and animated avatars. Select its output in Zoom, Meet, Teams, or another camera
application. Runs on **Windows, Ubuntu / Debian, and macOS**.

- **Backgrounds:** blur, image, looping video, solid color, or a second local camera.
- **Optional vision:** MediaPipe segmentation, RVM alpha matting, and NVIDIA CUDA.
- **Avatars:** expression tracking, idle animation, or Audio2Face lip sync.
- **Live controls:** browser preview and an authenticated HTTP / WebSocket API.

Camera processing stays local by default. Optional models need an initial
network download; a configured remote avatar renderer receives the selected
camera/audio stream.

## Install

**Version 0.4.0.** Install from source below, or check the
[Releases page](https://github.com/Denissvgn/custback/releases) for available
packaged builds. Signed Windows installers are outside the initial release scope.

| Platform | Recommended runtime | Virtual camera |
| --- | --- | --- |
| Windows 11 x64 | Python 3.12 | [OBS Studio](https://obsproject.com/download) |
| Ubuntu / Debian | Python 3.12 | v4l2loopback |
| macOS | Python 3.12 | [OBS Studio](https://obsproject.com/download) |

Core package metadata permits Python 3.10–3.14; initial package smoke coverage
uses 3.12. Optional backends depend on available wheels and hardware. Windows
ARM64, Windows 10, native Windows camera integration, and signed installers are
outside this initial package scope. The npm launcher supports Linux and macOS.

```bash
git clone https://github.com/Denissvgn/custback.git
cd custback
```

**Ubuntu / Debian and macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mediapipe]'
custback avatar --smoke
```

Configure the virtual camera with `./scripts/install_linux.sh` or
`./scripts/install_macos.sh`. On macOS, open OBS and start its virtual camera
once to register it, then stop OBS's output before using Custback.

**Windows 11 x64 — ordinary PowerShell, from the cloned folder**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e '.[windows,mediapipe]'
.\.venv\Scripts\custback.exe avatar --smoke
.\.venv\Scripts\custback.exe --mode blur
```

Install OBS with its virtual camera, and stop OBS's output before starting
Custback. Choose **OBS Virtual Camera** in your meeting app. Use a non-elevated
shell so managed files belong to your user.

For core-only installation, replace the extras with `-e .`; segmentation then
uses a heuristic fallback. The finite `avatar --smoke` command needs no camera.
See the [user guide](docs/user-guide.md) for optional backends and npm setup.

## Use

With the environment activated (on Windows, use the full executable path above):

```bash
custback --mode blur                       # blur your room
custback --image /path/to/background.jpg   # replace the background
custback --video /path/to/loop.mp4         # animated backdrop
custback -c config/default.yaml           # load your configuration
custback --synthetic --no-vcam --mode passthrough  # camera-free browser demo
```

Open **http://127.0.0.1:8710**, retrieve the local management token with
`custback --show-api-token`, and enter it in the browser. Keep the token out of
URLs and shared logs. The UI controls backgrounds, preview, and configured
avatars. Select Custback's virtual camera in your meeting app; press Ctrl+C
to stop the application.

For a local avatar, run `custback --mode remote` and `custback avatar` in
separate terminals. Remote rendering shows a fixed privacy slate when valid
avatar output is unavailable. Follow the [remote deployment guide](docs/remote-deployment.md)
before connecting another machine; remote access requires explicit TLS and
authentication configuration.

## Guides and limitations

| Need | Read |
| --- | --- |
| npm setup, camera troubleshooting, API, avatars, upgrades and removal | [User guide](docs/user-guide.md) |
| Full configuration | [Annotated YAML](config/default.yaml) |
| Rendering geometry and rollback | [Visual configuration](docs/visual-consistency-rollout.md) |
| Matte backends and current defaults | [Matte guide](docs/matte-quality-rollout.md) · [Policy rationale](docs/adr/0004-matte-quality-rollout.md) |
| Changes and upgrade status | [Changelog](CHANGELOG.md) |

Compatibility defaults remain active. Advanced presets, optional GPU paths,
physical camera compatibility, and sustained frame rates require validation
on your host; successful package startup does not establish those results.
Model weights and external camera drivers are downloaded or installed separately.

## Contribute and get help

See [CONTRIBUTING.md](CONTRIBUTING.md), report reproducible bugs through
[Issues](https://github.com/Denissvgn/custback/issues), or contact
[Denissvgn](https://github.com/Denissvgn). Report vulnerabilities through
[GitHub private reporting](https://github.com/Denissvgn/custback/security/advisories/new)
([security policy](SECURITY.md)).

[MIT licensed](LICENSE). Optional dependencies, models, and camera drivers keep
their own licenses.
