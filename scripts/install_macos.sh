#!/usr/bin/env bash
# custback setup for macOS.
# pyvirtualcam on macOS outputs through the OBS Virtual Camera system
# extension, so OBS must be installed once (the app itself does not need
# to be running after the extension is activated).
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required: https://brew.sh" >&2
  exit 1
fi

echo "==> Installing OBS (provides the macOS virtual camera extension)"
brew list --cask obs >/dev/null 2>&1 || brew install --cask obs

cat <<'EOF'

One-time activation:
  1. Open OBS once and click "Start Virtual Camera"
     (this registers the system extension; approve it in
      System Settings > Privacy & Security if prompted).
  2. You can then quit OBS.

Install custback:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -e '.[mediapipe,dev]'
  custback --mode blur            # or --image path.jpg / --video path.mp4

Then pick "OBS Virtual Camera" in Zoom/Meet/Teams/your browser.
Note: macOS will ask for camera permission on first run.
EOF
