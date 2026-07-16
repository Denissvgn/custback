#!/usr/bin/env bash
# custback automatic setup for Ubuntu and Debian.
# Installs the v4l2loopback kernel module and creates a persistent
# "custback Camera" virtual video device that meeting apps can select.
set -euo pipefail

OS_RELEASE_FILE="${CUSTBACK_OS_RELEASE_FILE:-/etc/os-release}"

os_release_value() {
  local wanted="$1"
  local key value
  while IFS='=' read -r key value; do
    if [[ "$key" == "$wanted" ]]; then
      value="${value#\"}"
      value="${value%\"}"
      value="${value#\'}"
      value="${value%\'}"
      printf '%s' "$value"
      return 0
    fi
  done < "$OS_RELEASE_FILE"
  return 1
}

if [[ ! -r "$OS_RELEASE_FILE" ]]; then
  echo "custback setup: unsupported Linux distribution (cannot read $OS_RELEASE_FILE)" >&2
  exit 1
fi

DISTRO_ID="$(os_release_value ID || true)"
DISTRO_ID_LIKE="$(os_release_value ID_LIKE || true)"
DISTRO_ID="${DISTRO_ID,,}"
DISTRO_ID_LIKE="${DISTRO_ID_LIKE,,}"

# The commands below are reviewed and tested only for Ubuntu and Debian. Do
# not optimistically run apt on derivatives: add and test a distro-specific
# installer before expanding this allowlist.
case "$DISTRO_ID" in
  ubuntu|debian) ;;
  *)
    echo "custback setup: unsupported Linux distribution "\
         "(ID=${DISTRO_ID:-unknown}, ID_LIKE=${DISTRO_ID_LIKE:-unknown}); "\
         "automatic setup supports Ubuntu and Debian only" >&2
    exit 1
    ;;
esac

echo "==> Installing v4l2loopback and Python build deps"
sudo apt-get update
sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils python3-venv python3-dev

echo "==> Configuring v4l2loopback (persistent across reboots)"
# exclusive_caps=1 is required for Chrome/Chromium and most Electron apps
# (Zoom, Teams, Slack) to list the device as a camera.
sudo tee /etc/modprobe.d/custback.conf > /dev/null <<'EOF'
options v4l2loopback devices=1 video_nr=10 card_label="custback Camera" exclusive_caps=1
EOF
echo "v4l2loopback" | sudo tee /etc/modules-load.d/custback.conf > /dev/null

echo "==> Loading module now"
sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="custback Camera" exclusive_caps=1

echo "==> Virtual camera ready at /dev/video10"
v4l2-ctl --list-devices 2>/dev/null | grep -A1 custback || true

cat <<'EOF'

Next steps:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -e '.[mediapipe,dev]'
  custback --mode blur            # or --image path.jpg / --video path.mp4

Then pick "custback Camera" in Zoom/Meet/Teams/your browser.
EOF
