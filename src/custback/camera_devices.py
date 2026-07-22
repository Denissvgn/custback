"""Camera device enumeration, backend selection, and output-loop prevention.

This is the one place that holds OS-specific *camera* behavior, kept out of
:mod:`custback.capture` so the reader/recovery machinery contains no camera
``sys.platform`` branch (the CC-4 seam discipline, applied to capture instead of
filesystem security).  It answers three Windows Phase-3 questions:

* **WIN-3.1** — friendly device names and *stable identifiers* rather than a
  bare index ``0`` that silently changes meaning when devices reorder.
* **WIN-3.2** — the ordered list of capture backends to open explicitly
  (``MSMF`` then ``DSHOW`` on Windows), analogous to the V4L2 preference the
  reader already applies on Linux.
* **WIN-3.4** — recognizing the project's own / OBS *output* camera so it is
  never offered back as an input (a capture→output→capture loop).

It also carries the **WIN-3.3** privacy-denial guidance so an "cannot open
camera" failure on Windows names the OS setting the user must change.

Enumeration is best-effort and never raises: a caller listing devices for a UI
must degrade to "no devices found" rather than crash.  The Linux path reads
sysfs/``/dev`` and is fully deterministic; the Windows/macOS paths probe indices
through OpenCV and attach friendly names when a name provider is available.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

try:  # OpenCV is optional at import time (see capture.py).
    import cv2
except ImportError:  # pragma: no cover - exercised only without OpenCV
    cv2 = None


#: How many bare indices to probe when the platform exposes no device list.
_DEFAULT_MAX_PROBE = 10

#: Substrings (case-insensitive) that identify a *virtual output* camera — the
#: sink this project (or OBS) publishes to.  Offering one as an input would
#: create a feedback loop, so :func:`filter_input_devices` drops them.  Kept as
#: plain substrings because vendors vary the surrounding text ("OBS Virtual
#: Camera", "OBS-Camera", "OBS Virtual Camera Device").
_VIRTUAL_OUTPUT_MARKERS = (
    "obs virtual camera",
    "obs-camera",
    "obs camera",
    "custback",
    "v4l2loopback",
    "dummy video device",  # v4l2loopback's default card label
    "unity video capture",
)


@dataclass(frozen=True)
class CameraDevice:
    """A discovered capture device.

    ``index`` is the OpenCV open ordinal.  ``stable_id`` is an identifier that
    survives a device reorder where the index does not (a ``/dev/v4l/by-id``
    path on Linux, a Media Foundation symbolic link on Windows when available);
    it falls back to a synthesized ``"<backend>:<index>"`` when the platform
    exposes nothing more durable.  ``is_virtual_output`` marks the project's own
    or OBS's output sink so callers can refuse it as an input.
    """

    index: int
    name: str
    stable_id: str
    backend: str = "unknown"
    is_virtual_output: bool = False


def is_virtual_output_name(name: str | None) -> bool:
    """Return ``True`` when ``name`` looks like a virtual *output* camera."""

    if not name:
        return False
    lowered = name.casefold()
    return any(marker in lowered for marker in _VIRTUAL_OUTPUT_MARKERS)


def filter_input_devices(devices: list[CameraDevice]) -> list[CameraDevice]:
    """Drop virtual-output sinks so no capture→output→capture loop is offered.

    WIN-3.4: the Custback/OBS output camera must never appear as a selectable
    *input*.  Detection is by ``is_virtual_output`` (set at enumeration time
    from the device name), so a renamed clone is still caught by name matching.
    """

    return [device for device in devices if not device.is_virtual_output]


# -- backend selection (WIN-3.2) --------------------------------------------
def preferred_capture_backends(
    cv2_module: Any | None = None, platform: str | None = None
) -> list[tuple[str, int]]:
    """Ordered ``(name, cv2 apiPreference)`` backends to try opening.

    Windows returns Media Foundation first, then DirectShow — the same
    "prefer the modern backend, fall back to the legacy one" shape the reader
    already relies on for V4L2.  Every other platform returns an empty list,
    which the reader treats as "open with OpenCV's default ``CAP_ANY``"; this
    keeps POSIX capture byte-for-byte identical to today (no explicit
    ``apiPreference`` is passed where none was before).

    Unknown backend constants (older OpenCV builds) are skipped rather than
    passed as ``0``/``CAP_ANY``, so a missing ``CAP_MSMF`` cannot silently
    degrade selection.
    """

    module = cv2_module if cv2_module is not None else cv2
    plat = platform if platform is not None else sys.platform
    if module is None or not plat.startswith("win"):
        return []
    ordered: list[tuple[str, int]] = []
    for name in ("CAP_MSMF", "CAP_DSHOW"):
        value = getattr(module, name, None)
        if isinstance(value, int) and value >= 0:
            ordered.append((name.removeprefix("CAP_"), value))
    return ordered


# -- privacy-denial guidance (WIN-3.3) --------------------------------------
def camera_open_hint(platform: str | None = None) -> str:
    """Actionable, credential-free guidance for a failed camera open.

    OpenCV surfaces a Windows *privacy* denial the same way it surfaces a
    genuinely absent device — ``isOpened()`` is ``False`` with no distinct code.
    Rather than mislabel it, the reader appends this hint to the outage message
    so the most common Windows cause (the per-app camera toggle) is named.
    """

    plat = platform if platform is not None else sys.platform
    if plat.startswith("win"):
        return (
            "if the device exists, check Windows Settings > Privacy & security > "
            "Camera and allow desktop apps to access the camera, then confirm no "
            "other application is using it"
        )
    if plat == "darwin":
        return (
            "check System Settings > Privacy & Security > Camera and confirm no "
            "other application is using the device"
        )
    return "confirm the device exists and is not in use by another application"


# -- Linux enumeration (deterministic, sysfs-backed) ------------------------
def _linux_stable_id(index: int, by_id_root: Path) -> str:
    """Resolve a ``/dev/v4l/by-id`` alias for ``/dev/video<index>`` if present."""

    try:
        entries = sorted(by_id_root.iterdir())
    except OSError:
        entries = []
    target = f"video{index}"
    for entry in entries:
        try:
            resolved = entry.resolve()
        except OSError:
            continue
        if resolved.name == target:
            return f"by-id/{entry.name}"
    return f"v4l2:{index}"


def linux_video_devices(
    sysfs_root: Path | str = "/sys/class/video4linux",
    by_id_root: Path | str = "/dev/v4l/by-id",
) -> list[CameraDevice]:
    """Enumerate V4L2 capture nodes with friendly names from sysfs.

    Pure filesystem reads — no OpenCV, no device opening — so it is safe to call
    from a UI/list path and is fully deterministic under test.  Non-``videoN``
    entries and unreadable nodes are skipped; ordering is by numeric index.
    """

    root = Path(sysfs_root)
    by_id = Path(by_id_root)
    devices: list[CameraDevice] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return devices
    indexed: list[tuple[int, Path]] = []
    for entry in entries:
        match = re.fullmatch(r"video(\d+)", entry.name)
        if match:
            indexed.append((int(match.group(1)), entry))
    for index, entry in sorted(indexed):
        try:
            name = (
                (entry / "name").read_text(encoding="utf-8", errors="replace").strip()
            )
        except OSError:
            name = ""
        friendly = name or f"/dev/video{index}"
        devices.append(
            CameraDevice(
                index=index,
                name=friendly,
                stable_id=_linux_stable_id(index, by_id),
                backend="V4L2",
                is_virtual_output=is_virtual_output_name(name),
            )
        )
    return devices


# -- OpenCV index probing (Windows / macOS / fallback) ----------------------
def probe_camera_indices(
    cv2_module: Any,
    *,
    max_probe: int = _DEFAULT_MAX_PROBE,
    backends: list[tuple[str, int]] | None = None,
    name_provider: Callable[[int], str | None] | None = None,
) -> list[CameraDevice]:
    """Open indices ``0..max_probe-1`` and keep the ones that open.

    Used where the OS exposes no cheap device list to OpenCV.  Each index is
    opened with the first working preferred backend (so the reported backend
    matches what capture will actually use) and released immediately.  Friendly
    names come from ``name_provider`` when supplied (e.g. a Media Foundation /
    DirectShow name source); otherwise a generic ``"Camera <index>"`` is used.
    """

    if cv2_module is None:
        return []
    order = backends if backends is not None else preferred_capture_backends(cv2_module)
    devices: list[CameraDevice] = []
    for index in range(max(0, max_probe)):
        opened_backend = _probe_single_index(cv2_module, index, order)
        if opened_backend is None:
            continue
        raw_name = None
        if name_provider is not None:
            try:
                raw_name = name_provider(index)
            except Exception:  # a name source must never break enumeration
                log.debug(
                    "camera name provider failed for index %d", index, exc_info=True
                )
        friendly = raw_name or f"Camera {index}"
        devices.append(
            CameraDevice(
                index=index,
                name=friendly,
                stable_id=f"{opened_backend.lower()}:{index}",
                backend=opened_backend,
                is_virtual_output=is_virtual_output_name(raw_name),
            )
        )
    return devices


def _probe_single_index(
    cv2_module: Any, index: int, backends: list[tuple[str, int]]
) -> str | None:
    """Return the backend name that opened ``index``, or ``None``."""

    candidates: list[tuple[str, int | None]]
    candidates = list(backends) if backends else [("default", None)]
    for name, api in candidates:
        cap = None
        try:
            cap = (
                cv2_module.VideoCapture(index)
                if api is None
                else cv2_module.VideoCapture(index, api)
            )
            if cap.isOpened():
                return name
        except Exception:
            log.debug("camera probe failed for index %d backend %s", index, name)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return None


# -- top-level enumeration ---------------------------------------------------
def enumerate_cameras(
    *,
    platform: str | None = None,
    cv2_module: Any | None = None,
    max_probe: int = _DEFAULT_MAX_PROBE,
    include_virtual_output: bool = False,
    name_provider: Callable[[int], str | None] | None = None,
) -> list[CameraDevice]:
    """Best-effort list of usable capture devices for the current platform.

    Linux reads sysfs; every other platform probes OpenCV indices.  Virtual
    *output* cameras are excluded by default (WIN-3.4); pass
    ``include_virtual_output=True`` only for diagnostics that want the full set.
    Never raises — any error degrades to an empty list.
    """

    plat = platform if platform is not None else sys.platform
    module = cv2_module if cv2_module is not None else cv2
    try:
        if plat.startswith("linux"):
            devices = linux_video_devices()
        else:
            devices = probe_camera_indices(
                module, max_probe=max_probe, name_provider=name_provider
            )
    except Exception:  # pragma: no cover - defensive; sub-calls already guard
        log.debug("camera enumeration failed on %s", plat, exc_info=True)
        return []
    if include_virtual_output:
        return devices
    return filter_input_devices(devices)
