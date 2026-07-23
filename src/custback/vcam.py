"""Virtual camera output.

Linux (Ubuntu): pyvirtualcam -> v4l2loopback device (see scripts/install_linux.sh).
macOS:          pyvirtualcam -> OBS Virtual Camera extension (see scripts/install_macos.sh).
Windows:        pyvirtualcam -> OBS Virtual Camera (install OBS Studio and start
                the virtual camera once; see WIN-3.7 in WINDOWS_IMPLEMENTATION_PLAN.md).

A NullOutput is provided for tests / API-only operation.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod

import numpy as np

from .config import OutputConfig

log = logging.getLogger(__name__)


def virtual_camera_setup_hint(platform: str | None = None) -> str:
    """Platform-specific, credential-free guidance for enabling a virtual camera.

    WIN-3.7: Windows and macOS both drive pyvirtualcam through OBS Virtual
    Camera, which is a *prerequisite the user installs* — this project must not
    silently redistribute OBS components (CC-5), so the message points at the
    install/enable step instead.  Linux keeps the existing v4l2loopback path.
    """

    plat = platform if platform is not None else sys.platform
    if plat.startswith("win"):
        return (
            "install OBS Studio and click 'Start Virtual Camera' once (or install "
            "the OBS Virtual Camera prerequisite); if it is already running, "
            "another application may be holding its single output slot"
        )
    if plat == "darwin":
        return (
            "install OBS Studio and enable its Virtual Camera extension "
            "(see scripts/install_macos.sh)"
        )
    return (
        "load the v4l2loopback kernel module and create an output device "
        "(see scripts/install_linux.sh)"
    )


class VideoOutput(ABC):
    #: True when send() itself blocks until the next frame slot (the pipeline
    #: must not add its own sleep on top).
    paces = False
    fallback_active = False
    fallback_reason = ""

    @abstractmethod
    def send(self, frame_bgr: np.ndarray) -> None: ...

    def close(self) -> None:
        pass


class NullOutput(VideoOutput):
    """Discards frames; useful when only the HTTP/WebSocket API is consumed."""

    def __init__(
        self, *, fallback_active: bool = False, fallback_reason: str = ""
    ) -> None:
        self.frames_sent = 0
        self.fallback_active = fallback_active
        self.fallback_reason = fallback_reason

    def send(self, frame_bgr: np.ndarray) -> None:
        self.frames_sent += 1


class PyVirtualCamOutput(VideoOutput):
    paces = True

    def __init__(self, cfg: OutputConfig, width: int, height: int):
        import pyvirtualcam

        kwargs: dict = {}
        if cfg.device:
            kwargs["device"] = cfg.device
        self.cam = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=cfg.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            **kwargs,
        )
        log.info(
            "virtual camera started: %s (%dx%d @ %d fps)",
            self.cam.device,
            width,
            height,
            cfg.fps,
        )

    def send(self, frame_bgr: np.ndarray) -> None:
        self.cam.send(frame_bgr)
        self.cam.sleep_until_next_frame()

    def close(self) -> None:
        self.cam.close()


def _classify_output_failure(exc: BaseException) -> str:
    """Map a pyvirtualcam open failure to a bounded, credential-free reason.

    OBS Virtual Camera exposes a *single* output slot; a second producer fails
    with a "in use"/"busy" style message.  Distinguishing that from "not
    installed" lets the status surface tell the user to close the other app
    rather than to (re)install OBS (WIN-3.7).
    """

    text = str(exc).casefold()
    if any(marker in text for marker in ("in use", "busy", "already", "in-use")):
        return "virtual-camera-in-use"
    return "virtual-camera-unavailable"


def open_output(cfg: OutputConfig, width: int, height: int) -> VideoOutput:
    if cfg.backend == "null":
        return NullOutput()
    if cfg.backend == "native":
        # WIN-6.1: the Windows 11 Media Foundation virtual camera.  Explicit
        # opt-in only — `auto` keeps the OBS/pyvirtualcam path until the native
        # camera passes its own clean-machine gate (feature-matrix guardrail).
        # An explicit backend fails loudly rather than silently degrading.
        from . import vcam_native

        return vcam_native.NativeVirtualCameraOutput(width, height)
    try:
        return PyVirtualCamOutput(cfg, width, height)
    except Exception as exc:
        hint = virtual_camera_setup_hint()
        if cfg.backend == "pyvirtualcam":
            raise RuntimeError(f"virtual camera unavailable: {exc} ({hint})") from exc
        reason = _classify_output_failure(exc)
        log.info(
            "virtual camera unavailable (%s); frames will only be served via the "
            "API. To enable the virtual camera: %s",
            reason,
            hint,
        )
        return NullOutput(
            fallback_active=True,
            fallback_reason=reason,
        )
