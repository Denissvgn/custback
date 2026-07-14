"""Virtual camera output.

Linux (Ubuntu): pyvirtualcam -> v4l2loopback device (see scripts/install_linux.sh).
macOS:          pyvirtualcam -> OBS Virtual Camera extension (see scripts/install_macos.sh).

A NullOutput is provided for tests / API-only operation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from .config import OutputConfig

log = logging.getLogger(__name__)


class VideoOutput(ABC):
    #: True when send() itself blocks until the next frame slot (the pipeline
    #: must not add its own sleep on top).
    paces = False

    @abstractmethod
    def send(self, frame_bgr: np.ndarray) -> None: ...

    def close(self) -> None:
        pass


class NullOutput(VideoOutput):
    """Discards frames; useful when only the HTTP/WebSocket API is consumed."""

    def __init__(self) -> None:
        self.frames_sent = 0

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
        log.info("virtual camera started: %s (%dx%d @ %d fps)",
                 self.cam.device, width, height, cfg.fps)

    def send(self, frame_bgr: np.ndarray) -> None:
        self.cam.send(frame_bgr)
        self.cam.sleep_until_next_frame()

    def close(self) -> None:
        self.cam.close()


def open_output(cfg: OutputConfig, width: int, height: int) -> VideoOutput:
    if cfg.backend == "null":
        return NullOutput()
    try:
        return PyVirtualCamOutput(cfg, width, height)
    except Exception as exc:
        if cfg.backend == "pyvirtualcam":
            raise
        log.warning(
            "virtual camera unavailable (%s); frames will only be served via "
            "the API. On Linux run scripts/install_linux.sh, on macOS run "
            "scripts/install_macos.sh.",
            exc,
        )
        return NullOutput()
