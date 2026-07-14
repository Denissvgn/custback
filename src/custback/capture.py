"""Camera capture sources.

All sources yield BGR uint8 frames of the configured size. A synthetic source
is included so the whole pipeline can run and be tested without hardware.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from .config import CameraConfig

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class CaptureSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if unavailable."""

    def close(self) -> None:
        pass


class OpenCVCapture(CaptureSource):
    """Real camera via OpenCV (V4L2 on Linux, AVFoundation on macOS)."""

    def __init__(self, cfg: CameraConfig):
        if cv2 is None:
            raise RuntimeError("opencv-python is required for camera capture")
        self.cfg = cfg
        device = cfg.device
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {cfg.device!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[1] != self.cfg.width or frame.shape[0] != self.cfg.height:
            frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))
        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        self.cap.release()


class SyntheticCapture(CaptureSource):
    """Test pattern: moving bright ellipse ("person") over a dark gradient.

    The ellipse is deliberately much brighter than the backdrop so the
    heuristic segmenter can find it — useful for end-to-end tests.
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.t0 = time.monotonic()
        h, w = cfg.height, cfg.width
        gradient = np.linspace(20, 70, w, dtype=np.uint8)
        self._bg = np.stack([np.tile(gradient, (h, 1))] * 3, axis=-1)

    def read(self) -> np.ndarray:
        h, w = self.cfg.height, self.cfg.width
        frame = self._bg.copy()
        t = time.monotonic() - self.t0
        cx = int(w / 2 + (w / 6) * np.sin(t))
        cy = int(h / 2)
        yy, xx = np.ogrid[:h, :w]
        ellipse = ((xx - cx) / (w * 0.14)) ** 2 + ((yy - cy) / (h * 0.3)) ** 2 <= 1.0
        frame[ellipse] = (200, 190, 210)
        return frame


def open_capture(cfg: CameraConfig) -> CaptureSource:
    if cfg.synthetic:
        return SyntheticCapture(cfg)
    return OpenCVCapture(cfg)
