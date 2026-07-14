"""FrameHub: thread-safe exchange point between the pipeline and the API.

Three flows meet here:
  1. Pipeline publishes each processed output frame -> MJPEG preview clients.
  2. Pipeline publishes raw camera frames + masks -> "remote" WebSocket clients
     (stage 2: an external avatar service consumes them).
  3. Remote clients push rendered frames back; in mode=remote the pipeline
     uses them as output, falling back to local compositing on timeout.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stats:
    frames_in: int = 0
    frames_out: int = 0
    fps: float = 0.0
    mode: str = ""
    segmentation_backend: str = ""
    segmentation_device: str = ""
    output_backend: str = ""
    remote_connected: bool = False
    remote_frames_used: int = 0
    remote_fallback_active: bool = False
    remote_fallback_mode: str = ""
    remote_fallback_count: int = 0
    remote_fallback_reason: str = ""
    config_version: int = 0
    started_at: float = field(default_factory=time.time)


class _Slot:
    """Latest-value slot with a change notification."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value: np.ndarray | None = None
        self._seq = 0
        self._ts = 0.0

    def put(self, value: np.ndarray) -> None:
        with self._cond:
            self._value = value
            self._seq += 1
            self._ts = time.monotonic()
            self._cond.notify_all()

    def get(self, last_seq: int = -1, timeout: float | None = None):
        """Return (frame, seq) newer than last_seq, or (None, last_seq)."""
        with self._cond:
            ready = self._cond.wait_for(
                lambda: self._value is not None and self._seq != last_seq,
                timeout=timeout,
            )
            if not ready:
                return None, last_seq
            return self._value, self._seq

    def clear(self) -> None:
        """Discard the value and wake waiters, which continue waiting for data."""
        with self._cond:
            self._value = None
            self._ts = 0.0
            self._seq += 1
            self._cond.notify_all()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._cond:
            return self._value, self._ts


class FrameHub:
    def __init__(self) -> None:
        self.output = _Slot()      # processed frames (what the vcam shows)
        self.raw = _Slot()         # raw camera frames (for remote avatar svc)
        self.remote_in = _Slot()   # frames rendered by the remote avatar svc
        self.stats = Stats()
        self._stats_lock = threading.Lock()
        self._remote_clients = 0
        self._remote_session = 0

    # -- pipeline side -------------------------------------------------
    def publish_output(self, frame: np.ndarray) -> None:
        self.output.put(frame)

    def publish_raw(self, frame: np.ndarray) -> None:
        self.raw.put(frame)

    def get_remote_frame(self, max_age_s: float) -> np.ndarray | None:
        """Latest remote-rendered frame if it is fresh enough, else None."""
        return self.remote_frame_status(max_age_s)[0]

    def remote_frame_status(
        self, max_age_s: float
    ) -> tuple[np.ndarray | None, str]:
        """Return a fresh frame or a deterministic local-fallback reason."""
        with self._stats_lock:
            if self._remote_clients == 0:
                return None, "no-client"
        frame, ts = self.remote_in.latest()
        if frame is None or (time.monotonic() - ts) > max_age_s:
            return None, "stale"
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            return None, "invalid"
        return frame, ""

    # -- API side ------------------------------------------------------
    def push_remote_frame(self, frame: np.ndarray, session_id: int | None = None) -> bool:
        """Publish a frame only for the currently connected remote session."""
        with self._stats_lock:
            if self._remote_clients == 0:
                return False
            if session_id is not None and session_id != self._remote_session:
                return False
            # Keep the stats lock through put so a final disconnect always
            # clears any frame published by that session.
            self.remote_in.put(frame)
            return True

    def remote_client_connected(self) -> int:
        with self._stats_lock:
            if self._remote_clients == 0:
                self._remote_session += 1
                self.remote_in.clear()
            self._remote_clients += 1
            self.stats.remote_connected = True
            return self._remote_session

    def remote_client_disconnected(self, session_id: int | None = None) -> None:
        with self._stats_lock:
            if session_id is not None and session_id != self._remote_session:
                return
            self._remote_clients = max(0, self._remote_clients - 1)
            self.stats.remote_connected = self._remote_clients > 0
            if self._remote_clients == 0:
                self.remote_in.clear()

    def clear_remote_frames(self) -> None:
        """Invalidate output from the current remote session on mode changes."""
        self.remote_in.clear()

    def update_stats(self, **kwargs) -> None:
        with self._stats_lock:
            for key, value in kwargs.items():
                setattr(self.stats, key, value)

    def stats_dict(self) -> dict:
        with self._stats_lock:
            return {
                "frames_in": self.stats.frames_in,
                "frames_out": self.stats.frames_out,
                "fps": round(self.stats.fps, 1),
                "mode": self.stats.mode,
                "segmentation_backend": self.stats.segmentation_backend,
                "segmentation_device": self.stats.segmentation_device,
                "output_backend": self.stats.output_backend,
                "remote_connected": self.stats.remote_connected,
                "remote_frames_used": self.stats.remote_frames_used,
                "remote_fallback_active": self.stats.remote_fallback_active,
                "remote_fallback_mode": self.stats.remote_fallback_mode,
                "remote_fallback_count": self.stats.remote_fallback_count,
                "remote_fallback_reason": self.stats.remote_fallback_reason,
                "config_version": self.stats.config_version,
                "uptime_s": round(time.time() - self.stats.started_at, 1),
            }
