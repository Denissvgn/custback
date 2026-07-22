"""FrameHub: thread-safe exchange point between the pipeline and the API.

Three flows meet here:
  1. Pipeline publishes each processed output frame -> MJPEG preview clients.
  2. Pipeline publishes raw camera frames + masks -> "remote" WebSocket clients
     (stage 2: an external avatar service consumes them).
  3. Remote clients push rendered frames back; in mode=remote the pipeline
     uses them as output, falling back to local compositing on timeout.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Stats:
    run_id: str = ""
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
    capture_backend: str = ""
    capture_fourcc: str | None = None
    capture_width: int | None = None
    capture_height: int | None = None
    capture_fps_reported: float | None = None
    capture_target_fps: int = 0
    capture_fps: float = 0.0
    capture_target_met: bool | None = None
    capture_frames_read: int = 0
    capture_dropped_frames: int = 0
    capture_read_failures: int = 0
    capture_restarts: int = 0
    capture_stalled: bool = False
    capture_frame_age_ms: float | None = None
    output_target_fps: int = 0
    fps_attainment_pct: float | None = None
    output_repeated_frames: int = 0
    processing_deadline_misses: int = 0
    capture_read_ms: float | None = None
    segmentation_ms: float | None = None
    background_ms: float | None = None
    composite_ms: float | None = None
    output_send_ms: float | None = None
    frame_processing_ms: float | None = None
    output_fallback_active: bool = False
    output_fallback_reason: str = ""
    segmentation_fallback_active: bool = False
    segmentation_fallback_reason: str = ""
    acceleration_mode: str = ""
    acceleration_requested_provider: str = ""
    acceleration_device_id: int = 0
    acceleration_state: str = ""
    acceleration_active_provider: str = ""
    acceleration_fallback_active: bool = False
    acceleration_fallback_reason: str = ""
    acceleration_fallback_count: int = 0
    acceleration_last_transition_ms: float | None = None
    background_video_source_fps: float | None = None
    background_video_timing_mode: str | None = None
    background_video_frames_displayed: int = 0
    background_video_frames_skipped: int = 0
    background_video_frames_reused: int = 0
    background_video_skip_ratio: float = 0.0
    background_video_seek_count: int = 0
    background_video_decode_failures: int = 0
    started_at: float = field(default_factory=time.time)


class _Slot:
    """Latest-value slot with a change notification."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value: np.ndarray | None = None
        self._seq = 0
        self._ts = 0.0
        self._subscribers: set[_AsyncSlotSubscription] = set()

    def put(self, value: np.ndarray) -> None:
        with self._cond:
            self._value = value
            self._seq += 1
            self._ts = time.monotonic()
            self._cond.notify_all()
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber._notify()

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
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber._notify()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._cond:
            return self._value, self._ts

    def subscribe(self) -> "_AsyncSlotSubscription":
        """Subscribe the current event loop to latest-only frame updates."""

        subscription = _AsyncSlotSubscription(self, asyncio.get_running_loop())
        with self._cond:
            self._subscribers.add(subscription)
        return subscription

    def _unsubscribe(self, subscription: "_AsyncSlotSubscription") -> None:
        with self._cond:
            self._subscribers.discard(subscription)


class _AsyncSlotSubscription:
    """Event-loop-native view of a thread-published latest-value slot."""

    def __init__(self, slot: _Slot, loop: asyncio.AbstractEventLoop) -> None:
        self._slot = slot
        self._loop = loop
        self._event = asyncio.Event()
        self._closed = False

    def _notify(self) -> None:
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self._event.set)
        except RuntimeError:
            # A loop can disappear during process/test teardown. Do not retain
            # a dead subscriber indefinitely in a long-lived frame hub.
            self.close()

    async def get(
        self, last_seq: int = -1, timeout: float | None = None
    ) -> tuple[np.ndarray | None, int]:
        """Return the newest frame after ``last_seq`` without a worker thread."""

        deadline = None if timeout is None else self._loop.time() + timeout
        while not self._closed:
            with self._slot._cond:
                if self._slot._value is not None and self._slot._seq != last_seq:
                    return self._slot._value, self._slot._seq
                # Clear while holding the publisher's lock so an update cannot
                # land in the gap between checking the sequence and waiting.
                self._event.clear()

            remaining = (
                None if deadline is None else max(0.0, deadline - self._loop.time())
            )
            if remaining == 0.0:
                return None, last_seq
            try:
                if remaining is None:
                    await self._event.wait()
                else:
                    await asyncio.wait_for(self._event.wait(), remaining)
            except asyncio.TimeoutError:
                return None, last_seq
        return None, last_seq

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._slot._unsubscribe(self)
        self._event.set()

    async def __aenter__(self) -> "_AsyncSlotSubscription":
        return self

    async def __aexit__(self, *_exc_info) -> None:
        self.close()


class FrameHub:
    def __init__(self, *, run_id: str = "") -> None:
        self.output = _Slot()  # processed frames (what the vcam shows)
        self.raw = _Slot()  # raw camera frames (for remote avatar svc)
        self.remote_in = _Slot()  # frames rendered by the remote avatar svc
        self.stats = Stats(run_id=run_id)
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

    def remote_frame_status(self, max_age_s: float) -> tuple[np.ndarray | None, str]:
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
    def push_remote_frame(
        self, frame: np.ndarray, session_id: int | None = None
    ) -> bool:
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

    def active_remote_session(self) -> int | None:
        """Return the authenticated renderer epoch, if one is currently live."""

        with self._stats_lock:
            return self._remote_session if self._remote_clients > 0 else None

    def remote_session_valid(self, session_id: int) -> bool:
        """Check a renderer lease without granting access to any other route."""

        with self._stats_lock:
            return self._remote_clients > 0 and session_id == self._remote_session

    def _invalidate_remote_session_locked(self, session_id: int | None) -> bool:
        if session_id is not None and session_id != self._remote_session:
            return False
        had_session = self._remote_clients > 0
        if had_session:
            # Advance immediately so an old WebSocket cannot publish in the gap
            # before the next authenticated connection arrives.
            self._remote_session += 1
        self._remote_clients = 0
        self.stats.remote_connected = False
        self.remote_in.clear()
        return had_session

    def invalidate_remote_session(self, session_id: int | None = None) -> bool:
        """Atomically revoke the renderer epoch and discard every queued frame."""

        with self._stats_lock:
            return self._invalidate_remote_session_locked(session_id)

    def reset_remote_session(self, reset: Callable[[], None]) -> bool:
        """Revoke stale output and reset privacy state at the same boundary."""

        with self._stats_lock:
            had_session = self._invalidate_remote_session_locked(None)
            reset()
            return had_session

    def clear_remote_frames(self) -> None:
        """Discard queued output without changing the authenticated epoch."""
        with self._stats_lock:
            self.remote_in.clear()

    def update_stats(self, **kwargs) -> None:
        with self._stats_lock:
            for key, value in kwargs.items():
                setattr(self.stats, key, value)

    def stats_dict(self) -> dict:
        with self._stats_lock:
            return {
                "run_id": self.stats.run_id,
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
                "capture_backend": self.stats.capture_backend,
                "capture_fourcc": self.stats.capture_fourcc,
                "capture_width": self.stats.capture_width,
                "capture_height": self.stats.capture_height,
                "capture_fps_reported": self._rounded_optional(
                    self.stats.capture_fps_reported, 2
                ),
                "capture_target_fps": self.stats.capture_target_fps,
                "capture_fps": round(self.stats.capture_fps, 1),
                "capture_target_met": self.stats.capture_target_met,
                "capture_frames_read": self.stats.capture_frames_read,
                "capture_dropped_frames": self.stats.capture_dropped_frames,
                "capture_read_failures": self.stats.capture_read_failures,
                "capture_restarts": self.stats.capture_restarts,
                "capture_stalled": self.stats.capture_stalled,
                "capture_frame_age_ms": self._rounded_optional(
                    self.stats.capture_frame_age_ms, 1
                ),
                "output_target_fps": self.stats.output_target_fps,
                "fps_attainment_pct": self._rounded_optional(
                    self.stats.fps_attainment_pct, 1
                ),
                "output_repeated_frames": self.stats.output_repeated_frames,
                "processing_deadline_misses": self.stats.processing_deadline_misses,
                "capture_read_ms": self._rounded_optional(
                    self.stats.capture_read_ms, 1
                ),
                "segmentation_ms": self._rounded_optional(
                    self.stats.segmentation_ms, 1
                ),
                "background_ms": self._rounded_optional(self.stats.background_ms, 1),
                "composite_ms": self._rounded_optional(self.stats.composite_ms, 1),
                "output_send_ms": self._rounded_optional(self.stats.output_send_ms, 1),
                "frame_processing_ms": self._rounded_optional(
                    self.stats.frame_processing_ms, 1
                ),
                "output_fallback_active": self.stats.output_fallback_active,
                "output_fallback_reason": self.stats.output_fallback_reason,
                "segmentation_fallback_active": self.stats.segmentation_fallback_active,
                "segmentation_fallback_reason": self.stats.segmentation_fallback_reason,
                "acceleration_mode": self.stats.acceleration_mode,
                "acceleration_requested_provider": (
                    self.stats.acceleration_requested_provider
                ),
                "acceleration_device_id": self.stats.acceleration_device_id,
                "acceleration_state": self.stats.acceleration_state,
                "acceleration_active_provider": self.stats.acceleration_active_provider,
                "acceleration_fallback_active": self.stats.acceleration_fallback_active,
                "acceleration_fallback_reason": self.stats.acceleration_fallback_reason,
                "acceleration_fallback_count": self.stats.acceleration_fallback_count,
                "acceleration_last_transition_ms": self._rounded_optional(
                    self.stats.acceleration_last_transition_ms, 1
                ),
                "background_video_source_fps": self._rounded_optional(
                    self.stats.background_video_source_fps, 2
                ),
                "background_video_timing_mode": self.stats.background_video_timing_mode,
                "background_video_frames_displayed": (
                    self.stats.background_video_frames_displayed
                ),
                "background_video_frames_skipped": (
                    self.stats.background_video_frames_skipped
                ),
                "background_video_frames_reused": (
                    self.stats.background_video_frames_reused
                ),
                "background_video_skip_ratio": round(
                    self.stats.background_video_skip_ratio, 4
                ),
                "background_video_seek_count": self.stats.background_video_seek_count,
                "background_video_decode_failures": (
                    self.stats.background_video_decode_failures
                ),
                "uptime_s": round(time.time() - self.stats.started_at, 1),
            }

    @staticmethod
    def _rounded_optional(value: float | None, digits: int) -> float | None:
        return None if value is None else round(value, digits)
