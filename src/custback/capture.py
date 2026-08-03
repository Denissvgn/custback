"""Camera capture sources and bounded capture-worker health telemetry.

All sources yield immutable envelopes around canonical-canvas, C-contiguous
BGR uint8 frames. Device negotiation remains tied to the configured
acquisition request. Real OpenCV capture runs in one dedicated reader because
``VideoCapture.read`` may block; the pipeline only ever consumes the newest
completed normalized frame.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .camera_devices import camera_open_hint, preferred_capture_backends
from .config import CameraConfig
from .geometry import (
    FrameValidationError,
    Size,
    TransformPlan,
    apply_transform,
    plan_transform,
    transform_frame,
    validate_bgr_frame,
)

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

# OpenCV is a compiled optional boundary. Keep its runtime ``None`` fallback
# while treating the dynamically exposed API as opaque to static analysis.
cv2: Any = _cv2


log = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """A camera can no longer provide usable frames."""


class CaptureModeError(CaptureError):
    """The camera could not activate the requested capture mode."""


class CaptureWorkerError(CaptureError):
    """The bounded camera reader could not be stopped safely."""


class _CaptureNegotiationRetry(CaptureError):
    """An automatic V4L2 MJPG request should receive one clean reopen."""


@dataclass(frozen=True)
class CapturedFrame:
    """One canonical pixel array and its atomic capture identity.

    ``captured_at_ns`` uses this process's monotonic clock at successful source
    read completion. The frozen envelope prevents consumers from accidentally
    re-associating metadata with different pixels; consumers must continue to
    treat the owned pixel array itself as read-only.
    """

    pixels: np.ndarray
    sequence: int
    captured_at_ns: int
    generation: int
    geometry_generation: int
    content_rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class CameraControlObservation:
    """One side-effect-free OpenCV camera-property observation.

    OpenCV documents zero as the unsupported-property sentinel, but zero is
    also a valid value for several camera controls.  The report therefore
    distinguishes a non-zero value reported by the backend from an ambiguous
    zero and never promotes either observation into permission to write.
    """

    name: str
    status: str
    value: float | None

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status, "value": self.value}


@dataclass(frozen=True)
class CameraControlReport:
    """Immutable, path-free camera-control capability observation."""

    policy: str = "preserve"
    backend_family: str = "other"
    qualification: str = "unqualified"
    writes_performed: bool = False
    generation: int = 0
    properties: tuple[CameraControlObservation, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "backend_family": self.backend_family,
            "qualification": self.qualification,
            "writes_performed": self.writes_performed,
            "generation": self.generation,
            "properties": {
                observation.name: observation.as_dict()
                for observation in self.properties
            },
        }


@dataclass(frozen=True)
class CaptureHealth:
    """Constant-space snapshot consumed by pipeline/status diagnostics."""

    # ``generation`` and ``geometry_generation`` describe the last frame
    # returned by ``read()``, not a newer frame waiting in the reader slot.
    # This lets downstream temporal state reset at the exact frame boundary.
    # ``sequence`` and ``captured_monotonic_ns`` mirror the matching envelope
    # for diagnostics. Temporal consumers use ``CapturedFrame`` directly so a
    # separate health sample can never race the pixel array.
    sequence: int = 0
    captured_monotonic_ns: int | None = None
    generation: int = 0
    geometry_generation: int = 0
    content_rect: tuple[int, int, int, int] | None = None
    backend: str = "unknown"
    fourcc: str | None = None
    width: int | None = None
    height: int | None = None
    delivered_width: int | None = None
    delivered_height: int | None = None
    oriented_width: int | None = None
    oriented_height: int | None = None
    normalized_width: int | None = None
    normalized_height: int | None = None
    geometry_transitions: int = 0
    fps_reported: float | None = None
    capture_fps: float = 0.0
    target_met: bool | None = None
    frames_read: int = 0
    dropped_frames: int = 0
    read_failures: int = 0
    restarts: int = 0
    stalled: bool = False
    frame_age_ms: float | None = None
    read_ms: float | None = None
    fatal_error: str = ""
    worker_alive: bool = False
    camera_controls: CameraControlReport = field(default_factory=CameraControlReport)


class CaptureSource(ABC):
    @abstractmethod
    def read(self) -> CapturedFrame | None:
        """Return the newest unread captured frame, or ``None`` if unavailable."""

    def health_snapshot(self) -> CaptureHealth:
        """Return capture health without blocking or mutating the source."""
        return CaptureHealth()

    def close(self) -> None:
        pass


def _device_value(device: int | str) -> int | str:
    if isinstance(device, str) and device.isdigit():
        return int(device)
    return device


def _safe_get(cap: Any, prop: int | None) -> float | None:
    if prop is None:
        return None
    try:
        value = float(cap.get(prop))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _fourcc_value(code: str) -> int:
    writer_fourcc = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(writer_fourcc):
        return int(cast(Any, writer_fourcc)(*code))
    return sum(ord(char) << (8 * index) for index, char in enumerate(code))


def _fourcc_name(value: float | None) -> str | None:
    if value is None or value <= 0:
        return None
    try:
        encoded = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    chars = "".join(chr((encoded >> (8 * index)) & 0xFF) for index in range(4))
    name = "".join(char for char in chars if 0x20 <= ord(char) < 0x7F).strip()
    return name.upper() or None


def _backend_name(cap: Any) -> str:
    try:
        name = cap.getBackendName()
    except Exception:
        name = ""
    if isinstance(name, str) and name.strip():
        return name.strip()

    backend_id = _safe_get(cap, getattr(cv2, "CAP_PROP_BACKEND", None))
    if backend_id is not None:
        registry = getattr(cv2, "videoio_registry", None)
        get_name = getattr(registry, "getBackendName", None)
        if callable(get_name):
            try:
                name = get_name(int(backend_id))
            except Exception:
                name = ""
            if isinstance(name, str) and name.strip():
                return name.strip()
        if int(backend_id) == int(getattr(cv2, "CAP_V4L2", -1)):
            return "V4L2"
        return str(int(backend_id))
    return "unknown"


def _is_v4l2(backend: str) -> bool:
    normalized = backend.upper().replace("_", "")
    return "V4L" in normalized


def _camera_backend_family(backend: str) -> str:
    normalized = "".join(
        character for character in backend.upper() if character.isalnum()
    )
    if "V4L" in normalized:
        return "v4l2"
    if "MSMF" in normalized or "MEDIAFOUNDATION" in normalized:
        return "msmf"
    if "DSHOW" in normalized or "DIRECTSHOW" in normalized:
        return "dshow"
    return "other"


def _observe_camera_controls(
    cap: Any,
    backend: str,
    *,
    generation: int,
) -> CameraControlReport:
    """Read the bounded control set once without calling ``VideoCapture.set``.

    Generic OpenCV does not expose property ranges, flags, or a trustworthy
    cross-backend support query.  A non-zero ``get`` result is recorded as
    backend-reported, zero remains explicitly indeterminate, and a missing
    constant/get failure is unavailable.  This keeps V4L2, MSMF, and DSHOW
    observations honest while the only implemented policy is ``preserve``.
    """

    observations: list[CameraControlObservation] = []
    for name, constant_name in (
        ("auto_white_balance", "CAP_PROP_AUTO_WB"),
        ("white_balance_temperature", "CAP_PROP_WB_TEMPERATURE"),
        ("auto_exposure", "CAP_PROP_AUTO_EXPOSURE"),
        ("exposure", "CAP_PROP_EXPOSURE"),
        ("gain", "CAP_PROP_GAIN"),
        ("gamma", "CAP_PROP_GAMMA"),
    ):
        prop = getattr(cv2, constant_name, None)
        value = _safe_get(cap, prop)
        if prop is None or value is None:
            status = "unavailable"
            value = None
        elif value == 0.0:
            status = "indeterminate-zero"
        else:
            status = "reported"
        observations.append(
            CameraControlObservation(name=name, status=status, value=value)
        )
    return CameraControlReport(
        backend_family=_camera_backend_family(backend),
        generation=generation,
        properties=tuple(observations),
    )


class OpenCVCapture(CaptureSource):
    """Real camera with verified mode negotiation and bounded recovery.

    ``read`` never calls OpenCV or performs recovery control work directly.
    One daemon reader owns blocking frame reads and publishes into a single
    latest-frame slot, while a separate controller owns open/release/join and
    backoff.  A stalled reader is released and joined before any replacement
    is created, ensuring that two readers never access the device concurrently.
    """

    _BACKOFFS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    _READER_JOIN_TIMEOUT_S = 1.0
    _RATE_WINDOW_S = 2.0
    _RATE_WARNING_AFTER_S = 5.0
    _EWMA_ALPHA = 0.1

    def __init__(self, cfg: CameraConfig, canvas_size: Size | None = None):
        if cv2 is None:
            raise RuntimeError("opencv-python is required for camera capture")
        self.cfg = cfg
        self.canvas_size = canvas_size or (cfg.width, cfg.height)
        if (
            not isinstance(self.canvas_size, tuple)
            or len(self.canvas_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.canvas_size)
        ):
            raise ValueError("capture canvas dimensions must be positive integers")
        self._device = _device_value(cfg.device)
        self._lock = threading.RLock()
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._controller_stop = threading.Event()
        self._controller_thread: threading.Thread | None = None

        self._cap: Any = None
        self._reader_stop: threading.Event | None = None
        self._reader_thread: threading.Thread | None = None
        self._generation = 0
        self._generation_started_at = time.monotonic()
        self._generation_has_frame = False
        self._open_attempts = 0

        self._slot: CapturedFrame | None = None
        self._slot_sequence = 0
        self._slot_generation = 0
        self._slot_geometry_generation = 0
        self._slot_content_rect: tuple[int, int, int, int] | None = None
        self._slot_identity = CaptureHealth()
        self._delivered_sequence = 0
        self._delivered_generation = 0
        self._delivered_geometry_generation = 0
        self._delivered_content_rect: tuple[int, int, int, int] | None = None
        self._delivered_identity = CaptureHealth()
        self._last_frame_at: float | None = None
        self._first_frame_at: float | None = None
        self._capture_timestamps: deque[float] = deque(maxlen=1024)

        self._backend = "unknown"
        self._fourcc: str | None = None
        self._width: int | None = None
        self._height: int | None = None
        self._delivered_width: int | None = None
        self._delivered_height: int | None = None
        self._oriented_width: int | None = None
        self._oriented_height: int | None = None
        self._normalized_width: int | None = None
        self._normalized_height: int | None = None
        self._generation_delivered_size: Size | None = None
        self._geometry_signature: tuple[object, ...] | None = None
        self._geometry_transitions = 0
        self._fps_reported: float | None = None
        self._target_met: bool | None = None
        self._logged_modes: set[tuple[Any, ...]] = set()
        self._camera_controls = CameraControlReport()

        self._frames_read = 0
        self._dropped_frames = 0
        self._read_failures = 0
        self._restarts = 0
        self._read_ms: float | None = None

        self._stall_after_s = max(2.0, 5.0 / cfg.fps)
        self._recovery_timeout_s = cfg.recovery_timeout_s
        self._backoffs = self._BACKOFFS
        # Initial acquisition is part of the same bounded outage contract as a
        # runtime reconnect.  This lets transient device contention recover
        # without blocking Pipeline.start() or failing on the first open.
        self._recovering = True
        self._recovery_started_at: float | None = self._generation_started_at
        self._next_reopen_at: float | None = self._generation_started_at
        self._backoff_index = 0
        self._auto_mjpeg_retries = 0

        self._rate_low_warned = False
        self._rate_low_since: float | None = None
        self._rate_recovery_windows = 0
        self._last_rate_evaluation = 0.0
        self._last_capture_log_state = ""
        # A compatibility warning belongs to the capture lifetime, not a
        # reconnect generation.  Once an operator has been told that a future
        # proportional-fit rollout will change this framing, repeating it on
        # every reconnect would only create log noise.
        self._aspect_upgrade_note_emitted = False

        controller = threading.Thread(
            target=self._controller_loop,
            name="camera-controller",
            daemon=True,
        )
        self._controller_thread = controller
        controller.start()

    # -- resource construction ---------------------------------------
    def _open_candidates(self) -> tuple[list[tuple[str | None, int | None]], bool]:
        """Ordered ``(name, apiPreference)`` opens plus an "explicit" flag.

        Windows returns an explicit MSMF→DSHOW order (WIN-3.2); every other
        platform returns a single ``(None, None)`` entry, i.e. the historical
        ``cv2.VideoCapture(device)`` with OpenCV's default backend — POSIX open
        behavior is byte-for-byte unchanged.  The flag drives whether a total
        open failure carries the platform privacy/contention hint (WIN-3.3).
        """

        backends = preferred_capture_backends(cv2)
        if not backends:
            return [(None, None)], False
        return [(name, api) for name, api in backends], True

    def _open_configured_capture(self) -> tuple[Any, str]:
        candidates, explicit = self._open_candidates()
        for _name, api in candidates:
            cap = (
                cv2.VideoCapture(self._device)
                if api is None
                else cv2.VideoCapture(self._device, api)
            )
            if not cap.isOpened():
                # Release and try the next backend; on Windows an MSMF failure
                # commonly succeeds under DSHOW (and vice versa).
                try:
                    cap.release()
                except Exception as exc:
                    log.debug(
                        "cannot release unopened camera (%s)",
                        type(exc).__name__,
                    )
                continue
            return self._configure_capture(cap)
        message = "cannot open configured camera"
        if explicit:
            message = f"{message} ({camera_open_hint()})"
        raise CaptureError(message)

    def _configure_capture(self, cap: Any) -> tuple[Any, str]:
        try:
            backend = _backend_name(cap)
            request_mjpeg = self.cfg.pixel_format == "mjpeg" or (
                self.cfg.pixel_format == "auto" and _is_v4l2(backend)
            )

            # V4L2 mode selection is order-sensitive: compressed format must
            # be selected before dimensions and rate to avoid a silent YUYV
            # fallback such as 1280x720@10.
            if request_mjpeg:
                accepted = cap.set(cv2.CAP_PROP_FOURCC, float(_fourcc_value("MJPG")))
                if self.cfg.pixel_format == "mjpeg" and not bool(accepted):
                    raise CaptureModeError("camera rejected explicit MJPG format")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.cfg.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.cfg.height))
            cap.set(cv2.CAP_PROP_FPS, float(self.cfg.fps))
            return cap, backend
        except BaseException:
            try:
                cap.release()
            except Exception as exc:
                log.debug(
                    "cannot release failed camera open (%s)",
                    type(exc).__name__,
                )
            raise

    def _start_generation(self) -> None:
        cap, backend = self._open_configured_capture()
        stop = threading.Event()
        with self._lock:
            if self._closed or self._fatal_error is not None:
                cap.release()
                raise CaptureError("camera acquisition ended while reopening")
            self._generation += 1
            generation = self._generation
            self._cap = cap
            self._backend = backend
            self._reader_stop = stop
            self._generation_started_at = time.monotonic()
            self._generation_has_frame = False
            self._generation_delivered_size = None
            self._geometry_signature = None
            self._next_reopen_at = None
            self._capture_timestamps.clear()
            self._first_frame_at = None
            self._last_rate_evaluation = 0.0
            self._rate_low_since = None
            self._rate_recovery_windows = 0
            thread = threading.Thread(
                target=self._reader_loop,
                args=(cap, stop, generation),
                name="camera-capture",
                daemon=True,
            )
            self._reader_thread = thread
        thread.start()

    # -- negotiation --------------------------------------------------
    def _record_negotiated_mode(
        self,
        cap: Any,
        raw_frame: np.ndarray,
        generation: int,
    ) -> CaptureError | None:
        backend = _backend_name(cap)
        frame_height, frame_width = raw_frame.shape[:2]
        prop_width = _safe_get(cap, getattr(cv2, "CAP_PROP_FRAME_WIDTH", None))
        prop_height = _safe_get(cap, getattr(cv2, "CAP_PROP_FRAME_HEIGHT", None))
        prop_fps = _safe_get(cap, getattr(cv2, "CAP_PROP_FPS", None))
        prop_fourcc = _safe_get(cap, getattr(cv2, "CAP_PROP_FOURCC", None))
        camera_controls = _observe_camera_controls(
            cap,
            backend,
            generation=generation,
        )

        # The backend properties describe the negotiated device mode, while the
        # delivered array describes what OpenCV handed to the application.  A
        # backend may scale internally, so using only the ndarray dimensions can
        # hide a lower real capture mode.  Prefer valid post-frame properties in
        # status and validate both surfaces below.
        width = (
            int(round(prop_width))
            if prop_width is not None and prop_width > 0
            else frame_width
        )
        height = (
            int(round(prop_height))
            if prop_height is not None and prop_height > 0
            else frame_height
        )
        if prop_width and prop_width > 0 and int(round(prop_width)) != frame_width:
            log.debug(
                "camera width property %s disagrees with delivered frame width %s",
                int(round(prop_width)),
                frame_width,
            )
        if prop_height and prop_height > 0 and int(round(prop_height)) != frame_height:
            log.debug(
                "camera height property %s disagrees with delivered frame height %s",
                int(round(prop_height)),
                frame_height,
            )
        fps = prop_fps if prop_fps and prop_fps > 0 else None
        fourcc = _fourcc_name(prop_fourcc)

        auto_mjpeg_fallback = (
            self.cfg.pixel_format == "auto"
            and _is_v4l2(backend)
            and fourcc not in {"MJPG", "JPEG"}
        )
        retry_auto_mjpeg = False
        with self._lock:
            if auto_mjpeg_fallback and self._auto_mjpeg_retries < 1:
                self._auto_mjpeg_retries += 1
                retry_auto_mjpeg = True
            elif not auto_mjpeg_fallback:
                self._auto_mjpeg_retries = 0
        if retry_auto_mjpeg:
            self._log_capture_transition(
                "auto-mjpeg-retry",
                "V4L2 MJPG negotiation did not stick (reported %s); retrying once",
                fourcc or "unknown",
            )
            return _CaptureNegotiationRetry(
                "automatic V4L2 MJPG negotiation did not stick"
            )

        mismatches: list[str] = []
        unverifiable: list[str] = []
        if width != self.cfg.width or height != self.cfg.height:
            mismatches.append(
                f"resolution {width}x{height} (requested "
                f"{self.cfg.width}x{self.cfg.height})"
            )
        if (frame_width, frame_height) != (self.cfg.width, self.cfg.height) and (
            frame_width,
            frame_height,
        ) != (width, height):
            mismatches.append(
                f"delivered resolution {frame_width}x{frame_height} (requested "
                f"{self.cfg.width}x{self.cfg.height})"
            )
        if fps is not None and self.cfg.fps - fps > max(1.0, self.cfg.fps * 0.1):
            mismatches.append(f"{fps:g} fps (requested {self.cfg.fps})")
        elif fps is None:
            unverifiable.append("reported fps")

        if mismatches:
            target_met: bool | None = False
        elif fps is None:
            target_met = None
        else:
            target_met = True

        with self._lock:
            self._backend = backend
            self._width = width
            self._height = height
            self._fps_reported = fps
            self._fourcc = fourcc
            self._target_met = target_met
            self._camera_controls = camera_controls
            signature = (
                self._backend,
                fourcc,
                width,
                height,
                fps,
                tuple(mismatches),
                tuple(unverifiable),
            )
            should_log = signature not in self._logged_modes
            self._logged_modes.add(signature)

        requested = f"{self.cfg.width}x{self.cfg.height} @ {self.cfg.fps} fps"
        negotiated = (
            f"{fourcc or 'unknown format'} {width}x{height} @ "
            f"{f'{fps:g}' if fps is not None else 'unknown'} fps"
        )
        if should_log:
            if mismatches or unverifiable:
                log.warning(
                    "camera mode %s on %s: requested %s, negotiated %s",
                    "mismatch" if mismatches else "could not be fully verified",
                    backend,
                    requested,
                    negotiated,
                )
            elif auto_mjpeg_fallback:
                log.warning(
                    "camera pixel-format fallback on %s: requested MJPG preference, "
                    "negotiated %s",
                    backend,
                    negotiated,
                )
            else:
                log.info("camera negotiated on %s: %s", backend, negotiated)

        if self.cfg.pixel_format == "mjpeg" and fourcc not in {"MJPG", "JPEG"}:
            return CaptureModeError(
                "camera did not activate explicit MJPG format "
                f"(reported {fourcc or 'unknown'})"
            )
        if mismatches and self.cfg.mode_mismatch == "error":
            return CaptureModeError("camera mode mismatch: " + ", ".join(mismatches))
        if unverifiable and self.cfg.mode_mismatch == "error":
            return CaptureModeError(
                "camera mode could not be verified: " + ", ".join(unverifiable)
            )
        return None

    def _normalize_delivered_frame(
        self,
        frame: np.ndarray,
        generation: int,
    ) -> tuple[np.ndarray, TransformPlan]:
        """Validate and normalize one delivered frame exactly once."""

        source = validate_bgr_frame(frame, name="camera frame")
        delivered_size = (source.shape[1], source.shape[0])
        with self._lock:
            previous_size = self._generation_delivered_size
            if previous_size is None:
                self._generation_delivered_size = delivered_size
            elif previous_size != delivered_size:
                if self.cfg.mode_mismatch == "error":
                    raise CaptureModeError(
                        "camera delivered resolution changed during generation "
                        f"from {previous_size[0]}x{previous_size[1]} to "
                        f"{delivered_size[0]}x{delivered_size[1]}"
                    )
                self._generation_delivered_size = delivered_size

        normalized, plan = transform_frame(
            source,
            self.canvas_size,
            rotation=self.cfg.rotation,
            mirror=self.cfg.mirror,
            fit=self.cfg.fit_mode,
            anchors=(self.cfg.anchor_x, self.cfg.anchor_y),
        )
        signature = (
            generation,
            delivered_size,
            plan.oriented_size,
            plan.target_size,
            plan.fit,
            plan.rotation,
            plan.mirror,
            self.cfg.anchor_x,
            self.cfg.anchor_y,
        )
        with self._lock:
            changed = signature != self._geometry_signature
            if changed:
                self._geometry_signature = signature
                self._geometry_transitions += 1
            self._delivered_width, self._delivered_height = delivered_size
            self._oriented_width, self._oriented_height = plan.oriented_size
            self._normalized_width, self._normalized_height = plan.target_size
            aspect_upgrade_note = (
                not self._aspect_upgrade_note_emitted
                and plan.fit == "stretch"
                and plan.oriented_size[0] * plan.target_size[1]
                != plan.oriented_size[1] * plan.target_size[0]
            )
            if aspect_upgrade_note:
                self._aspect_upgrade_note_emitted = True
        if changed:
            self._log_capture_transition(
                f"geometry:{signature!r}",
                "camera geometry generation=%d delivered=%dx%d oriented=%dx%d "
                "canvas=%dx%d fit=%s rotation=%d mirror=%s",
                generation,
                delivered_size[0],
                delivered_size[1],
                plan.oriented_size[0],
                plan.oriented_size[1],
                plan.target_size[0],
                plan.target_size[1],
                plan.fit,
                plan.rotation,
                plan.mirror,
            )
        if aspect_upgrade_note:
            log.warning(
                "visual-policy upgrade note: camera aspect %dx%d differs from "
                "canvas %dx%d; schema-v1 stretch preserves legacy distortion. "
                "The staged cover default will crop proportionally. Pin "
                "camera.fit_mode=stretch to retain current framing or preview "
                "camera.fit_mode=cover before upgrading",
                plan.oriented_size[0],
                plan.oriented_size[1],
                plan.target_size[0],
                plan.target_size[1],
            )
        return normalized, plan

    # -- reader -------------------------------------------------------
    def _reader_loop(self, cap: Any, stop: threading.Event, generation: int) -> None:
        negotiated = False
        try:
            while not stop.is_set():
                started = time.monotonic()
                try:
                    ok, frame = cap.read()
                except BaseException as exc:
                    if stop.is_set():
                        break
                    with self._lock:
                        if generation == self._generation and not self._closed:
                            self._read_failures += 1
                    self._log_capture_transition(
                        f"read-error:{type(exc).__name__}",
                        "camera read failed; recovery will retry (%s)",
                        type(exc).__name__,
                    )
                    return
                captured_at_ns = time.monotonic_ns()
                finished = captured_at_ns / 1_000_000_000.0
                if stop.is_set():
                    break
                if not ok or frame is None:
                    with self._lock:
                        if generation != self._generation or self._closed:
                            return
                        self._read_failures += 1
                    stop.wait(0.01)
                    continue
                try:
                    frame = validate_bgr_frame(frame, name="camera frame")
                except FrameValidationError as exc:
                    self._set_fatal(CaptureError(str(exc)))
                    return

                if not negotiated:
                    mode_error = self._record_negotiated_mode(cap, frame, generation)
                    negotiated = True
                    if isinstance(mode_error, _CaptureNegotiationRetry):
                        with self._lock:
                            if generation == self._generation and not self._closed:
                                self._read_failures += 1
                        return
                    if mode_error is not None:
                        self._set_fatal(mode_error)
                        return

                try:
                    frame, plan = self._normalize_delivered_frame(frame, generation)
                except CaptureModeError as exc:
                    self._set_fatal(exc)
                    return
                except BaseException as exc:
                    self._set_fatal(
                        CaptureError(f"camera frame conversion failed: {exc}")
                    )
                    return

                recovered = False
                announce_recovery = False
                rate_error: CaptureModeError | None = None
                with self._lock:
                    if generation != self._generation or self._closed:
                        return
                    if self._slot is not None and (
                        self._slot.sequence != self._delivered_sequence
                    ):
                        self._dropped_frames += 1
                    self._slot_sequence += 1
                    self._slot_generation = generation
                    self._slot_geometry_generation = self._geometry_transitions
                    content = plan.content_rect
                    self._slot_content_rect = (
                        content.left,
                        content.top,
                        content.right,
                        content.bottom,
                    )
                    self._slot = CapturedFrame(
                        pixels=frame,
                        sequence=self._slot_sequence,
                        captured_at_ns=captured_at_ns,
                        generation=generation,
                        geometry_generation=self._geometry_transitions,
                        content_rect=self._slot_content_rect,
                    )
                    self._slot_identity = CaptureHealth(
                        sequence=self._slot_sequence,
                        captured_monotonic_ns=captured_at_ns,
                        generation=generation,
                        geometry_generation=self._geometry_transitions,
                        content_rect=self._slot_content_rect,
                        backend=self._backend,
                        fourcc=self._fourcc,
                        width=self._width,
                        height=self._height,
                        delivered_width=plan.source_size[0],
                        delivered_height=plan.source_size[1],
                        oriented_width=plan.oriented_size[0],
                        oriented_height=plan.oriented_size[1],
                        normalized_width=plan.target_size[0],
                        normalized_height=plan.target_size[1],
                        geometry_transitions=self._geometry_transitions,
                        fps_reported=self._fps_reported,
                        target_met=self._target_met,
                        camera_controls=self._camera_controls,
                    )
                    self._frames_read += 1
                    self._generation_has_frame = True
                    self._last_frame_at = finished
                    if self._first_frame_at is None:
                        self._first_frame_at = finished
                    self._capture_timestamps.append(finished)
                    while (
                        self._capture_timestamps
                        and finished - self._capture_timestamps[0] > self._RATE_WINDOW_S
                    ):
                        self._capture_timestamps.popleft()
                    read_ms = (finished - started) * 1000.0
                    self._read_ms = (
                        read_ms
                        if self._read_ms is None
                        else self._EWMA_ALPHA * read_ms
                        + (1.0 - self._EWMA_ALPHA) * self._read_ms
                    )
                    recovered = self._recovering
                    # The initial asynchronous acquisition uses the same
                    # internal state as recovery.  It is not a recovery unless
                    # at least one open/generation attempt was retried.
                    announce_recovery = recovered and self._open_attempts > 1
                    if recovered:
                        self._recovering = False
                        self._recovery_started_at = None
                        self._next_reopen_at = None
                        self._backoff_index = 0
                        self._last_capture_log_state = ""
                    rate_error = self._evaluate_rate_locked(finished)
                if announce_recovery:
                    log.info("camera capture recovered")
                if rate_error is not None:
                    self._set_fatal(rate_error)
                    return
        finally:
            # The controller is the sole owner of release/join.  Reader-side
            # release can race a fatal wake-up and is not reliably idempotent
            # across native OpenCV backends.
            pass

    def _capture_fps_locked(self, now: float) -> float:
        recent = [
            stamp
            for stamp in self._capture_timestamps
            if now - stamp <= self._RATE_WINDOW_S
        ]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    def _evaluate_rate_locked(self, now: float) -> CaptureModeError | None:
        if self._first_frame_at is None:
            return None
        if now - self._first_frame_at < self._RATE_WINDOW_S:
            return None
        if self._last_rate_evaluation > 0.0:
            next_evaluation = self._last_rate_evaluation + self._RATE_WINDOW_S
            # Once a complete low window establishes that the rate has been
            # below target, evaluate exactly at the five-second warning edge
            # instead of waiting for a sixth/seventh second window boundary.
            if self._rate_low_since is not None and not self._rate_low_warned:
                next_evaluation = min(
                    next_evaluation,
                    self._rate_low_since + self._RATE_WARNING_AFTER_S,
                )
            if now < next_evaluation:
                return None
        self._last_rate_evaluation = now
        measured = self._capture_fps_locked(now)
        self._target_met = measured >= self.cfg.fps * 0.9
        if measured < self.cfg.fps * 0.9:
            self._rate_recovery_windows = 0
            if self._rate_low_since is None:
                # The full rate window is already below target, so its oldest
                # edge is the conservative start of the observed low period.
                self._rate_low_since = max(
                    self._first_frame_at,
                    now - self._RATE_WINDOW_S,
                )
            if now - self._rate_low_since < self._RATE_WARNING_AFTER_S:
                return None
            if not self._rate_low_warned:
                log.warning(
                    "camera capture rate %.1f fps is below 90%% of target %d fps",
                    measured,
                    self.cfg.fps,
                )
                self._rate_low_warned = True
            if self.cfg.mode_mismatch == "error":
                return CaptureModeError(
                    f"camera capture rate {measured:.1f} fps is below target "
                    f"{self.cfg.fps} fps"
                )
            return None
        self._rate_low_since = None
        if self._rate_low_warned and measured >= self.cfg.fps * 0.95:
            self._rate_recovery_windows += 1
            if self._rate_recovery_windows >= 2:
                log.info(
                    "camera capture rate recovered to %.1f fps (target %d fps)",
                    measured,
                    self.cfg.fps,
                )
                self._rate_low_warned = False
                self._rate_recovery_windows = 0
        else:
            self._rate_recovery_windows = 0
        return None

    # -- recovery -----------------------------------------------------
    def _log_capture_transition(self, state: str, message: str, *args: object) -> None:
        with self._lock:
            if self._last_capture_log_state == state:
                return
            self._last_capture_log_state = state
        log.warning(message, *args)

    def _set_fatal(self, error: BaseException) -> None:
        with self._lock:
            if self._fatal_error is None:
                self._fatal_error = error
            self._recovering = False
        # Wake the controller so it can release owned resources.  This event is
        # also safe when the controller itself discovered the failure.
        self._controller_stop.set()

    def _stop_current_worker(self) -> bool:
        with self._lock:
            cap = self._cap
            stop = self._reader_stop
            thread = self._reader_thread
        if stop is not None:
            stop.set()
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                log.debug(
                    "cannot release stalled camera (%s)",
                    type(exc).__name__,
                )
        if thread is not None and thread is not threading.current_thread():
            thread.join(self._READER_JOIN_TIMEOUT_S)
        alive = bool(thread is not None and thread.is_alive())
        with self._lock:
            if self._reader_thread is thread and not alive:
                self._reader_thread = None
                self._reader_stop = None
                self._cap = None
        if alive:
            self._set_fatal(
                CaptureWorkerError(
                    "camera reader remained blocked after release and a 1-second join"
                )
            )
            return False
        return True

    def _schedule_reopen_locked(self, now: float) -> None:
        delay = self._backoffs[min(self._backoff_index, len(self._backoffs) - 1)]
        self._backoff_index += 1
        self._next_reopen_at = now + delay

    def _begin_recovery(self, now: float, baseline: float) -> None:
        with self._lock:
            if self._recovering or self._closed:
                return
            self._recovering = True
            self._recovery_started_at = baseline
            self._backoff_index = 0
            self._auto_mjpeg_retries = 0
        self._log_capture_transition(
            "stalled",
            "camera capture stalled; releasing device and starting bounded recovery",
        )
        if not self._stop_current_worker():
            return
        with self._lock:
            self._schedule_reopen_locked(now)

    def _expire_recovery(self, now: float) -> BaseException | None:
        """Publish the terminal outage without performing blocking cleanup."""

        with self._lock:
            recovery_started = self._recovery_started_at
            if (
                self._closed
                or self._fatal_error is not None
                or not self._recovering
                or recovery_started is None
                or now - recovery_started < self._recovery_timeout_s
            ):
                return self._fatal_error
            error = CaptureError(
                f"camera produced no valid frame for "
                f"{self._recovery_timeout_s:g} seconds"
            )
        self._set_fatal(error)
        return error

    def _controller_step(self, now: float) -> None:
        """Advance capture ownership/recovery outside the output frame loop."""

        if self._expire_recovery(now) is not None:
            return
        with self._lock:
            if self._closed or self._fatal_error is not None:
                return
            last_frame = self._last_frame_at
            baseline = last_frame or self._generation_started_at
            recovering = self._recovering
            generation_has_frame = self._generation_has_frame
            generation_started = self._generation_started_at
            thread = self._reader_thread
            next_reopen = self._next_reopen_at

        if not recovering:
            if now - baseline >= self._stall_after_s:
                self._begin_recovery(now, baseline)
            return

        if thread is not None and thread.is_alive():
            if generation_has_frame:
                return
            if now - generation_started < self._stall_after_s:
                return
            self._log_capture_transition(
                "reopened-no-frame",
                "reopened camera still produced no frame; retrying",
            )
            if not self._stop_current_worker():
                return
            with self._lock:
                self._schedule_reopen_locked(now)
            return

        # Clear references left by a reader that exited on its own before
        # deciding whether the next attempt is due.
        if thread is not None and not self._stop_current_worker():
            return
        with self._lock:
            if self._closed or self._fatal_error is not None:
                return
            if self._next_reopen_at is None:
                self._schedule_reopen_locked(now)
            next_reopen = self._next_reopen_at
        if next_reopen is None or now < next_reopen:
            return

        with self._lock:
            self._next_reopen_at = None
            self._open_attempts += 1
            if self._open_attempts > 1:
                self._restarts += 1
        try:
            self._start_generation()
        except CaptureModeError as exc:
            self._set_fatal(exc)
        except BaseException as exc:
            with self._lock:
                if self._closed or self._fatal_error is not None:
                    return
                self._read_failures += 1
                self._schedule_reopen_locked(time.monotonic())
            _candidates, explicit = self._open_candidates()
            hint = f"; {camera_open_hint()}" if explicit else ""
            self._log_capture_transition(
                f"reopen-error:{type(exc).__name__}",
                "camera reopen failed (%s%s)",
                type(exc).__name__,
                hint,
            )

    def _controller_loop(self) -> None:
        """Own all potentially blocking camera lifecycle operations."""

        try:
            while not self._controller_stop.is_set():
                self._controller_step(time.monotonic())
                self._controller_stop.wait(0.01)
        except BaseException as exc:
            self._set_fatal(
                CaptureWorkerError(f"camera controller failed ({type(exc).__name__})")
            )
        finally:
            self._stop_current_worker()

    # -- public API ---------------------------------------------------
    def read(self) -> CapturedFrame | None:
        """Return the latest unread frame without blocking on camera control."""

        # A native open call may itself ignore backend timeouts.  Publishing
        # the deadline here remains constant-time and lets the pipeline fail at
        # the configured outage boundary even while the controller is stuck in
        # native code; process teardown is then the only safe cancellation.
        self._expire_recovery(time.monotonic())
        with self._lock:
            if self._closed:
                return None
            fatal = self._fatal_error
            if fatal is not None:
                raise fatal
            if self._slot is None or self._slot.sequence == self._delivered_sequence:
                return None
            captured = self._slot
            self._delivered_sequence = captured.sequence
            self._delivered_generation = captured.generation
            self._delivered_geometry_generation = captured.geometry_generation
            self._delivered_content_rect = captured.content_rect
            self._delivered_identity = self._slot_identity
            return captured

    def health_snapshot(self) -> CaptureHealth:
        now = time.monotonic()
        with self._lock:
            frame_age_ms = (
                None
                if self._last_frame_at is None
                else max(0.0, (now - self._last_frame_at) * 1000.0)
            )
            stalled = self._recovering or (
                not self._closed
                and self._fatal_error is None
                and now - (self._last_frame_at or self._generation_started_at)
                >= self._stall_after_s
            )
            thread = self._reader_thread
            measured_fps = self._capture_fps_locked(now)
            actual_target_met = (
                None
                if self._first_frame_at is None
                or now - self._first_frame_at < self._RATE_WINDOW_S
                else measured_fps >= self.cfg.fps * 0.9
            )
            identity = self._delivered_identity
            return CaptureHealth(
                sequence=identity.sequence,
                captured_monotonic_ns=identity.captured_monotonic_ns,
                generation=identity.generation,
                geometry_generation=identity.geometry_generation,
                content_rect=identity.content_rect,
                backend=identity.backend,
                fourcc=identity.fourcc,
                width=identity.width,
                height=identity.height,
                delivered_width=identity.delivered_width,
                delivered_height=identity.delivered_height,
                oriented_width=identity.oriented_width,
                oriented_height=identity.oriented_height,
                normalized_width=identity.normalized_width,
                normalized_height=identity.normalized_height,
                geometry_transitions=identity.geometry_transitions,
                fps_reported=identity.fps_reported,
                capture_fps=measured_fps,
                target_met=actual_target_met,
                frames_read=self._frames_read,
                dropped_frames=self._dropped_frames,
                read_failures=self._read_failures,
                restarts=self._restarts,
                stalled=stalled,
                frame_age_ms=frame_age_ms,
                read_ms=self._read_ms,
                fatal_error=str(self._fatal_error) if self._fatal_error else "",
                worker_alive=bool(thread is not None and thread.is_alive()),
                camera_controls=identity.camera_controls,
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            controller = self._controller_thread
        self._controller_stop.set()
        if controller is not None and controller is not threading.current_thread():
            controller.join(self._READER_JOIN_TIMEOUT_S + 0.25)
        if controller is not None and controller.is_alive():
            error = CaptureWorkerError(
                "camera controller remained blocked during bounded shutdown"
            )
            self._set_fatal(error)
            raise error
        if not self._stop_current_worker():
            with self._lock:
                fatal = self._fatal_error
            if fatal is not None:
                raise fatal


class SyntheticCapture(CaptureSource):
    """Test pattern: moving bright ellipse ("person") over a dark gradient."""

    def __init__(self, cfg: CameraConfig, canvas_size: Size | None = None):
        self.cfg = cfg
        self.canvas_size = canvas_size or (cfg.width, cfg.height)
        self._plan = plan_transform(
            (cfg.width, cfg.height),
            self.canvas_size,
            cfg.rotation,
            cfg.mirror,
            cfg.fit_mode,
            (cfg.anchor_x, cfg.anchor_y),
        )
        self.t0 = time.monotonic()
        h, w = cfg.height, cfg.width
        gradient = np.linspace(20, 70, w, dtype=np.uint8)
        self._bg = np.stack([np.tile(gradient, (h, 1))] * 3, axis=-1)
        self._lock = threading.Lock()
        self._closed = False
        self._frames_read = 0
        self._timestamps: deque[float] = deque(maxlen=1024)
        self._first_frame_at: float | None = None
        self._last_frame_at: float | None = None
        self._last_captured_at_ns: int | None = None
        self._read_ms: float | None = None

    def read(self) -> CapturedFrame | None:
        started = time.monotonic()
        with self._lock:
            if self._closed:
                return None
        h, w = self.cfg.height, self.cfg.width
        frame = self._bg.copy()
        t = time.monotonic() - self.t0
        cx = int(w / 2 + (w / 6) * np.sin(t))
        cy = int(h / 2)
        yy, xx = np.ogrid[:h, :w]
        ellipse = ((xx - cx) / (w * 0.14)) ** 2 + ((yy - cy) / (h * 0.3)) ** 2 <= 1.0
        frame[ellipse] = (200, 190, 210)
        frame = apply_transform(frame, self._plan)
        with self._lock:
            # Sequence and timestamp are assigned at the same serialized
            # boundary so concurrent synthetic readers cannot invert them.
            captured_at_ns = time.monotonic_ns()
            finished = captured_at_ns / 1_000_000_000.0
            self._frames_read += 1
            if self._first_frame_at is None:
                self._first_frame_at = finished
            self._last_frame_at = finished
            self._last_captured_at_ns = captured_at_ns
            self._timestamps.append(finished)
            while self._timestamps and finished - self._timestamps[0] > 2.0:
                self._timestamps.popleft()
            read_ms = (finished - started) * 1000.0
            self._read_ms = (
                read_ms
                if self._read_ms is None
                else 0.1 * read_ms + 0.9 * self._read_ms
            )
            sequence = self._frames_read
            content = self._plan.content_rect
            return CapturedFrame(
                pixels=frame,
                sequence=sequence,
                captured_at_ns=captured_at_ns,
                generation=1,
                geometry_generation=1,
                content_rect=(
                    content.left,
                    content.top,
                    content.right,
                    content.bottom,
                ),
            )

    def health_snapshot(self) -> CaptureHealth:
        now = time.monotonic()
        with self._lock:
            recent = [stamp for stamp in self._timestamps if now - stamp <= 2.0]
            if len(recent) >= 2 and recent[-1] > recent[0]:
                capture_fps = (len(recent) - 1) / (recent[-1] - recent[0])
            else:
                capture_fps = 0.0
            target_met = (
                None
                if self._first_frame_at is None or now - self._first_frame_at < 2.0
                else capture_fps >= self.cfg.fps * 0.9
            )
            return CaptureHealth(
                sequence=self._frames_read,
                captured_monotonic_ns=self._last_captured_at_ns,
                generation=1 if self._frames_read else 0,
                geometry_generation=1 if self._frames_read else 0,
                content_rect=(
                    self._plan.content_rect.left,
                    self._plan.content_rect.top,
                    self._plan.content_rect.right,
                    self._plan.content_rect.bottom,
                ),
                backend="synthetic",
                width=self.cfg.width,
                height=self.cfg.height,
                delivered_width=self.cfg.width,
                delivered_height=self.cfg.height,
                oriented_width=self._plan.oriented_size[0],
                oriented_height=self._plan.oriented_size[1],
                normalized_width=self._plan.target_size[0],
                normalized_height=self._plan.target_size[1],
                geometry_transitions=1 if self._frames_read else 0,
                fps_reported=float(self.cfg.fps),
                capture_fps=capture_fps,
                target_met=target_met,
                frames_read=self._frames_read,
                frame_age_ms=(
                    None
                    if self._last_frame_at is None
                    else max(0.0, (now - self._last_frame_at) * 1000.0)
                ),
                read_ms=self._read_ms,
                camera_controls=CameraControlReport(
                    backend_family="synthetic",
                    qualification="not-applicable",
                    generation=1 if self._frames_read else 0,
                ),
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True


def open_capture(cfg: CameraConfig, canvas_size: Size | None = None) -> CaptureSource:
    if cfg.synthetic:
        return SyntheticCapture(cfg, canvas_size)
    return OpenCVCapture(cfg, canvas_size)
