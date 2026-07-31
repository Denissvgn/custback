"""Backdrop providers: static and live.

Every provider returns a BGR frame matching the requested output size on each
call. Video providers follow their source clock; camera/stream providers read
the most recent available frame.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeGuard, cast

import numpy as np

from .color import ColorError, decode_image_to_srgb_bgr
from .config import BackgroundConfig, ResolvedBackdropTarget
from .geometry import (
    FitMode,
    FrameValidationError,
    Rect,
    RightAngleRotation,
    TransformPlan,
    apply_transform,
    orient_frame,
    plan_transform,
    validate_bgr_frame,
)
from .video_decoder import (
    ResolvedVideoColor,
    VideoColorMatrix,
    VideoColorOverrides,
    VideoColorPrimaries,
    VideoColorRange,
    VideoColorTransfer,
    VideoDecoderError,
    open_metadata_video,
)

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

# OpenCV is a compiled optional boundary. Keep its runtime ``None`` fallback
# while treating the dynamically exposed API as opaque to static analysis.
cv2: Any = _cv2
_ORIGINAL_OPENCV_VIDEO_CAPTURE = (
    getattr(_cv2, "VideoCapture", None) if _cv2 is not None else None
)

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".gif", ".avi"}
DEFAULT_IMAGE_MAX_PIXELS = 16_777_216
_IMAGE_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".webp": "WEBP",
}
# Shared by the API's upload endpoints and the preview window's n/p file
# cycling, so files uploaded through one show up in the other.
DEFAULT_BACKGROUNDS_DIR = Path.home() / ".local" / "share" / "custback" / "backgrounds"


def list_background_files(directory: Path) -> list[Path]:
    """Image and video files available for cycling, sorted by name."""
    if not directory.is_dir():
        return []
    exts = IMAGE_EXTS | VIDEO_EXTS
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


@dataclass(frozen=True)
class BackdropGeometry:
    """Hot-swappable backdrop presentation policy."""

    fit_mode: FitMode = "cover"
    anchor_x: float = 0.5
    anchor_y: float = 0.5

    def __post_init__(self) -> None:
        # Reuse the canonical planner's deterministic validation.
        plan_transform(
            (1, 1),
            (1, 1),
            fit=self.fit_mode,
            anchors=(self.anchor_x, self.anchor_y),
        )


@dataclass(frozen=True)
class VideoOrientationPolicy:
    """Observable result of OpenCV container-orientation negotiation."""

    rotation: RightAngleRotation
    metadata_rotation: int | None
    auto_rotation_disabled: bool
    status: str
    ambiguous: bool


_QUALIFIED_VIDEO_ORIENTATION_BACKENDS = frozenset(
    {"FFMPEG", "AVFOUNDATION", "PYAV_FFMPEG"}
)


def _open_video_capture(
    path: str,
    *,
    overrides: VideoColorOverrides,
    max_width: int,
    max_height: int,
) -> Any:
    """Open the production metadata-aware decoder.

    A replaced OpenCV constructor remains an intentional dependency-injection
    seam for deterministic scheduler/backend tests.  The unmodified runtime
    always uses PyAV and never asks OpenCV to infer video color.
    """

    current = getattr(cv2, "VideoCapture", None)
    if current is not None and current is not _ORIGINAL_OPENCV_VIDEO_CAPTURE:
        return current(path)
    return open_metadata_video(
        path,
        overrides=overrides,
        max_width=max_width,
        max_height=max_height,
    )


def _capture_property(cap: Any, prop: int | None) -> float | None:
    if prop is None:
        return None
    try:
        value = float(cap.get(prop))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _capture_backend_name(cap: Any) -> str | None:
    getter = getattr(cap, "getBackendName", None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except Exception:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized or None


def _configure_video_orientation(
    cap: Any,
) -> VideoOrientationPolicy:
    """Disable OpenCV auto-rotation and select only provable metadata rotation."""

    backend_name = _capture_backend_name(cap)
    if backend_name not in _QUALIFIED_VIDEO_ORIENTATION_BACKENDS:
        policy = VideoOrientationPolicy(
            rotation=0,
            metadata_rotation=None,
            auto_rotation_disabled=False,
            status="manual-only-unsupported-backend",
            ambiguous=True,
        )
        log.warning(
            "video orientation is ambiguous; backend %s has no "
            "qualified OpenCV orientation contract, using manual-only policy",
            backend_name or "unknown",
        )
        return policy

    metadata_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    auto_prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if metadata_prop is None or auto_prop is None:
        policy = VideoOrientationPolicy(
            rotation=0,
            metadata_rotation=None,
            auto_rotation_disabled=False,
            status="manual-only-unsupported",
            ambiguous=True,
        )
        log.warning(
            "video orientation is ambiguous; backend exposes no "
            "controllable orientation metadata, using manual-only policy",
        )
        return policy

    raw_metadata = _capture_property(cap, metadata_prop)
    metadata_rotation: int | None = None
    if raw_metadata is not None:
        rounded = int(round(raw_metadata))
        if abs(raw_metadata - rounded) <= 1e-6 and rounded in (0, 90, 180, 270):
            metadata_rotation = rounded

    try:
        control_accepted = bool(cap.set(auto_prop, 0.0))
    except Exception:
        control_accepted = False
    auto_readback = _capture_property(cap, auto_prop)
    auto_disabled = bool(
        control_accepted and auto_readback is not None and abs(auto_readback) <= 1e-6
    )
    if metadata_rotation is not None and auto_disabled:
        policy = VideoOrientationPolicy(
            rotation=cast(RightAngleRotation, metadata_rotation),
            metadata_rotation=metadata_rotation,
            auto_rotation_disabled=True,
            status="qualified-manual-metadata",
            ambiguous=False,
        )
        log.info(
            "video orientation qualified: metadata=%d auto-rotation=disabled",
            metadata_rotation,
        )
        return policy

    status = (
        "manual-only-invalid-metadata"
        if raw_metadata is not None and metadata_rotation is None
        else "manual-only-ambiguous-auto"
    )
    policy = VideoOrientationPolicy(
        rotation=0,
        metadata_rotation=metadata_rotation,
        auto_rotation_disabled=auto_disabled,
        status=status,
        ambiguous=True,
    )
    log.warning(
        "video orientation is ambiguous "
        "(metadata=%s auto-rotation-disabled=%s); using manual-only policy",
        "unknown" if metadata_rotation is None else metadata_rotation,
        auto_disabled,
    )
    return policy


class BackdropProvider(ABC):
    def __init__(
        self,
        *,
        fit_mode: FitMode = "cover",
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ) -> None:
        self._geometry = BackdropGeometry(fit_mode, anchor_x, anchor_y)
        # Scalar-only record of the pixels returned most recently.  The color
        # estimator uses this rectangle to exclude synthetic contain padding;
        # providers never retain an additional frame for that purpose.
        self._last_transform_plan: TransformPlan | None = None

    @abstractmethod
    def frame(self, width: int, height: int) -> np.ndarray:
        """Return the current backdrop as a BGR frame of the given size."""

    @property
    def geometry(self) -> BackdropGeometry:
        return self._geometry

    def set_geometry(
        self,
        fit_mode: FitMode,
        anchor_x: float,
        anchor_y: float,
    ) -> None:
        """Install validated policy and invalidate fitted pixels, not source state."""

        candidate = BackdropGeometry(fit_mode, anchor_x, anchor_y)
        if candidate != self._geometry:
            self._geometry = candidate
            self._last_transform_plan = None
            self._invalidate_geometry_cache()

    def _invalidate_geometry_cache(self) -> None:
        pass

    def _plan(self, frame: np.ndarray, width: int, height: int) -> TransformPlan:
        source = validate_bgr_frame(frame, name="backdrop frame")
        return plan_transform(
            (source.shape[1], source.shape[0]),
            (width, height),
            fit=self._geometry.fit_mode,
            anchors=(self._geometry.anchor_x, self._geometry.anchor_y),
        )

    def _fit_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        plan = self._plan(frame, width, height)
        fitted = apply_transform(frame, plan)
        self._last_transform_plan = plan
        return fitted

    @staticmethod
    def _full_content_rect(width: int, height: int) -> Rect:
        # Reuse the canonical size validator instead of maintaining a weaker
        # private interpretation of canvas dimensions.
        return plan_transform((1, 1), (width, height), fit="stretch").content_rect

    def transform_plan(self, width: int, height: int) -> TransformPlan:
        """Return the immutable plan for the most recently rendered canvas.

        Fitted providers must render the requested dimensions first.  Returning
        guessed geometry could let contain-mode padding influence illumination
        estimates or let correction survive a live source-size change, so a
        stale or missing plan fails deterministically.
        """

        self._full_content_rect(width, height)
        plan = self._last_transform_plan
        if plan is None or plan.target_size != (width, height):
            raise RuntimeError(
                "backdrop content rectangle is unavailable; transform plan "
                "does not match the requested canvas"
            )
        return plan

    def content_rect(self, width: int, height: int) -> Rect:
        """Return valid source content in the most recently rendered canvas."""

        return self.transform_plan(width, height).content_rect

    def close(self) -> None:
        pass

    def reset_stats(self) -> None:
        """Start a new public provider-scoped telemetry generation."""

        pass

    def diagnostic_frame_identity(self) -> dict[str, object]:
        """Return path-free scalar identity for the most recently used frame."""

        return {"provider": type(self).__name__}

    def stats_dict(self) -> dict[str, object]:
        """Return current-provider playback telemetry.

        Non-video providers deliberately expose the same zero/null shape so
        callers can reset public status atomically when modes change.
        """
        return {
            "background_video_source_fps": None,
            "background_video_timing_mode": None,
            "background_video_frames_displayed": 0,
            "background_video_frames_skipped": 0,
            "background_video_frames_reused": 0,
            "background_video_skip_ratio": 0.0,
            "background_video_seek_count": 0,
            "background_video_decode_failures": 0,
            "background_video_orientation_status": None,
            "background_video_metadata_rotation": None,
            "background_video_auto_rotation_disabled": None,
            "background_video_decoder_backend": None,
            "background_video_color_status": None,
            "background_video_input_color": None,
            "background_video_output_color": None,
            "background_video_color_assumed_fields": [],
            "background_video_color_overridden_fields": [],
        }


class ColorBackdrop(BackdropProvider):
    def __init__(
        self,
        color_bgr: tuple[int, int, int],
        *,
        fit_mode: FitMode = "cover",
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        super().__init__(
            fit_mode=fit_mode,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
        self.color = tuple(int(c) for c in color_bgr)
        self._cache: np.ndarray | None = None

    def frame(self, width: int, height: int) -> np.ndarray:
        if self._cache is None or self._cache.shape[:2] != (height, width):
            self._cache = np.full((height, width, 3), self.color, dtype=np.uint8)
        return self._cache

    def content_rect(self, width: int, height: int) -> Rect:
        return self._full_content_rect(width, height)


class ImageBackdrop(BackdropProvider):
    """Static image backdrop."""

    def __init__(
        self,
        path: str,
        *,
        max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
        fit_mode: FitMode = "cover",
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        super().__init__(
            fit_mode=fit_mode,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
        if cv2 is None:
            raise RuntimeError("opencv-python is required for image backdrops")
        if max_pixels <= 0:
            raise ValueError("image backdrop pixel limit must be positive")

        expected_format = _IMAGE_FORMATS.get(Path(path).suffix.lower())
        if expected_format is None:
            raise ValueError("unsupported background image format")
        try:
            self._image = decode_image_to_srgb_bgr(
                path,
                expected_format,
                max_pixels,
            )
        except FileNotFoundError:
            raise FileNotFoundError("cannot read background image") from None
        except ColorError as exc:
            if str(exc) == f"image exceeds {max_pixels} pixels":
                raise ValueError(
                    f"background image exceeds {max_pixels} pixels"
                ) from None
            raise ValueError("invalid background image") from None
        except (OSError, ValueError):
            raise ValueError("invalid background image") from None

        self._image = validate_bgr_frame(
            self._image,
            name="background image",
            require_contiguous=True,
        )
        oriented_height, oriented_width = self._image.shape[:2]
        if oriented_width <= 0 or oriented_height <= 0:
            raise ValueError("invalid background image dimensions")
        if oriented_width * oriented_height > max_pixels:
            raise ValueError(f"background image exceeds {max_pixels} pixels")
        self._cache: np.ndarray | None = None
        self._cache_key: tuple[object, ...] | None = None

    def frame(self, width: int, height: int) -> np.ndarray:
        key = (
            id(self._image),
            (self._image.shape[1], self._image.shape[0]),
            (width, height),
            self._geometry,
            (0, 0, 0),
        )
        if self._cache is None or self._cache_key != key:
            self._cache = self._fit_frame(self._image, width, height)
            self._cache_key = key
        return self._cache

    def _invalidate_geometry_cache(self) -> None:
        self._cache = None
        self._cache_key = None


class VideoBackdrop(BackdropProvider):
    """Monotonic-time-paced looping video backdrop.

    Calls made before the next source-frame deadline reuse the cached frame;
    late calls skip forward so playback speed is independent of pipeline FPS.
    """

    _DEFAULT_FPS = 30.0
    _MIN_FPS = 1.0
    _MAX_FPS = 240.0
    _MAX_SEQUENTIAL_SKIP = 8
    _MIN_TIMESTAMP_STEP_S = 1e-4
    _MAX_TIMESTAMP_STEP_S = 60.0

    def __init__(
        self,
        path: str,
        *,
        clock: Callable[[], float] | None = None,
        max_width: int = 3840,
        max_height: int = 2160,
        fit_mode: FitMode = "cover",
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
        color_matrix: VideoColorMatrix = "auto",
        color_range: VideoColorRange = "auto",
        color_primaries: VideoColorPrimaries = "auto",
        color_transfer: VideoColorTransfer = "auto",
    ):
        super().__init__(
            fit_mode=fit_mode,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
        if cv2 is None:
            raise RuntimeError("opencv-python is required for video backdrops")
        self.path = path
        self._max_width = max_width
        self._max_height = max_height
        self._color_overrides = VideoColorOverrides(
            matrix=color_matrix,
            range=color_range,
            primaries=color_primaries,
            transfer=color_transfer,
        )
        try:
            self.cap = _open_video_capture(
                path,
                overrides=self._color_overrides,
                max_width=max_width,
                max_height=max_height,
            )
        except FileNotFoundError:
            raise FileNotFoundError("cannot open background video") from None
        except VideoDecoderError as exc:
            raise VideoDecoderError(str(exc)) from None
        except RuntimeError:
            raise RuntimeError("cannot open background video") from None
        if not self.cap.isOpened():
            self.cap.release()
            raise FileNotFoundError("cannot open background video")
        self._orientation = _configure_video_orientation(self.cap)
        try:
            metadata_width = float(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata_height = float(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except (TypeError, ValueError, OverflowError):
            metadata_width = metadata_height = 0.0
        if (math.isfinite(metadata_width) and metadata_width > max_width) or (
            math.isfinite(metadata_height) and metadata_height > max_height
        ):
            self.cap.release()
            raise ValueError("background video metadata exceeds configured dimensions")
        try:
            source_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        except (TypeError, ValueError, OverflowError):
            source_fps = float("nan")
        if (
            not math.isfinite(source_fps)
            or source_fps < self._MIN_FPS
            or source_fps > self._MAX_FPS
        ):
            log.warning(
                "background video has implausible FPS metadata (%r); assuming 30 FPS",
                source_fps,
            )
            source_fps = self._DEFAULT_FPS
        try:
            frame_count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except (TypeError, ValueError, OverflowError):
            frame_count = 0.0
        self._frame_count = (
            int(round(frame_count))
            if math.isfinite(frame_count) and frame_count >= 1.0
            else 0
        )
        self._fps = source_fps
        self._clock = clock or time.monotonic
        self._epoch: float | None = None
        self._logical_index = -1
        self._source_index = -1
        self._current_pts_s: float | None = None
        self._current_deadline_s = 0.0
        self._timestamp_origin_s: float | None = None
        self._timestamp_origin_index: int | None = None
        self._last_reliable_pts_s: float | None = None
        self._last_reliable_source_index: int | None = None
        self._container_timing = False
        self._last_pts_step_s = 1.0 / self._fps
        self._duration_s = self._frame_count / self._fps if self._frame_count else 0.0
        self._pending: (
            tuple[
                np.ndarray,
                int,
                int,
                float | None,
                ResolvedVideoColor | None,
            ]
            | None
        ) = None
        self._retry_not_before_s = 0.0
        self._last_raw: np.ndarray | None = None
        self._last_fit: np.ndarray | None = None
        self._last_fit_key: tuple[object, ...] | None = None
        self._fatal_decode_error = False
        self._frames_displayed = 0
        self._frames_skipped = 0
        self._frames_reused = 0
        self._seek_count = 0
        self._decode_failures = 0
        self._last_returned_logical_index: int | None = None
        self._skip_warning_emitted = False
        self._displayed_color_contract: ResolvedVideoColor | None = None

    def _valid_decoded_frame(self, frame: object) -> TypeGuard[np.ndarray]:
        try:
            decoded = validate_bgr_frame(frame, name="background video frame")
        except FrameValidationError:
            valid = False
        else:
            valid = bool(
                decoded.shape[1] <= self._max_width
                and decoded.shape[0] <= self._max_height
            )
        if not valid and frame is not None:
            if (
                isinstance(frame, np.ndarray)
                and frame.ndim >= 2
                and (
                    frame.shape[1] > self._max_width
                    or frame.shape[0] > self._max_height
                )
            ):
                # A resolution-changing asset must not be decoded repeatedly:
                # retain the last good frame and quarantine further reads.
                self._fatal_decode_error = True
            log.warning(
                "ignoring invalid or oversized background video frame",
            )
        return valid

    def _capture_timestamp(self) -> float | None:
        try:
            value = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) and value >= 0.0 else None

    def _capture_index(self, fallback: int) -> int:
        try:
            # OpenCV reports the position of the *next* frame after read().
            value = float(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            index = int(round(value)) - 1
        except (TypeError, ValueError, OverflowError):
            return fallback
        if index < 0 or (self._frame_count and index >= self._frame_count):
            return fallback
        return index

    def _remember(
        self,
        frame: np.ndarray,
        source_index: int,
        logical_index: int,
        pts_s: float | None,
        deadline_s: float,
        color_contract: ResolvedVideoColor | None,
    ) -> None:
        self._last_raw = orient_frame(frame, self._orientation.rotation, False)
        self._source_index = source_index
        self._logical_index = logical_index
        self._current_pts_s = pts_s
        self._current_deadline_s = deadline_s
        self._displayed_color_contract = color_contract
        self._last_fit = None
        self._last_fit_key = None

    def _store_pending(
        self,
        frame: np.ndarray,
        expected_source_index: int,
        logical_index: int,
    ) -> None:
        source_index = self._capture_index(expected_source_index)
        pts_s = self._capture_timestamp()

        if self._timestamp_origin_s is None and pts_s is not None:
            self._timestamp_origin_s = pts_s
            self._timestamp_origin_index = source_index

        origin = self._timestamp_origin_s
        origin_index = self._timestamp_origin_index
        last_pts = self._last_reliable_pts_s
        last_index = self._last_reliable_source_index
        looped = source_index == 0 and self._source_index != 0

        if looped:
            # Estimate the unseen tail from the last *reliable* timestamp, not
            # the immediately displayed frame (whose PTS may have been absent).
            if (
                self._frame_count
                and origin is not None
                and last_pts is not None
                and last_index is not None
            ):
                remaining_intervals = max(1, self._frame_count - last_index)
                candidate = (
                    last_pts - origin + self._last_pts_step_s * remaining_intervals
                )
                if candidate > 0.0 and math.isfinite(candidate):
                    self._duration_s = candidate
            self._last_reliable_pts_s = None
            self._last_reliable_source_index = None
            # A rewind should return to the timestamp origin. Reject a backend
            # that instead exposes an unrelated timeline after looping.
            if pts_s is not None and origin is not None and origin_index == 0:
                tolerance_s = max(0.05, 2.0 * self._last_pts_step_s)
                if abs(pts_s - origin) > tolerance_s:
                    pts_s = None

        step_s: float | None = None
        if pts_s is not None and not looped:
            if (
                last_pts is not None
                and last_index is not None
                and source_index > last_index
            ):
                intervals = source_index - last_index
                step_s = (pts_s - last_pts) / intervals
            elif (
                last_pts is None
                and origin is not None
                and origin_index is not None
                and source_index > origin_index
            ):
                intervals = source_index - origin_index
                step_s = (pts_s - origin) / intervals
            elif last_index == source_index and last_pts is not None:
                tolerance_s = max(0.05, 2.0 * self._last_pts_step_s)
                if abs(pts_s - last_pts) > tolerance_s:
                    pts_s = None
            elif last_index is not None and source_index < last_index:
                pts_s = None

        if step_s is not None:
            if self._MIN_TIMESTAMP_STEP_S <= step_s <= self._MAX_TIMESTAMP_STEP_S:
                self._container_timing = True
                self._last_pts_step_s = step_s
                if self._frame_count and origin is not None and pts_s is not None:
                    # Until EOF is observed, extrapolate the final interval
                    # from the latest real PTS. This is substantially safer
                    # for VFR files than frame_count / nominal_fps.
                    remaining_intervals = self._frame_count - source_index
                    candidate = pts_s - origin + step_s * remaining_intervals
                    if candidate > 0.0 and math.isfinite(candidate):
                        self._duration_s = candidate
            else:
                # A non-increasing or absurd timestamp is not suitable as a
                # playback deadline. Keep the last reliable PTS so a good frame
                # after this one can still be validated against it.
                pts_s = None

        if pts_s is not None:
            self._last_reliable_pts_s = pts_s
            self._last_reliable_source_index = source_index

        color_contract = getattr(self.cap, "color_contract", None)
        self._pending = (
            frame,
            source_index,
            logical_index,
            pts_s,
            color_contract if isinstance(color_contract, ResolvedVideoColor) else None,
        )

    def _prefetch_next(self) -> bool:
        logical_index = self._logical_index + 1
        # Follow the decoder's actual sequential position. Frame-count metadata
        # is frequently off by one (or worse); modulo arithmetic here can make
        # an overstated count freeze forever or an understated count skip data.
        expected_source_index = self._source_index + 1
        ok, frame = self.cap.read()
        if (not ok or frame is None) and expected_source_index > 0:
            # A failed read may have advanced a decoder past a damaged packet.
            # Retry the expected frame explicitly before treating the boundary
            # as EOF and learning/correcting the advertised length from it.
            if self.cap.set(cv2.CAP_PROP_POS_FRAMES, expected_source_index):
                ok, frame = self.cap.read()
        if not ok or frame is None:
            # A sequential read failure followed by a successful rewind is a
            # stronger length signal than CAP_PROP_FRAME_COUNT.
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                return False
            ok, frame = self.cap.read()
            if ok and frame is not None and expected_source_index > 0:
                self._frame_count = expected_source_index
                self._duration_s = self._frame_count / self._fps
                expected_source_index = 0
        if not ok or not self._valid_decoded_frame(frame):
            return False
        if self._frame_count and expected_source_index >= self._frame_count:
            # The decoder produced data beyond an understated advertised count.
            self._frame_count = expected_source_index + 1
            self._duration_s = self._frame_count / self._fps
        self._store_pending(frame, expected_source_index, logical_index)
        self._retry_not_before_s = 0.0
        return True

    def _deadline_s(
        self,
        source_index: int,
        logical_index: int,
        pts_s: float | None,
    ) -> float:
        origin = self._timestamp_origin_s
        if self._container_timing and pts_s is not None and origin is not None:
            media_pts = max(0.0, pts_s - origin)
            if self._frame_count:
                loop = logical_index // self._frame_count
                deadline_s = loop * self._duration_s + media_pts
            else:
                deadline_s = media_pts
            return max(
                deadline_s,
                self._current_deadline_s + self._MIN_TIMESTAMP_STEP_S,
            )
        if self._container_timing:
            # Do not mix a relative container timeline with logical_index/fps,
            # which is anchored at epoch zero. A missing or rejected PTS falls
            # back one validated nominal interval from the displayed deadline.
            return self._current_deadline_s + 1.0 / self._fps
        return logical_index / self._fps

    def _promote_pending(self) -> None:
        assert self._pending is not None
        frame, source_index, logical_index, pts_s, color_contract = self._pending
        deadline_s = self._deadline_s(source_index, logical_index, pts_s)
        self._pending = None
        self._remember(
            frame,
            source_index,
            logical_index,
            pts_s,
            deadline_s,
            color_contract,
        )

    def _skip_stale_nominal(self, steps: int) -> bool:
        """Skip 2..8 CFR frames while decoding only the final image."""
        assert 2 <= steps <= self._MAX_SEQUENTIAL_SKIP
        assert self._pending is not None
        pending = self._pending
        target_logical_index = self._logical_index + steps
        target_source_index = (
            target_logical_index % self._frame_count
            if self._frame_count
            else self._source_index + steps
        )

        # The one-frame look-ahead has already consumed the first stale frame,
        # so grab begins with offset two. Keep the currently displayed frame
        # untouched until retrieve() succeeds.
        def restore_after_pending() -> None:
            _, pending_source_index, _, _, _ = pending
            resume_index = (
                (pending_source_index + 1) % self._frame_count
                if self._frame_count
                else pending_source_index + 1
            )
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, resume_index)

        for offset in range(2, steps + 1):
            logical_index = self._logical_index + offset
            expected_source_index = (
                logical_index % self._frame_count
                if self._frame_count
                else self._source_index + offset
            )
            ok = bool(self.cap.grab())
            if not ok and self._frame_count and expected_source_index == 0:
                if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    restore_after_pending()
                    return False
                ok = bool(self.cap.grab())
            if not ok:
                restore_after_pending()
                return False

        ok, frame = self.cap.retrieve()
        if not ok or not self._valid_decoded_frame(frame):
            restore_after_pending()
            return False
        source_index = self._capture_index(target_source_index)
        pts_s = self._capture_timestamp()
        self._pending = None
        self._remember(
            frame,
            source_index,
            target_logical_index,
            pts_s,
            target_logical_index / self._fps,
            (
                color
                if isinstance(
                    (color := getattr(self.cap, "color_contract", None)),
                    ResolvedVideoColor,
                )
                else None
            ),
        )
        if pts_s is not None:
            self._last_reliable_pts_s = pts_s
            self._last_reliable_source_index = source_index
        self._prefetch_next()
        return True

    def _seek_to_elapsed(self, elapsed_s: float) -> bool:
        """Seek to the current wall-clock phase without publishing failures."""
        used_timestamp_seek = False
        loop = 0
        if self._frame_count:
            duration = self._duration_s or (self._frame_count / self._fps)
            loop = int(elapsed_s // duration)
            phase_s = elapsed_s - loop * duration
            target_index = min(
                self._frame_count - 1,
                int(math.floor(phase_s * self._fps + 1e-9)),
            )
        else:
            phase_s = elapsed_s
            target_index = int(math.floor(elapsed_s * self._fps + 1e-9))

        timestamp_origin_s = self._timestamp_origin_s
        if self._container_timing and timestamp_origin_s is not None:
            used_timestamp_seek = bool(
                self.cap.set(
                    cv2.CAP_PROP_POS_MSEC,
                    (timestamp_origin_s + phase_s) * 1000.0,
                )
            )
        if not used_timestamp_seek:
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_index):
                return False

        ok, frame = self.cap.read()
        pts_s = self._capture_timestamp() if ok and frame is not None else None
        if used_timestamp_seek:
            assert timestamp_origin_s is not None
            target_pts_s = timestamp_origin_s + phase_s
            tolerance_s = max(0.05, 2.0 / self._fps, 2.0 * self._last_pts_step_s)
            timestamp_seek_is_accurate = (
                pts_s is not None and abs(pts_s - target_pts_s) <= tolerance_s
            )
            if not ok or frame is None or not timestamp_seek_is_accurate:
                # Some backends accept CAP_PROP_POS_MSEC but ignore it or only
                # jump to a distant keyframe. Fall back to bounded frame-index
                # seeking rather than publishing an obviously wrong phase.
                if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_index):
                    return False
                ok, frame = self.cap.read()
                pts_s = self._capture_timestamp() if ok and frame is not None else None
        if not ok or not self._valid_decoded_frame(frame):
            # In particular, do not reset to frame zero here: a failed random
            # seek must leave the last successfully decoded image on screen.
            return False

        source_index = self._capture_index(target_index)
        logical_index = (
            loop * self._frame_count + source_index
            if self._frame_count
            else target_index
        )
        self._pending = None
        color = getattr(self.cap, "color_contract", None)
        self._remember(
            frame,
            source_index,
            logical_index,
            pts_s,
            elapsed_s,
            color if isinstance(color, ResolvedVideoColor) else None,
        )
        if pts_s is not None:
            self._last_reliable_pts_s = pts_s
            self._last_reliable_source_index = source_index
        self._prefetch_next()
        self._seek_count += 1
        return True

    def _decode_first(self, now: float) -> None:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok and self._valid_decoded_frame(frame):
            source_index = self._capture_index(0)
            pts_s = self._capture_timestamp()
            self._timestamp_origin_s = pts_s
            self._timestamp_origin_index = source_index if pts_s is not None else None
            self._last_reliable_pts_s = pts_s
            self._last_reliable_source_index = (
                source_index if pts_s is not None else None
            )
            color = getattr(self.cap, "color_contract", None)
            self._remember(
                frame,
                source_index,
                0,
                pts_s,
                0.0,
                color if isinstance(color, ResolvedVideoColor) else None,
            )
            self._epoch = now
            # One-frame look-ahead is what makes an actual container timestamp
            # usable as a deadline instead of merely diagnostic metadata.
            self._prefetch_next()
            if self._fatal_decode_error:
                raise ValueError(
                    "background video contains a frame that exceeds configured dimensions"
                )
        else:
            self._decode_failures += 1

    def _advance_to_time(self, now: float) -> None:
        if self._fatal_decode_error:
            return
        if self._epoch is None:
            self._decode_first(now)
            return
        elapsed_s = max(0.0, now - self._epoch)
        if elapsed_s < self._retry_not_before_s:
            return

        if self._pending is None and not self._prefetch_next():
            self._retry_not_before_s = elapsed_s + 1.0 / self._fps
            self._decode_failures += 1
            log.warning(
                "cannot decode timed background video frame",
            )
            return

        assert self._pending is not None
        _, source_index, logical_index, pts_s, _color_contract = self._pending
        deadline_s = self._deadline_s(source_index, logical_index, pts_s)
        if elapsed_s + 1e-9 < deadline_s:
            return

        # At most eight stale frames are decoded sequentially. Beyond that,
        # seek directly to the retained wall-clock phase.
        timing_fps = (
            min(self._MAX_FPS, 1.0 / self._last_pts_step_s)
            if self._container_timing
            else self._fps
        )
        estimated_stale = max(
            1,
            int(math.floor((elapsed_s - deadline_s) * timing_fps + 1e-9)) + 1,
        )
        if estimated_stale > self._MAX_SEQUENTIAL_SKIP and self._frame_count:
            if not self._seek_to_elapsed(elapsed_s):
                self._retry_not_before_s = elapsed_s + 1.0 / self._fps
                self._decode_failures += 1
                log.warning(
                    "cannot decode timed background video frame",
                )
            return
        if not self._container_timing and estimated_stale > 1 and self._frame_count:
            if not self._skip_stale_nominal(estimated_stale):
                self._retry_not_before_s = elapsed_s + 1.0 / self._fps
                self._decode_failures += 1
                log.warning(
                    "cannot decode timed background video frame",
                )
            return

        advanced = 0
        while self._pending is not None and advanced < self._MAX_SEQUENTIAL_SKIP:
            _, source_index, logical_index, pts_s, _color_contract = self._pending
            deadline_s = self._deadline_s(source_index, logical_index, pts_s)
            if elapsed_s + 1e-9 < deadline_s:
                break
            self._promote_pending()
            advanced += 1
            if not self._prefetch_next():
                self._retry_not_before_s = elapsed_s + 1.0 / self._fps
                break

        if self._pending is not None:
            _, source_index, logical_index, pts_s, _color_contract = self._pending
            if elapsed_s + 1e-9 >= self._deadline_s(source_index, logical_index, pts_s):
                # A phase-preserving seek requires a duration. For containers
                # with unknown length, make bounded sequential progress until
                # EOF reveals the count instead of guessing a non-looping seek.
                if not self._frame_count:
                    return
                if not self._seek_to_elapsed(elapsed_s):
                    self._retry_not_before_s = elapsed_s + 1.0 / self._fps
                    self._decode_failures += 1
                    log.warning(
                        "cannot decode timed background video frame",
                    )

    def frame(self, width: int, height: int) -> np.ndarray:
        self._advance_to_time(self._clock())
        if self._last_raw is None:
            raise RuntimeError("background video has no decodable frame")
        previous = self._last_returned_logical_index
        current = self._logical_index
        if previous is None or current != previous:
            self._frames_displayed += 1
            if previous is not None:
                self._frames_skipped += max(0, current - previous - 1)
            self._last_returned_logical_index = current
        else:
            self._frames_reused += 1
        opportunities = self._frames_displayed + self._frames_skipped
        skip_ratio = self._frames_skipped / opportunities if opportunities else 0.0
        if (
            not self._skip_warning_emitted
            and opportunities >= 30
            and skip_ratio >= 0.10
        ):
            self._skip_warning_emitted = True
            log.warning(
                "background video is skipping %.1f%% of source frames to retain phase",
                skip_ratio * 100.0,
            )
        key = (
            self._logical_index,
            id(self._last_raw),
            (self._last_raw.shape[1], self._last_raw.shape[0]),
            self._orientation.rotation,
            self._orientation.status,
            (width, height),
            self._geometry,
            (0, 0, 0),
        )
        if self._last_fit is None or self._last_fit_key != key:
            self._last_fit = self._fit_frame(self._last_raw, width, height)
            self._last_fit_key = key
        return self._last_fit

    def stats_dict(self) -> dict[str, object]:
        opportunities = self._frames_displayed + self._frames_skipped
        color = self._displayed_color_contract
        return {
            "background_video_source_fps": self._fps,
            "background_video_timing_mode": (
                "container" if self._container_timing else "nominal"
            ),
            "background_video_frames_displayed": self._frames_displayed,
            "background_video_frames_skipped": self._frames_skipped,
            "background_video_frames_reused": self._frames_reused,
            "background_video_skip_ratio": (
                self._frames_skipped / opportunities if opportunities else 0.0
            ),
            "background_video_seek_count": self._seek_count,
            "background_video_decode_failures": self._decode_failures,
            "background_video_orientation_status": self._orientation.status,
            "background_video_metadata_rotation": self._orientation.metadata_rotation,
            "background_video_auto_rotation_disabled": (
                self._orientation.auto_rotation_disabled
            ),
            "background_video_decoder_backend": _capture_backend_name(self.cap),
            "background_video_color_status": (
                color.status if color is not None else "legacy-opencv-assumption"
            ),
            "background_video_input_color": (
                color.declared_input
                if color is not None
                else "opencv-opaque/assumed-srgb-full"
            ),
            "background_video_output_color": (
                color.output if color is not None else "srgb-full-bgr"
            ),
            "background_video_color_assumed_fields": (
                list(color.assumed_fields) if color is not None else ["all"]
            ),
            "background_video_color_overridden_fields": (
                list(color.overridden_fields) if color is not None else []
            ),
        }

    def diagnostic_frame_identity(self) -> dict[str, object]:
        """Identify the exact decoded video frame without exposing its path."""

        return {
            "provider": type(self).__name__,
            "source_index": self._source_index,
            "logical_index": self._logical_index,
            "pts_s": self._current_pts_s,
            "timing_mode": "container" if self._container_timing else "nominal",
        }

    def reset_stats(self) -> None:
        """Exclude activation trials from counters of the installed provider."""

        self._frames_displayed = 0
        self._frames_skipped = 0
        self._frames_reused = 0
        self._seek_count = 0
        self._decode_failures = 0
        self._last_returned_logical_index = None
        self._skip_warning_emitted = False

    def _invalidate_geometry_cache(self) -> None:
        self._last_fit = None
        self._last_fit_key = None

    def close(self) -> None:
        self.cap.release()


class CameraBackdrop(BackdropProvider):
    """Live backdrop from one validated, operator-owned local target."""

    def __init__(
        self,
        target: ResolvedBackdropTarget | int | str,
        *,
        fit_mode: FitMode = "cover",
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        super().__init__(
            fit_mode=fit_mode,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
        if cv2 is None:
            raise RuntimeError("opencv-python is required for camera backdrops")
        if isinstance(target, ResolvedBackdropTarget):
            self.target = target
            device = target.source
        else:
            # Direct construction remains available to local callers/tests, but
            # the application pipeline always supplies an immutable snapshot.
            self.target = ResolvedBackdropTarget("", target)
            device = target
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        self.cap = cv2.VideoCapture(device)
        try:
            opened = self.cap.isOpened()
        except BaseException as exc:
            with contextlib.suppress(Exception):
                self.cap.release()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError("cannot inspect backdrop source") from None
        if not opened:
            with contextlib.suppress(Exception):
                self.cap.release()
            raise RuntimeError("cannot open backdrop source")
        self._last_raw: np.ndarray | None = None
        self._last_fit: np.ndarray | None = None
        self._last_fit_key: tuple[object, ...] | None = None
        self._raw_generation = 0
        self._last_capture_monotonic_ns: int | None = None

    def frame(self, width: int, height: int) -> np.ndarray:
        ok, frame = self.cap.read()
        if ok and frame is not None:
            source = validate_bgr_frame(frame, name="backdrop camera frame")
            self._last_raw = np.ascontiguousarray(source)
            self._raw_generation += 1
            self._last_capture_monotonic_ns = time.monotonic_ns()
            self._last_fit = None
            self._last_fit_key = None
        if self._last_raw is None:
            raise RuntimeError("backdrop camera returned no frame")
        key = (
            self._raw_generation,
            id(self._last_raw),
            (self._last_raw.shape[1], self._last_raw.shape[0]),
            (width, height),
            self._geometry,
            (0, 0, 0),
        )
        if self._last_fit is None or self._last_fit_key != key:
            self._last_fit = self._fit_frame(self._last_raw, width, height)
            self._last_fit_key = key
        return self._last_fit

    def diagnostic_frame_identity(self) -> dict[str, object]:
        return {
            "provider": type(self).__name__,
            "generation": self._raw_generation,
            "capture_monotonic_ns": self._last_capture_monotonic_ns,
        }

    def _invalidate_geometry_cache(self) -> None:
        self._last_fit = None
        self._last_fit_key = None

    def close(self) -> None:
        self.cap.release()


class BlurBackdrop(BackdropProvider):
    """Backdrop is the blurred input frame; call set_source_frame() first.

    The blur runs at reduced resolution (same effective radius, ~10x cheaper
    at 720p). When a person mask is supplied, the person is excluded from the
    blur (normalized masked convolution) so their colors don't smear into the
    background as a ghost halo around the silhouette."""

    def __init__(
        self,
        strength: int,
        *,
        fit_mode: FitMode = "cover",
        anchor_x: float = 0.5,
        anchor_y: float = 0.5,
    ):
        super().__init__(
            fit_mode=fit_mode,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
        self.strength = strength if strength % 2 == 1 else strength + 1
        self._frame: np.ndarray | None = None

    def set_source_frame(
        self, frame: np.ndarray, mask: np.ndarray | None = None
    ) -> None:
        frame = validate_bgr_frame(frame, name="blur source frame")
        if cv2 is None:  # box-blur-ish fallback: downscale/upscale by striding
            small = frame[:: self.strength, :: self.strength]
            self._frame = np.repeat(
                np.repeat(small, self.strength, axis=0), self.strength, axis=1
            )[: frame.shape[0], : frame.shape[1]]
            return
        h, w = frame.shape[:2]
        scale = 4 if self.strength >= 13 and min(h, w) >= 128 else 1
        k = max(3, (self.strength // scale) | 1)
        src = frame
        if scale > 1:
            src = cv2.resize(
                frame, (w // scale, h // scale), interpolation=cv2.INTER_AREA
            )
        src = src.astype(np.float32)
        blurred = cv2.GaussianBlur(src, (k, k), 0)
        if mask is not None and mask.shape == (h, w):
            inv = 1.0 - mask
            if scale > 1:
                inv = cv2.resize(
                    inv, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA
                )
            bg_sum = cv2.GaussianBlur(src * inv[..., None], (k, k), 0)
            bg_weight = cv2.GaussianBlur(inv, (k, k), 0)[..., None]
            bg_only = bg_sum / (bg_weight + 1e-4)
            # Use the person-free blur wherever enough real background is
            # visible; deep inside the person keep the plain blur (that area
            # is covered by the person in the composite anyway).
            t = np.clip(bg_weight * 8.0, 0.0, 1.0)
            blurred = t * bg_only + (1.0 - t) * blurred
        if scale > 1:
            blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        self._frame = np.clip(blurred, 0, 255).astype(np.uint8)

    def frame(self, width: int, height: int) -> np.ndarray:
        if self._frame is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        frame = validate_bgr_frame(self._frame, name="blurred backdrop frame")
        if frame.shape != (height, width, 3):
            raise ValueError(
                "blurred backdrop must already match the canonical canvas: "
                f"expected {(height, width, 3)}, got {frame.shape}"
            )
        return np.ascontiguousarray(frame)

    def content_rect(self, width: int, height: int) -> Rect:
        return self._full_content_rect(width, height)


def create_backdrop(
    cfg: BackgroundConfig,
    *,
    camera_target: ResolvedBackdropTarget | None = None,
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
    video_max_width: int = 3840,
    video_max_height: int = 2160,
) -> BackdropProvider | None:
    """Build the active provider, including remote mode's preflight backdrop."""
    mode = cfg.mode
    if mode == "passthrough":
        return None
    if mode == "remote":
        mode = cfg.remote_fallback_mode
    geometry = {
        "fit_mode": cfg.fit_mode,
        "anchor_x": cfg.anchor_x,
        "anchor_y": cfg.anchor_y,
    }
    if mode == "blur":
        return BlurBackdrop(cfg.blur_strength, **geometry)
    if mode == "color":
        return ColorBackdrop(cfg.color, **geometry)
    if mode == "image":
        return ImageBackdrop(
            cfg.image_path,
            max_pixels=image_max_pixels,
            **geometry,
        )
    if mode == "video":
        return VideoBackdrop(
            cfg.video_path,
            max_width=video_max_width,
            max_height=video_max_height,
            color_matrix=cfg.video_color_matrix,
            color_range=cfg.video_color_range,
            color_primaries=cfg.video_color_primaries,
            color_transfer=cfg.video_color_transfer,
            **geometry,
        )
    if mode == "camera":
        if camera_target is None:
            if cfg.camera_target:
                raise ValueError(
                    "camera target was not resolved by the configuration owner"
                )
            if cfg.camera_device == "":
                raise ValueError("camera backdrop source is not configured")
            camera_target = ResolvedBackdropTarget("", cfg.camera_device)
        return CameraBackdrop(camera_target, **geometry)
    raise ValueError(f"unknown background mode: {mode!r}")
