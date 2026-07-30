"""Metadata-aware, bounded video decoding for file backdrops.

OpenCV's video API publishes already-converted BGR pixels without retaining
the range, YUV matrix, primaries, or transfer metadata that justified the
conversion.  This module keeps those declarations attached to each decoded
PyAV frame, supplies them explicitly to FFmpeg's reformatter, and normalizes
the result to custback's full-range display-referred sRGB BGR contract.

Only local, operator-owned files may use explicit metadata overrides.  Pixel
histograms are deliberately never consulted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np

from .color import linear_rgb_to_bgr_u8, srgb_eotf
from .geometry import validate_bgr_frame

try:
    import av as _av
except ImportError:  # pragma: no cover - guarded by a deterministic error
    _av = None

av: Any = _av

VideoColorMatrix: TypeAlias = Literal["auto", "bt601", "bt709"]
VideoColorRange: TypeAlias = Literal["auto", "limited", "full"]
VideoColorPrimaries: TypeAlias = Literal["auto", "bt709", "bt470bg", "smpte170m"]
VideoColorTransfer: TypeAlias = Literal["auto", "srgb", "bt709"]

# OpenCV property identifiers are ABI-stable and let the adapter preserve the
# existing VideoBackdrop scheduler without importing OpenCV or leaking its
# opaque decoded-color behavior into this path.
_CAP_PROP_POS_MSEC = 0
_CAP_PROP_POS_FRAMES = 1
_CAP_PROP_FRAME_WIDTH = 3
_CAP_PROP_FRAME_HEIGHT = 4
_CAP_PROP_FPS = 5
_CAP_PROP_FRAME_COUNT = 7
_CAP_PROP_ORIENTATION_META = 48
_CAP_PROP_ORIENTATION_AUTO = 49

_UNSPECIFIED_COLORSPACE = frozenset({0, 2})
_UNSPECIFIED_PRIMARIES = frozenset({0, 2})
_UNSPECIFIED_TRANSFER = frozenset({0, 2})
_UNSPECIFIED_RANGE = frozenset({0})
_MAX_SEEK_DECODE_FRAMES = 300
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_LOCAL_PROTOCOL_OPTIONS = {"protocol_whitelist": "file"}

# Linear source-primary RGB to linear sRGB (D65).  The values are derived from
# the published BT.709, BT.470 System B/G, and SMPTE-C chromaticities using the
# standard RGB-primary/white-point matrix construction.
_PRIMARY_TO_SRGB = {
    "bt709": np.eye(3, dtype=np.float32),
    "bt470bg": np.asarray(
        (
            (1.044043208763, -0.044043208763, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.011793378284, 0.988206621716),
        ),
        dtype=np.float32,
    ),
    "smpte170m": np.asarray(
        (
            (0.939542063773, 0.050181356860, 0.010276579367),
            (0.017772223144, 0.965792862497, 0.016434914360),
            (-0.001621599943, -0.004369749660, 1.005991349603),
        ),
        dtype=np.float32,
    ),
}


class VideoDecoderError(ValueError):
    """A video cannot satisfy the bounded, declared color contract."""


@dataclass(frozen=True)
class VideoColorOverrides:
    """Optional declarations for a reproducible operator-owned video file."""

    matrix: VideoColorMatrix = "auto"
    range: VideoColorRange = "auto"
    primaries: VideoColorPrimaries = "auto"
    transfer: VideoColorTransfer = "auto"

    @property
    def active(self) -> bool:
        return any(
            value != "auto"
            for value in (self.matrix, self.range, self.primaries, self.transfer)
        )


@dataclass(frozen=True)
class ResolvedVideoColor:
    """The declarations used to normalize one decoded frame."""

    matrix: Literal["bt601", "bt709"]
    range: Literal["limited", "full"]
    primaries: Literal["bt709", "bt470bg", "smpte170m"]
    transfer: Literal["srgb", "bt709"]
    status: Literal["tagged", "operator-override", "legacy-assumption"]
    assumed_fields: tuple[str, ...]
    overridden_fields: tuple[str, ...]
    output: str = "srgb-full-bgr"

    @property
    def declared_input(self) -> str:
        return f"{self.matrix}/{self.range}/{self.primaries}/{self.transfer}"


def _enum_int(value: object) -> int | None:
    try:
        converted = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return converted


def _metadata_value(
    frame: object,
    codec: object,
    name: str,
    unspecified: frozenset[int],
) -> int | None:
    for owner in (frame, codec):
        value = _enum_int(getattr(owner, name, None))
        if value is not None and value not in unspecified:
            return value
    return None


def _resolve_choice(
    *,
    field: str,
    override: str,
    metadata: int | None,
    mapping: dict[int, str],
    legacy: str,
    assumed: list[str],
    overridden: list[str],
) -> str:
    if override != "auto":
        overridden.append(field)
        return override
    if metadata is None:
        assumed.append(field)
        return legacy
    resolved = mapping.get(metadata)
    if resolved is None:
        raise VideoDecoderError(
            f"unsupported tagged video {field} declaration ({metadata})"
        )
    return resolved


def resolve_video_color(
    frame: object,
    codec: object,
    overrides: VideoColorOverrides,
) -> ResolvedVideoColor:
    """Resolve trusted frame/stream tags, explicit overrides, then legacy gaps."""

    assumed: list[str] = []
    overridden: list[str] = []
    matrix = _resolve_choice(
        field="matrix",
        override=overrides.matrix,
        metadata=_metadata_value(frame, codec, "colorspace", _UNSPECIFIED_COLORSPACE),
        mapping={1: "bt709", 5: "bt601", 6: "bt601"},
        legacy="bt709",
        assumed=assumed,
        overridden=overridden,
    )
    range_name = _resolve_choice(
        field="range",
        override=overrides.range,
        metadata=_metadata_value(frame, codec, "color_range", _UNSPECIFIED_RANGE),
        mapping={1: "limited", 2: "full"},
        legacy="full",
        assumed=assumed,
        overridden=overridden,
    )
    primaries = _resolve_choice(
        field="primaries",
        override=overrides.primaries,
        metadata=_metadata_value(
            frame, codec, "color_primaries", _UNSPECIFIED_PRIMARIES
        ),
        mapping={1: "bt709", 5: "bt470bg", 6: "smpte170m"},
        legacy="bt709",
        assumed=assumed,
        overridden=overridden,
    )
    transfer = _resolve_choice(
        field="transfer",
        override=overrides.transfer,
        metadata=_metadata_value(frame, codec, "color_trc", _UNSPECIFIED_TRANSFER),
        mapping={1: "bt709", 6: "bt709", 13: "srgb"},
        legacy="srgb",
        assumed=assumed,
        overridden=overridden,
    )
    status = (
        "operator-override"
        if overridden
        else "legacy-assumption"
        if assumed
        else "tagged"
    )
    return ResolvedVideoColor(
        matrix=matrix,  # type: ignore[arg-type]
        range=range_name,  # type: ignore[arg-type]
        primaries=primaries,  # type: ignore[arg-type]
        transfer=transfer,  # type: ignore[arg-type]
        status=status,
        assumed_fields=tuple(assumed),
        overridden_fields=tuple(overridden),
    )


def _bt709_eotf(encoded: np.ndarray) -> np.ndarray:
    """Decode the BT.709 OETF used by SDR video signals."""

    value = np.asarray(encoded, dtype=np.float32)
    result = np.where(
        value < 0.081,
        value / 4.5,
        np.power((value + 0.099) / 1.099, 1.0 / 0.45),
    )
    return np.ascontiguousarray(result, dtype=np.float32)


def normalize_video_frame(
    frame: object,
    codec: object,
    overrides: VideoColorOverrides,
    *,
    reformatter: Any | None = None,
) -> tuple[np.ndarray, ResolvedVideoColor]:
    """Convert one tagged PyAV frame to full-range display-referred sRGB BGR."""

    if av is None:
        raise RuntimeError(
            "PyAV is required for metadata-aware background video decoding"
        )
    resolved = resolve_video_color(frame, codec, overrides)
    colorspaces = av.video.reformatter.Colorspace
    ranges = av.video.reformatter.ColorRange
    source_matrix = (
        colorspaces.ITU709 if resolved.matrix == "bt709" else colorspaces.ITU601
    )
    source_range = ranges.MPEG if resolved.range == "limited" else ranges.JPEG
    converter = reformatter or av.video.reformatter.VideoReformatter()
    converted = converter.reformat(
        frame,
        format="bgr24",
        src_colorspace=source_matrix,
        dst_colorspace=colorspaces.ITU709,
        src_color_range=source_range,
        dst_color_range=ranges.JPEG,
    )
    bgr = validate_bgr_frame(
        np.ascontiguousarray(converted.to_ndarray()),
        name="decoded background video frame",
        require_contiguous=True,
    )
    if resolved.transfer == "srgb" and resolved.primaries == "bt709":
        return bgr, resolved

    encoded_rgb = np.ascontiguousarray(bgr[..., ::-1], dtype=np.float32) / np.float32(
        255.0
    )
    linear_source = (
        srgb_eotf(encoded_rgb)
        if resolved.transfer == "srgb"
        else _bt709_eotf(encoded_rgb)
    )
    primary_matrix = _PRIMARY_TO_SRGB[resolved.primaries]
    linear_srgb = np.matmul(linear_source, primary_matrix.T)
    return linear_rgb_to_bgr_u8(linear_srgb), resolved


def _finite_rate(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


class MetadataVideoCapture:
    """Small cv2.VideoCapture-compatible adapter backed by PyAV/FFmpeg.

    The adapter exists so the mature VideoBackdrop wall-clock scheduler remains
    unchanged.  Sequential decode, grab/retrieve skipping, bounded seeks, VFR
    timestamps, and loop behavior continue to be tested at that boundary.
    """

    def __init__(
        self,
        path: str,
        *,
        overrides: VideoColorOverrides | None = None,
        max_width: int = 3840,
        max_height: int = 2160,
    ):
        if av is None:
            raise RuntimeError(
                "PyAV is required for metadata-aware background video decoding"
            )
        self._path = path
        self._overrides = overrides or VideoColorOverrides()
        if (path.startswith(("//", "\\\\"))) or (
            _URI_SCHEME_RE.match(path) and not _WINDOWS_DRIVE_RE.match(path)
        ):
            raise VideoDecoderError(
                "background video requires a local operator-owned file"
            )
        source = Path(path)
        try:
            source_is_file = source.is_file()
        except OSError:
            raise FileNotFoundError("cannot open background video") from None
        if not source_is_file:
            raise FileNotFoundError("cannot open background video")
        try:
            # A local playlist/manifest is still capable of naming nested
            # protocols.  Constrain libavformat itself, at the point where
            # those nested resources would be resolved.
            self._container = av.open(
                path,
                mode="r",
                options=dict(_LOCAL_PROTOCOL_OPTIONS),
            )
        except Exception:
            # Native demuxer exceptions commonly include the local path or a
            # nested manifest target.  Do not retain that data in an exception
            # chain that application logs may render.
            raise FileNotFoundError("cannot open background video") from None
        streams = tuple(self._container.streams.video)
        if not streams:
            self._container.close()
            raise VideoDecoderError("background video has no video stream")
        self._stream = streams[0]
        self._codec = self._stream.codec_context
        self._width = int(getattr(self._codec, "width", 0) or 0)
        self._height = int(getattr(self._codec, "height", 0) or 0)
        if (
            self._width <= 0
            or self._height <= 0
            or self._width > max_width
            or self._height > max_height
        ):
            self._container.close()
            raise VideoDecoderError(
                "background video metadata exceeds configured dimensions"
            )
        self._max_width = max_width
        self._max_height = max_height
        self._fps = (
            _finite_rate(getattr(self._stream, "average_rate", None))
            or _finite_rate(getattr(self._stream, "base_rate", None))
            or _finite_rate(getattr(self._stream, "guessed_rate", None))
            or 0.0
        )
        self._frame_count = int(getattr(self._stream, "frames", 0) or 0)
        self._start_time = int(getattr(self._stream, "start_time", 0) or 0)
        raw_time_base = getattr(self._stream, "time_base", None)
        self._time_base = raw_time_base
        if not isinstance(raw_time_base, Fraction):
            try:
                self._time_base = Fraction(str(raw_time_base))
            except (TypeError, ValueError, ZeroDivisionError):
                self._time_base = Fraction(0, 1)
        self._iterator: Any = iter(self._container.decode(self._stream))
        self._prefetched: object | None = None
        self._grabbed: object | None = None
        self._last_frame: object | None = None
        self._last_source_index = -1
        self._next_fallback_index = 0
        self._pts_indexes: dict[int, int] = {}
        self._seek_target_s: float | None = None
        self._seek_target_index: int | None = None
        self._closed = False
        self._fatal_error: VideoDecoderError | None = None
        self._reformatter = av.video.reformatter.VideoReformatter()
        self.color_contract: ResolvedVideoColor | None = None
        self._orientation = 0

        try:
            first = self._decode_next_raw()
            if first is not None:
                self._prefetched = first
                self._orientation = self._normalized_rotation(first)
                # Resolve now so unsupported tagged HDR/wide-gamut content fails
                # during resource preflight rather than after activation.
                self.color_contract = resolve_video_color(
                    first, self._codec, self._overrides
                )
        except VideoDecoderError:
            self._container.close()
            self._closed = True
            raise

    @staticmethod
    def _normalized_rotation(frame: object) -> int:
        for side_data in getattr(frame, "side_data", ()):
            side_type = getattr(side_data, "type", None)
            if getattr(side_type, "name", None) != "DISPLAYMATRIX":
                continue
            raw = bytes(side_data)
            if len(raw) < 36:
                raise VideoDecoderError("invalid video display matrix")
            matrix = np.frombuffer(raw, dtype=np.int32, count=9)
            determinant = int(matrix[0]) * int(matrix[4]) - int(matrix[1]) * int(
                matrix[3]
            )
            if determinant < 0:
                raise VideoDecoderError(
                    "mirrored video display matrices are not supported"
                )
        value = _enum_int(getattr(frame, "rotation", 0))
        if value is None:
            raise VideoDecoderError("invalid video display rotation")
        counter_clockwise = value % 360
        if counter_clockwise not in (0, 90, 180, 270):
            raise VideoDecoderError("video display rotation must be a right angle")
        # PyAV exposes the display matrix in counter-clockwise degrees.
        # CAP_PROP_ORIENTATION_META and custback geometry use clockwise
        # degrees, so translate exactly once at this adapter boundary.
        return (-counter_clockwise) % 360

    def isOpened(self) -> bool:  # noqa: N802 - cv2 compatibility
        return not self._closed

    def getBackendName(self) -> str:  # noqa: N802 - cv2 compatibility
        return "PYAV_FFMPEG"

    def _relative_pts_s(self, frame: object) -> float | None:
        pts = _enum_int(getattr(frame, "pts", None))
        if pts is None or not self._time_base:
            return None
        seconds = float((pts - self._start_time) * self._time_base)
        return seconds if math.isfinite(seconds) and seconds >= 0.0 else None

    def _frame_index(self, frame: object) -> int:
        pts = _enum_int(getattr(frame, "pts", None))
        cached = self._pts_indexes.get(pts) if pts is not None else None
        if cached is not None:
            index = cached
        elif self._seek_target_s is not None and self._fps > 0.0:
            # A random-access decoder exposes timestamps but no ordinal. Use a
            # nominal estimate only for previously unseen seek results; every
            # sequentially observed PTS retains its exact ordinal in the map.
            pts_s = self._relative_pts_s(frame)
            index = (
                int(round(pts_s * self._fps))
                if pts_s is not None
                else self._next_fallback_index
            )
        else:
            index = self._next_fallback_index
        if pts is not None:
            self._pts_indexes[pts] = index
        self._next_fallback_index = max(self._next_fallback_index, index + 1)
        return max(0, index)

    def _decode_next_raw(self) -> object | None:
        if self._fatal_error is not None:
            raise self._fatal_error
        try:
            frame = next(self._iterator)
        except StopIteration:
            return None
        except Exception:
            error = VideoDecoderError("background video decode failed")
            self._fatal_error = error
            raise error from None
        self._validate_raw_frame(frame)
        return frame

    def _validate_raw_frame(self, frame: object) -> None:
        """Reject unsafe dimensions/color before scheduling can discard a frame."""

        width = int(getattr(frame, "width", 0) or 0)
        height = int(getattr(frame, "height", 0) or 0)
        if (
            width <= 0
            or height <= 0
            or width > self._max_width
            or height > self._max_height
        ):
            error = VideoDecoderError(
                "background video contains a frame that exceeds configured dimensions"
            )
            self._fatal_error = error
            raise error
        try:
            resolve_video_color(frame, self._codec, self._overrides)
        except VideoDecoderError as exc:
            self._fatal_error = exc
            raise

    def _next_selected_frame(self) -> object | None:
        if self._prefetched is not None:
            frame = self._prefetched
            self._prefetched = None
        else:
            frame = self._decode_next_raw()
        decoded = 0
        while frame is not None:
            decoded += 1
            index = self._frame_index(frame)
            pts_s = self._relative_pts_s(frame)
            index_ok = (
                self._seek_target_index is None or index >= self._seek_target_index
            )
            time_ok = self._seek_target_s is None or (
                pts_s is not None
                and pts_s + max(1e-6, 0.5 / max(self._fps, 1.0)) >= self._seek_target_s
            )
            if index_ok and time_ok:
                self._seek_target_index = None
                self._seek_target_s = None
                self._last_source_index = index
                self._last_frame = frame
                return frame
            if decoded >= _MAX_SEEK_DECODE_FRAMES:
                return None
            frame = self._decode_next_raw()
        return None

    def _convert(self, frame: object) -> np.ndarray:
        if self._fatal_error is not None:
            raise self._fatal_error
        width = int(getattr(frame, "width", 0) or 0)
        height = int(getattr(frame, "height", 0) or 0)
        if (
            width <= 0
            or height <= 0
            or width > self._max_width
            or height > self._max_height
        ):
            error = VideoDecoderError(
                "background video contains a frame that exceeds configured dimensions"
            )
            self._fatal_error = error
            raise error
        try:
            output, contract = normalize_video_frame(
                frame,
                self._codec,
                self._overrides,
                reformatter=self._reformatter,
            )
        except VideoDecoderError as exc:
            self._fatal_error = exc
            raise
        self.color_contract = contract
        return output

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self._next_selected_frame()
        if frame is None:
            return False, None
        return True, self._convert(frame)

    def grab(self) -> bool:
        self._grabbed = self._next_selected_frame()
        return self._grabbed is not None

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        frame = self._grabbed
        self._grabbed = None
        if frame is None:
            return False, None
        return True, self._convert(frame)

    def get(self, prop: int) -> float:
        if prop == _CAP_PROP_POS_MSEC:
            pts_s = (
                self._relative_pts_s(self._last_frame)
                if self._last_frame is not None
                else None
            )
            return (pts_s or 0.0) * 1000.0
        if prop == _CAP_PROP_POS_FRAMES:
            return float(self._last_source_index + 1)
        if prop == _CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == _CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == _CAP_PROP_FPS:
            return self._fps
        if prop == _CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        if prop == _CAP_PROP_ORIENTATION_META:
            return float(self._orientation)
        if prop == _CAP_PROP_ORIENTATION_AUTO:
            return 0.0
        return 0.0

    def _seek(self, seconds: float, index: int | None) -> bool:
        if not math.isfinite(seconds) or seconds < 0.0 or not self._time_base:
            return False
        offset = self._start_time + int(round(seconds / float(self._time_base)))
        try:
            self._container.seek(
                offset,
                stream=self._stream,
                backward=True,
                any_frame=False,
            )
        except Exception:
            return False
        self._iterator = iter(self._container.decode(self._stream))
        self._prefetched = None
        self._grabbed = None
        self._last_frame = None
        self._next_fallback_index = 0
        self._seek_target_s = seconds
        self._seek_target_index = index
        return True

    def set(self, prop: int, value: float) -> bool:
        if prop == _CAP_PROP_ORIENTATION_AUTO:
            return abs(float(value)) <= 1e-9
        if prop == _CAP_PROP_POS_MSEC:
            try:
                seconds = float(value) / 1000.0
            except (TypeError, ValueError, OverflowError):
                return False
            return self._seek(seconds, None)
        if prop == _CAP_PROP_POS_FRAMES:
            try:
                raw = float(value)
                index = int(round(raw))
            except (TypeError, ValueError, OverflowError):
                return False
            if (
                not math.isfinite(raw)
                or index < 0
                or abs(raw - index) > 1e-6
                or self._fps <= 0.0
            ):
                return False
            return self._seek(index / self._fps, index)
        return False

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._container.close()


def open_metadata_video(
    path: str,
    *,
    overrides: VideoColorOverrides | None = None,
    max_width: int = 3840,
    max_height: int = 2160,
) -> MetadataVideoCapture:
    """Open the mandatory metadata-aware local video decoder."""

    return MetadataVideoCapture(
        path,
        overrides=overrides,
        max_width=max_width,
        max_height=max_height,
    )
