"""Color-management, foreground estimation, and temporal harmonization.

External frames use the project's contiguous ``uint8 BGR`` contract.  Color
analysis and photometric transforms use finite ``float32`` linear sRGB.  The
estimator is pure and bounded to a 192-pixel analysis raster; the harmonizer
retains scalar parameters and signatures only.
"""

from __future__ import annotations

import io
import math
import os
import warnings
from concurrent.futures import Executor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from .geometry import Rect, apply_exif_orientation, validate_bgr_frame

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover - a required runtime dependency
    _cv2 = None

try:
    from PIL import Image, ImageCms
except ImportError:  # pragma: no cover - a required runtime dependency
    Image = ImageCms = None


cv2: Any = _cv2

ANALYSIS_LONG_EDGE = 192
MASK_CORE_THRESHOLD = 0.90
MASK_SUPPORT_THRESHOLD = 0.05
MASK_EROSION_SIZE = 5
TARGET_DILATION_SIZE = 19
MIN_SAMPLES = 96
MIN_MASK_COVERAGE = 0.01
MIN_CONFIDENCE = 0.45
NEAR_BLACK = 0.02
NEAR_CLIP = 0.98
NEUTRAL_SATURATION_MAX = 0.30
WB_GAIN_MIN = 0.86
WB_GAIN_MAX = 1.16
MAX_EXPOSURE_EV = 1.0

EXPOSURE_DEADBAND_EV = 0.010
WB_DEADBAND_LOG2 = 0.005
STEADY_EXPOSURE_SLEW_EV_S = 0.50
STEADY_WB_SLEW_LOG2_S = 0.20
FAST_EXPOSURE_SLEW_EV_S = 1.50
FAST_WB_SLEW_LOG2_S = 0.40
FAST_ACQUISITION_S = 1.0
LOW_CONFIDENCE_FREEZE_S = 0.5
STALE_DECAY_TAU_S = 1.5
STALE_CLEAR_S = 5.0
SCENE_CUT_LUMINANCE_EV = 0.75
SCENE_CUT_CHROMA_LOG2 = 0.20
CAMERA_TAU_MULTIPLIER = 2.0
CAMERA_SLEW_MULTIPLIER = 0.5

ContentRect: TypeAlias = Rect | tuple[int, int, int, int] | None


class ColorError(ValueError):
    """A color input or operation violates the bounded color contract."""


class ColorReason(str, Enum):
    """Bounded estimator outcome suitable for state and diagnostics."""

    OK = "ok"
    INSUFFICIENT_MASK = "insufficient-mask"
    INSUFFICIENT_NEUTRAL = "insufficient-neutral"
    CLIPPED = "clipped"
    MODE_EXCLUDED = "mode-excluded"
    SOLID_SATURATED = "solid-saturated"
    INVALID = "invalid"


class ColorBehavior(str, Enum):
    """Instantaneous transform selected by the estimator."""

    IDENTITY = "identity"
    EXPOSURE_ONLY = "exposure-only"
    EXPOSURE_WHITE_BALANCE = "exposure-white-balance"


class HarmonizerPhase(str, Enum):
    """Temporal-state phase; detailed public telemetry is added in VIS-3.1."""

    IDENTITY = "identity"
    WARMING = "warming"
    ACTIVE = "active"
    FROZEN = "frozen"
    STALE_DECAY = "stale-decay"
    SCENE_CUT = "scene-cut"


def _finite_float_array(value: np.ndarray | float, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ColorError(f"{name} must contain finite numeric values") from exc
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise ColorError(f"{name} must contain finite numeric values")
    return array


def srgb_eotf(encoded: np.ndarray | float) -> np.ndarray:
    """Decode normalized sRGB code values to linear-light sRGB."""

    value = _finite_float_array(encoded, "encoded sRGB")
    if float(np.min(value)) < 0.0 or float(np.max(value)) > 1.0:
        raise ColorError("encoded sRGB values must be in [0, 1]")
    result = np.where(
        value <= 0.04045,
        value / 12.92,
        np.power((value + 0.055) / 1.055, 2.4),
    )
    converted = np.asarray(result, dtype=np.float32)
    return converted if converted.ndim == 0 else np.ascontiguousarray(converted)


def srgb_oetf(linear: np.ndarray | float) -> np.ndarray:
    """Encode normalized linear sRGB to normalized sRGB code values."""

    value = _finite_float_array(linear, "linear sRGB")
    if float(np.min(value)) < 0.0 or float(np.max(value)) > 1.0:
        raise ColorError("linear sRGB values must be in [0, 1]")
    result = np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )
    converted = np.asarray(result, dtype=np.float32)
    return converted if converted.ndim == 0 else np.ascontiguousarray(converted)


_SRGB_U8_TO_LINEAR = srgb_eotf(np.arange(256, dtype=np.float32) / 255.0)


def _bgr_u8_to_linear_bgr_prevalidated(frame: np.ndarray) -> np.ndarray:
    """Decode a validated external frame while retaining its BGR channel order.

    This is the compiled production boundary used by the pipeline.  Callers must
    already have enforced the canonical contiguous ``uint8 BGR`` frame contract;
    keeping BGR here avoids a full-frame channel-reversal copy before OpenCV
    compositing.  Public color APIs continue to expose linear RGB.
    """

    if cv2 is not None:
        return np.ascontiguousarray(
            cv2.LUT(frame, _SRGB_U8_TO_LINEAR),
            dtype=np.float32,
        )
    return np.ascontiguousarray(_SRGB_U8_TO_LINEAR[frame], dtype=np.float32)


def _linear_bgr_to_bgr_u8_prevalidated(linear_bgr: np.ndarray) -> np.ndarray:
    """Encode a prevalidated linear-BGR production buffer through OpenCV.

    OpenCV's linear-BGR -> Lab -> encoded-BGR route applies the standard sRGB
    transfer function in compiled code.  It is equivalent to the public NumPy
    RGB OETF within one code value; exact alpha endpoints are restored by the
    compositor from the authoritative external BGR inputs.
    """

    if cv2 is None:
        return linear_rgb_to_bgr_u8(
            np.ascontiguousarray(linear_bgr[..., ::-1], dtype=np.float32)
        )
    # Production arithmetic is non-negative. OpenCV's linear-sRGB conversion
    # saturates values above the display-referred unit interval, matching the
    # public OETF's explicit upper clip without another full-frame pass.
    lab = cv2.cvtColor(linear_bgr, cv2.COLOR_LBGR2Lab)
    encoded = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return np.ascontiguousarray(cv2.convertScaleAbs(encoded, alpha=255.0))


def _consume_linear_bgr_to_bgr_u8_prevalidated(
    owned_linear_bgr: np.ndarray,
) -> np.ndarray:
    """Encode a dead-after-call linear-BGR buffer using in-place color stages.

    The caller transfers ownership of ``owned_linear_bgr`` to this helper and
    must not read it afterward.  Reusing that allocation for LBGR -> Lab ->
    encoded BGR removes two full-frame float32 temporaries from the production
    compositor.  The non-consuming helper above remains the contract for callers
    that retain their source.
    """

    if cv2 is None:
        return _linear_bgr_to_bgr_u8_prevalidated(owned_linear_bgr)
    lab = cv2.cvtColor(
        owned_linear_bgr,
        cv2.COLOR_LBGR2Lab,
        dst=owned_linear_bgr,
    )
    encoded = cv2.cvtColor(
        lab,
        cv2.COLOR_Lab2BGR,
        dst=owned_linear_bgr,
    )
    return np.ascontiguousarray(cv2.convertScaleAbs(encoded, alpha=255.0))


def bgr_u8_to_linear_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert one external BGR frame to contiguous linear-sRGB RGB."""

    try:
        source = validate_bgr_frame(
            frame,
            name="color input",
            require_contiguous=True,
        )
    except ValueError as exc:
        raise ColorError(str(exc)) from exc
    # A compiled LUT avoids a full-frame power operation. The final slice makes
    # the channel-order boundary explicit and returns independent storage.
    if cv2 is not None:
        decoded_bgr = cv2.LUT(source, _SRGB_U8_TO_LINEAR)
        return np.ascontiguousarray(decoded_bgr[..., ::-1], dtype=np.float32)
    return np.ascontiguousarray(_SRGB_U8_TO_LINEAR[source[..., ::-1]])


def linear_rgb_to_bgr_u8(linear_rgb: np.ndarray) -> np.ndarray:
    """Clip, sRGB-encode, round, and convert linear RGB to external BGR."""

    value = _finite_float_array(linear_rgb, "linear RGB frame")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ColorError("linear RGB frame must be a non-empty HxWx3 array")
    clipped = np.clip(value, 0.0, 1.0)
    encoded = srgb_oetf(clipped)
    quantized = np.rint(encoded * 255.0).astype(np.uint8)
    return np.ascontiguousarray(quantized[..., ::-1])


def linear_luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """Return finite Rec.709/sRGB relative luminance."""

    value = _finite_float_array(linear_rgb, "linear RGB")
    if value.shape[-1:] != (3,):
        raise ColorError("linear RGB must have a final dimension of three")
    result = (
        value[..., 0] * np.float32(0.2126729)
        + value[..., 1] * np.float32(0.7151522)
        + value[..., 2] * np.float32(0.0721750)
    )
    return np.ascontiguousarray(result, dtype=np.float32)


def linear_log_luminance(
    linear_rgb: np.ndarray,
    *,
    floor: float = 1e-6,
) -> np.ndarray:
    """Return base-2 log luminance with an explicit finite positive floor."""

    if (
        isinstance(floor, bool)
        or not isinstance(floor, (int, float))
        or not math.isfinite(float(floor))
        or float(floor) <= 0.0
    ):
        raise ColorError("log-luminance floor must be finite and positive")
    luminance = linear_luminance(linear_rgb)
    return np.ascontiguousarray(
        np.log2(np.maximum(luminance, np.float32(floor))),
        dtype=np.float32,
    )


def _selected_values(
    values: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    array = _finite_float_array(values, "statistic input")
    if mask is None:
        return (
            array.reshape((-1,) + array.shape[array.ndim - 1 :])
            if array.ndim > 1
            else array
        )
    selected_mask = np.asarray(mask)
    if (
        selected_mask.dtype != np.bool_
        or selected_mask.shape != array.shape[: selected_mask.ndim]
        or selected_mask.ndim == 0
    ):
        raise ColorError("statistic mask must be a matching boolean array")
    selected = array[selected_mask]
    if selected.size == 0:
        raise ColorError("statistic mask selects no values")
    return selected


def masked_median(
    values: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Compute a finite robust median, optionally after boolean selection."""

    selected = _selected_values(values, mask)
    result = np.median(selected, axis=axis)
    if np.ndim(result) == 0:
        return float(result)
    return np.ascontiguousarray(result, dtype=np.float32)


def masked_quantile(
    values: np.ndarray,
    quantile: float,
    mask: np.ndarray | None = None,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Compute a finite robust quantile, optionally after boolean selection."""

    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not math.isfinite(float(quantile))
        or not 0.0 <= float(quantile) <= 1.0
    ):
        raise ColorError("quantile must be a finite number in [0, 1]")
    selected = _selected_values(values, mask)
    result = np.quantile(selected, float(quantile), axis=axis)
    if np.ndim(result) == 0:
        return float(result)
    return np.ascontiguousarray(result, dtype=np.float32)


@dataclass(frozen=True)
class ColorTransform:
    """A bounded foreground transform in linear RGB."""

    exposure_ev: float = 0.0
    wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        exposure = float(self.exposure_ev)
        if not math.isfinite(exposure) or abs(exposure) > MAX_EXPOSURE_EV + 1e-9:
            raise ColorError("exposure transform must be finite and within +/-1 EV")
        if not isinstance(self.wb_gains, tuple) or len(self.wb_gains) != 3:
            raise ColorError("white-balance gains must be a three-value tuple")
        gains = tuple(float(value) for value in self.wb_gains)
        if any(
            not math.isfinite(value)
            or value < WB_GAIN_MIN - 1e-9
            or value > WB_GAIN_MAX + 1e-9
            for value in gains
        ):
            raise ColorError("white-balance gains are outside the bounded range")
        object.__setattr__(self, "exposure_ev", exposure)
        object.__setattr__(self, "wb_gains", gains)

    @property
    def is_identity(self) -> bool:
        return self.exposure_ev == 0.0 and self.wb_gains == (1.0, 1.0, 1.0)


IDENTITY_TRANSFORM = ColorTransform()


def bounded_color_transform(
    raw_exposure_ev: float,
    raw_wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    strength: float = 0.5,
    exposure_limit_ev: float = 0.85,
    white_balance_strength: float = 0.5,
) -> ColorTransform:
    """Clamp raw exposure/WB and interpolate each independently in log space."""

    numeric = (
        (raw_exposure_ev, "raw_exposure_ev"),
        (strength, "strength"),
        (exposure_limit_ev, "exposure_limit_ev"),
        (white_balance_strength, "white_balance_strength"),
    )
    for value, name in numeric:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ColorError(f"{name} must be finite")
    if not 0.0 <= float(strength) <= 1.0:
        raise ColorError("strength must be in [0, 1]")
    if not 0.0 <= float(white_balance_strength) <= 1.0:
        raise ColorError("white_balance_strength must be in [0, 1]")
    if not 0.0 <= float(exposure_limit_ev) <= MAX_EXPOSURE_EV:
        raise ColorError("exposure_limit_ev must be in [0, 1]")
    if not isinstance(raw_wb_gains, tuple) or len(raw_wb_gains) != 3:
        raise ColorError("raw_wb_gains must be a three-value tuple")
    gains = np.asarray(raw_wb_gains, dtype=np.float64)
    if not bool(np.isfinite(gains).all()) or bool(np.any(gains <= 0.0)):
        raise ColorError("raw_wb_gains must be finite and positive")
    gains /= np.exp(np.mean(np.log(gains)))
    gains = np.clip(gains, WB_GAIN_MIN, WB_GAIN_MAX)
    applied_gains = np.exp(np.log(gains) * float(white_balance_strength))
    applied_exposure = float(
        np.clip(
            float(raw_exposure_ev),
            -float(exposure_limit_ev),
            float(exposure_limit_ev),
        )
    ) * float(strength)
    return ColorTransform(
        exposure_ev=applied_exposure,
        wb_gains=(
            float(applied_gains[0]),
            float(applied_gains[1]),
            float(applied_gains[2]),
        ),
    )


def apply_color_transform(
    linear_rgb: np.ndarray,
    transform: ColorTransform,
) -> np.ndarray:
    """Apply bounded exposure and diagonal WB, retaining safe headroom."""

    if not isinstance(transform, ColorTransform):
        raise ColorError("transform must be a ColorTransform")
    value = _finite_float_array(linear_rgb, "linear RGB frame")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ColorError("linear RGB frame must be a non-empty HxWx3 array")
    gains = np.asarray(transform.wb_gains, dtype=np.float32)
    gains *= np.float32(2.0**transform.exposure_ev)
    result = np.clip(value, 0.0, 4.0) * gains
    return np.ascontiguousarray(np.clip(result, 0.0, 4.0), dtype=np.float32)


@dataclass(frozen=True)
class ColorSceneSignature:
    """Bounded scalar signature used only for temporal cut detection."""

    source_log_luminance: float
    target_log_luminance: float
    source_chroma_log2: tuple[float, float, float]
    target_chroma_log2: tuple[float, float, float]


@dataclass(frozen=True)
class ColorEstimate:
    """One pure instantaneous estimator result."""

    transform: ColorTransform
    behavior: ColorBehavior
    reason: ColorReason
    confidence: float
    exposure_confidence: float
    white_balance_confidence: float
    usable_source: int
    usable_target: int
    neutral_source: int
    neutral_target: int
    target_is_local: bool
    reliable: bool
    signature: ColorSceneSignature | None = None

    @property
    def exposure_ev(self) -> float:
        return self.transform.exposure_ev

    @property
    def wb_gains(self) -> tuple[float, float, float]:
        return self.transform.wb_gains

    @classmethod
    def identity(
        cls,
        reason: ColorReason,
        *,
        signature: ColorSceneSignature | None = None,
        usable_source: int = 0,
        usable_target: int = 0,
        target_is_local: bool = False,
    ) -> ColorEstimate:
        return cls(
            transform=IDENTITY_TRANSFORM,
            behavior=ColorBehavior.IDENTITY,
            reason=reason,
            confidence=0.0,
            exposure_confidence=0.0,
            white_balance_confidence=0.0,
            usable_source=usable_source,
            usable_target=usable_target,
            neutral_source=0,
            neutral_target=0,
            target_is_local=target_is_local,
            reliable=False,
            signature=signature,
        )


def _normalize_rect(
    rect: ContentRect, width: int, height: int
) -> tuple[int, int, int, int]:
    if rect is None:
        return (0, 0, width, height)
    if isinstance(rect, Rect):
        values = (rect.left, rect.top, rect.right, rect.bottom)
    elif isinstance(rect, tuple) and len(rect) == 4:
        values = rect
    else:
        raise ColorError("content rectangle must contain four integers")
    if any(type(value) is not int for value in values):
        raise ColorError("content rectangle must contain four integers")
    left, top, right, bottom = values
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ColorError("content rectangle is outside the frame")
    return values


def _safe_analysis_rect_mask(
    rect: tuple[int, int, int, int],
    source_size: tuple[int, int],
    analysis_size: tuple[int, int],
) -> np.ndarray:
    """Mark analysis bins whose complete area lies inside valid source content."""

    source_width, source_height = source_size
    analysis_width, analysis_height = analysis_size
    left, top, right, bottom = rect
    x = np.arange(analysis_width, dtype=np.int64)
    y = np.arange(analysis_height, dtype=np.int64)
    valid_x = (x * source_width >= left * analysis_width) & (
        (x + 1) * source_width <= right * analysis_width
    )
    valid_y = (y * source_height >= top * analysis_height) & (
        (y + 1) * source_height <= bottom * analysis_height
    )
    return np.ascontiguousarray(valid_y[:, None] & valid_x[None, :])


def _analysis_size(width: int, height: int) -> tuple[int, int]:
    if max(width, height) <= ANALYSIS_LONG_EDGE:
        return width, height
    scale = ANALYSIS_LONG_EDGE / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _linear_bgr_analysis_raster_prevalidated(
    linear_bgr: np.ndarray,
) -> np.ndarray:
    """Build one bounded AREA-resampled raster from trusted linear BGR."""

    if (
        not isinstance(linear_bgr, np.ndarray)
        or linear_bgr.dtype != np.float32
        or linear_bgr.ndim != 3
        or linear_bgr.shape[2] != 3
        or linear_bgr.size == 0
        or not linear_bgr.flags.c_contiguous
    ):
        raise ColorError(
            "linear BGR violates the prevalidated contiguous float32 contract"
        )
    height, width = linear_bgr.shape[:2]
    size = _analysis_size(width, height)
    if size == (width, height):
        return linear_bgr.copy()
    if cv2 is None:
        raise ColorError("opencv-python is required for color estimation")
    return np.ascontiguousarray(
        cv2.resize(linear_bgr, size, interpolation=cv2.INTER_AREA),
        dtype=np.float32,
    )


def _resize_linear_and_mask(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    *,
    executor: Executor | None = None,
    backdrop_analysis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    size = _analysis_size(width, height)
    if backdrop_analysis is not None:
        expected_shape = (size[1], size[0], 3)
        if (
            not isinstance(backdrop_analysis, np.ndarray)
            or backdrop_analysis.dtype != np.float32
            or backdrop_analysis.shape != expected_shape
            or not backdrop_analysis.flags.c_contiguous
        ):
            raise ColorError(
                "cached backdrop analysis violates the bounded float32 contract"
            )
    if size == (width, height):
        return (
            foreground.copy(),
            backdrop.copy() if backdrop_analysis is None else backdrop_analysis,
            mask.copy(),
        )
    if cv2 is None:
        raise ColorError("opencv-python is required for color estimation")
    if executor is not None:
        values = (
            (foreground, mask)
            if backdrop_analysis is not None
            else (foreground, backdrop, mask)
        )
        futures = tuple(
            executor.submit(
                cv2.resize,
                value,
                size,
                interpolation=cv2.INTER_AREA,
            )
            for value in values
        )
        resized = tuple(future.result() for future in futures)
        if backdrop_analysis is not None:
            return (
                np.ascontiguousarray(resized[0], dtype=np.float32),
                backdrop_analysis,
                np.ascontiguousarray(resized[1], dtype=np.float32),
            )
        return (
            np.ascontiguousarray(resized[0], dtype=np.float32),
            np.ascontiguousarray(resized[1], dtype=np.float32),
            np.ascontiguousarray(resized[2], dtype=np.float32),
        )
    return (
        np.ascontiguousarray(
            cv2.resize(foreground, size, interpolation=cv2.INTER_AREA),
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            (
                cv2.resize(backdrop, size, interpolation=cv2.INTER_AREA)
                if backdrop_analysis is None
                else backdrop_analysis
            ),
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            cv2.resize(mask, size, interpolation=cv2.INTER_AREA),
            dtype=np.float32,
        ),
    )


def _valid_luminance(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    luminance = linear_luminance(pixels)
    valid = (
        (luminance > NEAR_BLACK)
        & (np.max(pixels, axis=1) < NEAR_CLIP)
        & np.isfinite(pixels).all(axis=1)
    )
    return np.ascontiguousarray(valid), luminance


def _neutral_pixels(pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    maximum = np.max(pixels, axis=1)
    minimum = np.min(pixels, axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    return np.ascontiguousarray(valid & (saturation <= NEUTRAL_SATURATION_MAX))


def _centered_log_chroma(pixels: np.ndarray) -> tuple[float, float, float]:
    median = np.maximum(np.median(pixels, axis=0), 1e-6)
    logs = np.log2(median)
    logs -= np.mean(logs)
    return float(logs[0]), float(logs[1]), float(logs[2])


def _scene_signature(
    source: np.ndarray,
    target: np.ndarray,
    source_valid: np.ndarray,
    target_valid: np.ndarray,
    source_luminance: np.ndarray,
    target_luminance: np.ndarray,
) -> ColorSceneSignature:
    return ColorSceneSignature(
        source_log_luminance=float(np.median(np.log2(source_luminance[source_valid]))),
        target_log_luminance=float(np.median(np.log2(target_luminance[target_valid]))),
        source_chroma_log2=_centered_log_chroma(source[source_valid]),
        target_chroma_log2=_centered_log_chroma(target[target_valid]),
    )


def _is_solid_saturated(pixels: np.ndarray, neutral_count: int) -> bool:
    if neutral_count >= MIN_SAMPLES:
        return False
    maximum = np.max(pixels, axis=1)
    minimum = np.min(pixels, axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    spread = np.quantile(pixels, 0.90, axis=0) - np.quantile(pixels, 0.10, axis=0)
    return float(np.median(saturation)) > NEUTRAL_SATURATION_MAX and bool(
        np.max(spread) <= 0.02
    )


_KNOWN_BACKGROUND_MODES = {
    "passthrough",
    "blur",
    "color",
    "image",
    "video",
    "camera",
    "remote",
}
_ELIGIBLE_BACKGROUND_MODES = {"image", "video", "camera"}


def _estimate_color_transform(
    foreground_input: np.ndarray,
    backdrop_input: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str,
    strength: float = 0.5,
    exposure_limit_ev: float = 0.85,
    white_balance_strength: float = 0.5,
    foreground_content_rect: ContentRect = None,
    backdrop_content_rect: ContentRect = None,
    inputs_are_linear: bool = False,
    inputs_are_prevalidated_linear_bgr: bool = False,
    resize_executor: Executor | None = None,
    backdrop_analysis_linear_bgr: np.ndarray | None = None,
) -> ColorEstimate:
    try:
        if inputs_are_linear and inputs_are_prevalidated_linear_bgr:
            raise ColorError("linear input channel order is ambiguous")
        if inputs_are_prevalidated_linear_bgr:
            foreground_linear_bgr = foreground_input
            backdrop_linear_bgr = backdrop_input
            for value, name in (
                (foreground_linear_bgr, "foreground linear BGR"),
                (backdrop_linear_bgr, "backdrop linear BGR"),
            ):
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.float32
                    or value.ndim != 3
                    or value.shape[2] != 3
                    or value.size == 0
                    or not value.flags.c_contiguous
                ):
                    raise ColorError(
                        f"{name} violates the prevalidated float32 contract"
                    )
            frame_shape = foreground_linear_bgr.shape
            if frame_shape != backdrop_linear_bgr.shape:
                raise ColorError("foreground and backdrop shapes differ")
            foreground_linear = backdrop_linear = None
        elif inputs_are_linear:
            foreground_linear = np.asarray(foreground_input)
            backdrop_linear = np.asarray(backdrop_input)
            for value, name in (
                (foreground_linear, "foreground linear RGB"),
                (backdrop_linear, "backdrop linear RGB"),
            ):
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.float32
                    or value.ndim != 3
                    or value.shape[2] != 3
                    or value.size == 0
                    or not value.flags.c_contiguous
                    or not bool(np.isfinite(value).all())
                    or float(np.min(value)) < 0.0
                    or float(np.max(value)) > 1.0
                ):
                    raise ColorError(
                        f"{name} violates the finite contiguous float32 contract"
                    )
            frame_shape = foreground_linear.shape
            if frame_shape != backdrop_linear.shape:
                raise ColorError("foreground and backdrop shapes differ")
        else:
            foreground = validate_bgr_frame(
                foreground_input,
                name="foreground",
                require_contiguous=True,
            )
            backdrop = validate_bgr_frame(
                backdrop_input,
                name="backdrop",
                require_contiguous=True,
            )
            foreground_linear = backdrop_linear = None
            foreground_linear_bgr = backdrop_linear_bgr = None
            frame_shape = foreground.shape
            if frame_shape != backdrop.shape:
                raise ColorError("foreground and backdrop shapes differ")
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.float32
            or mask.ndim != 2
            or mask.shape != frame_shape[:2]
            or not mask.flags.c_contiguous
            or (
                not inputs_are_prevalidated_linear_bgr
                and (
                    not bool(np.isfinite(mask).all())
                    or float(np.min(mask)) < 0.0
                    or float(np.max(mask)) > 1.0
                )
            )
        ):
            raise ColorError("mask violates the finite contiguous float32 contract")
        if mode not in _KNOWN_BACKGROUND_MODES:
            raise ColorError("unknown background mode")
        for value, name in (
            (strength, "strength"),
            (white_balance_strength, "white_balance_strength"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ColorError(f"{name} must be in [0, 1]")
        if (
            isinstance(exposure_limit_ev, bool)
            or not isinstance(exposure_limit_ev, (int, float))
            or not math.isfinite(float(exposure_limit_ev))
            or not 0.0 <= float(exposure_limit_ev) <= MAX_EXPOSURE_EV
        ):
            raise ColorError("exposure_limit_ev must be in [0, 1]")
        height, width = mask.shape
        source_rect = _normalize_rect(foreground_content_rect, width, height)
        target_rect = _normalize_rect(backdrop_content_rect, width, height)
    except (ColorError, ValueError, TypeError, OverflowError):
        return ColorEstimate.identity(ColorReason.INVALID)

    if mode not in _ELIGIBLE_BACKGROUND_MODES:
        return ColorEstimate.identity(ColorReason.MODE_EXCLUDED)

    try:
        # Decode before area resampling.  This intentionally matches the Phase-0
        # estimator evidence instead of averaging gamma-encoded code values.
        if inputs_are_prevalidated_linear_bgr:
            assert foreground_linear_bgr is not None
            assert backdrop_linear_bgr is not None
            small_fg_bgr, small_bg_bgr, small_mask = _resize_linear_and_mask(
                foreground_linear_bgr,
                backdrop_linear_bgr,
                mask,
                executor=resize_executor,
                backdrop_analysis=backdrop_analysis_linear_bgr,
            )
            # Channel order matters only to RGB luminance/WB analysis.  Reverse
            # the bounded 192-pixel analysis rasters, never the full frames.
            small_fg = np.ascontiguousarray(small_fg_bgr[..., ::-1])
            small_bg = np.ascontiguousarray(small_bg_bgr[..., ::-1])
        else:
            if not inputs_are_linear:
                foreground_linear = bgr_u8_to_linear_rgb(foreground)
                backdrop_linear = bgr_u8_to_linear_rgb(backdrop)
            assert foreground_linear is not None
            assert backdrop_linear is not None
            small_fg, small_bg, small_mask = _resize_linear_and_mask(
                foreground_linear,
                backdrop_linear,
                mask,
            )
        analysis_height, analysis_width = small_mask.shape
        source_content = _safe_analysis_rect_mask(
            source_rect,
            (width, height),
            (analysis_width, analysis_height),
        )
        target_content = _safe_analysis_rect_mask(
            target_rect,
            (width, height),
            (analysis_width, analysis_height),
        )

        if cv2 is None:
            raise ColorError("opencv-python is required for color estimation")
        hard_core = ((small_mask >= MASK_CORE_THRESHOLD) & source_content).astype(
            np.uint8
        )
        core = cv2.erode(
            hard_core,
            np.ones((MASK_EROSION_SIZE, MASK_EROSION_SIZE), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        support = ((small_mask > MASK_SUPPORT_THRESHOLD) & source_content).astype(
            np.uint8
        )
        dilated = cv2.dilate(
            support,
            np.ones((TARGET_DILATION_SIZE, TARGET_DILATION_SIZE), dtype=np.uint8),
            iterations=1,
        )
        local_target = (dilated > 0) & (support == 0) & target_content
        target_is_local = int(local_target.sum()) >= MIN_SAMPLES
        target_region = local_target if target_is_local else target_content
        source_content_count = int(source_content.sum())
        mask_coverage = float(
            np.count_nonzero((small_mask >= MASK_CORE_THRESHOLD) & source_content)
            / max(1, source_content_count)
        )

        source = small_fg[core]
        target = small_bg[target_region]
        if (
            len(source) < MIN_SAMPLES
            or mask_coverage < MIN_MASK_COVERAGE
            or source_content_count < MIN_SAMPLES
        ):
            return ColorEstimate.identity(
                ColorReason.INSUFFICIENT_MASK,
                target_is_local=target_is_local,
            )
        if len(target) < MIN_SAMPLES:
            return ColorEstimate.identity(
                ColorReason.INVALID,
                target_is_local=target_is_local,
            )

        source_valid, source_luminance = _valid_luminance(source)
        target_valid, target_luminance = _valid_luminance(target)
        usable_source = int(source_valid.sum())
        usable_target = int(target_valid.sum())
        if usable_source < MIN_SAMPLES or usable_target < MIN_SAMPLES:
            return ColorEstimate.identity(
                ColorReason.CLIPPED,
                usable_source=usable_source,
                usable_target=usable_target,
                target_is_local=target_is_local,
            )

        signature = _scene_signature(
            source,
            target,
            source_valid,
            target_valid,
            source_luminance,
            target_luminance,
        )
        raw_ev = signature.target_log_luminance - signature.source_log_luminance
        exposure_transform = bounded_color_transform(
            raw_ev,
            strength=float(strength),
            exposure_limit_ev=float(exposure_limit_ev),
            white_balance_strength=0.0,
        )
        exposure_ev = exposure_transform.exposure_ev

        count_score = min(1.0, min(usable_source, usable_target) / 512.0)
        valid_score = min(
            usable_source / max(1, len(source)),
            usable_target / max(1, len(target)),
        )
        coverage_score = min(1.0, mask_coverage / 0.10)
        exposure_confidence = float(count_score * valid_score * coverage_score)
        if exposure_confidence < MIN_CONFIDENCE:
            return ColorEstimate.identity(
                ColorReason.INSUFFICIENT_MASK,
                signature=signature,
                usable_source=usable_source,
                usable_target=usable_target,
                target_is_local=target_is_local,
            )

        source_neutral = _neutral_pixels(source, source_valid)
        target_neutral = _neutral_pixels(target, target_valid)
        neutral_source = int(source_neutral.sum())
        neutral_target = int(target_neutral.sum())
        neutral_availability = (
            min(1.0, min(neutral_source, neutral_target) / MIN_SAMPLES)
            if target_is_local
            else 0.0
        )
        wb_confidence = float(exposure_confidence * neutral_availability)
        can_adapt_wb = (
            target_is_local
            and neutral_source >= MIN_SAMPLES
            and neutral_target >= MIN_SAMPLES
            and wb_confidence >= MIN_CONFIDENCE
        )
        if not can_adapt_wb:
            reason = (
                ColorReason.SOLID_SATURATED
                if _is_solid_saturated(target[target_valid], neutral_target)
                else ColorReason.INSUFFICIENT_NEUTRAL
            )
            return ColorEstimate(
                transform=ColorTransform(exposure_ev),
                behavior=ColorBehavior.EXPOSURE_ONLY,
                reason=reason,
                confidence=exposure_confidence,
                exposure_confidence=exposure_confidence,
                white_balance_confidence=wb_confidence,
                usable_source=usable_source,
                usable_target=usable_target,
                neutral_source=neutral_source,
                neutral_target=neutral_target,
                target_is_local=target_is_local,
                reliable=True,
                signature=signature,
            )

        source_rgb = np.median(source[source_neutral], axis=0)
        target_rgb = np.median(target[target_neutral], axis=0)
        raw_gains = target_rgb / np.maximum(source_rgb, 1e-6)
        transform = bounded_color_transform(
            raw_ev,
            (
                float(raw_gains[0]),
                float(raw_gains[1]),
                float(raw_gains[2]),
            ),
            strength=float(strength),
            exposure_limit_ev=float(exposure_limit_ev),
            white_balance_strength=float(white_balance_strength),
        )
        return ColorEstimate(
            transform=transform,
            behavior=ColorBehavior.EXPOSURE_WHITE_BALANCE,
            reason=ColorReason.OK,
            confidence=wb_confidence,
            exposure_confidence=exposure_confidence,
            white_balance_confidence=wb_confidence,
            usable_source=usable_source,
            usable_target=usable_target,
            neutral_source=neutral_source,
            neutral_target=neutral_target,
            target_is_local=True,
            reliable=True,
            signature=signature,
        )
    except (ColorError, ValueError, TypeError, OverflowError, FloatingPointError):
        return ColorEstimate.identity(ColorReason.INVALID)


def estimate_color_transform(
    foreground_bgr: np.ndarray,
    backdrop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str,
    strength: float = 0.5,
    exposure_limit_ev: float = 0.85,
    white_balance_strength: float = 0.5,
    foreground_content_rect: ContentRect = None,
    backdrop_content_rect: ContentRect = None,
) -> ColorEstimate:
    """Estimate from external BGR inputs, decoding each full frame once."""

    return _estimate_color_transform(
        foreground_bgr,
        backdrop_bgr,
        mask,
        mode=mode,
        strength=strength,
        exposure_limit_ev=exposure_limit_ev,
        white_balance_strength=white_balance_strength,
        foreground_content_rect=foreground_content_rect,
        backdrop_content_rect=backdrop_content_rect,
    )


def estimate_color_transform_linear(
    foreground_linear_rgb: np.ndarray,
    backdrop_linear_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str,
    strength: float = 0.5,
    exposure_limit_ev: float = 0.85,
    white_balance_strength: float = 0.5,
    foreground_content_rect: ContentRect = None,
    backdrop_content_rect: ContentRect = None,
) -> ColorEstimate:
    """Estimate from already-decoded full-frame linear RGB.

    The linear compositor can pass its once-decoded inputs here, avoiding a
    second full-frame EOTF while retaining the Phase-0 decode-before-resize
    analysis order.
    """

    return _estimate_color_transform(
        foreground_linear_rgb,
        backdrop_linear_rgb,
        mask,
        mode=mode,
        strength=strength,
        exposure_limit_ev=exposure_limit_ev,
        white_balance_strength=white_balance_strength,
        foreground_content_rect=foreground_content_rect,
        backdrop_content_rect=backdrop_content_rect,
        inputs_are_linear=True,
    )


def _estimate_color_transform_linear_bgr_prevalidated(
    foreground_linear_bgr: np.ndarray,
    backdrop_linear_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str,
    strength: float = 0.5,
    exposure_limit_ev: float = 0.85,
    white_balance_strength: float = 0.5,
    foreground_content_rect: ContentRect = None,
    backdrop_content_rect: ContentRect = None,
    resize_executor: Executor | None = None,
    backdrop_analysis_linear_bgr: np.ndarray | None = None,
) -> ColorEstimate:
    """Estimate from pipeline-validated linear BGR without full-frame rescans.

    The pipeline has already validated both canonical frames and the refined
    mask.  Analysis is area-downsampled in linear light before the small rasters
    are converted to the public estimator's RGB channel order.
    """

    return _estimate_color_transform(
        foreground_linear_bgr,
        backdrop_linear_bgr,
        mask,
        mode=mode,
        strength=strength,
        exposure_limit_ev=exposure_limit_ev,
        white_balance_strength=white_balance_strength,
        foreground_content_rect=foreground_content_rect,
        backdrop_content_rect=backdrop_content_rect,
        inputs_are_prevalidated_linear_bgr=True,
        resize_executor=resize_executor,
        backdrop_analysis_linear_bgr=backdrop_analysis_linear_bgr,
    )


def _image_format(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ColorError("expected image format must be non-empty")
    normalized = value.strip().upper()
    return "JPEG" if normalized in {"JPG", "JPEG"} else normalized


def _file_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def decode_image_to_srgb_bgr(
    path: str | os.PathLike[str],
    expected_format: str,
    max_pixels: int,
) -> np.ndarray:
    """Securely decode one still image through a single caller-owned descriptor.

    Tagged pixels are converted through LittleCMS to sRGB.  Untagged pixels use
    the documented sRGB assumption.  A malformed embedded profile is rejected
    rather than silently ignored.
    """

    if Image is None or ImageCms is None:
        raise ColorError("Pillow with ImageCms is required for image decoding")
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ColorError("max_pixels must be a positive integer")
    expected = _image_format(expected_format)
    try:
        source = open(Path(path), "rb")  # noqa: SIM115 - one owned descriptor
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ColorError("cannot open image") from exc

    try:
        with source:
            initial_identity = _file_identity(os.fstat(source.fileno()))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(source) as candidate:
                    if candidate.format != expected:
                        raise ColorError(
                            f"image format does not match expected {expected}"
                        )
                    width, height = candidate.size
                    if width <= 0 or height <= 0:
                        raise ColorError("image dimensions must be positive")
                    if width * height > max_pixels:
                        raise ColorError(f"image exceeds {max_pixels} pixels")
                    candidate.verify()

                if _file_identity(os.fstat(source.fileno())) != initial_identity:
                    raise ColorError("image changed during validation")
                source.seek(0)
                with Image.open(source) as decoded:
                    if decoded.format != expected or decoded.size != (width, height):
                        raise ColorError("image changed during validation")
                    orientation_value = decoded.getexif().get(274, 1)
                    if (
                        type(orientation_value) is not int
                        or not 1 <= orientation_value <= 8
                    ):
                        raise ColorError("invalid EXIF orientation")
                    decoded.load()
                    if _file_identity(os.fstat(source.fileno())) != initial_identity:
                        raise ColorError("image changed during validation")
                    if "icc_profile" in decoded.info:
                        profile_bytes = decoded.info["icc_profile"]
                        if (
                            not isinstance(profile_bytes, (bytes, bytearray))
                            or len(profile_bytes) == 0
                        ):
                            raise ColorError("invalid embedded ICC profile")
                        try:
                            source_profile = ImageCms.ImageCmsProfile(
                                io.BytesIO(profile_bytes)
                            )
                            destination_profile = ImageCms.createProfile("sRGB")
                            rgb_image = ImageCms.profileToProfile(
                                decoded,
                                source_profile,
                                destination_profile,
                                outputMode="RGB",
                            )
                            if rgb_image is None:
                                raise ColorError("invalid embedded ICC profile")
                        except Exception as exc:
                            if isinstance(exc, ColorError):
                                raise
                            raise ColorError("invalid embedded ICC profile") from exc
                    else:
                        rgb_image = decoded.convert("RGB")
                    rgb = np.asarray(rgb_image, dtype=np.uint8)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ColorError(f"image exceeds {max_pixels} pixels") from exc
    except ColorError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        SyntaxError,
        Warning,
    ) as exc:
        raise ColorError("invalid image") from exc

    if rgb.shape != (height, width, 3):
        raise ColorError("decoded image is not RGB")
    try:
        oriented = apply_exif_orientation(
            np.ascontiguousarray(rgb),
            orientation_value,
        )
    except ValueError as exc:
        raise ColorError("invalid EXIF orientation") from exc
    if oriented.shape[0] * oriented.shape[1] > max_pixels:
        raise ColorError(f"image exceeds {max_pixels} pixels")
    return np.ascontiguousarray(oriented[..., ::-1])


@dataclass(frozen=True)
class HarmonizerSnapshot:
    """Immutable scalar-only view of temporal state."""

    transform: ColorTransform
    phase: HarmonizerPhase
    reason: ColorReason
    confidence: float
    reliable: bool
    last_timestamp_s: float | None
    last_reliable_s: float | None
    low_confidence_since_s: float | None
    fast_until_s: float | None
    source_generation: int | None
    signature: ColorSceneSignature | None


def _finite_time(now_s: float) -> float:
    if (
        isinstance(now_s, bool)
        or not isinstance(now_s, (int, float))
        or not math.isfinite(float(now_s))
    ):
        raise ColorError("now_s must be a finite monotonic timestamp")
    return float(now_s)


def _scene_cut(
    previous: ColorSceneSignature,
    current: ColorSceneSignature,
) -> bool:
    if (
        abs(current.source_log_luminance - previous.source_log_luminance)
        >= SCENE_CUT_LUMINANCE_EV
        or abs(current.target_log_luminance - previous.target_log_luminance)
        >= SCENE_CUT_LUMINANCE_EV
    ):
        return True
    source_delta = max(
        abs(current_value - previous_value)
        for current_value, previous_value in zip(
            current.source_chroma_log2,
            previous.source_chroma_log2,
            strict=True,
        )
    )
    target_delta = max(
        abs(current_value - previous_value)
        for current_value, previous_value in zip(
            current.target_chroma_log2,
            previous.target_chroma_log2,
            strict=True,
        )
    )
    return max(source_delta, target_delta) >= SCENE_CUT_CHROMA_LOG2


def _fast_tau(adaptation_time_s: float) -> float:
    return max(0.05, min(0.20, adaptation_time_s / 4.0))


def _bounded_step(
    current: float,
    target: float,
    *,
    dt: float,
    tau: float,
    deadband: float,
    slew_per_s: float,
) -> float:
    delta = target - current
    if abs(delta) <= deadband or dt <= 0.0:
        return current
    alpha = -math.expm1(-dt / tau)
    requested = delta * alpha
    limit = slew_per_s * dt
    return current + min(limit, max(-limit, requested))


class ColorHarmonizer:
    """Elapsed-time color state with cut, freeze, decay, and reset behavior."""

    def __init__(
        self,
        adaptation_time_s: float = 0.8,
        *,
        mode: str = "image",
    ) -> None:
        if (
            isinstance(adaptation_time_s, bool)
            or not isinstance(adaptation_time_s, (int, float))
            or not math.isfinite(float(adaptation_time_s))
            or not 0.05 <= float(adaptation_time_s) <= 10.0
        ):
            raise ColorError("adaptation_time_s must be in [0.05, 10]")
        if mode not in _KNOWN_BACKGROUND_MODES:
            raise ColorError("unknown background mode")
        self.adaptation_time_s = float(adaptation_time_s)
        self.mode = mode
        self._exposure_ev = 0.0
        self._wb_log2 = (0.0, 0.0, 0.0)
        self._phase = HarmonizerPhase.IDENTITY
        self._reason = ColorReason.INSUFFICIENT_MASK
        self._confidence = 0.0
        self._reliable = False
        self._last_timestamp_s: float | None = None
        self._last_reliable_s: float | None = None
        self._low_confidence_since_s: float | None = None
        self._fast_until_s: float | None = None
        self._source_generation: int | None = None
        self._signature: ColorSceneSignature | None = None

    @property
    def transform(self) -> ColorTransform:
        wb_gains = tuple(
            float(np.clip(2.0**value, WB_GAIN_MIN, WB_GAIN_MAX))
            for value in self._wb_log2
        )
        return ColorTransform(
            exposure_ev=float(
                np.clip(self._exposure_ev, -MAX_EXPOSURE_EV, MAX_EXPOSURE_EV)
            ),
            wb_gains=(wb_gains[0], wb_gains[1], wb_gains[2]),
        )

    def snapshot(self) -> HarmonizerSnapshot:
        return HarmonizerSnapshot(
            transform=self.transform,
            phase=self._phase,
            reason=self._reason,
            confidence=self._confidence,
            reliable=self._reliable,
            last_timestamp_s=self._last_timestamp_s,
            last_reliable_s=self._last_reliable_s,
            low_confidence_since_s=self._low_confidence_since_s,
            fast_until_s=self._fast_until_s,
            source_generation=self._source_generation,
            signature=self._signature,
        )

    def clone(self) -> ColorHarmonizer:
        cloned = ColorHarmonizer(self.adaptation_time_s, mode=self.mode)
        cloned._exposure_ev = self._exposure_ev
        cloned._wb_log2 = self._wb_log2
        cloned._phase = self._phase
        cloned._reason = self._reason
        cloned._confidence = self._confidence
        cloned._reliable = self._reliable
        cloned._last_timestamp_s = self._last_timestamp_s
        cloned._last_reliable_s = self._last_reliable_s
        cloned._low_confidence_since_s = self._low_confidence_since_s
        cloned._fast_until_s = self._fast_until_s
        cloned._source_generation = self._source_generation
        cloned._signature = self._signature
        return cloned

    def reset(
        self,
        now_s: float,
        *,
        reason: ColorReason = ColorReason.INVALID,
        source_generation: int | None = None,
    ) -> ColorTransform:
        now = _finite_time(now_s)
        if source_generation is not None and (
            type(source_generation) is not int or source_generation < 0
        ):
            raise ColorError("source_generation must be a non-negative integer")
        self._exposure_ev = 0.0
        self._wb_log2 = (0.0, 0.0, 0.0)
        self._phase = HarmonizerPhase.IDENTITY
        self._reason = reason
        self._confidence = 0.0
        self._reliable = False
        self._last_timestamp_s = now
        self._last_reliable_s = None
        self._low_confidence_since_s = None
        self._fast_until_s = None
        self._source_generation = source_generation
        self._signature = None
        return self.transform

    def reset_and_update(
        self,
        estimate: ColorEstimate,
        now_s: float,
        *,
        source_generation: int | None = None,
    ) -> ColorTransform:
        """Hard-reset, then consume one estimate at that reset boundary.

        A normal repeated call at the same timestamp is intentionally
        idempotent.  A hard reset is different: its first reliable estimate
        must seed the replacement generation while still leaving the applied
        transform at identity.
        """

        if not isinstance(estimate, ColorEstimate):
            raise ColorError("estimate must be a ColorEstimate")
        now = _finite_time(now_s)
        self.reset(
            now,
            reason=ColorReason.INVALID,
            source_generation=source_generation,
        )
        # The reset timestamp guards callers from moving time backwards.
        # Clear it only inside this atomic operation so update() can seed the
        # replacement state at the same timestamp.
        self._last_timestamp_s = None
        return self.update(
            estimate,
            now,
            source_generation=source_generation,
        )

    def _install_transform(
        self, exposure_ev: float, wb_log2: tuple[float, ...]
    ) -> None:
        self._exposure_ev = float(
            np.clip(exposure_ev, -MAX_EXPOSURE_EV, MAX_EXPOSURE_EV)
        )
        lower = math.log2(WB_GAIN_MIN)
        upper = math.log2(WB_GAIN_MAX)
        self._wb_log2 = tuple(float(np.clip(value, lower, upper)) for value in wb_log2)

    def _decay_or_freeze(
        self,
        estimate: ColorEstimate,
        now: float,
        dt: float,
    ) -> ColorTransform:
        self._reason = estimate.reason
        self._confidence = estimate.confidence
        if not self._reliable:
            self._install_transform(0.0, (0.0, 0.0, 0.0))
            self._phase = HarmonizerPhase.IDENTITY
            self._last_timestamp_s = now
            return self.transform
        if self._low_confidence_since_s is None:
            self._low_confidence_since_s = now
        elapsed = now - self._low_confidence_since_s
        if elapsed <= LOW_CONFIDENCE_FREEZE_S:
            self._phase = HarmonizerPhase.FROZEN
            self._last_timestamp_s = now
            return self.transform
        if elapsed >= STALE_CLEAR_S:
            self._install_transform(0.0, (0.0, 0.0, 0.0))
            self._phase = HarmonizerPhase.IDENTITY
            self._reliable = False
            self._last_reliable_s = None
            self._signature = None
            self._fast_until_s = None
            self._last_timestamp_s = now
            return self.transform
        # A sparse update may straddle the freeze boundary.  Integrate only
        # the portion after that boundary so repeated-output/source gaps do
        # not make the frozen interval count as stale decay.
        decay_dt = min(dt, max(0.0, elapsed - LOW_CONFIDENCE_FREEZE_S))
        alpha = -math.expm1(-decay_dt / STALE_DECAY_TAU_S) if decay_dt > 0.0 else 0.0
        self._install_transform(
            self._exposure_ev * (1.0 - alpha),
            tuple(value * (1.0 - alpha) for value in self._wb_log2),
        )
        self._phase = HarmonizerPhase.STALE_DECAY
        self._last_timestamp_s = now
        return self.transform

    def update(
        self,
        estimate: ColorEstimate,
        now_s: float,
        *,
        source_generation: int | None = None,
    ) -> ColorTransform:
        if not isinstance(estimate, ColorEstimate):
            raise ColorError("estimate must be a ColorEstimate")
        now = _finite_time(now_s)
        if source_generation is not None and (
            type(source_generation) is not int or source_generation < 0
        ):
            raise ColorError("source_generation must be a non-negative integer")
        if self._last_timestamp_s is not None and now < self._last_timestamp_s:
            raise ColorError("now_s moved backwards")
        if (
            source_generation is not None
            and self._source_generation is not None
            and source_generation != self._source_generation
        ):
            return self.reset_and_update(
                estimate,
                now,
                source_generation=source_generation,
            )
        if source_generation is not None:
            self._source_generation = source_generation
        if self._last_timestamp_s is not None and now == self._last_timestamp_s:
            return self.transform

        dt = 0.0 if self._last_timestamp_s is None else now - self._last_timestamp_s
        if dt >= STALE_CLEAR_S and self._last_timestamp_s is not None:
            # A wall-clock gap is not called a reconnect.  It does, however,
            # prohibit one giant EMA step with stale state.
            self._install_transform(0.0, (0.0, 0.0, 0.0))
            self._reliable = False
            self._last_reliable_s = None
            self._low_confidence_since_s = None
            self._fast_until_s = None
            self._signature = None
            dt = 0.0

        if not estimate.reliable:
            return self._decay_or_freeze(estimate, now, dt)

        if (
            self._signature is not None
            and estimate.signature is not None
            and _scene_cut(self._signature, estimate.signature)
        ):
            # The cut-frame estimate is intentionally discarded.
            self._signature = estimate.signature
            self._phase = HarmonizerPhase.SCENE_CUT
            self._reason = estimate.reason
            self._confidence = estimate.confidence
            self._low_confidence_since_s = None
            self._last_reliable_s = now
            self._fast_until_s = now + FAST_ACQUISITION_S
            self._last_timestamp_s = now
            return self.transform

        if not self._reliable:
            # A hard reset or startup begins at identity.  The first reliable
            # frame seeds post-reset statistics but cannot flash a new transform.
            self._reliable = True
            self._signature = estimate.signature
            self._phase = HarmonizerPhase.WARMING
            self._reason = estimate.reason
            self._confidence = estimate.confidence
            self._low_confidence_since_s = None
            self._last_reliable_s = now
            self._fast_until_s = now + FAST_ACQUISITION_S
            self._last_timestamp_s = now
            return self.transform

        fast = self._fast_until_s is not None and now <= self._fast_until_s
        if fast:
            tau = _fast_tau(self.adaptation_time_s)
            exposure_slew = FAST_EXPOSURE_SLEW_EV_S
            wb_slew = FAST_WB_SLEW_LOG2_S
        else:
            tau = self.adaptation_time_s
            exposure_slew = STEADY_EXPOSURE_SLEW_EV_S
            wb_slew = STEADY_WB_SLEW_LOG2_S
            if self.mode == "camera":
                tau *= CAMERA_TAU_MULTIPLIER
                exposure_slew *= CAMERA_SLEW_MULTIPLIER
                wb_slew *= CAMERA_SLEW_MULTIPLIER

        target_logs = tuple(math.log2(value) for value in estimate.transform.wb_gains)
        new_exposure = _bounded_step(
            self._exposure_ev,
            estimate.transform.exposure_ev,
            dt=dt,
            tau=tau,
            deadband=EXPOSURE_DEADBAND_EV,
            slew_per_s=exposure_slew,
        )
        new_logs = tuple(
            _bounded_step(
                current,
                target,
                dt=dt,
                tau=tau,
                deadband=WB_DEADBAND_LOG2,
                slew_per_s=wb_slew,
            )
            for current, target in zip(self._wb_log2, target_logs, strict=True)
        )
        self._install_transform(new_exposure, new_logs)
        self._signature = estimate.signature
        self._phase = HarmonizerPhase.WARMING if fast else HarmonizerPhase.ACTIVE
        self._reason = estimate.reason
        self._confidence = estimate.confidence
        self._low_confidence_since_s = None
        self._last_reliable_s = now
        self._last_timestamp_s = now
        return self.transform

    def on_error(
        self,
        now_s: float,
        *,
        source_generation: int | None = None,
    ) -> ColorTransform:
        """Apply the same bounded policy as an invalid instantaneous estimate."""

        return self.update(
            ColorEstimate.identity(ColorReason.INVALID),
            now_s,
            source_generation=source_generation,
        )
