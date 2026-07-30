"""Canonical frame validation, orientation, and geometry normalization.

Planning is pure and cached by immutable values.  Pixel transforms never retain
live frames and always produce the external-frame contract used by the pipeline:
positive, C-contiguous ``H x W x 3 uint8`` BGR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from typing import Any, Literal

import numpy as np

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover - exercised by optional-dependency installs
    _cv2 = None

cv2: Any = _cv2

FitMode = Literal["cover", "contain", "stretch"]
RightAngleRotation = Literal[0, 90, 180, 270]
Interpolation = Literal["area", "linear"]
Size = tuple[int, int]


class GeometryError(ValueError):
    """A geometry setting or source/target size is invalid."""


class FrameValidationError(GeometryError):
    """A decoded frame violates the external BGR frame contract."""


@dataclass(frozen=True)
class Rect:
    """Half-open integer rectangle ``[left, right) x [top, bottom)``."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class Padding:
    """Opaque padding around valid source content in target coordinates."""

    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0


@dataclass(frozen=True)
class ResizeStep:
    """One direction-homogeneous OpenCV resize operation."""

    target_size: Size
    interpolation: Interpolation


def _validate_size(value: object, name: str) -> Size:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        raise GeometryError(f"{name} must be a (width, height) pair of integers")
    width, height = value
    if width <= 0 or height <= 0:
        raise GeometryError(f"{name} dimensions must be positive")
    return width, height


def _validate_anchor(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GeometryError(f"{name} must be a finite number in [0, 1]")
    anchor = float(value)
    if not math.isfinite(anchor) or not 0.0 <= anchor <= 1.0:
        raise GeometryError(f"{name} must be a finite number in [0, 1]")
    return anchor


@dataclass(frozen=True)
class GeometrySpec:
    """Immutable source-to-canvas geometry request."""

    source_size: Size
    target_size: Size
    rotation: RightAngleRotation = 0
    mirror: bool = False
    fit: FitMode = "cover"
    anchor_x: float = 0.5
    anchor_y: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_size", _validate_size(self.source_size, "source_size")
        )
        object.__setattr__(
            self, "target_size", _validate_size(self.target_size, "target_size")
        )
        if type(self.rotation) is not int or self.rotation not in (0, 90, 180, 270):
            raise GeometryError("rotation must be one of 0, 90, 180, or 270")
        if type(self.mirror) is not bool:
            raise GeometryError("mirror must be a boolean")
        if self.fit not in ("cover", "contain", "stretch"):
            raise GeometryError("fit must be one of cover, contain, or stretch")
        object.__setattr__(
            self, "anchor_x", _validate_anchor(self.anchor_x, "anchor_x")
        )
        object.__setattr__(
            self, "anchor_y", _validate_anchor(self.anchor_y, "anchor_y")
        )


@dataclass(frozen=True)
class TransformPlan:
    """Complete immutable transform telemetry and executable resize plan."""

    spec: GeometrySpec
    oriented_size: Size
    resized_size: Size
    scale_x: float
    scale_y: float
    crop_rect: Rect
    content_rect: Rect
    padding: Padding
    resize_steps: tuple[ResizeStep, ...]

    @property
    def source_size(self) -> Size:
        return self.spec.source_size

    @property
    def target_size(self) -> Size:
        return self.spec.target_size

    @property
    def fit(self) -> FitMode:
        return self.spec.fit

    @property
    def rotation(self) -> RightAngleRotation:
        return self.spec.rotation

    @property
    def mirror(self) -> bool:
        return self.spec.mirror

    @property
    def pad_rect(self) -> Rect:
        """Target rectangle occupied by valid pixels when padding is present."""

        return self.content_rect

    @property
    def is_exact_noop(self) -> bool:
        return (
            self.spec.rotation == 0
            and not self.spec.mirror
            and self.oriented_size == self.target_size
            and not self.resize_steps
            and self.crop_rect == Rect(0, 0, self.target_size[0], self.target_size[1])
            and self.padding == Padding()
        )

    def telemetry(self) -> dict[str, object]:
        """Return bounded, path-free values suitable for diagnostics."""

        return {
            "source_size": self.source_size,
            "oriented_size": self.oriented_size,
            "target_size": self.target_size,
            "resized_size": self.resized_size,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "crop_rect": (
                self.crop_rect.left,
                self.crop_rect.top,
                self.crop_rect.right,
                self.crop_rect.bottom,
            ),
            "content_rect": (
                self.content_rect.left,
                self.content_rect.top,
                self.content_rect.right,
                self.content_rect.bottom,
            ),
            "padding": (
                self.padding.left,
                self.padding.top,
                self.padding.right,
                self.padding.bottom,
            ),
            "fit": self.fit,
            "rotation": self.rotation,
            "mirror": self.mirror,
            "interpolation": tuple(step.interpolation for step in self.resize_steps),
        }


def _ceil_ratio(value: int, numerator: int, denominator: int) -> int:
    return (value * numerator + denominator - 1) // denominator


def _floor_ratio(value: int, numerator: int, denominator: int) -> int:
    return value * numerator // denominator


def _anchored_offset(anchor: float, excess: int) -> int:
    if excess <= 0:
        return 0
    # YAML decimal anchors such as 0.58 can multiply to one ULP below an
    # integer (0.58 * 50 == 28.999999999999996).  Move only to the next
    # representable value before floor so binary representation noise cannot
    # steal a pixel while genuine fractional offsets keep floor semantics.
    scaled = math.nextafter(anchor * excess, math.inf)
    return max(0, min(excess, math.floor(scaled)))


def _uniform_step(source: Size, target: Size) -> tuple[ResizeStep, ...]:
    if source == target:
        return ()
    source_width, source_height = source
    target_width, target_height = target
    downscale = target_width <= source_width and target_height <= source_height
    interpolation: Interpolation = "area" if downscale else "linear"
    return (ResizeStep(target, interpolation),)


def _stretch_steps(source: Size, target: Size) -> tuple[ResizeStep, ...]:
    source_width, source_height = source
    target_width, target_height = target
    x_direction = (target_width > source_width) - (target_width < source_width)
    y_direction = (target_height > source_height) - (target_height < source_height)
    if x_direction == 0 and y_direction == 0:
        return ()
    if x_direction * y_direction < 0:
        # Shrink first with area interpolation, then grow the orthogonal axis
        # with linear interpolation.  The unchanged axis is not resampled.
        if x_direction < 0:
            intermediate = (target_width, source_height)
        else:
            intermediate = (source_width, target_height)
        return (
            ResizeStep(intermediate, "area"),
            ResizeStep(target, "linear"),
        )
    interpolation: Interpolation = (
        "area" if x_direction <= 0 and y_direction <= 0 else "linear"
    )
    return (ResizeStep(target, interpolation),)


@lru_cache(maxsize=512)
def _plan_cached(spec: GeometrySpec) -> TransformPlan:
    source_width, source_height = spec.source_size
    if spec.rotation in (90, 270):
        oriented = (source_height, source_width)
    else:
        oriented = spec.source_size
    oriented_width, oriented_height = oriented
    target_width, target_height = spec.target_size

    if spec.fit == "stretch":
        resized = spec.target_size
        full = Rect(0, 0, target_width, target_height)
        return TransformPlan(
            spec=spec,
            oriented_size=oriented,
            resized_size=resized,
            scale_x=target_width / oriented_width,
            scale_y=target_height / oriented_height,
            crop_rect=full,
            content_rect=full,
            padding=Padding(),
            resize_steps=_stretch_steps(oriented, resized),
        )

    width_ratio_is_larger = (
        target_width * oriented_height >= target_height * oriented_width
    )
    if spec.fit == "cover":
        if width_ratio_is_larger:
            numerator, denominator = target_width, oriented_width
        else:
            numerator, denominator = target_height, oriented_height
        resized_width = max(
            target_width,
            _ceil_ratio(oriented_width, numerator, denominator),
        )
        resized_height = max(
            target_height,
            _ceil_ratio(oriented_height, numerator, denominator),
        )
        excess_x = resized_width - target_width
        excess_y = resized_height - target_height
        crop_x = _anchored_offset(spec.anchor_x, excess_x)
        crop_y = _anchored_offset(spec.anchor_y, excess_y)
        resized = (resized_width, resized_height)
        return TransformPlan(
            spec=spec,
            oriented_size=oriented,
            resized_size=resized,
            scale_x=resized_width / oriented_width,
            scale_y=resized_height / oriented_height,
            crop_rect=Rect(
                crop_x,
                crop_y,
                crop_x + target_width,
                crop_y + target_height,
            ),
            content_rect=Rect(0, 0, target_width, target_height),
            padding=Padding(),
            resize_steps=_uniform_step(oriented, resized),
        )

    if width_ratio_is_larger:
        numerator, denominator = target_height, oriented_height
    else:
        numerator, denominator = target_width, oriented_width
    resized_width = max(
        1,
        min(
            target_width,
            _floor_ratio(oriented_width, numerator, denominator),
        ),
    )
    resized_height = max(
        1,
        min(
            target_height,
            _floor_ratio(oriented_height, numerator, denominator),
        ),
    )
    padding_x = target_width - resized_width
    padding_y = target_height - resized_height
    left = _anchored_offset(spec.anchor_x, padding_x)
    top = _anchored_offset(spec.anchor_y, padding_y)
    right = padding_x - left
    bottom = padding_y - top
    resized = (resized_width, resized_height)
    return TransformPlan(
        spec=spec,
        oriented_size=oriented,
        resized_size=resized,
        scale_x=resized_width / oriented_width,
        scale_y=resized_height / oriented_height,
        crop_rect=Rect(0, 0, resized_width, resized_height),
        content_rect=Rect(left, top, left + resized_width, top + resized_height),
        padding=Padding(left, top, right, bottom),
        resize_steps=_uniform_step(oriented, resized),
    )


def plan_transform(
    source_size: Size,
    target_size: Size,
    rotation: RightAngleRotation = 0,
    mirror: bool = False,
    fit: FitMode = "cover",
    anchors: tuple[float, float] = (0.5, 0.5),
) -> TransformPlan:
    """Plan orientation and fitting without inspecting or retaining pixels."""

    if not isinstance(anchors, tuple) or len(anchors) != 2:
        raise GeometryError("anchors must be an (anchor_x, anchor_y) pair")
    spec = GeometrySpec(
        source_size=source_size,
        target_size=target_size,
        rotation=rotation,
        mirror=mirror,
        fit=fit,
        anchor_x=anchors[0],
        anchor_y=anchors[1],
    )
    return _plan_cached(spec)


def clear_plan_cache() -> None:
    """Clear immutable plan entries (primarily for bounded test isolation)."""

    _plan_cached.cache_clear()


def plan_cache_info() -> object:
    """Expose cache counters without exposing cached values or live pixels."""

    return _plan_cached.cache_info()


def validate_bgr_frame(
    frame: object,
    *,
    name: str = "frame",
    require_contiguous: bool = False,
) -> np.ndarray:
    """Validate a positive ``HxWx3 uint8`` decoded BGR frame."""

    if not isinstance(frame, np.ndarray):
        raise FrameValidationError(f"{name} must be a numpy array")
    if frame.dtype != np.uint8:
        raise FrameValidationError(f"{name} must have dtype uint8")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise FrameValidationError(f"{name} must have shape HxWx3")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise FrameValidationError(f"{name} dimensions must be positive")
    if require_contiguous and not frame.flags.c_contiguous:
        raise FrameValidationError(f"{name} must be C-contiguous")
    return frame


def apply_exif_orientation(frame: np.ndarray, orientation: int) -> np.ndarray:
    """Apply TIFF/EXIF orientation 1..8 exactly once to decoded pixels."""

    source = validate_bgr_frame(frame, name="EXIF source frame")
    if type(orientation) is not int or orientation not in range(1, 9):
        raise GeometryError("EXIF orientation must be an integer from 1 through 8")
    if orientation == 1:
        oriented = source
    elif orientation == 2:
        oriented = source[:, ::-1]
    elif orientation == 3:
        oriented = source[::-1, ::-1]
    elif orientation == 4:
        oriented = source[::-1, :]
    elif orientation == 5:
        oriented = source.transpose(1, 0, 2)
    elif orientation == 6:
        oriented = np.rot90(source, 3)
    elif orientation == 7:
        oriented = source.transpose(1, 0, 2)[::-1, ::-1]
    else:
        oriented = np.rot90(source, 1)
    return np.ascontiguousarray(oriented)


def orient_frame(
    frame: np.ndarray,
    rotation: RightAngleRotation = 0,
    mirror: bool = False,
) -> np.ndarray:
    """Apply clockwise rotation, then horizontal mirror in viewer coordinates."""

    source = validate_bgr_frame(frame, name="source frame")
    if type(rotation) is not int or rotation not in (0, 90, 180, 270):
        raise GeometryError("rotation must be one of 0, 90, 180, or 270")
    if type(mirror) is not bool:
        raise GeometryError("mirror must be a boolean")
    oriented = source if rotation == 0 else np.rot90(source, -rotation // 90)
    if mirror:
        oriented = oriented[:, ::-1]
    return np.ascontiguousarray(oriented)


def _resize(frame: np.ndarray, step: ResizeStep) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for geometry transforms")
    interpolation = cv2.INTER_AREA if step.interpolation == "area" else cv2.INTER_LINEAR
    return cv2.resize(frame, step.target_size, interpolation=interpolation)


def apply_transform(frame: np.ndarray, plan: TransformPlan) -> np.ndarray:
    """Execute one planned source-to-canvas transform."""

    source = validate_bgr_frame(frame, name="source frame")
    actual_size = (source.shape[1], source.shape[0])
    if actual_size != plan.source_size:
        raise GeometryError(
            "source frame size does not match transform plan: "
            f"expected {plan.source_size[0]}x{plan.source_size[1]}, "
            f"got {actual_size[0]}x{actual_size[1]}"
        )

    transformed = orient_frame(source, plan.rotation, plan.mirror)
    actual_oriented = (transformed.shape[1], transformed.shape[0])
    if actual_oriented != plan.oriented_size:  # pragma: no cover - plan invariant
        raise GeometryError("oriented frame size does not match transform plan")
    for step in plan.resize_steps:
        transformed = _resize(transformed, step)

    if plan.fit == "cover":
        crop = plan.crop_rect
        transformed = transformed[crop.top : crop.bottom, crop.left : crop.right]
    elif plan.fit == "contain" and plan.padding != Padding():
        target_width, target_height = plan.target_size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        content = plan.content_rect
        canvas[content.top : content.bottom, content.left : content.right] = transformed
        transformed = canvas

    target_width, target_height = plan.target_size
    if transformed.shape != (target_height, target_width, 3):
        raise GeometryError(
            "geometry transform produced an invalid target shape: "
            f"{transformed.shape!r}"
        )
    return np.ascontiguousarray(transformed, dtype=np.uint8)


def transform_frame(
    frame: np.ndarray,
    target_size: Size,
    *,
    rotation: RightAngleRotation = 0,
    mirror: bool = False,
    fit: FitMode = "cover",
    anchors: tuple[float, float] = (0.5, 0.5),
) -> tuple[np.ndarray, TransformPlan]:
    """Plan and execute a decoded BGR frame transform."""

    source = validate_bgr_frame(frame, name="source frame")
    plan = plan_transform(
        (source.shape[1], source.shape[0]),
        target_size,
        rotation,
        mirror,
        fit,
        anchors,
    )
    return apply_transform(source, plan), plan
