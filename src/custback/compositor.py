"""Alpha compositing of foreground (person) over a backdrop using the mask."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, MutableMapping

import numpy as np

from .color import (
    ColorError,
    ColorTransform,
    _bgr_u8_to_linear_bgr_prevalidated,
    _consume_linear_bgr_to_bgr_u8_prevalidated,
    apply_color_transform,
    bgr_u8_to_linear_rgb,
    linear_rgb_to_bgr_u8,
)
from .light_wrap import LightWrapFrameContext, LightWrapStabilizer

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

# OpenCV is a compiled optional boundary. Keep its runtime ``None`` fallback
# while treating the dynamically exposed API as opaque to static analysis.
cv2: Any = _cv2

BlendSpace = Literal["srgb_legacy", "linear_srgb"]

# The compositor owns only part of the complete 33.333334 ms presentation
# interval.  This ratified screen leaves at least 11.333334 ms for
# segmentation, backdrop selection, final guard/validation, and sink
# submission.  It is a compositor gate, never a full-path qualification.
COMPOSITOR_P95_SUB_BUDGET_MS = 22.0

COMPOSITOR_SUBSTAGE_NAMES = (
    "input_mask_validation",
    "color_transform_application",
    "edge_band",
    "model_foreground_replacement",
    "backdrop_blur_resize",
    "light_wrap_temporal_filter",
    "light_wrap_interpolation",
    "final_blend_conversion",
    "internal_output_validation",
)


@dataclass(frozen=True)
class PreparedLightWrap:
    """One full-canvas blurred wrap sample in the declared BGR working space."""

    pixels_bgr: np.ndarray
    blend_space: BlendSpace
    stabilized: bool


@dataclass(frozen=True)
class LegacyCompositorWorkspaceSnapshot:
    """Content-free ownership and allocation state for one legacy workspace."""

    shape: tuple[int, int, int]
    retained_bytes: int
    calls: int
    last_known_allocation_bytes: int
    closed: bool


class LegacyCompositorWorkspace:
    """Reusable exact-arithmetic buffers for the encoded legacy compositor.

    The workspace is generation-owned and deliberately not thread safe.  One
    pipeline frame lane may use it serially; outputs always receive independent
    storage and remain valid after the next call.  OpenCV ``multiply``/``add``
    are used instead of ``blendLinear`` because the latter normalizes weights
    and is not byte-exact with the frozen NumPy arithmetic.
    """

    def __init__(self, shape: tuple[int, int, int]) -> None:
        if (
            not isinstance(shape, tuple)
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[2] != 3
        ):
            raise ValueError("legacy compositor workspace shape must be HxWx3")
        height, width, _channels = shape
        self._shape = shape
        self._scalar_weight: np.ndarray | None = np.empty(
            (height, width),
            dtype=np.float32,
        )
        self._scalar_inverse: np.ndarray | None = np.empty(
            (height, width),
            dtype=np.float32,
        )
        self._weight_bgr: np.ndarray | None = np.empty(shape, dtype=np.float32)
        self._working_bgr: np.ndarray | None = np.empty(shape, dtype=np.float32)
        self._transformed_foreground_bgr: np.ndarray | None = np.empty(
            shape,
            dtype=np.uint8,
        )
        self._transformed_edge_bgr: np.ndarray | None = np.empty(
            shape,
            dtype=np.uint8,
        )
        self._color_transform_lut_bgr: np.ndarray | None = np.empty(
            (256, 1, 3),
            dtype=np.uint8,
        )
        self._color_transform_key: tuple[float, float, float, float] | None = None
        self._blur_small_bgr: np.ndarray | None = np.empty(
            (max(4, height // 8), max(4, width // 8), 3),
            dtype=np.uint8,
        )
        self._calls = 0
        self._last_known_allocation_bytes = 0
        self._closed = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def snapshot(self) -> LegacyCompositorWorkspaceSnapshot:
        arrays = (
            self._scalar_weight,
            self._scalar_inverse,
            self._weight_bgr,
            self._working_bgr,
            self._transformed_foreground_bgr,
            self._transformed_edge_bgr,
            self._color_transform_lut_bgr,
            self._blur_small_bgr,
        )
        return LegacyCompositorWorkspaceSnapshot(
            shape=self._shape,
            retained_bytes=sum(
                value.nbytes for value in arrays if isinstance(value, np.ndarray)
            ),
            calls=self._calls,
            last_known_allocation_bytes=self._last_known_allocation_bytes,
            closed=self._closed,
        )

    def close(self) -> None:
        """Release every retained raster; safe to call more than once."""

        self._scalar_weight = None
        self._scalar_inverse = None
        self._weight_bgr = None
        self._working_bgr = None
        self._transformed_foreground_bgr = None
        self._transformed_edge_bgr = None
        self._color_transform_lut_bgr = None
        self._color_transform_key = None
        self._blur_small_bgr = None
        self._last_known_allocation_bytes = 0
        self._closed = True

    def _buffers(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("legacy compositor workspace is closed")
        scalar_weight = self._scalar_weight
        scalar_inverse = self._scalar_inverse
        weight_bgr = self._weight_bgr
        working_bgr = self._working_bgr
        blur_small_bgr = self._blur_small_bgr
        if (
            scalar_weight is None
            or scalar_inverse is None
            or weight_bgr is None
            or working_bgr is None
            or blur_small_bgr is None
        ):  # pragma: no cover - invariant
            raise RuntimeError("legacy compositor workspace is incomplete")
        return (
            scalar_weight,
            scalar_inverse,
            weight_bgr,
            working_bgr,
            blur_small_bgr,
        )

    @staticmethod
    def _elapsed_ms(started_ns: int) -> float:
        return (time.perf_counter_ns() - started_ns) / 1_000_000.0

    @staticmethod
    def _three_channel_weight(
        scalar: np.ndarray,
        destination: np.ndarray,
        *,
        opencv: Any,
    ) -> None:
        opencv.merge((scalar, scalar, scalar), dst=destination)

    def transform_encoded(
        self,
        foreground_bgr: np.ndarray,
        edge_foreground_bgr: np.ndarray | None,
        transform: ColorTransform,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply the diagonal linear-sRGB transform through a retained LUT.

        The transform is separable by channel. Build its 256-entry BGR LUT
        with the deterministic public reference arithmetic, then let OpenCV
        write the full-frame result directly into generation-owned buffers.
        This is exact at alpha endpoints and avoids every transform-sized
        temporary on the steady path.
        """

        if self._closed:
            raise RuntimeError("legacy compositor workspace is closed")
        if foreground_bgr.shape != self._shape or (
            edge_foreground_bgr is not None and edge_foreground_bgr.shape != self._shape
        ):
            raise ValueError("legacy compositor workspace shape mismatch")
        opencv = cv2
        if opencv is None:  # pragma: no cover - package dependency invariant
            raise ColorError("opencv-python is required for compositor workspace")
        transformed_foreground = self._transformed_foreground_bgr
        transformed_edge = self._transformed_edge_bgr
        lut = self._color_transform_lut_bgr
        if transformed_foreground is None or transformed_edge is None or lut is None:
            raise RuntimeError("legacy compositor workspace is incomplete")
        red, green, blue = transform.wb_gains
        key = (
            float(transform.exposure_ev),
            float(red),
            float(green),
            float(blue),
        )
        if self._color_transform_key != key:
            ramp = np.repeat(
                np.arange(256, dtype=np.uint8).reshape(1, 256, 1),
                3,
                axis=2,
            )
            reference = linear_rgb_to_bgr_u8(
                apply_color_transform(
                    bgr_u8_to_linear_rgb(ramp),
                    transform,
                )
            )
            np.copyto(lut[:, 0, :], reference[0])
            self._color_transform_key = key
        opencv.LUT(foreground_bgr, lut, dst=transformed_foreground)
        if edge_foreground_bgr is None:
            return transformed_foreground, None
        opencv.LUT(edge_foreground_bgr, lut, dst=transformed_edge)
        return transformed_foreground, transformed_edge

    def blend(
        self,
        foreground: np.ndarray,
        backdrop: np.ndarray,
        mask: np.ndarray,
        *,
        light_wrap: float,
        edge_foreground: np.ndarray | None,
        prepared_light_wrap: PreparedLightWrap | None,
        diagnostics: MutableMapping[str, float] | None = None,
    ) -> np.ndarray:
        """Blend prevalidated inputs with frozen legacy float32 operation order."""

        opencv = cv2
        if opencv is None:  # pragma: no cover - package dependency invariant
            raise ColorError("opencv-python is required for compositor workspace")
        if (
            foreground.shape != self._shape
            or backdrop.shape != self._shape
            or mask.shape != self._shape[:2]
            or (edge_foreground is not None and edge_foreground.shape != self._shape)
        ):
            raise ValueError("legacy compositor workspace shape mismatch")

        (
            scalar_weight,
            scalar_inverse,
            weight_bgr,
            working_bgr,
            blur_small_bgr,
        ) = self._buffers()
        if diagnostics is not None:
            for name in COMPOSITOR_SUBSTAGE_NAMES:
                diagnostics.setdefault(name, 0.0)

        # uint8 -> float32 values are exact. ``copyto`` fills the retained
        # buffer without allocating the historical foreground cast.
        np.copyto(working_bgr, foreground, casting="unsafe")
        # The returned frame must have independent ownership because FrameHub
        # and output consumers may retain it. Before final conversion its fresh
        # storage is safe scratch for the stateless uint8 wrap upsample.
        output = np.empty(self._shape, dtype=np.uint8)
        known_allocation_bytes = output.nbytes

        if light_wrap > 0.0 or edge_foreground is not None:
            started_ns = time.perf_counter_ns() if diagnostics is not None else 0
            # Preserve the reference expression order:
            # ``4.0 * alpha * (1.0 - alpha)``.
            np.multiply(mask, np.float32(4.0), out=scalar_weight)
            np.subtract(np.float32(1.0), mask, out=scalar_inverse)
            np.multiply(scalar_weight, scalar_inverse, out=scalar_weight)
            if diagnostics is not None:
                diagnostics["edge_band"] += self._elapsed_ms(started_ns)

            if edge_foreground is not None:
                started_ns = time.perf_counter_ns() if diagnostics is not None else 0
                np.subtract(np.float32(1.0), scalar_weight, out=scalar_inverse)
                self._three_channel_weight(
                    scalar_inverse,
                    weight_bgr,
                    opencv=opencv,
                )
                opencv.multiply(working_bgr, weight_bgr, dst=working_bgr)
                self._three_channel_weight(
                    scalar_weight,
                    weight_bgr,
                    opencv=opencv,
                )
                opencv.multiply(
                    edge_foreground,
                    weight_bgr,
                    dst=weight_bgr,
                    dtype=opencv.CV_32F,
                )
                opencv.add(working_bgr, weight_bgr, dst=working_bgr)
                if diagnostics is not None:
                    diagnostics["model_foreground_replacement"] += self._elapsed_ms(
                        started_ns
                    )

            if light_wrap > 0.0:
                wrap: np.ndarray | None = None
                if prepared_light_wrap is not None:
                    wrap = _validate_prepared_light_wrap(
                        prepared_light_wrap,
                        blend_space="srgb_legacy",
                        expected_shape=foreground.shape,
                    )
                else:
                    started_ns = (
                        time.perf_counter_ns() if diagnostics is not None else 0
                    )
                    small_size = (
                        blur_small_bgr.shape[1],
                        blur_small_bgr.shape[0],
                    )
                    opencv.resize(
                        backdrop,
                        small_size,
                        dst=blur_small_bgr,
                        interpolation=opencv.INTER_AREA,
                    )
                    opencv.GaussianBlur(
                        blur_small_bgr,
                        (9, 9),
                        0,
                        dst=blur_small_bgr,
                    )
                    opencv.resize(
                        blur_small_bgr,
                        (self._shape[1], self._shape[0]),
                        dst=output,
                        interpolation=opencv.INTER_LINEAR,
                    )
                    wrap = output
                    if diagnostics is not None:
                        diagnostics["backdrop_blur_resize"] += self._elapsed_ms(
                            started_ns
                        )
                if wrap is not None:
                    started_ns = (
                        time.perf_counter_ns() if diagnostics is not None else 0
                    )
                    np.multiply(
                        scalar_weight,
                        np.float32(light_wrap),
                        out=scalar_weight,
                    )
                    np.subtract(
                        np.float32(1.0),
                        scalar_weight,
                        out=scalar_inverse,
                    )
                    self._three_channel_weight(
                        scalar_inverse,
                        weight_bgr,
                        opencv=opencv,
                    )
                    opencv.multiply(working_bgr, weight_bgr, dst=working_bgr)
                    self._three_channel_weight(
                        scalar_weight,
                        weight_bgr,
                        opencv=opencv,
                    )
                    opencv.multiply(
                        wrap,
                        weight_bgr,
                        dst=weight_bgr,
                        dtype=opencv.CV_32F,
                    )
                    opencv.add(working_bgr, weight_bgr, dst=working_bgr)
                    if diagnostics is not None:
                        diagnostics["light_wrap_interpolation"] += self._elapsed_ms(
                            started_ns
                        )

        started_ns = time.perf_counter_ns() if diagnostics is not None else 0
        self._three_channel_weight(mask, weight_bgr, opencv=opencv)
        opencv.multiply(working_bgr, weight_bgr, dst=working_bgr)
        np.subtract(np.float32(1.0), mask, out=scalar_inverse)
        self._three_channel_weight(scalar_inverse, weight_bgr, opencv=opencv)
        opencv.multiply(
            backdrop,
            weight_bgr,
            dst=weight_bgr,
            dtype=opencv.CV_32F,
        )
        opencv.add(working_bgr, weight_bgr, dst=working_bgr)
        np.copyto(output, working_bgr, casting="unsafe")
        if diagnostics is not None:
            diagnostics["final_blend_conversion"] += self._elapsed_ms(started_ns)
            started_ns = time.perf_counter_ns()
            if (
                output.dtype != np.uint8
                or output.shape != self._shape
                or not output.flags.c_contiguous
            ):  # pragma: no cover - construction invariant
                raise ColorError("legacy compositor produced an invalid frame")
            diagnostics["internal_output_validation"] += self._elapsed_ms(started_ns)

        self._calls += 1
        self._last_known_allocation_bytes = known_allocation_bytes
        return output


def _downscaled_blur_with_sample(
    image: np.ndarray,
    scale: int = 8,
    kernel: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native blurred pixels and their float32 temporal-state sample."""

    h, w = image.shape[:2]
    sw, sh = max(4, w // scale), max(4, h // scale)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (kernel, kernel), 0)
    return (
        blurred,
        np.ascontiguousarray(
            blurred,
            dtype=np.float32,
        ),
    )


def _downscaled_blur(image: np.ndarray, scale: int = 8, kernel: int = 9) -> np.ndarray:
    """Large soft blur on the cheap: blur at 1/scale resolution and upsample."""
    h, w = image.shape[:2]
    sw, sh = max(4, w // scale), max(4, h // scale)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (kernel, kernel), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def prepare_static_light_wrap(
    backdrop_bgr: np.ndarray,
    *,
    blend_space: BlendSpace,
    backdrop_linear_bgr: np.ndarray | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> PreparedLightWrap:
    """Prepare one immutable-backdrop wrap sample for generation caching.

    Unlike :func:`prepare_light_wrap`, this helper has no temporal authority.
    It is valid only when the caller has established that the backdrop pixels
    cannot advance.  Its resize/blur sequence is the exact stateless reference
    path, so a cached result changes cost and ownership, never pixels.
    """

    backdrop_bgr = _validate_frame(backdrop_bgr, name="backdrop_bgr")
    if blend_space not in ("srgb_legacy", "linear_srgb"):
        raise ValueError("blend_space must be 'srgb_legacy' or 'linear_srgb'")
    if cv2 is None:
        raise ColorError("opencv-python is required for light wrap")
    if diagnostics is not None:
        for name in COMPOSITOR_SUBSTAGE_NAMES:
            diagnostics.setdefault(name, 0.0)
    if blend_space == "srgb_legacy":
        if backdrop_linear_bgr is not None:
            raise ValueError(
                "backdrop_linear_bgr is only valid for linear_srgb light wrap"
            )
        working = backdrop_bgr
    else:
        working = (
            _bgr_u8_to_linear_bgr_prevalidated(backdrop_bgr)
            if backdrop_linear_bgr is None
            else _validate_prevalidated_linear_bgr(
                backdrop_linear_bgr,
                name="backdrop_linear_bgr",
                expected_shape=backdrop_bgr.shape,
            )
        )
    started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    try:
        pixels = np.ascontiguousarray(_downscaled_blur(working), dtype=np.float32)
    except cv2.error as exc:
        raise ColorError("OpenCV light-wrap preparation failed") from exc
    if diagnostics is not None:
        diagnostics["backdrop_blur_resize"] += (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000.0
    pixels.setflags(write=False)
    return PreparedLightWrap(
        pixels_bgr=pixels,
        blend_space=blend_space,
        stabilized=False,
    )


def prepare_light_wrap(
    backdrop_bgr: np.ndarray,
    *,
    blend_space: BlendSpace,
    stabilizer: LightWrapStabilizer,
    context: LightWrapFrameContext,
    backdrop_linear_bgr: np.ndarray | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> PreparedLightWrap:
    """Prepare and stabilize only the blurred backdrop wrap sample.

    Callers must not invoke this function when the effective wrap strength is
    zero.  That early bypass is what guarantees no blur work or temporal-state
    advancement for ``light_wrap: 0``.
    """

    backdrop_bgr = _validate_frame(backdrop_bgr, name="backdrop_bgr")
    if blend_space not in ("srgb_legacy", "linear_srgb"):
        raise ValueError("blend_space must be 'srgb_legacy' or 'linear_srgb'")
    if not isinstance(stabilizer, LightWrapStabilizer):
        raise ValueError("stabilizer must be a LightWrapStabilizer")
    if not isinstance(context, LightWrapFrameContext):
        raise ValueError("context must be a LightWrapFrameContext")
    if cv2 is None:
        raise ColorError("opencv-python is required for stabilized light wrap")
    if diagnostics is not None:
        for name in COMPOSITOR_SUBSTAGE_NAMES:
            diagnostics.setdefault(name, 0.0)

    if blend_space == "srgb_legacy":
        if backdrop_linear_bgr is not None:
            raise ValueError(
                "backdrop_linear_bgr is only valid for linear_srgb light wrap"
            )
        working = backdrop_bgr
        value_scale = 255.0
    else:
        working = (
            _bgr_u8_to_linear_bgr_prevalidated(backdrop_bgr)
            if backdrop_linear_bgr is None
            else _validate_prevalidated_linear_bgr(
                backdrop_linear_bgr,
                name="backdrop_linear_bgr",
                expected_shape=backdrop_bgr.shape,
            )
        )
        value_scale = 1.0

    height, width = backdrop_bgr.shape[:2]
    blur_started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    try:
        blurred_small, sample = _downscaled_blur_with_sample(working)
    except cv2.error as exc:
        raise ColorError("OpenCV light-wrap preparation failed") from exc
    if diagnostics is not None:
        diagnostics["backdrop_blur_resize"] += (
            time.perf_counter_ns() - blur_started_ns
        ) / 1_000_000.0
    filter_started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    filtered = stabilizer.update(
        sample,
        context,
        value_scale=value_scale,
        channel_order="bgr",
    )
    if diagnostics is not None:
        diagnostics["light_wrap_temporal_filter"] += (
            time.perf_counter_ns() - filter_started_ns
        ) / 1_000_000.0
    blur_started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    try:
        # When the stabilizer returns the current sample unchanged, upscale the
        # native blurred raster so schema-v1 stateless pixels remain exact.
        # The old path repeated downscale + blur solely to recover this raster.
        upscale_source = blurred_small if np.array_equal(filtered, sample) else filtered
        pixels = np.ascontiguousarray(
            cv2.resize(
                upscale_source,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            ),
            dtype=np.float32,
        )
    except cv2.error as exc:
        raise ColorError("OpenCV light-wrap preparation failed") from exc
    if diagnostics is not None:
        diagnostics["backdrop_blur_resize"] += (
            time.perf_counter_ns() - blur_started_ns
        ) / 1_000_000.0
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    pixels.setflags(write=False)
    return PreparedLightWrap(
        pixels_bgr=pixels,
        blend_space=blend_space,
        stabilized=True,
    )


def _validate_frame(
    frame: object,
    *,
    name: str,
    expected_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Validate one external BGR frame without copying or changing ownership."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.size == 0
    ):
        raise ValueError(f"{name} must be a non-empty HxWx3 uint8 BGR frame")
    if not frame.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if expected_shape is not None and frame.shape != expected_shape:
        raise ValueError(
            f"shape mismatch: {name} {frame.shape} vs foreground {expected_shape}"
        )
    return frame


def _validate_mask(mask: object, shape: tuple[int, int]) -> np.ndarray:
    """Validate the canonical finite float32 alpha-mask contract."""

    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.float32
        or mask.ndim != 2
        or mask.shape != shape
        or mask.size == 0
    ):
        raise ValueError(f"mask must be a non-empty {shape} float32 array")
    if not mask.flags.c_contiguous:
        raise ValueError("mask must be C-contiguous")
    if not np.isfinite(mask).all():
        raise ValueError("mask must contain only finite values")
    minimum = float(np.min(mask))
    maximum = float(np.max(mask))
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError("mask values must be in [0, 1]")
    return mask


def _validate_linear_frame(
    frame: object,
    *,
    name: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Validate an EOTF-decoded linear-sRGB input without coercion."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.float32
        or frame.ndim != 3
        or frame.shape != expected_shape
        or frame.size == 0
    ):
        raise ValueError(
            f"{name} must be a non-empty {expected_shape} float32 RGB array"
        )
    if not frame.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(frame).all():
        raise ValueError(f"{name} must contain only finite values")
    minimum = float(np.min(frame))
    maximum = float(np.max(frame))
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"{name} values must be in [0, 1]")
    return frame


def _validate_prepared_light_wrap(
    prepared: object,
    *,
    blend_space: BlendSpace,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Validate a prepared BGR sample at the compositor trust boundary."""

    if not isinstance(prepared, PreparedLightWrap):
        raise ValueError("prepared_light_wrap must be a PreparedLightWrap")
    if prepared.blend_space != blend_space:
        raise ValueError("prepared light-wrap blend space does not match compositor")
    if type(prepared.stabilized) is not bool:
        raise ValueError("prepared light-wrap stabilized flag must be boolean")
    pixels = prepared.pixels_bgr
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.float32
        or pixels.ndim != 3
        or pixels.shape != expected_shape
        or pixels.size == 0
        or not pixels.flags.c_contiguous
        or not np.isfinite(pixels).all()
    ):
        raise ValueError(
            "prepared light-wrap pixels must be finite contiguous float32 BGR"
        )
    maximum = 255.0 if blend_space == "srgb_legacy" else 1.0
    if float(np.min(pixels)) < 0.0 or float(np.max(pixels)) > maximum:
        raise ValueError("prepared light-wrap pixels lie outside the working space")
    return pixels


def _validated_light_wrap(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("light_wrap must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("light_wrap must be a finite number in [0, 1]")
    return result


def _is_identity_transform(transform: ColorTransform | None) -> bool:
    """Recognize the public identity policy without decoding an input twice."""

    return transform is None or transform.is_identity


def _validate_color_transform(transform: object) -> ColorTransform | None:
    if transform is not None and not isinstance(transform, ColorTransform):
        raise ValueError("color_transform must be a ColorTransform or None")
    return transform


def _legacy_blend_encoded(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    *,
    light_wrap: float,
    edge_foreground: np.ndarray | None,
    prepared_light_wrap: PreparedLightWrap | None,
    workspace: LegacyCompositorWorkspace | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> np.ndarray:
    """Run the frozen historical encoded-value arithmetic."""
    if workspace is not None and cv2 is not None:
        try:
            return workspace.blend(
                foreground,
                backdrop,
                mask,
                light_wrap=light_wrap,
                edge_foreground=edge_foreground,
                prepared_light_wrap=prepared_light_wrap,
                diagnostics=diagnostics,
            )
        except cv2.error:
            # Validated inputs can still encounter a native OpenCV failure.
            # Do not hand the partially written workspace back to the pipeline
            # for an identical retry; complete this frame through the frozen
            # allocation-heavy implementation below.  Structural Python
            # contract failures remain strict and are not swallowed.
            pass
    alpha = mask[..., None].astype(np.float32)
    fg = foreground.astype(np.float32)
    if light_wrap > 0.0 or edge_foreground is not None:
        # 0 in the person core and in pure background, 1 at the 50% edge.
        band = 4.0 * alpha * (1.0 - alpha)
        if edge_foreground is not None:
            fg = fg * (1.0 - band) + edge_foreground.astype(np.float32) * band
        if light_wrap > 0.0 and (cv2 is not None or prepared_light_wrap is not None):
            wrap = (
                _downscaled_blur(backdrop)
                if prepared_light_wrap is None
                else _validate_prepared_light_wrap(
                    prepared_light_wrap,
                    blend_space="srgb_legacy",
                    expected_shape=foreground.shape,
                )
            )
            k = light_wrap * band
            fg = fg * (1.0 - k) + wrap * k

    out = fg * alpha + backdrop.astype(np.float32) * (1.0 - alpha)
    return np.ascontiguousarray(out.astype(np.uint8))


def _legacy_composite(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    *,
    light_wrap: float,
    edge_foreground: np.ndarray | None,
    color_transform: ColorTransform | None,
    prepared_light_wrap: PreparedLightWrap | None,
    workspace: LegacyCompositorWorkspace | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> np.ndarray:
    """Preserve the historical path, decoding only a requested transform."""

    if _is_identity_transform(color_transform):
        transformed_foreground = foreground
        transformed_edge = edge_foreground
    else:
        assert color_transform is not None
        started_ns = time.perf_counter_ns() if diagnostics is not None else 0
        if workspace is not None and cv2 is not None:
            transformed_foreground, transformed_edge = workspace.transform_encoded(
                foreground,
                edge_foreground,
                color_transform,
            )
        else:
            transformed_foreground = _encode_transformed_linear_bgr_prevalidated(
                _bgr_u8_to_linear_bgr_prevalidated(foreground),
                color_transform,
            )
            transformed_edge = (
                None
                if edge_foreground is None
                else _encode_transformed_linear_bgr_prevalidated(
                    _bgr_u8_to_linear_bgr_prevalidated(edge_foreground),
                    color_transform,
                )
            )
        if diagnostics is not None:
            diagnostics["color_transform_application"] += (
                time.perf_counter_ns() - started_ns
            ) / 1_000_000.0
    return _legacy_blend_encoded(
        transformed_foreground,
        backdrop,
        mask,
        light_wrap=light_wrap,
        edge_foreground=transformed_edge,
        prepared_light_wrap=prepared_light_wrap,
        workspace=workspace,
        diagnostics=diagnostics,
    )


def composite_legacy_predecoded(
    foreground_bgr: np.ndarray,
    backdrop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    foreground_linear_rgb: np.ndarray,
    light_wrap: float = 0.0,
    edge_foreground_bgr: np.ndarray | None = None,
    edge_foreground_linear_rgb: np.ndarray | None = None,
    color_transform: ColorTransform | None = None,
    prepared_light_wrap: PreparedLightWrap | None = None,
    workspace: LegacyCompositorWorkspace | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> np.ndarray:
    """Apply a predecoded foreground transform, then legacy encoded blending."""

    validation_started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    foreground_bgr = _validate_frame(foreground_bgr, name="foreground_bgr")
    backdrop_bgr = _validate_frame(
        backdrop_bgr,
        name="backdrop_bgr",
        expected_shape=foreground_bgr.shape,
    )
    if (edge_foreground_bgr is None) != (edge_foreground_linear_rgb is None):
        raise ValueError(
            "edge_foreground_bgr and edge_foreground_linear_rgb "
            "must be provided together"
        )
    if edge_foreground_bgr is not None:
        edge_foreground_bgr = _validate_frame(
            edge_foreground_bgr,
            name="edge_foreground_bgr",
            expected_shape=foreground_bgr.shape,
        )
    mask = _validate_mask(mask, foreground_bgr.shape[:2])
    light_wrap = _validated_light_wrap(light_wrap)
    color_transform = _validate_color_transform(color_transform)
    foreground_linear_rgb = _validate_linear_frame(
        foreground_linear_rgb,
        name="foreground_linear_rgb",
        expected_shape=foreground_bgr.shape,
    )
    if edge_foreground_linear_rgb is not None:
        edge_foreground_linear_rgb = _validate_linear_frame(
            edge_foreground_linear_rgb,
            name="edge_foreground_linear_rgb",
            expected_shape=foreground_bgr.shape,
        )
    if workspace is not None and workspace.shape != foreground_bgr.shape:
        raise ValueError("legacy compositor workspace shape mismatch")
    if diagnostics is not None:
        for name in COMPOSITOR_SUBSTAGE_NAMES:
            diagnostics.setdefault(name, 0.0)
        diagnostics["input_mask_validation"] += (
            time.perf_counter_ns() - validation_started_ns
        ) / 1_000_000.0

    if _is_identity_transform(color_transform):
        transformed_foreground = foreground_bgr
        transformed_edge = edge_foreground_bgr
    else:
        assert color_transform is not None
        started_ns = time.perf_counter_ns() if diagnostics is not None else 0
        if workspace is not None and cv2 is not None:
            transformed_foreground, transformed_edge = workspace.transform_encoded(
                foreground_bgr,
                edge_foreground_bgr,
                color_transform,
            )
        else:
            transformed_foreground = linear_rgb_to_bgr_u8(
                apply_color_transform(foreground_linear_rgb, color_transform)
            )
            transformed_edge = (
                None
                if edge_foreground_linear_rgb is None
                else linear_rgb_to_bgr_u8(
                    apply_color_transform(
                        edge_foreground_linear_rgb,
                        color_transform,
                    )
                )
            )
        if diagnostics is not None:
            diagnostics["color_transform_application"] += (
                time.perf_counter_ns() - started_ns
            ) / 1_000_000.0
    return _legacy_blend_encoded(
        transformed_foreground,
        backdrop_bgr,
        mask,
        light_wrap=light_wrap,
        edge_foreground=transformed_edge,
        prepared_light_wrap=prepared_light_wrap,
        workspace=workspace,
        diagnostics=diagnostics,
    )


def composite_linear_predecoded(
    foreground_bgr: np.ndarray,
    backdrop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    foreground_linear_rgb: np.ndarray,
    backdrop_linear_rgb: np.ndarray,
    light_wrap: float = 0.0,
    edge_foreground_bgr: np.ndarray | None = None,
    edge_foreground_linear_rgb: np.ndarray | None = None,
    color_transform: ColorTransform | None = None,
    prepared_light_wrap: PreparedLightWrap | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> np.ndarray:
    """Composite already-decoded inputs without another full-frame EOTF pass.

    The BGR inputs are retained as authoritative external representations for
    bit-exact alpha endpoints. Their corresponding linear arrays must be
    EOTF-decoded, finite, contiguous float32 RGB values in [0, 1].
    """

    validation_started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    foreground_bgr = _validate_frame(foreground_bgr, name="foreground_bgr")
    backdrop_bgr = _validate_frame(
        backdrop_bgr,
        name="backdrop_bgr",
        expected_shape=foreground_bgr.shape,
    )
    if (edge_foreground_bgr is None) != (edge_foreground_linear_rgb is None):
        raise ValueError(
            "edge_foreground_bgr and edge_foreground_linear_rgb "
            "must be provided together"
        )
    if edge_foreground_bgr is not None:
        edge_foreground_bgr = _validate_frame(
            edge_foreground_bgr,
            name="edge_foreground_bgr",
            expected_shape=foreground_bgr.shape,
        )

    mask = _validate_mask(mask, foreground_bgr.shape[:2])
    light_wrap = _validated_light_wrap(light_wrap)
    color_transform = _validate_color_transform(color_transform)
    foreground_linear_rgb = _validate_linear_frame(
        foreground_linear_rgb,
        name="foreground_linear_rgb",
        expected_shape=foreground_bgr.shape,
    )
    backdrop_linear_rgb = _validate_linear_frame(
        backdrop_linear_rgb,
        name="backdrop_linear_rgb",
        expected_shape=foreground_bgr.shape,
    )
    if edge_foreground_linear_rgb is not None:
        edge_foreground_linear_rgb = _validate_linear_frame(
            edge_foreground_linear_rgb,
            name="edge_foreground_linear_rgb",
            expected_shape=foreground_bgr.shape,
        )
    if diagnostics is not None:
        for name in COMPOSITOR_SUBSTAGE_NAMES:
            diagnostics.setdefault(name, 0.0)
        diagnostics["input_mask_validation"] += (
            time.perf_counter_ns() - validation_started_ns
        ) / 1_000_000.0

    identity_transform = _is_identity_transform(color_transform)
    if identity_transform:
        transformed_foreground = foreground_linear_rgb
        transformed_edge = edge_foreground_linear_rgb
    else:
        assert color_transform is not None
        started_ns = time.perf_counter_ns() if diagnostics is not None else 0
        transformed_foreground = apply_color_transform(
            foreground_linear_rgb,
            color_transform,
        )
        transformed_edge = (
            None
            if edge_foreground_linear_rgb is None
            else apply_color_transform(
                edge_foreground_linear_rgb,
                color_transform,
            )
        )
        if diagnostics is not None:
            diagnostics["color_transform_application"] += (
                time.perf_counter_ns() - started_ns
            ) / 1_000_000.0

    alpha = mask[..., None]
    working_foreground = transformed_foreground
    if light_wrap > 0.0 or transformed_edge is not None:
        started_ns = time.perf_counter_ns() if diagnostics is not None else 0
        band = 4.0 * alpha * (1.0 - alpha)
        if diagnostics is not None:
            diagnostics["edge_band"] += (
                time.perf_counter_ns() - started_ns
            ) / 1_000_000.0
        if transformed_edge is not None:
            started_ns = time.perf_counter_ns() if diagnostics is not None else 0
            working_foreground = (
                working_foreground * (1.0 - band) + transformed_edge * band
            )
            if diagnostics is not None:
                diagnostics["model_foreground_replacement"] += (
                    time.perf_counter_ns() - started_ns
                ) / 1_000_000.0
        if light_wrap > 0.0 and (cv2 is not None or prepared_light_wrap is not None):
            started_ns = time.perf_counter_ns() if diagnostics is not None else 0
            wrap = (
                _downscaled_blur(backdrop_linear_rgb)
                if prepared_light_wrap is None
                else np.ascontiguousarray(
                    _validate_prepared_light_wrap(
                        prepared_light_wrap,
                        blend_space="linear_srgb",
                        expected_shape=foreground_bgr.shape,
                    )[..., ::-1]
                )
            )
            if diagnostics is not None and prepared_light_wrap is None:
                diagnostics["backdrop_blur_resize"] += (
                    time.perf_counter_ns() - started_ns
                ) / 1_000_000.0
            started_ns = time.perf_counter_ns() if diagnostics is not None else 0
            k = light_wrap * band
            working_foreground = working_foreground * (1.0 - k) + wrap * k
            if diagnostics is not None:
                diagnostics["light_wrap_interpolation"] += (
                    time.perf_counter_ns() - started_ns
                ) / 1_000_000.0

    started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    output_linear = working_foreground * alpha + backdrop_linear_rgb * (1.0 - alpha)
    encoded = linear_rgb_to_bgr_u8(output_linear)

    # EOTF/OETF round trips are specified within one code value, but exact alpha
    # endpoints are stronger: untouched input pixels remain byte-identical.
    background_endpoint = mask == 0.0
    if np.any(background_endpoint):
        encoded[background_endpoint] = backdrop_bgr[background_endpoint]
    if identity_transform:
        foreground_endpoint = mask == 1.0
        if np.any(foreground_endpoint):
            encoded[foreground_endpoint] = foreground_bgr[foreground_endpoint]
    encoded = np.ascontiguousarray(encoded)
    if diagnostics is not None:
        diagnostics["final_blend_conversion"] += (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000.0
        started_ns = time.perf_counter_ns()
        if (
            encoded.dtype != np.uint8
            or encoded.shape != foreground_bgr.shape
            or not encoded.flags.c_contiguous
        ):  # pragma: no cover - construction invariant
            raise ColorError("linear compositor produced an invalid frame")
        diagnostics["internal_output_validation"] += (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000.0
    return encoded


def _validate_prevalidated_linear_bgr(
    frame: object,
    *,
    name: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Check structural invariants without rescanning a validated full frame."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.float32
        or frame.ndim != 3
        or frame.shape != expected_shape
        or frame.size == 0
        or not frame.flags.c_contiguous
    ):
        raise ValueError(
            f"{name} must be a prevalidated contiguous {expected_shape} "
            "float32 linear-BGR array"
        )
    return frame


def _interpolate_linear_bgr(
    base: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Return ``base * (1-weight) + target * weight`` in compiled OpenCV."""

    inverse_weight = np.subtract(np.float32(1.0), weight)
    return cv2.blendLinear(base, target, inverse_weight, weight)


def _apply_color_transform_linear_bgr_prevalidated(
    linear_bgr: np.ndarray,
    transform: ColorTransform,
) -> np.ndarray:
    """Apply the public RGB gain policy to a trusted linear-BGR buffer."""

    exposure = np.float32(2.0**transform.exposure_ev)
    red, green, blue = transform.wb_gains
    matrix = np.diag(
        np.asarray(
            (blue * exposure, green * exposure, red * exposure),
            dtype=np.float32,
        )
    )
    return np.ascontiguousarray(cv2.transform(linear_bgr, matrix), dtype=np.float32)


def _encode_transformed_linear_bgr_prevalidated(
    linear_bgr: np.ndarray,
    transform: ColorTransform,
) -> np.ndarray:
    """Apply and encode a trusted linear-BGR foreground.

    OpenCV keeps the production path in native BGR order and compiled code.
    The deterministic NumPy fallback deliberately uses the public RGB
    reference operations so platforms without OpenCV retain the same bounded
    numerical contract.
    """

    if cv2 is None:
        linear_rgb = np.ascontiguousarray(linear_bgr[..., ::-1], dtype=np.float32)
        return linear_rgb_to_bgr_u8(apply_color_transform(linear_rgb, transform))
    transformed = _apply_color_transform_linear_bgr_prevalidated(
        linear_bgr,
        transform,
    )
    return _consume_linear_bgr_to_bgr_u8_prevalidated(transformed)


def _composite_legacy_bgr_prevalidated(
    foreground_bgr: np.ndarray,
    backdrop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    foreground_linear_bgr: np.ndarray,
    light_wrap: float = 0.0,
    edge_foreground_bgr: np.ndarray | None = None,
    edge_foreground_linear_bgr: np.ndarray | None = None,
    color_transform: ColorTransform | None = None,
    prepared_light_wrap: PreparedLightWrap | None = None,
    workspace: LegacyCompositorWorkspace | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> np.ndarray:
    """Legacy encoded blending over prevalidated linear-BGR foregrounds.

    This is the production integration seam for callers that already decoded
    the foreground.  It avoids channel-reversal copies and the NumPy transform
    path while preserving the public RGB predecoded API above.
    """

    validation_started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    foreground_bgr = _validate_frame(foreground_bgr, name="foreground_bgr")
    backdrop_bgr = _validate_frame(
        backdrop_bgr,
        name="backdrop_bgr",
        expected_shape=foreground_bgr.shape,
    )
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.float32
        or mask.ndim != 2
        or mask.shape != foreground_bgr.shape[:2]
        or mask.size == 0
        or not mask.flags.c_contiguous
    ):
        raise ValueError("mask must satisfy the prevalidated float32 contract")
    light_wrap = _validated_light_wrap(light_wrap)
    color_transform = _validate_color_transform(color_transform)
    foreground_linear_bgr = _validate_prevalidated_linear_bgr(
        foreground_linear_bgr,
        name="foreground_linear_bgr",
        expected_shape=foreground_bgr.shape,
    )
    if (edge_foreground_bgr is None) != (edge_foreground_linear_bgr is None):
        raise ValueError(
            "edge_foreground_bgr and edge_foreground_linear_bgr "
            "must be provided together"
        )
    if edge_foreground_bgr is not None:
        edge_foreground_bgr = _validate_frame(
            edge_foreground_bgr,
            name="edge_foreground_bgr",
            expected_shape=foreground_bgr.shape,
        )
        edge_foreground_linear_bgr = _validate_prevalidated_linear_bgr(
            edge_foreground_linear_bgr,
            name="edge_foreground_linear_bgr",
            expected_shape=foreground_bgr.shape,
        )
    if workspace is not None and workspace.shape != foreground_bgr.shape:
        raise ValueError("legacy compositor workspace shape mismatch")
    if diagnostics is not None:
        for name in COMPOSITOR_SUBSTAGE_NAMES:
            diagnostics.setdefault(name, 0.0)
        diagnostics["input_mask_validation"] += (
            time.perf_counter_ns() - validation_started_ns
        ) / 1_000_000.0

    if _is_identity_transform(color_transform):
        transformed_foreground = foreground_bgr
        transformed_edge = edge_foreground_bgr
    else:
        assert color_transform is not None
        started_ns = time.perf_counter_ns() if diagnostics is not None else 0
        if workspace is not None and cv2 is not None:
            transformed_foreground, transformed_edge = workspace.transform_encoded(
                foreground_bgr,
                edge_foreground_bgr,
                color_transform,
            )
        else:
            transformed_foreground = _encode_transformed_linear_bgr_prevalidated(
                foreground_linear_bgr,
                color_transform,
            )
            transformed_edge = (
                None
                if edge_foreground_linear_bgr is None
                else _encode_transformed_linear_bgr_prevalidated(
                    edge_foreground_linear_bgr,
                    color_transform,
                )
            )
        if diagnostics is not None:
            diagnostics["color_transform_application"] += (
                time.perf_counter_ns() - started_ns
            ) / 1_000_000.0

    return _legacy_blend_encoded(
        transformed_foreground,
        backdrop_bgr,
        mask,
        light_wrap=light_wrap,
        edge_foreground=transformed_edge,
        prepared_light_wrap=prepared_light_wrap,
        workspace=workspace,
        diagnostics=diagnostics,
    )


def _composite_linear_bgr_prevalidated(
    foreground_bgr: np.ndarray,
    backdrop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    foreground_linear_bgr: np.ndarray,
    backdrop_linear_bgr: np.ndarray,
    light_wrap: float = 0.0,
    edge_foreground_bgr: np.ndarray | None = None,
    edge_foreground_linear_bgr: np.ndarray | None = None,
    color_transform: ColorTransform | None = None,
    prepared_light_wrap: PreparedLightWrap | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> np.ndarray:
    """Accelerated production compositor over validated linear-BGR buffers.

    Public compositing APIs retain their linear-RGB contract and NumPy reference
    implementation.  This internal lane keeps OpenCV's native BGR order from
    EOTF through alpha blending and OETF, avoiding channel copies and Python
    full-frame arithmetic.  The caller-owned BGR frames remain authoritative at
    exact alpha endpoints.
    """

    validation_started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    foreground_bgr = _validate_frame(foreground_bgr, name="foreground_bgr")
    backdrop_bgr = _validate_frame(
        backdrop_bgr,
        name="backdrop_bgr",
        expected_shape=foreground_bgr.shape,
    )
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.float32
        or mask.ndim != 2
        or mask.shape != foreground_bgr.shape[:2]
        or mask.size == 0
        or not mask.flags.c_contiguous
    ):
        raise ValueError("mask must satisfy the prevalidated float32 contract")
    light_wrap = _validated_light_wrap(light_wrap)
    color_transform = _validate_color_transform(color_transform)
    foreground_linear_bgr = _validate_prevalidated_linear_bgr(
        foreground_linear_bgr,
        name="foreground_linear_bgr",
        expected_shape=foreground_bgr.shape,
    )
    backdrop_linear_bgr = _validate_prevalidated_linear_bgr(
        backdrop_linear_bgr,
        name="backdrop_linear_bgr",
        expected_shape=foreground_bgr.shape,
    )
    if (edge_foreground_bgr is None) != (edge_foreground_linear_bgr is None):
        raise ValueError(
            "edge_foreground_bgr and edge_foreground_linear_bgr "
            "must be provided together"
        )
    if edge_foreground_bgr is not None:
        edge_foreground_bgr = _validate_frame(
            edge_foreground_bgr,
            name="edge_foreground_bgr",
            expected_shape=foreground_bgr.shape,
        )
        edge_foreground_linear_bgr = _validate_prevalidated_linear_bgr(
            edge_foreground_linear_bgr,
            name="edge_foreground_linear_bgr",
            expected_shape=foreground_bgr.shape,
        )
    if diagnostics is not None:
        for name in COMPOSITOR_SUBSTAGE_NAMES:
            diagnostics.setdefault(name, 0.0)
        diagnostics["input_mask_validation"] += (
            time.perf_counter_ns() - validation_started_ns
        ) / 1_000_000.0

    if cv2 is None:
        return composite_linear_predecoded(
            foreground_bgr,
            backdrop_bgr,
            mask,
            foreground_linear_rgb=np.ascontiguousarray(
                foreground_linear_bgr[..., ::-1]
            ),
            backdrop_linear_rgb=np.ascontiguousarray(backdrop_linear_bgr[..., ::-1]),
            light_wrap=light_wrap,
            edge_foreground_bgr=edge_foreground_bgr,
            edge_foreground_linear_rgb=(
                None
                if edge_foreground_linear_bgr is None
                else np.ascontiguousarray(edge_foreground_linear_bgr[..., ::-1])
            ),
            color_transform=color_transform,
            prepared_light_wrap=prepared_light_wrap,
            diagnostics=diagnostics,
        )

    identity_transform = _is_identity_transform(color_transform)
    try:
        if identity_transform:
            transformed_foreground = foreground_linear_bgr
            transformed_edge = edge_foreground_linear_bgr
        else:
            assert color_transform is not None
            started_ns = time.perf_counter_ns() if diagnostics is not None else 0
            transformed_foreground = _apply_color_transform_linear_bgr_prevalidated(
                foreground_linear_bgr,
                color_transform,
            )
            transformed_edge = (
                None
                if edge_foreground_linear_bgr is None
                else _apply_color_transform_linear_bgr_prevalidated(
                    edge_foreground_linear_bgr,
                    color_transform,
                )
            )
            if diagnostics is not None:
                diagnostics["color_transform_application"] += (
                    time.perf_counter_ns() - started_ns
                ) / 1_000_000.0

        working_foreground = transformed_foreground
        if light_wrap > 0.0 or transformed_edge is not None:
            started_ns = time.perf_counter_ns() if diagnostics is not None else 0
            inverse_mask = np.subtract(np.float32(1.0), mask)
            band = cv2.multiply(mask, inverse_mask, scale=4.0)
            if diagnostics is not None:
                diagnostics["edge_band"] += (
                    time.perf_counter_ns() - started_ns
                ) / 1_000_000.0
            if transformed_edge is not None:
                started_ns = time.perf_counter_ns() if diagnostics is not None else 0
                working_foreground = _interpolate_linear_bgr(
                    working_foreground,
                    transformed_edge,
                    band,
                )
                if diagnostics is not None:
                    diagnostics["model_foreground_replacement"] += (
                        time.perf_counter_ns() - started_ns
                    ) / 1_000_000.0
            if light_wrap > 0.0:
                started_ns = time.perf_counter_ns() if diagnostics is not None else 0
                wrap = (
                    _downscaled_blur(backdrop_linear_bgr)
                    if prepared_light_wrap is None
                    else _validate_prepared_light_wrap(
                        prepared_light_wrap,
                        blend_space="linear_srgb",
                        expected_shape=foreground_bgr.shape,
                    )
                )
                if diagnostics is not None and prepared_light_wrap is None:
                    diagnostics["backdrop_blur_resize"] += (
                        time.perf_counter_ns() - started_ns
                    ) / 1_000_000.0
                started_ns = time.perf_counter_ns() if diagnostics is not None else 0
                working_foreground = _interpolate_linear_bgr(
                    working_foreground,
                    wrap,
                    np.multiply(band, np.float32(light_wrap)),
                )
                if diagnostics is not None:
                    diagnostics["light_wrap_interpolation"] += (
                        time.perf_counter_ns() - started_ns
                    ) / 1_000_000.0

        started_ns = time.perf_counter_ns() if diagnostics is not None else 0
        output_linear = _interpolate_linear_bgr(
            backdrop_linear_bgr,
            working_foreground,
            mask,
        )
        encoded = _consume_linear_bgr_to_bgr_u8_prevalidated(output_linear)
    except cv2.error as exc:
        # Structural contracts above intentionally remain strict ValueErrors.
        # Only a failure inside validated OpenCV photometric work enters the
        # pipeline's bounded identity-render fallback.
        raise ColorError("OpenCV photometric composition failed") from exc

    background_endpoint = cv2.compare(mask, 0.0, cv2.CMP_EQ)
    cv2.copyTo(backdrop_bgr, background_endpoint, encoded)
    if identity_transform:
        foreground_endpoint = cv2.compare(mask, 1.0, cv2.CMP_EQ)
        cv2.copyTo(foreground_bgr, foreground_endpoint, encoded)
    encoded = np.ascontiguousarray(encoded)
    if diagnostics is not None:
        diagnostics["final_blend_conversion"] += (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000.0
        started_ns = time.perf_counter_ns()
        if (
            encoded.dtype != np.uint8
            or encoded.shape != foreground_bgr.shape
            or not encoded.flags.c_contiguous
        ):  # pragma: no cover - construction invariant
            raise ColorError("linear compositor produced an invalid frame")
        diagnostics["internal_output_validation"] += (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000.0
    return encoded


def _composite_linear_bgr(
    foreground_bgr: np.ndarray,
    backdrop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    light_wrap: float = 0.0,
    edge_foreground_bgr: np.ndarray | None = None,
    color_transform: ColorTransform | None = None,
    prepared_light_wrap: PreparedLightWrap | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> np.ndarray:
    """Decode validated external inputs once and use the accelerated BGR lane."""

    foreground_linear_bgr = _bgr_u8_to_linear_bgr_prevalidated(foreground_bgr)
    backdrop_linear_bgr = _bgr_u8_to_linear_bgr_prevalidated(backdrop_bgr)
    edge_foreground_linear_bgr = (
        None
        if edge_foreground_bgr is None
        else _bgr_u8_to_linear_bgr_prevalidated(edge_foreground_bgr)
    )
    return _composite_linear_bgr_prevalidated(
        foreground_bgr,
        backdrop_bgr,
        mask,
        foreground_linear_bgr=foreground_linear_bgr,
        backdrop_linear_bgr=backdrop_linear_bgr,
        light_wrap=light_wrap,
        edge_foreground_bgr=edge_foreground_bgr,
        edge_foreground_linear_bgr=edge_foreground_linear_bgr,
        color_transform=color_transform,
        prepared_light_wrap=prepared_light_wrap,
        diagnostics=diagnostics,
    )


def composite(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    *,
    light_wrap: float = 0.0,
    edge_foreground: np.ndarray | None = None,
    blend_space: BlendSpace = "srgb_legacy",
    color_transform: ColorTransform | None = None,
    prepared_light_wrap: PreparedLightWrap | None = None,
    workspace: LegacyCompositorWorkspace | None = None,
    diagnostics: MutableMapping[str, float] | None = None,
) -> np.ndarray:
    """Blend a validated BGR foreground over a backdrop in an explicit space.

    foreground, backdrop: HxWx3 uint8 BGR; mask: HxW float32 in [0, 1].

    light_wrap (0..1): mixes a blurred copy of the backdrop into the person's
        edge band, simulating ambient light from the new background wrapping
        around the subject — the classic compositing trick for seamless edges.
    edge_foreground: clean-foreground prediction (HxWx3 uint8 BGR, e.g. from
        the rvm backend). Applied only inside the soft edge band, it replaces
        pixels contaminated by the original background (color spill in hair
        and along shoulders) with decontaminated ones.

    ``srgb_legacy`` preserves the original encoded-value arithmetic. A supplied
    transform is still evaluated in linear RGB before that legacy blend.
    ``linear_srgb`` decodes each input once, applies the foreground transform,
    edge replacement, light wrap, and alpha blend in linear RGB, then performs
    one final encode.
    """
    validation_started_ns = time.perf_counter_ns() if diagnostics is not None else 0
    foreground = _validate_frame(foreground, name="foreground")
    backdrop = _validate_frame(
        backdrop,
        name="backdrop",
        expected_shape=foreground.shape,
    )
    if edge_foreground is not None:
        edge_foreground = _validate_frame(
            edge_foreground,
            name="edge_foreground",
            expected_shape=foreground.shape,
        )
    mask = _validate_mask(mask, foreground.shape[:2])
    light_wrap = _validated_light_wrap(light_wrap)
    if blend_space not in ("srgb_legacy", "linear_srgb"):
        raise ValueError("blend_space must be 'srgb_legacy' or 'linear_srgb'")
    color_transform = _validate_color_transform(color_transform)
    if workspace is not None and workspace.shape != foreground.shape:
        raise ValueError("legacy compositor workspace shape mismatch")
    if diagnostics is not None:
        for name in COMPOSITOR_SUBSTAGE_NAMES:
            diagnostics.setdefault(name, 0.0)
        diagnostics["input_mask_validation"] += (
            time.perf_counter_ns() - validation_started_ns
        ) / 1_000_000.0

    if blend_space == "srgb_legacy":
        return _legacy_composite(
            foreground,
            backdrop,
            mask,
            light_wrap=light_wrap,
            edge_foreground=edge_foreground,
            color_transform=color_transform,
            prepared_light_wrap=prepared_light_wrap,
            workspace=workspace,
            diagnostics=diagnostics,
        )
    foreground_linear_rgb = bgr_u8_to_linear_rgb(foreground)
    backdrop_linear_rgb = bgr_u8_to_linear_rgb(backdrop)
    edge_foreground_linear_rgb = (
        None if edge_foreground is None else bgr_u8_to_linear_rgb(edge_foreground)
    )
    return composite_linear_predecoded(
        foreground,
        backdrop,
        mask,
        foreground_linear_rgb=foreground_linear_rgb,
        backdrop_linear_rgb=backdrop_linear_rgb,
        light_wrap=light_wrap,
        edge_foreground_bgr=edge_foreground,
        edge_foreground_linear_rgb=edge_foreground_linear_rgb,
        color_transform=color_transform,
        prepared_light_wrap=prepared_light_wrap,
        diagnostics=diagnostics,
    )
