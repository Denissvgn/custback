"""Alpha compositing of foreground (person) over a backdrop using the mask."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Literal

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

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

# OpenCV is a compiled optional boundary. Keep its runtime ``None`` fallback
# while treating the dynamically exposed API as opaque to static analysis.
cv2: Any = _cv2

BlendSpace = Literal["srgb_legacy", "linear_srgb"]


def _downscaled_blur(image: np.ndarray, scale: int = 8, kernel: int = 9) -> np.ndarray:
    """Large soft blur on the cheap: blur at 1/scale resolution and upsample."""
    h, w = image.shape[:2]
    sw, sh = max(4, w // scale), max(4, h // scale)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (kernel, kernel), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


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
) -> np.ndarray:
    """Run the frozen historical encoded-value arithmetic."""
    alpha = mask[..., None].astype(np.float32)
    fg = foreground.astype(np.float32)
    if light_wrap > 0.0 or edge_foreground is not None:
        # 0 in the person core and in pure background, 1 at the 50% edge.
        band = 4.0 * alpha * (1.0 - alpha)
        if edge_foreground is not None:
            fg = fg * (1.0 - band) + edge_foreground.astype(np.float32) * band
        if light_wrap > 0.0 and cv2 is not None:
            wrap = _downscaled_blur(backdrop)
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
) -> np.ndarray:
    """Preserve the historical path, decoding only a requested transform."""

    if _is_identity_transform(color_transform):
        transformed_foreground = foreground
        transformed_edge = edge_foreground
    else:
        assert color_transform is not None
        transformed_foreground = linear_rgb_to_bgr_u8(
            apply_color_transform(
                bgr_u8_to_linear_rgb(foreground),
                color_transform,
            )
        )
        transformed_edge = (
            None
            if edge_foreground is None
            else linear_rgb_to_bgr_u8(
                apply_color_transform(
                    bgr_u8_to_linear_rgb(edge_foreground),
                    color_transform,
                )
            )
        )
    return _legacy_blend_encoded(
        transformed_foreground,
        backdrop,
        mask,
        light_wrap=light_wrap,
        edge_foreground=transformed_edge,
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
) -> np.ndarray:
    """Apply a predecoded foreground transform, then legacy encoded blending."""

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

    if _is_identity_transform(color_transform):
        transformed_foreground = foreground_bgr
        transformed_edge = edge_foreground_bgr
    else:
        assert color_transform is not None
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
    return _legacy_blend_encoded(
        transformed_foreground,
        backdrop_bgr,
        mask,
        light_wrap=light_wrap,
        edge_foreground=transformed_edge,
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
) -> np.ndarray:
    """Composite already-decoded inputs without another full-frame EOTF pass.

    The BGR inputs are retained as authoritative external representations for
    bit-exact alpha endpoints. Their corresponding linear arrays must be
    EOTF-decoded, finite, contiguous float32 RGB values in [0, 1].
    """

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

    identity_transform = _is_identity_transform(color_transform)
    if identity_transform:
        transformed_foreground = foreground_linear_rgb
        transformed_edge = edge_foreground_linear_rgb
    else:
        assert color_transform is not None
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

    alpha = mask[..., None]
    working_foreground = transformed_foreground
    if light_wrap > 0.0 or transformed_edge is not None:
        band = 4.0 * alpha * (1.0 - alpha)
        if transformed_edge is not None:
            working_foreground = (
                working_foreground * (1.0 - band) + transformed_edge * band
            )
        if light_wrap > 0.0 and cv2 is not None:
            wrap = _downscaled_blur(backdrop_linear_rgb)
            k = light_wrap * band
            working_foreground = working_foreground * (1.0 - k) + wrap * k

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
    return np.ascontiguousarray(encoded)


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
) -> np.ndarray:
    """Accelerated production compositor over validated linear-BGR buffers.

    Public compositing APIs retain their linear-RGB contract and NumPy reference
    implementation.  This internal lane keeps OpenCV's native BGR order from
    EOTF through alpha blending and OETF, avoiding channel copies and Python
    full-frame arithmetic.  The caller-owned BGR frames remain authoritative at
    exact alpha endpoints.
    """

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
        )

    identity_transform = _is_identity_transform(color_transform)
    try:
        if identity_transform:
            transformed_foreground = foreground_linear_bgr
            transformed_edge = edge_foreground_linear_bgr
        else:
            assert color_transform is not None
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

        working_foreground = transformed_foreground
        if light_wrap > 0.0 or transformed_edge is not None:
            inverse_mask = np.subtract(np.float32(1.0), mask)
            band = cv2.multiply(mask, inverse_mask, scale=4.0)
            if transformed_edge is not None:
                working_foreground = _interpolate_linear_bgr(
                    working_foreground,
                    transformed_edge,
                    band,
                )
            if light_wrap > 0.0:
                wrap = _downscaled_blur(backdrop_linear_bgr)
                working_foreground = _interpolate_linear_bgr(
                    working_foreground,
                    wrap,
                    np.multiply(band, np.float32(light_wrap)),
                )

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
    return np.ascontiguousarray(encoded)


def _composite_linear_bgr(
    foreground_bgr: np.ndarray,
    backdrop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    light_wrap: float = 0.0,
    edge_foreground_bgr: np.ndarray | None = None,
    color_transform: ColorTransform | None = None,
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

    if blend_space == "srgb_legacy":
        return _legacy_composite(
            foreground,
            backdrop,
            mask,
            light_wrap=light_wrap,
            edge_foreground=edge_foreground,
            color_transform=color_transform,
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
    )
