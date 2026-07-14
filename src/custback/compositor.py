"""Alpha compositing of foreground (person) over a backdrop using the mask."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _downscaled_blur(image: np.ndarray, scale: int = 8, kernel: int = 9) -> np.ndarray:
    """Large soft blur on the cheap: blur at 1/scale resolution and upsample."""
    h, w = image.shape[:2]
    sw, sh = max(4, w // scale), max(4, h // scale)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (kernel, kernel), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def composite(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    *,
    light_wrap: float = 0.0,
    edge_foreground: np.ndarray | None = None,
) -> np.ndarray:
    """Blend: out = fg * mask + bg * (1 - mask), with optional edge treatments
    that make the person sit *in* the new background instead of on top of it.

    foreground, backdrop: HxWx3 uint8 BGR; mask: HxW float32 in [0, 1].

    light_wrap (0..1): mixes a blurred copy of the backdrop into the person's
        edge band, simulating ambient light from the new background wrapping
        around the subject — the classic compositing trick for seamless edges.
    edge_foreground: clean-foreground prediction (HxWx3 uint8 BGR, e.g. from
        the rvm backend). Applied only inside the soft edge band, it replaces
        pixels contaminated by the original background (color spill in hair
        and along shoulders) with decontaminated ones.
    """
    if foreground.shape != backdrop.shape:
        raise ValueError(
            f"shape mismatch: fg {foreground.shape} vs bg {backdrop.shape}"
        )
    if mask.shape != foreground.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != frame {foreground.shape[:2]}")

    alpha = mask[..., None].astype(np.float32)
    fg = foreground.astype(np.float32)

    if light_wrap > 0.0 or edge_foreground is not None:
        # 0 in the person core and in pure background, 1 at the 50% edge.
        band = 4.0 * alpha * (1.0 - alpha)
        if edge_foreground is not None and edge_foreground.shape == foreground.shape:
            fg = fg * (1.0 - band) + edge_foreground.astype(np.float32) * band
        if light_wrap > 0.0 and cv2 is not None:
            wrap = _downscaled_blur(backdrop)
            k = light_wrap * band
            fg = fg * (1.0 - k) + wrap * k

    out = fg * alpha + backdrop.astype(np.float32) * (1.0 - alpha)
    return out.astype(np.uint8)
