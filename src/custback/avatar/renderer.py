"""Place the avatar sprite over the selected background at output size.

The avatar service must return frames exactly the size of custback's raw
camera frames, so composition always targets the incoming frame geometry.
Backdrop providers are reused from :mod:`custback.backgrounds`; ``blur``
falls back to blurring the raw camera frame locally, which keeps a natural
scene behind the avatar without revealing the room sharply.
"""

from __future__ import annotations

import numpy as np

from ..backgrounds import (
    DEFAULT_IMAGE_MAX_PIXELS,
    BackdropProvider,
    ColorBackdrop,
    ImageBackdrop,
    VideoBackdrop,
)
from .config import AppearanceConfig, AvatarBackgroundConfig
from .rig import resize_straight_alpha

try:
    import cv2
except ImportError:  # pragma: no cover - required by the package, defensive
    cv2 = None


def create_avatar_backdrop(
    cfg: AvatarBackgroundConfig,
    *,
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
    video_max_width: int = 3840,
    video_max_height: int = 2160,
) -> BackdropProvider | None:
    """Build the static/scene provider; ``blur`` is handled per frame."""
    if cfg.mode == "color":
        return ColorBackdrop(cfg.color)
    if cfg.mode == "image":
        return ImageBackdrop(cfg.image_path, max_pixels=image_max_pixels)
    if cfg.mode == "video":
        return VideoBackdrop(
            cfg.video_path,
            max_width=video_max_width,
            max_height=video_max_height,
            color_matrix=cfg.video_color_matrix,
            color_range=cfg.video_color_range,
            color_primaries=cfg.video_color_primaries,
            color_transfer=cfg.video_color_transfer,
        )
    return None  # blur uses the raw frame


def blurred_room(frame: np.ndarray, blur_strength: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for the blur background")
    kernel = blur_strength if blur_strength % 2 else blur_strength + 1
    return cv2.GaussianBlur(frame, (kernel, kernel), 0)


def compose_avatar(
    sprite_bgra: np.ndarray,
    backdrop_bgr: np.ndarray,
    appearance: AppearanceConfig,
    framing_window: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Alpha-blend the sprite onto the backdrop like a person in the frame.

    ``framing_window`` selects the sprite rows (top/bottom fractions, from
    the rig's window for ``appearance.framing``) kept in shot — e.g. head
    and chest for the ``bust`` framing. ``scale`` is the framed avatar
    height relative to the frame height. The framed sprite's bottom-center
    anchors to the bottom edge of the frame; ``offset_x`` and ``offset_y``
    shift that anchor by up to half a frame right/down (negative values
    move left/up). Off-frame regions are cropped.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for avatar composition")
    height, width = backdrop_bgr.shape[:2]
    out = backdrop_bgr.copy()
    sprite_h, sprite_w = sprite_bgra.shape[:2]
    if sprite_h == 0 or sprite_w == 0:
        return out
    top = min(sprite_h, max(0, int(round(sprite_h * framing_window[0]))))
    bottom = min(sprite_h, max(0, int(round(sprite_h * framing_window[1]))))
    if bottom - top >= 1 and (top, bottom) != (0, sprite_h):
        sprite_bgra = sprite_bgra[top:bottom]
        sprite_h = bottom - top
    target_h = max(1, int(round(height * appearance.scale)))
    target_w = max(1, int(round(sprite_w * target_h / sprite_h)))
    interpolation = cv2.INTER_AREA if target_h < sprite_h else cv2.INTER_LINEAR
    sprite = resize_straight_alpha(
        sprite_bgra,
        (target_w, target_h),
        interpolation=interpolation,
    )

    anchor_x = width / 2.0 + appearance.offset_x * width / 2.0
    anchor_y = float(height) + appearance.offset_y * height / 2.0
    x0 = int(round(anchor_x - target_w / 2.0))
    y0 = int(round(anchor_y - target_h))

    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1 = min(width, x0 + target_w)
    dst_y1 = min(height, y0 + target_h)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return out
    region = sprite[
        src_y0 : src_y0 + (dst_y1 - dst_y0), src_x0 : src_x0 + (dst_x1 - dst_x0)
    ]
    alpha = region[..., 3:4].astype(np.float32) / 255.0
    target = out[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    blended = region[..., :3].astype(np.float32) * alpha + target * (1.0 - alpha)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out
