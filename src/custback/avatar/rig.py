"""Avatar rigs: turn a :class:`FaceState` into a BGRA avatar sprite.

Two rig kinds share one contract:

* :class:`BuiltinRig` — a procedural presenter drawn with OpenCV
  primitives. No assets, deterministic, and every part responds to the
  ARKit channels (blinks, brows, jaw, smile, gaze, head pose). Several
  characters (:data:`AVATAR_PRESETS`) and styles are selectable:
  ``cartoon`` (flat, bold features), ``realistic`` (natural proportions
  with soft shading), and ``sketch`` (pencil drawing).
* :class:`LayeredRig` — user-supplied PNG layers, one per part, with
  optional expression variants (``eyes_closed.png``, ``mouth_open.png``)
  and an optional ``rig.yaml`` for pivot/sway/framing tuning. PNG art is
  rendered as authored: ``sketch`` applies as a post-filter, ``realistic``
  is builtin-only.

``render`` returns a straight-alpha BGRA uint8 sprite: RGB retains the
authored color independently of coverage, and alpha is applied exactly once
when the renderer places the sprite. OpenCV geometric/filter transforms use a
temporary premultiplied copy to avoid transparent-edge color bleed, then
restore straight alpha. The renderer crops the rig's window for the configured
framing (``bust`` keeps head and chest in frame for meeting tiles, ``full``
shows everything, ``closeup`` fills the frame with the face).
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import numpy as np
import yaml

from .config import AVATAR_PARTS, AvatarFraming, AvatarStyle
from .state import FaceState

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover - required by the package, defensive
    _cv2 = None

# OpenCV is an optional-at-import-time native boundary: the runtime checks below
# preserve the graceful error path when it is absent, while its dynamically
# generated API intentionally remains opaque to static checking.
cv2: Any = _cv2

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is a required dependency
    Image = None

log = logging.getLogger(__name__)

# Parts that ride on the head group and follow the head pose.
HEAD_PARTS = frozenset({"head", "mouth", "nose", "eyes", "brows", "hair"})

AVATAR_STYLES: tuple[str, ...] = get_args(AvatarStyle)
AVATAR_FRAMINGS: tuple[str, ...] = get_args(AvatarFraming)

_BLINK_THRESHOLD = 0.5
_MOUTH_OPEN_THRESHOLD = 0.25
DEFAULT_RIG_LAYER_MAX_PIXELS = 16_777_216
DEFAULT_RIG_TOTAL_MAX_PIXELS = 134_217_728
DEFAULT_RIG_MANIFEST_MAX_BYTES = 64 * 1024


class RigError(ValueError):
    """The rig selection or its assets are unusable."""


def _require_cv2() -> None:
    if cv2 is None:
        raise RigError("opencv-python is required to render avatar rigs")


@dataclass(frozen=True)
class AvatarPreset:
    """Palette and feature set of one builtin presenter (colors are BGR)."""

    skin: tuple[int, int, int]
    skin_shadow: tuple[int, int, int]
    hair: tuple[int, int, int]
    shirt: tuple[int, int, int]
    shirt_trim: tuple[int, int, int]
    iris: tuple[int, int, int]
    lip: tuple[int, int, int]
    hair_style: str  # swept | short | long | curly
    glasses: bool = False


# Keys must match config.BUILTIN_AVATARS (the config layer validates names).
AVATAR_PRESETS: dict[str, AvatarPreset] = {
    "casey": AvatarPreset(
        skin=(140, 178, 235),
        skin_shadow=(110, 148, 205),
        hair=(46, 62, 88),
        shirt=(120, 86, 36),
        shirt_trim=(90, 62, 24),
        iris=(96, 122, 46),
        lip=(98, 84, 190),
        hair_style="swept",
    ),
    "robin": AvatarPreset(
        skin=(66, 100, 148),
        skin_shadow=(48, 76, 116),
        hair=(34, 30, 26),
        shirt=(150, 96, 44),
        shirt_trim=(110, 68, 30),
        iris=(40, 36, 32),
        lip=(84, 66, 140),
        hair_style="short",
        glasses=True,
    ),
    "alex": AvatarPreset(
        skin=(160, 196, 242),
        skin_shadow=(128, 162, 210),
        hair=(90, 168, 200),
        shirt=(84, 124, 64),
        shirt_trim=(58, 92, 44),
        iris=(150, 110, 60),
        lip=(110, 96, 196),
        hair_style="long",
    ),
    "nova": AvatarPreset(
        skin=(120, 162, 214),
        skin_shadow=(92, 132, 182),
        hair=(40, 58, 124),
        shirt=(116, 64, 108),
        shirt_trim=(84, 44, 78),
        iris=(60, 96, 120),
        lip=(96, 72, 176),
        hair_style="curly",
    ),
}


def sketch_filter(canvas: np.ndarray) -> None:
    """Pencil-drawing post-process in place; preserves the alpha channel."""
    _require_cv2()
    premultiplied = _premultiply_straight_alpha(canvas)
    gray = cv2.cvtColor(premultiplied[..., :3], cv2.COLOR_BGR2GRAY)
    blurred_inverse = cv2.GaussianBlur(255 - gray, (21, 21), 0)
    pencil = cv2.divide(gray, 255 - blurred_inverse, scale=256)
    # A faint warm paper tint keeps the strokes from looking clinical.
    coverage = premultiplied[..., 3:4] / 255.0
    styled = np.zeros_like(premultiplied)
    styled[..., 0] = pencil * 0.96 * coverage[..., 0]
    styled[..., 1] = pencil * 0.98 * coverage[..., 0]
    styled[..., 2] = pencil * coverage[..., 0]
    styled[..., 3:4] = premultiplied[..., 3:4]
    canvas[:] = _restore_straight_alpha(styled)


def alpha_over(base: np.ndarray, layer: np.ndarray) -> None:
    """Composite straight-alpha BGRA ``layer`` over ``base`` in place."""

    source_alpha = layer[..., 3:4].astype(np.float32) / 255.0
    destination_alpha = base[..., 3:4].astype(np.float32) / 255.0
    out_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
    numerator = layer[..., :3].astype(np.float32) * source_alpha + base[..., :3].astype(
        np.float32
    ) * destination_alpha * (1.0 - source_alpha)
    out_rgb = np.zeros_like(numerator)
    np.divide(numerator, out_alpha, out=out_rgb, where=out_alpha > 0.0)
    base[..., :3] = np.clip(np.rint(out_rgb), 0, 255).astype(np.uint8)
    base[..., 3:4] = np.clip(np.rint(out_alpha * 255.0), 0, 255).astype(np.uint8)


def _premultiply_straight_alpha(image: np.ndarray) -> np.ndarray:
    premultiplied = image.astype(np.float32)
    premultiplied[..., :3] *= premultiplied[..., 3:4] / 255.0
    return premultiplied


def _restore_straight_alpha(premultiplied: np.ndarray) -> np.ndarray:
    alpha = np.clip(premultiplied[..., 3:4], 0.0, 255.0)
    straight = np.zeros_like(premultiplied, dtype=np.float32)
    straight[..., 3:4] = alpha
    np.divide(
        premultiplied[..., :3] * 255.0,
        alpha,
        out=straight[..., :3],
        where=alpha > 1e-6,
    )
    return np.clip(np.rint(straight), 0, 255).astype(np.uint8)


def resize_straight_alpha(
    image: np.ndarray,
    size: tuple[int, int],
    *,
    interpolation: int,
) -> np.ndarray:
    """Resize BGRA while preserving its documented straight-alpha storage."""

    _require_cv2()
    transformed = cv2.resize(
        _premultiply_straight_alpha(image),
        size,
        interpolation=interpolation,
    )
    return _restore_straight_alpha(transformed)


def apply_head_pose(
    layer: np.ndarray,
    *,
    yaw: float,
    pitch: float,
    roll: float,
    pivot: tuple[float, float],
    sway_px: tuple[float, float],
) -> np.ndarray:
    """Rotate by roll around ``pivot`` and translate by yaw/pitch sway."""
    _require_cv2()
    height, width = layer.shape[:2]
    dx = max(-sway_px[0], min(sway_px[0], math.sin(yaw) * sway_px[0] * 2.0))
    dy = max(-sway_px[1], min(sway_px[1], math.sin(pitch) * sway_px[1] * 2.0))
    matrix = cv2.getRotationMatrix2D(pivot, math.degrees(roll), 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    transformed = cv2.warpAffine(
        _premultiply_straight_alpha(layer),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return _restore_straight_alpha(transformed)


class Rig(ABC):
    """A renderable avatar with individually selectable parts."""

    parts: tuple[str, ...] = AVATAR_PARTS

    # Vertical sprite windows (top/bottom fractions) per framing choice;
    # subclasses override with rig-appropriate values.
    FRAMING_WINDOWS: dict[str, tuple[float, float]] = {
        framing: (0.0, 1.0) for framing in AVATAR_FRAMINGS
    }

    @abstractmethod
    def render(
        self,
        state: FaceState,
        visible: frozenset[str],
        *,
        follow_pose: bool = True,
    ) -> np.ndarray:
        """Return the avatar as a straight-alpha BGRA uint8 sprite."""

    def framing_window(self, framing: str) -> tuple[float, float]:
        """The sprite rows (as fractions) that the framing keeps in shot."""
        return self.FRAMING_WINDOWS.get(framing, (0.0, 1.0))

    def close(self) -> None:
        pass


class BuiltinRig(Rig):
    """Procedural presenter; no assets required.

    ``avatar`` picks a character from :data:`AVATAR_PRESETS`; ``style``
    picks the treatment (cartoon, realistic, sketch). The sprite reaches
    the mid-torso so the ``full`` framing shows a waist-up shot and
    ``bust`` a webcam-style head-and-chest shot.
    """

    WIDTH = 480
    HEIGHT = 800
    PIVOT = (240.0, 400.0)
    SWAY_PX = (40.0, 26.0)

    # The head spans rows ~56-394; the chest continues to the bottom edge.
    FRAMING_WINDOWS = {
        "full": (0.0, 1.0),
        "bust": (0.0, 0.80),
        "closeup": (0.02, 0.54),
    }

    _EYE_WHITE = (248, 248, 248)
    _PUPIL = (28, 24, 20)
    _BROW = (40, 52, 72)
    _MOUTH_INNER = (36, 24, 60)
    _TEETH = (240, 244, 248)

    def __init__(self, avatar: str = "casey", style: str = "cartoon") -> None:
        _require_cv2()
        preset = AVATAR_PRESETS.get(avatar)
        if preset is None:
            raise RigError(
                f"unknown builtin avatar {avatar!r}; choose from {sorted(AVATAR_PRESETS)}"
            )
        if style not in AVATAR_STYLES:
            raise RigError(
                f"unknown avatar style {style!r}; choose from {list(AVATAR_STYLES)}"
            )
        self.avatar = avatar
        self.style = style
        self._SKIN = preset.skin
        self._SKIN_SHADOW = preset.skin_shadow
        self._HAIR = preset.hair
        self._SHIRT = preset.shirt
        self._SHIRT_TRIM = preset.shirt_trim
        self._IRIS = preset.iris
        self._LIP = preset.lip
        self._hair_style = preset.hair_style
        self._glasses = preset.glasses
        self._highlight = tuple(min(255, channel + 44) for channel in preset.skin)
        # Realistic style: human-scale features and soft shading; the
        # cartoon and sketch styles keep the bold cartoon geometry.
        realistic = style == "realistic"
        self._feature_scale = 0.75 if realistic else 1.0
        self._brow_px = 5 if realistic else 8
        self._lip_open_px = 4 if realistic else 5
        self._lip_closed_px = 5 if realistic else 7

    def render(
        self,
        state: FaceState,
        visible: frozenset[str],
        *,
        follow_pose: bool = True,
    ) -> np.ndarray:
        canvas = np.zeros((self.HEIGHT, self.WIDTH, 4), dtype=np.uint8)
        if "torso" in visible:
            self._draw_torso(canvas)
        head = np.zeros_like(canvas)
        drew_head = False
        # Hair before the facial features so brow raises stay visible
        # under the fringe.
        for part in ("head", "hair", "eyes", "brows", "nose", "mouth"):
            if part in visible:
                getattr(self, f"_draw_{part}")(head, state)
                drew_head = True
        if drew_head:
            if (
                follow_pose
                and state.present
                and (state.yaw or state.pitch or state.roll)
            ):
                head = apply_head_pose(
                    head,
                    yaw=state.yaw,
                    pitch=state.pitch,
                    roll=state.roll,
                    pivot=self.PIVOT,
                    sway_px=self.SWAY_PX,
                )
            alpha_over(canvas, head)
        if self.style == "realistic":
            self._soften_realistic(canvas)
        elif self.style == "sketch":
            sketch_filter(canvas)
        return canvas

    # -- torso ---------------------------------------------------------
    def _draw_torso(self, canvas: np.ndarray) -> None:
        cv2.rectangle(canvas, (208, 360), (272, 470), (*self._SKIN_SHADOW, 255), -1)
        cv2.ellipse(
            canvas, (240, 760), (232, 300), 0, 180, 360, (*self._SHIRT, 255), -1
        )
        cv2.rectangle(canvas, (8, 760), (472, self.HEIGHT), (*self._SHIRT, 255), -1)
        cv2.ellipse(
            canvas, (240, 760), (232, 300), 0, 180, 360, (*self._SHIRT_TRIM, 255), 6
        )
        # Collar, placket, and buttons read as clothing once the chest is
        # in frame (bust/full framing) instead of a bare color band.
        cv2.line(canvas, (206, 474), (240, 548), (*self._SHIRT_TRIM, 255), 5)
        cv2.line(canvas, (274, 474), (240, 548), (*self._SHIRT_TRIM, 255), 5)
        cv2.line(canvas, (240, 548), (240, self.HEIGHT), (*self._SHIRT_TRIM, 255), 3)
        for button_y in (596, 668, 740):
            cv2.circle(canvas, (240, button_y), 6, (*self._SHIRT_TRIM, 255), -1)
        cv2.ellipse(canvas, (240, 470), (54, 26), 0, 0, 180, (*self._SKIN, 255), -1)

    # -- head group ------------------------------------------------------
    def _draw_head(self, canvas: np.ndarray, _state: FaceState) -> None:
        cv2.ellipse(canvas, (240, 236), (128, 158), 0, 0, 360, (*self._SKIN, 255), -1)
        for cx in (116, 364):  # ears
            cv2.ellipse(canvas, (cx, 250), (18, 30), 0, 0, 360, (*self._SKIN, 255), -1)

    def _draw_hair(self, canvas: np.ndarray, _state: FaceState) -> None:
        getattr(self, f"_draw_hair_{self._hair_style}")(canvas)

    def _draw_hair_swept(self, canvas: np.ndarray) -> None:
        cv2.ellipse(canvas, (240, 176), (134, 120), 0, 180, 360, (*self._HAIR, 255), -1)
        cv2.ellipse(canvas, (240, 150), (136, 84), 0, 0, 180, (*self._HAIR, 255), -1)
        cv2.ellipse(canvas, (168, 176), (44, 52), 20, 0, 360, (*self._HAIR, 255), -1)
        cv2.ellipse(canvas, (312, 176), (44, 52), -20, 0, 360, (*self._HAIR, 255), -1)

    def _draw_hair_short(self, canvas: np.ndarray) -> None:
        cv2.ellipse(canvas, (240, 190), (130, 112), 0, 180, 360, (*self._HAIR, 255), -1)
        for cx in (118, 362):  # temple fades
            cv2.ellipse(canvas, (cx, 214), (16, 36), 0, 0, 360, (*self._HAIR, 255), -1)

    def _draw_hair_long(self, canvas: np.ndarray) -> None:
        self._draw_hair_swept(canvas)
        for cx in (150, 330):  # curtains falling to the shoulders
            cv2.ellipse(canvas, (cx, 320), (34, 150), 0, 0, 360, (*self._HAIR, 255), -1)

    def _draw_hair_curly(self, canvas: np.ndarray) -> None:
        cv2.ellipse(canvas, (240, 180), (128, 104), 0, 180, 360, (*self._HAIR, 255), -1)
        for cx, cy in ((130, 152), (180, 118), (240, 106), (300, 118), (350, 152)):
            cv2.circle(canvas, (cx, cy), 28, (*self._HAIR, 255), -1)

    def _draw_brows(self, canvas: np.ndarray, state: FaceState) -> None:
        raise_amount = (
            state.channel("browInnerUp")
            + (state.channel("browOuterUpLeft") + state.channel("browOuterUpRight"))
            / 2.0
        ) / 2.0
        for cx, down in (
            (178, state.channel("browDownLeft")),
            (302, state.channel("browDownRight")),
        ):
            dy = int(round(-20.0 * raise_amount + 10.0 * down))
            cv2.ellipse(
                canvas,
                (cx, 196 + dy),
                (34, 10),
                0,
                190,
                350,
                (*self._BROW, 255),
                self._brow_px,
            )

    def _draw_eyes(self, canvas: np.ndarray, state: FaceState) -> None:
        factor = self._feature_scale
        eye_rx = max(8, int(round(30.0 * factor)))
        look_x = (
            state.channel("eyeLookOutRight")
            + state.channel("eyeLookInLeft")
            - state.channel("eyeLookOutLeft")
            - state.channel("eyeLookInRight")
        ) / 2.0
        look_y = (
            state.channel("eyeLookDownLeft")
            + state.channel("eyeLookDownRight")
            - state.channel("eyeLookUpLeft")
            - state.channel("eyeLookUpRight")
        ) / 2.0
        for cx, blink, wide in (
            (180, state.channel("eyeBlinkLeft"), state.channel("eyeWideLeft")),
            (300, state.channel("eyeBlinkRight"), state.channel("eyeWideRight")),
        ):
            openness = max(0.0, (1.0 - blink) * (1.0 + 0.35 * wide))
            half_height = max(1, int(round(19.0 * factor * min(openness, 1.4))))
            if openness <= 0.12:  # closed: draw the lid line only
                cv2.line(
                    canvas,
                    (cx - eye_rx + 4, 240),
                    (cx + eye_rx - 4, 240),
                    (*self._SKIN_SHADOW, 255),
                    5,
                )
                continue
            cv2.ellipse(
                canvas,
                (cx, 240),
                (eye_rx, half_height),
                0,
                0,
                360,
                (*self._EYE_WHITE, 255),
                -1,
            )
            ix = cx + int(round(12.0 * factor * look_x))
            iy = 240 + int(round(min(half_height - 4, 8) * look_y))
            cv2.circle(
                canvas,
                (ix, iy),
                max(3, int(round(12 * factor))),
                (*self._IRIS, 255),
                -1,
            )
            cv2.circle(
                canvas,
                (ix, iy),
                max(2, int(round(6 * factor))),
                (*self._PUPIL, 255),
                -1,
            )
            cv2.ellipse(
                canvas,
                (cx, 240),
                (eye_rx, half_height),
                0,
                0,
                360,
                (*self._SKIN_SHADOW, 255),
                2,
            )
        if self._glasses:
            self._draw_glasses(canvas)

    def _draw_glasses(self, canvas: np.ndarray) -> None:
        frame_color = (48, 44, 40)
        for cx in (180, 300):
            cv2.ellipse(canvas, (cx, 240), (40, 32), 0, 0, 360, (*frame_color, 255), 5)
        cv2.line(canvas, (220, 234), (260, 234), (*frame_color, 255), 5)
        cv2.line(canvas, (140, 236), (114, 228), (*frame_color, 255), 5)
        cv2.line(canvas, (340, 236), (366, 228), (*frame_color, 255), 5)

    def _draw_nose(self, canvas: np.ndarray, _state: FaceState) -> None:
        cv2.ellipse(
            canvas, (240, 292), (12, 8), 0, 0, 180, (*self._SKIN_SHADOW, 255), 4
        )

    def _draw_mouth(self, canvas: np.ndarray, state: FaceState) -> None:
        jaw = state.channel("jawOpen")
        smile = (
            state.channel("mouthSmileLeft") + state.channel("mouthSmileRight")
        ) / 2.0
        frown = (
            state.channel("mouthFrownLeft") + state.channel("mouthFrownRight")
        ) / 2.0
        pucker = max(state.channel("mouthPucker"), state.channel("mouthFunnel"))
        center_y = 348 + int(round(10.0 * jaw))
        half_width = max(14, int(round(46.0 * (1.0 + 0.25 * smile - 0.45 * pucker))))
        open_half = int(round(2.0 + 30.0 * jaw))
        corner_dy = int(round(-14.0 * smile + 12.0 * frown))
        if jaw > 0.06:
            cv2.ellipse(
                canvas,
                (240, center_y),
                (half_width, open_half),
                0,
                0,
                360,
                (*self._MOUTH_INNER, 255),
                -1,
            )
            if open_half > 10:
                cv2.rectangle(
                    canvas,
                    (240 - half_width + 10, center_y - open_half + 3),
                    (240 + half_width - 10, center_y - max(2, open_half // 2)),
                    (*self._TEETH, 255),
                    -1,
                )
            cv2.ellipse(
                canvas,
                (240, center_y),
                (half_width, open_half),
                0,
                0,
                360,
                (*self._LIP, 255),
                self._lip_open_px,
            )
        else:
            # Closed lips: a parabola whose corners rise with smiles and
            # drop with frowns, sampled so the curve stays smooth.
            xs = np.linspace(-half_width, half_width, 15)
            ys = center_y + corner_dy * (xs / half_width) ** 2
            points = np.stack([240 + xs, ys], axis=1).astype(np.int32)
            cv2.polylines(
                canvas,
                [points],
                False,
                (*self._LIP, 255),
                self._lip_closed_px,
                cv2.LINE_AA,
            )

    # -- style post-processing -------------------------------------------
    def _soften_realistic(self, canvas: np.ndarray) -> None:
        """Soft light and shadow plus gentle edge smoothing, in place.

        Shading only touches the color channels: where alpha is zero the
        tint is invisible, so the silhouette stays exact.
        """
        overlay = canvas[..., :3].copy()
        cv2.ellipse(overlay, (214, 196), (64, 44), -12, 0, 360, self._highlight, -1)
        cv2.ellipse(overlay, (296, 306), (52, 64), 15, 0, 360, self._SKIN_SHADOW, -1)
        cv2.ellipse(overlay, (240, 486), (140, 70), 0, 0, 180, self._SKIN_SHADOW, -1)
        canvas[..., :3] = cv2.addWeighted(overlay, 0.22, canvas[..., :3], 0.78, 0)
        canvas[:] = _restore_straight_alpha(
            cv2.GaussianBlur(_premultiply_straight_alpha(canvas), (3, 3), 0)
        )


class LayeredRig(Rig):
    """PNG-layer rig loaded from a directory.

    Layout: one ``<part>.png`` per part (any subset of
    :data:`~custback.avatar.config.AVATAR_PARTS`), all with identical
    dimensions and an alpha channel. Optional variants replace their base
    layer while the expression is active: ``eyes_closed.png`` during blinks
    and ``mouth_open.png`` while the jaw is open. Optional ``rig.yaml`` keys:
    ``pivot: [x, y]`` (head rotation pivot, pixels), ``sway: [x, y]``
    (maximum yaw/pitch shift, pixels), ``head_parts`` (parts that follow
    the head pose; defaults to everything except the torso), and
    ``framing`` (mapping of framing name to ``[top, bottom]`` sprite-height
    fractions kept in shot).

    The ``sketch`` style applies as a pencil post-filter; ``cartoon`` and
    ``realistic`` leave the PNG art as authored.
    """

    FRAMING_WINDOWS = {
        "full": (0.0, 1.0),
        "bust": (0.0, 0.8),
        "closeup": (0.0, 0.5),
    }

    def __init__(
        self,
        directory: str | Path,
        style: str = "cartoon",
        *,
        rig_layer_max_pixels: int = DEFAULT_RIG_LAYER_MAX_PIXELS,
        rig_total_max_pixels: int = DEFAULT_RIG_TOTAL_MAX_PIXELS,
        rig_manifest_max_bytes: int = DEFAULT_RIG_MANIFEST_MAX_BYTES,
    ):
        _require_cv2()
        if style not in AVATAR_STYLES:
            raise RigError(
                f"unknown avatar style {style!r}; choose from {list(AVATAR_STYLES)}"
            )
        self.style = style
        self._framing_windows = dict(self.FRAMING_WINDOWS)
        self.directory = Path(directory).expanduser()
        if not self.directory.is_dir():
            raise RigError(f"rig directory does not exist: {self.directory}")
        for label, value in (
            ("rig_layer_max_pixels", rig_layer_max_pixels),
            ("rig_total_max_pixels", rig_total_max_pixels),
            ("rig_manifest_max_bytes", rig_manifest_max_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RigError(f"{label} must be a positive integer")
        self._rig_layer_max_pixels = rig_layer_max_pixels
        self._rig_total_max_pixels = rig_total_max_pixels
        self._rig_manifest_max_bytes = rig_manifest_max_bytes
        self._validate_manifest_size()
        self._layers: dict[str, np.ndarray] = {}
        self._variants: dict[str, np.ndarray] = {}
        discovered: list[tuple[str, Path]] = []
        for name in (*AVATAR_PARTS, "eyes_closed", "mouth_open"):
            path = self.directory / f"{name}.png"
            if path.is_file():
                discovered.append((name, path))
        if not any(name in AVATAR_PARTS for name, _path in discovered):
            raise RigError(
                f"rig directory {self.directory} contains no part layers "
                f"({', '.join(part + '.png' for part in AVATAR_PARTS)})"
            )
        size = self._preflight_layers([path for _name, path in discovered])
        for name, path in discovered:
            layer = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if (
                layer is None
                or layer.dtype != np.uint8
                or layer.ndim != 3
                or layer.shape[2] != 4
            ):
                raise RigError(f"rig layer must be a PNG with alpha: {path.name}")
            if layer.shape[:2] != size:
                raise RigError(
                    f"rig layer {path.name} size {layer.shape[1]}x{layer.shape[0]} "
                    f"does not match {size[1]}x{size[0]}"
                )
            if name in AVATAR_PARTS:
                self._layers[name] = layer
            else:
                self._variants[name] = layer
        height, width = size
        self.parts = tuple(part for part in AVATAR_PARTS if part in self._layers)
        self._pivot = (width / 2.0, height * 0.6)
        self._sway = (width * 0.08, height * 0.04)
        self._head_parts = frozenset(self._layers) & HEAD_PARTS
        self._load_manifest()

    @staticmethod
    def _has_alpha(image) -> bool:
        return "A" in image.getbands() or "transparency" in image.info

    def _validate_manifest_size(self) -> None:
        manifest = self.directory / "rig.yaml"
        try:
            if (
                manifest.is_file()
                and manifest.stat().st_size > self._rig_manifest_max_bytes
            ):
                raise RigError(f"rig.yaml exceeds {self._rig_manifest_max_bytes} bytes")
        except OSError as exc:
            raise RigError("cannot inspect rig.yaml") from exc

    def _preflight_layers(self, paths: list[Path]) -> tuple[int, int]:
        """Fully validate every layer with Pillow before OpenCV sees a path."""

        if Image is None:
            raise RigError("Pillow is required to validate PNG rig layers")
        expected_size: tuple[int, int] | None = None
        total_pixels = 0
        for path in paths:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with Image.open(path) as image:
                        if image.format != "PNG":
                            raise ValueError("layer header is not PNG")
                        width, height = image.size
                        if width <= 0 or height <= 0:
                            raise ValueError("invalid layer dimensions")
                        pixels = width * height
                        if pixels > self._rig_layer_max_pixels:
                            raise RigError(
                                f"rig layer {path.name} exceeds "
                                f"{self._rig_layer_max_pixels} pixels"
                            )
                        total_pixels += pixels
                        if total_pixels > self._rig_total_max_pixels:
                            raise RigError(
                                "rig decoded layers exceed "
                                f"{self._rig_total_max_pixels} pixels"
                            )
                        if expected_size is None:
                            expected_size = (width, height)
                        elif (width, height) != expected_size:
                            raise RigError(
                                f"rig layer {path.name} size {width}x{height} "
                                f"does not match {expected_size[0]}x"
                                f"{expected_size[1]}"
                            )
                        if not self._has_alpha(image):
                            raise RigError(
                                f"rig layer must be a PNG with alpha: {path.name}"
                            )
                        image.verify()
                    # verify() checks structure and CRC; a fresh decoder then
                    # forces full decompression before OpenCV can allocate.
                    with Image.open(path) as decoded:
                        if decoded.format != "PNG" or decoded.size != (width, height):
                            raise ValueError("layer changed during validation")
                        if not self._has_alpha(decoded):
                            raise ValueError("layer alpha changed during validation")
                        decoded.load()
            except RigError:
                raise
            except Exception as exc:
                # warnings.simplefilter("error") turns Pillow decompression
                # warnings into exceptions here; decoder/CRC/truncation and
                # unidentified-format failures share the same RigError API.
                raise RigError(f"invalid PNG rig layer: {path.name}") from exc
        assert expected_size is not None
        return expected_size[1], expected_size[0]

    def _load_manifest(self) -> None:
        manifest = self.directory / "rig.yaml"
        if not manifest.is_file():
            return
        self._validate_manifest_size()
        try:
            raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError, RecursionError) as exc:
            raise RigError("invalid rig.yaml") from exc
        if raw is None:
            return
        if not isinstance(raw, dict):
            raise RigError("rig.yaml root must be a mapping")
        unknown = set(raw) - {"pivot", "sway", "head_parts", "framing"}
        if unknown:
            raise RigError(f"unknown rig.yaml keys: {sorted(unknown)}")
        for key in ("pivot", "sway"):
            if key in raw:
                value = raw[key]
                converted: tuple[float, float] | None = None
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and all(
                        isinstance(item, (int, float)) and not isinstance(item, bool)
                        for item in value
                    )
                ):
                    try:
                        candidate = (float(value[0]), float(value[1]))
                    except (OverflowError, ValueError):
                        candidate = None
                    if candidate is not None and all(
                        math.isfinite(item) for item in candidate
                    ):
                        converted = candidate
                if converted is None:
                    raise RigError(f"rig.yaml {key} must be [x, y] finite numbers")
                setattr(self, f"_{key}", converted)
        if "head_parts" in raw:
            value = raw["head_parts"]
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item in AVATAR_PARTS for item in value
            ):
                raise RigError(
                    f"rig.yaml head_parts must list parts from {list(AVATAR_PARTS)}"
                )
            self._head_parts = frozenset(value) & set(self._layers)
        if "framing" in raw:
            value = raw["framing"]
            if not isinstance(value, dict):
                raise RigError(
                    "rig.yaml framing must map framing names to [top, bottom]"
                )
            for name, window in value.items():
                if name not in AVATAR_FRAMINGS:
                    raise RigError(
                        f"unknown framing {name!r}; choose from {list(AVATAR_FRAMINGS)}"
                    )
                converted_window: tuple[float, float] | None = None
                if (
                    isinstance(window, list)
                    and len(window) == 2
                    and all(
                        isinstance(item, (int, float)) and not isinstance(item, bool)
                        for item in window
                    )
                ):
                    try:
                        candidate = (float(window[0]), float(window[1]))
                    except (OverflowError, ValueError):
                        candidate = None
                    if (
                        candidate is not None
                        and all(math.isfinite(item) for item in candidate)
                        and 0.0 <= candidate[0] < candidate[1] <= 1.0
                    ):
                        converted_window = candidate
                if converted_window is None:
                    raise RigError(
                        f"rig.yaml framing {name} must be [top, bottom] fractions "
                        "with 0 <= top < bottom <= 1"
                    )
                self._framing_windows[name] = converted_window

    def framing_window(self, framing: str) -> tuple[float, float]:
        return self._framing_windows.get(framing, (0.0, 1.0))

    def _layer_for(self, part: str, state: FaceState) -> np.ndarray:
        if part == "eyes" and "eyes_closed" in self._variants:
            blink = (
                state.channel("eyeBlinkLeft") + state.channel("eyeBlinkRight")
            ) / 2.0
            if blink >= _BLINK_THRESHOLD:
                return self._variants["eyes_closed"]
        if part == "mouth" and "mouth_open" in self._variants:
            if state.channel("jawOpen") >= _MOUTH_OPEN_THRESHOLD:
                return self._variants["mouth_open"]
        return self._layers[part]

    def render(
        self,
        state: FaceState,
        visible: frozenset[str],
        *,
        follow_pose: bool = True,
    ) -> np.ndarray:
        sample = next(iter(self._layers.values()))
        canvas = np.zeros_like(sample)
        head = np.zeros_like(sample)
        drew_head = False
        for part in AVATAR_PARTS:
            if part not in self._layers or part not in visible:
                continue
            layer = self._layer_for(part, state)
            if part in self._head_parts:
                alpha_over(head, layer)
                drew_head = True
            else:
                alpha_over(canvas, layer)
        if drew_head:
            if (
                follow_pose
                and state.present
                and (state.yaw or state.pitch or state.roll)
            ):
                head = apply_head_pose(
                    head,
                    yaw=state.yaw,
                    pitch=state.pitch,
                    roll=state.roll,
                    pivot=self._pivot,
                    sway_px=self._sway,
                )
            alpha_over(canvas, head)
        if self.style == "sketch":
            sketch_filter(canvas)
        return canvas


def create_rig(
    selector: str,
    *,
    avatar: str = "casey",
    style: str = "cartoon",
    rig_layer_max_pixels: int = DEFAULT_RIG_LAYER_MAX_PIXELS,
    rig_total_max_pixels: int = DEFAULT_RIG_TOTAL_MAX_PIXELS,
    rig_manifest_max_bytes: int = DEFAULT_RIG_MANIFEST_MAX_BYTES,
) -> Rig:
    """Build a rig from the ``appearance`` selectors."""
    if selector == "builtin":
        return BuiltinRig(avatar=avatar, style=style)
    return LayeredRig(
        selector,
        style=style,
        rig_layer_max_pixels=rig_layer_max_pixels,
        rig_total_max_pixels=rig_total_max_pixels,
        rig_manifest_max_bytes=rig_manifest_max_bytes,
    )
