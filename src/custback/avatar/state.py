"""Animation state shared by all avatar drivers and renderers.

Every driver — vision tracking, Audio2Face-3D, or the synthetic idle
animator — normalizes its output into :class:`FaceState`: the 52 ARKit
blendshape channels plus a head pose. Renderers only ever consume this
contract, so animation sources are interchangeable at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# The ARKit face blendshape set, in MediaPipe's lowerCamelCase spelling.
# Audio2Face-3D emits the same channels in PascalCase; drivers normalize.
ARKIT_BLENDSHAPES: tuple[str, ...] = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
)

_CHANNEL_SET = frozenset(ARKIT_BLENDSHAPES)


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


@dataclass
class FaceState:
    """One animation sample: blendshape weights plus a coarse head pose.

    ``yaw``/``pitch``/``roll`` are radians in the camera frame (positive yaw
    turns the face to its left on screen). ``center_x``/``center_y`` locate
    the face in the source frame (normalized 0..1) and ``size`` is the face
    height as a fraction of the frame height; drivers without spatial
    tracking keep the defaults.
    """

    present: bool = False
    blendshapes: dict[str, float] = field(default_factory=dict)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    center_x: float = 0.5
    center_y: float = 0.5
    size: float = 0.35
    timestamp: float = 0.0

    @classmethod
    def neutral(cls, *, present: bool = True, timestamp: float = 0.0) -> "FaceState":
        return cls(present=present, timestamp=timestamp)

    def channel(self, name: str) -> float:
        """Return one blendshape weight clamped to [0, 1]."""
        if name not in _CHANNEL_SET:
            raise KeyError(f"unknown blendshape channel: {name!r}")
        return _clamp01(float(self.blendshapes.get(name, 0.0)))

    def set_channel(self, name: str, value: float) -> None:
        if name not in _CHANNEL_SET:
            raise KeyError(f"unknown blendshape channel: {name!r}")
        self.blendshapes[name] = _clamp01(float(value))


class StateSmoother:
    """Exponential smoothing over channels and pose to damp tracking jitter.

    ``factor`` is the weight of the previous sample (0 disables smoothing).
    Blink channels track instantly upward so blinks are never smeared away.
    """

    _INSTANT_UP = ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen")

    def __init__(self, factor: float):
        if not 0.0 <= factor <= 0.95:
            raise ValueError("smoothing factor must be within [0, 0.95]")
        self.factor = factor
        self._previous: FaceState | None = None

    def apply(self, state: FaceState) -> FaceState:
        previous = self._previous
        if previous is None or self.factor == 0.0 or not state.present:
            self._previous = state
            return state
        keep = self.factor
        blend = {}
        for name in _CHANNEL_SET | set(previous.blendshapes) | set(state.blendshapes):
            new = _clamp01(float(state.blendshapes.get(name, 0.0)))
            old = _clamp01(float(previous.blendshapes.get(name, 0.0)))
            if name in self._INSTANT_UP and new > old:
                blend[name] = new
            else:
                blend[name] = old * keep + new * (1.0 - keep)
        smoothed = FaceState(
            present=True,
            blendshapes=blend,
            yaw=previous.yaw * keep + state.yaw * (1.0 - keep),
            pitch=previous.pitch * keep + state.pitch * (1.0 - keep),
            roll=previous.roll * keep + state.roll * (1.0 - keep),
            center_x=previous.center_x * keep + state.center_x * (1.0 - keep),
            center_y=previous.center_y * keep + state.center_y * (1.0 - keep),
            size=previous.size * keep + state.size * (1.0 - keep),
            timestamp=state.timestamp,
        )
        self._previous = smoothed
        return smoothed

    def reset(self) -> None:
        self._previous = None
