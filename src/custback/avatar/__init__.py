"""Stage-2 avatar service: renders an animated avatar into custback.

The service is a client of custback's frame-forwarding API: it connects to
``WS /ws/frames?stream=raw``, tracks the person in the raw camera frames (or
follows an external Audio2Face-3D animation stream), renders an avatar over a
selected background, and returns the frames. With ``background.mode: remote``
custback shows exactly these frames on the virtual camera.

It runs either on the same machine as custback (``custback-avatar``) or on a
separate GPU host, pointing ``source.url`` at custback's TLS-protected API.
"""

from .config import AVATAR_PARTS, AvatarConfig, AvatarRuntime
from .state import ARKIT_BLENDSHAPES, FaceState

__all__ = [
    "ARKIT_BLENDSHAPES",
    "AVATAR_PARTS",
    "AvatarConfig",
    "AvatarRuntime",
    "FaceState",
]
