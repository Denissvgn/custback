"""Minimal example client for the stage-2 frame-forwarding API.

Connects to custback's WebSocket, receives raw camera frames, applies a
placeholder "avatar" transform, and sends frames back. When custback runs
with background.mode=remote, the returned frames become the virtual camera
output (with a fixed, camera-independent privacy slate if this client stalls).

The production stage-2 service ships with the package: `custback-avatar`
(the `custback.avatar` package) adds face tracking, Audio2Face-3D support,
rig rendering, and its own control API on top of this same protocol. This
example stays as the smallest possible starting point for a custom renderer.

Usage:
    custback --synthetic --mode remote --no-vcam &
    # Uses CUSTBACK_RENDERER_TOKEN first, then the renderer-only token file.
    python examples/avatar_client.py
"""

import asyncio
import inspect
import os
from pathlib import Path

import cv2
import numpy as np
import websockets

CUSTBACK_WS = "ws://127.0.0.1:8710/ws/frames?stream=raw"


def renderer_token() -> str:
    token = os.environ.get("CUSTBACK_RENDERER_TOKEN", "").strip()
    if not token:
        token = (
            Path.home() / ".config" / "custback" / "renderer-token"
        ).read_text().strip()
    if len(token) < 32:
        raise RuntimeError("custback renderer token is missing or invalid")
    return token


def transform(frame: np.ndarray) -> np.ndarray:
    """Placeholder avatar effect: posterize + label."""
    out = (frame // 64) * 64
    cv2.putText(out, "AVATAR (remote)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


async def main() -> None:
    headers = {"Authorization": f"Bearer {renderer_token()}"}
    # websockets 14 renamed extra_headers to additional_headers; custback
    # supports both sides of that transition.
    connect_parameters = inspect.signature(websockets.connect).parameters
    header_arg = (
        "additional_headers"
        if "additional_headers" in connect_parameters
        else "extra_headers"
    )
    proxy_args = {"proxy": None} if "proxy" in connect_parameters else {}
    async with websockets.connect(
        CUSTBACK_WS,
        max_size=16 * 1024 * 1024,
        **proxy_args,
        **{header_arg: headers},
    ) as ws:
        print(f"connected to {CUSTBACK_WS}")
        while True:
            data = await ws.recv()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            rendered = transform(frame)
            ok, jpeg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                await ws.send(jpeg.tobytes())


if __name__ == "__main__":
    asyncio.run(main())
