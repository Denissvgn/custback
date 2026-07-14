"""Example avatar-service client for the stage-2 frame-forwarding API.

Connects to custback's WebSocket, receives raw camera frames, applies a
placeholder "avatar" transform, and sends frames back. When custback runs
with background.mode=remote, the returned frames become the virtual camera
output (with a privacy-safe local blur fallback if this client stalls).

The real stage-2 avatar service replaces `transform()` with face tracking,
expression matching and avatar rendering.

Usage:
    custback --synthetic --mode remote --no-vcam &
    # Uses CUSTBACK_API_TOKEN first, then ~/.config/custback/api-token.
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


def api_token() -> str:
    token = os.environ.get("CUSTBACK_API_TOKEN", "").strip()
    if not token:
        token = (Path.home() / ".config" / "custback" / "api-token").read_text().strip()
    if len(token) < 32:
        raise RuntimeError("custback API token is missing or invalid")
    return token


def transform(frame: np.ndarray) -> np.ndarray:
    """Placeholder avatar effect: posterize + label."""
    out = (frame // 64) * 64
    cv2.putText(out, "AVATAR (remote)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


async def main() -> None:
    headers = {"Authorization": f"Bearer {api_token()}"}
    # websockets 14 renamed extra_headers to additional_headers; custback
    # supports both sides of that transition.
    header_arg = (
        "additional_headers"
        if "additional_headers" in inspect.signature(websockets.connect).parameters
        else "extra_headers"
    )
    async with websockets.connect(
        CUSTBACK_WS,
        max_size=16 * 1024 * 1024,
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
