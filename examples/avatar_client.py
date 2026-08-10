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
from typing import Any

import cv2
import numpy as np
import websockets

from custback.remote_protocol import decode_remote_frame, encode_remote_frame

CUSTBACK_WS = "ws://127.0.0.1:8710/ws/frames?stream=raw"
FRAME_MAX_BYTES = 16 * 1024 * 1024


def renderer_token() -> str:
    token = os.environ.get("CUSTBACK_RENDERER_TOKEN", "").strip()
    if not token:
        token = (
            (Path.home() / ".config" / "custback" / "renderer-token")
            .read_text()
            .strip()
        )
    if len(token) < 32:
        raise RuntimeError("custback renderer token is missing or invalid")
    return token


def transform(frame: np.ndarray) -> np.ndarray:
    """Placeholder avatar effect: posterize + label."""
    out = (frame // 64) * 64
    cv2.putText(
        out,
        "AVATAR (remote)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
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
    # The header keyword changed across supported websockets releases.  Keep
    # the dynamically selected keyword at this compatibility boundary rather
    # than making the rest of the connection call untyped.
    connect_kwargs: dict[str, Any] = {header_arg: headers}
    if "proxy" in connect_parameters:
        connect_kwargs["proxy"] = None
    async with websockets.connect(
        CUSTBACK_WS,
        max_size=FRAME_MAX_BYTES,
        **connect_kwargs,
    ) as ws:
        print(f"connected to {CUSTBACK_WS}")
        while True:
            data = await ws.recv()
            if not isinstance(data, bytes):
                raise RuntimeError("frame WebSocket returned a non-binary message")
            envelope = decode_remote_frame(
                data,
                expected_kind="raw-input",
                max_message_bytes=FRAME_MAX_BYTES,
            )
            # Renderer leases receive only positive, remotely admitted raw
            # epochs. Local-mode/management preview frames are never exposed
            # on this credentialed duplex lane.
            frame = cv2.imdecode(
                np.frombuffer(envelope.jpeg, np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None:
                continue
            rendered = transform(frame)
            ok, jpeg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                await ws.send(
                    encode_remote_frame(
                        "rendered-output",
                        envelope.raw_epoch,
                        jpeg.tobytes(),
                        max_message_bytes=FRAME_MAX_BYTES,
                    )
                )


if __name__ == "__main__":
    asyncio.run(main())
