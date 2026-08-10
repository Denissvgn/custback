"""Strict binary epoch envelope for the remote renderer WebSocket."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal


RemoteFrameKind = Literal["raw-input", "rendered-output"]

REMOTE_FRAME_PROTOCOL_VERSION = 1
REMOTE_FRAME_MAX_MESSAGE_BYTES = 2**31
_REMOTE_FRAME_MAGIC = b"CBFRAME\0"
_REMOTE_FRAME_HEADER = struct.Struct(">8sBBHQI")
REMOTE_FRAME_HEADER_BYTES = _REMOTE_FRAME_HEADER.size
_KIND_TO_CODE: dict[RemoteFrameKind, int] = {
    "raw-input": 1,
    "rendered-output": 2,
}
_CODE_TO_KIND: dict[int, RemoteFrameKind] = {
    code: kind for kind, code in _KIND_TO_CODE.items()
}


class RemoteFrameProtocolError(ValueError):
    """A renderer message does not match the bounded v1 wire contract."""


@dataclass(frozen=True, slots=True)
class RemoteFrameEnvelope:
    """One immutable JPEG payload and its exact source raw epoch."""

    kind: RemoteFrameKind
    raw_epoch: int
    jpeg: bytes


def _validated_limit(max_message_bytes: int) -> int:
    if (
        type(max_message_bytes) is not int
        or not REMOTE_FRAME_HEADER_BYTES < max_message_bytes
        or max_message_bytes > REMOTE_FRAME_MAX_MESSAGE_BYTES
    ):
        raise ValueError("remote frame message limit is outside the supported bound")
    return max_message_bytes


def _validated_epoch(raw_epoch: int, kind: RemoteFrameKind) -> int:
    if type(raw_epoch) is not int or not 0 <= raw_epoch <= 2**63 - 1:
        raise ValueError("remote frame epoch must be a bounded nonnegative integer")
    if kind == "rendered-output" and raw_epoch == 0:
        raise ValueError("rendered output requires a positive raw epoch")
    return raw_epoch


def encode_remote_frame(
    kind: RemoteFrameKind,
    raw_epoch: int,
    jpeg: bytes,
    *,
    max_message_bytes: int = REMOTE_FRAME_MAX_MESSAGE_BYTES,
) -> bytes:
    """Encode one exact-length v1 message without inspecting private pixels."""

    limit = _validated_limit(max_message_bytes)
    if kind not in _KIND_TO_CODE:
        raise ValueError("unsupported remote frame message kind")
    epoch = _validated_epoch(raw_epoch, kind)
    if not isinstance(jpeg, bytes) or not jpeg:
        raise TypeError("remote frame payload must be non-empty bytes")
    message_bytes = REMOTE_FRAME_HEADER_BYTES + len(jpeg)
    if message_bytes > limit:
        raise ValueError("remote frame message exceeds the configured byte limit")
    header = _REMOTE_FRAME_HEADER.pack(
        _REMOTE_FRAME_MAGIC,
        REMOTE_FRAME_PROTOCOL_VERSION,
        _KIND_TO_CODE[kind],
        0,
        epoch,
        len(jpeg),
    )
    return header + jpeg


def decode_remote_frame(
    data: bytes,
    *,
    expected_kind: RemoteFrameKind,
    max_message_bytes: int = REMOTE_FRAME_MAX_MESSAGE_BYTES,
) -> RemoteFrameEnvelope:
    """Decode a message only when every version, kind, size, and epoch agrees."""

    limit = _validated_limit(max_message_bytes)
    if expected_kind not in _KIND_TO_CODE:
        raise ValueError("unsupported expected remote frame message kind")
    if not isinstance(data, bytes):
        raise TypeError("remote frame message must be bytes")
    if not REMOTE_FRAME_HEADER_BYTES < len(data) <= limit:
        raise RemoteFrameProtocolError("remote frame message length is invalid")
    try:
        magic, version, kind_code, reserved, raw_epoch, payload_bytes = (
            _REMOTE_FRAME_HEADER.unpack_from(data)
        )
    except struct.error as exc:  # pragma: no cover - length guard is authoritative
        raise RemoteFrameProtocolError("remote frame header is invalid") from exc
    if magic != _REMOTE_FRAME_MAGIC or version != REMOTE_FRAME_PROTOCOL_VERSION:
        raise RemoteFrameProtocolError("unsupported remote frame protocol")
    if reserved != 0:
        raise RemoteFrameProtocolError("remote frame reserved bits must be zero")
    kind = _CODE_TO_KIND.get(kind_code)
    if kind is None or kind != expected_kind:
        raise RemoteFrameProtocolError("remote frame message kind is invalid")
    if payload_bytes != len(data) - REMOTE_FRAME_HEADER_BYTES:
        raise RemoteFrameProtocolError("remote frame payload length is inconsistent")
    try:
        epoch = _validated_epoch(raw_epoch, kind)
    except ValueError as exc:
        raise RemoteFrameProtocolError(str(exc)) from exc
    return RemoteFrameEnvelope(
        kind=kind,
        raw_epoch=epoch,
        jpeg=data[REMOTE_FRAME_HEADER_BYTES:],
    )
