"""Native Windows 11 virtual-camera frame transport (WIN-6.1).

The native Media Foundation virtual camera is a C++/WinRT media source that
runs inside the Windows Frame Server service process
(``packaging/windows/vcam/``).  The engine cannot hand it frames in-process;
the two sides share a named page-file section instead.  This module is the
*writer* half of that contract and the single source of truth for the shared
layout — ``packaging/windows/vcam/FrameRing.h`` mirrors these constants and
``tests/test_windows_vcam.py`` asserts the two never drift.

Transport design (latest-frame-wins seqlock, no queue):

* One fixed-size mapping: a 64-byte header followed by one BGRX frame.
* The writer publishes a frame by making ``seq`` odd, writing the payload and
  metadata, then making ``seq`` even again.  A reader that observes an odd
  ``seq``, or a ``seq`` that changed across its copy, discards the torn frame
  and retries; the Media Foundation stream serves its own sample clock and
  simply reads the newest complete frame on each ``RequestSample``.
* There is deliberately no cross-process event: a virtual camera must keep
  producing samples at its negotiated cadence even when the producer stalls,
  so the reader falls back to holding the last frame (or a placeholder when
  ``FLAG_ACTIVE`` is clear) rather than blocking on the engine.

Nothing here imports pywin32; the Windows mapping uses :mod:`mmap`'s named
``tagname`` support, so the ring logic stays testable on any platform against
a plain buffer.  Everything security-relevant about the section is local to
the user session (``Local\\`` namespace, per-user Frame Server registration).
"""

from __future__ import annotations

import logging
import mmap
import struct
import sys
import time
from typing import TypedDict

import numpy as np

from .vcam import VideoOutput

log = logging.getLogger(__name__)

#: COM class of the media-source activator (packaging/windows/vcam).  The
#: literal must match dllmain.cpp, VirtualCameraSession.cs, and Package.wxs;
#: tests/test_windows_vcam.py cross-checks all of them against this constant.
VCAM_CLSID = "{7A4C1B2E-9D35-4E6A-8B1F-52C84D9A6E01}"

#: Friendly name the camera registers with; meeting apps show this string.
#: It intentionally contains "Custback" so camera enumeration classifies it
#: as a virtual *output* and never offers it back as an input (WIN-3.4).
VCAM_FRIENDLY_NAME = "Custback Camera"

#: Session-local named section shared with the Frame Server media source.
SECTION_NAME = "Local\\CustbackVCamFrame0"

#: Raw header magic ("CBVC") and layout version.
MAGIC = b"CBVC"
PROTOCOL_VERSION = 1

#: Pixel format: MFVideoFormat_RGB32 memory order (B, G, R, X per pixel).
FOURCC = b"BGRX"
BYTES_PER_PIXEL = 4

#: Header layout, little-endian, 64 bytes:
#: magic 4s | version u32 | width u32 | height u32 | stride u32 | fourcc 4s |
#: seq u32 | flags u32 | timestamp_100ns u64 | frame_counter u64 | 16 reserved.
HEADER_FORMAT = "<4sIIII4sIIQQ16x"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

_SEQ_OFFSET = 24
_FLAGS_OFFSET = 28

#: ``flags`` bit 0: the writer is alive and publishing.  Cleared on close so
#: the media source can show its placeholder instead of a frozen last frame.
FLAG_ACTIVE = 0x1


def ring_size(width: int, height: int) -> int:
    """Total mapping size for one ``width``×``height`` BGRX frame."""

    return HEADER_SIZE + width * height * BYTES_PER_PIXEL


def _pack_header(
    width: int,
    height: int,
    *,
    seq: int,
    flags: int,
    timestamp_100ns: int,
    frame_counter: int,
) -> bytes:
    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        PROTOCOL_VERSION,
        width,
        height,
        width * BYTES_PER_PIXEL,
        FOURCC,
        seq & 0xFFFFFFFF,
        flags,
        timestamp_100ns & 0xFFFFFFFFFFFFFFFF,
        frame_counter & 0xFFFFFFFFFFFFFFFF,
    )


class FrameRingHeader(TypedDict):
    magic: bytes
    version: int
    width: int
    height: int
    stride: int
    fourcc: bytes
    seq: int
    flags: int
    timestamp_100ns: int
    frame_counter: int


def unpack_header(buffer) -> FrameRingHeader:
    """Decode the ring header; the reference for the C++ reader contract."""

    (
        magic,
        version,
        width,
        height,
        stride,
        fourcc,
        seq,
        flags,
        timestamp_100ns,
        frame_counter,
    ) = struct.unpack_from(HEADER_FORMAT, buffer, 0)
    return {
        "magic": magic,
        "version": version,
        "width": width,
        "height": height,
        "stride": stride,
        "fourcc": fourcc,
        "seq": seq,
        "flags": flags,
        "timestamp_100ns": timestamp_100ns,
        "frame_counter": frame_counter,
    }


class FrameRingWriter:
    """Seqlock writer over a shared frame mapping.

    ``buffer`` is any writable buffer of at least :func:`ring_size` bytes — a
    named :class:`mmap.mmap` in production, a plain ``bytearray`` under test.
    The writer owns the header: it stamps the geometry once at construction
    and republishes it with every frame so a reader that attaches late (the
    Frame Server starts on consumer demand) always sees a complete header.
    """

    def __init__(self, buffer, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("frame ring dimensions must be positive")
        if len(buffer) < ring_size(width, height):
            raise ValueError(
                f"frame ring buffer holds {len(buffer)} bytes; "
                f"{width}x{height} BGRX needs {ring_size(width, height)}"
            )
        self._buffer = buffer
        self.width = width
        self.height = height
        self._seq = 0
        self._frame_counter = 0
        # Publish a valid, inactive header immediately so an early reader
        # never parses uninitialized memory.
        self._write_header(flags=FLAG_ACTIVE, timestamp_100ns=self._now_100ns())

    @staticmethod
    def _now_100ns() -> int:
        return time.perf_counter_ns() // 100

    def _write_header(self, *, flags: int, timestamp_100ns: int) -> None:
        header = _pack_header(
            self.width,
            self.height,
            seq=self._seq,
            flags=flags,
            timestamp_100ns=timestamp_100ns,
            frame_counter=self._frame_counter,
        )
        self._buffer[:HEADER_SIZE] = header

    def publish(self, frame_bgr: np.ndarray) -> None:
        """Publish one BGR frame (torn-write-safe for concurrent readers)."""

        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame is {frame_bgr.shape[1]}x{frame_bgr.shape[0]}, "
                f"ring is {self.width}x{self.height}"
            )
        bgrx = np.empty((self.height, self.width, BYTES_PER_PIXEL), dtype=np.uint8)
        bgrx[:, :, :3] = frame_bgr
        bgrx[:, :, 3] = 255

        # Seqlock: odd seq marks the write in progress; the full header is
        # rewritten after the payload so geometry/seq/counter stay coherent.
        self._seq += 1
        struct.pack_into("<I", self._buffer, _SEQ_OFFSET, self._seq & 0xFFFFFFFF)
        self._buffer[HEADER_SIZE : HEADER_SIZE + bgrx.nbytes] = bgrx.tobytes()
        self._seq += 1
        self._frame_counter += 1
        self._write_header(flags=FLAG_ACTIVE, timestamp_100ns=self._now_100ns())

    def mark_inactive(self) -> None:
        """Clear ``FLAG_ACTIVE`` so the reader shows its placeholder."""

        struct.pack_into("<I", self._buffer, _FLAGS_OFFSET, 0)


def read_latest_frame(buffer, *, max_attempts: int = 4) -> np.ndarray | None:
    """Reference reader used by tests; mirrors the C++ ``FrameRing.h`` logic.

    Returns the newest complete BGRX frame, or ``None`` when the ring is
    unpublished, inactive, torn beyond ``max_attempts``, or malformed.  The
    real consumer is the Media Foundation stream, which applies exactly this
    protocol before wrapping the bytes in an ``IMFSample``.
    """

    for _ in range(max_attempts):
        header = unpack_header(buffer)
        if header["magic"] != MAGIC or header["version"] != PROTOCOL_VERSION:
            return None
        if not header["flags"] & FLAG_ACTIVE:
            return None
        seq = header["seq"]
        if seq % 2:  # write in progress
            continue
        width = int(header["width"])
        height = int(header["height"])
        if width <= 0 or height <= 0 or len(buffer) < ring_size(width, height):
            return None
        payload = bytes(
            buffer[HEADER_SIZE : HEADER_SIZE + width * height * BYTES_PER_PIXEL]
        )
        if unpack_header(buffer)["seq"] != seq:  # torn: writer moved on
            continue
        return np.frombuffer(payload, dtype=np.uint8).reshape(
            (height, width, BYTES_PER_PIXEL)
        )
    return None


def open_shared_ring(width: int, height: int, name: str = SECTION_NAME) -> mmap.mmap:
    """Create/open the named page-file section shared with the media source.

    Windows-only: ``tagname`` maps to a named section object in the caller's
    session (``Local\\``), which the Frame Server media source opens read-only
    by the same name.  Raises on any other platform — the native camera is a
    Windows 11 feature and must never silently no-op elsewhere (CC-1 spirit).
    """

    if sys.platform != "win32":
        raise RuntimeError(
            "the native virtual camera frame ring requires Windows 11 "
            "(use output.backend 'pyvirtualcam' elsewhere)"
        )
    return mmap.mmap(-1, ring_size(width, height), tagname=name)


class NativeVirtualCameraOutput(VideoOutput):
    """`VideoOutput` publishing frames to the native MF virtual camera.

    The camera device itself is created and owned by the desktop shell
    (``VirtualCameraSession.cs``) with *session* lifetime: it exists only
    while the shell runs, so nothing persists to clean up beyond the COM
    registration the installer already removes (WIN-5.7 / Package.wxs).  The
    engine's only job is to keep the shared ring fresh; pacing stays with the
    pipeline (``paces = False``), because the media source resamples on its
    own negotiated clock.
    """

    paces = False
    fallback_active = False
    fallback_reason = ""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        section_name: str = SECTION_NAME,
        buffer=None,
    ) -> None:
        self._mapping = None
        if buffer is None:
            buffer = self._mapping = open_shared_ring(width, height, section_name)
        self._writer = FrameRingWriter(buffer, width, height)
        self.frames_sent = 0
        log.info(
            "native virtual camera ring ready: %s (%dx%d BGRX)",
            section_name,
            width,
            height,
        )

    def send(self, frame_bgr: np.ndarray) -> None:
        self._writer.publish(frame_bgr)
        self.frames_sent += 1

    def close(self) -> None:
        try:
            self._writer.mark_inactive()
        finally:
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
