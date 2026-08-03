"""Private, bounded matte evidence recording and offline replay.

The normal pipeline and :class:`~custback.hub.FrameHub` deliberately retain
only latest frames.  This module is the opt-in exception for local quality
investigations: it persists identifiable RGB pixels and silhouettes in a
private directory with an explicit duration and byte bound.

Bundle format ``custback.matte-replay`` version 1 is documented in
``docs/matte-replay-bundle.md``.  Metric-authoritative arrays use NumPy's
lossless ``.npy`` representation with pickle loading disabled.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import math
import os
import queue
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np

from . import _platform as platform_fs
from .color import ColorTransform, IDENTITY_TRANSFORM
from .compositor import composite
from .config import (
    AccelerationConfig,
    BlendSpace,
    SegmentationConfig,
)
from .segmentation import (
    SegmentationFrameContext,
    SegmentationTimeline,
    TemporalResetReason,
    create_segmenter,
    refiner_for,
)

BUNDLE_SCHEMA = "custback.matte-replay"
BUNDLE_VERSION = 1
REPLAY_SCHEMA = "custback.matte-replay-output"
REPLAY_VERSION = 1
DEFAULT_DURATION_S = 20.0
DEFAULT_MAX_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MIN_BUNDLE_BYTES = 64 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
_QUEUE_DEPTH = 1

CaptureMode = Literal["full", "composite_only"]
ReplayMode = Literal["frozen", "rerun", "reference"]
MaskStage = Literal["raw", "refined"]


class MatteDiagnosticsError(ValueError):
    """A recorder or replay bundle violates the diagnostic contract."""


@dataclass(frozen=True)
class MatteCaptureMetadata:
    """Frame-aligned identity for one unique camera input."""

    bundle_sequence: int
    capture_sequence: int
    capture_monotonic_ns: int
    timestamp_source: str
    capture_generation: int
    geometry_generation: int


@dataclass
class MatteFrameEvidence:
    """Arrays and scalar provenance retained until asynchronous persistence."""

    metadata: MatteCaptureMetadata
    raw_frame: np.ndarray
    raw_mask: np.ndarray | None = None
    refined_mask: np.ndarray | None = None
    clean_foreground: np.ndarray | None = None
    backdrop_frame: np.ndarray | None = None
    base_composite: np.ndarray | None = None
    configured_controls: dict[str, Any] = field(default_factory=dict)
    effective_controls: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    compositor_substages_ms: dict[str, float] = field(default_factory=dict)
    resource_samples: dict[str, int] = field(default_factory=dict)
    segmentation_diagnostics: dict[str, Any] = field(default_factory=dict)
    backdrop_identity: dict[str, Any] = field(default_factory=dict)
    color_transform: ColorTransform = IDENTITY_TRANSFORM
    matte_authoritative: bool = True
    insufficiency_reason: str = ""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _private_directory(path: Path, *, create: bool) -> None:
    path = Path(path)
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise MatteDiagnosticsError(
                "diagnostic output directory already exists"
            ) from exc
        platform_fs.chmod_private(path, 0o700)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise MatteDiagnosticsError("diagnostic directory does not exist") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise MatteDiagnosticsError("diagnostic path must be a real directory")
    if not platform_fs.stat_owner_matches(before):
        raise PermissionError("diagnostic directory has another owner")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = platform_fs.open_nofollow(path, flags, directory=True)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not platform_fs.owner_matches(descriptor)
        ):
            raise PermissionError("diagnostic directory identity is unsafe")
        if create:
            platform_fs.set_private_mode(descriptor, 0o700)
        elif not platform_fs.is_private_to_owner(descriptor):
            raise PermissionError("diagnostic directory is not owner-only")
    finally:
        os.close(descriptor)


def _private_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = platform_fs.open_nofollow(path, flags, 0o600)
    try:
        platform_fs.set_private_mode(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_private_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _private_write(temporary, payload)
        os.replace(temporary, path)
        platform_fs.fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _copy_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, np.ndarray) or value.size == 0:
        raise MatteDiagnosticsError("diagnostic artifacts must be non-empty arrays")
    if value.dtype.hasobject:
        raise MatteDiagnosticsError("object arrays are forbidden in replay bundles")
    return np.array(value, copy=True, order="C")


def _bgr_copy(
    value: np.ndarray | None,
    *,
    name: str,
    shape: tuple[int, int, int] | None = None,
) -> np.ndarray | None:
    copied = _copy_array(value)
    if copied is None:
        return None
    if (
        copied.dtype != np.uint8
        or copied.ndim != 3
        or copied.shape[2] != 3
        or (shape is not None and copied.shape != shape)
    ):
        raise MatteDiagnosticsError(
            f"{name} must be a matching non-empty uint8 BGR frame"
        )
    return copied


def _mask_copy(
    value: np.ndarray | None,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray | None:
    copied = _copy_array(value)
    if copied is None:
        return None
    if (
        copied.dtype != np.float32
        or copied.ndim != 2
        or copied.shape != shape
        or not bool(np.isfinite(copied).all())
        or float(np.min(copied)) < 0.0
        or float(np.max(copied)) > 1.0
    ):
        raise MatteDiagnosticsError(
            f"{name} must be matching finite float32 alpha in [0, 1]"
        )
    return copied


def _finite_timings(values: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise MatteDiagnosticsError("timings must be finite non-negative scalars")
        result[name] = float(value)
    return result


def process_rss_bytes() -> int | None:
    """Return process RSS using only operating-system/Python facilities.

    Linux exposes a current resident-page count through ``/proc``.  Other
    POSIX platforms fall back to ``getrusage``; that value is a peak rather
    than an instantaneous sample, but remains a truthful conservative RSS
    observation.  Unsupported platforms return ``None`` instead of inventing
    a zero.
    """

    if sys.platform.startswith("linux"):
        try:
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return value if sys.platform == "darwin" else value * 1024


def _resource_samples(values: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, value in values.items():
        if (
            name
            not in (
                "allocation_bytes",
                "memory_bytes",
                "rss_bytes",
                "vram_bytes",
            )
            or type(value) is not int
            or value < 0
        ):
            raise MatteDiagnosticsError(
                "resource samples must be supported non-negative byte counts"
            )
        result[name] = value
    return result


_SEGMENTATION_DIAGNOSTIC_KEYS = frozenset(
    {
        "backend",
        "input_frame_shape",
        "model_mask_shape",
        "output_mask_shape",
        "effective_timestamp_delta_ms",
        "timestamp_adjustment_count",
        "timestamp_adjustment_ms",
        "last_timestamp_adjusted",
        "resize_interpolation",
    }
)
_MASK_RESIZE_INTERPOLATIONS = frozenset({"none", "area", "linear", "area+linear"})


def _diagnostic_shape(value: object, *, name: str) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise MatteDiagnosticsError(
            f"segmentation diagnostic {name} must be a positive HxW shape"
        )
    return [int(value[0]), int(value[1])]


def _segmentation_diagnostics(value: object) -> dict[str, Any]:
    """Validate and canonicalize optional, content-free segmenter evidence."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MatteDiagnosticsError(
            "segmentation diagnostics must be an object when present"
        )
    if not value:
        return {}
    keys = set(value)
    if keys != _SEGMENTATION_DIAGNOSTIC_KEYS:
        raise MatteDiagnosticsError(
            "segmentation diagnostics have missing or unsupported fields"
        )
    if value.get("backend") != "mediapipe":
        raise MatteDiagnosticsError(
            "segmentation diagnostics identify an unsupported backend"
        )

    input_shape = _diagnostic_shape(
        value.get("input_frame_shape"),
        name="input_frame_shape",
    )
    model_shape = _diagnostic_shape(
        value.get("model_mask_shape"),
        name="model_mask_shape",
    )
    output_shape = _diagnostic_shape(
        value.get("output_mask_shape"),
        name="output_mask_shape",
    )
    if output_shape != input_shape:
        raise MatteDiagnosticsError(
            "segmentation diagnostic output mask shape must match the input frame"
        )

    delta_ms = value.get("effective_timestamp_delta_ms")
    if delta_ms is not None and (type(delta_ms) is not int or delta_ms <= 0):
        raise MatteDiagnosticsError(
            "segmentation diagnostic timestamp delta must be positive or null"
        )
    adjustment_count = value.get("timestamp_adjustment_count")
    adjustment_ms = value.get("timestamp_adjustment_ms")
    adjusted = value.get("last_timestamp_adjusted")
    if type(adjustment_count) is not int or adjustment_count < 0:
        raise MatteDiagnosticsError(
            "segmentation diagnostic adjustment count must be non-negative"
        )
    if type(adjustment_ms) is not int or adjustment_ms < 0:
        raise MatteDiagnosticsError(
            "segmentation diagnostic adjustment must be non-negative"
        )
    if type(adjusted) is not bool or adjusted is not (adjustment_ms > 0):
        raise MatteDiagnosticsError(
            "segmentation diagnostic adjustment flag is inconsistent"
        )
    if adjusted and adjustment_count == 0:
        raise MatteDiagnosticsError(
            "segmentation diagnostic adjusted frame requires a counted adjustment"
        )

    interpolation = value.get("resize_interpolation")
    if interpolation not in _MASK_RESIZE_INTERPOLATIONS:
        raise MatteDiagnosticsError(
            "segmentation diagnostic resize interpolation is invalid"
        )
    shrinking = any(
        source > target for source, target in zip(model_shape, output_shape)
    )
    growing = any(source < target for source, target in zip(model_shape, output_shape))
    expected_interpolation = (
        "area+linear"
        if shrinking and growing
        else "area"
        if shrinking
        else "linear"
        if growing
        else "none"
    )
    if interpolation != expected_interpolation:
        raise MatteDiagnosticsError(
            "segmentation diagnostic resize interpolation contradicts mask shapes"
        )

    return {
        "backend": "mediapipe",
        "input_frame_shape": input_shape,
        "model_mask_shape": model_shape,
        "output_mask_shape": output_shape,
        "effective_timestamp_delta_ms": delta_ms,
        "timestamp_adjustment_count": adjustment_count,
        "timestamp_adjustment_ms": adjustment_ms,
        "last_timestamp_adjusted": adjusted,
        "resize_interpolation": interpolation,
    }


def segmenter_diagnostics_snapshot(segmenter: object) -> dict[str, Any]:
    """Read one successful MediaPipe snapshot without persisting its clock epoch."""

    snapshotter = getattr(segmenter, "telemetry_snapshot", None)
    if not callable(snapshotter):
        return {}
    snapshot = snapshotter()

    def read(name: str) -> object:
        if isinstance(snapshot, dict):
            return snapshot.get(name)
        return getattr(snapshot, name, None)

    # ``effective_timestamp_ms`` is intentionally not selected: the private
    # bundle already records the capture clock, while this object only needs
    # its content-free delta and quantization observability.
    return _segmentation_diagnostics(
        {
            "backend": "mediapipe",
            "input_frame_shape": read("input_frame_shape"),
            "model_mask_shape": read("model_mask_shape"),
            "output_mask_shape": read("output_mask_shape"),
            "effective_timestamp_delta_ms": read("effective_timestamp_delta_ms"),
            "timestamp_adjustment_count": read("timestamp_adjustment_count"),
            "timestamp_adjustment_ms": read("timestamp_adjustment_ms"),
            "last_timestamp_adjusted": read("last_timestamp_adjusted"),
            "resize_interpolation": read("resize_interpolation"),
        }
    )


def _freeze_evidence(
    evidence: MatteFrameEvidence,
    final_composite: np.ndarray,
    capture_mode: CaptureMode,
) -> tuple[MatteFrameEvidence, np.ndarray]:
    metadata = evidence.metadata
    if (
        type(metadata.bundle_sequence) is not int
        or metadata.bundle_sequence < 0
        or type(metadata.capture_sequence) is not int
        or metadata.capture_sequence < 0
        or type(metadata.capture_monotonic_ns) is not int
        or metadata.capture_monotonic_ns < 0
    ):
        raise MatteDiagnosticsError("frame sequence and timestamp must be non-negative")
    final = _bgr_copy(final_composite, name="final composite")
    assert final is not None
    frame_shape = final.shape
    raw_frame = (
        np.empty((0,), dtype=np.uint8)
        if capture_mode == "composite_only"
        else _bgr_copy(evidence.raw_frame, name="raw frame", shape=frame_shape)
    )
    assert raw_frame is not None
    frozen = MatteFrameEvidence(
        metadata=metadata,
        raw_frame=raw_frame,
        raw_mask=(
            None
            if capture_mode == "composite_only"
            else _mask_copy(
                evidence.raw_mask,
                name="raw mask",
                shape=frame_shape[:2],
            )
        ),
        refined_mask=(
            None
            if capture_mode == "composite_only"
            else _mask_copy(
                evidence.refined_mask,
                name="refined mask",
                shape=frame_shape[:2],
            )
        ),
        clean_foreground=(
            None
            if capture_mode == "composite_only"
            else _bgr_copy(
                evidence.clean_foreground,
                name="clean foreground",
                shape=frame_shape,
            )
        ),
        backdrop_frame=(
            None
            if capture_mode == "composite_only"
            else _bgr_copy(
                evidence.backdrop_frame,
                name="backdrop frame",
                shape=frame_shape,
            )
        ),
        base_composite=(
            None
            if capture_mode == "composite_only"
            else _bgr_copy(
                evidence.base_composite,
                name="base composite",
                shape=frame_shape,
            )
        ),
        configured_controls=dict(evidence.configured_controls),
        effective_controls=dict(evidence.effective_controls),
        timings_ms=_finite_timings(evidence.timings_ms),
        compositor_substages_ms=_finite_timings(evidence.compositor_substages_ms),
        resource_samples=_resource_samples(evidence.resource_samples),
        segmentation_diagnostics=_segmentation_diagnostics(
            evidence.segmentation_diagnostics
        ),
        backdrop_identity=dict(evidence.backdrop_identity),
        color_transform=evidence.color_transform,
        matte_authoritative=(
            capture_mode == "full" and bool(evidence.matte_authoritative)
        ),
        insufficiency_reason=(
            "final-composite-only capture"
            if capture_mode == "composite_only"
            else evidence.insufficiency_reason
        ),
    )
    if capture_mode == "full":
        required = (
            frozen.raw_frame,
            frozen.raw_mask,
            frozen.refined_mask,
            frozen.backdrop_frame,
            frozen.base_composite,
        )
        if frozen.matte_authoritative and any(value is None for value in required):
            raise MatteDiagnosticsError(
                "metric-authoritative frames require raw, mask, backdrop, and base tracks"
            )
    return frozen, final


class MatteDiagnosticRecorder:
    """Asynchronous bounded writer for one opt-in private replay bundle."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        duration_s: float = DEFAULT_DURATION_S,
        max_bytes: int = DEFAULT_MAX_BYTES,
        capture_mode: CaptureMode = "full",
    ):
        if (
            isinstance(duration_s, bool)
            or not isinstance(duration_s, (int, float))
            or not math.isfinite(float(duration_s))
            or float(duration_s) <= 0.0
        ):
            raise MatteDiagnosticsError(
                "recording duration must be positive and finite"
            )
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < MIN_BUNDLE_BYTES
        ):
            raise MatteDiagnosticsError(
                f"recording byte limit must be at least {MIN_BUNDLE_BYTES} bytes"
            )
        if capture_mode not in ("full", "composite_only"):
            raise MatteDiagnosticsError("unknown matte diagnostic capture mode")

        self.output_dir = Path(output_dir)
        self.duration_s = float(duration_s)
        self.max_bytes = max_bytes
        self.capture_mode: CaptureMode = capture_mode
        _private_directory(self.output_dir, create=True)
        self._frames_dir = self.output_dir / "frames"
        _private_directory(self._frames_dir, create=True)

        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[MatteFrameEvidence, np.ndarray]] = queue.Queue(
            maxsize=_QUEUE_DEPTH
        )
        self._stop_requested = threading.Event()
        self._accepting = True
        self._closed = False
        self._first_timestamp_ns: int | None = None
        self._last_timestamp_ns: int | None = None
        self._last_sequence = -1
        self._last_output_timestamp_ns: int | None = None
        self._output_event_bytes = 0
        self._output_events: list[dict[str, Any]] = []
        self._stop_reason = ""
        self._error = ""
        self._artifact_bytes = 0
        self._manifest: dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "version": BUNDLE_VERSION,
            "state": "recording",
            "capture_mode": capture_mode,
            "matte_metrics_authoritative": capture_mode == "full",
            "timestamp_clock": "monotonic",
            "timestamp_semantics": (
                "capture-completion when supplied by the capture source; "
                "otherwise unique-frame dequeue"
            ),
            "limits": {
                "duration_s": self.duration_s,
                "max_bytes": self.max_bytes,
            },
            "extension_points": {
                "post_base_final_output_provenance": {
                    "version": 1,
                    "present": False,
                }
            },
            "output_timeline": {
                "version": 1,
                "timestamp_clock": "monotonic",
                "complete": True,
                "events": [],
            },
            "frames": [],
        }
        _atomic_private_write(
            self.output_dir / "manifest.partial.json",
            _json_bytes(self._manifest),
        )
        self._worker = threading.Thread(
            target=self._writer_loop,
            name="matte-diagnostic-writer",
            daemon=True,
        )
        self._worker.start()

    @property
    def wants_intermediates(self) -> bool:
        return self.capture_mode == "full"

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._accepting

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def _request_stop(self, reason: str) -> None:
        with self._lock:
            if self._accepting:
                self._accepting = False
            if not self._stop_reason or self._stop_reason == "closed":
                self._stop_reason = reason
        self._stop_requested.set()

    def submit(
        self,
        evidence: MatteFrameEvidence,
        final_composite: np.ndarray,
    ) -> bool:
        """Queue one unique input without waiting for disk I/O.

        A full queue disables the recorder instead of delaying the next live
        output.  The already-sent output contract therefore remains unchanged.
        """

        timestamp_ns = evidence.metadata.capture_monotonic_ns
        with self._lock:
            if not self._accepting or self._closed:
                return False
            if evidence.metadata.bundle_sequence <= self._last_sequence:
                self._accepting = False
                self._stop_reason = "non-monotonic-sequence"
                self._error = "diagnostic frame sequence was not strictly increasing"
                self._stop_requested.set()
                return False
            if (
                self._last_timestamp_ns is not None
                and timestamp_ns < self._last_timestamp_ns
            ):
                self._accepting = False
                self._stop_reason = "non-monotonic-timestamp"
                self._error = "diagnostic capture timestamp moved backwards"
                self._stop_requested.set()
                return False
            if self._first_timestamp_ns is None:
                self._first_timestamp_ns = timestamp_ns
            elif timestamp_ns - self._first_timestamp_ns >= int(
                self.duration_s * 1_000_000_000
            ):
                self._accepting = False
                self._stop_reason = "duration-limit"
                self._stop_requested.set()
                return False
            self._last_sequence = evidence.metadata.bundle_sequence
            self._last_timestamp_ns = timestamp_ns
        try:
            frozen = _freeze_evidence(evidence, final_composite, self.capture_mode)
            self._queue.put_nowait(frozen)
            return True
        except queue.Full:
            self._request_stop("writer-backpressure")
            return False
        except Exception as exc:
            with self._lock:
                self._error = type(exc).__name__
            self._request_stop("invalid-frame")
            return False

    def submit_output_event(
        self,
        *,
        sent_monotonic_ns: int,
        source_bundle_sequence: int,
        base_updated: bool,
        exact_final_repeat: bool,
    ) -> bool:
        """Record one successful output send as bounded scalar provenance.

        Output repeats remain separate from the unique model-input track.  The
        post-base field is a typed, isolated extension seam so later stages
        cannot change camera/matte/base metric names or semantics.
        """

        if (
            type(sent_monotonic_ns) is not int
            or sent_monotonic_ns < 0
            or type(source_bundle_sequence) is not int
            or source_bundle_sequence < 0
            or type(base_updated) is not bool
            or type(exact_final_repeat) is not bool
        ):
            self._request_stop("invalid-output-event")
            return False
        with self._lock:
            if not self._accepting or self._closed:
                return False
            if source_bundle_sequence > self._last_sequence:
                self._accepting = False
                self._stop_reason = "invalid-output-event"
                self._error = "output event referenced an unknown input"
                self._stop_requested.set()
                return False
            if (
                self._last_output_timestamp_ns is not None
                and sent_monotonic_ns < self._last_output_timestamp_ns
            ):
                self._accepting = False
                self._stop_reason = "non-monotonic-output-timestamp"
                self._error = "diagnostic output timestamp moved backwards"
                self._stop_requested.set()
                return False
            event = {
                "sequence": len(self._output_events),
                "sent_monotonic_ns": sent_monotonic_ns,
                "source_bundle_sequence": source_bundle_sequence,
                "base_composite_sequence": source_bundle_sequence,
                "base_updated": base_updated,
                "exact_final_repeat": exact_final_repeat,
                "post_base_final_output_provenance": None,
            }
            event_bytes = len(_json_bytes(event))
            if (
                self._artifact_bytes + self._output_event_bytes + event_bytes + 4096
                > self.max_bytes
            ):
                self._accepting = False
                self._stop_reason = "size-limit"
                self._stop_requested.set()
                return False
            self._output_events.append(event)
            self._output_event_bytes += event_bytes
            self._last_output_timestamp_ns = sent_monotonic_ns
        return True

    def _artifact_payloads(
        self,
        evidence: MatteFrameEvidence,
        final_composite: np.ndarray,
    ) -> dict[str, tuple[str, bytes]]:
        arrays: list[tuple[str, str, np.ndarray | None]] = [
            ("raw_frame", "raw.npy", evidence.raw_frame),
            ("raw_mask", "raw_mask.npy", evidence.raw_mask),
            ("refined_mask", "refined_mask.npy", evidence.refined_mask),
            ("clean_foreground", "clean_foreground.npy", evidence.clean_foreground),
            ("backdrop_frame", "backdrop.npy", evidence.backdrop_frame),
            ("base_composite", "base_composite.npy", evidence.base_composite),
            ("final_composite", "final_composite.npy", final_composite),
        ]
        return {
            key: (name, _npy_bytes(array))
            for key, name, array in arrays
            if array is not None and array.size > 0
        }

    @staticmethod
    def _artifact_descriptor(
        path: str,
        payload: bytes,
        array: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

    def _frame_entry(
        self,
        evidence: MatteFrameEvidence,
        artifacts: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = evidence.metadata
        transform = evidence.color_transform
        entry: dict[str, Any] = {
            "sequence": metadata.bundle_sequence,
            "capture_sequence": metadata.capture_sequence,
            "capture_monotonic_ns": metadata.capture_monotonic_ns,
            "timestamp_source": metadata.timestamp_source,
            "capture_generation": metadata.capture_generation,
            "geometry_generation": metadata.geometry_generation,
            "matte_metrics_authoritative": evidence.matte_authoritative,
            "insufficiency_reason": evidence.insufficiency_reason,
            "configured_controls": evidence.configured_controls,
            "effective_controls": evidence.effective_controls,
            "timings_ms": evidence.timings_ms,
            "compositor_substages_ms": evidence.compositor_substages_ms,
            "resource_samples": evidence.resource_samples,
            "backdrop_identity": evidence.backdrop_identity,
            "color_transform": {
                "exposure_ev": transform.exposure_ev,
                "wb_gains": list(transform.wb_gains),
            },
            "artifacts": artifacts,
            "post_base_final_output_provenance": None,
        }
        if evidence.segmentation_diagnostics:
            entry["segmentation_diagnostics"] = evidence.segmentation_diagnostics
        return entry

    def _write_artifact(self, path: Path, payload: bytes) -> None:
        """Test seam for one private, lossless artifact write."""

        _private_write(path, payload)

    def _persist_frame(
        self,
        evidence: MatteFrameEvidence,
        final_composite: np.ndarray,
    ) -> bool:
        payloads = self._artifact_payloads(evidence, final_composite)
        frame_name = f"{evidence.metadata.bundle_sequence:08d}"
        artifacts: dict[str, dict[str, Any]] = {}
        array_by_key = {
            "raw_frame": evidence.raw_frame,
            "raw_mask": evidence.raw_mask,
            "refined_mask": evidence.refined_mask,
            "clean_foreground": evidence.clean_foreground,
            "backdrop_frame": evidence.backdrop_frame,
            "base_composite": evidence.base_composite,
            "final_composite": final_composite,
        }
        for key, (filename, payload) in payloads.items():
            array = array_by_key[key]
            assert array is not None
            artifacts[key] = self._artifact_descriptor(
                f"frames/{frame_name}/{filename}",
                payload,
                array,
            )
        if "base_composite" in artifacts and np.array_equal(
            evidence.base_composite, final_composite
        ):
            payloads.pop("base_composite")
            artifacts["base_composite"] = {"alias_of": "final_composite"}

        entry = self._frame_entry(evidence, artifacts)
        frames = self._manifest["frames"]
        assert isinstance(frames, list)
        projected: dict[str, Any] = dict(self._manifest)
        projected["frames"] = [*frames, entry]
        projected["state"] = "complete"
        projected["stop_reason"] = "closed"
        projected["artifact_bytes"] = self._artifact_bytes + sum(
            len(payload) for _name, payload in payloads.values()
        )
        projected["frame_count"] = len(frames) + 1
        projected_bytes = _json_bytes(projected)
        payload_bytes = sum(len(payload) for _name, payload in payloads.values())
        with self._lock:
            output_event_bytes = self._output_event_bytes
        if (
            self._artifact_bytes
            + payload_bytes
            + output_event_bytes
            + len(projected_bytes)
            + 512
            > self.max_bytes
        ):
            self._request_stop("size-limit")
            return False

        temporary_dir = self._frames_dir / f".{frame_name}.{uuid.uuid4().hex}.tmp"
        _private_directory(temporary_dir, create=True)
        for _key, (filename, payload) in payloads.items():
            self._write_artifact(temporary_dir / filename, payload)
        final_dir = self._frames_dir / frame_name
        try:
            platform_fs.rename_noreplace(temporary_dir, final_dir)
        except OSError as exc:
            # The private root prevents an attacker from introducing the target;
            # retain a portable fallback for filesystems without renameat2.
            if exc.errno not in (errno.ENOSYS, errno.ENOTSUP, errno.EINVAL):
                raise
            if final_dir.exists() or final_dir.is_symlink():
                raise
            os.rename(temporary_dir, final_dir)
            platform_fs.fsync_dir(self._frames_dir)
        frames.append(entry)
        self._artifact_bytes += payload_bytes
        if not evidence.matte_authoritative:
            self._manifest["matte_metrics_authoritative"] = False
        _atomic_private_write(
            self.output_dir / "manifest.partial.json",
            _json_bytes(self._manifest),
        )
        return True

    def _finalize_manifest(self, *, incomplete: bool) -> None:
        frames = self._manifest["frames"]
        assert isinstance(frames, list)
        with self._lock:
            output_events = [
                dict(event)
                for event in self._output_events
                if int(event["source_bundle_sequence"]) < len(frames)
            ]
            all_output_events_written = len(output_events) == len(self._output_events)
        for sequence, event in enumerate(output_events):
            event["sequence"] = sequence
        timeline = {
            "version": 1,
            "timestamp_clock": "monotonic",
            "complete": all_output_events_written,
            "events": output_events,
        }
        self._manifest["output_timeline"] = timeline
        self._manifest["state"] = "incomplete" if incomplete else "complete"
        self._manifest["stop_reason"] = self._stop_reason or "closed"
        self._manifest["artifact_bytes"] = self._artifact_bytes
        self._manifest["frame_count"] = len(frames)
        if not frames:
            self._manifest["matte_metrics_authoritative"] = False
        # Exact final accounting wins over the conservative enqueue estimate.
        # Discard newest scalar events before invalidating persisted frame data.
        while (
            self._artifact_bytes + len(_json_bytes(self._manifest)) > self.max_bytes
            and output_events
        ):
            output_events.pop()
            timeline["complete"] = False
            self._manifest["stop_reason"] = "size-limit"
        _atomic_private_write(
            self.output_dir / "manifest.json",
            _json_bytes(self._manifest),
        )
        try:
            (self.output_dir / "manifest.partial.json").unlink()
        except FileNotFoundError:
            pass
        platform_fs.fsync_dir(self.output_dir)

    def _writer_loop(self) -> None:
        incomplete = False
        try:
            while True:
                if self._stop_requested.is_set() and self._queue.empty():
                    break
                try:
                    evidence, final = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if not self._persist_frame(evidence, final):
                        continue
                except Exception as exc:
                    incomplete = True
                    with self._lock:
                        self._error = type(exc).__name__
                        self._stop_reason = "write-error"
                        self._accepting = False
                    self._stop_requested.set()
                finally:
                    self._queue.task_done()
                if incomplete:
                    while True:
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            break
                        else:
                            self._queue.task_done()
                    break
        finally:
            try:
                self._finalize_manifest(incomplete=incomplete)
            except Exception as exc:
                with self._lock:
                    self._error = type(exc).__name__
                    self._stop_reason = "manifest-write-error"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                worker = self._worker
            else:
                self._closed = True
                self._accepting = False
                if not self._stop_reason:
                    self._stop_reason = "closed"
                self._stop_requested.set()
                worker = self._worker
        worker.join()

    def __enter__(self) -> "MatteDiagnosticRecorder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _read_private_file(path: Path, *, max_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise MatteDiagnosticsError("bundle artifact is missing") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise MatteDiagnosticsError("bundle artifacts must be regular files")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = platform_fs.open_nofollow(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not platform_fs.owner_matches(descriptor)
            or not platform_fs.is_private_to_owner(descriptor)
        ):
            raise PermissionError("bundle artifact is not private to its owner")
        if opened.st_size > max_bytes:
            raise MatteDiagnosticsError("bundle artifact exceeds its declared bound")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise MatteDiagnosticsError("bundle artifact is truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise MatteDiagnosticsError("bundle artifact path is malformed")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.parts[0] != "frames"
        or len(relative.parts) != 3
    ):
        raise MatteDiagnosticsError("bundle artifact path escapes the bundle")
    return relative


class MatteReplayBundle:
    """Validated read-only view of a complete private version-1 bundle."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        _private_directory(self.root, create=False)
        manifest_path = self.root / "manifest.json"
        if (
            not manifest_path.exists()
            and (self.root / "manifest.partial.json").exists()
        ):
            raise MatteDiagnosticsError("bundle recording was interrupted")
        payload = _read_private_file(manifest_path, max_bytes=MAX_MANIFEST_BYTES)
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MatteDiagnosticsError("bundle manifest is malformed") from exc
        if not isinstance(manifest, dict):
            raise MatteDiagnosticsError("bundle manifest must be an object")
        if (
            manifest.get("schema") != BUNDLE_SCHEMA
            or manifest.get("version") != BUNDLE_VERSION
        ):
            raise MatteDiagnosticsError("unsupported matte replay bundle version")
        if manifest.get("state") != "complete":
            raise MatteDiagnosticsError("bundle is incomplete")
        capture_mode = manifest.get("capture_mode")
        if capture_mode not in ("full", "composite_only"):
            raise MatteDiagnosticsError("bundle capture mode is invalid")
        limits = manifest.get("limits")
        if not isinstance(limits, dict):
            raise MatteDiagnosticsError("bundle limits are missing")
        duration_s = limits.get("duration_s")
        max_bytes = limits.get("max_bytes")
        if (
            isinstance(duration_s, bool)
            or not isinstance(duration_s, (int, float))
            or not math.isfinite(float(duration_s))
            or float(duration_s) <= 0.0
            or type(max_bytes) is not int
            or max_bytes < MIN_BUNDLE_BYTES
        ):
            raise MatteDiagnosticsError("bundle limits are invalid")
        frames = manifest.get("frames")
        if not isinstance(frames, list):
            raise MatteDiagnosticsError("bundle frames must be a list")
        if len(frames) > 100_000:
            raise MatteDiagnosticsError("bundle contains too many frames")
        if manifest.get("frame_count") != len(frames):
            raise MatteDiagnosticsError("bundle frame count does not match manifest")
        output_events = self._validate_output_timeline(
            manifest.get("output_timeline"),
            frame_count=len(frames),
        )
        self.manifest = manifest
        validated_frames: list[dict[str, Any]] = []
        previous_timestamp = -1
        for index, frame in enumerate(frames):
            validated = self._validate_frame(frame, index)
            timestamp = validated["capture_monotonic_ns"]
            assert isinstance(timestamp, int)
            if timestamp < previous_timestamp:
                raise MatteDiagnosticsError("bundle timestamps are not monotonic")
            previous_timestamp = timestamp
            validated_frames.append(validated)
        if len(validated_frames) >= 2 and validated_frames[-1][
            "capture_monotonic_ns"
        ] - validated_frames[0]["capture_monotonic_ns"] >= int(
            float(duration_s) * 1_000_000_000
        ):
            raise MatteDiagnosticsError("bundle timestamps exceed the duration limit")
        declared_artifact_bytes = manifest.get("artifact_bytes")
        artifact_bytes = 0
        for frame in validated_frames:
            artifacts = frame["artifacts"]
            assert isinstance(artifacts, dict)
            for descriptor in artifacts.values():
                assert isinstance(descriptor, dict)
                if "alias_of" not in descriptor:
                    byte_count = descriptor.get("bytes")
                    assert isinstance(byte_count, int)
                    artifact_bytes += byte_count
        assert isinstance(max_bytes, int)
        if (
            declared_artifact_bytes != artifact_bytes
            or artifact_bytes + len(payload) > max_bytes
        ):
            raise MatteDiagnosticsError("bundle byte accounting is invalid")
        self.frames = tuple(validated_frames)
        self.output_events = tuple(output_events)

    @staticmethod
    def _validate_output_timeline(
        value: object,
        *,
        frame_count: int,
    ) -> list[dict[str, Any]]:
        # Version-1 bundles created before MATTE-0.2 have no send timeline.
        if value is None:
            return []
        if not isinstance(value, dict) or value.get("version") != 1:
            raise MatteDiagnosticsError("bundle output timeline is invalid")
        if value.get("timestamp_clock") != "monotonic":
            raise MatteDiagnosticsError("bundle output timestamp clock is invalid")
        if type(value.get("complete")) is not bool:
            raise MatteDiagnosticsError(
                "bundle output timeline completeness is invalid"
            )
        events = value.get("events")
        if not isinstance(events, list) or len(events) > 1_000_000:
            raise MatteDiagnosticsError("bundle output events are invalid")
        validated: list[dict[str, Any]] = []
        previous_timestamp = -1
        previous_source = -1
        for index, event in enumerate(events):
            if not isinstance(event, dict) or event.get("sequence") != index:
                raise MatteDiagnosticsError("bundle output event order is invalid")
            timestamp = event.get("sent_monotonic_ns")
            source = event.get("source_bundle_sequence")
            base = event.get("base_composite_sequence")
            if (
                type(timestamp) is not int
                or timestamp < previous_timestamp
                or type(source) is not int
                or source < 0
                or source >= frame_count
                or source < previous_source
                or base != source
                or type(event.get("base_updated")) is not bool
                or type(event.get("exact_final_repeat")) is not bool
            ):
                raise MatteDiagnosticsError("bundle output event is invalid")
            post_base = event.get("post_base_final_output_provenance")
            if post_base is not None and (
                not isinstance(post_base, dict)
                or post_base.get("schema")
                != "custback.matte-post-base-output-provenance"
                or post_base.get("version") != 1
                or not isinstance(post_base.get("stage"), str)
                or not isinstance(post_base.get("metrics"), dict)
            ):
                raise MatteDiagnosticsError(
                    "bundle post-base output provenance is invalid"
                )
            previous_timestamp = timestamp
            previous_source = source
            validated.append(event)
        return validated

    @staticmethod
    def _validate_frame(value: object, index: int) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise MatteDiagnosticsError("bundle frame entry must be an object")
        if value.get("sequence") != index:
            raise MatteDiagnosticsError("bundle frame order is not contiguous")
        timestamp = value.get("capture_monotonic_ns")
        if type(timestamp) is not int or timestamp < 0:
            raise MatteDiagnosticsError("bundle frame timestamp is invalid")
        if value.get("timestamp_source") not in (
            "capture-completion",
            "unique-frame-dequeue",
        ):
            raise MatteDiagnosticsError("bundle timestamp source is invalid")
        for name in (
            "capture_sequence",
            "capture_generation",
            "geometry_generation",
        ):
            scalar = value.get(name)
            if type(scalar) is not int or scalar < 0:
                raise MatteDiagnosticsError(f"bundle {name} is invalid")
        resources = value.get("resource_samples", {})
        if not isinstance(resources, dict):
            raise MatteDiagnosticsError("bundle resource samples are invalid")
        _resource_samples(cast(dict[str, int], resources))
        _segmentation_diagnostics(value.get("segmentation_diagnostics"))
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, dict) or "final_composite" not in artifacts:
            raise MatteDiagnosticsError("bundle frame artifacts are incomplete")
        paths: set[PurePosixPath] = set()
        for key, descriptor in artifacts.items():
            if not isinstance(key, str) or not isinstance(descriptor, dict):
                raise MatteDiagnosticsError("bundle artifact descriptor is malformed")
            if "alias_of" in descriptor:
                alias = descriptor.get("alias_of")
                if (
                    not isinstance(alias, str)
                    or alias == key
                    or alias not in artifacts
                    or "alias_of" in artifacts[alias]
                ):
                    raise MatteDiagnosticsError("bundle artifact alias is invalid")
                continue
            relative = _safe_relative_path(descriptor.get("path"))
            if relative in paths:
                raise MatteDiagnosticsError("bundle artifact path is duplicated")
            paths.add(relative)
            if (
                type(descriptor.get("bytes")) is not int
                or int(descriptor["bytes"]) <= 0
                or not isinstance(descriptor.get("sha256"), str)
                or len(str(descriptor["sha256"])) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(descriptor["sha256"])
                )
                or not isinstance(descriptor.get("shape"), list)
                or not isinstance(descriptor.get("dtype"), str)
            ):
                raise MatteDiagnosticsError("bundle artifact metadata is invalid")
        return value

    def _descriptor(self, frame: dict[str, Any], key: str) -> dict[str, Any]:
        artifacts = frame["artifacts"]
        assert isinstance(artifacts, dict)
        descriptor = artifacts.get(key)
        if not isinstance(descriptor, dict):
            raise MatteDiagnosticsError(f"bundle has no {key} track")
        alias = descriptor.get("alias_of")
        if isinstance(alias, str):
            target = artifacts.get(alias)
            if not isinstance(target, dict):
                raise MatteDiagnosticsError("bundle artifact alias target is missing")
            descriptor = target
        return descriptor

    def load_array(self, frame: dict[str, Any], key: str) -> np.ndarray:
        descriptor = self._descriptor(frame, key)
        relative = _safe_relative_path(descriptor.get("path"))
        path = self.root.joinpath(*relative.parts)
        _private_directory(path.parent, create=False)
        expected_value = descriptor.get("bytes")
        if (
            type(expected_value) is not int
            or expected_value <= 0
            or expected_value > MAX_ARTIFACT_BYTES
        ):
            raise MatteDiagnosticsError("bundle artifact byte count is invalid")
        expected_bytes = expected_value
        dtype_value = descriptor.get("dtype")
        shape_value = descriptor.get("shape")
        is_mask = key in {"raw_mask", "refined_mask"}
        expected_dtype = np.dtype(np.float32 if is_mask else np.uint8)
        expected_dimensions = 2 if is_mask else 3
        if (
            not isinstance(dtype_value, str)
            or not isinstance(shape_value, list)
            or len(shape_value) != expected_dimensions
            or any(
                type(dimension) is not int or dimension <= 0
                for dimension in shape_value
            )
            or (not is_mask and shape_value[2] != 3)
        ):
            raise MatteDiagnosticsError("bundle array shape contract is invalid")
        try:
            declared_dtype = np.dtype(dtype_value)
        except TypeError as exc:
            raise MatteDiagnosticsError("bundle array dtype is invalid") from exc
        if declared_dtype != expected_dtype:
            raise MatteDiagnosticsError("bundle array dtype contract is invalid")
        element_count = math.prod(shape_value)
        if element_count * declared_dtype.itemsize >= expected_bytes:
            # A valid .npy file is strictly larger than its raw data payload.
            raise MatteDiagnosticsError("bundle array dimensions exceed artifact bytes")
        payload = _read_private_file(path, max_bytes=expected_bytes)
        if len(payload) != expected_bytes:
            raise MatteDiagnosticsError("bundle artifact size does not match manifest")
        if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise MatteDiagnosticsError(
                "bundle artifact digest does not match manifest"
            )
        try:
            array = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise MatteDiagnosticsError("bundle NumPy artifact is malformed") from exc
        if (
            not isinstance(array, np.ndarray)
            or array.dtype.str != descriptor["dtype"]
            or list(array.shape) != descriptor["shape"]
            or array.size == 0
            or array.dtype.hasobject
        ):
            raise MatteDiagnosticsError("bundle array contract does not match manifest")
        return np.ascontiguousarray(array)


@dataclass(frozen=True)
class ReplayOptions:
    mode: ReplayMode = "frozen"
    mask_stage: MaskStage = "refined"
    light_wrap: float | None = None
    blend_space: str | None = None
    model_foreground: Literal["recorded", "on", "off"] = "recorded"
    realtime: bool = False


def _recorded_transform(frame: dict[str, Any]) -> ColorTransform:
    raw = frame.get("color_transform")
    if not isinstance(raw, dict):
        return IDENTITY_TRANSFORM
    gains = raw.get("wb_gains")
    if not isinstance(gains, list) or len(gains) != 3:
        raise MatteDiagnosticsError("recorded color transform is malformed")
    gain_values = (float(gains[0]), float(gains[1]), float(gains[2]))
    return ColorTransform(
        exposure_ev=float(raw.get("exposure_ev", 0.0)),
        wb_gains=gain_values,
    )


def _controls(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    configured = frame.get("configured_controls")
    if not isinstance(configured, dict):
        raise MatteDiagnosticsError("recorded configured controls are missing")
    segmentation = configured.get("segmentation")
    compositing = configured.get("compositing")
    if not isinstance(segmentation, dict) or not isinstance(compositing, dict):
        raise MatteDiagnosticsError("recorded matte controls are malformed")
    return segmentation, compositing


def _private_replay_output(output_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    _private_directory(output_dir, create=True)
    frames_dir = output_dir / "frames"
    _private_directory(frames_dir, create=True)
    return frames_dir, []


def replay_bundle(
    bundle_dir: Path | str,
    output_dir: Path | str,
    *,
    options: ReplayOptions | None = None,
) -> dict[str, Any]:
    """Replay a bundle without opening live capture or an output consumer."""

    selected = options or ReplayOptions()
    if selected.mode not in ("frozen", "rerun", "reference"):
        raise MatteDiagnosticsError("unknown replay mode")
    if selected.mask_stage not in ("raw", "refined"):
        raise MatteDiagnosticsError("unknown frozen mask stage")
    if selected.blend_space not in (None, "srgb_legacy", "linear_srgb"):
        raise MatteDiagnosticsError("unknown replay blend space")
    if selected.light_wrap is not None and (
        not math.isfinite(selected.light_wrap) or not 0.0 <= selected.light_wrap <= 1.0
    ):
        raise MatteDiagnosticsError("replay light wrap must be in [0, 1]")

    bundle = MatteReplayBundle(bundle_dir)
    if selected.mode in ("frozen", "rerun") and not bool(
        bundle.manifest.get("matte_metrics_authoritative")
    ):
        raise MatteDiagnosticsError(
            "final-composite-only bundles cannot run matte attribution replay"
        )
    output_path = Path(output_dir)
    frames_dir, results = _private_replay_output(output_path)
    attribution_variant = bool(
        selected.mask_stage != "refined"
        or selected.light_wrap is not None
        or selected.blend_space is not None
        or selected.model_foreground != "recorded"
    )
    segmenter: Any = None
    refiner: Any = None
    segmentation_timeline: SegmentationTimeline | None = None
    segmenter_controls_key = b""
    first_timestamp: int | None = None
    replay_started = time.monotonic_ns()
    try:
        for index, frame in enumerate(bundle.frames):
            timestamp_value = frame.get("capture_monotonic_ns")
            if type(timestamp_value) is not int:
                raise MatteDiagnosticsError("recorded timestamp is invalid")
            timestamp = timestamp_value
            if first_timestamp is None:
                first_timestamp = timestamp
            if selected.realtime and first_timestamp is not None:
                target_ns = replay_started + (timestamp - first_timestamp)
                delay_s = (target_ns - time.monotonic_ns()) / 1_000_000_000
                if delay_s > 0.0:
                    time.sleep(delay_s)

            reference = bundle.load_array(frame, "final_composite")
            reference_tolerance: int | None = None
            raw_mask: np.ndarray | None = None
            refined_mask: np.ndarray | None = None
            recorded_segmentation_diagnostics = _segmentation_diagnostics(
                frame.get("segmentation_diagnostics")
            )
            rerun_segmentation_diagnostics: dict[str, Any] = {}
            if selected.mode == "reference":
                rendered = reference.copy()
            else:
                raw = bundle.load_array(frame, "raw_frame")
                backdrop = bundle.load_array(frame, "backdrop_frame")
                segmentation_values, compositing_values = _controls(frame)
                if selected.mode == "frozen" and not attribution_variant:
                    correction = compositing_values.get("color_correction")
                    correction_mode = (
                        correction.get("mode")
                        if isinstance(correction, dict)
                        else "off"
                    )
                    reference_tolerance = 0 if correction_mode == "off" else 1
                if selected.mode == "rerun":
                    context = SegmentationFrameContext(
                        sequence=int(frame["capture_sequence"]),
                        timestamp_ns=timestamp,
                        generation=int(frame["capture_generation"]),
                        geometry_generation=int(frame["geometry_generation"]),
                        shape=raw.shape[:2],
                    )
                    configured_values = frame.get("configured_controls")
                    if not isinstance(configured_values, dict):
                        raise MatteDiagnosticsError(
                            "recorded configured controls are missing"
                        )
                    acceleration_values = configured_values.get("acceleration", {})
                    controls_key = _json_bytes(
                        {
                            "segmentation": segmentation_values,
                            "acceleration": acceleration_values,
                        }
                    )
                    if segmenter is None or controls_key != segmenter_controls_key:
                        replacing_segmenter = segmenter is not None
                        if segmenter is not None:
                            segmenter.close()
                        segmentation_cfg = SegmentationConfig.model_validate(
                            segmentation_values
                        )
                        acceleration_cfg = AccelerationConfig.model_validate(
                            acceleration_values
                        )
                        segmenter = create_segmenter(
                            segmentation_cfg,
                            acceleration=acceleration_cfg,
                        )
                        refiner = refiner_for(segmentation_cfg, segmenter)
                        segmenter_controls_key = controls_key
                        if segmentation_timeline is None:
                            segmentation_timeline = SegmentationTimeline()
                        elif replacing_segmenter:
                            segmentation_timeline.request_reset(
                                TemporalResetReason.SEGMENTATION_CONFIG
                            )
                    assert segmentation_timeline is not None
                    boundary = segmentation_timeline.observe(context)
                    if boundary.reset_reason is not None:
                        segmenter.reset_temporal_state(
                            boundary.reset_reason,
                            context.timestamp_ns,
                        )
                        refiner.reset_temporal_state(
                            boundary.reset_reason,
                            context.timestamp_ns,
                        )
                    reset_count_before = int(
                        getattr(segmenter, "temporal_reset_count", 0) or 0
                    )
                    raw_result = segmenter.segment(raw, context=context)
                    raw_mask = _mask_copy(
                        raw_result,
                        name="rerun raw mask",
                        shape=raw.shape[:2],
                    )
                    assert raw_mask is not None
                    reset_count_after = int(
                        getattr(segmenter, "temporal_reset_count", 0) or 0
                    )
                    reset_reason = getattr(
                        segmenter, "last_temporal_reset_reason", None
                    )
                    if reset_count_after > reset_count_before and isinstance(
                        reset_reason, TemporalResetReason
                    ):
                        refiner.reset_temporal_state(
                            reset_reason,
                            context.timestamp_ns,
                        )
                        if reset_reason is TemporalResetReason.BACKEND_RECOVERY:
                            segmentation_timeline.record_reset(
                                TemporalResetReason.BACKEND_RECOVERY
                            )
                    rerun_segmentation_diagnostics = segmenter_diagnostics_snapshot(
                        segmenter
                    )
                    refined_mask = np.ascontiguousarray(
                        refiner.refine(raw_mask, raw, context=context),
                        dtype=np.float32,
                    )
                    clean_foreground = getattr(segmenter, "last_foreground", None)
                else:
                    raw_mask = bundle.load_array(frame, "raw_mask")
                    refined_mask = bundle.load_array(frame, "refined_mask")
                    artifacts = frame.get("artifacts")
                    if isinstance(artifacts, dict) and "clean_foreground" in artifacts:
                        clean_foreground = bundle.load_array(frame, "clean_foreground")
                    else:
                        clean_foreground = None

                mask = raw_mask if selected.mask_stage == "raw" else refined_mask
                assert mask is not None
                recorded_use_foreground = bool(
                    compositing_values.get("use_model_foreground", False)
                )
                use_foreground = (
                    recorded_use_foreground
                    if selected.model_foreground == "recorded"
                    else selected.model_foreground == "on"
                )
                if selected.light_wrap is None:
                    recorded_light_wrap = compositing_values.get("light_wrap", 0.0)
                    if not isinstance(recorded_light_wrap, (int, float)):
                        raise MatteDiagnosticsError(
                            "recorded compositor light wrap is invalid"
                        )
                    light_wrap = float(recorded_light_wrap)
                else:
                    light_wrap = selected.light_wrap
                blend_space_value = (
                    str(compositing_values.get("blend_space", "srgb_legacy"))
                    if selected.blend_space is None
                    else selected.blend_space
                )
                if blend_space_value not in ("srgb_legacy", "linear_srgb"):
                    raise MatteDiagnosticsError(
                        "recorded compositor blend space is invalid"
                    )
                blend_space = cast(BlendSpace, blend_space_value)
                rendered = composite(
                    raw,
                    backdrop,
                    np.ascontiguousarray(mask, dtype=np.float32),
                    light_wrap=light_wrap,
                    edge_foreground=clean_foreground if use_foreground else None,
                    blend_space=blend_space,
                    color_transform=_recorded_transform(frame),
                )

            frame_dir = frames_dir / f"{index:08d}"
            _private_directory(frame_dir, create=True)
            output_payload = _npy_bytes(rendered)
            _private_write(frame_dir / "composite.npy", output_payload)
            delta = np.abs(rendered.astype(np.int16) - reference.astype(np.int16))
            maximum_delta = int(delta.max(initial=0))
            replay_frame: dict[str, Any] = {
                "sequence": index,
                "capture_sequence": frame.get("capture_sequence"),
                "capture_monotonic_ns": timestamp,
                "output": f"frames/{index:08d}/composite.npy",
                "output_sha256": hashlib.sha256(output_payload).hexdigest(),
                "reference_exact": bool(np.array_equal(rendered, reference)),
                "reference_max_channel_delta": maximum_delta,
                "reference_tolerance": reference_tolerance,
                "reference_within_tolerance": (
                    None
                    if reference_tolerance is None
                    else maximum_delta <= reference_tolerance
                ),
            }
            if recorded_segmentation_diagnostics or rerun_segmentation_diagnostics:
                replay_frame["segmentation_diagnostics"] = {
                    "recorded": recorded_segmentation_diagnostics or None,
                    "rerun": rerun_segmentation_diagnostics or None,
                }
            results.append(replay_frame)
    finally:
        if segmenter is not None:
            segmenter.close()

    report: dict[str, Any] = {
        "schema": REPLAY_SCHEMA,
        "version": REPLAY_VERSION,
        "source_bundle_schema": BUNDLE_SCHEMA,
        "source_bundle_version": BUNDLE_VERSION,
        "mode": selected.mode,
        "mask_stage": selected.mask_stage,
        "attribution_variant": attribution_variant,
        "baseline_reproduction_passed": (
            all(frame["reference_within_tolerance"] for frame in results)
            if selected.mode == "frozen" and not attribution_variant
            else None
        ),
        "variant": {
            "light_wrap": selected.light_wrap,
            "blend_space": selected.blend_space,
            "model_foreground": selected.model_foreground,
        },
        "timestamp_handling": (
            "paced from recorded monotonic deltas"
            if selected.realtime
            else "metadata retained; replay executed without wall-clock pacing"
        ),
        "frames": results,
    }
    _atomic_private_write(output_path / "replay.json", _json_bytes(report))
    return report


def build_replay_parser(
    *, prog: str = "custback matte-replay"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Replay a private matte diagnostic bundle without live capture",
    )
    parser.add_argument("bundle", help="complete private replay bundle directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new explicit private directory for replay outputs",
    )
    parser.add_argument(
        "--mode",
        choices=("frozen", "rerun", "reference"),
        default="frozen",
        help="frozen intermediates, model rerun, or final-composite extraction",
    )
    parser.add_argument(
        "--mask-stage",
        choices=("raw", "refined"),
        default="refined",
        help="recorded/rerun alpha stage supplied to the compositor",
    )
    parser.add_argument(
        "--light-wrap",
        type=float,
        help="override recorded light wrap with a value from 0 to 1",
    )
    parser.add_argument(
        "--blend-space",
        choices=("srgb_legacy", "linear_srgb"),
        help="override the recorded compositor blend space",
    )
    parser.add_argument(
        "--model-foreground",
        choices=("recorded", "on", "off"),
        default="recorded",
        help="retain or override clean-foreground edge replacement",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace unique inputs using recorded monotonic timestamp deltas",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback matte-replay",
) -> int:
    args = build_replay_parser(prog=prog).parse_args(argv)
    try:
        report = replay_bundle(
            args.bundle,
            args.output_dir,
            options=ReplayOptions(
                mode=args.mode,
                mask_stage=args.mask_stage,
                light_wrap=args.light_wrap,
                blend_space=args.blend_space,
                model_foreground=args.model_foreground,
                realtime=args.realtime,
            ),
        )
    except (OSError, MatteDiagnosticsError, ValueError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2
    print(f"replayed {len(report['frames'])} unique frame(s) into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
