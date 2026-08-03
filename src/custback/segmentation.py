"""Person segmentation backends.

Produces a float32 mask in [0, 1] with the same HxW as the input frame,
where 1.0 = person (keep), 0.0 = background (replace).

Backends:
  - rvm:        Robust Video Matting via onnxruntime. True alpha matting
                (hair-level edges) with a recurrent temporal state, plus a
                clean-foreground prediction used to remove background color
                spill. Runs on NVIDIA GPUs (CUDA) when onnxruntime-gpu is
                installed; CPU otherwise. Install: custback[rvm] or [gpu].
  - mediapipe:  MediaPipe Tasks ImageSegmenter (selfie segmentation model).
                Production quality, real-time on CPU; optional GPU delegate.
  - heuristic:  brightness/center-prior fallback so the pipeline still works
                without ML dependencies (and in tests / CI).
  - none:       full-frame mask (everything is "person") -> passthrough.

"auto" picks the best available: rvm > mediapipe > heuristic.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import os
import tempfile
import time
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Literal, cast

import numpy as np

from . import _platform as platform_fs
from .acceleration import (
    CPU_PROVIDER,
    AccelerationState,
    GpuRequiredError,
    preload_acceleration_dlls,
    provider_label,
    prove_rvm_provider,
    resolve_provider_candidates,
    warm_up_session,
)
from .config import (
    AccelerationConfig,
    CompositingConfig,
    SegmentationConfig,
    SpatialEdgeRefinementConfig,
    spatial_edge_refinement_radius,
)
from .matte_policy import MatteBackendKind, resolve_matte_policy

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

# OpenCV and the ML runtimes below are native/dynamically generated APIs. Keep
# the deliberate runtime fallback, but do not model their implementation
# details throughout the segmentation pipeline.
cv2: Any = _cv2

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Immutable identity for a model managed by custback."""

    backend: str
    url: str
    filename: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SegmenterPreparation:
    """ML backends whose dependencies and model bytes passed startup checks."""

    ready_backends: frozenset[str] = frozenset()


SEGMENTATION_TIMESTAMP_GAP_RESET_NS = 1_000_000_000
"""Reset temporal matte state after more than one second without an input.

One second is fifteen ordinary frame intervals even at 15 FPS, so normal
capture jitter and latest-slot overwrites remain continuous. It is still below
the real-camera stall/recovery boundary and prevents state from blending
across a visible acquisition outage.
"""


class TemporalResetReason(str, Enum):
    """Why temporal segmentation state was cleared before a boundary frame."""

    INITIAL = "initial"
    CAPTURE_GENERATION = "capture-generation-change"
    GEOMETRY = "geometry-change"
    NON_MONOTONIC_TIMESTAMP = "non-monotonic-timestamp"
    TIMESTAMP_GAP = "timestamp-gap"
    BACKEND_RECOVERY = "backend-recovery"
    SEGMENTATION_CONFIG = "segmentation-generation-change"


@dataclass(frozen=True)
class SegmentationFrameContext:
    """Capture identity supplied atomically with one segmentation input."""

    sequence: int
    timestamp_ns: int
    generation: int
    geometry_generation: int
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        """Reject malformed identity before a timeline can mutate its state."""

        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("segmentation sequence must be a non-negative integer")
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("segmentation timestamp must be a non-negative integer")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError(
                "segmentation capture generation must be a non-negative integer"
            )
        if type(self.geometry_generation) is not int or self.geometry_generation < 0:
            raise ValueError(
                "segmentation geometry generation must be a non-negative integer"
            )
        if (
            type(self.shape) is not tuple
            or len(self.shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.shape)
        ):
            raise ValueError(
                "segmentation frame shape must contain two positive integers"
            )


@dataclass(frozen=True)
class TemporalBoundary:
    """One timeline decision made before processing the associated frame."""

    reset_reason: TemporalResetReason | None
    elapsed_ns: int | None
    sequence_gap: int


@dataclass(frozen=True)
class SegmentationTimelineSnapshot:
    """Content-free temporal diagnostics for the current resource generation."""

    reset_count: int
    last_reset_reason: TemporalResetReason | None
    sequence_gap_events: int
    sequence_gap_frames: int
    last_sequence: int | None


@dataclass(frozen=True)
class MediaPipeTelemetry:
    """Content-free diagnostics for the last successful MediaPipe inference.

    Shapes use NumPy's ``(height, width)`` order.  The effective timestamp is
    useful to focused local tests, while persisted replay diagnostics omit it
    because the capture metadata already carries the source timestamp.
    """

    input_frame_shape: tuple[int, int] | None
    model_mask_shape: tuple[int, int] | None
    output_mask_shape: tuple[int, int] | None
    effective_timestamp_ms: int | None
    effective_timestamp_delta_ms: int | None
    timestamp_adjustment_count: int
    timestamp_adjustment_ms: int
    last_timestamp_adjusted: bool
    resize_interpolation: str | None


@dataclass(frozen=True)
class RVMTelemetry:
    """Content-free facts from the last successful RVM inference.

    Frame-derived fields remain ``None`` until a complete inference result has
    been validated and converted. Model and configured-detail identity remain
    available across temporal resets without retaining pixels, alpha, recurrent
    tensors, paths, or exception text.
    """

    input_frame_shape: tuple[int, int] | None
    output_alpha_shape: tuple[int, int] | None
    output_foreground_shape: tuple[int, int, int] | None
    configured_downsample_mode: Literal["auto", "explicit"]
    configured_downsample_ratio: float
    resolved_downsample_ratio: float | None
    preprocess_ms: float | None
    session_run_ms: float | None
    postprocess_ms: float | None
    model_builtin: bool
    model_identity: str
    model_sha256: str | None
    model_bytes: int | None
    acceleration_state: str
    acceleration_active_provider: str
    acceleration_fallback_active: bool
    acceleration_fallback_count: int


@dataclass(frozen=True)
class SegmentationTimelineCheckpoint:
    """Private-state checkpoint used to make boundary processing atomic."""

    last_context: SegmentationFrameContext | None
    pending_reset: TemporalResetReason | None
    reset_count: int
    last_reset_reason: TemporalResetReason | None
    sequence_gap_events: int
    sequence_gap_frames: int


class SegmentationTimeline:
    """Classify capture discontinuities at the exact next-frame boundary.

    A sequence gap is observable but is not itself a reset condition. A reset
    is requested only for initial/config boundaries, source or geometry
    changes, decreasing time, or an elapsed gap above the qualified limit.
    ``observe`` records the boundary frame after choosing one reason, ensuring
    multiple simultaneous discontinuities cause exactly one reset.
    """

    def __init__(
        self,
        *,
        gap_reset_ns: int = SEGMENTATION_TIMESTAMP_GAP_RESET_NS,
    ) -> None:
        if type(gap_reset_ns) is not int or gap_reset_ns <= 0:
            raise ValueError("segmentation timestamp gap limit must be positive")
        self.gap_reset_ns = gap_reset_ns
        self._last: SegmentationFrameContext | None = None
        self._pending_reset: TemporalResetReason | None = TemporalResetReason.INITIAL
        self._reset_count = 0
        self._last_reset_reason: TemporalResetReason | None = None
        self._sequence_gap_events = 0
        self._sequence_gap_frames = 0

    def request_reset(self, reason: TemporalResetReason) -> None:
        """Reset before the next unique input without consuming an input."""

        if not isinstance(reason, TemporalResetReason):
            raise TypeError("temporal reset reason must be a TemporalResetReason")
        self._pending_reset = reason

    def record_reset(self, reason: TemporalResetReason) -> None:
        """Record a backend-internal reset applied to the current frame."""

        if not isinstance(reason, TemporalResetReason):
            raise TypeError("temporal reset reason must be a TemporalResetReason")
        self._reset_count += 1
        self._last_reset_reason = reason

    def checkpoint(self) -> SegmentationTimelineCheckpoint:
        """Capture content-free state before attempting a boundary frame."""

        return SegmentationTimelineCheckpoint(
            last_context=self._last,
            pending_reset=self._pending_reset,
            reset_count=self._reset_count,
            last_reset_reason=self._last_reset_reason,
            sequence_gap_events=self._sequence_gap_events,
            sequence_gap_frames=self._sequence_gap_frames,
        )

    def restore(self, checkpoint: SegmentationTimelineCheckpoint) -> None:
        """Restore a checkpoint after reset or processing fails."""

        if not isinstance(checkpoint, SegmentationTimelineCheckpoint):
            raise TypeError("segmentation timeline requires its own checkpoint")
        self._last = checkpoint.last_context
        self._pending_reset = checkpoint.pending_reset
        self._reset_count = checkpoint.reset_count
        self._last_reset_reason = checkpoint.last_reset_reason
        self._sequence_gap_events = checkpoint.sequence_gap_events
        self._sequence_gap_frames = checkpoint.sequence_gap_frames

    def observe(self, context: SegmentationFrameContext) -> TemporalBoundary:
        """Record a unique input and return its pre-processing reset decision."""

        if not isinstance(context, SegmentationFrameContext):
            raise TypeError("segmentation timeline requires a frame context")
        previous = self._last
        if previous is not None and context.sequence <= previous.sequence:
            raise ValueError(
                "segmentation timeline sequence must increase for every input"
            )
        elapsed_ns = (
            None if previous is None else context.timestamp_ns - previous.timestamp_ns
        )
        sequence_gap = (
            0 if previous is None else max(0, context.sequence - previous.sequence - 1)
        )
        if sequence_gap:
            self._sequence_gap_events += 1
            self._sequence_gap_frames += sequence_gap

        reason = self._pending_reset
        if reason is None and previous is None:
            reason = TemporalResetReason.INITIAL
        if reason is None and previous is not None:
            if context.generation != previous.generation:
                reason = TemporalResetReason.CAPTURE_GENERATION
            elif (
                context.geometry_generation != previous.geometry_generation
                or context.shape != previous.shape
            ):
                reason = TemporalResetReason.GEOMETRY
            elif elapsed_ns is not None and elapsed_ns < 0:
                reason = TemporalResetReason.NON_MONOTONIC_TIMESTAMP
            elif elapsed_ns is not None and elapsed_ns > self.gap_reset_ns:
                reason = TemporalResetReason.TIMESTAMP_GAP

        self._pending_reset = None
        self._last = context
        if reason is not None:
            self._reset_count += 1
            self._last_reset_reason = reason
        return TemporalBoundary(
            reset_reason=reason,
            elapsed_ns=elapsed_ns,
            sequence_gap=sequence_gap,
        )

    def snapshot(self) -> SegmentationTimelineSnapshot:
        return SegmentationTimelineSnapshot(
            reset_count=self._reset_count,
            last_reset_reason=self._last_reset_reason,
            sequence_gap_events=self._sequence_gap_events,
            sequence_gap_frames=self._sequence_gap_frames,
            last_sequence=None if self._last is None else self._last.sequence,
        )


MEDIAPIPE_MODEL = ModelSpec(
    backend="mediapipe",
    url=(
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/1/selfie_segmenter.tflite"
    ),
    filename="selfie_segmenter.tflite",
    size=249_537,
    sha256="191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b",
)
RVM_MODEL = ModelSpec(
    backend="rvm",
    url=(
        "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
        "rvm_mobilenetv3_fp32.onnx"
    ),
    filename="rvm_mobilenetv3_fp32.onnx",
    size=14_975_696,
    sha256="88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
)
BUILTIN_MODELS = {spec.backend: spec for spec in (RVM_MODEL, MEDIAPIPE_MODEL)}
# Retain the URL names for downstream code that imported the old constants.
MEDIAPIPE_MODEL_URL = MEDIAPIPE_MODEL.url
RVM_MODEL_URL = RVM_MODEL.url
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "custback" / "models"
MODEL_CONNECT_TIMEOUT_S = 15.0
MODEL_DOWNLOAD_TIMEOUT_S = 120.0
MODEL_LOCK_TIMEOUT_S = 30.0
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class ModelAcquisitionError(RuntimeError):
    """A managed model could not be acquired and integrity-checked."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_cached_model(path: Path, spec: ModelSpec) -> bool:
    try:
        return (
            path.is_file()
            and path.stat().st_size == spec.size
            and _sha256_file(path) == spec.sha256
        )
    except OSError:
        return False


@contextmanager
def _model_lock(path: Path, timeout_s: float = MODEL_LOCK_TIMEOUT_S) -> Iterator[None]:
    """Serialize model writers without leaving an owned sentinel behind."""

    descriptor = platform_fs.open_nofollow(path, os.O_CREAT | os.O_RDWR, 0o600)
    platform_fs.set_private_mode(descriptor, 0o600)
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                platform_fs.lock_exclusive(descriptor)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise ModelAcquisitionError(
                        f"timed out waiting for model lock {path}"
                    ) from exc
                time.sleep(0.1)
        yield
    finally:
        try:
            platform_fs.unlock(descriptor)
        finally:
            os.close(descriptor)


def _sync_directory(path: Path) -> None:
    try:
        platform_fs.fsync_dir(path)
    except OSError:  # pragma: no cover - not supported by every filesystem
        pass


def _stream_model(
    spec: ModelSpec,
    output: BinaryIO,
    *,
    opener=urllib.request.urlopen,
) -> tuple[int, str]:
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "custback-model-fetch/1"},
    )
    started = time.monotonic()
    deadline = started + MODEL_DOWNLOAD_TIMEOUT_S
    digest = hashlib.sha256()
    size = 0
    connect_timeout = min(
        MODEL_CONNECT_TIMEOUT_S,
        max(0.001, deadline - time.monotonic()),
    )
    with opener(request, timeout=connect_timeout) as response:
        header = response.headers.get("Content-Length") if response.headers else None
        if header is not None:
            try:
                advertised = int(header)
            except ValueError as exc:
                raise ModelAcquisitionError(
                    f"invalid Content-Length for {spec.filename}: {header!r}"
                ) from exc
            if advertised != spec.size:
                raise ModelAcquisitionError(
                    f"unexpected size for {spec.filename}: server advertised "
                    f"{advertised}, expected {spec.size}"
                )
        next_progress = max(_DOWNLOAD_CHUNK_SIZE, spec.size // 4)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelAcquisitionError(
                    f"download timed out after {MODEL_DOWNLOAD_TIMEOUT_S:.0f}s: {spec.filename}"
                )
            _set_response_timeout(
                response,
                min(MODEL_CONNECT_TIMEOUT_S, max(0.001, remaining)),
            )
            read = getattr(response, "read1", None)
            if not callable(read):
                read = response.read
            chunk = cast(bytes, read(min(_DOWNLOAD_CHUNK_SIZE, spec.size - size + 1)))
            if time.monotonic() >= deadline:
                raise ModelAcquisitionError(
                    f"download timed out after {MODEL_DOWNLOAD_TIMEOUT_S:.0f}s: "
                    f"{spec.filename}"
                )
            if not chunk:
                break
            size += len(chunk)
            if size > spec.size:
                raise ModelAcquisitionError(
                    f"download exceeded expected size for {spec.filename}"
                )
            output.write(chunk)
            digest.update(chunk)
            if size >= next_progress and size < spec.size:
                log.info("downloading %s: %d/%d bytes", spec.filename, size, spec.size)
                next_progress += max(_DOWNLOAD_CHUNK_SIZE, spec.size // 4)
    return size, digest.hexdigest()


def _set_response_timeout(response: object, timeout_s: float) -> None:
    """Best-effort per-read socket deadline for urllib HTTP responses."""

    pending = [response]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        setter = getattr(candidate, "settimeout", None)
        if callable(setter):
            try:
                setter(timeout_s)
                return
            except OSError:
                return
        for attribute in ("fp", "raw", "_sock", "sock", "socket"):
            child = getattr(candidate, attribute, None)
            if child is not None:
                pending.append(child)


def acquire_model(
    spec: ModelSpec,
    model_dir: Path | None = None,
    *,
    opener=urllib.request.urlopen,
    allow_download: bool = True,
) -> Path:
    """Return a verified built-in model, downloading it atomically if needed."""

    if Path(spec.filename).name != spec.filename or spec.filename in {"", ".", ".."}:
        raise ValueError(f"unsafe model filename: {spec.filename!r}")
    if not spec.url.startswith("https://"):
        raise ValueError("managed model URLs must use HTTPS")
    if (
        spec.size <= 0
        or len(spec.sha256) != 64
        or any(char not in "0123456789abcdef" for char in spec.sha256)
    ):
        raise ValueError(f"invalid integrity metadata for {spec.filename}")
    directory = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.chmod(0o700)
    except OSError:  # pragma: no cover - best effort on unusual filesystems
        pass
    model_path = directory / spec.filename
    if _valid_cached_model(model_path, spec):
        log.debug("verified cached model %s (sha256 %s)", model_path, spec.sha256[:12])
        return model_path
    if not allow_download:
        raise ModelAcquisitionError(
            f"pre-acquired model is missing or failed integrity validation: "
            f"{spec.filename}"
        )

    lock_path = directory / f".{spec.filename}.lock"
    with _model_lock(lock_path):
        # Another process may have completed the download while we waited.
        if _valid_cached_model(model_path, spec):
            log.debug("verified cached model %s after lock wait", model_path)
            return model_path
        if model_path.exists():
            log.warning("cached model failed integrity validation: %s", model_path)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{spec.filename}.", suffix=".part", dir=directory
        )
        temporary = Path(temporary_name)
        started = time.monotonic()
        try:
            log.info("downloading %s to %s", spec.filename, model_path)
            with os.fdopen(descriptor, "wb") as output:
                size, digest = _stream_model(spec, output, opener=opener)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o600)
            if size != spec.size:
                raise ModelAcquisitionError(
                    f"truncated download for {spec.filename}: got {size}, expected {spec.size}"
                )
            if digest != spec.sha256:
                raise ModelAcquisitionError(
                    f"checksum mismatch for {spec.filename}: got {digest}, "
                    f"expected {spec.sha256}"
                )
            os.replace(temporary, model_path)
            _sync_directory(directory)
            log.info(
                "downloaded and verified %s (%d bytes, sha256 %s) in %.1fs",
                spec.filename,
                size,
                digest[:12],
                time.monotonic() - started,
            )
            return model_path
        except ModelAcquisitionError:
            raise
        except Exception as exc:
            raise ModelAcquisitionError(
                f"could not acquire {spec.filename}: {exc}"
            ) from exc
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def acquire_builtin_model(backend: str, model_dir: Path | None = None) -> Path:
    """Pre-acquire one selected managed model before runtime services start."""

    try:
        spec = BUILTIN_MODELS[backend]
    except KeyError as exc:
        raise ValueError(f"{backend!r} does not have a custback-managed model") from exc
    return acquire_model(spec, model_dir)


def _custom_model_backend(cfg: SegmentationConfig) -> str | None:
    """Return the backend selected by a custom model's file format.

    Model formats are backend-specific, so the suffix is authoritative when
    ``backend: auto`` is used: an ONNX model is never offered to MediaPipe and
    a TFLite model is never preceded by an unrelated RVM probe/download.
    Configuration validation rejects incompatible explicit backend/suffix
    combinations before this helper is called.
    """

    if not cfg.model_path:
        return None
    suffix = Path(cfg.model_path).suffix.lower()
    if suffix == ".onnx":
        return "rvm"
    if suffix == ".tflite":
        return "mediapipe"
    return None


def preacquire_segmenter_model(cfg: SegmentationConfig) -> SegmenterPreparation:
    """Resolve every viable startup fallback before camera resources open.

    Without a custom path, automatic selection prepares both installed ML
    backends. This avoids a second network attempt after capture starts when
    RVM activation fails and MediaPipe becomes the next candidate. A custom
    path selects exactly the backend matching its suffix; custom bytes remain
    user-owned and unpinned and receive only an existence/readability preflight.
    """

    module_names = {"rvm": "onnxruntime", "mediapipe": "mediapipe"}
    custom_backend = _custom_model_backend(cfg)
    if cfg.backend == "auto" and custom_backend is not None:
        candidates = ((custom_backend, module_names[custom_backend]),)
    elif cfg.backend == "auto":
        candidates = (
            ("rvm", module_names["rvm"]),
            ("mediapipe", module_names["mediapipe"]),
        )
    else:
        candidates = ((cfg.backend, module_names.get(cfg.backend, cfg.backend)),)
    ready: set[str] = set()
    custom_path = Path(cfg.model_path) if cfg.model_path else None
    for backend, module in candidates:
        if backend not in BUILTIN_MODELS:
            continue
        try:
            importlib.import_module(module)
            if backend == custom_backend:
                assert custom_path is not None
                if not custom_path.is_file():
                    raise FileNotFoundError(
                        f"custom {backend} model does not exist: {custom_path}"
                    )
                with custom_path.open("rb") as stream:
                    stream.read(1)
            else:
                acquire_builtin_model(backend)
            ready.add(backend)
        except Exception as exc:
            if cfg.backend != "auto":
                raise
            log.info("%s model pre-acquisition unavailable (%s)", backend, exc)
    return SegmenterPreparation(frozenset(ready))


class Segmenter(ABC):
    #: BGR uint8 clean-foreground prediction for the last frame, when the
    #: backend provides one (rvm). The compositor uses it inside the soft
    #: edge band to remove original-background color spill.
    last_foreground: np.ndarray | None = None
    #: Where inference runs ("cpu", "cuda", "gpu", "coreml") — shown in /status.
    device: str = "cpu"
    #: True when the backend outputs an edge-accurate alpha matte with its own
    #: temporal consistency; the refiner then skips redundant feathering/EMA.
    produces_matte: bool = False
    #: Alpha semantics used by the typed effective-policy resolver. Concrete
    #: backends must override this instead of relying on requested ``auto`` or
    #: on a loose ``produces_matte`` capability check.
    matte_backend_kind = MatteBackendKind.BINARY_COARSE

    def __init__(self) -> None:
        self.temporal_reset_count = 0
        self.last_temporal_reset_reason: TemporalResetReason | None = None
        self.last_temporal_reset_timestamp_ns: int | None = None
        self.last_input_sequence: int | None = None
        self.last_input_timestamp_ns: int | None = None

    def _accept_frame_context(
        self,
        context: SegmentationFrameContext | None,
        frame_shape: tuple[int, int],
    ) -> None:
        """Validate and publish one successful temporal input identity."""

        self._validate_frame_context(context, frame_shape)
        if context is None:
            return
        self.last_input_sequence = context.sequence
        self.last_input_timestamp_ns = context.timestamp_ns

    def _validate_frame_context(
        self,
        context: SegmentationFrameContext | None,
        frame_shape: tuple[int, int],
    ) -> None:
        """Reject identity regressions without advancing temporal state."""

        if context is None:
            return
        if not isinstance(context, SegmentationFrameContext):
            raise TypeError("segmenter requires a segmentation frame context")
        if context.shape != frame_shape:
            raise ValueError(
                "segmenter frame context shape does not match the input pixels"
            )
        last_sequence = getattr(self, "last_input_sequence", None)
        if last_sequence is not None and context.sequence <= last_sequence:
            raise ValueError("segmenter input sequence must increase after each frame")
        last_timestamp_ns = getattr(self, "last_input_timestamp_ns", None)
        if last_timestamp_ns is not None and context.timestamp_ns < last_timestamp_ns:
            raise ValueError(
                "segmenter input timestamp must not decrease without a reset"
            )

    def reset_temporal_state(
        self,
        reason: TemporalResetReason,
        timestamp_ns: int | None,
    ) -> None:
        """Clear temporal state before the frame at ``timestamp_ns``.

        Stateless implementations inherit this content-free telemetry-only
        implementation. Stateful backends clear their private state first and
        then call ``super()``.
        """

        if not isinstance(reason, TemporalResetReason):
            raise TypeError("temporal reset reason must be a TemporalResetReason")
        if timestamp_ns is not None and (
            type(timestamp_ns) is not int or timestamp_ns < 0
        ):
            raise ValueError("temporal reset timestamp must be a non-negative integer")
        self.last_foreground = None
        self.last_input_sequence = None
        self.last_input_timestamp_ns = None
        self.temporal_reset_count = (
            int(getattr(self, "temporal_reset_count", 0) or 0) + 1
        )
        self.last_temporal_reset_reason = reason
        self.last_temporal_reset_timestamp_ns = timestamp_ns

    @abstractmethod
    def segment(
        self,
        frame_bgr: np.ndarray,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        """Return float32 HxW mask in [0, 1] for one capture context."""

    def close(self) -> None:
        pass


class NullSegmenter(Segmenter):
    matte_backend_kind = MatteBackendKind.NULL_PASSTHROUGH

    def segment(
        self,
        frame_bgr: np.ndarray,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        self._accept_frame_context(context, frame_bgr.shape[:2])
        return np.ones(frame_bgr.shape[:2], dtype=np.float32)


class HeuristicSegmenter(Segmenter):
    """Crude person prior: bright, center-weighted regions.

    Not meant for production visuals — it keeps the pipeline functional when
    no ML backend is available, and drives hardware-free tests.
    """

    matte_backend_kind = MatteBackendKind.BINARY_COARSE

    def __init__(self, cfg: SegmentationConfig):
        super().__init__()
        self.cfg = cfg
        self._prior: np.ndarray | None = None

    def _center_prior(self, h: int, w: int) -> np.ndarray:
        if self._prior is None or self._prior.shape != (h, w):
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dist = ((xx - w / 2) / (w * 0.45)) ** 2 + ((yy - h / 2) / (h * 0.6)) ** 2
            self._prior = np.clip(1.5 - dist, 0.0, 1.0)
        assert self._prior is not None
        return self._prior

    def segment(
        self,
        frame_bgr: np.ndarray,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        self._accept_frame_context(context, frame_bgr.shape[:2])
        gray = frame_bgr.astype(np.float32).mean(axis=2) / 255.0
        h, w = gray.shape
        score = gray * self._center_prior(h, w)
        mask = (score > self.cfg.threshold * 0.8).astype(np.float32)
        return mask


class MediaPipeSegmenter(Segmenter):
    """MediaPipe Tasks ImageSegmenter with the selfie segmentation model."""

    matte_backend_kind = MatteBackendKind.CONFIDENCE_MASK_VIDEO

    def __init__(self, cfg: SegmentationConfig, *, allow_model_download: bool = True):
        super().__init__()
        mp: Any = importlib.import_module("mediapipe")
        mp_python: Any = importlib.import_module("mediapipe.tasks.python")
        mp_vision: Any = importlib.import_module("mediapipe.tasks.python.vision")

        if cfg.model_path and Path(cfg.model_path).suffix.lower() == ".tflite":
            model_path = Path(cfg.model_path)
        else:
            model_path = acquire_model(
                MEDIAPIPE_MODEL, allow_download=allow_model_download
            )

        def make(delegate):
            options = mp_vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path), delegate=delegate
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                output_confidence_masks=True,
            )
            return mp_vision.ImageSegmenter.create_from_options(options)

        self._make_segmenter = make
        self._active_delegate: Any = None
        self._segmenter: Any = None
        if cfg.delegate == "gpu":
            try:
                self._active_delegate = mp_python.BaseOptions.Delegate.GPU
                self._segmenter = make(self._active_delegate)
                self.device = "gpu"
            except Exception as exc:
                log.warning("mediapipe GPU delegate unavailable (%s); using CPU", exc)
                self._active_delegate = None
        if self._segmenter is None:
            self._segmenter = make(None)
        self._mp = mp
        self._timestamp_epoch_ns: int | None = None
        self._last_effective_timestamp_ms: int | None = None
        self._timestamp_adjustment_count = 0
        self._telemetry = MediaPipeTelemetry(
            input_frame_shape=None,
            model_mask_shape=None,
            output_mask_shape=None,
            effective_timestamp_ms=None,
            effective_timestamp_delta_ms=None,
            timestamp_adjustment_count=0,
            timestamp_adjustment_ms=0,
            last_timestamp_adjusted=False,
            resize_interpolation=None,
        )
        self._has_inference = False
        # Retained as a compatibility alias for focused tests and diagnostics
        # written before capture timestamps were carried with every frame.
        self._ts_ms = 0

    def _next_timestamp_ms(
        self,
        context: SegmentationFrameContext | None,
    ) -> tuple[int, int | None, int]:
        """Map capture time onto MediaPipe's strictly increasing VIDEO clock."""

        previous = getattr(self, "_last_effective_timestamp_ms", None)
        if context is None:
            # Direct callers without capture identity retain the historical
            # nominal cadence. The production pipeline always supplies a
            # context and therefore never enters this compatibility path;
            # context-free results are not valid for cadence qualification.
            candidate = (
                int(getattr(self, "_ts_ms", 0) or 0) + 33
                if bool(getattr(self, "_has_inference", False))
                else 33
            )
        else:
            epoch_ns = getattr(self, "_timestamp_epoch_ns", None)
            if epoch_ns is None:
                epoch_ns = context.timestamp_ns
                self._timestamp_epoch_ns = epoch_ns
            candidate = (context.timestamp_ns - epoch_ns) // 1_000_000

        effective = candidate
        if previous is not None and effective <= previous:
            effective = previous + 1
        adjustment_ms = effective - candidate
        delta_ms = None if previous is None else effective - previous

        self._last_effective_timestamp_ms = effective
        if adjustment_ms:
            self._timestamp_adjustment_count = (
                int(getattr(self, "_timestamp_adjustment_count", 0) or 0) + 1
            )
        self._ts_ms = effective
        self._has_inference = True
        return effective, delta_ms, adjustment_ms

    @staticmethod
    def _validated_confidence_mask(result: Any) -> np.ndarray:
        """Copy one finite float32 confidence mask out of a MediaPipe result."""

        masks = getattr(result, "confidence_masks", None)
        if masks is None:
            raise ValueError("MediaPipe result has no confidence masks")
        try:
            mask_count = len(masks)
        except TypeError as exc:
            raise ValueError(
                "MediaPipe confidence masks must be a sized collection"
            ) from exc
        if mask_count != 1:
            raise ValueError(
                f"MediaPipe result must contain exactly one confidence mask; "
                f"got {mask_count}"
            )
        numpy_view = getattr(masks[0], "numpy_view", None)
        if not callable(numpy_view):
            raise ValueError("MediaPipe confidence mask has no NumPy view")
        raw = numpy_view()
        if not isinstance(raw, np.ndarray):
            raise ValueError("MediaPipe confidence mask must be a NumPy array")
        if raw.dtype != np.dtype(np.float32):
            raise ValueError(
                f"MediaPipe confidence mask must use float32 values; got {raw.dtype}"
            )
        if raw.ndim != 2 or any(dimension <= 0 for dimension in raw.shape):
            raise ValueError(
                "MediaPipe confidence mask must have two positive dimensions"
            )
        if not np.isfinite(raw).all():
            raise ValueError("MediaPipe confidence mask contains non-finite values")
        # MediaPipe owns numpy_view() storage. Detach before its result leaves
        # scope and normalize the memory layout promised by Segmenter.
        return np.array(raw, dtype=np.float32, order="C", copy=True)

    @staticmethod
    def _resize_soft_mask(
        mask: np.ndarray,
        output_shape: tuple[int, int],
    ) -> tuple[np.ndarray, str]:
        """Resize soft alpha deterministically and return its policy label.

        Downsampling uses area resampling and upsampling uses linear
        interpolation. A mixed-axis change follows ADR 0001: shrink the first
        axis with area resampling, then grow the orthogonal axis with linear
        interpolation, so one OpenCV filter is never asked to serve opposite
        directions.
        """

        if mask.shape == output_shape:
            return mask, "none"
        if cv2 is None:  # pragma: no cover - MediaPipe installations need cv2
            raise RuntimeError("OpenCV is required to resize MediaPipe masks")
        input_height, input_width = mask.shape
        output_height, output_width = output_shape
        width_direction = (output_width > input_width) - (output_width < input_width)
        height_direction = (output_height > input_height) - (
            output_height < input_height
        )
        if width_direction * height_direction < 0:
            if width_direction < 0:
                intermediate_size = (output_width, input_height)
            else:
                intermediate_size = (input_width, output_height)
            resized = cv2.resize(
                mask,
                intermediate_size,
                interpolation=cv2.INTER_AREA,
            )
            resized = cv2.resize(
                resized,
                (output_width, output_height),
                interpolation=cv2.INTER_LINEAR,
            )
            interpolation_name = "area+linear"
        else:
            shrinking = width_direction <= 0 and height_direction <= 0
            interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
            interpolation_name = "area" if shrinking else "linear"
            resized = cv2.resize(
                mask,
                (output_width, output_height),
                interpolation=interpolation,
            )
        if resized.shape != output_shape:
            raise ValueError(
                "resized MediaPipe confidence mask does not match the input frame"
            )
        if resized.dtype != np.dtype(np.float32) or not np.isfinite(resized).all():
            raise ValueError("resized MediaPipe confidence mask is invalid")
        return resized, interpolation_name

    def telemetry_snapshot(self) -> MediaPipeTelemetry:
        """Return content-free state for the last successful inference."""

        snapshot = getattr(self, "_telemetry", None)
        if isinstance(snapshot, MediaPipeTelemetry):
            return snapshot
        return MediaPipeTelemetry(
            input_frame_shape=None,
            model_mask_shape=None,
            output_mask_shape=None,
            effective_timestamp_ms=None,
            effective_timestamp_delta_ms=None,
            timestamp_adjustment_count=int(
                getattr(self, "_timestamp_adjustment_count", 0) or 0
            ),
            timestamp_adjustment_ms=0,
            last_timestamp_adjusted=False,
            resize_interpolation=None,
        )

    def segment(
        self,
        frame_bgr: np.ndarray,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        self._accept_frame_context(context, frame_bgr.shape[:2])
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms, delta_ms, adjustment_ms = self._next_timestamp_ms(context)
        result = self._segmenter.segment_for_video(mp_image, timestamp_ms)
        mask = self._validated_confidence_mask(result)
        model_mask_shape = cast(tuple[int, int], mask.shape)
        mask, interpolation = self._resize_soft_mask(mask, frame_bgr.shape[:2])
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        np.clip(mask, 0.0, 1.0, out=mask)
        if not np.isfinite(mask).all():  # defensive check after native resampling
            raise ValueError("MediaPipe confidence mask became non-finite")
        # Publish one immutable value only after the model result and all
        # native resampling have satisfied the output contract.
        self._telemetry = MediaPipeTelemetry(
            input_frame_shape=frame_bgr.shape[:2],
            model_mask_shape=model_mask_shape,
            output_mask_shape=cast(tuple[int, int], mask.shape),
            effective_timestamp_ms=timestamp_ms,
            effective_timestamp_delta_ms=delta_ms,
            timestamp_adjustment_count=self._timestamp_adjustment_count,
            timestamp_adjustment_ms=adjustment_ms,
            last_timestamp_adjusted=adjustment_ms > 0,
            resize_interpolation=interpolation,
        )
        return mask

    def reset_temporal_state(
        self,
        reason: TemporalResetReason,
        timestamp_ns: int | None,
    ) -> None:
        # A newly constructed task has no prior VIDEO timestamp and therefore
        # needs no recreation for its initial/config boundary. Once inference
        # has advanced, MediaPipe requires a fresh task before restarting its
        # timestamp epoch.
        if bool(getattr(self, "_has_inference", False)) or bool(
            getattr(self, "_ts_ms", 0)
        ):
            replacement = self._make_segmenter(self._active_delegate)
            previous = self._segmenter
            self._segmenter = replacement
            try:
                previous.close()
            except Exception as exc:
                log.debug("cannot close reset MediaPipe segmenter (%s)", exc)
        self._timestamp_epoch_ns = None
        self._last_effective_timestamp_ms = None
        self._telemetry = MediaPipeTelemetry(
            input_frame_shape=None,
            model_mask_shape=None,
            output_mask_shape=None,
            effective_timestamp_ms=None,
            effective_timestamp_delta_ms=None,
            timestamp_adjustment_count=int(
                getattr(self, "_timestamp_adjustment_count", 0) or 0
            ),
            timestamp_adjustment_ms=0,
            last_timestamp_adjusted=False,
            resize_interpolation=None,
        )
        self._has_inference = False
        self._ts_ms = 0
        super().reset_temporal_state(reason, timestamp_ns)

    def close(self) -> None:
        self._segmenter.close()


class RVMSegmenter(Segmenter):
    """Robust Video Matting (https://github.com/PeterL1n/RobustVideoMatting)
    through onnxruntime.

    Outputs a real alpha matte plus a clean foreground prediction, and keeps
    a recurrent state across frames for temporal consistency. onnxruntime-gpu
    (the custback[gpu] extra) enables CUDA inference on NVIDIA cards; plain
    onnxruntime (custback[rvm]) runs on CPU.
    """

    produces_matte = True
    matte_backend_kind = MatteBackendKind.TRUE_ALPHA_RECURRENT
    _RECURRENT_STATE_COUNT = 4
    _EMPTY_RECURRENT_SHAPE = (1, 1, 1, 1)

    def __init__(
        self,
        cfg: SegmentationConfig,
        *,
        acceleration: AccelerationConfig | None = None,
        allow_model_download: bool = True,
    ):
        super().__init__()
        ort: Any = importlib.import_module("onnxruntime")

        custom_model = bool(
            cfg.model_path and Path(cfg.model_path).suffix.lower() == ".onnx"
        )
        if custom_model:
            model_path = Path(cfg.model_path)
        else:
            model_path = acquire_model(RVM_MODEL, allow_download=allow_model_download)

        self._ort = ort
        self._model_path = str(model_path)
        self._model_builtin = not custom_model
        self._model_identity = (
            RVM_MODEL.filename if self._model_builtin else model_path.name
        )
        try:
            model_payload = model_path.read_bytes()
        except OSError as exc:
            # Never fall back to a mutable path: provider proof, production
            # inference, recovery, and reported identity must all consume the
            # exact same immutable byte snapshot.
            raise RuntimeError("RVM model bytes could not be read") from exc
        # Hash and load the same immutable byte snapshot. This prevents a
        # mutable custom model (or a replaced cache file) from being attributed
        # to bytes different from those ONNX Runtime consumed.
        actual_sha256 = hashlib.sha256(model_payload).hexdigest()
        if self._model_builtin and (
            len(model_payload) != RVM_MODEL.size or actual_sha256 != RVM_MODEL.sha256
        ):
            raise RuntimeError("managed RVM model integrity changed before load")
        self._model_source: bytes = model_payload
        self._model_sha256: str = actual_sha256
        self._model_bytes: int = len(model_payload)
        self._accel_cfg = (
            acceleration if acceleration is not None else AccelerationConfig()
        )
        #: Truthful, latched acceleration lifecycle (read by /status and doctor).
        self.accel = AccelerationState(self._accel_cfg)
        self._session: Any = self._build_session()
        self._downsample = cfg.rvm_downsample
        self.last_downsample_ratio: float | None = None
        self._rec: list[np.ndarray] | None = None
        self._size: tuple[int, int] | None = None
        self._rvm_telemetry = self._empty_rvm_telemetry()

    def _empty_rvm_telemetry(self) -> RVMTelemetry:
        """Return resource identity without retaining facts from an old frame."""

        acceleration = self.accel.status()
        return RVMTelemetry(
            input_frame_shape=None,
            output_alpha_shape=None,
            output_foreground_shape=None,
            configured_downsample_mode=(
                "auto" if self._downsample == 0.0 else "explicit"
            ),
            configured_downsample_ratio=float(self._downsample),
            resolved_downsample_ratio=None,
            preprocess_ms=None,
            session_run_ms=None,
            postprocess_ms=None,
            model_builtin=self._model_builtin,
            model_identity=self._model_identity,
            model_sha256=self._model_sha256,
            model_bytes=self._model_bytes,
            acceleration_state=acceleration.state,
            acceleration_active_provider=acceleration.active_provider,
            acceleration_fallback_active=acceleration.fallback_active,
            acceleration_fallback_count=acceleration.fallback_count,
        )

    def rvm_telemetry_snapshot(self) -> RVMTelemetry:
        """Return immutable model/runtime facts for the last successful frame."""

        return self._rvm_telemetry

    # -- session construction / acceleration policy -------------------
    def _new_session_options(self) -> Any:
        options = self._ort.SessionOptions()
        options.log_severity_level = 3  # hide per-node provider assignment noise
        return options

    def _make_session(self, providers: list[Any]) -> Any:
        return self._ort.InferenceSession(
            self._model_source,
            sess_options=self._new_session_options(),
            providers=providers,
        )

    def _build_cpu_session(self) -> Any:
        """Construct a deterministic CPU-only session."""

        return self._make_session([CPU_PROVIDER])

    def _build_session(self) -> Any:
        """Resolve the acceleration policy into a proven production session.

        GPU providers are proven against the real RVM graph before use; a
        registered-but-unprovable provider is treated as absent.  ``auto`` falls
        back to CPU (latched) and ``gpu_required`` fails startup, so no code path
        silently pretends a GPU is active when it is not.
        """

        available = list(self._ort.get_available_providers())
        candidates = resolve_provider_candidates(self._accel_cfg, available)
        gpu_required = self._accel_cfg.mode == "gpu_required"

        if not candidates:
            if gpu_required:
                raise GpuRequiredError(
                    "acceleration.mode is gpu_required but no accelerator "
                    "execution provider is registered"
                )
            session = self._build_cpu_session()
            self.device = "cpu"
            self.accel.mark_cpu_active()
            return session

        self.accel.mark_probing()
        last_reason = "no accelerator provider could execute RVM"
        for candidate in candidates:
            preload_acceleration_dlls(candidate.name)
            proof = prove_rvm_provider(self._ort, self._model_source, candidate)
            if not proof.proven:
                last_reason = proof.error or last_reason
                log.info(
                    "RVM acceleration provider %s unavailable (%s)",
                    candidate.name,
                    proof.error or "not proven",
                )
                continue
            try:
                session = self._make_session([candidate.as_ort_arg(), CPU_PROVIDER])
                active = session.get_providers()[0]
                if active != candidate.name:
                    # Production session disagreed with the proof; do not trust it.
                    last_reason = "production session did not bind the proven provider"
                    continue
                warm_up_session(session, 64, 64)
            except Exception as exc:
                last_reason = " ".join(str(exc).split())[:200]
                log.info(
                    "RVM production session failed on %s (%s)",
                    candidate.name,
                    exc,
                )
                continue
            self.device = provider_label(candidate.name)
            self.accel.mark_gpu_active(candidate.name)
            log.info("RVM acceleration active on %s", candidate.name)
            return session

        if gpu_required:
            raise GpuRequiredError(
                f"acceleration.mode is gpu_required but no accelerator could "
                f"execute RVM: {last_reason}"
            )
        session = self._build_cpu_session()
        self.device = "cpu"
        self.accel.latch_fallback(last_reason)
        log.warning("RVM acceleration fell back to CPU: %s", last_reason)
        return session

    def _recover_to_cpu(self, exc: BaseException) -> None:
        """Rebuild a CPU-only session after a GPU/DLL/OOM inference failure."""

        log.warning("RVM GPU inference failed (%s); rebuilding a CPU-only session", exc)
        self._session = self._build_cpu_session()
        self.device = "cpu"
        self.accel.latch_fallback(str(exc))

    @classmethod
    def _zero_recurrent_state(cls) -> list[np.ndarray]:
        """Allocate four independent zero inputs for a clean RVM epoch."""

        return [
            np.zeros(cls._EMPTY_RECURRENT_SHAPE, dtype=np.float32)
            for _ in range(cls._RECURRENT_STATE_COUNT)
        ]

    def _start_temporal_epoch(self, frame_shape: tuple[int, int]) -> None:
        """Bind clean recurrent inputs to the first frame shape of an epoch."""

        self._rec = self._zero_recurrent_state()
        self._size = frame_shape

    def _feeds(self, frame_bgr: np.ndarray, ratio: float) -> dict[str, np.ndarray]:
        assert self._rec is not None
        if cv2 is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:  # pragma: no cover
            rgb = frame_bgr[..., ::-1]
        src = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return {
            "src": src,
            "r1i": self._rec[0],
            "r2i": self._rec[1],
            "r3i": self._rec[2],
            "r4i": self._rec[3],
            "downsample_ratio": np.asarray([ratio], dtype=np.float32),
        }

    def segment(
        self,
        frame_bgr: np.ndarray,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        # Validate first, but publish the identity only with the validated
        # alpha/recurrent/foreground/telemetry state below. This lets an exact
        # retry follow a failed inference without a phantom consumed frame.
        self._validate_frame_context(context, (h, w))
        if self._rec is None or self._size != (h, w):
            # Recurrent state is resolution-bound; reset on size changes.
            if self._size is not None and self._size != (h, w):
                self.reset_temporal_state(
                    TemporalResetReason.GEOMETRY,
                    None if context is None else context.timestamp_ns,
                )
            self._start_temporal_epoch((h, w))
        # Internal inference resolution: the model was trained to matte at a
        # reduced size and refine at full size; ~512 px on the long side is
        # the quality/speed sweet spot for webcam framing.
        ratio = self._downsample or min(1.0, max(0.125, 512.0 / max(h, w)))
        preprocess_ns = 0
        session_run_ns = 0
        started = time.monotonic_ns()
        feeds = self._feeds(frame_bgr, ratio)
        preprocess_ns += time.monotonic_ns() - started
        started = time.monotonic_ns()
        try:
            outputs = self._session.run(None, feeds)
        except Exception as exc:
            session_run_ns += time.monotonic_ns() - started
            # A GPU/DLL/OOM failure is recoverable once: rebuild a CPU session,
            # clear the recurrent state and foreground so the retry starts clean,
            # and stay on CPU. A CPU-side failure is not retried (it would loop).
            if not self.accel.on_gpu:
                raise
            self._recover_to_cpu(exc)
            self.reset_temporal_state(
                TemporalResetReason.BACKEND_RECOVERY,
                None if context is None else context.timestamp_ns,
            )
            self._start_temporal_epoch((h, w))
            started = time.monotonic_ns()
            feeds = self._feeds(frame_bgr, ratio)
            preprocess_ns += time.monotonic_ns() - started
            started = time.monotonic_ns()
            outputs = self._session.run(None, feeds)
            session_run_ns += time.monotonic_ns() - started
        else:
            session_run_ns += time.monotonic_ns() - started

        started = time.monotonic_ns()
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 6:
            raise ValueError("RVM returned an invalid output set")
        fgr, pha, *next_recurrent = outputs
        if (
            not isinstance(fgr, np.ndarray)
            or fgr.shape != (1, 3, h, w)
            or not np.issubdtype(fgr.dtype, np.floating)
            or not bool(np.isfinite(fgr).all())
        ):
            raise ValueError("RVM returned an invalid foreground")
        if (
            not isinstance(pha, np.ndarray)
            or pha.shape != (1, 1, h, w)
            or not np.issubdtype(pha.dtype, np.floating)
        ):
            raise ValueError("RVM returned an invalid alpha")
        alpha = np.ascontiguousarray(pha[0, 0].astype(np.float32))
        if (
            not bool(np.isfinite(alpha).all())
            or float(alpha.min(initial=0.0)) < 0.0
            or float(alpha.max(initial=1.0)) > 1.0
        ):
            raise ValueError("RVM returned alpha outside the finite [0, 1] contract")
        if len(next_recurrent) != self._RECURRENT_STATE_COUNT or any(
            not isinstance(state, np.ndarray)
            or state.ndim != 4
            or state.shape[0] != 1
            or state.size == 0
            or not np.issubdtype(state.dtype, np.floating)
            or not bool(np.isfinite(state).all())
            for state in next_recurrent
        ):
            raise ValueError("RVM returned invalid recurrent state")
        fgr_rgb = np.clip(fgr[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        foreground = np.ascontiguousarray(fgr_rgb[..., ::-1])
        postprocess_ns = time.monotonic_ns() - started
        acceleration = self.accel.status()

        # Publish frame-derived state as one final step. A failed inference or
        # malformed output leaves the prior successful snapshot truthful (or
        # the empty snapshot for a fresh/reset epoch).
        self._rec = next_recurrent
        self.last_foreground = foreground
        self.last_downsample_ratio = float(ratio)
        if context is not None:
            self.last_input_sequence = context.sequence
            self.last_input_timestamp_ns = context.timestamp_ns
        self._rvm_telemetry = RVMTelemetry(
            input_frame_shape=(h, w),
            output_alpha_shape=alpha.shape,
            output_foreground_shape=foreground.shape,
            configured_downsample_mode=(
                "auto" if self._downsample == 0.0 else "explicit"
            ),
            configured_downsample_ratio=float(self._downsample),
            resolved_downsample_ratio=float(ratio),
            preprocess_ms=preprocess_ns / 1_000_000.0,
            session_run_ms=session_run_ns / 1_000_000.0,
            postprocess_ms=postprocess_ns / 1_000_000.0,
            model_builtin=self._model_builtin,
            model_identity=self._model_identity,
            model_sha256=self._model_sha256,
            model_bytes=self._model_bytes,
            acceleration_state=acceleration.state,
            acceleration_active_provider=acceleration.active_provider,
            acceleration_fallback_active=acceleration.fallback_active,
            acceleration_fallback_count=acceleration.fallback_count,
        )
        return alpha

    def reset_temporal_state(
        self,
        reason: TemporalResetReason,
        timestamp_ns: int | None,
    ) -> None:
        """Discard every frame-derived value before the next RVM input."""

        self._rec = None
        self._size = None
        self.last_downsample_ratio = None
        super().reset_temporal_state(reason, timestamp_ns)
        self._rvm_telemetry = self._empty_rvm_telemetry()

    def close(self) -> None:
        self._session = None
        if isinstance(self._model_source, bytes):
            self._model_source = b""
        self._rec = None
        self._size = None
        self.last_downsample_ratio = None
        self.last_foreground = None
        self._rvm_telemetry = self._empty_rvm_telemetry()


def _watershed_edge_snap(
    mask: np.ndarray,
    frame_bgr: np.ndarray,
    radius: int = 8,
) -> np.ndarray:
    """Snap a coarse mask contour to nearby image edges within a bounded band.

    A guided filter can preserve an edge but cannot reliably move a displaced
    contour onto it. Marker watershed uses eroded foreground/background cores
    as immutable seeds and may move the boundary by at most ``radius`` pixels.
    """
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or not np.issubdtype(mask.dtype, np.number)
    ):
        return mask
    clipped = np.array(mask, dtype=np.float32, copy=True)
    np.nan_to_num(
        clipped,
        copy=False,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    np.clip(clipped, 0.0, 1.0, out=clipped)
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.dtype != np.uint8
        or radius <= 0
        or cv2 is None
    ):
        return clipped
    h, w = clipped.shape
    if h < 3 or w < 3 or frame_bgr.shape[:2] != (h, w):
        return clipped
    hard = (clipped >= 0.5).astype(np.uint8)
    if not hard.any() or hard.all():
        return clipped

    radius = min(radius, max(1, (min(h, w) - 1) // 4))
    try:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        sure_fg = cv2.erode(hard, kernel)
        sure_bg = cv2.erode(1 - hard, kernel)
        if not sure_fg.any() or not sure_bg.any():
            return clipped

        unknown = (sure_fg == 0) & (sure_bg == 0)

        # Erosion can remove a small connected component completely even when
        # another, larger component supplies the global foreground/background
        # marker. Watershed would then have no seed representing that component
        # and classify it away. Protect each seedless component and its bounded
        # uncertainty band independently.
        def seedless_components(binary: np.ndarray, seeds: np.ndarray) -> np.ndarray:
            contours, hierarchy = cv2.findContours(
                binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if hierarchy is None:
                return np.zeros_like(binary, dtype=bool)
            hierarchy = hierarchy[0]
            components = [
                index for index, relation in enumerate(hierarchy) if relation[3] == -1
            ]
            if len(components) <= 1:
                # The global non-empty seed check above proves the sole
                # component owns a marker.
                return np.zeros_like(binary, dtype=bool)

            result = np.zeros_like(binary, dtype=np.uint8)
            for index in components:
                x, y, width, height = cv2.boundingRect(contours[index])
                seed_crop = seeds[y : y + height, x : x + width]
                component = np.zeros((height, width), dtype=np.uint8)
                offset = np.asarray([[[x, y]]], dtype=contours[index].dtype)
                cv2.drawContours(
                    component, [contours[index] - offset], -1, 1, cv2.FILLED
                )
                child = hierarchy[index][2]
                while child != -1:
                    cv2.drawContours(
                        component, [contours[child] - offset], -1, 0, cv2.FILLED
                    )
                    child = hierarchy[child][0]
                if not np.any(component & seed_crop):
                    result[y : y + height, x : x + width] |= component
            return result != 0

        seedless = seedless_components(hard, sure_fg) | seedless_components(
            1 - hard, sure_bg
        )
        if seedless.any():
            protected = cv2.dilate(seedless.astype(np.uint8), kernel) != 0
            unknown &= ~protected
        if not unknown.any():
            return clipped

        # Watershed only needs the bounded uncertainty band and one marker
        # margin on either side. Cropping avoids a full-frame watershed and
        # keeps the 720p median comfortably inside the real-time budget while
        # preserving every pixel outside the band verbatim.
        active_rows = np.flatnonzero(unknown.any(axis=1))
        active_cols = np.flatnonzero(unknown.any(axis=0))
        if active_rows.size == 0 or active_cols.size == 0:
            return clipped
        margin = 2
        y0 = max(0, int(active_rows[0]) - margin)
        y1 = min(h, int(active_rows[-1]) + margin + 1)
        x0 = max(0, int(active_cols[0]) - margin)
        x1 = min(w, int(active_cols[-1]) + margin + 1)
        unknown_crop = unknown[y0:y1, x0:x1]
        guide = cv2.GaussianBlur(frame_bgr[y0:y1, x0:x1], (3, 3), 0)
        # Watershed on a featureless image degenerates to a distance split
        # between markers. That invents an edge rather than snapping to one,
        # so preserve the segmenter's contour when the uncertainty band has
        # no visible contrast at all.
        if np.ptp(guide[unknown_crop].astype(np.int16), axis=0).max(initial=0) == 0:
            return clipped

        markers = np.zeros(unknown_crop.shape, dtype=np.int32)
        markers[sure_bg[y0:y1, x0:x1] != 0] = 1
        markers[sure_fg[y0:y1, x0:x1] != 0] = 2
        cv2.watershed(guide, markers)
    except cv2.error as exc:
        log.warning("edge watershed failed; retaining segmenter mask: %s", exc)
        return clipped

    refined = clipped.copy()
    band = refined[y0:y1, x0:x1]
    band[unknown_crop & (markers == 1)] = 0.0
    band[unknown_crop & (markers == 2)] = 1.0
    band[unknown_crop & (markers == -1)] = 0.5
    return refined


STABLE_EDGE_MAX_BAND_AREA_FRACTION = 0.25
"""Maximum matte area eligible for stable spatial refinement."""

STABLE_EDGE_MAX_WORK_AREA_FRACTION = 0.85
"""Maximum summed padded component ROI area relative to the canonical frame."""

STABLE_EDGE_MAX_COMPONENTS = 64
"""Maximum number of independent boundary regions refined on one frame."""

_STABLE_EDGE_GUIDED_ITERATIONS = 4
_STABLE_EDGE_MIN_GRADIENT_LUMA = 10.0
_STABLE_EDGE_MIN_CONTRAST_LUMA = 20.0
_STABLE_EDGE_MIN_CORRELATION = 0.2
_STABLE_EDGE_MIN_MASK_VARIANCE = 0.002
_STABLE_EDGE_AMBIGUITY_RATIO = 0.82
_STABLE_EDGE_MAX_MIDTONE_FRACTION = 0.18
_STABLE_EDGE_THIN_DENSITY = 0.20


def _stable_guided_edge_refine(
    mask: np.ndarray,
    frame_bgr: np.ndarray,
    config: SpatialEdgeRefinementConfig,
) -> np.ndarray:
    """Move a coarse contour only toward strong, unique image-edge support.

    This opt-in path deliberately does not reinterpret the compatibility
    watershed. A bounded, iterated guided-alpha filter supplies a soft
    candidate. Local gradient, contrast, alpha/guide correlation and ambiguity
    gates decide which candidate segments may replace the current alpha.
    Every unsafe or unavailable path returns the sanitized current matte.
    """

    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or not np.issubdtype(mask.dtype, np.number)
    ):
        return mask
    clipped = np.array(mask, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(clipped).all():
        np.nan_to_num(
            clipped,
            copy=False,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
    np.clip(clipped, 0.0, 1.0, out=clipped)
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.dtype != np.uint8
        or not isinstance(config, SpatialEdgeRefinementConfig)
        or cv2 is None
    ):
        return clipped

    height, width = clipped.shape
    if height < 3 or width < 3 or frame_bgr.shape[:2] != (height, width):
        return clipped
    hard = (clipped >= 0.5).astype(np.uint8)
    if not hard.any() or hard.all():
        return clipped

    radius = spatial_edge_refinement_radius(config, (height, width))
    radius = min(radius, max(1, (min(height, width) - 1) // 4))
    if radius <= 0:
        return clipped

    try:
        search_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        band = (
            cv2.morphologyEx(
                hard,
                cv2.MORPH_GRADIENT,
                search_kernel,
            )
            != 0
        )
        if (
            not band.any()
            or np.count_nonzero(band)
            > STABLE_EDGE_MAX_BAND_AREA_FRACTION * clipped.size
        ):
            return clipped

        # A boundary pixel of an ordinary thick region occupies roughly half
        # its local search window. Hair-like foreground and narrow background
        # slits occupy much less. Keep those pixels authoritative so a guide
        # edge cannot erase a connected fine structure.
        local_foreground_density = cv2.boxFilter(
            hard.astype(np.float32),
            -1,
            (2 * radius + 1, 2 * radius + 1),
            normalize=True,
            borderType=cv2.BORDER_REFLECT101,
        )
        protected = (hard != 0) & (local_foreground_density < _STABLE_EDGE_THIN_DENSITY)
        protected |= (hard == 0) & (
            (1.0 - local_foreground_density) < _STABLE_EDGE_THIN_DENSITY
        )
        active = band & ~protected
        if not active.any():
            return clipped

        component_count, component_labels, component_stats, _centroids = (
            cv2.connectedComponentsWithStats(
                active.astype(np.uint8),
                connectivity=8,
            )
        )
        if component_count <= 1 or component_count - 1 > STABLE_EDGE_MAX_COMPONENTS:
            return clipped

        components: list[tuple[int, int, int, int, int]] = []
        work_area = 0
        for label in range(1, component_count):
            x, y, component_width, component_height, _area = component_stats[label]
            x0 = max(0, int(x) - radius)
            y0 = max(0, int(y) - radius)
            x1 = min(width, int(x + component_width) + radius)
            y1 = min(height, int(y + component_height) + radius)
            work_area += (x1 - x0) * (y1 - y0)
            components.append((label, x0, y0, x1, y1))
        if work_area > STABLE_EDGE_MAX_WORK_AREA_FRACTION * clipped.size:
            return clipped

        refined = clipped.copy()
        for label, x0, y0, x1, y1 in components:
            component = component_labels[y0:y1, x0:x1] == label
            if not component.any():
                continue
            alpha = clipped[y0:y1, x0:x1]
            alpha_hard = (alpha >= 0.5).astype(np.uint8)
            topology_before = (
                cv2.connectedComponents(alpha_hard, connectivity=8)[0],
                cv2.connectedComponents(1 - alpha_hard, connectivity=8)[0],
            )
            gray = cv2.cvtColor(
                frame_bgr[y0:y1, x0:x1],
                cv2.COLOR_BGR2GRAY,
            )
            if int(gray.max()) - int(gray.min()) < _STABLE_EDGE_MIN_CONTRAST_LUMA:
                continue

            # Scale only the denoising support, and keep it much smaller than
            # the search radius. Camera noise and 8x8 codec blocks are reduced
            # without merging genuinely separate nearby contours.
            sigma = max(0.8, radius / 6.0)
            blur_radius = min(4, max(2, int(math.ceil(2.0 * sigma))))
            blur_size = 2 * blur_radius + 1
            denoised = cv2.GaussianBlur(
                gray,
                (blur_size, blur_size),
                sigmaX=sigma,
                borderType=cv2.BORDER_REFLECT101,
            )
            guide = denoised.astype(np.float32) / 255.0
            window = (2 * radius + 1, 2 * radius + 1)

            def box_mean(value: np.ndarray) -> np.ndarray:
                return cv2.boxFilter(
                    value,
                    -1,
                    window,
                    normalize=True,
                    borderType=cv2.BORDER_REFLECT101,
                )

            guide_mean = box_mean(guide)
            guide_variance = np.maximum(
                box_mean(guide * guide) - guide_mean * guide_mean,
                0.0,
            )
            candidate = alpha.copy()
            correlation = np.zeros_like(candidate)
            mask_variance = np.zeros_like(candidate)
            for _ in range(_STABLE_EDGE_GUIDED_ITERATIONS):
                mask_mean = box_mean(candidate)
                mask_variance = np.maximum(
                    box_mean(candidate * candidate) - mask_mean * mask_mean,
                    0.0,
                )
                covariance = box_mean(guide * candidate) - guide_mean * mask_mean
                correlation = np.abs(covariance) / np.sqrt(
                    guide_variance * mask_variance + 1e-6
                )
                slope = covariance / (guide_variance + 1e-4)
                intercept = mask_mean - slope * guide_mean
                candidate = box_mean(slope) * guide + box_mean(intercept)
                np.clip(candidate, 0.0, 1.0, out=candidate)

            soft_count = np.count_nonzero((alpha > 1e-4) & (alpha < 1.0 - 1e-4))
            if soft_count >= max(4, int(0.001 * alpha.size)):
                # Retain part of the segmenter's own profile rather than
                # narrowing an already meaningful soft matte.
                candidate = 0.75 * candidate + 0.25 * alpha
            else:
                # A binary source still receives a narrow, multi-valued edge;
                # unlike watershed this never writes a fixed 0/1/0.5 band.
                candidate = np.clip((candidate - 0.375) / 0.25, 0.0, 1.0)
                candidate = candidate * candidate * (3.0 - 2.0 * candidate)

            candidate_hard = (candidate >= 0.5).astype(np.uint8)
            small_kernel = np.ones((3, 3), dtype=np.uint8)
            candidate_boundary = (
                cv2.morphologyEx(
                    candidate_hard,
                    cv2.MORPH_GRADIENT,
                    small_kernel,
                )
                != 0
            )
            if not candidate_boundary.any():
                continue

            gradient = cv2.morphologyEx(
                denoised,
                cv2.MORPH_GRADIENT,
                small_kernel,
            ).astype(np.float32)
            support_radius = max(1, min(3, int(math.ceil(sigma))))
            support_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * support_radius + 1, 2 * support_radius + 1),
            )
            nearby_gradient = cv2.dilate(gradient, support_kernel)
            exclusion = (
                cv2.dilate(candidate_boundary.astype(np.uint8), support_kernel) != 0
            )
            competing_gradient_source = gradient.copy()
            competing_gradient_source[exclusion] = 0.0
            competing_gradient = cv2.dilate(
                competing_gradient_source,
                search_kernel,
            )
            local_contrast = cv2.dilate(
                denoised,
                search_kernel,
            ).astype(np.float32) - cv2.erode(
                denoised,
                search_kernel,
            ).astype(np.float32)

            # A single strong step contains two dominant endpoint levels.
            # A second nearby edge introduces a material intermediate plateau.
            # Measuring that plateau on the unblurred guide makes alternating
            # near-equal gradients an exact no-op instead of a contour toggle.
            local_min = cv2.erode(gray, search_kernel).astype(np.float32)
            local_max = cv2.dilate(gray, search_kernel).astype(np.float32)
            local_span = local_max - local_min
            normalized = (gray.astype(np.float32) - local_min) / np.maximum(
                local_span,
                1.0,
            )
            midtone = ((normalized > 0.25) & (normalized < 0.75)).astype(np.float32)
            midtone_fraction = box_mean(midtone)

            support = candidate_boundary.copy()
            support &= nearby_gradient >= _STABLE_EDGE_MIN_GRADIENT_LUMA
            support &= local_contrast >= _STABLE_EDGE_MIN_CONTRAST_LUMA
            support &= correlation >= _STABLE_EDGE_MIN_CORRELATION
            support &= mask_variance >= _STABLE_EDGE_MIN_MASK_VARIANCE
            support &= (
                competing_gradient < _STABLE_EDGE_AMBIGUITY_RATIO * nearby_gradient
            )
            support &= midtone_fraction <= _STABLE_EDGE_MAX_MIDTONE_FRACTION
            if not support.any():
                continue

            strength_confidence = np.clip(
                (nearby_gradient - _STABLE_EDGE_MIN_GRADIENT_LUMA) / 64.0,
                0.0,
                1.0,
            )
            contrast_confidence = np.clip(
                (local_contrast - _STABLE_EDGE_MIN_CONTRAST_LUMA) / 96.0,
                0.0,
                1.0,
            )
            correlation_confidence = np.clip(
                (correlation - _STABLE_EDGE_MIN_CORRELATION)
                / (1.0 - _STABLE_EDGE_MIN_CORRELATION),
                0.0,
                1.0,
            )
            edge_confidence = np.minimum(
                np.minimum(strength_confidence, contrast_confidence),
                correlation_confidence,
            )
            support_confidence = np.where(
                support,
                0.97 + 0.03 * edge_confidence,
                0.0,
            ).astype(np.float32)
            confidence = cv2.dilate(support_confidence, search_kernel)
            apply = component & (confidence > 0.0)
            if not apply.any():
                continue
            output_crop = refined[y0:y1, x0:x1]
            blend = confidence[apply]
            output_crop[apply] += blend * (candidate[apply] - output_crop[apply])
            output_hard = (output_crop >= 0.5).astype(np.uint8)
            topology_after = (
                cv2.connectedComponents(output_hard, connectivity=8)[0],
                cv2.connectedComponents(1 - output_hard, connectivity=8)[0],
            )
            if topology_after != topology_before:
                return clipped

        np.clip(refined, 0.0, 1.0, out=refined)
    except cv2.error as exc:
        log.warning("stable edge refinement failed; retaining segmenter mask: %s", exc)
        return clipped

    return np.ascontiguousarray(refined, dtype=np.float32)


BOUNDARY_FLOW_MAX_LONG_EDGE = 320
"""Maximum long edge of the retained optical-flow guide."""

BOUNDARY_MAX_AREA_FRACTION = 0.25
"""Fail open to the current matte when its boundary band becomes too broad."""

_BOUNDARY_BAND_RADIUS = 3
_BOUNDARY_PHOTOMETRIC_LIMIT = 32.0
_BOUNDARY_FLOW_CONSISTENCY_LIMIT = 1.5
_BOUNDARY_MIN_CONFIDENCE = 0.2


class MaskRefiner:
    """Apply spatial refinement and one explicitly selected temporal policy.

    The historical adaptive EMA remains byte-for-byte selectable while the
    motion-aware path is opt-in. Motion state is deliberately limited to one
    full-resolution alpha plus a 320-pixel-long-edge grayscale guide and
    low-resolution hold metadata.
    """

    def __init__(self, cfg: SegmentationConfig):
        # Motion-aware stabilization replaces rather than stacks with the
        # historical unregistered EMA.  Retain the configured value in the
        # application snapshot, but expose the exercised refiner policy here.
        if (
            cfg.boundary_stabilization.mode == "motion_aware"
            and cfg.temporal_smoothing != 0.0
        ):
            cfg = cfg.model_copy(update={"temporal_smoothing": 0.0}, deep=True)
        self.cfg = cfg
        self._prev: np.ndarray | None = None
        self._prev_guide: np.ndarray | None = None
        self._prev_motion_timestamp_ns: int | None = None
        self._motion_hold_age: np.ndarray | None = None
        self._motion_direction: np.ndarray | None = None
        self.temporal_reset_count = 0
        self.last_temporal_reset_reason: TemporalResetReason | None = None
        self.last_temporal_reset_timestamp_ns: int | None = None
        self.last_input_sequence: int | None = None
        self.last_input_timestamp_ns: int | None = None

    def refine(
        self,
        mask: np.ndarray,
        frame_bgr: np.ndarray | None = None,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        next_sequence = self.last_input_sequence
        next_timestamp = self.last_input_timestamp_ns
        if context is not None:
            if not isinstance(context, SegmentationFrameContext):
                raise TypeError("mask refiner requires a segmentation frame context")
            if context.shape != mask.shape[:2] or (
                frame_bgr is not None and context.shape != frame_bgr.shape[:2]
            ):
                raise ValueError(
                    "mask refiner frame context shape does not match its inputs"
                )
            if (
                self.last_input_sequence is not None
                and context.sequence <= self.last_input_sequence
            ):
                raise ValueError(
                    "mask refiner input sequence must increase after each frame"
                )
            if (
                self.last_input_timestamp_ns is not None
                and context.timestamp_ns < self.last_input_timestamp_ns
            ):
                raise ValueError(
                    "mask refiner timestamp must not decrease without a reset"
                )
            next_sequence = context.sequence
            next_timestamp = context.timestamp_ns

        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
            raise ValueError("mask refiner requires a non-empty two-dimensional mask")
        mask = np.array(mask, dtype=np.float32, order="C", copy=True)
        np.nan_to_num(mask, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(mask, 0.0, 1.0, out=mask)
        if cv2 is not None:
            # Snap first. A later user-requested grow/shrink must remain an
            # intentional halo-control offset rather than being undone here.
            if self.cfg.edge_refine and frame_bgr is not None:
                if self.cfg.spatial_edge_refinement.mode == "legacy_watershed":
                    legacy_radius = spatial_edge_refinement_radius(
                        self.cfg.spatial_edge_refinement,
                        mask.shape,
                    )
                    if legacy_radius == 8:
                        # Keep the schema-v1/default call byte-compatible,
                        # including for tests and deployments that wrap the
                        # historical two-argument function.
                        mask = _watershed_edge_snap(mask, frame_bgr)
                    else:
                        mask = _watershed_edge_snap(
                            mask,
                            frame_bgr,
                            legacy_radius,
                        )
                else:
                    mask = _stable_guided_edge_refine(
                        mask,
                        frame_bgr,
                        self.cfg.spatial_edge_refinement,
                    )
            if self.cfg.mask_shift:
                r = abs(self.cfg.mask_shift)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)
                )
                op = cv2.dilate if self.cfg.mask_shift > 0 else cv2.erode
                mask = op(mask, kernel)
            if self.cfg.mask_blur:
                k = self.cfg.mask_blur
                mask = cv2.GaussianBlur(mask, (k, k), 0)

        # The input was sanitized above. OpenCV blur/morphology and the
        # bounded watershed preserve finite values and the unit interval.
        current = np.ascontiguousarray(mask, dtype=np.float32)

        next_guide: np.ndarray | None = None
        next_motion_timestamp_ns: int | None = None
        next_hold_age: np.ndarray | None = None
        next_direction: np.ndarray | None = None
        if self.cfg.boundary_stabilization.mode == "motion_aware":
            (
                result,
                next_guide,
                next_motion_timestamp_ns,
                next_hold_age,
                next_direction,
            ) = self._refine_motion_aware(current, frame_bgr, context)
        else:
            result = current
            # Historical compatibility EMA. Its coefficient remains per-frame
            # and is not silently converted into a time constant.
            alpha = self.cfg.temporal_smoothing
            if (
                alpha > 0
                and self._prev is not None
                and self._prev.shape == current.shape
            ):
                diff = np.abs(current - self._prev)
                if cv2 is not None:
                    diff = cv2.blur(diff, (7, 7))
                keep = alpha * np.clip(1.0 - 4.0 * diff, 0.0, 1.0)
                result = keep * self._prev + (1.0 - keep) * current

        result = np.ascontiguousarray(result, dtype=np.float32)
        if not np.isfinite(result).all():
            np.nan_to_num(result, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(result, 0.0, 1.0, out=result)

        # Publish all temporal state and input identity only after every
        # fallible operation produced a valid result.
        self._prev = result.copy()
        self._prev_guide = next_guide
        self._prev_motion_timestamp_ns = next_motion_timestamp_ns
        self._motion_hold_age = next_hold_age
        self._motion_direction = next_direction
        self.last_input_sequence = next_sequence
        self.last_input_timestamp_ns = next_timestamp
        return result

    @staticmethod
    def _empty_motion_history(
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Allocate bounded disagreement state for one current matte."""

        return (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.int8),
        )

    def _motion_guide(self, frame_bgr: np.ndarray | None) -> np.ndarray | None:
        """Return a privacy-local grayscale guide capped at a 320 px long edge."""

        if (
            cv2 is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.dtype != np.uint8
            or frame_bgr.size == 0
        ):
            return None
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, BOUNDARY_FLOW_MAX_LONG_EDGE / max(height, width))
        guide_width = max(2, int(round(width * scale)))
        guide_height = max(2, int(round(height * scale)))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if (guide_width, guide_height) != (width, height):
            gray = cv2.resize(
                gray,
                (guide_width, guide_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(gray, dtype=np.uint8)

    def _estimate_boundary_motion(
        self,
        current_guide: np.ndarray,
        previous_guide: np.ndarray,
        *,
        full_shape: tuple[int, int],
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Estimate bounded current-to-previous flow and correspondence confidence."""

        if (
            cv2 is None
            or current_guide.shape != previous_guide.shape
            or current_guide.ndim != 2
            or current_guide.size == 0
            or not math.isfinite(dt_s)
            or dt_s <= 0.0
        ):
            return None
        factory = getattr(cv2, "DISOpticalFlow_create", None)
        preset = getattr(cv2, "DISOPTICAL_FLOW_PRESET_ULTRAFAST", None)
        if not callable(factory) or preset is None:
            return None

        estimator: Any = factory(preset)
        # Backward flow maps each current-guide coordinate into the previous
        # guide, which is exactly the map needed to sample the previous alpha.
        backward = estimator.calc(current_guide, previous_guide, None)
        forward = estimator.calc(previous_guide, current_guide, None)
        backward = np.asarray(backward, dtype=np.float32)
        forward = np.asarray(forward, dtype=np.float32)
        expected_shape = (*current_guide.shape, 2)
        if (
            backward.shape != expected_shape
            or forward.shape != expected_shape
            or not np.isfinite(backward).all()
            or not np.isfinite(forward).all()
        ):
            return None

        guide_height, guide_width = current_guide.shape
        grid_y, grid_x = np.indices(
            (guide_height, guide_width),
            dtype=np.float32,
        )
        previous_x = grid_x + backward[..., 0]
        previous_y = grid_y + backward[..., 1]
        in_bounds = (
            (previous_x >= 0.0)
            & (previous_x <= guide_width - 1.0)
            & (previous_y >= 0.0)
            & (previous_y <= guide_height - 1.0)
        )
        if float(np.mean(in_bounds)) < 0.5:
            return None

        sampled_forward_x = cv2.remap(
            forward[..., 0],
            previous_x,
            previous_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        sampled_forward_y = cv2.remap(
            forward[..., 1],
            previous_x,
            previous_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        consistency = np.hypot(
            backward[..., 0] + sampled_forward_x,
            backward[..., 1] + sampled_forward_y,
        )
        warped_previous = cv2.remap(
            previous_guide,
            previous_x,
            previous_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        photometric = np.abs(
            current_guide.astype(np.float32) - warped_previous.astype(np.float32)
        )
        # A global scene discontinuity must seed the new frame rather than
        # creating isolated, apparently confident fragments.
        if (
            not np.isfinite(consistency).all()
            or float(np.median(photometric[in_bounds])) >= _BOUNDARY_PHOTOMETRIC_LIMIT
        ):
            return None

        gradient_x = cv2.Sobel(
            current_guide,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
        gradient_y = cv2.Sobel(
            current_guide,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )
        texture_confidence = np.clip(
            np.hypot(gradient_x, gradient_y) / 16.0,
            0.0,
            1.0,
        )
        consistency_confidence = np.clip(
            1.0 - consistency / _BOUNDARY_FLOW_CONSISTENCY_LIMIT,
            0.0,
            1.0,
        )
        photometric_confidence = np.clip(
            1.0 - photometric / _BOUNDARY_PHOTOMETRIC_LIMIT,
            0.0,
            1.0,
        )

        full_height, full_width = full_shape
        displacement_x = backward[..., 0] * (full_width / guide_width)
        displacement_y = backward[..., 1] * (full_height / guide_height)
        speed = np.hypot(displacement_x, displacement_y) / dt_s
        maximum_speed = self.cfg.boundary_stabilization.max_motion_px_per_s
        motion_confidence = np.clip(
            (maximum_speed - speed) / max(maximum_speed * 0.2, 1e-6),
            0.0,
            1.0,
        )
        confidence = (
            consistency_confidence
            * photometric_confidence
            * texture_confidence
            * motion_confidence
        )
        confidence[~in_bounds] = 0.0
        confidence[speed >= maximum_speed] = 0.0
        return (
            np.ascontiguousarray(backward, dtype=np.float32),
            np.ascontiguousarray(confidence, dtype=np.float32),
        )

    @staticmethod
    def _boundary_band(current: np.ndarray) -> np.ndarray | None:
        """Build the only region in which temporal output may differ."""

        if cv2 is None:
            return None
        hard = (current >= 0.5).astype(np.uint8)
        uncertain = ((current > 0.02) & (current < 0.98)).astype(np.uint8)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, edge_kernel)
        seeds = np.maximum(edge, uncertain)
        if not seeds.any():
            return np.zeros(current.shape, dtype=np.uint8)
        radius = _BOUNDARY_BAND_RADIUS
        band_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        band = cv2.dilate(seeds, band_kernel)
        if float(cv2.countNonZero(band)) / band.size > BOUNDARY_MAX_AREA_FRACTION:
            return None
        return band

    @staticmethod
    def _sample_sparse(
        source: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        interpolation: int,
    ) -> np.ndarray:
        """Sample an array only at active boundary coordinates."""

        sampled = cv2.remap(
            source,
            x.reshape(-1, 1).astype(np.float32, copy=False),
            y.reshape(-1, 1).astype(np.float32, copy=False),
            interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return sampled.reshape(-1)

    def _refine_motion_aware(
        self,
        current: np.ndarray,
        frame_bgr: np.ndarray | None,
        context: SegmentationFrameContext | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        int | None,
        np.ndarray,
        np.ndarray,
    ]:
        """Align and blend prior alpha only where correspondence is credible."""

        timestamp_ns = None if context is None else context.timestamp_ns
        try:
            current_guide = self._motion_guide(frame_bgr)
        except Exception as exc:
            log.debug("motion guide failed; using current matte: %s", exc)
            current_guide = None
        history_shape = (0, 0) if current_guide is None else current_guide.shape
        empty_age, empty_direction = self._empty_motion_history(history_shape)

        if (
            current_guide is None
            or timestamp_ns is None
            or self._prev is None
            or self._prev.shape != current.shape
            or self._prev_guide is None
            or self._prev_guide.shape != current_guide.shape
            or self._prev_motion_timestamp_ns is None
        ):
            return (
                current,
                current_guide,
                timestamp_ns,
                empty_age,
                empty_direction,
            )

        dt_s = (timestamp_ns - self._prev_motion_timestamp_ns) / 1_000_000_000.0
        policy = self.cfg.boundary_stabilization
        maximum_hold_s = min(4.0 * policy.time_constant_s, 0.25)
        if not math.isfinite(dt_s) or dt_s <= 0.0 or dt_s >= maximum_hold_s:
            return (
                current,
                current_guide,
                timestamp_ns,
                empty_age,
                empty_direction,
            )

        try:
            band = self._boundary_band(current)
            if band is None or cv2.countNonZero(band) == 0:
                return (
                    current,
                    current_guide,
                    timestamp_ns,
                    empty_age,
                    empty_direction,
                )
            estimate = self._estimate_boundary_motion(
                current_guide,
                self._prev_guide,
                full_shape=current.shape,
                dt_s=dt_s,
            )
            if estimate is None:
                return (
                    current,
                    current_guide,
                    timestamp_ns,
                    empty_age,
                    empty_direction,
                )

            backward, confidence_map = estimate
            if (
                backward.shape != (*current_guide.shape, 2)
                or confidence_map.shape != current_guide.shape
                or not np.isfinite(backward).all()
                or not np.isfinite(confidence_map).all()
            ):
                return (
                    current,
                    current_guide,
                    timestamp_ns,
                    empty_age,
                    empty_direction,
                )

            points = cv2.findNonZero(band)
            if points is None:
                return (
                    current,
                    current_guide,
                    timestamp_ns,
                    empty_age,
                    empty_direction,
                )
            points = np.asarray(points).reshape(-1, 2)
            columns = points[:, 0]
            rows = points[:, 1]
            full_height, full_width = current.shape
            guide_height, guide_width = current_guide.shape
            guide_x = (columns.astype(np.float32) + 0.5) * (
                guide_width / full_width
            ) - 0.5
            guide_y = (rows.astype(np.float32) + 0.5) * (
                guide_height / full_height
            ) - 0.5
            flow_x = self._sample_sparse(
                backward[..., 0],
                guide_x,
                guide_y,
                cv2.INTER_LINEAR,
            )
            flow_y = self._sample_sparse(
                backward[..., 1],
                guide_x,
                guide_y,
                cv2.INTER_LINEAR,
            )
            confidence = self._sample_sparse(
                confidence_map,
                guide_x,
                guide_y,
                cv2.INTER_LINEAR,
            )
            displacement_x = flow_x * (full_width / guide_width)
            displacement_y = flow_y * (full_height / guide_height)
            speed = np.hypot(displacement_x, displacement_y) / dt_s
            maximum_speed = policy.max_motion_px_per_s
            motion_confidence = np.clip(
                (maximum_speed - speed) / max(maximum_speed * 0.2, 1e-6),
                0.0,
                1.0,
            )
            confidence *= motion_confidence
            confidence[speed >= maximum_speed] = 0.0
            previous_x = columns.astype(np.float32) + flow_x * (
                full_width / guide_width
            )
            previous_y = rows.astype(np.float32) + flow_y * (full_height / guide_height)
            in_bounds = (
                (previous_x >= 0.0)
                & (previous_x <= full_width - 1.0)
                & (previous_y >= 0.0)
                & (previous_y <= full_height - 1.0)
            )
            confidence = np.clip(confidence, 0.0, 1.0)
            confidence[~in_bounds] = 0.0
            confidence = np.where(
                confidence >= _BOUNDARY_MIN_CONFIDENCE,
                (confidence - _BOUNDARY_MIN_CONFIDENCE)
                / (1.0 - _BOUNDARY_MIN_CONFIDENCE),
                0.0,
            ).astype(np.float32)
            previous_alpha = self._sample_sparse(
                self._prev,
                previous_x,
                previous_y,
                cv2.INTER_LINEAR,
            )

            previous_age = np.zeros(rows.size, dtype=np.float32)
            previous_direction = np.zeros(rows.size, dtype=np.int8)
            previous_guide_x = guide_x + flow_x
            previous_guide_y = guide_y + flow_y
            if (
                self._motion_hold_age is not None
                and self._motion_hold_age.shape == self._prev_guide.shape
            ):
                previous_age = self._sample_sparse(
                    self._motion_hold_age,
                    previous_guide_x,
                    previous_guide_y,
                    cv2.INTER_LINEAR,
                ).astype(np.float32, copy=False)
            if (
                self._motion_direction is not None
                and self._motion_direction.shape == self._prev_guide.shape
            ):
                sampled_direction = self._sample_sparse(
                    self._motion_direction.astype(np.float32),
                    previous_guide_x,
                    previous_guide_y,
                    cv2.INTER_NEAREST,
                )
                previous_direction = np.sign(sampled_direction).astype(np.int8)

            current_alpha = current[rows, columns]
            disagreement = current_alpha - previous_alpha
            direction = np.zeros(rows.size, dtype=np.int8)
            # Track any material float32 disagreement. A larger deadband makes
            # release time cadence-dependent because different FPS cross that
            # deadband on different frames.
            direction[disagreement > 1e-6] = 1
            direction[disagreement < -1e-6] = -1
            same_direction = (direction != 0) & (direction == previous_direction)
            hold_age = np.where(
                direction == 0,
                0.0,
                np.where(same_direction, previous_age + dt_s, dt_s),
            ).astype(np.float32)
            confidence[hold_age >= maximum_hold_s] = 0.0

            keep_previous = (
                math.exp(-dt_s / policy.time_constant_s) * confidence
            ).astype(np.float32)
            # Avoid introducing a rounding-only change when registered alpha
            # already agrees with the current matte.
            keep_previous[direction == 0] = 0.0
            stabilized = (
                keep_previous * previous_alpha + (1.0 - keep_previous) * current_alpha
            )
            result = current.copy()
            result[rows, columns] = stabilized

            active = (keep_previous > 0.0) & (direction != 0)
            next_age = empty_age
            next_direction = empty_direction
            if np.any(active):
                active_guide_x = np.clip(
                    np.rint(guide_x[active]).astype(np.intp),
                    0,
                    guide_width - 1,
                )
                active_guide_y = np.clip(
                    np.rint(guide_y[active]).astype(np.intp),
                    0,
                    guide_height - 1,
                )
                active_age = hold_age[active]
                np.maximum.at(
                    next_age,
                    (active_guide_y, active_guide_x),
                    active_age,
                )
                selected = active_age >= (
                    next_age[active_guide_y, active_guide_x] - 1e-6
                )
                active_directions = direction[active]
                next_direction[
                    active_guide_y[selected],
                    active_guide_x[selected],
                ] = active_directions[selected]
            return (
                result,
                current_guide,
                timestamp_ns,
                next_age,
                next_direction,
            )
        except Exception as exc:
            # Optical flow and sparse correspondence are optional quality
            # work. Any native/API failure must expose the current valid matte
            # and make it the sole state for the following frame.
            log.debug("boundary stabilization failed; using current matte: %s", exc)
            return (
                current,
                current_guide,
                timestamp_ns,
                empty_age,
                empty_direction,
            )

    def reset_temporal_state(
        self,
        reason: TemporalResetReason,
        timestamp_ns: int | None,
    ) -> None:
        """Clear previous alpha before processing a discontinuity frame."""

        if not isinstance(reason, TemporalResetReason):
            raise TypeError("temporal reset reason must be a TemporalResetReason")
        if timestamp_ns is not None and (
            type(timestamp_ns) is not int or timestamp_ns < 0
        ):
            raise ValueError("temporal reset timestamp must be a non-negative integer")
        self._prev = None
        self._prev_guide = None
        self._prev_motion_timestamp_ns = None
        self._motion_hold_age = None
        self._motion_direction = None
        self.last_input_sequence = None
        self.last_input_timestamp_ns = None
        self.temporal_reset_count += 1
        self.last_temporal_reset_reason = reason
        self.last_temporal_reset_timestamp_ns = timestamp_ns

    def reset(self) -> None:
        """Compatibility reset for callers without a capture timeline."""

        self._prev = None
        self._prev_guide = None
        self._prev_motion_timestamp_ns = None
        self._motion_hold_age = None
        self._motion_direction = None
        self.last_input_sequence = None
        self.last_input_timestamp_ns = None

    def close(self) -> None:
        """Release camera-derived temporal state without publishing telemetry."""

        self.reset()


def segmenter_matte_backend_kind(segmenter: object) -> MatteBackendKind:
    """Return explicit production capability with a legacy-test fallback."""

    kind = getattr(segmenter, "matte_backend_kind", None)
    if isinstance(kind, MatteBackendKind):
        return kind
    # Older extensions and narrow test doubles predate the explicit
    # capability. Preserve the one historical distinction they could express.
    if bool(getattr(segmenter, "produces_matte", False)):
        return MatteBackendKind.TRUE_ALPHA_RECURRENT
    return MatteBackendKind.BINARY_COARSE


def refiner_for(
    cfg: SegmentationConfig,
    segmenter: Segmenter,
    compositing: CompositingConfig | None = None,
) -> MaskRefiner:
    """Build the refiner selected by the sole backend-policy resolver.

    The optional compositor argument preserves the historical public helper
    signature for tests and offline callers; compositor values do not alter
    the effective refiner configuration.
    """

    policy = resolve_matte_policy(
        cfg,
        compositing or CompositingConfig(),
        segmenter_matte_backend_kind(segmenter),
        resolved_rvm_ratio=getattr(segmenter, "last_downsample_ratio", None),
    )
    return MaskRefiner(policy.effective_refiner_config(cfg))


def create_segmenter(
    cfg: SegmentationConfig,
    *,
    acceleration: AccelerationConfig | None = None,
    preparation: SegmenterPreparation | None = None,
) -> Segmenter:
    requested_backend = cfg.backend
    backend = (
        _custom_model_backend(cfg)
        if requested_backend == "auto" and cfg.model_path
        else requested_backend
    )
    # SegmentationConfig validates custom suffixes, but retain the automatic
    # behavior defensively if a caller supplies a non-standard path through a
    # model constructed without normal validation.
    if backend is None:
        backend = requested_backend
    prepared = preparation.ready_backends if preparation is not None else None
    if backend == "none":
        return NullSegmenter()
    if (
        prepared is not None
        and backend in {"rvm", "mediapipe"}
        and backend not in prepared
    ):
        raise ModelAcquisitionError(
            f"selected {backend} backend did not pass model pre-acquisition"
        )
    if backend in ("auto", "rvm") and (prepared is None or "rvm" in prepared):
        try:
            seg = RVMSegmenter(
                cfg,
                acceleration=acceleration,
                allow_model_download=preparation is None,
            )
            log.info("using rvm matting backend on %s", seg.device)
            return seg
        except GpuRequiredError:
            # gpu_required is an explicit operator demand for proven GPU
            # execution; never satisfy it by silently degrading to another
            # backend, regardless of backend=auto fallback.
            raise
        except Exception as exc:
            if requested_backend == "rvm":
                raise
            log.info("rvm backend unavailable (%s)", exc)
    if backend in ("auto", "mediapipe") and (
        prepared is None or "mediapipe" in prepared
    ):
        try:
            seg = MediaPipeSegmenter(cfg, allow_model_download=preparation is None)
            log.info("using mediapipe segmentation backend on %s", seg.device)
            return seg
        except Exception as exc:
            if requested_backend == "mediapipe":
                raise
            log.info("mediapipe unavailable (%s); falling back to heuristic", exc)
    log.info("using heuristic segmentation backend")
    return HeuristicSegmenter(cfg)
