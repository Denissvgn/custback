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
import os
import tempfile
import time
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, cast

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
from .config import AccelerationConfig, SegmentationConfig

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

    @abstractmethod
    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return float32 HxW mask in [0, 1]."""

    def close(self) -> None:
        pass


class NullSegmenter(Segmenter):
    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        return np.ones(frame_bgr.shape[:2], dtype=np.float32)


class HeuristicSegmenter(Segmenter):
    """Crude person prior: bright, center-weighted regions.

    Not meant for production visuals — it keeps the pipeline functional when
    no ML backend is available, and drives hardware-free tests.
    """

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._prior: np.ndarray | None = None

    def _center_prior(self, h: int, w: int) -> np.ndarray:
        if self._prior is None or self._prior.shape != (h, w):
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dist = ((xx - w / 2) / (w * 0.45)) ** 2 + ((yy - h / 2) / (h * 0.6)) ** 2
            self._prior = np.clip(1.5 - dist, 0.0, 1.0)
        assert self._prior is not None
        return self._prior

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = frame_bgr.astype(np.float32).mean(axis=2) / 255.0
        h, w = gray.shape
        score = gray * self._center_prior(h, w)
        mask = (score > self.cfg.threshold * 0.8).astype(np.float32)
        return mask


class MediaPipeSegmenter(Segmenter):
    """MediaPipe Tasks ImageSegmenter with the selfie segmentation model."""

    def __init__(self, cfg: SegmentationConfig, *, allow_model_download: bool = True):
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

        self._segmenter: Any = None
        if cfg.delegate == "gpu":
            try:
                self._segmenter = make(mp_python.BaseOptions.Delegate.GPU)
                self.device = "gpu"
            except Exception as exc:
                log.warning("mediapipe GPU delegate unavailable (%s); using CPU", exc)
        if self._segmenter is None:
            self._segmenter = make(None)
        self._mp = mp
        self._ts_ms = 0

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result = self._segmenter.segment_for_video(mp_image, self._ts_ms)
        mask = result.confidence_masks[0].numpy_view().astype(np.float32)
        if mask.shape != frame_bgr.shape[:2]:
            mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return mask

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

    def __init__(
        self,
        cfg: SegmentationConfig,
        *,
        acceleration: AccelerationConfig | None = None,
        allow_model_download: bool = True,
    ):
        ort: Any = importlib.import_module("onnxruntime")

        if cfg.model_path and Path(cfg.model_path).suffix.lower() == ".onnx":
            model_path = Path(cfg.model_path)
        else:
            model_path = acquire_model(RVM_MODEL, allow_download=allow_model_download)

        self._ort = ort
        self._model_path = str(model_path)
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

    # -- session construction / acceleration policy -------------------
    def _new_session_options(self) -> Any:
        options = self._ort.SessionOptions()
        options.log_severity_level = 3  # hide per-node provider assignment noise
        return options

    def _make_session(self, providers: list[Any]) -> Any:
        return self._ort.InferenceSession(
            self._model_path,
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
            proof = prove_rvm_provider(self._ort, self._model_path, candidate)
            if not proof.proven:
                last_reason = proof.error or last_reason
                log.info(
                    "RVM acceleration provider %s unavailable (%s)",
                    candidate.name,
                    proof.error or "not proven",
                )
                continue
            session = self._make_session([candidate.as_ort_arg(), CPU_PROVIDER])
            active = session.get_providers()[0]
            if active != candidate.name:
                # Production session disagreed with the proof; do not trust it.
                last_reason = "production session did not bind the proven provider"
                continue
            try:
                warm_up_session(session, 64, 64)
            except Exception as exc:  # pragma: no cover - defensive warm-up guard
                last_reason = " ".join(str(exc).split())[:200]
                log.info("RVM warm-up failed on %s (%s)", candidate.name, exc)
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

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self._rec is None or self._size != (h, w):
            # Recurrent state is resolution-bound; reset on size changes.
            self._rec = [np.zeros((1, 1, 1, 1), dtype=np.float32)] * 4
            self._size = (h, w)
        # Internal inference resolution: the model was trained to matte at a
        # reduced size and refine at full size; ~512 px on the long side is
        # the quality/speed sweet spot for webcam framing.
        ratio = self._downsample or min(1.0, max(0.125, 512.0 / max(h, w)))
        self.last_downsample_ratio = float(ratio)
        try:
            fgr, pha, *self._rec = self._session.run(
                None, self._feeds(frame_bgr, ratio)
            )
        except Exception as exc:
            # A GPU/DLL/OOM failure is recoverable once: rebuild a CPU session,
            # clear the recurrent state and foreground so the retry starts clean,
            # and stay on CPU. A CPU-side failure is not retried (it would loop).
            if not self.accel.on_gpu:
                raise
            self._recover_to_cpu(exc)
            self._rec = [np.zeros((1, 1, 1, 1), dtype=np.float32)] * 4
            self.last_foreground = None
            fgr, pha, *self._rec = self._session.run(
                None, self._feeds(frame_bgr, ratio)
            )
        fgr_rgb = np.clip(fgr[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        self.last_foreground = np.ascontiguousarray(fgr_rgb[..., ::-1])
        return np.ascontiguousarray(pha[0, 0].astype(np.float32))

    def close(self) -> None:
        self._session = None
        self._rec = None


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


class MaskRefiner:
    """Post-processing shared by all backends: mask grow/shrink, edge
    feathering, edge-aware refinement and adaptive temporal smoothing."""

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._prev: np.ndarray | None = None

    def refine(
        self, mask: np.ndarray, frame_bgr: np.ndarray | None = None
    ) -> np.ndarray:
        if cv2 is not None:
            # Snap first. A later user-requested grow/shrink must remain an
            # intentional halo-control offset rather than being undone here.
            if self.cfg.edge_refine and frame_bgr is not None:
                mask = _watershed_edge_snap(mask, frame_bgr)
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

        # Temporal smoothing, weighted by local change: static regions get the
        # full EMA (no flicker), moving edges track immediately (no ghosting).
        alpha = self.cfg.temporal_smoothing
        if alpha > 0 and self._prev is not None and self._prev.shape == mask.shape:
            diff = np.abs(mask - self._prev)
            if cv2 is not None:
                diff = cv2.blur(diff, (7, 7))
            keep = alpha * np.clip(1.0 - 4.0 * diff, 0.0, 1.0)
            mask = keep * self._prev + (1.0 - keep) * mask

        mask = np.clip(mask, 0.0, 1.0)
        self._prev = mask
        return mask

    def reset(self) -> None:
        self._prev = None


def refiner_for(cfg: SegmentationConfig, segmenter: Segmenter) -> MaskRefiner:
    """Build a refiner tuned to the segmenter: matting backends (rvm) already
    produce edge-accurate, temporally consistent alphas — re-feathering or
    re-smoothing them would only blur hair detail and add lag. mask_shift
    stays honored as the user's halo control."""
    if getattr(segmenter, "produces_matte", False):
        cfg = cfg.model_copy(
            update={
                "mask_blur": 0,
                "edge_refine": False,
                "temporal_smoothing": 0.0,
            },
            deep=True,
        )
    return MaskRefiner(cfg)


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
