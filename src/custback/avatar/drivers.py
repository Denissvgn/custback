"""Animation drivers: produce a :class:`FaceState` per rendered frame.

* :class:`IdleDriver` — deterministic procedural idling (blinks, gaze and
  head sway). Hardware-free; the fallback when no tracker is available and
  the base layer under audio-driven animation.
* :class:`VisionDriver` — MediaPipe Face Landmarker on the raw camera
  frames custback forwards: 52 ARKit blendshapes plus a head pose. Runs on
  CPU everywhere and fits any local GPU setup (RTX 3060 included).
* ``Audio2FaceDriver`` (in :mod:`.audio2face`) — NVIDIA Audio2Face-3D
  client for audio-driven animation from a gRPC endpoint.
"""

from __future__ import annotations

import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..segmentation import ModelAcquisitionError, ModelSpec, acquire_model
from .config import DriverConfig, VisionConfig
from .state import FaceState

try:
    import cv2
except ImportError:  # pragma: no cover - required by the package, defensive
    cv2 = None

log = logging.getLogger(__name__)

# MediaPipe Face Landmarker with blendshape and pose outputs; pinned like
# the segmentation models so every use is integrity-checked.
FACE_LANDMARKER_MODEL = ModelSpec(
    backend="face_landmarker",
    url=(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    filename="face_landmarker.task",
    size=3_758_596,
    sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
)


class DriverUnavailableError(RuntimeError):
    """The requested driver backend cannot run in this environment."""


class DriverStartupError(DriverUnavailableError):
    """A driver failed during startup and still requires lifecycle ownership."""

    def __init__(self, driver: Any, startup_error: BaseException):
        self.driver = driver
        self.startup_error = startup_error
        super().__init__(f"driver startup failed: {startup_error}")


@dataclass(frozen=True)
class _VisionBindings:
    """Imported MediaPipe surfaces needed to construct a vision driver."""

    mediapipe: Any
    tasks: Any
    vision: Any


@dataclass(frozen=True)
class DriverPreparation:
    """Assets prepared before entering the render/activation lane.

    The backend and configured model spelling bind the preparation to the
    candidate that produced it.  ``_vision_error`` is retained for ``auto`` so
    the eventual factory call preserves its historical idle fallback without
    repeating dependency probes, path checks, or model acquisition.
    """

    backend: str
    vision_model_setting: str
    vision_model_path: Path | None = None
    _vision_bindings: _VisionBindings | None = field(default=None, repr=False)
    _vision_error: DriverUnavailableError | ModelAcquisitionError | None = field(
        default=None, repr=False
    )


def _load_vision_bindings() -> _VisionBindings:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:
        raise DriverUnavailableError(
            "mediapipe is not installed; install the [mediapipe] extra "
            "for the vision avatar driver"
        ) from exc
    return _VisionBindings(mp, mp_tasks, mp_vision)


def _prepare_vision_model(
    cfg: VisionConfig, *, allow_model_download: bool
) -> Path:
    if cfg.model_path:
        model_path = Path(cfg.model_path).expanduser()
        if not model_path.is_file():
            raise DriverUnavailableError(
                f"vision model_path does not exist: {model_path}"
            )
        return model_path
    return acquire_model(
        FACE_LANDMARKER_MODEL, allow_download=allow_model_download
    )


def prepare_driver(
    cfg: DriverConfig, *, allow_model_download: bool = True
) -> DriverPreparation:
    """Prepare driver assets without constructing render-thread resources.

    Existing callers may continue to call :func:`create_driver` directly.  A
    transactional caller can instead run this function on an acquisition
    worker, then pass the result to ``create_driver(..., preparation=...)`` on
    its owned render lane.  Managed downloads and custom-path validation occur
    exactly once in the preparation step.
    """

    preparation = DriverPreparation(cfg.backend, cfg.vision.model_path)
    if cfg.backend not in ("auto", "vision"):
        return preparation
    try:
        if cfg.vision.model_path:
            # Custom paths are cheap to validate and should fail before any
            # optional backend import. Managed acquisition, by contrast, must
            # not download a model on hosts that cannot import MediaPipe.
            model_path = _prepare_vision_model(
                cfg.vision, allow_model_download=allow_model_download
            )
            bindings = _load_vision_bindings()
        else:
            bindings = _load_vision_bindings()
            model_path = _prepare_vision_model(
                cfg.vision, allow_model_download=allow_model_download
            )
    except (DriverUnavailableError, ModelAcquisitionError) as exc:
        if cfg.backend == "vision":
            raise
        return DriverPreparation(
            cfg.backend,
            cfg.vision.model_path,
            _vision_error=exc,
        )
    return DriverPreparation(
        cfg.backend,
        cfg.vision.model_path,
        vision_model_path=model_path,
        _vision_bindings=bindings,
    )


class FaceDriver(ABC):
    """Stateful animation source; ``update`` is called once per frame."""

    name: str = "driver"
    device: str = "cpu"

    @abstractmethod
    def update(self, frame_bgr: np.ndarray | None, timestamp: float) -> FaceState:
        """Return the animation state for a frame captured at ``timestamp``."""

    def close(self) -> None:
        pass


def idle_pose(timestamp: float) -> tuple[float, float, float]:
    """Gentle deterministic head sway (yaw, pitch, roll in radians)."""
    yaw = 0.045 * math.sin(2.0 * math.pi * timestamp / 9.0)
    pitch = 0.030 * math.sin(2.0 * math.pi * timestamp / 7.0 + 1.3)
    roll = 0.025 * math.sin(2.0 * math.pi * timestamp / 11.0 + 0.5)
    return yaw, pitch, roll


_BLINK_PERIOD_S = 3.7
_BLINK_DURATION_S = 0.22


def idle_blink(timestamp: float) -> float:
    """Deterministic blink weight: a short pulse every few seconds."""
    phase = timestamp % _BLINK_PERIOD_S
    if phase >= _BLINK_DURATION_S:
        return 0.0
    return math.sin(math.pi * phase / _BLINK_DURATION_S)


class IdleDriver(FaceDriver):
    """Synthetic presence: breathing-calm sway, periodic blinks, soft smile."""

    name = "idle"

    def update(self, frame_bgr: np.ndarray | None, timestamp: float) -> FaceState:
        yaw, pitch, roll = idle_pose(timestamp)
        state = FaceState(
            present=True, yaw=yaw, pitch=pitch, roll=roll, timestamp=timestamp
        )
        blink = idle_blink(timestamp)
        state.set_channel("eyeBlinkLeft", blink)
        state.set_channel("eyeBlinkRight", blink)
        state.set_channel("mouthSmileLeft", 0.18)
        state.set_channel("mouthSmileRight", 0.18)
        gaze = 0.5 + 0.5 * math.sin(2.0 * math.pi * timestamp / 13.0)
        state.set_channel("eyeLookOutRight", 0.25 * gaze)
        state.set_channel("eyeLookOutLeft", 0.25 * (1.0 - gaze))
        return state


def _pose_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Approximate (yaw, pitch, roll) from a facial transformation matrix."""
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    yaw = math.atan2(rotation[0, 2], rotation[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -rotation[1, 2])))
    roll = math.atan2(rotation[1, 0], rotation[1, 1])
    return yaw, pitch, roll


class VisionDriver(FaceDriver):
    """MediaPipe Face Landmarker tracking on the forwarded camera frames."""

    name = "vision"

    def __init__(
        self,
        cfg: VisionConfig,
        *,
        allow_model_download: bool = True,
        preparation: DriverPreparation | None = None,
    ):
        if cv2 is None:
            raise DriverUnavailableError("opencv-python is required for vision tracking")
        if preparation is not None:
            if preparation.vision_model_setting != cfg.model_path:
                raise ValueError(
                    "driver preparation does not match the vision configuration"
                )
            if preparation._vision_error is not None:
                raise preparation._vision_error
            if preparation.vision_model_path is None:
                raise ValueError("driver preparation contains no vision model")
            bindings = preparation._vision_bindings or _load_vision_bindings()
            model_path = preparation.vision_model_path
        else:
            bindings = _load_vision_bindings()
            model_path = _prepare_vision_model(
                cfg, allow_model_download=allow_model_download
            )
        self._mp = bindings.mediapipe
        options = bindings.vision.FaceLandmarkerOptions(
            base_options=bindings.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=bindings.vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        try:
            self._landmarker = bindings.vision.FaceLandmarker.create_from_options(
                options
            )
        except Exception as exc:  # mediapipe raises framework-specific types
            raise DriverUnavailableError(f"face landmarker failed to start: {exc}") from exc
        self._last_timestamp_ms = -1

    def update(self, frame_bgr: np.ndarray | None, timestamp: float) -> FaceState:
        if frame_bgr is None:
            return FaceState(present=False, timestamp=timestamp)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(self._last_timestamp_ms + 1, int(timestamp * 1000.0))
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return FaceState(present=False, timestamp=timestamp)
        state = FaceState(present=True, timestamp=timestamp)
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                name = category.category_name
                if name in state.blendshapes or name.startswith("_"):
                    continue
                try:
                    state.set_channel(name, category.score)
                except KeyError:
                    continue
        if result.facial_transformation_matrixes:
            state.yaw, state.pitch, state.roll = _pose_from_matrix(
                result.facial_transformation_matrixes[0]
            )
        landmarks = result.face_landmarks[0]
        xs = [landmark.x for landmark in landmarks]
        ys = [landmark.y for landmark in landmarks]
        state.center_x = min(1.0, max(0.0, (min(xs) + max(xs)) / 2.0))
        state.center_y = min(1.0, max(0.0, (min(ys) + max(ys)) / 2.0))
        state.size = min(1.0, max(0.0, max(ys) - min(ys)))
        return state

    def close(self) -> None:
        self._landmarker.close()


def create_driver(
    cfg: DriverConfig,
    *,
    allow_model_download: bool = True,
    preparation: DriverPreparation | None = None,
) -> FaceDriver:
    """Build the configured driver; ``auto`` degrades to the idle animator.

    ``preparation`` is optional for backward compatibility.  When supplied it
    must match the candidate configuration, and vision construction performs
    no model acquisition or path validation.
    """
    if preparation is not None and (
        preparation.backend != cfg.backend
        or preparation.vision_model_setting != cfg.vision.model_path
    ):
        raise ValueError("driver preparation does not match the driver configuration")
    if cfg.backend in ("auto", "vision"):
        try:
            if preparation is not None and preparation._vision_error is not None:
                raise preparation._vision_error
            return VisionDriver(
                cfg.vision,
                allow_model_download=allow_model_download,
                preparation=preparation,
            )
        except (DriverUnavailableError, ModelAcquisitionError) as exc:
            if cfg.backend == "vision":
                raise
            log.warning("vision driver unavailable (%s); using the idle animator", exc)
            return IdleDriver()
    if cfg.backend == "audio2face":
        from .audio2face import Audio2FaceDriver

        driver = Audio2FaceDriver(cfg.audio2face)
        try:
            driver.start()
        except BaseException as startup_error:
            # A source can exist even when worker construction/start fails.
            # Retry its teardown here; if it still survives, transfer the
            # concrete driver identity to the transactional owner.
            try:
                driver.close()
            except BaseException as cleanup_error:
                raise DriverStartupError(driver, startup_error) from cleanup_error
            raise
        return driver
    return IdleDriver()


class ThreadSafeLatestState:
    """Latest :class:`FaceState` shared between a driver thread and renders."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._blendshapes: dict[str, float] = {}
        self._updated_at: float | None = None

    def update_blendshapes(self, weights: dict[str, float], timestamp: float) -> None:
        with self._lock:
            self._blendshapes.update(weights)
            self._updated_at = timestamp

    def snapshot(self) -> tuple[dict[str, float], float | None]:
        with self._lock:
            return dict(self._blendshapes), self._updated_at
