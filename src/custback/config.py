"""Strict application configuration and atomic runtime state."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Output / compositing modes. Kept as a tuple for CLI/API compatibility.
MODES = ("passthrough", "blur", "image", "video", "color", "camera", "remote")
LOCAL_BACKGROUND_MODES = ("blur", "image", "video", "color", "camera")

BackgroundMode = Literal[
    "passthrough", "blur", "image", "video", "color", "camera", "remote"
]
LocalBackgroundMode = Literal["blur", "image", "video", "color", "camera"]
SegmentationBackend = Literal["auto", "rvm", "mediapipe", "heuristic", "none"]
SegmentationDelegate = Literal["cpu", "gpu"]
OutputBackend = Literal["auto", "pyvirtualcam", "null"]
ColorChannel = Annotated[int, Field(ge=0, le=255)]
SAFE_IMAGE_MAX_PIXELS = 89_478_485


def _clean_config_string(value: str, *, allow_empty: bool = True) -> str:
    """Reject invisible/path-breaking input without silently trimming it."""

    if value != value.strip():
        raise ValueError("must not contain leading or trailing whitespace")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError("must not contain control characters")
    if len(value) > 4096:
        raise ValueError("must not exceed 4096 characters")
    if not allow_empty and not value:
        raise ValueError("must not be empty")
    return value


def _clean_device(value: int | str, *, allow_empty: bool) -> int | str:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError("device index must be non-negative")
        return value
    if isinstance(value, str):
        return _clean_config_string(value, allow_empty=allow_empty)
    return value


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """Make failed cross-field assignment validation transactional.

        Pydantic restores neither the assigned field nor ``fields_set`` when
        an ``after`` model validator rejects ``validate_assignment``. Without
        this guard, catching a validation error could leave a supposedly
        strict config object in an invalid mode/path or half-TLS state.
        """
        fields = type(self).model_fields
        if name not in fields or name not in self.__dict__:
            super().__setattr__(name, value)
            return
        previous = self.__dict__[name]
        previous_fields_set = self.__pydantic_fields_set__.copy()
        try:
            super().__setattr__(name, value)
        except BaseException:
            self.__dict__[name] = previous
            self.__pydantic_fields_set__.clear()
            self.__pydantic_fields_set__.update(previous_fields_set)
            raise


class CameraConfig(_StrictModel):
    device: int | str = 0
    width: int = Field(default=1280, ge=16, le=7680)
    height: int = Field(default=720, ge=16, le=7680)
    fps: int = Field(default=30, ge=1, le=240)
    synthetic: bool = False
    mirror: bool = False

    @field_validator("device")
    @classmethod
    def _valid_device(cls, value: int | str) -> int | str:
        return _clean_device(value, allow_empty=False)


class BackgroundConfig(_StrictModel):
    mode: BackgroundMode = "blur"
    image_path: str = ""
    video_path: str = ""
    camera_device: int | str = ""
    color: tuple[ColorChannel, ColorChannel, ColorChannel] = (18, 100, 32)
    blur_strength: int = Field(default=31, ge=3, le=151)
    # Remote output must never reveal the unprocessed camera when the renderer
    # is absent or stale. This selects the local, privacy-safe fallback.
    remote_fallback_mode: LocalBackgroundMode = "blur"

    @field_validator("image_path", "video_path")
    @classmethod
    def _valid_optional_path(cls, value: str) -> str:
        return _clean_config_string(value)

    @field_validator("camera_device")
    @classmethod
    def _valid_camera_device(cls, value: int | str) -> int | str:
        return _clean_device(value, allow_empty=True)

    @field_validator("color", mode="before")
    @classmethod
    def _list_color_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("blur_strength", mode="before")
    @classmethod
    def _odd_blur_kernel(cls, value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value + 1 if value % 2 == 0 else value
        return value

    @model_validator(mode="after")
    def _active_source_is_configured(self) -> "BackgroundConfig":
        active_modes = {self.mode}
        if self.mode == "remote":
            active_modes.add(self.remote_fallback_mode)
        if "image" in active_modes and not self.image_path:
            raise ValueError("image_path is required for the active image background")
        if "video" in active_modes and not self.video_path:
            raise ValueError("video_path is required for the active video background")
        if "camera" in active_modes and self.camera_device == "":
            raise ValueError("camera_device is required for the active camera background")
        return self


class SegmentationConfig(_StrictModel):
    backend: SegmentationBackend = "auto"
    model_path: str = ""
    delegate: SegmentationDelegate = "cpu"
    rvm_downsample: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    mask_blur: int = Field(default=7, ge=0, le=151)
    edge_refine: bool = True
    mask_shift: int = Field(default=0, ge=-20, le=20)
    temporal_smoothing: float = Field(default=0.35, ge=0.0, le=0.95)

    @field_validator("model_path")
    @classmethod
    def _valid_model_path(cls, value: str) -> str:
        return _clean_config_string(value)

    @field_validator("mask_blur", mode="before")
    @classmethod
    def _odd_mask_kernel(cls, value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value + 1 if value % 2 == 0 else value
        return value

    @field_validator("rvm_downsample")
    @classmethod
    def _valid_rvm_downsample(cls, value: float) -> float:
        if value != 0.0 and value < 0.05:
            raise ValueError("rvm_downsample must be 0 (auto) or between 0.05 and 1")
        if not math.isfinite(value):
            raise ValueError("rvm_downsample must be finite")
        return value

    @model_validator(mode="after")
    def _model_matches_backend(self) -> "SegmentationConfig":
        if not self.model_path:
            if self.delegate == "gpu" and self.backend in ("rvm", "heuristic", "none"):
                raise ValueError("the GPU delegate is only used by auto/mediapipe")
            return self
        suffix = Path(self.model_path).suffix.lower()
        if self.backend == "rvm" and suffix != ".onnx":
            raise ValueError("rvm model_path must end in .onnx")
        if self.backend == "mediapipe" and suffix != ".tflite":
            raise ValueError("mediapipe model_path must end in .tflite")
        if self.backend == "auto" and suffix not in (".onnx", ".tflite"):
            raise ValueError("auto model_path must end in .onnx or .tflite")
        if self.backend in ("heuristic", "none"):
            raise ValueError(f"{self.backend} does not use model_path")
        if self.delegate == "gpu" and self.backend == "rvm":
            raise ValueError("the GPU delegate is only used by auto/mediapipe")
        return self


class CompositingConfig(_StrictModel):
    light_wrap: float = Field(default=0.25, ge=0.0, le=1.0)
    use_model_foreground: bool = True


class OutputConfig(_StrictModel):
    backend: OutputBackend = "auto"
    device: str = ""
    fps: int = Field(default=30, ge=1, le=240)
    preview: bool = False

    @field_validator("device")
    @classmethod
    def _valid_output_device(cls, value: str) -> str:
        return _clean_config_string(value)


class UploadLimits(_StrictModel):
    image_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1, le=2**31)
    video_max_bytes: int = Field(default=256 * 1024 * 1024, ge=1, le=2**31)
    # Stay at or below Pillow's decompression-bomb warning threshold across
    # the supported Pillow 10-12 range so the configured limit is authoritative.
    image_max_pixels: int = Field(
        default=16_777_216, ge=256, le=SAFE_IMAGE_MAX_PIXELS
    )
    video_max_width: int = Field(default=3840, ge=16, le=7680)
    video_max_height: int = Field(default=2160, ge=16, le=7680)
    storage_max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1, le=2**40)
    max_files: int = Field(default=100, ge=1, le=100_000)

    @model_validator(mode="after")
    def _storage_can_hold_one_file(self) -> "UploadLimits":
        largest = max(self.image_max_bytes, self.video_max_bytes)
        if self.storage_max_bytes < largest:
            raise ValueError("storage_max_bytes must be at least the largest per-file limit")
        return self


class ApiConfig(_StrictModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8710, ge=1, le=65535)
    remote_timeout_ms: int = Field(default=250, ge=1, le=60_000)
    allow_non_loopback: bool = False
    allowed_origins: tuple[str, ...] = ()
    token_file: str = "~/.config/custback/api-token"
    session_ttl_s: int = Field(default=28_800, ge=60, le=31_536_000)
    tls_certfile: str = ""
    tls_keyfile: str = ""
    ws_max_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=2**31)
    uploads: UploadLimits = Field(default_factory=UploadLimits)

    @field_validator("host")
    @classmethod
    def _valid_host(cls, value: str) -> str:
        from .api.security import normalize_bind_host

        normalized = normalize_bind_host(value)
        if normalized is None:
            raise ValueError("host must be a DNS name or IP literal without a port")
        return normalized

    @field_validator("token_file")
    @classmethod
    def _valid_token_file(cls, value: str) -> str:
        return _clean_config_string(value, allow_empty=False)

    @field_validator("tls_certfile", "tls_keyfile")
    @classmethod
    def _valid_optional_security_path(cls, value: str) -> str:
        return _clean_config_string(value)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _valid_origins(cls, value: Any) -> tuple[str, ...]:
        from .api.security import canonical_origin

        if isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, tuple):
            return value
        normalized: list[str] = []
        for origin in value:
            canonical = canonical_origin(origin)
            if canonical is None:
                raise ValueError(f"invalid exact HTTP(S) origin: {origin!r}")
            normalized.append(canonical)
        if len(normalized) != len(set(normalized)):
            raise ValueError("allowed_origins entries must be unique after normalization")
        return tuple(normalized)

    @model_validator(mode="after")
    def _tls_pair(self) -> "ApiConfig":
        if bool(self.tls_certfile) != bool(self.tls_keyfile):
            raise ValueError("tls_certfile and tls_keyfile must be configured together")
        return self


class AppConfig(_StrictModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    compositing: CompositingConfig = Field(default_factory=CompositingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return plain Python values, preserving tuple compatibility."""
        return self.model_dump(mode="python")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppConfig":
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("configuration root must be a mapping")
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path | None) -> "AppConfig":
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text())
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def validate(self) -> None:
        """Compatibility hook; assignment validation keeps the model valid."""
        type(self).model_validate(self.to_dict())

    def patched(self, patch: dict[str, Any]) -> "AppConfig":
        """Validate a one-level partial section patch without mutating self."""
        if not isinstance(patch, dict):
            raise TypeError("config patch must be a mapping")
        merged = self.to_dict()
        for section, values in patch.items():
            if section not in merged:
                # Preserve the unknown key so Pydantic reports it as forbidden.
                merged[section] = values
            elif not isinstance(values, dict):
                merged[section] = values
            else:
                current = merged[section]
                if not isinstance(current, dict):  # defensive; sections are models
                    merged[section] = values
                else:
                    current.update(values)
        return type(self).from_dict(merged)


@dataclass(frozen=True)
class ConfigState:
    config: AppConfig
    version: int


class ConfigVersionConflictError(RuntimeError):
    def __init__(self, expected_version: int, current_version: int):
        self.expected_version = expected_version
        self.current_version = current_version
        super().__init__(
            f"configuration changed concurrently: expected version "
            f"{expected_version}, current version is {current_version}"
        )


class _RuntimeConfigCoordinator:
    """Private mutation capability held by the live pipeline coordinator."""

    __slots__ = ("__runtime",)

    def __init__(self, runtime: "RuntimeConfig"):
        self.__runtime = runtime

    def commit(
        self, candidate: AppConfig, expected_version: int
    ) -> ConfigState:
        return self.__runtime._commit_with_activation(
            candidate, expected_version, lambda _: None
        )

    def commit_with_activation(
        self,
        candidate: AppConfig,
        expected_version: int,
        activate: Callable[[int], None],
    ) -> ConfigState:
        return self.__runtime._commit_with_activation(
            candidate, expected_version, activate
        )


class RuntimeConfig:
    """Atomic, versioned application configuration state."""

    def __init__(self, config: AppConfig):
        self._config = AppConfig.from_dict(config.to_dict())
        self._lock = threading.Lock()
        self._version = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def read(self) -> ConfigState:
        """Atomically return a matching configuration snapshot and version."""
        with self._lock:
            return ConfigState(self._config.model_copy(deep=True), self._version)

    def snapshot(self) -> AppConfig:
        """Compatibility wrapper around :meth:`read`."""
        return self.read().config

    def _coordinator_writer(self) -> _RuntimeConfigCoordinator:
        """Return the private write capability used by :class:`Pipeline`."""
        return _RuntimeConfigCoordinator(self)

    def commit(self, candidate: AppConfig, expected_version: int) -> ConfigState:
        """Reject config-only CAS writes that bypass resource activation."""
        raise RuntimeError(
            "direct RuntimeConfig.commit() is disabled; use the pipeline "
            "reconfiguration coordinator"
        )

    def commit_with_activation(
        self,
        candidate: AppConfig,
        expected_version: int,
        activate: Callable[[int], None],
    ) -> ConfigState:
        """Reject callers that do not hold the private coordinator capability."""
        raise RuntimeError(
            "direct RuntimeConfig.commit_with_activation() is disabled; use the "
            "pipeline reconfiguration coordinator"
        )

    def _commit_with_activation(
        self,
        candidate: AppConfig,
        expected_version: int,
        activate: Callable[[int], None],
    ) -> ConfigState:
        """Atomically activate resources and publish their matching config.

        ``activate`` runs while readers are excluded and receives the version
        that will be committed. It must only perform the already-prepared,
        non-blocking resource swap; construction and teardown belong outside
        this critical section. If it raises, configuration remains unchanged.
        """
        validated = AppConfig.from_dict(candidate.to_dict())
        with self._lock:
            if self._version != expected_version:
                raise ConfigVersionConflictError(expected_version, self._version)
            next_version = self._version + 1
            activate(next_version)
            self._config = validated
            self._version = next_version
            return ConfigState(self._config.model_copy(deep=True), self._version)

    def update(self, patch: dict[str, Any]) -> AppConfig:
        """Reject legacy config-first mutation that bypasses resource staging."""
        raise RuntimeError(
            "direct RuntimeConfig.update() is disabled; use the pipeline "
            "reconfiguration coordinator"
        )
