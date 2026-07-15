"""Strict avatar-service configuration and its versioned runtime state.

The models mirror :mod:`custback.config`: strict validation, transactional
assignment, YAML loading, and one-level section patches. ``AvatarRuntime``
is a simplified counterpart of ``RuntimeConfig`` — the avatar service has no
hardware to stage, so hot sections activate on the next rendered frame and
everything else requires a restart.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from ..config import _clean_config_string, _StrictModel
from .state import ARKIT_BLENDSHAPES  # noqa: F401  (re-exported contract)

DriverBackend = Literal["auto", "vision", "audio2face", "idle"]
AvatarBackgroundMode = Literal["color", "image", "video", "blur"]
AvatarStyle = Literal["cartoon", "realistic", "sketch"]
AvatarFraming = Literal["full", "bust", "closeup"]
ColorChannel = Annotated[int, Field(ge=0, le=255)]

# Renderable avatar layers, in compositing order (torso first, hair last).
AVATAR_PARTS: tuple[str, ...] = (
    "torso", "head", "mouth", "nose", "eyes", "brows", "hair",
)

# Builtin presenter characters; the preset art lives in rig.py.
BUILTIN_AVATARS: tuple[str, ...] = ("casey", "robin", "alex", "nova")

# Sections that activate live on the next rendered frame. Everything else
# (source connection, control-plane bind/security) needs a restart.
HOT_SECTIONS = frozenset({"appearance", "background", "driver", "render"})
RESTART_FIELDS = frozenset(
    {
        "driver.audio2face.url",
        "driver.audio2face.tls_ca_file",
        "driver.audio2face.tls_certfile",
        "driver.audio2face.tls_keyfile",
    }
)


class RestartRequiredError(RuntimeError):
    """The patch changes fields that only apply on service restart."""

    def __init__(self, fields: tuple[str, ...], current_version: int):
        self.fields = fields
        self.current_version = current_version
        super().__init__(
            "restart required to apply: " + ", ".join(fields)
        )


def _valid_ws_url(value: str) -> str:
    from ..api.security import validate_outbound_endpoint

    value = _clean_config_string(value, allow_empty=False)
    endpoint = validate_outbound_endpoint(
        value,
        kind="websocket",
        label="source.url",
    )
    assert endpoint is not None
    return endpoint.url


class SourceConfig(_StrictModel):
    """Where the custback frame API lives and how to authenticate to it."""

    url: str = "ws://127.0.0.1:8710"
    # Renderer-scoped frame token; CUSTBACK_RENDERER_TOKEN overrides the file.
    token_file: str = "~/.config/custback/renderer-token"
    tls_ca_file: str = ""
    tls_certfile: str = ""
    tls_keyfile: str = ""
    connect_timeout_s: float = Field(default=5.0, ge=0.5, le=60.0)
    reconnect_min_s: float = Field(default=0.5, ge=0.1, le=60.0)
    reconnect_max_s: float = Field(default=10.0, ge=0.5, le=300.0)
    frame_max_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=2**31)

    @field_validator("url")
    @classmethod
    def _valid_url(cls, value: str) -> str:
        return _valid_ws_url(value)

    @field_validator("token_file")
    @classmethod
    def _valid_token_file(cls, value: str) -> str:
        return _clean_config_string(value, allow_empty=False)

    @field_validator("tls_ca_file", "tls_certfile", "tls_keyfile")
    @classmethod
    def _valid_optional_tls_path(cls, value: str) -> str:
        return _clean_config_string(value)

    @model_validator(mode="after")
    def _backoff_window(self) -> "SourceConfig":
        from ..api.security import validate_client_tls, validate_outbound_endpoint

        if self.reconnect_max_s < self.reconnect_min_s:
            raise ValueError("reconnect_max_s must be at least reconnect_min_s")
        endpoint = validate_outbound_endpoint(
            self.url,
            kind="websocket",
            label="source.url",
        )
        validate_client_tls(
            endpoint,
            ca_file=self.tls_ca_file,
            certfile=self.tls_certfile,
            keyfile=self.tls_keyfile,
            label="source",
        )
        return self


class VisionConfig(_StrictModel):
    """MediaPipe Face Landmarker driver options."""

    model_path: str = ""  # optional .task override; empty = managed download

    @field_validator("model_path")
    @classmethod
    def _valid_model_path(cls, value: str) -> str:
        value = _clean_config_string(value)
        if value and Path(value).suffix.lower() != ".task":
            raise ValueError("vision model_path must end in .task")
        return value


class Audio2FaceConfig(_StrictModel):
    """Client options for an NVIDIA Audio2Face-3D gRPC service (NIM)."""

    url: str = ""  # grpc://numeric-loopback:port or grpcs://host:port
    tls_ca_file: str = ""
    tls_certfile: str = ""
    tls_keyfile: str = ""
    audio_source: str = "microphone"  # "microphone" or a 16 kHz mono .wav path
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    chunk_ms: int = Field(default=40, ge=10, le=1000)

    @field_validator("audio_source")
    @classmethod
    def _valid_audio_source(cls, value: str) -> str:
        return _clean_config_string(value)

    @field_validator("url")
    @classmethod
    def _valid_url(cls, value: str) -> str:
        from ..api.security import validate_outbound_endpoint

        value = _clean_config_string(value)
        if not value:
            return ""
        # Preserve the old numeric-loopback host:port spelling as a safe
        # migration, but persist the explicit transport form.
        candidate = value if "://" in value else f"grpc://{value}"
        endpoint = validate_outbound_endpoint(
            candidate,
            kind="grpc",
            label="driver.audio2face.url",
            require_port=True,
        )
        assert endpoint is not None
        return endpoint.url

    @field_validator("tls_ca_file", "tls_certfile", "tls_keyfile")
    @classmethod
    def _valid_optional_tls_path(cls, value: str) -> str:
        return _clean_config_string(value)

    @model_validator(mode="after")
    def _audio_source_configured(self) -> "Audio2FaceConfig":
        from ..api.security import validate_client_tls, validate_outbound_endpoint

        if not self.audio_source:
            raise ValueError("audio_source must be 'microphone' or a .wav path")
        endpoint = validate_outbound_endpoint(
            self.url,
            kind="grpc",
            label="driver.audio2face.url",
            allow_empty=True,
            require_port=True,
        )
        validate_client_tls(
            endpoint,
            ca_file=self.tls_ca_file,
            certfile=self.tls_certfile,
            keyfile=self.tls_keyfile,
            label="driver.audio2face",
        )
        return self


class DriverConfig(_StrictModel):
    backend: DriverBackend = "auto"
    smoothing: float = Field(default=0.4, ge=0.0, le=0.95)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    audio2face: Audio2FaceConfig = Field(default_factory=Audio2FaceConfig)

    @model_validator(mode="after")
    def _backend_is_configured(self) -> "DriverConfig":
        if self.backend == "audio2face" and not self.audio2face.url:
            raise ValueError("audio2face.url is required for the audio2face backend")
        return self


class AppearanceConfig(_StrictModel):
    rig: str = "builtin"  # "builtin" or a rig directory of PNG layers
    avatar: str = "casey"  # builtin character; ignored for PNG-layer rigs
    style: AvatarStyle = "cartoon"  # PNG-layer rigs support sketch only
    # bust keeps the head and chest in frame (webcam-style, right for
    # meeting tiles); full shows everything the rig has, closeup the face.
    framing: AvatarFraming = "bust"
    parts: tuple[str, ...] = AVATAR_PARTS
    scale: float = Field(default=1.0, ge=0.1, le=3.0)
    offset_x: float = Field(default=0.0, ge=-1.0, le=1.0)
    offset_y: float = Field(default=0.0, ge=-1.0, le=1.0)
    follow_pose: bool = True

    @field_validator("rig")
    @classmethod
    def _valid_rig(cls, value: str) -> str:
        return _clean_config_string(value, allow_empty=False)

    @field_validator("avatar")
    @classmethod
    def _known_avatar(cls, value: str) -> str:
        value = _clean_config_string(value, allow_empty=False)
        if value not in BUILTIN_AVATARS:
            raise ValueError(
                f"unknown builtin avatar {value!r}; choose from {list(BUILTIN_AVATARS)}"
            )
        return value

    @field_validator("parts", mode="before")
    @classmethod
    def _list_parts_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("parts")
    @classmethod
    def _known_unique_parts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one avatar part must be visible")
        unknown = [part for part in value if part not in AVATAR_PARTS]
        if unknown:
            raise ValueError(
                f"unknown avatar parts {unknown!r}; choose from {list(AVATAR_PARTS)}"
            )
        if len(set(value)) != len(value):
            raise ValueError("avatar parts must be unique")
        return value


class AvatarBackgroundConfig(_StrictModel):
    """The scene behind the avatar. ``blur`` shows the real room blurred."""

    mode: AvatarBackgroundMode = "color"
    color: tuple[ColorChannel, ColorChannel, ColorChannel] = (60, 46, 32)
    image_path: str = ""
    video_path: str = ""
    blur_strength: int = Field(default=31, ge=3, le=151)

    @field_validator("image_path", "video_path")
    @classmethod
    def _valid_optional_path(cls, value: str) -> str:
        return _clean_config_string(value)

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
    def _active_source_is_configured(self) -> "AvatarBackgroundConfig":
        if self.mode == "image" and not self.image_path:
            raise ValueError("image_path is required for the image background")
        if self.mode == "video" and not self.video_path:
            raise ValueError("video_path is required for the video background")
        return self


class RenderConfig(_StrictModel):
    max_fps: int = Field(default=30, ge=1, le=240)
    jpeg_quality: int = Field(default=85, ge=30, le=100)


class StorageConfig(_StrictModel):
    """On-disk stores for uploaded rigs and avatar scene media.

    These live on the avatar host (uploads arrive through the control API),
    so a remote deployment keeps its rigs and scene files next to the
    service that renders them.
    """

    rigs_dir: str = "~/.local/share/custback/rigs"
    backgrounds_dir: str = "~/.local/share/custback/avatar-backgrounds"
    image_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=2**31)
    video_max_bytes: int = Field(default=256 * 1024 * 1024, ge=1024, le=2**31)
    # Stays at or below Pillow's decompression-bomb warning threshold, like
    # custback's own upload limits, so the configured cap is authoritative.
    image_max_pixels: int = Field(default=16_777_216, ge=256, le=89_478_485)
    video_max_width: int = Field(default=3840, ge=16, le=7680)
    video_max_height: int = Field(default=2160, ge=16, le=7680)
    # Rig archives: compressed upload size, uncompressed payload, and the
    # number of layer files a single rig may contain.
    rig_zip_max_bytes: int = Field(default=64 * 1024 * 1024, ge=1024, le=2**31)
    rig_max_bytes: int = Field(default=128 * 1024 * 1024, ge=1024, le=2**31)
    rig_max_entries: int = Field(default=16, ge=1, le=64)
    storage_max_bytes: int = Field(
        default=1024 * 1024 * 1024, ge=1024, le=2**40
    )
    max_files: int = Field(default=100, ge=1, le=100_000)

    @field_validator("rigs_dir", "backgrounds_dir")
    @classmethod
    def _valid_directory(cls, value: str) -> str:
        return _clean_config_string(value, allow_empty=False)

    @model_validator(mode="after")
    def _storage_can_hold_one_file(self) -> "StorageConfig":
        largest = max(self.image_max_bytes, self.video_max_bytes)
        if self.storage_max_bytes < largest:
            raise ValueError(
                "storage_max_bytes must be at least the largest per-file limit"
            )
        return self


class AvatarApiConfig(_StrictModel):
    """The avatar service's own control plane (not custback's API)."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8711, ge=1, le=65535)
    token_file: str = "~/.config/custback/avatar-api-token"
    allow_non_loopback: bool = False
    allowed_origins: tuple[str, ...] = ()
    session_ttl_s: int = Field(default=28_800, ge=60, le=31_536_000)
    tls_certfile: str = ""
    tls_keyfile: str = ""
    # No WebSocket routes yet; kept for uvicorn parity with custback's API.
    ws_max_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=2**31)

    @field_validator("host")
    @classmethod
    def _valid_host(cls, value: str) -> str:
        from ..api.security import normalize_bind_host

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
    def _valid_origins(cls, value: Any) -> Any:
        from ..api.security import canonical_origin

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
    def _tls_pair(self) -> "AvatarApiConfig":
        if bool(self.tls_certfile) != bool(self.tls_keyfile):
            raise ValueError("tls_certfile and tls_keyfile must be configured together")
        return self


class AvatarConfig(_StrictModel):
    source: SourceConfig = Field(default_factory=SourceConfig)
    driver: DriverConfig = Field(default_factory=DriverConfig)
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    background: AvatarBackgroundConfig = Field(default_factory=AvatarBackgroundConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: AvatarApiConfig = Field(default_factory=AvatarApiConfig)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AvatarConfig":
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("configuration root must be a mapping")
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path | None) -> "AvatarConfig":
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

    def patched(self, patch: dict[str, Any]) -> "AvatarConfig":
        """Validate a one-level partial section patch without mutating self."""
        if not isinstance(patch, dict):
            raise TypeError("config patch must be a mapping")
        merged = self.to_dict()
        for section, values in patch.items():
            if section not in merged or not isinstance(values, dict):
                # Preserve the unknown key so Pydantic reports it as forbidden.
                merged[section] = values
            else:
                current = merged[section]
                if not isinstance(current, dict):  # defensive; sections are models
                    merged[section] = values
                else:
                    current.update(values)
        return type(self).from_dict(merged)


@dataclass(frozen=True)
class AvatarConfigState:
    config: AvatarConfig
    version: int


def _changed_fields(
    before: dict[str, Any], after: dict[str, Any], prefix: str = ""
) -> list[str]:
    changed: list[str] = []
    for key in sorted(set(before) | set(after)):
        path = f"{prefix}{key}"
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            changed.extend(_changed_fields(old, new, f"{path}."))
        elif old != new:
            changed.append(path)
    return changed


class AvatarRuntime:
    """Atomic, versioned avatar configuration.

    A patch that changes any field outside :data:`HOT_SECTIONS` raises
    :class:`RestartRequiredError` and applies nothing — including the hot
    part of a mixed patch, matching custback's transactional PATCH contract.
    """

    def __init__(self, config: AvatarConfig):
        self._config = AvatarConfig.from_dict(config.to_dict())
        self._lock = threading.Lock()
        self._version = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def read(self) -> AvatarConfigState:
        with self._lock:
            return AvatarConfigState(self._config.model_copy(deep=True), self._version)

    def apply_patch(self, patch: dict[str, Any]) -> AvatarConfigState:
        with self._lock:
            candidate = self._config.patched(patch)
            before = self._config.to_dict()
            after = candidate.to_dict()
            changed = _changed_fields(before, after)
            if not changed:
                return AvatarConfigState(self._config.model_copy(deep=True), self._version)
            restart = tuple(
                field for field in changed
                if field.split(".", 1)[0] not in HOT_SECTIONS
                or field in RESTART_FIELDS
            )
            if restart:
                raise RestartRequiredError(restart, self._version)
            self._config = candidate
            self._version += 1
            return AvatarConfigState(self._config.model_copy(deep=True), self._version)
