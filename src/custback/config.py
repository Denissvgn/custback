"""Strict application configuration and atomic runtime state."""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from . import _platform as platform_fs
from .config_merge import merge_patch

# Output / compositing modes. Kept as a tuple for CLI/API compatibility.
MODES = ("passthrough", "blur", "image", "video", "color", "camera", "remote")
LOCAL_BACKGROUND_MODES = ("blur", "image", "video", "color", "camera")

BackgroundMode = Literal[
    "passthrough", "blur", "image", "video", "color", "camera", "remote"
]
LocalBackgroundMode = Literal["blur", "image", "video", "color", "camera"]
SegmentationBackend = Literal["auto", "rvm", "mediapipe", "heuristic", "none"]
SegmentationDelegate = Literal["cpu", "gpu"]
BoundaryStabilizationMode = Literal["off", "motion_aware"]
SpatialEdgeRefinementMode = Literal["legacy_watershed", "stable_guided"]
AccelerationMode = Literal["auto", "cpu", "gpu_required"]
AccelerationProvider = Literal["auto", "cuda", "directml"]
OutputBackend = Literal["auto", "pyvirtualcam", "native", "null"]
CameraPixelFormat = Literal["auto", "mjpeg", "backend"]
CameraModeMismatch = Literal["warn", "error"]
FitMode = Literal["cover", "contain", "stretch"]
RightAngleRotation = Literal[0, 90, 180, 270]
BlendSpace = Literal["srgb_legacy", "linear_srgb"]
ColorCorrectionMode = Literal["off", "auto"]
LightWrapStabilizationMode = Literal["off", "temporal_bounded"]
VideoColorMatrix = Literal["auto", "bt601", "bt709"]
VideoColorRange = Literal["auto", "limited", "full"]
VideoColorPrimaries = Literal["auto", "bt709", "bt470bg", "smpte170m"]
VideoColorTransfer = Literal["auto", "srgb", "bt709"]
LEGACY_CONFIG_SCHEMA_VERSION = 1
CURRENT_CONFIG_SCHEMA_VERSION = 1
ColorChannel = Annotated[int, Field(ge=0, le=255)]
FrameDimension = Annotated[int, Field(ge=16, le=7680)]
Anchor = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
UnitStrength = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
ExposureLimitEv = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
AdaptationTimeSeconds = Annotated[float, Field(ge=0.05, le=10.0, allow_inf_nan=False)]
LightWrapTimeConstantSeconds = Annotated[
    float,
    Field(ge=0.01, le=1.0, allow_inf_nan=False),
]
SchemaVersion = Annotated[
    int,
    Field(
        strict=True,
        ge=LEGACY_CONFIG_SCHEMA_VERSION,
        le=CURRENT_CONFIG_SCHEMA_VERSION,
    ),
]
SAFE_IMAGE_MAX_PIXELS = 89_478_485
_BACKDROP_TARGET_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\\\/]")


def materialize_config_schema_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Copy a persisted mapping and bind absent fields to its schema semantics.

    Keeping this dispatch separate from Pydantic model defaults is what makes a
    versionless or explicit schema-v1 document retain v1 behavior after a later
    release introduces different new-install defaults.
    """

    version = data.get("schema_version", LEGACY_CONFIG_SCHEMA_VERSION)
    materialized = {"schema_version": version}
    materialized.update(
        (key, value) for key, value in data.items() if key != "schema_version"
    )
    if type(version) is not int or version != LEGACY_CONFIG_SCHEMA_VERSION:
        # Strict schema validation reports malformed and unsupported versions.
        # A later supported version must add its own explicit defaults here.
        return materialized

    def bind_section(section_name: str, defaults: dict[str, Any]) -> None:
        section = materialized.get(section_name)
        if section is None and section_name not in materialized:
            materialized[section_name] = dict(defaults)
        elif isinstance(section, dict):
            bound = dict(section)
            for field_name, value in defaults.items():
                bound.setdefault(field_name, value)
            materialized[section_name] = bound

    bind_section(
        "camera",
        {
            "fit_mode": "stretch",
            "anchor_x": 0.5,
            "anchor_y": 0.5,
            "rotation": 0,
        },
    )
    bind_section(
        "background",
        {
            "fit_mode": "cover",
            "anchor_x": 0.5,
            "anchor_y": 0.5,
            "video_color_matrix": "auto",
            "video_color_range": "auto",
            "video_color_primaries": "auto",
            "video_color_transfer": "auto",
        },
    )
    bind_section("output", {"width": None, "height": None})
    bind_section(
        "compositing",
        {
            "blend_space": "srgb_legacy",
            "light_wrap_stabilization": {
                "mode": "off",
                "time_constant_s": 0.12,
            },
        },
    )
    bind_section(
        "segmentation",
        {
            # Schema-v1 persisted configurations predate motion-aware
            # stabilization. Bind them to the exact legacy EMA policy rather
            # than letting a future new-install default reinterpret the file.
            "boundary_stabilization": {
                "mode": "off",
                "time_constant_s": 0.1,
                "max_motion_px_per_s": 720.0,
            },
            # Schema-v1 edge refinement is the fixed-radius watershed path.
            # Persisted configurations must never inherit a later install
            # default that would reinterpret ``edge_refine: true``.
            "spatial_edge_refinement": {
                "mode": "legacy_watershed",
                "reference_short_edge_px": 720,
                "radius_at_reference_px": 8,
                "min_radius_px": 2,
                "max_radius_px": 12,
            },
        },
    )
    segmentation = materialized.get("segmentation")
    if isinstance(segmentation, dict):
        spatial_defaults = {
            "mode": "legacy_watershed",
            "reference_short_edge_px": 720,
            "radius_at_reference_px": 8,
            "min_radius_px": 2,
            "max_radius_px": 12,
        }
        spatial = segmentation.get("spatial_edge_refinement")
        if isinstance(spatial, dict):
            bound_spatial = dict(spatial)
            for field_name, value in spatial_defaults.items():
                bound_spatial.setdefault(field_name, value)
            segmentation["spatial_edge_refinement"] = bound_spatial

    compositing = materialized.get("compositing")
    if isinstance(compositing, dict):
        correction_defaults = {
            "mode": "off",
            "strength": 0.5,
            "exposure_limit_ev": 0.85,
            "white_balance_strength": 0.5,
            "adaptation_time_s": 0.8,
        }
        correction = compositing.get("color_correction")
        if correction is None and "color_correction" not in compositing:
            compositing["color_correction"] = correction_defaults
        elif isinstance(correction, dict):
            bound_correction = dict(correction)
            for field_name, value in correction_defaults.items():
                bound_correction.setdefault(field_name, value)
            compositing["color_correction"] = bound_correction
    return materialized


# These fields select the avatar proxy's outbound security boundary.  They are
# consumed when the API application is constructed and cannot safely diverge
# from the immutable destination, credential-path, and TLS snapshot held by the
# proxy. The token value at that fixed path is deliberately read per request.
AVATAR_PROXY_RESTART_ONLY_FIELDS = frozenset(
    {
        "avatar.url",
        "avatar.token_file",
        "avatar.tls_ca_file",
        "avatar.tls_certfile",
        "avatar.tls_keyfile",
    }
)


def format_config_error(exc: BaseException) -> str:
    """Return one value-free line suitable for a CLI configuration error."""

    if isinstance(exc, ValidationError):
        errors = exc.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first.get("loc", ()))
            message = str(first.get("msg", "invalid value")).splitlines()[0]
            prefix = f"{location}: " if location else ""
            remaining = len(errors) - 1
            suffix = f" (+{remaining} more error(s))" if remaining else ""
            return f"{prefix}{message}{suffix}"
    text = str(exc).splitlines()
    return text[0] if text else type(exc).__name__


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


def _clean_backdrop_source(value: int | str, *, allow_empty: bool) -> int | str:
    """Normalize an operator-owned OpenCV source without accepting URLs.

    OpenCV does not expose enough redirect, proxy, certificate, or address-class
    controls to make a remote ``VideoCapture`` URL a verified transport.  Local
    device names and paths remain available as startup authority, but URI-like
    strings are rejected before OpenCV, DNS, or the filesystem is consulted.
    """

    value = _clean_device(value, allow_empty=allow_empty)
    if isinstance(value, str):
        is_windows_drive = bool(_WINDOWS_DRIVE_RE.match(value))
        if value and (
            value.startswith(("//", "\\\\"))
            or "://" in value
            or (_URI_SCHEME_RE.match(value) and not is_windows_drive)
        ):
            raise ValueError(
                "remote or URI camera backdrop sources are unsupported; "
                "configure a local device target"
            )
        if (
            value
            and not value.isdecimal()
            and not value.startswith("/")
            and not is_windows_drive
        ):
            raise ValueError(
                "camera backdrop source must be a numeric index or absolute local path"
            )
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
        values = cast(dict[str, Any], self.__dict__)
        if name not in fields or name not in values:
            super().__setattr__(name, value)
            return
        previous = values[name]
        previous_fields_set = self.__pydantic_fields_set__.copy()
        try:
            super().__setattr__(name, value)
        except BaseException:
            # Pydantic may replace the instance dictionary while applying an
            # assignment validator, so restore through its current mapping.
            cast(dict[str, Any], self.__dict__)[name] = previous
            self.__pydantic_fields_set__.clear()
            self.__pydantic_fields_set__.update(previous_fields_set)
            raise


class CameraConfig(_StrictModel):
    device: int | str = 0
    width: FrameDimension = 1280
    height: FrameDimension = 720
    fps: int = Field(default=30, ge=1, le=240)
    pixel_format: CameraPixelFormat = "auto"
    mode_mismatch: CameraModeMismatch = "warn"
    recovery_timeout_s: float = Field(default=10.0, ge=2.0, le=300.0)
    fit_mode: FitMode = "stretch"
    anchor_x: Anchor = 0.5
    anchor_y: Anchor = 0.5
    rotation: RightAngleRotation = 0
    synthetic: bool = False
    mirror: bool = False

    @field_validator("device")
    @classmethod
    def _valid_device(cls, value: int | str) -> int | str:
        return _clean_device(value, allow_empty=False)

    @field_validator("rotation", mode="before")
    @classmethod
    def _strict_right_angle_rotation(cls, value: object) -> object:
        # Pydantic's integer Literal matching otherwise accepts 90.0 as equal
        # to 90 even under a strict parent model.
        if type(value) is not int or value not in (0, 90, 180, 270):
            raise ValueError("rotation must be one of 0, 90, 180, or 270")
        return value

    @model_validator(mode="after")
    def _recovery_outlasts_stall_detection(self) -> "CameraConfig":
        stall_after_s = max(2.0, 5.0 / self.fps)
        if self.recovery_timeout_s <= stall_after_s:
            raise ValueError(
                "recovery_timeout_s must be greater than the camera stall threshold "
                f"({stall_after_s:g}s at {self.fps} fps)"
            )
        return self


class BackdropTargetConfig(_StrictModel):
    """One immutable, operator-owned live-backdrop source."""

    source: int | str

    @field_validator("source")
    @classmethod
    def _valid_source(cls, value: int | str) -> int | str:
        return _clean_backdrop_source(value, allow_empty=False)


@dataclass(frozen=True)
class ResolvedBackdropTarget:
    """Validated source snapshot handed to a candidate resource."""

    identifier: str
    source: int | str


class BackgroundConfig(_StrictModel):
    mode: BackgroundMode = "blur"
    image_path: str = ""
    video_path: str = ""
    camera_device: int | str = ""
    camera_target: str = ""
    color: tuple[ColorChannel, ColorChannel, ColorChannel] = (18, 100, 32)
    blur_strength: int = Field(default=31, ge=3, le=151)
    fit_mode: FitMode = "cover"
    anchor_x: Anchor = 0.5
    anchor_y: Anchor = 0.5
    # Metadata is resolved from each decoded frame first. These YAML-only
    # overrides are for reproducible operator-owned assets with absent/wrong
    # declarations; no pixel histogram inference is permitted.
    video_color_matrix: VideoColorMatrix = "auto"
    video_color_range: VideoColorRange = "auto"
    video_color_primaries: VideoColorPrimaries = "auto"
    video_color_transfer: VideoColorTransfer = "auto"
    # Local mode restored when remote/avatar mode is disabled. While remote is
    # active, every renderer failure emits the fixed input-independent slate.
    remote_fallback_mode: LocalBackgroundMode = "blur"

    @field_validator("image_path", "video_path")
    @classmethod
    def _valid_optional_path(cls, value: str) -> str:
        return _clean_config_string(value)

    @field_validator("camera_device")
    @classmethod
    def _valid_camera_device(cls, value: int | str) -> int | str:
        return _clean_backdrop_source(value, allow_empty=True)

    @field_validator("camera_target")
    @classmethod
    def _valid_camera_target(cls, value: str) -> str:
        value = _clean_config_string(value)
        if value and not _BACKDROP_TARGET_ID_RE.fullmatch(value):
            raise ValueError("camera_target must match [a-z][a-z0-9_-]{0,63}")
        return value

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
        if self.camera_device != "" and self.camera_target:
            raise ValueError("camera_device and camera_target are mutually exclusive")
        if (
            "camera" in active_modes
            and self.camera_device == ""
            and not self.camera_target
        ):
            raise ValueError(
                "camera_target or camera_device is required for the active "
                "camera background"
            )
        return self


class BoundaryStabilizationConfig(_StrictModel):
    """Opt-in, elapsed-time boundary stabilization policy.

    ``off`` deliberately means only this new policy is disabled; the separate
    ``temporal_smoothing`` compatibility EMA retains its historical meaning.
    """

    mode: BoundaryStabilizationMode = "off"
    time_constant_s: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        allow_inf_nan=False,
    )
    max_motion_px_per_s: float = Field(
        default=720.0,
        ge=1.0,
        le=30_720.0,
        allow_inf_nan=False,
    )


class SpatialEdgeRefinementConfig(_StrictModel):
    """Resolution-aware spatial edge refinement policy."""

    mode: SpatialEdgeRefinementMode = "legacy_watershed"
    reference_short_edge_px: int = Field(default=720, ge=16, le=7680)
    radius_at_reference_px: int = Field(default=8, ge=1, le=32)
    min_radius_px: int = Field(default=2, ge=1, le=32)
    max_radius_px: int = Field(default=12, ge=1, le=32)

    @model_validator(mode="after")
    def _radius_bounds_include_reference(self) -> "SpatialEdgeRefinementConfig":
        if not (
            self.min_radius_px <= self.radius_at_reference_px <= self.max_radius_px
        ):
            raise ValueError(
                "spatial edge refinement radius must satisfy "
                "min_radius_px <= radius_at_reference_px <= max_radius_px"
            )
        return self


class SegmentationConfig(_StrictModel):
    backend: SegmentationBackend = "auto"
    model_path: str = ""
    delegate: SegmentationDelegate = "cpu"
    rvm_downsample: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "RVM-only internal-resolution ratio; zero selects the runtime "
            "automatic ratio."
        ),
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Heuristic-backend score threshold; inapplicable to RVM native "
            "alpha and MediaPipe confidence masks."
        ),
    )
    mask_blur: int = Field(
        default=7,
        ge=0,
        le=151,
        description=(
            "Generic MediaPipe/heuristic mask blur; bypassed for RVM native alpha."
        ),
    )
    edge_refine: bool = Field(
        default=True,
        description=(
            "Generic MediaPipe/heuristic spatial-refinement gate; bypassed for "
            "RVM native alpha."
        ),
    )
    mask_shift: int = Field(
        default=0,
        ge=-20,
        le=20,
        description=(
            "Explicit grow/shrink halo control for RVM, MediaPipe, and heuristic "
            "mattes; inapplicable to null/passthrough."
        ),
    )
    temporal_smoothing: float = Field(
        default=0.35,
        ge=0.0,
        le=0.95,
        description=(
            "Generic compatibility EMA for MediaPipe/heuristic masks; bypassed "
            "by RVM recurrence and whenever motion-aware stabilization is active."
        ),
    )
    boundary_stabilization: BoundaryStabilizationConfig = Field(
        default_factory=BoundaryStabilizationConfig,
        description=(
            "Elapsed-time boundary policy for MediaPipe/heuristic masks and "
            "explicit experimental RVM qualification; motion-aware mode "
            "replaces the generic EMA."
        ),
    )
    spatial_edge_refinement: SpatialEdgeRefinementConfig = Field(
        default_factory=SpatialEdgeRefinementConfig,
        description=(
            "Resolution-aware MediaPipe/heuristic edge policy selected by "
            "edge_refine; bypassed for RVM native alpha."
        ),
    )

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


def spatial_edge_refinement_radius(
    config: SpatialEdgeRefinementConfig,
    shape: tuple[int, int],
) -> int:
    """Resolve a search radius from a canonical ``(height, width)`` shape."""

    height, width = shape
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("spatial edge refinement shape must contain positive integers")
    if config.mode == "legacy_watershed":
        # Compatibility mode preserves the historical fixed eight-pixel band
        # (or its explicitly configured replacement) at every resolution.
        return config.radius_at_reference_px
    short_edge = min(height, width)
    scaled = config.radius_at_reference_px * short_edge / config.reference_short_edge_px
    # Round half up instead of relying on Python's ties-to-even ``round`` so
    # the result is stable and unsurprising at exact half-pixel scale factors.
    resolved = math.floor(scaled + 0.5)
    return max(config.min_radius_px, min(config.max_radius_px, resolved))


class AccelerationConfig(_StrictModel):
    """RVM inference acceleration policy for onnxruntime.

    This is deliberately separate from ``segmentation.delegate`` (which only
    selects MediaPipe's CPU/GPU delegate).  It governs how the RVM ONNX session
    chooses an execution provider and how it behaves when a GPU provider is
    registered but cannot actually execute the graph:

    - ``mode: auto`` — prefer a GPU provider, prove it can run RVM, and fall
      back to CPU (latched) if it cannot.  This is the safe default.
    - ``mode: cpu`` — construct only ``CPUExecutionProvider``; never touch a GPU.
    - ``mode: gpu_required`` — fail startup unless real GPU execution is proven.

    ``provider: auto`` tries the platform's GPU providers in priority order;
    ``cuda`` / ``directml`` pin a specific one.  ``device_id`` selects the
    adapter when more than one is present.  Provider availability is resolved at
    runtime, so an unavailable provider is a start-time (or fallback) outcome,
    not a configuration error.
    """

    mode: AccelerationMode = "auto"
    provider: AccelerationProvider = "auto"
    device_id: int = Field(default=0, ge=0, le=64)

    @model_validator(mode="after")
    def _cpu_mode_forbids_gpu_provider(self) -> "AccelerationConfig":
        if self.mode == "cpu" and self.provider != "auto":
            raise ValueError(
                "acceleration.provider must be auto when mode is cpu; "
                "cpu mode never selects a GPU provider"
            )
        return self


class ColorCorrectionConfig(_StrictModel):
    """Bounded policy consumed by the foreground harmonization stage.

    Compatibility defaults deliberately describe an identity transform.  The
    implemented automatic mode and any future default flip are separately
    qualified.
    """

    mode: ColorCorrectionMode = "off"
    strength: UnitStrength = 0.5
    exposure_limit_ev: ExposureLimitEv = 0.85
    white_balance_strength: UnitStrength = 0.5
    adaptation_time_s: AdaptationTimeSeconds = 0.8


class LightWrapStabilizationConfig(_StrictModel):
    """Experimental dynamic-backdrop wrap-sample policy.

    ``off`` is the schema-v1 compatibility path.  Numeric values are inert
    until the explicit candidate mode is selected.
    """

    mode: LightWrapStabilizationMode = "off"
    time_constant_s: LightWrapTimeConstantSeconds = 0.12


class CompositingConfig(_StrictModel):
    light_wrap: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Soft-edge backdrop light wrap for active matte compositing; "
            "inapplicable to null/passthrough."
        ),
    )
    use_model_foreground: bool = Field(
        default=True,
        description=(
            "RVM-only clean-foreground edge substitution; other backends do "
            "not produce model foreground."
        ),
    )
    blend_space: BlendSpace = "srgb_legacy"
    light_wrap_stabilization: LightWrapStabilizationConfig = Field(
        default_factory=LightWrapStabilizationConfig,
        description=(
            "Experimental video/camera light-wrap sample stabilization; off "
            "preserves schema-v1 stateless compositing exactly."
        ),
    )
    color_correction: ColorCorrectionConfig = Field(
        default_factory=ColorCorrectionConfig
    )


class OutputConfig(_StrictModel):
    width: FrameDimension | None = None
    height: FrameDimension | None = None
    backend: OutputBackend = "auto"
    device: str = ""
    fps: int = Field(default=30, ge=1, le=240)
    preview: bool = False

    @field_validator("device")
    @classmethod
    def _valid_output_device(cls, value: str) -> str:
        return _clean_config_string(value)

    @model_validator(mode="after")
    def _paired_canvas_dimensions(self) -> "OutputConfig":
        if (self.width is None) != (self.height is None):
            raise ValueError("output width and height must be configured together")
        return self


class UploadLimits(_StrictModel):
    image_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1, le=2**31)
    video_max_bytes: int = Field(default=256 * 1024 * 1024, ge=1, le=2**31)
    # Stay at or below Pillow's decompression-bomb warning threshold across
    # the supported Pillow 10-12 range so the configured limit is authoritative.
    image_max_pixels: int = Field(default=16_777_216, ge=256, le=SAFE_IMAGE_MAX_PIXELS)
    video_max_width: int = Field(default=3840, ge=16, le=7680)
    video_max_height: int = Field(default=2160, ge=16, le=7680)
    storage_max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1, le=2**40)
    max_files: int = Field(default=100, ge=1, le=100_000)

    @model_validator(mode="after")
    def _storage_can_hold_one_file(self) -> "UploadLimits":
        largest = max(self.image_max_bytes, self.video_max_bytes)
        if self.storage_max_bytes < largest:
            raise ValueError(
                "storage_max_bytes must be at least the largest per-file limit"
            )
        return self


class ApiConfig(_StrictModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8710, ge=1, le=65535)
    remote_timeout_ms: int = Field(default=250, ge=1, le=60_000)
    allow_non_loopback: bool = False
    allowed_origins: tuple[str, ...] = ()
    token_file: str = "~/.config/custback/api-token"
    renderer_token_file: str = "~/.config/custback/renderer-token"
    session_ttl_s: int = Field(default=28_800, ge=60, le=31_536_000)
    tls_certfile: str = ""
    tls_keyfile: str = ""
    ws_max_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=2**31)
    max_stream_connections: int = Field(default=16, ge=1, le=10_000)
    uploads: UploadLimits = Field(default_factory=UploadLimits)

    @field_validator("host")
    @classmethod
    def _valid_host(cls, value: str) -> str:
        from .api.security import normalize_bind_host

        normalized = normalize_bind_host(value)
        if normalized is None:
            raise ValueError("host must be a DNS name or IP literal without a port")
        return normalized

    @field_validator("token_file", "renderer_token_file")
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
            raise ValueError(
                "allowed_origins entries must be unique after normalization"
            )
        return tuple(normalized)

    @model_validator(mode="after")
    def _tls_pair(self) -> "ApiConfig":
        if bool(self.tls_certfile) != bool(self.tls_keyfile):
            raise ValueError("tls_certfile and tls_keyfile must be configured together")
        return self


class AvatarRemoteConfig(_StrictModel):
    """Where the custback-avatar control API lives, for the ``/avatar/*`` proxy.

    An empty ``url`` disables the proxy. ``token_file`` is the avatar
    service's own control token (``custback-avatar --show-api-token``); the
    ``CUSTBACK_AVATAR_API_TOKEN`` environment variable overrides the file.
    The destination and credential path are restart-only because the proxy
    resolves both into an immutable startup snapshot. The token value is read
    securely for each request so a supervised avatar may mint the file after
    the core API starts. Timeout changes remain hot-configurable.
    """

    url: str = ""
    token_file: str = Field(
        default_factory=lambda: str(platform_fs.config_dir() / "avatar-api-token")
    )
    tls_ca_file: str = ""
    tls_certfile: str = ""
    tls_keyfile: str = ""
    connect_timeout_s: float = Field(default=3.0, ge=0.5, le=60.0)
    read_timeout_s: float = Field(default=30.0, ge=1.0, le=600.0)

    @field_validator("url")
    @classmethod
    def _valid_url(cls, value: str) -> str:
        from .api.security import validate_outbound_endpoint

        value = _clean_config_string(value)
        endpoint = validate_outbound_endpoint(
            value,
            kind="http",
            label="avatar.url",
            allow_empty=True,
        )
        return "" if endpoint is None else endpoint.url

    @field_validator("token_file")
    @classmethod
    def _valid_token_file(cls, value: str) -> str:
        return _clean_config_string(value, allow_empty=False)

    @field_validator("tls_ca_file", "tls_certfile", "tls_keyfile")
    @classmethod
    def _valid_optional_tls_path(cls, value: str) -> str:
        return _clean_config_string(value)

    @model_validator(mode="after")
    def _valid_client_tls(self) -> "AvatarRemoteConfig":
        from .api.security import validate_outbound_endpoint

        endpoint = validate_outbound_endpoint(
            self.url,
            kind="http",
            label="avatar.url",
            allow_empty=True,
        )
        if bool(self.tls_certfile) != bool(self.tls_keyfile):
            raise ValueError(
                "avatar TLS certificate and private key must be configured together"
            )
        tls_configured = bool(self.tls_ca_file or self.tls_certfile or self.tls_keyfile)
        if tls_configured and endpoint is None:
            raise ValueError("avatar TLS files require a configured secure endpoint")
        if tls_configured and endpoint is not None and not endpoint.secure:
            raise ValueError(
                "avatar TLS files cannot be used with a plaintext endpoint"
            )
        return self


class AppConfig(_StrictModel):
    schema_version: SchemaVersion = CURRENT_CONFIG_SCHEMA_VERSION
    camera: CameraConfig = Field(default_factory=CameraConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    backdrop_targets: dict[str, BackdropTargetConfig] = Field(default_factory=dict)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    compositing: CompositingConfig = Field(default_factory=CompositingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    avatar: AvatarRemoteConfig = Field(default_factory=AvatarRemoteConfig)

    @field_validator("backdrop_targets")
    @classmethod
    def _valid_backdrop_target_ids(
        cls, value: dict[str, BackdropTargetConfig]
    ) -> dict[str, BackdropTargetConfig]:
        for identifier in value:
            if not _BACKDROP_TARGET_ID_RE.fullmatch(identifier):
                raise ValueError("backdrop target IDs must match [a-z][a-z0-9_-]{0,63}")
        return value

    @model_validator(mode="after")
    def _selected_backdrop_target_exists(self) -> "AppConfig":
        selected = self.background.camera_target
        if selected and selected not in self.backdrop_targets:
            raise ValueError(f"unknown camera backdrop target ID: {selected!r}")
        return self

    @model_validator(mode="after")
    def _gpu_required_requires_rvm_eligibility(self) -> "AppConfig":
        if self.acceleration.mode != "gpu_required":
            return self
        backend = self.segmentation.backend
        format_constrained_mediapipe = (
            backend == "auto"
            and bool(self.segmentation.model_path)
            and Path(self.segmentation.model_path).suffix.lower() == ".tflite"
        )
        if backend not in {"auto", "rvm"} or format_constrained_mediapipe:
            raise ValueError(
                "acceleration.mode=gpu_required requires an RVM-eligible "
                "segmentation backend"
            )
        return self

    def resolved_backdrop_target(self) -> ResolvedBackdropTarget | None:
        """Return a detached, validated snapshot of the selected source."""

        def normalized(source: int | str) -> int | str:
            if isinstance(source, str) and source.isdecimal():
                return int(source)
            return source

        selected = self.background.camera_target
        if selected:
            configured = self.backdrop_targets[selected]
            return ResolvedBackdropTarget(selected, normalized(configured.source))
        source = self.background.camera_device
        if source != "":
            # Safe legacy startup-only configuration remains readable long
            # enough for the explicit ``custback migrate`` workflow. It is
            # never accepted as hot API authority.
            return ResolvedBackdropTarget("", normalized(source))
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return plain Python values, preserving tuple compatibility."""
        return self.model_dump(mode="python")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise TypeError("configuration root must be a mapping")
        return cls.model_validate(materialize_config_schema_defaults(data))

    @classmethod
    def load(cls, path: str | Path | None) -> "AppConfig":
        if path is None:
            return cls()
        try:
            raw = yaml.safe_load(Path(path).read_text())
        except (yaml.YAMLError, RecursionError) as exc:
            mark = getattr(exc, "problem_mark", None)
            location = (
                f" at line {mark.line + 1}, column {mark.column + 1}"
                if mark is not None
                else ""
            )
            problem = str(getattr(exc, "problem", "malformed YAML")).splitlines()[0]
            raise ValueError(
                f"invalid YAML configuration{location}: {problem}"
            ) from None
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
        """Validate an RFC 7396 merge patch without mutating this config."""
        if not isinstance(patch, dict):
            raise TypeError("config patch must be a mapping")
        return type(self).from_dict(merge_patch(self.to_dict(), patch))


def resolved_output_size(config: AppConfig) -> tuple[int, int]:
    """Return the one canonical canvas-size decision for a configuration.

    Capture, processing, remote protocols, and every output sink consume this
    helper.  Until paired output dimensions are configured, the historical
    camera request remains the canvas size.
    """

    width = config.output.width
    height = config.output.height
    if width is None or height is None:
        # OutputConfig validation guarantees that neither half can be configured
        # alone; retain a defensive paired check at this public resolver boundary.
        if width is not None or height is not None:  # pragma: no cover - invariant
            raise ValueError("output width and height must be configured together")
        return config.camera.width, config.camera.height
    return width, height


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

    def commit(self, candidate: AppConfig, expected_version: int) -> ConfigState:
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
        # Build the caller-visible snapshot before activation. Once ``activate``
        # returns, only non-failing pointer/integer assignments remain, so an
        # installed resource generation can never be mistaken for a failed
        # candidate and closed by the caller's rollback path.
        committed = ConfigState(
            validated.model_copy(deep=True),
            expected_version + 1,
        )
        with self._lock:
            if self._version != expected_version:
                raise ConfigVersionConflictError(expected_version, self._version)
            next_version = self._version + 1
            activate(next_version)
            self._config = validated
            self._version = next_version
            return committed

    def update(self, patch: dict[str, Any]) -> AppConfig:
        """Reject legacy config-first mutation that bypasses resource staging."""
        raise RuntimeError(
            "direct RuntimeConfig.update() is disabled; use the pipeline "
            "reconfiguration coordinator"
        )
