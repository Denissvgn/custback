"""Authenticated HTTP and WebSocket control plane for custback."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import errno
import io
import json
import logging
import math
import os
import queue
import re
import stat
import threading
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

try:
    from python_multipart.exceptions import FormParserError, MultipartParseError
    from python_multipart.multipart import MultipartParser, parse_options_header
except ImportError:  # Legacy namespace for downstream compatibility.
    from multipart.exceptions import FormParserError, MultipartParseError
    from multipart.multipart import MultipartParser, parse_options_header

from .. import __version__
from .. import _platform as platform_fs
from ..backgrounds import DEFAULT_BACKGROUNDS_DIR, IMAGE_EXTS, VIDEO_EXTS
from ..color import ColorError, decode_image_to_srgb_bgr
from ..config import (
    MODES,
    Anchor,
    BoundaryStabilizationMode,
    CameraConfig,
    CompositingConfig,
    FitMode,
    OutputConfig,
    RuntimeConfig,
    SchemaVersion,
    SegmentationConfig,
    UploadLimits,
    VideoColorMatrix,
    VideoColorPrimaries,
    VideoColorRange,
    VideoColorTransfer,
    resolved_output_size,
)
from ..hub import FrameHub
from ..remote_protocol import (
    RemoteFrameProtocolError,
    decode_remote_frame,
    encode_remote_frame,
)
from ..runtime_performance import (
    RUNTIME_PERFORMANCE_SAMPLE_LIMIT,
    RUNTIME_STAGE_NAMES,
    validate_runtime_performance_status,
)
from ..storage_tx import OwnedPath, OwnershipLedger, rename_noreplace
from .security import SESSION_COOKIE, SecurityPolicy
from .streaming import (
    ConnectionLimiter,
    JpegBroadcaster,
    LeasedStreamingResponse,
    stream_lifecycle_for,
)

try:
    import cv2
except ImportError:  # pragma: no cover - required by the package, defensive for docs
    cv2 = None

try:  # Pillow supplies a safe header/dimension check before OpenCV allocates.
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - OpenCV remains a compatibility fallback
    Image = None
    UnidentifiedImageError = OSError

log = logging.getLogger(__name__)

UPLOAD_DIR = DEFAULT_BACKGROUNDS_DIR
UPLOAD_CHUNK_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 2 * 1024 * 1024
SESSION_REQUEST_MAX_BYTES = 4096
CONFIG_REQUEST_MAX_BYTES = 64 * 1024
_ACTIVATION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="custback-api-activation"
)
_IMAGE_MEDIA_TYPES = {
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".png": {"image/png"},
    ".bmp": {"image/bmp", "image/x-ms-bmp"},
    ".webp": {"image/webp"},
}
_VIDEO_MEDIA_TYPES = {
    ".mp4": {"video/mp4"},
    ".webm": {"video/webm"},
    ".mov": {"video/quicktime"},
    ".mkv": {"video/x-matroska", "video/matroska"},
    ".gif": {"image/gif"},
    ".avi": {"video/x-msvideo", "video/avi"},
}
_IMAGE_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".webp": "WEBP",
}
_STAGED_UPLOAD_RE = re.compile(r"\.upload-[0-9a-f]{32}(?:\.part|\.[a-z0-9]+)\Z")
_FINAL_UPLOAD_RE = re.compile(r"[0-9a-f]{32}\.[a-z0-9]+\Z")
_VISUAL_CONFIG_PATCH_EXAMPLES = {
    "camera-cover-restart": {
        "summary": "Select proportional camera cover (restart required)",
        "description": (
            "Camera geometry is restart-only. A 409 response leaves the active "
            "configuration unchanged; persist the field and restart."
        ),
        "value": {"camera": {"fit_mode": "cover"}},
    },
    "backdrop-contain": {
        "summary": "Contain a backdrop and move its focal point",
        "description": "Backdrop geometry is hot and commits at a frame boundary.",
        "value": {
            "background": {
                "fit_mode": "contain",
                "anchor_x": 0.5,
                "anchor_y": 0.25,
            }
        },
    },
    "linear-compositing": {
        "summary": "Opt into linear-light compositing",
        "description": (
            "This hot opt-in does not change the schema-1 srgb_legacy default."
        ),
        "value": {"compositing": {"blend_space": "linear_srgb"}},
    },
    "automatic-color-correction": {
        "summary": "Opt into bounded automatic foreground correction",
        "description": (
            "Automatic correction is eligible only for image, video, and camera "
            "backdrops; low confidence remains an observable safe bypass."
        ),
        "value": {
            "compositing": {
                "color_correction": {
                    "mode": "auto",
                    "strength": 0.5,
                }
            }
        },
    },
    "visual-quality-hot": {
        "summary": "Atomically update backdrop geometry and color policy",
        "description": (
            "All fields in this example are hot. The candidate geometry and "
            "color state commit together or the old output remains active."
        ),
        "value": {
            "background": {
                "fit_mode": "cover",
                "anchor_x": 0.5,
                "anchor_y": 0.25,
            },
            "compositing": {
                "blend_space": "linear_srgb",
                "color_correction": {
                    "mode": "auto",
                    "strength": 0.5,
                },
            },
        },
    },
}

LOGIN_HTML = """<!doctype html>
<meta charset="utf-8"><title>custback login</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:2rem}
input,button{font:inherit;padding:.5rem;margin:.25rem}</style>
<h1>custback</h1><p>Enter the local API token to open the preview.</p>
<form id="login"><input id="token" type="password" autocomplete="current-password"
placeholder="API token" size="48" required><button>Sign in</button></form>
<p id="error" role="alert"></p>
<script>login.onsubmit=async(e)=>{e.preventDefault();error.textContent='';
let r=await fetch('/auth/session',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({token:token.value})});if(r.ok)location.reload();
else error.textContent='Authentication failed';};</script>
"""

DOCS_HTML = """<!doctype html>
<meta charset="utf-8"><title>custback API</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:2rem}
a{color:#8cf}code{background:#222;padding:.15rem .3rem}</style>
<h1>custback API</h1>
<p>The authenticated OpenAPI document is available as
<a href="/openapi.json"><code>/openapi.json</code></a>.</p>
<p>Native clients authenticate with <code>Authorization: Bearer …</code>.
Browser clients use the strict session established at <code>/auth/session</code>.</p>
"""


@dataclass(frozen=True)
class _UploadLimits:
    image_max_bytes: int = 20 * 1024**2
    video_max_bytes: int = 256 * 1024**2
    image_max_pixels: int = 16_777_216
    video_max_width: int = 3840
    video_max_height: int = 2160
    storage_max_bytes: int = 2 * 1024**3
    max_files: int = 100


@dataclass(frozen=True)
class _SavedUpload:
    path: Path
    final_path: Path
    original_name: str
    size: int
    width: int
    height: int
    kind: str


@dataclass(frozen=True)
class _TerminalThreadResult:
    value: Any
    cancellation: asyncio.CancelledError | None = None


async def _to_thread_terminal(
    function: Any, /, *args: Any, **kwargs: Any
) -> _TerminalThreadResult:
    """Run blocking work to completion even if its awaiting task is cancelled.

    ``asyncio.to_thread`` cancellation only abandons the asyncio wrapper; its
    worker keeps running.  Upload workers mutate files and quota reservations,
    so callers need both the terminal result and any cancellation that arrived
    while it was running.  They can then publish matching ownership state
    before re-raising the deferred cancellation.
    """

    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancellation: asyncio.CancelledError | None = None
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as exc:
            # The shield keeps outer cancellation away from ``worker``.  A
            # cancelled worker therefore means the callable itself raised
            # CancelledError, which must propagate rather than be retried.
            if worker.done() and worker.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except BaseException:
            break
    try:
        value = worker.result()
    except BaseException as exc:
        if cancellation is not None:
            raise cancellation from exc
        raise
    return _TerminalThreadResult(value, cancellation)


def _state(runtime: RuntimeConfig):
    """Read config+version atomically, with compatibility for older runtimes."""

    read = getattr(runtime, "read", None)
    if read is not None:
        return read()

    class LegacyState:
        config = runtime.snapshot()
        version = runtime.version

    return LegacyState()


def _state_body(state) -> dict[str, Any]:
    # Serialize through an explicit allow-list model so a future private field
    # cannot leak merely because a manual deny-list was not updated.
    private = state.config
    payload = private.to_dict()
    background = payload["background"]
    background["camera_targets"] = tuple(sorted(private.backdrop_targets))
    background["camera_source_configured"] = bool(
        private.background.camera_target or private.background.camera_device != ""
    )
    return PublicAppConfig.model_validate(payload).model_dump(mode="python")


def _upload_limits(runtime: RuntimeConfig) -> _UploadLimits:
    configured = getattr(_state(runtime).config.api, "uploads", None)
    defaults = _UploadLimits()
    if configured is None:
        return defaults
    return _UploadLimits(
        **{
            field: int(getattr(configured, field, getattr(defaults, field)))
            for field in defaults.__dataclass_fields__
        }
    )


def _encode_jpeg(frame: np.ndarray) -> bytes:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for JPEG encoding")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if Image is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG":
                    return None
                return image.size
    except (
        OSError,
        ValueError,
        Warning,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return None


class PublicApiConfig(BaseModel):
    """Browser-safe API settings with credential/private-key paths omitted."""

    enabled: bool
    host: str
    port: int
    remote_timeout_ms: int
    allow_non_loopback: bool
    allowed_origins: tuple[str, ...]
    session_ttl_s: int
    tls_certfile: str
    ws_max_bytes: int
    max_stream_connections: int
    uploads: UploadLimits


class PublicAvatarRemoteConfig(BaseModel):
    """Public proxy state; trust and credential locations stay operator-only."""

    url: str
    connect_timeout_s: float
    read_timeout_s: float


class PublicBackgroundConfig(BaseModel):
    """Browser-safe backdrop state with operator sources removed."""

    mode: str
    image_path: str
    video_path: str
    camera_target: str
    camera_targets: tuple[str, ...]
    camera_source_configured: bool
    color: tuple[int, int, int]
    blur_strength: int
    fit_mode: FitMode
    anchor_x: Anchor
    anchor_y: Anchor
    video_color_matrix: VideoColorMatrix
    video_color_range: VideoColorRange
    video_color_primaries: VideoColorPrimaries
    video_color_transfer: VideoColorTransfer
    remote_fallback_mode: str


class PublicAppConfig(BaseModel):
    schema_version: SchemaVersion
    camera: CameraConfig
    background: PublicBackgroundConfig
    segmentation: SegmentationConfig
    compositing: CompositingConfig
    output: OutputConfig
    api: PublicApiConfig
    avatar: PublicAvatarRemoteConfig


class _ConfigPatchResponse(BaseModel):
    config: PublicAppConfig
    config_version: int


class _UploadResponse(BaseModel):
    id: str
    original_name: str
    bytes: int
    width: int
    height: int
    kind: str
    config_version: int


class _TimingFieldsResponse(BaseModel):
    """Version-1 fixed duration registry; JSON keys are stable dotted paths."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    capture_read: float | None = Field(alias="capture.read")
    segmentation_total: float | None = Field(alias="segmentation.total")
    segmentation_preprocess: float | None = Field(alias="segmentation.preprocess")
    segmentation_inference: float | None = Field(alias="segmentation.inference")
    segmentation_postprocess: float | None = Field(alias="segmentation.postprocess")
    background_total: float | None = Field(alias="background.total")
    color_correction_total: float | None = Field(alias="color_correction.total")
    compositor_total: float | None = Field(alias="compositor.total")
    compositor_prepare: float | None = Field(alias="compositor.prepare")
    compositor_blend: float | None = Field(alias="compositor.blend")
    output_send_total: float | None = Field(alias="output.send_total")
    output_submission: float | None = Field(alias="output.submission")
    output_sink_pacing_wait: float | None = Field(alias="output.sink_pacing_wait")
    output_application_pacing_wait: float | None = Field(
        alias="output.application_pacing_wait"
    )
    output_schedule_lateness: float | None = Field(alias="output.schedule_lateness")
    pipeline_processing_only: float | None = Field(alias="pipeline.processing_only")
    pipeline_new_frame_service: float | None = Field(alias="pipeline.new_frame_service")
    pipeline_new_frame_serialized_loop: float | None = Field(
        alias="pipeline.new_frame_serialized_loop"
    )


class _PostBaseStageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: str = Field(pattern=r"^[a-z][a-z0-9-]{0,31}$")
    update_count: int = Field(ge=0)
    update_fps: float = Field(ge=0.0)
    base_reuse_update_count: int = Field(ge=0)
    base_reuse_update_fps: float = Field(ge=0.0)


class _PostBaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: Literal["custback.post-base-cadence"] = Field(alias="schema")
    version: Literal[1]
    stages: list[_PostBaseStageResponse] = Field(max_length=8)


class _StatusExtensionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    post_base: _PostBaseResponse


_BackendName = Literal["rvm", "mediapipe", "heuristic", "none"]
_QualityTier = Literal["matting", "segmentation", "heuristic", "none"]
_SelectionReasonCategory = Literal[
    "none",
    "runtime-not-installed",
    "runtime-unavailable",
    "model-unavailable",
    "permission-denied",
    "preparation-unavailable",
    "activation-failed",
    "all-ml-backends-unavailable",
]
_PolicyText = Literal[
    "off",
    "motion_aware",
    "legacy_watershed",
    "stable_guided",
    "temporal_bounded",
    "native_soft_alpha",
    "confidence_soft_mask",
    "thresholded_binary_mask",
    "opaque_passthrough",
    "model_alpha_no_calibration",
    "confidence_mask_no_calibration",
    "heuristic_threshold",
    "none",
    "mask_shift_only",
    "generic_postprocess",
    "model_only",
    "explicit_motion_aware",
    "generic_temporal_policy",
    "model-alpha-no-calibration;generic-postprocess",
    "model-alpha-no-calibration;mask-shift-only",
    "confidence_mask_no_calibration;generic-postprocess",
    "heuristic_threshold;generic-postprocess",
]
_PolicyValue = _PolicyText | bool | int | float | None
_PolicyReason = Literal[
    "awaiting-first-rvm-inference",
    "backdrop-has-no-dynamic-timeline",
    "configured-active",
    "configured-off",
    "configured-ratio-resolved",
    "generic-mask-policy",
    "heuristic-score-cutoff",
    "heuristic-threshold-produces-binary-mask",
    "light-wrap-strength-is-zero",
    "mediapipe-confidence-mask-does-not-use-threshold",
    "null-or-passthrough-has-no-mask-threshold",
    "null-or-passthrough-has-no-matte",
    "null-or-passthrough-has-no-matte-refiner",
    "null-or-passthrough-has-no-model-foreground",
    "null-or-passthrough-has-no-rvm-inference",
    "null-or-passthrough-has-no-soft-edge-composite",
    "null-or-passthrough-has-no-temporal-matte",
    "opaque-core-calibration-requires-separate-evidence",
    "preserve-mediapipe-confidence-mask",
    "preserve-rvm-pha-without-threshold",
    "replaced-by-motion-aware",
    "runtime-auto-ratio",
    "rvm-native-alpha-bypasses-generic-blur",
    "rvm-native-alpha-bypasses-generic-edge-refinement",
    "rvm-native-alpha-is-never-hard-thresholded",
    "rvm-recurrence-bypasses-generic-ema",
    "selected-backend-does-not-produce-clean-foreground",
    "selected-backend-does-not-use-rvm-ratio",
]


class _SegmentationSelectionAttemptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: _BackendName
    quality_tier: _QualityTier
    preparation_result: Literal[
        "ready",
        "unavailable",
        "not-run",
        "not-applicable",
    ]
    activation_result: Literal["selected", "failed", "not-attempted"]
    reason_category: _SelectionReasonCategory
    reason: str = Field(max_length=240)
    guidance: str = Field(max_length=240)


class _SegmentationSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["custback.backend-selection"] = Field(alias="schema")
    version: Literal[1]
    requested_backend: Literal["auto", "rvm", "mediapipe", "heuristic", "none"]
    selected_backend: _BackendName
    quality_tier: _QualityTier
    selection_mode: Literal["automatic", "explicit", "model-format"]
    fallback_active: bool
    fallback_category: _SelectionReasonCategory
    fallback_reason: str = Field(max_length=240)
    guidance: str = Field(max_length=240)
    active_device: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    active_provider: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    attempts: list[_SegmentationSelectionAttemptResponse] = Field(
        min_length=1,
        max_length=4,
    )


class _MattePolicyConfiguredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rvm_downsample_ratio: float
    threshold: float
    mask_blur: int
    edge_refine: bool
    edge_refinement_mode: Literal["off", "legacy_watershed", "stable_guided"]
    edge_refinement_reference_short_edge_px: int
    edge_refinement_radius_at_reference_px: int
    edge_refinement_min_radius_px: int
    edge_refinement_max_radius_px: int
    mask_shift: int
    temporal_smoothing: float
    boundary_stabilization_mode: BoundaryStabilizationMode
    boundary_stabilization_time_constant_s: float
    boundary_stabilization_max_motion_px_per_s: float
    use_model_foreground: bool
    light_wrap: float
    light_wrap_stabilization_mode: Literal["off", "temporal_bounded"]
    light_wrap_stabilization_time_constant_s: float


class _MattePolicyEffectiveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_alpha_mode: Literal[
        "native_soft_alpha",
        "confidence_soft_mask",
        "thresholded_binary_mask",
        "opaque_passthrough",
    ]
    opaque_core_mode: Literal[
        "model_alpha_no_calibration",
        "confidence_mask_no_calibration",
        "heuristic_threshold",
        "none",
    ]
    halo_mode: Literal["mask_shift_only", "generic_postprocess", "none"]
    residual_temporal_mode: Literal[
        "model_only",
        "explicit_motion_aware",
        "generic_temporal_policy",
        "none",
    ]
    rvm_downsample_ratio: float | None
    threshold: float | None
    mask_blur: int
    edge_refine: bool
    edge_refinement_mode: Literal["off", "legacy_watershed", "stable_guided"]
    edge_refinement_radius_px: int
    mask_shift: int
    temporal_smoothing: float
    boundary_stabilization_mode: BoundaryStabilizationMode
    boundary_stabilization_time_constant_s: float
    boundary_stabilization_max_motion_px_per_s: float
    use_model_foreground: bool
    light_wrap: float
    light_wrap_stabilization_mode: Literal["off", "temporal_bounded"]
    light_wrap_stabilization_time_constant_s: float


class _MattePolicyControlResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configured: _PolicyValue
    effective: _PolicyValue
    state: Literal["effective", "bypassed", "inapplicable"]
    reason: _PolicyReason


class _MattePolicyControlsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rvm_downsample_ratio: _MattePolicyControlResponse
    raw_alpha: _MattePolicyControlResponse
    threshold: _MattePolicyControlResponse
    mask_blur: _MattePolicyControlResponse
    edge_refine: _MattePolicyControlResponse
    mask_shift: _MattePolicyControlResponse
    temporal_smoothing: _MattePolicyControlResponse
    boundary_stabilization: _MattePolicyControlResponse
    use_model_foreground: _MattePolicyControlResponse
    light_wrap: _MattePolicyControlResponse
    light_wrap_stabilization: _MattePolicyControlResponse
    opaque_core_halo: _MattePolicyControlResponse


class _MattePolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["custback.matte-policy"] = Field(alias="schema")
    version: Literal[1]
    blend_space: Literal["srgb_legacy", "linear_srgb"]
    selected_backend_kind: Literal[
        "true_alpha_recurrent",
        "confidence_mask_video",
        "binary_coarse",
        "null_passthrough",
    ]
    backend_kind: Literal[
        "true_alpha_recurrent",
        "confidence_mask_video",
        "binary_coarse",
        "null_passthrough",
    ]
    passthrough: bool
    experimental_rvm_generic: bool
    configured: _MattePolicyConfiguredResponse
    effective: _MattePolicyEffectiveResponse
    controls: _MattePolicyControlsResponse


class _MatteRolloutResponse(BaseModel):
    """Path-free process-local MATTE-5.4 canary and rollback telemetry."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["custback.matte-rollout-status"] = Field(alias="schema")
    version: Literal[1]
    stage: Literal["compatibility_hold"]
    decision: Literal["held_pending_physical_qualification"]
    configured_schema_version: int = Field(ge=1)
    config_version: int = Field(ge=0)
    qualified_default_active: Literal[False]
    preset_catalog_version: Literal[1]
    preset_evidence_status: Literal["not_qualified"]
    legacy_policy_available: Literal[True]
    legacy_policy_active: bool
    rollback_patch_id: Literal["matte-legacy-v1"]
    patch_attempts: int = Field(ge=0)
    patch_in_flight: int = Field(ge=0)
    patch_successes: int = Field(ge=0)
    patch_failures: int = Field(ge=0)
    legacy_rollbacks: int = Field(ge=0)
    last_outcome: Literal["none", "attempt", "success", "failure", "rollback"]

    @model_validator(mode="after")
    def _counters_are_consistent(self) -> "_MatteRolloutResponse":
        if self.patch_attempts != (
            self.patch_in_flight + self.patch_successes + self.patch_failures
        ):
            raise ValueError("matte rollout attempt counters are inconsistent")
        if self.legacy_rollbacks > self.patch_successes:
            raise ValueError("matte rollout rollback count exceeds successes")
        return self


_RuntimePerformanceState = Literal["warming", "healthy", "degraded", "failed"]
_RuntimeCadenceStatus = Literal[
    "warming",
    "matched",
    "intentional-repeat",
    "unexpected-shortfall",
    "failed",
]
_RuntimePerformanceReason = Literal[
    "none",
    "output-attainment",
    "unique-attainment",
    "output-and-unique-attainment",
    "processing-deadline-miss",
    "multiple-performance-gates",
    "publisher-failed",
    "pipeline-failed",
]
_RuntimeStageName = Literal[
    "capture.read",
    "segmentation.total",
    "background.total",
    "color_correction.estimate",
    "color_correction.apply",
    "compositor.input_mask_validation",
    "compositor.color_transform_application",
    "compositor.edge_band",
    "compositor.model_foreground_replacement",
    "compositor.backdrop_blur_resize",
    "compositor.light_wrap_temporal_filter",
    "compositor.light_wrap_interpolation",
    "compositor.final_blend_conversion",
    "compositor.internal_output_validation",
    "compositor.light_wrap",
    "compositor.prepare",
    "compositor.blend",
    "compositor.total",
    "pipeline.processing_only",
    "output.submission",
]


class _RuntimeStageMapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    capture_read: float | None = Field(alias="capture.read", ge=0.0, le=3_600_000.0)
    segmentation_total: float | None = Field(
        alias="segmentation.total", ge=0.0, le=3_600_000.0
    )
    background_total: float | None = Field(
        alias="background.total", ge=0.0, le=3_600_000.0
    )
    color_correction_estimate: float | None = Field(
        alias="color_correction.estimate", ge=0.0, le=3_600_000.0
    )
    color_correction_apply: float | None = Field(
        alias="color_correction.apply", ge=0.0, le=3_600_000.0
    )
    compositor_input_mask_validation: float | None = Field(
        alias="compositor.input_mask_validation", ge=0.0, le=3_600_000.0
    )
    compositor_color_transform_application: float | None = Field(
        alias="compositor.color_transform_application", ge=0.0, le=3_600_000.0
    )
    compositor_edge_band: float | None = Field(
        alias="compositor.edge_band", ge=0.0, le=3_600_000.0
    )
    compositor_model_foreground_replacement: float | None = Field(
        alias="compositor.model_foreground_replacement", ge=0.0, le=3_600_000.0
    )
    compositor_backdrop_blur_resize: float | None = Field(
        alias="compositor.backdrop_blur_resize", ge=0.0, le=3_600_000.0
    )
    compositor_light_wrap_temporal_filter: float | None = Field(
        alias="compositor.light_wrap_temporal_filter", ge=0.0, le=3_600_000.0
    )
    compositor_light_wrap_interpolation: float | None = Field(
        alias="compositor.light_wrap_interpolation", ge=0.0, le=3_600_000.0
    )
    compositor_final_blend_conversion: float | None = Field(
        alias="compositor.final_blend_conversion", ge=0.0, le=3_600_000.0
    )
    compositor_internal_output_validation: float | None = Field(
        alias="compositor.internal_output_validation", ge=0.0, le=3_600_000.0
    )
    compositor_light_wrap: float | None = Field(
        alias="compositor.light_wrap", ge=0.0, le=3_600_000.0
    )
    compositor_prepare: float | None = Field(
        alias="compositor.prepare", ge=0.0, le=3_600_000.0
    )
    compositor_blend: float | None = Field(
        alias="compositor.blend", ge=0.0, le=3_600_000.0
    )
    compositor_total: float | None = Field(
        alias="compositor.total", ge=0.0, le=3_600_000.0
    )
    pipeline_processing_only: float | None = Field(
        alias="pipeline.processing_only", ge=0.0, le=3_600_000.0
    )
    output_submission: float | None = Field(
        alias="output.submission", ge=0.0, le=3_600_000.0
    )


class _RuntimeMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_duration_s: float = Field(ge=0.0, le=5.0)
    window_sample_count: int = Field(ge=0, le=RUNTIME_PERFORMANCE_SAMPLE_LIMIT)
    output_send_fps: float = Field(ge=0.0, le=10_000.0)
    processing_completed_fps: float = Field(ge=0.0, le=10_000.0)
    sent_unique_base_fps: float = Field(ge=0.0, le=10_000.0)
    output_attainment: float = Field(ge=0.0, le=10_000.0)
    unique_attainment: float = Field(ge=0.0, le=10_000.0)
    processing_deadline_miss_ratio: float = Field(ge=0.0, le=1.0)
    output_schedule_late_ratio: float = Field(ge=0.0, le=1.0)
    stage_p50_ms: _RuntimeStageMapResponse
    stage_p95_ms: _RuntimeStageMapResponse
    dominant_stage: _RuntimeStageName | None


class _RuntimeEpochKeyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int = Field(ge=0, le=2**63 - 1)
    capture_generation: int = Field(ge=0, le=2**63 - 1)
    segmentation_generation: int = Field(ge=0, le=2**63 - 1)
    backdrop_generation: int = Field(ge=0, le=2**63 - 1)


class _RuntimeCountersResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processing_completed_count: int = Field(ge=0, le=2**63 - 1)
    output_send_count: int = Field(ge=0, le=2**63 - 1)
    sent_unique_base_count: int = Field(ge=0, le=2**63 - 1)
    processing_deadline_miss_count: int = Field(ge=0, le=2**63 - 1)
    output_schedule_late_count: int = Field(ge=0, le=2**63 - 1)


class _RuntimeEpochResponse(_RuntimeMetricsResponse, _RuntimeCountersResponse):
    model_config = ConfigDict(extra="forbid")

    key: _RuntimeEpochKeyResponse
    state: _RuntimePerformanceState
    reason: _RuntimePerformanceReason
    duration_s: float = Field(ge=0.0, le=1_000_000_000.0)


class _RuntimePublisherResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["uninitialized", "sink-paced", "deadline-paced"]
    state: Literal["starting", "running", "stopped", "failed"]
    handoff_overwrite_count: int = Field(ge=0, le=2**63 - 1)
    missed_slot_count: int = Field(ge=0, le=2**63 - 1)
    slate_send_count: int = Field(ge=0, le=2**63 - 1)
    pending_depth: int = Field(ge=0, le=1)
    output_base_config_version: int = Field(ge=0, le=2**63 - 1)


class _ColorOffPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["off"]


class _DisableColorCompositingPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color_correction: _ColorOffPatchResponse


class _DisableWrapCompositingPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    light_wrap: float = Field(ge=0.0, le=0.0)


class _DisableColorAndWrapCompositingPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color_correction: _ColorOffPatchResponse
    light_wrap: float = Field(ge=0.0, le=0.0)


class _DisableColorPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compositing: _DisableColorCompositingPatchResponse


class _DisableWrapPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compositing: _DisableWrapCompositingPatchResponse


class _DisableColorAndWrapPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compositing: _DisableColorAndWrapCompositingPatchResponse


class _EmptyMitigationPatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _DisableColorMitigationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int = Field(ge=0, le=2**63 - 1)
    kind: Literal["disable-color-correction"]
    patch: _DisableColorPatchResponse


class _DisableWrapMitigationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int = Field(ge=0, le=2**63 - 1)
    kind: Literal["disable-light-wrap"]
    patch: _DisableWrapPatchResponse


class _DisableColorAndWrapMitigationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int = Field(ge=0, le=2**63 - 1)
    kind: Literal["disable-color-and-light-wrap"]
    patch: _DisableColorAndWrapPatchResponse


class _ReviewBackendMitigationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int = Field(ge=0, le=2**63 - 1)
    kind: Literal["review-backend-or-diagnostic-target"]
    patch: _EmptyMitigationPatchResponse


_RuntimeMitigationResponse = (
    _DisableColorMitigationResponse
    | _DisableWrapMitigationResponse
    | _DisableColorAndWrapMitigationResponse
    | _ReviewBackendMitigationResponse
)


class _RuntimePerformanceResponse(_RuntimeMetricsResponse):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    state: _RuntimePerformanceState
    reason: _RuntimePerformanceReason
    cadence_status: _RuntimeCadenceStatus
    target_fps: float = Field(
        ge=0.0,
        le=1000.0,
        description="Compatibility alias for transport_target_fps.",
    )
    transport_target_fps: float = Field(
        ge=0.0,
        le=1000.0,
        description="Target rate for sink publication, including exact repeats.",
    )
    unique_target_fps: float = Field(
        ge=0.0,
        le=1000.0,
        description="Target rate for newly adopted guarded bases.",
    )
    transport_deadline_ms: float = Field(
        ge=0.0,
        le=3_600_000.0,
        description="Output schedule and serialized transport interval.",
    )
    processing_deadline_ms: float = Field(
        ge=0.0,
        le=3_600_000.0,
        description="Processing-only interval in the unique-base domain.",
    )
    output_healthy: bool
    unique_healthy: bool
    current_epoch: _RuntimeEpochResponse
    last_closed_epoch: _RuntimeEpochResponse | None
    startup: _RuntimeCountersResponse
    publisher: _RuntimePublisherResponse
    recommended_mitigation: _RuntimeMitigationResponse | None

    @model_validator(mode="after")
    def _runtime_contract_is_consistent(self) -> "_RuntimePerformanceResponse":
        public = self.model_dump(mode="python", by_alias=True)
        validate_runtime_performance_status(public)
        return self


if (
    tuple(
        _RuntimeStageMapResponse.model_fields[field].alias
        for field in _RuntimeStageMapResponse.model_fields
    )
    != RUNTIME_STAGE_NAMES
):  # pragma: no cover - schema invariant
    raise RuntimeError("runtime performance stage schema drifted")


class _StatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    frames_in: int
    frames_out: int
    fps: float
    mode: str
    segmentation_backend: str
    segmentation_device: str
    segmentation_selection: _SegmentationSelectionResponse
    matte_policy: _MattePolicyResponse
    matte_rollout: _MatteRolloutResponse
    segmentation_generation: int
    capture_sequence: int
    capture_sequence_gap_count: int
    capture_missing_input_count: int
    matte_reset_count: int
    matte_last_reset_reason: str
    segmentation_produces_matte: bool
    effective_rvm_downsample_ratio: float | None
    effective_mask_blur: int
    effective_edge_refine: bool
    effective_edge_refinement_mode: Literal[
        "off",
        "legacy_watershed",
        "stable_guided",
    ]
    effective_edge_refinement_radius_px: int
    effective_mask_shift: int
    effective_temporal_smoothing: float
    effective_boundary_stabilization_mode: BoundaryStabilizationMode
    effective_boundary_stabilization_time_constant_s: float
    effective_boundary_stabilization_max_motion_px_per_s: float
    effective_use_model_foreground: bool
    effective_light_wrap: float
    output_backend: str
    native_ring: str
    remote_connected: bool
    remote_frames_used: int
    remote_fallback_active: bool
    remote_fallback_mode: str
    remote_fallback_count: int
    remote_fallback_reason: str
    config_version: int
    capture_backend: str
    capture_fourcc: str | None
    capture_width: int | None
    capture_height: int | None
    capture_delivered_width: int | None
    capture_delivered_height: int | None
    capture_oriented_width: int | None
    capture_oriented_height: int | None
    capture_normalized_width: int | None
    capture_normalized_height: int | None
    capture_generation: int
    capture_geometry_transitions: int
    camera_fit: str
    camera_rotation: int
    camera_mirror: bool
    camera_scale_x: float | None
    camera_scale_y: float | None
    camera_crop_left: int | None
    camera_crop_top: int | None
    camera_crop_right: int | None
    camera_crop_bottom: int | None
    camera_pad_left: int
    camera_pad_top: int
    camera_pad_right: int
    camera_pad_bottom: int
    camera_controls: dict[str, object]
    capture_fps_reported: float | None
    capture_target_fps: int
    capture_fps: float
    capture_target_met: bool | None
    capture_frames_read: int
    capture_dropped_frames: int
    capture_read_failures: int
    capture_restarts: int
    capture_stalled: bool
    capture_frame_age_ms: float | None
    output_target_fps: int
    output_width: int | None
    output_height: int | None
    output_fps: int | None
    output_effective_fps: float
    fps_attainment_pct: float | None
    output_repeated_frames: int
    segmentation_update_count: int
    segmentation_update_fps: float
    base_composite_update_count: int
    base_composite_update_fps: float
    base_composite_reuse_count: int
    base_composite_reuse_fps: float
    base_composite_reuse_ratio: float
    exact_final_output_repeat_count: int
    exact_final_output_repeat_fps: float
    exact_final_output_repeat_ratio: float
    output_send_count: int
    output_send_fps: float
    processing_completed_count: int
    processing_completed_fps: float
    output_base_config_version: int
    output_handoff_overwrite_count: int
    output_schedule_skipped_slots: int
    output_privacy_slate_send_count: int
    output_publisher_state: Literal["starting", "running", "stopped", "failed"]
    last_unique_frame_age_ms: float | None
    capture_timestamp_delta_p50_ms: float | None
    capture_timestamp_delta_p95_ms: float | None
    output_send_delta_p50_ms: float | None
    output_send_delta_p95_ms: float | None
    output_send_jitter_p50_ms: float | None
    output_send_jitter_p95_ms: float | None
    base_composite_delta_p50_ms: float | None
    base_composite_delta_p95_ms: float | None
    cadence_mismatch_active: bool
    processing_deadline_misses: int
    serialized_new_frame_deadline_misses: int
    output_sink_pacing_events: int
    output_sink_recovery_events: int
    application_pacing_events: int
    output_schedule_late_events: int
    capture_read_ms: float | None
    segmentation_ms: float | None
    background_ms: float | None
    color_correction_ms: float | None
    background_fit: str
    background_rotation: int
    background_mirror: bool
    background_scale_x: float | None
    background_scale_y: float | None
    background_crop_left: int | None
    background_crop_top: int | None
    background_crop_right: int | None
    background_crop_bottom: int | None
    background_pad_left: int
    background_pad_top: int
    background_pad_right: int
    background_pad_bottom: int
    background_geometry_transitions: int
    background_fallback_active: bool
    background_fallback_reason: Literal["", "asset-unavailable"]
    color_correction_mode: str
    color_correction_active: bool
    color_correction_effective_mode: str
    color_correction_state: str
    color_correction_reason: str
    color_correction_confidence: float
    color_correction_exposure_ev: float
    color_correction_wb_gain_r: float
    color_correction_wb_gain_g: float
    color_correction_wb_gain_b: float
    color_correction_wb_active: bool
    color_correction_exposure_clamped: bool
    color_correction_exposure_clamp_count: int
    color_correction_exposure_clamp_time_s: float
    color_correction_wb_clamped: bool
    color_correction_wb_clamp_count: int
    color_correction_wb_clamp_time_s: float
    color_correction_warming: bool
    color_correction_stale: bool
    color_correction_applied_frames: int
    color_correction_bypassed_frames: int
    color_correction_scene_cuts: int
    color_correction_transitions: int
    color_correction_reason_transitions: int
    color_input_assumption: str
    composite_ms: float | None
    output_send_ms: float | None
    output_submission_ms: float | None
    output_sink_pacing_wait_ms: float | None
    application_pacing_wait_ms: float | None
    output_schedule_lateness_ms: float | None
    frame_processing_ms: float | None
    new_frame_service_ms: float | None
    new_frame_serialized_loop_ms: float | None
    timing_schema_version: Literal[1]
    timing_ms: _TimingFieldsResponse
    runtime_performance: _RuntimePerformanceResponse
    output_fallback_active: bool
    output_fallback_reason: str
    segmentation_fallback_active: bool
    segmentation_fallback_reason: str
    acceleration_mode: str
    acceleration_requested_provider: str
    acceleration_device_id: int
    acceleration_state: str
    acceleration_active_provider: str
    acceleration_fallback_active: bool
    acceleration_fallback_reason: str
    acceleration_fallback_count: int
    acceleration_last_transition_ms: float | None
    background_video_source_fps: float | None
    background_video_timing_mode: str | None
    background_video_frames_displayed: int
    background_video_frames_skipped: int
    background_video_frames_reused: int
    background_video_skip_ratio: float
    background_video_seek_count: int
    background_video_decode_failures: int
    background_video_lifetime_frames_displayed: int = Field(ge=0, le=2**63 - 1)
    background_video_lifetime_frames_skipped: int = Field(ge=0, le=2**63 - 1)
    background_video_lifetime_frames_reused: int = Field(ge=0, le=2**63 - 1)
    background_video_lifetime_seek_count: int = Field(ge=0, le=2**63 - 1)
    background_video_lifetime_decode_failures: int = Field(ge=0, le=2**63 - 1)
    background_video_orientation_status: str | None
    background_video_metadata_rotation: int | None
    background_video_auto_rotation_disabled: bool | None
    background_video_decoder_backend: str | None
    background_video_color_status: str | None
    background_video_input_color: str | None
    background_video_output_color: str | None
    background_video_color_assumed_fields: list[str]
    background_video_color_overridden_fields: list[str]
    extensions: _StatusExtensionsResponse
    uptime_s: float

    @model_validator(mode="after")
    def _selection_matches_matte_policy(self) -> "_StatusResponse":
        expected_kind = {
            "rvm": "true_alpha_recurrent",
            "mediapipe": "confidence_mask_video",
            "heuristic": "binary_coarse",
            "none": "null_passthrough",
        }[self.segmentation_selection.selected_backend]
        policy = self.matte_policy
        if policy.selected_backend_kind != expected_kind:
            raise ValueError("selected backend and matte policy kind do not match")
        effective_kind = "null_passthrough" if policy.passthrough else expected_kind
        if policy.backend_kind != effective_kind:
            raise ValueError("matte policy passthrough/effective kind is inconsistent")
        if policy.experimental_rvm_generic and (
            expected_kind != "true_alpha_recurrent" or policy.passthrough
        ):
            raise ValueError("experimental RVM policy requires active RVM matting")
        if self.matte_rollout.config_version != self.config_version:
            raise ValueError(
                "matte rollout and public config_version must describe one frame"
            )
        return self


class _BackgroundListResponse(BaseModel):
    files: list[str]
    modes: list[str]
    # Absolute store directory, so clients can build config path patches
    # for entries in ``files``.
    directory: str


def _decode_jpeg(data: bytes, expected_size: tuple[int, int]) -> np.ndarray | None:
    if not data.startswith(b"\xff\xd8") or cv2 is None or Image is None:
        return None
    dimensions = _jpeg_dimensions(data)
    # A Pillow parse failure is a hard rejection. Falling through to OpenCV
    # would allocate from an untrusted payload without the header check.
    if dimensions is None or dimensions != expected_size:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return None
    if (frame.shape[1], frame.shape[0]) != expected_size:
        return None
    return frame


def _error(status: int, code: str, message: str, **extra: Any) -> HTTPException:
    detail: dict[str, Any] = {"code": code, "message": message}
    detail.update(extra)
    return HTTPException(status_code=status, detail=detail)


def _map_apply_error(exc: BaseException) -> HTTPException:
    name = type(exc).__name__
    if name == "RestartRequiredError":
        return _error(
            409,
            "restart_required",
            str(exc),
            fields=list(getattr(exc, "fields", ())),
            current_version=getattr(exc, "current_version", None),
        )
    if name in ("ConfigConflictError", "ConfigVersionConflictError"):
        return _error(409, "config_conflict", str(exc))
    if name in ("ActivationError", "ConfigApplyError"):
        return _error(
            422,
            "activation_failed",
            "candidate configuration could not be activated",
        )
    if name == "ReconfigurationUnavailable" or isinstance(exc, TimeoutError):
        return _error(503, "reconfiguration_unavailable", str(exc))
    if isinstance(exc, OSError):
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            return _error(507, "insufficient_storage", "cannot promote background")
        if isinstance(exc, FileNotFoundError):
            return _error(422, "activation_failed", "staged background is unavailable")
        if isinstance(exc, FileExistsError):
            return _error(
                409, "config_conflict", "background destination already exists"
            )
        return _error(
            507, "insufficient_storage", "background storage operation failed"
        )
    if isinstance(exc, ValidationError):
        errors = []
        for error in exc.errors(
            include_url=False, include_context=False, include_input=False
        ):
            errors.append(
                {
                    "field": ".".join(str(part) for part in error.get("loc", ())),
                    "message": error.get("msg", "invalid value"),
                    "type": error.get("type", "value_error"),
                }
            )
        return _error(
            422,
            "invalid_config",
            "configuration validation failed",
            errors=errors,
        )
    if isinstance(exc, (ValueError, TypeError)):
        return _error(422, "invalid_config", str(exc))
    raise exc


def _apply_patch(
    runtime: RuntimeConfig,
    coordinator: Any,
    patch: dict[str, Any],
    *,
    expected_version: int | None = None,
):
    try:
        kwargs: dict[str, Any] = {"origin": "api"}
        if expected_version is not None:
            kwargs["expected_version"] = expected_version
        return coordinator.apply_config_patch(patch, timeout=5.0, **kwargs)
    except BaseException as exc:
        raise _map_apply_error(exc) from exc


def _conditional_config_version(request: Request) -> int | None:
    raw = request.headers.get("x-expected-config-version")
    if raw is None:
        return None
    if (
        len(raw) > 19
        or not raw.isascii()
        or not raw.isdecimal()
        or (len(raw) > 1 and raw.startswith("0"))
    ):
        raise _error(
            422,
            "invalid_content",
            "X-Expected-Config-Version must be a canonical non-negative integer",
        )
    value = int(raw)
    if value > 2**63 - 1:
        raise _error(
            422,
            "invalid_content",
            "X-Expected-Config-Version is outside the supported range",
        )
    return value


def _version(runtime: RuntimeConfig) -> str:
    return str(_state(runtime).version)


async def _limited_json(
    request: Request,
    limit: int,
    *,
    media_types: frozenset[str] = frozenset({"application/json"}),
) -> Any:
    media_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if media_type not in media_types:
        raise _error(
            415,
            "unsupported_media_type",
            f"one of {', '.join(sorted(media_types))} is required",
        )
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise _error(
                413,
                "request_too_large",
                "JSON request is too large",
                limit=limit,
            )
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise _error(422, "invalid_content", "valid JSON content is required") from exc


def _auth_failure(path: str) -> Response:
    headers = {"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"}
    if path == "/":
        return HTMLResponse(LOGIN_HTML, status_code=401, headers=headers)
    return JSONResponse(
        {"detail": {"code": "unauthorized", "message": "valid API token required"}},
        status_code=401,
        headers=headers,
    )


class _UploadStore:
    def __init__(self, directory: Path, limits: _UploadLimits):
        self.directory = directory
        self.limits = limits
        self._lock = threading.Lock()
        self._ledger = OwnershipLedger(self.directory)
        self._reserved_bytes = self._ledger.reserved_bytes
        self._reserved_files = self._ledger.reserved_slots
        self._active_temps: set[Path] = set()
        self._transactions: dict[Path, OwnedPath] = {}
        self._cleanup_pending: dict[Path, tuple[tuple[Path, ...], int, bool]] = {}
        self._cleanup_retry_lock = threading.Lock()

    @staticmethod
    def _remove_owned_path(path: Path, kind: str) -> None:
        if kind != "file":
            raise OSError(errno.EINVAL, "core upload ownership must be a file")
        path.unlink()

    def _sync_reservations_locked(self) -> None:
        self._reserved_bytes = self._ledger.reserved_bytes
        self._reserved_files = self._ledger.reserved_slots

    def _forget_transaction_locked(self, record: OwnedPath) -> None:
        for initial, owned in tuple(self._transactions.items()):
            if owned is record:
                self._transactions.pop(initial, None)
                self._cleanup_pending.pop(initial, None)

    def ensure_directory(self) -> None:
        """Create or repair the user-owned managed directory as mode 0700."""

        with self._lock:
            self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
            before = os.lstat(self.directory)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise OSError(errno.ENOTDIR, "upload storage is not a directory")
            if not platform_fs.stat_owner_matches(before):
                raise PermissionError(
                    errno.EPERM,
                    "upload storage must be owned by the current user",
                    self.directory,
                )
            # chmod by name first so a restrictive umask cannot leave a newly
            # created directory unopenable. Verify the inode did not change,
            # then bind the final mode to a no-follow directory descriptor.
            platform_fs.chmod_private(self.directory, 0o700)
            after = os.lstat(self.directory)
            if (
                not stat.S_ISDIR(after.st_mode)
                or stat.S_ISLNK(after.st_mode)
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise OSError(errno.EAGAIN, "upload storage changed while securing it")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            descriptor = platform_fs.open_nofollow(
                self.directory, flags, directory=True
            )
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISDIR(opened.st_mode) or (
                    opened.st_dev,
                    opened.st_ino,
                ) != (after.st_dev, after.st_ino):
                    raise OSError(
                        errno.EAGAIN,
                        "upload storage changed while securing it",
                    )
                # Authoritative owner check on the bound descriptor (SID on Windows).
                if not platform_fs.owner_matches(descriptor):
                    raise PermissionError(
                        errno.EPERM,
                        "upload storage must be owned by the current user",
                        self.directory,
                    )
                platform_fs.set_private_mode(descriptor, 0o700)
            finally:
                os.close(descriptor)

    def _usage(self) -> tuple[int, int]:
        total = count = 0
        if not self.directory.is_dir():
            return total, count
        for path in self.directory.iterdir():
            if path.is_file() and not path.is_symlink():
                # In-progress files are represented by reservations. Hidden
                # files left by a dead process have no reservation and must
                # count toward both aggregate quotas until safely reclaimed.
                if path in self._active_temps:
                    continue
                if self._ledger.owns(path):
                    continue
                count += 1
                # A stat failure must not turn an on-disk inode into zero-byte
                # quota usage.  Fail the quota scan and therefore the upload.
                total += path.stat().st_size
        return total, count

    def _reserve_file(self, temporary: Path) -> None:
        with self._lock:
            _total, count = self._usage()
            if count + self._reserved_files >= self.limits.max_files:
                raise _error(
                    507, "upload_quota_exceeded", "background file quota reached"
                )
            if temporary.exists() or temporary.is_symlink():
                raise FileExistsError(
                    f"upload staging path already exists: {temporary.name}"
                )
            record = self._ledger.begin(temporary, kind="file", reserved_slots=1)
            self._transactions[temporary] = record
            self._sync_reservations_locked()
            self._active_temps.add(temporary)

    def _reserve_bytes(self, amount: int, temporary: Path | None = None) -> None:
        with self._lock:
            total, _count = self._usage()
            if total + self._reserved_bytes + amount > self.limits.storage_max_bytes:
                raise _error(
                    507, "upload_quota_exceeded", "background storage quota reached"
                )
            record = self._ledger.find(
                temporary if temporary is not None else next(iter(self._active_temps))
            )
            if record is None:
                raise ValueError("upload byte reservation has no owner")
            self._ledger.set_charge(
                record, reserved_bytes=record.reserved_bytes + amount
            )
            self._sync_reservations_locked()

    def _bind_created(self, temporary: Path) -> None:
        with self._lock:
            record = self._ledger.find(temporary)
            if record is None:
                raise ValueError("upload staging has no ownership record")
            self._ledger.bind(record, temporary)

    def _abandon_unbound(self, temporary: Path) -> None:
        with self._lock:
            record = self._transactions.pop(temporary, None)
            if record is not None:
                self._ledger.abandon_unbound(record)
            self._active_temps.discard(temporary)
            self._sync_reservations_locked()

    def _release(
        self, temporary: Path, reserved_bytes: int, reserved_file: bool
    ) -> None:
        with self._lock:
            self._release_locked(temporary, reserved_bytes, reserved_file)

    def _release_locked(
        self, temporary: Path, reserved_bytes: int, reserved_file: bool
    ) -> None:
        self._active_temps.discard(temporary)
        self._sync_reservations_locked()

    def _commit(self, temporary: Path, staged: Path, reserved_bytes: int) -> None:
        """Move an owned upload to its hidden activation name."""
        with self._lock:
            self._rename_private_owned(temporary, staged)
            self._active_temps.discard(temporary)
            self._active_temps.add(staged)
            self._sync_reservations_locked()

    @staticmethod
    def _open_private_file(path: Path):
        descriptor: int | None = None
        try:
            descriptor = platform_fs.open_nofollow(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            # A restrictive umask may remove owner bits; the final contract is
            # an exact private mode on the inode bound to this descriptor.
            platform_fs.set_private_mode(descriptor, 0o600)
            destination = os.fdopen(descriptor, "wb")
            descriptor = None
            return destination
        finally:
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)

    @staticmethod
    def _secure_private_file(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = platform_fs.open_nofollow(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise OSError(errno.EINVAL, "upload is not a regular file", path)
            if not platform_fs.owner_matches(descriptor):
                raise PermissionError(
                    errno.EPERM,
                    "upload must be owned by the current user",
                    path,
                )
            platform_fs.set_private_mode(descriptor, 0o600)
        finally:
            os.close(descriptor)

    def _rename_private_owned(self, source: Path, destination: Path) -> None:
        """Rename/harden while retaining both possible names on failure."""

        record = self._ledger.find(source, destination)
        if record is None:
            raise ValueError("upload rename has no ownership record")
        self._ledger.prepare_rename(record, source, destination)
        rename_noreplace(source, destination)
        try:
            self._ledger.finish_rename(record, destination)
            self._secure_private_file(destination)
        except BaseException as primary:
            # Rollback is best effort, but unlike the historical helper its
            # failure cannot discard ownership: both candidates were persisted
            # before either rename and reconcile follows the inode.
            try:
                if destination in self._ledger.reconcile(record):
                    self._ledger.prepare_rename(record, destination, source)
                    rename_noreplace(destination, source)
                    self._ledger.finish_rename(record, source)
            except OSError:
                try:
                    self._ledger.reconcile(record)
                except OSError:
                    pass
            raise primary

    def _finish_transaction(self, *paths: Path) -> None:
        """Transfer a successfully activated inode to committed disk usage."""

        with self._lock:
            record = self._ledger.find(*paths)
            if record is None:
                return
            self._ledger.commit(record)
            for path in record.paths:
                self._active_temps.discard(path)
            self._forget_transaction_locked(record)
            self._sync_reservations_locked()

    @staticmethod
    def _sync_and_close(destination: Any) -> None:
        try:
            destination.flush()
            os.fsync(destination.fileno())
        finally:
            destination.close()

    def _cleanup_save(
        self,
        destination: Any,
        temporary: Path,
        staged: Path | None,
        reserved_bytes: int,
        reserved_file: bool,
    ) -> None:
        """Release all pre-handoff upload ownership in one blocking worker."""

        if destination is not None:
            with contextlib.suppress(OSError):
                destination.close()
        paths = tuple(path for path in (temporary, staged) if path is not None)
        with self._cleanup_retry_lock:
            with self._lock:
                record = self._transactions.get(temporary) or self._ledger.find(*paths)
                if record is not None:
                    try:
                        self._ledger.mark_cleanup(record)
                    except OSError:
                        pass
                    error: OSError | None = None
                    # Retain the historical bounded immediate retry, but keep
                    # the durable record and quota charge after all attempts.
                    for _attempt in range(3):
                        error = self._ledger.cleanup(record, self._remove_owned_path)
                        if error is None:
                            break
                    if error is not None:
                        actual = record.paths
                        self._active_temps.update(actual)
                        self._cleanup_pending[temporary] = (
                            actual,
                            record.reserved_bytes,
                            bool(record.reserved_slots),
                        )
                        self._sync_reservations_locked()
                        log.warning(
                            "cannot remove failed upload staging file(s): %s",
                            ", ".join(path.name for path in actual),
                        )
                        return
                    for path in paths:
                        self._active_temps.discard(path)
                    self._cleanup_pending.pop(temporary, None)
                    self._transactions.pop(temporary, None)
                    self._sync_reservations_locked()
                    return
            remaining = self._unlink_cleanup_paths(paths)
            if remaining:
                # Keep the reservation/active-file ownership until a later
                # upload retries deletion. This prevents a transient unlink
                # failure from turning into an unowned hidden staging inode.
                with self._lock:
                    self._cleanup_pending[temporary] = (
                        remaining,
                        reserved_bytes,
                        reserved_file,
                    )
                log.warning(
                    "cannot remove failed upload staging file(s): %s",
                    ", ".join(path.name for path in remaining),
                )
                return
            # This takes the store lock and can wait behind a quota scan, so it
            # belongs in the same worker rather than on the ASGI event loop.
            self._release(temporary, reserved_bytes, reserved_file)

    @staticmethod
    def _unlink_cleanup_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
        """Retry transient unlinks and return paths still owned by the store."""

        remaining: list[Path] = []
        for path in paths:
            for attempt in range(3):
                try:
                    path.unlink()
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    if attempt == 2:
                        remaining.append(path)
        return tuple(remaining)

    def _retry_pending_cleanup(self) -> None:
        """Retry failed staging deletion without dropping quota ownership."""

        with self._cleanup_retry_lock:
            with self._lock:
                self._ledger.retry_cleanup(self._remove_owned_path)
                pending_records = tuple(
                    record
                    for record in self._ledger.records
                    if record.state == "cleanup"
                )
                active_temps = {
                    path for record in pending_records for path in record.paths
                }
                active_temps.update(
                    path
                    for path in self._active_temps
                    if (owned := self._ledger.find(path)) is not None
                    and owned.state == "active"
                )
                self._active_temps = active_temps
                for temporary, record in tuple(self._cleanup_pending.items()):
                    owned = self._ledger.find(*record[0])
                    if owned is None:
                        self._cleanup_pending.pop(temporary, None)
                        self._transactions.pop(temporary, None)
                    else:
                        self._cleanup_pending[temporary] = (
                            owned.paths,
                            owned.reserved_bytes,
                            bool(owned.reserved_slots),
                        )
                self._sync_reservations_locked()
            with self._lock:
                pending = tuple(self._cleanup_pending.items())
            for temporary, record in pending:
                paths, reserved_bytes, reserved_file = record
                if self._ledger.find(*paths) is not None:
                    continue
                remaining = self._unlink_cleanup_paths(paths)
                with self._lock:
                    if self._cleanup_pending.get(temporary) != record:
                        continue
                    if remaining:
                        self._cleanup_pending[temporary] = (
                            remaining,
                            reserved_bytes,
                            reserved_file,
                        )
                    else:
                        self._cleanup_pending.pop(temporary, None)
                        self._release_locked(temporary, reserved_bytes, reserved_file)

    @staticmethod
    def _active_background_path(config: Any) -> str:
        background = config.background
        if background.mode == "image":
            return background.image_path
        if background.mode == "video":
            return background.video_path
        if background.mode == "remote":
            if background.remote_fallback_mode == "image":
                return background.image_path
            if background.remote_fallback_mode == "video":
                return background.video_path
        return ""

    def _is_active(self, path: Path, config: Any) -> bool:
        active = self._active_background_path(config)
        return bool(
            active
            and Path(active).expanduser().resolve(strict=False)
            == path.resolve(strict=False)
        )

    def discard_failed_upload(
        self,
        staged: Path,
        final: Path,
        *,
        final_owned: bool,
        config: Any,
    ) -> None:
        """Remove every inactive inode still owned by a failed transaction."""
        if (
            staged.parent != self.directory
            or final.parent != self.directory
            or _STAGED_UPLOAD_RE.fullmatch(staged.name) is None
            or _FINAL_UPLOAD_RE.fullmatch(final.name) is None
            or staged.suffix != final.suffix
        ):
            raise ValueError("invalid failed-upload cleanup paths")
        with self._lock:
            record = self._ledger.find(staged, final)
            if record is None:
                return
            current = self._ledger.reconcile(record)
            if any(self._is_active(path, config) for path in current):
                self._ledger.commit(record)
                for path in current:
                    self._active_temps.discard(path)
                self._forget_transaction_locked(record)
                self._sync_reservations_locked()
                return
            try:
                self._ledger.mark_cleanup(record)
            except OSError:
                pass
            error = self._ledger.cleanup(record, self._remove_owned_path)
            self._sync_reservations_locked()
            if error is not None:
                self._cleanup_pending.setdefault(
                    staged,
                    (
                        record.paths,
                        record.reserved_bytes,
                        bool(record.reserved_slots),
                    ),
                )
                self._active_temps.update(record.paths)
                raise error
            self._active_temps.discard(staged)
            self._active_temps.discard(final)
            self._forget_transaction_locked(record)
            self._cleanup_pending.pop(staged, None)

    def promote(self, staged: Path, final: Path) -> None:
        if (
            staged.parent != self.directory
            or final.parent != self.directory
            or _STAGED_UPLOAD_RE.fullmatch(staged.name) is None
            or _FINAL_UPLOAD_RE.fullmatch(final.name) is None
            or staged.suffix != final.suffix
        ):
            raise ValueError("invalid upload promotion paths")
        with self._lock:
            if staged.is_symlink() or not staged.is_file():
                raise FileNotFoundError(f"staged upload is unavailable: {staged.name}")
            if final.exists() or final.is_symlink():
                raise FileExistsError(
                    f"upload destination already exists: {final.name}"
                )
            self._rename_private_owned(staged, final)
            self._active_temps.discard(staged)
            self._active_temps.add(final)

    def rollback_promotion(self, staged: Path, final: Path) -> None:
        with self._lock:
            if final.is_symlink() or not final.is_file():
                raise FileNotFoundError(f"promoted upload is unavailable: {final.name}")
            if staged.exists() or staged.is_symlink():
                raise FileExistsError(
                    f"upload staging path already exists: {staged.name}"
                )
            self._rename_private_owned(final, staged)
            self._active_temps.discard(final)
            self._active_temps.add(staged)

    def cleanup_staged(self, config: Any) -> None:
        """Recover only explicitly marked crash-left owned inodes."""
        try:
            os.lstat(self.directory)
        except FileNotFoundError:
            return
        self.ensure_directory()
        with self._lock:
            for record in tuple(self._ledger.records):
                current = self._ledger.reconcile(record)
                if record.state == "committed" or any(
                    self._is_active(path, config) for path in current
                ):
                    self._ledger.commit(record)
                    continue
                try:
                    self._ledger.mark_cleanup(record)
                except OSError:
                    pass
                error = self._ledger.cleanup(record, self._remove_owned_path)
                if error is not None:
                    self._active_temps.update(record.paths)
                    log.warning(
                        "cannot remove marked staged upload %s",
                        ", ".join(path.name for path in record.paths),
                    )
            self._sync_reservations_locked()

    async def save(self, request: Request, kind: str) -> _SavedUpload:
        """Parse one multipart file directly into an owned mode-0600 file.

        FastAPI's ``UploadFile`` dependency is intentionally not used: it
        would spool the complete multipart body before endpoint limits run.
        """
        maximum = (
            self.limits.image_max_bytes
            if kind == "image"
            else self.limits.video_max_bytes
        )
        content_type = request.headers.get("content-type", "")
        try:
            media_type, options = parse_options_header(content_type.encode("latin-1"))
        except (UnicodeEncodeError, ValueError) as exc:
            raise _error(
                415, "unsupported_media_type", "invalid multipart Content-Type"
            ) from exc
        boundary = options.get(b"boundary")
        if media_type != b"multipart/form-data" or not boundary:
            raise _error(
                415, "unsupported_media_type", "multipart/form-data is required"
            )

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > maximum + MULTIPART_OVERHEAD_BYTES:
                    raise _error(
                        413,
                        "upload_too_large",
                        f"request exceeds {maximum} byte {kind} limit",
                        limit=maximum,
                    )
            except ValueError as exc:
                raise _error(
                    400, "invalid_content_length", "invalid Content-Length"
                ) from exc

        try:
            mkdir_result = await _to_thread_terminal(self.ensure_directory)
        except OSError as exc:
            raise _error(
                507, "insufficient_storage", "cannot create upload storage"
            ) from exc
        if mkdir_result.cancellation is not None:
            raise mkdir_result.cancellation
        retry_result = await _to_thread_terminal(self._retry_pending_cleanup)
        if retry_result.cancellation is not None:
            raise retry_result.cancellation
        upload_id = uuid.uuid4().hex
        temporary = self.directory / f".upload-{upload_id}.part"
        reserved_file = False
        reserved_bytes = 0
        staged: Path | None = None
        staged_owned: Path | None = None
        final: Path | None = None
        handed_off = False
        original = ""
        suffix = ""
        size = 0
        body_size = 0
        file_seen = file_complete = message_complete = False
        current_file = False
        headers: dict[bytes, bytes] = {}
        header_name: list[bytes] = []
        header_value: list[bytes] = []
        destination = None

        def on_part_begin() -> None:
            nonlocal headers, current_file
            headers = {}
            current_file = False

        def on_header_field(data: bytes, start: int, end: int) -> None:
            header_name.append(data[start:end])

        def on_header_value(data: bytes, start: int, end: int) -> None:
            header_value.append(data[start:end])

        def on_header_end() -> None:
            headers[b"".join(header_name).lower()] = b"".join(header_value).strip()
            header_name.clear()
            header_value.clear()

        def on_headers_finished() -> None:
            nonlocal current_file, file_seen, original, suffix, staged, final
            if file_seen:
                raise _error(
                    422, "invalid_multipart", "exactly one file part is required"
                )
            disposition, values = parse_options_header(
                headers.get(b"content-disposition")
            )
            filename = values.get(b"filename")
            if (
                disposition != b"form-data"
                or values.get(b"name") != b"file"
                or not filename
            ):
                raise _error(
                    422, "invalid_multipart", "a file field named 'file' is required"
                )
            if headers.get(b"content-transfer-encoding", b"binary").lower() not in {
                b"binary",
                b"8bit",
                b"7bit",
            }:
                raise _error(
                    415,
                    "unsupported_media_type",
                    "encoded multipart files are unsupported",
                )
            original = Path(filename.decode("latin-1").replace("\\", "/")).name
            if not original or "\x00" in original:
                raise _error(400, "missing_filename", "upload filename is required")
            suffix = Path(original).suffix.lower()
            allowed = IMAGE_EXTS if kind == "image" else VIDEO_EXTS
            if suffix not in allowed:
                raise _error(
                    415,
                    "unsupported_media_type",
                    f"unsupported {kind} extension {suffix!r}",
                    allowed=sorted(allowed),
                )
            declared = headers.get(b"content-type", b"").decode("latin-1").lower()
            expected = (_IMAGE_MEDIA_TYPES if kind == "image" else _VIDEO_MEDIA_TYPES)[
                suffix
            ]
            if declared not in expected:
                raise _error(
                    415,
                    "unsupported_media_type",
                    f"declared media type {declared!r} does not match {suffix}",
                    allowed=sorted(expected),
                )
            staged = self.directory / f".upload-{upload_id}{suffix}"
            final = self.directory / f"{upload_id}{suffix}"
            current_file = file_seen = True

        def on_part_data(data: bytes, start: int, end: int) -> None:
            nonlocal size, reserved_bytes
            if not current_file or destination is None:
                raise _error(422, "invalid_multipart", "unexpected multipart field")
            length = end - start
            if size + length > maximum:
                raise _error(
                    413,
                    "upload_too_large",
                    f"{kind} exceeds {maximum} byte limit",
                    limit=maximum,
                )
            self._reserve_bytes(length, temporary)
            reserved_bytes += length
            destination.write(data[start:end])
            size += length

        def on_part_end() -> None:
            nonlocal file_complete, current_file
            if current_file:
                file_complete = True
            current_file = False

        def on_end() -> None:
            nonlocal message_complete
            message_complete = True

        try:
            reserve_result = await _to_thread_terminal(self._reserve_file, temporary)
            reserved_file = True
            if reserve_result.cancellation is not None:
                raise reserve_result.cancellation

            try:
                open_result = await _to_thread_terminal(
                    self._open_private_file, temporary
                )
            except FileExistsError as exc:
                await _to_thread_terminal(self._abandon_unbound, temporary)
                handed_off = True  # the colliding inode was never ours
                raise _error(
                    507,
                    "insufficient_storage",
                    "upload staging name collided",
                ) from exc
            destination = open_result.value
            bind_result = await _to_thread_terminal(self._bind_created, temporary)
            if bind_result.cancellation is not None:
                raise bind_result.cancellation
            if open_result.cancellation is not None:
                raise open_result.cancellation
            parser = MultipartParser(
                boundary,
                callbacks={
                    "on_part_begin": on_part_begin,
                    "on_header_field": on_header_field,
                    "on_header_value": on_header_value,
                    "on_header_end": on_header_end,
                    "on_headers_finished": on_headers_finished,
                    "on_part_data": on_part_data,
                    "on_part_end": on_part_end,
                    "on_end": on_end,
                },
                max_size=maximum + MULTIPART_OVERHEAD_BYTES,
            )
            async for received in request.stream():
                body_size += len(received)
                if body_size > maximum + MULTIPART_OVERHEAD_BYTES:
                    raise _error(
                        413,
                        "upload_too_large",
                        "multipart request is too large",
                        limit=maximum,
                    )
                for offset in range(0, len(received), UPLOAD_CHUNK_BYTES):
                    write_result = await _to_thread_terminal(
                        parser.write,
                        received[offset : offset + UPLOAD_CHUNK_BYTES],
                    )
                    if write_result.cancellation is not None:
                        raise write_result.cancellation
            finalize_result = await _to_thread_terminal(parser.finalize)
            if finalize_result.cancellation is not None:
                raise finalize_result.cancellation
            sync_result = await _to_thread_terminal(self._sync_and_close, destination)
            destination = None
            if sync_result.cancellation is not None:
                raise sync_result.cancellation
            if not message_complete or not file_seen or not file_complete:
                raise _error(422, "invalid_multipart", "incomplete multipart upload")
            if size == 0 or staged is None or final is None:
                raise _error(422, "invalid_media", "uploaded file is empty")
            validation_result = await _to_thread_terminal(
                self._validate, temporary, kind, suffix
            )
            width, height = validation_result.value
            if validation_result.cancellation is not None:
                raise validation_result.cancellation
            commit_result = await _to_thread_terminal(
                self._commit, temporary, staged, reserved_bytes
            )
            reserved_file = False
            reserved_bytes = 0
            staged_owned = staged
            if commit_result.cancellation is not None:
                raise commit_result.cancellation
            saved = _SavedUpload(staged, final, original, size, width, height, kind)
            handed_off = True
            return saved
        except (FormParserError, MultipartParseError) as exc:
            raise _error(
                422, "invalid_multipart", "malformed multipart upload"
            ) from exc
        except OSError as exc:
            raise _error(
                507, "insufficient_storage", "cannot store background"
            ) from exc
        finally:
            if not handed_off:
                cleanup_result = await _to_thread_terminal(
                    self._cleanup_save,
                    destination,
                    temporary,
                    staged_owned,
                    reserved_bytes,
                    reserved_file,
                )
                if cleanup_result.cancellation is not None:
                    raise cleanup_result.cancellation

    def _validate(
        self, path: Path, kind: str, suffix: str | None = None
    ) -> tuple[int, int]:
        if cv2 is None:
            raise _error(503, "decoder_unavailable", "opencv-python is unavailable")
        suffix = (suffix or path.suffix).lower()
        if kind == "image":
            try:
                image = decode_image_to_srgb_bgr(
                    path,
                    _IMAGE_FORMATS.get(suffix, ""),
                    self.limits.image_max_pixels,
                )
            except ColorError as exc:
                if str(exc) == (f"image exceeds {self.limits.image_max_pixels} pixels"):
                    raise _error(
                        422,
                        "image_dimensions_exceeded",
                        "image pixel limit exceeded",
                        limit=self.limits.image_max_pixels,
                    ) from exc
                raise _error(
                    422,
                    "invalid_media",
                    "image cannot be decoded",
                ) from exc
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise _error(
                    422,
                    "invalid_media",
                    "image cannot be decoded",
                ) from exc
            height, width = image.shape[:2]
            if width <= 0 or height <= 0:
                raise _error(
                    422,
                    "invalid_media",
                    "image cannot be decoded",
                )
            return width, height

        try:
            with path.open("rb") as media_file:
                header = media_file.read(16)
        except OSError:
            raise
        header_matches = {
            ".mp4": len(header) >= 8 and header[4:8] == b"ftyp",
            ".mov": len(header) >= 8 and header[4:8] == b"ftyp",
            ".webm": header.startswith(b"\x1aE\xdf\xa3"),
            ".mkv": header.startswith(b"\x1aE\xdf\xa3"),
            ".gif": header.startswith((b"GIF87a", b"GIF89a")),
            ".avi": header.startswith(b"RIFF") and header[8:12] == b"AVI ",
        }.get(suffix, False)
        if not header_matches:
            raise _error(
                422,
                "invalid_media",
                f"video container header does not match {suffix or 'filename'}",
            )
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise _error(422, "invalid_media", "video container cannot be opened")
            metadata_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (
                not math.isfinite(metadata_width)
                or not math.isfinite(metadata_height)
                or metadata_width < 1
                or metadata_height < 1
            ):
                raise _error(422, "invalid_media", "video dimensions are missing")
            if (
                metadata_width > self.limits.video_max_width
                or metadata_height > self.limits.video_max_height
            ):
                raise _error(
                    422,
                    "video_dimensions_exceeded",
                    "video metadata exceeds configured dimensions",
                    max_width=self.limits.video_max_width,
                    max_height=self.limits.video_max_height,
                )
            width = height = 0
            decoded = 0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise _error(
                        422, "invalid_media", "video contains an invalid frame"
                    )
                frame_height, frame_width = frame.shape[:2]
                if (
                    frame_width > self.limits.video_max_width
                    or frame_height > self.limits.video_max_height
                ):
                    raise _error(
                        422,
                        "video_dimensions_exceeded",
                        "video contains a frame that exceeds configured limits",
                        max_width=self.limits.video_max_width,
                        max_height=self.limits.video_max_height,
                    )
                if decoded == 0:
                    width, height = frame_width, frame_height
                decoded += 1
            if decoded == 0:
                raise _error(422, "invalid_media", "video has no decodable frame")
            return width, height
        finally:
            capture.release()

    def remove(self, identifier: str, config: Any) -> None:
        if Path(identifier).name != identifier or identifier.startswith("."):
            raise _error(404, "background_not_found", "background does not exist")
        with self._lock:
            path = self.directory / identifier
            if self._ledger.owns(path):
                raise _error(404, "background_not_found", "background does not exist")
            if path.is_symlink():
                raise _error(404, "background_not_found", "background does not exist")
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError):
                raise _error(404, "background_not_found", "background does not exist")
            if resolved.parent != self.directory.resolve() or not resolved.is_file():
                raise _error(404, "background_not_found", "background does not exist")

            if self._is_active(resolved, config):
                raise _error(
                    409, "background_in_use", "active background cannot be deleted"
                )
            resolved.unlink()


@dataclass
class _CleanupTask:
    saved: _SavedUpload
    final_owned: threading.Event
    done: threading.Event


class _StagedCleanupQueue:
    """Eventually serialize failed-candidate cleanup without holding ASGI."""

    def __init__(self, runtime: RuntimeConfig, coordinator: Any, store: _UploadStore):
        self.runtime = runtime
        self.coordinator = coordinator
        self.store = store
        self._tasks: queue.Queue[_CleanupTask] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(
        self, saved: _SavedUpload, final_owned: threading.Event
    ) -> threading.Event:
        done = threading.Event()
        self._tasks.put(_CleanupTask(saved, final_owned, done))
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run,
                    name="custback-upload-cleanup",
                    daemon=True,
                )
                self._worker.start()
        return done

    def _discard(self, task: _CleanupTask, config: Any) -> None:
        self.store.discard_failed_upload(
            task.saved.path,
            task.saved.final_path,
            final_owned=task.final_owned.is_set(),
            config=config,
        )

    def _attempt(self, task: _CleanupTask) -> bool:
        try:
            if not getattr(self.coordinator, "running", False):
                self._discard(task, _state(self.runtime).config)
                return True
            self.coordinator.apply_storage_mutation(
                lambda config: self._discard(task, config),
                1.0,
            )
            return True
        except BaseException as exc:
            if type(exc).__name__ in {
                "ReconfigurationUnavailable",
                "ConfigConflictError",
                "ConfigVersionConflictError",
            }:
                return False
            log.warning(
                "cannot yet discard failed staged upload %s",
                task.saved.path,
                exc_info=True,
            )
            return False

    def _run(self) -> None:
        while True:
            try:
                task = self._tasks.get(timeout=1.0)
            except queue.Empty:
                with self._lock:
                    if self._tasks.empty():
                        self._worker = None
                        return
                continue
            try:
                if self._attempt(task):
                    task.done.set()
                else:
                    # Give the frame worker time to finish a slow stage, then
                    # rotate the task behind any other pending candidates.
                    threading.Event().wait(0.25)
                    self._tasks.put(task)
            finally:
                self._tasks.task_done()


async def _deny_ws(ws: WebSocket, status: int, code: int, detail: str) -> None:
    response = JSONResponse({"detail": detail}, status_code=status)
    denial = getattr(ws, "send_denial_response", None)
    if denial is not None and "websocket.http.response" in ws.scope.get(
        "extensions", {}
    ):
        await denial(response)
    else:
        await ws.close(code=code, reason=detail)


def create_app(
    runtime: RuntimeConfig,
    hub: FrameHub,
    coordinator: Any,
    *,
    security: SecurityPolicy,
    upload_dir: Path | None = None,
    avatar_client_factory: Any = None,
    on_shutdown: Callable[[], None] | None = None,
    profile_service: Any = None,
) -> FastAPI:
    """Create the authenticated API bound to the active pipeline coordinator.

    ``on_shutdown`` wires the private lifecycle channel (WIN-5.3): when supplied,
    ``POST /lifecycle/shutdown`` invokes it to request a graceful stop.  It is
    restricted to the management bearer so a browser/WebView session cannot stop
    the engine, and the route is absent-in-effect (503) when no supervisor wired
    a handler.
    """

    from ..avatar.store import ThumbnailCache
    from .avatar_proxy import register_avatar_proxy

    if (
        not hasattr(coordinator, "apply_config_patch")
        or not hasattr(coordinator, "apply_staged_config_patch")
        or not hasattr(coordinator, "apply_storage_mutation")
    ):
        raise TypeError("an active transactional pipeline coordinator is required")
    app = FastAPI(
        title="custback",
        version=__version__,
        docs_url=None,
        redoc_url=None,
    )
    stream_lifecycle = stream_lifecycle_for(app)
    startup_api = _state(runtime).config.api
    stream_connections = ConnectionLimiter(startup_api.max_stream_connections)
    output_jpegs = JpegBroadcaster(hub.output, _encode_jpeg)
    raw_jpegs = JpegBroadcaster(hub.raw, _encode_jpeg)
    # Exposed for lifecycle diagnostics and deterministic admission tests.
    app.state.stream_connections = stream_connections
    store = _UploadStore(upload_dir or UPLOAD_DIR, _upload_limits(runtime))
    store.cleanup_staged(_state(runtime).config)
    cleanup_queue = _StagedCleanupQueue(runtime, coordinator, store)
    thumbnails = ThumbnailCache()
    register_avatar_proxy(
        app,
        runtime,
        stream_connections=stream_connections,
        client_factory=avatar_client_factory,
    )
    default_openapi = app.openapi

    def authenticated_openapi():
        if app.openapi_schema is None:
            schema = default_openapi()
            schema.setdefault("components", {}).setdefault("securitySchemes", {})[
                "BearerAuth"
            ] = {"type": "http", "scheme": "bearer"}
            schema["security"] = [{"BearerAuth": []}]
            # This endpoint authenticates the token carried in its JSON body.
            session_path = schema.get("paths", {}).get("/auth/session", {})
            if "post" in session_path:
                session_path["post"]["security"] = []
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = authenticated_openapi

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError):
        # Do not echo Pydantic's input value: request bodies may contain
        # sensitive data and the public contract only needs a stable code.
        return JSONResponse(
            {
                "detail": {
                    "code": "invalid_content",
                    "message": "request content does not match the API contract",
                }
            },
            status_code=422,
        )

    @app.middleware("http")
    async def security_boundary(request: Request, call_next):
        hosts = request.headers.getlist("host")
        origins = request.headers.getlist("origin")
        authorizations = request.headers.getlist("authorization")
        valid_header_shape = (
            len(hosts) == 1 and len(origins) <= 1 and len(authorizations) <= 1
        )
        host = hosts[0] if len(hosts) == 1 else None
        origin = origins[0] if len(origins) == 1 else None
        authorization = authorizations[0] if len(authorizations) == 1 else None
        try:
            if not valid_header_shape or not security.context_allowed(host, origin):
                response: Response = JSONResponse(
                    {
                        "detail": {
                            "code": "forbidden_origin",
                            "message": "request context rejected",
                        }
                    },
                    status_code=403,
                )
            elif request.url.path == "/auth/session" and request.method == "POST":
                response = await call_next(request)
            elif not security.authenticated(
                authorization, request.cookies.get(SESSION_COOKIE)
            ):
                response = _auth_failure(request.url.path)
            else:
                response = await call_next(request)
        except Exception:
            log.exception("unhandled API request failure")
            response = JSONResponse(
                {"detail": {"code": "internal_error", "message": "internal API error"}},
                status_code=500,
            )
        if "X-Config-Version" not in response.headers:
            response.headers["X-Config-Version"] = _version(runtime)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.post(
        "/auth/session",
        status_code=204,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["token"],
                            "additionalProperties": False,
                            "properties": {
                                "token": {"type": "string", "format": "password"}
                            },
                        }
                    }
                },
            }
        },
    )
    async def create_session(request: Request) -> Response:
        body = await _limited_json(request, SESSION_REQUEST_MAX_BYTES)
        if (
            not isinstance(body, dict)
            or set(body) != {"token"}
            or not isinstance(body.get("token"), str)
        ):
            raise _error(
                422,
                "invalid_content",
                "session request must contain exactly one string token",
            )
        supplied = body["token"]
        if not security.bearer_valid(f"Bearer {supplied}"):
            response = _auth_failure(request.url.path)
            response.status_code = 401
            return response
        session = security.sessions.issue()
        response = Response(status_code=204)
        response.set_cookie(
            SESSION_COOKIE,
            session,
            max_age=security.session_ttl_s,
            httponly=True,
            secure=security.secure_cookie,
            samesite="strict",
            path="/",
        )
        return response

    @app.delete("/auth/session", status_code=204)
    async def delete_session(request: Request) -> Response:
        security.sessions.revoke(request.cookies.get(SESSION_COOKIE))
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.post("/lifecycle/shutdown", status_code=202, include_in_schema=False)
    async def lifecycle_shutdown(request: Request) -> Response:
        # Private lifecycle channel for the supervising desktop shell (WIN-5.3).
        # Restricted to the management bearer: a WebView session (HttpOnly cookie
        # only, no bearer) must never be able to stop the engine.  The
        # Host/Origin boundary + SameSite=strict already reject cross-site use,
        # and the bearer is unforgeable from page script, so this cannot be
        # driven by loaded web content.
        if not security.bearer_valid(request.headers.get("authorization")):
            raise _error(
                403,
                "forbidden",
                "lifecycle shutdown requires the management bearer token",
            )
        if on_shutdown is None:
            raise _error(
                503,
                "shutdown_unavailable",
                "no lifecycle shutdown channel is wired to this process",
            )
        on_shutdown()
        return Response(status_code=202)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        from .webui import WEBUI_HTML

        return WEBUI_HTML

    @app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
    async def docs() -> str:
        return DOCS_HTML

    @app.get("/status", response_model=_StatusResponse)
    async def status() -> JSONResponse:
        from ..vcam_native import native_ring_status

        body = hub.stats_dict()
        body["native_ring"] = native_ring_status()
        try:
            public = _StatusResponse.model_validate(body).model_dump(
                mode="json",
                by_alias=True,
            )
        except ValidationError:
            log.error("public status failed schema validation")
            return JSONResponse(
                {
                    "detail": {
                        "code": "internal_error",
                        "message": "internal API error",
                    }
                },
                status_code=500,
                headers={
                    "X-Config-Version": str(body.get("config_version", "unknown"))
                },
            )
        return JSONResponse(
            public,
            headers={"X-Config-Version": str(public["config_version"])},
        )

    @app.get("/config", response_model=PublicAppConfig)
    async def get_config() -> JSONResponse:
        state = _state(runtime)
        return JSONResponse(
            _state_body(state),
            headers={"X-Config-Version": str(state.version)},
        )

    @app.patch(
        "/config",
        response_model=_ConfigPatchResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"type": "object"},
                        "examples": _VISUAL_CONFIG_PATCH_EXAMPLES,
                    },
                    "application/merge-patch+json": {
                        "schema": {"type": "object"},
                        "examples": _VISUAL_CONFIG_PATCH_EXAMPLES,
                    },
                },
            }
        },
    )
    async def patch_config(request: Request) -> JSONResponse:
        patch = await _limited_json(
            request,
            CONFIG_REQUEST_MAX_BYTES,
            media_types=frozenset({"application/json", "application/merge-patch+json"}),
        )
        if not isinstance(patch, dict):
            raise _error(422, "invalid_content", "config patch must be a JSON object")
        expected_version = _conditional_config_version(request)
        state = await asyncio.to_thread(
            _apply_patch,
            runtime,
            coordinator,
            patch,
            expected_version=expected_version,
        )
        return JSONResponse(
            {"config": _state_body(state), "config_version": state.version},
            headers={"X-Config-Version": str(state.version)},
        )

    @app.get("/profiles")
    async def get_profiles() -> JSONResponse:
        if profile_service is None:
            raise _error(
                503,
                "profiles_unavailable",
                "managed profile preferences are not available in this process",
            )
        try:
            body = await asyncio.to_thread(profile_service.status)
        except Exception as exc:
            from ..profile_service import ProfileServiceError

            if isinstance(exc, ProfileServiceError):
                raise _error(exc.status, exc.code, str(exc)) from exc
            log.exception("profile status failed")
            raise _error(500, "profile_error", "profile status is unavailable") from exc
        return JSONResponse(
            body,
            headers={"X-Config-Version": str(body["config_version"])},
        )

    @app.post("/profiles/apply")
    async def apply_profile(request: Request) -> JSONResponse:
        if profile_service is None:
            raise _error(
                503,
                "profiles_unavailable",
                "managed profile preferences are not available in this process",
            )
        body = await _limited_json(request, CONFIG_REQUEST_MAX_BYTES)
        if not isinstance(body, dict) or set(body) != {
            "selections",
            "expected_config_version",
            "expected_preferences_revision",
            "accept_experimental",
        }:
            raise _error(422, "invalid_content", "profile apply request is invalid")
        selections = body["selections"]
        if (
            not isinstance(selections, dict)
            or not selections
            or len(selections) > 2
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in selections.items()
            )
            or type(body["expected_config_version"]) is not int
            or type(body["expected_preferences_revision"]) is not int
            or type(body["accept_experimental"]) is not bool
        ):
            raise _error(422, "invalid_content", "profile apply request is invalid")
        try:
            result = await asyncio.to_thread(
                profile_service.apply,
                selections,
                expected_config_version=body["expected_config_version"],
                expected_preferences_revision=body["expected_preferences_revision"],
                accept_experimental=body["accept_experimental"],
            )
        except Exception as exc:
            from ..profile_service import ProfileServiceError

            if isinstance(exc, ProfileServiceError):
                raise _error(exc.status, exc.code, str(exc)) from exc
            log.exception("profile apply failed")
            raise _error(
                500, "profile_error", "profile selection could not be saved"
            ) from exc
        return JSONResponse(
            result,
            headers={"X-Config-Version": str(result["config_version"])},
        )

    @app.post("/profiles/reset")
    async def reset_profiles(request: Request) -> JSONResponse:
        if profile_service is None:
            raise _error(
                503,
                "profiles_unavailable",
                "managed profile preferences are not available in this process",
            )
        body = await _limited_json(request, CONFIG_REQUEST_MAX_BYTES)
        if not isinstance(body, dict) or set(body) != {
            "axes",
            "expected_config_version",
            "expected_preferences_revision",
        }:
            raise _error(422, "invalid_content", "profile reset request is invalid")
        axes = body["axes"]
        if (
            not isinstance(axes, list)
            or not 1 <= len(axes) <= 2
            or any(not isinstance(axis, str) for axis in axes)
            or type(body["expected_config_version"]) is not int
            or type(body["expected_preferences_revision"]) is not int
        ):
            raise _error(422, "invalid_content", "profile reset request is invalid")
        try:
            result = await asyncio.to_thread(
                profile_service.reset,
                axes,
                expected_config_version=body["expected_config_version"],
                expected_preferences_revision=body["expected_preferences_revision"],
            )
        except Exception as exc:
            from ..profile_service import ProfileServiceError

            if isinstance(exc, ProfileServiceError):
                raise _error(exc.status, exc.code, str(exc)) from exc
            log.exception("profile reset failed")
            raise _error(
                500, "profile_error", "profile preferences could not be reset"
            ) from exc
        return JSONResponse(
            result,
            headers={"X-Config-Version": str(result["config_version"])},
        )

    @app.get("/backgrounds", response_model=_BackgroundListResponse)
    async def list_backgrounds() -> dict:
        await asyncio.to_thread(store.ensure_directory)
        files = await asyncio.to_thread(
            lambda: sorted(
                p.name
                for p in store.directory.iterdir()
                if p.is_file()
                and not p.is_symlink()
                and not store._ledger.owns(p)
                and not p.name.startswith(".upload-")
                and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
            )
        )
        return {
            "files": files,
            "modes": list(MODES),
            "directory": str(store.directory),
        }

    @app.get(
        "/backgrounds/{identifier}/thumbnail.jpg",
        response_class=Response,
        responses={200: {"content": {"image/jpeg": {}}}},
    )
    async def background_thumbnail(identifier: str) -> Response:
        from ..avatar.store import StoreError, render_media_thumbnail

        if (
            not identifier
            or identifier.startswith(".")
            or os.path.basename(identifier.replace("\\", "/")) != identifier
        ):
            raise _error(404, "media_not_found", "no such stored file")
        path = store.directory / identifier
        suffix = path.suffix.lower()
        kind = "image" if suffix in IMAGE_EXTS else "video"
        if (
            suffix not in IMAGE_EXTS | VIDEO_EXTS
            or not path.is_file()
            or path.is_symlink()
            or store._ledger.owns(path)
        ):
            raise _error(404, "media_not_found", "no such stored file")
        stat_result = path.stat()
        key = (str(path), stat_result.st_mtime_ns, stat_result.st_size)
        cached = thumbnails.get(key)
        if cached is None:
            try:
                cached = await asyncio.to_thread(
                    render_media_thumbnail,
                    path,
                    kind,
                    max_pixels=store.limits.image_max_pixels,
                )
            except StoreError as exc:
                raise _error(exc.status, exc.code, str(exc)) from exc
            thumbnails.put(key, cached)
        return Response(content=cached, media_type="image/jpeg")

    async def upload(request: Request, kind: str) -> JSONResponse:
        saved = await store.save(request, kind)
        field = "image_path" if kind == "image" else "video_path"
        promoted = threading.Event()

        def promote_candidate() -> None:
            store.promote(saved.path, saved.final_path)
            promoted.set()

        def rollback_candidate() -> None:
            store.rollback_promotion(saved.path, saved.final_path)
            promoted.clear()

        # Keep the authoritative concurrent Future separate from asyncio's
        # cancellation scopes. Cancelling an asyncio.to_thread wrapper can
        # mark that wrapper cancelled while its worker continues, which made
        # it unsafe to decide whether the committed file should be removed.
        apply_future = _ACTIVATION_EXECUTOR.submit(
            coordinator.apply_staged_config_patch,
            {"background": {"mode": kind, field: str(saved.final_path)}},
            {"background": {"mode": kind, field: str(saved.path)}},
            promote_candidate,
            rollback_candidate,
            5.0,
        )

        def discard_failed_candidate() -> threading.Event:
            return cleanup_queue.submit(saved, promoted)

        def settle_activation_ownership(
            completed: concurrent.futures.Future,
        ) -> None:
            try:
                completed.result()
            except BaseException:
                discard_failed_candidate()
            else:
                try:
                    store._finish_transaction(saved.path, saved.final_path)
                except OSError:
                    # The active config remains authoritative.  Its durable
                    # ownership record is charged and startup recovery will
                    # adopt it rather than deleting the active inode.
                    log.warning(
                        "cannot yet finalize activated upload ownership %s",
                        saved.final_path,
                        exc_info=True,
                    )

        # Settlement belongs to the authoritative concurrent future, not the
        # request task.  It therefore runs even if cancellation lands after
        # apply succeeds or while an asyncio worker would be admitted.
        apply_future.add_done_callback(settle_activation_ownership)

        try:
            state = await asyncio.shield(asyncio.wrap_future(apply_future))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            cleanup_done = discard_failed_candidate()
            # Responsive pipelines normally clean on the next frame, keeping
            # the synchronous failure contract deterministic. If the worker
            # is stalled, return promptly while the daemon queue keeps retrying.
            await asyncio.to_thread(cleanup_done.wait, 0.5)
            if isinstance(exc, HTTPException):
                raise
            raise _map_apply_error(exc) from exc
        try:
            await asyncio.to_thread(
                store._finish_transaction, saved.path, saved.final_path
            )
        except OSError:
            # Activation has already committed.  Keep the ownership charge and
            # let active-config-aware restart recovery adopt this exact inode.
            log.warning(
                "cannot yet finalize activated upload ownership %s",
                saved.final_path,
                exc_info=True,
            )
        body = {
            "id": saved.final_path.name,
            "original_name": saved.original_name,
            "bytes": saved.size,
            "width": saved.width,
            "height": saved.height,
            "kind": saved.kind,
            "config_version": state.version,
        }
        return JSONResponse(
            body,
            status_code=201,
            headers={"X-Config-Version": str(state.version)},
        )

    upload_openapi = {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                }
            },
        }
    }

    @app.post(
        "/background/image",
        status_code=201,
        response_model=_UploadResponse,
        openapi_extra=upload_openapi,
    )
    async def upload_image(request: Request) -> JSONResponse:
        return await upload(request, "image")

    @app.post(
        "/background/video",
        status_code=201,
        response_model=_UploadResponse,
        openapi_extra=upload_openapi,
    )
    async def upload_video(request: Request) -> JSONResponse:
        return await upload(request, "video")

    @app.delete("/backgrounds/{identifier}", status_code=204)
    async def delete_background(identifier: str) -> Response:
        def mutate(config) -> None:
            store.remove(identifier, config)

        try:
            state = await asyncio.to_thread(
                coordinator.apply_storage_mutation, mutate, 5.0
            )
        except HTTPException:
            raise
        except BaseException as exc:
            raise _map_apply_error(exc) from exc
        return Response(
            status_code=204,
            headers={"X-Config-Version": str(state.version)},
        )

    @app.get(
        "/video/snapshot.jpg",
        response_class=Response,
        responses={
            200: {
                "description": "Latest composited frame as JPEG",
                "content": {
                    "image/jpeg": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
    )
    async def snapshot() -> Response:
        frame, _ = hub.output.latest()
        if frame is None:
            raise _error(503, "frame_unavailable", "no frame yet")
        height, width = frame.shape[:2]
        return Response(
            content=await asyncio.to_thread(_encode_jpeg, frame),
            media_type="image/jpeg",
            headers={
                "X-Frame-Width": str(width),
                "X-Frame-Height": str(height),
            },
        )

    @app.get(
        "/video/mjpeg",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "Continuous multipart JPEG stream",
                "content": {
                    "multipart/x-mixed-replace": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def mjpeg() -> StreamingResponse:
        boundary = "custbackframe"
        canvas_width, canvas_height = resolved_output_size(_state(runtime).config)
        lease = stream_connections.try_acquire()
        if lease is None:
            raise _error(
                429,
                "stream_limit",
                "authenticated stream connection limit reached",
            )
        registration = stream_lifecycle.register("core_mjpeg")
        if registration is None:
            lease.release()
            raise _error(503, "api_shutting_down", "the API is shutting down")

        async def gen():
            try:
                async with output_jpegs.subscribe() as subscription:
                    seq = -1
                    while True:
                        jpeg, seq = await subscription.get(seq, 1.0)
                        if jpeg is None:
                            continue
                        yield (
                            (
                                f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                                f"X-Frame-Width: {canvas_width}\r\n"
                                f"X-Frame-Height: {canvas_height}\r\n"
                                f"Content-Length: {len(jpeg)}\r\n\r\n"
                            ).encode()
                            + jpeg
                            + b"\r\n"
                        )
            finally:
                # Direct iterator consumers and the response wrapper may both
                # release; ConnectionLease is deliberately idempotent.
                lease.release()

        return LeasedStreamingResponse(
            gen(),
            lease=lease,
            registration=registration,
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        )

    @app.websocket("/ws/frames")
    async def ws_frames(ws: WebSocket, stream: str = "raw") -> None:
        hosts = ws.headers.getlist("host")
        origins = ws.headers.getlist("origin")
        authorizations = ws.headers.getlist("authorization")
        if (
            len(hosts) != 1
            or len(origins) > 1
            or len(authorizations) > 1
            or not security.context_allowed(hosts[0], origins[0] if origins else None)
        ):
            await _deny_ws(ws, 403, 4403, "request origin or host rejected")
            return
        authorization = authorizations[0] if authorizations else None
        management_authenticated = security.authenticated(
            authorization,
            ws.cookies.get(SESSION_COOKIE),
        )
        renderer_authenticated = stream == "raw" and security.renderer_bearer_valid(
            authorization
        )
        if not (management_authenticated or renderer_authenticated):
            await _deny_ws(ws, 401, 4401, "valid route credential required")
            return
        if stream not in ("raw", "output"):
            await _deny_ws(ws, 400, 4400, "stream must be raw|output")
            return

        lease = stream_connections.try_acquire()
        if lease is None:
            await _deny_ws(
                ws,
                429,
                4429,
                "authenticated stream connection limit reached",
            )
            return

        accepted = False

        async def close_for_shutdown() -> None:
            if accepted:
                with contextlib.suppress(RuntimeError):
                    await ws.close(code=1012, reason="API shutting down")

        renderer_connection = bool(stream == "raw" and renderer_authenticated)
        registration = stream_lifecycle.register(
            "renderer_websocket" if renderer_connection else "management_websocket",
            close=close_for_shutdown,
        )
        if registration is None:
            lease.release()
            await _deny_ws(ws, 503, 1012, "API shutting down")
            return

        remote_session = None
        try:
            canvas_width, canvas_height = resolved_output_size(_state(runtime).config)
            await ws.accept(
                headers=[
                    (b"x-custback-frame-width", str(canvas_width).encode("ascii")),
                    (b"x-custback-frame-height", str(canvas_height).encode("ascii")),
                ]
            )
            accepted = True
            remote_session = (
                hub.remote_client_connected() if renderer_connection else None
            )
            jpegs = raw_jpegs if stream == "raw" else output_jpegs
            stop = asyncio.Event()
            ws_limit = int(
                getattr(
                    _state(runtime).config.api,
                    "ws_max_bytes",
                    16 * 1024**2,
                )
            )

            async def sender() -> None:
                async with jpegs.subscribe() as subscription:
                    seq = -1
                    while not stop.is_set():
                        if remote_session is not None and not hub.remote_session_valid(
                            remote_session
                        ):
                            await ws.close(
                                code=1012,
                                reason="renderer session invalidated",
                            )
                            return
                        jpeg, seq = await subscription.get(seq, 0.5)
                        if jpeg is not None:
                            if renderer_connection:
                                raw_epoch = hub.remote_raw_epoch_for_sequence(seq)
                                if raw_epoch is None or raw_epoch <= 0:
                                    # The bounded binding may expire behind an
                                    # unusually stalled encoder. Epoch zero is
                                    # management-preview-only local input.
                                    # Never relabel either as renderer input.
                                    continue
                                try:
                                    jpeg = encode_remote_frame(
                                        "raw-input",
                                        raw_epoch,
                                        jpeg,
                                        max_message_bytes=ws_limit,
                                    )
                                except ValueError:
                                    await ws.close(
                                        code=1009,
                                        reason=("frame exceeds configured byte limit"),
                                    )
                                    return
                            await ws.send_bytes(jpeg)

            async def receiver() -> None:
                if not renderer_connection:
                    while True:
                        message = await ws.receive()
                        if message.get("type") == "websocket.disconnect":
                            return
                        await ws.close(
                            code=1008,
                            reason="stream is read-only",
                        )
                        return
                expected_size = resolved_output_size(_state(runtime).config)
                while True:
                    message = await ws.receive()
                    kind = message.get("type")
                    if kind == "websocket.disconnect":
                        return
                    data = message.get("bytes")
                    if data is None:
                        await ws.close(
                            code=1003,
                            reason="binary JPEG frames required",
                        )
                        return
                    if len(data) > ws_limit:
                        await ws.close(
                            code=1009,
                            reason="frame exceeds configured byte limit",
                        )
                        return
                    try:
                        envelope = decode_remote_frame(
                            data,
                            expected_kind="rendered-output",
                            max_message_bytes=ws_limit,
                        )
                    except (RemoteFrameProtocolError, TypeError, ValueError):
                        await ws.close(
                            code=1007,
                            reason="invalid renderer frame envelope",
                        )
                        return
                    frame = await asyncio.to_thread(
                        _decode_jpeg,
                        envelope.jpeg,
                        expected_size,
                    )
                    if frame is None:
                        await ws.close(
                            code=1007,
                            reason="invalid JPEG or frame dimensions",
                        )
                        return
                    assert remote_session is not None
                    hub.push_remote_frame(
                        frame,
                        raw_epoch=envelope.raw_epoch,
                        session_id=remote_session,
                    )

            send_task = asyncio.create_task(sender())
            receive_task = asyncio.create_task(receiver())
            try:
                done, _pending = await asyncio.wait(
                    {send_task, receive_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    try:
                        task.result()
                    except Exception:
                        log.debug(
                            "WebSocket frame task stopped",
                            exc_info=True,
                        )
            finally:
                stop.set()
                for task in (send_task, receive_task):
                    task.cancel()
                await asyncio.gather(
                    send_task,
                    receive_task,
                    return_exceptions=True,
                )
        finally:
            if remote_session is not None:
                hub.remote_client_disconnected(remote_session)
            registration.release()
            lease.release()

    return app
