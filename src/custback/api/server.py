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
from pydantic import BaseModel, ValidationError

try:  # python-multipart >= 0.0.12 canonical namespace
    from python_multipart.exceptions import FormParserError, MultipartParseError
    from python_multipart.multipart import MultipartParser, parse_options_header
except ImportError:  # python-multipart 0.0.9 minimum compatibility
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
from ..storage_tx import OwnedPath, OwnershipLedger, rename_noreplace
from .security import SESSION_COOKIE, SecurityPolicy
from .streaming import ConnectionLimiter, JpegBroadcaster, LeasedStreamingResponse

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


class _StatusResponse(BaseModel):
    run_id: str
    frames_in: int
    frames_out: int
    fps: float
    mode: str
    segmentation_backend: str
    segmentation_device: str
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
    processing_deadline_misses: int
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
    color_correction_warming: bool
    color_correction_stale: bool
    color_correction_applied_frames: int
    color_correction_bypassed_frames: int
    color_correction_scene_cuts: int
    color_correction_transitions: int
    color_input_assumption: str
    composite_ms: float | None
    output_send_ms: float | None
    frame_processing_ms: float | None
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
    background_video_orientation_status: str | None
    background_video_metadata_rotation: int | None
    background_video_auto_rotation_disabled: bool | None
    background_video_decoder_backend: str | None
    background_video_color_status: str | None
    background_video_input_color: str | None
    background_video_output_color: str | None
    background_video_color_assumed_fields: list[str]
    background_video_color_overridden_fields: list[str]
    uptime_s: float


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
        return _error(422, "activation_failed", str(exc))
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


def _apply_patch(runtime: RuntimeConfig, coordinator: Any, patch: dict[str, Any]):
    try:
        return coordinator.apply_config_patch(patch, timeout=5.0, origin="api")
    except BaseException as exc:
        raise _map_apply_error(exc) from exc


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
    register_avatar_proxy(app, runtime, client_factory=avatar_client_factory)
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
        return JSONResponse(
            body, headers={"X-Config-Version": str(body["config_version"])}
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
        state = await asyncio.to_thread(_apply_patch, runtime, coordinator, patch)
        return JSONResponse(
            {"config": _state_body(state), "config_version": state.version},
            headers={"X-Config-Version": str(state.version)},
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

        remote_session = None
        try:
            canvas_width, canvas_height = resolved_output_size(_state(runtime).config)
            await ws.accept(
                headers=[
                    (b"x-custback-frame-width", str(canvas_width).encode("ascii")),
                    (b"x-custback-frame-height", str(canvas_height).encode("ascii")),
                ]
            )
            remote_session = hub.remote_client_connected() if stream == "raw" else None
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
                            await ws.send_bytes(jpeg)

            async def receiver() -> None:
                if stream == "output":
                    while True:
                        message = await ws.receive()
                        if message.get("type") == "websocket.disconnect":
                            return
                        await ws.close(
                            code=1008,
                            reason="output stream is read-only",
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
                    frame = await asyncio.to_thread(
                        _decode_jpeg,
                        data,
                        expected_size,
                    )
                    if frame is None:
                        await ws.close(
                            code=1007,
                            reason="invalid JPEG or frame dimensions",
                        )
                        return
                    assert remote_session is not None
                    hub.push_remote_frame(frame, remote_session)

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
            lease.release()

    return app
