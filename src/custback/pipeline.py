"""Main processing loop and transactional live reconfiguration."""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Mapping
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeout,
)
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import Any, Callable, cast

import numpy as np
import cv2

from .backgrounds import (
    BackdropFrameTiming,
    BlurBackdrop,
    ColorBackdrop,
    ImageBackdrop,
    create_backdrop,
)
from .capture import CapturedFrame, open_capture
from .cadence import CadenceTracker
from .color import (
    ANALYSIS_LONG_EDGE,
    ColorError,
    ColorHarmonizer,
    ColorReason,
    ColorTransform,
    HarmonizerPhase,
    HarmonizerSnapshot,
    IDENTITY_TRANSFORM,
    _bgr_u8_to_linear_bgr_prevalidated as bgr_u8_to_linear_bgr,
    _estimate_color_transform_linear_bgr_prevalidated as estimate_color_transform_linear,
    _linear_bgr_analysis_raster_prevalidated,
)
from .compositor import (
    COMPOSITOR_SUBSTAGE_NAMES,
    LegacyCompositorWorkspace,
    PreparedLightWrap,
    _composite_linear_bgr_prevalidated as composite_linear_predecoded,
    composite,
    _composite_legacy_bgr_prevalidated as composite_legacy_predecoded,
    prepare_light_wrap,
    prepare_static_light_wrap,
)
from .config import (
    AppConfig,
    BlendSpace,
    ConfigState,
    ConfigVersionConflictError,
    RuntimeConfig,
    changed_config_paths,
    restart_only_config_paths,
    resolved_output_size,
)
from .diagnostics import sanitized_config_summary
from .geometry import Size, apply_transform, plan_transform, validate_bgr_frame
from .hub import FrameHub, TIMING_FIELD_NAMES, TIMING_SCHEMA_VERSION
from .matte_diagnostics import (
    MatteCaptureMetadata,
    MatteDiagnosticRecorder,
    MatteFrameEvidence,
    process_rss_bytes,
    segmenter_diagnostics_snapshot,
)
from .matte_live_diagnostics import LocalMatteDiagnosticMonitor
from .matte_policy import MattePolicySnapshot, resolve_matte_policy
from .matte_rollout import (
    MatteRolloutAttempt,
    MatteRolloutTelemetry,
    classify_matte_patch,
)
from .output_scheduler import (
    OutputPublisher,
    OutputPublisherError,
    OutputSendReceipt,
    PublicationPolicy,
    RemoteOutputProof,
    SafeBaseFrame,
    thaw_status,
)
from .runtime_performance import (
    RUNTIME_PERFORMANCE_MIN_ATTAINMENT,
    PerformanceEpochKey,
    PublisherTelemetry,
    RuntimePerformanceTracker,
)
from .light_wrap import (
    LightWrapFrameContext,
    LightWrapSnapshot,
    LightWrapStabilizer,
)
from .segmentation import (
    MaskRefiner,
    NullSegmenter,
    SegmentationFrameContext,
    SegmentationTimeline,
    Segmenter,
    SegmenterPreparation,
    RVMTelemetry,
    TemporalResetReason,
    create_segmenter,
    refiner_for,
    segmenter_matte_backend_kind,
    segmenter_selection_status,
)
from .vcam import OutputSendTiming, open_output

log = logging.getLogger(__name__)

_VIDEO_STATS_DEFAULTS: dict[str, object] = {
    "background_video_source_fps": None,
    "background_video_timing_mode": None,
    "background_video_frames_displayed": 0,
    "background_video_frames_skipped": 0,
    "background_video_frames_reused": 0,
    "background_video_skip_ratio": 0.0,
    "background_video_seek_count": 0,
    "background_video_decode_failures": 0,
    "background_video_orientation_status": None,
    "background_video_metadata_rotation": None,
    "background_video_auto_rotation_disabled": None,
    "background_video_decoder_backend": None,
    "background_video_color_status": None,
    "background_video_input_color": None,
    "background_video_output_color": None,
    "background_video_color_assumed_fields": [],
    "background_video_color_overridden_fields": [],
}

_VIDEO_LIFETIME_FIELD_PAIRS = (
    (
        "background_video_frames_displayed",
        "background_video_lifetime_frames_displayed",
    ),
    (
        "background_video_frames_skipped",
        "background_video_lifetime_frames_skipped",
    ),
    (
        "background_video_frames_reused",
        "background_video_lifetime_frames_reused",
    ),
    (
        "background_video_seek_count",
        "background_video_lifetime_seek_count",
    ),
    (
        "background_video_decode_failures",
        "background_video_lifetime_decode_failures",
    ),
)
_MAX_VIDEO_LIFETIME_COUNT = 2**63 - 1


def _video_counter(stats: Mapping[str, object], name: str) -> int:
    value = stats.get(name, 0)
    if type(value) is not int or value < 0:
        return 0
    return min(value, _MAX_VIDEO_LIFETIME_COUNT)


@dataclass
class _BackgroundVideoLifetimeCounters:
    """Retain scalar playback totals after a provider generation is retired."""

    retired: dict[str, int] = field(
        default_factory=lambda: {
            lifetime_name: 0
            for _current_name, lifetime_name in _VIDEO_LIFETIME_FIELD_PAIRS
        }
    )

    @staticmethod
    def _provider_stats(provider: object) -> Mapping[str, object]:
        getter = getattr(provider, "stats_dict", None)
        if not callable(getter):
            return {}
        try:
            values = getter()
        except Exception:
            # Teardown telemetry is best effort and must never revoke an
            # already committed configuration change.
            return {}
        return values if isinstance(values, Mapping) else {}

    def retire(self, provider: object) -> None:
        """Fold one quiescent provider's final counters into run totals."""

        stats = self._provider_stats(provider)
        for current_name, lifetime_name in _VIDEO_LIFETIME_FIELD_PAIRS:
            self.retired[lifetime_name] = min(
                _MAX_VIDEO_LIFETIME_COUNT,
                self.retired[lifetime_name] + _video_counter(stats, current_name),
            )

    def project(self, current_stats: Mapping[str, object]) -> dict[str, int]:
        """Return retired plus active-provider counters without mutating either."""

        return {
            lifetime_name: min(
                _MAX_VIDEO_LIFETIME_COUNT,
                self.retired[lifetime_name]
                + _video_counter(current_stats, current_name),
            )
            for current_name, lifetime_name in _VIDEO_LIFETIME_FIELD_PAIRS
        }


_ACCELERATION_STATS_DEFAULTS: dict[str, object] = {
    "acceleration_mode": "",
    "acceleration_requested_provider": "",
    "acceleration_device_id": 0,
    "acceleration_state": "",
    "acceleration_active_provider": "",
    "acceleration_fallback_active": False,
    "acceleration_fallback_reason": "",
    "acceleration_fallback_count": 0,
    "acceleration_last_transition_ms": None,
}

_COLOR_ELIGIBLE_MODES = frozenset({"image", "video", "camera"})
_COLOR_REASON_DEBOUNCE_S = 0.5


def _acceleration_stats(segmenter: Any) -> dict[str, object]:
    """Read the segmenter's latched acceleration status, if it has one.

    Only the RVM segmenter runs an ONNX Runtime execution provider; the
    heuristic/mediapipe/null backends report the neutral defaults.  Reading the
    latched state each frame is what makes a mid-run GPU->CPU fallback visible
    in ``/status`` without any extra notification path.
    """

    accel = getattr(segmenter, "accel", None)
    if accel is None:
        return dict(_ACCELERATION_STATS_DEFAULTS)
    status = accel.status()
    fallback_active = bool(status.fallback_active)
    return {
        "acceleration_mode": status.requested_mode,
        "acceleration_requested_provider": status.requested_provider,
        "acceleration_device_id": status.device_id,
        "acceleration_state": status.state,
        "acceleration_active_provider": status.active_provider,
        "acceleration_fallback_active": fallback_active,
        "acceleration_fallback_reason": (
            "requested accelerator unavailable; using CPU" if fallback_active else ""
        ),
        "acceleration_fallback_count": status.fallback_count,
        "acceleration_last_transition_ms": status.last_transition_ms,
    }


def _acceleration_evidence(segmenter: Any) -> dict[str, object]:
    """Return bounded, path-free provider evidence for a private frame.

    Public status retains a short diagnostic reason for operators.  Replay
    evidence instead records only the fact and count of fallback plus a stable
    reason code, so an exception containing a model or user path cannot enter
    an otherwise content-free qualification report.
    """

    accel = getattr(segmenter, "accel", None)
    if accel is None:
        return {
            "applicable": False,
            "requested_mode": "",
            "requested_provider": "",
            "device_id": 0,
            "state": "",
            "active_provider": "",
            "fallback_active": False,
            "fallback_count": 0,
            "fallback_reason_code": "",
        }
    status = accel.status()
    fallback_active = bool(status.fallback_active)
    requested_mode = str(status.requested_mode)
    requested_provider = str(status.requested_provider)
    state = str(status.state)
    active_provider = str(status.active_provider)
    return {
        "applicable": True,
        "requested_mode": (
            requested_mode
            if requested_mode in {"auto", "cpu", "gpu_required"}
            else "unknown"
        ),
        "requested_provider": (
            requested_provider
            if requested_provider in {"auto", "cuda", "directml"}
            else "unknown"
        ),
        "device_id": int(status.device_id),
        "state": (
            state
            if state in {"starting", "gpu_probing", "gpu_active", "cpu_fallback"}
            else "unknown"
        ),
        "active_provider": (
            active_provider
            if active_provider in {"cpu", "cuda", "directml", "coreml"}
            else "unknown"
        ),
        "fallback_active": fallback_active,
        "fallback_count": int(status.fallback_count),
        "fallback_reason_code": ("provider-fallback" if fallback_active else ""),
    }


def _rvm_telemetry_evidence(segmenter: Any) -> dict[str, object]:
    """Return path-free model/detail/timing facts for the last valid RVM frame."""

    snapshotter = getattr(segmenter, "rvm_telemetry_snapshot", None)
    if not callable(snapshotter):
        return {"applicable": False}
    try:
        snapshot = snapshotter()
    except Exception:
        return {"applicable": False}
    if not isinstance(snapshot, RVMTelemetry):
        return {"applicable": False}

    def shape(value: object, dimensions: int) -> list[int] | None:
        if value is None:
            return None
        if (
            not isinstance(value, tuple)
            or len(value) != dimensions
            or any(type(item) is not int or item <= 0 for item in value)
        ):
            raise ValueError
        return list(value)

    def optional_finite(value: object, *, maximum: float | None = None) -> float | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (maximum is not None and float(value) > maximum)
        ):
            raise ValueError
        return float(value)

    try:
        input_shape = shape(snapshot.input_frame_shape, 2)
        alpha_shape = shape(snapshot.output_alpha_shape, 2)
        foreground_shape = shape(snapshot.output_foreground_shape, 3)
        configured_mode = snapshot.configured_downsample_mode
        configured_ratio = optional_finite(
            snapshot.configured_downsample_ratio,
            maximum=1.0,
        )
        resolved_ratio = optional_finite(
            snapshot.resolved_downsample_ratio,
            maximum=1.0,
        )
        if (
            configured_ratio is None
            or configured_mode not in {"auto", "explicit"}
            or (configured_mode == "auto" and configured_ratio != 0.0)
            or (configured_mode == "explicit" and configured_ratio <= 0.0)
            or type(snapshot.model_builtin) is not bool
            or not isinstance(snapshot.model_identity, str)
            or not 1 <= len(snapshot.model_identity) <= 128
            or any(
                not (
                    character.isascii() and (character.isalnum() or character in "._-")
                )
                for character in snapshot.model_identity
            )
            or (
                snapshot.model_sha256 is not None
                and (
                    not isinstance(snapshot.model_sha256, str)
                    or len(snapshot.model_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in snapshot.model_sha256
                    )
                )
            )
            or (
                snapshot.model_bytes is not None
                and (type(snapshot.model_bytes) is not int or snapshot.model_bytes <= 0)
            )
            or snapshot.acceleration_state
            not in {"starting", "gpu_probing", "gpu_active", "cpu_fallback"}
            or snapshot.acceleration_active_provider
            not in {"", "cpu", "cuda", "directml", "coreml"}
            or type(snapshot.acceleration_fallback_active) is not bool
            or type(snapshot.acceleration_fallback_count) is not int
            or snapshot.acceleration_fallback_count < 0
        ):
            raise ValueError
        preprocess_ms = optional_finite(snapshot.preprocess_ms)
        session_run_ms = optional_finite(snapshot.session_run_ms)
        postprocess_ms = optional_finite(snapshot.postprocess_ms)
    except (TypeError, ValueError):
        return {"applicable": False}

    return {
        "applicable": True,
        "input_frame_shape": input_shape,
        "output_alpha_shape": alpha_shape,
        "output_foreground_shape": foreground_shape,
        "configured_downsample_mode": configured_mode,
        "configured_downsample_ratio": configured_ratio,
        "resolved_downsample_ratio": resolved_ratio,
        "preprocess_ms": preprocess_ms,
        "session_run_ms": session_run_ms,
        "postprocess_ms": postprocess_ms,
        "model_builtin": snapshot.model_builtin,
        "model_identity": snapshot.model_identity,
        "model_sha256": snapshot.model_sha256,
        "model_bytes": snapshot.model_bytes,
        "acceleration_state": snapshot.acceleration_state,
        "acceleration_active_provider": snapshot.acceleration_active_provider,
        "acceleration_fallback_active": snapshot.acceleration_fallback_active,
        "acceleration_fallback_count": snapshot.acceleration_fallback_count,
    }


def _output_sink_evidence(output: Any) -> dict[str, object]:
    """Return the actual sink's path-free pacing contract for timing evidence."""

    if output is None:
        return {
            "applicable": False,
            "backend": "unknown",
            "paces": None,
        }
    backend = {
        "NullOutput": "null",
        "PyVirtualCamOutput": "pyvirtualcam",
        "NativeVirtualCameraOutput": "native",
    }.get(type(output).__name__, "unknown")
    paces = getattr(output, "paces", None)
    return {
        "applicable": True,
        "backend": backend,
        "paces": paces if type(paces) is bool else None,
    }


# Retain compact downsampled fingerprints for the complete pipeline session.
# The bound is a fail-closed admission limit, never an eviction policy: once it
# is reached, the renderer session is invalidated and output remains the slate.
_RAW_FINGERPRINT_HISTORY = 1_000_000
_RAW_FINGERPRINT_SIZE = (16, 12)
_RAW_ECHO_PIXEL_TOLERANCE = 3
_RAW_ECHO_CHANGED_FRACTION = 0.02
_RAW_ECHO_MEAN_DELTA = 4.0
_RAW_ECHO_LOWRES_MEAN_DELTA = 20.0
_RAW_ECHO_LOWRES_CORRELATION = 0.93


def _segmenter_key(cfg: AppConfig) -> tuple[object, ...]:
    """Inputs that require rebuilding the segmenter (and its ORT session).

    The acceleration policy is a separate top-level section but is consumed at
    segmenter construction (provider selection / proof), so a change to it must
    re-stage the segmenter exactly like a segmentation change would.
    """

    return (
        cfg.segmentation.model_dump(mode="python"),
        cfg.acceleration.model_dump(mode="python"),
    )


def _backdrop_provider_key(cfg: AppConfig) -> tuple[object, ...]:
    """Only inputs that require constructing a different backdrop provider.

    Presentation-only geometry is deliberately excluded: changing fit or anchor
    must not rewind a live video or reopen a camera backdrop.
    """

    background = cfg.background
    mode = (
        background.remote_fallback_mode
        if background.mode == "remote"
        else background.mode
    )
    if mode == "passthrough":
        return ("passthrough",)
    if mode == "blur":
        return ("blur", background.blur_strength)
    if mode == "color":
        return ("color", background.color)
    if mode == "image":
        return ("image", background.image_path, cfg.api.uploads.image_max_pixels)
    if mode == "video":
        return (
            "video",
            background.video_path,
            cfg.api.uploads.video_max_width,
            cfg.api.uploads.video_max_height,
            background.video_color_matrix,
            background.video_color_range,
            background.video_color_primaries,
            background.video_color_transfer,
        )
    if mode == "camera":
        target = cfg.resolved_backdrop_target()
        if target is None:
            raise ValueError("camera backdrop target is not configured")
        return ("camera", target.identifier, target.source)
    raise ValueError(f"unknown background mode: {mode!r}")


def _backdrop_visual_key(cfg: AppConfig) -> tuple[object, ...]:
    """Backdrop identity plus presentation policy for fitted-cache generations."""

    background = cfg.background
    return (
        background.mode,
        _backdrop_provider_key(cfg),
        background.fit_mode,
        background.anchor_x,
        background.anchor_y,
    )


def _visual_state_key(cfg: AppConfig) -> tuple[object, ...]:
    """Geometry/working-space inputs that invalidate temporal color state."""

    camera = cfg.camera
    correction = cfg.compositing.color_correction
    return (
        camera.width,
        camera.height,
        camera.fit_mode,
        camera.anchor_x,
        camera.anchor_y,
        camera.rotation,
        camera.mirror,
        resolved_output_size(cfg),
        _backdrop_visual_key(cfg),
        cfg.compositing.blend_space,
        correction.mode,
    )


def _color_scalar_policy_key(cfg: AppConfig) -> tuple[object, ...]:
    """Hot scalar policy that can retain an already-applied transform."""

    correction = cfg.compositing.color_correction
    return (
        correction.strength,
        correction.exposure_limit_ev,
        correction.white_balance_strength,
        correction.adaptation_time_s,
    )


def _color_state_key(cfg: AppConfig) -> tuple[object, ...]:
    """Hard reset inputs: mode/source/geometry/model/working-space changes."""

    return (_visual_state_key(cfg), _segmenter_key(cfg))


def _new_color_harmonizer(cfg: AppConfig) -> ColorHarmonizer:
    correction = cfg.compositing.color_correction
    return ColorHarmonizer(
        correction.adaptation_time_s,
        mode=cfg.background.mode,
    )


def _light_wrap_state_key(cfg: AppConfig) -> tuple[object, ...]:
    """Every input that gives a dynamic wrap sample different semantics."""

    policy = cfg.compositing.light_wrap_stabilization
    enabled = (
        policy.mode == "temporal_bounded"
        and cfg.compositing.light_wrap > 0.0
        and cfg.background.mode in {"video", "camera"}
    )
    if not enabled:
        return ("off",)
    return (
        policy.mode,
        policy.time_constant_s,
        cfg.compositing.light_wrap,
        cfg.compositing.blend_space,
        resolved_output_size(cfg),
        _backdrop_visual_key(cfg),
    )


def _new_light_wrap_stabilizer(cfg: AppConfig) -> LightWrapStabilizer | None:
    if _light_wrap_state_key(cfg) == ("off",):
        return None
    return LightWrapStabilizer(cfg.compositing.light_wrap_stabilization.time_constant_s)


def _plan_stats(prefix: str, plan: Any) -> dict[str, object]:
    """Flatten one immutable geometry plan into the public status contract."""

    crop = plan.crop_rect
    padding = plan.padding
    return {
        f"{prefix}_fit": plan.fit,
        f"{prefix}_rotation": plan.rotation,
        f"{prefix}_mirror": plan.mirror,
        f"{prefix}_scale_x": plan.scale_x,
        f"{prefix}_scale_y": plan.scale_y,
        f"{prefix}_crop_left": crop.left,
        f"{prefix}_crop_top": crop.top,
        f"{prefix}_crop_right": crop.right,
        f"{prefix}_crop_bottom": crop.bottom,
        f"{prefix}_pad_left": padding.left,
        f"{prefix}_pad_top": padding.top,
        f"{prefix}_pad_right": padding.right,
        f"{prefix}_pad_bottom": padding.bottom,
    }


def _empty_plan_stats(
    prefix: str,
    *,
    fit: str,
    rotation: int = 0,
    mirror: bool = False,
) -> dict[str, object]:
    """Return an explicit unknown-plan state without inventing geometry."""

    return {
        f"{prefix}_fit": fit,
        f"{prefix}_rotation": rotation,
        f"{prefix}_mirror": mirror,
        f"{prefix}_scale_x": None,
        f"{prefix}_scale_y": None,
        f"{prefix}_crop_left": None,
        f"{prefix}_crop_top": None,
        f"{prefix}_crop_right": None,
        f"{prefix}_crop_bottom": None,
        f"{prefix}_pad_left": 0,
        f"{prefix}_pad_top": 0,
        f"{prefix}_pad_right": 0,
        f"{prefix}_pad_bottom": 0,
    }


def _camera_plan_stats(
    cfg: AppConfig,
    canvas_size: Size,
    capture_health: Any,
) -> dict[str, object]:
    """Reconstruct the exact delivered-camera plan from scalar health state."""

    delivered_width = getattr(capture_health, "delivered_width", None)
    delivered_height = getattr(capture_health, "delivered_height", None)
    if not delivered_width or not delivered_height:
        return _empty_plan_stats(
            "camera",
            fit=cfg.camera.fit_mode,
            rotation=cfg.camera.rotation,
            mirror=cfg.camera.mirror,
        )
    try:
        plan = plan_transform(
            (int(delivered_width), int(delivered_height)),
            canvas_size,
            rotation=cfg.camera.rotation,
            mirror=cfg.camera.mirror,
            fit=cfg.camera.fit_mode,
            anchors=(cfg.camera.anchor_x, cfg.camera.anchor_y),
        )
    except (TypeError, ValueError):
        return _empty_plan_stats(
            "camera",
            fit=cfg.camera.fit_mode,
            rotation=cfg.camera.rotation,
            mirror=cfg.camera.mirror,
        )
    return _plan_stats("camera", plan)


def _camera_controls_stats(capture_health: Any) -> dict[str, object]:
    """Serialize the side-effect-free capture snapshot without native handles."""

    controls = getattr(capture_health, "camera_controls", None)
    serializer = getattr(controls, "as_dict", None)
    if callable(serializer):
        controls = serializer()
    if not isinstance(controls, dict):
        return {}
    return dict(controls)


def _background_plan_stats(resources: "_Resources") -> dict[str, object]:
    """Read only the plan that produced the current provider pixels."""

    cfg = resources.cfg
    provider = resources.backdrop
    effective_mode = (
        cfg.background.remote_fallback_mode
        if cfg.background.mode == "remote"
        else cfg.background.mode
    )
    if provider is None:
        return _empty_plan_stats(
            "background",
            fit=cfg.background.fit_mode,
        )
    getter = getattr(provider, "transform_plan", None)
    if not callable(getter):
        plan = None
    else:
        try:
            plan = getter(resources.canvas_size[0], resources.canvas_size[1])
        except (RuntimeError, TypeError, ValueError):
            plan = None
    if plan is None and effective_mode in {"blur", "color"}:
        plan = plan_transform(
            resources.canvas_size,
            resources.canvas_size,
            fit=cfg.background.fit_mode,
            anchors=(cfg.background.anchor_x, cfg.background.anchor_y),
        )
    if plan is None:
        return _empty_plan_stats("background", fit=cfg.background.fit_mode)
    values = _plan_stats("background", plan)
    # Qualified video orientation is applied before the fit plan. Include it
    # in the total source-to-canvas rotation instead of reporting a false zero.
    stats_getter = getattr(provider, "stats_dict", None)
    if callable(stats_getter):
        provider_stats = stats_getter()
        if isinstance(provider_stats, Mapping):
            orientation_status = provider_stats.get(
                "background_video_orientation_status"
            )
            metadata_rotation = provider_stats.get("background_video_metadata_rotation")
        else:
            orientation_status = metadata_rotation = None
        if (
            orientation_status == "qualified-manual-metadata"
            and type(metadata_rotation) is int
        ):
            values["background_rotation"] = metadata_rotation
    return values


def _effective_color_mode(transform: ColorTransform) -> str:
    exposure = abs(transform.exposure_ev) > 1e-9
    white_balance = any(abs(gain - 1.0) > 1e-9 for gain in transform.wb_gains)
    if exposure and white_balance:
        return "exposure-white-balance"
    if exposure:
        return "exposure"
    if white_balance:
        return "white-balance"
    return "identity"


def _color_stats(
    cfg: AppConfig,
    snapshot: HarmonizerSnapshot | None,
    *,
    applied_transform: ColorTransform | None = None,
    application_failed: bool = False,
    exposure_clamped: bool = False,
    white_balance_clamped: bool = False,
) -> dict[str, object]:
    """Map configured, temporal, and actually applied correction state."""

    configured_mode = cfg.compositing.color_correction.mode
    eligible = (
        configured_mode == "auto" and cfg.background.mode in _COLOR_ELIGIBLE_MODES
    )
    if configured_mode != "auto":
        state = "disabled"
        reason = "disabled"
        transform = IDENTITY_TRANSFORM
        effective_mode = "off"
    elif not eligible:
        state = "mode-excluded"
        reason = ColorReason.MODE_EXCLUDED.value
        transform = IDENTITY_TRANSFORM
        effective_mode = "bypass"
    else:
        transform = (
            applied_transform
            if applied_transform is not None
            else snapshot.transform
            if snapshot is not None
            else IDENTITY_TRANSFORM
        )
        if snapshot is None:
            state = "warming"
            reason = "initializing"
        else:
            state = {
                HarmonizerPhase.IDENTITY: "low-confidence",
                HarmonizerPhase.WARMING: "warming",
                HarmonizerPhase.ACTIVE: "active",
                HarmonizerPhase.FROZEN: "low-confidence",
                HarmonizerPhase.STALE_DECAY: "stale-decay",
                HarmonizerPhase.SCENE_CUT: "scene-cut",
            }[snapshot.phase]
            reason = snapshot.reason.value
        effective_mode = (
            "bypass" if application_failed else _effective_color_mode(transform)
        )
        if application_failed:
            transform = IDENTITY_TRANSFORM
            reason = "application-error"

    confidence = snapshot.confidence if snapshot is not None and eligible else 0.0
    gains = transform.wb_gains
    active = eligible and not application_failed and not transform.is_identity
    clamp_eligible = eligible and not application_failed
    return {
        "color_correction_mode": configured_mode,
        "color_correction_active": active,
        "color_correction_effective_mode": effective_mode,
        "color_correction_state": state,
        "color_correction_reason": reason,
        "color_correction_confidence": confidence,
        "color_correction_exposure_ev": transform.exposure_ev,
        "color_correction_wb_gain_r": gains[0],
        "color_correction_wb_gain_g": gains[1],
        "color_correction_wb_gain_b": gains[2],
        "color_correction_wb_active": any(abs(gain - 1.0) > 1e-9 for gain in gains),
        "color_correction_exposure_clamped": (
            bool(exposure_clamped) if clamp_eligible else False
        ),
        "color_correction_wb_clamped": (
            bool(white_balance_clamped) if clamp_eligible else False
        ),
        "color_correction_warming": state in {"warming", "scene-cut"},
        "color_correction_stale": state == "stale-decay",
        "color_input_assumption": "display-referred-srgb-bt709-full-range",
    }


def _build_backdrop(cfg: AppConfig) -> Any:
    kwargs: dict[str, Any] = {
        "image_max_pixels": cfg.api.uploads.image_max_pixels,
        "video_max_width": cfg.api.uploads.video_max_width,
        "video_max_height": cfg.api.uploads.video_max_height,
    }
    if _backdrop_provider_key(cfg)[0] == "camera":
        kwargs["camera_target"] = cfg.resolved_backdrop_target()
    return create_backdrop(cfg.background, **kwargs)


def _startup_backdrop(cfg: AppConfig) -> tuple[Any, str]:
    """Open the persisted backdrop without making its asset an API dependency.

    A missing or unreadable operator-owned image/video is recoverable at
    startup: the configured intent remains active and a later transactional
    PATCH can replace it.  Other provider failures (for example a camera
    device or invalid non-asset mode) retain the ordinary fail-fast contract.
    """

    effective_mode = (
        cfg.background.remote_fallback_mode
        if cfg.background.mode == "remote"
        else cfg.background.mode
    )
    try:
        return _build_backdrop(cfg), ""
    except (OSError, RuntimeError, ValueError):
        if effective_mode not in {"image", "video"}:
            raise
        return None, "asset-unavailable"


def _ewma(previous: float | None, sample: float, alpha: float = 0.1) -> float:
    return sample if previous is None else previous + alpha * (sample - previous)


def _send_output_with_timing(
    output: Any,
    frame: np.ndarray,
    *,
    copy_frame: bool = False,
) -> OutputSendTiming:
    """Use the typed sink boundary while retaining duck-typed test outputs."""

    started_at_ns = time.monotonic_ns()
    payload = frame.copy() if copy_frame else frame
    copy_completed_at_ns = time.monotonic_ns()
    sender = getattr(output, "send_with_timing", None)
    if callable(sender):
        timing = sender(payload)
        if isinstance(timing, OutputSendTiming):
            if not (
                copy_completed_at_ns <= timing.submitted_at_ns <= timing.completed_at_ns
            ):
                raise RuntimeError("output sink returned non-monotonic timing")
            return OutputSendTiming(
                submitted_at_ns=timing.submitted_at_ns,
                completed_at_ns=timing.completed_at_ns,
                submission_ms=(
                    (copy_completed_at_ns - started_at_ns) / 1_000_000.0
                    + timing.submission_ms
                ),
                pacing_wait_ms=timing.pacing_wait_ms,
                pacing_events=timing.pacing_events,
                recovery_events=timing.recovery_events,
            )
        # A legacy/mock output may expose an untyped dynamic attribute. Its
        # call already performed the send, so derive the conservative boundary
        # without sending a second time.
    else:
        output.send(payload)
    completed_at_ns = max(started_at_ns, time.monotonic_ns())
    return OutputSendTiming(
        submitted_at_ns=completed_at_ns,
        completed_at_ns=completed_at_ns,
        submission_ms=(completed_at_ns - started_at_ns) / 1_000_000.0,
        pacing_wait_ms=0.0,
    )


def _timing_fields(
    stage_ewma: Mapping[str, float | None],
    *,
    capture_health: Any,
    segmenter: Any,
) -> dict[str, float | None]:
    """Project fixed version-1 timer names from the current run-owned EWMAs."""

    rvm = _rvm_telemetry_evidence(segmenter)

    def optional_duration(value: object) -> float | None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            return None
        return min(float(value), 3_600_000.0)

    values: dict[str, float | None] = {
        "capture.read": optional_duration(getattr(capture_health, "read_ms", None)),
        "segmentation.total": stage_ewma.get("segmentation_ms"),
        "segmentation.preprocess": optional_duration(
            rvm.get("preprocess_ms") if rvm.get("applicable") is True else None
        ),
        "segmentation.inference": optional_duration(
            rvm.get("session_run_ms") if rvm.get("applicable") is True else None
        ),
        "segmentation.postprocess": optional_duration(
            rvm.get("postprocess_ms") if rvm.get("applicable") is True else None
        ),
        "background.total": stage_ewma.get("background_ms"),
        "color_correction.total": stage_ewma.get("color_correction_ms"),
        "compositor.total": stage_ewma.get("composite_ms"),
        "compositor.prepare": stage_ewma.get("composite_prepare_ms"),
        "compositor.blend": stage_ewma.get("composite_blend_ms"),
        "output.send_total": stage_ewma.get("output_send_ms"),
        "output.submission": stage_ewma.get("output_submission_ms"),
        "output.sink_pacing_wait": stage_ewma.get("output_sink_pacing_wait_ms"),
        "output.application_pacing_wait": stage_ewma.get("application_pacing_wait_ms"),
        "output.schedule_lateness": stage_ewma.get("output_schedule_lateness_ms"),
        "pipeline.processing_only": stage_ewma.get("frame_processing_ms"),
        "pipeline.new_frame_service": stage_ewma.get("new_frame_service_ms"),
        "pipeline.new_frame_serialized_loop": stage_ewma.get(
            "new_frame_serialized_loop_ms"
        ),
    }
    if set(values) != set(TIMING_FIELD_NAMES):  # pragma: no cover - schema invariant
        raise RuntimeError("pipeline timing projection drifted from hub schema")
    return values


def _cadence_status(
    tracker: CadenceTracker,
    *,
    now_ns: int,
    target_output_fps: int,
) -> dict[str, object]:
    """Project tracker truth plus compatibility aliases into public status."""

    snapshot = tracker.snapshot(now_ns=now_ns)
    values = snapshot.as_dict()
    # The public contract names successful capture-derived output as a base
    # update. These tracker-only aliases are useful in unit tests but would
    # duplicate and blur that stable vocabulary in /status.
    values.pop("unique_capture_count")
    values.pop("unique_capture_fps")
    send_count = snapshot.output_send_count
    base_count = snapshot.base_composite_update_count
    send_fps = snapshot.output_send_fps
    values.update(
        {
            "frames_in": base_count,
            "frames_out": send_count,
            "fps": send_fps,
            "output_effective_fps": send_fps,
            "fps_attainment_pct": (
                min(100.0, send_fps / target_output_fps * 100.0)
                if send_count >= 2
                else None
            ),
            "output_repeated_frames": snapshot.base_composite_reuse_count,
        }
    )
    return values


def _capture_health_stats(
    capture_health: Any,
    *,
    fallback_frames_read: int,
) -> dict[str, object]:
    return {
        "capture_fps": getattr(capture_health, "capture_fps", 0.0),
        "capture_target_met": getattr(capture_health, "target_met", None),
        "capture_frames_read": getattr(
            capture_health,
            "frames_read",
            fallback_frames_read,
        ),
        "capture_dropped_frames": getattr(capture_health, "dropped_frames", 0),
        "capture_read_failures": getattr(capture_health, "read_failures", 0),
        "capture_restarts": getattr(capture_health, "restarts", 0),
        "capture_stalled": getattr(capture_health, "stalled", False),
        "capture_frame_age_ms": getattr(capture_health, "frame_age_ms", None),
        "capture_read_ms": getattr(capture_health, "read_ms", None),
    }


class RestartRequiredError(RuntimeError):
    """The patch is valid but changes resources that cannot be swapped live."""

    status_code = 409

    def __init__(self, fields: list[str] | tuple[str, ...], current_version: int):
        self.fields = tuple(sorted(fields))
        self.current_version = current_version
        super().__init__(
            "restart required for configuration fields: " + ", ".join(self.fields)
        )


class ActivationError(RuntimeError):
    """A validated hot configuration could not activate its resources."""

    status_code = 422


class ReconfigurationUnavailable(RuntimeError):
    """The pipeline cannot currently acknowledge a reconfiguration request."""

    status_code = 503


class ConfigConflictError(RuntimeError):
    """The patch was based on a configuration version that is no longer current."""

    status_code = 409

    def __init__(self, expected_version: int, current_version: int):
        self.expected_version = expected_version
        self.current_version = current_version
        super().__init__(
            f"configuration changed concurrently: expected version {expected_version}, "
            f"current version is {current_version}"
        )


class _PrivacyViolation(ValueError):
    """A frame or mask violated the remote-output privacy boundary."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class _RawFingerprint:
    jpeg_features: tuple[int, ...]
    thumbnail: np.ndarray


class _RawReplayHistory:
    """Compact, bounded Bloom-style index with no permissive eviction."""

    _MAX_BYTES = 16 * 1024 * 1024
    _MIN_BYTES = 4 * 1024
    _JPEG_QUANTIZATION_WIDTH = 64
    _JPEG_QUANTIZATION_STEP = 4
    _MATCHING_SIGNATURES = 8

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._byte_count = min(
            self._MAX_BYTES,
            max(self._MIN_BYTES, capacity * 64),
        )
        self._bits: bytearray | None = None
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def clear(self) -> None:
        self._bits = None
        self._count = 0

    @staticmethod
    def _keys(fingerprint: _RawFingerprint):
        # JPEG preserves low-frequency block/DC values even when edge-based
        # perceptual hashes flip near equal comparisons.  Hash the complete
        # coarse feature vector under sixteen overlapping quantizers.  A small
        # JPEG drift crosses only a subset of their staggered boundaries, so
        # multiple whole-frame keys remain identical without retaining a
        # reversible historical thumbnail.
        width = _RawReplayHistory._JPEG_QUANTIZATION_WIDTH
        step = _RawReplayHistory._JPEG_QUANTIZATION_STEP
        for phase_index, offset in enumerate(range(0, width, step)):
            key = phase_index
            for value in fingerprint.jpeg_features:
                quantized = min(3, (value + offset) // width)
                key = (key << 2) | quantized
            yield key

    @staticmethod
    def _positions(key: int, bit_count: int):
        mask = (1 << 64) - 1
        for salt in (
            0x9E3779B97F4A7C15,
            0xD1B54A32D192ED03,
            0x94D049BB133111EB,
        ):
            value = (key + salt) & mask
            value ^= value >> 30
            value = (value * 0xBF58476D1CE4E5B9) & mask
            value ^= value >> 27
            value = (value * 0x94D049BB133111EB) & mask
            value ^= value >> 31
            yield value % bit_count

    def append(self, fingerprint: _RawFingerprint) -> None:
        if self._bits is None:
            self._bits = bytearray(self._byte_count)
        bit_count = len(self._bits) * 8
        for key in self._keys(fingerprint):
            for position in self._positions(key, bit_count):
                self._bits[position >> 3] |= 1 << (position & 7)
        self._count += 1

    def matches(self, fingerprint: _RawFingerprint) -> bool:
        if self._bits is None:
            return False
        bit_count = len(self._bits) * 8
        matches = 0
        for key in self._keys(fingerprint):
            if all(
                self._bits[position >> 3] & (1 << (position & 7))
                for position in self._positions(key, bit_count)
            ):
                matches += 1
                if matches >= self._MATCHING_SIGNATURES:
                    return True
        return False


def _safe_close(resource: Any, label: str) -> None:
    if resource is None:
        return
    try:
        resource.close()
    except Exception:
        log.exception("cannot close %s", label)


@dataclass(frozen=True)
class _CaptureSequenceSnapshot:
    """Content-free identity of unique captures accepted by the pipeline."""

    last_sequence: int | None
    gap_events: int
    missing_inputs: int


class _CaptureSequenceTimeline:
    """Count source-sequence gaps independently of segmentation decisions."""

    def __init__(self) -> None:
        self._last_sequence: int | None = None
        self._gap_events = 0
        self._missing_inputs = 0

    def observe(self, sequence: int) -> None:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("capture sequence must be a non-negative integer")
        previous = self._last_sequence
        if previous is not None:
            if sequence <= previous:
                raise ValueError(
                    "capture sequence must increase for every accepted input"
                )
            missing = sequence - previous - 1
            if missing:
                self._gap_events += 1
                self._missing_inputs += missing
        self._last_sequence = sequence

    def snapshot(self) -> _CaptureSequenceSnapshot:
        return _CaptureSequenceSnapshot(
            last_sequence=self._last_sequence,
            gap_events=self._gap_events,
            missing_inputs=self._missing_inputs,
        )


@dataclass(frozen=True)
class _TemporalStateOwner:
    """One generation-owned segmenter/refiner pair and its exact policy."""

    policy: tuple[object, ...]
    generation: int | None
    segmenter: Any = field(repr=False)
    refiner: Any = field(repr=False)

    def matches(
        self,
        *,
        policy: tuple[object, ...],
        generation: int | None,
        segmenter: Any,
        refiner: Any,
    ) -> bool:
        """Return whether policy, generation, and both stateful objects agree."""

        return (
            self.policy == policy
            and self.generation == generation
            and self.segmenter is segmenter
            and self.refiner is refiner
        )


@dataclass
class _Resources:
    cfg: AppConfig
    version: int
    capture: Any
    segmenter: Any
    refiner: Any
    backdrop: Any
    output: Any
    output_factory: Callable[[], Any] | None = None
    output_owned_by_publisher: bool = False
    background_fallback_reason: str = ""
    background_startup_asset_guard: bool = False
    visual_generation: int = 0
    segmentation_generation: int = 0
    light_wrap_generation: int = 0
    harmonizer: ColorHarmonizer | None = None
    light_wrap_stabilizer: LightWrapStabilizer | None = None
    color_reset_token: tuple[object, ...] | None = None
    background_geometry_token: tuple[object, ...] | None = None
    background_geometry_transitions: int = 0
    color_correction_applied_frames: int = 0
    color_correction_bypassed_frames: int = 0
    color_correction_scene_cuts: int = 0
    color_correction_transitions: int = 0
    color_correction_reason_transitions: int = 0
    color_correction_exposure_clamped: bool = False
    color_correction_exposure_clamp_count: int = 0
    color_correction_exposure_clamp_time_s: float = 0.0
    color_correction_wb_clamped: bool = False
    color_correction_wb_clamp_count: int = 0
    color_correction_wb_clamp_time_s: float = 0.0
    _color_correction_last_unique_at_s: float | None = field(
        default=None,
        repr=False,
    )
    background_video_lifetime: _BackgroundVideoLifetimeCounters = field(
        default_factory=_BackgroundVideoLifetimeCounters,
        repr=False,
    )
    last_compositor_substages_ms: dict[str, float] = field(default_factory=dict)
    canvas_size: Size = field(init=False)
    capture_sequence_timeline: _CaptureSequenceTimeline = field(
        init=False,
        repr=False,
    )
    segmentation_timeline: SegmentationTimeline = field(init=False, repr=False)
    temporal_state_owner: _TemporalStateOwner = field(init=False, repr=False)
    color_analysis_executor: ThreadPoolExecutor = field(init=False, repr=False)
    color_backdrop_analysis_token: tuple[object, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    color_backdrop_analysis_linear_bgr: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )
    static_light_wrap_token: tuple[object, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    static_light_wrap: PreparedLightWrap | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _legacy_compositor_workspace: LegacyCompositorWorkspace | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _capture_closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.canvas_size = resolved_output_size(self.cfg)
        self.capture_sequence_timeline = _CaptureSequenceTimeline()
        self.segmentation_timeline = SegmentationTimeline()
        self.temporal_state_owner = _TemporalStateOwner(
            policy=_segmenter_key(self.cfg),
            generation=self.segmentation_generation,
            segmenter=self.segmenter,
            refiner=self.refiner,
        )
        # The three independent linear-light analysis resizes release the GIL.
        # A resource-owned pool runs them concurrently and is closed with the
        # pipeline generation, avoiding process-global worker lifetime.
        self.color_analysis_executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="custback-color-analysis",
        )
        if self.harmonizer is None:
            self.harmonizer = _new_color_harmonizer(self.cfg)
        if self.light_wrap_stabilizer is None:
            self.light_wrap_stabilizer = _new_light_wrap_stabilizer(self.cfg)

    @property
    def canvas_shape(self) -> tuple[int, int, int]:
        return (self.canvas_size[1], self.canvas_size[0], 3)

    def image_backdrop_analysis(
        self,
        backdrop_frame: np.ndarray,
        backdrop_linear_bgr: np.ndarray,
    ) -> np.ndarray | None:
        """Return one generation-scoped bounded raster for immutable images."""

        provider = self.backdrop
        if (
            not isinstance(provider, ImageBackdrop)
            or getattr(provider, "_cache", None) is not backdrop_frame
        ):
            return None
        token = (
            self.visual_generation,
            id(provider),
            getattr(provider, "_cache_key", None),
            id(backdrop_frame),
            backdrop_frame.shape,
        )
        if (
            self.color_backdrop_analysis_token != token
            or self.color_backdrop_analysis_linear_bgr is None
        ):
            analysis = _linear_bgr_analysis_raster_prevalidated(backdrop_linear_bgr)
            if max(analysis.shape[:2]) > ANALYSIS_LONG_EDGE:
                raise ColorError("cached backdrop analysis exceeds its bound")
            self.color_backdrop_analysis_token = token
            self.color_backdrop_analysis_linear_bgr = analysis
        return self.color_backdrop_analysis_linear_bgr

    def invalidate_color_backdrop_analysis(self) -> None:
        self.color_backdrop_analysis_token = None
        self.color_backdrop_analysis_linear_bgr = None

    def static_light_wrap_sample(
        self,
        backdrop_frame: np.ndarray,
        *,
        blend_space: BlendSpace,
        backdrop_linear_bgr: np.ndarray | None,
        diagnostics: dict[str, float],
    ) -> PreparedLightWrap | None:
        """Return a generation-owned sample only for immutable providers."""

        provider = self.backdrop
        if not isinstance(provider, (ColorBackdrop, ImageBackdrop)):
            return None
        if getattr(provider, "_cache", None) is not backdrop_frame:
            return None
        token = (
            self.visual_generation,
            id(provider),
            provider.geometry,
            getattr(provider, "_cache_key", None),
            getattr(provider, "color", None),
            id(backdrop_frame),
            backdrop_frame.shape,
            blend_space,
        )
        if self.static_light_wrap_token != token or self.static_light_wrap is None:
            self.static_light_wrap = prepare_static_light_wrap(
                backdrop_frame,
                blend_space=blend_space,
                backdrop_linear_bgr=(
                    backdrop_linear_bgr if blend_space == "linear_srgb" else None
                ),
                diagnostics=diagnostics,
            )
            self.static_light_wrap_token = token
        return self.static_light_wrap

    def invalidate_static_light_wrap(self) -> None:
        self.static_light_wrap_token = None
        self.static_light_wrap = None

    def retire_background_video_provider(self, provider: object) -> None:
        """Preserve final video counters before a provider is closed."""

        self.background_video_lifetime.retire(provider)

    def background_video_lifetime_stats(
        self,
        current_stats: Mapping[str, object],
    ) -> dict[str, int]:
        """Project cumulative counters including the active provider."""

        return self.background_video_lifetime.project(current_stats)

    def legacy_compositor_workspace(self) -> LegacyCompositorWorkspace:
        """Return the lazily allocated generation-owned legacy work buffers."""

        workspace = self._legacy_compositor_workspace
        if workspace is None:
            workspace = LegacyCompositorWorkspace(self.canvas_shape)
            self._legacy_compositor_workspace = workspace
        return workspace

    def close_capture(self) -> None:
        """Stop capture once so terminal health can be sampled without a race."""

        if self._capture_closed:
            return
        self._capture_closed = True
        _safe_close(self.capture, "capture")

    def close(self) -> None:
        # Close independently so one faulty backend cannot strand the others.
        # Capture is stopped first so no further latest-slot overwrite can race
        # teardown or the final operator-health snapshot.
        self.close_capture()
        self.invalidate_color_backdrop_analysis()
        self.invalidate_static_light_wrap()
        _safe_close(self._legacy_compositor_workspace, "legacy compositor workspace")
        self._legacy_compositor_workspace = None
        try:
            self.color_analysis_executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            log.exception("cannot close color analysis workers")
        if not self.output_owned_by_publisher:
            _safe_close(self.output, "video output")
        _safe_close(self.backdrop, "backdrop")
        _safe_close(self.light_wrap_stabilizer, "light-wrap stabilizer")
        _safe_close(self.refiner, "mask refiner")
        _safe_close(self.segmenter, "segmenter")


@dataclass
class _Activation:
    candidate: AppConfig
    replace_segmenter: bool = False
    segmenter: Any = None
    refiner: Any = None
    temporal_state_owner: _TemporalStateOwner | None = field(
        default=None,
        repr=False,
    )
    temporal_trial_context: SegmentationFrameContext | None = None
    temporal_trial_dirty: bool = False
    replace_backdrop: bool = False
    backdrop: Any = None
    visual_state_changed: bool = False
    replace_harmonizer: bool = False
    reconfigure_harmonizer: bool = False
    harmonizer: ColorHarmonizer | None = None
    replace_light_wrap_stabilizer: bool = False
    light_wrap_stabilizer: LightWrapStabilizer | None = None
    promoted: bool = False

    def take_segmenter(self) -> Any:
        """Detach and return a staged segmenter still owned by this activation."""

        if self.promoted or not self.replace_segmenter:
            return None
        segmenter = self.segmenter
        self.replace_segmenter = False
        self.segmenter = None
        self.refiner = None
        self.temporal_state_owner = None
        self.temporal_trial_context = None
        self.temporal_trial_dirty = False
        return segmenter

    def take_backdrop(self) -> Any:
        """Detach and return a staged backdrop still owned by this activation."""

        if self.promoted or not self.replace_backdrop:
            return None
        backdrop = self.backdrop
        self.replace_backdrop = False
        self.backdrop = None
        return backdrop

    def mark_promoted(self) -> None:
        """Transfer every staged object to live resources exactly once."""

        if self.promoted:
            return
        self.promoted = True
        self.replace_segmenter = False
        self.segmenter = None
        self.refiner = None
        self.temporal_state_owner = None
        self.temporal_trial_context = None
        self.temporal_trial_dirty = False
        self.replace_backdrop = False
        self.backdrop = None
        self.replace_harmonizer = False
        self.reconfigure_harmonizer = False
        self.harmonizer = None
        self.replace_light_wrap_stabilizer = False
        self.light_wrap_stabilizer = None

    def discard(self) -> None:
        _safe_close(self.take_backdrop(), "staged backdrop")
        if self.replace_light_wrap_stabilizer:
            _safe_close(
                self.light_wrap_stabilizer,
                "staged light-wrap stabilizer",
            )
            self.replace_light_wrap_stabilizer = False
            self.light_wrap_stabilizer = None
        refiner = self.refiner
        segmenter = self.take_segmenter()
        _safe_close(refiner, "staged mask refiner")
        _safe_close(segmenter, "staged segmenter")


@dataclass(frozen=True)
class _PreparedColorFrame:
    """Frame-local decoded inputs and one bounded foreground transform."""

    transform: ColorTransform = IDENTITY_TRANSFORM
    foreground_linear_bgr: np.ndarray | None = None
    backdrop_linear_bgr: np.ndarray | None = None
    edge_foreground_linear_bgr: np.ndarray | None = None
    snapshot: HarmonizerSnapshot | None = None
    exposure_clamped: bool = False
    white_balance_clamped: bool = False


@dataclass(frozen=True)
class _PreflightResult:
    """Already-sent startup output and the capture it consumed."""

    output: np.ndarray | None
    captured: CapturedFrame
    send_timing: OutputSendTiming
    base_ready_at_ns: int
    segmentation_updated: bool
    frame_processing_ms: float
    new_frame_service_ms: float
    new_frame_serialized_loop_ms: float
    timings: dict[str, float]
    color_status: dict[str, object]
    remote_fallback_active: bool
    remote_fallback_reason: str
    matte_evidence: MatteFrameEvidence | None = field(
        default=None,
        repr=False,
    )

    @property
    def capture_sequence(self) -> int:
        return self.captured.sequence


@dataclass
class _PublisherBridge:
    """Bounded scalar/evidence bridge between compute and publication lanes."""

    cadence: CadenceTracker
    performance: RuntimePerformanceTracker
    publisher_mode: str
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    last_status: dict[str, object] = field(default_factory=dict, repr=False)
    # Current run/config identity authored only by the processing lane. A
    # paced publisher may still repeat pixels from an older immutable
    # envelope after a hot commit; this overlay keeps public policy and
    # selection fields current while ``output_base_config_version`` preserves
    # the pixels' producer version.
    committed_status: dict[str, object] = field(default_factory=dict, repr=False)
    pending_evidence: dict[int, MatteFrameEvidence] = field(
        default_factory=dict,
        repr=False,
    )
    retained_base_epochs: dict[int, PerformanceEpochKey] = field(
        default_factory=dict,
        repr=False,
    )
    last_matte_source_sequence: int | None = None
    processing_completed_count: int = 0
    output_base_config_version: int = 0
    resolved_remote_raw_epoch: int = 0
    performance_state: str = "warming"
    output_ewma: dict[str, float | None] = field(
        default_factory=lambda: {
            "output_send_ms": None,
            "output_submission_ms": None,
            "output_sink_pacing_wait_ms": None,
            "application_pacing_wait_ms": None,
            "output_schedule_lateness_ms": None,
            "new_frame_service_ms": None,
        },
        repr=False,
    )

    def retain_base(
        self,
        base_id: int,
        epoch_key: PerformanceEpochKey,
        evidence: MatteFrameEvidence | None,
    ) -> None:
        with self.lock:
            if base_id in self.retained_base_epochs:
                raise RuntimeError("publisher base id was retained twice")
            if not self.performance.retain_epoch(epoch_key):
                raise RuntimeError("publisher base references a finalized epoch")
            self.retained_base_epochs[base_id] = epoch_key
            if evidence is not None:
                self.pending_evidence[base_id] = evidence

    def discard_evidence(self, base_id: int) -> None:
        with self.lock:
            self.pending_evidence.pop(base_id, None)

    def release_base(self, base_id: int) -> None:
        """Release one envelope after it leaves publisher-owned retention."""

        with self.lock:
            self.pending_evidence.pop(base_id, None)
            epoch_key = self.retained_base_epochs.pop(base_id, None)
        if epoch_key is None:
            raise RuntimeError("publisher released an unknown base id")
        self.performance.release_epoch(epoch_key)
        _log_finalized_performance_epochs(self.performance)


_COMMITTED_OUTPUT_STATUS_NAMES = (
    "config_version",
    "mode",
    "matte_rollout",
    "matte_policy",
    "segmentation_selection",
    "color_correction_mode",
    "effective_rvm_downsample_ratio",
    "effective_mask_blur",
    "effective_edge_refine",
    "effective_edge_refinement_mode",
    "effective_edge_refinement_radius_px",
    "effective_mask_shift",
    "effective_temporal_smoothing",
    "effective_boundary_stabilization_mode",
    "effective_boundary_stabilization_time_constant_s",
    "effective_boundary_stabilization_max_motion_px_per_s",
    "effective_use_model_foreground",
    "effective_light_wrap",
    "output_target_fps",
    "capture_target_fps",
    "remote_fallback_mode",
)


def _committed_output_status(identity: Mapping[str, object]) -> dict[str, object]:
    """Select config/policy truth without relabelling old pixels.

    Capture/provider/color-estimator counters and generations remain frozen in
    each ``SafeBaseFrame``. Only fields whose meaning changes at the atomic
    config commit are overlaid while that older base is repeated.
    """

    missing = [name for name in _COMMITTED_OUTPUT_STATUS_NAMES if name not in identity]
    if missing:  # pragma: no cover - internal status-schema invariant
        raise RuntimeError("committed output status is incomplete")
    return {name: identity[name] for name in _COMMITTED_OUTPUT_STATUS_NAMES}


def _log_finalized_performance_epochs(
    performance: RuntimePerformanceTracker,
) -> None:
    """Emit one bounded, path-free record after an epoch can no longer mutate."""

    for closed in performance.take_finalized_epoch_summaries():
        raw_key = closed.get("key")
        key = raw_key if isinstance(raw_key, Mapping) else {}
        state = str(closed.get("state", "warming"))
        output_attainment = cast(float, closed.get("output_attainment", 0.0))
        unique_attainment = cast(float, closed.get("unique_attainment", 0.0))
        if state in {"warming", "failed"}:
            cadence_status = state
        elif (
            output_attainment < RUNTIME_PERFORMANCE_MIN_ATTAINMENT
            or unique_attainment < RUNTIME_PERFORMANCE_MIN_ATTAINMENT
        ):
            cadence_status = "unexpected-shortfall"
        elif performance.unique_target_fps < performance.transport_target_fps:
            cadence_status = "intentional-repeat"
        else:
            cadence_status = "matched"
        log.info(
            "runtime performance epoch finalized config_version=%s "
            "capture_generation=%s segmentation_generation=%s "
            "backdrop_generation=%s state=%s reason=%s cadence_status=%s "
            "transport_target_fps=%.3f unique_target_fps=%.3f "
            "output_attainment=%.6f unique_attainment=%.6f "
            "deadline_miss_ratio=%.6f late_send_ratio=%.6f",
            key.get("config_version", 0),
            key.get("capture_generation", 0),
            key.get("segmentation_generation", 0),
            key.get("backdrop_generation", 0),
            state,
            closed.get("reason", "none"),
            cadence_status,
            performance.transport_target_fps,
            performance.unique_target_fps,
            output_attainment,
            unique_attainment,
            cast(float, closed.get("processing_deadline_miss_ratio", 0.0)),
            cast(float, closed.get("output_schedule_late_ratio", 0.0)),
        )


@dataclass
class _PatchRequest:
    candidate: AppConfig
    expected_version: int
    origin: str = "internal"
    activation_candidate: AppConfig | None = None
    before_activate: Callable[[], None] | None = None
    rollback_activate: Callable[[], None] | None = None
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: bool = False
    result: ConfigState | None = None
    error: BaseException | None = None
    storage_epoch: int = 0
    prepared_activation: _Activation | None = None
    rollout_attempt: MatteRolloutAttempt | None = None

    def cancel_and_take_activation(self) -> tuple[bool, _Activation | None]:
        """Cancel and reclaim an activation not yet claimed by the frame lane."""

        with self.lock:
            if self.result is not None or self.error is not None:
                return False, None
            self.cancelled = True
            activation = self.prepared_activation
            self.prepared_activation = None
            return True, activation

    def fail(self, error: BaseException) -> None:
        with self.lock:
            if self.result is None and self.error is None:
                self.error = error
                if self.rollout_attempt is not None:
                    self.rollout_attempt.fail()
        self.done.set()


@dataclass
class _MutationRequest:
    """Serialized side effect tied to the effective config generation."""

    mutate: Callable[[AppConfig], None]
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: bool = False
    result: ConfigState | None = None
    error: BaseException | None = None

    def cancel(self) -> bool:
        with self.lock:
            if self.result is not None or self.error is not None:
                return False
            self.cancelled = True
            return True

    def fail(self, error: BaseException) -> None:
        with self.lock:
            if self.result is None and self.error is None:
                self.error = error
        self.done.set()


def _changed_paths(old: Any, new: Any, prefix: str = "") -> list[str]:
    return changed_config_paths(old, new, prefix)


def _restart_only_changes(old: AppConfig, new: AppConfig) -> list[str]:
    return restart_only_config_paths(old, new)


class Pipeline:
    def __init__(
        self,
        runtime: RuntimeConfig,
        hub: FrameHub,
        *,
        model_preparation: SegmenterPreparation | None = None,
        raw_fingerprint_capacity: int = _RAW_FINGERPRINT_HISTORY,
        matte_recorder: MatteDiagnosticRecorder | None = None,
        matte_monitor: LocalMatteDiagnosticMonitor | None = None,
    ):
        if (
            not isinstance(raw_fingerprint_capacity, int)
            or isinstance(raw_fingerprint_capacity, bool)
            or raw_fingerprint_capacity < 1
        ):
            raise ValueError("raw fingerprint capacity must be a positive integer")
        self.runtime = runtime
        self.hub = hub
        self._stop = threading.Event()
        self._startup_done = threading.Event()
        self._thread: threading.Thread | None = None
        self._output_publisher: OutputPublisher | None = None
        self._runtime_performance: RuntimePerformanceTracker | None = None
        self._publisher_policy = PublicationPolicy(0, "local")
        self._publisher_policy_epoch = 0
        self._publisher_raw_epoch = 0
        self._publisher_policy_lock = threading.RLock()
        self._error: BaseException | None = None
        self._requests: queue.Queue[_PatchRequest | _MutationRequest] = queue.Queue()
        self._request_enqueue_lock = threading.Lock()
        # A preparation stays registered until its result has either moved to
        # request ownership or all abandoned-result cleanup has finished.  The
        # condition lets stop() enforce its deadline without calling an
        # unbounded executor join while a backend's close() is still running in
        # a Future callback.  RLock is intentional: cancelling queued futures
        # during executor.shutdown() can invoke their callbacks synchronously.
        self._preparation_lock = threading.RLock()
        self._preparation_condition = threading.Condition(self._preparation_lock)
        self._preparation_executor: ThreadPoolExecutor | None = None
        self._preparation_futures: set[Future[_Activation]] = set()
        self._storage_epoch_lock = threading.Lock()
        self._storage_epoch = 0
        self._lifecycle_lock = threading.Lock()
        self._active_state: ConfigState | None = None
        self._teardown_lock = threading.Lock()
        self._teardown_threads: set[threading.Thread] = set()
        self._deferred_closes: list[tuple[Any, str]] = []
        self._runtime_writer = runtime._coordinator_writer()
        self._model_preparation = model_preparation
        self._matte_recorder = matte_recorder
        self._matte_monitor = matte_monitor
        self._matte_bundle_sequence = 0
        self._matte_last_source_sequence: int | None = None
        self._matte_rollout = MatteRolloutTelemetry()
        initial_state = self.runtime.read()
        update_hub_stats = getattr(self.hub, "update_stats", None)
        if callable(update_hub_stats):
            update_hub_stats(
                config_version=initial_state.version,
                matte_rollout=self._matte_rollout.snapshot(
                    initial_state.config,
                    initial_state.version,
                ),
            )
        self._fallback_log_states: dict[str, tuple[bool, str]] = {}
        self._geometry_log_states: dict[str, tuple[object, ...]] = {}
        self._color_phase_log_state: tuple[object, ...] | None = None
        self._color_reason_log_state: str | None = None
        self._color_reason_log_candidate: tuple[str, float] | None = None
        self._raw_fingerprint_capacity = raw_fingerprint_capacity
        self._recent_raw_fingerprints = _RawReplayHistory(raw_fingerprint_capacity)
        self._privacy_history_exhausted = False
        self._privacy_invalidated_session: int | None = None
        self._latest_raw_frame: np.ndarray | None = None
        lifecycle_listener = getattr(
            self.hub,
            "set_remote_lifecycle_listener",
            None,
        )
        if callable(lifecycle_listener):
            lifecycle_listener(self._remote_lifecycle_changed)

    def start(self, timeout: float | None = None) -> None:
        """Start and synchronously acknowledge resource activation."""
        effective_timeout = (
            max(30.0, self.runtime.snapshot().camera.recovery_timeout_s + 5.0)
            if timeout is None
            else timeout
        )
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                raise ReconfigurationUnavailable("pipeline is already running")
            self._stop.clear()
            self._startup_done.clear()
            self._error = None
            self._output_publisher = None
            self._runtime_performance = None
            self._publisher_policy = PublicationPolicy(0, "local")
            self._publisher_policy_epoch = 0
            self._publisher_raw_epoch = 0
            self._active_state = None
            self._fallback_log_states.clear()
            self._geometry_log_states.clear()
            self._color_phase_log_state = None
            self._color_reason_log_state = None
            self._color_reason_log_candidate = None
            self._matte_bundle_sequence = 0
            self._matte_last_source_sequence = None
            with self._preparation_lock:
                if self._preparation_executor is not None:
                    raise ReconfigurationUnavailable(
                        "pipeline candidate preparation is still shutting down"
                    )
                self._preparation_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="custback-segmentation-prepare",
                )
            self._thread = threading.Thread(
                target=self._run, name="pipeline", daemon=True
            )
            self._thread.start()
        if not self._startup_done.wait(effective_timeout):
            self._stop.set()
            thread = self._thread
            if thread is not None:
                thread.join(timeout=max(0.1, min(5.0, effective_timeout)))
            if thread is not None and thread.is_alive():
                raise ReconfigurationUnavailable(
                    "pipeline startup timed out; worker is still running after "
                    "shutdown was requested"
                )
            if self._error is not None:
                raise ActivationError(str(self._error)) from self._error
            raise ReconfigurationUnavailable(
                "pipeline startup timed out; worker stopped during shutdown"
            )
        if self._error is not None:
            thread = self._thread
            if thread is not None:
                thread.join(timeout=1.0)
                if thread.is_alive():
                    raise ReconfigurationUnavailable(
                        "pipeline startup failed and its worker survived teardown"
                    ) from self._error
            raise ActivationError(str(self._error)) from self._error

    def stop(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        self._stop.set()
        publisher = self._output_publisher
        if publisher is not None:
            publisher.request_stop()
        with self._request_enqueue_lock:
            self._fail_pending(ReconfigurationUnavailable("pipeline is stopping"))
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
        frame_survived = thread is not None and thread.is_alive()
        with self._teardown_lock:
            teardown_threads = tuple(self._teardown_threads)
        for teardown in teardown_threads:
            teardown.join(max(0.0, deadline - time.monotonic()))
        with self._teardown_lock:
            survivors = [
                worker for worker in self._teardown_threads if worker.is_alive()
            ]
        preparation_error: ReconfigurationUnavailable | None = None
        try:
            # Always initiate executor shutdown, even if another worker has
            # survived its deadline. A later stop() can finish joining any
            # model constructor that could not be interrupted on this pass.
            self._shutdown_preparation_executor(deadline, timeout)
        except ReconfigurationUnavailable as exc:
            preparation_error = exc
        if frame_survived:
            raise ReconfigurationUnavailable(
                f"pipeline worker did not stop within {timeout:.1f}s"
            )
        if survivors:
            raise ReconfigurationUnavailable(
                f"{len(survivors)} resource teardown worker(s) did not stop "
                f"within {timeout:.1f}s"
            )
        if preparation_error is not None:
            raise preparation_error
        if self._error is not None:
            raise self._error

    @property
    def error(self) -> BaseException | None:
        """Return the authoritative worker failure recorded by the pipeline."""

        return self._error

    @property
    def running(self) -> bool:
        publisher = self._output_publisher
        return bool(
            self._error is None
            and self._thread is not None
            and self._thread.is_alive()
            and publisher is not None
            and publisher.snapshot().state in {"starting", "running"}
        )

    def apply_config_patch(
        self,
        patch: dict[str, Any],
        timeout: float = 5.0,
        *,
        origin: str = "internal",
        expected_version: int | None = None,
    ) -> ConfigState:
        """Apply one patch and record only sanitized matte rollout outcomes."""

        rollout_kind = classify_matte_patch(patch)
        if rollout_kind is None:
            if expected_version is None:
                return self._apply_config_patch(patch, timeout, origin=origin)
            return self._apply_config_patch(
                patch,
                timeout,
                origin=origin,
                expected_version=expected_version,
            )
        rollout_attempt = self._matte_rollout.begin(rollout_kind)
        try:
            if expected_version is None:
                state = self._apply_config_patch(
                    patch,
                    timeout,
                    origin=origin,
                    rollout_attempt=rollout_attempt,
                )
            else:
                state = self._apply_config_patch(
                    patch,
                    timeout,
                    origin=origin,
                    rollout_attempt=rollout_attempt,
                    expected_version=expected_version,
                )
        except BaseException:
            if rollout_attempt.fail():
                current = self.runtime.read()
                self.hub.update_stats(
                    matte_rollout=self._matte_rollout.snapshot(
                        current.config,
                        current.version,
                    )
                )
            raise
        if rollout_attempt.succeed():
            self.hub.update_stats(
                matte_rollout=self._matte_rollout.snapshot(
                    state.config,
                    state.version,
                )
            )
        return state

    def _apply_config_patch(
        self,
        patch: dict[str, Any],
        timeout: float = 5.0,
        *,
        origin: str = "internal",
        rollout_attempt: MatteRolloutAttempt | None = None,
        expected_version: int | None = None,
    ) -> ConfigState:
        """Validate, activate, commit, and acknowledge a hot configuration patch."""
        if isinstance(patch, dict):
            restart_fields = []
            if "schema_version" in patch:
                restart_fields.append("schema_version")
            background_patch = patch.get("background")
            if (
                isinstance(background_patch, dict)
                and "camera_device" in background_patch
            ):
                restart_fields.append("background.camera_device")
            if "backdrop_targets" in patch:
                restart_fields.append("backdrop_targets")
            if restart_fields:
                raise RestartRequiredError(restart_fields, self.runtime.version)
        while True:
            base = self.runtime.read()
            if expected_version is not None and base.version != expected_version:
                raise ConfigConflictError(expected_version, base.version)
            candidate = base.config.patched(patch)
            if candidate != base.config:
                break
            # Establish a linearization point for a no-op response. If a
            # commit won between validation and this read, reapply the patch
            # to that generation instead of returning a stale header/body.
            current = self.runtime.read()
            if current.version == base.version:
                log.info(
                    "config update accepted origin=%s version=%d fields=none "
                    "summary=none (no change)",
                    origin,
                    current.version,
                )
                return current
        restart_fields = _restart_only_changes(base.config, candidate)
        if restart_fields:
            raise RestartRequiredError(restart_fields, base.version)
        if not self.running or not self._startup_done.is_set():
            raise ReconfigurationUnavailable("pipeline is not running")
        if threading.current_thread() is self._thread:
            raise ReconfigurationUnavailable(
                "cannot synchronously reconfigure from the pipeline worker"
            )

        deadline = time.monotonic() + timeout
        request = _PatchRequest(
            candidate,
            base.version,
            origin=origin,
            storage_epoch=self._read_storage_epoch(),
            rollout_attempt=rollout_attempt,
        )
        self._prepare_patch_request(request, base.config, deadline)
        return self._submit_prepared_patch(request, deadline)

    def apply_staged_config_patch(
        self,
        patch: dict[str, Any],
        staging_patch: dict[str, Any],
        before_activate: Callable[[], None],
        rollback_activate: Callable[[], None],
        timeout: float = 5.0,
    ) -> ConfigState:
        """Preflight hidden resources, then promote and commit at one boundary.

        ``patch`` describes the final effective configuration while
        ``staging_patch`` points constructors at a hidden candidate asset.
        The promotion callback runs only after construction and trial succeed,
        inside the same compare-and-swap activation that publishes config.
        Its rollback pair must tolerate a partially completed promotion.
        """
        base = self.runtime.read()
        candidate = base.config.patched(patch)
        staging_candidate = base.config.patched(staging_patch)
        final_paths = set(_changed_paths(base.config, candidate))
        staging_paths = set(_changed_paths(base.config, staging_candidate))
        if not final_paths or final_paths != staging_paths:
            raise ValueError(
                "staged and final patches must change the same configuration fields"
            )
        staged_differences = set(_changed_paths(candidate, staging_candidate))
        allowed_asset_path = {
            "image": "background.image_path",
            "video": "background.video_path",
        }.get(candidate.background.mode)
        if candidate.background.mode == "remote":
            allowed_asset_path = {
                "image": "background.image_path",
                "video": "background.video_path",
            }.get(candidate.background.remote_fallback_mode)
        if allowed_asset_path is None or staged_differences != {allowed_asset_path}:
            raise ValueError(
                "staged and final configurations may differ only in the active "
                "image/video asset path"
            )
        restart_fields = _restart_only_changes(base.config, candidate)
        if restart_fields:
            raise RestartRequiredError(restart_fields, base.version)
        if not self.running or not self._startup_done.is_set():
            raise ReconfigurationUnavailable("pipeline is not running")
        if threading.current_thread() is self._thread:
            raise ReconfigurationUnavailable(
                "cannot synchronously reconfigure from the pipeline worker"
            )
        request = _PatchRequest(
            candidate,
            base.version,
            origin="api-upload",
            activation_candidate=staging_candidate,
            before_activate=before_activate,
            rollback_activate=rollback_activate,
            storage_epoch=self._read_storage_epoch(),
        )
        deadline = time.monotonic() + timeout
        self._prepare_patch_request(request, base.config, deadline)
        return self._submit_prepared_patch(request, deadline)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise ReconfigurationUnavailable(
                "pipeline candidate preparation exceeded the reconfiguration deadline"
            )
        return remaining

    def _read_storage_epoch(self) -> int:
        with self._storage_epoch_lock:
            return self._storage_epoch

    def _track_preparation_future(self, future: Future[_Activation]) -> None:
        with self._preparation_condition:
            self._preparation_futures.add(future)

    def _untrack_preparation_future(self, future: Future[_Activation]) -> None:
        with self._preparation_condition:
            self._preparation_futures.discard(future)
            self._preparation_condition.notify_all()

    def _discard_preparation_result(self, future: Future[_Activation]) -> None:
        """Close an abandoned result before releasing its lifecycle token."""

        try:
            if future.cancelled():
                return
            activation = future.result()
        except BaseException:
            return
        else:
            activation.discard()
        finally:
            # Future.result() becomes observable before its callbacks finish.
            # Removing this token only here prevents stop() from mistaking a
            # blocked candidate close for a terminal preparation worker.
            self._untrack_preparation_future(future)

    def _abandon_preparation_future(self, future: Future[_Activation]) -> None:
        """Transfer a not-yet-returned future to deterministic cleanup."""

        if future.cancel():
            self._untrack_preparation_future(future)
            return
        # The constructor is already running or completed concurrently.
        # add_done_callback also runs immediately for a completed future.
        future.add_done_callback(self._discard_preparation_result)

    def _prepare_patch_request(
        self,
        request: _PatchRequest,
        current: AppConfig,
        deadline: float,
    ) -> None:
        """Build fallible candidate resources on the dedicated executor."""

        candidate = request.activation_candidate or request.candidate
        with self._preparation_lock:
            executor = self._preparation_executor
            if executor is None:
                raise ReconfigurationUnavailable(
                    "pipeline candidate preparation is unavailable"
                )
            try:
                future = executor.submit(
                    self._prepare_activation_off_lane,
                    current.model_copy(deep=True),
                    candidate.model_copy(deep=True),
                )
                # Register under the same lock as submit so shutdown cannot
                # snapshot the executor in the handoff gap between the two.
                self._track_preparation_future(future)
            except RuntimeError as exc:
                raise ReconfigurationUnavailable(
                    "pipeline candidate preparation is shutting down"
                ) from exc
        cleanup_deferred = False
        try:
            try:
                remaining = self._remaining(deadline)
            except BaseException:
                cleanup_deferred = True
                self._abandon_preparation_future(future)
                raise
            try:
                activation = future.result(timeout=remaining)
            except FutureTimeout as exc:
                cleanup_deferred = True
                self._abandon_preparation_future(future)
                raise ReconfigurationUnavailable(
                    "pipeline candidate preparation exceeded the reconfiguration deadline"
                ) from exc
            except ActivationError:
                raise
            except BaseException as exc:
                raise ActivationError(str(exc)) from exc
            if not self.running or not self._startup_done.is_set():
                activation.discard()
                raise ReconfigurationUnavailable("pipeline stopped during preparation")
            request.prepared_activation = activation
        finally:
            if not cleanup_deferred:
                self._untrack_preparation_future(future)

    def _submit_prepared_patch(
        self, request: _PatchRequest, deadline: float
    ) -> ConfigState:
        """Transfer a prepared activation to the frame queue or discard it."""

        try:
            remaining = self._remaining(deadline)
        except BaseException:
            activation = request.prepared_activation
            request.prepared_activation = None
            self._schedule_discard_activation(activation)
            raise
        return self._submit_patch(request, remaining)

    def _shutdown_preparation_executor(self, deadline: float, timeout: float) -> None:
        with self._preparation_condition:
            executor = self._preparation_executor
            if executor is None:
                return
            executor.shutdown(wait=False, cancel_futures=True)
            while self._preparation_futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise ReconfigurationUnavailable(
                        "pipeline candidate preparation worker did not stop "
                        f"within {timeout:.1f}s"
                    )
                self._preparation_condition.wait(remaining)
        # All running callbacks have reached terminal state, so this wait is
        # now bounded to executor bookkeeping and worker exit.
        executor.shutdown(wait=True, cancel_futures=True)
        with self._preparation_condition:
            if self._preparation_executor is executor:
                self._preparation_executor = None

    def _submit_patch(self, request: _PatchRequest, timeout: float) -> ConfigState:
        try:
            self._enqueue_request(request)
        except BaseException:
            _cancelled, activation = request.cancel_and_take_activation()
            # The request never entered the owned frame queue, so the caller
            # remains the sole owner and closes it before returning.
            if activation is not None:
                activation.discard()
            raise
        if not request.done.wait(timeout):
            cancelled, activation = request.cancel_and_take_activation()
            if cancelled:
                self._schedule_discard_activation(activation)
                raise ReconfigurationUnavailable(
                    f"pipeline did not acknowledge reconfiguration within {timeout:.1f}s"
                )
            # Completion won the lock concurrently; wait for its event publish.
            request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:  # defensive invariant
            raise ReconfigurationUnavailable(
                "pipeline returned no configuration result"
            )
        return request.result

    def _enqueue_request(self, request: _PatchRequest | _MutationRequest) -> None:
        """Atomically enqueue only while the frame worker still accepts work."""

        with self._request_enqueue_lock:
            if (
                self._stop.is_set()
                or not self.running
                or not self._startup_done.is_set()
            ):
                raise ReconfigurationUnavailable("pipeline is not accepting requests")
            self._requests.put(request)

    def apply_storage_mutation(
        self,
        mutate: Callable[[AppConfig], None],
        timeout: float = 5.0,
    ) -> ConfigState:
        """Serialize an asset mutation with frame-boundary reconfiguration.

        Storage changes are ordered in the same queue as PATCH activation, so
        a candidate is either preflighted before deletion (and makes the asset
        active) or afterward (and fails to open it). Because effective config
        does not change, a successful mutation must not advance its version.
        """
        if not self.running or not self._startup_done.is_set():
            raise ReconfigurationUnavailable("pipeline is not running")
        if threading.current_thread() is self._thread:
            raise ReconfigurationUnavailable(
                "cannot synchronously mutate storage from the pipeline worker"
            )
        request = _MutationRequest(mutate)
        self._enqueue_request(request)
        if not request.done.wait(timeout):
            if request.cancel():
                raise ReconfigurationUnavailable(
                    f"pipeline did not acknowledge storage mutation within {timeout:.1f}s"
                )
            request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise ReconfigurationUnavailable("pipeline returned no storage result")
        return request.result

    # -- lifecycle -----------------------------------------------------
    def _open_resources(
        self,
        state: ConfigState,
        *,
        defer_output: bool = False,
    ) -> _Resources:
        cfg = state.config
        canvas_size = resolved_output_size(cfg)
        # ExitStack protects every successfully opened backend if a later
        # constructor fails. Once complete, _Resources owns deterministic close.
        with ExitStack() as startup:
            capture = open_capture(cfg.camera, canvas_size)
            startup.callback(_safe_close, capture, "capture")
            if self._model_preparation is not None:
                segmenter = create_segmenter(
                    cfg.segmentation,
                    acceleration=cfg.acceleration,
                    preparation=self._model_preparation,
                )
            else:
                segmenter = create_segmenter(
                    cfg.segmentation, acceleration=cfg.acceleration
                )
            startup.callback(_safe_close, segmenter, "segmenter")
            refiner = refiner_for(cfg.segmentation, segmenter)
            backdrop, background_fallback_reason = _startup_backdrop(cfg)
            if backdrop is not None:
                startup.callback(_safe_close, backdrop, "backdrop")
            output_factory: Callable[[], Any] | None = None
            if defer_output:
                output_cfg = cfg.output.model_copy(deep=True)

                def deferred_output_factory() -> Any:
                    return open_output(output_cfg, canvas_size[0], canvas_size[1])

                output_factory = deferred_output_factory
                output = None
            else:
                output = open_output(cfg.output, canvas_size[0], canvas_size[1])
                startup.callback(_safe_close, output, "video output")
            resources = _Resources(
                cfg,
                state.version,
                capture,
                segmenter,
                refiner,
                backdrop,
                output,
                output_factory=output_factory,
                output_owned_by_publisher=defer_output,
                background_fallback_reason=background_fallback_reason,
                background_startup_asset_guard=True,
            )
            # Backend callbacks above already own their individual teardown.
            # Register only the executor created by _Resources so a later
            # construction/configuration failure cannot leak its worker pool or
            # close any backend twice.
            startup.callback(
                resources.color_analysis_executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
            if (
                resources.canvas_size != canvas_size
            ):  # pragma: no cover - resolver invariant
                raise RuntimeError("resource canvas changed during construction")
            if hasattr(self.hub, "configure_canvas"):
                self.hub.configure_canvas(canvas_size)
            startup.pop_all()
            return resources

    @staticmethod
    def _performance_epoch_key(
        resources: _Resources,
        capture_health: Any,
    ) -> PerformanceEpochKey:
        return PerformanceEpochKey(
            config_version=resources.version,
            capture_generation=int(getattr(capture_health, "generation", 0) or 0),
            segmentation_generation=resources.segmentation_generation,
            backdrop_generation=resources.visual_generation,
        )

    @staticmethod
    def _performance_epoch_tuple(
        key: PerformanceEpochKey,
    ) -> tuple[int, int, int, int]:
        return (
            key.config_version,
            key.capture_generation,
            key.segmentation_generation,
            key.backdrop_generation,
        )

    @staticmethod
    def _runtime_stage_samples(
        timings: Mapping[str, float],
        frame_processing_ms: float,
        compositor_substages: Mapping[str, float] | None = None,
    ) -> dict[str, float | None]:
        """Project frame-local timers onto the additive performance schema."""

        substages = compositor_substages or {}
        light_wrap_ms = sum(
            float(substages.get(name, 0.0) or 0.0)
            for name in (
                "backdrop_blur_resize",
                "light_wrap_temporal_filter",
                "light_wrap_interpolation",
            )
        )
        return {
            "capture.read": timings.get("capture_read_ms"),
            "segmentation.total": timings.get("segmentation_ms"),
            "background.total": timings.get("background_ms"),
            "color_correction.estimate": timings.get("color_correction_ms"),
            "color_correction.apply": substages.get("color_transform_application"),
            **{
                f"compositor.{name}": substages.get(name)
                for name in COMPOSITOR_SUBSTAGE_NAMES
            },
            "compositor.light_wrap": light_wrap_ms,
            "compositor.prepare": timings.get("composite_prepare_ms"),
            "compositor.blend": timings.get("composite_blend_ms"),
            "compositor.total": timings.get("composite_ms"),
            "pipeline.processing_only": frame_processing_ms,
            "output.submission": None,
        }

    def _publisher_policy_transition(
        self,
        mode: str,
        *,
        reason: str,
        raw_epoch: int | None = None,
    ) -> PublicationPolicy:
        """Fence publication before changing any privacy-relevant epoch."""

        with self._publisher_policy_lock:
            publisher = self._output_publisher
            self._publisher_policy_epoch += 1
            if mode == "remote":
                if raw_epoch is not None:
                    self._publisher_raw_epoch = raw_epoch
                session = self.hub.active_remote_session() or 0
                policy = PublicationPolicy(
                    self._publisher_policy_epoch,
                    "remote",
                    raw_epoch=self._publisher_raw_epoch,
                    renderer_session=session,
                )
            else:
                policy = PublicationPolicy(self._publisher_policy_epoch, "local")
            if publisher is not None:
                publisher.fence_to_slate(policy, reason=reason, timeout=0.25)
            self._publisher_policy = policy
            return policy

    def _remote_lifecycle_changed(self, event: str, _session_id: int) -> None:
        """Fence retained renderer pixels at the renderer lease boundary."""

        reason = {
            "connected": "renderer-session-changed",
            "disconnected": "privacy-renderer-disconnected",
            "invalidated": "privacy-renderer-invalidated",
        }.get(event)
        if reason is None:
            raise ValueError("unknown remote lifecycle event")
        try:
            with self._publisher_policy_lock:
                if (
                    self._output_publisher is None
                    or self._publisher_policy.mode != "remote"
                ):
                    return
                if self._privacy_history_exhausted:
                    reason = "privacy-history-exhausted"
                self._publisher_policy_transition("remote", reason=reason)
        except BaseException as exc:
            self._publisher_failure(exc)
            raise

    def _publisher_failure(self, exc: BaseException) -> None:
        tracker = self._runtime_performance
        if tracker is not None:
            try:
                tracker.mark_failed("publisher-failed")
                self._publish_publisher_state("failed")
            except Exception:
                pass
        self._error = OutputPublisherError("output publication failed")
        self._error.__cause__ = exc
        self._stop.set()

    def _publish_publisher_state(self, state: str) -> None:
        """Latch a terminal publisher state even when no later send occurs."""

        publisher = self._output_publisher
        tracker = self._runtime_performance
        if publisher is None or tracker is None:
            return
        snapshot = publisher.snapshot()
        performance = tracker.snapshot()
        raw_publisher = performance.get("publisher")
        previous = raw_publisher if isinstance(raw_publisher, Mapping) else {}
        tracker.update_publisher(
            PublisherTelemetry(
                mode=str(previous.get("mode", "deadline-paced")),  # type: ignore[arg-type]
                state=("failed" if state == "failed" else "stopped"),
                handoff_overwrite_count=snapshot.handoff_overwrite_count,
                missed_slot_count=snapshot.schedule_skipped_slots,
                slate_send_count=snapshot.privacy_slate_send_count,
                pending_depth=int(snapshot.pending),
                output_base_config_version=int(
                    previous.get("output_base_config_version", 0) or 0
                ),
            )
        )
        self.hub.update_stats(
            output_publisher_state=state,
            runtime_performance=tracker.snapshot(),
        )

    def _publisher_pacing_complete(
        self,
        timing: OutputSendTiming,
        *,
        bridge: _PublisherBridge,
    ) -> None:
        """Install measured sink pacing after accepted pixels are visible."""

        with bridge.lock:
            output_send_ms = timing.submission_ms + timing.pacing_wait_ms
            bridge.output_ewma["output_send_ms"] = _ewma(
                bridge.output_ewma["output_send_ms"],
                output_send_ms,
            )
            bridge.output_ewma["output_sink_pacing_wait_ms"] = _ewma(
                bridge.output_ewma["output_sink_pacing_wait_ms"],
                timing.pacing_wait_ms,
            )
            raw_timing = bridge.last_status.get("timing_ms")
            timing_map = dict(raw_timing) if isinstance(raw_timing, Mapping) else {}
            timing_map["output.send_total"] = bridge.output_ewma["output_send_ms"]
            timing_map["output.sink_pacing_wait"] = bridge.output_ewma[
                "output_sink_pacing_wait_ms"
            ]
            update = {
                "output_send_ms": bridge.output_ewma["output_send_ms"],
                "output_sink_pacing_wait_ms": bridge.output_ewma[
                    "output_sink_pacing_wait_ms"
                ],
                "timing_ms": timing_map,
            }
            bridge.last_status.update(update)
        self.hub.update_stats(**update)

    def _publish_output_receipt(
        self,
        receipt: OutputSendReceipt,
        *,
        resources: _Resources,
        bridge: _PublisherBridge,
    ) -> None:
        """Install one sink-accepted frame and its exact matching public state."""

        base = receipt.base
        with bridge.lock:
            if base is not None:
                raw_status = thaw_status(base.status)
                if not isinstance(raw_status, dict):  # pragma: no cover - invariant
                    raise RuntimeError("publisher base status is not a mapping")
                status = raw_status
                bridge.output_base_config_version = base.config_version
                if receipt.base_updated:
                    bridge.resolved_remote_raw_epoch = max(
                        bridge.resolved_remote_raw_epoch,
                        base.resolved_remote_raw_epoch,
                    )
            else:
                status = dict(bridge.last_status)

            # This processing-lane snapshot carries the currently committed
            # policy/selection identity. An old safe base may continue to be
            # repeated after a hot commit, but only
            # ``output_base_config_version`` below is allowed to retain the
            # old pixel provenance.
            status.update(bridge.committed_status)

            # A PATCH acknowledgement represents the committed runtime
            # configuration even while the paced lane is still repeating the
            # last safe pixels.  Preserve the pixel-producing version in
            # ``output_base_config_version`` below, but never let an immutable
            # old envelope roll the authoritative public config version or
            # rollout decision backwards.
            committed_state = self.runtime.read()
            status["config_version"] = committed_state.version
            status["mode"] = committed_state.config.background.mode
            status["matte_rollout"] = self._matte_rollout.snapshot(
                committed_state.config,
                committed_state.version,
            )

            processing_missed = bool(
                receipt.base_updated
                and base is not None
                and base.processing_deadline_missed
            )
            schedule_late = receipt.schedule_lateness_ms > 1.0
            cadence_has_prior = (
                bridge.cadence.snapshot(
                    now_ns=receipt.timing.submitted_at_ns
                ).output_send_count
                > 0
            )
            cadence_base_updated = bool(
                receipt.base_updated or (not cadence_has_prior and base is not None)
            )
            bridge.cadence.record_send(
                sent_at_ns=receipt.timing.submitted_at_ns,
                capture_sequence=(
                    base.capture_sequence
                    if cadence_base_updated and base is not None
                    else None
                ),
                captured_at_ns=(
                    base.captured_at_ns
                    if cadence_base_updated and base is not None
                    else None
                ),
                base_ready_at_ns=(
                    base.base_ready_at_ns
                    if cadence_base_updated and base is not None
                    else None
                ),
                base_updated=cadence_base_updated,
                segmentation_updated=bool(
                    receipt.base_updated
                    and base is not None
                    and base.segmentation_updated
                ),
                exact_final_repeat=(
                    receipt.exact_final_repeat if cadence_has_prior else False
                ),
                processing_deadline_missed=processing_missed,
                serialized_new_frame_deadline_missed=False,
                output_sink_pacing_events=receipt.timing.pacing_events,
                output_sink_recovery_events=receipt.timing.recovery_events,
                application_pacing_events=receipt.application_pacing_events,
                output_schedule_late=schedule_late,
            )

            output_send_ms = (
                receipt.timing.submission_ms
                + receipt.timing.pacing_wait_ms
                + receipt.application_pacing_wait_ms
            )
            output_samples = {
                "output_send_ms": output_send_ms,
                "output_submission_ms": receipt.timing.submission_ms,
                "output_sink_pacing_wait_ms": receipt.timing.pacing_wait_ms,
                "application_pacing_wait_ms": receipt.application_pacing_wait_ms,
                "output_schedule_lateness_ms": receipt.schedule_lateness_ms,
            }
            for name, sample in output_samples.items():
                if (
                    bool(getattr(resources.output, "paces", False))
                    and receipt.timing.pacing_events > 0
                    and receipt.timing.pacing_wait_ms == 0.0
                    and name in {"output_send_ms", "output_sink_pacing_wait_ms"}
                ):
                    # Acceptance-aware sinks expose the submitted pixels before
                    # their deliberate wait. Completion installs these two
                    # measurements without delaying the matching hub frame.
                    continue
                bridge.output_ewma[name] = _ewma(bridge.output_ewma[name], sample)

            new_frame_service_ms: float | None = None
            if receipt.base_updated and base is not None:
                service_ns = receipt.timing.submitted_at_ns - base.captured_at_ns
                if service_ns < 0 or service_ns > 3_600_000_000_000:
                    # Focused/test captures and a few third-party capture
                    # adapters may not share the process monotonic epoch.  Do
                    # not publish an impossible duration; retain the bounded
                    # handoff-to-first-submission component instead.
                    service_ns = receipt.timing.submitted_at_ns - base.base_ready_at_ns
                new_frame_service_ms = max(0.0, service_ns / 1_000_000.0)
                bridge.output_ewma["new_frame_service_ms"] = _ewma(
                    bridge.output_ewma["new_frame_service_ms"],
                    new_frame_service_ms,
                )

            # Processing can rotate epochs before its new envelope is adopted.
            # The tracker compares this producer key atomically with recording,
            # so an old-generation repeat cannot leak into the new bucket.
            # Built-in slates have no producer key and count in the active
            # publication epoch.
            expected_epoch = (
                None if base is None else PerformanceEpochKey(*base.performance_epoch)
            )
            bridge.performance.record_output(
                unique_base=receipt.base_updated,
                schedule_late=schedule_late,
                submission_ms=receipt.timing.submission_ms,
                expected_epoch=expected_epoch,
            )
            publisher_state = receipt.snapshot.state
            tracker_state = (
                "failed"
                if publisher_state == "failed"
                else "stopped"
                if publisher_state in {"stopping", "stopped"}
                else "running"
            )
            bridge.performance.update_publisher(
                PublisherTelemetry(
                    mode=bridge.publisher_mode,  # type: ignore[arg-type]
                    state=tracker_state,  # type: ignore[arg-type]
                    handoff_overwrite_count=(receipt.snapshot.handoff_overwrite_count),
                    missed_slot_count=receipt.snapshot.schedule_skipped_slots,
                    slate_send_count=receipt.snapshot.privacy_slate_send_count,
                    pending_depth=int(receipt.snapshot.pending),
                    output_base_config_version=bridge.output_base_config_version,
                )
            )
            performance = bridge.performance.snapshot()
            cadence_snapshot = bridge.cadence.snapshot(
                now_ns=receipt.timing.completed_at_ns
            )

            previous_state = bridge.performance_state
            current_state = str(performance["state"])
            if current_state != previous_state:
                if current_state == "degraded":
                    log.warning(
                        "runtime performance degraded reason=%s output_fps=%.3f "
                        "unique_fps=%.3f cadence_status=%s "
                        "transport_target_fps=%.3f unique_target_fps=%.3f "
                        "dominant_stage=%s exact_repeat_pct=%.1f",
                        performance["reason"],
                        performance["output_send_fps"],
                        performance["sent_unique_base_fps"],
                        performance["cadence_status"],
                        performance["transport_target_fps"],
                        performance["unique_target_fps"],
                        performance["dominant_stage"] or "none",
                        cadence_snapshot.exact_final_output_repeat_ratio * 100.0,
                    )
                elif previous_state == "degraded" and current_state == "healthy":
                    log.warning(
                        "runtime performance recovered cadence_status=%s "
                        "transport_target_fps=%.3f unique_target_fps=%.3f",
                        performance["cadence_status"],
                        performance["transport_target_fps"],
                        performance["unique_target_fps"],
                    )
                bridge.performance_state = current_state

            status.update(
                _cadence_status(
                    bridge.cadence,
                    now_ns=receipt.timing.completed_at_ns,
                    target_output_fps=resources.cfg.output.fps,
                )
            )
            timing = status.get("timing_ms")
            timing_map = dict(timing) if isinstance(timing, Mapping) else {}
            timing_map.update(
                {
                    "output.send_total": bridge.output_ewma["output_send_ms"],
                    "output.submission": bridge.output_ewma["output_submission_ms"],
                    "output.sink_pacing_wait": bridge.output_ewma[
                        "output_sink_pacing_wait_ms"
                    ],
                    "output.application_pacing_wait": bridge.output_ewma[
                        "application_pacing_wait_ms"
                    ],
                    "output.schedule_lateness": bridge.output_ewma[
                        "output_schedule_lateness_ms"
                    ],
                    "pipeline.new_frame_service": bridge.output_ewma[
                        "new_frame_service_ms"
                    ],
                    # The compute and publication lanes are no longer one
                    # serialized service scope. Keep the v1 key nullable.
                    "pipeline.new_frame_serialized_loop": None,
                }
            )
            remote_privacy_slate = (
                resources.cfg.background.mode == "remote" and receipt.privacy_slate
            )
            status.update(
                {
                    "output_send_ms": bridge.output_ewma["output_send_ms"],
                    "output_submission_ms": bridge.output_ewma["output_submission_ms"],
                    "output_sink_pacing_wait_ms": bridge.output_ewma[
                        "output_sink_pacing_wait_ms"
                    ],
                    "application_pacing_wait_ms": bridge.output_ewma[
                        "application_pacing_wait_ms"
                    ],
                    "output_schedule_lateness_ms": bridge.output_ewma[
                        "output_schedule_lateness_ms"
                    ],
                    "new_frame_service_ms": bridge.output_ewma["new_frame_service_ms"],
                    "new_frame_serialized_loop_ms": None,
                    "timing_schema_version": TIMING_SCHEMA_VERSION,
                    "timing_ms": timing_map,
                    "processing_completed_count": bridge.processing_completed_count,
                    "processing_completed_fps": performance["processing_completed_fps"],
                    "output_base_config_version": bridge.output_base_config_version,
                    "output_handoff_overwrite_count": (
                        receipt.snapshot.handoff_overwrite_count
                    ),
                    "output_schedule_skipped_slots": (
                        receipt.snapshot.schedule_skipped_slots
                    ),
                    "output_privacy_slate_send_count": (
                        receipt.snapshot.privacy_slate_send_count
                    ),
                    "output_publisher_state": receipt.snapshot.state,
                    "runtime_performance": performance,
                    "remote_fallback_active": remote_privacy_slate,
                    "remote_fallback_reason": (
                        receipt.privacy_reason if remote_privacy_slate else ""
                    ),
                }
            )
            bridge.last_status = dict(status)
            evidence = (
                bridge.pending_evidence.pop(base.base_id, None)
                if receipt.base_updated and base is not None
                else None
            )

        self.hub.publish_output(receipt.pixels, stats=status)
        if evidence is not None:
            evidence.timings_ms.update(
                {
                    "output_send_ms": output_send_ms,
                    "output_submission_ms": receipt.timing.submission_ms,
                    "output_sink_pacing_wait_ms": receipt.timing.pacing_wait_ms,
                    "application_pacing_wait_ms": (receipt.application_pacing_wait_ms),
                    "output_schedule_lateness_ms": receipt.schedule_lateness_ms,
                }
            )
            if new_frame_service_ms is not None:
                evidence.timings_ms["new_frame_service_ms"] = new_frame_service_ms
            rss_bytes = process_rss_bytes()
            if rss_bytes is not None:
                evidence.resource_samples["rss_bytes"] = rss_bytes
            if self._submit_matte_evidence(evidence, receipt.pixels):
                bridge.last_matte_source_sequence = evidence.metadata.bundle_sequence
                self._matte_last_source_sequence = bridge.last_matte_source_sequence
            self._submit_live_matte_evidence(evidence, status=status)
        recorder = self._matte_recorder
        output_matte_source = None if base is None else base.matte_bundle_sequence
        if recorder is not None and output_matte_source is not None:
            recorder.submit_output_event(
                sent_monotonic_ns=receipt.timing.submitted_at_ns,
                source_bundle_sequence=output_matte_source,
                base_updated=receipt.base_updated,
                exact_final_repeat=receipt.exact_final_repeat,
            )
        elif recorder is not None:
            recorder.mark_output_timeline_incomplete("unattributed-safe-slate")

    def _run(self) -> None:
        resources: _Resources | None = None
        cadence_tracker: CadenceTracker | None = None
        publisher: OutputPublisher | None = None
        bridge: _PublisherBridge | None = None
        try:
            # Reset replay evidence only after every stale renderer lease and
            # queued remote frame have been invalidated at one hub boundary.
            self.hub.reset_remote_session(self._reset_raw_replay_history)
            state = self.runtime.read()
            resources = self._open_resources(state, defer_output=True)
            self._active_state = state
            capture_health = (
                resources.capture.health_snapshot()
                if hasattr(resources.capture, "health_snapshot")
                else None
            )
            cadence_tracker = CadenceTracker(resources.cfg.output.fps)
            # A slower requested camera intentionally supplies fewer unique
            # bases than the transport publishes. Keep that processing domain
            # distinct from the sink schedule so 15 -> 30 repeats are healthy.
            unique_target_fps = min(
                resources.cfg.camera.fps,
                resources.cfg.output.fps,
            )
            performance = RuntimePerformanceTracker(
                resources.cfg.output.fps,
                self._performance_epoch_key(resources, capture_health),
                unique_target_fps=unique_target_fps,
                color_correction_mode=resources.cfg.compositing.color_correction.mode,
                light_wrap=resources.cfg.compositing.light_wrap,
            )
            self._runtime_performance = performance
            bridge = _PublisherBridge(
                cadence=cadence_tracker,
                performance=performance,
                publisher_mode="deadline-paced",
            )

            initial_mode = resources.cfg.background.mode
            initial_session = self.hub.active_remote_session() or 0
            self._publisher_policy = PublicationPolicy(
                0,
                "remote" if initial_mode == "remote" else "local",
                raw_epoch=0,
                renderer_session=(initial_session if initial_mode == "remote" else 0),
            )
            if resources.output_factory is None:  # pragma: no cover - invariant
                raise RuntimeError("deferred output factory is missing")
            publisher = OutputPublisher(
                resources.output_factory,
                width=resources.canvas_size[0],
                height=resources.canvas_size[1],
                target_fps=resources.cfg.output.fps,
                slate_pixels=self._privacy_slate(resources.canvas_shape),
                initial_policy=self._publisher_policy,
                on_send=lambda receipt: self._publish_output_receipt(
                    receipt,
                    resources=resources,
                    bridge=bridge,
                ),
                on_pacing_complete=lambda timing: self._publisher_pacing_complete(
                    timing,
                    bridge=bridge,
                ),
                on_failure=self._publisher_failure,
                on_overwrite=bridge.discard_evidence,
                on_release=bridge.release_base,
            )
            self._output_publisher = publisher
            publisher.start()
            resources.output = publisher.wait_opened(5.0)
            bridge.publisher_mode = (
                "sink-paced"
                if bool(getattr(resources.output, "paces", False))
                else "deadline-paced"
            )

            preflight = self._preflight(resources, send_output=False)
            processing_deadline_ms = 1000.0 / unique_target_fps
            performance.record_processing(
                deadline_missed=(
                    preflight.frame_processing_ms > processing_deadline_ms
                ),
                stages_ms=self._runtime_stage_samples(
                    preflight.timings,
                    preflight.frame_processing_ms,
                    resources.last_compositor_substages_ms,
                ),
            )
            bridge.processing_completed_count = 1
            stage_ewma: dict[str, float | None] = {
                **preflight.timings,
                "output_send_ms": None,
                "output_submission_ms": None,
                "output_sink_pacing_wait_ms": None,
                "application_pacing_wait_ms": None,
                "output_schedule_lateness_ms": None,
                "frame_processing_ms": preflight.frame_processing_ms,
                "new_frame_service_ms": None,
                "new_frame_serialized_loop_ms": None,
            }
            capture_health = (
                resources.capture.health_snapshot()
                if hasattr(resources.capture, "health_snapshot")
                else None
            )
            initial_stats = self._identity_stats(
                resources,
                capture_health=capture_health,
                color_status=preflight.color_status,
            )
            initial_stats.update(
                {
                    **_cadence_status(
                        cadence_tracker,
                        now_ns=preflight.base_ready_at_ns,
                        target_output_fps=resources.cfg.output.fps,
                    ),
                    **_capture_health_stats(
                        capture_health,
                        fallback_frames_read=1,
                    ),
                    "remote_frames_used": 0,
                    "remote_fallback_active": preflight.remote_fallback_active,
                    "remote_fallback_count": (
                        1 if preflight.remote_fallback_active else 0
                    ),
                    "remote_fallback_reason": preflight.remote_fallback_reason,
                    "segmentation_ms": stage_ewma["segmentation_ms"],
                    "background_ms": stage_ewma["background_ms"],
                    "color_correction_ms": stage_ewma["color_correction_ms"],
                    "composite_ms": stage_ewma["composite_ms"],
                    "output_send_ms": stage_ewma["output_send_ms"],
                    "output_submission_ms": stage_ewma["output_submission_ms"],
                    "output_sink_pacing_wait_ms": stage_ewma[
                        "output_sink_pacing_wait_ms"
                    ],
                    "application_pacing_wait_ms": stage_ewma[
                        "application_pacing_wait_ms"
                    ],
                    "output_schedule_lateness_ms": stage_ewma[
                        "output_schedule_lateness_ms"
                    ],
                    "frame_processing_ms": stage_ewma["frame_processing_ms"],
                    "new_frame_service_ms": stage_ewma["new_frame_service_ms"],
                    "new_frame_serialized_loop_ms": stage_ewma[
                        "new_frame_serialized_loop_ms"
                    ],
                    "timing_schema_version": TIMING_SCHEMA_VERSION,
                    "timing_ms": _timing_fields(
                        stage_ewma,
                        capture_health=capture_health,
                        segmenter=resources.segmenter,
                    ),
                    "processing_completed_count": 1,
                    "processing_completed_fps": 0.0,
                    "output_base_config_version": resources.version,
                    "output_handoff_overwrite_count": 0,
                    "output_schedule_skipped_slots": 0,
                    "output_privacy_slate_send_count": 0,
                    "output_publisher_state": "running",
                    "runtime_performance": performance.snapshot(),
                }
            )
            if resources.cfg.background.mode == "remote":
                self._publisher_raw_epoch += 1
                self._publisher_policy_transition(
                    "remote",
                    reason="awaiting-renderer",
                    raw_epoch=self._publisher_raw_epoch,
                )
                self._publish_remote_raw_frame(
                    preflight.captured.pixels,
                    self._publisher_raw_epoch,
                )
            else:
                self.hub.publish_raw(preflight.captured.pixels)
            bridge.last_status = dict(initial_stats)
            bridge.committed_status = _committed_output_status(
                self._identity_stats(
                    resources,
                    capture_health=capture_health,
                    color_status=preflight.color_status,
                )
            )
            bridge.output_base_config_version = resources.version
            initial_asset_slate = bool(resources.background_fallback_reason)
            initial_privacy_slate = (
                initial_asset_slate or resources.cfg.background.mode == "remote"
            )
            if preflight.output is None:  # pragma: no cover - startup invariant
                raise RuntimeError("startup preflight produced no output")
            base = SafeBaseFrame.create(
                preflight.output,
                base_id=0,
                config_version=resources.version,
                capture_sequence=preflight.captured.sequence,
                captured_at_ns=preflight.captured.captured_at_ns,
                base_ready_at_ns=preflight.base_ready_at_ns,
                policy_epoch=self._publisher_policy.epoch,
                segmentation_updated=preflight.segmentation_updated,
                processing_deadline_missed=(
                    preflight.frame_processing_ms > processing_deadline_ms
                ),
                status=initial_stats,
                performance_epoch=self._performance_epoch_tuple(
                    performance.current_epoch_key
                ),
                privacy_slate=initial_privacy_slate,
                privacy_reason=(
                    resources.background_fallback_reason
                    if initial_asset_slate
                    else preflight.remote_fallback_reason or "no-client"
                    if resources.cfg.background.mode == "remote"
                    else ""
                ),
                matte_bundle_sequence=(
                    None
                    if initial_asset_slate or preflight.matte_evidence is None
                    else preflight.matte_evidence.metadata.bundle_sequence
                ),
            )
            # SafeBaseFrame owns its immutable pixels. Drop the preflight copy
            # before the paced lane starts retaining current/pending bases.
            retained_preflight_evidence = (
                None if initial_asset_slate else preflight.matte_evidence
            )
            preflight = replace(preflight, output=None, matte_evidence=None)
            with bridge.lock:
                bridge.retain_base(
                    0,
                    PerformanceEpochKey(*base.performance_epoch),
                    retained_preflight_evidence,
                )
                try:
                    accepted = publisher.submit(base)
                except BaseException:
                    bridge.release_base(0)
                    raise
                if not accepted:
                    bridge.release_base(0)
                    raise OutputPublisherError("initial output base was rejected")
            retained_preflight_evidence = None
            publisher.wait_ready(5.0)
            del base
            performance.mark_ready()
            # Pre-ready capture churn belongs to the startup epoch.  Begin the
            # public steady-state sequence-gap accounting at the next accepted
            # frame while retaining the preflight sequence only for duplicate
            # rejection inside the processing loop.  The compatibility cadence
            # tracker is deliberately *not* reset: ``frames_in`` remains the
            # count of every unique base that reached the sink, including the
            # startup base.  RuntimePerformanceTracker owns the separate
            # pre-ready/steady-state split.
            resources.capture_sequence_timeline = _CaptureSequenceTimeline()
            self.hub.update_stats(runtime_performance=performance.snapshot())
            self._startup_done.set()
            self._loop(
                resources,
                preflight=preflight,
                cadence_tracker=cadence_tracker,
                stage_ewma=stage_ewma,
                publisher=publisher,
                publisher_bridge=bridge,
            )
            if publisher.error is not None:
                raise OutputPublisherError(
                    "output publisher failed"
                ) from publisher.error
        except BaseException as exc:
            log.exception("pipeline crashed")
            if self._error is None:
                self._error = exc
            if self._runtime_performance is not None:
                try:
                    self._runtime_performance.mark_failed("pipeline-failed")
                except Exception:
                    pass
        finally:
            self._stop.set()
            with self._request_enqueue_lock:
                self._fail_pending(
                    ReconfigurationUnavailable("pipeline worker stopped")
                )
            if resources is not None:
                resources.close_capture()
                if publisher is not None:
                    publisher.request_stop()
                    try:
                        self._publish_publisher_state("stopping")
                    except Exception:
                        log.exception("cannot publish stopping output state")
                    try:
                        publisher.close(timeout=5.0)
                    except OutputPublisherError as exc:
                        if self._error is None:
                            self._error = exc
                        # A blocked/ambiguous sink call still owns the output
                        # backend and may still invoke its acceptance callback.
                        # Keep this pipeline worker, bridge, and resources alive
                        # until that owner thread actually exits.  stop() stays
                        # bounded by reporting this frame worker as a survivor,
                        # and start() rejects a new run meanwhile.
                        if publisher.is_alive:
                            self._startup_done.set()
                            while not publisher.wait_terminated(0.1):
                                pass
                    terminal_state = (
                        "failed"
                        if publisher.snapshot().state == "failed"
                        else "stopped"
                    )
                    try:
                        self._publish_publisher_state(terminal_state)
                    except Exception:
                        log.exception("cannot publish terminal output state")
                self._publish_terminal_cadence(resources, cadence_tracker)
                resources.close()
            self._output_publisher = None
            if self._matte_recorder is not None:
                self._matte_recorder.close()
            if self._matte_monitor is not None:
                self._matte_monitor.clear_history()
            self._drain_deferred_closes()
            # On startup failure, readiness is not published until teardown
            # finishes. A blocked close is therefore observed as a surviving
            # startup worker rather than being hidden behind the root error.
            self._startup_done.set()

    def _publish_terminal_cadence(
        self,
        resources: _Resources,
        cadence_tracker: CadenceTracker | None,
    ) -> None:
        """Latch final drop/reset/cadence evidence after capture has stopped."""

        if cadence_tracker is None or not hasattr(self.hub, "update_stats"):
            return
        try:
            snapshot = cadence_tracker.snapshot(now_ns=time.monotonic_ns())
            capture_health = (
                resources.capture.health_snapshot()
                if hasattr(resources.capture, "health_snapshot")
                else None
            )
            capture_sequence = resources.capture_sequence_timeline.snapshot()
            temporal = resources.segmentation_timeline.snapshot()
            values = _cadence_status(
                cadence_tracker,
                now_ns=time.monotonic_ns(),
                target_output_fps=resources.cfg.output.fps,
            )
            values.update(
                _capture_health_stats(
                    capture_health,
                    fallback_frames_read=snapshot.unique_capture_count,
                )
            )
            values.update(
                {
                    "capture_sequence": capture_sequence.last_sequence or 0,
                    "capture_sequence_gap_count": capture_sequence.gap_events,
                    "capture_missing_input_count": capture_sequence.missing_inputs,
                    "matte_reset_count": temporal.reset_count,
                    "matte_last_reset_reason": (
                        ""
                        if temporal.last_reset_reason is None
                        else temporal.last_reset_reason.value
                    ),
                }
            )
            self.hub.update_stats(**values)
        except Exception:
            # Teardown must continue even if a third-party test double violates
            # the health/status contract. The original pipeline error remains
            # authoritative.
            log.exception("cannot publish terminal cadence health")

    def _fail_pending(self, error: BaseException) -> None:
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            if isinstance(request, _PatchRequest):
                with request.lock:
                    activation = request.prepared_activation
                    request.prepared_activation = None
                    if request.result is None and request.error is None:
                        request.error = error
                        if request.rollout_attempt is not None:
                            request.rollout_attempt.fail()
                request.done.set()
                self._schedule_discard_activation(activation)
            else:
                request.fail(error)

    # -- reconfiguration ----------------------------------------------
    @staticmethod
    def _prepare_activation_off_lane(
        current: AppConfig,
        candidate: AppConfig,
    ) -> _Activation:
        """Construct changed resources without occupying the frame worker."""

        activation = _Activation(candidate=candidate)
        try:
            if _segmenter_key(candidate) != _segmenter_key(current):
                activation.segmenter = create_segmenter(
                    candidate.segmentation, acceleration=candidate.acceleration
                )
                activation.replace_segmenter = True
                activation.refiner = refiner_for(
                    candidate.segmentation,
                    activation.segmenter,
                )
                activation.temporal_state_owner = _TemporalStateOwner(
                    policy=_segmenter_key(candidate),
                    generation=None,
                    segmenter=activation.segmenter,
                    refiner=activation.refiner,
                )
            if _backdrop_provider_key(candidate) != _backdrop_provider_key(current):
                activation.backdrop = _build_backdrop(candidate)
                activation.replace_backdrop = True
            activation.visual_state_changed = _visual_state_key(
                candidate
            ) != _visual_state_key(current)
            activation.replace_harmonizer = _color_state_key(
                candidate
            ) != _color_state_key(current)
            if activation.replace_harmonizer:
                activation.harmonizer = _new_color_harmonizer(candidate)
            activation.reconfigure_harmonizer = (
                not activation.replace_harmonizer
                and _color_scalar_policy_key(candidate)
                != _color_scalar_policy_key(current)
            )
            activation.replace_light_wrap_stabilizer = _light_wrap_state_key(
                candidate
            ) != _light_wrap_state_key(current)
            if activation.replace_light_wrap_stabilizer:
                activation.light_wrap_stabilizer = _new_light_wrap_stabilizer(candidate)
        except Exception as exc:
            activation.discard()
            raise ActivationError(str(exc)) from exc
        return activation

    def _stage_activation(
        self,
        resources: _Resources,
        candidate: AppConfig,
        prepared: _Activation | None,
    ) -> _Activation:
        """Attach reused live pointers to an already constructed candidate."""

        old_cfg = resources.cfg
        if prepared is None:
            raise ActivationError("candidate resources were not prepared off-lane")
        activation = prepared
        if activation.promoted:
            raise ActivationError("prepared activation was already promoted")
        activation.candidate = candidate
        old_segmentation_policy = _segmenter_key(old_cfg)
        candidate_segmentation_policy = _segmenter_key(candidate)
        segmentation_changed = candidate_segmentation_policy != old_segmentation_policy
        background_changed = _backdrop_provider_key(
            candidate
        ) != _backdrop_provider_key(old_cfg)
        visual_state_changed = _visual_state_key(candidate) != _visual_state_key(
            old_cfg
        )
        color_state_changed = _color_state_key(candidate) != _color_state_key(old_cfg)
        color_scalar_policy_changed = _color_scalar_policy_key(
            candidate
        ) != _color_scalar_policy_key(old_cfg)
        light_wrap_state_changed = _light_wrap_state_key(
            candidate
        ) != _light_wrap_state_key(old_cfg)
        if segmentation_changed != activation.replace_segmenter:
            raise ActivationError("prepared segmentation candidate is stale")
        if background_changed != activation.replace_backdrop:
            raise ActivationError("prepared background candidate is stale")
        if visual_state_changed != activation.visual_state_changed:
            raise ActivationError("prepared visual-state candidate is stale")
        if color_state_changed != activation.replace_harmonizer:
            raise ActivationError("prepared color-state candidate is stale")
        if (
            color_scalar_policy_changed and not color_state_changed
        ) != activation.reconfigure_harmonizer:
            raise ActivationError("prepared color scalar-policy candidate is stale")
        if light_wrap_state_changed != activation.replace_light_wrap_stabilizer:
            raise ActivationError("prepared light-wrap-state candidate is stale")
        live_owner = resources.temporal_state_owner
        if not live_owner.matches(
            policy=old_segmentation_policy,
            generation=resources.segmentation_generation,
            segmenter=resources.segmenter,
            refiner=resources.refiner,
        ):
            raise ActivationError("live temporal-state ownership is inconsistent")
        if segmentation_changed:
            candidate_owner = activation.temporal_state_owner
            if candidate_owner is None or not candidate_owner.matches(
                policy=candidate_segmentation_policy,
                generation=None,
                segmenter=activation.segmenter,
                refiner=activation.refiner,
            ):
                raise ActivationError(
                    "prepared segmenter/refiner policy ownership is stale"
                )
            activation.temporal_state_owner = _TemporalStateOwner(
                policy=candidate_owner.policy,
                generation=resources.segmentation_generation + 1,
                segmenter=candidate_owner.segmenter,
                refiner=candidate_owner.refiner,
            )
        elif activation.temporal_state_owner is not None:
            raise ActivationError(
                "unchanged segmentation policy cannot own candidate temporal state"
            )
        if not background_changed:
            activation.backdrop = resources.backdrop
        if activation.reconfigure_harmonizer:
            if resources.harmonizer is None:
                raise ActivationError("live color harmonizer is missing")
            correction = candidate.compositing.color_correction
            activation.harmonizer = resources.harmonizer.clone(
                adaptation_time_s=correction.adaptation_time_s,
            )
            old_limit = old_cfg.compositing.color_correction.exposure_limit_ev
            if correction.exposure_limit_ev < old_limit:
                activation.harmonizer.clamp_exposure(correction.exposure_limit_ev)
        elif not color_state_changed:
            activation.harmonizer = resources.harmonizer
        elif activation.harmonizer is None:
            raise ActivationError("prepared color harmonizer is missing")
        if not light_wrap_state_changed:
            activation.light_wrap_stabilizer = resources.light_wrap_stabilizer
        elif (
            _light_wrap_state_key(candidate) != ("off",)
            and activation.light_wrap_stabilizer is None
        ):
            raise ActivationError("prepared light-wrap stabilizer is missing")
        return activation

    def _trial_activation(
        self,
        resources: _Resources,
        activation: _Activation,
        captured: CapturedFrame,
    ) -> None:
        """Exercise only staged/synthetic state before committing.

        Existing working segmenter, refiner, and backdrop objects are never
        called here, so a failed trial preserves recurrent state, temporal mask,
        video position, and blur caches exactly.
        """
        # A trial backend is untrusted until it succeeds. Give it a detached
        # raster so it cannot mutate or retain the authoritative capture slot.
        frame = np.ascontiguousarray(captured.pixels).copy()
        old_cfg = resources.cfg
        # Resource ownership follows `_segmenter_key`, which includes both the
        # segmentation policy and acceleration/provider policy. Trial that exact
        # replacement set so an acceleration-only commit cannot install an
        # unexercised backend.
        segmentation_changed = _segmenter_key(activation.candidate) != _segmenter_key(
            old_cfg
        )
        background_changed = _backdrop_provider_key(
            activation.candidate
        ) != _backdrop_provider_key(old_cfg)
        background_geometry_changed = (
            activation.candidate.background.fit_mode,
            activation.candidate.background.anchor_x,
            activation.candidate.background.anchor_y,
        ) != (
            old_cfg.background.fit_mode,
            old_cfg.background.anchor_x,
            old_cfg.background.anchor_y,
        )
        compositing_changed = activation.candidate.compositing != old_cfg.compositing
        if not (
            segmentation_changed
            or background_changed
            or background_geometry_changed
            or compositing_changed
        ):
            return
        try:
            if background_geometry_changed:
                # Exercise the candidate planner and pixel path on detached,
                # asymmetric pixels.  Never call a reused live video/camera
                # provider during a geometry-only trial.
                synthetic = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
                trial_size = (
                    min(resources.canvas_size[0], 64),
                    min(resources.canvas_size[1], 64),
                )
                geometry_plan = plan_transform(
                    (7, 5),
                    trial_size,
                    fit=activation.candidate.background.fit_mode,
                    anchors=(
                        activation.candidate.background.anchor_x,
                        activation.candidate.background.anchor_y,
                    ),
                )
                fitted = apply_transform(synthetic, geometry_plan)
                self._validate_canvas_frame(
                    fitted,
                    trial_size,
                    boundary="candidate backdrop geometry",
                )
            if segmentation_changed:
                context = self._segmentation_context(captured)
                candidate_owner = activation.temporal_state_owner
                if candidate_owner is None or not candidate_owner.matches(
                    policy=_segmenter_key(activation.candidate),
                    generation=resources.segmentation_generation + 1,
                    segmenter=activation.segmenter,
                    refiner=activation.refiner,
                ):
                    raise ActivationError(
                        "candidate trial does not own its temporal state"
                    )
                activation.temporal_trial_context = context
                activation.temporal_trial_dirty = True
                self._reset_temporal_pair(
                    activation.segmenter,
                    activation.refiner,
                    TemporalResetReason.SEGMENTATION_CONFIG,
                    context.timestamp_ns,
                )
                mask = self._segment_and_refine_mask(
                    activation.segmenter,
                    activation.refiner,
                    frame,
                    privacy_safe=activation.candidate.background.mode == "remote",
                    context=context,
                )
            else:
                # A deterministic synthetic edge exercises backdrop/compositor
                # contracts without touching the working processing state.
                mask = np.full(frame.shape[:2], 0.5, dtype=np.float32)

            if background_changed and activation.backdrop is not None:
                if isinstance(activation.backdrop, BlurBackdrop):
                    activation.backdrop.set_source_frame(frame, mask)
                bg = activation.backdrop.frame(
                    resources.canvas_size[0],
                    resources.canvas_size[1],
                )
                self._validate_canvas_frame(
                    bg,
                    resources.canvas_size,
                    boundary="candidate backdrop",
                )
            else:
                bg = np.zeros_like(frame)

            candidate_segmenter = (
                activation.segmenter if segmentation_changed else resources.segmenter
            )
            candidate_ratio = getattr(
                candidate_segmenter,
                "last_downsample_ratio",
                None,
            )
            candidate_policy = resolve_matte_policy(
                activation.candidate.segmentation,
                activation.candidate.compositing,
                segmenter_matte_backend_kind(candidate_segmenter),
                resolved_rvm_ratio=(
                    float(candidate_ratio)
                    if isinstance(candidate_ratio, (int, float))
                    and not isinstance(candidate_ratio, bool)
                    and math.isfinite(float(candidate_ratio))
                    else None
                ),
                passthrough=(activation.candidate.background.mode == "passthrough"),
                canvas_shape=(resources.canvas_size[1], resources.canvas_size[0]),
                light_wrap_stabilization_eligible=(
                    activation.candidate.background.mode in {"video", "camera"}
                ),
            )
            edge_fg = (
                activation.segmenter.last_foreground
                if segmentation_changed
                and candidate_policy.effective.use_model_foreground
                else None
            )
            if edge_fg is not None:
                edge_fg = self._validate_canvas_frame(
                    edge_fg,
                    resources.canvas_size,
                    boundary="candidate model foreground",
                )
            base_harmonizer = activation.harmonizer or resources.harmonizer
            if base_harmonizer is None:
                raise ActivationError("candidate color harmonizer is missing")
            trial_harmonizer = base_harmonizer.clone()
            prepared_color = self._prepare_color_frame(
                resources,
                frame,
                bg,
                mask,
                edge_fg,
                now_s=captured.captured_at_ns / 1_000_000_000.0,
                cfg=activation.candidate,
                harmonizer=trial_harmonizer,
                track_live_state=False,
            )
            prepared_light_wrap: PreparedLightWrap | None = None
            base_wrap_stabilizer = (
                activation.light_wrap_stabilizer
                if activation.replace_light_wrap_stabilizer
                else resources.light_wrap_stabilizer
            )
            if (
                candidate_policy.effective.light_wrap > 0.0
                and base_wrap_stabilizer is not None
            ):
                trial_wrap_stabilizer = base_wrap_stabilizer.clone()
                prepared_light_wrap = prepare_light_wrap(
                    bg,
                    blend_space=activation.candidate.compositing.blend_space,
                    stabilizer=trial_wrap_stabilizer,
                    context=LightWrapFrameContext(
                        frame_id=0,
                        timestamp_ns=captured.captured_at_ns,
                        source_token=("activation-trial", id(activation)),
                    ),
                    backdrop_linear_bgr=(
                        prepared_color.backdrop_linear_bgr
                        if activation.candidate.compositing.blend_space == "linear_srgb"
                        else None
                    ),
                )
            trial_workspace = (
                LegacyCompositorWorkspace(resources.canvas_shape)
                if activation.candidate.compositing.blend_space == "srgb_legacy"
                else None
            )
            try:
                out = self._composite_prepared_color(
                    activation.candidate,
                    frame,
                    bg,
                    mask,
                    edge_fg,
                    prepared_color,
                    light_wrap=candidate_policy.effective.light_wrap,
                    prepared_light_wrap=prepared_light_wrap,
                    legacy_workspace=trial_workspace,
                )
            finally:
                if trial_workspace is not None:
                    trial_workspace.close()
            self._validate_output_frame(out, resources.canvas_size)
            if segmentation_changed:
                # Trials are validation-only. Never publish recurrence,
                # foreground, refiner history, or input identity derived from
                # an unsent trial. The normal timeline still records a config
                # reset on the first authoritative input after commit.
                assert activation.temporal_trial_context is not None
                self._reset_temporal_pair(
                    activation.segmenter,
                    activation.refiner,
                    TemporalResetReason.SEGMENTATION_CONFIG,
                    activation.temporal_trial_context.timestamp_ns,
                )
                activation.temporal_trial_dirty = False
        except Exception as exc:
            raise ActivationError(str(exc)) from exc

    def _install_activation(
        self,
        resources: _Resources,
        activation: _Activation,
        version: int,
    ) -> tuple[Any, Any, AppConfig]:
        """Swap prepared pointers using only precomputed, reversible operations."""
        old_cfg = resources.cfg
        old_segmenter = resources.segmenter
        old_refiner = resources.refiner
        old_temporal_state_owner = getattr(
            resources,
            "temporal_state_owner",
            None,
        )
        old_backdrop = resources.backdrop
        old_background_fallback_reason = getattr(
            resources,
            "background_fallback_reason",
            "",
        )
        old_harmonizer = getattr(resources, "harmonizer", None)
        old_light_wrap_stabilizer = getattr(
            resources,
            "light_wrap_stabilizer",
            None,
        )
        old_color_reset_token = getattr(resources, "color_reset_token", None)
        old_backdrop_analysis_token = getattr(
            resources,
            "color_backdrop_analysis_token",
            None,
        )
        old_backdrop_analysis = getattr(
            resources,
            "color_backdrop_analysis_linear_bgr",
            None,
        )
        old_version = resources.version
        old_visual_generation = resources.visual_generation
        old_segmentation_generation = getattr(resources, "segmentation_generation", 0)
        old_light_wrap_generation = getattr(resources, "light_wrap_generation", 0)
        old_segmentation_timeline = (
            resources.segmentation_timeline.checkpoint()
            if activation.replace_segmenter
            else None
        )
        old_active_state = self._active_state
        old_backdrop_geometry = (
            resources.backdrop.geometry
            if not activation.replace_backdrop
            and resources.backdrop is not None
            and hasattr(resources.backdrop, "geometry")
            else None
        )
        background_geometry_changed = (
            activation.candidate.background.fit_mode,
            activation.candidate.background.anchor_x,
            activation.candidate.background.anchor_y,
        ) != (
            old_cfg.background.fit_mode,
            old_cfg.background.anchor_x,
            old_cfg.background.anchor_y,
        )
        # Allocate/copy before the first effective pointer changes. The actual
        # swap below is assignment-only and has an explicit rollback guard.
        new_active_state = ConfigState(
            activation.candidate.model_copy(deep=True), version
        )
        candidate_owner = activation.temporal_state_owner
        if activation.replace_segmenter:
            if (
                activation.temporal_trial_context is None
                or activation.temporal_trial_dirty
                or candidate_owner is None
                or not candidate_owner.matches(
                    policy=_segmenter_key(activation.candidate),
                    generation=old_segmentation_generation + 1,
                    segmenter=activation.segmenter,
                    refiner=activation.refiner,
                )
            ):
                raise ActivationError(
                    "candidate temporal state was not cleanly trialed"
                )
        elif (
            old_temporal_state_owner is not None
            and not old_temporal_state_owner.matches(
                policy=_segmenter_key(activation.candidate),
                generation=old_segmentation_generation,
                segmenter=old_segmenter,
                refiner=old_refiner,
            )
        ):
            raise ActivationError(
                "unchanged segmentation policy lost temporal-state ownership"
            )
        try:
            if activation.replace_segmenter:
                assert candidate_owner is not None
                resources.segmenter = activation.segmenter
                resources.refiner = activation.refiner
                resources.segmentation_generation = old_segmentation_generation + 1
                resources.temporal_state_owner = candidate_owner
                resources.segmentation_timeline.request_reset(
                    TemporalResetReason.SEGMENTATION_CONFIG
                )
            if activation.replace_backdrop:
                resources.backdrop = activation.backdrop
                # Candidate asset providers are constructed and exercised
                # before this commit.  Replacing the unavailable persisted
                # provider therefore clears the startup-only slate latch.
                resources.background_fallback_reason = ""
            if activation.replace_harmonizer or activation.reconfigure_harmonizer:
                if activation.harmonizer is None:
                    raise ActivationError("staged color harmonizer is missing")
                resources.harmonizer = activation.harmonizer
            if activation.replace_harmonizer:
                resources.color_reset_token = None
            if activation.replace_light_wrap_stabilizer:
                resources.light_wrap_stabilizer = activation.light_wrap_stabilizer
                resources.light_wrap_generation = old_light_wrap_generation + 1
            if activation.visual_state_changed:
                resources.visual_generation += 1
                invalidate_analysis = getattr(
                    resources,
                    "invalidate_color_backdrop_analysis",
                    None,
                )
                if callable(invalidate_analysis):
                    invalidate_analysis()
            resources.cfg = activation.candidate
            resources.version = version
            self._active_state = new_active_state
            if (
                background_geometry_changed
                and not activation.replace_backdrop
                and resources.backdrop is not None
                and hasattr(resources.backdrop, "set_geometry")
            ):
                resources.backdrop.set_geometry(
                    activation.candidate.background.fit_mode,
                    activation.candidate.background.anchor_x,
                    activation.candidate.background.anchor_y,
                )
        except BaseException:
            resources.segmenter = old_segmenter
            resources.refiner = old_refiner
            if old_temporal_state_owner is not None:
                resources.temporal_state_owner = old_temporal_state_owner
            resources.backdrop = old_backdrop
            resources.background_fallback_reason = old_background_fallback_reason
            resources.harmonizer = old_harmonizer
            resources.light_wrap_stabilizer = old_light_wrap_stabilizer
            resources.color_reset_token = old_color_reset_token
            if hasattr(resources, "color_backdrop_analysis_token"):
                resources.color_backdrop_analysis_token = old_backdrop_analysis_token
            if hasattr(resources, "color_backdrop_analysis_linear_bgr"):
                resources.color_backdrop_analysis_linear_bgr = old_backdrop_analysis
            resources.cfg = old_cfg
            resources.version = old_version
            resources.visual_generation = old_visual_generation
            resources.segmentation_generation = old_segmentation_generation
            resources.light_wrap_generation = old_light_wrap_generation
            if old_segmentation_timeline is not None:
                resources.segmentation_timeline.restore(old_segmentation_timeline)
            self._active_state = old_active_state
            if (
                old_backdrop_geometry is not None
                and resources.backdrop is old_backdrop
                and hasattr(resources.backdrop, "set_geometry")
            ):
                resources.backdrop.set_geometry(
                    old_backdrop_geometry.fit_mode,
                    old_backdrop_geometry.anchor_x,
                    old_backdrop_geometry.anchor_y,
                )
            raise
        return (
            old_backdrop if activation.replace_backdrop else None,
            old_segmenter if activation.replace_segmenter else None,
            old_cfg,
        )

    def _post_install_activation(
        self, resources: _Resources, old_cfg: AppConfig
    ) -> None:
        """Run non-critical hub side effects without invalidating a commit."""
        if (
            old_cfg.compositing.blend_space == "srgb_legacy"
            and resources.cfg.compositing.blend_space != "srgb_legacy"
        ):
            _safe_close(
                resources._legacy_compositor_workspace,
                "legacy compositor workspace",
            )
            resources._legacy_compositor_workspace = None
        if _segmenter_key(old_cfg) != _segmenter_key(resources.cfg):
            selection = segmenter_selection_status(
                resources.segmenter,
                resources.cfg.segmentation.backend,
            )
            self._log_fallback_transition(
                "segmentation",
                bool(selection["fallback_active"]),
                str(selection["fallback_reason"]),
            )
        if old_cfg.background.mode != resources.cfg.background.mode:
            try:
                self.hub.invalidate_remote_session()
            except Exception:
                log.exception("cannot invalidate remote session after mode switch")
                self._privacy_history_exhausted = True

    def _handle_patch_request(
        self,
        resources: _Resources,
        request: _PatchRequest,
        trial_frame: CapturedFrame,
        *,
        publisher_bridge: _PublisherBridge | None = None,
    ) -> None:
        with request.lock:
            if request.cancelled:
                activation = request.prepared_activation
                request.prepared_activation = None
                request.done.set()
                self._schedule_discard_activation(activation)
                return
            # Ownership leaves the request before any fallible trial work.
            # A concurrent timeout can only reclaim an activation that the
            # frame lane has not claimed yet.
            activation = request.prepared_activation
            request.prepared_activation = None
        current = self.runtime.read()
        storage_epoch = self._read_storage_epoch()
        if (
            current.version != request.expected_version
            or resources.version != request.expected_version
        ):
            self._schedule_discard_activation(activation)
            request.fail(ConfigConflictError(request.expected_version, current.version))
            return
        if storage_epoch != request.storage_epoch:
            self._schedule_discard_activation(activation)
            request.fail(ActivationError("candidate assets changed during preparation"))
            return
        try:
            activation = self._stage_activation(
                resources,
                request.activation_candidate or request.candidate,
                activation,
            )
            self._trial_activation(resources, activation, trial_frame)
            if activation.replace_backdrop and hasattr(
                activation.backdrop, "reset_stats"
            ):
                # Candidate trials are deliberately unsent.  The provider
                # object is installed after the trial, so reset only its public
                # counters (not playback state) before publishing the new
                # current-provider telemetry generation.
                activation.backdrop.reset_stats()
            # Resources were exercised through the hidden staging path, but
            # the effective snapshot published below must contain only the
            # promoted final path.
            activation.candidate = request.candidate
        except BaseException as exc:
            self._schedule_discard_activation(activation)
            request.fail(exc)
            return

        old_backdrop = old_segmenter = old_light_wrap_stabilizer = None
        old_cfg: AppConfig | None = None
        old_publication_policy = self._publisher_policy
        publication_fenced = False
        with request.lock:
            if request.cancelled:
                self._schedule_discard_activation(activation)
                request.done.set()
                return
            try:

                def activate(next_version: int) -> None:
                    nonlocal old_backdrop
                    nonlocal old_segmenter
                    nonlocal old_light_wrap_stabilizer
                    nonlocal old_cfg
                    nonlocal publication_fenced
                    promotion_attempted = request.before_activate is not None
                    try:
                        current_mode = resources.cfg.background.mode
                        candidate_mode = request.candidate.background.mode
                        if self._output_publisher is not None and (
                            current_mode == "remote"
                            or candidate_mode == "remote"
                            or current_mode != candidate_mode
                        ):
                            self._publisher_policy_transition(
                                candidate_mode,
                                reason="config-transition",
                            )
                            publication_fenced = True
                        if request.before_activate is not None:
                            request.before_activate()
                        replaced_light_wrap_stabilizer = (
                            resources.light_wrap_stabilizer
                            if activation.replace_light_wrap_stabilizer
                            else None
                        )
                        old_backdrop, old_segmenter, old_cfg = self._install_activation(
                            resources, activation, next_version
                        )
                        old_light_wrap_stabilizer = replaced_light_wrap_stabilizer
                    except BaseException:
                        if publication_fenced and self._output_publisher is not None:
                            self._output_publisher.fence_to_slate(
                                old_publication_policy,
                                reason="config-rollback",
                                timeout=0.25,
                            )
                            self._publisher_policy = old_publication_policy
                            publication_fenced = False
                        if (
                            promotion_attempted
                            and request.rollback_activate is not None
                        ):
                            # If rollback itself fails, surface that cleanup
                            # failure: the API must report that storage
                            # ownership could not be restored.
                            request.rollback_activate()
                        raise

                committed = self._runtime_writer.commit_with_activation(
                    request.candidate, request.expected_version, activate
                )
            except ConfigVersionConflictError as exc:
                if publication_fenced and self._output_publisher is not None:
                    self._output_publisher.fence_to_slate(
                        old_publication_policy,
                        reason="config-rollback",
                        timeout=0.25,
                    )
                    self._publisher_policy = old_publication_policy
                self._schedule_discard_activation(activation)
                request.error = ConfigConflictError(
                    exc.expected_version, exc.current_version
                )
                if request.rollout_attempt is not None:
                    request.rollout_attempt.fail()
            except BaseException as exc:
                if publication_fenced and self._output_publisher is not None:
                    self._output_publisher.fence_to_slate(
                        old_publication_policy,
                        reason="config-rollback",
                        timeout=0.25,
                    )
                    self._publisher_policy = old_publication_policy
                self._schedule_discard_activation(activation)
                request.error = exc
                if request.rollout_attempt is not None:
                    request.rollout_attempt.fail()
            else:
                # From here onward the candidate belongs exclusively to live
                # resources. Detach it before any best-effort side effect can
                # fail and accidentally route it through candidate cleanup.
                activation.mark_promoted()
                if old_backdrop is not None:
                    # The old provider is now quiescent but has not yet been
                    # closed. Fold its final counters exactly once so the new
                    # current-provider generation cannot erase run history.
                    resources.retire_background_video_provider(old_backdrop)
                request.result = committed
                if request.rollout_attempt is not None:
                    request.rollout_attempt.succeed()
                if publisher_bridge is not None:
                    capture_health = (
                        resources.capture.health_snapshot()
                        if hasattr(resources.capture, "health_snapshot")
                        else None
                    )
                    publisher_bridge.performance.bind_epoch(
                        self._performance_epoch_key(resources, capture_health),
                        color_correction_mode=(
                            resources.cfg.compositing.color_correction.mode
                        ),
                        light_wrap=resources.cfg.compositing.light_wrap,
                    )
                    _log_finalized_performance_epochs(publisher_bridge.performance)
                    committed_status = self._identity_stats(
                        resources,
                        capture_health=capture_health,
                    )
                    with publisher_bridge.lock:
                        publisher_bridge.committed_status = _committed_output_status(
                            committed_status
                        )
                if old_cfg is not None:
                    try:
                        self._post_install_activation(resources, old_cfg)
                    except Exception:
                        log.exception("cannot apply committed config side effects")
                try:
                    changed = _changed_paths(
                        old_cfg or resources.cfg,
                        request.candidate,
                    )
                    log.info(
                        "config update accepted origin=%s version=%d fields=%s "
                        "summary=%s",
                        request.origin,
                        committed.version,
                        ",".join(changed) or "none",
                        sanitized_config_summary(request.candidate, changed),
                    )
                except Exception:
                    log.exception("cannot audit committed config update")
            # Publish success/failure before teardown. A blocking or faulty
            # old backend must not delay or invalidate an already-committed ack.
            request.done.set()
        self._schedule_close(old_backdrop, "replaced backdrop")
        self._schedule_close(old_segmenter, "replaced segmenter")
        self._schedule_close(
            old_light_wrap_stabilizer,
            "replaced light-wrap stabilizer",
        )

    def _schedule_discard_activation(self, activation: _Activation | None) -> None:
        if activation is None:
            return
        backdrop = activation.take_backdrop()
        if backdrop is not None:
            self._schedule_close(backdrop, "discarded staged backdrop")
        if activation.replace_light_wrap_stabilizer:
            stabilizer = activation.light_wrap_stabilizer
            activation.replace_light_wrap_stabilizer = False
            activation.light_wrap_stabilizer = None
            if stabilizer is not None:
                self._schedule_close(
                    stabilizer,
                    "discarded staged light-wrap stabilizer",
                )
        refiner = activation.refiner if activation.replace_segmenter else None
        segmenter = activation.take_segmenter()
        if refiner is not None:
            self._schedule_close(refiner, "discarded staged mask refiner")
        if segmenter is not None:
            self._schedule_close(segmenter, "discarded staged segmenter")

    def _handle_mutation_request(
        self, resources: _Resources, request: _MutationRequest
    ) -> None:
        with request.lock:
            if request.cancelled:
                request.done.set()
                return
            try:
                # Give the callback a read-only-by-convention copy so a
                # storage operation cannot mutate the live resource config.
                request.mutate(resources.cfg.model_copy(deep=True))
                request.result = self.runtime.read()
                if request.result.version != resources.version:
                    raise ConfigConflictError(resources.version, request.result.version)
                with self._storage_epoch_lock:
                    self._storage_epoch += 1
            except BaseException as exc:
                request.error = exc
            request.done.set()

    def _schedule_close(self, resource: Any, label: str) -> None:
        """Close replaced resources off the frame worker and track liveness."""
        if resource is None:
            return

        worker: threading.Thread

        def close_resource() -> None:
            try:
                _safe_close(resource, label)
            finally:
                with self._teardown_lock:
                    self._teardown_threads.discard(worker)

        worker = threading.Thread(
            target=close_resource,
            name=f"teardown-{label.replace(' ', '-')}",
            daemon=True,
        )
        with self._teardown_lock:
            self._teardown_threads.add(worker)
        try:
            worker.start()
        except BaseException:
            with self._teardown_lock:
                self._teardown_threads.discard(worker)
                self._deferred_closes.append((resource, label))
            # The new config/resource pair was committed before teardown was
            # scheduled. Thread exhaustion must not revoke that success or kill
            # the frame worker; retain the old resource for shutdown cleanup.
            log.exception(
                "cannot schedule %s teardown; deferring until shutdown", label
            )

    def _drain_deferred_closes(self) -> None:
        """Close resources whose post-commit teardown thread could not start."""
        with self._teardown_lock:
            deferred = tuple(self._deferred_closes)
            self._deferred_closes.clear()
        for resource, label in deferred:
            _safe_close(resource, f"deferred {label}")

    def _identity_stats(
        self,
        resources: _Resources,
        *,
        capture_health: Any = None,
        color_status: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build path-free identity state for one matching output boundary."""

        cfg = resources.cfg
        if capture_health is None and hasattr(resources.capture, "health_snapshot"):
            capture_health = resources.capture.health_snapshot()
        output_fallback = bool(getattr(resources.output, "fallback_active", False))
        segmentation_selection = segmenter_selection_status(
            resources.segmenter,
            cfg.segmentation.backend,
        )
        segmentation_fallback = bool(segmentation_selection["fallback_active"])
        segmentation_fallback_reason = str(segmentation_selection["fallback_reason"])
        background_fallback_reason = resources.background_fallback_reason
        background_fallback_active = bool(background_fallback_reason)
        video_stats = (
            resources.backdrop.stats_dict()
            if resources.backdrop is not None
            and hasattr(resources.backdrop, "stats_dict")
            else dict(_VIDEO_STATS_DEFAULTS)
        )
        if not isinstance(video_stats, Mapping):
            video_stats = dict(_VIDEO_STATS_DEFAULTS)
        video_lifetime_stats = resources.background_video_lifetime_stats(video_stats)
        camera_geometry = _camera_plan_stats(
            cfg,
            resources.canvas_size,
            capture_health,
        )
        background_geometry = _background_plan_stats(resources)
        self._log_geometry_transitions(
            resources,
            capture_health,
            camera_geometry,
            background_geometry,
        )
        if color_status is None:
            snapshot = (
                resources.harmonizer.snapshot()
                if resources.harmonizer is not None
                else None
            )
            color_status = _color_stats(
                cfg,
                snapshot,
                exposure_clamped=resources.color_correction_exposure_clamped,
                white_balance_clamped=resources.color_correction_wb_clamped,
            )
        self._log_color_transition(resources, color_status)
        self._log_fallback_transition(
            "output",
            output_fallback,
            getattr(resources.output, "fallback_reason", "") if output_fallback else "",
        )
        self._log_fallback_transition(
            "segmentation",
            segmentation_fallback,
            segmentation_fallback_reason,
        )
        self._log_fallback_transition(
            "background",
            background_fallback_active,
            background_fallback_reason,
        )
        policy_snapshot = self._matte_policy_snapshot(resources)
        effective_controls = self._effective_matte_controls(
            resources,
            policy_snapshot=policy_snapshot,
        )
        effective_refiner = effective_controls["refiner"]
        if not isinstance(effective_refiner, dict):
            effective_refiner = {}
        effective_boundary = effective_refiner.get("boundary_stabilization", {})
        if not isinstance(effective_boundary, dict):
            effective_boundary = {}
        effective_edge_refine = bool(effective_refiner.get("edge_refine", False))
        effective_edge_refinement_mode = str(
            effective_controls.get("edge_refinement_mode", "off") or "off"
        )
        raw_edge_refinement_radius = effective_controls.get(
            "edge_refinement_radius_px",
            0,
        )
        effective_edge_refinement_radius_px = (
            raw_edge_refinement_radius if type(raw_edge_refinement_radius) is int else 0
        )
        produces_matte = bool(effective_controls["produces_matte"])
        effective_mask_shift = effective_controls["mask_shift"]
        if type(effective_mask_shift) is not int:
            effective_mask_shift = 0
        effective_light_wrap = effective_controls["light_wrap"]
        if not isinstance(effective_light_wrap, (int, float)) or isinstance(
            effective_light_wrap, bool
        ):
            effective_light_wrap = 0.0
        capture_sequence = resources.capture_sequence_timeline.snapshot()
        temporal = resources.segmentation_timeline.snapshot()
        matte_policy = self._public_matte_policy(
            resources,
            policy_snapshot=policy_snapshot,
        )
        return {
            "mode": cfg.background.mode,
            "segmentation_backend": type(resources.segmenter).__name__,
            "segmentation_device": resources.segmenter.device,
            "segmentation_selection": segmentation_selection,
            "matte_policy": matte_policy,
            "matte_rollout": self._matte_rollout.snapshot(cfg, resources.version),
            "segmentation_generation": resources.segmentation_generation,
            "capture_sequence": capture_sequence.last_sequence or 0,
            "capture_sequence_gap_count": capture_sequence.gap_events,
            "capture_missing_input_count": capture_sequence.missing_inputs,
            "matte_reset_count": temporal.reset_count,
            "matte_last_reset_reason": (
                ""
                if temporal.last_reset_reason is None
                else temporal.last_reset_reason.value
            ),
            "segmentation_produces_matte": produces_matte,
            "effective_rvm_downsample_ratio": effective_controls[
                "rvm_downsample_ratio"
            ],
            "effective_mask_blur": int(effective_refiner.get("mask_blur", 0) or 0),
            "effective_edge_refine": effective_edge_refine,
            "effective_edge_refinement_mode": effective_edge_refinement_mode,
            "effective_edge_refinement_radius_px": (
                effective_edge_refinement_radius_px
            ),
            "effective_mask_shift": effective_mask_shift,
            "effective_temporal_smoothing": float(
                effective_refiner.get("temporal_smoothing", 0.0) or 0.0
            ),
            "effective_boundary_stabilization_mode": str(
                effective_boundary.get("mode", "off") or "off"
            ),
            "effective_boundary_stabilization_time_constant_s": float(
                effective_boundary.get("time_constant_s", 0.1) or 0.1
            ),
            "effective_boundary_stabilization_max_motion_px_per_s": float(
                effective_boundary.get("max_motion_px_per_s", 720.0) or 720.0
            ),
            "effective_use_model_foreground": bool(
                effective_controls["use_model_foreground"] and produces_matte
            ),
            "effective_light_wrap": float(effective_light_wrap),
            "output_backend": type(resources.output).__name__,
            "output_target_fps": cfg.output.fps,
            "output_width": getattr(
                resources.output,
                "width",
                resources.canvas_size[0],
            ),
            "output_height": getattr(
                resources.output,
                "height",
                resources.canvas_size[1],
            ),
            "output_fps": getattr(resources.output, "fps", cfg.output.fps),
            "output_fallback_active": output_fallback,
            "output_fallback_reason": (
                getattr(resources.output, "fallback_reason", "")
                if output_fallback
                else ""
            ),
            "segmentation_fallback_active": segmentation_fallback,
            "segmentation_fallback_reason": segmentation_fallback_reason,
            "capture_backend": (
                getattr(capture_health, "backend", type(resources.capture).__name__)
                if capture_health is not None
                else type(resources.capture).__name__
            ),
            "capture_fourcc": getattr(capture_health, "fourcc", None),
            # width/height retain their original negotiated-mode meaning.
            "capture_width": getattr(capture_health, "width", cfg.camera.width),
            "capture_height": getattr(capture_health, "height", cfg.camera.height),
            "capture_delivered_width": getattr(capture_health, "delivered_width", None),
            "capture_delivered_height": getattr(
                capture_health, "delivered_height", None
            ),
            "capture_oriented_width": getattr(capture_health, "oriented_width", None),
            "capture_oriented_height": getattr(capture_health, "oriented_height", None),
            "capture_normalized_width": getattr(
                capture_health, "normalized_width", resources.canvas_size[0]
            ),
            "capture_normalized_height": getattr(
                capture_health, "normalized_height", resources.canvas_size[1]
            ),
            "capture_generation": getattr(capture_health, "generation", 0),
            "capture_geometry_transitions": getattr(
                capture_health, "geometry_transitions", 0
            ),
            "camera_controls": _camera_controls_stats(capture_health),
            "capture_fps_reported": getattr(
                capture_health, "fps_reported", float(cfg.camera.fps)
            ),
            "capture_target_fps": cfg.camera.fps,
            "remote_fallback_mode": (
                "privacy-slate" if cfg.background.mode == "remote" else ""
            ),
            "background_fallback_active": background_fallback_active,
            "background_fallback_reason": background_fallback_reason,
            "background_geometry_transitions": (
                resources.background_geometry_transitions
            ),
            "color_correction_applied_frames": (
                resources.color_correction_applied_frames
            ),
            "color_correction_bypassed_frames": (
                resources.color_correction_bypassed_frames
            ),
            "color_correction_scene_cuts": resources.color_correction_scene_cuts,
            "color_correction_transitions": resources.color_correction_transitions,
            "color_correction_reason_transitions": (
                resources.color_correction_reason_transitions
            ),
            "color_correction_exposure_clamp_count": (
                resources.color_correction_exposure_clamp_count
            ),
            "color_correction_exposure_clamp_time_s": (
                resources.color_correction_exposure_clamp_time_s
            ),
            "color_correction_wb_clamp_count": (
                resources.color_correction_wb_clamp_count
            ),
            "color_correction_wb_clamp_time_s": (
                resources.color_correction_wb_clamp_time_s
            ),
            "config_version": resources.version,
            **_acceleration_stats(resources.segmenter),
            **video_stats,
            **video_lifetime_stats,
            **camera_geometry,
            **background_geometry,
            **color_status,
        }

    def _update_identity_stats(self, resources: _Resources) -> None:
        cfg = resources.cfg
        output_fallback = bool(getattr(resources.output, "fallback_active", False))
        selection = segmenter_selection_status(
            resources.segmenter,
            cfg.segmentation.backend,
        )
        segmentation_fallback = bool(selection["fallback_active"])
        self.hub.update_stats(**self._identity_stats(resources))
        self._log_fallback_transition(
            "output",
            output_fallback,
            getattr(resources.output, "fallback_reason", "") if output_fallback else "",
        )
        self._log_fallback_transition(
            "segmentation",
            segmentation_fallback,
            str(selection["fallback_reason"]),
        )

    def _log_geometry_transitions(
        self,
        resources: _Resources,
        capture_health: Any,
        camera: dict[str, object],
        background: dict[str, object],
    ) -> None:
        """Log the first complete plan and later scalar-only plan changes."""

        camera_state = (
            getattr(capture_health, "generation", 0),
            getattr(capture_health, "delivered_width", None),
            getattr(capture_health, "delivered_height", None),
            getattr(capture_health, "oriented_width", None),
            getattr(capture_health, "oriented_height", None),
            *camera.values(),
        )
        if self._geometry_log_states.get("camera") != camera_state:
            self._geometry_log_states["camera"] = camera_state
            log.info(
                "camera transform generation=%d delivered=%sx%s oriented=%sx%s "
                "output=%dx%d fit=%s rotation=%d mirror=%s "
                "scale=%s,%s crop=%s,%s,%s,%s pad=%d,%d,%d,%d",
                getattr(capture_health, "generation", 0),
                getattr(capture_health, "delivered_width", None),
                getattr(capture_health, "delivered_height", None),
                getattr(capture_health, "oriented_width", None),
                getattr(capture_health, "oriented_height", None),
                resources.canvas_size[0],
                resources.canvas_size[1],
                camera["camera_fit"],
                camera["camera_rotation"],
                camera["camera_mirror"],
                camera["camera_scale_x"],
                camera["camera_scale_y"],
                camera["camera_crop_left"],
                camera["camera_crop_top"],
                camera["camera_crop_right"],
                camera["camera_crop_bottom"],
                camera["camera_pad_left"],
                camera["camera_pad_top"],
                camera["camera_pad_right"],
                camera["camera_pad_bottom"],
            )
            delivered_width = int(getattr(capture_health, "delivered_width", 0) or 0)
            delivered_height = int(getattr(capture_health, "delivered_height", 0) or 0)
            oriented_width = int(getattr(capture_health, "oriented_width", 0) or 0)
            oriented_height = int(getattr(capture_health, "oriented_height", 0) or 0)
            negotiation_mismatch = bool(
                delivered_width > 0
                and delivered_height > 0
                and (
                    delivered_width != resources.cfg.camera.width
                    or delivered_height != resources.cfg.camera.height
                )
            )
            aspect_distortion = 0.0
            if (
                resources.cfg.camera.fit_mode == "stretch"
                and oriented_width > 0
                and oriented_height > 0
            ):
                source_aspect = oriented_width / oriented_height
                target_aspect = resources.canvas_size[0] / resources.canvas_size[1]
                aspect_distortion = abs(source_aspect / target_aspect - 1.0)
            if negotiation_mismatch or aspect_distortion > 0.01:
                log.warning(
                    "camera geometry mismatch negotiation=%s "
                    "delivered=%dx%d requested=%dx%d "
                    "stretch_aspect_distortion_pct=%.3f",
                    negotiation_mismatch,
                    delivered_width,
                    delivered_height,
                    resources.cfg.camera.width,
                    resources.cfg.camera.height,
                    aspect_distortion * 100.0,
                )

        effective_mode = (
            resources.cfg.background.remote_fallback_mode
            if resources.cfg.background.mode == "remote"
            else resources.cfg.background.mode
        )
        background_state = (
            effective_mode,
            id(resources.backdrop),
            *background.values(),
        )
        # Fitted providers have no truthful plan between a geometry change and
        # their next rendered frame. Do not log/count that temporary unknown.
        plan_available = background["background_scale_x"] is not None
        if (
            plan_available
            and self._geometry_log_states.get("background") != background_state
        ):
            self._geometry_log_states["background"] = background_state
            resources.background_geometry_token = background_state
            resources.background_geometry_transitions += 1
            log.info(
                "background transform mode=%s output=%dx%d fit=%s rotation=%d "
                "mirror=%s scale=%s,%s crop=%s,%s,%s,%s pad=%d,%d,%d,%d",
                effective_mode,
                resources.canvas_size[0],
                resources.canvas_size[1],
                background["background_fit"],
                background["background_rotation"],
                background["background_mirror"],
                background["background_scale_x"],
                background["background_scale_y"],
                background["background_crop_left"],
                background["background_crop_top"],
                background["background_crop_right"],
                background["background_crop_bottom"],
                background["background_pad_left"],
                background["background_pad_top"],
                background["background_pad_right"],
                background["background_pad_bottom"],
            )

    def _log_color_transition(
        self,
        resources: _Resources,
        status: dict[str, object],
        *,
        now_s: float | None = None,
    ) -> None:
        """Log phase immediately and debounce estimator-reason chatter."""

        phase_state = (
            status["color_correction_mode"],
            status["color_correction_active"],
            status["color_correction_effective_mode"],
            status["color_correction_state"],
        )
        if self._color_phase_log_state != phase_state:
            self._color_phase_log_state = phase_state
            resources.color_correction_transitions += 1
            log.info(
                "color correction phase state=%s configured=%s effective=%s "
                "active=%s exposure_ev=%+.3f wb=%.4f,%.4f,%.4f",
                status["color_correction_state"],
                status["color_correction_mode"],
                status["color_correction_effective_mode"],
                status["color_correction_active"],
                status["color_correction_exposure_ev"],
                status["color_correction_wb_gain_r"],
                status["color_correction_wb_gain_g"],
                status["color_correction_wb_gain_b"],
            )

        observed = time.monotonic() if now_s is None else float(now_s)
        if not math.isfinite(observed):
            raise ValueError("color log observation time must be finite")
        reason = str(status["color_correction_reason"])
        if self._color_reason_log_state is None:
            emit_reason = True
        elif reason == self._color_reason_log_state:
            self._color_reason_log_candidate = None
            emit_reason = False
        else:
            candidate = self._color_reason_log_candidate
            if candidate is None or candidate[0] != reason:
                self._color_reason_log_candidate = (reason, observed)
                emit_reason = False
            else:
                emit_reason = observed - candidate[1] >= _COLOR_REASON_DEBOUNCE_S

        if emit_reason:
            self._color_reason_log_state = reason
            self._color_reason_log_candidate = None
            resources.color_correction_reason_transitions += 1
            log.info(
                "color correction estimator reason=%s confidence=%.3f",
                reason,
                status["color_correction_confidence"],
            )

    @staticmethod
    def _record_color_output(
        resources: _Resources,
        status: dict[str, object],
        *,
        processed: bool,
        observed_at_s: float | None = None,
    ) -> None:
        if status["color_correction_active"]:
            resources.color_correction_applied_frames += 1
        else:
            resources.color_correction_bypassed_frames += 1
        if processed and status["color_correction_state"] == "scene-cut":
            resources.color_correction_scene_cuts += 1
        if not processed:
            return

        observed = time.monotonic() if observed_at_s is None else float(observed_at_s)
        if not math.isfinite(observed):
            raise ValueError("color observation time must be finite")
        previous_observed = resources._color_correction_last_unique_at_s
        if previous_observed is not None:
            # Accumulate a hold-last interval only at unique-frame boundaries;
            # exact repeated output must never inflate clamp duration.
            observed = max(observed, previous_observed)
            elapsed = observed - previous_observed
            if resources.color_correction_exposure_clamped:
                resources.color_correction_exposure_clamp_time_s += elapsed
            if resources.color_correction_wb_clamped:
                resources.color_correction_wb_clamp_time_s += elapsed

        exposure_clamped = status.get("color_correction_exposure_clamped") is True
        white_balance_clamped = status.get("color_correction_wb_clamped") is True
        resources.color_correction_exposure_clamped = exposure_clamped
        resources.color_correction_wb_clamped = white_balance_clamped
        if exposure_clamped:
            resources.color_correction_exposure_clamp_count += 1
        if white_balance_clamped:
            resources.color_correction_wb_clamp_count += 1
        resources._color_correction_last_unique_at_s = observed

    def _log_fallback_transition(self, kind: str, active: bool, reason: str) -> None:
        """Emit one record only when a fallback state or reason changes."""

        state = (active, reason if active else "")
        previous = self._fallback_log_states.get(kind)
        if previous == state:
            return
        self._fallback_log_states[kind] = state
        if active:
            if kind == "output":
                log.warning(
                    "output fallback active reason=%s; frames are API-only. "
                    "Run the platform setup script to restore the virtual camera",
                    reason or "unknown",
                )
            else:
                log.warning(
                    "%s fallback active reason=%s",
                    kind,
                    reason or "unknown",
                )
        elif previous is not None and previous[0]:
            log.info("%s fallback recovered", kind)

    # -- frame processing ---------------------------------------------
    @staticmethod
    def _validate_canvas_frame(
        frame: np.ndarray,
        canvas_size: Size,
        *,
        boundary: str,
    ) -> np.ndarray:
        try:
            validated = validate_bgr_frame(
                frame,
                name=f"{boundary} frame",
                require_contiguous=True,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        expected_shape = (canvas_size[1], canvas_size[0], 3)
        if validated.shape != expected_shape:
            raise ValueError(
                f"{boundary} frame must match canonical canvas {expected_shape}, "
                f"got {validated.shape}"
            )
        return validated

    @classmethod
    def _validate_output_frame(cls, out: np.ndarray, canvas_size: Size) -> None:
        cls._validate_canvas_frame(
            out,
            canvas_size,
            boundary="processed output",
        )

    @staticmethod
    def _privacy_slate(shape: tuple[int, ...]) -> np.ndarray:
        """Return a fixed opaque slate containing no camera-derived pixels."""

        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"privacy slate requires an HxWx3 shape, got {shape}")
        height, width, _channels = shape
        if height <= 0 or width <= 0:
            raise ValueError(f"privacy slate requires a non-empty shape, got {shape}")
        # A two-tone neutral checker remains visibly a privacy fallback and,
        # unlike a mean-color or blurred fallback, cannot reveal source color,
        # silhouettes, text, or other spatial detail. The pattern also avoids
        # becoming byte-identical to an ordinary uniform camera frame.
        tile = max(2, min(height, width) // 8)
        yy, xx = np.indices((height, width), dtype=np.int32)
        checker = ((yy // tile) + (xx // tile)) & 1
        slate = np.empty((height, width, 3), dtype=np.uint8)
        slate[checker == 0] = (24, 27, 32)
        slate[checker == 1] = (36, 40, 48)
        return slate

    @staticmethod
    def _emergency_blur(frame: np.ndarray) -> np.ndarray:
        """Compatibility name for the input-independent privacy slate."""

        return Pipeline._privacy_slate(frame.shape)

    @staticmethod
    def _validate_mask(
        mask: np.ndarray,
        frame: np.ndarray,
        *,
        privacy_safe: bool,
    ) -> np.ndarray:
        """Validate a mask before any compositor or backdrop can consume it."""

        if (
            not isinstance(mask, np.ndarray)
            or mask.ndim != 2
            or mask.shape != frame.shape[:2]
            or mask.dtype != np.float32
            or mask.size == 0
        ):
            raise _PrivacyViolation(
                "segmentation-invalid-mask",
                "segmenter returned a mask with an invalid type or shape",
            )
        if not np.isfinite(mask).all():
            raise _PrivacyViolation(
                "segmentation-invalid-mask",
                "segmenter returned a non-finite mask",
            )
        minimum = float(np.min(mask))
        maximum = float(np.max(mask))
        if minimum < 0.0 or maximum > 1.0:
            raise _PrivacyViolation(
                "segmentation-invalid-mask",
                "segmenter returned a mask outside [0, 1]",
            )
        validated = np.ascontiguousarray(mask, dtype=np.float32)
        if privacy_safe and bool(np.all(validated >= (1.0 - 1e-6))):
            raise _PrivacyViolation(
                "segmentation-all-foreground",
                "remote fallback mask exposes the entire camera frame",
            )
        return validated

    @classmethod
    def _segment_and_refine_masks(
        cls,
        segmenter: Any,
        refiner: Any,
        frame: np.ndarray,
        *,
        privacy_safe: bool,
        context: SegmentationFrameContext | None = None,
        stage_timings: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and retain both backend and post-refiner alpha stages."""

        started = time.monotonic_ns() if stage_timings is not None else 0
        reset_count_before = int(getattr(segmenter, "temporal_reset_count", 0) or 0)
        raw_mask = (
            segmenter.segment(frame, context=context)
            if isinstance(segmenter, Segmenter)
            else segmenter.segment(frame)
        )
        raw = cls._validate_mask(
            raw_mask,
            frame,
            privacy_safe=False,
        )
        reset_count_after = int(getattr(segmenter, "temporal_reset_count", 0) or 0)
        reset_reason = getattr(segmenter, "last_temporal_reset_reason", None)
        if reset_count_after > reset_count_before and isinstance(
            reset_reason, TemporalResetReason
        ):
            # Stateful backends can discover a discontinuity internally (RVM
            # geometry and provider recovery). Pair the refiner only after a
            # valid alpha exists, and before it sees this exact boundary frame.
            reset_refiner = getattr(refiner, "reset_temporal_state", None)
            if callable(reset_refiner):
                reset_refiner(
                    reset_reason,
                    None if context is None else context.timestamp_ns,
                )
        if stage_timings is not None:
            stage_timings["backend_inference_ms"] = (
                time.monotonic_ns() - started
            ) / 1_000_000.0
        started = time.monotonic_ns() if stage_timings is not None else 0
        refined = (
            refiner.refine(raw, frame, context=context)
            if isinstance(refiner, MaskRefiner)
            else refiner.refine(raw, frame)
        )
        validated = cls._validate_mask(
            refined,
            frame,
            privacy_safe=privacy_safe,
        )
        if stage_timings is not None:
            stage_timings["refinement_ms"] = (
                time.monotonic_ns() - started
            ) / 1_000_000.0
        return raw, validated

    @classmethod
    def _segment_and_refine_mask(
        cls,
        segmenter: Any,
        refiner: Any,
        frame: np.ndarray,
        *,
        privacy_safe: bool,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        """Compatibility wrapper for callers that need only final alpha."""

        _raw, refined = cls._segment_and_refine_masks(
            segmenter,
            refiner,
            frame,
            privacy_safe=privacy_safe,
            context=context,
        )
        return refined

    @staticmethod
    def _segmentation_context(captured: CapturedFrame) -> SegmentationFrameContext:
        return SegmentationFrameContext(
            sequence=captured.sequence,
            timestamp_ns=captured.captured_at_ns,
            generation=captured.generation,
            geometry_generation=captured.geometry_generation,
            shape=captured.pixels.shape[:2],
        )

    @staticmethod
    def _reset_temporal_pair(
        segmenter: Any,
        refiner: Any,
        reason: TemporalResetReason,
        timestamp_ns: int,
    ) -> None:
        """Reset both temporal owners before the discontinuity frame."""

        reset_segmenter = getattr(segmenter, "reset_temporal_state", None)
        if callable(reset_segmenter):
            reset_segmenter(reason, timestamp_ns)
        reset_refiner = getattr(refiner, "reset_temporal_state", None)
        if callable(reset_refiner):
            reset_refiner(reason, timestamp_ns)

    def _segment_resource_masks(
        self,
        resources: _Resources,
        captured: CapturedFrame,
        *,
        privacy_safe: bool,
        stage_timings: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply one timeline decision, then process its exact boundary frame."""

        context = self._segmentation_context(captured)
        checkpoint = resources.segmentation_timeline.checkpoint()
        boundary = resources.segmentation_timeline.observe(context)
        reset_count_before: int | None = None
        try:
            if boundary.reset_reason is not None:
                self._reset_temporal_pair(
                    resources.segmenter,
                    resources.refiner,
                    boundary.reset_reason,
                    context.timestamp_ns,
                )
            reset_count_before = int(
                getattr(resources.segmenter, "temporal_reset_count", 0) or 0
            )
            result = self._segment_and_refine_masks(
                resources.segmenter,
                resources.refiner,
                captured.pixels,
                privacy_safe=privacy_safe,
                context=context,
                stage_timings=stage_timings,
            )
        except BaseException:
            backend_recovered = (
                reset_count_before is not None
                and int(getattr(resources.segmenter, "temporal_reset_count", 0) or 0)
                > reset_count_before
                and getattr(resources.segmenter, "last_temporal_reset_reason", None)
                is TemporalResetReason.BACKEND_RECOVERY
            )
            resources.segmentation_timeline.restore(checkpoint)
            retry_reason = (
                TemporalResetReason.BACKEND_RECOVERY
                if backend_recovered
                else boundary.reset_reason
            )
            if retry_reason is not None:
                # The failed input is rolled back, so publish the reset only
                # when it is paired with the next successfully accepted
                # boundary frame rather than counting the failed attempt too.
                resources.segmentation_timeline.request_reset(retry_reason)
            raise
        backend_recovered = (
            reset_count_before is not None
            and int(getattr(resources.segmenter, "temporal_reset_count", 0) or 0)
            > reset_count_before
            and getattr(resources.segmenter, "last_temporal_reset_reason", None)
            is TemporalResetReason.BACKEND_RECOVERY
        )
        if backend_recovered:
            resources.segmentation_timeline.record_reset(
                TemporalResetReason.BACKEND_RECOVERY
            )
        applied_reset = (
            TemporalResetReason.BACKEND_RECOVERY
            if backend_recovered
            else boundary.reset_reason
        )
        if applied_reset is not None:
            elapsed_gap_ms = (
                "none"
                if boundary.elapsed_ns is None
                else f"{max(0.0, boundary.elapsed_ns / 1_000_000.0):.3f}"
            )
            log.info(
                "matte temporal reset reason=%s sequence_gap=%d "
                "elapsed_gap_ms=%s capture_generation=%d "
                "geometry_generation=%d segmentation_generation=%d "
                "config_version=%d",
                applied_reset.value,
                boundary.sequence_gap,
                elapsed_gap_ms,
                context.generation,
                context.geometry_generation,
                resources.segmentation_generation,
                resources.version,
            )
        return result

    def _matte_evidence_requested(self) -> bool:
        recorder = self._matte_recorder
        monitor = self._matte_monitor
        return bool(
            (recorder is not None and recorder.accepting)
            or (monitor is not None and monitor.accepting)
        )

    def _new_matte_evidence(
        self,
        resources: _Resources,
        captured: CapturedFrame,
    ) -> MatteFrameEvidence | None:
        if not self._matte_evidence_requested():
            return None
        metadata = MatteCaptureMetadata(
            bundle_sequence=self._matte_bundle_sequence,
            capture_sequence=captured.sequence,
            capture_monotonic_ns=captured.captured_at_ns,
            timestamp_source="capture-completion",
            capture_generation=captured.generation,
            geometry_generation=captured.geometry_generation,
        )
        self._matte_bundle_sequence += 1
        cfg = resources.cfg
        return MatteFrameEvidence(
            metadata=metadata,
            raw_frame=captured.pixels,
            configured_controls={
                "segmentation": cfg.segmentation.model_dump(mode="json"),
                "acceleration": cfg.acceleration.model_dump(mode="json"),
                "compositing": cfg.compositing.model_dump(mode="json"),
                "background": {
                    "mode": cfg.background.mode,
                    "fit_mode": cfg.background.fit_mode,
                    "anchor_x": cfg.background.anchor_x,
                    "anchor_y": cfg.background.anchor_y,
                },
            },
            matte_authoritative=False,
            insufficiency_reason="no matte/composite path was recorded",
        )

    @staticmethod
    def _matte_policy_snapshot(resources: _Resources) -> MattePolicySnapshot:
        """Resolve the current policy from actual live backend/runtime facts."""

        segmenter = resources.segmenter
        ratio = getattr(segmenter, "last_downsample_ratio", None)
        return resolve_matte_policy(
            resources.cfg.segmentation,
            resources.cfg.compositing,
            segmenter_matte_backend_kind(segmenter),
            resolved_rvm_ratio=(
                float(ratio)
                if isinstance(ratio, (int, float))
                and not isinstance(ratio, bool)
                and math.isfinite(float(ratio))
                else None
            ),
            passthrough=resources.cfg.background.mode == "passthrough",
            canvas_shape=(resources.canvas_size[1], resources.canvas_size[0]),
            light_wrap_stabilization_eligible=(
                resources.cfg.background.mode in {"video", "camera"}
            ),
        )

    @staticmethod
    def _public_matte_policy(
        resources: _Resources,
        *,
        policy_snapshot: MattePolicySnapshot | None = None,
    ) -> dict[str, object]:
        """Return the versioned, path-free configured/effective matte policy."""

        policy = policy_snapshot or Pipeline._matte_policy_snapshot(resources)
        return {
            "schema": "custback.matte-policy",
            "version": 1,
            "blend_space": resources.cfg.compositing.blend_space,
            **policy.to_dict(),
        }

    @staticmethod
    def _effective_matte_controls(
        resources: _Resources,
        *,
        rvm_telemetry: dict[str, object] | None = None,
        policy_snapshot: MattePolicySnapshot | None = None,
    ) -> dict[str, object]:
        """Project compatibility fields from the typed policy snapshot."""

        segmenter = resources.segmenter
        policy = policy_snapshot or Pipeline._matte_policy_snapshot(resources)
        effective = policy.effective
        refiner_cfg = policy.effective_refiner_config(resources.cfg.segmentation)
        return {
            "segmentation_backend": type(segmenter).__name__,
            "segmentation_device": str(getattr(segmenter, "device", "unknown")),
            "acceleration": _acceleration_evidence(segmenter),
            "rvm_telemetry": (
                rvm_telemetry
                if rvm_telemetry is not None
                else _rvm_telemetry_evidence(segmenter)
            ),
            "output_sink": _output_sink_evidence(
                getattr(resources, "output", None),
            ),
            "produces_matte": (
                policy.selected_backend_kind.value == "true_alpha_recurrent"
                and not policy.passthrough
            ),
            "rvm_downsample_ratio": effective.rvm_downsample_ratio,
            "refiner": refiner_cfg.model_dump(mode="json"),
            "edge_refinement_mode": effective.edge_refinement_mode,
            "edge_refinement_radius_px": effective.edge_refinement_radius_px,
            "mask_shift": effective.mask_shift,
            "use_model_foreground": effective.use_model_foreground,
            "light_wrap": effective.light_wrap,
            "light_wrap_stabilization": Pipeline._light_wrap_evidence(resources),
            "blend_space": resources.cfg.compositing.blend_space,
            "matte_policy": policy.to_dict(),
        }

    @staticmethod
    def _backdrop_diagnostic_identity(resources: _Resources) -> dict[str, object]:
        backdrop = resources.backdrop
        if backdrop is None:
            return {
                "provider": "none",
                "visual_generation": resources.visual_generation,
            }
        identity = getattr(backdrop, "diagnostic_frame_identity", None)
        if callable(identity):
            value = identity()
            if isinstance(value, dict):
                return {
                    **value,
                    "visual_generation": resources.visual_generation,
                }
        return {
            "provider": type(backdrop).__name__,
            "visual_generation": resources.visual_generation,
        }

    @staticmethod
    def _light_wrap_frame_context(
        resources: _Resources,
    ) -> LightWrapFrameContext | None:
        """Resolve typed backdrop timing without consulting diagnostics/stats."""

        backdrop = resources.backdrop
        timing_method = getattr(backdrop, "temporal_frame_timing", None)
        if not callable(timing_method):
            return None
        timing = timing_method()
        if timing is None:
            return None
        if not isinstance(timing, BackdropFrameTiming):
            raise ValueError("backdrop temporal timing contract is invalid")
        return LightWrapFrameContext(
            frame_id=timing.frame_id,
            timestamp_ns=timing.timestamp_ns,
            source_token=(
                resources.light_wrap_generation,
                id(backdrop),
                resources.canvas_size,
            ),
            discontinuity_revision=timing.discontinuity_revision,
        )

    @staticmethod
    def _light_wrap_evidence(
        resources: _Resources,
    ) -> dict[str, object]:
        configured = resources.cfg.compositing.light_wrap_stabilization
        stabilizer = getattr(resources, "light_wrap_stabilizer", None)
        snapshot: LightWrapSnapshot | None = (
            None if stabilizer is None else stabilizer.snapshot()
        )
        effective_mode = (
            configured.mode
            if stabilizer is not None and resources.cfg.compositing.light_wrap > 0.0
            else "off"
        )
        return {
            "configured_mode": configured.mode,
            "effective_mode": effective_mode,
            "time_constant_s": configured.time_constant_s,
            "generation": getattr(resources, "light_wrap_generation", 0),
            "updates": 0 if snapshot is None else snapshot.updates,
            "repeated_frames": (0 if snapshot is None else snapshot.repeated_frames),
            "reset_count": 0 if snapshot is None else snapshot.reset_count,
            "scene_cut_count": (0 if snapshot is None else snapshot.scene_cut_count),
            "last_reset_reason": (
                ""
                if snapshot is None or snapshot.last_reset_reason is None
                else snapshot.last_reset_reason.value
            ),
            "last_dt_s": None if snapshot is None else snapshot.last_dt_s,
            "retained_bytes": (0 if snapshot is None else snapshot.retained_bytes),
        }

    def _submit_matte_evidence(
        self,
        evidence: MatteFrameEvidence | None,
        final_composite: np.ndarray,
    ) -> bool:
        """Submit persistent replay evidence without involving live preview."""

        recorder = self._matte_recorder
        if recorder is not None and evidence is not None:
            return recorder.submit(evidence, final_composite)
        return False

    def _submit_live_matte_evidence(
        self,
        evidence: MatteFrameEvidence | None,
        *,
        status: Mapping[str, object],
    ) -> None:
        """Fan out to the explicit local sink after matching output/status.

        The diagnostic monitor is intentionally independent from FrameHub and
        the configured output sink. A monitor failure must never replace or
        delay an already-published production frame.
        """

        monitor = self._matte_monitor
        if monitor is None or evidence is None or not monitor.accepting:
            return
        try:
            monitor.submit(evidence, status=status)
        except Exception:
            log.exception("cannot submit local matte diagnostic evidence")

    @staticmethod
    def _raw_fingerprint(frame: np.ndarray) -> _RawFingerprint:
        """Build a compact low-pass fingerprint without retaining raw pixels."""

        thumbnail = cv2.resize(
            frame,
            _RAW_FINGERPRINT_SIZE,
            interpolation=cv2.INTER_AREA,
        )
        coarse = cv2.resize(
            frame,
            (4, 3),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint16)
        luminance = (
            coarse[..., 0] * 29 + coarse[..., 1] * 150 + coarse[..., 2] * 77
        ) >> 8
        means = tuple(
            int(value)
            for value in np.rint(thumbnail.mean(axis=(0, 1), dtype=np.float64))
        )
        jpeg_features = tuple(int(value) for value in luminance.reshape(-1)) + means
        thumbnail.setflags(write=False)
        return _RawFingerprint(
            jpeg_features=jpeg_features,
            thumbnail=thumbnail,
        )

    def _reset_raw_replay_history(self) -> None:
        """Begin a new pipeline privacy session after renderer invalidation."""

        self._recent_raw_fingerprints.clear()
        self._privacy_history_exhausted = False
        self._privacy_invalidated_session = None
        self._latest_raw_frame = None

    def _remember_raw_frame(self, frame: np.ndarray) -> bool:
        # Keep one current copy for the precise near-raw comparison. Historical
        # frames are represented only by small, non-reversible fingerprints.
        self._latest_raw_frame = frame.copy()
        if self._privacy_history_exhausted:
            return False
        if len(self._recent_raw_fingerprints) >= self._raw_fingerprint_capacity:
            # Never evict evidence into a permissive state.  The caller revokes
            # the renderer lease before any subsequent publication can proceed.
            self._privacy_history_exhausted = True
            return False
        self._recent_raw_fingerprints.append(self._raw_fingerprint(frame))
        return True

    def _record_remote_raw_frame(self, frame: np.ndarray) -> None:
        """Remember frames exposed to a renderer and revoke on exhaustion."""

        session = self.hub.active_remote_session()
        if session is None:
            self._latest_raw_frame = frame.copy()
            return
        if self._remember_raw_frame(frame):
            return
        if self._privacy_invalidated_session == session:
            return
        self.hub.invalidate_remote_session(session)
        self._privacy_invalidated_session = session

    def _publish_remote_raw_frame(self, frame: np.ndarray, raw_epoch: int) -> bool:
        """Atomically admit and expose raw pixels to the exclusive renderer."""

        admitted_session = 0

        def admit(session: int) -> bool:
            nonlocal admitted_session
            admitted_session = session
            if self._remember_raw_frame(frame):
                return True
            self._privacy_invalidated_session = session
            return False

        published = self.hub.publish_remote_raw(
            frame,
            raw_epoch,
            admit=admit,
        )
        if not published and admitted_session > 0:
            self.hub.invalidate_remote_session(admitted_session)
        return published

    @staticmethod
    def _lowres_raw_similarity(
        candidate: np.ndarray,
        raw_thumbnail: np.ndarray,
    ) -> bool:
        candidate_thumbnail = cv2.resize(
            candidate,
            _RAW_FINGERPRINT_SIZE,
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        reference = raw_thumbnail.astype(np.float32)
        mean_delta = float(np.abs(candidate_thumbnail - reference).mean())
        if mean_delta > _RAW_ECHO_LOWRES_MEAN_DELTA:
            return False
        candidate_centered = candidate_thumbnail - float(candidate_thumbnail.mean())
        reference_centered = reference - float(reference.mean())
        denominator = float(
            np.sqrt(
                np.sum(candidate_centered * candidate_centered, dtype=np.float64)
                * np.sum(reference_centered * reference_centered, dtype=np.float64)
            )
        )
        if denominator <= 1e-9:
            # Uniform/near-uniform frames have no stable correlation; a small
            # low-resolution delta is sufficient to treat them as an echo.
            return mean_delta <= _RAW_ECHO_MEAN_DELTA
        correlation = float(
            np.sum(
                candidate_centered * reference_centered,
                dtype=np.float64,
            )
            / denominator
        )
        return correlation >= _RAW_ECHO_LOWRES_CORRELATION

    @staticmethod
    def _is_near_raw(candidate: np.ndarray, raw: np.ndarray) -> bool:
        if candidate.shape != raw.shape:
            return False
        delta = np.abs(candidate.astype(np.int16) - raw.astype(np.int16))
        if not np.any(delta):
            return True
        changed = np.any(delta > _RAW_ECHO_PIXEL_TOLERANCE, axis=2)
        changed_fraction = float(np.count_nonzero(changed)) / changed.size
        pixel_near = (
            changed_fraction <= _RAW_ECHO_CHANGED_FRACTION
            or float(delta.mean()) <= _RAW_ECHO_MEAN_DELTA
        )
        if pixel_near:
            return True
        return Pipeline._lowres_raw_similarity(
            candidate,
            Pipeline._raw_fingerprint(raw).thumbnail,
        )

    def _matches_recent_raw(self, candidate: np.ndarray) -> bool:
        if not self._recent_raw_fingerprints:
            return False
        candidate_fingerprint = self._raw_fingerprint(candidate)
        return self._recent_raw_fingerprints.matches(candidate_fingerprint)

    def _guard_remote_output(
        self,
        candidate: np.ndarray,
        raw: np.ndarray,
        *,
        privacy_safe: bool,
    ) -> tuple[np.ndarray, str]:
        """Apply the final fail-closed gate shared by every output sink."""

        if not privacy_safe:
            return candidate, ""
        if self._privacy_history_exhausted:
            return self._privacy_slate(raw.shape), "privacy-history-exhausted"
        try:
            self._validate_output_frame(candidate, (raw.shape[1], raw.shape[0]))
        except Exception:
            return self._privacy_slate(raw.shape), "privacy-invalid-output"
        if self._is_near_raw(candidate, raw):
            return self._privacy_slate(raw.shape), "privacy-raw-echo"
        if self._matches_recent_raw(candidate):
            return self._privacy_slate(raw.shape), "privacy-delayed-raw-echo"
        return candidate, ""

    def _render_local_mode(
        self,
        resources: _Resources,
        captured: CapturedFrame,
        *,
        color_outcome: dict[str, object] | None = None,
        matte_evidence: MatteFrameEvidence | None = None,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        frame = captured.pixels
        mode = resources.cfg.background.mode
        if mode == "passthrough":
            return frame
        if mode == "remote" and isinstance(resources.segmenter, NullSegmenter):
            return self._emergency_blur(frame)
        out, _ = self._local_composite(
            resources,
            frame,
            captured=captured,
            privacy_safe=mode == "remote",
            color_outcome=color_outcome,
            matte_evidence=matte_evidence,
            timings=timings,
        )
        return out

    @staticmethod
    def _composite_prepared_color(
        cfg: AppConfig,
        frame: np.ndarray,
        backdrop_frame: np.ndarray,
        mask: np.ndarray,
        edge_foreground: np.ndarray | None,
        prepared: _PreparedColorFrame,
        *,
        light_wrap: float,
        prepared_light_wrap: PreparedLightWrap | None = None,
        legacy_workspace: LegacyCompositorWorkspace | None = None,
        diagnostics: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Render with shared decoded inputs when correction prepared them."""

        compositing = cfg.compositing
        foreground_linear = prepared.foreground_linear_bgr
        backdrop_linear = prepared.backdrop_linear_bgr
        edge_linear = prepared.edge_foreground_linear_bgr
        edge_pair_ready = edge_foreground is None or edge_linear is not None
        if compositing.blend_space == "linear_srgb":
            if foreground_linear is None:
                foreground_linear = bgr_u8_to_linear_bgr(frame)
            if backdrop_linear is None:
                backdrop_linear = bgr_u8_to_linear_bgr(backdrop_frame)
            if edge_foreground is not None and edge_linear is None:
                edge_linear = bgr_u8_to_linear_bgr(edge_foreground)
            return composite_linear_predecoded(
                frame,
                backdrop_frame,
                mask,
                foreground_linear_bgr=foreground_linear,
                backdrop_linear_bgr=backdrop_linear,
                light_wrap=light_wrap,
                edge_foreground_bgr=edge_foreground,
                edge_foreground_linear_bgr=edge_linear,
                color_transform=prepared.transform,
                prepared_light_wrap=prepared_light_wrap,
                diagnostics=diagnostics,
            )
        if (
            compositing.blend_space == "srgb_legacy"
            and foreground_linear is not None
            and edge_pair_ready
        ):
            return composite_legacy_predecoded(
                frame,
                backdrop_frame,
                mask,
                foreground_linear_bgr=foreground_linear,
                light_wrap=light_wrap,
                edge_foreground_bgr=edge_foreground,
                edge_foreground_linear_bgr=edge_linear,
                color_transform=prepared.transform,
                prepared_light_wrap=prepared_light_wrap,
                workspace=legacy_workspace,
                diagnostics=diagnostics,
            )
        return composite(
            frame,
            backdrop_frame,
            mask,
            light_wrap=light_wrap,
            edge_foreground=edge_foreground,
            blend_space=compositing.blend_space,
            color_transform=prepared.transform,
            prepared_light_wrap=prepared_light_wrap,
            workspace=legacy_workspace,
            diagnostics=diagnostics,
        )

    def _prepare_color_frame(
        self,
        resources: _Resources,
        frame: np.ndarray,
        backdrop_frame: np.ndarray,
        mask: np.ndarray,
        edge_foreground: np.ndarray | None,
        *,
        now_s: float,
        captured: CapturedFrame | None = None,
        cfg: AppConfig | None = None,
        harmonizer: ColorHarmonizer | None = None,
        track_live_state: bool = True,
    ) -> _PreparedColorFrame:
        """Estimate one transform and retain decoded inputs for the compositor.

        Expected low-confidence estimates are state-machine inputs, not errors.
        Unexpected estimator failures are isolated here so structural
        compositor/output validation remains strict.
        """

        effective_cfg = resources.cfg if cfg is None else cfg
        correction = effective_cfg.compositing.color_correction
        if correction.mode != "auto" or effective_cfg.background.mode not in {
            "image",
            "video",
            "camera",
        }:
            return _PreparedColorFrame()

        active_harmonizer = harmonizer or resources.harmonizer
        if active_harmonizer is None:  # defensive resource invariant
            raise RuntimeError("active color harmonizer is missing")

        foreground_linear_bgr: np.ndarray | None = None
        backdrop_linear_bgr: np.ndarray | None = None
        backdrop_analysis_linear_bgr: np.ndarray | None = None
        edge_linear_bgr: np.ndarray | None = None
        source_generation: int | None = None
        pending_reset_token: tuple[object, ...] | None = None
        reset_required = False
        exposure_clamped = False
        white_balance_clamped = False
        try:
            foreground_content_rect: tuple[int, int, int, int] | None = None
            backdrop_content_rect: Any = None
            backdrop_geometry_token: object = None
            if track_live_state:
                geometry_generation = 0
                if captured is not None:
                    source_generation = captured.generation
                    foreground_content_rect = captured.content_rect
                    geometry_generation = captured.geometry_generation
                provider = resources.backdrop
                if provider is not None and hasattr(provider, "content_rect"):
                    backdrop_content_rect = provider.content_rect(
                        resources.canvas_size[0],
                        resources.canvas_size[1],
                    )
                    transform_plan = getattr(provider, "transform_plan", None)
                    backdrop_geometry_token = (
                        transform_plan(
                            resources.canvas_size[0],
                            resources.canvas_size[1],
                        )
                        if callable(transform_plan)
                        else backdrop_content_rect
                    )
                pending_reset_token = (
                    resources.visual_generation,
                    source_generation,
                    geometry_generation,
                    foreground_content_rect,
                    id(provider),
                    backdrop_geometry_token,
                    resources.canvas_size,
                )
                reset_required = pending_reset_token != resources.color_reset_token

            foreground_linear_bgr = bgr_u8_to_linear_bgr(frame)
            backdrop_linear_bgr = bgr_u8_to_linear_bgr(backdrop_frame)
            if cfg is None and harmonizer is None and track_live_state:
                backdrop_analysis_linear_bgr = resources.image_backdrop_analysis(
                    backdrop_frame,
                    backdrop_linear_bgr,
                )
            if edge_foreground is not None:
                edge_linear_bgr = bgr_u8_to_linear_bgr(edge_foreground)
            estimate = estimate_color_transform_linear(
                foreground_linear_bgr,
                backdrop_linear_bgr,
                mask,
                mode=effective_cfg.background.mode,
                strength=correction.strength,
                exposure_limit_ev=correction.exposure_limit_ev,
                white_balance_strength=correction.white_balance_strength,
                foreground_content_rect=foreground_content_rect,
                backdrop_content_rect=backdrop_content_rect,
                resize_executor=resources.color_analysis_executor,
                backdrop_analysis_linear_bgr=backdrop_analysis_linear_bgr,
            )
            if reset_required:
                transform = active_harmonizer.reset_and_update(
                    estimate,
                    now_s,
                    source_generation=source_generation,
                )
                resources.color_reset_token = pending_reset_token
            else:
                transform = active_harmonizer.update(
                    estimate,
                    now_s,
                    source_generation=source_generation,
                )
            exposure_clamped = getattr(estimate, "exposure_clamped", False) is True
            white_balance_clamped = (
                getattr(estimate, "white_balance_clamped", False) is True
            )
            if track_live_state:
                self._log_fallback_transition(
                    "color-correction",
                    False,
                    "",
                )
        except Exception as exc:
            try:
                if reset_required:
                    active_harmonizer.reset(
                        now_s,
                        reason=ColorReason.INVALID,
                        source_generation=source_generation,
                    )
                    resources.color_reset_token = pending_reset_token
                transform = active_harmonizer.on_error(
                    now_s,
                    source_generation=source_generation,
                )
            except Exception:
                transform = IDENTITY_TRANSFORM
            if track_live_state:
                self._log_fallback_transition(
                    "color-correction",
                    True,
                    type(exc).__name__,
                )
        return _PreparedColorFrame(
            transform=transform,
            foreground_linear_bgr=foreground_linear_bgr,
            backdrop_linear_bgr=backdrop_linear_bgr,
            edge_foreground_linear_bgr=edge_linear_bgr,
            snapshot=active_harmonizer.snapshot(),
            exposure_clamped=exposure_clamped,
            white_balance_clamped=white_balance_clamped,
        )

    def _preflight(
        self,
        resources: _Resources,
        *,
        send_output: bool = True,
    ) -> _PreflightResult:
        """Read and process a real frame before reporting startup readiness."""
        accepted_started_ns = time.monotonic_ns()
        camera_wait = (
            2.0
            if resources.cfg.camera.synthetic
            else resources.cfg.camera.recovery_timeout_s + 2.0
        )
        deadline = time.monotonic() + camera_wait
        captured: CapturedFrame | None = None
        while (
            captured is None and time.monotonic() < deadline and not self._stop.is_set()
        ):
            captured = resources.capture.read()
            if captured is None:
                self._stop.wait(0.05)
        if captured is None:
            raise ActivationError("capture returned no frame during startup preflight")
        diagnostic_frame_started_ns = (
            time.monotonic_ns() if self._matte_evidence_requested() else None
        )
        frame = captured.pixels
        try:
            frame = self._validate_canvas_frame(
                frame,
                resources.canvas_size,
                boundary="capture",
            )
        except ValueError as exc:
            raise ActivationError(str(exc)) from exc
        resources.capture_sequence_timeline.observe(captured.sequence)
        remote_mode = resources.cfg.background.mode == "remote"
        matte_evidence = self._new_matte_evidence(
            resources,
            captured,
        )
        privacy_reason = ""
        color_outcome: dict[str, object] = {}
        timings: dict[str, float] = {
            **(
                {"capture_read_ms": captured.read_ms}
                if captured.read_ms is not None
                else {}
            ),
            "segmentation_ms": 0.0,
            "background_ms": 0.0,
            "color_correction_ms": 0.0,
            "composite_prepare_ms": 0.0,
            "composite_blend_ms": 0.0,
            "composite_ms": 0.0,
        }
        segmentation_sequence_before = (
            resources.segmentation_timeline.snapshot().last_sequence
        )
        process_started_ns = time.monotonic_ns()
        try:
            if resources.cfg.background.mode == "passthrough":
                # Passthrough does not need a mask to render, but the segmenter
                # is already part of the hot-swappable resource generation.
                # Exercise it so a later background-only PATCH cannot reveal a
                # backend failure for the first time.
                self._segment_resource_masks(
                    resources,
                    captured,
                    privacy_safe=False,
                )
            if remote_mode:
                # Exercise the configured fallback without publishing it. A
                # remote startup probes the real output backend only with the
                # fixed slate, never with a camera-derived composite.
                _candidate, privacy_reason = self._local_composite(
                    resources,
                    frame,
                    captured=captured,
                    privacy_safe=True,
                    timings=timings,
                    color_outcome=color_outcome,
                )
                out = self._privacy_slate(frame.shape)
                privacy_reason = privacy_reason or "startup-slate"
            else:
                self._latest_raw_frame = frame.copy()
                out = self._render_local_mode(
                    resources,
                    captured,
                    color_outcome=color_outcome,
                    matte_evidence=matte_evidence,
                    timings=timings,
                )
            validation_started_ns = time.monotonic_ns()
            self._validate_output_frame(out, resources.canvas_size)
            if matte_evidence is not None:
                matte_evidence.timings_ms["pre_guard_output_validation_ms"] = (
                    time.monotonic_ns() - validation_started_ns
                ) / 1_000_000.0
        except _PrivacyViolation as exc:
            raise ActivationError(f"invalid mask: {exc}") from exc
        segmentation_updated = (
            resources.segmentation_timeline.snapshot().last_sequence
            != segmentation_sequence_before
        )
        frame_processing_ms = (time.monotonic_ns() - process_started_ns) / 1_000_000.0
        # The privacy gate sits at the final publication boundary. Startup uses
        # the same fail-closed path as the steady-state loop, so a bad mask or
        # raw-looking preflight result can never reach the real output backend.
        guard_started_ns = time.monotonic_ns()
        out, gate_reason = self._guard_remote_output(
            out,
            frame,
            privacy_safe=remote_mode,
        )
        privacy_reason = gate_reason or privacy_reason
        self._validate_output_frame(out, resources.canvas_size)
        guard_output_validation_ms = (
            time.monotonic_ns() - guard_started_ns
        ) / 1_000_000.0
        base_ready_at_ns = time.monotonic_ns()
        if send_output:
            send_started = time.monotonic_ns()
            send_timing = _send_output_with_timing(
                resources.output,
                out,
                copy_frame=remote_mode,
            )
            output_send_ms = (send_timing.completed_at_ns - send_started) / 1_000_000.0
        else:
            # Production publication is owned by OutputPublisher.  Preserve the
            # preflight result shape for focused synchronous callers without
            # pretending the compute lane submitted a frame.
            send_timing = OutputSendTiming(
                submitted_at_ns=base_ready_at_ns,
                completed_at_ns=base_ready_at_ns,
                submission_ms=0.0,
                pacing_wait_ms=0.0,
            )
            output_send_ms = 0.0
        new_frame_service_ms = (
            send_timing.submitted_at_ns - accepted_started_ns
        ) / 1_000_000.0
        new_frame_serialized_loop_ms = (
            send_timing.completed_at_ns - accepted_started_ns
        ) / 1_000_000.0
        if matte_evidence is not None:
            matte_evidence.timings_ms["output_send_ms"] = output_send_ms
            matte_evidence.timings_ms["guard_output_validation_ms"] = (
                guard_output_validation_ms
            )
            matte_evidence.timings_ms["output_submission_ms"] = (
                send_timing.submission_ms
            )
            matte_evidence.timings_ms["output_sink_pacing_wait_ms"] = (
                send_timing.pacing_wait_ms
            )
            matte_evidence.timings_ms["application_pacing_wait_ms"] = 0.0
            matte_evidence.timings_ms["output_schedule_lateness_ms"] = 0.0
            matte_evidence.timings_ms["frame_processing_ms"] = frame_processing_ms
            matte_evidence.timings_ms["new_frame_service_ms"] = new_frame_service_ms
            matte_evidence.timings_ms["new_frame_serialized_loop_ms"] = (
                new_frame_serialized_loop_ms
            )
            if diagnostic_frame_started_ns is not None:
                matte_evidence.timings_ms["frame_total_ms"] = (
                    send_timing.completed_at_ns - diagnostic_frame_started_ns
                ) / 1_000_000.0
            rss_bytes = process_rss_bytes()
            if rss_bytes is not None:
                matte_evidence.resource_samples["rss_bytes"] = rss_bytes
        if send_output and self._submit_matte_evidence(matte_evidence, out):
            assert matte_evidence is not None
            self._matte_last_source_sequence = matte_evidence.metadata.bundle_sequence
            recorder = self._matte_recorder
            assert recorder is not None
            recorder.submit_output_event(
                sent_monotonic_ns=send_timing.submitted_at_ns,
                source_bundle_sequence=self._matte_last_source_sequence,
                base_updated=True,
                exact_final_repeat=False,
            )
        if not color_outcome:
            snapshot = (
                resources.harmonizer.snapshot()
                if resources.harmonizer is not None
                else None
            )
            color_outcome.update(_color_stats(resources.cfg, snapshot))
        self._record_color_output(
            resources,
            color_outcome,
            processed=True,
            observed_at_s=captured.captured_at_ns / 1_000_000_000.0,
        )
        if privacy_reason:
            log.warning("remote privacy fallback active reason=%s", privacy_reason)
        resources.background_startup_asset_guard = False
        background_fallback_reason = getattr(
            resources,
            "background_fallback_reason",
            "",
        )
        if background_fallback_reason:
            matte_evidence = None
        return _PreflightResult(
            output=out,
            captured=captured,
            send_timing=send_timing,
            base_ready_at_ns=base_ready_at_ns,
            segmentation_updated=segmentation_updated,
            frame_processing_ms=frame_processing_ms,
            new_frame_service_ms=new_frame_service_ms,
            new_frame_serialized_loop_ms=new_frame_serialized_loop_ms,
            timings=timings,
            color_status=color_outcome,
            remote_fallback_active=remote_mode,
            remote_fallback_reason=("no-client" if remote_mode else ""),
            matte_evidence=matte_evidence,
        )

    def _local_composite(
        self,
        resources: _Resources,
        frame: np.ndarray,
        *,
        captured: CapturedFrame | None = None,
        privacy_safe: bool,
        timings: dict[str, float] | None = None,
        color_outcome: dict[str, object] | None = None,
        matte_evidence: MatteFrameEvidence | None = None,
    ) -> tuple[np.ndarray, str]:
        cfg = resources.cfg
        backdrop = resources.backdrop
        background_fallback_reason = getattr(
            resources,
            "background_fallback_reason",
            "",
        )
        if background_fallback_reason:
            # Persisted image/video activation failed before any camera pixels
            # were eligible for composition.  Keep this path input-independent
            # until a valid transactional background replacement commits.
            return (
                self._privacy_slate(resources.canvas_shape),
                background_fallback_reason,
            )
        if privacy_safe and isinstance(resources.segmenter, NullSegmenter):
            return self._emergency_blur(frame), "segmentation-none"

        try:
            started = time.monotonic_ns()
            diagnostic_stage_timings: dict[str, float] | None = (
                {} if matte_evidence is not None else None
            )
            if captured is None:
                raw_mask, mask = self._segment_and_refine_masks(
                    resources.segmenter,
                    resources.refiner,
                    frame,
                    privacy_safe=privacy_safe,
                    stage_timings=diagnostic_stage_timings,
                )
            else:
                raw_mask, mask = self._segment_resource_masks(
                    resources,
                    captured,
                    privacy_safe=privacy_safe,
                    stage_timings=diagnostic_stage_timings,
                )
            rvm_telemetry: dict[str, object] | None = None
            if matte_evidence is not None:
                # Capture one sanitized immutable snapshot for this exact
                # successful segmentation. Reuse it for both timing and
                # controls so later code cannot accidentally attribute a
                # different inference to this frame.
                rvm_telemetry = _rvm_telemetry_evidence(resources.segmenter)
                if (
                    diagnostic_stage_timings is not None
                    and rvm_telemetry.get("applicable") is True
                ):
                    for evidence_key, timing_key in (
                        ("preprocess_ms", "rvm_preprocess_ms"),
                        ("session_run_ms", "rvm_session_run_ms"),
                        ("postprocess_ms", "rvm_postprocess_ms"),
                    ):
                        value = rvm_telemetry.get(evidence_key)
                        if isinstance(value, float):
                            diagnostic_stage_timings[timing_key] = value
            if timings is not None:
                timings["segmentation_ms"] = (
                    time.monotonic_ns() - started
                ) / 1_000_000.0
            started = time.monotonic_ns()
            if isinstance(backdrop, BlurBackdrop):
                backdrop.set_source_frame(frame, mask)
            if backdrop is None:
                if timings is not None:
                    timings["background_ms"] = (
                        time.monotonic_ns() - started
                    ) / 1_000_000.0
                if privacy_safe:
                    return self._emergency_blur(frame), "local-failure"
                return frame, ""
            try:
                bg = backdrop.frame(
                    resources.canvas_size[0],
                    resources.canvas_size[1],
                )
            except (OSError, RuntimeError, ValueError):
                effective_mode = (
                    cfg.background.remote_fallback_mode
                    if cfg.background.mode == "remote"
                    else cfg.background.mode
                )
                if not (
                    resources.background_startup_asset_guard
                    and effective_mode in {"image", "video"}
                ):
                    raise
                # Some video containers open successfully but cannot decode a
                # first frame.  Treat only this startup provider boundary as
                # asset unavailability; unrelated segmentation/compositor
                # exceptions retain their existing fail-fast behavior.
                resources.retire_background_video_provider(backdrop)
                resources.backdrop = None
                resources.background_fallback_reason = "asset-unavailable"
                _safe_close(backdrop, "unavailable startup backdrop")
                if timings is not None:
                    timings["background_ms"] = (
                        time.monotonic_ns() - started
                    ) / 1_000_000.0
                return (
                    self._privacy_slate(resources.canvas_shape),
                    resources.background_fallback_reason,
                )
            self._validate_canvas_frame(
                bg,
                resources.canvas_size,
                boundary="backdrop",
            )
            if timings is not None:
                timings["background_ms"] = (time.monotonic_ns() - started) / 1_000_000.0
            matte_policy = self._matte_policy_snapshot(resources)
            backend_clean_foreground = resources.segmenter.last_foreground
            edge_fg = (
                backend_clean_foreground
                if matte_policy.effective.use_model_foreground
                else None
            )
            if edge_fg is not None:
                edge_fg = self._validate_canvas_frame(
                    edge_fg,
                    resources.canvas_size,
                    boundary="model foreground",
                )
            correction = cfg.compositing.color_correction
            color_eligible = correction.mode == "auto" and cfg.background.mode in {
                "image",
                "video",
                "camera",
            }
            if color_eligible:
                started = time.monotonic_ns()
                prepared_color = self._prepare_color_frame(
                    resources,
                    frame,
                    bg,
                    mask,
                    edge_fg,
                    now_s=(
                        captured.captured_at_ns / 1_000_000_000.0
                        if captured is not None
                        else time.monotonic()
                    ),
                    captured=captured,
                )
                if timings is not None:
                    timings["color_correction_ms"] = (
                        time.monotonic_ns() - started
                    ) / 1_000_000.0
            else:
                prepared_color = _PreparedColorFrame()
                if timings is not None:
                    timings["color_correction_ms"] = 0.0
            composite_started_ns = time.monotonic_ns()
            prepare_started_ns = composite_started_ns
            compositor_diagnostics = (
                matte_evidence.compositor_substages_ms
                if matte_evidence is not None
                else {name: 0.0 for name in COMPOSITOR_SUBSTAGE_NAMES}
            )
            blend_started_ns: int | None = None
            next_light_wrap_stabilizer: LightWrapStabilizer | None = None
            diagnostic_prepared_light_wrap: PreparedLightWrap | None = None
            diagnostic_color_transform = prepared_color.transform
            legacy_workspace = (
                resources.legacy_compositor_workspace()
                if cfg.compositing.blend_space == "srgb_legacy"
                and isinstance(resources, _Resources)
                else None
            )
            try:
                prepared_light_wrap: PreparedLightWrap | None = None
                active_light_wrap_stabilizer = getattr(
                    resources,
                    "light_wrap_stabilizer",
                    None,
                )
                if (
                    matte_policy.effective.light_wrap > 0.0
                    and active_light_wrap_stabilizer is not None
                ):
                    wrap_context = self._light_wrap_frame_context(resources)
                    if wrap_context is not None:
                        candidate_light_wrap_stabilizer: LightWrapStabilizer = (
                            active_light_wrap_stabilizer.clone()
                        )
                        next_light_wrap_stabilizer = candidate_light_wrap_stabilizer
                        backdrop_linear = prepared_color.backdrop_linear_bgr
                        if (
                            cfg.compositing.blend_space == "linear_srgb"
                            and backdrop_linear is None
                        ):
                            backdrop_linear = bgr_u8_to_linear_bgr(bg)
                            prepared_color = _PreparedColorFrame(
                                transform=prepared_color.transform,
                                foreground_linear_bgr=(
                                    prepared_color.foreground_linear_bgr
                                ),
                                backdrop_linear_bgr=backdrop_linear,
                                edge_foreground_linear_bgr=(
                                    prepared_color.edge_foreground_linear_bgr
                                ),
                                snapshot=prepared_color.snapshot,
                                exposure_clamped=prepared_color.exposure_clamped,
                                white_balance_clamped=(
                                    prepared_color.white_balance_clamped
                                ),
                            )
                        prepared_light_wrap = prepare_light_wrap(
                            bg,
                            blend_space=cfg.compositing.blend_space,
                            stabilizer=candidate_light_wrap_stabilizer,
                            context=wrap_context,
                            backdrop_linear_bgr=(
                                backdrop_linear
                                if cfg.compositing.blend_space == "linear_srgb"
                                else None
                            ),
                            diagnostics=compositor_diagnostics,
                        )
                elif matte_policy.effective.light_wrap > 0.0 and isinstance(
                    resources, _Resources
                ):
                    backdrop_linear = prepared_color.backdrop_linear_bgr
                    if (
                        cfg.compositing.blend_space == "linear_srgb"
                        and backdrop_linear is None
                    ):
                        backdrop_linear = bgr_u8_to_linear_bgr(bg)
                        prepared_color = _PreparedColorFrame(
                            transform=prepared_color.transform,
                            foreground_linear_bgr=(
                                prepared_color.foreground_linear_bgr
                            ),
                            backdrop_linear_bgr=backdrop_linear,
                            edge_foreground_linear_bgr=(
                                prepared_color.edge_foreground_linear_bgr
                            ),
                            snapshot=prepared_color.snapshot,
                            exposure_clamped=prepared_color.exposure_clamped,
                            white_balance_clamped=(
                                prepared_color.white_balance_clamped
                            ),
                        )
                    prepared_light_wrap = resources.static_light_wrap_sample(
                        bg,
                        blend_space=cfg.compositing.blend_space,
                        backdrop_linear_bgr=backdrop_linear,
                        diagnostics=compositor_diagnostics,
                    )
                if timings is not None:
                    timings["composite_prepare_ms"] = (
                        time.monotonic_ns() - prepare_started_ns
                    ) / 1_000_000.0
                blend_started_ns = time.monotonic_ns()
                rendered = self._composite_prepared_color(
                    cfg,
                    frame,
                    bg,
                    mask,
                    edge_fg,
                    prepared_color,
                    light_wrap=matte_policy.effective.light_wrap,
                    prepared_light_wrap=prepared_light_wrap,
                    legacy_workspace=legacy_workspace,
                    diagnostics=compositor_diagnostics,
                )
                diagnostic_prepared_light_wrap = prepared_light_wrap
            except ColorError as exc:
                # A photometric transform/conversion failure follows the same
                # bounded correction fallback, while structural ValueError
                # contracts remain strict and are never swallowed here.
                self._log_fallback_transition(
                    "color-correction",
                    True,
                    type(exc).__name__,
                )
                # Preparation is transactional with the render. A fallback
                # that did not consume the prepared sample cannot publish its
                # candidate temporal history.
                next_light_wrap_stabilizer = None
                diagnostic_color_transform = IDENTITY_TRANSFORM
                if blend_started_ns is None:
                    if timings is not None:
                        timings["composite_prepare_ms"] = (
                            time.monotonic_ns() - prepare_started_ns
                        ) / 1_000_000.0
                    blend_started_ns = time.monotonic_ns()
                fallback_prepared_light_wrap = (
                    prepared_light_wrap
                    if prepared_light_wrap is not None
                    and not prepared_light_wrap.stabilized
                    else None
                )
                rendered = composite(
                    frame,
                    bg,
                    mask,
                    light_wrap=matte_policy.effective.light_wrap,
                    edge_foreground=edge_fg,
                    blend_space=cfg.compositing.blend_space,
                    color_transform=IDENTITY_TRANSFORM,
                    prepared_light_wrap=fallback_prepared_light_wrap,
                    workspace=legacy_workspace,
                    diagnostics=compositor_diagnostics,
                )
                diagnostic_prepared_light_wrap = fallback_prepared_light_wrap
                if color_outcome is not None:
                    color_outcome.update(
                        _color_stats(
                            cfg,
                            prepared_color.snapshot,
                            applied_transform=IDENTITY_TRANSFORM,
                            application_failed=True,
                            exposure_clamped=prepared_color.exposure_clamped,
                            white_balance_clamped=(
                                prepared_color.white_balance_clamped
                            ),
                        )
                    )
            else:
                if color_outcome is not None:
                    color_outcome.update(
                        _color_stats(
                            cfg,
                            prepared_color.snapshot,
                            applied_transform=prepared_color.transform,
                            exposure_clamped=prepared_color.exposure_clamped,
                            white_balance_clamped=(
                                prepared_color.white_balance_clamped
                            ),
                        )
                    )
            if timings is not None:
                assert blend_started_ns is not None
                timings["composite_blend_ms"] = (
                    time.monotonic_ns() - blend_started_ns
                ) / 1_000_000.0
            validation_started_ns = (
                time.monotonic_ns() if matte_evidence is not None else 0
            )
            self._validate_output_frame(rendered, resources.canvas_size)
            if matte_evidence is not None:
                matte_evidence.timings_ms["post_composite_validation_ms"] = (
                    time.monotonic_ns() - validation_started_ns
                ) / 1_000_000.0
            if next_light_wrap_stabilizer is not None:
                resources.light_wrap_stabilizer = next_light_wrap_stabilizer
            if timings is not None:
                timings["composite_ms"] = (
                    time.monotonic_ns() - composite_started_ns
                ) / 1_000_000.0
            resources.last_compositor_substages_ms = {
                name: float(compositor_diagnostics.get(name, 0.0))
                for name in COMPOSITOR_SUBSTAGE_NAMES
            }
            if matte_evidence is not None and not privacy_safe:
                matte_evidence.raw_mask = raw_mask
                matte_evidence.refined_mask = mask
                matte_evidence.clean_foreground = backend_clean_foreground
                matte_evidence.backdrop_frame = bg
                matte_evidence.base_composite = rendered
                matte_evidence.prepared_light_wrap = diagnostic_prepared_light_wrap
                matte_evidence.segmentation_diagnostics = (
                    segmenter_diagnostics_snapshot(resources.segmenter)
                )
                matte_evidence.effective_controls = self._effective_matte_controls(
                    resources,
                    rvm_telemetry=rvm_telemetry,
                )
                legacy_workspace = resources._legacy_compositor_workspace
                if legacy_workspace is not None:
                    workspace_snapshot = legacy_workspace.snapshot()
                    # These are application-visible lower bounds: retained
                    # workspace bytes plus the known independently owned output
                    # allocation. Native OpenCV internals are not guessed.
                    matte_evidence.resource_samples["memory_bytes"] = (
                        workspace_snapshot.retained_bytes
                    )
                    matte_evidence.resource_samples["allocation_bytes"] = (
                        workspace_snapshot.last_known_allocation_bytes
                    )
                if diagnostic_stage_timings is not None:
                    matte_evidence.timings_ms.update(diagnostic_stage_timings)
                if timings is not None:
                    matte_evidence.timings_ms.update(timings)
                matte_evidence.backdrop_identity = self._backdrop_diagnostic_identity(
                    resources
                )
                matte_evidence.color_transform = diagnostic_color_transform
                matte_evidence.matte_authoritative = True
                matte_evidence.insufficiency_reason = ""
            return (
                rendered,
                "",
            )
        except _PrivacyViolation as exc:
            if not privacy_safe:
                raise
            log.warning("local remote-mode privacy fallback reason=%s", exc.reason)
            return self._privacy_slate(frame.shape), exc.reason
        except Exception:
            if not privacy_safe:
                raise
            log.exception("local remote-mode fallback failed")
            return self._privacy_slate(frame.shape), "local-failure"

    def _privacy_checked(
        self, candidate: np.ndarray, frame: np.ndarray, privacy_safe: bool
    ) -> np.ndarray:
        """Compatibility wrapper for direct privacy-gate callers and tests."""

        guarded, _reason = self._guard_remote_output(
            candidate,
            frame,
            privacy_safe=privacy_safe,
        )
        return guarded

    def _loop(
        self,
        resources: _Resources,
        *,
        preflight: _PreflightResult | None = None,
        cadence_tracker: CadenceTracker | None = None,
        stage_ewma: dict[str, float | None] | None = None,
        publisher: OutputPublisher | None = None,
        publisher_bridge: _PublisherBridge | None = None,
    ) -> None:
        # Publication/reuse remains output-paced. Processing gets the wider
        # unique-base budget when the requested camera cadence is lower; this
        # changes only deadline accounting, never exact-repeat scheduling.
        transport_interval_ns = round(1_000_000_000 / resources.cfg.output.fps)
        transport_interval_ms = transport_interval_ns / 1_000_000.0
        processing_deadline_ms = 1000.0 / min(
            resources.cfg.camera.fps,
            resources.cfg.output.fps,
        )
        if cadence_tracker is None:
            cadence_tracker = CadenceTracker(resources.cfg.output.fps)
        if stage_ewma is None:
            stage_ewma = {
                "segmentation_ms": None,
                "background_ms": None,
                "color_correction_ms": None,
                "composite_prepare_ms": None,
                "composite_blend_ms": None,
                "composite_ms": None,
                "output_send_ms": None,
                "output_submission_ms": None,
                "output_sink_pacing_wait_ms": None,
                "application_pacing_wait_ms": None,
                "output_schedule_lateness_ms": None,
                "frame_processing_ms": None,
                "new_frame_service_ms": None,
                "new_frame_serialized_loop_ms": None,
            }
        else:
            # Older embedders and focused tests may provide the schema-v0
            # stage map. The fixed public timing schema is nullable, so add
            # the new compositor split without breaking those callers.
            stage_ewma.setdefault("composite_prepare_ms", None)
            stage_ewma.setdefault("composite_blend_ms", None)
        initial_remote_slate = (
            preflight is not None and resources.cfg.background.mode == "remote"
        )
        remote_used = 0
        fallback_count = 1 if initial_remote_slate else 0
        last_output = (
            preflight.output if preflight is not None and publisher is None else None
        )
        fallback_active = initial_remote_slate
        fallback_reason = "startup-slate" if initial_remote_slate else ""
        last_capture_sequence = (
            None if preflight is None else preflight.capture_sequence
        )
        previous_remote_fallback: tuple[bool, str] = (
            fallback_active,
            fallback_reason,
        )
        last_color_status = (
            _color_stats(
                resources.cfg,
                resources.harmonizer.snapshot()
                if resources.harmonizer is not None
                else None,
            )
            if preflight is None
            else preflight.color_status
        )
        loop_clock_ns = time.monotonic_ns()
        next_cycle_start_ns = (
            loop_clock_ns
            if preflight is None
            else preflight.send_timing.submitted_at_ns + transport_interval_ns
        )
        next_output_deadline_ns = (
            loop_clock_ns + transport_interval_ns
            if preflight is None
            else next_cycle_start_ns
        )
        next_base_id = 1
        remote_pending_capture = (
            preflight.captured
            if preflight is not None and resources.cfg.background.mode == "remote"
            else None
        )
        remote_pending_raw_epoch = (
            self._publisher_raw_epoch if remote_pending_capture is not None else 0
        )
        remote_pending_config_version = (
            resources.version if remote_pending_capture is not None else -1
        )
        remote_pending_renderer_session = (
            self._publisher_policy.renderer_session
            if remote_pending_capture is not None
            else -1
        )
        remote_pending_started_ns = (
            time.monotonic_ns() if remote_pending_capture is not None else 0
        )
        remote_submitted_response_ns = 0

        while not self._stop.is_set():
            serialized_started_ns = time.monotonic_ns()
            application_pacing_wait_ms = 0.0
            application_pacing_events = 0
            if publisher is not None or not resources.output.paces:
                pacing_started_ns = time.monotonic_ns()
                remaining_ns = next_cycle_start_ns - pacing_started_ns
                if remaining_ns > 0:
                    self._stop.wait(remaining_ns / 1_000_000_000.0)
                    pacing_completed_ns = time.monotonic_ns()
                    application_pacing_wait_ms = max(
                        0.0,
                        (pacing_completed_ns - pacing_started_ns) / 1_000_000.0,
                    )
                    application_pacing_events = 1
                    if self._stop.is_set():
                        break
            service_started_ns = time.monotonic_ns()
            captured = resources.capture.read()
            if captured is not None and last_capture_sequence is not None:
                if captured.sequence < last_capture_sequence:
                    raise RuntimeError(
                        "capture sequence reordered from "
                        f"{last_capture_sequence} to {captured.sequence}"
                    )
                if captured.sequence == last_capture_sequence:
                    # A source must not return the already-consumed preflight
                    # slot as a new input. Ignore an accidental duplicate
                    # without re-running any temporal consumer.
                    captured = None
            frame = captured.pixels if captured is not None else None
            matte_evidence: MatteFrameEvidence | None = None
            used_remote_candidate = False
            remote_proof: RemoteOutputProof | None = None
            remote_probe: tuple[np.ndarray | None, str, int, int, int] | None = None
            resolved_remote_raw_epoch = 0
            remote_candidate_published_ns = 0
            timings = {
                **(
                    {"capture_read_ms": captured.read_ms}
                    if captured is not None and captured.read_ms is not None
                    else {}
                ),
                "segmentation_ms": 0.0,
                "background_ms": 0.0,
                "color_correction_ms": 0.0,
                "composite_prepare_ms": 0.0,
                "composite_blend_ms": 0.0,
                "composite_ms": 0.0,
            }
            processed = frame is not None
            segmentation_updated = False
            processing_deadline_missed = False
            frame_processing_ms: float | None = None
            diagnostic_frame_started_ns = (
                time.monotonic_ns()
                if processed and self._matte_evidence_requested()
                else None
            )
            if processed:
                assert frame is not None
                assert captured is not None
                # Substage telemetry is frame-local. Modes that bypass the
                # compositor report measured-inapplicable zeros instead of
                # inheriting an image/video sample from the prior epoch.
                resources.last_compositor_substages_ms = {
                    name: 0.0 for name in COMPOSITOR_SUBSTAGE_NAMES
                }
                frame = self._validate_canvas_frame(
                    frame,
                    resources.canvas_size,
                    boundary="capture",
                )
                resources.capture_sequence_timeline.observe(captured.sequence)
                last_capture_sequence = captured.sequence
                if publisher is None:
                    self.hub.publish_raw(frame)
                segmentation_sequence_before = (
                    resources.segmentation_timeline.snapshot().last_sequence
                )

                # A real current frame is the activation trial input. No candidate
                # is committed until this preflight succeeds.
                try:
                    request = self._requests.get_nowait()
                except queue.Empty:
                    request = None
                if request is not None:
                    if isinstance(request, _PatchRequest):
                        self._handle_patch_request(
                            resources,
                            request,
                            captured,
                            publisher_bridge=publisher_bridge,
                        )
                    else:
                        self._handle_mutation_request(resources, request)

                mode = resources.cfg.background.mode
                if publisher is not None and mode == "remote":
                    if publisher_bridge is None:  # pragma: no cover - invariant
                        raise RuntimeError("publisher bridge is missing")
                    with publisher_bridge.lock:
                        publisher_resolved_epoch = (
                            publisher_bridge.resolved_remote_raw_epoch
                        )
                    now_ns = time.monotonic_ns()
                    remote_timeout_ns = resources.cfg.api.remote_timeout_ms * 1_000_000
                    raw_timed_out = bool(
                        remote_pending_capture is not None
                        and now_ns - remote_pending_started_ns > remote_timeout_ns
                    )
                    needs_new_raw = bool(
                        remote_pending_capture is None
                        or remote_pending_config_version != resources.version
                        or remote_pending_renderer_session
                        != self._publisher_policy.renderer_session
                        or publisher_resolved_epoch >= remote_pending_raw_epoch
                        or raw_timed_out
                    )
                    if needs_new_raw:
                        self._publisher_raw_epoch += 1
                        self._publisher_policy_transition(
                            "remote",
                            reason="awaiting-renderer",
                            raw_epoch=self._publisher_raw_epoch,
                        )
                        raw_published = self._publish_remote_raw_frame(
                            frame,
                            self._publisher_raw_epoch,
                        )
                        remote_pending_capture = captured if raw_published else None
                        remote_pending_raw_epoch = (
                            self._publisher_raw_epoch if raw_published else 0
                        )
                        remote_pending_config_version = (
                            resources.version if raw_published else -1
                        )
                        remote_pending_renderer_session = (
                            self._publisher_policy.renderer_session
                            if raw_published
                            else -1
                        )
                        remote_pending_started_ns = now_ns if raw_published else 0
                        remote_submitted_response_ns = 0
                        # The publisher already emits its fixed slate after the
                        # fence. Do not manufacture a processed "unique" base
                        # before a response for this exact raw epoch exists.
                        next_cycle_start_ns = service_started_ns + transport_interval_ns
                        continue

                    proof_reader = getattr(
                        self.hub,
                        "remote_frame_proof_status",
                        None,
                    )
                    if callable(proof_reader):
                        typed_probe_reader = cast(
                            Callable[
                                [float],
                                tuple[np.ndarray | None, str, int, int, int],
                            ],
                            proof_reader,
                        )
                        remote_probe = typed_probe_reader(
                            resources.cfg.api.remote_timeout_ms / 1000.0
                        )
                    else:
                        probe_frame, probe_reason = self.hub.remote_frame_status(
                            max_age_s=resources.cfg.api.remote_timeout_ms / 1000.0
                        )
                        remote_probe = (probe_frame, probe_reason, 0, 0, 0)
                    if remote_probe[0] is None and remote_probe[4] == 0:
                        next_cycle_start_ns = service_started_ns + transport_interval_ns
                        continue
                    if (
                        remote_probe[4] > 0
                        and remote_probe[4] == remote_submitted_response_ns
                    ):
                        # This exact response already has one immutable base in
                        # the handoff/publication lane. Do not manufacture a
                        # second unique base with the same capture provenance
                        # while waiting for its first-send receipt.
                        next_cycle_start_ns = service_started_ns + transport_interval_ns
                        continue
                    assert remote_pending_capture is not None
                    captured = remote_pending_capture
                    frame = captured.pixels
                elif publisher is not None:
                    self.hub.publish_raw(frame)
                    remote_pending_capture = None
                    remote_pending_raw_epoch = 0
                    remote_pending_config_version = -1
                    remote_pending_renderer_session = -1
                    remote_pending_started_ns = 0
                    remote_submitted_response_ns = 0

                matte_evidence = (
                    None
                    if mode == "remote"
                    else self._new_matte_evidence(resources, captured)
                )
                process_started = time.monotonic_ns()
                cfg = resources.cfg
                mode = cfg.background.mode
                color_outcome: dict[str, object] = {}
                fallback_active = False
                fallback_reason = ""
                if mode != "remote":
                    self._latest_raw_frame = frame.copy()

                if mode == "remote":
                    if remote_probe is not None:
                        remote, fallback_reason = remote_probe[:2]
                    else:
                        remote, fallback_reason = self.hub.remote_frame_status(
                            max_age_s=cfg.api.remote_timeout_ms / 1000.0
                        )
                    if remote is not None:
                        if (
                            remote.dtype != np.uint8
                            or remote.ndim != 3
                            or remote.shape[2] != 3
                        ):
                            remote = None
                            fallback_reason = "invalid"
                        elif remote.shape != resources.canvas_shape:
                            remote = None
                            fallback_reason = "wrong-size"
                    if remote is not None:
                        out_frame = remote
                        used_remote_candidate = True
                        fallback_reason = ""
                    else:
                        # A segmented local composite necessarily preserves the
                        # foreground and is therefore not a privacy boundary.
                        # Missing, stale, or malformed remote output always
                        # fails closed to the input-independent slate.
                        out_frame = self._privacy_slate(frame.shape)
                        fallback_active = True
                elif resources.background_fallback_reason:
                    out_frame = self._privacy_slate(resources.canvas_shape)
                    matte_evidence = None
                elif mode == "passthrough" or resources.backdrop is None:
                    out_frame = frame
                else:
                    out_frame, _ = self._local_composite(
                        resources,
                        frame,
                        captured=captured,
                        privacy_safe=False,
                        timings=timings,
                        color_outcome=color_outcome,
                        matte_evidence=matte_evidence,
                    )
                if not color_outcome:
                    color_outcome.update(
                        _color_stats(
                            cfg,
                            resources.harmonizer.snapshot()
                            if resources.harmonizer is not None
                            else None,
                        )
                    )
                segmentation_updated = (
                    resources.segmentation_timeline.snapshot().last_sequence
                    != segmentation_sequence_before
                )
                last_color_status = color_outcome
                frame_processing_ms = (
                    time.monotonic_ns() - process_started
                ) / 1_000_000.0
                if matte_evidence is not None:
                    matte_evidence.timings_ms["frame_processing_ms"] = (
                        frame_processing_ms
                    )
                processing_deadline_missed = (
                    frame_processing_ms > processing_deadline_ms
                )
                samples = (
                    *timings.items(),
                    ("frame_processing_ms", frame_processing_ms),
                )
                for name, sample in samples:
                    stage_ewma[name] = _ewma(stage_ewma.get(name), sample)
            else:
                if publisher is not None:
                    # The publication lane owns exact target-paced reuse.  A
                    # missing capture must never re-enter segmentation,
                    # backdrop, colour, matte, or harmonizer state.
                    self._stop.wait(min(0.002, transport_interval_ms / 1000.0))
                    continue
                if last_output is None:
                    self._stop.wait(min(0.01, transport_interval_ms / 1000.0))
                    continue
                out_frame = last_output

            mode = resources.cfg.background.mode
            guard_source = frame if frame is not None else self._latest_raw_frame
            if guard_source is None:
                # Defensive invariant: startup preflight always establishes the
                # source associated with ``initial_output`` before this loop.
                raise RuntimeError("output publication has no associated source frame")
            guard_started_ns = time.monotonic_ns()
            out_frame, privacy_reason = self._guard_remote_output(
                out_frame,
                guard_source,
                privacy_safe=mode == "remote",
            )
            if privacy_reason:
                fallback_active = True
                fallback_reason = privacy_reason
                used_remote_candidate = False
            # Repeats retain only the already-guarded output, so loss of camera
            # input cannot resurrect an unsafe pre-gate candidate.
            self._validate_output_frame(out_frame, resources.canvas_size)
            base_ready_at_ns = time.monotonic_ns() if processed else None
            exact_final_repeat = bool(
                publisher is None
                and last_output is not None
                and np.array_equal(out_frame, last_output)
            )

            if mode == "remote" and processed:
                assert frame is not None
                # A renderer may connect or publish while sink/application
                # pacing waits. Re-read the latest-only slot at the final send
                # boundary so a stale decision cannot obscure a newer
                # invalid/valid candidate for one extra output interval.
                proof_reader = getattr(self.hub, "remote_frame_proof_status", None)
                if publisher is not None and callable(proof_reader):
                    typed_proof_reader = cast(
                        Callable[[float], tuple[np.ndarray | None, str, int, int, int]],
                        proof_reader,
                    )
                    (
                        latest_remote,
                        latest_reason,
                        proof_raw_epoch,
                        proof_session,
                        proof_published_ns,
                    ) = typed_proof_reader(resources.cfg.api.remote_timeout_ms / 1000.0)
                    policy = self._publisher_policy
                    matching_response = bool(
                        proof_published_ns > 0
                        and policy.mode == "remote"
                        and proof_raw_epoch == policy.raw_epoch
                        and proof_session == policy.renderer_session
                    )
                    if matching_response:
                        resolved_remote_raw_epoch = proof_raw_epoch
                        remote_candidate_published_ns = proof_published_ns
                    structurally_valid_remote = bool(
                        isinstance(latest_remote, np.ndarray)
                        and latest_remote.dtype == np.uint8
                        and latest_remote.ndim == 3
                        and latest_remote.shape == resources.canvas_shape
                    )
                    if structurally_valid_remote and (
                        policy.mode != "remote"
                        or proof_raw_epoch != policy.raw_epoch
                        or proof_session != policy.renderer_session
                    ):
                        # Never accept a frame carrying stale scalar proof, but
                        # still classify a raw/delayed-raw echo before dropping
                        # it.  The producer can outrun the paced sink and rotate
                        # the required raw epoch before this rejection reaches
                        # output; retaining the security reason keeps that
                        # transition observable while pixels remain the slate.
                        assert isinstance(latest_remote, np.ndarray)
                        _guarded, proof_privacy_reason = self._guard_remote_output(
                            latest_remote,
                            frame,
                            privacy_safe=True,
                        )
                        latest_remote = None
                        latest_reason = proof_privacy_reason or "wrong-epoch"
                    elif latest_remote is not None:
                        remote_proof = RemoteOutputProof(
                            proof_raw_epoch,
                            proof_session,
                            proof_published_ns
                            + resources.cfg.api.remote_timeout_ms * 1_000_000,
                        )
                else:
                    latest_remote, latest_reason = self.hub.remote_frame_status(
                        max_age_s=resources.cfg.api.remote_timeout_ms / 1000.0
                    )
                if latest_remote is not None:
                    if (
                        latest_remote.dtype != np.uint8
                        or latest_remote.ndim != 3
                        or latest_remote.shape[2] != 3
                    ):
                        latest_remote = None
                        latest_reason = "invalid"
                    elif latest_remote.shape != resources.canvas_shape:
                        latest_remote = None
                        latest_reason = "wrong-size"
                if latest_remote is None:
                    out_frame = self._privacy_slate(frame.shape)
                    fallback_active = True
                    fallback_reason = latest_reason
                    used_remote_candidate = False
                else:
                    out_frame = latest_remote
                    fallback_active = False
                    fallback_reason = ""
                    used_remote_candidate = True
                out_frame, privacy_reason = self._guard_remote_output(
                    out_frame,
                    frame,
                    privacy_safe=True,
                )
                if privacy_reason:
                    fallback_active = True
                    fallback_reason = privacy_reason
                    used_remote_candidate = False
                self._validate_output_frame(out_frame, resources.canvas_size)
                base_ready_at_ns = time.monotonic_ns()
                exact_final_repeat = bool(
                    publisher is None
                    and last_output is not None
                    and np.array_equal(out_frame, last_output)
                )
            guard_output_validation_ms = (
                time.monotonic_ns() - guard_started_ns
            ) / 1_000_000.0
            if publisher is not None:
                if publisher_bridge is None:  # pragma: no cover - invariant
                    raise RuntimeError("publisher bridge is missing")
                assert captured is not None
                assert base_ready_at_ns is not None
                capture_health = (
                    resources.capture.health_snapshot()
                    if hasattr(resources.capture, "health_snapshot")
                    else None
                )
                publisher_bridge.performance.bind_epoch(
                    self._performance_epoch_key(resources, capture_health),
                    color_correction_mode=(
                        resources.cfg.compositing.color_correction.mode
                    ),
                    light_wrap=resources.cfg.compositing.light_wrap,
                )
                _log_finalized_performance_epochs(publisher_bridge.performance)
                publisher_bridge.performance.record_processing(
                    deadline_missed=processing_deadline_missed,
                    stages_ms=self._runtime_stage_samples(
                        timings,
                        frame_processing_ms or 0.0,
                        resources.last_compositor_substages_ms,
                    ),
                )
                publisher_bridge.processing_completed_count += 1
                self._record_color_output(
                    resources,
                    last_color_status,
                    processed=True,
                    observed_at_s=captured.captured_at_ns / 1_000_000_000.0,
                )
                if mode == "remote":
                    if used_remote_candidate:
                        remote_used += 1
                    elif fallback_active:
                        fallback_count += 1

                producer_status = self._identity_stats(
                    resources,
                    capture_health=capture_health,
                    color_status=last_color_status,
                )
                producer_status.update(
                    {
                        **_capture_health_stats(
                            capture_health,
                            fallback_frames_read=(
                                resources.capture_sequence_timeline.snapshot().last_sequence
                                or 0
                            ),
                        ),
                        "remote_frames_used": remote_used,
                        "remote_fallback_active": fallback_active,
                        "remote_fallback_count": fallback_count,
                        "remote_fallback_reason": fallback_reason,
                        "segmentation_ms": stage_ewma["segmentation_ms"],
                        "background_ms": stage_ewma["background_ms"],
                        "color_correction_ms": stage_ewma["color_correction_ms"],
                        "composite_ms": stage_ewma["composite_ms"],
                        "output_send_ms": publisher_bridge.output_ewma[
                            "output_send_ms"
                        ],
                        "output_submission_ms": publisher_bridge.output_ewma[
                            "output_submission_ms"
                        ],
                        "output_sink_pacing_wait_ms": publisher_bridge.output_ewma[
                            "output_sink_pacing_wait_ms"
                        ],
                        "application_pacing_wait_ms": publisher_bridge.output_ewma[
                            "application_pacing_wait_ms"
                        ],
                        "output_schedule_lateness_ms": publisher_bridge.output_ewma[
                            "output_schedule_lateness_ms"
                        ],
                        "frame_processing_ms": stage_ewma["frame_processing_ms"],
                        "new_frame_service_ms": publisher_bridge.output_ewma[
                            "new_frame_service_ms"
                        ],
                        "new_frame_serialized_loop_ms": None,
                        "timing_schema_version": TIMING_SCHEMA_VERSION,
                        "timing_ms": _timing_fields(
                            stage_ewma,
                            capture_health=capture_health,
                            segmenter=resources.segmenter,
                        ),
                        "processing_completed_count": (
                            publisher_bridge.processing_completed_count
                        ),
                        "processing_completed_fps": (
                            publisher_bridge.performance.snapshot()[
                                "processing_completed_fps"
                            ]
                        ),
                        "output_base_config_version": resources.version,
                        "output_handoff_overwrite_count": (
                            publisher.snapshot().handoff_overwrite_count
                        ),
                        "output_schedule_skipped_slots": (
                            publisher.snapshot().schedule_skipped_slots
                        ),
                        "output_privacy_slate_send_count": (
                            publisher.snapshot().privacy_slate_send_count
                        ),
                        "output_publisher_state": publisher.snapshot().state,
                        "runtime_performance": (
                            publisher_bridge.performance.snapshot()
                        ),
                    }
                )
                if matte_evidence is not None:
                    matte_evidence.timings_ms["guard_output_validation_ms"] = (
                        guard_output_validation_ms
                    )
                asset_slate = bool(resources.background_fallback_reason)
                privacy_slate = asset_slate or (
                    mode == "remote" and not used_remote_candidate
                )
                reason_code = str(
                    resources.background_fallback_reason
                    if asset_slate
                    else fallback_reason or "renderer-unavailable"
                ).replace("_", "-")
                if not reason_code or any(
                    char not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                    for char in reason_code
                ):
                    reason_code = "renderer-unavailable"
                if mode == "remote" and used_remote_candidate and remote_proof is None:
                    out_frame = self._privacy_slate(resources.canvas_shape)
                    privacy_slate = True
                    used_remote_candidate = False
                    reason_code = "proof-unavailable"
                base = SafeBaseFrame.create(
                    out_frame,
                    base_id=next_base_id,
                    config_version=resources.version,
                    capture_sequence=captured.sequence,
                    captured_at_ns=captured.captured_at_ns,
                    base_ready_at_ns=base_ready_at_ns,
                    policy_epoch=self._publisher_policy.epoch,
                    segmentation_updated=segmentation_updated,
                    processing_deadline_missed=processing_deadline_missed,
                    status=producer_status,
                    performance_epoch=self._performance_epoch_tuple(
                        publisher_bridge.performance.current_epoch_key
                    ),
                    remote_proof=(None if privacy_slate else remote_proof),
                    resolved_remote_raw_epoch=resolved_remote_raw_epoch,
                    privacy_slate=privacy_slate,
                    privacy_reason=(reason_code if privacy_slate else ""),
                    matte_bundle_sequence=(
                        None
                        if matte_evidence is None
                        else matte_evidence.metadata.bundle_sequence
                    ),
                )
                with publisher_bridge.lock:
                    publisher_bridge.retain_base(
                        next_base_id,
                        PerformanceEpochKey(*base.performance_epoch),
                        matte_evidence,
                    )
                    try:
                        accepted = publisher.submit(base)
                    except BaseException:
                        publisher_bridge.release_base(next_base_id)
                        raise
                    if not accepted:
                        publisher_bridge.release_base(next_base_id)
                if (
                    accepted
                    and resolved_remote_raw_epoch > 0
                    and resolved_remote_raw_epoch == remote_pending_raw_epoch
                    and remote_candidate_published_ns > 0
                ):
                    remote_submitted_response_ns = remote_candidate_published_ns
                next_base_id += 1
                # Publisher custody now owns the immutable copy. The producer
                # must not retain an older output raster/envelope between cycles.
                del base
                del out_frame
                next_cycle_start_ns = service_started_ns + transport_interval_ns
                if publisher.error is not None:
                    raise OutputPublisherError(
                        "output publisher failed"
                    ) from publisher.error
                continue
            send_started_ns = time.monotonic_ns()
            send_timing = _send_output_with_timing(
                resources.output,
                out_frame,
                copy_frame=mode == "remote",
            )
            output_send_ms = (
                send_timing.completed_at_ns - send_started_ns
            ) / 1_000_000.0
            schedule_lateness_ms = max(
                0.0,
                (send_timing.submitted_at_ns - next_output_deadline_ns) / 1_000_000.0,
            )
            schedule_late = schedule_lateness_ms > 1.0
            new_frame_service_ms: float | None = None
            new_frame_serialized_loop_ms: float | None = None
            serialized_deadline_missed = False
            if processed:
                new_frame_service_ms = max(
                    0.0,
                    (send_timing.submitted_at_ns - service_started_ns) / 1_000_000.0,
                )
                new_frame_serialized_loop_ms = max(
                    0.0,
                    (send_timing.completed_at_ns - serialized_started_ns) / 1_000_000.0,
                )
                serialized_deadline_missed = (
                    new_frame_serialized_loop_ms > transport_interval_ms + 1.0
                )

            cadence_tracker.record_send(
                sent_at_ns=send_timing.submitted_at_ns,
                capture_sequence=(captured.sequence if captured is not None else None),
                captured_at_ns=(
                    captured.captured_at_ns if captured is not None else None
                ),
                base_ready_at_ns=base_ready_at_ns,
                base_updated=processed,
                segmentation_updated=segmentation_updated,
                exact_final_repeat=exact_final_repeat,
                processing_deadline_missed=processing_deadline_missed,
                serialized_new_frame_deadline_missed=serialized_deadline_missed,
                output_sink_pacing_events=send_timing.pacing_events,
                output_sink_recovery_events=send_timing.recovery_events,
                application_pacing_events=application_pacing_events,
                output_schedule_late=schedule_late,
            )
            if processed and mode == "remote":
                if used_remote_candidate:
                    remote_used += 1
                elif fallback_active:
                    fallback_count += 1
            # Preserve the legacy total-loop pacing contract: a fast cycle
            # waits only for its remaining budget, while an over-budget cycle
            # starts the next capture immediately instead of adding a full
            # interval of avoidable delay. The wait is carried to the top of
            # the next cycle so capture is sampled after, never before, pacing.
            next_cycle_start_ns = service_started_ns + transport_interval_ns
            next_output_deadline_ns = (
                send_timing.submitted_at_ns + transport_interval_ns
            )
            last_output = out_frame

            if matte_evidence is not None:
                matte_evidence.timings_ms["output_send_ms"] = output_send_ms
                matte_evidence.timings_ms["guard_output_validation_ms"] = (
                    guard_output_validation_ms
                )
                matte_evidence.timings_ms["output_submission_ms"] = (
                    send_timing.submission_ms
                )
                matte_evidence.timings_ms["output_sink_pacing_wait_ms"] = (
                    send_timing.pacing_wait_ms
                )
                matte_evidence.timings_ms["application_pacing_wait_ms"] = (
                    application_pacing_wait_ms
                )
                matte_evidence.timings_ms["output_schedule_lateness_ms"] = (
                    schedule_lateness_ms
                )
                if new_frame_service_ms is not None:
                    matte_evidence.timings_ms["new_frame_service_ms"] = (
                        new_frame_service_ms
                    )
                if new_frame_serialized_loop_ms is not None:
                    matte_evidence.timings_ms["new_frame_serialized_loop_ms"] = (
                        new_frame_serialized_loop_ms
                    )
                if diagnostic_frame_started_ns is not None:
                    matte_evidence.timings_ms["frame_total_ms"] = (
                        send_timing.completed_at_ns - diagnostic_frame_started_ns
                    ) / 1_000_000.0
                rss_bytes = process_rss_bytes()
                if rss_bytes is not None:
                    matte_evidence.resource_samples["rss_bytes"] = rss_bytes
                if self._submit_matte_evidence(matte_evidence, out_frame):
                    self._matte_last_source_sequence = (
                        matte_evidence.metadata.bundle_sequence
                    )
            recorder = self._matte_recorder
            if recorder is not None and self._matte_last_source_sequence is not None:
                recorder.submit_output_event(
                    sent_monotonic_ns=send_timing.submitted_at_ns,
                    source_bundle_sequence=self._matte_last_source_sequence,
                    base_updated=processed,
                    exact_final_repeat=exact_final_repeat,
                )
            stage_ewma["output_send_ms"] = _ewma(
                stage_ewma["output_send_ms"], output_send_ms
            )
            stage_ewma["output_submission_ms"] = _ewma(
                stage_ewma["output_submission_ms"],
                send_timing.submission_ms,
            )
            stage_ewma["output_sink_pacing_wait_ms"] = _ewma(
                stage_ewma["output_sink_pacing_wait_ms"],
                send_timing.pacing_wait_ms,
            )
            stage_ewma["application_pacing_wait_ms"] = _ewma(
                stage_ewma["application_pacing_wait_ms"],
                application_pacing_wait_ms,
            )
            stage_ewma["output_schedule_lateness_ms"] = _ewma(
                stage_ewma["output_schedule_lateness_ms"],
                schedule_lateness_ms,
            )
            if new_frame_service_ms is not None:
                stage_ewma["new_frame_service_ms"] = _ewma(
                    stage_ewma["new_frame_service_ms"],
                    new_frame_service_ms,
                )
            if new_frame_serialized_loop_ms is not None:
                stage_ewma["new_frame_serialized_loop_ms"] = _ewma(
                    stage_ewma["new_frame_serialized_loop_ms"],
                    new_frame_serialized_loop_ms,
                )
            capture_health = (
                resources.capture.health_snapshot()
                if hasattr(resources.capture, "health_snapshot")
                else None
            )
            remote_state = (fallback_active, fallback_reason)
            if remote_state != previous_remote_fallback:
                if fallback_active:
                    log.warning(
                        "remote fallback active mode=%s reason=%s",
                        "privacy-slate",
                        fallback_reason,
                    )
                elif previous_remote_fallback[0]:
                    log.info("remote renderer recovered; privacy fallback inactive")
                previous_remote_fallback = remote_state
            self._record_color_output(
                resources,
                last_color_status,
                processed=processed,
                observed_at_s=(
                    captured.captured_at_ns / 1_000_000_000.0
                    if captured is not None
                    else None
                ),
            )
            frame_stats = self._identity_stats(
                resources,
                capture_health=capture_health,
                color_status=last_color_status,
            )
            frame_stats.update(
                {
                    **_cadence_status(
                        cadence_tracker,
                        now_ns=send_timing.completed_at_ns,
                        target_output_fps=resources.cfg.output.fps,
                    ),
                    **_capture_health_stats(
                        capture_health,
                        fallback_frames_read=(
                            cadence_tracker.snapshot(
                                now_ns=send_timing.completed_at_ns
                            ).unique_capture_count
                        ),
                    ),
                    "remote_frames_used": remote_used,
                    "remote_fallback_active": fallback_active,
                    "remote_fallback_count": fallback_count,
                    "remote_fallback_reason": fallback_reason,
                    "segmentation_ms": stage_ewma["segmentation_ms"],
                    "background_ms": stage_ewma["background_ms"],
                    "color_correction_ms": stage_ewma["color_correction_ms"],
                    "composite_ms": stage_ewma["composite_ms"],
                    "output_send_ms": stage_ewma["output_send_ms"],
                    "output_submission_ms": stage_ewma["output_submission_ms"],
                    "output_sink_pacing_wait_ms": stage_ewma[
                        "output_sink_pacing_wait_ms"
                    ],
                    "application_pacing_wait_ms": stage_ewma[
                        "application_pacing_wait_ms"
                    ],
                    "output_schedule_lateness_ms": stage_ewma[
                        "output_schedule_lateness_ms"
                    ],
                    "frame_processing_ms": stage_ewma["frame_processing_ms"],
                    "new_frame_service_ms": stage_ewma["new_frame_service_ms"],
                    "new_frame_serialized_loop_ms": stage_ewma[
                        "new_frame_serialized_loop_ms"
                    ],
                    "timing_schema_version": TIMING_SCHEMA_VERSION,
                    "timing_ms": _timing_fields(
                        stage_ewma,
                        capture_health=capture_health,
                        segmenter=resources.segmenter,
                    ),
                }
            )
            self.hub.publish_output(out_frame, stats=frame_stats)
            self._submit_live_matte_evidence(
                matte_evidence,
                status=frame_stats,
            )
            # Preflight status is installed before readiness. Waiting through
            # one steady-state publication prevents startup callers from
            # racing an already-sent preflight slot with their first command.
            self._startup_done.set()
