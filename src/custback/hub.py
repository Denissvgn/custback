"""FrameHub: thread-safe exchange point between the pipeline and the API.

Three flows meet here:
  1. Pipeline publishes each processed output frame -> MJPEG preview clients.
  2. Pipeline publishes raw camera frames + masks -> "remote" WebSocket clients
     (stage 2: an external avatar service consumes them).
  3. Remote clients push rendered frames back; in mode=remote the pipeline
     uses them as output, falling back to local compositing on timeout.
"""

from __future__ import annotations

import asyncio
import copy
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .geometry import Size, validate_bgr_frame


TIMING_SCHEMA_VERSION = 1
TIMING_FIELD_NAMES = (
    "capture.read",
    "segmentation.total",
    "segmentation.preprocess",
    "segmentation.inference",
    "segmentation.postprocess",
    "background.total",
    "color_correction.total",
    "compositor.total",
    "compositor.prepare",
    "compositor.blend",
    "output.send_total",
    "output.submission",
    "output.sink_pacing_wait",
    "output.application_pacing_wait",
    "output.schedule_lateness",
    "pipeline.processing_only",
    "pipeline.new_frame_service",
    "pipeline.new_frame_serialized_loop",
)
_TIMING_FIELD_SET = frozenset(TIMING_FIELD_NAMES)
_MAX_PUBLIC_DURATION_MS = 3_600_000.0
_POST_BASE_NAMESPACE_RE = re.compile(r"[a-z][a-z0-9-]{0,31}\Z")
_MAX_POST_BASE_STAGES = 8
_SELECTION_KEYS = frozenset(
    {
        "schema",
        "version",
        "requested_backend",
        "selected_backend",
        "quality_tier",
        "selection_mode",
        "fallback_active",
        "fallback_category",
        "fallback_reason",
        "guidance",
        "active_device",
        "active_provider",
        "attempts",
    }
)
_SELECTION_ATTEMPT_KEYS = frozenset(
    {
        "backend",
        "quality_tier",
        "preparation_result",
        "activation_result",
        "reason_category",
        "reason",
        "guidance",
    }
)
_QUALITY_BY_BACKEND = {
    "rvm": "matting",
    "mediapipe": "segmentation",
    "heuristic": "heuristic",
    "none": "none",
}
_BACKEND_KIND_BY_SELECTED = {
    "rvm": "true_alpha_recurrent",
    "mediapipe": "confidence_mask_video",
    "heuristic": "binary_coarse",
    "none": "null_passthrough",
}
_SELECTION_REASON_CATEGORIES = frozenset(
    {
        "none",
        "runtime-not-installed",
        "runtime-unavailable",
        "model-unavailable",
        "permission-denied",
        "preparation-unavailable",
        "activation-failed",
        "all-ml-backends-unavailable",
    }
)
_SELECTION_PUBLIC_DIAGNOSTICS: dict[
    str,
    frozenset[tuple[str, str]],
] = {
    "none": frozenset({("", "")}),
    "runtime-not-installed": frozenset(
        {
            (
                "RVM unavailable: runtime not installed",
                "Install the RVM runtime profile and restart.",
            ),
            (
                "MediaPipe unavailable: runtime not installed",
                "Install the MediaPipe runtime profile and restart.",
            ),
        }
    ),
    "runtime-unavailable": frozenset(
        {
            (
                "RVM unavailable: runtime failed to load",
                "Run custback doctor and repair the reported runtime profile.",
            ),
            (
                "MediaPipe unavailable: runtime failed to load",
                "Run custback doctor and repair the reported runtime profile.",
            ),
        }
    ),
    "model-unavailable": frozenset(
        {
            (
                "RVM unavailable: model is not ready",
                "Run the installer or rebuild command to repair managed model assets.",
            ),
            (
                "MediaPipe unavailable: model is not ready",
                "Run the installer or rebuild command to repair managed model assets.",
            ),
        }
    ),
    "permission-denied": frozenset(
        {
            (
                "RVM unavailable: model is not readable",
                "Repair the managed model cache or select a readable custom model.",
            ),
            (
                "MediaPipe unavailable: model is not readable",
                "Repair the managed model cache or select a readable custom model.",
            ),
        }
    ),
    "preparation-unavailable": frozenset(
        {
            (
                "RVM unavailable: startup preparation did not succeed",
                "Run custback doctor and repair the reported runtime profile.",
            ),
            (
                "MediaPipe unavailable: startup preparation did not succeed",
                "Run custback doctor and repair the reported runtime profile.",
            ),
            (
                "Preferred segmentation backend unavailable",
                "Run custback doctor and repair the reported runtime profile.",
            ),
        }
    ),
    "activation-failed": frozenset(
        {
            (
                "RVM unavailable: backend activation failed",
                "Run custback doctor and repair the reported runtime profile.",
            ),
            (
                "MediaPipe unavailable: backend activation failed",
                "Run custback doctor and repair the reported runtime profile.",
            ),
        }
    ),
    "all-ml-backends-unavailable": frozenset(
        {
            (
                "RVM and MediaPipe unavailable: using heuristic detection",
                "Install the RVM runtime profile for matting or the MediaPipe "
                "profile for segmentation.",
            )
        }
    ),
}
_MATTE_POLICY_KEYS = frozenset(
    {
        "schema",
        "version",
        "blend_space",
        "selected_backend_kind",
        "backend_kind",
        "passthrough",
        "experimental_rvm_generic",
        "configured",
        "effective",
        "controls",
    }
)
_MATTE_CONFIGURED_KEYS = frozenset(
    {
        "rvm_downsample_ratio",
        "threshold",
        "mask_blur",
        "edge_refine",
        "edge_refinement_mode",
        "edge_refinement_reference_short_edge_px",
        "edge_refinement_radius_at_reference_px",
        "edge_refinement_min_radius_px",
        "edge_refinement_max_radius_px",
        "mask_shift",
        "temporal_smoothing",
        "boundary_stabilization_mode",
        "boundary_stabilization_time_constant_s",
        "boundary_stabilization_max_motion_px_per_s",
        "use_model_foreground",
        "light_wrap",
        "light_wrap_stabilization_mode",
        "light_wrap_stabilization_time_constant_s",
    }
)
_MATTE_EFFECTIVE_KEYS = frozenset(
    {
        "raw_alpha_mode",
        "opaque_core_mode",
        "halo_mode",
        "residual_temporal_mode",
        "rvm_downsample_ratio",
        "threshold",
        "mask_blur",
        "edge_refine",
        "edge_refinement_mode",
        "edge_refinement_radius_px",
        "mask_shift",
        "temporal_smoothing",
        "boundary_stabilization_mode",
        "boundary_stabilization_time_constant_s",
        "boundary_stabilization_max_motion_px_per_s",
        "use_model_foreground",
        "light_wrap",
        "light_wrap_stabilization_mode",
        "light_wrap_stabilization_time_constant_s",
    }
)
_MATTE_CONTROL_KEYS = frozenset(
    {
        "rvm_downsample_ratio",
        "raw_alpha",
        "threshold",
        "mask_blur",
        "edge_refine",
        "mask_shift",
        "temporal_smoothing",
        "boundary_stabilization",
        "use_model_foreground",
        "light_wrap",
        "light_wrap_stabilization",
        "opaque_core_halo",
    }
)
_MATTE_CONTROL_VALUE_KEYS = frozenset({"configured", "effective", "state", "reason"})
_MATTE_POLICY_TEXT_VALUES = frozenset(
    {
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
    }
)
_MATTE_CONTROL_REASONS = frozenset(
    {
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
    }
)


def _empty_segmentation_selection() -> dict[str, object]:
    return {
        "schema": "custback.backend-selection",
        "version": 1,
        "requested_backend": "none",
        "selected_backend": "none",
        "quality_tier": "none",
        "selection_mode": "explicit",
        "fallback_active": False,
        "fallback_category": "none",
        "fallback_reason": "",
        "guidance": "",
        "active_device": "none",
        "active_provider": "none",
        "attempts": [
            {
                "backend": "none",
                "quality_tier": "none",
                "preparation_result": "not-applicable",
                "activation_result": "selected",
                "reason_category": "none",
                "reason": "",
                "guidance": "",
            }
        ],
    }


def _empty_matte_policy() -> dict[str, object]:
    # Keep the hub's pre-start status valid against the same typed resolver
    # used by the live pipeline instead of maintaining a second policy table.
    from .config import CompositingConfig, SegmentationConfig
    from .matte_policy import MatteBackendKind, resolve_matte_policy

    compositing = CompositingConfig()
    return {
        "schema": "custback.matte-policy",
        "version": 1,
        "blend_space": compositing.blend_space,
        **resolve_matte_policy(
            SegmentationConfig(backend="none"),
            compositing,
            MatteBackendKind.NULL_PASSTHROUGH,
        ).to_dict(),
    }


def _empty_timing_fields() -> dict[str, float | None]:
    return dict.fromkeys(TIMING_FIELD_NAMES)


@dataclass(frozen=True)
class PostBaseProvenance:
    """Typed cadence owned by one optional post-base output stage.

    The stage can report its own output changes, including changes made while
    the safe base was reused. It cannot publish or relabel capture,
    segmentation, base-composite, reuse, or output-send truth.
    """

    update_count: int = 0
    update_fps: float = 0.0
    base_reuse_update_count: int = 0
    base_reuse_update_fps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("update_count", "base_reuse_update_count"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 2**63 - 1:
                raise ValueError(f"post-base {name} must be a bounded nonnegative int")
        for name in ("update_fps", "base_reuse_update_fps"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 10_000.0
            ):
                raise ValueError(
                    f"post-base {name} must be a bounded nonnegative finite number"
                )


@dataclass
class Stats:
    run_id: str = ""
    frames_in: int = 0
    frames_out: int = 0
    fps: float = 0.0
    mode: str = ""
    segmentation_backend: str = ""
    segmentation_device: str = ""
    segmentation_selection: dict[str, object] = field(
        default_factory=_empty_segmentation_selection
    )
    matte_policy: dict[str, object] = field(default_factory=_empty_matte_policy)
    segmentation_generation: int = 0
    capture_sequence: int = 0
    capture_sequence_gap_count: int = 0
    capture_missing_input_count: int = 0
    matte_reset_count: int = 0
    matte_last_reset_reason: str = ""
    segmentation_produces_matte: bool = False
    effective_rvm_downsample_ratio: float | None = None
    effective_mask_blur: int = 0
    effective_edge_refine: bool = False
    effective_edge_refinement_mode: str = "off"
    effective_edge_refinement_radius_px: int = 0
    effective_mask_shift: int = 0
    effective_temporal_smoothing: float = 0.0
    effective_boundary_stabilization_mode: str = "off"
    effective_boundary_stabilization_time_constant_s: float = 0.1
    effective_boundary_stabilization_max_motion_px_per_s: float = 720.0
    effective_use_model_foreground: bool = False
    effective_light_wrap: float = 0.0
    output_backend: str = ""
    remote_connected: bool = False
    remote_frames_used: int = 0
    remote_fallback_active: bool = False
    remote_fallback_mode: str = ""
    remote_fallback_count: int = 0
    remote_fallback_reason: str = ""
    config_version: int = 0
    capture_backend: str = ""
    capture_fourcc: str | None = None
    capture_width: int | None = None
    capture_height: int | None = None
    capture_delivered_width: int | None = None
    capture_delivered_height: int | None = None
    capture_oriented_width: int | None = None
    capture_oriented_height: int | None = None
    capture_normalized_width: int | None = None
    capture_normalized_height: int | None = None
    capture_generation: int = 0
    capture_geometry_transitions: int = 0
    camera_fit: str = ""
    camera_rotation: int = 0
    camera_mirror: bool = False
    camera_scale_x: float | None = None
    camera_scale_y: float | None = None
    camera_crop_left: int | None = None
    camera_crop_top: int | None = None
    camera_crop_right: int | None = None
    camera_crop_bottom: int | None = None
    camera_pad_left: int = 0
    camera_pad_top: int = 0
    camera_pad_right: int = 0
    camera_pad_bottom: int = 0
    camera_controls: dict[str, object] = field(default_factory=dict)
    capture_fps_reported: float | None = None
    capture_target_fps: int = 0
    capture_fps: float = 0.0
    capture_target_met: bool | None = None
    capture_frames_read: int = 0
    capture_dropped_frames: int = 0
    capture_read_failures: int = 0
    capture_restarts: int = 0
    capture_stalled: bool = False
    capture_frame_age_ms: float | None = None
    output_target_fps: int = 0
    output_width: int | None = None
    output_height: int | None = None
    output_fps: int | None = None
    output_effective_fps: float = 0.0
    fps_attainment_pct: float | None = None
    output_repeated_frames: int = 0
    segmentation_update_count: int = 0
    segmentation_update_fps: float = 0.0
    base_composite_update_count: int = 0
    base_composite_update_fps: float = 0.0
    base_composite_reuse_count: int = 0
    base_composite_reuse_fps: float = 0.0
    base_composite_reuse_ratio: float = 0.0
    exact_final_output_repeat_count: int = 0
    exact_final_output_repeat_fps: float = 0.0
    exact_final_output_repeat_ratio: float = 0.0
    output_send_count: int = 0
    output_send_fps: float = 0.0
    last_unique_frame_age_ms: float | None = None
    capture_timestamp_delta_p50_ms: float | None = None
    capture_timestamp_delta_p95_ms: float | None = None
    output_send_delta_p50_ms: float | None = None
    output_send_delta_p95_ms: float | None = None
    output_send_jitter_p50_ms: float | None = None
    output_send_jitter_p95_ms: float | None = None
    base_composite_delta_p50_ms: float | None = None
    base_composite_delta_p95_ms: float | None = None
    cadence_mismatch_active: bool = False
    processing_deadline_misses: int = 0
    serialized_new_frame_deadline_misses: int = 0
    output_sink_pacing_events: int = 0
    output_sink_recovery_events: int = 0
    application_pacing_events: int = 0
    output_schedule_late_events: int = 0
    capture_read_ms: float | None = None
    segmentation_ms: float | None = None
    background_ms: float | None = None
    color_correction_ms: float | None = None
    background_fit: str = ""
    background_rotation: int = 0
    background_mirror: bool = False
    background_scale_x: float | None = None
    background_scale_y: float | None = None
    background_crop_left: int | None = None
    background_crop_top: int | None = None
    background_crop_right: int | None = None
    background_crop_bottom: int | None = None
    background_pad_left: int = 0
    background_pad_top: int = 0
    background_pad_right: int = 0
    background_pad_bottom: int = 0
    background_geometry_transitions: int = 0
    color_correction_mode: str = "off"
    color_correction_active: bool = False
    color_correction_effective_mode: str = "off"
    color_correction_state: str = "disabled"
    color_correction_reason: str = "disabled"
    color_correction_confidence: float = 0.0
    color_correction_exposure_ev: float = 0.0
    color_correction_wb_gain_r: float = 1.0
    color_correction_wb_gain_g: float = 1.0
    color_correction_wb_gain_b: float = 1.0
    color_correction_wb_active: bool = False
    color_correction_warming: bool = False
    color_correction_stale: bool = False
    color_correction_applied_frames: int = 0
    color_correction_bypassed_frames: int = 0
    color_correction_scene_cuts: int = 0
    color_correction_transitions: int = 0
    color_input_assumption: str = "display-referred-srgb-bt709-full-range"
    composite_ms: float | None = None
    output_send_ms: float | None = None
    output_submission_ms: float | None = None
    output_sink_pacing_wait_ms: float | None = None
    application_pacing_wait_ms: float | None = None
    output_schedule_lateness_ms: float | None = None
    frame_processing_ms: float | None = None
    new_frame_service_ms: float | None = None
    new_frame_serialized_loop_ms: float | None = None
    timing_schema_version: int = TIMING_SCHEMA_VERSION
    timing_ms: dict[str, float | None] = field(default_factory=_empty_timing_fields)
    output_fallback_active: bool = False
    output_fallback_reason: str = ""
    segmentation_fallback_active: bool = False
    segmentation_fallback_reason: str = ""
    acceleration_mode: str = ""
    acceleration_requested_provider: str = ""
    acceleration_device_id: int = 0
    acceleration_state: str = ""
    acceleration_active_provider: str = ""
    acceleration_fallback_active: bool = False
    acceleration_fallback_reason: str = ""
    acceleration_fallback_count: int = 0
    acceleration_last_transition_ms: float | None = None
    background_video_source_fps: float | None = None
    background_video_timing_mode: str | None = None
    background_video_frames_displayed: int = 0
    background_video_frames_skipped: int = 0
    background_video_frames_reused: int = 0
    background_video_skip_ratio: float = 0.0
    background_video_seek_count: int = 0
    background_video_decode_failures: int = 0
    background_video_orientation_status: str | None = None
    background_video_metadata_rotation: int | None = None
    background_video_auto_rotation_disabled: bool | None = None
    background_video_decoder_backend: str | None = None
    background_video_color_status: str | None = None
    background_video_input_color: str | None = None
    background_video_output_color: str | None = None
    background_video_color_assumed_fields: list[str] = field(default_factory=list)
    background_video_color_overridden_fields: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


class _Slot:
    """Latest-value slot with a change notification."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._value: np.ndarray | None = None
        self._seq = 0
        self._ts = 0.0
        self._subscribers: set[_AsyncSlotSubscription] = set()

    def put(self, value: np.ndarray) -> None:
        with self._cond:
            self._value = value
            self._seq += 1
            self._ts = time.monotonic()
            self._cond.notify_all()
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber._notify()

    def get(self, last_seq: int = -1, timeout: float | None = None):
        """Return (frame, seq) newer than last_seq, or (None, last_seq)."""
        with self._cond:
            ready = self._cond.wait_for(
                lambda: self._value is not None and self._seq != last_seq,
                timeout=timeout,
            )
            if not ready:
                return None, last_seq
            return self._value, self._seq

    def clear(self) -> None:
        """Discard the value and wake waiters, which continue waiting for data."""
        with self._cond:
            self._value = None
            self._ts = 0.0
            self._seq += 1
            self._cond.notify_all()
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber._notify()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._cond:
            return self._value, self._ts

    def subscribe(self) -> "_AsyncSlotSubscription":
        """Subscribe the current event loop to latest-only frame updates."""

        subscription = _AsyncSlotSubscription(self, asyncio.get_running_loop())
        with self._cond:
            self._subscribers.add(subscription)
        return subscription

    def _unsubscribe(self, subscription: "_AsyncSlotSubscription") -> None:
        with self._cond:
            self._subscribers.discard(subscription)


class _AsyncSlotSubscription:
    """Event-loop-native view of a thread-published latest-value slot."""

    def __init__(self, slot: _Slot, loop: asyncio.AbstractEventLoop) -> None:
        self._slot = slot
        self._loop = loop
        self._event = asyncio.Event()
        self._closed = False

    def _notify(self) -> None:
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self._event.set)
        except RuntimeError:
            # A loop can disappear during process/test teardown. Do not retain
            # a dead subscriber indefinitely in a long-lived frame hub.
            self.close()

    async def get(
        self, last_seq: int = -1, timeout: float | None = None
    ) -> tuple[np.ndarray | None, int]:
        """Return the newest frame after ``last_seq`` without a worker thread."""

        deadline = None if timeout is None else self._loop.time() + timeout
        while not self._closed:
            with self._slot._cond:
                if self._slot._value is not None and self._slot._seq != last_seq:
                    return self._slot._value, self._slot._seq
                # Clear while holding the publisher's lock so an update cannot
                # land in the gap between checking the sequence and waiting.
                self._event.clear()

            remaining = (
                None if deadline is None else max(0.0, deadline - self._loop.time())
            )
            if remaining == 0.0:
                return None, last_seq
            try:
                if remaining is None:
                    await self._event.wait()
                else:
                    await asyncio.wait_for(self._event.wait(), remaining)
            except asyncio.TimeoutError:
                return None, last_seq
        return None, last_seq

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._slot._unsubscribe(self)
        self._event.set()

    async def __aenter__(self) -> "_AsyncSlotSubscription":
        return self

    async def __aexit__(self, *_exc_info) -> None:
        self.close()


class FrameHub:
    def __init__(self, *, run_id: str = "") -> None:
        self.output = _Slot()  # processed frames (what the vcam shows)
        self.raw = _Slot()  # raw camera frames (for remote avatar svc)
        self.remote_in = _Slot()  # frames rendered by the remote avatar svc
        self.stats = Stats(run_id=run_id)
        self._stats_lock = threading.Lock()
        self._remote_clients = 0
        self._remote_session = 0
        self._canvas_size: Size | None = None
        self._post_base_provenance: dict[str, PostBaseProvenance] = {}

    def configure_canvas(self, canvas_size: Size) -> None:
        """Freeze the optional pipeline publication contract for this run."""

        if (
            not isinstance(canvas_size, tuple)
            or len(canvas_size) != 2
            or any(type(value) is not int or value <= 0 for value in canvas_size)
        ):
            raise ValueError("hub canvas dimensions must be positive integers")
        with self._stats_lock:
            changed = self._canvas_size is not None and self._canvas_size != canvas_size
            self._canvas_size = canvas_size
        if changed:
            # A process-level restart may reuse the hub object.  Never let
            # subscribers or a renderer observe pixels from the old canvas.
            self.raw.clear()
            self.output.clear()
            self.remote_in.clear()

    def _validate_publication(self, frame: np.ndarray, boundary: str) -> None:
        canvas_size = self._canvas_size
        if canvas_size is None:
            return
        validated = validate_bgr_frame(
            frame,
            name=f"hub {boundary} frame",
            require_contiguous=True,
        )
        expected = (canvas_size[1], canvas_size[0], 3)
        if validated.shape != expected:
            raise ValueError(
                f"hub {boundary} frame must match canvas {expected}, "
                f"got {validated.shape}"
            )

    # -- pipeline side -------------------------------------------------
    def publish_output(
        self,
        frame: np.ndarray,
        *,
        stats: Mapping[str, object] | None = None,
    ) -> None:
        """Publish one output and its matching status at one hub boundary.

        Status is installed before the slot is notified while the stats lock
        remains held. A consumer awakened by this publication therefore
        cannot read status from the preceding frame.
        """

        self._validate_publication(frame, "output")
        with self._stats_lock:
            if stats is not None:
                self._update_stats_locked(stats)
            self.output.put(frame)

    def publish_raw(self, frame: np.ndarray) -> None:
        self._validate_publication(frame, "raw")
        self.raw.put(frame)

    def get_remote_frame(self, max_age_s: float) -> np.ndarray | None:
        """Latest remote-rendered frame if it is fresh enough, else None."""
        return self.remote_frame_status(max_age_s)[0]

    def remote_frame_status(self, max_age_s: float) -> tuple[np.ndarray | None, str]:
        """Return a fresh frame or a deterministic local-fallback reason."""
        with self._stats_lock:
            if self._remote_clients == 0:
                return None, "no-client"
        frame, ts = self.remote_in.latest()
        if frame is None or (time.monotonic() - ts) > max_age_s:
            return None, "stale"
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            return None, "invalid"
        return frame, ""

    # -- API side ------------------------------------------------------
    def push_remote_frame(
        self, frame: np.ndarray, session_id: int | None = None
    ) -> bool:
        """Publish a frame only for the currently connected remote session."""
        with self._stats_lock:
            if self._remote_clients == 0:
                return False
            if session_id is not None and session_id != self._remote_session:
                return False
            # Keep the stats lock through put so a final disconnect always
            # clears any frame published by that session.
            self.remote_in.put(frame)
            return True

    def remote_client_connected(self) -> int:
        with self._stats_lock:
            if self._remote_clients == 0:
                self._remote_session += 1
                self.remote_in.clear()
            self._remote_clients += 1
            self.stats.remote_connected = True
            return self._remote_session

    def remote_client_disconnected(self, session_id: int | None = None) -> None:
        with self._stats_lock:
            if session_id is not None and session_id != self._remote_session:
                return
            self._remote_clients = max(0, self._remote_clients - 1)
            self.stats.remote_connected = self._remote_clients > 0
            if self._remote_clients == 0:
                self.remote_in.clear()

    def active_remote_session(self) -> int | None:
        """Return the authenticated renderer epoch, if one is currently live."""

        with self._stats_lock:
            return self._remote_session if self._remote_clients > 0 else None

    def remote_session_valid(self, session_id: int) -> bool:
        """Check a renderer lease without granting access to any other route."""

        with self._stats_lock:
            return self._remote_clients > 0 and session_id == self._remote_session

    def _invalidate_remote_session_locked(self, session_id: int | None) -> bool:
        if session_id is not None and session_id != self._remote_session:
            return False
        had_session = self._remote_clients > 0
        if had_session:
            # Advance immediately so an old WebSocket cannot publish in the gap
            # before the next authenticated connection arrives.
            self._remote_session += 1
        self._remote_clients = 0
        self.stats.remote_connected = False
        self.remote_in.clear()
        return had_session

    def invalidate_remote_session(self, session_id: int | None = None) -> bool:
        """Atomically revoke the renderer epoch and discard every queued frame."""

        with self._stats_lock:
            return self._invalidate_remote_session_locked(session_id)

    def reset_remote_session(self, reset: Callable[[], None]) -> bool:
        """Revoke stale output and reset privacy state at the same boundary."""

        with self._stats_lock:
            had_session = self._invalidate_remote_session_locked(None)
            reset()
            return had_session

    def clear_remote_frames(self) -> None:
        """Discard queued output without changing the authenticated epoch."""
        with self._stats_lock:
            self.remote_in.clear()

    def update_stats(self, **kwargs) -> None:
        with self._stats_lock:
            self._update_stats_locked(kwargs)

    def publish_post_base_provenance(
        self,
        namespace: str,
        provenance: PostBaseProvenance,
    ) -> None:
        """Publish one bounded optional-stage cadence namespace.

        This deliberately separate method prevents an optional downstream
        stage from writing core cadence fields through a generic mapping.
        """

        if not isinstance(namespace, str) or not _POST_BASE_NAMESPACE_RE.fullmatch(
            namespace
        ):
            raise ValueError("post-base namespace must match [a-z][a-z0-9-]{0,31}")
        if not isinstance(provenance, PostBaseProvenance):
            raise TypeError("post-base provenance must be PostBaseProvenance")
        with self._stats_lock:
            if (
                namespace not in self._post_base_provenance
                and len(self._post_base_provenance) >= _MAX_POST_BASE_STAGES
            ):
                raise ValueError("post-base provenance stage limit reached")
            self._post_base_provenance[namespace] = provenance

    def _update_stats_locked(self, values: Mapping[str, object]) -> None:
        staged: dict[str, object] = {}
        for key, value in values.items():
            if key == "started_at" or not hasattr(self.stats, key):
                raise KeyError(f"unknown public stats field: {key}")
            if key == "timing_schema_version" and value != TIMING_SCHEMA_VERSION:
                raise ValueError(
                    f"timing_schema_version must be {TIMING_SCHEMA_VERSION}"
                )
            if key == "timing_ms":
                value = self._validated_timing_fields(value)
            if key == "segmentation_selection":
                value = self._validated_segmentation_selection(value)
            if key == "matte_policy":
                value = self._validated_matte_policy(value)
            if isinstance(value, (dict, list)):
                value = copy.deepcopy(value)
            staged[key] = value

        candidate_selection = staged.get(
            "segmentation_selection",
            self.stats.segmentation_selection,
        )
        candidate_policy = staged.get("matte_policy", self.stats.matte_policy)
        if "segmentation_selection" in staged or "matte_policy" in staged:
            self._validate_selection_policy_pair(
                candidate_selection,
                candidate_policy,
            )
        for key, value in staged.items():
            setattr(self.stats, key, value)

    @staticmethod
    def _validate_selection_policy_pair(
        selection: object,
        policy: object,
    ) -> None:
        if not isinstance(selection, Mapping) or not isinstance(policy, Mapping):
            raise TypeError("selection and matte policy must be mappings")
        selected = selection["selected_backend"]
        expected_kind = _BACKEND_KIND_BY_SELECTED[selected]
        if policy["selected_backend_kind"] != expected_kind:
            raise ValueError("selected backend and matte policy kind do not match")
        passthrough = policy["passthrough"]
        expected_effective_kind = "null_passthrough" if passthrough else expected_kind
        if policy["backend_kind"] != expected_effective_kind:
            raise ValueError("matte policy passthrough/effective kind is inconsistent")
        if policy["experimental_rvm_generic"] and (
            expected_kind != "true_alpha_recurrent" or passthrough
        ):
            raise ValueError("experimental RVM policy requires active RVM matting")

    @staticmethod
    def _bounded_public_text(value: object, field_name: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) > 240
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError(f"{field_name} must be bounded one-line text")
        return value

    @staticmethod
    def _validated_policy_value(value: object, field_name: str) -> object:
        if value is None or type(value) is bool:
            return value
        if type(value) is int:
            if abs(value) > 1_000_000_000:
                raise ValueError(f"{field_name} integer is outside the public bound")
            return value
        if type(value) is float:
            if not math.isfinite(value) or abs(value) > 1_000_000_000.0:
                raise ValueError(f"{field_name} number is outside the public bound")
            return value
        if isinstance(value, str) and value in _MATTE_POLICY_TEXT_VALUES:
            return value
        raise ValueError(f"{field_name} is not an allowed public policy value")

    @classmethod
    def _validated_segmentation_selection(
        cls,
        value: object,
    ) -> dict[str, object]:
        if not isinstance(value, Mapping) or set(value) != _SELECTION_KEYS:
            raise ValueError(
                "segmentation_selection must contain exactly the versioned keys"
            )
        if value["schema"] != "custback.backend-selection" or value["version"] != 1:
            raise ValueError("unsupported segmentation_selection schema")
        requested = value["requested_backend"]
        selected = value["selected_backend"]
        tier = value["quality_tier"]
        mode = value["selection_mode"]
        if requested not in {"auto", "rvm", "mediapipe", "heuristic", "none"}:
            raise ValueError("invalid requested segmentation backend")
        if selected not in _QUALITY_BY_BACKEND:
            raise ValueError("invalid selected segmentation backend")
        if tier != _QUALITY_BY_BACKEND[selected]:
            raise ValueError("selected backend and quality tier do not match")
        if mode not in {"automatic", "explicit", "model-format"}:
            raise ValueError("invalid segmentation selection mode")
        if mode == "explicit":
            if requested == "auto" or selected != requested:
                raise ValueError("explicit selection must select the requested backend")
            expected_fallback = False
        elif mode == "automatic":
            if requested != "auto" or selected == "none":
                raise ValueError("automatic selection requires an auto backend request")
            expected_fallback = selected != "rvm"
        else:
            if requested != "auto" or selected == "none":
                raise ValueError(
                    "model-format selection requires an auto backend request"
                )
            expected_fallback = selected == "heuristic"
        fallback_active = value["fallback_active"]
        category = value["fallback_category"]
        if type(fallback_active) is not bool:
            raise TypeError("segmentation fallback_active must be boolean")
        if category not in _SELECTION_REASON_CATEGORIES:
            raise ValueError("invalid segmentation fallback category")
        if fallback_active != (category != "none"):
            raise ValueError("segmentation fallback flag/category are inconsistent")
        if fallback_active != expected_fallback:
            raise ValueError(
                "selection mode/backend fallback semantics are inconsistent"
            )
        reason = cls._bounded_public_text(
            value["fallback_reason"], "segmentation fallback reason"
        )
        guidance = cls._bounded_public_text(
            value["guidance"], "segmentation fallback guidance"
        )
        if fallback_active and (not reason or not guidance):
            raise ValueError("active segmentation fallback requires guidance")
        if not fallback_active and (reason or guidance):
            raise ValueError("inactive segmentation fallback must have empty guidance")
        if (reason, guidance) not in _SELECTION_PUBLIC_DIAGNOSTICS[category]:
            raise ValueError("segmentation fallback diagnostic is not sanitized")
        for key in ("active_device", "active_provider"):
            text = cls._bounded_public_text(value[key], f"segmentation {key}")
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", text):
                raise ValueError(f"segmentation {key} contains unsupported characters")

        raw_attempts = value["attempts"]
        if not isinstance(raw_attempts, list) or not 1 <= len(raw_attempts) <= 4:
            raise ValueError("segmentation attempts must contain one to four entries")
        attempts: list[dict[str, str]] = []
        seen: set[object] = set()
        selected_attempts = 0
        for raw in raw_attempts:
            if not isinstance(raw, Mapping) or set(raw) != _SELECTION_ATTEMPT_KEYS:
                raise ValueError("segmentation attempt has unknown or missing keys")
            backend = raw["backend"]
            if backend not in _QUALITY_BY_BACKEND or backend in seen:
                raise ValueError(
                    "segmentation attempt backend must be valid and unique"
                )
            seen.add(backend)
            if raw["quality_tier"] != _QUALITY_BY_BACKEND[backend]:
                raise ValueError("segmentation attempt tier does not match backend")
            if raw["preparation_result"] not in {
                "ready",
                "unavailable",
                "not-run",
                "not-applicable",
            }:
                raise ValueError("invalid segmentation preparation result")
            activation = raw["activation_result"]
            if activation not in {"selected", "failed", "not-attempted"}:
                raise ValueError("invalid segmentation activation result")
            attempt_category = raw["reason_category"]
            if attempt_category not in _SELECTION_REASON_CATEGORIES:
                raise ValueError("invalid segmentation attempt reason category")
            attempt_reason = cls._bounded_public_text(
                raw["reason"], "segmentation attempt reason"
            )
            attempt_guidance = cls._bounded_public_text(
                raw["guidance"], "segmentation attempt guidance"
            )
            if (attempt_category == "none") != (
                not attempt_reason and not attempt_guidance
            ):
                raise ValueError("attempt reason fields/category are inconsistent")
            if (
                attempt_reason,
                attempt_guidance,
            ) not in _SELECTION_PUBLIC_DIAGNOSTICS[attempt_category]:
                raise ValueError("segmentation attempt diagnostic is not sanitized")
            preparation_result = raw["preparation_result"]
            if activation == "selected" and (
                attempt_category != "none" or preparation_result == "unavailable"
            ):
                raise ValueError("selected attempt cannot be unavailable or failed")
            if activation == "failed" and attempt_category == "none":
                raise ValueError("failed attempt requires a reason category")
            if preparation_result == "unavailable" and (
                activation != "not-attempted" or attempt_category == "none"
            ):
                raise ValueError(
                    "unavailable preparation cannot report backend activation"
                )
            if activation == "selected":
                selected_attempts += 1
                if backend != selected:
                    raise ValueError("selected attempt does not match selected backend")
            attempts.append({key: str(raw[key]) for key in _SELECTION_ATTEMPT_KEYS})
        if selected_attempts != 1:
            raise ValueError("segmentation attempts must identify one selected backend")
        attempted_backends = [attempt["backend"] for attempt in attempts]
        if mode == "explicit":
            expected_attempts = [selected]
        elif mode == "automatic":
            expected_attempts = {
                "rvm": ["rvm", "mediapipe"],
                "mediapipe": ["rvm", "mediapipe"],
                "heuristic": ["rvm", "mediapipe", "heuristic"],
            }[selected]
        elif selected == "heuristic":
            if attempted_backends[0] not in {"rvm", "mediapipe"}:
                raise ValueError("model-format fallback must retain its ML candidate")
            expected_attempts = [attempted_backends[0], "heuristic"]
        else:
            expected_attempts = [selected]
        if attempted_backends != expected_attempts:
            raise ValueError("selection attempts do not match the selection mode")
        selected_index = attempted_backends.index(selected)
        if any(
            attempt["reason_category"] == "none"
            for attempt in attempts[:selected_index]
        ):
            raise ValueError(
                "preferred candidates before a fallback require diagnostics"
            )

        validated = dict(value)
        validated["attempts"] = attempts
        return validated

    @classmethod
    def _validated_matte_policy(cls, value: object) -> dict[str, object]:
        if not isinstance(value, Mapping) or set(value) != _MATTE_POLICY_KEYS:
            raise ValueError("matte_policy must contain exactly the versioned keys")
        if value["schema"] != "custback.matte-policy" or value["version"] != 1:
            raise ValueError("unsupported matte_policy schema")
        if value["blend_space"] not in {"srgb_legacy", "linear_srgb"}:
            raise ValueError("invalid matte_policy blend space")
        backend_kinds = {
            "true_alpha_recurrent",
            "confidence_mask_video",
            "binary_coarse",
            "null_passthrough",
        }
        if (
            value["selected_backend_kind"] not in backend_kinds
            or value["backend_kind"] not in backend_kinds
        ):
            raise ValueError("invalid matte_policy backend kind")
        if (
            type(value["passthrough"]) is not bool
            or type(value["experimental_rvm_generic"]) is not bool
        ):
            raise TypeError("matte_policy flags must be boolean")
        configured = value["configured"]
        effective = value["effective"]
        controls = value["controls"]
        if (
            not isinstance(configured, Mapping)
            or set(configured) != _MATTE_CONFIGURED_KEYS
        ):
            raise ValueError("matte_policy configured fields do not match schema")
        if (
            not isinstance(effective, Mapping)
            or set(effective) != _MATTE_EFFECTIVE_KEYS
        ):
            raise ValueError("matte_policy effective fields do not match schema")
        if not isinstance(controls, Mapping) or set(controls) != _MATTE_CONTROL_KEYS:
            raise ValueError("matte_policy controls do not match schema")
        validated_configured = {
            str(key): cls._validated_policy_value(
                configured[key],
                f"matte_policy configured {key!r}",
            )
            for key in configured
        }
        validated_effective = {
            str(key): cls._validated_policy_value(
                effective[key],
                f"matte_policy effective {key!r}",
            )
            for key in effective
        }
        validated_controls: dict[str, object] = {}
        for name, raw_control in controls.items():
            if (
                not isinstance(raw_control, Mapping)
                or set(raw_control) != _MATTE_CONTROL_VALUE_KEYS
            ):
                raise ValueError(f"matte_policy control {name!r} does not match schema")
            if raw_control["state"] not in {
                "effective",
                "bypassed",
                "inapplicable",
            }:
                raise ValueError(f"matte_policy control {name!r} has invalid state")
            reason = cls._bounded_public_text(
                raw_control["reason"], f"matte_policy control {name!r} reason"
            )
            if reason not in _MATTE_CONTROL_REASONS:
                raise ValueError(
                    f"matte_policy control {name!r} reason is not sanitized"
                )
            validated_controls[str(name)] = {
                "configured": cls._validated_policy_value(
                    raw_control["configured"],
                    f"matte_policy control {name!r} configured",
                ),
                "effective": cls._validated_policy_value(
                    raw_control["effective"],
                    f"matte_policy control {name!r} effective",
                ),
                "state": raw_control["state"],
                "reason": reason,
            }
        validated = dict(value)
        validated["configured"] = validated_configured
        validated["effective"] = validated_effective
        validated["controls"] = validated_controls
        return validated

    @staticmethod
    def _validated_timing_fields(value: object) -> dict[str, float | None]:
        if not isinstance(value, Mapping):
            raise TypeError("timing_ms must be a mapping")
        if set(value) != _TIMING_FIELD_SET:
            raise ValueError("timing_ms must contain exactly the versioned timing keys")
        validated: dict[str, float | None] = {}
        for key in TIMING_FIELD_NAMES:
            sample = value[key]
            if sample is None:
                validated[key] = None
                continue
            if (
                isinstance(sample, bool)
                or not isinstance(sample, (int, float))
                or not math.isfinite(float(sample))
                or not 0.0 <= float(sample) <= _MAX_PUBLIC_DURATION_MS
            ):
                raise ValueError(
                    f"timing_ms[{key!r}] must be null or a bounded "
                    "nonnegative finite duration"
                )
            validated[key] = float(sample)
        return validated

    def _post_base_extensions_locked(self) -> dict[str, object]:
        return {
            "post_base": {
                "schema": "custback.post-base-cadence",
                "version": 1,
                "stages": [
                    {
                        "namespace": namespace,
                        "update_count": provenance.update_count,
                        "update_fps": round(float(provenance.update_fps), 3),
                        "base_reuse_update_count": (provenance.base_reuse_update_count),
                        "base_reuse_update_fps": round(
                            float(provenance.base_reuse_update_fps),
                            3,
                        ),
                    }
                    for namespace, provenance in sorted(
                        self._post_base_provenance.items()
                    )
                ],
            }
        }

    def stats_dict(self) -> dict:
        with self._stats_lock:
            return {
                "run_id": self.stats.run_id,
                "frames_in": self.stats.frames_in,
                "frames_out": self.stats.frames_out,
                "fps": round(self.stats.fps, 1),
                "mode": self.stats.mode,
                "segmentation_backend": self.stats.segmentation_backend,
                "segmentation_device": self.stats.segmentation_device,
                "segmentation_selection": copy.deepcopy(
                    self.stats.segmentation_selection
                ),
                "matte_policy": copy.deepcopy(self.stats.matte_policy),
                "segmentation_generation": self.stats.segmentation_generation,
                "capture_sequence": self.stats.capture_sequence,
                "capture_sequence_gap_count": self.stats.capture_sequence_gap_count,
                "capture_missing_input_count": self.stats.capture_missing_input_count,
                "matte_reset_count": self.stats.matte_reset_count,
                "matte_last_reset_reason": self.stats.matte_last_reset_reason,
                "segmentation_produces_matte": self.stats.segmentation_produces_matte,
                "effective_rvm_downsample_ratio": self._rounded_optional(
                    self.stats.effective_rvm_downsample_ratio, 6
                ),
                "effective_mask_blur": self.stats.effective_mask_blur,
                "effective_edge_refine": self.stats.effective_edge_refine,
                "effective_edge_refinement_mode": (
                    self.stats.effective_edge_refinement_mode
                ),
                "effective_edge_refinement_radius_px": (
                    self.stats.effective_edge_refinement_radius_px
                ),
                "effective_mask_shift": self.stats.effective_mask_shift,
                "effective_temporal_smoothing": round(
                    self.stats.effective_temporal_smoothing, 4
                ),
                "effective_boundary_stabilization_mode": (
                    self.stats.effective_boundary_stabilization_mode
                ),
                "effective_boundary_stabilization_time_constant_s": round(
                    self.stats.effective_boundary_stabilization_time_constant_s,
                    4,
                ),
                "effective_boundary_stabilization_max_motion_px_per_s": round(
                    self.stats.effective_boundary_stabilization_max_motion_px_per_s,
                    4,
                ),
                "effective_use_model_foreground": (
                    self.stats.effective_use_model_foreground
                ),
                "effective_light_wrap": round(self.stats.effective_light_wrap, 4),
                "output_backend": self.stats.output_backend,
                "remote_connected": self.stats.remote_connected,
                "remote_frames_used": self.stats.remote_frames_used,
                "remote_fallback_active": self.stats.remote_fallback_active,
                "remote_fallback_mode": self.stats.remote_fallback_mode,
                "remote_fallback_count": self.stats.remote_fallback_count,
                "remote_fallback_reason": self.stats.remote_fallback_reason,
                "config_version": self.stats.config_version,
                "capture_backend": self.stats.capture_backend,
                "capture_fourcc": self.stats.capture_fourcc,
                "capture_width": self.stats.capture_width,
                "capture_height": self.stats.capture_height,
                "capture_delivered_width": self.stats.capture_delivered_width,
                "capture_delivered_height": self.stats.capture_delivered_height,
                "capture_oriented_width": self.stats.capture_oriented_width,
                "capture_oriented_height": self.stats.capture_oriented_height,
                "capture_normalized_width": self.stats.capture_normalized_width,
                "capture_normalized_height": self.stats.capture_normalized_height,
                "capture_generation": self.stats.capture_generation,
                "capture_geometry_transitions": self.stats.capture_geometry_transitions,
                "camera_fit": self.stats.camera_fit,
                "camera_rotation": self.stats.camera_rotation,
                "camera_mirror": self.stats.camera_mirror,
                "camera_scale_x": self._rounded_optional(self.stats.camera_scale_x, 6),
                "camera_scale_y": self._rounded_optional(self.stats.camera_scale_y, 6),
                "camera_crop_left": self.stats.camera_crop_left,
                "camera_crop_top": self.stats.camera_crop_top,
                "camera_crop_right": self.stats.camera_crop_right,
                "camera_crop_bottom": self.stats.camera_crop_bottom,
                "camera_pad_left": self.stats.camera_pad_left,
                "camera_pad_top": self.stats.camera_pad_top,
                "camera_pad_right": self.stats.camera_pad_right,
                "camera_pad_bottom": self.stats.camera_pad_bottom,
                "camera_controls": copy.deepcopy(self.stats.camera_controls),
                "capture_fps_reported": self._rounded_optional(
                    self.stats.capture_fps_reported, 2
                ),
                "capture_target_fps": self.stats.capture_target_fps,
                "capture_fps": round(self.stats.capture_fps, 1),
                "capture_target_met": self.stats.capture_target_met,
                "capture_frames_read": self.stats.capture_frames_read,
                "capture_dropped_frames": self.stats.capture_dropped_frames,
                "capture_read_failures": self.stats.capture_read_failures,
                "capture_restarts": self.stats.capture_restarts,
                "capture_stalled": self.stats.capture_stalled,
                "capture_frame_age_ms": self._rounded_optional(
                    self.stats.capture_frame_age_ms, 1
                ),
                "output_target_fps": self.stats.output_target_fps,
                "output_width": self.stats.output_width,
                "output_height": self.stats.output_height,
                "output_fps": self.stats.output_fps,
                "output_effective_fps": round(self.stats.output_effective_fps, 1),
                "fps_attainment_pct": self._rounded_optional(
                    self.stats.fps_attainment_pct, 1
                ),
                "output_repeated_frames": self.stats.output_repeated_frames,
                "segmentation_update_count": self.stats.segmentation_update_count,
                "segmentation_update_fps": round(self.stats.segmentation_update_fps, 3),
                "base_composite_update_count": (self.stats.base_composite_update_count),
                "base_composite_update_fps": round(
                    self.stats.base_composite_update_fps, 3
                ),
                "base_composite_reuse_count": self.stats.base_composite_reuse_count,
                "base_composite_reuse_fps": round(
                    self.stats.base_composite_reuse_fps, 3
                ),
                "base_composite_reuse_ratio": round(
                    self.stats.base_composite_reuse_ratio, 4
                ),
                "exact_final_output_repeat_count": (
                    self.stats.exact_final_output_repeat_count
                ),
                "exact_final_output_repeat_fps": round(
                    self.stats.exact_final_output_repeat_fps, 3
                ),
                "exact_final_output_repeat_ratio": round(
                    self.stats.exact_final_output_repeat_ratio, 4
                ),
                "output_send_count": self.stats.output_send_count,
                "output_send_fps": round(self.stats.output_send_fps, 3),
                "last_unique_frame_age_ms": self._rounded_optional(
                    self.stats.last_unique_frame_age_ms, 3
                ),
                "capture_timestamp_delta_p50_ms": self._rounded_optional(
                    self.stats.capture_timestamp_delta_p50_ms, 3
                ),
                "capture_timestamp_delta_p95_ms": self._rounded_optional(
                    self.stats.capture_timestamp_delta_p95_ms, 3
                ),
                "output_send_delta_p50_ms": self._rounded_optional(
                    self.stats.output_send_delta_p50_ms, 3
                ),
                "output_send_delta_p95_ms": self._rounded_optional(
                    self.stats.output_send_delta_p95_ms, 3
                ),
                "output_send_jitter_p50_ms": self._rounded_optional(
                    self.stats.output_send_jitter_p50_ms, 3
                ),
                "output_send_jitter_p95_ms": self._rounded_optional(
                    self.stats.output_send_jitter_p95_ms, 3
                ),
                "base_composite_delta_p50_ms": self._rounded_optional(
                    self.stats.base_composite_delta_p50_ms, 3
                ),
                "base_composite_delta_p95_ms": self._rounded_optional(
                    self.stats.base_composite_delta_p95_ms, 3
                ),
                "cadence_mismatch_active": self.stats.cadence_mismatch_active,
                "processing_deadline_misses": self.stats.processing_deadline_misses,
                "serialized_new_frame_deadline_misses": (
                    self.stats.serialized_new_frame_deadline_misses
                ),
                "output_sink_pacing_events": self.stats.output_sink_pacing_events,
                "output_sink_recovery_events": self.stats.output_sink_recovery_events,
                "application_pacing_events": self.stats.application_pacing_events,
                "output_schedule_late_events": (self.stats.output_schedule_late_events),
                "capture_read_ms": self._rounded_optional(
                    self.stats.capture_read_ms, 1
                ),
                "segmentation_ms": self._rounded_optional(
                    self.stats.segmentation_ms, 1
                ),
                "background_ms": self._rounded_optional(self.stats.background_ms, 1),
                "color_correction_ms": self._rounded_optional(
                    self.stats.color_correction_ms, 1
                ),
                "background_fit": self.stats.background_fit,
                "background_rotation": self.stats.background_rotation,
                "background_mirror": self.stats.background_mirror,
                "background_scale_x": self._rounded_optional(
                    self.stats.background_scale_x, 6
                ),
                "background_scale_y": self._rounded_optional(
                    self.stats.background_scale_y, 6
                ),
                "background_crop_left": self.stats.background_crop_left,
                "background_crop_top": self.stats.background_crop_top,
                "background_crop_right": self.stats.background_crop_right,
                "background_crop_bottom": self.stats.background_crop_bottom,
                "background_pad_left": self.stats.background_pad_left,
                "background_pad_top": self.stats.background_pad_top,
                "background_pad_right": self.stats.background_pad_right,
                "background_pad_bottom": self.stats.background_pad_bottom,
                "background_geometry_transitions": (
                    self.stats.background_geometry_transitions
                ),
                "color_correction_mode": self.stats.color_correction_mode,
                "color_correction_active": self.stats.color_correction_active,
                "color_correction_effective_mode": (
                    self.stats.color_correction_effective_mode
                ),
                "color_correction_state": self.stats.color_correction_state,
                "color_correction_reason": self.stats.color_correction_reason,
                "color_correction_confidence": round(
                    self.stats.color_correction_confidence, 3
                ),
                "color_correction_exposure_ev": round(
                    self.stats.color_correction_exposure_ev, 3
                ),
                "color_correction_wb_gain_r": round(
                    self.stats.color_correction_wb_gain_r, 4
                ),
                "color_correction_wb_gain_g": round(
                    self.stats.color_correction_wb_gain_g, 4
                ),
                "color_correction_wb_gain_b": round(
                    self.stats.color_correction_wb_gain_b, 4
                ),
                "color_correction_wb_active": self.stats.color_correction_wb_active,
                "color_correction_warming": self.stats.color_correction_warming,
                "color_correction_stale": self.stats.color_correction_stale,
                "color_correction_applied_frames": (
                    self.stats.color_correction_applied_frames
                ),
                "color_correction_bypassed_frames": (
                    self.stats.color_correction_bypassed_frames
                ),
                "color_correction_scene_cuts": (self.stats.color_correction_scene_cuts),
                "color_correction_transitions": (
                    self.stats.color_correction_transitions
                ),
                "color_input_assumption": self.stats.color_input_assumption,
                "composite_ms": self._rounded_optional(self.stats.composite_ms, 1),
                "output_send_ms": self._rounded_optional(self.stats.output_send_ms, 1),
                "output_submission_ms": self._rounded_optional(
                    self.stats.output_submission_ms, 3
                ),
                "output_sink_pacing_wait_ms": self._rounded_optional(
                    self.stats.output_sink_pacing_wait_ms, 3
                ),
                "application_pacing_wait_ms": self._rounded_optional(
                    self.stats.application_pacing_wait_ms, 3
                ),
                "output_schedule_lateness_ms": self._rounded_optional(
                    self.stats.output_schedule_lateness_ms, 3
                ),
                "frame_processing_ms": self._rounded_optional(
                    self.stats.frame_processing_ms, 1
                ),
                "new_frame_service_ms": self._rounded_optional(
                    self.stats.new_frame_service_ms, 3
                ),
                "new_frame_serialized_loop_ms": self._rounded_optional(
                    self.stats.new_frame_serialized_loop_ms, 3
                ),
                "timing_schema_version": self.stats.timing_schema_version,
                "timing_ms": {
                    key: self._rounded_optional(self.stats.timing_ms[key], 3)
                    for key in TIMING_FIELD_NAMES
                },
                "output_fallback_active": self.stats.output_fallback_active,
                "output_fallback_reason": self.stats.output_fallback_reason,
                "segmentation_fallback_active": self.stats.segmentation_fallback_active,
                "segmentation_fallback_reason": self.stats.segmentation_fallback_reason,
                "acceleration_mode": self.stats.acceleration_mode,
                "acceleration_requested_provider": (
                    self.stats.acceleration_requested_provider
                ),
                "acceleration_device_id": self.stats.acceleration_device_id,
                "acceleration_state": self.stats.acceleration_state,
                "acceleration_active_provider": self.stats.acceleration_active_provider,
                "acceleration_fallback_active": self.stats.acceleration_fallback_active,
                "acceleration_fallback_reason": self.stats.acceleration_fallback_reason,
                "acceleration_fallback_count": self.stats.acceleration_fallback_count,
                "acceleration_last_transition_ms": self._rounded_optional(
                    self.stats.acceleration_last_transition_ms, 1
                ),
                "background_video_source_fps": self._rounded_optional(
                    self.stats.background_video_source_fps, 2
                ),
                "background_video_timing_mode": self.stats.background_video_timing_mode,
                "background_video_frames_displayed": (
                    self.stats.background_video_frames_displayed
                ),
                "background_video_frames_skipped": (
                    self.stats.background_video_frames_skipped
                ),
                "background_video_frames_reused": (
                    self.stats.background_video_frames_reused
                ),
                "background_video_skip_ratio": round(
                    self.stats.background_video_skip_ratio, 4
                ),
                "background_video_seek_count": self.stats.background_video_seek_count,
                "background_video_decode_failures": (
                    self.stats.background_video_decode_failures
                ),
                "background_video_orientation_status": (
                    self.stats.background_video_orientation_status
                ),
                "background_video_metadata_rotation": (
                    self.stats.background_video_metadata_rotation
                ),
                "background_video_auto_rotation_disabled": (
                    self.stats.background_video_auto_rotation_disabled
                ),
                "background_video_decoder_backend": (
                    self.stats.background_video_decoder_backend
                ),
                "background_video_color_status": (
                    self.stats.background_video_color_status
                ),
                "background_video_input_color": (
                    self.stats.background_video_input_color
                ),
                "background_video_output_color": (
                    self.stats.background_video_output_color
                ),
                "background_video_color_assumed_fields": list(
                    self.stats.background_video_color_assumed_fields
                ),
                "background_video_color_overridden_fields": list(
                    self.stats.background_video_color_overridden_fields
                ),
                "extensions": self._post_base_extensions_locked(),
                "uptime_s": round(time.time() - self.stats.started_at, 1),
            }

    @staticmethod
    def _rounded_optional(value: float | None, digits: int) -> float | None:
        return None if value is None else round(value, digits)
