"""Bounded temporal state for dynamic-backdrop light-wrap samples.

The compositor remains the owner of alpha arithmetic.  This module filters
only the already blurred backdrop sample used by light wrap, and therefore
cannot change the matte or the ordinary backdrop contribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

WorkingChannelOrder = Literal["bgr", "rgb"]

_NS_PER_SECOND = 1_000_000_000
_MAX_GAP_NS = 750_000_000
_SCENE_CUT_LUMA = 0.32
_SCENE_CUT_CHROMA = 0.45
_MAX_LUMA_CHANGE_PER_S = 1.50
_MAX_CHROMA_CHANGE_PER_S = 2.00
_BGR_LUMA = np.asarray((0.0722, 0.7152, 0.2126), dtype=np.float32)
_RGB_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


class LightWrapResetReason(str, Enum):
    """Why a light-wrap history was discarded."""

    INITIAL = "initial"
    BACKDROP_CHANGE = "backdrop-change"
    SEEK = "seek"
    SCENE_CUT = "scene-cut"
    NON_MONOTONIC_TIME = "non-monotonic-time"
    LONG_GAP = "long-gap"
    SHAPE_CHANGE = "shape-change"
    WORKING_SPACE_CHANGE = "working-space-change"


@dataclass(frozen=True)
class LightWrapFrameContext:
    """Identity and actual presentation time of one backdrop frame."""

    frame_id: int
    timestamp_ns: int
    source_token: tuple[object, ...]
    discontinuity_revision: int = 0

    def __post_init__(self) -> None:
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("light-wrap frame_id must be a non-negative integer")
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("light-wrap timestamp_ns must be a non-negative integer")
        if type(self.source_token) is not tuple or not self.source_token:
            raise ValueError("light-wrap source_token must be a non-empty tuple")
        if (
            type(self.discontinuity_revision) is not int
            or self.discontinuity_revision < 0
        ):
            raise ValueError(
                "light-wrap discontinuity_revision must be a non-negative integer"
            )


@dataclass(frozen=True)
class LightWrapSnapshot:
    """Content-free state and reset telemetry for evidence and tests."""

    updates: int
    repeated_frames: int
    reset_count: int
    scene_cut_count: int
    last_reset_reason: LightWrapResetReason | None
    last_frame_id: int | None
    last_timestamp_ns: int | None
    last_dt_s: float | None
    last_scene_luma_delta: float | None
    last_scene_chroma_delta: float | None
    retained_bytes: int
    state_shape: tuple[int, int, int] | None


def _validated_sample(sample: object, value_scale: float) -> np.ndarray:
    if (
        not isinstance(sample, np.ndarray)
        or sample.dtype != np.float32
        or sample.ndim != 3
        or sample.shape[2] != 3
        or sample.size == 0
        or not sample.flags.c_contiguous
    ):
        raise ValueError(
            "light-wrap sample must be a non-empty contiguous HxWx3 float32 array"
        )
    if (
        isinstance(value_scale, bool)
        or not isinstance(value_scale, (int, float))
        or not math.isfinite(float(value_scale))
        or float(value_scale) <= 0.0
    ):
        raise ValueError("light-wrap value_scale must be a positive finite number")
    if not np.isfinite(sample).all():
        raise ValueError("light-wrap sample must contain only finite values")
    minimum = float(np.min(sample))
    maximum = float(np.max(sample))
    if minimum < 0.0 or maximum > float(value_scale):
        raise ValueError("light-wrap sample lies outside its working-space range")
    return sample


def _scene_delta(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    value_scale: float,
    channel_order: WorkingChannelOrder,
) -> tuple[float, float]:
    """Return cut-sensitive luminance/chroma displacement in normalized units.

    Global channel displacement catches palette and illumination cuts. Mean
    per-pixel displacement over the already blurred low-resolution sample also
    catches hard spatial rearrangements with unchanged global means. Ordinary
    modest backdrop motion stays below the deliberately high cut thresholds;
    excessive motion safely resets instead of smearing old wrap color.
    """

    pixel_delta = (current.astype(np.float64) - previous.astype(np.float64)) / float(
        value_scale
    )
    delta = (
        current.mean(axis=(0, 1), dtype=np.float64)
        - previous.mean(axis=(0, 1), dtype=np.float64)
    ) / float(value_scale)
    weights = _BGR_LUMA if channel_order == "bgr" else _RGB_LUMA
    weights64 = weights.astype(np.float64)
    global_luma = float(np.dot(delta, weights64))
    global_chroma = delta - global_luma
    spatial_luma = np.sum(pixel_delta * weights64, axis=2, dtype=np.float64)
    spatial_chroma = pixel_delta - spatial_luma[..., None]
    return (
        max(
            abs(global_luma),
            float(np.mean(np.abs(spatial_luma), dtype=np.float64)),
        ),
        max(
            float(np.linalg.norm(global_chroma)),
            float(
                np.mean(
                    np.linalg.norm(spatial_chroma, axis=2),
                    dtype=np.float64,
                )
            ),
        ),
    )


class LightWrapStabilizer:
    """Elapsed-time low-pass with bounded luminance and chroma movement."""

    def __init__(self, time_constant_s: float) -> None:
        if (
            isinstance(time_constant_s, bool)
            or not isinstance(time_constant_s, (int, float))
            or not math.isfinite(float(time_constant_s))
            or not 0.01 <= float(time_constant_s) <= 1.0
        ):
            raise ValueError("light-wrap time_constant_s must be in [0.01, 1.0]")
        self.time_constant_s = float(time_constant_s)
        self._filtered: np.ndarray | None = None
        self._previous_raw: np.ndarray | None = None
        self._last_context: LightWrapFrameContext | None = None
        self._value_scale: float | None = None
        self._channel_order: WorkingChannelOrder | None = None
        self._updates = 0
        self._repeated_frames = 0
        self._reset_count = 0
        self._scene_cut_count = 0
        self._last_reset_reason: LightWrapResetReason | None = None
        self._last_dt_s: float | None = None
        self._last_scene_luma_delta: float | None = None
        self._last_scene_chroma_delta: float | None = None

    def clone(self) -> "LightWrapStabilizer":
        """Return a detached copy suitable for transactional activation trials."""

        clone = LightWrapStabilizer(self.time_constant_s)
        clone._filtered = None if self._filtered is None else self._filtered.copy()
        clone._previous_raw = (
            None if self._previous_raw is None else self._previous_raw.copy()
        )
        clone._last_context = self._last_context
        clone._value_scale = self._value_scale
        clone._channel_order = self._channel_order
        clone._updates = self._updates
        clone._repeated_frames = self._repeated_frames
        clone._reset_count = self._reset_count
        clone._scene_cut_count = self._scene_cut_count
        clone._last_reset_reason = self._last_reset_reason
        clone._last_dt_s = self._last_dt_s
        clone._last_scene_luma_delta = self._last_scene_luma_delta
        clone._last_scene_chroma_delta = self._last_scene_chroma_delta
        if clone._filtered is not None:
            clone._filtered.setflags(write=False)
        if clone._previous_raw is not None:
            clone._previous_raw.setflags(write=False)
        return clone

    def _install_current(
        self,
        sample: np.ndarray,
        context: LightWrapFrameContext,
        *,
        value_scale: float,
        channel_order: WorkingChannelOrder,
        reason: LightWrapResetReason,
        count_reset: bool,
    ) -> np.ndarray:
        filtered = sample.copy()
        previous_raw = sample.copy()
        filtered.setflags(write=False)
        previous_raw.setflags(write=False)
        self._filtered = filtered
        self._previous_raw = previous_raw
        self._last_context = context
        self._value_scale = float(value_scale)
        self._channel_order = channel_order
        self._updates += 1
        self._last_dt_s = None
        self._last_reset_reason = reason
        if reason is not LightWrapResetReason.SCENE_CUT:
            self._last_scene_luma_delta = None
            self._last_scene_chroma_delta = None
        if count_reset:
            self._reset_count += 1
        if reason is LightWrapResetReason.SCENE_CUT:
            self._scene_cut_count += 1
        return sample

    def update(
        self,
        sample: np.ndarray,
        context: LightWrapFrameContext,
        *,
        value_scale: float,
        channel_order: WorkingChannelOrder,
    ) -> np.ndarray:
        """Filter one unique backdrop sample in its active compositing space."""

        sample = _validated_sample(sample, value_scale)
        if not isinstance(context, LightWrapFrameContext):
            raise ValueError("light-wrap context must be a LightWrapFrameContext")
        if channel_order not in ("bgr", "rgb"):
            raise ValueError("light-wrap channel_order must be 'bgr' or 'rgb'")

        previous_context = self._last_context
        if previous_context is None:
            return self._install_current(
                sample,
                context,
                value_scale=value_scale,
                channel_order=channel_order,
                reason=LightWrapResetReason.INITIAL,
                count_reset=False,
            )

        if context.discontinuity_revision != previous_context.discontinuity_revision:
            return self._install_current(
                sample,
                context,
                value_scale=value_scale,
                channel_order=channel_order,
                reason=LightWrapResetReason.SEEK,
                count_reset=True,
            )
        if (
            self._filtered is None
            or self._previous_raw is None
            or self._filtered.shape != sample.shape
        ):
            return self._install_current(
                sample,
                context,
                value_scale=value_scale,
                channel_order=channel_order,
                reason=LightWrapResetReason.SHAPE_CHANGE,
                count_reset=True,
            )
        if context.source_token != previous_context.source_token:
            return self._install_current(
                sample,
                context,
                value_scale=value_scale,
                channel_order=channel_order,
                reason=LightWrapResetReason.BACKDROP_CHANGE,
                count_reset=True,
            )
        if (
            self._value_scale != float(value_scale)
            or self._channel_order != channel_order
        ):
            return self._install_current(
                sample,
                context,
                value_scale=value_scale,
                channel_order=channel_order,
                reason=LightWrapResetReason.WORKING_SPACE_CHANGE,
                count_reset=True,
            )
        if context.frame_id == previous_context.frame_id:
            self._repeated_frames += 1
            return self._filtered
        if context.frame_id < previous_context.frame_id:
            return self._install_current(
                sample,
                context,
                value_scale=value_scale,
                channel_order=channel_order,
                reason=LightWrapResetReason.NON_MONOTONIC_TIME,
                count_reset=True,
            )

        dt_ns = context.timestamp_ns - previous_context.timestamp_ns
        if dt_ns <= 0:
            return self._install_current(
                sample,
                context,
                value_scale=value_scale,
                channel_order=channel_order,
                reason=LightWrapResetReason.NON_MONOTONIC_TIME,
                count_reset=True,
            )
        if dt_ns > _MAX_GAP_NS:
            return self._install_current(
                sample,
                context,
                value_scale=value_scale,
                channel_order=channel_order,
                reason=LightWrapResetReason.LONG_GAP,
                count_reset=True,
            )

        luma_scene, chroma_scene = _scene_delta(
            self._previous_raw,
            sample,
            value_scale=float(value_scale),
            channel_order=channel_order,
        )
        self._last_scene_luma_delta = luma_scene
        self._last_scene_chroma_delta = chroma_scene
        if luma_scene >= _SCENE_CUT_LUMA or chroma_scene >= _SCENE_CUT_CHROMA:
            return self._install_current(
                sample,
                context,
                value_scale=value_scale,
                channel_order=channel_order,
                reason=LightWrapResetReason.SCENE_CUT,
                count_reset=True,
            )

        dt_s = dt_ns / _NS_PER_SECOND
        current_weight = 1.0 - math.exp(-dt_s / self.time_constant_s)
        proposed_delta = np.multiply(
            np.subtract(sample, self._filtered),
            np.float32(current_weight),
        )
        weights = _BGR_LUMA if channel_order == "bgr" else _RGB_LUMA
        luma_delta = np.sum(
            proposed_delta * weights,
            axis=2,
            dtype=np.float32,
        )
        luma_limit = np.float32(_MAX_LUMA_CHANGE_PER_S * dt_s * float(value_scale))
        bounded_luma = np.clip(luma_delta, -luma_limit, luma_limit)
        chroma_delta = proposed_delta - luma_delta[..., None]
        chroma_norm = np.sqrt(
            np.sum(chroma_delta * chroma_delta, axis=2, dtype=np.float32)
        )
        chroma_limit = np.float32(_MAX_CHROMA_CHANGE_PER_S * dt_s * float(value_scale))
        chroma_scale = np.minimum(
            np.float32(1.0),
            np.divide(
                chroma_limit,
                np.maximum(chroma_norm, np.float32(1e-12)),
            ),
        )
        bounded_delta = bounded_luma[..., None] + (
            chroma_delta * chroma_scale[..., None]
        )
        filtered = np.clip(
            self._filtered + bounded_delta,
            0.0,
            float(value_scale),
        )
        installed_filtered = np.ascontiguousarray(filtered, dtype=np.float32)
        previous_raw = sample.copy()
        installed_filtered.setflags(write=False)
        previous_raw.setflags(write=False)
        self._filtered = installed_filtered
        self._previous_raw = previous_raw
        self._last_context = context
        self._updates += 1
        self._last_dt_s = dt_s
        return self._filtered

    def snapshot(self) -> LightWrapSnapshot:
        arrays = (self._filtered, self._previous_raw)
        return LightWrapSnapshot(
            updates=self._updates,
            repeated_frames=self._repeated_frames,
            reset_count=self._reset_count,
            scene_cut_count=self._scene_cut_count,
            last_reset_reason=self._last_reset_reason,
            last_frame_id=(
                None if self._last_context is None else self._last_context.frame_id
            ),
            last_timestamp_ns=(
                None if self._last_context is None else self._last_context.timestamp_ns
            ),
            last_dt_s=self._last_dt_s,
            last_scene_luma_delta=self._last_scene_luma_delta,
            last_scene_chroma_delta=self._last_scene_chroma_delta,
            retained_bytes=sum(array.nbytes for array in arrays if array is not None),
            state_shape=(None if self._filtered is None else self._filtered.shape),
        )

    def close(self) -> None:
        """Release camera-derived temporal samples immediately."""

        self._filtered = None
        self._previous_raw = None
        self._last_context = None
        self._value_scale = None
        self._channel_order = None
