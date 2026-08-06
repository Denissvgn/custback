"""Explicit, native-preview-only matte diagnostic views.

This module deliberately has no API or :class:`~custback.hub.FrameHub`
integration.  A local HighGUI preview selects one view, the pipeline freezes
the already-produced matte evidence, and a depth-one daemon worker renders the
newest sample.  When no view is selected, :meth:`submit` returns before copying
or inspecting any full-frame array.

Live opaque-core, hole, and halo views are morphology-based *proxies*.  They
are useful for locating a suspect stage, but metric-authoritative defect
classification still requires the private replay annotations consumed by
``custback matte-diagnose``.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

import numpy as np

from .color import ColorTransform
from .compositor import PreparedLightWrap, composite
from .config import BlendSpace
from .matte_diagnostics import MatteCaptureMetadata, MatteFrameEvidence

try:
    import cv2
except ImportError:  # pragma: no cover - required by the application
    cv2 = None

log = logging.getLogger(__name__)


DiagnosticView = Literal[
    "raw_camera",
    "raw_alpha",
    "refined_alpha",
    "clean_foreground",
    "backdrop",
    "alpha_over_source",
    "uncertain_boundary",
    "opaque_core_deficit",
    "foreground_holes",
    "exterior_halo",
    "model_foreground_only",
    "light_wrap_only",
    "final_composite_contribution",
    "instability",
]


@dataclass(frozen=True)
class DiagnosticViewSpec:
    name: DiagnosticView
    label: str
    interpretation: str


DIAGNOSTIC_VIEWS: tuple[DiagnosticViewSpec, ...] = (
    DiagnosticViewSpec("raw_camera", "Raw camera", "exact source pixels"),
    DiagnosticViewSpec("raw_alpha", "Raw model alpha", "exact backend alpha"),
    DiagnosticViewSpec(
        "refined_alpha",
        "Refined alpha",
        "exact post-refiner/stabilizer alpha",
    ),
    DiagnosticViewSpec(
        "clean_foreground",
        "RVM clean foreground",
        "exact backend foreground when available",
    ),
    DiagnosticViewSpec("backdrop", "Exact backdrop", "exact composited backdrop"),
    DiagnosticViewSpec(
        "alpha_over_source",
        "Alpha over source",
        "refined alpha colour scale over exact source",
    ),
    DiagnosticViewSpec(
        "uncertain_boundary",
        "Uncertain boundary",
        "strict 0.05 < refined alpha < 0.95",
    ),
    DiagnosticViewSpec(
        "opaque_core_deficit",
        "Inferred opaque-core deficit",
        "morphology proxy; not ground-truth qualification evidence",
    ),
    DiagnosticViewSpec(
        "foreground_holes",
        "Inferred foreground holes",
        "enclosed-alpha proxy; not ground-truth qualification evidence",
    ),
    DiagnosticViewSpec(
        "exterior_halo",
        "Inferred exterior halo",
        "exterior-alpha proxy; not ground-truth qualification evidence",
    ),
    DiagnosticViewSpec(
        "model_foreground_only",
        "Model foreground only",
        "held-input counterfactual with model foreground and no light wrap",
    ),
    DiagnosticViewSpec(
        "light_wrap_only",
        "Light wrap only",
        "held-input counterfactual with light wrap and source foreground",
    ),
    DiagnosticViewSpec(
        "final_composite_contribution",
        "Final composite contribution",
        "pre-reaction base delta from the held-input plain alpha blend",
    ),
    DiagnosticViewSpec(
        "instability",
        "Flow-compensated instability",
        "red=registered alpha residual; cyan=held-input downstream edge-colour",
    ),
)

_VIEW_BY_NAME = {spec.name: spec for spec in DIAGNOSTIC_VIEWS}
_VIEW_NAMES: tuple[DiagnosticView, ...] = tuple(spec.name for spec in DIAGNOSTIC_VIEWS)
_UNAVAILABLE_COLOUR = (24, 24, 24)


@dataclass(frozen=True)
class MatteTemporalTelemetry:
    """Frame-paired temporal facts computed only for an active local sink."""

    capture_sequence_delta: int | None = None
    capture_timestamp_delta_ms: float | None = None
    raw_alpha_abs_diff: float | None = None
    raw_alpha_compensated_abs_diff: float | None = None
    refined_alpha_abs_diff: float | None = None
    refined_alpha_compensated_abs_diff: float | None = None
    edge_colour_abs_diff: float | None = None
    edge_colour_state: str = "warming"
    registration_dx_px: float | None = None
    registration_dy_px: float | None = None
    registration_response: float | None = None
    registration_overlap_fraction: float | None = None
    registration_state: str = "warming"
    history_state: str = "warming"

    def to_dict(self) -> dict[str, object]:
        return {
            "capture_sequence_delta": self.capture_sequence_delta,
            "capture_timestamp_delta_ms": self.capture_timestamp_delta_ms,
            "raw_alpha_abs_diff": self.raw_alpha_abs_diff,
            "raw_alpha_compensated_abs_diff": (self.raw_alpha_compensated_abs_diff),
            "refined_alpha_abs_diff": self.refined_alpha_abs_diff,
            "refined_alpha_compensated_abs_diff": (
                self.refined_alpha_compensated_abs_diff
            ),
            "edge_colour_abs_diff": self.edge_colour_abs_diff,
            "edge_colour_state": self.edge_colour_state,
            "registration_dx_px": self.registration_dx_px,
            "registration_dy_px": self.registration_dy_px,
            "registration_response": self.registration_response,
            "registration_overlap_fraction": self.registration_overlap_fraction,
            "registration_state": self.registration_state,
            "history_state": self.history_state,
        }


@dataclass(frozen=True)
class _FrozenEvidence:
    metadata: MatteCaptureMetadata
    raw_frame: np.ndarray
    raw_mask: np.ndarray | None
    refined_mask: np.ndarray | None
    clean_foreground: np.ndarray | None
    backdrop_frame: np.ndarray | None
    base_composite: np.ndarray | None
    prepared_light_wrap: PreparedLightWrap | None
    effective_controls: dict[str, Any]
    timings_ms: dict[str, float]
    compositor_substages_ms: dict[str, float]
    segmentation_diagnostics: dict[str, Any]
    color_transform: ColorTransform


@dataclass(frozen=True)
class _RenderedDiagnostic:
    pixels: np.ndarray
    available: bool
    reason: str


@dataclass(frozen=True)
class LocalMatteDiagnosticFrame:
    """One rendered private frame and its content-free local telemetry."""

    view: DiagnosticView
    label: str
    interpretation: str
    pixels: np.ndarray
    available: bool
    unavailable_reason: str
    capture_sequence: int
    capture_monotonic_ns: int
    capture_generation: int
    geometry_generation: int
    temporal: MatteTemporalTelemetry
    effective_controls: Mapping[str, Any]
    timings_ms: Mapping[str, float]
    compositor_substages_ms: Mapping[str, float]
    segmentation_diagnostics: Mapping[str, Any]
    status: Mapping[str, object]


def diagnostic_view_names() -> tuple[DiagnosticView, ...]:
    """Return the stable native-preview cycle order."""

    return _VIEW_NAMES


def diagnostic_view_spec(view: DiagnosticView) -> DiagnosticViewSpec:
    """Return the honest label and interpretation for one view."""

    return _VIEW_BY_NAME[view]


def _readonly_array(
    value: np.ndarray | None,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | type[np.generic] | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    if (
        not isinstance(value, np.ndarray)
        or value.size == 0
        or (shape is not None and value.shape != shape)
        or (dtype is not None and value.dtype != np.dtype(dtype))
    ):
        raise ValueError(f"{name} has an invalid diagnostic array contract")
    copied = np.array(value, copy=True, order="C")
    if copied.dtype == np.float32 and (
        not bool(np.isfinite(copied).all())
        or float(copied.min()) < 0.0
        or float(copied.max()) > 1.0
    ):
        raise ValueError(f"{name} must contain finite alpha in [0, 1]")
    copied.setflags(write=False)
    return copied


def _freeze_evidence(evidence: MatteFrameEvidence) -> _FrozenEvidence:
    if not isinstance(evidence, MatteFrameEvidence):
        raise TypeError("live matte diagnostics require MatteFrameEvidence")
    source = _readonly_array(
        evidence.raw_frame,
        name="raw frame",
        dtype=np.uint8,
    )
    assert source is not None
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("raw frame must be a uint8 BGR frame")
    frame_shape = source.shape
    mask_shape = frame_shape[:2]
    return _FrozenEvidence(
        metadata=evidence.metadata,
        raw_frame=source,
        raw_mask=_readonly_array(
            evidence.raw_mask,
            name="raw mask",
            shape=mask_shape,
            dtype=np.float32,
        ),
        refined_mask=_readonly_array(
            evidence.refined_mask,
            name="refined mask",
            shape=mask_shape,
            dtype=np.float32,
        ),
        clean_foreground=_readonly_array(
            evidence.clean_foreground,
            name="clean foreground",
            shape=frame_shape,
            dtype=np.uint8,
        ),
        backdrop_frame=_readonly_array(
            evidence.backdrop_frame,
            name="backdrop frame",
            shape=frame_shape,
            dtype=np.uint8,
        ),
        base_composite=_readonly_array(
            evidence.base_composite,
            name="base composite",
            shape=frame_shape,
            dtype=np.uint8,
        ),
        prepared_light_wrap=_freeze_prepared_light_wrap(
            evidence.prepared_light_wrap,
            frame_shape=frame_shape,
        ),
        effective_controls=copy.deepcopy(evidence.effective_controls),
        timings_ms={
            str(key): float(value) for key, value in evidence.timings_ms.items()
        },
        compositor_substages_ms={
            str(key): float(value)
            for key, value in evidence.compositor_substages_ms.items()
        },
        segmentation_diagnostics=copy.deepcopy(evidence.segmentation_diagnostics),
        color_transform=evidence.color_transform,
    )


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _mean_abs(first: np.ndarray, second: np.ndarray) -> float:
    return _round_metric(
        float(
            np.mean(
                np.abs(first.astype(np.float32) - second.astype(np.float32)),
                dtype=np.float64,
            )
        )
    )


def _mean_abs_on_support(
    first: np.ndarray,
    second: np.ndarray,
    support: np.ndarray,
) -> float:
    delta = np.abs(first.astype(np.float32) - second.astype(np.float32))
    return _round_metric(float(np.mean(delta[support], dtype=np.float64)))


def _freeze_prepared_light_wrap(
    prepared: PreparedLightWrap | None,
    *,
    frame_shape: tuple[int, int, int],
) -> PreparedLightWrap | None:
    if prepared is None:
        return None
    if not isinstance(prepared, PreparedLightWrap):
        raise ValueError("prepared light wrap has an invalid diagnostic contract")
    maximum = 255.0 if prepared.blend_space == "srgb_legacy" else 1.0
    pixels = prepared.pixels_bgr
    if (
        prepared.blend_space not in ("srgb_legacy", "linear_srgb")
        or type(prepared.stabilized) is not bool
        or not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.float32
        or pixels.shape != frame_shape
        or pixels.size == 0
        or not bool(np.isfinite(pixels).all())
        or float(pixels.min()) < 0.0
        or float(pixels.max()) > maximum
    ):
        raise ValueError("prepared light wrap has an invalid diagnostic contract")
    copied = np.array(pixels, copy=True, order="C")
    copied.setflags(write=False)
    return PreparedLightWrap(
        pixels_bgr=copied,
        blend_space=prepared.blend_space,
        stabilized=prepared.stabilized,
    )


@dataclass(frozen=True)
class _Registration:
    affine: np.ndarray
    dx: float | None
    dy: float | None
    response: float | None
    available: bool
    state: str


def _identity_registration(
    state: str,
    *,
    response: float | None = None,
) -> _Registration:
    return _Registration(
        affine=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        dx=None,
        dy=None,
        response=response,
        available=False,
        state=state,
    )


def _translation(
    previous: np.ndarray,
    current: np.ndarray,
) -> _Registration:
    if cv2 is None:
        return _identity_registration("opencv-unavailable")
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY).astype(np.float32)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
    try:
        (dx, dy), response = cv2.phaseCorrelate(previous_gray, current_gray)
    except cv2.error:
        return _identity_registration("registration-error")
    if not math.isfinite(dx) or not math.isfinite(dy) or not math.isfinite(response):
        return _identity_registration("non-finite")
    rounded_response = _round_metric(response)
    if response < 0.10:
        return _identity_registration(
            "low-confidence",
            response=rounded_response,
        )
    height, width = previous_gray.shape
    overlap_x = max(0.0, float(width) - abs(float(dx))) / float(width)
    overlap_y = max(0.0, float(height) - abs(float(dy))) / float(height)
    if overlap_x < 0.5 or overlap_y < 0.5:
        return _identity_registration(
            "implausible-displacement",
            response=rounded_response,
        )
    affine = np.asarray(
        [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        dtype=np.float32,
    )
    return _Registration(
        affine=affine,
        dx=float(dx),
        dy=float(dy),
        response=rounded_response,
        available=True,
        state="ready",
    )


def _warp(array: np.ndarray, affine: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if cv2 is None:
        return np.array(array, copy=True, order="C")
    return cv2.warpAffine(
        array,
        affine,
        (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _valid_overlap(
    affine: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    if cv2 is None:  # pragma: no cover - registration is unavailable without cv2
        return np.ones(shape, dtype=bool)
    source_support = np.ones(shape, dtype=np.uint8)
    warped = cv2.warpAffine(
        source_support,
        affine,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(bool)


def _same_temporal_epoch(
    previous: _FrozenEvidence,
    current: _FrozenEvidence,
    previous_status: Mapping[str, object],
    current_status: Mapping[str, object],
) -> bool:
    return (
        previous.raw_frame.shape == current.raw_frame.shape
        and previous.metadata.capture_generation == current.metadata.capture_generation
        and previous.metadata.geometry_generation
        == current.metadata.geometry_generation
        and current.metadata.capture_sequence > previous.metadata.capture_sequence
        and current.metadata.capture_monotonic_ns
        > previous.metadata.capture_monotonic_ns
        and previous_status.get("segmentation_generation")
        == current_status.get("segmentation_generation")
        and previous_status.get("matte_reset_count")
        == current_status.get("matte_reset_count")
        and previous_status.get("config_version")
        == current_status.get("config_version")
    )


def _temporal_metrics(
    previous: _FrozenEvidence | None,
    current: _FrozenEvidence,
    previous_status: Mapping[str, object] | None,
    current_status: Mapping[str, object],
) -> tuple[MatteTemporalTelemetry, np.ndarray | None, np.ndarray | None]:
    if previous is None or previous_status is None:
        return MatteTemporalTelemetry(), None, None
    sequence_delta = (
        current.metadata.capture_sequence - previous.metadata.capture_sequence
    )
    timestamp_delta_ms = (
        current.metadata.capture_monotonic_ns - previous.metadata.capture_monotonic_ns
    ) / 1_000_000.0
    if not _same_temporal_epoch(
        previous,
        current,
        previous_status,
        current_status,
    ):
        return (
            MatteTemporalTelemetry(
                capture_sequence_delta=sequence_delta,
                capture_timestamp_delta_ms=_round_metric(timestamp_delta_ms),
                edge_colour_state="reset",
                registration_state="reset",
                history_state="reset",
            ),
            None,
            None,
        )
    registration = _translation(previous.raw_frame, current.raw_frame)
    shape = current.raw_frame.shape[:2]
    overlap = (
        _valid_overlap(registration.affine, shape) if registration.available else None
    )
    overlap_fraction = (
        _round_metric(float(np.mean(overlap, dtype=np.float64)))
        if overlap is not None
        else None
    )
    raw_abs = raw_comp = refined_abs = refined_comp = None
    refined_delta: np.ndarray | None = None
    edge_colour_delta: np.ndarray | None = None
    if previous.raw_mask is not None and current.raw_mask is not None:
        raw_abs = _mean_abs(current.raw_mask, previous.raw_mask)
        if registration.available and overlap is not None:
            warped_raw = _warp(previous.raw_mask, registration.affine, shape)
            raw_comp = _mean_abs_on_support(current.raw_mask, warped_raw, overlap)
    if previous.refined_mask is not None and current.refined_mask is not None:
        refined_abs = _mean_abs(current.refined_mask, previous.refined_mask)
        if registration.available and overlap is not None:
            warped_refined = _warp(
                previous.refined_mask,
                registration.affine,
                shape,
            )
            refined_delta = np.abs(
                current.refined_mask.astype(np.float32)
                - warped_refined.astype(np.float32)
            )
            refined_delta = np.where(overlap, refined_delta, 0.0).astype(
                np.float32,
                copy=False,
            )
            refined_comp = _round_metric(
                float(np.mean(refined_delta[overlap], dtype=np.float64))
            )
        edge_state = "registration-unavailable"
        if (
            registration.available
            and refined_delta is not None
            and overlap is not None
            and previous.base_composite is not None
            and current.base_composite is not None
            and previous.backdrop_frame is not None
            and current.backdrop_frame is not None
        ):
            previous_plain = _plain_composite(previous)
            current_plain = _plain_composite(current)
            assert previous_plain is not None
            assert current_plain is not None
            previous_contribution = (
                previous.base_composite.astype(np.float32)
                - previous_plain.astype(np.float32)
            ) / 255.0
            current_contribution = (
                current.base_composite.astype(np.float32)
                - current_plain.astype(np.float32)
            ) / 255.0
            warped_previous_contribution = _warp(
                previous_contribution,
                registration.affine,
                shape,
            )
            rgb_delta = np.mean(
                np.abs(current_contribution - warped_previous_contribution),
                axis=2,
                dtype=np.float32,
            )
            uncertain = (
                (current.refined_mask > 0.05)
                & (current.refined_mask < 0.95)
                & (refined_delta <= 0.01)
                & overlap
            )
            edge_colour_delta = np.where(uncertain, rgb_delta, 0.0).astype(
                np.float32,
                copy=False,
            )
            if bool(np.any(uncertain)):
                edge_metric = _round_metric(
                    float(np.mean(rgb_delta[uncertain], dtype=np.float64))
                )
                edge_state = "ready"
            else:
                edge_metric = None
                edge_state = "no-stable-edge-support"
        else:
            edge_metric = None
            if registration.available:
                edge_state = "contribution-unavailable"
    else:
        edge_metric = None
        edge_state = "alpha-unavailable"
    return (
        MatteTemporalTelemetry(
            capture_sequence_delta=sequence_delta,
            capture_timestamp_delta_ms=_round_metric(timestamp_delta_ms),
            raw_alpha_abs_diff=raw_abs,
            raw_alpha_compensated_abs_diff=raw_comp,
            refined_alpha_abs_diff=refined_abs,
            refined_alpha_compensated_abs_diff=refined_comp,
            edge_colour_abs_diff=edge_metric,
            edge_colour_state=edge_state,
            registration_dx_px=(
                None if registration.dx is None else _round_metric(registration.dx)
            ),
            registration_dy_px=(
                None if registration.dy is None else _round_metric(registration.dy)
            ),
            registration_response=registration.response,
            registration_overlap_fraction=overlap_fraction,
            registration_state=registration.state,
            history_state="ready",
        ),
        refined_delta,
        edge_colour_delta,
    )


def _gray_alpha(alpha: np.ndarray) -> np.ndarray:
    gray = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=2))


def _heatmap(values: np.ndarray, *, support: np.ndarray | None = None) -> np.ndarray:
    normalized = np.clip(values.astype(np.float32), 0.0, 1.0)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    if cv2 is None:
        colored = np.repeat(gray[..., None], 3, axis=2)
    else:
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    if support is not None:
        colored[~support.astype(bool)] = 0
    return np.ascontiguousarray(colored)


def _unavailable(sample: _FrozenEvidence, reason: str) -> _RenderedDiagnostic:
    slate = np.full(sample.raw_frame.shape, _UNAVAILABLE_COLOUR, dtype=np.uint8)
    return _RenderedDiagnostic(slate, False, reason)


def _required_matte(
    sample: _FrozenEvidence,
) -> tuple[np.ndarray, np.ndarray] | None:
    if sample.refined_mask is None or sample.backdrop_frame is None:
        return None
    return sample.refined_mask, sample.backdrop_frame


def _blend_space(sample: _FrozenEvidence) -> BlendSpace:
    value = sample.effective_controls.get("blend_space", "srgb_legacy")
    return cast(
        BlendSpace,
        value if value in ("srgb_legacy", "linear_srgb") else "srgb_legacy",
    )


def _light_wrap(sample: _FrozenEvidence) -> float:
    value = sample.effective_controls.get("light_wrap", 0.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _plain_composite(sample: _FrozenEvidence) -> np.ndarray | None:
    required = _required_matte(sample)
    if required is None:
        return None
    alpha, backdrop = required
    return composite(
        sample.raw_frame,
        backdrop,
        alpha,
        blend_space=_blend_space(sample),
        color_transform=sample.color_transform,
    )


def _inferred_holes(binary: np.ndarray) -> np.ndarray:
    inverse = (~binary).astype(np.uint8)
    if cv2 is None:
        return np.zeros_like(binary)
    padded = np.pad(inverse, 1, mode="constant", constant_values=1)
    exterior = padded.copy()
    flood_mask = np.zeros(
        (exterior.shape[0] + 2, exterior.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(exterior, flood_mask, (0, 0), 2)
    exterior = exterior[1:-1, 1:-1] == 2
    return inverse.astype(bool) & ~exterior


def render_diagnostic_view(
    sample: _FrozenEvidence,
    view: DiagnosticView,
    *,
    refined_instability: np.ndarray | None = None,
    edge_colour_instability: np.ndarray | None = None,
    instability_unavailable_reason: str | None = None,
) -> _RenderedDiagnostic:
    """Render one view without mutating any frozen or production array."""

    if view == "raw_camera":
        return _RenderedDiagnostic(sample.raw_frame.copy(), True, "")
    if view == "raw_alpha":
        if sample.raw_mask is None:
            return _unavailable(sample, "raw model alpha is unavailable")
        return _RenderedDiagnostic(_gray_alpha(sample.raw_mask), True, "")
    if view == "refined_alpha":
        if sample.refined_mask is None:
            return _unavailable(sample, "refined alpha is unavailable")
        return _RenderedDiagnostic(_gray_alpha(sample.refined_mask), True, "")
    if view == "clean_foreground":
        if sample.clean_foreground is None:
            return _unavailable(
                sample,
                "the active backend did not provide a clean foreground",
            )
        return _RenderedDiagnostic(sample.clean_foreground.copy(), True, "")
    if view == "backdrop":
        if sample.backdrop_frame is None:
            return _unavailable(sample, "this output mode has no matte backdrop")
        return _RenderedDiagnostic(sample.backdrop_frame.copy(), True, "")
    if sample.refined_mask is None:
        return _unavailable(sample, "this output mode did not produce matte evidence")

    alpha = sample.refined_mask
    if view == "alpha_over_source":
        heat = _heatmap(alpha)
        mixed = np.rint(
            sample.raw_frame.astype(np.float32) * 0.45 + heat.astype(np.float32) * 0.55
        ).astype(np.uint8)
        return _RenderedDiagnostic(np.ascontiguousarray(mixed), True, "")
    if view == "uncertain_boundary":
        uncertain = (alpha > 0.05) & (alpha < 0.95)
        values = np.where(uncertain, 4.0 * alpha * (1.0 - alpha), 0.0)
        return _RenderedDiagnostic(
            _heatmap(values, support=uncertain),
            True,
            "",
        )
    binary = alpha >= 0.5
    if view == "opaque_core_deficit":
        if cv2 is None:
            core = binary
        else:
            radius = max(1, min(15, round(min(alpha.shape) / 160)))
            kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
            core = cv2.erode(binary.astype(np.uint8), kernel).astype(bool)
        return _RenderedDiagnostic(
            _heatmap((1.0 - alpha) * core, support=core),
            True,
            "",
        )
    if view == "foreground_holes":
        holes = _inferred_holes(binary)
        return _RenderedDiagnostic(
            _heatmap((1.0 - alpha) * holes, support=holes),
            True,
            "",
        )
    if view == "exterior_halo":
        exterior = ~binary & ~_inferred_holes(binary)
        support = exterior & (alpha > 0.05)
        return _RenderedDiagnostic(
            _heatmap(alpha * exterior, support=support),
            True,
            "",
        )
    required = _required_matte(sample)
    if required is None:
        return _unavailable(sample, "the held-input compositor tracks are unavailable")
    alpha, backdrop = required
    if view == "model_foreground_only":
        if sample.clean_foreground is None:
            return _unavailable(
                sample,
                "the active backend did not provide a clean foreground",
            )
        rendered = composite(
            sample.raw_frame,
            backdrop,
            alpha,
            edge_foreground=sample.clean_foreground,
            blend_space=_blend_space(sample),
            color_transform=sample.color_transform,
        )
        return _RenderedDiagnostic(rendered, True, "")
    if view == "light_wrap_only":
        rendered = composite(
            sample.raw_frame,
            backdrop,
            alpha,
            light_wrap=_light_wrap(sample),
            blend_space=_blend_space(sample),
            color_transform=sample.color_transform,
            prepared_light_wrap=sample.prepared_light_wrap,
        )
        return _RenderedDiagnostic(rendered, True, "")
    if view == "final_composite_contribution":
        if sample.base_composite is None:
            return _unavailable(
                sample, "the pre-reaction base composite is unavailable"
            )
        plain = _plain_composite(sample)
        if plain is None:  # pragma: no cover - guarded by required matte above
            return _unavailable(sample, "the plain alpha counterfactual is unavailable")
        magnitude = (
            np.mean(
                np.abs(
                    sample.base_composite.astype(np.float32) - plain.astype(np.float32)
                ),
                axis=2,
                dtype=np.float32,
            )
            / 255.0
        )
        return _RenderedDiagnostic(_heatmap(magnitude), True, "")
    if view == "instability":
        if refined_instability is None:
            return _unavailable(
                sample,
                instability_unavailable_reason
                or "a prior compatible unique input is required",
            )
        alpha_signal = np.clip(refined_instability * 4.0, 0.0, 1.0)
        colour_signal = (
            np.zeros_like(alpha_signal)
            if edge_colour_instability is None
            else np.clip(edge_colour_instability * 4.0, 0.0, 1.0)
        )
        rendered = np.zeros((*alpha.shape, 3), dtype=np.uint8)
        rendered[..., 0] = np.rint(colour_signal * 255.0).astype(np.uint8)
        rendered[..., 1] = rendered[..., 0]
        rendered[..., 2] = np.rint(alpha_signal * 255.0).astype(np.uint8)
        return _RenderedDiagnostic(rendered, True, "")
    raise ValueError(f"unknown diagnostic view: {view}")


class LocalMatteDiagnosticMonitor:
    """Depth-one private monitor activated only by the native preview."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = threading.Event()
        self._view: DiagnosticView | None = None
        self._epoch = 0
        self._pending: tuple[_FrozenEvidence, dict[str, object], int] | None = None
        self._latest_sample: _FrozenEvidence | None = None
        self._previous_sample: _FrozenEvidence | None = None
        self._previous_status: dict[str, object] | None = None
        self._latest_frame: LocalMatteDiagnosticFrame | None = None
        self._sequence = 0
        self._worker: threading.Thread | None = None
        self._closed = False

    @property
    def accepting(self) -> bool:
        """Cheap inactive-path probe used by the frame worker."""

        return self._active.is_set()

    @property
    def selected_view(self) -> DiagnosticView | None:
        with self._condition:
            return self._view

    def select(self, view: DiagnosticView | None) -> DiagnosticView | None:
        if view is not None and view not in _VIEW_BY_NAME:
            raise ValueError(f"unknown diagnostic view: {view}")
        with self._condition:
            if self._closed:
                return None
            if view is None:
                self._deactivate_locked()
                return None
            changed = self._view != view
            self._view = view
            self._active.set()
            if changed:
                self._epoch += 1
                self._latest_frame = None
            self._ensure_worker_locked()
            self._condition.notify_all()
            return view

    def select_next(self, *, reverse: bool = False) -> DiagnosticView | None:
        """Cycle output -> views -> output, clearing sensitive history at output."""

        current = self.selected_view
        if current is None:
            return self.select(_VIEW_NAMES[-1] if reverse else _VIEW_NAMES[0])
        index = _VIEW_NAMES.index(current) + (-1 if reverse else 1)
        if index < 0 or index >= len(_VIEW_NAMES):
            return self.select(None)
        return self.select(_VIEW_NAMES[index])

    def deactivate(self) -> None:
        self.select(None)

    def _deactivate_locked(self) -> None:
        self._epoch += 1
        self._view = None
        self._active.clear()
        self._pending = None
        self._latest_sample = None
        self._previous_sample = None
        self._previous_status = None
        self._latest_frame = None
        self._sequence += 1
        self._condition.notify_all()

    def clear_history(self) -> None:
        """Forget sensitive samples while preserving the selected view."""

        with self._condition:
            self._epoch += 1
            self._pending = None
            self._latest_sample = None
            self._previous_sample = None
            self._previous_status = None
            self._latest_frame = None
            self._sequence += 1
            self._condition.notify_all()

    def close(self) -> None:
        worker: threading.Thread | None
        with self._condition:
            if self._closed:
                return
            self._deactivate_locked()
            self._closed = True
            worker = self._worker
            self._condition.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5.0)
            if worker.is_alive():
                log.warning(
                    "local matte diagnostic renderer did not stop within 5 seconds"
                )

    def submit(
        self,
        evidence: MatteFrameEvidence,
        *,
        status: Mapping[str, object],
    ) -> bool:
        """Freeze and enqueue the newest sample, dropping superseded work."""

        if not self._active.is_set():
            return False
        frozen = _freeze_evidence(evidence)
        status_copy = copy.deepcopy(dict(status))
        with self._condition:
            if self._closed or self._view is None:
                return False
            self._pending = (frozen, status_copy, self._epoch)
            self._ensure_worker_locked()
            self._condition.notify_all()
        return True

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run,
            name="matte-live-diagnostics",
            daemon=True,
        )
        self._worker.start()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._pending is not None
                )
                if self._closed:
                    return
                pending = self._pending
                self._pending = None
                view = self._view
                previous = self._previous_sample
                previous_status = self._previous_status
            if pending is None or view is None:
                continue
            self._render_one(pending, view, previous, previous_status)
            # Do not retain private rasters in an idle worker stack after the
            # monitor's owned references are cleared on deactivation.
            pending = None
            previous = None
            previous_status = None
            view = None

    def _render_one(
        self,
        pending: tuple[_FrozenEvidence, dict[str, object], int],
        view: DiagnosticView,
        previous: _FrozenEvidence | None,
        previous_status: Mapping[str, object] | None,
    ) -> None:
        sample, status, epoch = pending
        try:
            temporal, alpha_instability, colour_instability = _temporal_metrics(
                previous,
                sample,
                previous_status,
                status,
            )
            rendered = render_diagnostic_view(
                sample,
                view,
                refined_instability=alpha_instability,
                edge_colour_instability=colour_instability,
                instability_unavailable_reason=(
                    (
                        "source registration is "
                        f"{temporal.registration_state.replace('-', ' ')}"
                    )
                    if temporal.history_state == "ready"
                    and temporal.registration_state != "ready"
                    else (
                        "refined alpha is unavailable for this frame pair"
                        if temporal.history_state == "ready"
                        and sample.refined_mask is None
                        else None
                    )
                ),
            )
        except Exception:
            log.exception("cannot render local matte diagnostic view %s", view)
            temporal = MatteTemporalTelemetry(history_state="render-error")
            rendered = _unavailable(sample, "diagnostic rendering failed")
        spec = diagnostic_view_spec(view)
        pixels = np.array(rendered.pixels, copy=True, order="C")
        pixels.setflags(write=False)
        frame = LocalMatteDiagnosticFrame(
            view=view,
            label=spec.label,
            interpretation=spec.interpretation,
            pixels=pixels,
            available=rendered.available,
            unavailable_reason=rendered.reason,
            capture_sequence=sample.metadata.capture_sequence,
            capture_monotonic_ns=sample.metadata.capture_monotonic_ns,
            capture_generation=sample.metadata.capture_generation,
            geometry_generation=sample.metadata.geometry_generation,
            temporal=temporal,
            effective_controls=copy.deepcopy(sample.effective_controls),
            timings_ms=dict(sample.timings_ms),
            compositor_substages_ms=dict(sample.compositor_substages_ms),
            segmentation_diagnostics=copy.deepcopy(sample.segmentation_diagnostics),
            status=copy.deepcopy(status),
        )
        with self._condition:
            if self._closed or self._view != view or self._epoch != epoch:
                return
            self._previous_sample = sample
            self._previous_status = status
            self._latest_sample = sample
            self._latest_frame = frame
            self._sequence += 1
            self._condition.notify_all()

    def get(
        self,
        last_sequence: int = -1,
        timeout: float | None = None,
    ) -> tuple[LocalMatteDiagnosticFrame | None, int]:
        """Return a rendered frame newer than ``last_sequence``."""

        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self._closed
                    or (
                        self._latest_frame is not None
                        and self._sequence != last_sequence
                    )
                ),
                timeout=timeout,
            )
            if not ready or self._latest_frame is None:
                return None, last_sequence
            return self._latest_frame, self._sequence

    def latest(self) -> LocalMatteDiagnosticFrame | None:
        with self._condition:
            return self._latest_frame
