"""Typed backend-specific matte policy resolution.

Persisted segmentation and compositing values describe configured intent.  The
active backend can legitimately make some of those values effective, bypassed,
or inapplicable.  This module is the single authority for that distinction.

The complete snapshot is the sole runtime/evidence policy contract. MATTE-4.1
publishes that same path-free snapshot through the versioned status transport;
the older flat status fields remain compatibility projections of it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal

from .config import (
    CompositingConfig,
    SegmentationConfig,
    spatial_edge_refinement_radius,
)

PolicyValue = bool | int | float | str | None
RawAlphaMode = Literal[
    "native_soft_alpha",
    "confidence_soft_mask",
    "thresholded_binary_mask",
    "opaque_passthrough",
]
OpaqueCoreMode = Literal[
    "model_alpha_no_calibration",
    "confidence_mask_no_calibration",
    "heuristic_threshold",
    "none",
]
HaloMode = Literal["mask_shift_only", "generic_postprocess", "none"]
ResidualTemporalMode = Literal[
    "model_only",
    "explicit_motion_aware",
    "generic_temporal_policy",
    "none",
]


class MatteBackendKind(str, Enum):
    """Matte semantics supplied by the actually selected runtime backend."""

    TRUE_ALPHA_RECURRENT = "true_alpha_recurrent"
    CONFIDENCE_MASK_VIDEO = "confidence_mask_video"
    BINARY_COARSE = "binary_coarse"
    NULL_PASSTHROUGH = "null_passthrough"


class MatteControlState(str, Enum):
    """Whether one configured control can affect the active matte path."""

    EFFECTIVE = "effective"
    BYPASSED = "bypassed"
    INAPPLICABLE = "inapplicable"


@dataclass(frozen=True)
class MatteControl:
    """Configured intent and the value actually consumed by the runtime."""

    configured: PolicyValue
    effective: PolicyValue
    state: MatteControlState
    reason: str


@dataclass(frozen=True)
class ConfiguredMattePolicy:
    """Exact configured matte/compositor intent relevant to backend policy."""

    rvm_downsample_ratio: float
    threshold: float
    mask_blur: int
    edge_refine: bool
    edge_refinement_mode: str
    edge_refinement_reference_short_edge_px: int
    edge_refinement_radius_at_reference_px: int
    edge_refinement_min_radius_px: int
    edge_refinement_max_radius_px: int
    mask_shift: int
    temporal_smoothing: float
    boundary_stabilization_mode: str
    boundary_stabilization_time_constant_s: float
    boundary_stabilization_max_motion_px_per_s: float
    use_model_foreground: bool
    light_wrap: float
    light_wrap_stabilization_mode: str
    light_wrap_stabilization_time_constant_s: float


@dataclass(frozen=True)
class EffectiveMattePolicy:
    """Values and alpha semantics exercised by the active runtime path."""

    raw_alpha_mode: RawAlphaMode
    opaque_core_mode: OpaqueCoreMode
    halo_mode: HaloMode
    residual_temporal_mode: ResidualTemporalMode
    rvm_downsample_ratio: float | None
    threshold: float | None
    mask_blur: int
    edge_refine: bool
    edge_refinement_mode: str
    edge_refinement_radius_px: int
    mask_shift: int
    temporal_smoothing: float
    boundary_stabilization_mode: str
    boundary_stabilization_time_constant_s: float
    boundary_stabilization_max_motion_px_per_s: float
    use_model_foreground: bool
    light_wrap: float
    light_wrap_stabilization_mode: str
    light_wrap_stabilization_time_constant_s: float


@dataclass(frozen=True)
class MattePolicyControls:
    """Per-control applicability metadata for operator and evidence consumers."""

    rvm_downsample_ratio: MatteControl
    raw_alpha: MatteControl
    threshold: MatteControl
    mask_blur: MatteControl
    edge_refine: MatteControl
    mask_shift: MatteControl
    temporal_smoothing: MatteControl
    boundary_stabilization: MatteControl
    use_model_foreground: MatteControl
    light_wrap: MatteControl
    light_wrap_stabilization: MatteControl
    opaque_core_halo: MatteControl


@dataclass(frozen=True)
class MattePolicySnapshot:
    """One immutable configured-versus-effective backend policy snapshot."""

    selected_backend_kind: MatteBackendKind
    backend_kind: MatteBackendKind
    passthrough: bool
    experimental_rvm_generic: bool
    configured: ConfiguredMattePolicy
    effective: EffectiveMattePolicy
    controls: MattePolicyControls

    def effective_refiner_config(
        self,
        configured: SegmentationConfig,
    ) -> SegmentationConfig:
        """Return the refiner config mechanically selected by this snapshot."""

        boundary = configured.boundary_stabilization.model_copy(
            update={"mode": self.effective.boundary_stabilization_mode},
            deep=True,
        )
        return configured.model_copy(
            update={
                "mask_blur": self.effective.mask_blur,
                "edge_refine": self.effective.edge_refine,
                "mask_shift": self.effective.mask_shift,
                "temporal_smoothing": self.effective.temporal_smoothing,
                "boundary_stabilization": boundary,
            },
            deep=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible evidence representation."""

        return asdict(self)


def _configured(segmentation: SegmentationConfig, compositing: CompositingConfig):
    spatial = segmentation.spatial_edge_refinement
    boundary = segmentation.boundary_stabilization
    return ConfiguredMattePolicy(
        rvm_downsample_ratio=segmentation.rvm_downsample,
        threshold=segmentation.threshold,
        mask_blur=segmentation.mask_blur,
        edge_refine=segmentation.edge_refine,
        edge_refinement_mode=spatial.mode,
        edge_refinement_reference_short_edge_px=spatial.reference_short_edge_px,
        edge_refinement_radius_at_reference_px=spatial.radius_at_reference_px,
        edge_refinement_min_radius_px=spatial.min_radius_px,
        edge_refinement_max_radius_px=spatial.max_radius_px,
        mask_shift=segmentation.mask_shift,
        temporal_smoothing=segmentation.temporal_smoothing,
        boundary_stabilization_mode=boundary.mode,
        boundary_stabilization_time_constant_s=boundary.time_constant_s,
        boundary_stabilization_max_motion_px_per_s=boundary.max_motion_px_per_s,
        use_model_foreground=compositing.use_model_foreground,
        light_wrap=compositing.light_wrap,
        light_wrap_stabilization_mode=(compositing.light_wrap_stabilization.mode),
        light_wrap_stabilization_time_constant_s=(
            compositing.light_wrap_stabilization.time_constant_s
        ),
    )


def _control(
    configured: PolicyValue,
    effective: PolicyValue,
    state: MatteControlState,
    reason: str,
) -> MatteControl:
    return MatteControl(configured, effective, state, reason)


def _enabled_control(
    configured: PolicyValue,
    effective: PolicyValue,
    enabled: bool,
) -> MatteControl:
    if enabled:
        return _control(
            configured,
            effective,
            MatteControlState.EFFECTIVE,
            "configured-active",
        )
    return _control(
        configured,
        effective,
        MatteControlState.BYPASSED,
        "configured-off",
    )


def _neutral_boundary(
    configured: ConfiguredMattePolicy,
) -> tuple[str, float, float]:
    return (
        "off",
        configured.boundary_stabilization_time_constant_s,
        configured.boundary_stabilization_max_motion_px_per_s,
    )


def resolve_matte_policy(
    segmentation: SegmentationConfig,
    compositing: CompositingConfig,
    backend_kind: MatteBackendKind,
    *,
    resolved_rvm_ratio: float | None = None,
    passthrough: bool = False,
    canvas_shape: tuple[int, int] | None = None,
    experimental_rvm_generic: bool = False,
    light_wrap_stabilization_eligible: bool = True,
) -> MattePolicySnapshot:
    """Resolve configured intent for one actually selected backend.

    ``backend_kind`` must describe the constructed backend; requested
    ``segmentation.backend == "auto"`` is not sufficient because it may have
    selected any of the four policies.  ``resolved_rvm_ratio`` is deliberately
    runtime input so an automatic ratio remains unknown until RVM successfully
    processes its first frame.
    """

    if not isinstance(backend_kind, MatteBackendKind):
        raise TypeError("matte policy requires an actual MatteBackendKind")
    if type(passthrough) is not bool:
        raise TypeError("matte policy passthrough flag must be boolean")
    if type(experimental_rvm_generic) is not bool:
        raise TypeError("experimental RVM generic policy flag must be boolean")
    if type(light_wrap_stabilization_eligible) is not bool:
        raise TypeError("light-wrap stabilization eligibility flag must be boolean")
    if experimental_rvm_generic and (
        backend_kind is not MatteBackendKind.TRUE_ALPHA_RECURRENT or passthrough
    ):
        raise ValueError(
            "experimental RVM generic policy requires an active RVM matte path"
        )
    if resolved_rvm_ratio is not None and (
        isinstance(resolved_rvm_ratio, bool)
        or not isinstance(resolved_rvm_ratio, (int, float))
        or not 0.0 < float(resolved_rvm_ratio) <= 1.0
    ):
        raise ValueError("resolved RVM ratio must be in (0, 1]")
    if canvas_shape is not None:
        if (
            type(canvas_shape) is not tuple
            or len(canvas_shape) != 2
            or any(type(value) is not int or value <= 0 for value in canvas_shape)
        ):
            raise ValueError(
                "matte policy canvas shape must contain positive integer "
                "(height, width)"
            )

    configured = _configured(segmentation, compositing)
    effective_kind = MatteBackendKind.NULL_PASSTHROUGH if passthrough else backend_kind

    if effective_kind is MatteBackendKind.NULL_PASSTHROUGH:
        boundary_mode, boundary_tau, boundary_motion = _neutral_boundary(configured)
        effective = EffectiveMattePolicy(
            raw_alpha_mode="opaque_passthrough",
            opaque_core_mode="none",
            halo_mode="none",
            residual_temporal_mode="none",
            rvm_downsample_ratio=None,
            threshold=None,
            mask_blur=0,
            edge_refine=False,
            edge_refinement_mode="off",
            edge_refinement_radius_px=0,
            mask_shift=0,
            temporal_smoothing=0.0,
            boundary_stabilization_mode=boundary_mode,
            boundary_stabilization_time_constant_s=boundary_tau,
            boundary_stabilization_max_motion_px_per_s=boundary_motion,
            use_model_foreground=False,
            light_wrap=0.0,
            light_wrap_stabilization_mode="off",
            light_wrap_stabilization_time_constant_s=(
                configured.light_wrap_stabilization_time_constant_s
            ),
        )
        controls = MattePolicyControls(
            rvm_downsample_ratio=_control(
                configured.rvm_downsample_ratio,
                None,
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-rvm-inference",
            ),
            raw_alpha=_control(
                None,
                "opaque_passthrough",
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-matte",
            ),
            threshold=_control(
                configured.threshold,
                None,
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-mask-threshold",
            ),
            mask_blur=_control(
                configured.mask_blur,
                0,
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-matte-refiner",
            ),
            edge_refine=_control(
                configured.edge_refine,
                False,
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-matte-refiner",
            ),
            mask_shift=_control(
                configured.mask_shift,
                0,
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-matte-refiner",
            ),
            temporal_smoothing=_control(
                configured.temporal_smoothing,
                0.0,
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-temporal-matte",
            ),
            boundary_stabilization=_control(
                configured.boundary_stabilization_mode,
                "off",
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-temporal-matte",
            ),
            use_model_foreground=_control(
                configured.use_model_foreground,
                False,
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-model-foreground",
            ),
            light_wrap=_control(
                configured.light_wrap,
                0.0,
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-soft-edge-composite",
            ),
            light_wrap_stabilization=_control(
                configured.light_wrap_stabilization_mode,
                "off",
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-soft-edge-composite",
            ),
            opaque_core_halo=_control(
                None,
                "none",
                MatteControlState.INAPPLICABLE,
                "null-or-passthrough-has-no-matte",
            ),
        )
        return MattePolicySnapshot(
            selected_backend_kind=backend_kind,
            backend_kind=effective_kind,
            passthrough=passthrough,
            experimental_rvm_generic=False,
            configured=configured,
            effective=effective,
            controls=controls,
        )

    boundary_active = configured.boundary_stabilization_mode == "motion_aware"
    boundary_mode = "motion_aware" if boundary_active else "off"
    temporal_smoothing = 0.0 if boundary_active else configured.temporal_smoothing
    boundary_control = _enabled_control(
        configured.boundary_stabilization_mode,
        boundary_mode,
        boundary_active,
    )
    if boundary_active:
        temporal_control = _control(
            configured.temporal_smoothing,
            0.0,
            MatteControlState.BYPASSED,
            "replaced-by-motion-aware",
        )
    else:
        temporal_control = _enabled_control(
            configured.temporal_smoothing,
            temporal_smoothing,
            temporal_smoothing > 0.0,
        )

    mask_shift = configured.mask_shift
    mask_shift_control = _enabled_control(mask_shift, mask_shift, mask_shift != 0)
    light_wrap = configured.light_wrap
    light_wrap_control = _enabled_control(
        light_wrap,
        light_wrap,
        light_wrap > 0.0,
    )
    light_wrap_stabilization_active = (
        light_wrap > 0.0
        and configured.light_wrap_stabilization_mode == "temporal_bounded"
        and light_wrap_stabilization_eligible
    )
    light_wrap_stabilization_mode = (
        "temporal_bounded" if light_wrap_stabilization_active else "off"
    )
    if light_wrap <= 0.0:
        light_wrap_stabilization_control = _control(
            configured.light_wrap_stabilization_mode,
            "off",
            MatteControlState.BYPASSED,
            "light-wrap-strength-is-zero",
        )
    elif (
        configured.light_wrap_stabilization_mode == "temporal_bounded"
        and not light_wrap_stabilization_eligible
    ):
        light_wrap_stabilization_control = _control(
            configured.light_wrap_stabilization_mode,
            "off",
            MatteControlState.BYPASSED,
            "backdrop-has-no-dynamic-timeline",
        )
    else:
        light_wrap_stabilization_control = _enabled_control(
            configured.light_wrap_stabilization_mode,
            light_wrap_stabilization_mode,
            light_wrap_stabilization_active,
        )

    if effective_kind is MatteBackendKind.TRUE_ALPHA_RECURRENT:
        rvm_edge_refine = experimental_rvm_generic and configured.edge_refine
        rvm_edge_radius = (
            spatial_edge_refinement_radius(
                segmentation.spatial_edge_refinement,
                canvas_shape,
            )
            if rvm_edge_refine and canvas_shape is not None
            else 0
        )
        rvm_mask_blur = configured.mask_blur if experimental_rvm_generic else 0
        rvm_temporal_smoothing = temporal_smoothing if experimental_rvm_generic else 0.0
        rvm_generic_halo = rvm_edge_refine or rvm_mask_blur > 0
        resolved_ratio = (
            None if resolved_rvm_ratio is None else float(resolved_rvm_ratio)
        )
        ratio_reason = (
            "awaiting-first-rvm-inference"
            if resolved_ratio is None
            else (
                "runtime-auto-ratio"
                if configured.rvm_downsample_ratio == 0.0
                else "configured-ratio-resolved"
            )
        )
        ratio_control = _control(
            configured.rvm_downsample_ratio,
            resolved_ratio,
            MatteControlState.EFFECTIVE,
            ratio_reason,
        )
        effective = EffectiveMattePolicy(
            raw_alpha_mode="native_soft_alpha",
            opaque_core_mode="model_alpha_no_calibration",
            halo_mode=(
                "generic_postprocess" if rvm_generic_halo else "mask_shift_only"
            ),
            residual_temporal_mode=(
                "explicit_motion_aware"
                if boundary_active
                else (
                    "generic_temporal_policy"
                    if rvm_temporal_smoothing > 0.0
                    else "model_only"
                )
            ),
            rvm_downsample_ratio=resolved_ratio,
            threshold=None,
            mask_blur=rvm_mask_blur,
            edge_refine=rvm_edge_refine,
            edge_refinement_mode=(
                configured.edge_refinement_mode if rvm_edge_refine else "off"
            ),
            edge_refinement_radius_px=rvm_edge_radius,
            mask_shift=mask_shift,
            temporal_smoothing=rvm_temporal_smoothing,
            boundary_stabilization_mode=boundary_mode,
            boundary_stabilization_time_constant_s=(
                configured.boundary_stabilization_time_constant_s
            ),
            boundary_stabilization_max_motion_px_per_s=(
                configured.boundary_stabilization_max_motion_px_per_s
            ),
            use_model_foreground=configured.use_model_foreground,
            light_wrap=light_wrap,
            light_wrap_stabilization_mode=light_wrap_stabilization_mode,
            light_wrap_stabilization_time_constant_s=(
                configured.light_wrap_stabilization_time_constant_s
            ),
        )
        controls = MattePolicyControls(
            rvm_downsample_ratio=ratio_control,
            raw_alpha=_control(
                None,
                "native_soft_alpha",
                MatteControlState.EFFECTIVE,
                "preserve-rvm-pha-without-threshold",
            ),
            threshold=_control(
                configured.threshold,
                None,
                MatteControlState.INAPPLICABLE,
                "rvm-native-alpha-is-never-hard-thresholded",
            ),
            mask_blur=(
                _enabled_control(
                    configured.mask_blur,
                    rvm_mask_blur,
                    rvm_mask_blur > 0,
                )
                if experimental_rvm_generic
                else _control(
                    configured.mask_blur,
                    0,
                    MatteControlState.BYPASSED,
                    "rvm-native-alpha-bypasses-generic-blur",
                )
            ),
            edge_refine=(
                _enabled_control(
                    configured.edge_refine,
                    rvm_edge_refine,
                    rvm_edge_refine,
                )
                if experimental_rvm_generic
                else _control(
                    configured.edge_refine,
                    False,
                    MatteControlState.BYPASSED,
                    "rvm-native-alpha-bypasses-generic-edge-refinement",
                )
            ),
            mask_shift=mask_shift_control,
            temporal_smoothing=(
                temporal_control
                if experimental_rvm_generic
                else _control(
                    configured.temporal_smoothing,
                    0.0,
                    MatteControlState.BYPASSED,
                    "rvm-recurrence-bypasses-generic-ema",
                )
            ),
            boundary_stabilization=boundary_control,
            use_model_foreground=_enabled_control(
                configured.use_model_foreground,
                configured.use_model_foreground,
                configured.use_model_foreground,
            ),
            light_wrap=light_wrap_control,
            light_wrap_stabilization=light_wrap_stabilization_control,
            opaque_core_halo=_control(
                None,
                (
                    "model-alpha-no-calibration;generic-postprocess"
                    if rvm_generic_halo
                    else "model-alpha-no-calibration;mask-shift-only"
                ),
                MatteControlState.EFFECTIVE,
                "opaque-core-calibration-requires-separate-evidence",
            ),
        )
        return MattePolicySnapshot(
            selected_backend_kind=backend_kind,
            backend_kind=effective_kind,
            passthrough=False,
            experimental_rvm_generic=experimental_rvm_generic,
            configured=configured,
            effective=effective,
            controls=controls,
        )

    edge_active = configured.edge_refine
    edge_radius = (
        spatial_edge_refinement_radius(
            segmentation.spatial_edge_refinement,
            canvas_shape,
        )
        if edge_active and canvas_shape is not None
        else 0
    )
    edge_control = _enabled_control(edge_active, edge_active, edge_active)
    blur_control = _enabled_control(
        configured.mask_blur,
        configured.mask_blur,
        configured.mask_blur > 0,
    )

    if effective_kind is MatteBackendKind.CONFIDENCE_MASK_VIDEO:
        raw_alpha_mode: RawAlphaMode = "confidence_soft_mask"
        opaque_core_mode: OpaqueCoreMode = "confidence_mask_no_calibration"
        threshold = None
        threshold_control = _control(
            configured.threshold,
            None,
            MatteControlState.INAPPLICABLE,
            "mediapipe-confidence-mask-does-not-use-threshold",
        )
    else:
        raw_alpha_mode = "thresholded_binary_mask"
        opaque_core_mode = "heuristic_threshold"
        # Preserve the schema-v1 HeuristicSegmenter cutoff exactly.
        threshold = configured.threshold * 0.8
        threshold_control = _control(
            configured.threshold,
            threshold,
            MatteControlState.EFFECTIVE,
            "heuristic-score-cutoff",
        )

    effective = EffectiveMattePolicy(
        raw_alpha_mode=raw_alpha_mode,
        opaque_core_mode=opaque_core_mode,
        halo_mode="generic_postprocess",
        residual_temporal_mode=(
            "generic_temporal_policy"
            if boundary_active or temporal_smoothing > 0.0
            else "none"
        ),
        rvm_downsample_ratio=None,
        threshold=threshold,
        mask_blur=configured.mask_blur,
        edge_refine=edge_active,
        edge_refinement_mode=(
            configured.edge_refinement_mode if edge_active else "off"
        ),
        edge_refinement_radius_px=edge_radius,
        mask_shift=mask_shift,
        temporal_smoothing=temporal_smoothing,
        boundary_stabilization_mode=boundary_mode,
        boundary_stabilization_time_constant_s=(
            configured.boundary_stabilization_time_constant_s
        ),
        boundary_stabilization_max_motion_px_per_s=(
            configured.boundary_stabilization_max_motion_px_per_s
        ),
        use_model_foreground=False,
        light_wrap=light_wrap,
        light_wrap_stabilization_mode=light_wrap_stabilization_mode,
        light_wrap_stabilization_time_constant_s=(
            configured.light_wrap_stabilization_time_constant_s
        ),
    )
    controls = MattePolicyControls(
        rvm_downsample_ratio=_control(
            configured.rvm_downsample_ratio,
            None,
            MatteControlState.INAPPLICABLE,
            "selected-backend-does-not-use-rvm-ratio",
        ),
        raw_alpha=_control(
            None,
            raw_alpha_mode,
            MatteControlState.EFFECTIVE,
            (
                "preserve-mediapipe-confidence-mask"
                if effective_kind is MatteBackendKind.CONFIDENCE_MASK_VIDEO
                else "heuristic-threshold-produces-binary-mask"
            ),
        ),
        threshold=threshold_control,
        mask_blur=blur_control,
        edge_refine=edge_control,
        mask_shift=mask_shift_control,
        temporal_smoothing=temporal_control,
        boundary_stabilization=boundary_control,
        use_model_foreground=_control(
            configured.use_model_foreground,
            False,
            MatteControlState.INAPPLICABLE,
            "selected-backend-does-not-produce-clean-foreground",
        ),
        light_wrap=light_wrap_control,
        light_wrap_stabilization=light_wrap_stabilization_control,
        opaque_core_halo=_control(
            None,
            f"{opaque_core_mode};generic-postprocess",
            MatteControlState.EFFECTIVE,
            "generic-mask-policy",
        ),
    )
    return MattePolicySnapshot(
        selected_backend_kind=backend_kind,
        backend_kind=effective_kind,
        passthrough=False,
        experimental_rvm_generic=False,
        configured=configured,
        effective=effective,
        controls=controls,
    )
