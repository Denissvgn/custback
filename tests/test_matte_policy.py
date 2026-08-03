"""Backend-specific matte-policy contract tests.

These tests deliberately exercise the typed policy resolver rather than
reconstructing effective controls from a refiner or from public status fields.
That keeps one owner for configured-versus-effective and applicability
semantics while MATTE-4.1 remains responsible for the eventual public schema.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import custback.pipeline as pipeline_mod
from custback.capture import CapturedFrame
from custback.compositor import composite
from custback.config import (
    AppConfig,
    BoundaryStabilizationConfig,
    CompositingConfig,
    LightWrapStabilizationConfig,
    RuntimeConfig,
    SegmentationConfig,
    SpatialEdgeRefinementConfig,
)
from custback.hub import FrameHub
from custback.matte_policy import (
    ConfiguredMattePolicy,
    EffectiveMattePolicy,
    MatteBackendKind,
    MatteControl,
    MatteControlState,
    MattePolicyControls,
    MattePolicySnapshot,
    resolve_matte_policy,
)
from custback.pipeline import ActivationError, Pipeline, _PatchRequest, _Resources
from custback.segmentation import (
    HeuristicSegmenter,
    MediaPipeSegmenter,
    NullSegmenter,
    RVMSegmenter,
    Segmenter,
    refiner_for,
)


_CONTROL_ALIASES = {
    "rvm_ratio": ("rvm_ratio", "rvm_downsample_ratio", "rvm_downsample"),
    "raw_alpha": ("raw_alpha",),
    "threshold": ("threshold",),
    "blur": ("blur", "mask_blur"),
    "edge_refine": ("edge_refine",),
    "edge_refinement_mode": ("edge_refinement_mode",),
    "edge_refinement_radius": (
        "edge_refinement_radius",
        "edge_refinement_radius_px",
    ),
    "mask_shift": ("mask_shift",),
    "generic_smoothing": ("generic_smoothing", "temporal_smoothing"),
    "residual_stabilization": (
        "residual_stabilization",
        "boundary_stabilization",
    ),
    "model_foreground": ("model_foreground", "use_model_foreground"),
    "light_wrap": ("light_wrap",),
    "light_wrap_stabilization": ("light_wrap_stabilization",),
    "opaque_core_halo": ("opaque_core_halo",),
}


def _control(snapshot: MattePolicySnapshot, semantic_name: str) -> MatteControl:
    for field_name in _CONTROL_ALIASES[semantic_name]:
        value = getattr(snapshot.controls, field_name, None)
        if isinstance(value, MatteControl):
            return value
    raise AssertionError(
        f"policy controls do not expose the {semantic_name!r} semantic"
    )


def _configured_policy() -> tuple[SegmentationConfig, CompositingConfig]:
    return (
        SegmentationConfig(
            backend="auto",
            rvm_downsample=0.55,
            threshold=0.83,
            mask_blur=9,
            edge_refine=True,
            mask_shift=-2,
            temporal_smoothing=0.61,
            boundary_stabilization=BoundaryStabilizationConfig(
                mode="off",
                time_constant_s=0.07,
                max_motion_px_per_s=480.0,
            ),
            spatial_edge_refinement=SpatialEdgeRefinementConfig(
                mode="stable_guided",
                reference_short_edge_px=720,
                radius_at_reference_px=6,
                min_radius_px=1,
                max_radius_px=10,
            ),
        ),
        CompositingConfig(
            use_model_foreground=True,
            light_wrap=0.37,
            light_wrap_stabilization=LightWrapStabilizationConfig(
                mode="temporal_bounded",
                time_constant_s=0.18,
            ),
        ),
    )


def _resolve(
    backend_kind: MatteBackendKind,
    *,
    segmentation: SegmentationConfig | None = None,
    compositing: CompositingConfig | None = None,
    resolved_rvm_ratio: float | None = 0.42,
) -> MattePolicySnapshot:
    default_segmentation, default_compositing = _configured_policy()
    return resolve_matte_policy(
        segmentation or default_segmentation,
        compositing or default_compositing,
        backend_kind,
        resolved_rvm_ratio=resolved_rvm_ratio,
        canvas_shape=(720, 1280),
    )


class _PolicySegmenter(Segmenter):
    """Small explicit-capability backend for policy/activation integration."""

    def __init__(
        self,
        backend_kind: MatteBackendKind,
        *,
        ratio: float | None = None,
        fail: bool = False,
    ) -> None:
        super().__init__()
        self.matte_backend_kind = backend_kind
        self.produces_matte = backend_kind is MatteBackendKind.TRUE_ALPHA_RECURRENT
        self.device = (
            "cuda" if backend_kind is MatteBackendKind.TRUE_ALPHA_RECURRENT else "cpu"
        )
        self._ratio = ratio
        self._fail = fail
        self.last_downsample_ratio: float | None = None
        self.closed = False

    def segment(
        self,
        frame_bgr: np.ndarray,
        *,
        context=None,
    ) -> np.ndarray:
        self._accept_frame_context(context, frame_bgr.shape[:2])
        if self._fail:
            raise RuntimeError("candidate policy trial failed")
        if self.matte_backend_kind is MatteBackendKind.TRUE_ALPHA_RECURRENT:
            self.last_downsample_ratio = self._ratio
            self.last_foreground = frame_bgr.copy()
        return np.full(frame_bgr.shape[:2], 0.5, dtype=np.float32)

    def reset_temporal_state(self, reason, timestamp_ns) -> None:
        self.last_downsample_ratio = None
        super().reset_temporal_state(reason, timestamp_ns)

    def close(self) -> None:
        self.closed = True
        super().close()


def _activation_config(backend: str) -> AppConfig:
    return AppConfig.from_dict(
        {
            "camera": {
                "synthetic": True,
                "width": 32,
                "height": 24,
                "fps": 30,
            },
            "background": {"mode": "color", "color": [20, 30, 40]},
            "segmentation": {
                "backend": backend,
                "threshold": 0.73,
                "mask_blur": 9,
                "edge_refine": True,
                "mask_shift": -1,
                "temporal_smoothing": 0.6,
            },
            "compositing": {
                "use_model_foreground": True,
                "light_wrap": 0.3,
            },
            "output": {"backend": "null"},
            "api": {"enabled": False},
        }
    )


def _captured_frame(sequence: int) -> CapturedFrame:
    frame = np.full((24, 32, 3), 90, dtype=np.uint8)
    return CapturedFrame(
        pixels=frame,
        sequence=sequence,
        captured_at_ns=1_000_000_000 + sequence * 33_333_333,
        generation=1,
        geometry_generation=1,
        content_rect=(0, 0, 32, 24),
    )


@pytest.mark.parametrize(
    ("backend_kind", "expected_states"),
    [
        (
            MatteBackendKind.TRUE_ALPHA_RECURRENT,
            {
                "rvm_ratio": MatteControlState.EFFECTIVE,
                "raw_alpha": MatteControlState.EFFECTIVE,
                "threshold": MatteControlState.INAPPLICABLE,
                "blur": MatteControlState.BYPASSED,
                "edge_refine": MatteControlState.BYPASSED,
                "mask_shift": MatteControlState.EFFECTIVE,
                "generic_smoothing": MatteControlState.BYPASSED,
                "residual_stabilization": MatteControlState.BYPASSED,
                "model_foreground": MatteControlState.EFFECTIVE,
                "light_wrap": MatteControlState.EFFECTIVE,
                "light_wrap_stabilization": MatteControlState.EFFECTIVE,
                "opaque_core_halo": MatteControlState.EFFECTIVE,
            },
        ),
        (
            MatteBackendKind.CONFIDENCE_MASK_VIDEO,
            {
                "rvm_ratio": MatteControlState.INAPPLICABLE,
                "raw_alpha": MatteControlState.EFFECTIVE,
                "threshold": MatteControlState.INAPPLICABLE,
                "blur": MatteControlState.EFFECTIVE,
                "edge_refine": MatteControlState.EFFECTIVE,
                "mask_shift": MatteControlState.EFFECTIVE,
                "generic_smoothing": MatteControlState.EFFECTIVE,
                "residual_stabilization": MatteControlState.BYPASSED,
                "model_foreground": MatteControlState.INAPPLICABLE,
                "light_wrap": MatteControlState.EFFECTIVE,
                "light_wrap_stabilization": MatteControlState.EFFECTIVE,
                "opaque_core_halo": MatteControlState.EFFECTIVE,
            },
        ),
        (
            MatteBackendKind.BINARY_COARSE,
            {
                "rvm_ratio": MatteControlState.INAPPLICABLE,
                "raw_alpha": MatteControlState.EFFECTIVE,
                "threshold": MatteControlState.EFFECTIVE,
                "blur": MatteControlState.EFFECTIVE,
                "edge_refine": MatteControlState.EFFECTIVE,
                "mask_shift": MatteControlState.EFFECTIVE,
                "generic_smoothing": MatteControlState.EFFECTIVE,
                "residual_stabilization": MatteControlState.BYPASSED,
                "model_foreground": MatteControlState.INAPPLICABLE,
                "light_wrap": MatteControlState.EFFECTIVE,
                "light_wrap_stabilization": MatteControlState.EFFECTIVE,
                "opaque_core_halo": MatteControlState.EFFECTIVE,
            },
        ),
        (
            MatteBackendKind.NULL_PASSTHROUGH,
            {
                "rvm_ratio": MatteControlState.INAPPLICABLE,
                "raw_alpha": MatteControlState.INAPPLICABLE,
                "threshold": MatteControlState.INAPPLICABLE,
                "blur": MatteControlState.INAPPLICABLE,
                "edge_refine": MatteControlState.INAPPLICABLE,
                "mask_shift": MatteControlState.INAPPLICABLE,
                "generic_smoothing": MatteControlState.INAPPLICABLE,
                "residual_stabilization": MatteControlState.INAPPLICABLE,
                "model_foreground": MatteControlState.INAPPLICABLE,
                "light_wrap": MatteControlState.INAPPLICABLE,
                "light_wrap_stabilization": MatteControlState.INAPPLICABLE,
                "opaque_core_halo": MatteControlState.INAPPLICABLE,
            },
        ),
    ],
)
def test_effective_policy_matrix_is_explicit_for_every_backend(
    backend_kind: MatteBackendKind,
    expected_states: dict[str, MatteControlState],
) -> None:
    snapshot = _resolve(backend_kind)

    assert isinstance(snapshot, MattePolicySnapshot)
    assert isinstance(snapshot.configured, ConfiguredMattePolicy)
    assert isinstance(snapshot.effective, EffectiveMattePolicy)
    assert isinstance(snapshot.controls, MattePolicyControls)
    assert snapshot.backend_kind is backend_kind
    assert snapshot.configured is not snapshot.effective

    for semantic_name, expected_state in expected_states.items():
        control = _control(snapshot, semantic_name)
        assert control.state is expected_state, semantic_name
        if expected_state is not MatteControlState.EFFECTIVE:
            assert control.reason, semantic_name

    # The operator's persisted intent remains observable even where this
    # backend neutralizes or rejects it.
    assert _control(snapshot, "threshold").configured == 0.83
    assert _control(snapshot, "blur").configured == 9
    assert _control(snapshot, "edge_refine").configured is True
    assert _control(snapshot, "mask_shift").configured == -2
    assert _control(snapshot, "generic_smoothing").configured == 0.61
    assert _control(snapshot, "model_foreground").configured is True
    assert _control(snapshot, "light_wrap").configured == 0.37
    assert (
        _control(snapshot, "light_wrap_stabilization").configured == "temporal_bounded"
    )


def test_rvm_policy_neutralizes_generic_controls_without_erasing_intent() -> None:
    snapshot = _resolve(MatteBackendKind.TRUE_ALPHA_RECURRENT)

    assert _control(snapshot, "threshold").effective is None
    assert _control(snapshot, "blur").effective == 0
    assert _control(snapshot, "edge_refine").effective is False
    assert snapshot.configured.edge_refinement_mode == "stable_guided"
    assert snapshot.effective.edge_refinement_mode == "off"
    assert snapshot.effective.edge_refinement_radius_px == 0
    assert _control(snapshot, "generic_smoothing").effective == 0.0
    assert _control(snapshot, "mask_shift").effective == -2
    assert _control(snapshot, "model_foreground").effective is True
    assert _control(snapshot, "light_wrap").effective == 0.37
    assert snapshot.effective.light_wrap_stabilization_mode == "temporal_bounded"
    assert snapshot.effective.light_wrap_stabilization_time_constant_s == 0.18


def test_resolver_is_pure_immutable_and_rejects_requested_backend_labels() -> None:
    segmentation, compositing = _configured_policy()
    configured_before = segmentation.model_dump(mode="python")
    compositing_before = compositing.model_dump(mode="python")

    snapshot = resolve_matte_policy(
        segmentation,
        compositing,
        MatteBackendKind.CONFIDENCE_MASK_VIDEO,
        canvas_shape=(720, 1280),
    )

    assert segmentation.model_dump(mode="python") == configured_before
    assert compositing.model_dump(mode="python") == compositing_before
    with pytest.raises(FrozenInstanceError):
        snapshot.passthrough = True  # type: ignore[misc]
    with pytest.raises(TypeError, match="actual MatteBackendKind"):
        resolve_matte_policy(
            segmentation,
            compositing,
            cast(Any, "auto"),
        )


def test_light_wrap_stabilization_is_bypassed_without_dynamic_backdrop_time() -> None:
    segmentation, compositing = _configured_policy()

    snapshot = resolve_matte_policy(
        segmentation,
        compositing,
        MatteBackendKind.CONFIDENCE_MASK_VIDEO,
        canvas_shape=(720, 1280),
        light_wrap_stabilization_eligible=False,
    )

    control = _control(snapshot, "light_wrap_stabilization")
    assert snapshot.effective.light_wrap == 0.37
    assert snapshot.effective.light_wrap_stabilization_mode == "off"
    assert control.configured == "temporal_bounded"
    assert control.effective == "off"
    assert control.state is MatteControlState.BYPASSED
    assert control.reason == "backdrop-has-no-dynamic-timeline"


@pytest.mark.parametrize(
    ("ratio", "shape"),
    [
        (0.0, (720, 1280)),
        (1.01, (720, 1280)),
        (float("nan"), (720, 1280)),
        (0.4, (0, 1280)),
        (0.4, (720, True)),
    ],
)
def test_resolver_rejects_invalid_runtime_facts(
    ratio: float,
    shape: tuple[Any, Any],
) -> None:
    segmentation, compositing = _configured_policy()
    with pytest.raises(ValueError):
        resolve_matte_policy(
            segmentation,
            compositing,
            MatteBackendKind.TRUE_ALPHA_RECURRENT,
            resolved_rvm_ratio=ratio,
            canvas_shape=cast(tuple[int, int], shape),
        )


def test_resolver_rejects_non_boolean_light_wrap_eligibility() -> None:
    segmentation, compositing = _configured_policy()
    with pytest.raises(TypeError, match="eligibility flag"):
        resolve_matte_policy(
            segmentation,
            compositing,
            MatteBackendKind.CONFIDENCE_MASK_VIDEO,
            light_wrap_stabilization_eligible=cast(Any, 1),
        )


def test_experimental_generic_policy_is_restricted_to_active_rvm() -> None:
    segmentation, compositing = _configured_policy()
    with pytest.raises(ValueError, match="active RVM"):
        resolve_matte_policy(
            segmentation,
            compositing,
            MatteBackendKind.CONFIDENCE_MASK_VIDEO,
            experimental_rvm_generic=True,
        )
    with pytest.raises(ValueError, match="active RVM"):
        resolve_matte_policy(
            segmentation,
            compositing,
            MatteBackendKind.TRUE_ALPHA_RECURRENT,
            passthrough=True,
            experimental_rvm_generic=True,
        )


def test_heuristic_threshold_reports_the_legacy_score_cutoff_it_exercises() -> None:
    snapshot = _resolve(MatteBackendKind.BINARY_COARSE)
    threshold = _control(snapshot, "threshold")

    assert threshold.configured == 0.83
    assert threshold.effective == pytest.approx(0.83 * 0.8)
    assert threshold.state is MatteControlState.EFFECTIVE


@pytest.mark.parametrize(
    "backend_kind",
    [
        MatteBackendKind.TRUE_ALPHA_RECURRENT,
        MatteBackendKind.CONFIDENCE_MASK_VIDEO,
        MatteBackendKind.BINARY_COARSE,
    ],
)
def test_motion_aware_policy_is_the_only_effective_temporal_owner(
    backend_kind: MatteBackendKind,
) -> None:
    segmentation, compositing = _configured_policy()
    segmentation_payload = segmentation.model_dump(mode="python")
    segmentation_payload["boundary_stabilization"] = {
        "mode": "motion_aware",
        "time_constant_s": 0.05,
        "max_motion_px_per_s": 360.0,
    }
    segmentation = SegmentationConfig.model_validate(segmentation_payload)

    snapshot = _resolve(
        backend_kind,
        segmentation=segmentation,
        compositing=compositing,
    )

    generic = _control(snapshot, "generic_smoothing")
    residual = _control(snapshot, "residual_stabilization")
    assert generic.state is MatteControlState.BYPASSED
    assert generic.configured == 0.61
    assert generic.effective == 0.0
    assert residual.state is MatteControlState.EFFECTIVE
    assert residual.effective == "motion_aware"


@pytest.mark.parametrize(
    ("backend_kind", "supported_controls"),
    [
        (
            MatteBackendKind.TRUE_ALPHA_RECURRENT,
            {
                "mask_shift",
                "residual_stabilization",
                "model_foreground",
                "light_wrap",
                "light_wrap_stabilization",
            },
        ),
        (
            MatteBackendKind.CONFIDENCE_MASK_VIDEO,
            {
                "blur",
                "edge_refine",
                "mask_shift",
                "generic_smoothing",
                "residual_stabilization",
                "light_wrap",
                "light_wrap_stabilization",
            },
        ),
        (
            MatteBackendKind.BINARY_COARSE,
            {
                "blur",
                "edge_refine",
                "mask_shift",
                "generic_smoothing",
                "residual_stabilization",
                "light_wrap",
                "light_wrap_stabilization",
            },
        ),
    ],
)
def test_supported_controls_configured_off_are_bypassed_not_inapplicable(
    backend_kind: MatteBackendKind,
    supported_controls: set[str],
) -> None:
    segmentation = SegmentationConfig(
        backend="auto",
        rvm_downsample=0.0,
        threshold=0.0,
        mask_blur=0,
        edge_refine=False,
        mask_shift=0,
        temporal_smoothing=0.0,
        boundary_stabilization=BoundaryStabilizationConfig(mode="off"),
    )
    compositing = CompositingConfig(
        use_model_foreground=False,
        light_wrap=0.0,
    )

    snapshot = _resolve(
        backend_kind,
        segmentation=segmentation,
        compositing=compositing,
        resolved_rvm_ratio=None,
    )

    for semantic_name in supported_controls:
        control = _control(snapshot, semantic_name)
        assert control.state is MatteControlState.BYPASSED, semantic_name
        assert control.reason, semantic_name


@pytest.mark.parametrize(
    "selected_kind",
    [
        MatteBackendKind.TRUE_ALPHA_RECURRENT,
        MatteBackendKind.CONFIDENCE_MASK_VIDEO,
        MatteBackendKind.BINARY_COARSE,
    ],
)
def test_passthrough_neutralizes_controls_but_retains_selected_backend(
    selected_kind: MatteBackendKind,
) -> None:
    segmentation, compositing = _configured_policy()
    snapshot = resolve_matte_policy(
        segmentation,
        compositing,
        selected_kind,
        resolved_rvm_ratio=0.42,
        passthrough=True,
        canvas_shape=(720, 1280),
    )

    assert snapshot.selected_backend_kind is selected_kind
    assert snapshot.backend_kind is MatteBackendKind.NULL_PASSTHROUGH
    assert snapshot.passthrough is True
    for control_field in fields(snapshot.controls):
        control = getattr(snapshot.controls, control_field.name)
        assert control.state is MatteControlState.INAPPLICABLE, control_field.name
        assert control.reason, control_field.name
    assert snapshot.effective.mask_blur == 0
    assert snapshot.effective.edge_refine is False
    assert snapshot.effective.mask_shift == 0
    assert snapshot.effective.temporal_smoothing == 0.0
    assert snapshot.effective.use_model_foreground is False
    assert snapshot.effective.light_wrap == 0.0


def test_rvm_ratio_distinguishes_auto_unresolved_resolved_and_explicit() -> None:
    auto = SegmentationConfig(backend="rvm", rvm_downsample=0.0)
    explicit = SegmentationConfig(backend="rvm", rvm_downsample=0.57)
    compositing = CompositingConfig()

    unresolved = _resolve(
        MatteBackendKind.TRUE_ALPHA_RECURRENT,
        segmentation=auto,
        compositing=compositing,
        resolved_rvm_ratio=None,
    )
    resolved = _resolve(
        MatteBackendKind.TRUE_ALPHA_RECURRENT,
        segmentation=auto,
        compositing=compositing,
        resolved_rvm_ratio=0.4,
    )
    configured = _resolve(
        MatteBackendKind.TRUE_ALPHA_RECURRENT,
        segmentation=explicit,
        compositing=compositing,
        resolved_rvm_ratio=None,
    )
    wrong_backend = _resolve(
        MatteBackendKind.CONFIDENCE_MASK_VIDEO,
        segmentation=explicit,
        compositing=compositing,
        resolved_rvm_ratio=0.57,
    )

    assert _control(unresolved, "rvm_ratio").configured == 0.0
    assert _control(unresolved, "rvm_ratio").effective is None
    assert _control(unresolved, "rvm_ratio").state is MatteControlState.EFFECTIVE
    assert _control(resolved, "rvm_ratio").effective == pytest.approx(0.4)
    assert _control(resolved, "rvm_ratio").state is MatteControlState.EFFECTIVE
    assert _control(configured, "rvm_ratio").effective is None
    assert _control(configured, "rvm_ratio").state is MatteControlState.EFFECTIVE
    assert _control(wrong_backend, "rvm_ratio").effective is None
    assert _control(wrong_backend, "rvm_ratio").state is MatteControlState.INAPPLICABLE


def test_threshold_changes_remain_visible_but_cannot_calibrate_rvm_alpha() -> None:
    segmentation, compositing = _configured_policy()
    low = segmentation.model_copy(update={"threshold": 0.1})
    high = segmentation.model_copy(update={"threshold": 0.9})

    low_policy = _resolve(
        MatteBackendKind.TRUE_ALPHA_RECURRENT,
        segmentation=low,
        compositing=compositing,
    )
    high_policy = _resolve(
        MatteBackendKind.TRUE_ALPHA_RECURRENT,
        segmentation=high,
        compositing=compositing,
    )

    assert _control(low_policy, "threshold").configured == 0.1
    assert _control(high_policy, "threshold").configured == 0.9
    assert _control(low_policy, "threshold").effective is None
    assert _control(high_policy, "threshold").effective is None
    assert _control(low_policy, "threshold").state is MatteControlState.INAPPLICABLE
    assert _control(high_policy, "threshold").state is MatteControlState.INAPPLICABLE


def test_rvm_refiner_preserves_soft_alpha_with_generic_controls_enabled() -> None:
    segmentation, _compositing = _configured_policy()
    segmentation = segmentation.model_copy(
        update={
            "threshold": 0.99,
            "mask_blur": 151,
            "edge_refine": True,
            "temporal_smoothing": 0.95,
            "mask_shift": 0,
        }
    )

    # The policy resolver only requires the backend's declared kind. Avoid
    # constructing an ONNX session in this pure contract test.
    segmenter = object.__new__(RVMSegmenter)
    refiner = refiner_for(segmentation, segmenter)
    alpha = np.linspace(0.01, 0.99, 63, dtype=np.float32).reshape(7, 9)
    guide = np.tile(np.arange(9, dtype=np.uint8), (7, 1))
    guide = np.repeat(guide[:, :, None], 3, axis=2)

    refined = refiner.refine(alpha, guide)

    np.testing.assert_array_equal(refined, alpha)
    assert np.count_nonzero((refined > 0.0) & (refined < 1.0)) == refined.size
    assert refiner.cfg.mask_blur == 0
    assert refiner.cfg.edge_refine is False
    assert refiner.cfg.temporal_smoothing == 0.0


def test_builtin_segmenters_declare_policy_kind_without_class_name_inference() -> None:
    assert RVMSegmenter.matte_backend_kind is MatteBackendKind.TRUE_ALPHA_RECURRENT
    assert (
        MediaPipeSegmenter.matte_backend_kind is MatteBackendKind.CONFIDENCE_MASK_VIDEO
    )
    assert HeuristicSegmenter.matte_backend_kind is MatteBackendKind.BINARY_COARSE
    assert NullSegmenter.matte_backend_kind is MatteBackendKind.NULL_PASSTHROUGH


def test_config_api_schema_documents_backend_applicability() -> None:
    segmentation_properties = cast(
        dict[str, dict[str, Any]],
        SegmentationConfig.model_json_schema()["properties"],
    )
    compositing_properties = cast(
        dict[str, dict[str, Any]],
        CompositingConfig.model_json_schema()["properties"],
    )

    assert "Heuristic" in segmentation_properties["threshold"]["description"]
    assert "RVM" in segmentation_properties["mask_blur"]["description"]
    assert "RVM" in segmentation_properties["edge_refine"]["description"]
    assert "RVM" in segmentation_properties["mask_shift"]["description"]
    assert "MediaPipe" in segmentation_properties["temporal_smoothing"]["description"]
    assert "RVM-only" in compositing_properties["use_model_foreground"]["description"]
    assert "null/passthrough" in compositing_properties["light_wrap"]["description"]


@pytest.mark.parametrize("backend_kind", list(MatteBackendKind))
def test_policy_snapshot_has_a_json_safe_replay_representation(
    backend_kind: MatteBackendKind,
) -> None:
    snapshot = _resolve(backend_kind)

    payload = snapshot.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["backend_kind"] == backend_kind.value
    assert set(payload) >= {"backend_kind", "configured", "effective", "controls"}

    serialized_controls = payload["controls"]
    assert isinstance(serialized_controls, dict)
    for control_field in fields(snapshot.controls):
        control_payload = serialized_controls[control_field.name]
        assert set(control_payload) == {
            "configured",
            "effective",
            "state",
            "reason",
        }
        assert control_payload["state"] in {
            "effective",
            "bypassed",
            "inapplicable",
        }


@pytest.mark.parametrize(
    ("backend_kind", "expected_kind"),
    [
        (
            MatteBackendKind.TRUE_ALPHA_RECURRENT,
            MatteBackendKind.TRUE_ALPHA_RECURRENT,
        ),
        (
            MatteBackendKind.CONFIDENCE_MASK_VIDEO,
            MatteBackendKind.CONFIDENCE_MASK_VIDEO,
        ),
        (MatteBackendKind.BINARY_COARSE, MatteBackendKind.BINARY_COARSE),
        (MatteBackendKind.NULL_PASSTHROUGH, MatteBackendKind.NULL_PASSTHROUGH),
    ],
)
def test_pipeline_evidence_projection_consumes_the_typed_snapshot(
    backend_kind: MatteBackendKind,
    expected_kind: MatteBackendKind,
) -> None:
    cfg = _activation_config("auto")
    segmenter = _PolicySegmenter(backend_kind, ratio=0.42)
    if backend_kind is MatteBackendKind.TRUE_ALPHA_RECURRENT:
        segmenter.last_downsample_ratio = 0.42
    resources = cast(
        _Resources,
        SimpleNamespace(
            cfg=cfg,
            segmenter=segmenter,
            canvas_size=(32, 24),
        ),
    )

    projected = Pipeline._effective_matte_controls(resources)
    policy = cast(dict[str, Any], projected["matte_policy"])
    effective = cast(dict[str, Any], policy["effective"])
    refiner = cast(dict[str, Any], projected["refiner"])

    assert policy["selected_backend_kind"] == backend_kind.value
    assert policy["backend_kind"] == expected_kind.value
    assert projected["rvm_downsample_ratio"] == effective["rvm_downsample_ratio"]
    assert projected["mask_shift"] == effective["mask_shift"]
    assert projected["use_model_foreground"] == effective["use_model_foreground"]
    assert projected["light_wrap"] == effective["light_wrap"]
    assert refiner["mask_blur"] == effective["mask_blur"]
    assert refiner["edge_refine"] == effective["edge_refine"]


def test_null_runtime_consumes_neutral_compositor_policy() -> None:
    cfg = _activation_config("none")
    frame = np.full((24, 32, 3), 100, dtype=np.uint8)
    backdrop_frame = np.full((24, 32, 3), 20, dtype=np.uint8)
    mask = np.full((24, 32), 0.5, dtype=np.float32)
    segmenter = _PolicySegmenter(MatteBackendKind.NULL_PASSTHROUGH)
    segmenter.last_foreground = np.full((24, 32, 3), 220, dtype=np.uint8)
    segmenter.segment = lambda _frame, **_kwargs: mask.copy()  # type: ignore[method-assign]
    refiner = refiner_for(cfg.segmentation, segmenter, cfg.compositing)

    class Backdrop:
        def frame(self, width: int, height: int) -> np.ndarray:
            assert (width, height) == (32, 24)
            return backdrop_frame.copy()

    resources = cast(
        _Resources,
        SimpleNamespace(
            cfg=cfg,
            segmenter=segmenter,
            refiner=refiner,
            backdrop=Backdrop(),
            canvas_size=(32, 24),
        ),
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())

    rendered, reason = pipeline._local_composite(
        resources,
        frame,
        privacy_safe=False,
    )
    expected = composite(
        frame,
        backdrop_frame,
        mask,
        light_wrap=0.0,
        edge_foreground=None,
        blend_space=cfg.compositing.blend_space,
    )

    assert reason == ""
    np.testing.assert_array_equal(rendered, expected)


def test_backend_switch_installs_fresh_policy_and_failed_switch_rolls_back(
    monkeypatch,
) -> None:
    current = _activation_config("heuristic")
    live_segmenter = _PolicySegmenter(MatteBackendKind.BINARY_COARSE)
    live_refiner = refiner_for(
        current.segmentation,
        live_segmenter,
        current.compositing,
    )
    resources = _Resources(
        current,
        0,
        None,
        live_segmenter,
        live_refiner,
        None,
        None,
    )
    pipeline = Pipeline(RuntimeConfig(current), FrameHub())
    created: list[_PolicySegmenter] = []

    def create_segmenter(cfg, **_kwargs):
        if cfg.backend == "rvm":
            segmenter = _PolicySegmenter(
                MatteBackendKind.TRUE_ALPHA_RECURRENT,
                ratio=0.5,
            )
        else:
            segmenter = _PolicySegmenter(
                MatteBackendKind.CONFIDENCE_MASK_VIDEO,
                fail=True,
            )
        created.append(segmenter)
        return segmenter

    monkeypatch.setattr(pipeline_mod, "create_segmenter", create_segmenter)
    first_candidate = current.patched(
        {
            "segmentation": {
                "backend": "rvm",
                "rvm_downsample": 0.5,
            }
        }
    )
    first_activation = pipeline._prepare_activation_off_lane(
        current,
        first_candidate,
    )
    fresh_refiner = first_activation.refiner
    first_request = _PatchRequest(
        first_candidate,
        0,
        prepared_activation=first_activation,
    )

    try:
        pipeline._handle_patch_request(
            resources,
            first_request,
            _captured_frame(1),
        )

        assert first_request.error is None
        assert first_request.result is not None
        assert resources.segmenter is created[0]
        assert resources.refiner is fresh_refiner
        assert resources.refiner is not live_refiner
        assert resources.segmentation_generation == 1
        assert resources.refiner._prev is None
        committed_policy = pipeline._matte_policy_snapshot(resources)
        assert (
            committed_policy.selected_backend_kind
            is MatteBackendKind.TRUE_ALPHA_RECURRENT
        )
        assert (
            committed_policy.controls.threshold.state is MatteControlState.INAPPLICABLE
        )
        assert committed_policy.controls.mask_blur.state is MatteControlState.BYPASSED

        pipeline._segment_resource_masks(
            resources,
            _captured_frame(1),
            privacy_safe=False,
        )
        committed_policy = pipeline._matte_policy_snapshot(resources)
        assert committed_policy.effective.rvm_downsample_ratio == 0.5
        assert resources.refiner._prev is not None
        live_alpha_before = resources.refiner._prev.copy()
        live_segmenter_before = resources.segmenter
        live_refiner_before = resources.refiner
        live_generation_before = resources.segmentation_generation

        rejected_candidate = first_candidate.patched(
            {"segmentation": {"backend": "mediapipe"}}
        )
        rejected_activation = pipeline._prepare_activation_off_lane(
            first_candidate,
            rejected_candidate,
        )
        rejected_request = _PatchRequest(
            rejected_candidate,
            1,
            prepared_activation=rejected_activation,
        )
        pipeline._handle_patch_request(
            resources,
            rejected_request,
            _captured_frame(2),
        )

        assert isinstance(rejected_request.error, ActivationError)
        assert rejected_request.result is None
        assert resources.segmenter is live_segmenter_before
        assert resources.refiner is live_refiner_before
        assert resources.segmentation_generation == live_generation_before
        assert resources.cfg is first_candidate
        assert resources.version == 1
        assert pipeline.runtime.version == 1
        assert pipeline._matte_policy_snapshot(resources) == committed_policy
        np.testing.assert_array_equal(resources.refiner._prev, live_alpha_before)
    finally:
        resources.close()
        pipeline.stop(timeout=1.0)
