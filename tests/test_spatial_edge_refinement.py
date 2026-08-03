"""Deterministic contracts for the opt-in stable guided edge refiner.

All fixtures are generated arrays.  Tests avoid depending on the exact marker
placement of the legacy watershed path and use relational, scale-normalized,
or topology assertions for the new candidate.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pytest

import custback.segmentation as segmentation_mod
from custback.config import (
    SegmentationConfig,
    SpatialEdgeRefinementConfig,
    spatial_edge_refinement_radius,
)
from custback.segmentation import MaskRefiner, _stable_guided_edge_refine

cv2 = pytest.importorskip("cv2")


def _policy(**updates: object) -> SpatialEdgeRefinementConfig:
    values: dict[str, object] = {
        "mode": "stable_guided",
        "reference_short_edge_px": 720,
        "radius_at_reference_px": 8,
        "min_radius_px": 2,
        "max_radius_px": 12,
    }
    values.update(updates)
    return SpatialEdgeRefinementConfig.model_validate(values)


def _refiner_config(
    *,
    edge_refine: bool = True,
    policy: SpatialEdgeRefinementConfig | None = None,
) -> SegmentationConfig:
    return SegmentationConfig(
        mask_blur=0,
        edge_refine=edge_refine,
        mask_shift=0,
        temporal_smoothing=0.0,
        spatial_edge_refinement=_policy() if policy is None else policy,
    )


def _hard_step(shape: tuple[int, int], edge_x: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    mask[:, edge_x:] = 1.0
    return mask


def _step_guide(
    shape: tuple[int, int],
    edge_x: int,
    *,
    left: int = 20,
    right: int = 220,
) -> np.ndarray:
    guide = np.full((*shape, 3), left, dtype=np.uint8)
    guide[:, edge_x:] = right
    return guide


def _soft_ramp(
    shape: tuple[int, int],
    edge_x: int,
    width: int,
) -> np.ndarray:
    height, frame_width = shape
    x = np.arange(frame_width, dtype=np.float32)
    profile = np.clip(
        (x - (float(edge_x) - width / 2.0)) / float(width),
        0.0,
        1.0,
    )
    return np.ascontiguousarray(np.broadcast_to(profile, (height, frame_width)))


def _vertical_contour_positions(mask: np.ndarray) -> np.ndarray:
    positions = np.empty(mask.shape[0], dtype=np.float64)
    for row_index, row in enumerate(mask):
        foreground = np.flatnonzero(row >= 0.5)
        assert foreground.size, "fixture must contain a foreground contour"
        right = int(foreground[0])
        left = max(0, right - 1)
        left_alpha = float(row[left])
        right_alpha = float(row[right])
        if right == left or left_alpha == right_alpha:
            positions[row_index] = float(right)
        else:
            positions[row_index] = left + (
                (0.5 - left_alpha) / (right_alpha - left_alpha)
            )
    return positions


def _median_contour_x(mask: np.ndarray) -> float:
    return float(np.median(_vertical_contour_positions(mask)))


def _contour_displacement_p95(masks: Sequence[np.ndarray]) -> float:
    contours = [_vertical_contour_positions(mask) for mask in masks]
    values = np.concatenate(
        [
            np.abs(current - previous)
            for previous, current in zip(contours, contours[1:])
        ]
    )
    return float(np.quantile(values, 0.95))


def _area_drift_p95(masks: Sequence[np.ndarray]) -> float:
    reference = float(masks[0].sum(dtype=np.float64))
    assert reference > 0.0
    values = [
        abs(float(mask.sum(dtype=np.float64)) - reference) / reference for mask in masks
    ]
    return float(np.quantile(values, 0.95))


def _component_count(binary: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents(
        binary.astype(np.uint8),
        connectivity=8,
    )
    return int(count) - 1


def _deterministic_camera_noise(
    shape: tuple[int, int],
    phase: int,
    *,
    amplitude: int = 3,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.int32)
    modulus = 2 * amplitude + 1
    return ((xx * 17 + yy * 11 + phase * 13) % modulus) - amplitude


def _noisy_step_guide(
    shape: tuple[int, int],
    edge_x: int,
    phase: int,
) -> np.ndarray:
    base = _step_guide(shape, edge_x, left=28, right=220).astype(np.int16)
    noise = _deterministic_camera_noise(shape, phase)
    return np.clip(base + noise[..., None], 0, 255).astype(np.uint8)


def _block_noisy_step_guide(
    shape: tuple[int, int],
    edge_x: int,
    phase: int,
) -> np.ndarray:
    base = _step_guide(shape, edge_x, left=32, right=216).astype(np.int16)
    yy, xx = np.indices(shape, dtype=np.int32)
    blocks = np.where(((xx // 8 + yy // 8 + phase) & 1) == 0, -12, 12)
    return np.clip(base + blocks[..., None], 0, 255).astype(np.uint8)


def test_default_policy_retains_legacy_watershed_through_mask_refiner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SegmentationConfig(
        mask_blur=0,
        edge_refine=True,
        temporal_smoothing=0.0,
    )
    assert cfg.spatial_edge_refinement.mode == "legacy_watershed"
    mask = _hard_step((32, 48), 28)
    guide = _step_guide(mask.shape, 24)
    expected = np.full(mask.shape, 0.25, dtype=np.float32)
    calls: list[tuple[int, int]] = []

    def legacy(candidate: np.ndarray, frame: np.ndarray) -> np.ndarray:
        calls.append(frame.shape[:2])
        np.testing.assert_array_equal(candidate, mask)
        return expected

    monkeypatch.setattr(segmentation_mod, "_watershed_edge_snap", legacy)
    monkeypatch.setattr(
        segmentation_mod,
        "_stable_guided_edge_refine",
        lambda *_args, **_kwargs: pytest.fail(
            "the compatibility default must not select stable_guided"
        ),
    )

    output = MaskRefiner(cfg).refine(mask, guide)

    np.testing.assert_array_equal(output, expected)
    assert calls == [mask.shape]


def test_edge_refine_false_bypasses_opt_in_candidate_through_mask_refiner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = _hard_step((32, 48), 28)
    guide = _step_guide(mask.shape, 24)
    monkeypatch.setattr(
        segmentation_mod,
        "_stable_guided_edge_refine",
        lambda *_args, **_kwargs: pytest.fail(
            "edge_refine=false must bypass every spatial candidate"
        ),
    )

    output = MaskRefiner(_refiner_config(edge_refine=False)).refine(mask, guide)

    np.testing.assert_array_equal(output, mask)


@pytest.mark.parametrize(
    ("shape", "expected_radius"),
    [
        pytest.param((360, 640), 4, id="360p"),
        pytest.param((720, 1280), 8, id="720p"),
        pytest.param((1080, 1920), 12, id="1080p"),
    ],
)
def test_radius_scales_from_reference_short_edge(
    shape: tuple[int, int],
    expected_radius: int,
) -> None:
    assert spatial_edge_refinement_radius(_policy(), shape) == expected_radius


@pytest.mark.parametrize(
    ("shape", "expected_radius"),
    [
        pytest.param((90, 160), 2, id="minimum-clamp"),
        pytest.param((2160, 3840), 12, id="maximum-clamp"),
    ],
)
def test_radius_obeys_explicit_minimum_and_maximum(
    shape: tuple[int, int],
    expected_radius: int,
) -> None:
    assert spatial_edge_refinement_radius(_policy(), shape) == expected_radius


@pytest.mark.parametrize(
    ("offset_px", "left", "right"),
    [
        pytest.param(-3, 20, 220, id="mask-left-foreground-bright"),
        pytest.param(3, 20, 220, id="mask-right-foreground-bright"),
        pytest.param(-3, 220, 20, id="mask-left-foreground-dark"),
        pytest.param(3, 220, 20, id="mask-right-foreground-dark"),
    ],
)
def test_displaced_straight_edge_improves_from_both_sides(
    offset_px: int,
    left: int,
    right: int,
) -> None:
    shape = (360, 640)
    true_edge_x = shape[1] // 2
    ideal = _hard_step(shape, true_edge_x)
    coarse = _hard_step(shape, true_edge_x + offset_px)
    guide = _step_guide(shape, true_edge_x, left=left, right=right)

    output = _stable_guided_edge_refine(coarse, guide, _policy())

    error_before = float(np.mean(np.abs(coarse - ideal), dtype=np.float64))
    error_after = float(np.mean(np.abs(output - ideal), dtype=np.float64))
    assert error_after <= error_before * 0.60
    assert abs(_median_contour_x(output) - _median_contour_x(ideal)) <= 1.5


def test_displaced_curved_edge_improves_without_changing_topology() -> None:
    shape = (360, 640)
    yy, xx = np.indices(shape)
    center = (shape[1] // 2, shape[0] // 2)
    ideal = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= 72**2).astype(np.float32)
    coarse = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= 75**2).astype(np.float32)
    guide = np.full((*shape, 3), 24, dtype=np.uint8)
    guide[ideal.astype(bool)] = 220

    output = _stable_guided_edge_refine(coarse, guide, _policy())

    before = float(np.mean(np.abs(coarse - ideal), dtype=np.float64))
    after = float(np.mean(np.abs(output - ideal), dtype=np.float64))
    assert after <= before * 0.70
    assert _component_count(output >= 0.5) == _component_count(ideal >= 0.5) == 1


@pytest.mark.parametrize(
    "guide",
    [
        pytest.param(
            np.full((360, 640, 3), 80, dtype=np.uint8),
            id="uniform",
        ),
        pytest.param(
            _step_guide((360, 640), 320, left=80, right=84),
            id="low-contrast",
        ),
    ],
)
def test_featureless_or_low_contrast_guide_is_exact_noop(
    guide: np.ndarray,
) -> None:
    mask = _hard_step(guide.shape[:2], 323)

    output = _stable_guided_edge_refine(mask, guide, _policy())

    np.testing.assert_array_equal(output, mask)


def test_equal_competing_gradients_are_exact_noop() -> None:
    shape = (360, 640)
    mask = _hard_step(shape, 323)
    guide = np.full((*shape, 3), 20, dtype=np.uint8)
    guide[:, 320:] = 100
    guide[:, 322:] = 180

    output = _stable_guided_edge_refine(mask, guide, _policy())

    np.testing.assert_array_equal(output, mask)


def test_alternating_nearby_gradient_preference_is_exact_noop() -> None:
    shape = (360, 640)
    mask = _hard_step(shape, 323)
    outputs = []
    for middle in (101, 99, 101, 99, 101, 99):
        guide = np.full((*shape, 3), 20, dtype=np.uint8)
        guide[:, 320:] = middle
        guide[:, 322:] = 180
        outputs.append(_stable_guided_edge_refine(mask, guide, _policy()))

    for output in outputs:
        np.testing.assert_array_equal(output, mask)
    assert _contour_displacement_p95(outputs) == 0.0


@pytest.mark.parametrize(
    "guide_factory",
    [_noisy_step_guide, _block_noisy_step_guide],
    ids=("seeded-camera-noise", "eight-pixel-block-noise"),
)
def test_noise_does_not_make_stationary_contour_or_area_drift(
    guide_factory,
) -> None:
    shape = (360, 640)
    true_edge_x = shape[1] // 2
    mask = _hard_step(shape, true_edge_x + 3)
    outputs = [
        _stable_guided_edge_refine(
            mask,
            guide_factory(shape, true_edge_x, phase),
            _policy(),
        )
        for phase in range(12)
    ]

    assert _contour_displacement_p95(outputs) <= 0.5
    assert _area_drift_p95(outputs) <= 0.01
    assert (
        max(abs(_median_contour_x(output) - (true_edge_x - 0.5)) for output in outputs)
        <= 1.5
    )


def test_resolution_normalized_behavior_is_equivalent() -> None:
    policy = _policy()
    canonical_errors: list[float] = []
    for shape in ((360, 640), (720, 1280), (1080, 1920)):
        scale = shape[0] / policy.reference_short_edge_px
        true_edge_x = shape[1] // 2
        offset_px = round(6.0 * scale)
        ideal = _hard_step(shape, true_edge_x)
        coarse = _hard_step(shape, true_edge_x + offset_px)
        guide = _step_guide(shape, true_edge_x)

        output = _stable_guided_edge_refine(coarse, guide, policy)

        before = float(np.mean(np.abs(coarse - ideal), dtype=np.float64))
        after = float(np.mean(np.abs(output - ideal), dtype=np.float64))
        assert after <= before * 0.60
        canonical_error = (
            abs(_median_contour_x(output) - _median_contour_x(ideal)) / scale
        )
        canonical_errors.append(canonical_error)
        assert canonical_error <= 1.5

    assert max(canonical_errors) - min(canonical_errors) <= 1.0


def test_soft_ramp_retains_soft_values_and_uncertain_area() -> None:
    shape = (360, 640)
    mask = _soft_ramp(shape, shape[1] // 2, width=16)
    guide = _step_guide(shape, shape[1] // 2)
    uncertain_before = (mask > 0.05) & (mask < 0.95)

    output = _stable_guided_edge_refine(mask, guide, _policy())

    uncertain_after = (output > 0.05) & (output < 0.95)
    assert np.count_nonzero(uncertain_after) >= np.count_nonzero(uncertain_before) * 0.5
    assert np.unique(output[(output > 0.0) & (output < 1.0)]).size >= 4
    np.testing.assert_array_equal(output[:, :300], mask[:, :300])
    np.testing.assert_array_equal(output[:, 340:], mask[:, 340:])
    assert np.all(np.diff(output[shape[0] // 2]) >= -1e-6)


@pytest.mark.parametrize("value", (0.0, 1.0))
def test_uniform_endpoint_mattes_are_exact_noop(value: float) -> None:
    mask = np.full((96, 160), value, dtype=np.float32)
    guide = _noisy_step_guide(mask.shape, 80, phase=3)

    output = _stable_guided_edge_refine(mask, guide, _policy())

    np.testing.assert_array_equal(output, mask)


@pytest.mark.parametrize(
    "guide",
    [
        pytest.param(np.zeros((96, 160), dtype=np.uint8), id="two-dimensional"),
        pytest.param(np.zeros((96, 160, 4), dtype=np.uint8), id="four-channels"),
        pytest.param(np.zeros((96, 160, 3), dtype=np.float32), id="wrong-dtype"),
        pytest.param(np.zeros((95, 160, 3), dtype=np.uint8), id="wrong-shape"),
    ],
)
def test_malformed_guides_return_sanitized_current_alpha(guide: np.ndarray) -> None:
    mask = _soft_ramp((96, 160), 80, width=12)

    output = _stable_guided_edge_refine(mask, guide, _policy())

    np.testing.assert_array_equal(output, mask)


def test_native_failure_returns_exact_current_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = _hard_step((96, 160), 84)
    guide = _step_guide(mask.shape, 80)

    def fail_blur(*_args, **_kwargs):
        raise cv2.error("forced stable-guided failure")

    monkeypatch.setattr(segmentation_mod.cv2, "GaussianBlur", fail_blur)

    output = _stable_guided_edge_refine(mask, guide, _policy())

    np.testing.assert_array_equal(output, mask)


def test_overlarge_boundary_work_is_exact_noop() -> None:
    yy, xx = np.indices((96, 160))
    mask = ((xx + yy) % 2).astype(np.float32)
    guide = np.where(mask[..., None] != 0, 220, 20).astype(np.uint8)
    guide = np.repeat(guide, 3, axis=2)

    output = _stable_guided_edge_refine(mask, guide, _policy())

    np.testing.assert_array_equal(output, mask)


@pytest.mark.parametrize("feature", ("foreground-hair", "background-slit"))
def test_connected_thin_features_and_topology_are_preserved(feature: str) -> None:
    shape = (360, 640)
    mask = np.zeros(shape, dtype=np.float32)
    if feature == "foreground-hair":
        mask[100:270, 150:330] = 1.0
        feature_region = np.zeros(shape, dtype=bool)
        feature_region[180:181, 329:500] = True
        mask[feature_region] = 1.0
        expected_feature = output_feature = lambda alpha: alpha >= 0.5
    else:
        mask[70:290, 100:540] = 1.0
        feature_region = np.zeros(shape, dtype=bool)
        feature_region[70:210, 320:321] = True
        mask[feature_region] = 0.0
        expected_feature = output_feature = lambda alpha: alpha < 0.5
    guide = np.full((*shape, 3), 24, dtype=np.uint8)
    guide[mask >= 0.5] = 220
    foreground_components_before = _component_count(mask >= 0.5)
    background_components_before = _component_count(mask < 0.5)

    output = _stable_guided_edge_refine(mask, guide, _policy())

    expected = expected_feature(mask)[feature_region]
    retained = output_feature(output)[feature_region]
    assert np.count_nonzero(retained == expected) / expected.size >= 0.95
    assert _component_count(output >= 0.5) == foreground_components_before
    assert _component_count(output < 0.5) == background_components_before


def test_candidate_returns_finite_contiguous_float32_through_mask_refiner() -> None:
    shape = (96, 160)
    mask = np.asfortranarray(_soft_ramp(shape, 80, width=12).astype(np.float64))
    mask[0, 0] = np.nan
    mask[0, 1] = np.inf
    mask[0, 2] = -np.inf
    guide = _step_guide(shape, 80)

    output = MaskRefiner(_refiner_config()).refine(mask, guide)

    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert np.isfinite(output).all()
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0


def test_repeated_candidate_calls_are_bit_deterministic() -> None:
    shape = (360, 640)
    mask = _hard_step(shape, 323)
    guide = _noisy_step_guide(shape, 320, phase=7)
    policy = _policy()

    first = _stable_guided_edge_refine(mask, guide, policy)
    second = _stable_guided_edge_refine(mask, guide, policy)

    np.testing.assert_array_equal(first, second)


def test_stable_guided_720p_performance_sanity() -> None:
    shape = (720, 1280)
    mask = _hard_step(shape, 646)
    guide = _step_guide(shape, 640)
    policy = _policy()
    for _ in range(3):
        _stable_guided_edge_refine(mask, guide, policy)

    samples: list[float] = []
    for _ in range(11):
        started = time.perf_counter()
        output = _stable_guided_edge_refine(mask, guide, policy)
        samples.append(time.perf_counter() - started)

    assert output.shape == shape
    median = float(np.median(samples))
    assert median < 0.020, (
        f"720p median stable-guided refinement took {median * 1000.0:.2f} ms"
    )
