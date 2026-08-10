"""Deterministic contracts for elapsed-time, motion-aware matte stabilization.

The optical-flow estimator is deliberately replaced with exact synthetic
backward flow in the numerical tests.  This keeps the tests independent of
OpenCV optical-flow implementation and platform details while still exercising
the production warp, confidence, boundary-band, and elapsed-time paths.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

import custback.segmentation as segmentation_mod
from custback.config import BoundaryStabilizationConfig, SegmentationConfig
from custback.segmentation import (
    BOUNDARY_FLOW_MAX_LONG_EDGE,
    BOUNDARY_MAX_AREA_FRACTION,
    MaskRefiner,
    SegmentationFrameContext,
    TemporalResetReason,
)


_NS_PER_SECOND = 1_000_000_000
_BASE_TIMESTAMP_NS = 10 * _NS_PER_SECOND


def _context(
    sequence: int,
    relative_timestamp_ns: int,
    shape: tuple[int, int],
) -> SegmentationFrameContext:
    return SegmentationFrameContext(
        sequence=sequence,
        timestamp_ns=_BASE_TIMESTAMP_NS + relative_timestamp_ns,
        generation=1,
        geometry_generation=1,
        shape=shape,
    )


def _motion_config(
    *,
    time_constant_s: float = 0.1,
    max_motion_px_per_s: float = 720.0,
) -> SegmentationConfig:
    return SegmentationConfig(
        mask_blur=0,
        edge_refine=False,
        temporal_smoothing=0.0,
        boundary_stabilization=BoundaryStabilizationConfig(
            mode="motion_aware",
            time_constant_s=time_constant_s,
            max_motion_px_per_s=max_motion_px_per_s,
        ),
    )


def _soft_vertical_edge(
    *,
    shape: tuple[int, int] = (48, 64),
    edge_x: int = 32,
) -> np.ndarray:
    height, width = shape
    x = np.arange(width, dtype=np.float32)
    profile = np.clip((float(edge_x) + 3.0 - x) / 6.0, 0.0, 1.0)
    return np.ascontiguousarray(np.broadcast_to(profile, (height, width)))


def _soft_box(
    *,
    shape: tuple[int, int] = (64, 96),
    bounds: tuple[int, int, int, int] = (20, 14, 44, 50),
    feather_px: float = 3.0,
) -> np.ndarray:
    height, width = shape
    left, top, right, bottom = bounds
    yy, xx = np.indices(shape, dtype=np.float32)
    signed_distance = np.minimum.reduce(
        (
            xx - float(left),
            float(right - 1) - xx,
            yy - float(top),
            float(bottom - 1) - yy,
        )
    )
    return np.ascontiguousarray(
        np.clip(
            (signed_distance + feather_px) / (2.0 * feather_px),
            0.0,
            1.0,
        ).astype(np.float32)
    )


def _shift(mask: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    result = np.zeros_like(mask)
    source_x0 = max(0, -dx)
    source_x1 = min(mask.shape[1], mask.shape[1] - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(mask.shape[0], mask.shape[0] - dy)
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    if source_x1 > source_x0 and source_y1 > source_y0:
        result[target_y0:target_y1, target_x0:target_x1] = mask[
            source_y0:source_y1,
            source_x0:source_x1,
        ]
    return np.ascontiguousarray(result)


def _dummy_frame(shape: tuple[int, int], *, phase: int = 0) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.uint16)
    base = ((xx * 17 + yy * 11 + phase * 29) % 251).astype(np.uint8)
    return np.ascontiguousarray(
        np.stack((base, np.roll(base, 1, axis=1), 255 - base), axis=-1)
    )


def _vertical_contour_positions(mask: np.ndarray) -> np.ndarray:
    """Return subpixel 0.5 crossings for the generated vertical-edge fixture."""

    positions = np.empty(mask.shape[0], dtype=np.float64)
    for row_index, row in enumerate(mask):
        below = np.flatnonzero(row < 0.5)
        assert below.size, "fixture must contain a foreground/background crossing"
        right = int(below[0])
        left = max(0, right - 1)
        left_alpha = float(row[left])
        right_alpha = float(row[right])
        if left_alpha == right_alpha:
            positions[row_index] = float(right)
        else:
            positions[row_index] = left + (
                (left_alpha - 0.5) / (left_alpha - right_alpha)
            )
    return positions


def _contour_displacement_p95(masks: Sequence[np.ndarray]) -> float:
    """Adjacent registered contour displacement for a fixed-guide fixture."""

    contours = [_vertical_contour_positions(mask) for mask in masks]
    displacement = np.concatenate(
        [
            np.abs(current - previous)
            for previous, current in zip(contours, contours[1:])
        ]
    )
    return float(np.quantile(displacement, 0.95))


def _stationary_area_drift_p95(masks: Sequence[np.ndarray]) -> float:
    reference_area = float(masks[0].sum(dtype=np.float64))
    assert reference_area > 0.0
    drift = [
        abs(float(mask.sum(dtype=np.float64)) - reference_area) / reference_area
        for mask in masks
    ]
    return float(np.quantile(drift, 0.95))


def _install_exact_motion(
    monkeypatch: pytest.MonkeyPatch,
    refiner: MaskRefiner,
    *,
    flow_x: float = 0.0,
    flow_y: float = 0.0,
    confidence: float = 1.0,
    calls: list[dict[str, object]] | None = None,
) -> None:
    """Replace guide/flow estimation while retaining production warping."""

    monkeypatch.setattr(
        refiner,
        "_motion_guide",
        lambda frame: np.zeros(frame.shape[:2], dtype=np.float32),
    )

    def estimate(
        current_guide: np.ndarray,
        previous_guide: np.ndarray,
        *,
        full_shape: tuple[int, int],
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if calls is not None:
            calls.append(
                {
                    "current_shape": current_guide.shape,
                    "previous_shape": previous_guide.shape,
                    "full_shape": full_shape,
                    "dt_s": dt_s,
                }
            )
        flow = np.empty((*current_guide.shape, 2), dtype=np.float32)
        flow[..., 0] = flow_x
        flow[..., 1] = flow_y
        certainty = np.full(
            current_guide.shape,
            confidence,
            dtype=np.float32,
        )
        return flow, certainty

    monkeypatch.setattr(refiner, "_estimate_boundary_motion", estimate)


def _cadence_result(
    monkeypatch: pytest.MonkeyPatch,
    timestamps_ns: Sequence[int],
) -> np.ndarray:
    shape = (48, 64)
    initial = _soft_vertical_edge(shape=shape)
    target = initial.copy()
    uncertain = (target > 0.0) & (target < 1.0)
    target[uncertain] = 0.5 + 0.8 * (target[uncertain] - 0.5)
    target = np.ascontiguousarray(target, dtype=np.float32)
    frame = _dummy_frame(shape)
    refiner = MaskRefiner(_motion_config(time_constant_s=0.2))
    _install_exact_motion(monkeypatch, refiner)

    output = refiner.refine(
        initial,
        frame,
        context=_context(0, int(timestamps_ns[0]), shape),
    )
    for sequence, timestamp_ns in enumerate(timestamps_ns[1:], start=1):
        output = refiner.refine(
            target,
            frame,
            context=_context(sequence, int(timestamp_ns), shape),
        )
    return output


def test_equivalent_elapsed_time_is_invariant_at_15_30_and_60_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = []
    for fps in (15, 30, 60):
        timestamps = [round(index * _NS_PER_SECOND / fps) for index in range(fps + 1)]
        outputs.append(_cadence_result(monkeypatch, timestamps))

    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(outputs[1], outputs[2], atol=1e-5, rtol=0.0)


def test_irregular_cadence_matches_uniform_cadence_at_equal_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uniform = [round(index * _NS_PER_SECOND / 30) for index in range(31)]
    irregular = [
        0,
        17_000_000,
        80_000_000,
        145_000_000,
        333_000_000,
        601_000_000,
        _NS_PER_SECOND,
    ]

    regular_output = _cadence_result(monkeypatch, uniform)
    irregular_output = _cadence_result(monkeypatch, irregular)

    np.testing.assert_allclose(
        irregular_output,
        regular_output,
        atol=1e-5,
        rtol=0.0,
    )


@pytest.mark.parametrize("fps", (15, 30, 60))
def test_stationary_two_pixel_jitter_meets_contour_and_area_gates(
    monkeypatch: pytest.MonkeyPatch,
    fps: int,
) -> None:
    shape = (48, 64)
    centered = _soft_vertical_edge(shape=shape, edge_x=33)
    left = _soft_vertical_edge(shape=shape, edge_x=32)
    right = _soft_vertical_edge(shape=shape, edge_x=34)
    refiner = MaskRefiner(_motion_config(time_constant_s=0.2))
    _install_exact_motion(monkeypatch, refiner)
    frame = _dummy_frame(shape)

    outputs = [
        refiner.refine(
            centered,
            frame,
            context=_context(0, 0, shape),
        )
    ]
    raw_masks = [centered]
    for sequence in range(1, fps + 1):
        raw = left if sequence % 2 else right
        raw_masks.append(raw)
        outputs.append(
            refiner.refine(
                raw,
                frame,
                context=_context(
                    sequence,
                    round(sequence * _NS_PER_SECOND / fps),
                    shape,
                ),
            )
        )

    # The generated defect is genuinely a two-pixel alternating contour,
    # while the stabilized output clears both ratified stationary gates.
    assert _contour_displacement_p95(raw_masks[1:]) == pytest.approx(
        2.0,
        abs=1e-6,
    )
    assert _contour_displacement_p95(outputs) <= 1.2
    assert _stationary_area_drift_p95(outputs) <= 0.01


def test_exact_backward_translation_tracks_motion_without_a_double_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (64, 96)
    previous = _soft_box(shape=shape)
    translation_px = 4
    current = _shift(previous, translation_px)
    refiner = MaskRefiner(_motion_config(time_constant_s=0.5))
    # Backward flow maps each current pixel to its prior coordinate.
    _install_exact_motion(
        monkeypatch,
        refiner,
        flow_x=-float(translation_px),
    )

    refiner.refine(
        previous,
        _dummy_frame(shape),
        context=_context(0, 0, shape),
    )
    output = refiner.refine(
        current,
        _dummy_frame(shape, phase=1),
        context=_context(1, 33_333_333, shape),
    )

    np.testing.assert_allclose(output, current, atol=1e-6, rtol=0.0)
    old_only = (previous >= 0.95) & (current <= 0.05)
    new_only = (current >= 0.95) & (previous <= 0.05)
    assert float(output[old_only].max(initial=0.0)) <= 0.05
    assert float(output[new_only].min(initial=1.0)) >= 0.95


def test_exact_two_degree_rotation_follows_current_contour_without_doubling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    shape = (128, 128)
    previous = _soft_box(
        shape=shape,
        bounds=(38, 26, 90, 104),
        feather_px=3.0,
    )
    center = ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)
    forward = cv2.getRotationMatrix2D(center, 2.0, 1.0).astype(np.float32)
    current = cv2.warpAffine(
        previous,
        forward,
        (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32)

    # cv2.warpAffine samples the source through the inverse affine transform.
    # Convert that exact current->previous map into the backward-flow contract.
    backward_affine = cv2.invertAffineTransform(forward)
    yy, xx = np.indices(shape, dtype=np.float32)
    previous_x = (
        backward_affine[0, 0] * xx + backward_affine[0, 1] * yy + backward_affine[0, 2]
    )
    previous_y = (
        backward_affine[1, 0] * xx + backward_affine[1, 1] * yy + backward_affine[1, 2]
    )
    backward_flow = np.ascontiguousarray(
        np.stack((previous_x - xx, previous_y - yy), axis=-1),
        dtype=np.float32,
    )

    refiner = MaskRefiner(_motion_config(time_constant_s=0.5))
    monkeypatch.setattr(
        refiner,
        "_motion_guide",
        lambda frame: np.zeros(frame.shape[:2], dtype=np.float32),
    )
    monkeypatch.setattr(
        refiner,
        "_estimate_boundary_motion",
        lambda *_args, **_kwargs: (
            backward_flow,
            np.ones(shape, dtype=np.float32),
        ),
    )
    frame = _dummy_frame(shape)

    refiner.refine(
        previous,
        frame,
        context=_context(0, 0, shape),
    )
    output = refiner.refine(
        current,
        frame,
        context=_context(1, 33_333_333, shape),
    )

    np.testing.assert_allclose(output, current, atol=2e-4, rtol=0.0)
    contour_disagreement = np.mean((output >= 0.5) != (current >= 0.5))
    assert float(contour_disagreement) <= 0.001
    assert cv2.connectedComponents((output >= 0.5).astype(np.uint8))[0] - 1 == 1


@pytest.mark.parametrize(
    ("translation_px", "confidence"),
    [
        pytest.param(30, 1.0, id="motion-above-configured-maximum"),
        pytest.param(2, 0.0, id="zero-confidence-correspondence"),
    ],
)
def test_untrusted_motion_falls_back_to_the_exact_current_matte(
    monkeypatch: pytest.MonkeyPatch,
    translation_px: int,
    confidence: float,
) -> None:
    shape = (64, 96)
    previous = _soft_box(shape=shape)
    current = _shift(previous, translation_px)
    refiner = MaskRefiner(
        _motion_config(
            time_constant_s=0.5,
            max_motion_px_per_s=720.0,
        )
    )
    _install_exact_motion(
        monkeypatch,
        refiner,
        flow_x=-float(translation_px),
        confidence=confidence,
    )

    refiner.refine(
        previous,
        _dummy_frame(shape),
        context=_context(0, 0, shape),
    )
    output = refiner.refine(
        current,
        _dummy_frame(shape, phase=1),
        context=_context(1, 33_333_333, shape),
    )

    np.testing.assert_array_equal(output, current)


def test_zero_confidence_occlusion_is_local_while_other_boundary_stabilizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (128, 192)
    previous_left = _soft_box(
        shape=shape,
        bounds=(18, 35, 58, 94),
        feather_px=3.0,
    )
    previous_right = _soft_box(
        shape=shape,
        bounds=(132, 35, 172, 94),
        feather_px=3.0,
    )
    previous = np.maximum(previous_left, previous_right).astype(np.float32)

    # The left contour has small estimator jitter. The right contour is newly
    # occluded and receives no correspondence confidence.
    current_left = _soft_box(
        shape=shape,
        bounds=(19, 35, 59, 94),
        feather_px=3.0,
    )
    current_right = _soft_box(
        shape=shape,
        bounds=(132, 35, 172, 94),
        feather_px=3.0,
    )
    current_right[:, 151:] = 0.0
    current = np.maximum(current_left, current_right).astype(np.float32)
    confidence = np.ones(shape, dtype=np.float32)
    confidence[:, shape[1] // 2 :] = 0.0

    refiner = MaskRefiner(_motion_config(time_constant_s=0.5))
    monkeypatch.setattr(
        refiner,
        "_motion_guide",
        lambda frame: np.zeros(frame.shape[:2], dtype=np.float32),
    )
    monkeypatch.setattr(
        refiner,
        "_estimate_boundary_motion",
        lambda *_args, **_kwargs: (
            np.zeros((*shape, 2), dtype=np.float32),
            confidence,
        ),
    )
    frame = _dummy_frame(shape)
    refiner.refine(
        previous,
        frame,
        context=_context(0, 0, shape),
    )
    output = refiner.refine(
        current,
        frame,
        context=_context(1, 33_333_333, shape),
    )

    midpoint = shape[1] // 2
    np.testing.assert_array_equal(output[:, midpoint:], current[:, midpoint:])
    left_output = output[:, :midpoint]
    left_current = current[:, :midpoint]
    left_previous = previous[:, :midpoint]
    assert np.count_nonzero(np.abs(left_output - left_current) > 1e-5) > 0
    assert np.mean(np.abs(left_output - left_previous)) < np.mean(
        np.abs(left_current - left_previous)
    )


def test_zero_confidence_disocclusion_uses_exact_new_foreground_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (128, 192)
    previous_left = _soft_box(
        shape=shape,
        bounds=(18, 35, 58, 94),
        feather_px=3.0,
    )
    previous_right = _soft_box(
        shape=shape,
        bounds=(132, 35, 172, 94),
        feather_px=3.0,
    )
    previous_right[:, 151:] = 0.0
    previous = np.maximum(previous_left, previous_right).astype(np.float32)

    # The left contour has ordinary estimator jitter. The missing half of the
    # right subject is newly revealed and deliberately has no correspondence.
    current_left = _soft_box(
        shape=shape,
        bounds=(19, 35, 59, 94),
        feather_px=3.0,
    )
    current_right = _soft_box(
        shape=shape,
        bounds=(132, 35, 172, 94),
        feather_px=3.0,
    )
    current = np.maximum(current_left, current_right).astype(np.float32)
    confidence = np.ones(shape, dtype=np.float32)
    confidence[:, shape[1] // 2 :] = 0.0

    refiner = MaskRefiner(_motion_config(time_constant_s=0.5))
    monkeypatch.setattr(
        refiner,
        "_motion_guide",
        lambda frame: np.zeros(frame.shape[:2], dtype=np.float32),
    )
    monkeypatch.setattr(
        refiner,
        "_estimate_boundary_motion",
        lambda *_args, **_kwargs: (
            np.zeros((*shape, 2), dtype=np.float32),
            confidence,
        ),
    )
    frame = _dummy_frame(shape)
    refiner.refine(
        previous,
        frame,
        context=_context(0, 0, shape),
    )
    output = refiner.refine(
        current,
        frame,
        context=_context(1, 33_333_333, shape),
    )

    midpoint = shape[1] // 2
    np.testing.assert_array_equal(output[:, midpoint:], current[:, midpoint:])
    newly_revealed = (previous <= 0.05) & (current >= 0.95)
    assert np.any(newly_revealed[:, midpoint:])
    np.testing.assert_array_equal(output[newly_revealed], current[newly_revealed])
    left_output = output[:, :midpoint]
    left_current = current[:, :midpoint]
    left_previous = previous[:, :midpoint]
    assert np.count_nonzero(np.abs(left_output - left_current) > 1e-5) > 0
    assert np.mean(np.abs(left_output - left_previous)) < np.mean(
        np.abs(left_current - left_previous)
    )


def test_stabilization_changes_only_the_boundary_and_preserves_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (48, 64)
    previous = _soft_vertical_edge(shape=shape)
    current = previous.copy()
    uncertain = (current > 0.0) & (current < 1.0)
    current[uncertain] = np.clip(
        current[uncertain] + np.where(current[uncertain] < 0.5, 0.08, -0.08),
        0.0,
        1.0,
    )
    # Exercise normalization rather than relying on the pipeline's validated
    # float32/C-contiguous input contract.
    current_input = np.asfortranarray(current.astype(np.float64))
    refiner = MaskRefiner(_motion_config(time_constant_s=0.5))
    _install_exact_motion(monkeypatch, refiner)

    refiner.refine(
        previous,
        _dummy_frame(shape),
        context=_context(0, 0, shape),
    )
    output = refiner.refine(
        current_input,
        _dummy_frame(shape),
        context=_context(1, 33_333_333, shape),
    )

    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert np.isfinite(output).all()
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0
    np.testing.assert_array_equal(output[:, :24], current[:, :24])
    np.testing.assert_array_equal(output[:, 40:], current[:, 40:])
    assert np.array_equal(output[current == 0.0], current[current == 0.0])
    assert np.array_equal(output[current == 1.0], current[current == 1.0])
    soft_values = np.unique(output[(output > 0.0) & (output < 1.0)])
    assert soft_values.size >= 5
    assert np.all(np.diff(output[shape[0] // 2]) <= 1e-6)


def test_connected_hair_like_soft_feature_survives_temporal_stabilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    shape = (96, 128)
    body = _soft_box(
        shape=shape,
        bounds=(38, 38, 90, 84),
        feather_px=3.0,
    )
    previous = body.copy()
    current = body.copy()
    tip_row = 10
    join_row = 41
    core_column = 64
    feature_rows = slice(tip_row, join_row + 1)
    previous[feature_rows, core_column] = 1.0
    current[feature_rows, core_column] = 1.0
    previous[feature_rows, core_column - 1] = np.maximum(
        previous[feature_rows, core_column - 1],
        0.65,
    )
    previous[feature_rows, core_column + 1] = np.maximum(
        previous[feature_rows, core_column + 1],
        0.65,
    )
    current[feature_rows, core_column - 1] = np.maximum(
        current[feature_rows, core_column - 1],
        0.35,
    )
    current[feature_rows, core_column + 1] = np.maximum(
        current[feature_rows, core_column + 1],
        0.35,
    )
    previous = np.ascontiguousarray(previous, dtype=np.float32)
    current = np.ascontiguousarray(current, dtype=np.float32)

    refiner = MaskRefiner(_motion_config(time_constant_s=0.5))
    _install_exact_motion(monkeypatch, refiner)
    frame = _dummy_frame(shape)
    refiner.refine(
        previous,
        frame,
        context=_context(0, 0, shape),
    )
    output = refiner.refine(
        current,
        frame,
        context=_context(1, 33_333_333, shape),
    )

    exposed_feature_rows = slice(tip_row, 38)
    flank = np.zeros(shape, dtype=bool)
    flank[exposed_feature_rows, core_column - 1] = True
    flank[exposed_feature_rows, core_column + 1] = True
    assert np.any(np.abs(output[flank] - current[flank]) > 1e-5)
    assert np.all(output[flank] > 0.35)
    assert np.all(output[flank] < 0.65)
    np.testing.assert_array_equal(
        output[feature_rows, core_column],
        np.ones(join_row - tip_row + 1, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        output[tip_row - 1, core_column - 1 : core_column + 2],
        np.zeros(3, dtype=np.float32),
    )
    assert cv2.connectedComponents((current >= 0.5).astype(np.uint8))[0] - 1 == 1
    assert cv2.connectedComponents((output >= 0.5).astype(np.uint8))[0] - 1 == 1
    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert np.isfinite(output).all()


@pytest.mark.parametrize("case", ("zero", "one", "tiny", "invalid"))
def test_degenerate_and_invalid_masks_return_a_finite_contiguous_unit_alpha(
    case: str,
) -> None:
    shape = (24, 32)
    if case == "zero":
        mask = np.zeros(shape, dtype=np.float32)
    elif case == "one":
        mask = np.ones(shape, dtype=np.float32)
    elif case == "tiny":
        mask = np.zeros(shape, dtype=np.float32)
        mask[11:13, 15:17] = 1.0
    else:
        mask = np.zeros(shape, dtype=np.float32)
        mask[0, :4] = (np.nan, np.inf, -np.inf, 2.0)
        mask[1, 0] = -1.0

    output = MaskRefiner(_motion_config()).refine(
        mask,
        _dummy_frame(shape),
        context=_context(0, 0, shape),
    )

    assert output.dtype == np.float32
    assert output.flags.c_contiguous
    assert np.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    if case in ("zero", "one", "tiny"):
        np.testing.assert_array_equal(output, mask)


def test_fail_soft_invalid_mask_cannot_poison_following_temporal_history() -> None:
    """Direct callers are sanitized even though pipeline boundaries reject this."""

    shape = (12, 20)
    invalid_row = np.asarray(
        [np.nan, np.inf, -np.inf, 2.0, -1.0],
        dtype=np.float32,
    )
    invalid = np.ascontiguousarray(np.tile(invalid_row, (shape[0], 4)))
    expected_row = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    expected = np.ascontiguousarray(np.tile(expected_row, (shape[0], 4)))
    valid = np.full(shape, 0.5, dtype=np.float32)
    refiner = MaskRefiner(
        SegmentationConfig(
            mask_blur=0,
            edge_refine=False,
            temporal_smoothing=0.75,
        )
    )

    sanitized = refiner.refine(
        invalid,
        _dummy_frame(shape),
        context=_context(0, 0, shape),
    )
    np.testing.assert_array_equal(sanitized, expected)
    np.testing.assert_array_equal(refiner._prev, expected)

    # Every sanitized-history value differs from the valid matte by 0.5, so the
    # legacy disagreement gate releases it instead of propagating bad history.
    output = refiner.refine(
        valid,
        _dummy_frame(shape, phase=1),
        context=_context(1, 33_333_333, shape),
    )

    np.testing.assert_array_equal(output, valid)
    np.testing.assert_array_equal(refiner._prev, valid)
    assert np.isfinite(refiner._prev).all()
    assert refiner.last_input_sequence == 1
    assert refiner.last_input_timestamp_ns == _BASE_TIMESTAMP_NS + 33_333_333


def test_nonfinite_flow_falls_back_to_exact_current_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (48, 64)
    previous = _soft_vertical_edge(shape=shape, edge_x=30)
    current = _soft_vertical_edge(shape=shape, edge_x=32)
    refiner = MaskRefiner(_motion_config())
    monkeypatch.setattr(
        refiner,
        "_motion_guide",
        lambda frame: np.zeros(frame.shape[:2], dtype=np.float32),
    )
    flow = np.zeros((*shape, 2), dtype=np.float32)
    flow[0, 0, 0] = np.nan
    monkeypatch.setattr(
        refiner,
        "_estimate_boundary_motion",
        lambda *_args, **_kwargs: (
            flow,
            np.ones(shape, dtype=np.float32),
        ),
    )
    frame = _dummy_frame(shape)
    refiner.refine(
        previous,
        frame,
        context=_context(0, 0, shape),
    )

    output = refiner.refine(
        current,
        frame,
        context=_context(1, 33_333_333, shape),
    )

    np.testing.assert_array_equal(output, current)


def test_persistent_same_direction_disagreement_hard_releases_by_hold_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (48, 64)
    previous = _soft_vertical_edge(shape=shape, edge_x=30)
    current = _soft_vertical_edge(shape=shape, edge_x=32)
    time_constant_s = 0.05
    maximum_hold_s = min(4.0 * time_constant_s, 0.25)
    step_ns = 25_000_000
    deadline_ns = round(maximum_hold_s * _NS_PER_SECOND)
    refiner = MaskRefiner(
        _motion_config(
            time_constant_s=time_constant_s,
        )
    )
    _install_exact_motion(monkeypatch, refiner)
    frame = _dummy_frame(shape)
    refiner.refine(
        previous,
        frame,
        context=_context(0, 0, shape),
    )

    release_times_ns: list[int] = []
    first_output: np.ndarray | None = None
    for sequence, timestamp_ns in enumerate(
        range(step_ns, deadline_ns + step_ns, step_ns),
        start=1,
    ):
        output = refiner.refine(
            current,
            frame,
            context=_context(sequence, timestamp_ns, shape),
        )
        if first_output is None:
            first_output = output
        if np.array_equal(output, current):
            release_times_ns.append(timestamp_ns)

    assert first_output is not None
    assert not np.array_equal(first_output, current)
    assert release_times_ns
    assert release_times_ns[0] <= deadline_ns


@pytest.mark.parametrize(
    "reason",
    list(TemporalResetReason),
    ids=lambda reason: reason.value,
)
def test_every_reset_reason_clears_motion_state_before_the_boundary_frame(
    monkeypatch: pytest.MonkeyPatch,
    reason: TemporalResetReason,
) -> None:
    shape = (48, 64)
    first = _soft_vertical_edge(shape=shape, edge_x=28)
    second = _soft_vertical_edge(shape=shape, edge_x=30)
    after_reset = _soft_vertical_edge(shape=shape, edge_x=36)
    estimator_calls: list[dict[str, object]] = []
    refiner = MaskRefiner(_motion_config(time_constant_s=0.5))
    _install_exact_motion(
        monkeypatch,
        refiner,
        flow_x=-2.0,
        calls=estimator_calls,
    )

    refiner.refine(
        first,
        _dummy_frame(shape),
        context=_context(0, 0, shape),
    )
    refiner.refine(
        second,
        _dummy_frame(shape, phase=1),
        context=_context(1, 33_333_333, shape),
    )
    assert len(estimator_calls) == 1

    reset_timestamp_ns = _BASE_TIMESTAMP_NS + 1_000_000_000
    refiner.reset_temporal_state(reason, reset_timestamp_ns)
    assert refiner._prev is None
    assert refiner._prev_guide is None
    assert refiner._prev_motion_timestamp_ns is None
    assert refiner._motion_hold_age is None
    assert refiner._motion_direction is None
    output = refiner.refine(
        after_reset,
        _dummy_frame(shape, phase=2),
        context=_context(0, 1_000_000_000, shape),
    )

    np.testing.assert_array_equal(output, after_reset)
    assert len(estimator_calls) == 1
    assert refiner.temporal_reset_count == 1
    assert refiner.last_temporal_reset_reason is reason
    assert refiner.last_temporal_reset_timestamp_ns == reset_timestamp_ns
    assert refiner.last_input_sequence == 0
    assert refiner.last_input_timestamp_ns == reset_timestamp_ns


def test_close_clears_populated_temporal_state_without_reset_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (48, 64)
    reset_timestamp_ns = _BASE_TIMESTAMP_NS
    refiner = MaskRefiner(_motion_config(time_constant_s=0.5))
    _install_exact_motion(monkeypatch, refiner)
    refiner.reset_temporal_state(
        TemporalResetReason.SEGMENTATION_CONFIG,
        reset_timestamp_ns,
    )
    refiner.refine(
        _soft_vertical_edge(shape=shape, edge_x=30),
        _dummy_frame(shape),
        context=_context(0, 0, shape),
    )

    assert refiner._prev is not None
    assert refiner._prev_guide is not None
    assert refiner._prev_motion_timestamp_ns is not None
    assert refiner._motion_hold_age is not None
    assert refiner._motion_direction is not None
    telemetry = (
        refiner.temporal_reset_count,
        refiner.last_temporal_reset_reason,
        refiner.last_temporal_reset_timestamp_ns,
    )

    refiner.close()

    assert refiner._prev is None
    assert refiner._prev_guide is None
    assert refiner._prev_motion_timestamp_ns is None
    assert refiner._motion_hold_age is None
    assert refiner._motion_direction is None
    assert refiner.last_input_sequence is None
    assert refiner.last_input_timestamp_ns is None
    assert (
        refiner.temporal_reset_count,
        refiner.last_temporal_reset_reason,
        refiner.last_temporal_reset_timestamp_ns,
    ) == telemetry

    # Teardown may race with another owner; close remains safe and idempotent.
    refiner.close()


def test_flow_estimator_exception_seeds_current_for_the_next_valid_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (48, 64)
    stale = np.ascontiguousarray(
        (_soft_vertical_edge(shape=shape, edge_x=28) >= 0.5).astype(np.float32)
    )
    current = np.ascontiguousarray(
        (_soft_vertical_edge(shape=shape, edge_x=32) >= 0.5).astype(np.float32)
    )
    refiner = MaskRefiner(_motion_config(time_constant_s=0.5))
    monkeypatch.setattr(
        refiner,
        "_motion_guide",
        lambda frame: np.zeros(frame.shape[:2], dtype=np.float32),
    )
    estimator_calls = [0]

    def estimate(
        current_guide: np.ndarray,
        previous_guide: np.ndarray,
        *,
        full_shape: tuple[int, int],
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del previous_guide, full_shape, dt_s
        estimator_calls[0] += 1
        if estimator_calls[0] == 1:
            raise RuntimeError("synthetic flow failure")
        return (
            np.zeros((*current_guide.shape, 2), dtype=np.float32),
            np.ones(current_guide.shape, dtype=np.float32),
        )

    monkeypatch.setattr(refiner, "_estimate_boundary_motion", estimate)
    frame = _dummy_frame(shape)
    refiner.refine(
        stale,
        frame,
        context=_context(0, 0, shape),
    )

    failed_flow_output = refiner.refine(
        current,
        frame,
        context=_context(1, 33_333_333, shape),
    )
    recovered_output = refiner.refine(
        current,
        frame,
        context=_context(2, 66_666_667, shape),
    )

    np.testing.assert_array_equal(failed_flow_output, current)
    np.testing.assert_array_equal(recovered_output, current)
    assert estimator_calls[0] == 2
    assert refiner.last_input_sequence == 2
    assert refiner.last_input_timestamp_ns == _BASE_TIMESTAMP_NS + 66_666_667


def test_spatial_refinement_failure_does_not_publish_input_or_motion_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (48, 64)
    cfg = _motion_config(time_constant_s=0.5).model_copy(
        update={"mask_blur": 3},
    )
    refiner = MaskRefiner(cfg)
    _install_exact_motion(monkeypatch, refiner)
    frame = _dummy_frame(shape)
    first = _soft_vertical_edge(shape=shape, edge_x=30)
    second = _soft_vertical_edge(shape=shape, edge_x=32)
    refiner.refine(
        first,
        frame,
        context=_context(0, 0, shape),
    )
    assert refiner._prev is not None
    assert refiner._prev_guide is not None
    assert refiner._motion_hold_age is not None
    assert refiner._motion_direction is not None
    state = (
        refiner._prev.copy(),
        refiner._prev_guide.copy(),
        refiner._prev_motion_timestamp_ns,
        refiner._motion_hold_age.copy(),
        refiner._motion_direction.copy(),
        refiner.last_input_sequence,
        refiner.last_input_timestamp_ns,
    )
    real_blur = segmentation_mod.cv2.GaussianBlur
    monkeypatch.setattr(
        segmentation_mod.cv2,
        "GaussianBlur",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic spatial failure")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic spatial failure"):
        refiner.refine(
            second,
            frame,
            context=_context(1, 33_333_333, shape),
        )

    np.testing.assert_array_equal(refiner._prev, state[0])
    np.testing.assert_array_equal(refiner._prev_guide, state[1])
    assert refiner._prev_motion_timestamp_ns == state[2]
    np.testing.assert_array_equal(refiner._motion_hold_age, state[3])
    np.testing.assert_array_equal(refiner._motion_direction, state[4])
    assert refiner.last_input_sequence == state[5]
    assert refiner.last_input_timestamp_ns == state[6]

    monkeypatch.setattr(segmentation_mod.cv2, "GaussianBlur", real_blur)
    recovered = refiner.refine(
        second,
        frame,
        context=_context(1, 33_333_333, shape),
    )
    assert recovered.shape == shape
    assert refiner.last_input_sequence == 1


def test_motion_guide_is_bounded_before_optical_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (720, 1280)
    previous = _soft_box(
        shape=shape,
        bounds=(440, 180, 840, 650),
        feather_px=5.0,
    )
    current = _shift(previous, 2)
    refiner = MaskRefiner(_motion_config())
    observed: list[tuple[tuple[int, ...], tuple[int, int]]] = []

    def estimate(
        current_guide: np.ndarray,
        previous_guide: np.ndarray,
        *,
        full_shape: tuple[int, int],
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del previous_guide, dt_s
        observed.append((current_guide.shape, full_shape))
        flow = np.zeros((*current_guide.shape, 2), dtype=np.float32)
        confidence = np.ones(current_guide.shape, dtype=np.float32)
        return flow, confidence

    monkeypatch.setattr(refiner, "_estimate_boundary_motion", estimate)
    frame = _dummy_frame(shape)
    refiner.refine(
        previous,
        frame,
        context=_context(0, 0, shape),
    )
    refiner.refine(
        current,
        frame,
        context=_context(1, 33_333_333, shape),
    )

    assert len(observed) == 1
    guide_shape, full_shape = observed[0]
    assert full_shape == shape
    assert max(guide_shape) == BOUNDARY_FLOW_MAX_LONG_EDGE
    assert np.prod(guide_shape) <= np.prod(shape) * BOUNDARY_MAX_AREA_FRACTION


def test_real_dis_flow_smoke_uses_current_to_previous_direction() -> None:
    cv2 = pytest.importorskip("cv2")
    shape = (96, 128)
    previous = np.random.default_rng(4).integers(
        0,
        256,
        shape,
        dtype=np.uint8,
    )
    current = cv2.warpAffine(
        previous,
        np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (shape[1], shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    refiner = MaskRefiner(_motion_config())

    estimate = refiner._estimate_boundary_motion(
        current,
        previous,
        full_shape=shape,
        dt_s=1.0 / 30.0,
    )

    assert estimate is not None
    backward, confidence = estimate
    trusted = confidence > 0.2
    assert float(np.mean(trusted)) > 0.1
    # A +3 px source translation requires a roughly -3 px backward sample.
    assert -5.0 < float(np.median(backward[..., 0][trusted])) < -1.0
    assert float(np.median(np.abs(backward[..., 1][trusted]))) < 2.0


def test_overlarge_boundary_falls_back_without_estimating_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (48, 64)
    previous = np.full(shape, 0.45, dtype=np.float32)
    current = np.full(shape, 0.55, dtype=np.float32)
    refiner = MaskRefiner(_motion_config())
    monkeypatch.setattr(
        refiner,
        "_estimate_boundary_motion",
        lambda *_args, **_kwargs: pytest.fail(
            "overlarge boundary must not invoke optical flow"
        ),
    )
    frame = _dummy_frame(shape)

    refiner.refine(
        previous,
        frame,
        context=_context(0, 0, shape),
    )
    output = refiner.refine(
        current,
        frame,
        context=_context(1, 33_333_333, shape),
    )

    np.testing.assert_array_equal(output, current)


def test_boundary_stabilization_defaults_off_and_preserves_legacy_ema() -> None:
    cfg = SegmentationConfig(
        mask_blur=0,
        edge_refine=False,
        temporal_smoothing=0.5,
    )
    assert cfg.boundary_stabilization.mode == "off"
    refiner = MaskRefiner(cfg)
    previous = np.full((10, 10), 0.5, dtype=np.float32)
    current = np.full((10, 10), 0.6, dtype=np.float32)

    first = refiner.refine(previous)
    output = refiner.refine(current)

    np.testing.assert_array_equal(first, previous)
    # Legacy behavior:
    # diff=0.1, keep=0.5*clamp(1-4*0.1, 0, 1)=0.3,
    # result=0.3*0.5 + 0.7*0.6 = 0.57.
    np.testing.assert_allclose(output, 0.57, atol=1e-7, rtol=0.0)
    assert output.dtype == np.float32
