"""Deterministic MATTE-2.4 light-wrap stabilization tests."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import custback.compositor as compositor_mod
from custback.color import bgr_u8_to_linear_rgb
from custback.compositor import PreparedLightWrap, composite, prepare_light_wrap
from custback.light_wrap import (
    LightWrapFrameContext,
    LightWrapResetReason,
    LightWrapStabilizer,
)

HEIGHT = 72
WIDTH = 96
START_NS = 4_000_000_000
FRAME_NS = 33_333_333


def _context(
    frame_id: int,
    *,
    timestamp_ns: int | None = None,
    source_token: tuple[object, ...] = ("video", "fixture"),
    discontinuity_revision: int = 0,
) -> LightWrapFrameContext:
    return LightWrapFrameContext(
        frame_id=frame_id,
        timestamp_ns=(
            START_NS + frame_id * FRAME_NS if timestamp_ns is None else timestamp_ns
        ),
        source_token=source_token,
        discontinuity_revision=discontinuity_revision,
    )


def _constant_sample(value: float, shape: tuple[int, int] = (5, 7)) -> np.ndarray:
    return np.full((*shape, 3), value, dtype=np.float32)


def _foreground() -> np.ndarray:
    yy, xx = np.indices((HEIGHT, WIDTH))
    return np.stack(
        (
            70 + (yy % 11),
            95 + (xx % 13),
            135 + ((xx + yy) % 17),
        ),
        axis=2,
    ).astype(np.uint8)


def _model_foreground() -> np.ndarray:
    foreground = _foreground().astype(np.int16)
    foreground[..., 0] += 18
    foreground[..., 1] -= 12
    foreground[..., 2] += 22
    return np.clip(foreground, 0, 255).astype(np.uint8)


def _mask(kind: Literal["narrow", "broad"]) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    if kind == "narrow":
        mask[:, :42] = 1.0
        mask[:, 42:55] = np.linspace(
            0.95,
            0.05,
            13,
            dtype=np.float32,
        )[None, :]
    else:
        # Deliberately broad, under-opaque coverage like the Run-B failure
        # family: a valid core followed by a wide alpha=0.4 contamination band.
        mask[:, :28] = 1.0
        mask[:, 28:64] = 0.4
        mask[:, 64:72] = np.linspace(
            0.35,
            0.0,
            8,
            dtype=np.float32,
        )[None, :]
    return np.ascontiguousarray(mask)


def _static_backdrop() -> np.ndarray:
    yy, xx = np.indices((HEIGHT, WIDTH))
    return np.stack(
        (
            35 + (xx * 2 + yy) % 90,
            85 + (yy * 3) % 100,
            115 + (xx + yy * 2) % 110,
        ),
        axis=2,
    ).astype(np.uint8)


def _moving_backdrop(frame_id: int) -> np.ndarray:
    """Translate a two-color field while preserving exact global channel means."""

    left = np.array((32, 104, 206), dtype=np.uint8)
    right = np.array((218, 164, 46), dtype=np.uint8)
    pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[:, : WIDTH // 2] = left
    pixels[:, WIDTH // 2 :] = right
    return np.ascontiguousarray(np.roll(pixels, frame_id * 11, axis=1))


def _edge_variation(
    frames: list[np.ndarray],
    edge_band: np.ndarray,
) -> float:
    values = []
    for previous, current in zip(frames, frames[1:]):
        values.append(
            float(
                np.mean(
                    np.abs(
                        current[edge_band].astype(np.float32)
                        - previous[edge_band].astype(np.float32)
                    ),
                    dtype=np.float64,
                )
            )
            / 255.0
        )
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def test_zero_wrap_is_an_exact_stateless_bypass(monkeypatch) -> None:
    foreground = _foreground()
    backdrop = _static_backdrop()
    mask = _mask("narrow")
    mask_before = mask.copy()
    stabilizer = LightWrapStabilizer(0.2)
    state_before = stabilizer.snapshot()
    malformed_unused_sample = PreparedLightWrap(
        pixels_bgr=np.zeros((1, 1), dtype=np.float64),
        blend_space="srgb_legacy",
        stabilized=True,
    )

    monkeypatch.setattr(
        compositor_mod,
        "_downscaled_blur",
        lambda *_args, **_kwargs: pytest.fail(
            "zero light wrap must not prepare or consume a wrap sample"
        ),
    )
    expected = (
        foreground.astype(np.float32) * mask[..., None]
        + backdrop.astype(np.float32) * (1.0 - mask[..., None])
    ).astype(np.uint8)
    actual = composite(
        foreground,
        backdrop,
        mask,
        light_wrap=0.0,
        edge_foreground=None,
        prepared_light_wrap=malformed_unused_sample,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(mask, mask_before)
    assert stabilizer.snapshot() == state_before


def test_opencv_preparation_failure_is_bounded_and_does_not_advance_state(
    monkeypatch,
) -> None:
    stabilizer = LightWrapStabilizer(0.2)
    before = stabilizer.snapshot()
    monkeypatch.setattr(
        compositor_mod,
        "_downscaled_blur_with_sample",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            compositor_mod.cv2.error("forced light-wrap resize failure")
        ),
    )

    with pytest.raises(
        compositor_mod.ColorError,
        match="light-wrap preparation",
    ):
        prepare_light_wrap(
            _static_backdrop(),
            blend_space="srgb_legacy",
            stabilizer=stabilizer,
            context=_context(0),
        )

    assert stabilizer.snapshot() == before


def test_repeated_backdrop_frame_does_not_advance_temporal_state() -> None:
    stabilizer = LightWrapStabilizer(0.2)
    context = _context(3)
    first = _constant_sample(70.0)
    contradictory_repeat = _constant_sample(180.0)

    seeded = stabilizer.update(
        first,
        context,
        value_scale=255.0,
        channel_order="bgr",
    )
    before = stabilizer.snapshot()
    repeated = stabilizer.update(
        contradictory_repeat,
        context,
        value_scale=255.0,
        channel_order="bgr",
    )
    after = stabilizer.snapshot()

    np.testing.assert_array_equal(repeated, seeded)
    assert after.updates == before.updates == 1
    assert after.repeated_frames == before.repeated_frames + 1
    assert after.last_timestamp_ns == before.last_timestamp_ns
    assert after.retained_bytes == before.retained_bytes


def test_elapsed_time_filter_is_independent_of_update_partition() -> None:
    initial = _constant_sample(100.0)
    target = _constant_sample(120.0)
    single = LightWrapStabilizer(0.2)
    split = LightWrapStabilizer(0.2)
    origin = _context(0, timestamp_ns=START_NS)

    single.update(initial, origin, value_scale=255.0, channel_order="bgr")
    split.update(initial, origin, value_scale=255.0, channel_order="bgr")
    one_step = single.update(
        target,
        _context(1, timestamp_ns=START_NS + 200_000_000),
        value_scale=255.0,
        channel_order="bgr",
    )
    split.update(
        target,
        _context(1, timestamp_ns=START_NS + 50_000_000),
        value_scale=255.0,
        channel_order="bgr",
    )
    split.update(
        target,
        _context(2, timestamp_ns=START_NS + 125_000_000),
        value_scale=255.0,
        channel_order="bgr",
    )
    partitioned = split.update(
        target,
        _context(3, timestamp_ns=START_NS + 200_000_000),
        value_scale=255.0,
        channel_order="bgr",
    )

    np.testing.assert_allclose(partitioned, one_step, atol=2e-5, rtol=0.0)
    expected = 100.0 + (120.0 - 100.0) * (1.0 - math.exp(-1.0))
    np.testing.assert_allclose(one_step, expected, atol=2e-5, rtol=0.0)


def test_elapsed_time_luminance_bound_is_active_and_exact() -> None:
    stabilizer = LightWrapStabilizer(0.01)
    stabilizer.update(
        _constant_sample(100.0),
        _context(0, timestamp_ns=START_NS),
        value_scale=255.0,
        channel_order="bgr",
    )

    bounded = stabilizer.update(
        _constant_sample(120.0),
        _context(1, timestamp_ns=START_NS + 10_000_000),
        value_scale=255.0,
        channel_order="bgr",
    )

    expected = 100.0 + 1.5 * 0.01 * 255.0
    np.testing.assert_allclose(bounded, expected, atol=2e-5, rtol=0.0)


def test_elapsed_time_chroma_bound_is_active() -> None:
    stabilizer = LightWrapStabilizer(0.01)
    initial = _constant_sample(100.0)
    current = initial.copy()
    current[..., 0] += 30.0
    current[..., 2] -= np.float32(30.0 * 0.0722 / 0.2126)
    stabilizer.update(
        initial,
        _context(0, timestamp_ns=START_NS),
        value_scale=255.0,
        channel_order="bgr",
    )

    bounded = stabilizer.update(
        current,
        _context(1, timestamp_ns=START_NS + 10_000_000),
        value_scale=255.0,
        channel_order="bgr",
    )

    delta = bounded - initial
    weights = np.asarray((0.0722, 0.7152, 0.2126), np.float32)
    luma = np.sum(delta * weights, axis=2)
    chroma = delta - luma[..., None]
    chroma_norm = np.sqrt(np.sum(chroma * chroma, axis=2))
    np.testing.assert_allclose(luma, 0.0, atol=2e-5, rtol=0.0)
    assert float(np.max(chroma_norm)) <= 2.0 * 0.01 * 255.0 + 2e-5
    assert float(np.min(chroma_norm)) >= 2.0 * 0.01 * 255.0 - 2e-5


@pytest.mark.parametrize(
    "partitions_ns",
    [
        [200_000_000],
        [66_666_667, 66_666_667, 66_666_666],
        [33_333_333] * 5 + [33_333_335],
        [17_000_000, 49_000_000, 23_000_000, 71_000_000, 40_000_000],
    ],
)
def test_active_luminance_bound_depends_on_elapsed_time_not_cadence(
    partitions_ns: list[int],
) -> None:
    stabilizer = LightWrapStabilizer(0.01)
    initial = _constant_sample(100.0)
    target = _constant_sample(180.0)
    timestamp_ns = START_NS
    stabilizer.update(
        initial,
        _context(0, timestamp_ns=timestamp_ns),
        value_scale=255.0,
        channel_order="bgr",
    )
    resolved = initial
    for frame_id, dt_ns in enumerate(partitions_ns, start=1):
        timestamp_ns += dt_ns
        resolved = stabilizer.update(
            target,
            _context(frame_id, timestamp_ns=timestamp_ns),
            value_scale=255.0,
            channel_order="bgr",
        )

    assert timestamp_ns == START_NS + 200_000_000
    expected = 100.0 + 1.5 * 0.2 * 255.0
    np.testing.assert_allclose(resolved, expected, atol=5e-5, rtol=0.0)


def test_scene_cut_uses_only_the_current_wrap_sample() -> None:
    stabilizer = LightWrapStabilizer(0.2)
    before = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    after = np.full((HEIGHT, WIDTH, 3), 230, dtype=np.uint8)
    foreground = _foreground()
    mask = _mask("narrow")

    prepare_light_wrap(
        before,
        blend_space="srgb_legacy",
        stabilizer=stabilizer,
        context=_context(0),
    )
    prepared = prepare_light_wrap(
        after,
        blend_space="srgb_legacy",
        stabilizer=stabilizer,
        context=_context(1),
    )
    stabilized = composite(
        foreground,
        after,
        mask,
        light_wrap=0.8,
        prepared_light_wrap=prepared,
    )
    current_only = composite(
        foreground,
        after,
        mask,
        light_wrap=0.8,
    )
    snapshot = stabilizer.snapshot()

    np.testing.assert_array_equal(stabilized, current_only)
    assert snapshot.last_reset_reason is LightWrapResetReason.SCENE_CUT
    assert snapshot.scene_cut_count == 1
    assert snapshot.reset_count == 1
    assert snapshot.last_scene_luma_delta is not None
    assert snapshot.last_scene_luma_delta >= 0.32


@pytest.mark.parametrize("blend_space", ["srgb_legacy", "linear_srgb"])
def test_equal_mean_spatial_scene_cut_does_not_smear_old_layout(
    blend_space: Literal["srgb_legacy", "linear_srgb"],
) -> None:
    before = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    before[:, WIDTH // 2 :] = 255
    after = np.ascontiguousarray(before[:, ::-1])
    np.testing.assert_array_equal(
        before.mean(axis=(0, 1)),
        after.mean(axis=(0, 1)),
    )
    stabilizer = LightWrapStabilizer(0.2)
    prepare_light_wrap(
        before,
        blend_space=blend_space,
        stabilizer=stabilizer,
        context=_context(0),
    )

    prepared = prepare_light_wrap(
        after,
        blend_space=blend_space,
        stabilizer=stabilizer,
        context=_context(1),
    )
    stabilized = composite(
        _foreground(),
        after,
        _mask("narrow"),
        light_wrap=0.8,
        blend_space=blend_space,
        prepared_light_wrap=prepared,
    )
    current_only = composite(
        _foreground(),
        after,
        _mask("narrow"),
        light_wrap=0.8,
        blend_space=blend_space,
    )

    np.testing.assert_array_equal(stabilized, current_only)
    snapshot = stabilizer.snapshot()
    assert snapshot.last_reset_reason is LightWrapResetReason.SCENE_CUT
    assert snapshot.scene_cut_count == 1


@pytest.mark.parametrize(
    ("change", "expected_reason"),
    [
        ("seek", LightWrapResetReason.SEEK),
        ("source", LightWrapResetReason.BACKDROP_CHANGE),
        ("shape", LightWrapResetReason.SHAPE_CHANGE),
    ],
)
def test_seek_source_and_shape_changes_reset_to_the_current_sample(
    change: str,
    expected_reason: LightWrapResetReason,
) -> None:
    stabilizer = LightWrapStabilizer(0.2)
    first = _constant_sample(80.0, (8, 12))
    current = _constant_sample(118.0, (8, 12))
    previous_context = _context(0)
    current_context = _context(1)
    if change == "seek":
        current_context = replace(current_context, discontinuity_revision=1)
    elif change == "source":
        current_context = replace(current_context, source_token=("video", "new"))
    else:
        current = _constant_sample(118.0, (9, 13))

    stabilizer.update(
        first,
        previous_context,
        value_scale=255.0,
        channel_order="bgr",
    )
    resolved = stabilizer.update(
        current,
        current_context,
        value_scale=255.0,
        channel_order="bgr",
    )

    np.testing.assert_array_equal(resolved, current)
    snapshot = stabilizer.snapshot()
    assert snapshot.last_reset_reason is expected_reason
    assert snapshot.reset_count == 1
    assert snapshot.last_frame_id == 1


@pytest.mark.parametrize(
    ("context", "expected_reason"),
    [
        (
            _context(1, timestamp_ns=START_NS),
            LightWrapResetReason.NON_MONOTONIC_TIME,
        ),
        (
            _context(1, timestamp_ns=START_NS + 750_000_001),
            LightWrapResetReason.LONG_GAP,
        ),
    ],
)
def test_invalid_or_stale_elapsed_time_resets_to_the_current_sample(
    context: LightWrapFrameContext,
    expected_reason: LightWrapResetReason,
) -> None:
    stabilizer = LightWrapStabilizer(0.2)
    initial = _constant_sample(80.0)
    current = _constant_sample(118.0)
    stabilizer.update(
        initial,
        _context(0, timestamp_ns=START_NS),
        value_scale=255.0,
        channel_order="bgr",
    )

    resolved = stabilizer.update(
        current,
        context,
        value_scale=255.0,
        channel_order="bgr",
    )

    np.testing.assert_array_equal(resolved, current)
    assert stabilizer.snapshot().last_reset_reason is expected_reason


def test_working_space_change_resets_instead_of_reinterpreting_history() -> None:
    stabilizer = LightWrapStabilizer(0.2)
    encoded = _constant_sample(0.5)
    linear = _constant_sample(0.25)
    stabilizer.update(
        encoded,
        _context(0),
        value_scale=255.0,
        channel_order="bgr",
    )

    resolved = stabilizer.update(
        linear,
        _context(1),
        value_scale=1.0,
        channel_order="bgr",
    )

    np.testing.assert_array_equal(resolved, linear)
    assert (
        stabilizer.snapshot().last_reset_reason
        is LightWrapResetReason.WORKING_SPACE_CHANGE
    )


def test_state_is_low_resolution_clone_is_detached_and_close_releases_it() -> None:
    backdrop = np.full((360, 640, 3), 96, dtype=np.uint8)
    original = LightWrapStabilizer(0.2)
    prepared = prepare_light_wrap(
        backdrop,
        blend_space="srgb_legacy",
        stabilizer=original,
        context=_context(0),
    )
    snapshot = original.snapshot()

    assert snapshot.state_shape == (45, 80, 3)
    assert snapshot.retained_bytes == 2 * 45 * 80 * 3 * np.dtype(np.float32).itemsize
    assert snapshot.retained_bytes < backdrop.nbytes
    assert prepared.pixels_bgr.shape == backdrop.shape
    assert prepared.pixels_bgr.flags.writeable is False

    clone = original.clone()
    assert clone.snapshot() == snapshot
    clone.update(
        _constant_sample(110.0, (45, 80)),
        _context(1),
        value_scale=255.0,
        channel_order="bgr",
    )
    assert clone.snapshot().updates == snapshot.updates + 1
    assert original.snapshot() == snapshot

    clone.close()
    closed = clone.snapshot()
    assert closed.retained_bytes == 0
    assert closed.state_shape is None
    assert closed.last_frame_id is None
    assert original.snapshot() == snapshot


@pytest.mark.parametrize("blend_space", ["srgb_legacy", "linear_srgb"])
@pytest.mark.parametrize("mask_kind", ["narrow", "broad"])
@pytest.mark.parametrize("use_model_foreground", [False, True])
@pytest.mark.parametrize("light_wrap", [0.0, 0.65])
def test_static_backdrop_factorial_preserves_historical_appearance(
    blend_space: Literal["srgb_legacy", "linear_srgb"],
    mask_kind: Literal["narrow", "broad"],
    use_model_foreground: bool,
    light_wrap: float,
) -> None:
    foreground = _foreground()
    backdrop = _static_backdrop()
    mask = _mask(mask_kind)
    edge_foreground = _model_foreground() if use_model_foreground else None
    stabilizer = LightWrapStabilizer(0.2)
    initial_snapshot = stabilizer.snapshot()
    expected = composite(
        foreground,
        backdrop,
        mask,
        light_wrap=light_wrap,
        edge_foreground=edge_foreground,
        blend_space=blend_space,
    )

    for frame_id in range(3):
        prepared = (
            prepare_light_wrap(
                backdrop,
                blend_space=blend_space,
                stabilizer=stabilizer,
                context=_context(frame_id),
            )
            if light_wrap > 0.0
            else None
        )
        actual = composite(
            foreground,
            backdrop,
            mask,
            light_wrap=light_wrap,
            edge_foreground=edge_foreground,
            blend_space=blend_space,
            prepared_light_wrap=prepared,
        )
        np.testing.assert_array_equal(actual, expected)
    if light_wrap == 0.0:
        assert stabilizer.snapshot() == initial_snapshot


def test_linear_stabilization_filters_in_linear_light() -> None:
    black = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    middle = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    stabilizer = LightWrapStabilizer(0.2)
    prepare_light_wrap(
        black,
        blend_space="linear_srgb",
        stabilizer=stabilizer,
        context=_context(0, timestamp_ns=START_NS),
    )
    prepared = prepare_light_wrap(
        middle,
        blend_space="linear_srgb",
        stabilizer=stabilizer,
        context=_context(1, timestamp_ns=START_NS + 100_000_000),
    )

    decoded_middle = float(bgr_u8_to_linear_rgb(middle)[0, 0, 0])
    expected = decoded_middle * (1.0 - math.exp(-0.5))
    np.testing.assert_allclose(
        prepared.pixels_bgr,
        expected,
        atol=2e-6,
        rtol=0.0,
    )
    encoded_space_average = 128.0 * (1.0 - math.exp(-0.5)) / 255.0
    assert abs(float(prepared.pixels_bgr.mean()) - encoded_space_average) > 0.1


@pytest.mark.parametrize("mask_kind", ["narrow", "broad"])
@pytest.mark.parametrize("use_model_foreground", [False, True])
@pytest.mark.parametrize("blend_space", ["srgb_legacy", "linear_srgb"])
def test_dynamic_wrap_variation_drops_thirty_percent_without_changing_alpha(
    mask_kind: Literal["narrow", "broad"],
    use_model_foreground: bool,
    blend_space: Literal["srgb_legacy", "linear_srgb"],
) -> None:
    foreground = _foreground()
    mask = _mask(mask_kind)
    mask_before = mask.copy()
    edge_band = (mask > 0.05) & (mask < 0.95)
    edge_foreground = _model_foreground() if use_model_foreground else None
    stabilizer = LightWrapStabilizer(0.3)
    plain_frames: list[np.ndarray] = []
    legacy_frames: list[np.ndarray] = []
    stabilized_frames: list[np.ndarray] = []

    for frame_id in range(18):
        backdrop = _moving_backdrop(frame_id)
        plain_frames.append(
            composite(
                foreground,
                backdrop,
                mask,
                light_wrap=0.0,
                edge_foreground=edge_foreground,
                blend_space=blend_space,
            )
        )
        legacy_frames.append(
            composite(
                foreground,
                backdrop,
                mask,
                light_wrap=0.8,
                edge_foreground=edge_foreground,
                blend_space=blend_space,
            )
        )
        prepared = prepare_light_wrap(
            backdrop,
            blend_space=blend_space,
            stabilizer=stabilizer,
            context=_context(frame_id),
        )
        stabilized_frames.append(
            composite(
                foreground,
                backdrop,
                mask,
                light_wrap=0.8,
                edge_foreground=edge_foreground,
                blend_space=blend_space,
                prepared_light_wrap=prepared,
            )
        )
        np.testing.assert_array_equal(mask, mask_before)

    # Attribute only wrap-induced motion. Ordinary (1-alpha)*backdrop motion
    # is present in every composite and cannot be removed by a wrap-only state.
    legacy_wrap = [
        wrapped.astype(np.int16) - plain.astype(np.int16)
        for wrapped, plain in zip(legacy_frames, plain_frames)
    ]
    stabilized_wrap = [
        wrapped.astype(np.int16) - plain.astype(np.int16)
        for wrapped, plain in zip(stabilized_frames, plain_frames)
    ]
    legacy_variation = _edge_variation(legacy_wrap, edge_band)
    stabilized_variation = _edge_variation(stabilized_wrap, edge_band)

    assert legacy_variation > 0.01
    assert stabilized_variation <= 0.70 * legacy_variation
    assert _edge_variation(stabilized_frames, edge_band) < _edge_variation(
        legacy_frames,
        edge_band,
    )
    assert stabilizer.snapshot().scene_cut_count == 0
    np.testing.assert_array_equal(mask, mask_before)
