"""Contract tests for the canonical VIS-1.1 geometry implementation."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Any, cast

import numpy as np
import pytest

import custback.geometry as geometry
from custback.geometry import (
    FrameValidationError,
    GeometryError,
    GeometrySpec,
    Padding,
    Rect,
    ResizeStep,
    apply_exif_orientation,
    apply_transform,
    orient_frame,
    plan_transform,
    transform_frame,
    validate_bgr_frame,
)


FULL_1280_720 = Rect(0, 0, 1280, 720)
NO_PADDING = Padding()


def _corner_frame(height: int = 3, width: int = 5) -> np.ndarray:
    """Return an asymmetric BGR frame with stable IDs in each corner."""

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[0, 0] = (1, 11, 21)
    frame[0, -1] = (2, 12, 22)
    frame[-1, 0] = (3, 13, 23)
    frame[-1, -1] = (4, 14, 24)
    return frame


def _corner_ids(frame: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(frame[0, 0, 0]),
        int(frame[0, -1, 0]),
        int(frame[-1, 0, 0]),
        int(frame[-1, -1, 0]),
    )


@pytest.mark.parametrize(
    (
        "source",
        "target",
        "fit",
        "rotation",
        "resized",
        "crop",
        "content",
        "padding",
        "interpolation",
    ),
    [
        pytest.param(
            (640, 480),
            (1280, 720),
            "cover",
            0,
            (1280, 960),
            Rect(0, 120, 1280, 840),
            FULL_1280_720,
            NO_PADDING,
            ("linear",),
            id="4:3-to-16:9-cover",
        ),
        pytest.param(
            (640, 480),
            (1280, 720),
            "contain",
            0,
            (960, 720),
            Rect(0, 0, 960, 720),
            Rect(160, 0, 1120, 720),
            Padding(left=160, right=160),
            ("linear",),
            id="4:3-to-16:9-contain",
        ),
        pytest.param(
            (1920, 1080),
            (640, 480),
            "cover",
            0,
            (854, 480),
            Rect(107, 0, 747, 480),
            Rect(0, 0, 640, 480),
            NO_PADDING,
            ("area",),
            id="16:9-to-4:3-cover",
        ),
        pytest.param(
            (1920, 1080),
            (640, 480),
            "contain",
            0,
            (640, 360),
            Rect(0, 0, 640, 360),
            Rect(0, 60, 640, 420),
            Padding(top=60, bottom=60),
            ("area",),
            id="16:9-to-4:3-contain",
        ),
        pytest.param(
            (720, 1280),
            (1280, 720),
            "cover",
            0,
            (1280, 2276),
            Rect(0, 778, 1280, 1498),
            FULL_1280_720,
            NO_PADDING,
            ("linear",),
            id="portrait-to-landscape-cover",
        ),
        pytest.param(
            (720, 1280),
            (1280, 720),
            "cover",
            90,
            (1280, 720),
            FULL_1280_720,
            FULL_1280_720,
            NO_PADDING,
            (),
            id="portrait-rotated-to-exact-landscape",
        ),
        pytest.param(
            (641, 479),
            (1281, 721),
            "cover",
            0,
            (1281, 958),
            Rect(0, 118, 1281, 839),
            Rect(0, 0, 1281, 721),
            NO_PADDING,
            ("linear",),
            id="odd-cover",
        ),
        pytest.param(
            (641, 479),
            (1281, 721),
            "contain",
            0,
            (964, 721),
            Rect(0, 0, 964, 721),
            Rect(158, 0, 1122, 721),
            Padding(left=158, right=159),
            ("linear",),
            id="odd-contain",
        ),
    ],
)
def test_plans_match_every_adr_worked_example(
    source,
    target,
    fit,
    rotation,
    resized,
    crop,
    content,
    padding,
    interpolation,
):
    plan = plan_transform(
        source,
        target,
        rotation=rotation,
        fit=fit,
    )

    assert plan.source_size == source
    assert plan.target_size == target
    assert plan.oriented_size == (
        (source[1], source[0]) if rotation in (90, 270) else source
    )
    assert plan.resized_size == resized
    assert plan.crop_rect == crop
    assert plan.content_rect == content
    assert plan.pad_rect == content
    assert plan.padding == padding
    assert tuple(step.interpolation for step in plan.resize_steps) == interpolation
    assert plan.scale_x == pytest.approx(resized[0] / plan.oriented_size[0])
    assert plan.scale_y == pytest.approx(resized[1] / plan.oriented_size[1])


@pytest.mark.parametrize(
    ("fit", "anchors", "expected_crop", "expected_content", "expected_padding"),
    [
        (
            "cover",
            (0.5, 0.0),
            Rect(0, 0, 1280, 720),
            FULL_1280_720,
            NO_PADDING,
        ),
        (
            "cover",
            (0.5, 1.0),
            Rect(0, 240, 1280, 960),
            FULL_1280_720,
            NO_PADDING,
        ),
        (
            "contain",
            (0.0, 0.5),
            Rect(0, 0, 960, 720),
            Rect(0, 0, 960, 720),
            Padding(right=320),
        ),
        (
            "contain",
            (1.0, 0.5),
            Rect(0, 0, 960, 720),
            Rect(320, 0, 1280, 720),
            Padding(left=320),
        ),
    ],
)
def test_anchor_edges_use_viewer_coordinates_and_exact_excess(
    fit,
    anchors,
    expected_crop,
    expected_content,
    expected_padding,
):
    plan = plan_transform((640, 480), (1280, 720), fit=fit, anchors=anchors)

    assert plan.crop_rect == expected_crop
    assert plan.content_rect == expected_content
    assert plan.padding == expected_padding


def test_decimal_anchor_does_not_lose_integer_offset_to_binary_rounding():
    cover = plan_transform(
        (100, 50),
        (50, 50),
        fit="cover",
        anchors=(0.58, 0.5),
    )
    contain = plan_transform(
        (50, 50),
        (100, 50),
        fit="contain",
        anchors=(0.58, 0.5),
    )

    assert cover.crop_rect.left == 29
    assert contain.content_rect.left == 29


def test_plan_telemetry_is_complete_bounded_and_path_free():
    plan = plan_transform(
        (641, 479),
        (1281, 721),
        rotation=180,
        mirror=True,
        fit="contain",
        anchors=(0.25, 0.75),
    )

    assert plan.telemetry() == {
        "source_size": (641, 479),
        "oriented_size": (641, 479),
        "target_size": (1281, 721),
        "resized_size": (964, 721),
        "scale_x": 964 / 641,
        "scale_y": 721 / 479,
        "crop_rect": (0, 0, 964, 721),
        "content_rect": (79, 0, 1043, 721),
        "padding": (79, 0, 238, 0),
        "fit": "contain",
        "rotation": 180,
        "mirror": True,
        "interpolation": ("linear",),
    }


def test_seeded_plan_properties_hold_for_odd_and_extreme_sizes():
    rng = np.random.default_rng(0xC057BAC)
    fits = ("cover", "contain", "stretch")
    rotations = (0, 90, 180, 270)

    for _ in range(300):
        source = tuple(int(value) for value in rng.integers(1, 2049, size=2))
        target = tuple(int(value) for value in rng.integers(1, 2049, size=2))
        fit = fits[int(rng.integers(0, len(fits)))]
        rotation = rotations[int(rng.integers(0, len(rotations)))]
        anchors = tuple(float(value) for value in rng.random(2))
        plan = plan_transform(
            cast(tuple[int, int], source),
            cast(tuple[int, int], target),
            rotation=rotation,
            mirror=bool(rng.integers(0, 2)),
            fit=fit,
            anchors=cast(tuple[float, float], anchors),
        )
        oriented_width, oriented_height = plan.oriented_size
        resized_width, resized_height = plan.resized_size
        target_width, target_height = plan.target_size

        assert oriented_width > 0 and oriented_height > 0
        assert resized_width > 0 and resized_height > 0
        assert math.isfinite(plan.scale_x) and plan.scale_x > 0.0
        assert math.isfinite(plan.scale_y) and plan.scale_y > 0.0
        assert plan.scale_x == resized_width / oriented_width
        assert plan.scale_y == resized_height / oriented_height

        if fit == "cover":
            assert resized_width >= target_width
            assert resized_height >= target_height
            assert plan.crop_rect.width == target_width
            assert plan.crop_rect.height == target_height
            assert 0 <= plan.crop_rect.left <= plan.crop_rect.right <= resized_width
            assert 0 <= plan.crop_rect.top <= plan.crop_rect.bottom <= resized_height
            assert plan.content_rect == Rect(0, 0, target_width, target_height)
            assert plan.padding == NO_PADDING
        elif fit == "contain":
            assert resized_width <= target_width
            assert resized_height <= target_height
            assert plan.crop_rect == Rect(0, 0, resized_width, resized_height)
            assert plan.content_rect.width == resized_width
            assert plan.content_rect.height == resized_height
            assert (
                0 <= plan.content_rect.left <= plan.content_rect.right <= target_width
            )
            assert (
                0 <= plan.content_rect.top <= plan.content_rect.bottom <= target_height
            )
            assert (
                plan.padding.left + resized_width + plan.padding.right == target_width
            )
            assert (
                plan.padding.top + resized_height + plan.padding.bottom == target_height
            )
        else:
            assert plan.resized_size == plan.target_size
            assert plan.crop_rect == Rect(0, 0, target_width, target_height)
            assert plan.content_rect == plan.crop_rect
            assert plan.padding == NO_PADDING
            assert len(plan.resize_steps) <= 2

        if plan.resize_steps:
            assert plan.resize_steps[-1].target_size == plan.resized_size
        else:
            assert plan.resized_size == plan.oriented_size


def test_plans_are_frozen_hashable_cached_and_cache_is_bounded():
    geometry.clear_plan_cache()
    first = plan_transform(
        (640, 480),
        (1280, 720),
        rotation=90,
        mirror=True,
        fit="contain",
        anchors=(0.25, 0.75),
    )
    second = plan_transform(
        (640, 480),
        (1280, 720),
        rotation=90,
        mirror=True,
        fit="contain",
        anchors=(0.25, 0.75),
    )

    assert second is first
    assert hash(first) == hash(second)
    assert getattr(geometry.plan_cache_info(), "hits") == 1
    with pytest.raises(FrozenInstanceError):
        setattr(first, "scale_x", 123.0)
    with pytest.raises(FrozenInstanceError):
        setattr(first.spec, "fit", "cover")

    variants = (
        plan_transform((641, 480), (1280, 720), rotation=90, mirror=True),
        plan_transform((640, 480), (1281, 720), rotation=90, mirror=True),
        plan_transform((640, 480), (1280, 720), rotation=180, mirror=True),
        plan_transform((640, 480), (1280, 720), rotation=90, mirror=False),
        plan_transform((640, 480), (1280, 720), rotation=90, mirror=True, fit="cover"),
        plan_transform(
            (640, 480),
            (1280, 720),
            rotation=90,
            mirror=True,
            anchors=(0.0, 0.75),
        ),
    )
    assert all(candidate != first for candidate in variants)

    geometry.clear_plan_cache()
    for width in range(1, 601):
        plan_transform((width, 17), (31, 23))
    info = geometry.plan_cache_info()
    assert getattr(info, "maxsize") == 512
    assert getattr(info, "currsize") == 512


@pytest.mark.parametrize(
    ("rotation", "expected_size", "expected_corners"),
    [
        (0, (5, 3), (1, 2, 3, 4)),
        (90, (3, 5), (3, 1, 4, 2)),
        (180, (5, 3), (4, 3, 2, 1)),
        (270, (3, 5), (2, 4, 1, 3)),
    ],
)
def test_manual_rotations_have_documented_corner_positions(
    rotation,
    expected_size,
    expected_corners,
):
    source = _corner_frame()
    output, plan = transform_frame(
        source,
        expected_size,
        rotation=rotation,
        fit="cover",
    )

    assert plan.oriented_size == expected_size
    assert _corner_ids(output) == expected_corners
    assert output.flags.c_contiguous


def test_rotation_precedes_horizontal_mirror_in_viewer_coordinates():
    source = _corner_frame()
    rotated_then_mirrored, _ = transform_frame(
        source,
        (3, 5),
        rotation=90,
        mirror=True,
    )
    mirrored_then_rotated = np.rot90(source[:, ::-1], 3)

    assert _corner_ids(rotated_then_mirrored) == (1, 3, 2, 4)
    assert not np.array_equal(rotated_then_mirrored, mirrored_then_rotated)


@pytest.mark.parametrize(
    ("orientation", "expected_shape", "expected_corners"),
    [
        (1, (3, 5, 3), (1, 2, 3, 4)),
        (2, (3, 5, 3), (2, 1, 4, 3)),
        (3, (3, 5, 3), (4, 3, 2, 1)),
        (4, (3, 5, 3), (3, 4, 1, 2)),
        (5, (5, 3, 3), (1, 3, 2, 4)),
        (6, (5, 3, 3), (3, 1, 4, 2)),
        (7, (5, 3, 3), (4, 2, 3, 1)),
        (8, (5, 3, 3), (2, 4, 1, 3)),
    ],
)
def test_all_exif_orientation_pixel_operations_are_complete_and_contiguous(
    orientation,
    expected_shape,
    expected_corners,
):
    output = apply_exif_orientation(_corner_frame(), orientation)

    assert output.shape == expected_shape
    assert _corner_ids(output) == expected_corners
    assert output.flags.c_contiguous


@pytest.mark.parametrize("orientation", (0, 9, -1, True, 1.0, None))
def test_invalid_exif_orientation_fails_deterministically(orientation):
    with pytest.raises(GeometryError, match="EXIF orientation"):
        apply_exif_orientation(_corner_frame(), cast(Any, orientation))


def test_cover_pixel_crop_uses_actual_anchor_coordinate():
    source = np.zeros((6, 8, 3), dtype=np.uint8)
    source[:] = np.arange(6, dtype=np.uint8)[:, None, None]

    top, _ = transform_frame(source, (8, 4), fit="cover", anchors=(0.5, 0.0))
    center, _ = transform_frame(source, (8, 4), fit="cover", anchors=(0.5, 0.5))
    bottom, _ = transform_frame(source, (8, 4), fit="cover", anchors=(0.5, 1.0))

    assert top[:, 0, 0].tolist() == [0, 1, 2, 3]
    assert center[:, 0, 0].tolist() == [1, 2, 3, 4]
    assert bottom[:, 0, 0].tolist() == [2, 3, 4, 5]


def test_contain_pixel_placement_and_opaque_black_padding_are_exact():
    source = np.full((2, 4, 3), (17, 29, 43), dtype=np.uint8)
    output, plan = transform_frame(
        source,
        (4, 4),
        fit="contain",
        anchors=(0.5, 0.5),
    )

    assert plan.content_rect == Rect(0, 1, 4, 3)
    assert plan.padding == Padding(top=1, bottom=1)
    assert np.array_equal(output[1:3], source)
    assert not output[0].any()
    assert not output[3].any()


@pytest.mark.parametrize("fit", ("cover", "contain"))
def test_circle_remains_circular_under_proportional_fit(fit):
    height, width = 240, 320
    yy, xx = np.indices((height, width))
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[(xx - 160) ** 2 + (yy - 120) ** 2 <= 50**2] = 255

    output, _ = transform_frame(source, (320, 180), fit=fit)
    points = np.argwhere(output[:, :, 0] >= 128)
    assert len(points) > 0
    fitted_height = int(points[:, 0].max() - points[:, 0].min() + 1)
    fitted_width = int(points[:, 1].max() - points[:, 1].min() + 1)

    assert abs(fitted_width / fitted_height - 1.0) <= 0.03


class _ResizeSpy:
    INTER_AREA = 31
    INTER_LINEAR = 73

    def __init__(self):
        self.calls: list[tuple[tuple[int, int], int, tuple[int, ...]]] = []

    def resize(self, frame, size, *, interpolation):
        self.calls.append((size, interpolation, frame.shape))
        width, height = size
        return np.zeros((height, width, 3), dtype=frame.dtype)


@pytest.mark.parametrize(
    ("source_size", "target_size", "fit", "expected"),
    [
        ((1920, 1080), (640, 480), "cover", [((854, 480), 31)]),
        ((1920, 1080), (640, 480), "contain", [((640, 360), 31)]),
        ((640, 480), (1280, 720), "cover", [((1280, 960), 73)]),
        ((640, 480), (1280, 720), "contain", [((960, 720), 73)]),
        ((640, 480), (320, 720), "stretch", [((320, 480), 31), ((320, 720), 73)]),
        ((640, 480), (960, 240), "stretch", [((640, 240), 31), ((960, 240), 73)]),
        ((640, 480), (640, 480), "cover", []),
    ],
)
def test_interpolation_selection_and_mixed_stretch_order_are_explicit(
    monkeypatch,
    source_size,
    target_size,
    fit,
    expected,
):
    spy = _ResizeSpy()
    monkeypatch.setattr(geometry, "cv2", spy)
    source = np.zeros((source_size[1], source_size[0], 3), dtype=np.uint8)

    output, plan = transform_frame(source, target_size, fit=fit)

    assert [(size, interpolation) for size, interpolation, _ in spy.calls] == expected
    assert tuple(step.target_size for step in plan.resize_steps) == tuple(
        size for size, _ in expected
    )
    assert output.shape == (target_size[1], target_size[0], 3)


@pytest.mark.parametrize(
    ("source_size", "target_size"),
    [
        ((640, 480), (320, 240)),
        ((640, 480), (640, 240)),
        ((640, 480), (320, 480)),
    ],
)
def test_stretch_pure_downscale_uses_one_area_step(source_size, target_size):
    plan = plan_transform(source_size, target_size, fit="stretch")
    assert plan.resize_steps == (ResizeStep(target_size, "area"),)


@pytest.mark.parametrize(
    ("source_size", "target_size"),
    [
        ((640, 480), (1280, 960)),
        ((640, 480), (1280, 480)),
        ((640, 480), (640, 960)),
    ],
)
def test_stretch_pure_upscale_uses_one_linear_step(source_size, target_size):
    plan = plan_transform(source_size, target_size, fit="stretch")
    assert plan.resize_steps == (ResizeStep(target_size, "linear"),)


@pytest.mark.parametrize(
    ("bad_size", "message"),
    [
        ((0, 10), "dimensions must be positive"),
        ((10, 0), "dimensions must be positive"),
        ((-1, 10), "dimensions must be positive"),
        ((10, -1), "dimensions must be positive"),
        ((True, 10), "pair of integers"),
        ((10, False), "pair of integers"),
        ((10.0, 20), "pair of integers"),
        ((10, 20.0), "pair of integers"),
        ([10, 20], "pair of integers"),
        ((10,), "pair of integers"),
        ((10, 20, 30), "pair of integers"),
    ],
)
def test_invalid_source_and_target_sizes_fail_deterministically(bad_size, message):
    with pytest.raises(GeometryError, match=message):
        plan_transform(cast(Any, bad_size), (10, 10))
    with pytest.raises(GeometryError, match=message):
        plan_transform((10, 10), cast(Any, bad_size))


@pytest.mark.parametrize("rotation", (-90, 45, 360, True, 90.0, "90", None))
def test_invalid_rotation_fails_deterministically(rotation):
    with pytest.raises(GeometryError, match="rotation"):
        plan_transform((10, 10), (20, 20), rotation=cast(Any, rotation))
    with pytest.raises(GeometryError, match="rotation"):
        orient_frame(_corner_frame(), rotation=cast(Any, rotation))


@pytest.mark.parametrize("mirror", (0, 1, "true", None, np.bool_(True)))
def test_non_boolean_mirror_fails_deterministically(mirror):
    with pytest.raises(GeometryError, match="mirror must be a boolean"):
        plan_transform((10, 10), (20, 20), mirror=cast(Any, mirror))
    with pytest.raises(GeometryError, match="mirror must be a boolean"):
        orient_frame(_corner_frame(), mirror=cast(Any, mirror))


@pytest.mark.parametrize("fit", ("crop", "", None, 1))
def test_invalid_fit_fails_deterministically(fit):
    with pytest.raises(GeometryError, match="fit must be one of"):
        plan_transform((10, 10), (20, 20), fit=cast(Any, fit))


@pytest.mark.parametrize(
    "anchors",
    (
        [0.5, 0.5],
        (0.5,),
        (0.5, 0.5, 0.5),
        (float("nan"), 0.5),
        (float("inf"), 0.5),
        (-0.001, 0.5),
        (1.001, 0.5),
        (True, 0.5),
        ("0.5", 0.5),
    ),
)
def test_invalid_anchors_fail_deterministically(anchors):
    with pytest.raises(GeometryError, match="anchor"):
        plan_transform((10, 10), (20, 20), anchors=cast(Any, anchors))


@pytest.mark.parametrize(
    "bad_frame",
    (
        None,
        [],
        np.zeros((3, 5, 3), dtype=np.float32),
        np.zeros((3, 5, 3), dtype=np.int16),
        np.zeros((3, 5), dtype=np.uint8),
        np.zeros((3, 5, 4), dtype=np.uint8),
        np.zeros((3, 5, 0), dtype=np.uint8),
        np.zeros((0, 5, 3), dtype=np.uint8),
        np.zeros((3, 0, 3), dtype=np.uint8),
        np.zeros((), dtype=np.uint8),
    ),
)
def test_invalid_frames_fail_at_the_shared_boundary(bad_frame):
    with pytest.raises(FrameValidationError):
        validate_bgr_frame(bad_frame, name="test frame")
    with pytest.raises(FrameValidationError):
        transform_frame(cast(Any, bad_frame), (8, 6))


def test_contiguity_can_be_enforced_at_a_sink_and_is_normalized_by_transform():
    source = np.arange(6 * 16 * 3, dtype=np.uint8).reshape(6, 16, 3)[:, ::2]
    assert source.shape == (6, 8, 3)
    assert not source.flags.c_contiguous
    assert validate_bgr_frame(source) is source
    with pytest.raises(FrameValidationError, match="C-contiguous"):
        validate_bgr_frame(source, require_contiguous=True)

    output, plan = transform_frame(source, (8, 6), fit="cover")

    assert plan.is_exact_noop
    assert output.flags.c_contiguous
    assert np.array_equal(output, source)


def test_exact_noop_is_pixel_preserving_and_does_not_resize(monkeypatch):
    source = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    class NoResize:
        INTER_AREA = 1
        INTER_LINEAR = 2

        @staticmethod
        def resize(*_args, **_kwargs):
            pytest.fail("exact geometry must not call resize")

    monkeypatch.setattr(geometry, "cv2", NoResize())
    output, plan = transform_frame(source, (8, 6), fit="cover")

    assert plan.is_exact_noop
    assert not plan.resize_steps
    assert np.array_equal(output, source)
    assert output.flags.c_contiguous
    assert np.shares_memory(output, source)


@pytest.mark.parametrize(
    ("fit", "rotation", "mirror", "target"),
    [
        ("cover", 0, False, (17, 9)),
        ("contain", 90, False, (17, 9)),
        ("stretch", 180, True, (17, 9)),
        ("cover", 270, True, (9, 17)),
    ],
)
def test_transform_result_always_satisfies_external_frame_contract(
    fit,
    rotation,
    mirror,
    target,
):
    source = np.arange(7 * 11 * 3, dtype=np.uint16).reshape(7, 11, 3)
    source = (source % 251).astype(np.uint8)

    output, _ = transform_frame(
        source,
        target,
        fit=fit,
        rotation=rotation,
        mirror=mirror,
        anchors=(0.37, 0.81),
    )

    assert output.shape == (target[1], target[0], 3)
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous
    assert np.isfinite(output).all()


def test_apply_transform_rejects_a_frame_with_stale_source_dimensions():
    plan = plan_transform((8, 6), (16, 9))
    with pytest.raises(GeometryError, match="does not match transform plan"):
        apply_transform(np.zeros((7, 8, 3), dtype=np.uint8), plan)


def test_geometry_values_are_immutable_even_when_constructed_directly():
    spec = GeometrySpec(
        source_size=(8, 6),
        target_size=(16, 9),
        fit="stretch",
        anchor_x=cast(Any, np.float64(0.25)),
        anchor_y=cast(Any, np.float32(0.75)),
    )
    plan = plan_transform(
        spec.source_size,
        spec.target_size,
        fit=spec.fit,
        anchors=(spec.anchor_x, spec.anchor_y),
    )

    assert spec.anchor_x == 0.25
    assert spec.anchor_y == 0.75
    assert hash(spec)
    assert hash(plan)
    with pytest.raises(FrozenInstanceError):
        setattr(plan.crop_rect, "left", 99)
    with pytest.raises(FrozenInstanceError):
        setattr(plan.padding, "left", 99)
