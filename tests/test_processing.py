import time

import numpy as np
import pytest

import custback.backgrounds as backgrounds_mod
import custback.compositor as compositor_mod
import custback.segmentation as segmentation_mod
from custback.backgrounds import (
    BlurBackdrop,
    ColorBackdrop,
    VideoBackdrop,
    create_backdrop,
)
from custback.color import (
    ColorError,
    ColorTransform,
    apply_color_transform,
    bgr_u8_to_linear_rgb,
    linear_rgb_to_bgr_u8,
)
from custback.compositor import (
    composite,
    composite_legacy_predecoded,
    composite_linear_predecoded,
)
from custback.config import BackgroundConfig, SegmentationConfig
from custback.segmentation import (
    HeuristicSegmenter,
    MaskRefiner,
    NullSegmenter,
    _watershed_edge_snap,
)


def frame(h=72, w=128, value=100):
    return np.full((h, w, 3), value, dtype=np.uint8)


def legacy_composite_reference(
    foreground,
    backdrop,
    mask,
    *,
    light_wrap=0.0,
    edge_foreground=None,
):
    """Frozen pre-VIS-2.2 encoded-value arithmetic for compatibility tests."""

    alpha = mask[..., None].astype(np.float32)
    working = foreground.astype(np.float32)
    if light_wrap > 0.0 or edge_foreground is not None:
        band = 4.0 * alpha * (1.0 - alpha)
        if edge_foreground is not None:
            working = working * (1.0 - band) + edge_foreground.astype(np.float32) * band
        if light_wrap > 0.0 and compositor_mod.cv2 is not None:
            wrap = compositor_mod._downscaled_blur(backdrop)
            amount = light_wrap * band
            working = working * (1.0 - amount) + wrap * amount
    return (working * alpha + backdrop.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )


class FakeVideoCapture:
    """Deterministic subset of VideoCapture used by playback-clock tests."""

    def __init__(
        self,
        values,
        *,
        fps=10.0,
        timestamps_s=None,
        fail_indices=(),
        timestamp_seek=True,
        reported_frame_count=None,
    ):
        self.frames = [frame(h=4, w=6, value=value) for value in values]
        self.fps = fps
        self.timestamps_s = timestamps_s
        self.fail_indices = set(fail_indices)
        self.timestamp_seek = timestamp_seek
        self.reported_frame_count = reported_frame_count
        self.pos = 0
        self.last_index = None
        self.grabbed_frame = None
        self.read_calls = 0
        self.grab_calls = 0
        self.set_calls = []
        self.released = False

    def isOpened(self):
        return True

    def release(self):
        self.released = True

    def get(self, prop):
        cv2 = backgrounds_mod.cv2
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return (
                len(self.frames)
                if self.reported_frame_count is None
                else self.reported_frame_count
            )
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.pos
        if prop == cv2.CAP_PROP_POS_MSEC:
            if self.timestamps_s is None or self.last_index is None:
                return float("nan")
            return self.timestamps_s[self.last_index] * 1000.0
        return 0.0

    def set(self, prop, value):
        cv2 = backgrounds_mod.cv2
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = max(0, int(value))
            return True
        if prop == cv2.CAP_PROP_POS_MSEC:
            if not self.timestamp_seek or self.timestamps_s is None:
                return False
            seconds = float(value) / 1000.0
            self.pos = max(
                0,
                min(
                    len(self.frames) - 1,
                    int(np.searchsorted(self.timestamps_s, seconds, side="right") - 1),
                ),
            )
            return True
        return False

    def read(self):
        self.read_calls += 1
        index = self.pos
        if index >= len(self.frames) or index in self.fail_indices:
            return False, None
        self.pos += 1
        self.last_index = index
        return True, self.frames[index].copy()

    def grab(self):
        self.grab_calls += 1
        index = self.pos
        if index >= len(self.frames) or index in self.fail_indices:
            self.grabbed_frame = None
            return False
        self.pos += 1
        self.last_index = index
        self.grabbed_frame = self.frames[index].copy()
        return True

    def retrieve(self):
        if self.grabbed_frame is None:
            return False, None
        return True, self.grabbed_frame.copy()


class TestCompositor:
    def test_full_mask_keeps_foreground(self):
        fg, bg = frame(value=200), frame(value=10)
        out = composite(fg, bg, np.ones((72, 128), np.float32))
        assert (out == 200).all()

    def test_zero_mask_shows_backdrop(self):
        fg, bg = frame(value=200), frame(value=10)
        out = composite(fg, bg, np.zeros((72, 128), np.float32))
        assert (out == 10).all()

    def test_half_mask_blends(self):
        fg, bg = frame(value=200), frame(value=0)
        out = composite(fg, bg, np.full((72, 128), 0.5, np.float32))
        assert abs(int(out[0, 0, 0]) - 100) <= 1

    def test_linear_half_mask_blends_in_linear_light(self):
        fg, bg = frame(value=200), frame(value=0)
        out = composite(
            fg,
            bg,
            np.full((72, 128), 0.5, np.float32),
            blend_space="linear_srgb",
        )
        assert abs(int(out[0, 0, 0]) - 146) <= 1

    def test_linear_mask_endpoints_are_bit_exact_and_owned(self):
        fg = np.arange(3 * 4 * 6, dtype=np.uint8).reshape(4, 6, 3)
        bg = np.ascontiguousarray(255 - fg)
        fg_before = fg.copy()
        bg_before = bg.copy()
        mask = np.array(
            [
                [0.0, 1.0, 0.25, 0.75, 0.0, 1.0],
                [1.0, 0.0, 0.5, 0.5, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.25, 0.75],
                [1.0, 1.0, 0.0, 0.0, 0.75, 0.25],
            ],
            dtype=np.float32,
        )

        out = composite(fg, bg, mask, blend_space="linear_srgb")

        assert np.array_equal(out[mask == 0.0], bg[mask == 0.0])
        assert np.array_equal(out[mask == 1.0], fg[mask == 1.0])
        assert out.dtype == np.uint8
        assert out.flags.c_contiguous
        assert not np.shares_memory(out, fg)
        assert not np.shares_memory(out, bg)
        assert np.array_equal(fg, fg_before)
        assert np.array_equal(bg, bg_before)

    def test_predecoded_linear_compositor_uses_bgr_endpoint_authority(self):
        fg = np.array([[[11, 22, 33], [44, 55, 66], [77, 88, 99]]], np.uint8)
        bg = np.array([[[199, 188, 177], [166, 155, 144], [133, 122, 111]]], np.uint8)
        mask = np.array([[0.0, 1.0, 0.5]], np.float32)
        # Deliberately disagree with the BGR references. The predecoded arrays
        # own soft-pixel math, while external BGR owns exact alpha endpoints.
        foreground_linear = np.zeros((1, 3, 3), np.float32)
        backdrop_linear = np.ones((1, 3, 3), np.float32)

        out = composite_linear_predecoded(
            fg,
            bg,
            mask,
            foreground_linear_rgb=foreground_linear,
            backdrop_linear_rgb=backdrop_linear,
        )

        assert np.array_equal(out[0, 0], bg[0, 0])
        assert np.array_equal(out[0, 1], fg[0, 1])
        assert out.dtype == np.uint8
        assert out.flags.c_contiguous

    def test_accelerated_linear_bgr_ramp_matches_public_rgb_and_exact_endpoints(self):
        codes = np.arange(256, dtype=np.uint8)[None, :]
        fg = np.stack(
            (codes, np.roll(codes, 31, axis=1), np.roll(codes, 97, axis=1)),
            axis=2,
        )
        bg = np.ascontiguousarray(255 - fg)
        mask = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
        foreground_linear_bgr = compositor_mod._bgr_u8_to_linear_bgr_prevalidated(fg)
        backdrop_linear_bgr = compositor_mod._bgr_u8_to_linear_bgr_prevalidated(bg)
        expected = composite_linear_predecoded(
            fg,
            bg,
            mask,
            foreground_linear_rgb=np.ascontiguousarray(
                foreground_linear_bgr[..., ::-1]
            ),
            backdrop_linear_rgb=np.ascontiguousarray(backdrop_linear_bgr[..., ::-1]),
        )

        actual = compositor_mod._composite_linear_bgr_prevalidated(
            fg,
            bg,
            mask,
            foreground_linear_bgr=foreground_linear_bgr,
            backdrop_linear_bgr=backdrop_linear_bgr,
        )

        assert np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))) <= 1
        np.testing.assert_array_equal(actual[mask == 0.0], bg[mask == 0.0])
        np.testing.assert_array_equal(actual[mask == 1.0], fg[mask == 1.0])

    def test_accelerated_linear_bgr_random_transform_edge_and_wrap_match_public_rgb(
        self,
    ):
        rng = np.random.default_rng(0xB6C0)
        shape = (127, 193, 3)
        fg = rng.integers(0, 256, shape, dtype=np.uint8)
        bg = rng.integers(0, 256, shape, dtype=np.uint8)
        edge = rng.integers(0, 256, shape, dtype=np.uint8)
        mask = rng.random(shape[:2], dtype=np.float32)
        mask[0, :] = 0.0
        mask[-1, :] = 1.0
        transform = ColorTransform(
            exposure_ev=0.63,
            wb_gains=(1.11, 0.97, 0.89),
        )
        foreground_linear_bgr = compositor_mod._bgr_u8_to_linear_bgr_prevalidated(fg)
        backdrop_linear_bgr = compositor_mod._bgr_u8_to_linear_bgr_prevalidated(bg)
        edge_linear_bgr = compositor_mod._bgr_u8_to_linear_bgr_prevalidated(edge)
        expected = composite_linear_predecoded(
            fg,
            bg,
            mask,
            foreground_linear_rgb=np.ascontiguousarray(
                foreground_linear_bgr[..., ::-1]
            ),
            backdrop_linear_rgb=np.ascontiguousarray(backdrop_linear_bgr[..., ::-1]),
            light_wrap=0.25,
            edge_foreground_bgr=edge,
            edge_foreground_linear_rgb=np.ascontiguousarray(edge_linear_bgr[..., ::-1]),
            color_transform=transform,
        )

        actual = compositor_mod._composite_linear_bgr_prevalidated(
            fg,
            bg,
            mask,
            foreground_linear_bgr=foreground_linear_bgr,
            backdrop_linear_bgr=backdrop_linear_bgr,
            light_wrap=0.25,
            edge_foreground_bgr=edge,
            edge_foreground_linear_bgr=edge_linear_bgr,
            color_transform=transform,
        )

        assert np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))) <= 1
        np.testing.assert_array_equal(actual[mask == 0.0], bg[mask == 0.0])

    @pytest.mark.parametrize("operation", ["transform", "encode"])
    def test_accelerated_opencv_photometric_error_becomes_color_error(
        self,
        monkeypatch,
        operation,
    ):
        fg = frame(h=8, w=12, value=80)
        bg = frame(h=8, w=12, value=20)
        mask = np.full((8, 12), 0.5, np.float32)
        foreground_linear_bgr = compositor_mod._bgr_u8_to_linear_bgr_prevalidated(fg)
        backdrop_linear_bgr = compositor_mod._bgr_u8_to_linear_bgr_prevalidated(bg)

        def fail(*_args, **_kwargs):
            raise compositor_mod.cv2.error(f"forced {operation} failure")

        if operation == "transform":
            monkeypatch.setattr(compositor_mod.cv2, "transform", fail)
        else:
            monkeypatch.setattr(
                compositor_mod,
                "_consume_linear_bgr_to_bgr_u8_prevalidated",
                fail,
            )

        with pytest.raises(
            ColorError,
            match="OpenCV photometric composition failed",
        ) as raised:
            compositor_mod._composite_linear_bgr_prevalidated(
                fg,
                bg,
                mask,
                foreground_linear_bgr=foreground_linear_bgr,
                backdrop_linear_bgr=backdrop_linear_bgr,
                color_transform=ColorTransform(exposure_ev=0.25),
            )

        assert isinstance(raised.value.__cause__, compositor_mod.cv2.error)

    def test_accelerated_malformed_structure_remains_strict_before_opencv(
        self,
        monkeypatch,
    ):
        fg = frame(h=8, w=12, value=80)
        bg = frame(h=8, w=12, value=20)
        mask = np.full((8, 12), 0.5, np.float32)
        foreground_linear_bgr = (
            compositor_mod._bgr_u8_to_linear_bgr_prevalidated(fg)
        ).astype(np.float64)
        backdrop_linear_bgr = compositor_mod._bgr_u8_to_linear_bgr_prevalidated(bg)
        encode_calls = []

        def unexpected_encode(*_args, **_kwargs):
            encode_calls.append(True)
            raise compositor_mod.cv2.error("encode should not run")

        monkeypatch.setattr(
            compositor_mod,
            "_consume_linear_bgr_to_bgr_u8_prevalidated",
            unexpected_encode,
        )

        with pytest.raises(ValueError, match="foreground_linear_bgr"):
            compositor_mod._composite_linear_bgr_prevalidated(
                fg,
                bg,
                mask,
                foreground_linear_bgr=foreground_linear_bgr,
                backdrop_linear_bgr=backdrop_linear_bgr,
                color_transform=ColorTransform(exposure_ev=0.25),
            )
        assert not encode_calls

    def test_legacy_identity_path_is_byte_identical_with_edge_and_wrap(self):
        pytest.importorskip("cv2")
        rng = np.random.default_rng(0xC057BAC)
        fg = rng.integers(0, 256, size=(72, 128, 3), dtype=np.uint8)
        bg = rng.integers(0, 256, size=(72, 128, 3), dtype=np.uint8)
        clean = rng.integers(0, 256, size=(72, 128, 3), dtype=np.uint8)
        mask = rng.random((72, 128), dtype=np.float32)
        expected = legacy_composite_reference(
            fg,
            bg,
            mask,
            light_wrap=0.35,
            edge_foreground=clean,
        )

        implicit = composite(
            fg,
            bg,
            mask,
            light_wrap=0.35,
            edge_foreground=clean,
        )
        explicit = composite(
            fg,
            bg,
            mask,
            light_wrap=0.35,
            edge_foreground=clean,
            blend_space="srgb_legacy",
            color_transform=ColorTransform(),
        )
        predecoded = composite_legacy_predecoded(
            fg,
            bg,
            mask,
            foreground_linear_rgb=bgr_u8_to_linear_rgb(fg),
            light_wrap=0.35,
            edge_foreground_bgr=clean,
            edge_foreground_linear_rgb=bgr_u8_to_linear_rgb(clean),
            color_transform=ColorTransform(),
        )

        assert np.array_equal(implicit, expected)
        assert np.array_equal(explicit, expected)
        assert np.array_equal(predecoded, expected)

    def test_legacy_transform_is_applied_before_encoded_blending(self):
        fg = frame(h=4, w=6, value=48)
        bg = frame(h=4, w=6, value=8)
        clean = frame(h=4, w=6, value=72)
        mask = np.full((4, 6), 0.5, np.float32)
        transform = ColorTransform(exposure_ev=0.5)
        transformed_fg = linear_rgb_to_bgr_u8(
            apply_color_transform(bgr_u8_to_linear_rgb(fg), transform)
        )
        transformed_clean = linear_rgb_to_bgr_u8(
            apply_color_transform(bgr_u8_to_linear_rgb(clean), transform)
        )
        expected = legacy_composite_reference(
            transformed_fg,
            bg,
            mask,
            edge_foreground=transformed_clean,
        )

        out = composite(
            fg,
            bg,
            mask,
            edge_foreground=clean,
            color_transform=transform,
        )

        assert np.array_equal(out, expected)

    def test_legacy_predecoded_reuses_linear_foregrounds(self, monkeypatch):
        fg = frame(h=4, w=6, value=48)
        bg = frame(h=4, w=6, value=8)
        clean = frame(h=4, w=6, value=72)
        mask = np.full((4, 6), 0.5, np.float32)
        transform = ColorTransform(exposure_ev=0.5)
        foreground_linear = bgr_u8_to_linear_rgb(fg)
        edge_linear = bgr_u8_to_linear_rgb(clean)
        expected = composite(
            fg,
            bg,
            mask,
            edge_foreground=clean,
            color_transform=transform,
        )

        def unexpected_decode(_value):
            raise AssertionError("predecoded legacy compositor decoded an input")

        monkeypatch.setattr(
            compositor_mod,
            "bgr_u8_to_linear_rgb",
            unexpected_decode,
        )
        out = composite_legacy_predecoded(
            fg,
            bg,
            mask,
            foreground_linear_rgb=foreground_linear,
            edge_foreground_bgr=clean,
            edge_foreground_linear_rgb=edge_linear,
            color_transform=transform,
        )

        assert np.array_equal(out, expected)

    def test_linear_transform_applies_identically_to_camera_and_rvm_foreground(self):
        fg = frame(h=4, w=6, value=48)
        bg = frame(h=4, w=6, value=8)
        clean = frame(h=4, w=6, value=72)
        mask = np.full((4, 6), 0.5, np.float32)
        mask[:, :2] = 1.0
        transform = ColorTransform(
            exposure_ev=0.5,
            wb_gains=(1.05, 1.0, 0.95),
        )
        transformed_fg = apply_color_transform(
            bgr_u8_to_linear_rgb(fg),
            transform,
        )
        transformed_clean = apply_color_transform(
            bgr_u8_to_linear_rgb(clean),
            transform,
        )
        bg_linear = bgr_u8_to_linear_rgb(bg)
        expected_core = linear_rgb_to_bgr_u8(transformed_fg)
        expected_edge = linear_rgb_to_bgr_u8(transformed_clean * 0.5 + bg_linear * 0.5)

        out = composite(
            fg,
            bg,
            mask,
            edge_foreground=clean,
            blend_space="linear_srgb",
            color_transform=transform,
        )

        assert np.array_equal(out[:, :2], expected_core[:, :2])
        # At alpha=.5 the edge band is exactly one, so the RVM foreground fully
        # replaces camera pixels before the same transformed linear blend.
        assert np.array_equal(out[:, 2:], expected_edge[:, 2:])

    def test_linear_path_decodes_each_input_once_and_encodes_once(self, monkeypatch):
        calls = {"decode": 0, "encode": 0, "transform": 0, "predecoded": 0}
        real_decode = compositor_mod.bgr_u8_to_linear_rgb
        real_encode = compositor_mod.linear_rgb_to_bgr_u8
        real_transform = compositor_mod.apply_color_transform
        real_predecoded = compositor_mod.composite_linear_predecoded

        def counted_decode(value):
            calls["decode"] += 1
            return real_decode(value)

        def counted_encode(value):
            calls["encode"] += 1
            return real_encode(value)

        def counted_transform(value, transform):
            calls["transform"] += 1
            return real_transform(value, transform)

        def counted_predecoded(*args, **kwargs):
            calls["predecoded"] += 1
            return real_predecoded(*args, **kwargs)

        monkeypatch.setattr(
            compositor_mod,
            "bgr_u8_to_linear_rgb",
            counted_decode,
        )
        monkeypatch.setattr(
            compositor_mod,
            "linear_rgb_to_bgr_u8",
            counted_encode,
        )
        monkeypatch.setattr(
            compositor_mod,
            "apply_color_transform",
            counted_transform,
        )
        monkeypatch.setattr(
            compositor_mod,
            "composite_linear_predecoded",
            counted_predecoded,
        )

        composite(
            frame(value=60),
            frame(value=20),
            np.full((72, 128), 0.5, np.float32),
            edge_foreground=frame(value=80),
            blend_space="linear_srgb",
            color_transform=ColorTransform(exposure_ev=0.25),
        )

        assert calls == {
            "decode": 3,
            "encode": 1,
            "transform": 2,
            "predecoded": 1,
        }

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            composite(frame(), frame(h=10, w=10), np.ones((72, 128), np.float32))

    @staticmethod
    def edge_mask():
        """Person on the left, background right, soft edge column at x=64."""
        mask = np.zeros((72, 128), np.float32)
        mask[:, :64] = 1.0
        mask[:, 64] = 0.5
        return mask

    @pytest.mark.parametrize("blend_space", ["srgb_legacy", "linear_srgb"])
    def test_light_wrap_tints_only_the_edge_band(self, blend_space):
        pytest.importorskip("cv2")
        fg = frame(value=50)
        bg = np.zeros((72, 128, 3), np.uint8)
        bg[:, :, 1] = 200  # green backdrop
        mask = self.edge_mask()
        plain = composite(fg, bg, mask, blend_space=blend_space)
        wrapped = composite(
            fg,
            bg,
            mask,
            light_wrap=0.5,
            blend_space=blend_space,
        )
        # the person's edge picks up backdrop light...
        assert int(wrapped[36, 64, 1]) > int(plain[36, 64, 1])
        # ...but the person core and the pure background are untouched
        assert (wrapped[:, :40] == plain[:, :40]).all()
        assert (wrapped[:, 100:] == plain[:, 100:]).all()
        assert int(wrapped.max()) <= 200

    def test_edge_foreground_used_only_in_band(self):
        fg = frame(value=200)
        bg = frame(value=0)
        clean = np.zeros((72, 128, 3), np.uint8)
        clean[:, :, 2] = 255  # "decontaminated" foreground is pure red
        mask = self.edge_mask()
        out = composite(fg, bg, mask, edge_foreground=clean)
        # band pixel (alpha=0.5, band=1): fg replaced by clean -> red*0.5
        assert abs(int(out[36, 64, 2]) - 127) <= 2
        assert int(out[36, 64, 0]) <= 1
        # core person pixel unchanged
        assert tuple(out[36, 10]) == (200, 200, 200)

    def test_edge_foreground_shape_mismatch_raises(self):
        fg, bg = frame(value=200), frame(value=0)
        with pytest.raises(ValueError, match="shape mismatch"):
            composite(
                fg,
                bg,
                self.edge_mask(),
                edge_foreground=np.zeros((10, 10, 3), np.uint8),
            )

    @pytest.mark.parametrize(
        "bad_frame",
        [
            np.zeros((72, 128, 3), np.float32),
            np.zeros((72, 128), np.uint8),
            np.zeros((72, 128, 4), np.uint8),
            np.zeros((0, 128, 3), np.uint8),
            np.zeros((72, 256, 3), np.uint8)[:, ::2],
        ],
        ids=["dtype", "rank", "channels", "empty", "noncontiguous"],
    )
    def test_invalid_foreground_contract_raises(self, bad_frame):
        with pytest.raises(ValueError, match="foreground"):
            composite(
                bad_frame,
                frame(),
                np.ones((72, 128), np.float32),
            )

    @pytest.mark.parametrize("argument", ["backdrop", "edge_foreground"])
    @pytest.mark.parametrize(
        "bad_frame",
        [
            np.zeros((72, 128, 3), np.float32),
            np.zeros((72, 128), np.uint8),
            np.zeros((72, 128, 4), np.uint8),
            np.zeros((0, 128, 3), np.uint8),
            np.zeros((72, 256, 3), np.uint8)[:, ::2],
        ],
        ids=["dtype", "rank", "channels", "empty", "noncontiguous"],
    )
    def test_invalid_secondary_frame_contract_raises(self, argument, bad_frame):
        kwargs = {argument: bad_frame}
        foreground = frame()
        backdrop = kwargs.pop("backdrop", frame())
        with pytest.raises(ValueError, match=argument):
            composite(
                foreground,
                backdrop,
                np.ones((72, 128), np.float32),
                **kwargs,
            )

    @pytest.mark.parametrize(
        "argument",
        [
            "foreground_linear_rgb",
            "backdrop_linear_rgb",
            "edge_foreground_linear_rgb",
        ],
    )
    @pytest.mark.parametrize(
        "bad_linear",
        [
            np.zeros((72, 128, 3), np.float64),
            np.zeros((72, 128), np.float32),
            np.zeros((72, 128, 4), np.float32),
            np.zeros((10, 10, 3), np.float32),
            np.zeros((0, 128, 3), np.float32),
            np.zeros((72, 256, 3), np.float32)[:, ::2],
            np.full((72, 128, 3), np.nan, np.float32),
            np.full((72, 128, 3), np.inf, np.float32),
            np.full((72, 128, 3), -0.01, np.float32),
            np.full((72, 128, 3), 1.01, np.float32),
        ],
        ids=[
            "dtype",
            "rank",
            "channels",
            "shape",
            "empty",
            "noncontiguous",
            "nan",
            "infinity",
            "negative",
            "above-one",
        ],
    )
    def test_invalid_predecoded_frame_contract_raises(self, argument, bad_linear):
        edge = frame(value=80)
        foreground_linear = bgr_u8_to_linear_rgb(frame())
        backdrop_linear = bgr_u8_to_linear_rgb(frame())
        edge_linear = bgr_u8_to_linear_rgb(edge)
        if argument == "foreground_linear_rgb":
            foreground_linear = bad_linear
        elif argument == "backdrop_linear_rgb":
            backdrop_linear = bad_linear
        else:
            edge_linear = bad_linear

        with pytest.raises(ValueError, match=argument):
            composite_linear_predecoded(
                frame(),
                frame(),
                np.ones((72, 128), np.float32),
                foreground_linear_rgb=foreground_linear,
                backdrop_linear_rgb=backdrop_linear,
                edge_foreground_bgr=edge,
                edge_foreground_linear_rgb=edge_linear,
            )

    @pytest.mark.parametrize(
        "missing",
        ["edge_foreground_bgr", "edge_foreground_linear_rgb"],
    )
    def test_predecoded_edge_pair_must_be_complete(self, missing):
        kwargs = {
            "edge_foreground_bgr": frame(value=80),
            "edge_foreground_linear_rgb": bgr_u8_to_linear_rgb(frame(value=80)),
        }
        kwargs[missing] = None

        with pytest.raises(ValueError, match="must be provided together"):
            composite_linear_predecoded(
                frame(),
                frame(),
                np.ones((72, 128), np.float32),
                foreground_linear_rgb=bgr_u8_to_linear_rgb(frame()),
                backdrop_linear_rgb=bgr_u8_to_linear_rgb(frame()),
                **kwargs,
            )

    def test_invalid_legacy_predecoded_foreground_raises(self):
        with pytest.raises(ValueError, match="foreground_linear_rgb"):
            composite_legacy_predecoded(
                frame(),
                frame(),
                np.ones((72, 128), np.float32),
                foreground_linear_rgb=np.zeros((72, 128, 3), np.float64),
            )

    @pytest.mark.parametrize(
        "bad_mask",
        [
            np.ones((72, 128), np.float64),
            np.ones((72, 128), np.int32),
            np.ones((72, 128, 1), np.float32),
            np.ones((10, 10), np.float32),
            np.ones((0, 128), np.float32),
            np.full((72, 128), np.nan, np.float32),
            np.full((72, 128), np.inf, np.float32),
            np.full((72, 128), -0.01, np.float32),
            np.full((72, 128), 1.01, np.float32),
            np.ones((72, 256), np.float32)[:, ::2],
        ],
        ids=[
            "float64",
            "integer",
            "rank",
            "shape",
            "empty",
            "nan",
            "infinity",
            "negative",
            "above-one",
            "noncontiguous",
        ],
    )
    def test_invalid_mask_contract_raises(self, bad_mask):
        with pytest.raises(ValueError, match="mask"):
            composite(frame(), frame(), bad_mask)

    @pytest.mark.parametrize(
        "light_wrap",
        [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"],
    )
    def test_invalid_light_wrap_raises(self, light_wrap):
        with pytest.raises(ValueError, match="light_wrap"):
            composite(
                frame(),
                frame(),
                np.ones((72, 128), np.float32),
                light_wrap=light_wrap,
            )

    def test_invalid_blend_space_and_transform_raise(self):
        mask = np.ones((72, 128), np.float32)
        with pytest.raises(ValueError, match="blend_space"):
            composite(
                frame(),
                frame(),
                mask,
                blend_space="display_p3",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="color_transform"):
            composite(
                frame(),
                frame(),
                mask,
                color_transform=object(),  # type: ignore[arg-type]
            )


class TestBackdrops:
    def test_color_backdrop(self):
        bd = ColorBackdrop((1, 2, 3))
        out = bd.frame(64, 32)
        assert out.shape == (32, 64, 3)
        assert tuple(out[0, 0]) == (1, 2, 3)

    def test_blur_backdrop_smooths(self):
        bd = BlurBackdrop(strength=15)
        src = frame(value=0)
        src[:, 64:] = 255  # hard vertical edge
        bd.set_source_frame(src)
        out = bd.frame(128, 72)
        edge_band = out[:, 60:68, 0].astype(int)
        assert 0 < edge_band.mean() < 255  # edge got blended

    def test_masked_blur_keeps_person_out_of_backdrop(self):
        pytest.importorskip("cv2")
        img = np.full((128, 128, 3), 20, np.uint8)
        img[44:84, 44:84] = (0, 0, 255)  # bright red "person"
        mask = np.zeros((128, 128), np.float32)
        mask[44:84, 44:84] = 1.0

        plain = BlurBackdrop(strength=31)
        plain.set_source_frame(img)
        masked = BlurBackdrop(strength=31)
        masked.set_source_frame(img, mask)

        # red channel just above the person: plain blur smears the person's
        # color into the backdrop, the masked blur must not
        ring = (slice(38, 44), slice(48, 80), 2)
        assert plain.frame(128, 128)[ring].mean() > 60
        assert masked.frame(128, 128)[ring].mean() < 40

    def test_image_backdrop_from_file(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "bg.png"
        cv2.imwrite(str(path), frame(h=50, w=50, value=42))
        cfg = BackgroundConfig(mode="image", image_path=str(path))
        bd = create_backdrop(cfg)
        assert bd is not None
        out = bd.frame(128, 72)
        assert out is not None
        assert out.shape == (72, 128, 3)
        assert (out == 42).all()

    def test_image_backdrop_rejects_pixel_bomb_before_opencv(
        self, tmp_path, monkeypatch
    ):
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "oversized.png"
        assert cv2.imwrite(str(path), frame(h=9, w=9, value=42))
        monkeypatch.setattr(
            backgrounds_mod.cv2,
            "imread",
            lambda *_args, **_kwargs: pytest.fail(
                "OpenCV must not see an image over the configured pixel cap"
            ),
        )

        with pytest.raises(ValueError, match="exceeds 64 pixels"):
            create_backdrop(
                BackgroundConfig(mode="image", image_path=str(path)),
                image_max_pixels=64,
            )

    def test_image_backdrop_uses_shared_secure_color_decoder(self, monkeypatch):
        pytest.importorskip("cv2")
        decode_calls = []

        def fail_decode(path, expected_format, max_pixels):
            decode_calls.append((path, expected_format, max_pixels))
            raise backgrounds_mod.ColorError("invalid image")

        monkeypatch.setattr(
            backgrounds_mod,
            "decode_image_to_srgb_bgr",
            fail_decode,
        )
        monkeypatch.setattr(
            backgrounds_mod.cv2,
            "imread",
            lambda *_args, **_kwargs: pytest.fail(
                "OpenCV must not see a Pillow decode failure"
            ),
        )

        with pytest.raises(ValueError, match="invalid background image"):
            create_backdrop(BackgroundConfig(mode="image", image_path="corrupt.png"))
        assert decode_calls == [("corrupt.png", "PNG", 16_777_216)]

    def test_video_backdrop_loops(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "bg.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48)
        )
        for value in (10, 200):
            writer.write(frame(h=48, w=64, value=value))
        writer.release()
        cfg = BackgroundConfig(mode="video", video_path=str(path))
        bd = create_backdrop(cfg)
        assert bd is not None
        # read more frames than the file has -> must loop, not fail
        frames = [bd.frame(64, 48) for _ in range(5)]
        for rendered in frames:
            assert rendered is not None
            assert rendered.shape == (48, 64, 3)
        bd.close()

    def test_video_backdrop_uses_source_time_not_call_count(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        path = tmp_path / "timed.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48)
        )
        assert writer.isOpened()
        for value in (10, 100, 200):
            writer.write(frame(h=48, w=64, value=value))
        writer.release()

        now = [0.0]
        bd = VideoBackdrop(str(path), clock=lambda: now[0])
        try:
            first = bd.frame(64, 48).copy()
            first_timing = bd.temporal_frame_timing()
            assert np.array_equal(bd.frame(64, 48), first)  # same instant: hold
            assert bd.temporal_frame_timing() == first_timing
            now[0] = 0.5
            second = bd.frame(64, 48).copy()
            second_timing = bd.temporal_frame_timing()
            now[0] = 1.0
            third = bd.frame(64, 48).copy()
            third_timing = bd.temporal_frame_timing()
            now[0] = 1.5
            looped = bd.frame(64, 48).copy()
            looped_timing = bd.temporal_frame_timing()
            stats = bd.stats_dict()
        finally:
            bd.close()
        assert second.mean() > first.mean() + 50
        assert third.mean() > second.mean() + 50
        assert np.allclose(looped, first, atol=3)
        assert stats["background_video_frames_displayed"] == 4
        assert stats["background_video_frames_reused"] == 1
        assert stats["background_video_frames_skipped"] == 0
        assert first_timing is not None
        assert second_timing is not None
        assert third_timing is not None
        assert looped_timing is not None
        assert [
            first_timing.frame_id,
            second_timing.frame_id,
            third_timing.frame_id,
            looped_timing.frame_id,
        ] == [0, 1, 2, 3]
        assert (
            first_timing.timestamp_ns
            < second_timing.timestamp_ns
            < third_timing.timestamp_ns
            < looped_timing.timestamp_ns
        )
        assert looped_timing.discontinuity_revision > (
            third_timing.discontinuity_revision
        )

    def test_video_backdrop_stats_can_reset_after_unsent_activation_trial(
        self, monkeypatch
    ):
        pytest.importorskip("cv2")
        capture = FakeVideoCapture([10, 20], fps=24.0)
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        backdrop = VideoBackdrop("trial.avi", clock=lambda: 0.0)
        try:
            backdrop.frame(6, 4)  # candidate activation trial, never sent
            assert backdrop.stats_dict()["background_video_frames_displayed"] == 1
            backdrop.reset_stats()
            reset = backdrop.stats_dict()
            assert reset["background_video_frames_displayed"] == 0
            assert reset["background_video_frames_reused"] == 0
            backdrop.frame(6, 4)  # first frame of the installed provider
            installed = backdrop.stats_dict()
            assert installed["background_video_frames_displayed"] == 1
            assert installed["background_video_frames_reused"] == 0
        finally:
            backdrop.close()

    @pytest.mark.parametrize("reported_fps", [0.0, 0.25, 241.0, float("nan")])
    def test_video_backdrop_bounds_bad_fps_metadata(self, monkeypatch, reported_fps):
        pytest.importorskip("cv2")
        capture = FakeVideoCapture([10, 20], fps=reported_fps)
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        backdrop = VideoBackdrop("fake.avi", clock=lambda: 0.0)
        try:
            assert backdrop._fps == VideoBackdrop._DEFAULT_FPS
        finally:
            backdrop.close()

    def test_video_backdrop_prefers_reliable_container_deadlines(self, monkeypatch):
        pytest.importorskip("cv2")
        # Deliberately contradict the nominal 30 FPS metadata. Frame 1 is not
        # due until 400 ms according to the container.
        capture = FakeVideoCapture(
            [10, 100, 200], fps=30.0, timestamps_s=[0.0, 0.4, 0.9]
        )
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        now = [0.0]
        backdrop = VideoBackdrop("vfr.avi", clock=lambda: now[0])
        try:
            first = backdrop.frame(6, 4).copy()
            reads_after_lookahead = capture.read_calls
            now[0] = 0.2
            assert np.array_equal(backdrop.frame(6, 4), first)
            assert capture.read_calls == reads_after_lookahead  # early reuse
            now[0] = 0.4
            assert int(backdrop.frame(6, 4).mean()) == 100
            now[0] = 0.8
            assert int(backdrop.frame(6, 4).mean()) == 100
            now[0] = 0.9
            assert int(backdrop.frame(6, 4).mean()) == 200
        finally:
            backdrop.close()

    def test_video_backdrop_mixed_invalid_pts_uses_relative_fps_fallback(
        self, monkeypatch
    ):
        pytest.importorskip("cv2")
        capture = FakeVideoCapture(
            [0, 50, 100, 150],
            fps=30.0,
            timestamps_s=[0.0, 0.5, float("nan"), 1.5],
        )
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        now = [0.0]
        backdrop = VideoBackdrop("mixed-pts.avi", clock=lambda: now[0])
        try:
            assert int(backdrop.frame(6, 4).mean()) == 0
            now[0] = 0.5
            assert int(backdrop.frame(6, 4).mean()) == 50
            # The missing PTS falls back one nominal interval after 500 ms;
            # it must not use logical index 2 / 30 FPS from epoch zero.
            now[0] = 0.53
            assert int(backdrop.frame(6, 4).mean()) == 50
            now[0] = 0.54
            assert int(backdrop.frame(6, 4).mean()) == 100
        finally:
            backdrop.close()

    def test_video_backdrop_rejects_outlier_after_a_missing_pts(self, monkeypatch):
        pytest.importorskip("cv2")
        capture = FakeVideoCapture(
            [0, 50, 100, 150],
            fps=30.0,
            timestamps_s=[0.0, 0.5, float("nan"), 1000.0],
        )
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        now = [0.0]
        backdrop = VideoBackdrop("outlier-after-gap.avi", clock=lambda: now[0])
        try:
            assert int(backdrop.frame(6, 4).mean()) == 0
            now[0] = 0.5
            assert int(backdrop.frame(6, 4).mean()) == 50
            now[0] = 0.54
            assert int(backdrop.frame(6, 4).mean()) == 100
            # The 1000-second timestamp follows a missing PTS, but is still
            # checked against the last reliable source timestamp. Playback
            # therefore uses one nominal interval instead of freezing.
            now[0] = 0.57
            assert int(backdrop.frame(6, 4).mean()) == 150
        finally:
            backdrop.close()

    def test_video_backdrop_learns_unknown_length_and_loops(self, monkeypatch):
        pytest.importorskip("cv2")
        capture = FakeVideoCapture([10, 200], fps=2.0, reported_frame_count=0)
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        now = [0.0]
        backdrop = VideoBackdrop("unknown-length.avi", clock=lambda: now[0])
        try:
            assert int(backdrop.frame(6, 4).mean()) == 10
            # Jump past an entire loop before EOF has exposed the length. The
            # bounded sequential path must discover two frames and retain the
            # 1.5-second wall-clock phase (frame 1), not freeze or seek blindly.
            now[0] = 1.5
            assert int(backdrop.frame(6, 4).mean()) == 200
            assert backdrop._frame_count == 2
            now[0] = 2.0
            assert int(backdrop.frame(6, 4).mean()) == 10
        finally:
            backdrop.close()

    def test_video_backdrop_corrects_overstated_frame_count_at_eof(self, monkeypatch):
        pytest.importorskip("cv2")
        capture = FakeVideoCapture([10, 200], fps=2.0, reported_frame_count=3)
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        now = [0.0]
        backdrop = VideoBackdrop("overstated-length.avi", clock=lambda: now[0])
        try:
            assert int(backdrop.frame(6, 4).mean()) == 10
            now[0] = 0.5
            assert int(backdrop.frame(6, 4).mean()) == 200
            now[0] = 1.0
            assert int(backdrop.frame(6, 4).mean()) == 10
            assert backdrop._frame_count == 2
            now[0] = 1.5
            assert int(backdrop.frame(6, 4).mean()) == 200
        finally:
            backdrop.close()

    def test_video_backdrop_skips_small_gaps_then_seeks_large_gaps(self, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        capture = FakeVideoCapture(range(20), fps=10.0)
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        now = [0.0]
        backdrop = VideoBackdrop("untimed.avi", clock=lambda: now[0])
        try:
            assert int(backdrop.frame(6, 4).mean()) == 0
            now[0] = 0.5
            assert int(backdrop.frame(6, 4).mean()) == 5
            before_seek = backdrop.temporal_frame_timing()
            # Orientation-control probing is construction-only; playback still
            # needs no seek for this five-frame stale interval.
            assert not [
                call
                for call in capture.set_calls
                if call[0] != cv2.CAP_PROP_ORIENTATION_AUTO
            ]
            assert capture.grab_calls == 4  # decode only the final skipped image
            now[0] = 1.5
            assert int(backdrop.frame(6, 4).mean()) == 15
            after_seek = backdrop.temporal_frame_timing()
            assert (cv2.CAP_PROP_POS_FRAMES, 15) in capture.set_calls
            stats = backdrop.stats_dict()
            assert stats["background_video_frames_displayed"] == 3
            assert stats["background_video_frames_skipped"] == 13
            assert stats["background_video_seek_count"] == 1
            assert stats["background_video_skip_ratio"] == pytest.approx(13 / 16)
            assert before_seek is not None and after_seek is not None
            assert after_seek.timestamp_ns > before_seek.timestamp_ns
            assert (
                after_seek.discontinuity_revision > before_seek.discontinuity_revision
            )
            revision = after_seek.discontinuity_revision
            backdrop.reset_stats()
            timing_after_reset = backdrop.temporal_frame_timing()
            assert timing_after_reset is not None
            assert timing_after_reset.discontinuity_revision == revision
        finally:
            backdrop.close()

    def test_video_skip_crossing_loop_resets_even_when_target_is_not_frame_zero(
        self,
        monkeypatch,
    ):
        pytest.importorskip("cv2")
        capture = FakeVideoCapture(range(5), fps=10.0)
        monkeypatch.setattr(
            backgrounds_mod.cv2,
            "VideoCapture",
            lambda _path: capture,
        )
        now = [0.0]
        backdrop = VideoBackdrop("skipped-loop.avi", clock=lambda: now[0])
        try:
            assert int(backdrop.frame(6, 4).mean()) == 0
            before = backdrop.temporal_frame_timing()
            now[0] = 0.6
            assert int(backdrop.frame(6, 4).mean()) == 1
            after = backdrop.temporal_frame_timing()
        finally:
            backdrop.close()

        assert before is not None
        assert after is not None
        assert after.frame_id == 6
        assert after.discontinuity_revision > before.discontinuity_revision

    def test_video_backdrop_uses_timestamp_seek_and_keeps_loop_phase(self, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        capture = FakeVideoCapture(
            range(6), fps=2.0, timestamps_s=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        )
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        now = [0.0]
        backdrop = VideoBackdrop("timed-loop.avi", clock=lambda: now[0])
        try:
            assert int(backdrop.frame(6, 4).mean()) == 0
            # 10 seconds is three full 3-second loops plus one second: frame 2.
            now[0] = 10.0
            assert int(backdrop.frame(6, 4).mean()) == 2
            assert any(prop == cv2.CAP_PROP_POS_MSEC for prop, _ in capture.set_calls)
        finally:
            backdrop.close()

    def test_video_backdrop_failed_seek_retains_last_good_frame(self, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        capture = FakeVideoCapture(range(20), fps=10.0, fail_indices={15})
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        now = [0.0]
        backdrop = VideoBackdrop("broken-seek.avi", clock=lambda: now[0])
        try:
            backdrop.frame(6, 4)
            now[0] = 0.2
            last_good = backdrop.frame(6, 4).copy()
            assert int(last_good.mean()) == 2
            now[0] = 1.5
            assert np.array_equal(backdrop.frame(6, 4), last_good)
            assert (cv2.CAP_PROP_POS_FRAMES, 15) in capture.set_calls
            # Failed random access must not silently publish frame zero.
            assert capture.set_calls[-1] != (cv2.CAP_PROP_POS_FRAMES, 0)
        finally:
            backdrop.close()

    def test_video_backdrop_rejects_oversized_lookahead_during_preflight(
        self, monkeypatch
    ):
        pytest.importorskip("cv2")
        capture = FakeVideoCapture([10, 200], fps=2.0)
        capture.frames[1] = frame(h=5, w=7, value=200)
        monkeypatch.setattr(backgrounds_mod.cv2, "VideoCapture", lambda _path: capture)
        now = [0.0]
        backdrop = VideoBackdrop(
            "changing-resolution.avi",
            clock=lambda: now[0],
            max_width=6,
            max_height=4,
        )
        try:
            with pytest.raises(ValueError, match="exceeds configured dimensions"):
                backdrop.frame(6, 4)
            reads = capture.read_calls
            now[0] = 0.5
            backdrop.frame(6, 4)
            assert capture.read_calls == reads
        finally:
            backdrop.close()

    def test_passthrough_has_no_backdrop(self):
        assert create_backdrop(BackgroundConfig(mode="passthrough")) is None

    def test_missing_image_raises(self):
        pytest.importorskip("cv2")
        cfg = BackgroundConfig(mode="image", image_path="/nonexistent.png")
        with pytest.raises(FileNotFoundError):
            create_backdrop(cfg)


class TestSegmentation:
    def test_null_segmenter_full_mask(self):
        mask = NullSegmenter().segment(frame())
        assert mask.shape == (72, 128)
        assert (mask == 1.0).all()

    def test_heuristic_finds_bright_center(self):
        cfg = SegmentationConfig(backend="heuristic")
        seg = HeuristicSegmenter(cfg)
        img = frame(value=15)
        img[20:52, 44:84] = 230  # bright centered "person"
        mask = seg.segment(img)
        assert mask[36, 64] == 1.0  # inside the person
        assert mask[5, 5] == 0.0  # dark corner
        assert 0.0 < mask.mean() < 1.0

    def test_refiner_damps_small_fluctuations(self):
        cfg = SegmentationConfig(mask_blur=0, edge_refine=False, temporal_smoothing=0.5)
        refiner = MaskRefiner(cfg)
        refiner.refine(np.full((10, 10), 0.5, np.float32))
        out = refiner.refine(np.full((10, 10), 0.6, np.float32))
        assert 0.5 < out.mean() < 0.6  # pulled back toward the previous mask

    def test_refiner_tracks_large_changes_immediately(self):
        # Adaptive smoothing: real motion must not leave a ghost trail.
        cfg = SegmentationConfig(mask_blur=0, edge_refine=False, temporal_smoothing=0.5)
        refiner = MaskRefiner(cfg)
        refiner.refine(np.zeros((10, 10), np.float32))
        out = refiner.refine(np.ones((10, 10), np.float32))
        assert out.mean() > 0.9

    def test_refiner_output_in_range(self):
        cfg = SegmentationConfig(mask_blur=7, temporal_smoothing=0.3)
        refiner = MaskRefiner(cfg)
        noisy = np.random.default_rng(0).random((40, 40)).astype(np.float32)
        out = refiner.refine(noisy)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_edge_refine_snaps_mask_to_image_edge(self):
        pytest.importorskip("cv2")
        # Real image edge at x=64; the segmenter's mask edge is off by 6 px.
        img = frame(value=10)
        img[:, 64:] = 240
        mask = np.zeros((72, 128), np.float32)
        mask[:, 70:] = 1.0
        ideal = np.zeros((72, 128), np.float32)
        ideal[:, 64:] = 1.0
        cfg = SegmentationConfig(mask_blur=9, edge_refine=True, temporal_smoothing=0.0)
        refined = MaskRefiner(cfg).refine(mask, img)
        err_before = np.abs(mask - ideal).mean()
        err_after = np.abs(refined - ideal).mean()
        assert err_after < err_before  # moved toward the true edge

    @pytest.mark.parametrize("mask_edge", [58, 70])
    def test_edge_refine_snaps_from_both_sides(self, mask_edge):
        pytest.importorskip("cv2")
        img = frame(value=10)
        img[:, 64:] = 240
        mask = np.zeros((72, 128), np.float32)
        mask[:, mask_edge:] = 1.0
        ideal = np.zeros_like(mask)
        ideal[:, 64:] = 1.0
        cfg = SegmentationConfig(mask_blur=9, edge_refine=True, temporal_smoothing=0.0)
        refined = MaskRefiner(cfg).refine(mask, img)
        assert np.abs(refined - ideal).mean() < np.abs(mask - ideal).mean() * 0.5

    def test_edge_snap_is_noop_on_uniform_guide(self):
        pytest.importorskip("cv2")
        mask = np.zeros((96, 128), np.float32)
        mask[:, 70:] = 1.0
        guide = frame(h=96, w=128, value=80)
        refined = _watershed_edge_snap(mask, guide)
        assert np.array_equal(refined, mask)

    def test_edge_snap_preserves_thin_foreground_without_safe_markers(self):
        pytest.importorskip("cv2")
        mask = np.zeros((96, 128), np.float32)
        mask[:, 63:65] = 1.0
        guide = frame(h=96, w=128, value=10)
        guide[:, 63:65] = 240
        refined = _watershed_edge_snap(mask, guide)
        assert np.array_equal(refined, mask)

    @pytest.mark.parametrize("component_value", [0.0, 1.0])
    def test_edge_snap_preserves_each_seedless_thin_component(self, component_value):
        pytest.importorskip("cv2")
        base_value = 1.0 - component_value
        mask = np.full((120, 180), base_value, np.float32)
        mask[30:90, 20:80] = component_value  # supplies its own eroded marker
        mask[20:100, 140:142] = component_value  # no component-local marker
        guide = frame(h=120, w=180, value=int(10 + 230 * base_value))
        guide[mask == component_value] = int(10 + 230 * component_value)
        refined = _watershed_edge_snap(mask, guide)
        assert np.array_equal(refined[:, 140:142], mask[:, 140:142])

    def test_edge_snap_falls_back_on_opencv_error(self, monkeypatch):
        cv2 = pytest.importorskip("cv2")
        mask = np.zeros((96, 128), np.float32)
        mask[:, 70:] = 1.0
        guide = frame(h=96, w=128, value=10)
        guide[:, 64:] = 240

        def fail_watershed(*_args, **_kwargs):
            raise cv2.error("forced watershed failure")

        monkeypatch.setattr(segmentation_mod.cv2, "watershed", fail_watershed)
        refined = _watershed_edge_snap(mask, guide)
        assert np.array_equal(refined, mask)

    def test_edge_snap_follows_curved_boundary(self):
        pytest.importorskip("cv2")
        y, x = np.ogrid[:128, :128]
        ideal = (((x - 64) ** 2 + (y - 64) ** 2) <= 30**2).astype(np.float32)
        mask = (((x - 64) ** 2 + (y - 64) ** 2) <= 36**2).astype(np.float32)
        guide = frame(h=128, w=128, value=10)
        guide[ideal.astype(bool)] = 240
        refined = _watershed_edge_snap(mask, guide)
        assert np.abs(refined - ideal).mean() < np.abs(mask - ideal).mean() * 0.25

    def test_edge_snap_720p_performance_sanity(self):
        pytest.importorskip("cv2")
        mask = np.zeros((720, 1280), np.float32)
        mask[:, 646:] = 1.0
        guide = frame(h=720, w=1280, value=10)
        guide[:, 640:] = 240
        _watershed_edge_snap(mask, guide)  # allocate/OpenCV warm-up
        samples = []
        for _ in range(11):
            started = time.perf_counter()
            refined = _watershed_edge_snap(mask, guide)
            samples.append(time.perf_counter() - started)
        assert refined.shape == mask.shape
        median = float(np.median(samples))
        assert median < 0.010, f"720p median refinement took {median * 1000:.2f} ms"

    def test_mask_shift_shrinks_and_grows(self):
        pytest.importorskip("cv2")
        mask = np.zeros((40, 40), np.float32)
        mask[10:30, 10:30] = 1.0
        shrunk = MaskRefiner(
            SegmentationConfig(
                mask_shift=-2,
                mask_blur=0,
                edge_refine=False,
                temporal_smoothing=0.0,
            )
        ).refine(mask)
        grown = MaskRefiner(
            SegmentationConfig(
                mask_shift=2,
                mask_blur=0,
                edge_refine=False,
                temporal_smoothing=0.0,
            )
        ).refine(mask)
        assert shrunk.sum() < mask.sum() < grown.sum()


@pytest.mark.parametrize(
    ("use_model_foreground", "light_wrap"),
    [
        pytest.param(False, 0.0, id="plain"),
        pytest.param(True, 0.0, id="model-foreground"),
        pytest.param(False, 0.25, id="light-wrap"),
        pytest.param(True, 0.25, id="model-foreground-and-light-wrap"),
    ],
)
def test_legacy_workspace_is_byte_exact_and_preserves_frame_contracts(
    use_model_foreground,
    light_wrap,
):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0x4D41545445)
    shape = (32, 48, 3)
    foreground = rng.integers(0, 256, shape, dtype=np.uint8)
    backdrop = rng.integers(0, 256, shape, dtype=np.uint8)
    clean_foreground = rng.integers(0, 256, shape, dtype=np.uint8)
    mask = rng.random(shape[:2], dtype=np.float32)
    mask[0] = 0.0
    mask[-1] = 1.0
    inputs_before = tuple(
        value.copy() for value in (foreground, backdrop, clean_foreground, mask)
    )
    edge = clean_foreground if use_model_foreground else None
    reference = composite(
        foreground,
        backdrop,
        mask,
        light_wrap=light_wrap,
        edge_foreground=edge,
        blend_space="srgb_legacy",
    )
    assert np.array_equal(
        reference,
        legacy_composite_reference(
            foreground,
            backdrop,
            mask,
            light_wrap=light_wrap,
            edge_foreground=edge,
        ),
    )
    workspace = compositor_mod.LegacyCompositorWorkspace(shape)
    diagnostics = {}

    optimized = composite(
        foreground,
        backdrop,
        mask,
        light_wrap=light_wrap,
        edge_foreground=edge,
        blend_space="srgb_legacy",
        workspace=workspace,
        diagnostics=diagnostics,
    )

    assert np.array_equal(optimized, reference)
    assert optimized.dtype == np.uint8
    assert optimized.shape == shape
    assert optimized.flags.c_contiguous
    assert int(optimized.min()) >= 0
    assert int(optimized.max()) <= 255
    assert np.array_equal(optimized[0], backdrop[0])
    assert np.array_equal(optimized[-1], foreground[-1])
    for value, before in zip(
        (foreground, backdrop, clean_foreground, mask),
        inputs_before,
        strict=True,
    ):
        assert np.array_equal(value, before)
    assert set(diagnostics) == set(compositor_mod.COMPOSITOR_SUBSTAGE_NAMES)
    assert all(
        isinstance(value, float) and np.isfinite(value) and value >= 0.0
        for value in diagnostics.values()
    )
    if not use_model_foreground:
        assert diagnostics["model_foreground_replacement"] == 0.0
    if light_wrap == 0.0:
        assert diagnostics["backdrop_blur_resize"] == 0.0
        assert diagnostics["light_wrap_interpolation"] == 0.0


def test_legacy_workspace_outputs_are_deterministic_and_independently_owned():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0x0BADC0DE)
    shape = (24, 40, 3)
    foreground = rng.integers(0, 256, shape, dtype=np.uint8)
    backdrop = rng.integers(0, 256, shape, dtype=np.uint8)
    clean_foreground = rng.integers(0, 256, shape, dtype=np.uint8)
    mask = rng.random(shape[:2], dtype=np.float32)
    workspace = compositor_mod.LegacyCompositorWorkspace(shape)

    first = composite(
        foreground,
        backdrop,
        mask,
        light_wrap=0.25,
        edge_foreground=clean_foreground,
        workspace=workspace,
    )
    retained_first = first.copy()
    second = composite(
        foreground,
        backdrop,
        mask,
        light_wrap=0.25,
        edge_foreground=clean_foreground,
        workspace=workspace,
    )

    assert first is not second
    assert not np.shares_memory(first, second)
    assert np.array_equal(first, retained_first)
    assert np.array_equal(second, retained_first)
    second.fill(0)
    assert np.array_equal(first, retained_first)
    assert workspace.snapshot().calls == 2


def test_legacy_workspace_opencv_failure_uses_exact_reference_and_recovers(
    monkeypatch,
):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0xFA11BAC)
    shape = (24, 40, 3)
    foreground = rng.integers(0, 256, shape, dtype=np.uint8)
    backdrop = rng.integers(0, 256, shape, dtype=np.uint8)
    clean_foreground = rng.integers(0, 256, shape, dtype=np.uint8)
    mask = rng.random(shape[:2], dtype=np.float32)
    expected = composite(
        foreground,
        backdrop,
        mask,
        light_wrap=0.25,
        edge_foreground=clean_foreground,
    )
    workspace = compositor_mod.LegacyCompositorWorkspace(shape)
    real_add = compositor_mod.cv2.add
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise compositor_mod.cv2.error("forced compiled legacy failure")
        return real_add(*args, **kwargs)

    monkeypatch.setattr(compositor_mod.cv2, "add", fail_once)
    recovered = composite(
        foreground,
        backdrop,
        mask,
        light_wrap=0.25,
        edge_foreground=clean_foreground,
        workspace=workspace,
    )

    assert calls == 1
    assert workspace.snapshot().calls == 0
    np.testing.assert_array_equal(recovered, expected)

    subsequent = composite(
        foreground,
        backdrop,
        mask,
        light_wrap=0.25,
        edge_foreground=clean_foreground,
        workspace=workspace,
    )
    np.testing.assert_array_equal(subsequent, expected)
    assert workspace.snapshot().calls == 1


def test_legacy_workspace_does_not_swallow_structural_failure(monkeypatch):
    shape = (8, 12, 3)
    workspace = compositor_mod.LegacyCompositorWorkspace(shape)

    def fail_contract(*_args, **_kwargs):
        raise ValueError("forced strict workspace contract")

    monkeypatch.setattr(workspace, "blend", fail_contract)
    with pytest.raises(ValueError, match="strict workspace contract"):
        composite(
            frame(h=shape[0], w=shape[1], value=200),
            frame(h=shape[0], w=shape[1], value=10),
            np.full(shape[:2], 0.5, np.float32),
            workspace=workspace,
        )


def test_legacy_workspace_reports_bounded_retained_and_transient_bytes():
    shape = (24, 40, 3)
    height, width, _channels = shape
    workspace = compositor_mod.LegacyCompositorWorkspace(shape)
    initial = workspace.snapshot()
    expected_retained = (
        2 * height * width * np.dtype(np.float32).itemsize
        + 2 * height * width * 3 * np.dtype(np.float32).itemsize
        + max(4, height // 8) * max(4, width // 8) * 3
    )

    assert initial.retained_bytes == expected_retained
    assert initial.last_known_allocation_bytes == 0
    assert initial.calls == 0
    assert initial.closed is False

    output = composite(
        frame(h=height, w=width, value=200),
        frame(h=height, w=width, value=10),
        np.full((height, width), 0.5, np.float32),
        workspace=workspace,
    )
    used = workspace.snapshot()

    assert used.retained_bytes == expected_retained
    assert used.last_known_allocation_bytes == output.nbytes
    assert used.calls == 1
    assert used.closed is False


def test_legacy_workspace_rejects_shape_mismatch_and_use_after_close():
    pytest.importorskip("cv2")
    workspace = compositor_mod.LegacyCompositorWorkspace((24, 40, 3))
    other_foreground = frame(h=20, w=40, value=200)
    other_backdrop = frame(h=20, w=40, value=10)
    other_mask = np.full((20, 40), 0.5, np.float32)

    with pytest.raises(ValueError, match="workspace shape mismatch"):
        composite(
            other_foreground,
            other_backdrop,
            other_mask,
            workspace=workspace,
        )

    workspace.close()
    workspace.close()
    closed = workspace.snapshot()
    assert closed.closed is True
    assert closed.retained_bytes == 0
    assert closed.last_known_allocation_bytes == 0

    with pytest.raises(RuntimeError, match="workspace is closed"):
        composite(
            frame(h=24, w=40, value=200),
            frame(h=24, w=40, value=10),
            np.full((24, 40), 0.5, np.float32),
            workspace=workspace,
        )
