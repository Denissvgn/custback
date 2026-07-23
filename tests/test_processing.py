import time

import numpy as np
import pytest

import custback.backgrounds as backgrounds_mod
import custback.segmentation as segmentation_mod
from custback.backgrounds import (
    BlurBackdrop,
    ColorBackdrop,
    VideoBackdrop,
    _fit,
    create_backdrop,
)
from custback.compositor import composite
from custback.config import BackgroundConfig, SegmentationConfig
from custback.segmentation import (
    HeuristicSegmenter,
    MaskRefiner,
    NullSegmenter,
    _watershed_edge_snap,
)


def frame(h=72, w=128, value=100):
    return np.full((h, w, 3), value, dtype=np.uint8)


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

    def test_light_wrap_tints_only_the_edge_band(self):
        pytest.importorskip("cv2")
        fg = frame(value=50)
        bg = np.zeros((72, 128, 3), np.uint8)
        bg[:, :, 1] = 200  # green backdrop
        mask = self.edge_mask()
        plain = composite(fg, bg, mask)
        wrapped = composite(fg, bg, mask, light_wrap=0.5)
        # the person's edge picks up backdrop light...
        assert int(wrapped[36, 64, 1]) > int(plain[36, 64, 1])
        # ...but the person core and the pure background are untouched
        assert (wrapped[:, :40] == plain[:, :40]).all()
        assert (wrapped[:, 100:] == plain[:, 100:]).all()

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

    def test_edge_foreground_shape_mismatch_ignored(self):
        fg, bg = frame(value=200), frame(value=0)
        out = composite(
            fg,
            bg,
            self.edge_mask(),
            edge_foreground=np.zeros((10, 10, 3), np.uint8),
        )
        assert tuple(out[36, 10]) == (200, 200, 200)


class TestBackdrops:
    def test_fit_center_crops_to_exact_size(self):
        out = _fit(frame(h=100, w=100), 128, 72)
        assert out.shape == (72, 128, 3)

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

    def test_image_backdrop_fully_decodes_with_pillow_before_opencv(self, monkeypatch):
        pytest.importorskip("cv2")
        open_calls = []

        class FakeImage:
            format = "PNG"
            size = (8, 8)

            def __init__(self, decode):
                self.decode = decode

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def verify(self):
                return None

            def load(self):
                if self.decode:
                    raise OSError("corrupt compressed pixels")

        def fake_open(_path):
            open_calls.append(len(open_calls))
            return FakeImage(decode=len(open_calls) == 2)

        monkeypatch.setattr(backgrounds_mod.Image, "open", fake_open)
        monkeypatch.setattr(
            backgrounds_mod.cv2,
            "imread",
            lambda *_args, **_kwargs: pytest.fail(
                "OpenCV must not see a Pillow decode failure"
            ),
        )

        with pytest.raises(ValueError, match="invalid background image"):
            create_backdrop(BackgroundConfig(mode="image", image_path="corrupt.png"))
        assert len(open_calls) == 2

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
            assert np.array_equal(bd.frame(64, 48), first)  # same instant: hold
            now[0] = 0.5
            second = bd.frame(64, 48).copy()
            now[0] = 1.0
            third = bd.frame(64, 48).copy()
            now[0] = 1.5
            looped = bd.frame(64, 48).copy()
            stats = bd.stats_dict()
        finally:
            bd.close()
        assert second.mean() > first.mean() + 50
        assert third.mean() > second.mean() + 50
        assert np.allclose(looped, first, atol=3)
        assert stats["background_video_frames_displayed"] == 4
        assert stats["background_video_frames_reused"] == 1
        assert stats["background_video_frames_skipped"] == 0

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
            assert not capture.set_calls  # five stale frames: sequential
            assert capture.grab_calls == 4  # decode only the final skipped image
            now[0] = 1.5
            assert int(backdrop.frame(6, 4).mean()) == 15
            assert (cv2.CAP_PROP_POS_FRAMES, 15) in capture.set_calls
            stats = backdrop.stats_dict()
            assert stats["background_video_frames_displayed"] == 3
            assert stats["background_video_frames_skipped"] == 13
            assert stats["background_video_seek_count"] == 1
            assert stats["background_video_skip_ratio"] == pytest.approx(13 / 16)
        finally:
            backdrop.close()

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
