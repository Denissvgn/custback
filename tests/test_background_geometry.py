"""Provider-level regressions for canonical backdrop geometry and orientation."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from PIL import Image

import custback.backgrounds as backgrounds_mod
from custback.backgrounds import (
    BlurBackdrop,
    CameraBackdrop,
    ColorBackdrop,
    ImageBackdrop,
    VideoBackdrop,
)
from custback.geometry import FrameValidationError, Rect

cv2 = pytest.importorskip("cv2")


def _pattern(height: int = 4, width: int = 6, *, offset: int = 0) -> np.ndarray:
    """Return a lossless, coordinate-labeled BGR source."""

    yy, xx = np.indices((height, width), dtype=np.uint16)
    return np.stack(
        (
            (17 * xx + 3 * yy + offset) % 256,
            (5 * xx + 29 * yy + offset) % 256,
            (31 * xx + 11 * yy + offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _save_png(path, bgr: np.ndarray, *, orientation: int = 1) -> None:
    exif = Image.Exif()
    exif[274] = orientation
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", exif=exif)


def _expected_exif(frame: np.ndarray, orientation: int) -> np.ndarray:
    """Independent TIFF-orientation oracle frozen by ADR 0001."""

    operations = {
        1: lambda value: value,
        2: lambda value: value[:, ::-1],
        3: lambda value: value[::-1, ::-1],
        4: lambda value: value[::-1, :],
        5: lambda value: value.transpose(1, 0, 2),
        6: lambda value: np.rot90(value, 3),
        7: lambda value: value.transpose(1, 0, 2)[::-1, ::-1],
        8: lambda value: np.rot90(value, 1),
    }
    return np.ascontiguousarray(operations[orientation](frame))


class _SequenceCapture:
    """Small OpenCV capture double with controllable orientation properties."""

    def __init__(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float = 1.0,
        metadata_rotation: float = 0.0,
        auto_rotation: float = 1.0,
        auto_set_accepted: bool = True,
        auto_readback_stuck: bool = False,
        backend_name: str = "FFMPEG",
    ) -> None:
        self.frames = list(frames)
        self.fps = fps
        self.metadata_rotation = metadata_rotation
        self.auto_rotation = auto_rotation
        self.auto_set_accepted = auto_set_accepted
        self.auto_readback_stuck = auto_readback_stuck
        self.backend_name = backend_name
        self.pos = 0
        self.last_index: int | None = None
        self.grabbed: np.ndarray | None = None
        self.read_calls = 0
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def getBackendName(self) -> str:
        return self.backend_name

    def release(self) -> None:
        self.released = True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frames[0].shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frames[0].shape[0])
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return float("nan")
        if prop == cv2.CAP_PROP_ORIENTATION_META:
            return self.metadata_rotation
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
            return self.auto_rotation
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
            if not self.auto_set_accepted:
                return False
            if not self.auto_readback_stuck:
                self.auto_rotation = value
            return True
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = max(0, int(value))
            return True
        if prop == cv2.CAP_PROP_POS_MSEC:
            return False
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_calls += 1
        if self.pos >= len(self.frames):
            return False, None
        index = self.pos
        self.pos += 1
        self.last_index = index
        return True, self.frames[index].copy()

    def grab(self) -> bool:
        ok, frame = self.read()
        self.grabbed = frame if ok else None
        return ok

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        if self.grabbed is None:
            return False, None
        return True, self.grabbed.copy()


class _CV2CaptureHarness:
    """Only the OpenCV surface exercised by backdrop capture providers."""

    def __init__(
        self,
        captures: Sequence[_SequenceCapture],
        *,
        orientation_controls: bool = True,
    ) -> None:
        self.captures = list(captures)
        self.CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
        self.CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        self.CAP_PROP_FPS = cv2.CAP_PROP_FPS
        self.CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        self.CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
        self.CAP_PROP_POS_MSEC = cv2.CAP_PROP_POS_MSEC
        if orientation_controls:
            self.CAP_PROP_ORIENTATION_META = cv2.CAP_PROP_ORIENTATION_META
            self.CAP_PROP_ORIENTATION_AUTO = cv2.CAP_PROP_ORIENTATION_AUTO

    def VideoCapture(self, _source) -> _SequenceCapture:
        assert self.captures, "unexpected VideoCapture open"
        return self.captures.pop(0)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_image_backdrop_uses_authoritative_pillow_exif_decode_once(
    tmp_path,
    monkeypatch,
    orientation,
) -> None:
    source = _pattern(3, 5)
    path = tmp_path / f"orientation-{orientation}.png"
    _save_png(path, source, orientation=orientation)
    monkeypatch.setattr(
        backgrounds_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: pytest.fail(
            "ImageBackdrop must not independently decode pixels with OpenCV"
        ),
    )

    backdrop = ImageBackdrop(str(path))
    expected = _expected_exif(source, orientation)
    output = backdrop.frame(expected.shape[1], expected.shape[0])

    assert np.array_equal(output, expected)
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous


def test_image_backdrop_rejects_malformed_embedded_icc(tmp_path) -> None:
    source = _pattern(8, 10)
    path = tmp_path / "malformed-profile.png"
    Image.fromarray(
        np.ascontiguousarray(source[:, :, ::-1]),
        mode="RGB",
    ).save(path, format="PNG", icc_profile=b"not-an-icc-profile")

    with pytest.raises(ValueError, match="invalid background image"):
        ImageBackdrop(str(path))


def test_video_qualified_metadata_rotation_is_applied_exactly_once(
    monkeypatch,
) -> None:
    source = _pattern(2, 3)
    capture = _SequenceCapture([source], metadata_rotation=90)
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture]),
    )

    backdrop = VideoBackdrop("qualified.mp4", clock=lambda: 0.0)
    try:
        output = backdrop.frame(2, 3)
        stats = backdrop.stats_dict()
    finally:
        backdrop.close()

    assert np.array_equal(output, np.ascontiguousarray(np.rot90(source, 3)))
    assert stats["background_video_orientation_status"] == ("qualified-manual-metadata")
    assert stats["background_video_metadata_rotation"] == 90
    assert stats["background_video_auto_rotation_disabled"] is True
    assert (cv2.CAP_PROP_ORIENTATION_AUTO, 0.0) in capture.set_calls


@pytest.mark.parametrize(
    ("accepted", "readback_stuck"),
    ((False, False), (True, True)),
)
def test_video_ignored_or_unverifiable_auto_rotation_is_ambiguous_and_not_reapplied(
    monkeypatch,
    accepted,
    readback_stuck,
) -> None:
    source = _pattern(2, 3)
    capture = _SequenceCapture(
        [source],
        metadata_rotation=90,
        auto_set_accepted=accepted,
        auto_readback_stuck=readback_stuck,
    )
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture]),
    )

    backdrop = VideoBackdrop("opaque.mp4", clock=lambda: 0.0)
    try:
        output = backdrop.frame(3, 2)
        stats = backdrop.stats_dict()
    finally:
        backdrop.close()

    assert np.array_equal(output, source)
    assert stats["background_video_orientation_status"] == (
        "manual-only-ambiguous-auto"
    )
    assert stats["background_video_metadata_rotation"] == 90
    assert stats["background_video_auto_rotation_disabled"] is False


def test_video_missing_orientation_controls_uses_observable_manual_only_policy(
    monkeypatch,
) -> None:
    source = _pattern(2, 3)
    capture = _SequenceCapture([source], metadata_rotation=90)
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture], orientation_controls=False),
    )

    backdrop = VideoBackdrop("unsupported.mp4", clock=lambda: 0.0)
    try:
        assert np.array_equal(backdrop.frame(3, 2), source)
        stats = backdrop.stats_dict()
    finally:
        backdrop.close()

    assert stats["background_video_orientation_status"] == ("manual-only-unsupported")
    assert stats["background_video_metadata_rotation"] is None
    assert stats["background_video_auto_rotation_disabled"] is False
    assert all(
        prop != cv2.CAP_PROP_ORIENTATION_AUTO for prop, _value in capture.set_calls
    )


def test_video_unsupported_backend_cannot_fake_orientation_qualification(
    monkeypatch,
) -> None:
    source = _pattern(2, 3)
    capture = _SequenceCapture(
        [source],
        metadata_rotation=90,
        auto_rotation=0,
        backend_name="GSTREAMER",
    )
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture]),
    )

    backdrop = VideoBackdrop("unsupported-backend.mp4", clock=lambda: 0.0)
    try:
        assert np.array_equal(backdrop.frame(3, 2), source)
        stats = backdrop.stats_dict()
    finally:
        backdrop.close()

    assert stats["background_video_orientation_status"] == (
        "manual-only-unsupported-backend"
    )
    assert stats["background_video_metadata_rotation"] is None
    assert stats["background_video_auto_rotation_disabled"] is False
    assert all(
        prop != cv2.CAP_PROP_ORIENTATION_AUTO for prop, _value in capture.set_calls
    )


def test_video_invalid_rotation_metadata_is_not_guessed(monkeypatch) -> None:
    source = _pattern(2, 3)
    capture = _SequenceCapture([source], metadata_rotation=45)
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture]),
    )

    backdrop = VideoBackdrop("invalid-metadata.mp4", clock=lambda: 0.0)
    try:
        assert np.array_equal(backdrop.frame(3, 2), source)
        stats = backdrop.stats_dict()
    finally:
        backdrop.close()

    assert stats["background_video_orientation_status"] == (
        "manual-only-invalid-metadata"
    )
    assert stats["background_video_metadata_rotation"] is None
    assert stats["background_video_auto_rotation_disabled"] is True


def test_video_diagnostics_and_exception_tracebacks_are_asset_path_free(
    monkeypatch,
    caplog,
) -> None:
    private_path = "/private/customer/acme-token-9381/background.mp4"
    good = _pattern()
    malformed = np.zeros((4, 6), dtype=np.uint8)
    capture = _SequenceCapture(
        [good, malformed],  # type: ignore[list-item]
        fps=0.0,
        backend_name="GSTREAMER",
    )
    now = [0.0]
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture]),
    )
    caplog.set_level(logging.WARNING, logger=backgrounds_mod.__name__)

    backdrop = VideoBackdrop(private_path, clock=lambda: now[0])
    try:
        backdrop.frame(4, 4)
        now[0] = 1.0
        backdrop.frame(4, 4)
    finally:
        backdrop.close()

    messages = caplog.text
    assert "video orientation is ambiguous" in messages
    assert "implausible FPS metadata" in messages
    assert "invalid or oversized background video frame" in messages
    assert "cannot decode timed background video frame" in messages
    assert private_path not in messages
    assert "acme-token-9381" not in messages

    def fail_open(*_args, **_kwargs):
        raise FileNotFoundError(private_path)

    monkeypatch.setattr(backgrounds_mod, "_open_video_capture", fail_open)
    with pytest.raises(FileNotFoundError) as raised:
        VideoBackdrop(private_path)
    rendered = "".join(traceback.format_exception(raised.type, raised.value, raised.tb))
    assert private_path not in rendered
    assert "acme-token-9381" not in rendered


def test_image_video_and_camera_backdrops_share_exact_geometry(
    tmp_path,
    monkeypatch,
) -> None:
    source = _pattern()
    path = tmp_path / "equivalent.png"
    _save_png(path, source)
    video_capture = _SequenceCapture([source])
    camera_capture = _SequenceCapture([source])
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([video_capture, camera_capture]),
    )

    image = ImageBackdrop(str(path), anchor_x=1.0)
    video = VideoBackdrop(
        "equivalent.mp4",
        clock=lambda: 0.0,
        anchor_x=1.0,
    )
    camera = CameraBackdrop(7, anchor_x=1.0)
    try:
        image_output = image.frame(4, 4)
        video_output = video.frame(4, 4)
        camera_output = camera.frame(4, 4)
    finally:
        video.close()
        camera.close()

    expected = np.ascontiguousarray(source[:, 2:6])
    assert np.array_equal(image_output, expected)
    assert np.array_equal(video_output, expected)
    assert np.array_equal(camera_output, expected)


def test_image_geometry_change_invalidates_only_fitted_pixels(
    tmp_path,
) -> None:
    source = _pattern()
    path = tmp_path / "anchored.png"
    _save_png(path, source)
    backdrop = ImageBackdrop(str(path), anchor_x=0.0)

    left = backdrop.frame(4, 4)
    assert backdrop.frame(4, 4) is left
    backdrop.set_geometry("cover", 1.0, 0.5)
    right = backdrop.frame(4, 4)

    assert right is not left
    assert np.array_equal(left, source[:, :4])
    assert np.array_equal(right, source[:, 2:6])


def test_contain_content_rect_excludes_padding_and_tracks_cached_plan(tmp_path) -> None:
    source = _pattern(4, 6)
    path = tmp_path / "contained.png"
    _save_png(path, source)
    backdrop = ImageBackdrop(str(path), fit_mode="contain")

    with pytest.raises(RuntimeError, match="content rectangle is unavailable"):
        backdrop.content_rect(8, 8)

    output = backdrop.frame(8, 8)
    expected = Rect(0, 1, 8, 6)
    assert backdrop.content_rect(8, 8) == expected
    assert backdrop.frame(8, 8) is output
    assert backdrop.content_rect(8, 8) == expected
    assert not output[: expected.top].any()
    assert not output[expected.bottom :].any()

    with pytest.raises(RuntimeError, match="content rectangle is unavailable"):
        backdrop.content_rect(9, 8)
    backdrop.set_geometry("cover", 0.5, 0.5)
    with pytest.raises(RuntimeError, match="content rectangle is unavailable"):
        backdrop.content_rect(8, 8)
    backdrop.frame(8, 8)
    assert backdrop.content_rect(8, 8) == Rect(0, 0, 8, 8)


def test_intrinsically_full_backdrops_report_the_complete_canvas() -> None:
    assert ColorBackdrop((1, 2, 3)).content_rect(9, 7) == Rect(0, 0, 9, 7)

    blur = BlurBackdrop(strength=3)
    blur.set_source_frame(_pattern(7, 9))
    assert blur.frame(9, 7).shape == (7, 9, 3)
    assert blur.content_rect(9, 7) == Rect(0, 0, 9, 7)


def test_video_geometry_change_invalidates_fit_without_advancing_source(
    monkeypatch,
) -> None:
    source = _pattern()
    capture = _SequenceCapture([source])
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture]),
    )
    backdrop = VideoBackdrop(
        "anchored.mp4",
        clock=lambda: 0.0,
        anchor_x=0.0,
    )
    try:
        left = backdrop.frame(4, 4)
        logical_index = backdrop._logical_index
        reads = capture.read_calls
        backdrop.set_geometry("cover", 1.0, 0.5)
        right = backdrop.frame(4, 4)
    finally:
        backdrop.close()

    assert np.array_equal(left, source[:, :4])
    assert np.array_equal(right, source[:, 2:6])
    assert backdrop._logical_index == logical_index
    assert capture.read_calls == reads


def test_video_valid_resolution_change_replans_without_reusing_stale_fit(
    monkeypatch,
) -> None:
    landscape = _pattern(4, 6)
    portrait = _pattern(6, 4, offset=83)
    capture = _SequenceCapture([landscape, portrait], fps=1.0)
    now = [0.0]
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture]),
    )
    backdrop = VideoBackdrop(
        "resolution-change.mp4",
        clock=lambda: now[0],
        anchor_x=0.0,
        anchor_y=0.0,
        max_width=8,
        max_height=8,
    )
    try:
        first = backdrop.frame(4, 4).copy()
        first_plan = backdrop.transform_plan(4, 4)
        now[0] = 1.01
        second = backdrop.frame(4, 4).copy()
        second_plan = backdrop.transform_plan(4, 4)
    finally:
        backdrop.close()

    assert np.array_equal(first, landscape[:, :4])
    assert np.array_equal(second, portrait[:4, :])
    assert first_plan.source_size == (6, 4)
    assert second_plan.source_size == (4, 6)
    assert first_plan != second_plan
    assert not backdrop._fatal_decode_error
    assert second.flags.c_contiguous


def test_video_malformed_frame_is_rejected_and_last_good_frame_is_retained(
    monkeypatch,
) -> None:
    good = _pattern()
    malformed = np.zeros((4, 6), dtype=np.uint8)
    capture = _SequenceCapture([good, malformed], fps=1.0)  # type: ignore[list-item]
    now = [0.0]
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture]),
    )
    backdrop = VideoBackdrop("malformed.mp4", clock=lambda: now[0])
    try:
        last_good = backdrop.frame(4, 4).copy()
        now[0] = 1.01
        recovered = backdrop.frame(4, 4).copy()
        stats = backdrop.stats_dict()
    finally:
        backdrop.close()

    assert np.array_equal(recovered, last_good)
    assert cast(int, stats["background_video_decode_failures"]) >= 1
    assert not backdrop._fatal_decode_error


@pytest.mark.parametrize(
    "malformed",
    (
        np.zeros((4, 6), dtype=np.uint8),
        np.zeros((4, 6, 4), dtype=np.uint8),
        np.zeros((4, 6, 3), dtype=np.float32),
    ),
)
def test_camera_backdrop_rejects_malformed_frames_at_source(
    monkeypatch,
    malformed,
) -> None:
    capture = _SequenceCapture([malformed])  # type: ignore[list-item]
    monkeypatch.setattr(
        backgrounds_mod,
        "cv2",
        _CV2CaptureHarness([capture]),
    )
    backdrop = CameraBackdrop(7)
    try:
        with pytest.raises(FrameValidationError):
            backdrop.frame(4, 4)
    finally:
        backdrop.close()


def test_blur_backdrop_never_runs_a_second_geometry_fit(monkeypatch) -> None:
    source = _pattern()
    backdrop = BlurBackdrop(strength=3)
    backdrop.set_source_frame(source)
    monkeypatch.setattr(
        backdrop,
        "_fit_frame",
        lambda *_args, **_kwargs: pytest.fail(
            "an already-canonical blur backdrop must not be fitted again"
        ),
    )

    output = backdrop.frame(6, 4)

    assert output.shape == source.shape
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous


def test_blur_backdrop_rejects_canvas_mismatch_instead_of_resizing() -> None:
    backdrop = BlurBackdrop(strength=3)
    backdrop.set_source_frame(_pattern())

    with pytest.raises(
        ValueError,
        match="must already match the canonical canvas",
    ):
        backdrop.frame(4, 4)


def test_blur_backdrop_gaussian_remains_encoded_space() -> None:
    source = np.zeros((9, 11, 3), dtype=np.uint8)
    source[:, 5:] = (40, 120, 220)
    expected = cv2.GaussianBlur(source.astype(np.float32), (3, 3), 0)

    backdrop = BlurBackdrop(strength=3)
    backdrop.set_source_frame(source)
    output = backdrop.frame(11, 9)

    assert np.array_equal(output, np.clip(expected, 0, 255).astype(np.uint8))
