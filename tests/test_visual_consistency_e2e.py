"""VIS-4.1 integrated geometry, color, privacy, and sink regressions.

The focused geometry/color modules prove the algorithms in isolation.  These
tests deliberately enter through :meth:`Pipeline._loop` so the same generated
frame crosses capture validation, segmentation, backdrop fitting, temporal
color state, privacy guards, hub publication, and output sinks.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
from PIL import Image, ImageCms

cv2 = pytest.importorskip("cv2")

import custback.capture as capture_mod
import custback.pipeline as pipeline_mod
import custback.preview as preview_mod
from custback.api.server import _encode_jpeg
from custback.backgrounds import ImageBackdrop
from custback.capture import CapturedFrame, CaptureHealth, OpenCVCapture
from custback.color import ColorBehavior, ColorReason
from custback.config import AppConfig, CameraConfig, RuntimeConfig
from custback.geometry import (
    FitMode,
    Rect,
    RightAngleRotation,
    TransformPlan,
    transform_frame,
)
from custback.hub import FrameHub
from custback.matte_diagnostics import MatteDiagnosticRecorder, MatteReplayBundle
from custback.pipeline import Pipeline, _Resources
from custback.vcam import PyVirtualCamOutput
from custback.vcam_native import (
    NativeVirtualCameraOutput,
    read_latest_frame,
    ring_size,
)


Canvas = tuple[int, int]


def _pattern(width: int, height: int, *, offset: int = 0) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint32)
    return np.stack(
        (
            (13 * x + 3 * y + offset) % 251,
            (5 * x + 17 * y + offset) % 251,
            (29 * x + 7 * y + offset) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)


def _camera_circle() -> np.ndarray:
    height, width = 480, 640
    y, x = np.ogrid[:height, :width]
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[(x - width // 2) ** 2 + (y - height // 2) ** 2 <= 72**2] = 255
    # Asymmetric non-white markers make rotation/mirror mistakes observable
    # without contaminating the white-circle proportionality measurement.
    source[8:28, 12:42] = (220, 10, 20)
    source[-31:-11, -47:-17] = (10, 220, 20)
    return source


class _GeometryCapture:
    def __init__(
        self,
        source: np.ndarray,
        canvas: Canvas,
        *,
        rotation: RightAngleRotation,
        mirror: bool,
    ) -> None:
        self.source = source
        self.frame, self.plan = transform_frame(
            source,
            canvas,
            fit="cover",
            rotation=rotation,
            mirror=mirror,
            anchors=(0.5, 0.5),
        )
        self._available = True
        self.frames_read = 0
        self.captured_at_ns: int | None = None

    def read(self) -> CapturedFrame | None:
        if not self._available:
            return None
        self._available = False
        self.frames_read = 1
        self.captured_at_ns = int(time.monotonic() * 1_000_000_000)
        content = self.plan.content_rect
        return CapturedFrame(
            pixels=self.frame,
            sequence=self.frames_read,
            captured_at_ns=self.captured_at_ns,
            generation=1,
            geometry_generation=1,
            content_rect=(
                content.left,
                content.top,
                content.right,
                content.bottom,
            ),
        )

    def health_snapshot(self) -> CaptureHealth:
        content = self.plan.content_rect
        oriented_width, oriented_height = self.plan.oriented_size
        return CaptureHealth(
            sequence=self.frames_read,
            captured_monotonic_ns=self.captured_at_ns,
            generation=1,
            geometry_generation=1,
            content_rect=(
                content.left,
                content.top,
                content.right,
                content.bottom,
            ),
            backend="vis-4.1-generated",
            width=self.source.shape[1],
            height=self.source.shape[0],
            delivered_width=self.source.shape[1],
            delivered_height=self.source.shape[0],
            oriented_width=oriented_width,
            oriented_height=oriented_height,
            normalized_width=self.frame.shape[1],
            normalized_height=self.frame.shape[0],
            geometry_transitions=1,
            fps_reported=30.0,
            frames_read=self.frames_read,
        )

    def close(self) -> None:
        pass


class _SequenceCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        generation: int = 1,
    ) -> None:
        self.frames = list(frames)
        assert self.frames
        self.width = self.frames[0].shape[1]
        self.height = self.frames[0].shape[0]
        self.generation = generation
        self.frames_read = 0
        self.captured_at_ns: int | None = None

    def read(self) -> CapturedFrame | None:
        if not self.frames:
            return None
        self.frames_read += 1
        self.captured_at_ns = int(time.monotonic() * 1_000_000_000)
        return CapturedFrame(
            pixels=self.frames.pop(0),
            sequence=self.frames_read,
            captured_at_ns=self.captured_at_ns,
            generation=self.generation,
            geometry_generation=self.generation,
            content_rect=(0, 0, self.width, self.height),
        )

    def health_snapshot(self) -> CaptureHealth:
        return CaptureHealth(
            sequence=self.frames_read,
            captured_monotonic_ns=self.captured_at_ns,
            generation=self.generation,
            geometry_generation=self.generation,
            content_rect=(0, 0, self.width, self.height),
            backend="vis-4.1-sequence",
            width=self.width,
            height=self.height,
            delivered_width=self.width,
            delivered_height=self.height,
            oriented_width=self.width,
            oriented_height=self.height,
            normalized_width=self.width,
            normalized_height=self.height,
            geometry_transitions=self.generation,
            fps_reported=30.0,
            frames_read=self.frames_read,
        )

    def close(self) -> None:
        pass


class _MaskSegmenter:
    device = "cpu"
    produces_matte = False
    last_foreground = None

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = np.ascontiguousarray(mask, dtype=np.float32)
        self.frames: list[np.ndarray] = []

    def segment(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return self.mask.copy()

    def close(self) -> None:
        pass


class _IdentityRefiner:
    def refine(self, mask: np.ndarray, _frame: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(mask, dtype=np.float32)

    def close(self) -> None:
        pass


class _GeometryBackdrop:
    def __init__(
        self,
        source: np.ndarray,
        *,
        fit: FitMode,
        anchors: tuple[float, float],
    ) -> None:
        self.source = source
        self.fit: FitMode = fit
        self.anchors = anchors
        self.plan: TransformPlan | None = None
        self.calls = 0

    def frame(self, width: int, height: int) -> np.ndarray:
        self.calls += 1
        output, self.plan = transform_frame(
            self.source,
            (width, height),
            fit=self.fit,
            anchors=self.anchors,
        )
        return output

    def transform_plan(self, width: int, height: int) -> TransformPlan:
        assert self.plan is not None
        assert self.plan.target_size == (width, height)
        return self.plan

    def content_rect(self, width: int, height: int) -> Rect:
        return self.transform_plan(width, height).content_rect

    def close(self) -> None:
        pass


class _CountingImageBackdrop(ImageBackdrop):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.fit_calls = 0
        super().__init__(*args, **kwargs)

    def _fit_frame(
        self,
        frame: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        self.fit_calls += 1
        return super()._fit_frame(frame, width, height)

    @property
    def calls(self) -> int:
        return self.fit_calls

    @property
    def plan(self) -> TransformPlan | None:
        return self._last_transform_plan


class _StopOutput:
    paces = True
    fallback_active = False
    fallback_reason = ""

    def __init__(self, pipeline: Pipeline, count: int, canvas: Canvas) -> None:
        self.pipeline = pipeline
        self.count = count
        self.width, self.height = canvas
        self.fps = 30
        self.frames: list[np.ndarray] = []

    def send(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())
        if len(self.frames) >= self.count:
            self.pipeline._stop.set()

    def close(self) -> None:
        pass


def _background_config(
    provider: Literal["image", "video", "camera"],
    *,
    fit: str,
    anchors: tuple[float, float],
) -> dict[str, object]:
    value: dict[str, object] = {
        "mode": provider,
        "fit_mode": fit,
        "anchor_x": anchors[0],
        "anchor_y": anchors[1],
    }
    if provider == "image":
        value["image_path"] = "/generated/vis-4.1.png"
    elif provider == "video":
        value["video_path"] = "/generated/vis-4.1.mp4"
    else:
        value["camera_device"] = 7
    return value


def _pipeline_config(
    canvas: Canvas,
    *,
    background: dict[str, object],
    correction: str = "off",
) -> AppConfig:
    return AppConfig.from_dict(
        {
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30,
                "fit_mode": "cover",
            },
            "background": background,
            "segmentation": {
                "backend": "heuristic",
                "temporal_smoothing": 0.0,
                "edge_refine": False,
                "mask_blur": 0,
            },
            "compositing": {
                "blend_space": "linear_srgb",
                "light_wrap": 0.0,
                "color_correction": {"mode": correction},
            },
            "output": {
                "backend": "null",
                "width": canvas[0],
                "height": canvas[1],
                "fps": 30,
            },
            "api": {"enabled": False},
        }
    )


@dataclass(frozen=True)
class _GeometryCase:
    provider: Literal["image", "video", "camera"]
    source_size: Canvas
    fit: Literal["cover", "contain", "stretch"]
    anchors: tuple[float, float]
    rotation: Literal[0, 90, 180, 270]
    mirror: bool
    canvas: Canvas


@pytest.mark.parametrize(
    "case",
    (
        pytest.param(
            _GeometryCase(
                "image",
                (201, 201),
                "cover",
                (0.0, 0.0),
                0,
                False,
                (1280, 720),
            ),
            id="square-icc-image-cover-anchor-start-720p",
        ),
        pytest.param(
            _GeometryCase(
                "video",
                (181, 319),
                "contain",
                (1.0, 1.0),
                90,
                True,
                (1920, 1080),
            ),
            id="odd-portrait-video-contain-anchor-end-1080p",
        ),
        pytest.param(
            _GeometryCase(
                "image",
                (181, 319),
                "cover",
                (1.0, 0.0),
                180,
                True,
                (1280, 720),
            ),
            id="odd-portrait-icc-image-cover-focal-anchor-720p",
        ),
        pytest.param(
            _GeometryCase(
                "camera",
                (319, 179),
                "stretch",
                (0.0, 1.0),
                270,
                True,
                (1280, 720),
            ),
            id="odd-sixteen-nine-live-stretch-mixed-anchor-720p",
        ),
    ),
)
def test_geometry_provider_matrix_crosses_the_complete_frame_lane(
    case: _GeometryCase,
    tmp_path,
) -> None:
    camera = _GeometryCapture(
        _camera_circle(),
        case.canvas,
        rotation=case.rotation,
        mirror=case.mirror,
    )
    source = _pattern(*case.source_size, offset=37)
    if case.provider == "image":
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        path = tmp_path / "portrait-profile.png"
        Image.fromarray(source[..., ::-1], mode="RGB").save(
            path,
            format="PNG",
            icc_profile=profile,
        )
        backdrop: Any = _CountingImageBackdrop(
            str(path),
            fit_mode=case.fit,
            anchor_x=case.anchors[0],
            anchor_y=case.anchors[1],
        )
    else:
        backdrop = _GeometryBackdrop(
            source,
            fit=case.fit,
            anchors=case.anchors,
        )

    cfg = _pipeline_config(
        case.canvas,
        background=_background_config(
            case.provider,
            fit=case.fit,
            anchors=case.anchors,
        ),
    )
    hub = FrameHub()
    hub.configure_canvas(case.canvas)
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    output = _StopOutput(pipeline, 1, case.canvas)
    mask = np.zeros((case.canvas[1], case.canvas[0]), dtype=np.float32)
    segmenter = _MaskSegmenter(mask)
    resources = _Resources(
        cfg,
        0,
        camera,
        segmenter,
        _IdentityRefiner(),
        backdrop,
        output,
    )
    try:
        pipeline._loop(resources)

        raw = hub.raw.latest()[0]
        published = hub.output.latest()[0]
        assert raw is not None and published is not None
        assert raw.shape == published.shape == (case.canvas[1], case.canvas[0], 3)
        assert np.array_equal(raw, camera.frame)
        assert segmenter.frames == [raw]
        assert len(output.frames) == 1
        assert np.array_equal(output.frames[0], published)
        assert backdrop.calls == 1
        assert backdrop.plan is not None
        assert backdrop.plan.fit == case.fit
        assert backdrop.plan.target_size == case.canvas

        expected, expected_plan = transform_frame(
            source,
            case.canvas,
            fit=case.fit,
            anchors=case.anchors,
        )
        np.testing.assert_array_equal(published, expected)
        assert backdrop.plan.crop_rect == expected_plan.crop_rect
        assert backdrop.plan.content_rect == expected_plan.content_rect

        # Both 4:3 camera normalization and proportional backdrop modes plan
        # one resize per source. Stretch remains an explicit legacy mode.
        assert len(camera.plan.resize_steps) == 1
        assert camera.plan.scale_x == pytest.approx(camera.plan.scale_y, rel=0.002)
        if case.fit != "stretch":
            assert len(backdrop.plan.resize_steps) == 1
            assert backdrop.plan.scale_x == pytest.approx(
                backdrop.plan.scale_y,
                rel=0.003,
            )

        white = np.all(raw >= 240, axis=2)
        ys, xs = np.where(white)
        circle_width = int(xs.max() - xs.min() + 1)
        circle_height = int(ys.max() - ys.min() + 1)
        assert circle_width / circle_height == pytest.approx(1.0, rel=0.015)
    finally:
        resources.close()


class _CanvasBackdrop:
    def __init__(self, pixels: np.ndarray) -> None:
        self.pixels = pixels

    def frame(self, width: int, height: int) -> np.ndarray:
        assert self.pixels.shape == (height, width, 3)
        return self.pixels

    def close(self) -> None:
        pass


class _SequenceBackdrop:
    def __init__(self, frames: list[np.ndarray]) -> None:
        assert frames
        self.frames = list(frames)
        self.last = frames[-1]
        self.calls = 0

    def frame(self, width: int, height: int) -> np.ndarray:
        self.calls += 1
        value = self.frames.pop(0) if self.frames else self.last
        assert value.shape == (height, width, 3)
        return value

    def close(self) -> None:
        pass


class _IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


@pytest.mark.parametrize(
    ("background_mode", "foreground_center", "backdrop_center", "expected"),
    (
        pytest.param(
            "image",
            (80, 80, 80),
            (150, 150, 150),
            "positive-ev",
            id="image-positive-ev",
        ),
        pytest.param(
            "video",
            (150, 150, 150),
            (80, 80, 80),
            "negative-ev",
            id="video-negative-ev",
        ),
        pytest.param(
            "camera",
            (95, 100, 105),
            (105, 100, 95),
            "warm-to-cool",
            id="live-camera-warm-to-cool",
        ),
        pytest.param(
            "image",
            (105, 100, 95),
            (95, 100, 105),
            "cool-to-warm",
            id="image-cool-to-warm",
        ),
    ),
)
def test_real_color_pairs_cross_pipeline_without_changing_raw(
    monkeypatch: pytest.MonkeyPatch,
    background_mode: Literal["image", "video", "camera"],
    foreground_center: tuple[int, int, int],
    backdrop_center: tuple[int, int, int],
    expected: str,
) -> None:
    width, height = 160, 120
    y, x = np.indices((height, width), dtype=np.int16)
    variation = ((x + y) % 31) - 15
    foreground = np.clip(
        np.asarray(foreground_center, dtype=np.int16)[None, None, :]
        + variation[..., None],
        1,
        250,
    ).astype(np.uint8)
    backdrop_pixels = np.clip(
        np.asarray(backdrop_center, dtype=np.int16)[None, None, :]
        + variation[..., None],
        1,
        250,
    ).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[20:100, 40:120] = 1.0
    cfg = _pipeline_config(
        (width, height),
        background=_background_config(
            background_mode,
            fit="cover",
            anchors=(0.5, 0.5),
        ),
        correction="auto",
    )
    hub = FrameHub()
    hub.configure_canvas((width, height))
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    output = _StopOutput(pipeline, 2, (width, height))
    resources = _Resources(
        cfg,
        0,
        _SequenceCapture([foreground.copy(), foreground.copy()]),
        _MaskSegmenter(mask),
        _IdentityRefiner(),
        _CanvasBackdrop(backdrop_pixels),
        output,
    )
    # Exercise the real time-based harmonizer deterministically while the
    # complete pipeline still owns timestamps and publication.
    monkeypatch.setattr(pipeline_mod.time, "monotonic", _IncrementingClock())
    try:
        pipeline._loop(resources)

        raw = hub.raw.latest()[0]
        rendered = hub.output.latest()[0]
        assert raw is not None and rendered is not None
        np.testing.assert_array_equal(raw, foreground)
        np.testing.assert_array_equal(
            rendered[mask == 0.0], backdrop_pixels[mask == 0.0]
        )
        assert not np.array_equal(rendered[mask == 1.0], foreground[mask == 1.0])
        raw_jpeg = cv2.imdecode(
            np.frombuffer(_encode_jpeg(raw), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert raw_jpeg is not None
        assert (
            float(
                np.mean(np.abs(raw_jpeg.astype(np.int16) - foreground.astype(np.int16)))
            )
            < 3.0
        )
        assert not np.array_equal(raw_jpeg[mask == 1.0], rendered[mask == 1.0])

        assert resources.harmonizer is not None
        snapshot = resources.harmonizer.snapshot()
        if expected == "positive-ev":
            assert snapshot.transform.exposure_ev > 0.0
        elif expected == "negative-ev":
            assert snapshot.transform.exposure_ev < 0.0
        elif expected == "warm-to-cool":
            red, _green, blue = snapshot.transform.wb_gains
            assert red < 1.0 < blue
        else:
            red, _green, blue = snapshot.transform.wb_gains
            assert blue < 1.0 < red

        stats = hub.stats_dict()
        assert stats["color_correction_active"] is True
        assert stats["color_correction_applied_frames"] >= 1
        assert stats["color_correction_confidence"] > 0.0
    finally:
        resources.close()


def _color_edge_sequence(
    case: Literal["clipped", "saturated", "low-confidence", "noisy"],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    height, width = 120, 160
    y, x = np.indices((height, width), dtype=np.int16)
    texture = ((7 * x + 11 * y) % 17) - 8
    foreground = np.clip(
        np.asarray((90, 92, 94), dtype=np.int16)[None, None, :] + texture[..., None],
        1,
        250,
    ).astype(np.uint8)
    backdrop = np.clip(
        np.asarray((158, 150, 142), dtype=np.int16)[None, None, :] + texture[..., None],
        1,
        250,
    ).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[20:100, 40:120] = 1.0

    if case == "clipped":
        foreground[:] = 255
    elif case == "saturated":
        backdrop[:] = (188, 26, 233)
    elif case == "low-confidence":
        mask[:] = 0.0
        mask[58:62, 78:82] = 1.0
    else:
        rng = np.random.default_rng(4101)
        foregrounds = []
        backdrops = []
        for _ in range(36):
            foreground_noise = rng.normal(0.0, 3.0, foreground.shape)
            backdrop_noise = rng.normal(0.0, 3.0, backdrop.shape)
            noisy_foreground = np.clip(
                foreground.astype(np.float32) + foreground_noise,
                1,
                250,
            ).astype(np.uint8)
            noisy_backdrop = np.clip(
                backdrop.astype(np.float32) + backdrop_noise,
                1,
                250,
            ).astype(np.uint8)
            # A fixed observed core separates temporal correction movement from
            # sensor noise while the estimator still sees the noisy full scene.
            noisy_foreground[58:63, 78:83] = (90, 92, 94)
            foregrounds.append(noisy_foreground)
            backdrops.append(noisy_backdrop)
        return foregrounds, backdrops, mask

    return [foreground.copy(), foreground.copy()], [backdrop.copy(), backdrop], mask


@pytest.mark.parametrize(
    ("case", "expected_reason", "expected_behavior"),
    (
        pytest.param(
            "clipped",
            ColorReason.CLIPPED,
            ColorBehavior.IDENTITY,
            id="clipped-identity",
        ),
        pytest.param(
            "saturated",
            ColorReason.SOLID_SATURATED,
            ColorBehavior.EXPOSURE_ONLY,
            id="saturated-exposure-only",
        ),
        pytest.param(
            "low-confidence",
            ColorReason.INSUFFICIENT_MASK,
            ColorBehavior.IDENTITY,
            id="low-confidence-identity",
        ),
        pytest.param(
            "noisy",
            ColorReason.OK,
            ColorBehavior.EXPOSURE_WHITE_BALANCE,
            id="static-noise-bounded",
        ),
    ),
)
def test_real_color_edge_pixels_fail_safe_or_remain_bounded_in_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    case: Literal["clipped", "saturated", "low-confidence", "noisy"],
    expected_reason: ColorReason,
    expected_behavior: ColorBehavior,
) -> None:
    foregrounds, backdrops, mask = _color_edge_sequence(case)
    height, width = mask.shape
    cfg = _pipeline_config(
        (width, height),
        background=_background_config(
            "image",
            fit="cover",
            anchors=(0.5, 0.5),
        ),
        correction="auto",
    )
    hub = FrameHub()
    hub.configure_canvas((width, height))
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    output = _StopOutput(pipeline, len(foregrounds), (width, height))
    backdrop = _SequenceBackdrop(backdrops)
    resources = _Resources(
        cfg,
        0,
        _SequenceCapture(foregrounds),
        _MaskSegmenter(mask),
        _IdentityRefiner(),
        backdrop,
        output,
    )
    estimates = []
    production_estimator = pipeline_mod.estimate_color_transform_linear

    def observe_estimate(*args: Any, **kwargs: Any):
        estimate = production_estimator(*args, **kwargs)
        estimates.append(estimate)
        return estimate

    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        observe_estimate,
    )
    monkeypatch.setattr(pipeline_mod.time, "monotonic", _IncrementingClock())
    try:
        pipeline._loop(resources)

        assert len(estimates) == len(output.frames) == len(foregrounds)
        assert all(estimate.reason is expected_reason for estimate in estimates)
        assert all(estimate.behavior is expected_behavior for estimate in estimates)
        np.testing.assert_array_equal(hub.raw.latest()[0], foregrounds[-1])
        np.testing.assert_array_equal(
            output.frames[-1][mask == 0.0],
            backdrops[-1][mask == 0.0],
        )

        assert resources.harmonizer is not None
        snapshot = resources.harmonizer.snapshot()
        stats = hub.stats_dict()
        assert snapshot.reason is expected_reason
        assert stats["color_correction_reason"] == expected_reason.value
        assert abs(float(stats["color_correction_exposure_ev"])) <= (
            cfg.compositing.color_correction.exposure_limit_ev
            * cfg.compositing.color_correction.strength
        )
        gains = (
            float(stats["color_correction_wb_gain_r"]),
            float(stats["color_correction_wb_gain_g"]),
            float(stats["color_correction_wb_gain_b"]),
        )
        assert all(0.86 <= gain <= 1.16 for gain in gains)

        if expected_behavior is ColorBehavior.IDENTITY:
            assert snapshot.transform.is_identity
            assert stats["color_correction_active"] is False
            assert stats["color_correction_effective_mode"] == "identity"
            assert stats["color_correction_applied_frames"] == 0
            np.testing.assert_array_equal(
                output.frames[-1][mask == 1.0],
                foregrounds[-1][mask == 1.0],
            )
        else:
            assert snapshot.reliable
            assert stats["color_correction_active"] is True
            assert stats["color_correction_applied_frames"] >= 1
            assert not np.array_equal(
                output.frames[-1][mask == 1.0],
                foregrounds[-1][mask == 1.0],
            )
            if expected_behavior is ColorBehavior.EXPOSURE_ONLY:
                assert stats["color_correction_effective_mode"] == "exposure"
                assert stats["color_correction_wb_active"] is False
            else:
                assert stats["color_correction_effective_mode"] == (
                    "exposure-white-balance"
                )
                assert stats["color_correction_wb_active"] is True
                center = np.asarray(
                    [frame[60, 80] for frame in output.frames],
                    dtype=np.int16,
                )
                # Once acquisition settles, real pixel noise cannot introduce
                # visible correction oscillation at the fixed observed core.
                assert int(np.max(np.abs(np.diff(center[-10:], axis=0)))) <= 1
    finally:
        resources.close()


@pytest.mark.parametrize("background_mode", ("blur", "color", "image"))
def test_disabled_correction_is_byte_identity_across_local_modes(
    monkeypatch: pytest.MonkeyPatch,
    background_mode: Literal["blur", "color", "image"],
) -> None:
    width, height = 160, 120
    foreground = _pattern(width, height, offset=83)
    backdrop = _pattern(width, height, offset=197)
    background: dict[str, object]
    if background_mode == "blur":
        background = {"mode": "blur", "blur_strength": 3}
    elif background_mode == "color":
        background = {"mode": "color", "color": [12, 34, 56]}
    else:
        background = _background_config(
            "image",
            fit="cover",
            anchors=(0.5, 0.5),
        )
    cfg = _pipeline_config(
        (width, height),
        background=background,
        correction="off",
    )
    hub = FrameHub()
    hub.configure_canvas((width, height))
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    output = _StopOutput(pipeline, 1, (width, height))
    resources = _Resources(
        cfg,
        0,
        _SequenceCapture([foreground]),
        _MaskSegmenter(np.ones((height, width), dtype=np.float32)),
        _IdentityRefiner(),
        _CanvasBackdrop(backdrop),
        output,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "estimate_color_transform_linear",
        lambda *_args, **_kwargs: pytest.fail(
            "disabled correction must not decode or analyze pixels"
        ),
    )
    try:
        pipeline._loop(resources)

        assert len(output.frames) == 1
        np.testing.assert_array_equal(output.frames[0], foreground)
        np.testing.assert_array_equal(hub.output.latest()[0], foreground)
        stats = hub.stats_dict()
        assert stats["color_correction_mode"] == "off"
        assert stats["color_correction_state"] == "disabled"
        assert stats["color_correction_reason"] == "disabled"
        assert stats["color_correction_effective_mode"] == "off"
        assert stats["color_correction_active"] is False
        assert stats["color_correction_applied_frames"] == 0
        assert stats["color_correction_bypassed_frames"] == 1
    finally:
        resources.close()


def _run_video_color_sequence(
    monkeypatch: pytest.MonkeyPatch,
    backdrop_frames: list[np.ndarray],
) -> tuple[_Resources, _StopOutput, FrameHub, _SequenceBackdrop]:
    height, width = backdrop_frames[0].shape[:2]
    foreground = np.full((height, width, 3), 90, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[20 : height - 20, 30 : width - 30] = 1.0
    cfg = _pipeline_config(
        (width, height),
        background=_background_config(
            "video",
            fit="cover",
            anchors=(0.5, 0.5),
        ),
        correction="auto",
    )
    hub = FrameHub()
    hub.configure_canvas((width, height))
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    output = _StopOutput(pipeline, len(backdrop_frames), (width, height))
    backdrop = _SequenceBackdrop(backdrop_frames)
    resources = _Resources(
        cfg,
        0,
        _SequenceCapture([foreground.copy() for _ in backdrop_frames]),
        _MaskSegmenter(mask),
        _IdentityRefiner(),
        backdrop,
        output,
    )
    monkeypatch.setattr(pipeline_mod.time, "monotonic", _IncrementingClock())
    pipeline._loop(resources)
    return resources, output, hub, backdrop


def test_slow_video_color_drift_is_bounded_in_the_complete_frame_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 120, 160
    levels = np.linspace(90, 150, num=60)
    backdrops = [
        np.full((height, width, 3), int(round(level)), dtype=np.uint8)
        for level in levels
    ]
    resources, output, hub, backdrop = _run_video_color_sequence(
        monkeypatch,
        backdrops,
    )
    try:
        assert len(output.frames) == backdrop.calls == len(backdrops)
        core_values = np.asarray(
            [int(frame[height // 2, width // 2, 0]) for frame in output.frames]
        )
        steps = np.diff(core_values)
        assert int(steps.min()) >= 0
        assert int(steps.max()) <= 4
        assert int(core_values[-1]) > int(core_values[0])
        assert resources.harmonizer is not None
        assert 0.0 < resources.harmonizer.snapshot().transform.exposure_ev <= 0.425
        assert hub.stats_dict()["color_correction_scene_cuts"] == 0
    finally:
        resources.close()


def test_video_hard_cut_holds_cut_frame_then_fast_acquires_without_phase_rewind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 120, 160
    before = np.full((height, width, 3), 150, dtype=np.uint8)
    after = np.full((height, width, 3), 45, dtype=np.uint8)
    backdrops = [before.copy() for _ in range(5)] + [after.copy() for _ in range(5)]
    resources, output, hub, backdrop = _run_video_color_sequence(
        monkeypatch,
        backdrops,
    )
    try:
        assert len(output.frames) == backdrop.calls == len(backdrops)
        center = (height // 2, width // 2)
        # The first hard-cut estimate is deliberately held; the video source
        # still advances exactly once while foreground output cannot flash.
        np.testing.assert_array_equal(
            output.frames[5][center], output.frames[4][center]
        )
        assert not np.array_equal(output.frames[-1][center], output.frames[5][center])
        assert resources.color_correction_scene_cuts == 1
        assert hub.stats_dict()["color_correction_scene_cuts"] == 1
        assert resources.harmonizer is not None
        transform = resources.harmonizer.snapshot().transform
        assert abs(transform.exposure_ev) <= 0.85
        assert all(0.86 <= gain <= 1.16 for gain in transform.wb_gains)
    finally:
        resources.close()


class _FakeVirtualCamera:
    instances: list["_FakeVirtualCamera"] = []

    def __init__(self, *, width: int, height: int, fps: int, **_kwargs: Any) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.device = "VIS-4.1 fake pyvirtualcam"
        self.frames: list[np.ndarray] = []
        self.closed = False
        type(self).instances.append(self)

    def send(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def sleep_until_next_frame(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _QualificationHighGUI:
    """Record the physical-window handoff without requiring a display."""

    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 4
    error = type("error", (Exception,), {})

    def __init__(self) -> None:
        self.shown: list[np.ndarray] = []

    def namedWindow(self, *_args: object) -> None:
        pass

    def imshow(self, _title: str, frame: np.ndarray) -> None:
        self.shown.append(frame.copy())

    def waitKey(self, _delay_ms: int) -> int:
        return ord("q")

    def getWindowProperty(self, *_args: object) -> float:
        return 1.0

    def destroyWindow(self, _title: str) -> None:
        pass


class _SinkFanout:
    paces = True
    fallback_active = False
    fallback_reason = ""

    def __init__(
        self,
        pipeline: Pipeline,
        pyvirtualcam: PyVirtualCamOutput,
        native: NativeVirtualCameraOutput,
        *,
        stop_after: int,
    ) -> None:
        self.pipeline = pipeline
        self.pyvirtualcam = pyvirtualcam
        self.native = native
        self.stop_after = stop_after
        self.width = native.width
        self.height = native.height
        self.fps = native.fps
        self.frames: list[np.ndarray] = []

    def send(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())
        self.pyvirtualcam.send(frame)
        self.native.send(frame)
        if len(self.frames) >= self.stop_after:
            self.pipeline._stop.set()

    def close(self) -> None:
        self.pyvirtualcam.close()
        self.native.close()


_CAPTURE_RELEASED = object()


class _ReconnectCaptureHandle:
    """Minimal blocking OpenCV handle for one integrated reconnect lane."""

    def __init__(
        self,
        *,
        property_size: Canvas,
        frames: tuple[np.ndarray, ...] = (),
        fail_when_empty: bool,
    ) -> None:
        self.values = {
            _ReconnectCV2.CAP_PROP_FRAME_WIDTH: float(property_size[0]),
            _ReconnectCV2.CAP_PROP_FRAME_HEIGHT: float(property_size[1]),
            _ReconnectCV2.CAP_PROP_FPS: 30.0,
            _ReconnectCV2.CAP_PROP_FOURCC: float(
                sum(ord(char) << (8 * index) for index, char in enumerate("YUYV"))
            ),
            _ReconnectCV2.CAP_PROP_BACKEND: 999.0,
            _ReconnectCV2.CAP_PROP_ORIENTATION_META: 0.0,
            _ReconnectCV2.CAP_PROP_ORIENTATION_AUTO: 0.0,
        }
        self.frames: queue.Queue[object] = queue.Queue()
        for frame in frames:
            self.frames.put(frame)
        self.fail_when_empty = fail_when_empty
        self.released = False

    def isOpened(self) -> bool:
        return True

    def getBackendName(self) -> str:
        return "V4L2"

    def set(self, prop: int, value: float) -> bool:
        self.values[prop] = float(value)
        return True

    def get(self, prop: int) -> float:
        return self.values.get(prop, 0.0)

    def push(self, frame: np.ndarray) -> None:
        self.frames.put(frame)

    def read(self) -> tuple[bool, object | None]:
        while not self.released:
            try:
                value = self.frames.get(timeout=0.002 if self.fail_when_empty else 0.01)
            except queue.Empty:
                if self.fail_when_empty:
                    return False, None
                continue
            if value is _CAPTURE_RELEASED:
                return False, None
            assert isinstance(value, np.ndarray)
            return True, value.copy()
        return False, None

    def release(self) -> None:
        self.released = True
        self.frames.put(_CAPTURE_RELEASED)


class _ReconnectCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6
    CAP_PROP_BACKEND = 42
    CAP_PROP_ORIENTATION_META = 48
    CAP_PROP_ORIENTATION_AUTO = 49
    CAP_PROP_AUTO_WB = 100
    CAP_PROP_WB_TEMPERATURE = 101
    CAP_PROP_AUTO_EXPOSURE = 102
    CAP_PROP_EXPOSURE = 103
    CAP_PROP_GAIN = 104
    CAP_PROP_GAMMA = 105
    CAP_V4L2 = 200

    def __init__(self, handles: list[_ReconnectCaptureHandle]) -> None:
        self.handles = list(handles)

    def VideoCapture(
        self,
        _device: int | str,
        _api_preference: int | None = None,
    ) -> _ReconnectCaptureHandle:
        del _api_preference
        if not self.handles:
            raise AssertionError("capture unexpectedly reopened")
        return self.handles.pop(0)

    @staticmethod
    def VideoWriter_fourcc(*chars: str) -> int:
        return sum(
            ord(char) << (8 * index) for index, char in enumerate("".join(chars))
        )


class _ReconnectSinkFanout:
    """Fan out every paced frame but stop only after two distinct sources."""

    paces = True
    fallback_active = False
    fallback_reason = ""

    def __init__(
        self,
        pipeline: Pipeline,
        pyvirtualcam: PyVirtualCamOutput,
        native: NativeVirtualCameraOutput,
        *,
        second_handle: _ReconnectCaptureHandle,
        second_source: np.ndarray,
    ) -> None:
        self.pipeline = pipeline
        self.pyvirtualcam = pyvirtualcam
        self.native = native
        self.second_handle = second_handle
        self.second_source = second_source
        self.width = native.width
        self.height = native.height
        self.fps = native.fps
        self.unique_frames: list[np.ndarray] = []

    def send(self, frame: np.ndarray) -> None:
        self.pyvirtualcam.send(frame)
        self.native.send(frame)
        marker = int(frame[0, 0, 0])
        if self.unique_frames and int(self.unique_frames[-1][0, 0, 0]) == marker:
            return
        self.unique_frames.append(frame.copy())
        if len(self.unique_frames) == 1:
            self.second_handle.push(self.second_source)
        elif len(self.unique_frames) == 2:
            self.pipeline._stop.set()

    def close(self) -> None:
        self.pyvirtualcam.close()
        self.native.close()


def test_opencv_reconnect_size_change_crosses_pipeline_and_both_vcam_sinks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canvas = (40, 30)
    first_source = np.full((24, 32, 3), 11, dtype=np.uint8)
    second_source = np.full((32, 16, 3), 99, dtype=np.uint8)
    first_handle = _ReconnectCaptureHandle(
        property_size=(32, 24),
        frames=(first_source,),
        fail_when_empty=True,
    )
    second_handle = _ReconnectCaptureHandle(
        property_size=(32, 24),
        fail_when_empty=False,
    )
    monkeypatch.setattr(
        capture_mod,
        "cv2",
        _ReconnectCV2([first_handle, second_handle]),
    )
    capture = OpenCVCapture(
        CameraConfig(
            width=32,
            height=24,
            fps=30,
            fit_mode="cover",
            mode_mismatch="warn",
        ),
        canvas,
    )
    capture._stall_after_s = 0.03
    capture._recovery_timeout_s = 0.5
    capture._backoffs = (0.005,)

    cfg = _pipeline_config(canvas, background={"mode": "passthrough"})
    hub = FrameHub()
    hub.configure_canvas(canvas)
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    _FakeVirtualCamera.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "pyvirtualcam",
        SimpleNamespace(
            Camera=_FakeVirtualCamera,
            PixelFormat=SimpleNamespace(BGR=object()),
        ),
    )
    pyvirtualcam = PyVirtualCamOutput(cfg.output, *canvas)
    ring = bytearray(ring_size(*canvas))
    native = NativeVirtualCameraOutput(
        *canvas,
        fps=cfg.output.fps,
        buffer=ring,
    )
    output = _ReconnectSinkFanout(
        pipeline,
        pyvirtualcam,
        native,
        second_handle=second_handle,
        second_source=second_source,
    )
    resources = _Resources(
        cfg,
        0,
        capture,
        _MaskSegmenter(np.zeros((canvas[1], canvas[0]), dtype=np.float32)),
        _IdentityRefiner(),
        None,
        output,
    )
    try:
        deadline = time.monotonic() + 2.0
        pipeline._loop(resources)
        assert time.monotonic() < deadline

        assert [int(frame[0, 0, 0]) for frame in output.unique_frames] == [11, 99]
        assert all(
            frame.shape == (canvas[1], canvas[0], 3) for frame in output.unique_frames
        )
        health = capture.health_snapshot()
        assert health.generation == health.geometry_generation == 2
        assert health.geometry_transitions == 2
        assert (health.delivered_width, health.delivered_height) == (16, 32)
        assert (health.normalized_width, health.normalized_height) == canvas
        assert health.restarts == 1
        assert first_handle.released

        raw = hub.raw.latest()[0]
        preview = hub.output.latest()[0]
        assert raw is not None and preview is not None
        np.testing.assert_array_equal(raw, output.unique_frames[-1])
        np.testing.assert_array_equal(preview, raw)
        np.testing.assert_array_equal(
            _FakeVirtualCamera.instances[-1].frames[-1],
            raw,
        )
        native_bgrx = read_latest_frame(ring)
        assert native_bgrx is not None
        np.testing.assert_array_equal(native_bgrx[..., :3], raw)
    finally:
        resources.close()


def test_four_three_camera_is_proportional_and_equal_at_every_local_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width, height = 1280, 720
    canvas = (width, height)
    cfg = _pipeline_config(
        canvas,
        background={"mode": "passthrough"},
    )
    hub = FrameHub()
    hub.configure_canvas(canvas)
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    capture = _GeometryCapture(
        _camera_circle(),
        canvas,
        rotation=0,
        mirror=False,
    )

    _FakeVirtualCamera.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "pyvirtualcam",
        SimpleNamespace(
            Camera=_FakeVirtualCamera,
            PixelFormat=SimpleNamespace(BGR=object()),
        ),
    )
    pyvirtualcam = PyVirtualCamOutput(cfg.output, width, height)
    ring = bytearray(ring_size(width, height))
    native = NativeVirtualCameraOutput(
        width,
        height,
        fps=cfg.output.fps,
        buffer=ring,
    )
    output = _SinkFanout(
        pipeline,
        pyvirtualcam,
        native,
        stop_after=1,
    )
    resources = _Resources(
        cfg,
        0,
        capture,
        _MaskSegmenter(np.zeros((height, width), dtype=np.float32)),
        _IdentityRefiner(),
        None,
        output,
    )
    try:
        pipeline._loop(resources)

        raw = hub.raw.latest()[0]
        preview = hub.output.latest()[0]
        assert raw is not None and preview is not None
        np.testing.assert_array_equal(preview, raw)
        np.testing.assert_array_equal(output.frames[0], raw)
        np.testing.assert_array_equal(
            _FakeVirtualCamera.instances[-1].frames[0],
            raw,
        )
        native_bgrx = read_latest_frame(ring)
        assert native_bgrx is not None
        np.testing.assert_array_equal(native_bgrx[..., :3], raw)

        jpeg = _encode_jpeg(preview)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape == raw.shape
        assert (
            float(np.mean(np.abs(decoded.astype(np.int16) - raw.astype(np.int16))))
            < 4.0
        )

        crop = capture.plan.crop_rect
        assert (crop.left, crop.top, crop.right, crop.bottom) == (
            0,
            120,
            1280,
            840,
        )
        white = np.all(raw >= 240, axis=2)
        ys, xs = np.where(white)
        assert (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) == (
            pytest.approx(1.0, rel=0.015)
        )
    finally:
        resources.close()


def test_one_matte_generation_is_identical_at_replay_preview_and_sink_seams(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind MATTE-5.2 comparisons to one production composite generation."""

    width, height = 96, 54
    canvas = (width, height)
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_ramp = x_grid / float(width - 1)
    y_ramp = y_grid / float(height - 1)
    source = np.stack(
        (
            35.0 + 90.0 * x_ramp,
            55.0 + 100.0 * y_ramp,
            80.0 + 70.0 * (0.6 * x_ramp + 0.4 * y_ramp),
        ),
        axis=-1,
    ).astype(np.uint8)
    backdrop = np.stack(
        (
            155.0 + 45.0 * y_ramp,
            105.0 + 55.0 * x_ramp,
            45.0 + 65.0 * (1.0 - y_ramp),
        ),
        axis=-1,
    ).astype(np.uint8)
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt(
        ((x - width / 2.0) / 25.0) ** 2 + ((y - height / 2.0) / 19.0) ** 2
    )
    alpha = np.clip(1.25 - distance, 0.0, 1.0).astype(np.float32)
    cfg = _pipeline_config(
        canvas,
        background={
            "mode": "image",
            "image_path": "/generated/matte-5.2-background.png",
        },
    )
    hub = FrameHub()
    hub.configure_canvas(canvas)
    bundle_root = tmp_path / "same-generation-bundle"
    recorder = MatteDiagnosticRecorder(bundle_root, max_bytes=8_000_000)
    pipeline = Pipeline(RuntimeConfig(cfg), hub, matte_recorder=recorder)

    _FakeVirtualCamera.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "pyvirtualcam",
        SimpleNamespace(
            Camera=_FakeVirtualCamera,
            PixelFormat=SimpleNamespace(BGR=object()),
        ),
    )
    pyvirtualcam = PyVirtualCamOutput(cfg.output, width, height)
    ring = bytearray(ring_size(width, height))
    native = NativeVirtualCameraOutput(
        width,
        height,
        fps=cfg.output.fps,
        buffer=ring,
    )
    output = _SinkFanout(pipeline, pyvirtualcam, native, stop_after=1)
    resources = _Resources(
        cfg,
        0,
        _SequenceCapture([source], generation=17),
        _MaskSegmenter(alpha),
        _IdentityRefiner(),
        _CanvasBackdrop(backdrop),
        output,
    )
    try:
        pipeline._loop(resources)
        recorder.close()

        bundle = MatteReplayBundle(bundle_root)
        assert len(bundle.frames) == 1
        recorded = bundle.frames[0]
        assert recorded["capture_sequence"] == 1
        assert recorded["capture_generation"] == 17
        assert recorded["geometry_generation"] == 17
        base = bundle.load_array(recorded, "base_composite")
        final = bundle.load_array(recorded, "final_composite")
        np.testing.assert_array_equal(final, base)
        assert all(
            event["post_base_final_output_provenance"] is None
            for event in bundle.output_events
        )

        published = hub.output.latest()[0]
        assert published is not None
        np.testing.assert_array_equal(published, final)
        np.testing.assert_array_equal(output.frames[0], final)
        np.testing.assert_array_equal(
            _FakeVirtualCamera.instances[-1].frames[0],
            final,
        )
        native_bgrx = read_latest_frame(ring)
        assert native_bgrx is not None
        np.testing.assert_array_equal(native_bgrx[..., :3], final)

        pre_overlay: list[np.ndarray] = []

        def capture_pre_overlay(frame, *_args, **_kwargs):
            pre_overlay.append(frame.copy())
            shown = frame.copy()
            shown[0, 0] = (255, 0, 255)
            return shown

        fake_highgui = _QualificationHighGUI()
        monkeypatch.setattr(preview_mod, "cv2", fake_highgui)
        monkeypatch.setattr(preview_mod, "_draw_overlay", capture_pre_overlay)
        assert preview_mod.run_preview(
            RuntimeConfig(cfg),
            hub,
            threading.Event(),
        )
        np.testing.assert_array_equal(pre_overlay[0], final)
        assert not np.array_equal(fake_highgui.shown[0], final)

        jpeg = _encode_jpeg(published)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape == final.shape
        assert (
            float(np.mean(np.abs(decoded.astype(np.int16) - final.astype(np.int16))))
            < 4.0
        )
    finally:
        recorder.close()
        resources.close()


@pytest.mark.parametrize(
    "candidate_kind",
    ("wrong-size", "current-raw-echo", "delayed-raw-echo"),
)
def test_remote_privacy_slate_is_identical_at_preview_mjpeg_and_vcam_sinks(
    monkeypatch: pytest.MonkeyPatch,
    candidate_kind: str,
) -> None:
    width, height = 128, 72
    first = _pattern(width, height, offset=11)
    second = _pattern(width, height, offset=137)
    raw_frames = [first] if candidate_kind != "delayed-raw-echo" else [first, second]
    if candidate_kind == "wrong-size":
        candidate = _pattern(64, 48, offset=79)
        expected_reason = "wrong-size"
    elif candidate_kind == "current-raw-echo":
        candidate = first.copy()
        expected_reason = "privacy-raw-echo"
    else:
        candidate = first.copy()
        expected_reason = "privacy-delayed-raw-echo"

    cfg = _pipeline_config(
        (width, height),
        background={
            "mode": "remote",
            "remote_fallback_mode": "color",
            "color": [3, 5, 7],
        },
        correction="auto",
    )
    hub = FrameHub()
    hub.configure_canvas((width, height))
    pipeline = Pipeline(RuntimeConfig(cfg), hub)

    _FakeVirtualCamera.instances.clear()
    fake_module = SimpleNamespace(
        Camera=_FakeVirtualCamera,
        PixelFormat=SimpleNamespace(BGR=object()),
    )
    monkeypatch.setitem(sys.modules, "pyvirtualcam", fake_module)
    pyvirtualcam = PyVirtualCamOutput(cfg.output, width, height)
    ring = bytearray(ring_size(width, height))
    native = NativeVirtualCameraOutput(width, height, fps=cfg.output.fps, buffer=ring)
    output = _SinkFanout(
        pipeline,
        pyvirtualcam,
        native,
        stop_after=len(raw_frames),
    )
    resources = _Resources(
        cfg,
        0,
        _SequenceCapture(raw_frames),
        _MaskSegmenter(np.zeros((height, width), dtype=np.float32)),
        _IdentityRefiner(),
        None,
        output,
    )
    session = hub.remote_client_connected()
    try:
        assert hub.push_remote_frame(candidate, session)
        pipeline._loop(resources)

        slate = Pipeline._privacy_slate((height, width, 3))
        preview = hub.output.latest()[0]
        raw = hub.raw.latest()[0]
        assert preview is not None and raw is not None
        np.testing.assert_array_equal(preview, slate)
        np.testing.assert_array_equal(raw, raw_frames[-1])
        assert not np.array_equal(preview, raw)
        assert all(np.array_equal(frame, slate) for frame in output.frames)

        fake_camera = _FakeVirtualCamera.instances[-1]
        assert all(np.array_equal(frame, slate) for frame in fake_camera.frames)
        native_bgrx = read_latest_frame(ring)
        assert native_bgrx is not None
        np.testing.assert_array_equal(native_bgrx[..., :3], slate)

        jpeg = _encode_jpeg(preview)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == slate.shape
        # MJPEG is intentionally lossy, so equality is asserted as the same
        # guarded image within the production encoder's bounded error.
        error = np.abs(decoded.astype(np.int16) - slate.astype(np.int16))
        assert float(np.mean(error)) < 4.0

        stats = hub.stats_dict()
        assert stats["remote_fallback_mode"] == "privacy-slate"
        assert stats["remote_fallback_reason"] == expected_reason
        assert stats["color_correction_applied_frames"] == 0
    finally:
        hub.remote_client_disconnected(session)
        resources.close()
