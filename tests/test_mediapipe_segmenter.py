"""Focused MATTE-1.3 tests for MediaPipe time and soft-mask contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import custback.segmentation as segmentation_mod
from custback.capture import CapturedFrame
from custback.config import AppConfig, RuntimeConfig, SegmentationConfig
from custback.hub import FrameHub
from custback.pipeline import Pipeline, _Resources
from custback.segmentation import (
    SEGMENTATION_TIMESTAMP_GAP_RESET_NS,
    MaskRefiner,
    MediaPipeSegmenter,
    MediaPipeTelemetry,
    SegmentationFrameContext,
    TemporalResetReason,
)


class _FakeImage:
    def __init__(self, *, image_format: object, data: np.ndarray) -> None:
        self.image_format = image_format
        self.data = data


class _FakeBaseOptions:
    class Delegate:
        GPU = "gpu"

    def __init__(self, **values: object) -> None:
        self.values = values


class _FakeImageSegmenterOptions:
    def __init__(self, **values: object) -> None:
        self.values = values


class _FakeConfidenceMask:
    def __init__(self, value: object) -> None:
        self.value = value

    def numpy_view(self) -> object:
        return self.value


class _FakeTask:
    def __init__(self, results: Sequence[object]) -> None:
        self._results = list(results)
        self.timestamps_ms: list[int] = []
        self.images: list[_FakeImage] = []
        self.close_calls = 0

    def segment_for_video(
        self,
        image: _FakeImage,
        timestamp_ms: int,
    ) -> SimpleNamespace:
        index = len(self.timestamps_ms)
        if index >= len(self._results):
            raise AssertionError("fake MediaPipe task received an unexpected frame")
        self.timestamps_ms.append(timestamp_ms)
        self.images.append(image)
        result = self._results[index]
        if isinstance(result, SimpleNamespace):
            return result
        return SimpleNamespace(
            confidence_masks=[
                _FakeConfidenceMask(value) for value in cast(Sequence[object], result)
            ]
        )

    def close(self) -> None:
        self.close_calls += 1


class _FakeMediaPipeRuntime:
    def __init__(
        self,
        task_results: Sequence[Sequence[object]],
    ) -> None:
        self._task_results = list(task_results)
        self.tasks: list[_FakeTask] = []
        self.options: list[_FakeImageSegmenterOptions] = []

    def create_from_options(
        self,
        options: _FakeImageSegmenterOptions,
    ) -> _FakeTask:
        index = len(self.tasks)
        if index >= len(self._task_results):
            raise AssertionError("fake MediaPipe runtime created an unexpected task")
        task = _FakeTask(self._task_results[index])
        self.tasks.append(task)
        self.options.append(options)
        return task


def _new_segmenter(
    monkeypatch: pytest.MonkeyPatch,
    task_results: Sequence[Sequence[object]],
) -> tuple[MediaPipeSegmenter, _FakeMediaPipeRuntime]:
    runtime = _FakeMediaPipeRuntime(task_results)
    fake_mp = SimpleNamespace(
        Image=_FakeImage,
        ImageFormat=SimpleNamespace(SRGB="srgb"),
    )
    fake_python = SimpleNamespace(BaseOptions=_FakeBaseOptions)
    fake_vision = SimpleNamespace(
        ImageSegmenterOptions=_FakeImageSegmenterOptions,
        RunningMode=SimpleNamespace(VIDEO="video"),
        ImageSegmenter=SimpleNamespace(
            create_from_options=runtime.create_from_options,
        ),
    )
    modules = {
        "mediapipe": fake_mp,
        "mediapipe.tasks.python": fake_python,
        "mediapipe.tasks.python.vision": fake_vision,
    }
    monkeypatch.setattr(
        segmentation_mod.importlib,
        "import_module",
        lambda name: modules[name],
    )
    segmenter = MediaPipeSegmenter(
        SegmentationConfig(
            backend="mediapipe",
            model_path="/fake/selfie-segmenter.tflite",
            delegate="cpu",
        )
    )
    return segmenter, runtime


def _context(
    sequence: int,
    timestamp_ns: int,
    shape: tuple[int, int],
    *,
    generation: int = 1,
) -> SegmentationFrameContext:
    return SegmentationFrameContext(
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        generation=generation,
        geometry_generation=generation,
        shape=shape,
    )


def _captured(
    sequence: int,
    timestamp_ns: int,
    shape: tuple[int, int],
    *,
    generation: int = 1,
) -> CapturedFrame:
    height, width = shape
    return CapturedFrame(
        pixels=np.zeros((height, width, 3), dtype=np.uint8),
        sequence=sequence,
        captured_at_ns=timestamp_ns,
        generation=generation,
        geometry_generation=generation,
        content_rect=(0, 0, width, height),
    )


@pytest.mark.parametrize(
    ("relative_timestamps_ns", "expected_timestamps_ms"),
    [
        pytest.param(
            (0, 66_666_667, 133_333_334, 200_000_001),
            (0, 66, 133, 200),
            id="15-fps",
        ),
        pytest.param(
            (0, 33_333_333, 66_666_666, 99_999_999),
            (0, 33, 66, 99),
            id="30-fps",
        ),
        pytest.param(
            (0, 16_666_667, 33_333_334, 50_000_001),
            (0, 16, 33, 50),
            id="60-fps",
        ),
        pytest.param(
            (0, 4_500_000, 80_000_000, 101_250_000),
            (0, 4, 80, 101),
            id="irregular",
        ),
    ],
)
def test_capture_time_drives_relative_mediapipe_video_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    relative_timestamps_ns: tuple[int, ...],
    expected_timestamps_ms: tuple[int, ...],
) -> None:
    shape = (3, 4)
    mask = np.full(shape, 0.5, dtype=np.float32)
    segmenter, runtime = _new_segmenter(
        monkeypatch,
        [[[mask] for _ in relative_timestamps_ns]],
    )
    base_timestamp_ns = 9_000_000_000

    for sequence, relative_ns in enumerate(relative_timestamps_ns):
        frame = np.full((*shape, 3), sequence, dtype=np.uint8)
        segmenter.segment(
            frame,
            context=_context(
                sequence,
                base_timestamp_ns + relative_ns,
                shape,
            ),
        )

    assert runtime.tasks[0].timestamps_ms == list(expected_timestamps_ms)
    assert all(
        current > previous
        for previous, current in zip(
            runtime.tasks[0].timestamps_ms,
            runtime.tasks[0].timestamps_ms[1:],
        )
    )
    telemetry = segmenter.telemetry_snapshot()
    assert isinstance(telemetry, MediaPipeTelemetry)
    assert telemetry.input_frame_shape == shape
    assert telemetry.model_mask_shape == shape
    assert telemetry.output_mask_shape == shape
    assert telemetry.effective_timestamp_ms == expected_timestamps_ms[-1]
    assert (
        telemetry.effective_timestamp_delta_ms
        == expected_timestamps_ms[-1] - expected_timestamps_ms[-2]
    )
    assert telemetry.timestamp_adjustment_count == 0
    assert telemetry.last_timestamp_adjusted is False
    assert telemetry.resize_interpolation == "none"


def test_same_millisecond_inputs_receive_minimal_monotonic_bumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 3)
    mask = np.full(shape, 0.5, dtype=np.float32)
    segmenter, runtime = _new_segmenter(
        monkeypatch,
        [[[mask], [mask], [mask], [mask]]],
    )
    base_timestamp_ns = 5_000_000_000

    for sequence, relative_ns in enumerate((0, 100_000, 900_000, 1_100_000)):
        segmenter.segment(
            np.zeros((*shape, 3), dtype=np.uint8),
            context=_context(
                sequence,
                base_timestamp_ns + relative_ns,
                shape,
            ),
        )

    assert runtime.tasks[0].timestamps_ms == [0, 1, 2, 3]
    telemetry = segmenter.telemetry_snapshot()
    assert telemetry.effective_timestamp_ms == 3
    assert telemetry.effective_timestamp_delta_ms == 1
    assert telemetry.timestamp_adjustment_count == 3
    assert telemetry.timestamp_adjustment_ms == 2
    assert telemetry.last_timestamp_adjusted is True
    with pytest.raises(FrozenInstanceError):
        setattr(telemetry, "effective_timestamp_ms", 99)


def test_reset_recreates_used_task_and_starts_a_fresh_timestamp_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 3)
    mask = np.full(shape, 0.5, dtype=np.float32)
    segmenter, runtime = _new_segmenter(
        monkeypatch,
        [
            [[mask]],
            [[mask]],
        ],
    )
    frame = np.zeros((*shape, 3), dtype=np.uint8)

    segmenter.segment(
        frame,
        context=_context(1, 4_000_000_000, shape),
    )
    first_task = runtime.tasks[0]
    assert first_task.timestamps_ms == [0]

    segmenter.reset_temporal_state(
        TemporalResetReason.CAPTURE_GENERATION,
        9_000_000_000,
    )
    assert first_task.close_calls == 1
    assert len(runtime.tasks) == 2

    segmenter.segment(
        frame,
        context=_context(
            2,
            9_000_000_000,
            shape,
            generation=2,
        ),
    )
    assert runtime.tasks[1].timestamps_ms == [0]
    telemetry = segmenter.telemetry_snapshot()
    assert telemetry.effective_timestamp_ms == 0
    assert telemetry.effective_timestamp_delta_ms is None
    assert telemetry.last_timestamp_adjusted is False
    assert segmenter.temporal_reset_count == 1
    assert (
        segmenter.last_temporal_reset_reason is TemporalResetReason.CAPTURE_GENERATION
    )
    assert segmenter.last_temporal_reset_timestamp_ns == 9_000_000_000


@pytest.mark.parametrize(
    (
        "second_sequence",
        "timestamp_delta_ns",
        "second_generation",
        "expected_reason",
        "expected_task_timestamps",
    ),
    [
        pytest.param(
            4,
            100_000_000,
            1,
            None,
            [[0, 100]],
            id="ordinary-sequence-gap",
        ),
        pytest.param(
            2,
            SEGMENTATION_TIMESTAMP_GAP_RESET_NS + 1,
            1,
            TemporalResetReason.TIMESTAMP_GAP,
            [[0], [0]],
            id="long-timestamp-gap",
        ),
        pytest.param(
            2,
            100_000_000,
            2,
            TemporalResetReason.CAPTURE_GENERATION,
            [[0], [0]],
            id="capture-generation",
        ),
        pytest.param(
            2,
            -1,
            1,
            TemporalResetReason.NON_MONOTONIC_TIMESTAMP,
            [[0], [0]],
            id="decreasing-timestamp",
        ),
    ],
)
def test_pipeline_timeline_resets_mediapipe_task_at_exact_capture_boundary(
    monkeypatch: pytest.MonkeyPatch,
    second_sequence: int,
    timestamp_delta_ns: int,
    second_generation: int,
    expected_reason: TemporalResetReason | None,
    expected_task_timestamps: list[list[int]],
) -> None:
    shape = (16, 16)
    mask = np.full(shape, 0.5, dtype=np.float32)
    task_results = (
        [[[mask], [mask]]]
        if expected_reason is None
        else [
            [[mask]],
            [[mask]],
        ]
    )
    segmenter, runtime = _new_segmenter(monkeypatch, task_results)
    cfg = AppConfig.from_dict(
        {
            "camera": {
                "synthetic": True,
                "width": shape[1],
                "height": shape[0],
            },
            "background": {"mode": "color"},
            "segmentation": {
                "backend": "mediapipe",
                "model_path": "/fake/selfie-segmenter.tflite",
                "mask_blur": 0,
                "edge_refine": False,
                "temporal_smoothing": 0.0,
            },
            "output": {"backend": "null"},
            "api": {"enabled": False},
        }
    )
    resources = _Resources(
        cfg,
        0,
        None,
        segmenter,
        MaskRefiner(cfg.segmentation),
        None,
        None,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    base_timestamp_ns = 2_000_000_000
    try:
        pipeline._segment_resource_masks(
            resources,
            _captured(1, base_timestamp_ns, shape),
            privacy_safe=False,
        )
        pipeline._segment_resource_masks(
            resources,
            _captured(
                second_sequence,
                base_timestamp_ns + timestamp_delta_ns,
                shape,
                generation=second_generation,
            ),
            privacy_safe=False,
        )

        assert [task.timestamps_ms for task in runtime.tasks] == (
            expected_task_timestamps
        )
        timeline = resources.segmentation_timeline.snapshot()
        assert timeline.reset_count == (1 if expected_reason is None else 2)
        assert timeline.last_reset_reason is (
            TemporalResetReason.INITIAL if expected_reason is None else expected_reason
        )
        if expected_reason is None:
            assert timeline.sequence_gap_events == 1
            assert timeline.sequence_gap_frames == 2
            assert runtime.tasks[0].close_calls == 0
            telemetry = segmenter.telemetry_snapshot()
            assert telemetry.effective_timestamp_ms == 100
            assert telemetry.effective_timestamp_delta_ms == 100
        else:
            assert timeline.sequence_gap_events == 0
            assert runtime.tasks[0].close_calls == 1
            telemetry = segmenter.telemetry_snapshot()
            assert telemetry.effective_timestamp_ms == 0
            assert telemetry.effective_timestamp_delta_ms is None
    finally:
        resources.close()


def test_failed_mask_validation_does_not_publish_partial_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 3)
    valid = np.full(shape, 0.5, dtype=np.float32)
    invalid = np.full(shape, np.nan, dtype=np.float32)
    segmenter, runtime = _new_segmenter(
        monkeypatch,
        [
            [
                [valid],
                [invalid],
                [valid],
            ]
        ],
    )
    frame = np.zeros((*shape, 3), dtype=np.uint8)
    base_timestamp_ns = 7_000_000_000

    segmenter.segment(
        frame,
        context=_context(1, base_timestamp_ns, shape),
    )
    successful = segmenter.telemetry_snapshot()

    with pytest.raises(ValueError, match="confidence mask"):
        segmenter.segment(
            frame,
            context=_context(2, base_timestamp_ns + 33_000_000, shape),
        )
    assert segmenter.telemetry_snapshot() == successful

    segmenter.segment(
        frame,
        context=_context(3, base_timestamp_ns + 66_000_000, shape),
    )
    assert runtime.tasks[0].timestamps_ms == [0, 33, 66]
    recovered = segmenter.telemetry_snapshot()
    assert recovered.effective_timestamp_ms == 66
    assert recovered.effective_timestamp_delta_ms == 33
    assert recovered.input_frame_shape == shape
    assert recovered.model_mask_shape == shape
    assert recovered.output_mask_shape == shape


@pytest.mark.parametrize(
    "confidence_masks",
    [
        pytest.param([], id="empty-count"),
        pytest.param(
            [
                np.zeros((2, 3), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32),
            ],
            id="multiple-count",
        ),
        pytest.param([object()], id="not-an-array"),
        pytest.param(
            [np.zeros((2, 3), dtype=np.float64)],
            id="wrong-dtype",
        ),
        pytest.param(
            [np.full((2, 3), np.nan, dtype=np.float32)],
            id="nan",
        ),
        pytest.param(
            [np.full((2, 3), np.inf, dtype=np.float32)],
            id="infinite",
        ),
        pytest.param(
            [np.zeros((6,), dtype=np.float32)],
            id="one-dimensional",
        ),
        pytest.param(
            [np.zeros((2, 3, 1), dtype=np.float32)],
            id="three-dimensional",
        ),
        pytest.param(
            [np.zeros((0, 3), dtype=np.float32)],
            id="empty-dimension",
        ),
    ],
)
def test_invalid_confidence_mask_contract_is_rejected_before_resize(
    monkeypatch: pytest.MonkeyPatch,
    confidence_masks: list[object],
) -> None:
    segmenter, _runtime = _new_segmenter(
        monkeypatch,
        [[confidence_masks]],
    )
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="confidence mask"):
        segmenter.segment(
            frame,
            context=_context(1, 1_000_000_000, (2, 3)),
        )


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(
            SimpleNamespace(),
            id="missing-confidence-masks",
        ),
        pytest.param(
            SimpleNamespace(confidence_masks=object()),
            id="non-sequence-confidence-masks",
        ),
    ],
)
def test_missing_or_non_sequence_confidence_masks_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    result: SimpleNamespace,
) -> None:
    segmenter, _runtime = _new_segmenter(
        monkeypatch,
        [[result]],
    )

    with pytest.raises(ValueError, match="confidence mask"):
        segmenter.segment(
            np.zeros((2, 3, 3), dtype=np.uint8),
            context=_context(1, 1_000_000_000, (2, 3)),
        )


@pytest.mark.parametrize(
    ("source_shape", "target_shape", "expected_steps"),
    [
        pytest.param(
            (2, 3),
            (5, 7),
            (((7, 5), "linear"),),
            id="linear-upsample",
        ),
        pytest.param(
            (5, 7),
            (2, 3),
            (((3, 2), "area"),),
            id="area-downsample",
        ),
        pytest.param(
            (2, 8),
            (4, 5),
            (
                ((5, 2), "area"),
                ((5, 4), "linear"),
            ),
            id="mixed-axis-shrink-width-then-grow-height",
        ),
        pytest.param(
            (4, 5),
            (2, 8),
            (
                ((5, 2), "area"),
                ((8, 2), "linear"),
            ),
            id="mixed-axis-shrink-height-then-grow-width",
        ),
        pytest.param(
            (2, 6),
            (3, 4),
            (
                ((4, 2), "area"),
                ((4, 3), "linear"),
            ),
            id="mixed-axis-equal-area-remains-two-step",
        ),
    ],
)
def test_soft_mask_resize_uses_explicit_directional_interpolation(
    monkeypatch: pytest.MonkeyPatch,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    expected_steps: tuple[tuple[tuple[int, int], str], ...],
) -> None:
    source = np.arange(np.prod(source_shape), dtype=np.float32).reshape(source_shape)
    source /= float(source.size - 1)
    real_resize = segmentation_mod.cv2.resize
    interpolation_codes = {
        "area": segmentation_mod.cv2.INTER_AREA,
        "linear": segmentation_mod.cv2.INTER_LINEAR,
    }
    expected = source
    for size, interpolation in expected_steps:
        expected = real_resize(
            expected,
            size,
            interpolation=interpolation_codes[interpolation],
        )
    resize_calls: list[tuple[tuple[int, int], int]] = []

    def recording_resize(
        value: np.ndarray,
        size: tuple[int, int],
        *args: object,
        **kwargs: Any,
    ) -> np.ndarray:
        resize_calls.append((size, kwargs["interpolation"]))
        return real_resize(value, size, *args, **kwargs)

    monkeypatch.setattr(segmentation_mod.cv2, "resize", recording_resize)
    segmenter, _runtime = _new_segmenter(
        monkeypatch,
        [[[source]]],
    )

    result = segmenter.segment(
        np.zeros((*target_shape, 3), dtype=np.uint8),
        context=_context(1, 2_000_000_000, target_shape),
    )

    assert resize_calls == [
        (size, interpolation_codes[interpolation])
        for size, interpolation in expected_steps
    ]
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-7)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0
    telemetry = segmenter.telemetry_snapshot()
    assert telemetry.input_frame_shape == target_shape
    assert telemetry.model_mask_shape == source_shape
    assert telemetry.output_mask_shape == target_shape
    assert telemetry.effective_timestamp_ms == 0
    assert telemetry.effective_timestamp_delta_ms is None
    assert telemetry.timestamp_adjustment_count == 0
    assert telemetry.last_timestamp_adjusted is False
    assert telemetry.resize_interpolation == "+".join(
        interpolation for _size, interpolation in expected_steps
    )


def test_unresized_mask_is_clamped_float32_and_contiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backing = np.asarray(
        [
            [-0.25, 9.0, 0.2, 9.0, 0.8, 9.0],
            [1.25, 9.0, 0.4, 9.0, 0.6, 9.0],
        ],
        dtype=np.float32,
    )
    source = backing[:, ::2]
    assert not source.flags.c_contiguous
    segmenter, _runtime = _new_segmenter(
        monkeypatch,
        [[[source]]],
    )

    result = segmenter.segment(
        np.zeros((2, 3, 3), dtype=np.uint8),
        context=_context(1, 3_000_000_000, (2, 3)),
    )

    np.testing.assert_array_equal(
        result,
        np.asarray(
            [
                [0.0, 0.2, 0.8],
                [1.0, 0.4, 0.6],
            ],
            dtype=np.float32,
        ),
    )
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    telemetry = segmenter.telemetry_snapshot()
    assert telemetry.input_frame_shape == (2, 3)
    assert telemetry.model_mask_shape == (2, 3)
    assert telemetry.output_mask_shape == (2, 3)
    assert telemetry.resize_interpolation == "none"
