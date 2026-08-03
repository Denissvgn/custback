"""Focused MATTE-1.2 capture-time and temporal-reset contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from custback.capture import CapturedFrame
from custback.config import AppConfig, RuntimeConfig, SegmentationConfig
from custback.hub import FrameHub
from custback.pipeline import (
    Pipeline,
    _Activation,
    _PatchRequest,
    _Resources,
    _TemporalStateOwner,
    _segmenter_key,
)
from custback.segmentation import (
    SEGMENTATION_TIMESTAMP_GAP_RESET_NS,
    MaskRefiner,
    MediaPipeSegmenter,
    NullSegmenter,
    SegmentationFrameContext,
    SegmentationTimeline,
    Segmenter,
    TemporalResetReason,
)


def _context(
    sequence: int,
    timestamp_ns: int,
    *,
    generation: int = 1,
    geometry_generation: int = 1,
    shape: tuple[int, int] = (6, 8),
) -> SegmentationFrameContext:
    return SegmentationFrameContext(
        sequence=sequence,
        timestamp_ns=timestamp_ns,
        generation=generation,
        geometry_generation=geometry_generation,
        shape=shape,
    )


def _captured(
    pixels: np.ndarray,
    sequence: int,
    timestamp_ns: int,
    *,
    generation: int = 1,
    geometry_generation: int = 1,
) -> CapturedFrame:
    height, width = pixels.shape[:2]
    return CapturedFrame(
        pixels=pixels,
        sequence=sequence,
        captured_at_ns=timestamp_ns,
        generation=generation,
        geometry_generation=geometry_generation,
        content_rect=(0, 0, width, height),
    )


def _config() -> AppConfig:
    return AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 16, "height": 16, "fps": 30},
            "background": {"mode": "color"},
            "segmentation": {
                "backend": "heuristic",
                "mask_blur": 0,
                "edge_refine": False,
                "temporal_smoothing": 0.0,
            },
            "output": {"backend": "null", "fps": 30},
            "api": {"enabled": False},
        }
    )


class _TrackingSegmenter(Segmenter):
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        super().__init__()
        self.events = events

    def reset_temporal_state(
        self,
        reason: TemporalResetReason,
        timestamp_ns: int | None,
    ) -> None:
        self.events.append(("segmenter-reset", reason, timestamp_ns))
        super().reset_temporal_state(reason, timestamp_ns)

    def segment(
        self,
        frame_bgr: np.ndarray,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        assert context is not None
        self._accept_frame_context(context, frame_bgr.shape[:2])
        self.events.append(("segment", context.sequence, context.timestamp_ns))
        return np.zeros(frame_bgr.shape[:2], dtype=np.float32)


class _TrackingRefiner(MaskRefiner):
    def __init__(
        self,
        cfg: SegmentationConfig,
        events: list[tuple[object, ...]],
    ) -> None:
        super().__init__(cfg)
        self.events = events

    def reset_temporal_state(
        self,
        reason: TemporalResetReason,
        timestamp_ns: int | None,
    ) -> None:
        self.events.append(("refiner-reset", reason, timestamp_ns))
        super().reset_temporal_state(reason, timestamp_ns)

    def refine(
        self,
        mask: np.ndarray,
        frame_bgr: np.ndarray | None = None,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        assert context is not None
        self.events.append(("refine", context.sequence, context.timestamp_ns))
        return super().refine(mask, frame_bgr, context=context)


class _RecoveringSegmenter(_TrackingSegmenter):
    """Model one backend failure followed by a clean same-frame retry."""

    def segment(
        self,
        frame_bgr: np.ndarray,
        *,
        context: SegmentationFrameContext | None = None,
    ) -> np.ndarray:
        assert context is not None
        self._accept_frame_context(context, frame_bgr.shape[:2])
        self.reset_temporal_state(
            TemporalResetReason.BACKEND_RECOVERY,
            context.timestamp_ns,
        )
        self._accept_frame_context(context, frame_bgr.shape[:2])
        self.events.append(("segment", context.sequence, context.timestamp_ns))
        return np.zeros(frame_bgr.shape[:2], dtype=np.float32)


def _pipeline_resources() -> tuple[
    Pipeline,
    _Resources,
    _TrackingSegmenter,
    _TrackingRefiner,
    list[tuple[object, ...]],
]:
    cfg = _config()
    events: list[tuple[object, ...]] = []
    segmenter = _TrackingSegmenter(events)
    refiner = _TrackingRefiner(cfg.segmentation, events)
    resources = _Resources(
        cfg,
        0,
        None,
        segmenter,
        refiner,
        None,
        None,
    )
    return (
        Pipeline(RuntimeConfig(cfg), FrameHub()),
        resources,
        segmenter,
        refiner,
        events,
    )


def test_first_frame_resets_pair_before_processing_and_identical_gap_is_continuous():
    pipeline, resources, segmenter, refiner, events = _pipeline_resources()
    pixels = np.full((6, 8, 3), 47, dtype=np.uint8)
    try:
        pipeline._segment_resource_masks(
            resources,
            _captured(pixels, 1, 100),
            privacy_safe=False,
        )
        assert events == [
            ("segmenter-reset", TemporalResetReason.INITIAL, 100),
            ("refiner-reset", TemporalResetReason.INITIAL, 100),
            ("segment", 1, 100),
            ("refine", 1, 100),
        ]

        events.clear()
        pipeline._segment_resource_masks(
            resources,
            _captured(pixels.copy(), 2, 200),
            privacy_safe=False,
        )
        assert events == [("segment", 2, 200), ("refine", 2, 200)]

        events.clear()
        pipeline._segment_resource_masks(
            resources,
            _captured(pixels.copy(), 5, 300),
            privacy_safe=False,
        )
        assert events == [("segment", 5, 300), ("refine", 5, 300)]
        assert segmenter.temporal_reset_count == 1
        assert refiner.temporal_reset_count == 1
        snapshot = resources.segmentation_timeline.snapshot()
        assert snapshot.reset_count == 1
        assert snapshot.last_reset_reason is TemporalResetReason.INITIAL
        assert snapshot.sequence_gap_events == 1
        assert snapshot.sequence_gap_frames == 2
        assert snapshot.last_sequence == 5
    finally:
        resources.close()


@pytest.mark.parametrize(
    ("delta_ns", "expected_reason"),
    [
        (SEGMENTATION_TIMESTAMP_GAP_RESET_NS, None),
        (
            SEGMENTATION_TIMESTAMP_GAP_RESET_NS + 1,
            TemporalResetReason.TIMESTAMP_GAP,
        ),
    ],
)
def test_timestamp_gap_resets_only_above_the_qualified_limit(
    delta_ns: int,
    expected_reason: TemporalResetReason | None,
):
    timeline = SegmentationTimeline()
    assert timeline.observe(_context(1, 10)).reset_reason is TemporalResetReason.INITIAL

    boundary = timeline.observe(_context(4, 10 + delta_ns))

    assert boundary.elapsed_ns == delta_ns
    assert boundary.sequence_gap == 2
    assert boundary.reset_reason is expected_reason
    snapshot = timeline.snapshot()
    assert snapshot.reset_count == (1 if expected_reason is None else 2)
    assert snapshot.sequence_gap_events == 1
    assert snapshot.sequence_gap_frames == 2


def test_generation_change_wins_over_simultaneous_sequence_and_timestamp_gap():
    timeline = SegmentationTimeline()
    timeline.observe(_context(1, 100, generation=1))

    boundary = timeline.observe(
        _context(
            4,
            100 + SEGMENTATION_TIMESTAMP_GAP_RESET_NS + 1,
            generation=2,
        )
    )

    assert boundary.reset_reason is TemporalResetReason.CAPTURE_GENERATION
    assert boundary.sequence_gap == 2
    snapshot = timeline.snapshot()
    assert snapshot.reset_count == 2
    assert snapshot.last_reset_reason is TemporalResetReason.CAPTURE_GENERATION
    assert snapshot.sequence_gap_events == 1
    assert snapshot.sequence_gap_frames == 2


def test_requested_segmentation_generation_reset_applies_once_to_next_frame():
    timeline = SegmentationTimeline()
    timeline.observe(_context(1, 100))
    timeline.request_reset(TemporalResetReason.SEGMENTATION_CONFIG)
    assert timeline.snapshot().reset_count == 1

    boundary = timeline.observe(_context(2, 200))
    assert boundary.reset_reason is TemporalResetReason.SEGMENTATION_CONFIG
    assert TemporalResetReason.SEGMENTATION_CONFIG.value == (
        "segmentation-generation-change"
    )
    assert timeline.observe(_context(3, 300)).reset_reason is None
    snapshot = timeline.snapshot()
    assert snapshot.reset_count == 2
    assert snapshot.last_reset_reason is TemporalResetReason.SEGMENTATION_CONFIG


@pytest.mark.parametrize("sequence", [2, 1], ids=["duplicate", "reordered"])
def test_timeline_rejects_nonincreasing_sequence_without_mutating_state(
    sequence: int,
):
    timeline = SegmentationTimeline()
    timeline.observe(_context(2, 100))
    before = timeline.snapshot()

    with pytest.raises(ValueError, match="sequence"):
        timeline.observe(_context(sequence, 200))

    assert timeline.snapshot() == before
    assert timeline.observe(_context(3, 200)).reset_reason is None


@pytest.mark.parametrize(
    "second",
    [
        _context(2, 200, geometry_generation=2),
        _context(2, 200, shape=(7, 8)),
    ],
    ids=["geometry-generation", "pixel-shape"],
)
def test_geometry_generation_or_shape_change_resets(second: SegmentationFrameContext):
    timeline = SegmentationTimeline()
    timeline.observe(_context(1, 100))

    boundary = timeline.observe(second)

    assert boundary.reset_reason is TemporalResetReason.GEOMETRY
    assert timeline.snapshot().reset_count == 2


@pytest.mark.parametrize(
    ("second_pixels", "second_generation", "expected_reason"),
    [
        (
            np.zeros((6, 8, 3), dtype=np.uint8),
            2,
            TemporalResetReason.CAPTURE_GENERATION,
        ),
        (
            np.zeros((7, 8, 3), dtype=np.uint8),
            1,
            TemporalResetReason.GEOMETRY,
        ),
    ],
    ids=["same-resolution-generation", "resize"],
)
def test_generation_or_resize_resets_pair_once_before_boundary_processing(
    second_pixels: np.ndarray,
    second_generation: int,
    expected_reason: TemporalResetReason,
):
    pipeline, resources, segmenter, refiner, events = _pipeline_resources()
    try:
        pipeline._segment_resource_masks(
            resources,
            _captured(np.zeros((6, 8, 3), dtype=np.uint8), 1, 100),
            privacy_safe=False,
        )

        events.clear()
        pipeline._segment_resource_masks(
            resources,
            _captured(
                second_pixels,
                2,
                200,
                generation=second_generation,
            ),
            privacy_safe=False,
        )

        assert events == [
            ("segmenter-reset", expected_reason, 200),
            ("refiner-reset", expected_reason, 200),
            ("segment", 2, 200),
            ("refine", 2, 200),
        ]
        assert segmenter.temporal_reset_count == 2
        assert refiner.temporal_reset_count == 2
        snapshot = resources.segmentation_timeline.snapshot()
        assert snapshot.reset_count == 2
        assert snapshot.last_reset_reason is expected_reason
    finally:
        resources.close()


def test_equal_timestamp_is_continuous_but_decreasing_timestamp_resets_pair():
    pipeline, resources, segmenter, refiner, events = _pipeline_resources()
    pixels = np.zeros((6, 8, 3), dtype=np.uint8)
    try:
        pipeline._segment_resource_masks(
            resources,
            _captured(pixels, 1, 100),
            privacy_safe=False,
        )

        events.clear()
        pipeline._segment_resource_masks(
            resources,
            _captured(pixels.copy(), 2, 100),
            privacy_safe=False,
        )
        assert events == [("segment", 2, 100), ("refine", 2, 100)]

        events.clear()
        pipeline._segment_resource_masks(
            resources,
            _captured(pixels.copy(), 3, 99),
            privacy_safe=False,
        )
        assert events == [
            ("segmenter-reset", TemporalResetReason.NON_MONOTONIC_TIMESTAMP, 99),
            ("refiner-reset", TemporalResetReason.NON_MONOTONIC_TIMESTAMP, 99),
            ("segment", 3, 99),
            ("refine", 3, 99),
        ]
        assert segmenter.temporal_reset_count == 2
        assert refiner.temporal_reset_count == 2
    finally:
        resources.close()


def test_backend_recovery_resets_refiner_before_same_boundary_refine():
    cfg = _config()
    events: list[tuple[object, ...]] = []
    segmenter = _RecoveringSegmenter(events)
    refiner = _TrackingRefiner(cfg.segmentation, events)
    resources = _Resources(
        cfg,
        0,
        None,
        segmenter,
        refiner,
        None,
        None,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    try:
        pipeline._segment_resource_masks(
            resources,
            _captured(np.zeros((6, 8, 3), dtype=np.uint8), 1, 100),
            privacy_safe=False,
        )

        assert events == [
            ("segmenter-reset", TemporalResetReason.INITIAL, 100),
            ("refiner-reset", TemporalResetReason.INITIAL, 100),
            ("segmenter-reset", TemporalResetReason.BACKEND_RECOVERY, 100),
            ("segment", 1, 100),
            ("refiner-reset", TemporalResetReason.BACKEND_RECOVERY, 100),
            ("refine", 1, 100),
        ]
        assert segmenter.temporal_reset_count == 2
        assert refiner.temporal_reset_count == 2
        snapshot = resources.segmentation_timeline.snapshot()
        assert snapshot.reset_count == 2
        assert snapshot.last_reset_reason is TemporalResetReason.BACKEND_RECOVERY
    finally:
        resources.close()


def test_segmenter_context_requires_increasing_sequence_with_nondecreasing_time():
    segmenter = NullSegmenter()
    pixels = np.zeros((6, 8, 3), dtype=np.uint8)
    segmenter.segment(pixels, context=_context(1, 100))
    segmenter.segment(pixels, context=_context(2, 100))

    with pytest.raises(ValueError, match="sequence"):
        segmenter.segment(pixels, context=_context(2, 101))
    with pytest.raises(ValueError, match="timestamp"):
        segmenter.segment(pixels, context=_context(3, 99))

    segmenter.reset_temporal_state(TemporalResetReason.CAPTURE_GENERATION, 50)
    segmenter.segment(pixels, context=_context(1, 50, generation=2))


def test_segmenter_rejects_context_shape_that_does_not_match_pixels():
    pixels = np.zeros((6, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        NullSegmenter().segment(
            pixels,
            context=_context(1, 100, shape=(5, 8)),
        )


def test_frame_context_preserves_zero_based_bundle_sequences():
    assert _context(0, 100).sequence == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sequence": -1}, "sequence"),
        ({"sequence": True}, "sequence"),
        ({"timestamp_ns": -1}, "timestamp"),
        ({"timestamp_ns": True}, "timestamp"),
        ({"generation": -1}, "capture generation"),
        ({"geometry_generation": -1}, "geometry generation"),
        ({"shape": (0, 8)}, "frame shape"),
        ({"shape": [6, 8]}, "frame shape"),
    ],
)
def test_frame_context_rejects_malformed_atomic_identity(
    changes: dict[str, object],
    message: str,
):
    values: dict[str, object] = {
        "sequence": 1,
        "timestamp_ns": 100,
        "generation": 1,
        "geometry_generation": 1,
        "shape": (6, 8),
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        SegmentationFrameContext(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("gap_reset_ns", [0, -1, True, 1.5])
def test_timeline_rejects_invalid_gap_limits(gap_reset_ns: object):
    with pytest.raises(ValueError, match="gap limit"):
        SegmentationTimeline(gap_reset_ns=gap_reset_ns)  # type: ignore[arg-type]


def test_mediapipe_reset_recreates_only_an_already_used_video_task():
    class _Task:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    segmenter = object.__new__(MediaPipeSegmenter)
    Segmenter.__init__(segmenter)
    initial_task = _Task()
    replacement_task = _Task()
    delegate = object()
    make_calls: list[object] = []
    segmenter._segmenter = initial_task
    segmenter._active_delegate = delegate
    segmenter._make_segmenter = lambda delegate: (
        make_calls.append(delegate) or replacement_task
    )
    segmenter._ts_ms = 0

    segmenter.reset_temporal_state(TemporalResetReason.INITIAL, 100)
    assert make_calls == []
    assert initial_task.close_calls == 0
    assert segmenter._segmenter is initial_task

    segmenter._ts_ms = 33
    segmenter.reset_temporal_state(
        TemporalResetReason.CAPTURE_GENERATION,
        200,
    )
    assert make_calls == [delegate]
    assert initial_task.close_calls == 1
    assert segmenter._segmenter is replacement_task
    assert segmenter._ts_ms == 0
    assert segmenter.temporal_reset_count == 2
    assert (
        segmenter.last_temporal_reset_reason is TemporalResetReason.CAPTURE_GENERATION
    )


def test_failed_boundary_reset_is_retried_before_next_successful_frame():
    class _Task:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class _ResettableMediaPipe(MediaPipeSegmenter):
        def segment(
            self,
            frame_bgr: np.ndarray,
            *,
            context: SegmentationFrameContext | None = None,
        ) -> np.ndarray:
            assert context is not None
            self._accept_frame_context(context, frame_bgr.shape[:2])
            self._ts_ms += 33
            return np.zeros(frame_bgr.shape[:2], dtype=np.float32)

    cfg = _config()
    segmenter = object.__new__(_ResettableMediaPipe)
    Segmenter.__init__(segmenter)
    initial_task = _Task()
    replacement_task = _Task()
    make_calls = 0

    def make(delegate):
        nonlocal make_calls
        del delegate
        make_calls += 1
        if make_calls == 1:
            raise RuntimeError("task recreation failed")
        return replacement_task

    segmenter._segmenter = initial_task
    segmenter._active_delegate = None
    segmenter._make_segmenter = make
    segmenter._ts_ms = 0
    refiner = MaskRefiner(cfg.segmentation)
    resources = _Resources(
        cfg,
        0,
        None,
        segmenter,
        refiner,
        None,
        None,
    )
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    pixels = np.zeros((6, 8, 3), dtype=np.uint8)
    try:
        pipeline._segment_resource_masks(
            resources,
            _captured(pixels, 1, 100),
            privacy_safe=False,
        )

        with pytest.raises(RuntimeError, match="task recreation failed"):
            pipeline._segment_resource_masks(
                resources,
                _captured(pixels, 2, 99),
                privacy_safe=False,
            )
        failed = resources.segmentation_timeline.snapshot()
        assert failed.reset_count == 1
        assert failed.last_reset_reason is TemporalResetReason.INITIAL
        assert failed.last_sequence == 1
        assert segmenter._segmenter is initial_task
        assert refiner.temporal_reset_count == 1

        pipeline._segment_resource_masks(
            resources,
            _captured(pixels, 3, 110),
            privacy_safe=False,
        )
        recovered = resources.segmentation_timeline.snapshot()
        assert recovered.reset_count == 2
        assert (
            recovered.last_reset_reason is TemporalResetReason.NON_MONOTONIC_TIMESTAMP
        )
        assert recovered.last_sequence == 3
        assert make_calls == 2
        assert initial_task.close_calls == 1
        assert segmenter._segmenter is replacement_task
        assert segmenter.temporal_reset_count == 2
        assert refiner.temporal_reset_count == 2
    finally:
        resources.close()


def test_acceleration_only_activation_trials_pair_before_commit_and_resets_live_pair():
    cfg = _config()
    candidate = cfg.patched({"acceleration": {"mode": "cpu"}})
    runtime = RuntimeConfig(cfg)
    pipeline = Pipeline(runtime, FrameHub())
    old_events: list[tuple[object, ...]] = []
    candidate_events: list[tuple[object, ...]] = []
    segment_versions: list[int] = []
    refine_versions: list[int] = []

    class _VersionedSegmenter(_TrackingSegmenter):
        def segment(
            self,
            frame_bgr: np.ndarray,
            *,
            context: SegmentationFrameContext | None = None,
        ) -> np.ndarray:
            segment_versions.append(runtime.version)
            return super().segment(frame_bgr, context=context)

    class _VersionedRefiner(_TrackingRefiner):
        def refine(
            self,
            mask: np.ndarray,
            frame_bgr: np.ndarray | None = None,
            *,
            context: SegmentationFrameContext | None = None,
        ) -> np.ndarray:
            refine_versions.append(runtime.version)
            return super().refine(mask, frame_bgr, context=context)

    staged_segmenter = _VersionedSegmenter(candidate_events)
    staged_refiner = _VersionedRefiner(
        candidate.segmentation,
        candidate_events,
    )
    resources = _Resources(
        cfg,
        0,
        None,
        _TrackingSegmenter(old_events),
        _TrackingRefiner(cfg.segmentation, old_events),
        None,
        None,
    )
    assert resources.harmonizer is not None
    captured = _captured(np.zeros((16, 16, 3), dtype=np.uint8), 1, 100)
    request = _PatchRequest(
        candidate,
        0,
        prepared_activation=_Activation(
            candidate=candidate,
            replace_segmenter=True,
            segmenter=staged_segmenter,
            refiner=staged_refiner,
            temporal_state_owner=_TemporalStateOwner(
                policy=_segmenter_key(candidate),
                generation=None,
                segmenter=staged_segmenter,
                refiner=staged_refiner,
            ),
            replace_harmonizer=True,
            harmonizer=resources.harmonizer.clone(),
        ),
    )
    try:
        pipeline._handle_patch_request(resources, request, captured)

        assert request.result is not None
        assert request.error is None
        assert runtime.version == 1
        assert segment_versions == [0]
        assert refine_versions == [0]
        assert old_events == []
        assert candidate_events == [
            ("segmenter-reset", TemporalResetReason.SEGMENTATION_CONFIG, 100),
            ("refiner-reset", TemporalResetReason.SEGMENTATION_CONFIG, 100),
            ("segment", 1, 100),
            ("refine", 1, 100),
            ("segmenter-reset", TemporalResetReason.SEGMENTATION_CONFIG, 100),
            ("refiner-reset", TemporalResetReason.SEGMENTATION_CONFIG, 100),
        ]

        candidate_events.clear()
        pipeline._segment_resource_masks(
            resources,
            captured,
            privacy_safe=False,
        )
        assert segment_versions == [0, 1]
        assert refine_versions == [0, 1]
        assert candidate_events == [
            ("segmenter-reset", TemporalResetReason.SEGMENTATION_CONFIG, 100),
            ("refiner-reset", TemporalResetReason.SEGMENTATION_CONFIG, 100),
            ("segment", 1, 100),
            ("refine", 1, 100),
        ]
        temporal = resources.segmentation_timeline.snapshot()
        assert temporal.reset_count == 1
        assert temporal.last_reset_reason is TemporalResetReason.SEGMENTATION_CONFIG
    finally:
        resources.close()
        pipeline.stop(timeout=1.0)


def test_output_repeat_does_not_segment_refine_or_reset_again():
    cfg = _config()
    events: list[tuple[object, ...]] = []
    segmenter = _TrackingSegmenter(events)
    refiner = _TrackingRefiner(cfg.segmentation, events)
    pixels = np.full((16, 16, 3), 91, dtype=np.uint8)
    captured = _captured(pixels, 1, 100)

    class _Capture:
        def __init__(self) -> None:
            self.values: list[CapturedFrame | None] = [captured, None]

        def read(self) -> CapturedFrame | None:
            return self.values.pop(0) if self.values else None

        def close(self) -> None:
            pass

    class _Backdrop:
        def frame(self, width: int, height: int) -> np.ndarray:
            return np.zeros((height, width, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())

    class _Output:
        paces = True
        fallback_active = False
        fallback_reason = ""

        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []

        def send(self, frame: np.ndarray) -> None:
            self.frames.append(frame.copy())
            if len(self.frames) == 2:
                pipeline._stop.set()

        def close(self) -> None:
            pass

    output = _Output()
    resources = _Resources(
        cfg,
        0,
        _Capture(),
        segmenter,
        refiner,
        _Backdrop(),
        output,
    )
    try:
        pipeline._loop(resources)

        assert events == [
            ("segmenter-reset", TemporalResetReason.INITIAL, 100),
            ("refiner-reset", TemporalResetReason.INITIAL, 100),
            ("segment", 1, 100),
            ("refine", 1, 100),
        ]
        assert segmenter.temporal_reset_count == 1
        assert refiner.temporal_reset_count == 1
        assert len(output.frames) == 2
        np.testing.assert_array_equal(output.frames[1], output.frames[0])
        assert pipeline.hub.stats_dict()["output_repeated_frames"] == 1
    finally:
        resources.close()
