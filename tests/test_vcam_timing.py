"""Monotonic output submission and sink-pacing timing contracts."""

from __future__ import annotations

import sys
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

import custback.pipeline as pipeline
import custback.vcam as vcam
from custback.config import OutputConfig
from custback.vcam import NullOutput, OutputSendTiming, PyVirtualCamOutput


def _frame() -> np.ndarray:
    return np.zeros((6, 8, 3), dtype=np.uint8)


def test_default_send_timing_uses_legacy_send_completion_as_submission(
    monkeypatch,
) -> None:
    ticks = iter((1_000_000_000, 1_003_500_000))
    monkeypatch.setattr(vcam.time, "monotonic_ns", lambda: next(ticks))
    output = NullOutput(width=8, height=6, fps=30)

    timing = output.send_with_timing(_frame())

    assert timing == OutputSendTiming(
        submitted_at_ns=1_003_500_000,
        completed_at_ns=1_003_500_000,
        submission_ms=3.5,
        pacing_wait_ms=0.0,
        pacing_events=0,
        recovery_events=0,
    )
    assert output.frames_sent == 1


def test_default_send_timing_never_reports_a_negative_duration(monkeypatch) -> None:
    ticks = iter((10, 9))
    monkeypatch.setattr(vcam.time, "monotonic_ns", lambda: next(ticks))
    output = NullOutput(width=8, height=6, fps=30)

    timing = output.send_with_timing(_frame())

    assert timing.submitted_at_ns == timing.completed_at_ns == 10
    assert timing.submission_ms == timing.pacing_wait_ms == 0.0


def test_pyvirtualcam_timing_separates_submission_from_pacing(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeCamera:
        def __init__(self, **kwargs) -> None:
            self.width = kwargs["width"]
            self.height = kwargs["height"]
            self.fps = kwargs["fps"]
            self.device = "fake"

        def send(self, frame: np.ndarray) -> None:
            assert frame.shape == (6, 8, 3)
            calls.append("send")

        def sleep_until_next_frame(self) -> None:
            calls.append("pace")

        def close(self) -> None:
            calls.append("close")

    module = SimpleNamespace(
        Camera=FakeCamera,
        PixelFormat=SimpleNamespace(BGR="BGR"),
    )
    monkeypatch.setitem(sys.modules, "pyvirtualcam", module)
    ticks = iter((2_000_000_000, 2_002_000_000, 2_009_250_000))
    monkeypatch.setattr(vcam.time, "monotonic_ns", lambda: next(ticks))
    output = PyVirtualCamOutput(
        OutputConfig(backend="pyvirtualcam", fps=30),
        8,
        6,
    )

    timing = output.send_with_timing(_frame())

    assert calls == ["send", "pace"]
    assert timing == OutputSendTiming(
        submitted_at_ns=2_002_000_000,
        completed_at_ns=2_009_250_000,
        submission_ms=2.0,
        pacing_wait_ms=7.25,
        pacing_events=1,
        recovery_events=0,
    )
    output.close()
    assert calls[-1] == "close"


def test_pyvirtualcam_unpaced_submission_keeps_validation_and_omits_sleep(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeCamera:
        width = 8
        height = 6
        fps = 30
        device = "fake"

        def __init__(self, **_kwargs) -> None:
            pass

        def send(self, _frame: np.ndarray) -> None:
            calls.append("send")

        def sleep_until_next_frame(self) -> None:
            calls.append("pace")

        def close(self) -> None:
            pass

    module = SimpleNamespace(
        Camera=FakeCamera,
        PixelFormat=SimpleNamespace(BGR="BGR"),
    )
    monkeypatch.setitem(sys.modules, "pyvirtualcam", module)
    ticks = iter((3_000_000_000, 3_001_500_000))
    monkeypatch.setattr(vcam.time, "monotonic_ns", lambda: next(ticks))
    output = PyVirtualCamOutput(OutputConfig(backend="pyvirtualcam"), 8, 6)

    timing = output.submit_unpaced_with_timing(_frame())

    assert calls == ["send"]
    assert timing == OutputSendTiming(
        submitted_at_ns=3_001_500_000,
        completed_at_ns=3_001_500_000,
        submission_ms=1.5,
        pacing_wait_ms=0.0,
    )


def test_send_compatibility_still_submits_and_paces(monkeypatch) -> None:
    calls: list[str] = []

    class FakeCamera:
        width = 8
        height = 6
        fps = 30
        device = "fake"

        def __init__(self, **_kwargs) -> None:
            pass

        def send(self, _frame: np.ndarray) -> None:
            calls.append("send")

        def sleep_until_next_frame(self) -> None:
            calls.append("pace")

        def close(self) -> None:
            pass

    module = SimpleNamespace(
        Camera=FakeCamera,
        PixelFormat=SimpleNamespace(BGR="BGR"),
    )
    monkeypatch.setitem(sys.modules, "pyvirtualcam", module)
    monkeypatch.setattr(vcam.time, "monotonic_ns", lambda: 1_000_000_000)
    output = PyVirtualCamOutput(OutputConfig(backend="pyvirtualcam"), 8, 6)

    result = output.send(_frame())

    assert result is None
    assert calls == ["send", "pace"]


def test_output_send_timing_contains_only_monotonic_scalar_provenance() -> None:
    assert {field.name for field in fields(OutputSendTiming)} == {
        "submitted_at_ns",
        "completed_at_ns",
        "submission_ms",
        "pacing_wait_ms",
        "pacing_events",
        "recovery_events",
    }


def test_pipeline_output_timing_includes_required_privacy_copy(monkeypatch) -> None:
    source = _frame()
    received: list[np.ndarray] = []

    class TimedOutput:
        def send_with_timing(self, frame: np.ndarray) -> OutputSendTiming:
            received.append(frame)
            return OutputSendTiming(
                submitted_at_ns=1_004_000_000,
                completed_at_ns=1_006_000_000,
                submission_ms=1.5,
                pacing_wait_ms=2.0,
                pacing_events=1,
            )

    ticks = iter((1_000_000_000, 1_002_000_000))
    monkeypatch.setattr(pipeline.time, "monotonic_ns", lambda: next(ticks))

    timing = pipeline._send_output_with_timing(  # noqa: SLF001
        TimedOutput(),
        source,
        copy_frame=True,
    )

    assert len(received) == 1
    assert received[0] is not source
    assert np.array_equal(received[0], source)
    assert timing.submission_ms == 3.5
    assert timing.pacing_wait_ms == 2.0
