"""VIS-1.3 camera-to-canvas geometry contract tests."""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import custback.capture as capture_mod
from custback.capture import (
    CaptureError,
    CaptureModeError,
    OpenCVCapture,
    SyntheticCapture,
)
from custback.config import CameraConfig, FitMode
from custback.geometry import apply_transform


def _fourcc(code: str) -> int:
    return sum(ord(char) << (8 * index) for index, char in enumerate(code))


_RELEASED = object()


class _ControlledCap:
    """A capture whose reader advances only when the test supplies a frame."""

    def __init__(
        self,
        *,
        property_size: tuple[int, int],
        property_fps: float = 30.0,
        locked_properties: tuple[int, ...] = (),
        initial_frames: tuple[object, ...] = (),
        backend_name: str = "TEST",
        control_values: dict[int, float] | None = None,
    ) -> None:
        width, height = property_size
        self.values = {
            _FakeCV2.CAP_PROP_FRAME_WIDTH: float(width),
            _FakeCV2.CAP_PROP_FRAME_HEIGHT: float(height),
            _FakeCV2.CAP_PROP_FPS: property_fps,
            _FakeCV2.CAP_PROP_FOURCC: float(_fourcc("YUYV")),
            _FakeCV2.CAP_PROP_BACKEND: 999.0,
            _FakeCV2.CAP_PROP_ORIENTATION_META: 0.0,
            _FakeCV2.CAP_PROP_ORIENTATION_AUTO: 0.0,
        }
        self.values.update(control_values or {})
        self.backend_name = backend_name
        self.locked_properties = set(locked_properties)
        self.frames: queue.Queue[object] = queue.Queue()
        for frame in initial_frames:
            self.frames.put(frame)
        self.set_calls: list[tuple[int, float]] = []
        self.get_calls: list[int] = []
        self.read_calls = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def getBackendName(self) -> str:
        return self.backend_name

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        if prop not in self.locked_properties:
            self.values[prop] = float(value)
        return True

    def get(self, prop: int) -> float:
        self.get_calls.append(prop)
        return self.values.get(prop, 0.0)

    def push(self, frame: object) -> None:
        self.frames.put(frame)

    def read(self) -> tuple[bool, object | None]:
        self.read_calls += 1
        while not self.released:
            try:
                frame = self.frames.get(timeout=0.01)
            except queue.Empty:
                continue
            if frame is _RELEASED:
                return False, None
            if isinstance(frame, np.ndarray):
                return True, frame.copy()
            return True, frame
        return False, None

    def release(self) -> None:
        self.released = True
        self.frames.put(_RELEASED)


class _FailAfterFramesCap(_ControlledCap):
    """Return the scripted frames, then continuously report acquisition failure."""

    def read(self) -> tuple[bool, object | None]:
        self.read_calls += 1
        if self.released:
            return False, None
        try:
            frame = self.frames.get_nowait()
        except queue.Empty:
            time.sleep(0.002)
            return False, None
        if isinstance(frame, np.ndarray):
            return True, frame.copy()
        return True, frame


class _FakeCV2:
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

    def __init__(self, captures: list[_ControlledCap]) -> None:
        self.captures = list(captures)
        self.opened: list[_ControlledCap] = []

    def VideoCapture(
        self, _device: int | str, _api_preference: int | None = None
    ) -> _ControlledCap:
        if not self.captures:
            raise AssertionError("capture unexpectedly reopened")
        capture = self.captures.pop(0)
        self.opened.append(capture)
        return capture

    @staticmethod
    def VideoWriter_fourcc(*chars: str) -> int:
        return _fourcc("".join(chars))


def _wait_for_frame(
    capture: OpenCVCapture,
    *,
    predicate: Callable[[np.ndarray], bool] | None = None,
    timeout: float = 1.0,
) -> np.ndarray:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = capture.read()
        if frame is not None and (predicate is None or predicate(frame)):
            return frame
        time.sleep(0.002)
    raise AssertionError("capture produced no matching frame")


def _wait_for_error(
    capture: OpenCVCapture,
    error_type: type[BaseException],
    *,
    timeout: float = 1.0,
) -> BaseException:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            capture.read()
        except error_type as exc:
            return exc
        time.sleep(0.002)
    raise AssertionError(f"capture did not raise {error_type.__name__}")


def _install_camera(
    monkeypatch: pytest.MonkeyPatch,
    capture: _ControlledCap,
    *,
    cfg: CameraConfig,
    canvas_size: tuple[int, int],
) -> OpenCVCapture:
    monkeypatch.setattr(capture_mod, "cv2", _FakeCV2([capture]))
    return OpenCVCapture(cfg, canvas_size)


def test_acquisition_stays_640x480_while_cover_normalizes_circle_to_1280x720(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = _ControlledCap(property_size=(640, 480))
    real_transform = capture_mod.transform_frame
    plans: list[Any] = []

    def count_transform(frame: np.ndarray, *args: Any, **kwargs: Any):
        normalized, plan = real_transform(frame, *args, **kwargs)
        plans.append(plan)
        return normalized, plan

    monkeypatch.setattr(capture_mod, "transform_frame", count_transform)
    capture = _install_camera(
        monkeypatch,
        cap,
        cfg=CameraConfig(width=640, height=480, fps=30, fit_mode="cover"),
        canvas_size=(1280, 720),
    )
    try:
        yy, xx = np.ogrid[:480, :640]
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        raw[(xx - 320) ** 2 + (yy - 240) ** 2 <= 80**2] = 255
        cap.push(raw)

        frame = _wait_for_frame(capture)

        assert frame.shape == (720, 1280, 3)
        assert frame.dtype == np.uint8
        assert frame.flags.c_contiguous
        requested = {prop: value for prop, value in cap.set_calls}
        assert requested[_FakeCV2.CAP_PROP_FRAME_WIDTH] == 640.0
        assert requested[_FakeCV2.CAP_PROP_FRAME_HEIGHT] == 480.0
        assert len(plans) == 1
        crop = plans[0].crop_rect
        assert (crop.left, crop.top, crop.right, crop.bottom) == (
            0,
            120,
            1280,
            840,
        )

        foreground_y, foreground_x = np.where(frame[:, :, 0] >= 128)
        circle_width = int(foreground_x.max() - foreground_x.min() + 1)
        circle_height = int(foreground_y.max() - foreground_y.min() + 1)
        assert abs(circle_width - circle_height) <= 1
        assert 319 <= circle_width <= 323
    finally:
        capture.close()


def test_health_separates_property_delivered_oriented_and_normalized_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    locked = (
        _FakeCV2.CAP_PROP_FRAME_WIDTH,
        _FakeCV2.CAP_PROP_FRAME_HEIGHT,
    )
    cap = _ControlledCap(
        property_size=(64, 48),
        locked_properties=locked,
    )
    capture = _install_camera(
        monkeypatch,
        cap,
        cfg=CameraConfig(width=32, height=24, fps=30, fit_mode="contain"),
        canvas_size=(40, 20),
    )
    try:
        cap.push(np.full((24, 32, 3), 17, dtype=np.uint8))
        frame = _wait_for_frame(capture)
        health = capture.health_snapshot()

        assert frame.shape == (20, 40, 3)
        assert (health.width, health.height) == (64, 48)
        assert (health.delivered_width, health.delivered_height) == (32, 24)
        assert (health.oriented_width, health.oriented_height) == (32, 24)
        assert (health.normalized_width, health.normalized_height) == (40, 20)
        assert health.geometry_transitions == 1
        assert health.generation == 1
        assert health.geometry_generation == 1
        assert health.content_rect == (7, 0, 33, 20)
        assert "camera mode mismatch" in caplog.text
    finally:
        capture.close()


def test_camera_control_report_is_generation_bound_truthful_and_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_values = {
        _FakeCV2.CAP_PROP_AUTO_WB: 1.0,
        _FakeCV2.CAP_PROP_WB_TEMPERATURE: 0.0,
        _FakeCV2.CAP_PROP_AUTO_EXPOSURE: 0.75,
        _FakeCV2.CAP_PROP_EXPOSURE: -5.0,
        _FakeCV2.CAP_PROP_GAIN: 0.0,
        _FakeCV2.CAP_PROP_GAMMA: float("nan"),
    }
    cap = _ControlledCap(
        property_size=(32, 24),
        backend_name="V4L2",
        control_values=control_values,
    )
    capture = _install_camera(
        monkeypatch,
        cap,
        cfg=CameraConfig(
            width=32,
            height=24,
            fps=30,
            pixel_format="backend",
        ),
        canvas_size=(32, 24),
    )
    try:
        cap.push(np.full((24, 32, 3), 17, dtype=np.uint8))
        _wait_for_frame(capture)

        report = capture.health_snapshot().camera_controls.as_dict()

        assert report == {
            "policy": "preserve",
            "backend_family": "v4l2",
            "qualification": "unqualified",
            "writes_performed": False,
            "generation": 1,
            "properties": {
                "auto_white_balance": {"status": "reported", "value": 1.0},
                "white_balance_temperature": {
                    "status": "indeterminate-zero",
                    "value": 0.0,
                },
                "auto_exposure": {"status": "reported", "value": 0.75},
                "exposure": {"status": "reported", "value": -5.0},
                "gain": {"status": "indeterminate-zero", "value": 0.0},
                "gamma": {"status": "unavailable", "value": None},
            },
        }
        control_ids = set(control_values)
        assert not any(prop in control_ids for prop, _value in cap.set_calls)
        assert [prop for prop in cap.get_calls if prop in control_ids] == list(
            control_values
        )
        assert "device" not in repr(report).lower()
    finally:
        capture.close()


@pytest.mark.parametrize(
    ("backend", "family"),
    [
        ("V4L2", "v4l2"),
        ("MSMF", "msmf"),
        ("Media Foundation", "msmf"),
        ("DSHOW", "dshow"),
        ("DirectShow", "dshow"),
        ("AVFoundation", "other"),
        ("unknown", "other"),
    ],
)
def test_camera_control_backends_are_characterized_independently(
    backend: str,
    family: str,
) -> None:
    assert capture_mod._camera_backend_family(backend) == family


def test_dynamic_delivered_size_warn_replans_and_keeps_canonical_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = _ControlledCap(property_size=(32, 24))
    capture = _install_camera(
        monkeypatch,
        cap,
        cfg=CameraConfig(
            width=32,
            height=24,
            fps=30,
            fit_mode="cover",
            mode_mismatch="warn",
        ),
        canvas_size=(40, 30),
    )
    try:
        cap.push(np.full((24, 32, 3), 11, dtype=np.uint8))
        first = _wait_for_frame(capture)
        cap.push(np.full((32, 16, 3), 99, dtype=np.uint8))
        second = _wait_for_frame(capture)
        health = capture.health_snapshot()

        assert first.shape == second.shape == (30, 40, 3)
        assert int(first[0, 0, 0]) == 11
        assert int(second[0, 0, 0]) == 99
        assert (health.delivered_width, health.delivered_height) == (16, 32)
        assert (health.oriented_width, health.oriented_height) == (16, 32)
        assert (health.normalized_width, health.normalized_height) == (40, 30)
        assert health.geometry_transitions == 2
    finally:
        capture.close()


def test_legacy_stretch_aspect_upgrade_note_is_emitted_once_per_capture(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cap = _ControlledCap(property_size=(32, 24))
    capture = _install_camera(
        monkeypatch,
        cap,
        cfg=CameraConfig(
            width=32,
            height=24,
            fps=30,
            fit_mode="stretch",
            mode_mismatch="warn",
        ),
        canvas_size=(40, 20),
    )
    try:
        cap.push(np.full((24, 32, 3), 11, dtype=np.uint8))
        _wait_for_frame(capture)
        cap.push(np.full((32, 16, 3), 99, dtype=np.uint8))
        _wait_for_frame(
            capture,
            predicate=lambda frame: int(frame[0, 0, 0]) == 99,
        )

        note = "visual-policy upgrade note"
        assert caplog.text.count(note) == 1
        assert "schema-v1 stretch preserves legacy distortion" in caplog.text
        assert "camera.fit_mode=stretch" in caplog.text
        assert "camera.fit_mode=cover" in caplog.text
        assert "/dev/" not in caplog.text
    finally:
        capture.close()


def test_cover_and_aspect_matched_stretch_do_not_emit_upgrade_note(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cases: tuple[tuple[FitMode, tuple[int, int]], ...] = (
        ("cover", (40, 20)),
        ("stretch", (40, 30)),
    )
    for fit, canvas in cases:
        cap = _ControlledCap(property_size=(32, 24))
        capture = _install_camera(
            monkeypatch,
            cap,
            cfg=CameraConfig(width=32, height=24, fps=30, fit_mode=fit),
            canvas_size=canvas,
        )
        try:
            cap.push(np.full((24, 32, 3), 11, dtype=np.uint8))
            _wait_for_frame(capture)
        finally:
            capture.close()

    assert "visual-policy upgrade note" not in caplog.text


def test_dynamic_delivered_size_is_fatal_in_error_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = _ControlledCap(property_size=(32, 24))
    capture = _install_camera(
        monkeypatch,
        cap,
        cfg=CameraConfig(
            width=32,
            height=24,
            fps=30,
            mode_mismatch="error",
        ),
        canvas_size=(40, 30),
    )
    try:
        cap.push(np.full((24, 32, 3), 11, dtype=np.uint8))
        assert _wait_for_frame(capture).shape == (30, 40, 3)
        cap.push(np.full((32, 16, 3), 99, dtype=np.uint8))

        error = _wait_for_error(capture, CaptureModeError)

        assert "changed during generation from 32x24 to 16x32" in str(error)
        assert capture.health_snapshot().frames_read == 1
    finally:
        capture.close()


def test_reconnect_replans_from_the_new_generation_delivered_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_cap = _FailAfterFramesCap(
        property_size=(32, 24),
        initial_frames=(np.full((24, 32, 3), 11, dtype=np.uint8),),
    )
    second_cap = _ControlledCap(
        property_size=(32, 24),
        initial_frames=(np.full((32, 16, 3), 99, dtype=np.uint8),),
    )
    monkeypatch.setattr(capture_mod, "cv2", _FakeCV2([first_cap, second_cap]))
    capture = OpenCVCapture(
        CameraConfig(
            width=32,
            height=24,
            fps=30,
            fit_mode="cover",
            mode_mismatch="warn",
        ),
        (40, 30),
    )
    try:
        first = _wait_for_frame(
            capture,
            predicate=lambda frame: int(frame[0, 0, 0]) == 11,
        )
        first_health = capture.health_snapshot()
        capture._stall_after_s = 0.03
        capture._recovery_timeout_s = 0.5
        capture._backoffs = (0.005,)

        second = _wait_for_frame(
            capture,
            predicate=lambda frame: int(frame[0, 0, 0]) == 99,
            timeout=0.8,
        )
        health = capture.health_snapshot()

        assert first.shape == second.shape == (30, 40, 3)
        assert first_health.generation == 1
        assert first_health.geometry_generation == 1
        assert health.generation == 2
        assert health.geometry_generation == 2
        assert health.restarts == 1
        assert health.camera_controls.generation == 2
        assert health.geometry_transitions == 2
        assert (health.delivered_width, health.delivered_height) == (16, 32)
        assert (health.normalized_width, health.normalized_height) == (40, 30)
        assert first_cap.released
    finally:
        capture.close()


def test_reconnect_identity_is_promoted_only_with_its_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_cap = _FailAfterFramesCap(
        property_size=(32, 24),
        initial_frames=(np.full((24, 32, 3), 11, dtype=np.uint8),),
    )
    second_cap = _ControlledCap(
        property_size=(32, 24),
        initial_frames=(np.full((32, 16, 3), 99, dtype=np.uint8),),
    )
    monkeypatch.setattr(capture_mod, "cv2", _FakeCV2([first_cap, second_cap]))
    capture = OpenCVCapture(
        CameraConfig(
            width=32,
            height=24,
            fps=30,
            fit_mode="cover",
            mode_mismatch="warn",
        ),
        (40, 30),
    )
    try:
        _wait_for_frame(
            capture,
            predicate=lambda frame: int(frame[0, 0, 0]) == 11,
        )
        first_health = capture.health_snapshot()
        capture._stall_after_s = 0.03
        capture._recovery_timeout_s = 0.5
        capture._backoffs = (0.005,)

        deadline = time.monotonic() + 0.8
        queued_generation = 0
        while time.monotonic() < deadline:
            capture._expire_recovery(time.monotonic())
            with capture._lock:
                queued_generation = capture._slot_generation
                queued = capture._slot_sequence != capture._delivered_sequence
            if queued_generation == 2 and queued:
                break
            time.sleep(0.002)
        assert queued_generation == 2

        # Negotiation and normalization for generation 2 have completed, but
        # status must still describe the generation-1 frame returned by read().
        before_promotion = capture.health_snapshot()
        assert before_promotion.generation == first_health.generation == 1
        assert before_promotion.camera_controls.generation == 1
        assert (
            before_promotion.delivered_width,
            before_promotion.delivered_height,
        ) == (32, 24)
        assert before_promotion.geometry_transitions == 1

        promoted = capture.read()
        assert promoted is not None and int(promoted[0, 0, 0]) == 99
        after_promotion = capture.health_snapshot()
        assert after_promotion.generation == 2
        assert after_promotion.camera_controls.generation == 2
        assert (
            after_promotion.delivered_width,
            after_promotion.delivered_height,
        ) == (16, 32)
        assert after_promotion.geometry_transitions == 2
    finally:
        capture.close()


def test_rotation_then_viewer_horizontal_mirror_occurs_before_slot_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = _ControlledCap(property_size=(16, 32))
    capture = _install_camera(
        monkeypatch,
        cap,
        cfg=CameraConfig(
            width=16,
            height=32,
            fps=30,
            fit_mode="stretch",
            rotation=90,
            mirror=True,
        ),
        canvas_size=(32, 16),
    )
    try:
        yy, xx = np.indices((32, 16))
        raw = np.stack((xx, yy, xx + yy), axis=-1).astype(np.uint8)
        cap.push(raw)

        frame = _wait_for_frame(capture)
        expected = np.ascontiguousarray(np.rot90(raw, -1)[:, ::-1])

        np.testing.assert_array_equal(frame, expected)
        health = capture.health_snapshot()
        assert (health.delivered_width, health.delivered_height) == (16, 32)
        assert (health.oriented_width, health.oriented_height) == (32, 16)
        assert (health.normalized_width, health.normalized_height) == (32, 16)
        assert health.generation == 1
        assert health.geometry_generation == 1
        assert health.content_rect == (0, 0, 32, 16)
    finally:
        capture.close()


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        (np.zeros((24, 32), dtype=np.uint8), "shape HxWx3"),
        (np.zeros((24, 32, 4), dtype=np.uint8), "shape HxWx3"),
        (np.zeros((24, 32, 3), dtype=np.float32), "dtype uint8"),
        (np.zeros((0, 32, 3), dtype=np.uint8), "dimensions must be positive"),
        ([[[0, 0, 0]]], "must be a numpy array"),
    ],
    ids=["gray", "bgra", "float", "zero-height", "non-array"],
)
def test_malformed_delivered_frames_are_rejected_before_slot_publish(
    monkeypatch: pytest.MonkeyPatch,
    malformed: object,
    message: str,
) -> None:
    cap = _ControlledCap(property_size=(32, 24))
    capture = _install_camera(
        monkeypatch,
        cap,
        cfg=CameraConfig(width=32, height=24, fps=30),
        canvas_size=(40, 30),
    )
    try:
        cap.push(malformed)
        error = _wait_for_error(capture, CaptureError)

        assert message in str(error)
        health = capture.health_snapshot()
        assert health.frames_read == 0
        assert health.normalized_width is None
    finally:
        capture.close()


def test_each_delivered_frame_is_transformed_exactly_once_before_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = _ControlledCap(property_size=(32, 24))
    real_transform = capture_mod.transform_frame
    calls: list[tuple[int, int]] = []

    def count_transform(frame: np.ndarray, *args: Any, **kwargs: Any):
        calls.append((frame.shape[1], frame.shape[0]))
        return real_transform(frame, *args, **kwargs)

    monkeypatch.setattr(capture_mod, "transform_frame", count_transform)
    capture = _install_camera(
        monkeypatch,
        cap,
        cfg=CameraConfig(width=32, height=24, fps=30),
        canvas_size=(40, 30),
    )
    try:
        cap.push(np.full((24, 32, 3), 1, dtype=np.uint8))
        assert int(_wait_for_frame(capture)[0, 0, 0]) == 1
        cap.push(np.full((24, 32, 3), 2, dtype=np.uint8))
        assert int(_wait_for_frame(capture)[0, 0, 0]) == 2

        assert calls == [(32, 24), (32, 24)]
        assert capture.health_snapshot().frames_read == 2
    finally:
        capture.close()


def test_synthetic_capture_uses_the_same_rotation_mirror_and_canvas_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_time = SimpleNamespace(monotonic=lambda: 100.0)
    monkeypatch.setattr(capture_mod, "time", fixed_time)
    real_apply = capture_mod.apply_transform
    calls = 0

    def count_apply(frame: np.ndarray, plan: Any) -> np.ndarray:
        nonlocal calls
        calls += 1
        return real_apply(frame, plan)

    monkeypatch.setattr(capture_mod, "apply_transform", count_apply)
    cfg = CameraConfig(
        synthetic=True,
        width=16,
        height=32,
        fps=30,
        fit_mode="stretch",
        rotation=90,
        mirror=True,
    )
    capture_source = capture_mod.open_capture(cfg, (32, 16))
    assert isinstance(capture_source, SyntheticCapture)
    capture = capture_source
    try:
        raw = capture._bg.copy()
        yy, xx = np.ogrid[: cfg.height, : cfg.width]
        ellipse = ((xx - cfg.width // 2) / (cfg.width * 0.14)) ** 2 + (
            (yy - cfg.height // 2) / (cfg.height * 0.3)
        ) ** 2 <= 1.0
        raw[ellipse] = (200, 190, 210)
        expected = apply_transform(raw, capture._plan)

        frame = capture.read()

        assert frame is not None
        np.testing.assert_array_equal(frame, expected)
        assert calls == 1
        health = capture.health_snapshot()
        assert (health.width, health.height) == (16, 32)
        assert (health.delivered_width, health.delivered_height) == (16, 32)
        assert (health.oriented_width, health.oriented_height) == (32, 16)
        assert (health.normalized_width, health.normalized_height) == (32, 16)
        assert health.camera_controls.as_dict() == {
            "policy": "preserve",
            "backend_family": "synthetic",
            "qualification": "not-applicable",
            "writes_performed": False,
            "generation": 1,
            "properties": {},
        }
    finally:
        capture.close()
