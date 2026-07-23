"""Camera negotiation and bounded latest-frame worker tests."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import custback.capture as capture_mod
from custback.capture import (
    CaptureError,
    CaptureModeError,
    CaptureWorkerError,
    OpenCVCapture,
    SyntheticCapture,
)
from custback.config import CameraConfig


def fourcc(code: str) -> int:
    return sum(ord(char) << (8 * index) for index, char in enumerate(code))


class FakeCap:
    def __init__(
        self,
        *,
        backend="V4L2",
        width=128,
        height=72,
        fps=30.0,
        pixel_format="YUYV",
        locked=(),
        rejected=(),
        fail_forever=False,
        delay=0.003,
        frame=None,
    ):
        self.backend = backend
        self.values = {
            FakeCV2.CAP_PROP_FRAME_WIDTH: float(width),
            FakeCV2.CAP_PROP_FRAME_HEIGHT: float(height),
            FakeCV2.CAP_PROP_FPS: float(fps),
            FakeCV2.CAP_PROP_FOURCC: float(fourcc(pixel_format)),
            FakeCV2.CAP_PROP_BACKEND: float(FakeCV2.CAP_V4L2),
        }
        self.locked = set(locked)
        self.rejected = set(rejected)
        self.fail_forever = fail_forever
        self.delay = delay
        self.frame = (
            np.full((height, width, 3), 17, np.uint8) if frame is None else frame
        )
        self.set_calls = []
        self.read_calls = 0
        self.released = False

    def isOpened(self):
        return True

    def getBackendName(self):
        return self.backend

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if prop in self.rejected:
            return False
        if prop not in self.locked:
            self.values[prop] = float(value)
        return True

    def get(self, prop):
        return self.values.get(prop, 0.0)

    def read(self):
        time.sleep(self.delay)
        self.read_calls += 1
        if self.released or self.fail_forever:
            return False, None
        return True, self.frame.copy()

    def release(self):
        self.released = True


class BlockingCap(FakeCap):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.unblock = threading.Event()

    def read(self):
        self.entered.set()
        self.unblock.wait(2.0)
        return False, None

    def release(self):
        self.released = True
        self.unblock.set()


class UninterruptibleCap(FakeCap):
    """A pathological backend whose release does not wake a blocked read."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.unblock = threading.Event()

    def read(self):
        self.entered.set()
        self.unblock.wait(2.0)
        return False, None

    def release(self):
        self.released = True


class ClosedCap(FakeCap):
    def isOpened(self):
        return False


class SlowReleaseCap(FakeCap):
    def __init__(self):
        super().__init__(fail_forever=True, delay=0.001)
        self.release_entered = threading.Event()
        self.release_unblock = threading.Event()

    def release(self):
        self.released = True
        self.release_entered.set()
        self.release_unblock.wait(2.0)


class FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6
    CAP_PROP_BACKEND = 42
    CAP_V4L2 = 200
    CAP_MSMF = 1400
    CAP_DSHOW = 700

    def __init__(self, captures):
        self.captures = list(captures)
        self.opened = []

    def VideoCapture(self, device, apiPreference=None):
        assert self.captures, f"unexpected open for {device!r}"
        cap = self.captures.pop(0)
        self.opened.append((device, time.monotonic(), cap, apiPreference))
        return cap

    @staticmethod
    def VideoWriter_fourcc(*chars):
        return fourcc("".join(chars))

    @staticmethod
    def resize(frame, size):
        width, height = size
        return np.full((height, width, 3), frame[0, 0], np.uint8)

    @staticmethod
    def flip(frame, axis):
        return np.flip(frame, axis=axis).copy()


class BlockingOpenCV2(FakeCV2):
    def __init__(self, captures):
        super().__init__(captures)
        self.entered = threading.Event()
        self.unblock = threading.Event()

    def VideoCapture(self, device):
        self.entered.set()
        self.unblock.wait(2.0)
        return super().VideoCapture(device)


def wait_for_frame(capture, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = capture.read()
        if frame is not None:
            return frame
        time.sleep(0.003)
    raise AssertionError("capture produced no frame")


def wait_for_error(capture, error_type, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            capture.read()
        except error_type as exc:
            return exc
        time.sleep(0.003)
    raise AssertionError(f"capture did not raise {error_type.__name__}")


def test_v4l2_auto_requests_mjpg_before_dimensions_and_reads_back_mode(monkeypatch):
    cap = FakeCap(width=128, height=72, fps=30.0)
    fake_cv2 = FakeCV2([cap])
    monkeypatch.setattr(capture_mod, "cv2", fake_cv2)

    capture = OpenCVCapture(CameraConfig(device="0", width=128, height=72, fps=30))
    try:
        frame = wait_for_frame(capture)
        assert frame.shape == (72, 128, 3)
        assert fake_cv2.opened[0][0] == 0
        assert [prop for prop, _ in cap.set_calls] == [
            FakeCV2.CAP_PROP_FOURCC,
            FakeCV2.CAP_PROP_FRAME_WIDTH,
            FakeCV2.CAP_PROP_FRAME_HEIGHT,
            FakeCV2.CAP_PROP_FPS,
        ]
        health = capture.health_snapshot()
        assert health.backend == "V4L2"
        assert health.fourcc == "MJPG"
        assert (health.width, health.height, health.fps_reported) == (128, 72, 30.0)
        # Actual target attainment remains unknown until a full rate window.
        assert health.target_met is None
        assert health.frames_read >= 1
    finally:
        capture.close()
    assert cap.released
    assert not capture.health_snapshot().worker_alive


def test_first_successful_generation_is_not_logged_as_recovery(monkeypatch, caplog):
    cap = FakeCap()
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        wait_for_frame(capture)
    finally:
        capture.close()
    assert "camera capture recovered" not in caplog.text


def test_auto_preserves_non_v4l2_backend_pixel_format(monkeypatch):
    cap = FakeCap(backend="AVFOUNDATION")
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        wait_for_frame(capture)
        assert [prop for prop, _ in cap.set_calls] == [
            FakeCV2.CAP_PROP_FRAME_WIDTH,
            FakeCV2.CAP_PROP_FRAME_HEIGHT,
            FakeCV2.CAP_PROP_FPS,
        ]
        assert capture.health_snapshot().fourcc == "YUYV"
    finally:
        capture.close()


def test_backend_is_read_back_after_the_first_valid_frame(monkeypatch):
    cap = FakeCap(backend="V4L2")

    def backend_after_negotiation():
        return "V4L2" if cap.read_calls == 0 else "V4L2-negotiated"

    cap.getBackendName = backend_after_negotiation
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        wait_for_frame(capture)
        assert capture.health_snapshot().backend == "V4L2-negotiated"
    finally:
        capture.close()


def test_initial_open_failure_retries_with_backoff_and_recovers(monkeypatch):
    unavailable = ClosedCap()
    recovered = FakeCap(delay=0.002)
    fake_cv2 = FakeCV2([unavailable, recovered])
    monkeypatch.setattr(capture_mod, "cv2", fake_cv2)
    monkeypatch.setattr(OpenCVCapture, "_BACKOFFS", (0.01,))

    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        frame = wait_for_frame(capture, timeout=0.5)
        assert frame.shape == (72, 128, 3)
        health = capture.health_snapshot()
        assert health.restarts == 1
        assert health.read_failures == 1
        assert unavailable.released
    finally:
        capture.close()


def test_auto_retries_once_when_v4l2_mjpg_does_not_stick(monkeypatch, caplog):
    first = FakeCap(locked={FakeCV2.CAP_PROP_FOURCC})
    recovered = FakeCap()
    fake_cv2 = FakeCV2([first, recovered])
    monkeypatch.setattr(capture_mod, "cv2", fake_cv2)
    monkeypatch.setattr(OpenCVCapture, "_BACKOFFS", (0.01,))

    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        wait_for_frame(capture, timeout=0.5)
        health = capture.health_snapshot()
        assert health.fourcc == "MJPG"
        assert health.restarts == 1
        assert health.read_failures == 1
        assert "retrying once" in caplog.text
    finally:
        capture.close()
    assert first.released
    assert len(fake_cv2.opened) == 2


def test_auto_accepts_backend_format_after_one_failed_mjpg_retry(monkeypatch, caplog):
    locked = {FakeCV2.CAP_PROP_FOURCC}
    first = FakeCap(locked=locked)
    fallback = FakeCap(locked=locked)
    fake_cv2 = FakeCV2([first, fallback])
    monkeypatch.setattr(capture_mod, "cv2", fake_cv2)
    monkeypatch.setattr(OpenCVCapture, "_BACKOFFS", (0.01,))

    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        wait_for_frame(capture, timeout=0.5)
        assert capture.health_snapshot().fourcc == "YUYV"
        assert "pixel-format fallback" in caplog.text
    finally:
        capture.close()
    assert len(fake_cv2.opened) == 2


def test_blocking_native_open_does_not_block_read_or_outage_deadline(monkeypatch):
    fake_cv2 = BlockingOpenCV2([FakeCap()])
    monkeypatch.setattr(capture_mod, "cv2", fake_cv2)
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    capture._recovery_timeout_s = 0.05
    try:
        assert fake_cv2.entered.wait(0.2)
        started = time.monotonic()
        assert capture.read() is None
        assert time.monotonic() - started < 0.02
        error = wait_for_error(capture, CaptureError, timeout=0.3)
        assert "0.05 seconds" in str(error)
    finally:
        # The fake models a native open that ignores its timeout.  Release it so
        # controller ownership can finish and the test process has no survivor.
        fake_cv2.unblock.set()
        capture.close()


def test_negotiated_resolution_uses_post_frame_property_readback(monkeypatch, caplog):
    requested_frame = np.full((72, 128, 3), 23, np.uint8)
    cap = FakeCap(
        width=64,
        height=48,
        frame=requested_frame,
        locked={FakeCV2.CAP_PROP_FRAME_WIDTH, FakeCV2.CAP_PROP_FRAME_HEIGHT},
    )
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        frame = wait_for_frame(capture)
        assert frame.shape == (72, 128, 3)
        health = capture.health_snapshot()
        assert (health.width, health.height) == (64, 48)
        assert "negotiated MJPG 64x48" in caplog.text
    finally:
        capture.close()


def test_explicit_mjpeg_rejection_closes_handle(monkeypatch):
    cap = FakeCap(rejected={FakeCV2.CAP_PROP_FOURCC})
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(
        CameraConfig(width=128, height=72, fps=30, pixel_format="mjpeg")
    )
    try:
        error = wait_for_error(capture, CaptureModeError)
        assert "rejected explicit MJPG" in str(error)
    finally:
        capture.close()
    assert cap.released


def test_warn_mode_reports_negotiated_mismatch_and_resizes(monkeypatch, caplog):
    locked = {
        FakeCV2.CAP_PROP_FRAME_WIDTH,
        FakeCV2.CAP_PROP_FRAME_HEIGHT,
        FakeCV2.CAP_PROP_FPS,
    }
    cap = FakeCap(width=64, height=48, fps=10.0, locked=locked)
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        frame = wait_for_frame(capture)
        health = capture.health_snapshot()
        assert frame.shape == (72, 128, 3)
        assert (health.width, health.height, health.fps_reported) == (64, 48, 10.0)
        # Negotiated mismatch is reported separately; this field represents
        # measured runtime rate and remains unknown during warm-up.
        assert health.target_met is None
        assert "camera mode mismatch" in caplog.text
    finally:
        capture.close()


def test_error_mode_fails_on_negotiated_mismatch(monkeypatch):
    locked = {
        FakeCV2.CAP_PROP_FRAME_WIDTH,
        FakeCV2.CAP_PROP_FRAME_HEIGHT,
        FakeCV2.CAP_PROP_FPS,
    }
    cap = FakeCap(width=64, height=48, fps=10.0, locked=locked)
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(
        CameraConfig(width=128, height=72, fps=30, mode_mismatch="error")
    )
    try:
        error = wait_for_error(capture, CaptureModeError)
        assert "resolution 64x48" in str(error)
    finally:
        capture.close()


def test_explicit_mjpeg_must_be_verified_after_first_frame(monkeypatch):
    cap = FakeCap(
        pixel_format="YUYV",
        locked={FakeCV2.CAP_PROP_FOURCC},
    )
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(
        CameraConfig(width=128, height=72, fps=30, pixel_format="mjpeg")
    )
    try:
        error = wait_for_error(capture, CaptureModeError)
        assert "did not activate explicit MJPG" in str(error)
    finally:
        capture.close()


def test_latest_frame_slot_counts_overwrites_without_backlog(monkeypatch):
    cap = FakeCap(delay=0.001)
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            health = capture.health_snapshot()
            if health.frames_read >= 8:
                break
            time.sleep(0.003)
        else:
            raise AssertionError("reader did not fill latest-frame slot")
        assert health.dropped_frames >= 7
        assert capture.read() is not None
        assert capture.read() is None
    finally:
        capture.close()


def test_stalled_capture_reopens_once_and_recovers(monkeypatch):
    stalled = FakeCap(fail_forever=True, delay=0.001)
    recovered = FakeCap(delay=0.002)
    fake_cv2 = FakeCV2([stalled, recovered])
    monkeypatch.setattr(capture_mod, "cv2", fake_cv2)
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    capture._stall_after_s = 0.03
    capture._recovery_timeout_s = 0.5
    capture._backoffs = (0.01, 0.02, 0.04, 0.08)
    try:
        frame = wait_for_frame(capture, timeout=0.5)
        assert frame.shape == (72, 128, 3)
        health = capture.health_snapshot()
        assert health.restarts == 1
        assert health.read_failures > 0
        assert health.stalled is False
        assert stalled.released
        assert len(fake_cv2.opened) == 2
        assert fake_cv2.opened[1][1] - fake_cv2.opened[0][1] >= 0.035
    finally:
        capture.close()


def test_reconnect_backoff_sequence_is_exact_and_saturates(monkeypatch):
    cap = FakeCap()
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        delays = []
        with capture._lock:
            capture._backoff_index = 0
            for now in (10.0, 20.0, 30.0, 40.0, 50.0):
                capture._schedule_reopen_locked(now)
                next_reopen_at = capture._next_reopen_at
                assert next_reopen_at is not None
                delays.append(next_reopen_at - now)
        assert delays == [0.5, 1.0, 2.0, 4.0, 4.0]
    finally:
        capture.close()


def test_terminal_outage_uses_the_configured_deadline(monkeypatch):
    stalled = FakeCap(fail_forever=True, delay=0.001)
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([stalled]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    capture._stall_after_s = 0.01
    capture._recovery_timeout_s = 0.08
    capture._backoffs = (0.2,)
    try:
        error = wait_for_error(capture, CaptureError, timeout=0.4)
        assert "0.08 seconds" in str(error)
    finally:
        capture.close()


def test_slow_release_does_not_block_latest_slot_reads(monkeypatch):
    cap = SlowReleaseCap()
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    capture._stall_after_s = 0.01
    capture._recovery_timeout_s = 0.06
    try:
        assert cap.release_entered.wait(0.3)
        started = time.monotonic()
        assert capture.read() is None
        assert time.monotonic() - started < 0.02
        error = wait_for_error(capture, CaptureError, timeout=0.3)
        assert "0.06 seconds" in str(error)
    finally:
        cap.release_unblock.set()
        capture.close()


def test_repeated_no_frame_recovery_warning_is_deduplicated(monkeypatch, caplog):
    captures = [FakeCap(fail_forever=True, delay=0.001) for _ in range(12)]
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2(captures))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    capture._stall_after_s = 0.01
    capture._recovery_timeout_s = 0.12
    capture._backoffs = (0.002,)
    try:
        wait_for_error(capture, CaptureError, timeout=0.4)
    finally:
        capture.close()
    assert caplog.text.count("reopened camera still produced no frame") == 1


def test_shutdown_during_reconnect_backoff_does_not_reopen(monkeypatch):
    stalled = FakeCap(fail_forever=True, delay=0.001)
    unused = FakeCap()
    fake_cv2 = FakeCV2([stalled, unused])
    monkeypatch.setattr(capture_mod, "cv2", fake_cv2)
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    capture._stall_after_s = 0.01
    capture._backoffs = (0.2,)
    deadline = time.monotonic() + 0.2
    while not capture.health_snapshot().stalled and time.monotonic() < deadline:
        capture.read()
        time.sleep(0.002)
    assert capture.health_snapshot().stalled
    capture.close()
    time.sleep(0.02)
    assert len(fake_cv2.opened) == 1
    assert not capture.health_snapshot().worker_alive


def test_strict_low_rate_requires_a_continuous_low_period(monkeypatch):
    cap = FakeCap(fps=240.0, delay=0.006)
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(
        CameraConfig(
            width=128,
            height=72,
            fps=240,
            mode_mismatch="error",
        )
    )
    capture._RATE_WINDOW_S = 0.02
    capture._RATE_WARNING_AFTER_S = 0.05
    try:
        # A low first window starts the duration clock; it must not fail yet.
        time.sleep(0.035)
        assert capture.read() is not None
        error = wait_for_error(capture, CaptureModeError, timeout=0.3)
        assert "capture rate" in str(error)
    finally:
        capture.close()


def test_close_releases_blocking_reader_and_joins_it(monkeypatch):
    cap = BlockingCap()
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    assert cap.entered.wait(0.5)
    capture.close()
    assert cap.released
    assert not capture.health_snapshot().worker_alive


def test_reader_surviving_release_and_join_is_terminal(monkeypatch):
    cap = UninterruptibleCap()
    monkeypatch.setattr(capture_mod, "cv2", FakeCV2([cap]))
    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    capture._stall_after_s = 0.01
    capture._READER_JOIN_TIMEOUT_S = 0.02
    try:
        assert cap.entered.wait(0.2)
        error = wait_for_error(capture, CaptureWorkerError, timeout=0.3)
        assert "remained blocked" in str(error)
        deadline = time.monotonic() + 0.2
        controller = capture._controller_thread
        while (
            controller is not None
            and controller.is_alive()
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        assert controller is None or not controller.is_alive()
    finally:
        # Native Python threads cannot be force-cancelled safely.  Production
        # exits the failed pipeline/process; unblock only this fake so the test
        # interpreter can verify deterministic final cleanup.
        cap.unblock.set()
        capture.close()
    assert not capture.health_snapshot().worker_alive


def _force_windows_platform(monkeypatch):
    # Drives both the MSMF/DSHOW backend order and the privacy-denial hint,
    # which read sys.platform through the camera_devices module.
    import custback.camera_devices as camera_devices_mod

    monkeypatch.setattr(camera_devices_mod.sys, "platform", "win32")


def test_windows_capture_selects_msmf_then_dshow(monkeypatch):
    _force_windows_platform(monkeypatch)
    msmf_closed = ClosedCap()  # MSMF cannot open
    dshow_ok = FakeCap(delay=0.002)  # DSHOW opens
    fake_cv2 = FakeCV2([msmf_closed, dshow_ok])
    monkeypatch.setattr(capture_mod, "cv2", fake_cv2)

    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        frame = wait_for_frame(capture, timeout=0.5)
        assert frame.shape == (72, 128, 3)
        assert msmf_closed.released
        # Both backends were opened explicitly, in order, with apiPreference set.
        assert fake_cv2.opened[0][3] == FakeCV2.CAP_MSMF
        assert fake_cv2.opened[1][3] == FakeCV2.CAP_DSHOW
    finally:
        capture.close()


def test_windows_total_open_failure_logs_privacy_hint(monkeypatch, caplog):
    _force_windows_platform(monkeypatch)

    class AlwaysClosedCV2(FakeCV2):
        def VideoCapture(self, device, apiPreference=None):
            cap = ClosedCap()
            self.opened.append((device, time.monotonic(), cap, apiPreference))
            return cap

    monkeypatch.setattr(capture_mod, "cv2", AlwaysClosedCV2([]))
    monkeypatch.setattr(OpenCVCapture, "_BACKOFFS", (0.01,))

    capture = OpenCVCapture(CameraConfig(width=128, height=72, fps=30))
    try:
        with caplog.at_level("WARNING"):
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline and "Privacy" not in caplog.text:
                time.sleep(0.01)
        assert "Privacy" in caplog.text and "Camera" in caplog.text
    finally:
        capture.close()


def test_synthetic_capture_exposes_compatible_health_and_mirror():
    cfg = CameraConfig(synthetic=True, width=64, height=32, fps=30, mirror=True)
    capture = SyntheticCapture(cfg)
    try:
        frame = capture.read()
        assert frame is not None
        assert frame.shape == (32, 64, 3)
        health = capture.health_snapshot()
        assert health.backend == "synthetic"
        assert health.target_met is None
        assert health.frames_read == 1
        assert health.frame_age_ms is not None
    finally:
        capture.close()
