"""Preview tests with a stubbed cv2 GUI (real windows can't open in CI)."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import custback.preview as preview_mod
from custback.__main__ import build_parser, config_from_args
from custback.config import AppConfig, RuntimeConfig
from custback.hub import FrameHub


class FakeCV2:
    """Minimal highgui stand-in recording calls and scripting key presses."""

    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 4
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    error = type("error", (Exception,), {})

    def __init__(
        self, keys=(ord("q"),), fail_named_window=False, fail_wait_key=False
    ):
        self.keys = list(keys)
        self.fail_named_window = fail_named_window
        self.fail_wait_key = fail_wait_key
        self.shown: list = []
        self.destroyed = False

    def namedWindow(self, *_):
        if self.fail_named_window:
            raise self.error("no display")

    def imshow(self, _title, frame):
        self.shown.append(frame)

    def waitKey(self, _ms):
        if self.fail_wait_key:
            raise self.error("display disappeared")
        return self.keys.pop(0) if self.keys else -1

    def getWindowProperty(self, *_):
        return 1.0

    def putText(self, *_args, **_kwargs):
        pass

    def rectangle(self, *_args, **_kwargs):
        pass

    def addWeighted(self, *_args, **kwargs):
        pass

    def destroyWindow(self, _title):
        self.destroyed = True


def hub_with_frame():
    hub = FrameHub()
    hub.publish_output(np.zeros((72, 128, 3), np.uint8))
    return hub


def make_runtime(**bg) -> RuntimeConfig:
    return RuntimeConfig(AppConfig.from_dict({"background": bg}) if bg else AppConfig())


class ImmediateCoordinator:
    """Synchronous stand-in for the pipeline's serialized test boundary."""

    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime
        self.writer = runtime._coordinator_writer()
        self.patches = []

    def apply_config_patch(self, patch, timeout):
        self.patches.append((patch, timeout))
        state = self.runtime.read()
        candidate = state.config.patched(patch)
        if candidate != state.config:
            self.writer.commit(candidate, state.version)


def make_controller(runtime, backgrounds_dir=None):
    return preview_mod._PreviewController(
        runtime,
        backgrounds_dir=backgrounds_dir,
        coordinator=ImmediateCoordinator(runtime),
    )


def feed(hub, n=20, interval=0.01):
    def run():
        for _ in range(n):
            hub.publish_output(np.zeros((72, 128, 3), np.uint8))
            time.sleep(interval)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_cli_flag_sets_preview():
    args = build_parser().parse_args(["--preview", "--synthetic"])
    cfg = config_from_args(args)
    assert cfg.output.preview is True
    assert config_from_args(build_parser().parse_args([])).output.preview is False


def test_cli_populates_required_background_source_before_switching_mode():
    image = config_from_args(build_parser().parse_args(["--image", "missing.jpg"]))
    assert image.background.mode == "image"
    assert image.background.image_path == "missing.jpg"

    camera = config_from_args(build_parser().parse_args(["--bg-camera", "2"]))
    assert camera.background.mode == "camera"
    assert camera.background.camera_device == "2"

    with pytest.raises(ValueError, match="image_path"):
        config_from_args(build_parser().parse_args(["--mode", "image"]))


def test_preview_shows_frames_and_quits_on_q(monkeypatch):
    fake = FakeCV2(keys=[-1, ord("q")])
    monkeypatch.setattr(preview_mod, "cv2", fake)
    hub = hub_with_frame()
    stop = threading.Event()
    feed(hub)
    preview_mod.run_preview(make_runtime(), hub, stop)
    assert stop.is_set()          # quitting the preview stops the app
    assert len(fake.shown) >= 1
    assert fake.destroyed


def test_preview_stops_when_stop_event_set(monkeypatch):
    fake = FakeCV2(keys=[])       # user never presses a key
    monkeypatch.setattr(preview_mod, "cv2", fake)
    hub = hub_with_frame()
    stop = threading.Event()
    timer = threading.Timer(0.3, stop.set)
    timer.start()
    t0 = time.monotonic()
    preview_mod.run_preview(make_runtime(), hub, stop)
    assert time.monotonic() - t0 < 3.0
    timer.cancel()


def test_preview_headless_fallback(monkeypatch):
    """A late HighGUI error returns without blocking or raising."""
    fake = FakeCV2(fail_named_window=True)
    monkeypatch.setattr(preview_mod, "cv2", fake)
    stop = threading.Event()
    stop.set()                    # return immediately from the fallback wait
    preview_mod.run_preview(make_runtime(), hub_with_frame(), stop)
    assert fake.shown == []


def test_late_preview_failure_does_not_stop_pipeline(monkeypatch):
    fake = FakeCV2(fail_wait_key=True)
    monkeypatch.setattr(preview_mod, "cv2", fake)
    stop = threading.Event()
    result = preview_mod.run_preview(make_runtime(), hub_with_frame(), stop)
    assert result is False
    assert not stop.is_set()
    assert fake.destroyed


def test_preview_probe_rejects_headless_linux_without_spawning(monkeypatch):
    called = False
    monkeypatch.setattr(preview_mod, "cv2", object())

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(preview_mod.subprocess, "run", unexpected)
    available, reason = preview_mod.preview_available(
        environ={}, platform="linux", timeout=0.1
    )
    assert not available
    assert "DISPLAY" in reason
    assert not called


def test_preview_probe_contains_child_abort(monkeypatch):
    monkeypatch.setattr(preview_mod, "cv2", object())
    monkeypatch.setattr(
        preview_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=-6),
    )
    available, reason = preview_mod.preview_available(
        environ={"DISPLAY": ":99"}, platform="linux", timeout=0.1
    )
    assert not available
    assert "signal 6" in reason


def test_preview_probe_success(monkeypatch):
    monkeypatch.setattr(preview_mod, "cv2", object())
    monkeypatch.setattr(
        preview_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    assert preview_mod.preview_available(
        environ={"DISPLAY": ":0"}, platform="linux", timeout=0.1
    ) == (True, "")


class TestPreviewController:
    """Keyboard control logic, independent of the GUI event loop."""

    def test_mode_key_switches_mode(self):
        runtime = make_runtime(mode="passthrough")
        ctl = make_controller(runtime)
        ctl.handle_key(ord("1"))  # blur
        assert runtime.snapshot().background.mode == "blur"
        assert "blur" in ctl.current_message()

    def test_blur_keys_adjust_strength_and_clamp(self):
        runtime = make_runtime(mode="blur", blur_strength=145)
        ctl = make_controller(runtime)
        ctl.handle_key(ord("]"))
        assert runtime.snapshot().background.blur_strength == preview_mod.BLUR_MAX
        for _ in range(20):
            ctl.handle_key(ord("["))
        assert runtime.snapshot().background.blur_strength == preview_mod.BLUR_MIN

    def test_blur_key_notes_when_not_in_blur_mode(self):
        runtime = make_runtime(mode="color")
        ctl = make_controller(runtime)
        ctl.handle_key(ord("]"))
        assert "switch to blur mode" in ctl.current_message()

    def test_color_cycle_wraps(self):
        runtime = make_runtime(mode="color")
        ctl = make_controller(runtime)
        seen = set()
        for _ in range(len(preview_mod.COLOR_PRESETS)):
            ctl.handle_key(ord("n"))
            seen.add(tuple(runtime.snapshot().background.color))
        assert seen == set(preview_mod.COLOR_PRESETS)

    def test_image_mode_with_no_files_flashes_hint(self, tmp_path):
        runtime = make_runtime(mode="passthrough")
        ctl = make_controller(runtime, tmp_path)
        ctl.handle_key(ord("3"))  # image mode
        assert runtime.snapshot().background.mode == "passthrough"  # unchanged
        assert "no background image files" in ctl.current_message()

    def test_image_mode_picks_up_directory_file_and_cycles(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.png").write_bytes(b"")
        runtime = make_runtime(mode="passthrough")
        ctl = make_controller(runtime, tmp_path)
        ctl.handle_key(ord("3"))
        cfg = runtime.snapshot().background
        assert cfg.mode == "image"
        assert cfg.image_path.endswith("a.jpg")

        ctl.handle_key(ord("n"))
        assert runtime.snapshot().background.image_path.endswith("b.png")
        ctl.handle_key(ord("n"))  # wraps back around
        assert runtime.snapshot().background.image_path.endswith("a.jpg")

    def test_cycle_video_file_switches_mode_to_video(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.mp4").write_bytes(b"")
        runtime = make_runtime(mode="image", image_path=str(tmp_path / "a.jpg"))
        ctl = make_controller(runtime, tmp_path)
        ctl.handle_key(ord("n"))
        cfg = runtime.snapshot().background
        assert cfg.mode == "video"
        assert cfg.video_path.endswith("b.mp4")

    def test_camera_mode_without_device_flashes_hint(self):
        runtime = make_runtime(mode="passthrough")
        ctl = make_controller(runtime)
        ctl.handle_key(ord("5"))
        assert runtime.snapshot().background.mode == "passthrough"
        assert "camera_device" in ctl.current_message()

    def test_camera_mode_accepts_integer_zero_device(self):
        runtime = make_runtime(mode="passthrough", camera_device=0)
        ctl = make_controller(runtime)
        ctl.handle_key(ord("5"))
        assert runtime.snapshot().background.mode == "camera"

    def test_help_toggle(self):
        ctl = make_controller(make_runtime())
        assert ctl.show_help is False
        ctl.handle_key(ord("h"))
        assert ctl.show_help is True
        ctl.handle_key(ord("h"))
        assert ctl.show_help is False

    def test_message_expires(self, monkeypatch):
        ctl = make_controller(make_runtime())
        ctl.flash("hello")
        assert ctl.current_message() == "hello"
        future = time.monotonic() + preview_mod.MESSAGE_SECONDS + 1
        monkeypatch.setattr(preview_mod.time, "monotonic", lambda: future)
        assert ctl.current_message() is None

    def test_unmapped_key_is_noop(self):
        runtime = make_runtime(mode="passthrough")
        ctl = make_controller(runtime)
        ctl.handle_key(ord("z"))
        assert runtime.snapshot().background.mode == "passthrough"
        assert ctl.current_message() is None

    def test_updates_use_pipeline_coordinator(self):
        runtime = make_runtime(mode="passthrough")

        coordinator = ImmediateCoordinator(runtime)
        ctl = preview_mod._PreviewController(
            runtime, backgrounds_dir=None, coordinator=coordinator
        )
        ctl.handle_key(ord("1"))
        assert coordinator.patches == [({"background": {"mode": "blur"}}, 5.0)]
        assert runtime.snapshot().background.mode == "blur"


def test_preview_end_to_end_key_sequence(monkeypatch, tmp_path):
    """Full loop: press '1' (blur) then ']' (raise strength) then quit."""
    fake = FakeCV2(keys=[ord("1"), ord("]"), ord("q")])
    monkeypatch.setattr(preview_mod, "cv2", fake)
    hub = hub_with_frame()
    runtime = make_runtime(mode="passthrough", blur_strength=31)
    stop = threading.Event()
    feed(hub, n=10)
    preview_mod.run_preview(
        runtime,
        hub,
        stop,
        backgrounds_dir=tmp_path,
        coordinator=ImmediateCoordinator(runtime),
    )
    cfg = runtime.snapshot().background
    assert cfg.mode == "blur"
    assert cfg.blur_strength == 41
