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
from custback.matte_diagnostics import MatteCaptureMetadata, MatteFrameEvidence
from custback.matte_live_diagnostics import (
    LocalMatteDiagnosticFrame,
    LocalMatteDiagnosticMonitor,
    MatteTemporalTelemetry,
    diagnostic_view_names,
)


class FakeCV2:
    """Minimal highgui stand-in recording calls and scripting key presses."""

    WINDOW_NORMAL = 0
    WND_PROP_VISIBLE = 4
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    error = type("error", (Exception,), {})

    def __init__(self, keys=(ord("q"),), fail_named_window=False, fail_wait_key=False):
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

    def apply_config_patch(self, patch, timeout, *, origin="internal"):
        self.patches.append((patch, timeout, origin))
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


def make_diagnostic_frame() -> LocalMatteDiagnosticFrame:
    return LocalMatteDiagnosticFrame(
        view="refined_alpha",
        label="Refined alpha",
        interpretation="exact post-refiner/stabilizer alpha",
        pixels=np.full((72, 128, 3), 42, np.uint8),
        available=True,
        unavailable_reason="",
        capture_sequence=11,
        capture_monotonic_ns=2_066_670_000,
        capture_generation=3,
        geometry_generation=4,
        temporal=MatteTemporalTelemetry(
            capture_sequence_delta=2,
            capture_timestamp_delta_ms=66.67,
            raw_alpha_abs_diff=0.2,
            raw_alpha_compensated_abs_diff=0.1,
            refined_alpha_abs_diff=0.08,
            refined_alpha_compensated_abs_diff=0.04,
            edge_colour_abs_diff=0.03,
            edge_colour_state="ready",
            registration_dx_px=1.25,
            registration_dy_px=-0.5,
            registration_response=0.92,
            registration_overlap_fraction=0.97,
            registration_state="ready",
            history_state="ready",
        ),
        effective_controls={
            "rvm_downsample_ratio": 0.25,
            "mask_shift": 2,
            "use_model_foreground": True,
            "light_wrap": 0.3,
        },
        timings_ms={
            "backend_inference_ms": 7.1,
            "rvm_preprocess_ms": 0.2,
            "rvm_session_run_ms": 6.3,
            "rvm_postprocess_ms": 0.4,
            "refinement_ms": 1.2,
            "segmentation_ms": 8.3,
            "background_ms": 0.8,
            "color_correction_ms": 0.6,
            "composite_prepare_ms": 0.4,
            "composite_blend_ms": 0.7,
            "composite_ms": 1.1,
            "post_composite_validation_ms": 0.05,
        },
        compositor_substages_ms={
            "input_mask_validation": 0.01,
            "edge_band": 0.02,
            "model_foreground_replacement": 0.03,
            "backdrop_blur_resize": 0.04,
            "light_wrap_temporal_filter": 0.05,
            "light_wrap_interpolation": 0.06,
            "final_blend_conversion": 0.07,
            "internal_output_validation": 0.08,
        },
        segmentation_diagnostics={},
        status={},
    )


def active_diagnostic_monitor() -> LocalMatteDiagnosticMonitor:
    monitor = LocalMatteDiagnosticMonitor()
    monitor.select("raw_camera")
    source = np.full((12, 16, 3), 17, np.uint8)
    assert monitor.submit(
        MatteFrameEvidence(
            metadata=MatteCaptureMetadata(
                bundle_sequence=1,
                capture_sequence=1,
                capture_monotonic_ns=33_333_333,
                timestamp_source="test",
                capture_generation=0,
                geometry_generation=0,
            ),
            raw_frame=source,
        ),
        status={"segmentation_generation": 0, "matte_reset_count": 0},
    )
    frame, _sequence = monitor.get(timeout=2.0)
    assert frame is not None
    return monitor


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


def test_cli_sets_renderer_token_file():
    cfg = config_from_args(
        build_parser().parse_args(
            ["--renderer-token-file", "/tmp/custback-renderer-token"]
        )
    )
    assert cfg.api.renderer_token_file == "/tmp/custback-renderer-token"


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
    assert stop.is_set()  # quitting the preview stops the app
    assert len(fake.shown) >= 1
    assert fake.destroyed


def test_preview_stops_when_stop_event_set(monkeypatch):
    fake = FakeCV2(keys=[])  # user never presses a key
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
    stop.set()  # return immediately from the fallback wait
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


@pytest.mark.parametrize(
    ("fake", "expected_destroyed"),
    [
        (FakeCV2(keys=[ord("q")]), True),
        (FakeCV2(fail_wait_key=True), True),
        (FakeCV2(fail_named_window=True), False),
    ],
    ids=["quit", "late-gui-failure", "window-open-failure"],
)
def test_preview_exit_deactivates_and_clears_diagnostic_monitor(
    monkeypatch,
    fake,
    expected_destroyed,
):
    monitor = active_diagnostic_monitor()
    assert monitor.accepting
    assert monitor.latest() is not None
    monkeypatch.setattr(preview_mod, "cv2", fake)

    try:
        preview_mod.run_preview(
            make_runtime(),
            hub_with_frame(),
            threading.Event(),
            matte_monitor=monitor,
        )

        assert monitor.selected_view is None
        assert not monitor.accepting
        assert monitor.latest() is None
        assert fake.destroyed is expected_destroyed
    finally:
        monitor.close()


def test_return_to_output_repaints_cached_production_frame_when_hub_stalls(
    monkeypatch,
):
    production = np.full((24, 32, 3), 31, dtype=np.uint8)
    diagnostic_pixels = np.full_like(production, 219)
    diagnostic = SimpleNamespace(
        view="raw_camera",
        pixels=diagnostic_pixels,
        status={},
    )

    class ScriptedMonitor:
        selected_view = None

        def select_next(self, *, reverse=False):
            del reverse
            self.selected_view = "raw_camera" if self.selected_view is None else None
            return self.selected_view

        def get(self, last_sequence, timeout=None):
            del last_sequence, timeout
            return diagnostic, 1

        def deactivate(self):
            self.selected_view = None

    monitor = ScriptedMonitor()
    fake = FakeCV2(keys=[ord("d"), ord("D"), ord("q")])
    monkeypatch.setattr(preview_mod, "cv2", fake)
    monkeypatch.setattr(
        preview_mod,
        "_draw_overlay",
        lambda frame, *_args, **_kwargs: frame.copy(),
    )
    hub = FrameHub()
    hub.publish_output(production)

    preview_mod.run_preview(
        make_runtime(),
        hub,
        threading.Event(),
        matte_monitor=monitor,  # type: ignore[arg-type]
    )

    assert len(fake.shown) >= 3
    np.testing.assert_array_equal(fake.shown[0], production)
    np.testing.assert_array_equal(fake.shown[1], diagnostic_pixels)
    np.testing.assert_array_equal(fake.shown[2], production)


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


def test_highgui_env_removes_only_opencv_injected_missing_font_dir(tmp_path):
    missing = tmp_path / "cv2" / "qt" / "fonts"
    normalized, reason = preview_mod.normalize_highgui_environment(
        environ={"DISPLAY": ":0", "QT_QPA_FONTDIR": str(missing)},
        inherited_environ={},
        platform="linux",
        cv2_module=object(),
    )
    assert reason == ""
    assert "QT_QPA_FONTDIR" not in normalized

    preserved, reason = preview_mod.normalize_highgui_environment(
        environ={"DISPLAY": ":0", "QT_QPA_FONTDIR": str(missing)},
        inherited_environ={"QT_QPA_FONTDIR": str(missing)},
        platform="linux",
        cv2_module=object(),
    )
    assert reason == ""
    assert preserved["QT_QPA_FONTDIR"] == str(missing)


def test_highgui_env_recognizes_cv2_font_default_imported_earlier(tmp_path):
    package = tmp_path / "cv2"
    missing = package / "qt" / "fonts"
    module = SimpleNamespace(__file__=str(package / "__init__.py"))
    normalized, reason = preview_mod.normalize_highgui_environment(
        environ={"DISPLAY": ":0", "QT_QPA_FONTDIR": str(missing)},
        # Simulate backgrounds.py having imported cv2 before preview.py.
        inherited_environ={"QT_QPA_FONTDIR": str(missing)},
        platform="linux",
        cv2_module=module,
    )
    assert reason == ""
    assert "QT_QPA_FONTDIR" not in normalized


def test_highgui_env_selects_xcb_for_xwayland(monkeypatch):
    monkeypatch.setattr(
        preview_mod, "_qt_platform_plugins", lambda *_args, **_kwargs: {"xcb"}
    )
    normalized, reason = preview_mod.normalize_highgui_environment(
        environ={
            "DISPLAY": ":0",
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_SESSION_TYPE": "wayland",
        },
        inherited_environ={},
        platform="linux",
        cv2_module=object(),
    )
    assert reason == ""
    assert normalized["QT_QPA_PLATFORM"] == "xcb"


def test_highgui_env_does_not_force_xcb_when_wayland_plugin_exists(monkeypatch):
    monkeypatch.setattr(
        preview_mod,
        "_qt_platform_plugins",
        lambda *_args, **_kwargs: {"xcb", "wayland"},
    )
    normalized, reason = preview_mod.normalize_highgui_environment(
        environ={
            "DISPLAY": ":0",
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_SESSION_TYPE": "wayland",
        },
        inherited_environ={},
        platform="linux",
        cv2_module=object(),
    )
    assert reason == ""
    assert "QT_QPA_PLATFORM" not in normalized


def test_highgui_env_preserves_explicit_platform(monkeypatch):
    monkeypatch.setattr(
        preview_mod, "_qt_platform_plugins", lambda *_args, **_kwargs: {"xcb"}
    )
    normalized, reason = preview_mod.normalize_highgui_environment(
        environ={
            "DISPLAY": ":0",
            "WAYLAND_DISPLAY": "wayland-0",
            "QT_QPA_PLATFORM": "minimal",
        },
        inherited_environ={"QT_QPA_PLATFORM": "minimal"},
        platform="linux",
        cv2_module=object(),
    )
    assert reason == ""
    assert normalized["QT_QPA_PLATFORM"] == "minimal"


def test_highgui_env_rejects_wayland_only_without_plugin(monkeypatch):
    monkeypatch.setattr(
        preview_mod, "_qt_platform_plugins", lambda *_args, **_kwargs: {"xcb"}
    )
    _normalized, reason = preview_mod.normalize_highgui_environment(
        environ={"WAYLAND_DISPLAY": "wayland-0"},
        inherited_environ={},
        platform="linux",
        cv2_module=object(),
    )
    assert "browser preview" in reason
    assert "XWayland" in reason


def test_preview_probe_reports_deduplicated_bounded_stderr(monkeypatch):
    monkeypatch.setattr(preview_mod, "cv2", object())
    warning = "QFontDatabase: missing fonts"
    stderr = ((warning + "\n") * 100 + "x" * 5000).encode()
    monkeypatch.setattr(
        preview_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr=stderr),
    )
    available, reason = preview_mod.preview_available(
        environ={"DISPLAY": ":0"}, platform="linux", timeout=0.1
    )
    assert not available
    assert reason.count(warning) == 1
    assert len(reason) <= preview_mod.PROBE_STDERR_LIMIT + 80


def test_qt_bootstrap_filter_suppresses_only_known_noise():
    captured = (
        b"Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use xcb.\n"
        b"QFontDatabase: Cannot find font directory /missing/fonts.\n"
        b"Note that Qt no longer ships fonts. Deploy some or use fontconfig.\n"
        b"QFontDatabase: Cannot find font directory /missing/fonts.\n"
        b"real late HighGUI error\n"
    )
    remaining, suppressed = preview_mod._filter_qt_bootstrap_stderr(captured)
    assert suppressed == 4
    assert remaining == b"real late HighGUI error\n"


def test_named_window_filter_replays_unknown_stderr(monkeypatch, capfd):
    class NoisyCV2(FakeCV2):
        def namedWindow(self, *_):
            import os

            os.write(
                2,
                b"QFontDatabase: Cannot find font directory /missing/fonts.\n"
                b"real late HighGUI error\n",
            )

    monkeypatch.setattr(preview_mod, "cv2", NoisyCV2())
    preview_mod._named_window_with_filtered_qt_stderr()
    assert capfd.readouterr().err == "real late HighGUI error\n"


def test_status_overlay_shows_actual_backends_cpu_and_fallbacks():
    status, warnings = preview_mod._status_overlay_lines(
        {
            "mode": "video",
            "capture_fps": 9.8,
            "fps": 10.0,
            "capture_target_fps": 30,
            "output_target_fps": 30,
            "capture_width": 1280,
            "capture_height": 720,
            "capture_fourcc": "YUYV",
            "capture_fps_reported": 10.0,
            "capture_backend": "V4L2",
            "segmentation_backend": "RVMSegmenter",
            "segmentation_device": "cpu",
            "output_backend": "NullOutput",
            "config_version": 4,
            "capture_target_met": False,
            "output_fallback_active": True,
            "output_fallback_reason": "virtual-camera-unavailable",
            "remote_fallback_active": True,
            "remote_fallback_reason": "stale",
            "background_video_source_fps": 24.0,
            "background_video_timing_mode": "clocked",
            "background_video_frames_displayed": 100,
            "background_video_skip_ratio": 0.58,
        }
    )
    rendered = "\n".join(status)
    warning_text = "\n".join(warnings)
    assert "IN 9.8/30" in rendered
    assert "OUT 10.0/30" in rendered
    assert "rvm/cpu" in rendered
    assert "NullOutput" in rendered
    assert "CONFIG v4" in rendered
    assert "1280x720" in rendered and "YUYV" in rendered
    assert "skip 58%" in rendered
    assert "CAPTURE BELOW TARGET" in warning_text
    assert "OUTPUT FALLBACK" in warning_text
    assert "REMOTE FALLBACK" in warning_text


def test_status_overlay_shows_structured_backend_downgrade_and_policy():
    status, warnings = preview_mod._status_overlay_lines(
        {
            "segmentation_backend": "MediaPipeSegmenter",
            "segmentation_device": "cpu",
            "output_backend": "NullOutput",
            "segmentation_selection": {
                "requested_backend": "auto",
                "selected_backend": "mediapipe",
                "quality_tier": "segmentation",
                "selection_mode": "automatic",
                "fallback_active": True,
                "fallback_category": "runtime-not-installed",
                "fallback_reason": "RVM unavailable: runtime not installed",
                "guidance": "Install the RVM runtime profile and restart.",
                "active_device": "cpu",
                "active_provider": "cpu",
                "attempts": [],
            },
            "matte_policy": {
                "effective": {
                    "raw_alpha_mode": "confidence_soft_mask",
                    "rvm_downsample_ratio": None,
                    "edge_refine": True,
                    "residual_temporal_mode": "explicit_motion_aware",
                    "light_wrap": 0.15,
                }
            },
        }
    )

    rendered = "\n".join(status)
    assert (
        "SEG mediapipe/cpu  TIER segmentation  PROVIDER cpu  SELECT automatic"
    ) in rendered
    assert (
        "MATTE ALPHA confidence_soft_mask  EDGE on  "
        "TEMPORAL explicit_motion_aware  LIGHT WRAP 0.15"
    ) in rendered
    assert warnings == [
        "SEGMENTATION FALLBACK [runtime-not-installed]: "
        "RVM unavailable: runtime not installed",
        "SEGMENTATION ACTION: Install the RVM runtime profile and restart.",
    ]


def test_status_overlay_does_not_warn_for_explicit_mediapipe():
    _status, warnings = preview_mod._status_overlay_lines(
        {
            # Structured selection is authoritative over the legacy flag.
            "segmentation_fallback_active": True,
            "segmentation_fallback_reason": "legacy-misclassification",
            "segmentation_selection": {
                "requested_backend": "mediapipe",
                "selected_backend": "mediapipe",
                "quality_tier": "segmentation",
                "selection_mode": "explicit",
                "fallback_active": False,
                "fallback_category": "none",
                "fallback_reason": "",
                "guidance": "",
                "active_device": "cpu",
                "active_provider": "cpu",
                "attempts": [],
            },
        }
    )

    assert warnings == []


@pytest.mark.parametrize(
    ("backend", "tier", "device", "provider"),
    [
        ("rvm", "matting", "cuda", "cuda"),
        ("mediapipe", "segmentation", "cpu", "cpu"),
        ("heuristic", "heuristic", "cpu", "cpu"),
        ("none", "none", "none", "none"),
    ],
)
def test_status_overlay_covers_every_backend_quality_tier(
    backend,
    tier,
    device,
    provider,
):
    status, warnings = preview_mod._status_overlay_lines(
        {
            "segmentation_selection": {
                "requested_backend": backend,
                "selected_backend": backend,
                "quality_tier": tier,
                "selection_mode": "explicit",
                "fallback_active": False,
                "fallback_category": "none",
                "fallback_reason": "",
                "guidance": "",
                "active_device": device,
                "active_provider": provider,
                "attempts": [],
            }
        }
    )

    assert (
        f"SEG {backend}/{device}  TIER {tier}  PROVIDER {provider}  SELECT explicit"
    ) in status[1]
    assert warnings == []


def test_status_overlay_marks_stalled_capture_age():
    _status, warnings = preview_mod._status_overlay_lines(
        {"capture_stalled": True, "capture_frame_age_ms": 2450.0}
    )
    assert warnings == ["CAPTURE STALLED (2450 ms since last frame)"]


def test_status_overlay_distinguishes_visual_updates_from_output_sends():
    status, warnings = preview_mod._status_overlay_lines(
        {
            "base_composite_update_fps": 15.0,
            "segmentation_update_fps": 14.9,
            "output_send_fps": 29.7,
            "base_composite_reuse_ratio": 564 / 1105,
            "exact_final_output_repeat_ratio": 564 / 1104,
            "cadence_mismatch_active": True,
        }
    )

    assert status == [
        "starting  IN -- fps  OUT -- fps",
        "SEG unknown/unknown  OUTPUT unknown  CONFIG v0",
        "CADENCE VIS 15.0  SEG 14.9  SEND 29.7  BASE REUSE 51%  EXACT FINAL REPEAT 51%",
    ]
    assert warnings == ["VISUAL UPDATES 15 FPS; OUTPUT REPEATS TO 30 FPS"]


def test_status_overlay_does_not_infer_mismatch_from_partial_cadence():
    status, warnings = preview_mod._status_overlay_lines(
        {
            "base_composite_update_fps": 15.0,
            "output_send_fps": 29.7,
            "cadence_mismatch_active": False,
        }
    )

    assert status[-1] == "CADENCE VIS 15.0  SEND 29.7"
    assert warnings == []


def test_diagnostic_overlay_formats_frame_and_public_telemetry():
    diagnostic = make_diagnostic_frame()
    lines, warnings = preview_mod._diagnostic_overlay_lines(
        diagnostic,
        selected_view="refined_alpha",
        stats={
            "matte_policy": {
                "effective": {
                    "rvm_downsample_ratio": 0.25,
                    "mask_shift": 2,
                    "use_model_foreground": True,
                    "light_wrap": 0.3,
                },
                "blend_space": "linear_srgb",
            },
            "matte_reset_count": 5,
            "matte_last_reset_reason": "capture_generation_changed",
            "timing_ms": {
                "segmentation.total": 8.4,
                "compositor.total": 1.1,
                "pipeline.new_frame_service": 11.2,
            },
        },
    )

    assert warnings == []
    assert "PRE-REACTION / NEVER SENT TO OUTPUT" in lines[0]
    assert "INPUT seq 11  delta 2  dt 66.67 ms  history ready" == lines[1]
    assert (
        "TEMP ALPHA raw 0.2000 (registered 0.1000)  refined 0.0800 (registered 0.0400)"
    ) == lines[2]
    assert "TEMP EDGE downstream contribution RGB 0.0300  state ready" == lines[3]
    assert (
        "REGISTRATION ready dx 1.2500 dy -0.5000 response 0.9200 overlap 0.9700"
    ) == lines[4]
    assert (
        "EFFECTIVE RVM ratio 0.2500  shift 2  model foreground on"
        "  wrap 0.3000  blend linear_srgb"
    ) == lines[5]
    assert "RESET count 5  reason capture_generation_changed" == lines[6]
    assert "backend 7.10" in lines[7]
    assert "RVM-run 6.30" in lines[7]
    assert "total 8.30" in lines[7]
    assert "color 0.60" in lines[8]
    assert "validate 0.05" in lines[8]
    assert "seg 8.40  compositor 1.10  new-frame 11.20" in lines[9]
    assert "input-mask-validation 0.01" in lines[10]
    assert "backdrop-blur-resize 0.04" in lines[10]
    assert "light-wrap-temporal-filter 0.05" in lines[11]
    assert "internal-output-validation 0.08" in lines[11]


def test_diagnostic_overlay_waiting_line_uses_selected_view_label():
    lines, warnings = preview_mod._diagnostic_overlay_lines(
        None,
        selected_view="raw_alpha",
        stats={},
    )

    assert lines == [
        "LOCAL MATTE DIAGNOSTIC — Raw model alpha — waiting for a unique input"
    ]
    assert warnings == []


def test_diagnostic_overlay_is_drawn_on_an_unshared_copy(monkeypatch):
    fake = FakeCV2()
    monkeypatch.setattr(preview_mod, "cv2", fake)
    diagnostic = make_diagnostic_frame()
    original = diagnostic.pixels.copy()
    monitor = LocalMatteDiagnosticMonitor()
    monitor.select("refined_alpha")
    controller = preview_mod._PreviewController(
        make_runtime(),
        backgrounds_dir=None,
        matte_monitor=monitor,
    )

    labeled = preview_mod._draw_overlay(
        diagnostic.pixels,
        {},
        controller,
        diagnostic=diagnostic,
    )

    assert not np.shares_memory(labeled, diagnostic.pixels)
    labeled[:] = 0
    np.testing.assert_array_equal(diagnostic.pixels, original)
    monitor.close()


class TestPreviewController:
    """Keyboard control logic, independent of the GUI event loop."""

    def test_mode_key_switches_mode(self):
        runtime = make_runtime(mode="passthrough")
        ctl = make_controller(runtime)
        ctl.handle_key(ord("1"))  # blur
        assert runtime.snapshot().background.mode == "blur"
        message = ctl.current_message()
        assert message is not None
        assert "blur" in message

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
        message = ctl.current_message()
        assert message is not None
        assert "switch to blur mode" in message

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
        message = ctl.current_message()
        assert message is not None
        assert "no background image files" in message

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
        message = ctl.current_message()
        assert message is not None
        assert "camera_target" in message

    def test_camera_mode_accepts_integer_zero_device(self):
        runtime = make_runtime(mode="passthrough", camera_device=0)
        ctl = make_controller(runtime)
        ctl.handle_key(ord("5"))
        assert runtime.snapshot().background.mode == "camera"

    def test_camera_mode_accepts_operator_target_id(self):
        runtime = RuntimeConfig(
            AppConfig.from_dict(
                {
                    "background": {
                        "mode": "passthrough",
                        "camera_target": "side-camera",
                    },
                    "backdrop_targets": {"side-camera": {"source": 2}},
                }
            )
        )
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

    def test_help_and_hint_describe_local_diagnostic_cycle(self):
        assert "[d/D] matte view" in preview_mod.HINT_LINE
        assert any(
            "d / D" in line and "LOCAL-ONLY" in line and "cycles to output" in line
            for line in preview_mod.HELP_LINES
        )

    def test_diagnostic_keys_cycle_both_directions_and_back_to_output(self):
        monitor = LocalMatteDiagnosticMonitor()
        ctl = preview_mod._PreviewController(
            make_runtime(),
            backgrounds_dir=None,
            matte_monitor=monitor,
        )
        views = diagnostic_view_names()

        ctl.handle_key(ord("d"))
        assert ctl.diagnostic_view == views[0]
        assert monitor.accepting
        assert "LOCAL matte view" in (ctl.current_message() or "")

        ctl.handle_key(ord("d"))
        assert ctl.diagnostic_view == views[1]
        ctl.handle_key(ord("D"))
        assert ctl.diagnostic_view == views[0]

        ctl.handle_key(ord("D"))
        assert ctl.diagnostic_view is None
        assert not monitor.accepting
        assert "showing production output" in (ctl.current_message() or "")

        ctl.handle_key(ord("D"))
        assert ctl.diagnostic_view == views[-1]
        ctl.handle_key(ord("d"))
        assert ctl.diagnostic_view is None
        assert not monitor.accepting
        monitor.close()

    def test_returning_to_output_clears_retained_diagnostic_pixels(self):
        monitor = active_diagnostic_monitor()
        ctl = preview_mod._PreviewController(
            make_runtime(),
            backgrounds_dir=None,
            matte_monitor=monitor,
        )

        ctl.handle_key(ord("D"))

        assert ctl.diagnostic_view is None
        assert not monitor.accepting
        assert monitor.latest() is None
        monitor.close()

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
        assert coordinator.patches == [
            ({"background": {"mode": "blur"}}, 5.0, "preview")
        ]
        assert runtime.snapshot().background.mode == "blur"

    def test_successful_update_identifies_preview_origin(self):
        runtime = make_runtime(mode="passthrough")
        coordinator = ImmediateCoordinator(runtime)
        ctl = preview_mod._PreviewController(
            runtime, backgrounds_dir=None, coordinator=coordinator
        )
        ctl.handle_key(ord("1"))
        assert coordinator.patches == [
            ({"background": {"mode": "blur"}}, 5.0, "preview")
        ]


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
