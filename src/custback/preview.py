"""On-screen preview window for output and explicit local matte diagnosis.

The default view shows exactly what the virtual camera sends. The ``d`` key
explicitly selects a private, pre-reaction diagnostic sink that is never
published to the virtual camera, FrameHub, WebUI, or API.

Must run on the main thread: OpenCV's GUI (highgui) requires that on macOS,
so `run_preview` is called from `__main__.run()` in place of the idle wait
loop while the pipeline and API threads do the work.

Controls (also drawn on screen as a hint bar):
  q / ESC     quit (stops the whole app)
  h           toggle the full help overlay
  0-5         switch mode: passthrough/blur/color/image/video/camera
  [ / ]       decrease / increase blur strength
  n / p       next / previous background file (image+video mode) or
              color preset (color mode)
  d / D       next / previous local matte diagnostic view; cycles back to output
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

# OpenCV's Qt bootstrap mutates these variables during import.  Remember what
# the user supplied so we can remove only OpenCV's broken defaults while
# preserving intentional overrides.
_QT_ENV_BEFORE_CV2 = {
    key: value
    for key in ("QT_QPA_FONTDIR", "QT_QPA_PLATFORM")
    if (value := os.environ.get(key)) is not None
}

from .backgrounds import DEFAULT_BACKGROUNDS_DIR, IMAGE_EXTS, list_background_files
from .compositor import COMPOSITOR_SUBSTAGE_NAMES
from .config import RuntimeConfig
from .hub import FrameHub
from .matte_live_diagnostics import (
    LocalMatteDiagnosticFrame,
    LocalMatteDiagnosticMonitor,
    diagnostic_view_spec,
)

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

# OpenCV is a compiled optional boundary. Keep its runtime ``None`` fallback
# while treating the dynamically exposed API as opaque to static analysis.
cv2: Any = _cv2

log = logging.getLogger(__name__)

WINDOW_TITLE = "custback preview"
QUIT_KEYS = (ord("q"), 27)  # q, ESC
HELP_KEY = ord("h")
NEXT_KEY, PREV_KEY = ord("n"), ord("p")
BLUR_DOWN_KEY, BLUR_UP_KEY = ord("["), ord("]")
DIAGNOSTIC_NEXT_KEY, DIAGNOSTIC_PREV_KEY = ord("d"), ord("D")
MODE_KEYS = {
    ord("0"): "passthrough",
    ord("1"): "blur",
    ord("2"): "color",
    ord("3"): "image",
    ord("4"): "video",
    ord("5"): "camera",
}
COLOR_PRESETS: list[tuple[int, int, int]] = [
    (18, 100, 32),  # green
    (140, 90, 20),  # blue
    (60, 60, 60),  # gray
    (245, 245, 245),  # white
    (0, 0, 0),  # black
]
BLUR_STEP = 10
BLUR_MIN, BLUR_MAX = 3, 151
MESSAGE_SECONDS = 2.5
PREVIEW_PROBE_TIMEOUT_S = 3.0
PROBE_STDERR_LIMIT = 2048
_AMBER = (0, 191, 255)

# Some Linux OpenCV wheels bundle Qt/XCB but no fonts.  Even after removing
# OpenCV's invalid QT_QPA_FONTDIR and explicitly selecting its xcb plugin, Qt
# writes these harmless bootstrap diagnostics directly to file descriptor 2.
# Capture only the first real window creation and discard these known lines;
# replay every other byte so genuine HighGUI diagnostics are not hidden.
_QT_BOOTSTRAP_NOISE_PREFIXES = (
    "Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome.",
    "QFontDatabase: Cannot find font directory ",
    "Note that Qt no longer ships fonts.",
)

_HIGHGUI_PROBE = (
    "import cv2; "
    "cv2.namedWindow('custback highgui probe', cv2.WINDOW_NORMAL); "
    "cv2.destroyWindow('custback highgui probe')"
)

HINT_LINE = (
    "[0-5] mode(passthru/blur/color/image/video/cam)  [n/p] file or color  "
    "[ [ ] ] blur  [d/D] matte view  [h] help  [q] quit"
)
HELP_LINES = [
    "0 passthrough   1 blur   2 color   3 image   4 video   5 camera",
    "n / p    next / previous background file (image+video) or color preset",
    "[  /  ]  decrease / increase blur strength",
    "d / D    next / previous LOCAL-ONLY pre-reaction matte view; cycles to output",
    "h        toggle this help",
    "q / ESC  quit",
]


def _qt_platform_plugins(
    cv2_module=None,
    *,
    environ: Mapping[str, str] | None = None,
) -> set[str]:
    """Return Qt platform plugin names visible to OpenCV, when discoverable."""

    module = cv2 if cv2_module is None else cv2_module
    roots: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        package_dir = Path(module_file).resolve().parent
        roots.extend(
            [
                package_dir / "qt" / "plugins" / "platforms",
                package_dir / "plugins" / "platforms",
            ]
        )
    env = os.environ if environ is None else environ
    for entry in env.get("QT_QPA_PLATFORM_PLUGIN_PATH", "").split(os.pathsep):
        if entry:
            roots.append(Path(entry))

    plugins: set[str] = set()
    for root in roots:
        try:
            entries = tuple(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            name = entry.name.lower()
            if "xcb" in name:
                plugins.add("xcb")
            if "wayland" in name:
                plugins.add("wayland")
    return plugins


def _normalize_highgui_environment(
    environ: Mapping[str, str],
    *,
    inherited_environ: Mapping[str, str],
    platform: str,
    cv2_module=None,
) -> tuple[dict[str, str], str, tuple[str, ...]]:
    env = dict(environ)
    notes: list[str] = []
    if not platform.startswith("linux"):
        return env, "", ()

    font_dir = env.get("QT_QPA_FONTDIR")
    module = cv2 if cv2_module is None else cv2_module
    module_file = getattr(module, "__file__", None)
    expected_cv2_font_dir = (
        Path(module_file).resolve().parent / "qt" / "fonts" if module_file else None
    )
    looks_opencv_injected = bool(
        font_dir
        and expected_cv2_font_dir is not None
        and Path(font_dir).resolve() == expected_cv2_font_dir
    )
    # Other custback modules can import cv2 before preview.py.  In that case
    # our import-time snapshot already contains OpenCV's unconditional wheel
    # default, so recognize its exact package-relative path as injected too.
    font_was_user_supplied = (
        inherited_environ.get("QT_QPA_FONTDIR") is not None
        and not looks_opencv_injected
    )
    if font_dir and not font_was_user_supplied and not Path(font_dir).is_dir():
        env.pop("QT_QPA_FONTDIR", None)
        notes.append("removed OpenCV's missing Qt font directory; using fontconfig")

    wayland = bool(env.get("WAYLAND_DISPLAY")) or (
        env.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )
    if not wayland:
        return env, "", tuple(notes)

    plugins = _qt_platform_plugins(cv2_module, environ=env)
    explicit_platform = inherited_environ.get("QT_QPA_PLATFORM") is not None
    has_xwayland = bool(env.get("DISPLAY"))
    if (
        has_xwayland
        and not explicit_platform
        and "xcb" in plugins
        and "wayland" not in plugins
    ):
        env["QT_QPA_PLATFORM"] = "xcb"
        notes.append("selected OpenCV's xcb plugin for the Wayland/XWayland session")
        return env, "", tuple(notes)

    if not has_xwayland and "wayland" not in plugins:
        return (
            env,
            "Wayland is available but OpenCV has no Qt Wayland platform plugin; "
            "use the browser preview or run with XWayland (DISPLAY set)",
            tuple(notes),
        )
    return env, "", tuple(notes)


def normalize_highgui_environment(
    *,
    environ: Mapping[str, str] | None = None,
    inherited_environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    cv2_module=None,
) -> tuple[dict[str, str], str]:
    """Normalize OpenCV Qt variables without overriding user choices.

    The returned mapping is suitable for the HighGUI child probe.  The caller
    can supply ``inherited_environ`` to distinguish values present before
    OpenCV import from defaults injected by OpenCV itself.
    """

    env = os.environ if environ is None else environ
    inherited = env if inherited_environ is None else inherited_environ
    normalized, reason, _notes = _normalize_highgui_environment(
        env,
        inherited_environ=inherited,
        platform=sys.platform if platform is None else platform,
        cv2_module=cv2_module,
    )
    return normalized, reason


def _bounded_probe_stderr(value: object, limit: int = PROBE_STDERR_LIMIT) -> str:
    if not value:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    unique: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if line and line not in seen:
            seen.add(line)
            unique.append(line)
    summary = " | ".join(unique)
    if len(summary) > limit:
        summary = summary[: max(0, limit - 1)] + "…"
    return summary


def _probe_failure(reason: str, stderr: object = None) -> tuple[bool, str]:
    detail = _bounded_probe_stderr(stderr)
    return False, f"{reason}: {detail}" if detail else reason


def _read_probe_stderr(stream) -> bytes:
    try:
        stream.flush()
        stream.seek(0)
        return stream.read(PROBE_STDERR_LIMIT)
    except OSError:
        return b""


def _filter_qt_bootstrap_stderr(value: bytes) -> tuple[bytes, int]:
    """Remove only known OpenCV/Qt bootstrap noise from captured stderr."""

    kept: list[bytes] = []
    suppressed = 0
    for line in value.splitlines(keepends=True):
        text = line.decode("utf-8", errors="replace").lstrip()
        if text.startswith(_QT_BOOTSTRAP_NOISE_PREFIXES):
            suppressed += 1
        else:
            kept.append(line)
    return b"".join(kept), suppressed


def _named_window_with_filtered_qt_stderr() -> None:
    """Create the first native window while filtering known Qt wheel noise.

    Qt writes these messages below Python's ``sys.stderr`` layer, so a normal
    redirect cannot intercept them.  A temporary file avoids pipe-buffer
    deadlocks, and unknown diagnostics are replayed to the original fd before
    any captured exception is re-raised.
    """

    try:
        saved_stderr = os.dup(2)
    except OSError:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        return

    captured = b""
    error: tuple[BaseException, object] | None = None
    try:
        with tempfile.TemporaryFile(mode="w+b") as stderr_stream:
            try:
                os.dup2(stderr_stream.fileno(), 2)
                try:
                    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                except BaseException as exc:
                    error = (exc, exc.__traceback__)
            finally:
                os.dup2(saved_stderr, 2)
            try:
                stderr_stream.flush()
                stderr_stream.seek(0)
                captured = stderr_stream.read()
            except OSError:
                captured = b""

        remaining, suppressed = _filter_qt_bootstrap_stderr(captured)
        if remaining:
            try:
                os.write(saved_stderr, remaining)
            except OSError:
                pass
        if suppressed:
            log.debug(
                "suppressed %d known OpenCV/Qt bootstrap diagnostic lines",
                suppressed,
            )
    finally:
        os.close(saved_stderr)

    if error is not None:
        exc, traceback = error
        raise exc.with_traceback(traceback)


def preview_available(
    *,
    environ: dict[str, str] | None = None,
    platform: str | None = None,
    timeout: float = PREVIEW_PROBE_TIMEOUT_S,
) -> tuple[bool, str]:
    """Safely determine whether HighGUI can initialize.

    Qt aborts the process rather than raising ``cv2.error`` for several
    headless/display-plugin failures.  Probe in a child process so that a
    native abort cannot take down the camera pipeline.
    """

    if cv2 is None:
        return False, "opencv-python is not installed"
    source_env = os.environ if environ is None else environ
    current_platform = sys.platform if platform is None else platform
    inherited = _QT_ENV_BEFORE_CV2 if environ is None else source_env
    env, normalization_error, notes = _normalize_highgui_environment(
        source_env,
        inherited_environ=inherited,
        platform=current_platform,
        cv2_module=cv2,
    )
    if normalization_error:
        return False, normalization_error
    for note in notes:
        log.debug("HighGUI environment: %s", note)
    if environ is None:
        # Apply the same carefully scoped changes to the parent process; the
        # successful probe and the real preview must use identical Qt settings.
        for key in ("QT_QPA_FONTDIR", "QT_QPA_PLATFORM"):
            if key in env:
                os.environ[key] = env[key]
            else:
                os.environ.pop(key, None)
    if current_platform.startswith("linux") and not (
        env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")
    ):
        return False, "no DISPLAY or WAYLAND_DISPLAY is available"
    try:
        with tempfile.TemporaryFile(mode="w+b") as stderr_stream:
            try:
                result = subprocess.run(
                    [sys.executable, "-c", _HIGHGUI_PROBE],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_stream,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return _probe_failure(
                    f"HighGUI probe timed out after {timeout:g} seconds",
                    _read_probe_stderr(stderr_stream),
                )
            captured_stderr = getattr(result, "stderr", None)
            if captured_stderr is None:
                captured_stderr = _read_probe_stderr(stderr_stream)
    except OSError as exc:
        return False, f"HighGUI probe failed: {exc}"
    if result.returncode != 0:
        if result.returncode < 0:
            return _probe_failure(
                f"HighGUI probe terminated by signal {-result.returncode}",
                captured_stderr,
            )
        return _probe_failure(
            f"HighGUI probe exited with status {result.returncode}",
            captured_stderr,
        )
    return True, ""


class _PreviewController:
    """Keyboard-driven live control, mirroring PATCH /config."""

    def __init__(
        self,
        runtime: RuntimeConfig,
        backgrounds_dir: Path | None,
        coordinator=None,
        matte_monitor: LocalMatteDiagnosticMonitor | None = None,
    ):
        self.runtime = runtime
        self.backgrounds_dir = backgrounds_dir
        self.coordinator = coordinator
        self.matte_monitor = matte_monitor
        self.show_help = False
        self._message: tuple[str, float] | None = None

    def flash(self, text: str) -> None:
        self._message = (text, time.monotonic() + MESSAGE_SECONDS)

    def current_message(self) -> str | None:
        if self._message and time.monotonic() < self._message[1]:
            return self._message[0]
        return None

    def handle_key(self, key: int) -> None:
        try:
            if key == HELP_KEY:
                self.show_help = not self.show_help
            elif key in MODE_KEYS:
                self._set_mode(MODE_KEYS[key])
            elif key in (BLUR_DOWN_KEY, BLUR_UP_KEY):
                self._adjust_blur(-BLUR_STEP if key == BLUR_DOWN_KEY else BLUR_STEP)
            elif key in (NEXT_KEY, PREV_KEY):
                self._cycle(+1 if key == NEXT_KEY else -1)
            elif key in (DIAGNOSTIC_NEXT_KEY, DIAGNOSTIC_PREV_KEY):
                self._cycle_diagnostic(reverse=key == DIAGNOSTIC_PREV_KEY)
        except Exception as exc:
            log.warning("preview config update rejected: %s", exc)
            self.flash(f"change rejected: {exc}")

    @property
    def diagnostic_view(self) -> str | None:
        monitor = self.matte_monitor
        return None if monitor is None else monitor.selected_view

    def _cycle_diagnostic(self, *, reverse: bool) -> None:
        monitor = self.matte_monitor
        if monitor is None:
            self.flash("local matte diagnostics are unavailable")
            return
        view = monitor.select_next(reverse=reverse)
        if view is None:
            self.flash("matte diagnostics off — showing production output")
            return
        spec = diagnostic_view_spec(view)
        self.flash(f"LOCAL matte view -> {spec.label}")

    def _update(self, patch: dict) -> None:
        if self.coordinator is None or not hasattr(
            self.coordinator, "apply_config_patch"
        ):
            raise RuntimeError("preview controls require the pipeline coordinator")
        self.coordinator.apply_config_patch(
            patch,
            timeout=5.0,
            origin="preview",
        )

    def _set_mode(self, mode: str) -> None:
        cfg = self.runtime.snapshot().background
        patch: dict = {"mode": mode}
        backgrounds_dir = self.backgrounds_dir or DEFAULT_BACKGROUNDS_DIR
        if mode in ("image", "video") and not (
            cfg.image_path if mode == "image" else cfg.video_path
        ):
            files = [
                f
                for f in list_background_files(backgrounds_dir)
                if (f.suffix.lower() in IMAGE_EXTS) == (mode == "image")
            ]
            if not files:
                self.flash(f"no background {mode} files in {backgrounds_dir}")
                return
            patch["image_path" if mode == "image" else "video_path"] = str(files[0])
        if mode == "camera" and cfg.camera_device == "" and not cfg.camera_target:
            self.flash(
                "configure background.camera_target or a startup camera device first"
            )
            return
        self._update({"background": patch})
        self.flash(f"mode -> {mode}")

    def _adjust_blur(self, delta: int) -> None:
        cfg = self.runtime.snapshot().background
        value = max(BLUR_MIN, min(BLUR_MAX, cfg.blur_strength + delta))
        self._update({"background": {"blur_strength": value}})
        note = "" if cfg.mode == "blur" else "  (press 1 to switch to blur mode)"
        self.flash(f"blur strength -> {value}{note}")

    def _cycle(self, direction: int) -> None:
        cfg = self.runtime.snapshot().background
        if cfg.mode == "color":
            try:
                color = cast(tuple[int, int, int], tuple(cfg.color))
                idx = COLOR_PRESETS.index(color)
            except ValueError:
                idx = -1
            idx = (idx + direction) % len(COLOR_PRESETS)
            self._update({"background": {"color": list(COLOR_PRESETS[idx])}})
            self.flash(f"color preset {idx + 1}/{len(COLOR_PRESETS)}")
        elif cfg.mode in ("image", "video"):
            backgrounds_dir = self.backgrounds_dir or DEFAULT_BACKGROUNDS_DIR
            files = list_background_files(backgrounds_dir)
            if not files:
                self.flash(f"no background files in {backgrounds_dir}")
                return
            current = Path(cfg.image_path if cfg.mode == "image" else cfg.video_path)
            idx = files.index(current) if current in files else -1
            nxt = files[(idx + direction) % len(files)]
            is_image = nxt.suffix.lower() in IMAGE_EXTS
            new_mode = "image" if is_image else "video"
            key = "image_path" if is_image else "video_path"
            self._update({"background": {"mode": new_mode, key: str(nxt)}})
            self.flash(f"{new_mode} -> {nxt.name}")
        else:
            self.flash("n/p applies in image, video or color mode (press 2/3/4)")


def _as_float(value: object) -> float | None:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _signed_metric_text(value: object) -> str:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    return f"{number:.4f}" if math.isfinite(number) else "n/a"


def _as_int(value: object) -> int | None:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        return None


def _fps_text(actual: object, target: object) -> str:
    actual_fps = _as_float(actual)
    target_fps = _as_float(target)
    shown_actual = "--" if actual_fps is None else f"{actual_fps:.1f}"
    if target_fps in (None, 0):
        return shown_actual
    return f"{shown_actual}/{target_fps:.0f}"


def _compact_status_text(value: object, default: str = "") -> str:
    """Return a single-line, bounded status value suitable for the overlay."""

    text = " ".join(str(value or default).split())
    return text[:160]


def _status_overlay_lines(
    stats: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    """Build compact status and warning lines without assuming warm metrics."""

    mode = str(stats.get("mode") or "starting")
    output_target = stats.get("output_target_fps")
    capture_target = stats.get("capture_target_fps", output_target)
    status = [
        f"{mode}  IN {_fps_text(stats.get('capture_fps'), capture_target)} fps  "
        f"OUT {_fps_text(stats.get('fps'), output_target)} fps"
    ]

    raw_selection = stats.get("segmentation_selection")
    selection = raw_selection if isinstance(raw_selection, Mapping) else None
    if selection is not None:
        backend = _compact_status_text(
            selection.get("selected_backend"), "unknown"
        ).lower()
        device = _compact_status_text(
            selection.get("active_device"),
            str(stats.get("segmentation_device") or "unknown"),
        ).lower()
        quality_tier = _compact_status_text(
            selection.get("quality_tier"), "unknown"
        ).lower()
        provider = _compact_status_text(selection.get("active_provider"), "unknown")
        selection_mode = _compact_status_text(
            selection.get("selection_mode"), "unknown"
        ).lower()
    else:
        backend = str(stats.get("segmentation_backend") or "unknown")
        backend = backend.removesuffix("Segmenter").lower()
        device = str(stats.get("segmentation_device") or "unknown").lower()
        quality_tier = ""
        provider = ""
        selection_mode = ""
    output_backend = str(stats.get("output_backend") or "unknown")
    version = stats.get("config_version", 0)
    segmenter_status = f"SEG {backend}/{device}"
    if selection is not None:
        segmenter_status += (
            f"  TIER {quality_tier}  PROVIDER {provider}  SELECT {selection_mode}"
        )
    status.append(f"{segmenter_status}  OUTPUT {output_backend}  CONFIG v{version}")

    raw_policy = stats.get("matte_policy")
    policy = raw_policy if isinstance(raw_policy, Mapping) else None
    raw_effective = policy.get("effective") if policy is not None else None
    effective = raw_effective if isinstance(raw_effective, Mapping) else None
    if effective is not None:
        alpha_mode = _compact_status_text(
            effective.get("raw_alpha_mode"), "unknown"
        ).lower()
        matte_parts = [f"ALPHA {alpha_mode}"]
        rvm_ratio = _as_float(effective.get("rvm_downsample_ratio"))
        if rvm_ratio is not None:
            matte_parts.append(f"RVM RATIO {rvm_ratio:.3f}".rstrip("0").rstrip("."))
        matte_parts.append(
            "EDGE " + ("on" if effective.get("edge_refine") is True else "off")
        )
        temporal = _compact_status_text(effective.get("residual_temporal_mode"))
        if temporal:
            matte_parts.append(f"TEMPORAL {temporal.lower()}")
        light_wrap = _as_float(effective.get("light_wrap"))
        if light_wrap is not None:
            matte_parts.append(f"LIGHT WRAP {light_wrap:.2f}".rstrip("0").rstrip("."))
        status.append("MATTE " + "  ".join(matte_parts))

    capture_parts: list[str] = []
    width, height = stats.get("capture_width"), stats.get("capture_height")
    if width and height:
        capture_parts.append(f"{width}x{height}")
    fourcc = str(stats.get("capture_fourcc") or "").strip()
    if fourcc:
        capture_parts.append(fourcc)
    reported = _as_float(stats.get("capture_fps_reported"))
    if reported:
        capture_parts.append(f"reported {reported:.1f} fps")
    if capture_parts:
        capture_backend = str(stats.get("capture_backend") or "camera")
        status.append(f"CAPTURE {capture_backend}: " + "  ".join(capture_parts))

    delivered_width = stats.get("capture_delivered_width")
    delivered_height = stats.get("capture_delivered_height")
    output_width = stats.get("output_width")
    output_height = stats.get("output_height")
    camera_fit = str(stats.get("camera_fit") or "")
    if (
        delivered_width
        and delivered_height
        and output_width
        and output_height
        and camera_fit
    ):
        status.append(
            f"GEOMETRY {delivered_width}x{delivered_height} -> "
            f"{camera_fit} -> {output_width}x{output_height}"
        )

    correction_state = str(stats.get("color_correction_state") or "")
    if correction_state:
        effective = str(stats.get("color_correction_effective_mode") or "unknown")
        reason = str(stats.get("color_correction_reason") or "unknown")
        confidence = _as_float(stats.get("color_correction_confidence")) or 0.0
        exposure_ev = _as_float(stats.get("color_correction_exposure_ev")) or 0.0
        status.append(
            f"COLOR {correction_state}/{effective}  {exposure_ev:+.2f} EV  "
            f"confidence {confidence:.2f}  reason {reason}"
        )

    timing_mode = str(stats.get("background_video_timing_mode") or "")
    source_fps = _as_float(stats.get("background_video_source_fps"))
    video_frames = _as_int(stats.get("background_video_frames_displayed")) or 0
    if timing_mode or source_fps or video_frames:
        skip_ratio = _as_float(stats.get("background_video_skip_ratio")) or 0.0
        status.append(
            "VIDEO "
            + (f"{source_fps:.1f} fps  " if source_fps else "")
            + (f"{timing_mode}  " if timing_mode else "")
            + f"skip {skip_ratio * 100:.0f}%"
        )

    base_update_fps = _as_float(stats.get("base_composite_update_fps"))
    segmentation_update_fps = _as_float(stats.get("segmentation_update_fps"))
    output_send_fps = _as_float(stats.get("output_send_fps"))
    base_reuse_ratio = _as_float(stats.get("base_composite_reuse_ratio"))
    exact_repeat_ratio = _as_float(stats.get("exact_final_output_repeat_ratio"))
    cadence_parts: list[str] = []
    if base_update_fps is not None:
        cadence_parts.append(f"VIS {base_update_fps:.1f}")
    if segmentation_update_fps is not None:
        cadence_parts.append(f"SEG {segmentation_update_fps:.1f}")
    if output_send_fps is not None:
        cadence_parts.append(f"SEND {output_send_fps:.1f}")
    if base_reuse_ratio is not None:
        cadence_parts.append(f"BASE REUSE {base_reuse_ratio * 100:.0f}%")
    if exact_repeat_ratio is not None:
        cadence_parts.append(f"EXACT FINAL REPEAT {exact_repeat_ratio * 100:.0f}%")
    if cadence_parts:
        status.append("CADENCE " + "  ".join(cadence_parts))

    warnings: list[str] = []
    if stats.get("capture_stalled"):
        age = _as_float(stats.get("capture_frame_age_ms"))
        warnings.append(
            "CAPTURE STALLED" + (f" ({age:.0f} ms since last frame)" if age else "")
        )
    elif stats.get("capture_target_met") is False:
        warnings.append("CAPTURE BELOW TARGET")

    if stats.get("cadence_mismatch_active"):
        if base_update_fps is not None and output_send_fps is not None:
            warnings.append(
                f"VISUAL UPDATES {base_update_fps:.0f} FPS; "
                f"OUTPUT REPEATS TO {output_send_fps:.0f} FPS"
            )
        else:
            warnings.append("VISUAL UPDATE CADENCE MISMATCH")

    if correction_state == "low-confidence":
        reason = str(stats.get("color_correction_reason") or "unspecified")
        warnings.append(f"COLOR LOW CONFIDENCE: {reason[:120]}")
    elif correction_state == "stale-decay":
        warnings.append("COLOR CORRECTION STALE: decaying to identity")

    fallback_fields = [
        ("output_fallback_active", "output_fallback_reason", "OUTPUT FALLBACK"),
        ("remote_fallback_active", "remote_fallback_reason", "REMOTE FALLBACK"),
    ]
    if selection is None:
        fallback_fields.insert(
            1,
            (
                "segmentation_fallback_active",
                "segmentation_fallback_reason",
                "SEGMENTATION FALLBACK",
            ),
        )
    for active_key, reason_key, label in fallback_fields:
        if stats.get(active_key):
            reason = str(stats.get(reason_key) or "unspecified")
            warnings.append(f"{label}: {reason[:120]}")

    if selection is not None and selection.get("fallback_active") is True:
        category = _compact_status_text(
            selection.get("fallback_category"), "unavailable"
        )
        reason = _compact_status_text(
            selection.get("fallback_reason"), "preferred backend unavailable"
        )
        warnings.append(f"SEGMENTATION FALLBACK [{category[:64]}]: {reason[:120]}")
        guidance = _compact_status_text(selection.get("guidance"))
        if guidance:
            warnings.append(f"SEGMENTATION ACTION: {guidance[:140]}")
    return status, warnings


def _metric_text(value: object, *, scale: float = 1.0, suffix: str = "") -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number * scale:.4f}{suffix}"


def _timing_text(values: Mapping[str, object], name: str) -> str:
    value = _as_float(values.get(name))
    return "n/a" if value is None else f"{value:.2f}"


def _diagnostic_overlay_lines(
    diagnostic: LocalMatteDiagnosticFrame | None,
    *,
    selected_view: str | None,
    stats: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    """Build frame-local lines while reusing public policy/status semantics."""

    if selected_view is None:
        return [], []
    if diagnostic is None or diagnostic.view != selected_view:
        label = (
            diagnostic_view_spec(cast(Any, selected_view)).label
            if selected_view
            else "diagnostic"
        )
        return (
            [f"LOCAL MATTE DIAGNOSTIC — {label} — waiting for a unique input"],
            [],
        )

    temporal = diagnostic.temporal.to_dict()
    sequence_delta = temporal["capture_sequence_delta"]
    timestamp_delta = temporal["capture_timestamp_delta_ms"]
    delta_text = "n/a" if sequence_delta is None else str(sequence_delta)
    timestamp_delta_value = _as_float(timestamp_delta)
    dt_text = (
        "n/a" if timestamp_delta_value is None else f"{timestamp_delta_value:.2f} ms"
    )
    lines = [
        "LOCAL MATTE DIAGNOSTIC — "
        f"{diagnostic.label} — PRE-REACTION / NEVER SENT TO OUTPUT",
        f"INPUT seq {diagnostic.capture_sequence}  delta {delta_text}  "
        f"dt {dt_text}  history {temporal['history_state']}",
        "TEMP ALPHA raw "
        f"{_metric_text(temporal['raw_alpha_abs_diff'])} "
        f"(registered "
        f"{_metric_text(temporal['raw_alpha_compensated_abs_diff'])})  "
        "refined "
        f"{_metric_text(temporal['refined_alpha_abs_diff'])} "
        f"(registered "
        f"{_metric_text(temporal['refined_alpha_compensated_abs_diff'])})",
        "TEMP EDGE downstream contribution RGB "
        f"{_metric_text(temporal['edge_colour_abs_diff'])}  "
        f"state {temporal['edge_colour_state']}",
        f"REGISTRATION {temporal['registration_state']} "
        f"dx {_signed_metric_text(temporal['registration_dx_px'])} "
        f"dy {_signed_metric_text(temporal['registration_dy_px'])} "
        f"response {_metric_text(temporal['registration_response'])} "
        f"overlap {_metric_text(temporal['registration_overlap_fraction'])}",
    ]

    raw_policy = stats.get("matte_policy")
    policy = raw_policy if isinstance(raw_policy, Mapping) else {}
    raw_effective = policy.get("effective")
    effective = raw_effective if isinstance(raw_effective, Mapping) else {}
    ratio = effective.get("rvm_downsample_ratio")
    ratio_text = "n/a" if ratio is None else _metric_text(ratio)
    lines.append(
        "EFFECTIVE "
        f"RVM ratio {ratio_text}  shift {effective.get('mask_shift', 'n/a')}  "
        "model foreground "
        f"{'on' if effective.get('use_model_foreground') is True else 'off'}  "
        f"wrap {_metric_text(effective.get('light_wrap'))}  "
        f"blend {policy.get('blend_space', 'unknown')}"
    )
    lines.append(
        "RESET count "
        f"{stats.get('matte_reset_count', 0)}  reason "
        f"{stats.get('matte_last_reset_reason') or 'none'}"
    )

    frame_timings = diagnostic.timings_ms
    lines.append(
        "FRAME SEG ms "
        f"backend {_timing_text(frame_timings, 'backend_inference_ms')}  "
        f"RVM-pre {_timing_text(frame_timings, 'rvm_preprocess_ms')}  "
        f"RVM-run {_timing_text(frame_timings, 'rvm_session_run_ms')}  "
        f"RVM-post {_timing_text(frame_timings, 'rvm_postprocess_ms')}  "
        f"refine {_timing_text(frame_timings, 'refinement_ms')}  "
        f"total {_timing_text(frame_timings, 'segmentation_ms')}"
    )
    lines.append(
        "FRAME COMPOSITOR ms "
        f"background {_timing_text(frame_timings, 'background_ms')}  "
        f"color {_timing_text(frame_timings, 'color_correction_ms')}  "
        f"prepare {_timing_text(frame_timings, 'composite_prepare_ms')}  "
        f"blend {_timing_text(frame_timings, 'composite_blend_ms')}  "
        f"total {_timing_text(frame_timings, 'composite_ms')}  "
        "validate "
        f"{_timing_text(frame_timings, 'post_composite_validation_ms')}"
    )
    raw_public_timing = stats.get("timing_ms")
    public_timing = raw_public_timing if isinstance(raw_public_timing, Mapping) else {}
    lines.append(
        "PUBLIC EWMA ms "
        f"seg {_timing_text(public_timing, 'segmentation.total')}  "
        f"compositor {_timing_text(public_timing, 'compositor.total')}  "
        f"new-frame {_timing_text(public_timing, 'pipeline.new_frame_service')}"
    )
    substages = diagnostic.compositor_substages_ms
    for offset in range(0, len(COMPOSITOR_SUBSTAGE_NAMES), 4):
        names = COMPOSITOR_SUBSTAGE_NAMES[offset : offset + 4]
        lines.append(
            "COMPOSITOR SUBSTAGES ms "
            + "  ".join(
                f"{name.replace('_', '-')} {_timing_text(substages, name)}"
                for name in names
            )
        )
    warnings = []
    if not diagnostic.available:
        warnings.append(
            "VIEW UNAVAILABLE: "
            + (diagnostic.unavailable_reason or "required evidence is absent")
        )
    elif "proxy" in diagnostic.interpretation:
        warnings.append(
            "NON-AUTHORITATIVE LIVE PROXY — use private replay annotations "
            "for qualification"
        )
    return lines, warnings


def _draw_bar(
    frame,
    y0: int,
    y1: int,
    lines: list[str],
    alpha: float = 0.55,
    colors: list[tuple[int, int, int]] | None = None,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (frame.shape[1], y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    for i, line in enumerate(lines):
        color = colors[i] if colors and i < len(colors) else (255, 255, 255)
        cv2.putText(
            frame,
            line,
            (10, y0 + 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_overlay(
    frame,
    stats: dict,
    controller: _PreviewController,
    diagnostic: LocalMatteDiagnosticFrame | None = None,
):
    labeled = frame.copy()
    h, _w = labeled.shape[:2]

    status_lines, warning_lines = _status_overlay_lines(stats)
    diagnostic_lines, diagnostic_warnings = _diagnostic_overlay_lines(
        diagnostic,
        selected_view=controller.diagnostic_view,
        stats=stats,
    )
    status_lines = diagnostic_lines + status_lines
    warning_lines = diagnostic_warnings + warning_lines
    top_lines = status_lines + warning_lines
    top_h = 22 * len(top_lines) + 12
    colors = [(0, 255, 0)] + [(255, 255, 255)] * (len(status_lines) - 1)
    colors.extend([_AMBER] * len(warning_lines))
    _draw_bar(labeled, 0, top_h, top_lines, alpha=0.6, colors=colors)

    if controller.show_help:
        help_y = top_h + 8
        _draw_bar(
            labeled,
            help_y,
            help_y + 22 * len(HELP_LINES) + 14,
            HELP_LINES,
        )

    message = controller.current_message()
    bottom_lines = [message] if message else []
    bottom_lines.append(HINT_LINE)
    bar_h = 24 * len(bottom_lines) + 12
    _draw_bar(labeled, h - bar_h, h, bottom_lines)
    return labeled


def run_preview(
    runtime: RuntimeConfig,
    hub: FrameHub,
    stop: threading.Event,
    backgrounds_dir: Path | None = None,
    coordinator=None,
    matte_monitor: LocalMatteDiagnosticMonitor | None = None,
) -> bool:
    """Display output frames with interactive controls until `stop` is set
    or the user quits.

    Return ``True`` when the user explicitly closes/quits the preview.  A
    late HighGUI failure returns ``False`` without setting ``stop`` so the
    caller can continue the already-running pipeline in headless mode.  The
    caller must run :func:`preview_available` before starting the pipeline;
    the local exception guard remains for failures after that probe.
    """
    if cv2 is None:
        if matte_monitor is not None:
            matte_monitor.deactivate()
        log.warning("opencv-python not installed; preview disabled")
        return False

    try:
        _named_window_with_filtered_qt_stderr()
    except cv2.error as exc:
        if matte_monitor is not None:
            matte_monitor.deactivate()
        log.warning("no GUI available, preview disabled (%s)", exc)
        return False

    controller = _PreviewController(
        runtime,
        backgrounds_dir or DEFAULT_BACKGROUNDS_DIR,
        coordinator=coordinator,
        matte_monitor=matte_monitor,
    )
    log.info(
        "preview window open — press h for controls, d for local matte views, "
        "q or ESC to quit"
    )
    output_sequence = -1
    diagnostic_sequence = -1
    last_output = None
    last_diagnostic: LocalMatteDiagnosticFrame | None = None
    user_quit = False
    try:
        while not stop.is_set():
            frame = None
            selected_view = controller.diagnostic_view
            if selected_view is None or (
                last_diagnostic is not None and last_diagnostic.view != selected_view
            ):
                last_diagnostic = None
            diagnostic = None
            if selected_view is not None and matte_monitor is not None:
                diagnostic, diagnostic_sequence = matte_monitor.get(
                    diagnostic_sequence,
                    timeout=0.05,
                )
                if diagnostic is not None:
                    last_diagnostic = diagnostic
                elif (
                    last_diagnostic is not None
                    and last_diagnostic.view == selected_view
                ):
                    diagnostic = last_diagnostic
                frame = diagnostic.pixels if diagnostic is not None else last_output
            else:
                new_output, output_sequence = hub.output.get(
                    output_sequence,
                    timeout=0.2,
                )
                if new_output is not None:
                    last_output = new_output
                frame = last_output
                diagnostic = None
            if frame is not None:
                stats = (
                    dict(diagnostic.status)
                    if diagnostic is not None
                    else hub.stats_dict()
                )
                cv2.imshow(
                    WINDOW_TITLE,
                    _draw_overlay(
                        frame,
                        stats,
                        controller,
                        diagnostic=diagnostic,
                    ),
                )
            # waitKey pumps the GUI event loop; required even without frames.
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                user_quit = True
                break
            if key != 0xFF:  # 0xFF (from -1) means "no key pressed"
                controller.handle_key(key)
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                user_quit = True
                break  # user closed the window
    except cv2.error as exc:
        log.warning("preview disabled after a display failure: %s", exc)
    finally:
        if matte_monitor is not None:
            matte_monitor.deactivate()
        if user_quit:
            stop.set()
        try:
            cv2.destroyWindow(WINDOW_TITLE)
        except cv2.error:
            pass
    return user_quit
