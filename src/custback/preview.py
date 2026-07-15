"""On-screen preview window for verifying the processed output locally.

Shows exactly what the virtual camera sends, with keyboard controls to
change background mode, cycle background files, and adjust blur strength
live — the same knobs as `PATCH /config`, without leaving the window.

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
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path

# OpenCV's Qt bootstrap mutates these variables during import.  Remember what
# the user supplied so we can remove only OpenCV's broken defaults while
# preserving intentional overrides.
_QT_ENV_BEFORE_CV2 = {
    key: os.environ.get(key)
    for key in ("QT_QPA_FONTDIR", "QT_QPA_PLATFORM")
}

from .backgrounds import DEFAULT_BACKGROUNDS_DIR, IMAGE_EXTS, list_background_files
from .config import RuntimeConfig
from .hub import FrameHub

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

log = logging.getLogger(__name__)

WINDOW_TITLE = "custback preview"
QUIT_KEYS = (ord("q"), 27)  # q, ESC
HELP_KEY = ord("h")
NEXT_KEY, PREV_KEY = ord("n"), ord("p")
BLUR_DOWN_KEY, BLUR_UP_KEY = ord("["), ord("]")
MODE_KEYS = {
    ord("0"): "passthrough",
    ord("1"): "blur",
    ord("2"): "color",
    ord("3"): "image",
    ord("4"): "video",
    ord("5"): "camera",
}
COLOR_PRESETS: list[tuple[int, int, int]] = [
    (18, 100, 32),      # green
    (140, 90, 20),      # blue
    (60, 60, 60),       # gray
    (245, 245, 245),    # white
    (0, 0, 0),          # black
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
    "[ [ ] ] blur  [h] help  [q] quit"
)
HELP_LINES = [
    "0 passthrough   1 blur   2 color   3 image   4 video   5 camera",
    "n / p    next / previous background file (image+video) or color preset",
    "[  /  ]  decrease / increase blur strength",
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
        Path(module_file).resolve().parent / "qt" / "fonts"
        if module_file
        else None
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
    ):
        self.runtime = runtime
        self.backgrounds_dir = backgrounds_dir
        self.coordinator = coordinator
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
        except Exception as exc:
            log.warning("preview config update rejected: %s", exc)
            self.flash(f"change rejected: {exc}")

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
        if mode in ("image", "video") and not (cfg.image_path if mode == "image" else cfg.video_path):
            files = [f for f in list_background_files(self.backgrounds_dir)
                     if (f.suffix.lower() in IMAGE_EXTS) == (mode == "image")]
            if not files:
                self.flash(f"no background {mode} files in {self.backgrounds_dir}")
                return
            patch["image_path" if mode == "image" else "video_path"] = str(files[0])
        if mode == "camera" and cfg.camera_device == "":
            self.flash("set background.camera_device first (API/CLI) — no default second camera")
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
                idx = COLOR_PRESETS.index(tuple(cfg.color))
            except ValueError:
                idx = -1
            idx = (idx + direction) % len(COLOR_PRESETS)
            self._update({"background": {"color": list(COLOR_PRESETS[idx])}})
            self.flash(f"color preset {idx + 1}/{len(COLOR_PRESETS)}")
        elif cfg.mode in ("image", "video"):
            files = list_background_files(self.backgrounds_dir)
            if not files:
                self.flash(f"no background files in {self.backgrounds_dir}")
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
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _fps_text(actual: object, target: object) -> str:
    actual_fps = _as_float(actual)
    target_fps = _as_float(target)
    shown_actual = "--" if actual_fps is None else f"{actual_fps:.1f}"
    if target_fps in (None, 0):
        return shown_actual
    return f"{shown_actual}/{target_fps:.0f}"


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

    backend = str(stats.get("segmentation_backend") or "unknown")
    backend = backend.removesuffix("Segmenter").lower()
    device = str(stats.get("segmentation_device") or "unknown").lower()
    output_backend = str(stats.get("output_backend") or "unknown")
    version = stats.get("config_version", 0)
    status.append(
        f"SEG {backend}/{device}  OUTPUT {output_backend}  CONFIG v{version}"
    )

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

    timing_mode = str(stats.get("background_video_timing_mode") or "")
    source_fps = _as_float(stats.get("background_video_source_fps"))
    video_frames = int(stats.get("background_video_frames_displayed") or 0)
    if timing_mode or source_fps or video_frames:
        skip_ratio = _as_float(stats.get("background_video_skip_ratio")) or 0.0
        status.append(
            "VIDEO "
            + (f"{source_fps:.1f} fps  " if source_fps else "")
            + (f"{timing_mode}  " if timing_mode else "")
            + f"skip {skip_ratio * 100:.0f}%"
        )

    warnings: list[str] = []
    if stats.get("capture_stalled"):
        age = _as_float(stats.get("capture_frame_age_ms"))
        warnings.append(
            "CAPTURE STALLED" + (f" ({age:.0f} ms since last frame)" if age else "")
        )
    elif stats.get("capture_target_met") is False:
        warnings.append("CAPTURE BELOW TARGET")

    fallback_fields = (
        ("output_fallback_active", "output_fallback_reason", "OUTPUT FALLBACK"),
        (
            "segmentation_fallback_active",
            "segmentation_fallback_reason",
            "SEGMENTATION FALLBACK",
        ),
        ("remote_fallback_active", "remote_fallback_reason", "REMOTE FALLBACK"),
    )
    for active_key, reason_key, label in fallback_fields:
        if stats.get(active_key):
            reason = str(stats.get(reason_key) or "unspecified")
            warnings.append(f"{label}: {reason[:120]}")
    return status, warnings


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
        cv2.putText(frame, line, (10, y0 + 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)


def _draw_overlay(frame, stats: dict, controller: _PreviewController):
    labeled = frame.copy()
    h, _w = labeled.shape[:2]

    status_lines, warning_lines = _status_overlay_lines(stats)
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
        log.warning("opencv-python not installed; preview disabled")
        return False

    try:
        _named_window_with_filtered_qt_stderr()
    except cv2.error as exc:
        log.warning("no GUI available, preview disabled (%s)", exc)
        return False

    controller = _PreviewController(
        runtime,
        backgrounds_dir or DEFAULT_BACKGROUNDS_DIR,
        coordinator=coordinator,
    )
    log.info("preview window open — press h for controls, q or ESC to quit")
    seq = -1
    user_quit = False
    try:
        while not stop.is_set():
            frame, seq = hub.output.get(seq, timeout=0.2)
            if frame is not None:
                cv2.imshow(WINDOW_TITLE, _draw_overlay(frame, hub.stats_dict(), controller))
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
        if user_quit:
            stop.set()
        try:
            cv2.destroyWindow(WINDOW_TITLE)
        except cv2.error:
            pass
    return user_quit
