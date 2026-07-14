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
import threading
import time
from pathlib import Path

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
    env = dict(os.environ if environ is None else environ)
    current_platform = sys.platform if platform is None else platform
    if current_platform.startswith("linux") and not (
        env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")
    ):
        return False, "no DISPLAY or WAYLAND_DISPLAY is available"
    try:
        result = subprocess.run(
            [sys.executable, "-c", _HIGHGUI_PROBE],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"HighGUI probe failed: {exc}"
    if result.returncode != 0:
        if result.returncode < 0:
            return False, f"HighGUI probe terminated by signal {-result.returncode}"
        return False, f"HighGUI probe exited with status {result.returncode}"
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
        self.coordinator.apply_config_patch(patch, timeout=5.0)

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


def _draw_bar(frame, y0: int, y1: int, lines: list[str], alpha: float = 0.55) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (frame.shape[1], y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, y0 + 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_overlay(frame, stats: dict, controller: _PreviewController):
    labeled = frame.copy()
    h, w = labeled.shape[:2]

    label = f"{stats['mode']}  {stats['fps']:.0f} fps"
    backend = stats.get("segmentation_backend", "").removesuffix("Segmenter").lower()
    if backend:
        device = stats.get("segmentation_device", "")
        label += f"  {backend}" + (f"/{device}" if device and device != "cpu" else "")
    cv2.putText(labeled, label, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    if controller.show_help:
        _draw_bar(labeled, 40, 40 + 22 * len(HELP_LINES) + 14, HELP_LINES)

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
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
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
