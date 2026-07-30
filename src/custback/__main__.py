"""CLI entry point: `custback` or `python -m custback`."""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

from .backgrounds import IMAGE_EXTS, VIDEO_EXTS
from .config import MODES, AppConfig, RuntimeConfig, format_config_error
from .diagnostics import sanitized_config_summary
from .hub import FrameHub
from .pipeline import Pipeline

log = logging.getLogger(__name__)

EXIT_RUNTIME = 1
EXIT_CONFIG = 2
EXIT_API = 3
API_START_TIMEOUT_S = 5.0
SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None
_VISUAL_POLICY_DIAGNOSTIC_FIELDS = (
    "schema_version",
    "camera.fit_mode",
    "camera.anchor_x",
    "camera.anchor_y",
    "camera.rotation",
    "background.fit_mode",
    "background.anchor_x",
    "background.anchor_y",
    "output.width",
    "output.height",
    "compositing.blend_space",
    "compositing.color_correction.mode",
    "compositing.color_correction.strength",
    "compositing.color_correction.exposure_limit_ev",
    "compositing.color_correction.white_balance_strength",
    "compositing.color_correction.adaptation_time_s",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="custback",
        description="Virtual camera with background replacement",
    )
    parser.add_argument("-c", "--config", help="path to YAML config file")
    parser.add_argument("--camera", help="camera device (index or path/URL)")
    parser.add_argument("--width", type=int, help="capture width")
    parser.add_argument("--height", type=int, help="capture height")
    parser.add_argument("--fps", type=int, help="target fps")
    parser.add_argument(
        "--camera-pixel-format",
        choices=("auto", "mjpeg", "backend"),
        help="capture pixel-format policy",
    )
    parser.add_argument(
        "--camera-mode-mismatch",
        choices=("warn", "error"),
        help="whether a negotiated camera-mode mismatch is fatal",
    )
    parser.add_argument(
        "--camera-recovery-timeout",
        type=float,
        metavar="SECONDS",
        help="maximum runtime camera outage before exiting",
    )
    parser.add_argument("--mirror", action="store_true", help="mirror the camera")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use a synthetic test source instead of a real camera",
    )
    parser.add_argument("--mode", choices=MODES, help="background mode")
    parser.add_argument("--image", help="background image path (implies --mode image)")
    parser.add_argument("--video", help="background video path (implies --mode video)")
    parser.add_argument(
        "--bg-camera",
        help="operator-approved local camera/device as live backdrop (implies --mode camera)",
    )
    parser.add_argument("--blur", type=int, help="blur strength (implies --mode blur)")
    api_toggle = parser.add_mutually_exclusive_group()
    api_toggle.add_argument(
        "--enable-api",
        action="store_true",
        help="enable the HTTP API even when the config file disables it",
    )
    api_toggle.add_argument(
        "--no-api", action="store_true", help="disable the HTTP API"
    )
    parser.add_argument("--api-host", help="API bind host (default 127.0.0.1)")
    parser.add_argument("--api-port", type=int, help="API port (default 8710)")
    parser.add_argument("--api-token-file", help="path to the mode-0600 API token file")
    parser.add_argument(
        "--renderer-token-file",
        help="path to the mode-0600 renderer-scoped frame token file",
    )
    parser.add_argument(
        "--avatar-url",
        help="custback-avatar control API root for the /avatar/* proxy",
    )
    parser.add_argument(
        "--avatar-token-file",
        help="path to custback-avatar's existing control-token file",
    )
    parser.add_argument(
        "--allow-non-loopback-api",
        action="store_true",
        help="allow a TLS-protected API bind outside loopback",
    )
    parser.add_argument("--api-tls-cert", help="TLS certificate for the API")
    parser.add_argument("--api-tls-key", help="TLS private key for the API")
    parser.add_argument(
        "--api-plaintext",
        action="store_true",
        help="clear configured API TLS credentials (numeric loopback only)",
    )
    parser.add_argument(
        "--avatar-plaintext",
        action="store_true",
        help="clear configured avatar-proxy TLS credentials (numeric loopback only)",
    )
    show_token = parser.add_mutually_exclusive_group()
    show_token.add_argument(
        "--show-api-token",
        action="store_true",
        help="print the resolved API token and exit",
    )
    show_token.add_argument(
        "--show-renderer-token",
        action="store_true",
        help="provision and print the renderer-scoped frame token, then exit",
    )
    parser.add_argument(
        "--no-vcam",
        action="store_true",
        help="do not open a virtual camera (serve frames only via the API)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show the processed output in an on-screen window (q/ESC quits)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log-file",
        metavar="PATH",
        help="override the default rotating diagnostics log",
    )
    log_group.add_argument(
        "--no-file-log",
        action="store_true",
        help="disable the default rotating diagnostics log",
    )
    parser.add_argument(
        "--dump-config",
        metavar="PATH",
        help="write the effective config to PATH and exit",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="list detected input cameras (friendly name + stable id) and exit",
    )
    return parser


def list_cameras() -> int:
    """Print detected input cameras and exit.

    A no-hardware-mutating surface for choosing a stable device identifier
    instead of a bare index that can change meaning when devices reorder
    (WIN-3.1).  The project's own / OBS output camera is excluded so it is
    never selected as an input (WIN-3.4).
    """

    from .camera_devices import enumerate_cameras

    devices = enumerate_cameras()
    if not devices:
        print("no input cameras detected")
        return 0
    width = max(len(str(device.index)) for device in devices)
    for device in devices:
        print(
            f"[{device.index:>{width}}] {device.name}  "
            f"(id={device.stable_id}, backend={device.backend})"
        )
    return 0


def config_from_args(args: argparse.Namespace) -> AppConfig:
    # Apply every CLI override to plain data and validate the resulting
    # configuration once.  Valid combinations must not depend on assignment
    # order when two overrides jointly satisfy a cross-field invariant.
    values = AppConfig.load(args.config).to_dict()
    cam = values["camera"]
    bg = values["background"]
    api = values["api"]
    avatar = values["avatar"]
    output = values["output"]
    if args.camera is not None:
        cam["device"] = args.camera
    if args.width is not None:
        cam["width"] = args.width
    if args.height is not None:
        cam["height"] = args.height
    if args.fps is not None:
        cam["fps"] = args.fps
        output["fps"] = args.fps
    if args.camera_pixel_format is not None:
        cam["pixel_format"] = args.camera_pixel_format
    if args.camera_mode_mismatch is not None:
        cam["mode_mismatch"] = args.camera_mode_mismatch
    if args.camera_recovery_timeout is not None:
        cam["recovery_timeout_s"] = args.camera_recovery_timeout
    if args.mirror:
        cam["mirror"] = True
    if args.synthetic:
        cam["synthetic"] = True
    if args.image:
        # Route by extension rather than trusting the flag name: users
        # commonly reach for whichever flag they remember, and preview.py's
        # n/p file cycling already auto-detects the same way.
        if Path(args.image).suffix.lower() in VIDEO_EXTS:
            bg["video_path"] = args.image
            bg["mode"] = "video"
        else:
            bg["image_path"] = args.image
            bg["mode"] = "image"
    if args.video:
        if Path(args.video).suffix.lower() in IMAGE_EXTS:
            bg["image_path"] = args.video
            bg["mode"] = "image"
        else:
            bg["video_path"] = args.video
            bg["mode"] = "video"
    if args.bg_camera:
        bg["camera_device"] = args.bg_camera
        bg["mode"] = "camera"
    if args.blur is not None:
        bg["blur_strength"] = args.blur
        bg["mode"] = "blur"
    if args.mode:
        bg["mode"] = args.mode
    if args.enable_api:
        api["enabled"] = True
    if args.no_api:
        api["enabled"] = False
    if args.api_host:
        api["host"] = args.api_host
    if args.api_port is not None:
        api["port"] = args.api_port
    if args.api_token_file:
        api["token_file"] = args.api_token_file
    if args.renderer_token_file:
        api["renderer_token_file"] = args.renderer_token_file
    if args.avatar_url:
        avatar["url"] = args.avatar_url
    if args.avatar_token_file:
        avatar["token_file"] = args.avatar_token_file
    if args.avatar_plaintext:
        from .api.security import validate_outbound_endpoint

        endpoint = validate_outbound_endpoint(
            avatar["url"],
            kind="http",
            label="avatar.url",
            allow_empty=True,
        )
        if endpoint is None or endpoint.secure:
            raise ValueError(
                "--avatar-plaintext requires an http:// numeric-loopback --avatar-url"
            )
        avatar["tls_ca_file"] = ""
        avatar["tls_certfile"] = ""
        avatar["tls_keyfile"] = ""
    if args.allow_non_loopback_api:
        api["allow_non_loopback"] = True
    if args.api_tls_cert:
        api["tls_certfile"] = args.api_tls_cert
    if args.api_tls_key:
        api["tls_keyfile"] = args.api_tls_key
    if args.api_plaintext:
        from .api.security import is_numeric_loopback_host

        if args.api_tls_cert or args.api_tls_key:
            raise ValueError(
                "--api-plaintext cannot be combined with API TLS arguments"
            )
        if not is_numeric_loopback_host(api["host"]):
            raise ValueError("--api-plaintext requires a numeric-loopback --api-host")
        api["tls_certfile"] = ""
        api["tls_keyfile"] = ""
    if args.no_vcam:
        output["backend"] = "null"
    if args.preview:
        output["preview"] = True
    return AppConfig.from_dict(values)


class ApiStartupError(RuntimeError):
    """The configured API could not become or remain available."""


class _ShutdownSignal(BaseException):
    """Unwind startup/runtime work so signal-driven shutdown can clean up."""


class _ApiRunner:
    """Pre-bound, observable Uvicorn server running beside the pipeline."""

    def __init__(self, app, cfg):
        import uvicorn

        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=cfg.host,
                port=cfg.port,
                log_level="warning",
                ssl_certfile=cfg.tls_certfile or None,
                ssl_keyfile=cfg.tls_keyfile or None,
                ws_max_size=cfg.ws_max_bytes,
            )
        )
        self._thread: threading.Thread | None = None
        self._socket = None
        self._error: BaseException | None = None
        self._shutdown_requested = False
        self._unexpected_exit = False

    @property
    def failed(self) -> bool:
        if self._error is not None:
            self._unexpected_exit = True
        if (
            self._thread is not None
            and not self._thread.is_alive()
            and not self._shutdown_requested
        ):
            self._unexpected_exit = True
        return self._unexpected_exit

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self, timeout: float = API_START_TIMEOUT_S) -> None:
        try:
            bound_socket = self.server.config.bind_socket()
            self._socket = bound_socket
        except SystemExit as exc:
            raise ApiStartupError("API socket bind failed") from exc
        except OSError as exc:
            raise ApiStartupError(f"API socket bind failed: {exc}") from exc

        def serve() -> None:
            try:
                self.server.run(sockets=[bound_socket])
            except BaseException as exc:
                self._error = exc

        self._thread = threading.Thread(target=serve, name="api", daemon=True)
        try:
            self._thread.start()
        except BaseException as exc:
            if self._socket is not None:
                with contextlib.suppress(OSError):
                    self._socket.close()
            self._thread = None
            raise ApiStartupError(f"API thread startup failed: {exc}") from exc
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.server.started:
                return
            if self.failed:
                self.stop()
                raise ApiStartupError(
                    f"API startup failed: {self._error or 'server exited'}"
                )
            threading.Event().wait(0.02)
        self.stop()
        raise ApiStartupError(f"API did not start within {timeout:g} seconds")

    def stop(self, timeout: float = 3.0) -> bool:
        self._shutdown_requested = True
        self.server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                self.server.force_exit = True
                self._thread.join(1.0)
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        return self._thread is None or not self._thread.is_alive()


def _watch_pipeline(
    pipeline: Pipeline,
    stop: threading.Event,
    api_runner: _ApiRunner | None = None,
) -> None:
    """Set `stop` if either required service dies."""
    while not stop.is_set():
        if not pipeline.running or (api_runner is not None and api_runner.failed):
            stop.set()
            return
        stop.wait(0.5)


def _wait_headless(
    pipeline: Pipeline,
    stop: threading.Event,
    api_runner: _ApiRunner | None,
) -> None:
    """Monitor required services until shutdown; classification is centralized."""
    while not stop.is_set() and pipeline.running:
        if api_runner is not None and api_runner.failed:
            stop.set()
            return
        stop.wait(0.5)


def _classify_service_failure(
    pipeline: Pipeline, api_runner: _ApiRunner | None
) -> tuple[int, str]:
    """Return one authoritative required-service failure, if present."""
    if api_runner is not None and api_runner.failed:
        detail = api_runner.error or "server thread exited without an exception"
        log.error("API server stopped unexpectedly: %s", detail)
        return EXIT_API, "api-failure"
    if not pipeline.running:
        log.error("pipeline stopped unexpectedly")
        return EXIT_RUNTIME, "pipeline-failure"
    return 0, ""


def _log_shutdown_summary(hub: FrameHub, reason: str, exit_code: int) -> None:
    stats = hub.stats_dict()
    log.info(
        "shutdown reason=%s exit=%d uptime=%.1fs frames_in=%d frames_out=%d "
        "capture_fps=%.1f output_fps=%.1f read_failures=%d restarts=%d "
        "repeats=%d video_skips=%d capture_read_ms=%s segmentation_ms=%s "
        "background_ms=%s color_correction_ms=%s composite_ms=%s output_send_ms=%s "
        "frame_processing_ms=%s capture_generation=%d camera_geometry=%d "
        "background_geometry=%d corrections_applied=%d corrections_bypassed=%d "
        "color_scene_cuts=%d color_transitions=%d",
        reason,
        exit_code,
        stats["uptime_s"],
        stats["frames_in"],
        stats["frames_out"],
        stats["capture_fps"],
        stats["fps"],
        stats["capture_read_failures"],
        stats["capture_restarts"],
        stats["output_repeated_frames"],
        stats["background_video_frames_skipped"],
        stats["capture_read_ms"],
        stats["segmentation_ms"],
        stats["background_ms"],
        stats["color_correction_ms"],
        stats["composite_ms"],
        stats["output_send_ms"],
        stats["frame_processing_ms"],
        stats["capture_generation"],
        stats["capture_geometry_transitions"],
        stats["background_geometry_transitions"],
        stats["color_correction_applied_frames"],
        stats["color_correction_bypassed_frames"],
        stats["color_correction_scene_cuts"],
        stats["color_correction_transitions"],
    )


def _security_policy(cfg: AppConfig):
    from .api.security import (
        SecurityConfigurationError,
        SecurityPolicy,
        is_loopback_host,
        resolve_api_token,
        resolve_renderer_token,
        validate_bind_security,
    )

    api = cfg.api
    validate_bind_security(
        api.host,
        allow_non_loopback=api.allow_non_loopback,
        tls_certfile=api.tls_certfile,
        tls_keyfile=api.tls_keyfile,
    )
    if not is_loopback_host(api.host) and not api.allowed_origins:
        raise SecurityConfigurationError(
            "non-loopback API binds require explicit api.allowed_origins"
        )
    token = resolve_api_token(api.token_file)
    if token.created:
        log.info("created API token file %s", token.path)
    renderer_token = resolve_renderer_token(api.renderer_token_file, create=True)
    if renderer_token.created:
        log.info("created renderer token file %s", renderer_token.path)
    return SecurityPolicy.for_bind(
        token.value,
        api.host,
        api.port,
        allowed_origins=api.allowed_origins,
        tls=bool(api.tls_certfile),
        session_ttl_s=api.session_ttl_s,
        renderer_token=renderer_token.value,
    )


def run(cfg: AppConfig, *, run_id: str = "") -> int:
    hub = FrameHub(run_id=run_id)
    stop = threading.Event()
    shutdown_reason = "normal"
    previous_signal_handlers: dict[signal.Signals, SignalHandler] = {}
    if threading.current_thread() is threading.main_thread():

        def request_shutdown(signum, _frame) -> None:
            nonlocal shutdown_reason
            if stop.is_set():
                return
            shutdown_reason = signal.Signals(signum).name.lower()
            stop.set()
            raise _ShutdownSignal()

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signal_handlers[sig] = signal.signal(sig, request_shutdown)

    exit_code = 0
    security = None
    pipeline: Pipeline | None = None
    api_runner: _ApiRunner | None = None
    try:
        if cfg.api.enabled:
            # Resolve token and validate the bind policy before activating hardware,
            # but do not advertise a listening API until the pipeline is ready.
            security = _security_policy(cfg)

        from .segmentation import preacquire_segmenter_model

        try:
            model_preparation = preacquire_segmenter_model(cfg.segmentation)
        except Exception:
            log.exception("segmentation model acquisition failed")
            exit_code = EXIT_RUNTIME
            shutdown_reason = "model-acquisition-failure"

        # Probe preview only after security and managed models are resolved.
        # This keeps the hardware/API readiness sequence deterministic while
        # still normalizing Qt before the real native window is opened.
        if exit_code == 0 and not stop.is_set() and cfg.output.preview:
            from .preview import preview_available

            available, reason = preview_available()
            if not available:
                log.warning("preview disabled: %s", reason)
                cfg.output.preview = False

        runtime = RuntimeConfig(cfg)
        if exit_code == 0 and not stop.is_set():
            pipeline = Pipeline(
                runtime,
                hub,
                model_preparation=model_preparation,
            )
            try:
                pipeline.start()
            except _ShutdownSignal:
                raise
            except BaseException:
                log.exception("pipeline startup failed")
                exit_code = EXIT_RUNTIME
                shutdown_reason = "pipeline-startup-failure"

        api_address = "disabled"
        if (
            exit_code == 0
            and not stop.is_set()
            and cfg.api.enabled
            and pipeline is not None
        ):
            from .api.server import create_app

            assert security is not None

            def request_lifecycle_shutdown() -> None:
                # Invoked from the API thread by POST /lifecycle/shutdown
                # (WIN-5.3). Setting the event unwinds the wait loop below; the
                # finally block then stops the API and drains the pipeline, so
                # camera/output cleanup still runs (WIN-5.4).
                nonlocal shutdown_reason
                if not stop.is_set():
                    shutdown_reason = "lifecycle-shutdown"
                stop.set()

            app = create_app(
                runtime,
                hub,
                pipeline,
                security=security,
                on_shutdown=request_lifecycle_shutdown,
            )
            api_runner = _ApiRunner(app, cfg.api)
            try:
                api_runner.start()
            except ApiStartupError:
                log.exception("API startup failed")
                exit_code = EXIT_API
                shutdown_reason = "api-startup-failure"
            else:
                scheme = "https" if cfg.api.tls_certfile else "http"
                api_address = f"{scheme}://{cfg.api.host}:{cfg.api.port}"

        if exit_code == 0 and not stop.is_set() and pipeline is not None:
            ready = hub.stats_dict()
            log.info(
                "ready api=%s camera_requested=%s/%sx%s@%s "
                "camera_negotiated=%s/%s %sx%s@%s segmenter=%s/%s output=%s "
                "%sx%s@%s preview=%s visual_policy=%s",
                api_address,
                cfg.camera.pixel_format,
                cfg.camera.width,
                cfg.camera.height,
                cfg.camera.fps,
                ready["capture_backend"],
                ready["capture_fourcc"] or "backend",
                ready["capture_width"],
                ready["capture_height"],
                ready["capture_fps_reported"],
                ready["segmentation_backend"],
                ready["segmentation_device"],
                ready["output_backend"],
                ready["output_width"],
                ready["output_height"],
                ready["output_fps"],
                "native" if cfg.output.preview else "disabled",
                sanitized_config_summary(cfg, list(_VISUAL_POLICY_DIAGNOSTIC_FIELDS)),
            )

            if cfg.output.preview:
                # Runs on the main thread (required by OpenCV GUI on macOS);
                # returns when the user quits the window or `stop` is set.
                from .preview import run_preview

                watchdog = threading.Thread(
                    target=_watch_pipeline,
                    args=(pipeline, stop, api_runner),
                    name="watchdog",
                    daemon=True,
                )
                watchdog.start()
                user_quit = run_preview(runtime, hub, stop, coordinator=pipeline)
                failure_code, failure_reason = _classify_service_failure(
                    pipeline, api_runner
                )
                if failure_code:
                    exit_code = failure_code
                    shutdown_reason = failure_reason
                    stop.set()
                elif user_quit:
                    shutdown_reason = "preview-quit"
                    stop.set()
                elif not stop.is_set():
                    log.warning("continuing without the preview window")
                    _wait_headless(pipeline, stop, api_runner)
            else:
                _wait_headless(pipeline, stop, api_runner)

            if exit_code == 0:
                failure_code, failure_reason = _classify_service_failure(
                    pipeline, api_runner
                )
                if failure_code:
                    exit_code = failure_code
                    shutdown_reason = failure_reason
    except (_ShutdownSignal, KeyboardInterrupt):
        stop.set()
    except (OSError, ValueError):
        exit_code = EXIT_CONFIG
        shutdown_reason = "configuration-failure"
        raise
    except Exception:
        log.exception("runtime coordination failed")
        if exit_code == 0:
            exit_code = EXIT_RUNTIME
            shutdown_reason = "runtime-failure"
    finally:
        if api_runner is not None:
            stop_api = getattr(api_runner, "stop", None)
            if callable(stop_api) and not stop_api():
                log.error("API server did not stop cleanly")
                exit_code = EXIT_API
                shutdown_reason = "api-shutdown-failure"
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                log.exception("pipeline error")
                if exit_code == 0:
                    exit_code = EXIT_RUNTIME
                    shutdown_reason = "pipeline-failure"
        if api_runner is not None and getattr(api_runner, "failed", False):
            exit_code = EXIT_API
            shutdown_reason = "api-failure"
        _log_shutdown_summary(hub, shutdown_reason, exit_code)
        if threading.current_thread() is threading.main_thread():
            for sig, previous in previous_signal_handlers.items():
                if previous is not None:
                    signal.signal(sig, previous)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv[:1] == ["migrate"]:
        # Migration must inspect legacy YAML before the current strict runtime
        # parser tries to load it.  It also must not initialize diagnostics or
        # any camera/network resource merely to repair on-disk state.
        from .migration import main as migration_main

        return migration_main(effective_argv[1:], prog="custback migrate")
    if effective_argv[:1] == ["avatar"]:
        # Keep the canonical avatar surface available from every Python
        # installation.  The separate ``custback-avatar`` console script is a
        # compatibility alias that enters the same implementation directly.
        from .avatar.__main__ import main as avatar_main

        return avatar_main(effective_argv[1:], prog="custback avatar")

    args = build_parser().parse_args(effective_argv)
    from .diagnostics import LoggingConfigurationError, configure_logging

    try:
        logging_session = configure_logging(
            verbose=args.verbose,
            log_file=args.log_file,
            no_file_log=args.no_file_log,
        )
    except (LoggingConfigurationError, ValueError) as exc:
        print(f"custback: logging configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    if args.list_cameras:
        # Enumeration touches no config and opens no output; keep it ahead of
        # config load so a broken config file does not block device discovery.
        return list_cameras()
    try:
        try:
            cfg = config_from_args(args)
        except (OSError, ValueError) as exc:
            log.error("invalid configuration: %s", format_config_error(exc))
            return EXIT_CONFIG
        if args.dump_config:
            cfg.save(args.dump_config)
            print(f"config written to {args.dump_config}")
            return 0
        if args.show_api_token:
            try:
                from .api.security import resolve_api_token

                print(resolve_api_token(cfg.api.token_file).value)
                return 0
            except (OSError, ValueError) as exc:
                log.error("cannot resolve API token: %s", format_config_error(exc))
                return EXIT_CONFIG
        if args.show_renderer_token:
            try:
                from .api.security import resolve_renderer_token

                print(
                    resolve_renderer_token(
                        cfg.api.renderer_token_file,
                        create=True,
                    ).value
                )
                return 0
            except (OSError, ValueError) as exc:
                log.error(
                    "cannot resolve renderer token: %s",
                    format_config_error(exc),
                )
                return EXIT_CONFIG
        try:
            return run(cfg, run_id=logging_session.run_id)
        except (OSError, ValueError) as exc:
            log.error(
                "startup configuration error: %s",
                format_config_error(exc),
            )
            return EXIT_CONFIG
    finally:
        logging_session.close()


if __name__ == "__main__":
    raise SystemExit(main())
