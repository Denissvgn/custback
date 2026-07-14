"""CLI entry point: `custback` or `python -m custback`."""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import threading
import time
from pathlib import Path

from .backgrounds import IMAGE_EXTS, VIDEO_EXTS
from .config import MODES, AppConfig, RuntimeConfig
from .hub import FrameHub
from .pipeline import Pipeline

log = logging.getLogger(__name__)

EXIT_RUNTIME = 1
EXIT_CONFIG = 2
EXIT_API = 3
API_START_TIMEOUT_S = 5.0


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
    parser.add_argument("--mirror", action="store_true", help="mirror the camera")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="use a synthetic test source instead of a real camera",
    )
    parser.add_argument("--mode", choices=MODES, help="background mode")
    parser.add_argument("--image", help="background image path (implies --mode image)")
    parser.add_argument("--video", help="background video path (implies --mode video)")
    parser.add_argument(
        "--bg-camera",
        help="second camera / stream URL as live backdrop (implies --mode camera)",
    )
    parser.add_argument("--blur", type=int, help="blur strength (implies --mode blur)")
    parser.add_argument("--no-api", action="store_true", help="disable the HTTP API")
    parser.add_argument("--api-host", help="API bind host (default 127.0.0.1)")
    parser.add_argument("--api-port", type=int, help="API port (default 8710)")
    parser.add_argument("--api-token-file", help="path to the mode-0600 API token file")
    parser.add_argument(
        "--allow-non-loopback-api", action="store_true",
        help="allow a TLS-protected API bind outside loopback",
    )
    parser.add_argument("--api-tls-cert", help="TLS certificate for the API")
    parser.add_argument("--api-tls-key", help="TLS private key for the API")
    parser.add_argument(
        "--show-api-token", action="store_true",
        help="print the resolved API token and exit",
    )
    parser.add_argument(
        "--no-vcam", action="store_true",
        help="do not open a virtual camera (serve frames only via the API)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="show the processed output in an on-screen window (q/ESC quits)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--dump-config", metavar="PATH",
        help="write the effective config to PATH and exit",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.load(args.config)
    cam, bg, api = cfg.camera, cfg.background, cfg.api
    if args.camera is not None:
        cam.device = args.camera
    if args.width is not None:
        cam.width = args.width
    if args.height is not None:
        cam.height = args.height
    if args.fps is not None:
        cam.fps = args.fps
        cfg.output.fps = args.fps
    if args.mirror:
        cam.mirror = True
    if args.synthetic:
        cam.synthetic = True
    if args.image:
        # Route by extension rather than trusting the flag name: users
        # commonly reach for whichever flag they remember, and preview.py's
        # n/p file cycling already auto-detects the same way.
        if Path(args.image).suffix.lower() in VIDEO_EXTS:
            bg.video_path = args.image
            bg.mode = "video"
        else:
            bg.image_path = args.image
            bg.mode = "image"
    if args.video:
        if Path(args.video).suffix.lower() in IMAGE_EXTS:
            bg.image_path = args.video
            bg.mode = "image"
        else:
            bg.video_path = args.video
            bg.mode = "video"
    if args.bg_camera:
        bg.camera_device = args.bg_camera
        bg.mode = "camera"
    if args.blur is not None:
        bg.blur_strength = args.blur
        bg.mode = "blur"
    if args.mode:
        bg.mode = args.mode
    if args.no_api:
        api.enabled = False
    if args.api_host:
        api.host = args.api_host
    if args.api_port is not None:
        api.port = args.api_port
    if args.api_token_file:
        api.token_file = args.api_token_file
    if args.allow_non_loopback_api:
        api.allow_non_loopback = True
    if args.api_tls_cert or args.api_tls_key:
        api_values = api.model_dump(mode="python")
        if args.api_tls_cert:
            api_values["tls_certfile"] = args.api_tls_cert
        if args.api_tls_key:
            api_values["tls_keyfile"] = args.api_tls_key
        cfg.api = type(api).model_validate(api_values)
    if args.no_vcam:
        cfg.output.backend = "null"
    if args.preview:
        cfg.output.preview = True
    cfg.validate()
    return cfg


class ApiStartupError(RuntimeError):
    """The configured API could not become or remain available."""


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
            self._socket = self.server.config.bind_socket()
        except SystemExit as exc:
            raise ApiStartupError("API socket bind failed") from exc
        except OSError as exc:
            raise ApiStartupError(f"API socket bind failed: {exc}") from exc

        def serve() -> None:
            try:
                self.server.run(sockets=[self._socket])
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
                raise ApiStartupError(f"API startup failed: {self._error or 'server exited'}")
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
) -> int:
    """Monitor required services until shutdown; return the selected status."""
    while not stop.is_set() and pipeline.running:
        if api_runner is not None and api_runner.failed:
            log.error("API server stopped unexpectedly: %s", api_runner.error)
            stop.set()
            return EXIT_API
        stop.wait(0.5)
    return 0


def _security_policy(cfg: AppConfig):
    from .api.security import (
        SecurityConfigurationError,
        SecurityPolicy,
        is_loopback_host,
        resolve_api_token,
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
    return SecurityPolicy.for_bind(
        token.value,
        api.host,
        api.port,
        allowed_origins=api.allowed_origins,
        tls=bool(api.tls_certfile),
        session_ttl_s=api.session_ttl_s,
    )


def run(cfg: AppConfig) -> int:
    if cfg.output.preview:
        from .preview import preview_available

        available, reason = preview_available()
        if not available:
            log.warning("preview disabled: %s", reason)
            cfg.output.preview = False

    runtime = RuntimeConfig(cfg)
    hub = FrameHub()
    pipeline = Pipeline(runtime, hub)

    api_runner: _ApiRunner | None = None
    if cfg.api.enabled:
        from .api.server import create_app

        # Security validation is unconditional: accepting an injected policy
        # here could otherwise bypass non-loopback/TLS/origin checks while the
        # server still binds to cfg.api.host.
        security = _security_policy(cfg)
        app = create_app(runtime, hub, pipeline, security=security)
        api_runner = _ApiRunner(app, cfg.api)
        try:
            api_runner.start()
        except ApiStartupError:
            log.exception("API startup failed")
            return EXIT_API
        scheme = "https" if cfg.api.tls_certfile else "http"
        log.info("API listening on %s://%s:%d", scheme, cfg.api.host, cfg.api.port)

    try:
        pipeline.start()
    except BaseException:
        api_stop_failed = api_runner is not None and not api_runner.stop()
        log.exception("pipeline startup failed")
        if api_stop_failed:
            log.error("API server survived failed pipeline startup")
            return EXIT_API
        return EXIT_RUNTIME

    stop = threading.Event()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())

    exit_code = 0
    try:
        if cfg.output.preview:
            # Runs on the main thread (required by OpenCV GUI on macOS);
            # returns when the user quits the window or `stop` is set.
            from .preview import run_preview

            watchdog = threading.Thread(
                target=_watch_pipeline, args=(pipeline, stop, api_runner),
                name="watchdog", daemon=True,
            )
            watchdog.start()
            user_quit = run_preview(runtime, hub, stop, coordinator=pipeline)
            if not user_quit and not stop.is_set():
                log.warning("continuing without the preview window")
                exit_code = _wait_headless(pipeline, stop, api_runner)
        else:
            exit_code = _wait_headless(pipeline, stop, api_runner)
    finally:
        # No `return` in this block: a return inside `finally` silences any
        # exception still propagating from the `try` above, which would mask
        # a real crash behind an unrelated pipeline.stop() failure.
        pipeline_stop_failed = False
        try:
            pipeline.stop()
        except Exception:
            log.exception("pipeline error")
            pipeline_stop_failed = True
        if api_runner is not None:
            if api_runner.failed:
                exit_code = EXIT_API
            if not api_runner.stop():
                log.error("API server did not stop cleanly")
                exit_code = EXIT_API
    if pipeline_stop_failed:
        return EXIT_RUNTIME
    if api_runner is not None and api_runner.failed:
        return EXIT_API
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        cfg = config_from_args(args)
    except (OSError, ValueError) as exc:
        log.error("invalid configuration: %s", exc)
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
            log.error("cannot resolve API token: %s", exc)
            return EXIT_CONFIG
    try:
        return run(cfg)
    except (OSError, ValueError) as exc:
        log.error("startup configuration error: %s", exc)
        return EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
