"""CLI entry point: `custback-avatar` or `python -m custback.avatar`."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
from importlib import resources
from pathlib import Path

from typing import get_args

from .. import _platform as platform_fs
from ..config import format_config_error
from .config import (
    AVATAR_PARTS,
    BUILTIN_AVATARS,
    AvatarConfig,
    AvatarFraming,
    AvatarRuntime,
    AvatarStyle,
)

log = logging.getLogger(__name__)

EXIT_RUNTIME = 1
EXIT_CONFIG = 2
EXIT_API = 3


def build_parser(*, prog: str = "custback-avatar") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Avatar renderer for custback: consumes raw camera frames over "
            "the custback WebSocket API and returns avatar frames "
            "(run custback with --mode remote)"
        ),
    )
    parser.add_argument("-c", "--config", help="path to YAML config file")
    parser.add_argument(
        "--source",
        help="custback API base URL, ws://host:8710 or wss://host:8710 "
        "(default ws://127.0.0.1:8710)",
    )
    parser.add_argument(
        "--source-token-file",
        help="path to custback's renderer-only frame token "
        "(CUSTBACK_RENDERER_TOKEN overrides)",
    )
    parser.add_argument(
        "--driver",
        choices=("auto", "vision", "audio2face", "idle"),
        help="animation driver (default auto: vision if available, else idle)",
    )
    parser.add_argument(
        "--a2f-url",
        help="Audio2Face-3D endpoint: grpc://127.0.0.1:port or "
        "grpcs://host:port (implies --driver audio2face)",
    )
    parser.add_argument(
        "--rig", help="avatar rig: 'builtin' or a directory of PNG layers"
    )
    parser.add_argument(
        "--avatar",
        choices=BUILTIN_AVATARS,
        help="builtin avatar character (default casey; ignored for PNG rigs)",
    )
    parser.add_argument(
        "--style",
        choices=get_args(AvatarStyle),
        help="render style (default cartoon; PNG rigs support sketch only)",
    )
    parser.add_argument(
        "--framing",
        choices=get_args(AvatarFraming),
        help="how much of the avatar stays in frame "
        "(default bust: head and chest, right for meeting tiles)",
    )
    parser.add_argument(
        "--parts",
        help=f"comma-separated visible parts from: {', '.join(AVATAR_PARTS)}",
    )
    parser.add_argument(
        "--scale", type=float, help="avatar height / frame height (0.1-3)"
    )
    parser.add_argument("--offset-x", type=float, help="horizontal shift, -1..1")
    parser.add_argument(
        "--offset-y", type=float, help="vertical shift, -1..1 (negative = up)"
    )
    parser.add_argument(
        "--bg-mode", choices=("color", "image", "video", "blur"), help="background mode"
    )
    parser.add_argument(
        "--bg-image", help="background image path (implies --bg-mode image)"
    )
    parser.add_argument(
        "--bg-video", help="background video path (implies --bg-mode video)"
    )
    parser.add_argument("--no-api", action="store_true", help="disable the control API")
    parser.add_argument("--api-host", help="control API bind host (default 127.0.0.1)")
    parser.add_argument("--api-port", type=int, help="control API port (default 8711)")
    parser.add_argument(
        "--api-token-file", help="path to the mode-0600 avatar API token file"
    )
    parser.add_argument(
        "--allow-non-loopback-api",
        action="store_true",
        help="allow a TLS-protected control API bind outside loopback",
    )
    parser.add_argument("--api-tls-cert", help="TLS certificate for the control API")
    parser.add_argument("--api-tls-key", help="TLS private key for the control API")
    parser.add_argument(
        "--show-api-token",
        action="store_true",
        help="print the resolved avatar control API token and exit",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="initialize and tear down a hardware-free idle renderer, then exit",
    )
    storage_permissions = parser.add_mutually_exclusive_group()
    storage_permissions.add_argument(
        "--check-storage-permissions",
        action="store_true",
        help="audit existing rig/background store ownership and modes, then exit",
    )
    storage_permissions.add_argument(
        "--fix-storage-permissions",
        action="store_true",
        help="repair user-owned rig/background store modes, then exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log-file", metavar="PATH", help="write a rotating diagnostics log to PATH"
    )
    parser.add_argument(
        "--dump-config",
        metavar="PATH",
        help="write the effective config to PATH and exit",
    )
    return parser


def avatar_config_bytes() -> bytes:
    """Return the installed, annotated avatar configuration template."""

    return resources.files("custback.avatar").joinpath("avatar.yaml").read_bytes()


def export_avatar_config(
    destination: str | os.PathLike[str] | None,
    *,
    output=None,
) -> int:
    """Print or exclusively create a private copy of the bundled template."""

    contents = avatar_config_bytes()
    if destination is None or os.fspath(destination) == "-":
        stream = output if output is not None else sys.stdout.buffer
        stream.write(contents)
        return 0

    path = Path(destination).expanduser()
    descriptor = platform_fs.open_nofollow(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        platform_fs.set_private_mode(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return 0


def _config_export_destination(argv: list[str]) -> str | None | object:
    """Parse the command without letting the runtime parser consume it."""

    if argv[:2] != ["config", "export"]:
        return _NOT_CONFIG_EXPORT
    if len(argv) > 3:
        raise ValueError("usage: custback avatar config export [PATH]")
    return argv[2] if len(argv) == 3 else None


_NOT_CONFIG_EXPORT = object()


def config_from_args(args: argparse.Namespace) -> AvatarConfig:
    # Assemble one candidate so related overrides are validated atomically.
    values = AvatarConfig.load(args.config).to_dict()
    source = values["source"]
    driver = values["driver"]
    audio2face = driver["audio2face"]
    appearance = values["appearance"]
    background = values["background"]
    api = values["api"]
    if args.source:
        source["url"] = args.source
    if args.source_token_file:
        source["token_file"] = args.source_token_file
    if args.a2f_url:
        audio2face["url"] = args.a2f_url
        driver["backend"] = "audio2face"
    if args.driver:
        driver["backend"] = args.driver
    if args.rig:
        appearance["rig"] = args.rig
    if args.avatar:
        appearance["avatar"] = args.avatar
    if args.style:
        appearance["style"] = args.style
    if args.framing:
        appearance["framing"] = args.framing
    if args.parts:
        appearance["parts"] = tuple(
            part.strip() for part in args.parts.split(",") if part.strip()
        )
    if args.scale is not None:
        appearance["scale"] = args.scale
    if args.offset_x is not None:
        appearance["offset_x"] = args.offset_x
    if args.offset_y is not None:
        appearance["offset_y"] = args.offset_y
    if args.bg_image:
        background["image_path"] = args.bg_image
        background["mode"] = "image"
    if args.bg_video:
        background["video_path"] = args.bg_video
        background["mode"] = "video"
    if args.bg_mode:
        background["mode"] = args.bg_mode
    if args.no_api:
        api["enabled"] = False
    if args.api_host:
        api["host"] = args.api_host
    if args.api_port is not None:
        api["port"] = args.api_port
    if args.api_token_file:
        api["token_file"] = args.api_token_file
    if args.allow_non_loopback_api:
        api["allow_non_loopback"] = True
    if args.api_tls_cert:
        api["tls_certfile"] = args.api_tls_cert
    if args.api_tls_key:
        api["tls_keyfile"] = args.api_tls_key
    return AvatarConfig.from_dict(values)


def _storage_permission_command(cfg: AvatarConfig, *, fix: bool) -> int:
    """Run the operator-facing storage permission doctor/fix command."""

    from .store import (
        StoreError,
        audit_storage_permissions,
        repair_storage_permissions,
    )

    try:
        repaired = repair_storage_permissions(cfg.storage) if fix else ()
        remaining = audit_storage_permissions(cfg.storage)
    except (OSError, StoreError) as exc:
        print(
            f"custback-avatar: storage permission check failed: {exc}",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    if remaining:
        for issue in remaining:
            print(
                f"{issue.reason}: {issue.path} "
                f"mode={issue.actual_mode:04o} expected={issue.expected_mode:04o}",
                file=sys.stderr,
            )
        print(
            "custback-avatar: storage permissions need repair; "
            "run with --fix-storage-permissions",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    if fix:
        print(f"storage permissions repaired: {len(repaired)} path(s)")
    else:
        print("storage permissions ok")
    return 0


def _hardware_free_smoke() -> int:
    """Exercise installed renderer resources without sockets or hardware."""

    from .service import AvatarService

    cfg = AvatarConfig.from_dict(
        {
            "driver": {"backend": "idle"},
            "background": {"mode": "color"},
            "api": {"enabled": False},
        }
    )
    service = AvatarService(AvatarRuntime(cfg))
    try:
        service.activate_initial()
    finally:
        service.close()
    print("avatar hardware-free smoke ok")
    return 0


def _security_policy(cfg: AvatarConfig):
    from ..api.security import (
        SecurityConfigurationError,
        SecurityPolicy,
        is_loopback_host,
        validate_bind_security,
    )
    from .api import resolve_avatar_api_token

    api = cfg.api
    validate_bind_security(
        api.host,
        allow_non_loopback=api.allow_non_loopback,
        tls_certfile=api.tls_certfile,
        tls_keyfile=api.tls_keyfile,
    )
    if not is_loopback_host(api.host) and not api.allowed_origins:
        raise SecurityConfigurationError(
            "non-loopback control API binds require explicit api.allowed_origins"
        )
    token = resolve_avatar_api_token(api.token_file)
    if token.created:
        log.info("created avatar API token file %s", token.path)
    return SecurityPolicy.for_bind(
        token.value,
        api.host,
        api.port,
        allowed_origins=api.allowed_origins,
        tls=bool(api.tls_certfile),
        session_ttl_s=api.session_ttl_s,
    )


async def _serve(service, stop_signals: bool) -> int:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    registered: list[signal.Signals] = []
    if stop_signals:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, stop.set)
                registered.append(sig)
    try:
        await service.run(stop)
    finally:
        for sig in registered:
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(sig)
    return 0


def run(cfg: AvatarConfig) -> int:
    from ..__main__ import ApiStartupError, _ApiRunner
    from .api import create_avatar_app
    from .service import AvatarService

    security = None
    if cfg.api.enabled:
        security = _security_policy(cfg)

    runtime = AvatarRuntime(cfg)
    service = AvatarService(runtime)

    api_runner = None
    exit_code = 0
    try:
        # Do not publish the control plane until version zero has a complete,
        # trialed resource generation behind it. ``service.run`` observes this
        # generation and treats the call as an idempotent startup check.
        service.activate_initial()
        if cfg.api.enabled:
            assert security is not None
            app = create_avatar_app(runtime, service, security=security)
            api_runner = _ApiRunner(app, cfg.api)
            try:
                api_runner.start()
            except ApiStartupError:
                log.exception("avatar control API startup failed")
                return EXIT_API
            scheme = "https" if cfg.api.tls_certfile else "http"
            log.info(
                "avatar control API ready at %s://%s:%s",
                scheme,
                cfg.api.host,
                cfg.api.port,
            )
        log.info(
            "avatar service starting: driver=%s rig=%s avatar=%s style=%s "
            "framing=%s parts=%s scale=%.2f",
            cfg.driver.backend,
            "builtin" if cfg.appearance.rig == "builtin" else "custom",
            cfg.appearance.avatar,
            cfg.appearance.style,
            cfg.appearance.framing,
            ",".join(cfg.appearance.parts),
            cfg.appearance.scale,
        )
        exit_code = asyncio.run(_serve(service, stop_signals=True))
    except KeyboardInterrupt:
        exit_code = 0
    except Exception:
        log.exception("avatar service failed")
        exit_code = EXIT_RUNTIME
    finally:
        if api_runner is not None and not api_runner.stop():
            log.error("avatar control API did not stop cleanly")
            exit_code = exit_code or EXIT_API
        try:
            service.close()
        except Exception:
            log.exception("avatar service resource teardown failed")
            exit_code = exit_code or EXIT_RUNTIME
    return exit_code


def main(argv: list[str] | None = None, *, prog: str = "custback-avatar") -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        destination = _config_export_destination(effective_argv)
    except ValueError as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    if destination is not _NOT_CONFIG_EXPORT:
        try:
            return export_avatar_config(destination)
        except (OSError, ValueError) as exc:
            print(f"{prog}: cannot export avatar config: {exc}", file=sys.stderr)
            return EXIT_CONFIG

    args = build_parser(prog=prog).parse_args(effective_argv)
    from ..diagnostics import LoggingConfigurationError, configure_logging

    try:
        # Console logging by default: the rotating file log belongs to the
        # custback process; two writers must not rotate the same file.
        logging_session = configure_logging(
            verbose=args.verbose,
            log_file=args.log_file,
            no_file_log=not args.log_file,
        )
    except (LoggingConfigurationError, ValueError) as exc:
        print(f"custback-avatar: logging configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        try:
            cfg = config_from_args(args)
        except (OSError, ValueError) as exc:
            log.error("invalid configuration: %s", format_config_error(exc))
            return EXIT_CONFIG
        if args.check_storage_permissions or args.fix_storage_permissions:
            return _storage_permission_command(
                cfg,
                fix=args.fix_storage_permissions,
            )
        if args.smoke:
            try:
                return _hardware_free_smoke()
            except Exception:
                log.exception("avatar hardware-free smoke failed")
                return EXIT_RUNTIME
        if args.dump_config:
            cfg.save(args.dump_config)
            print(f"config written to {args.dump_config}")
            return 0
        if args.show_api_token:
            try:
                from .api import resolve_avatar_api_token

                print(resolve_avatar_api_token(cfg.api.token_file).value)
                return 0
            except (OSError, ValueError) as exc:
                log.error(
                    "cannot resolve avatar API token: %s",
                    format_config_error(exc),
                )
                return EXIT_CONFIG
        try:
            return run(cfg)
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
