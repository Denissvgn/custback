"""Process diagnostics shared by the CLI and interactive preview.

The command line entry point deliberately keeps argument parsing and lifecycle
coordination in ``__main__``.  This module owns the lower-level, testable parts
of diagnostics: secure rotating logs, per-run correlation, and value-free
configuration audit records.
"""

from __future__ import annotations

import errno
import logging
import logging.handlers
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from . import _platform as platform_fs

DEFAULT_LOG_BYTES = 5 * 1024 * 1024
DEFAULT_LOG_BACKUPS = 3
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [run=%(run_id)s]: %(message)s"
_URL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>'\"]+")
_SAFE_CONFIG_STRINGS = {
    "acceleration.mode",
    "acceleration.provider",
    "background.fit_mode",
    "background.mode",
    "background.remote_fallback_mode",
    "camera.fit_mode",
    "camera.mode_mismatch",
    "camera.pixel_format",
    "compositing.blend_space",
    "compositing.color_correction.mode",
    "output.backend",
    "segmentation.backend",
    "segmentation.delegate",
}
_SENSITIVE_FIELD_PARTS = {
    "cert",
    "credential",
    "device",
    "file",
    "key",
    "origin",
    "password",
    "path",
    "secret",
    "token",
    "url",
}


class LoggingConfigurationError(OSError):
    """An explicitly requested log destination could not be configured."""


def sanitized_source(value: object) -> str:
    """Render a local source while redacting every URL path and credential.

    Camera/background source errors are persisted in the durable log.  A full
    RTSP/HTTP URL may contain both user information and bearer-style query
    parameters, so URL-like values are reduced to a scheme-only label before
    they are placed in an exception or log record.
    """

    text = str(value)
    try:
        parsed = urlsplit(text)
    except ValueError:
        parsed = None
    if parsed is not None and parsed.scheme and (parsed.netloc or "://" in text):
        scheme = parsed.scheme.lower()[:32]
        return f"{scheme}://<redacted>"
    return text


def redact_sensitive_text(value: object) -> str:
    """Strip URL credentials, paths, queries and fragments from log text."""

    def redact(match: re.Match[str]) -> str:
        raw = match.group(0)
        trailing = ""
        while raw and raw[-1] in ").,;]}":
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        try:
            parsed = urlsplit(raw)
            hostname = parsed.hostname
            port = parsed.port
        except (ValueError, AttributeError):
            return "<redacted-url>" + trailing
        if not parsed.scheme or hostname is None:
            return "<redacted-url>" + trailing
        shown_host = f"[{hostname}]" if ":" in hostname else hostname
        origin = f"{parsed.scheme.lower()}://{shown_host}"
        if port is not None:
            origin += f":{port}"
        sensitive = bool(
            parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        )
        return (origin + ("/<redacted>" if sensitive else "")) + trailing

    return _URL_RE.sub(redact, str(value))


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive_text(super().format(record))


def sanitized_config_summary(
    config: object, fields: tuple[str, ...] | list[str]
) -> str:
    """Summarize acknowledged values using a deliberately narrow whitelist."""

    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        data = model_dump(mode="python")
    elif isinstance(config, Mapping):
        data = config
    else:
        return "none"

    entries: list[str] = []
    for field_name in fields:
        value: object = data
        try:
            for component in field_name.split("."):
                if not isinstance(value, Mapping):
                    raise KeyError(component)
                value = value[component]
        except KeyError:
            entries.append(f"{field_name}=<unavailable>")
            continue

        components = set(field_name.lower().replace("-", "_").split("."))
        sensitive = any(
            part in component.split("_")
            for component in components
            for part in _SENSITIVE_FIELD_PARTS
        )
        if sensitive:
            shown = "<redacted>"
        elif isinstance(value, bool) or value is None:
            shown = str(value).lower()
        elif isinstance(value, (int, float)):
            shown = str(value)
        elif isinstance(value, str) and field_name in _SAFE_CONFIG_STRINGS:
            shown = value
        elif (
            isinstance(value, (list, tuple))
            and len(value) <= 4
            and all(isinstance(item, (int, float, bool)) for item in value)
        ):
            shown = "[" + ",".join(str(item) for item in value) + "]"
        else:
            shown = "<redacted>"
        entries.append(f"{field_name}={shown}")
    return ";".join(entries) or "none"


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        # A handler filter is used instead of a root-logger filter: logger
        # filters are not applied to records propagated from child loggers.
        record.run_id = self.run_id
        return True


class SecureRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A rotating handler whose current and backup files remain mode 0600."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            # RotatingFileHandler opens (and therefore secures) the current
            # file, but leaves old backups untouched until a later rollover.
            self._secure_existing_files()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _secure_existing_file(path: str) -> None:
        try:
            before = os.lstat(path)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(before.st_mode):
            raise OSError(
                errno.ELOOP if stat.S_ISLNK(before.st_mode) else errno.EINVAL,
                "refusing to secure a non-regular log file",
                path,
            )

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        fd = platform_fs.open_nofollow(path, flags)
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
            ):
                raise OSError(
                    errno.EAGAIN,
                    "log file changed while permissions were being secured",
                    path,
                )
            platform_fs.set_private_mode(fd, 0o600)
        finally:
            os.close(fd)

    def _secure_existing_files(self) -> None:
        for index in range(self.backupCount + 1):
            path = self.baseFilename if index == 0 else f"{self.baseFilename}.{index}"
            self._secure_existing_file(path)

    def _open(self):
        flags = os.O_WRONLY | os.O_CREAT
        flags |= os.O_APPEND if self.mode.startswith("a") else os.O_TRUNC
        flags |= getattr(os, "O_CLOEXEC", 0)
        fd = platform_fs.open_nofollow(self.baseFilename, flags, 0o600)
        try:
            platform_fs.set_private_mode(fd, 0o600)
            return os.fdopen(
                fd,
                self.mode,
                encoding=self.encoding,
                errors=self.errors,
            )
        except BaseException:
            os.close(fd)
            raise

    def doRollover(self) -> None:  # noqa: N802 - logging API spelling
        super().doRollover()
        self._secure_existing_files()


@dataclass
class LoggingSession:
    """Result of :func:`configure_logging`, including the public run ID."""

    run_id: str
    log_path: Path | None
    file_logging: bool
    _logger: logging.Logger = field(repr=False)
    _handlers: tuple[logging.Handler, ...] = field(repr=False)

    def close(self) -> None:
        """Flush and detach only the handlers installed for this session."""
        for handler in self._handlers:
            handler.flush()
            self._logger.removeHandler(handler)
            handler.close()


def default_log_path(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the XDG-compatible per-user log path.

    A relative ``XDG_STATE_HOME`` is invalid per the XDG base-directory
    specification and is ignored rather than being resolved against an
    unpredictable working directory.
    """

    env = os.environ if environ is None else environ
    state_home = env.get("XDG_STATE_HOME", "")
    if state_home and Path(state_home).is_absolute():
        base = Path(state_home)
    else:
        resolved_home = home
        if resolved_home is None:
            configured_home = env.get("HOME")
            resolved_home = Path(configured_home) if configured_home else Path.home()
        base = resolved_home / ".local" / "state"
    return base / "custback" / "custback.log"


def _prepare_log_directory(path: Path, *, managed_default: bool) -> None:
    parent = path.parent
    existed = parent.exists()
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    # Always secure custback's managed state directory.  For an explicit path,
    # do not unexpectedly chmod a pre-existing directory owned by the user.
    if managed_default or not existed:
        parent.chmod(0o700)


def configure_logging(
    *,
    verbose: bool = False,
    log_file: str | os.PathLike[str] | None = None,
    no_file_log: bool = False,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    run_id: str | None = None,
    logger: logging.Logger | None = None,
    max_bytes: int = DEFAULT_LOG_BYTES,
    backup_count: int = DEFAULT_LOG_BACKUPS,
) -> LoggingSession:
    """Configure console and optional secure rotating file logging.

    Failure of the managed default path is non-fatal and leaves console
    logging active.  Failure of an explicitly supplied path raises
    :class:`LoggingConfigurationError`, allowing the CLI to return its
    configuration-error exit code.
    """

    if no_file_log and log_file is not None:
        raise ValueError("--log-file and --no-file-log cannot be used together")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if backup_count < 0:
        raise ValueError("backup_count must be non-negative")

    target_logger = logging.getLogger() if logger is None else logger
    target_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    target_logger.propagate = False if logger is not None else target_logger.propagate
    for existing in tuple(target_logger.handlers):
        if getattr(existing, "_custback_diagnostics_handler", False):
            target_logger.removeHandler(existing)
            existing.close()

    resolved_run_id = run_id or secrets.token_hex(4)
    formatter = _RedactingFormatter(LOG_FORMAT)
    run_filter = _RunIdFilter(resolved_run_id)
    console = logging.StreamHandler()
    console._custback_diagnostics_handler = True  # type: ignore[attr-defined]
    console.setFormatter(formatter)
    console.addFilter(run_filter)
    target_logger.addHandler(console)
    handlers: list[logging.Handler] = [console]

    if no_file_log:
        return LoggingSession(
            resolved_run_id, None, False, target_logger, tuple(handlers)
        )

    managed_default = log_file is None
    path = (
        default_log_path(environ=environ, home=home)
        if managed_default
        else Path(log_file).expanduser()
    )
    try:
        _prepare_log_directory(path, managed_default=managed_default)
        file_handler = SecureRotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler._custback_diagnostics_handler = True  # type: ignore[attr-defined]
        file_handler.setFormatter(formatter)
        file_handler.addFilter(run_filter)
        target_logger.addHandler(file_handler)
        handlers.append(file_handler)
    except (OSError, ValueError) as exc:
        if not managed_default:
            raise LoggingConfigurationError(
                f"cannot configure log file {path}: {exc}"
            ) from exc
        target_logger.warning(
            "cannot enable default file log %s; continuing with stderr: %s",
            path,
            exc,
        )
        return LoggingSession(
            resolved_run_id, path, False, target_logger, tuple(handlers)
        )

    return LoggingSession(resolved_run_id, path, True, target_logger, tuple(handlers))


def changed_field_names(patch: Mapping[str, object]) -> tuple[str, ...]:
    """Return sorted dotted field names without retaining their values."""

    fields: list[str] = []

    def visit(value: Mapping[str, object], prefix: str) -> None:
        for key, child in value.items():
            field_name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, Mapping):
                visit(child, field_name)
            else:
                fields.append(field_name)

    visit(patch, "")
    return tuple(sorted(fields))


def audit_config_change(
    origin: str,
    patch: Mapping[str, object],
    *,
    version: int | None,
    logger: logging.Logger | None = None,
) -> tuple[str, ...]:
    """Log a successful configuration change without logging field values."""

    fields = changed_field_names(patch)
    audit_log = logging.getLogger(__name__) if logger is None else logger
    audit_log.info(
        "config change accepted origin=%s fields=%s config_version=%s",
        origin,
        ",".join(fields) or "none",
        "unknown" if version is None else version,
    )
    return fields
