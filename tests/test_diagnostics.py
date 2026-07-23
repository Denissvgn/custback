"""Tests for durable, privacy-safe runtime diagnostics."""

from __future__ import annotations

import io
import logging
import os
import stat

import pytest

from custback.diagnostics import (
    LoggingConfigurationError,
    audit_config_change,
    changed_field_names,
    configure_logging,
    default_log_path,
    redact_sensitive_text,
    sanitized_config_summary,
    sanitized_source,
)


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_default_log_path_prefers_absolute_xdg_state_home(tmp_path):
    assert default_log_path(environ={"XDG_STATE_HOME": str(tmp_path)}) == (
        tmp_path / "custback" / "custback.log"
    )


def test_default_log_path_ignores_relative_xdg_state_home(tmp_path):
    assert (
        default_log_path(environ={"XDG_STATE_HOME": "relative"}, home=tmp_path)
        == tmp_path / ".local" / "state" / "custback" / "custback.log"
    )


def test_configure_logging_creates_secure_default_log_and_run_id(tmp_path):
    logger = logging.Logger("custback-test")
    session = configure_logging(
        environ={"XDG_STATE_HOME": str(tmp_path)},
        logger=logger,
        run_id="run12345",
    )
    try:
        logger.info("ready")
        assert session.file_logging is True
        assert session.log_path == tmp_path / "custback" / "custback.log"
        log_path = session.log_path
        assert log_path is not None
        assert "[run=run12345]" in log_path.read_text(encoding="utf-8")
        assert _mode(log_path.parent) == 0o700
        assert _mode(log_path) == 0o600
    finally:
        session.close()


def test_rotated_logs_remain_mode_0600(tmp_path):
    logger = logging.Logger("custback-rollover-test")
    path = tmp_path / "logs" / "custback.log"
    session = configure_logging(
        log_file=path,
        logger=logger,
        run_id="rollover",
        max_bytes=80,
        backup_count=3,
    )
    try:
        for index in range(20):
            logger.info("message %d %s", index, "x" * 40)
    finally:
        session.close()
    files = sorted(path.parent.glob("custback.log*"))
    assert path in files
    assert 2 <= len(files) <= 4
    assert all(_mode(candidate) == 0o600 for candidate in files)


def test_existing_log_and_backups_are_secured_immediately_at_startup(tmp_path):
    logger = logging.Logger("custback-existing-log-modes-test")
    path = tmp_path / "logs" / "custback.log"
    path.parent.mkdir()
    existing = [
        path,
        path.with_name("custback.log.1"),
        path.with_name("custback.log.2"),
    ]
    for candidate in existing:
        candidate.write_text("old log\n", encoding="utf-8")
        candidate.chmod(0o666)

    session = configure_logging(
        log_file=path,
        logger=logger,
        run_id="existing-modes",
        backup_count=3,
    )
    try:
        assert all(_mode(candidate) == 0o600 for candidate in existing)
    finally:
        session.close()


def test_existing_backup_symlink_is_never_followed(tmp_path):
    logger = logging.Logger("custback-backup-symlink-test")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    path = log_dir / "custback.log"
    path.write_text("current\n", encoding="utf-8")
    target = tmp_path / "outside.log"
    target.write_text("outside\n", encoding="utf-8")
    target.chmod(0o644)
    os.symlink(target, log_dir / "custback.log.1")

    with pytest.raises(LoggingConfigurationError, match="non-regular log file"):
        configure_logging(
            log_file=path,
            logger=logger,
            run_id="backup-symlink",
            backup_count=3,
        )
    assert _mode(target) == 0o644


def test_no_file_log_keeps_only_console_handler():
    logger = logging.Logger("custback-console-test")
    session = configure_logging(no_file_log=True, logger=logger, run_id="console")
    try:
        assert session.file_logging is False
        assert session.log_path is None
        assert len(logger.handlers) == 1
    finally:
        session.close()


def test_explicit_log_failure_is_a_configuration_error(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked", encoding="utf-8")
    logger = logging.Logger("custback-explicit-failure-test")
    with pytest.raises(LoggingConfigurationError, match="cannot configure log file"):
        configure_logging(log_file=blocker / "custback.log", logger=logger)


def test_default_log_failure_falls_back_to_console(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked", encoding="utf-8")
    logger = logging.Logger("custback-default-failure-test")
    session = configure_logging(
        environ={"XDG_STATE_HOME": str(blocker)},
        logger=logger,
        run_id="fallback",
    )
    try:
        assert session.file_logging is False
        assert session.log_path == blocker / "custback" / "custback.log"
        assert len(logger.handlers) == 1
    finally:
        session.close()


def test_log_file_and_disable_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="cannot be used together"):
        configure_logging(
            log_file=tmp_path / "custback.log",
            no_file_log=True,
            logger=logging.Logger("custback-conflict-test"),
        )


def test_config_audit_logs_only_field_names_and_version():
    stream = io.StringIO()
    logger = logging.Logger("custback-audit-test")
    logger.addHandler(logging.StreamHandler(stream))
    secret_url = "https://alice:secret@example.invalid/private"
    patch = {
        "background": {"mode": "remote", "remote_url": secret_url},
        "api": {"token": "super-secret"},
    }
    assert changed_field_names(patch) == (
        "api.token",
        "background.mode",
        "background.remote_url",
    )
    audit_config_change("preview", patch, version=7, logger=logger)
    message = stream.getvalue()
    assert "origin=preview" in message
    assert "config_version=7" in message
    assert "background.remote_url" in message
    assert secret_url not in message
    assert "super-secret" not in message


def test_remote_sources_are_reduced_to_credential_free_labels():
    source = "rtsp://camera-user:camera-password@example.test/live?token=secret"
    assert sanitized_source(source) == "rtsp://<redacted>"
    assert sanitized_source("/dev/video2") == "/dev/video2"
    message = redact_sensitive_text(f"failed to open {source}")
    assert message == "failed to open rtsp://example.test/<redacted>"
    assert "camera-password" not in message


def test_log_formatter_redacts_urls_inside_messages_and_tracebacks(tmp_path):
    logger = logging.Logger("custback-redaction-test")
    session = configure_logging(
        log_file=tmp_path / "custback.log",
        logger=logger,
        run_id="redaction",
    )
    secret = "rtsp://alice:password@example.test/live?token=secret"
    try:
        try:
            raise RuntimeError(f"camera failed for {secret}")
        except RuntimeError:
            logger.exception("source startup failed: %s", secret)
    finally:
        session.close()
    text = (tmp_path / "custback.log").read_text(encoding="utf-8")
    assert "alice" not in text
    assert "password" not in text
    assert "token=secret" not in text
    assert "rtsp://example.test/<redacted>" in text


def test_config_summary_whitelists_safe_values_and_redacts_sources():
    config = {
        "background": {
            "mode": "video",
            "video_path": "/private/beach.mp4",
            "blur_strength": 41,
        },
        "api": {"allowed_origins": ["https://alice:secret@example.test/app"]},
    }
    summary = sanitized_config_summary(
        config,
        [
            "background.mode",
            "background.video_path",
            "background.blur_strength",
            "api.allowed_origins",
        ],
    )
    assert "background.mode=video" in summary
    assert "background.blur_strength=41" in summary
    assert "background.video_path=<redacted>" in summary
    assert "api.allowed_origins=<redacted>" in summary
    assert "beach.mp4" not in summary
    assert "alice" not in summary
