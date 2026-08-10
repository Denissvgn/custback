"""Tests for durable, privacy-safe runtime diagnostics."""

from __future__ import annotations

import io
import logging
import os
import stat
from pathlib import Path

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


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "model load failed at /home/alice/Private Models/rvm model.onnx; using cpu",
            "model load failed at <redacted-path>; using cpu",
        ),
        (
            r"model load failed at C:\Program Files (x86)\Private Models\rvm "
            r"model.onnx; "
            "using cpu",
            "model load failed at <redacted-path>; using cpu",
        ),
        (
            r"model load failed at \\studio-server\private share\Models\rvm.onnx; "
            "using cpu",
            "model load failed at <redacted-path>; using cpu",
        ),
        (
            "device=/dev/v4l/by-id/private-camera "
            "model=/mnt/private/models/rvm.onnx "
            "token_file=/home/alice/.config/custback/api-token",
            "device=<redacted-path> model=<redacted-path> token_file=<redacted-path>",
        ),
    ],
)
def test_local_source_paths_are_redacted_without_losing_public_context(
    message,
    expected,
):
    assert redact_sensitive_text(message) == expected


def test_log_formatter_redacts_exception_messages_and_traceback_sources(tmp_path):
    logger = logging.Logger("custback-local-path-redaction-test")
    session = configure_logging(
        log_file=tmp_path / "custback.log",
        logger=logger,
        run_id="local-path-redaction",
    )
    source_path = "/home/alice/Private Source/startup module.py"
    model_path = r"C:\Users\Alice\Private Models\rvm model.onnx"
    try:
        try:
            code = compile(
                f"raise RuntimeError({model_path!r})",
                source_path,
                "exec",
            )
            exec(code, {})
        except RuntimeError:
            logger.exception("model startup failed for %s", "/dev/video99")
        logger.info(
            "created API token file %s; permissions=private",
            Path("/home/alice/Private Config/api token"),
        )
    finally:
        session.close()

    text = (tmp_path / "custback.log").read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in text
    assert "RuntimeError" in text
    assert text.count("<redacted-path>") >= 4
    assert "/home/alice" not in text
    assert "/mnt/data/projects" not in text
    assert r"C:\Users\Alice" not in text
    assert "/dev/video99" not in text
    assert "Private Config" not in text


def test_benign_one_line_diagnostics_and_safe_basenames_are_preserved():
    message = (
        "ready backend=rvm/cpu provider=CUDAExecutionProvider "
        "model=rvm_mobilenetv3_fp32.onnx device=cuda route=/status fps=30/30"
    )
    assert redact_sensitive_text(message) == message
    assert redact_sensitive_text("api_token=private-value ready") == (
        "api_token=<redacted> ready"
    )
    assert (
        redact_sensitive_text("/home/alice/Private Config/api token")
        == "<redacted-path>"
    )
    assert (
        redact_sensitive_text('  File "relative/private_source.py", line 7, in start')
        == '  File "<redacted-path>", line 7, in start'
    )


def test_config_summary_whitelists_safe_values_and_redacts_sources():
    config = {
        "schema_version": 1,
        "background": {
            "mode": "video",
            "video_path": "/private/beach.mp4",
            "blur_strength": 41,
            "fit_mode": "cover",
        },
        "camera": {"fit_mode": "stretch"},
        "compositing": {
            "blend_space": "srgb_legacy",
            "color_correction": {"mode": "off", "strength": 0.5},
        },
        "api": {"allowed_origins": ["https://alice:secret@example.test/app"]},
    }
    summary = sanitized_config_summary(
        config,
        [
            "schema_version",
            "background.mode",
            "background.video_path",
            "background.blur_strength",
            "background.fit_mode",
            "camera.fit_mode",
            "compositing.blend_space",
            "compositing.color_correction.mode",
            "compositing.color_correction.strength",
            "api.allowed_origins",
        ],
    )
    assert "schema_version=1" in summary
    assert "background.mode=video" in summary
    assert "background.blur_strength=41" in summary
    assert "background.fit_mode=cover" in summary
    assert "camera.fit_mode=stretch" in summary
    assert "compositing.blend_space=srgb_legacy" in summary
    assert "compositing.color_correction.mode=off" in summary
    assert "compositing.color_correction.strength=0.5" in summary
    assert "background.video_path=<redacted>" in summary
    assert "api.allowed_origins=<redacted>" in summary
    assert "beach.mp4" not in summary
    assert "alice" not in summary
