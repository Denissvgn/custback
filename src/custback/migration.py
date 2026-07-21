"""Explicit, crash-recoverable migration for legacy custback installations.

The migration entry point intentionally operates before the normal application
parser.  Legacy configuration must be inspected as data: loading it through the
current strict model first would reject the very values for which an operator
needs a safe migration report.

Configuration replacement follows a small write-ahead protocol.  The original
bytes are retained in a private backup, a private journal records both content
digests, and the candidate is published with a same-directory ``os.replace``.
Digest checks, rather than a journal phase alone, make retries converge after a
process is interrupted at any durable boundary.
"""

from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

from . import _platform as platform_fs
from .config import AppConfig, format_config_error

MAX_CONFIG_BYTES = 1024 * 1024
MAX_JOURNAL_BYTES = 16 * 1024
JOURNAL_SCHEMA = 1
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
EXIT_OPERATOR_ACTION_REQUIRED = 4

_TARGET_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\\\/]")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

DURABLE_BOUNDARIES = (
    "backup-durable",
    "journal-prepared",
    "candidate-durable",
    "journal-candidate-durable",
    "config-replaced",
    "journal-committed",
    "cleanup-durable",
)

BoundaryHook = Callable[[str], None]


class MigrationError(ValueError):
    """A migration could not proceed without risking user data."""


class StorageMigrationError(MigrationError):
    """A managed store contains an entry that repair must not mutate."""

    def __init__(self, message: str, issues: Iterable["StorageIssue"] = ()):
        super().__init__(message)
        self.issues = tuple(issues)


class MigrationStatus(str, Enum):
    MIGRATED = "migrated"
    ALREADY_CURRENT = "already_current"
    OPERATOR_ACTION_REQUIRED = "operator_action_required"


@dataclass(frozen=True)
class MigrationArtifacts:
    """Private files associated with one configuration migration."""

    backup: Path
    candidate: Path
    journal: Path


@dataclass(frozen=True)
class ConfigMigrationResult:
    status: MigrationStatus
    config_path: Path
    backup_path: Path | None
    detail: str
    recovered: bool = False

    @property
    def changed(self) -> bool:
        return self.status is MigrationStatus.MIGRATED


@dataclass(frozen=True)
class StorageIssue:
    path: Path
    reason: str  # mode | symlink | owner | type | unreadable | changed
    expected_mode: int | None
    actual_mode: int | None
    device: int | None = None
    inode: int | None = None


@dataclass(frozen=True)
class StorageAudit:
    roots: tuple[Path, ...]
    issues: tuple[StorageIssue, ...]

    @property
    def clean(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    data: bytes
    device: int
    inode: int
    mode: int
    uid: int


@dataclass(frozen=True)
class _ConfigPlan:
    status: MigrationStatus
    detail: str
    migrated: bytes | None = None


class _MigrationLoader(yaml.SafeLoader):
    """Safe loader that rejects aliases and ambiguous duplicate keys."""

    def compose_node(self, parent, index):  # type: ignore[no-untyped-def]
        if self.check_event(AliasEvent):
            raise ConstructorError(None, None, "YAML aliases are not allowed", None)
        return super().compose_node(parent, index)

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, found {node.id}",
                node.start_mark,
            )
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _lexical_path(path: str | os.PathLike[str]) -> Path:
    expanded = os.path.expanduser(os.fspath(path))
    return Path(os.path.abspath(expanded))


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _checkpoint(boundary: str, hook: BoundaryHook | None) -> None:
    if hook is not None:
        hook(boundary)


def migration_artifacts(
    config_path: str | os.PathLike[str],
) -> MigrationArtifacts:
    path = _lexical_path(config_path)
    prefix = f".{path.name}.custback-migration"
    return MigrationArtifacts(
        backup=path.with_name(f"{prefix}.backup"),
        candidate=path.with_name(f"{prefix}.candidate"),
        journal=path.with_name(f"{prefix}.json"),
    )


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _read_regular_nofollow(
    path: Path,
    *,
    maximum: int,
    required_mode: int | None = None,
) -> _FileSnapshot:
    before = _lstat(path)
    if before is None:
        raise MigrationError(f"file does not exist: {path}")
    if stat.S_ISLNK(before.st_mode):
        raise MigrationError(f"refusing to follow symbolic link: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise MigrationError(f"expected a regular file: {path}")
    if not platform_fs.stat_owner_matches(before):
        raise MigrationError(f"file is not owned by the current user: {path}")
    actual_mode = stat.S_IMODE(before.st_mode)
    if required_mode is not None and actual_mode != required_mode:
        raise MigrationError(
            f"private migration file must have mode {required_mode:04o}: {path}"
        )
    if before.st_size > maximum:
        raise MigrationError(f"file exceeds the migration size limit: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = platform_fs.open_nofollow(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise MigrationError(f"refusing to follow symbolic link: {path}") from exc
        raise
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise MigrationError(f"file changed while opening: {path}")
        # Authoritative owner check on the bound descriptor (SID on Windows).
        if not platform_fs.owner_matches(descriptor):
            raise MigrationError(f"file is not owned by the current user: {path}")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > maximum:
            raise MigrationError(f"file exceeds the migration size limit: {path}")
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise MigrationError(f"file changed while reading: {path}")
        return _FileSnapshot(
            path=path,
            data=data,
            device=opened.st_dev,
            inode=opened.st_ino,
            mode=stat.S_IMODE(opened.st_mode),
            uid=opened.st_uid,
        )
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    try:
        platform_fs.fsync_dir(directory)
    except NotADirectoryError as exc:
        raise MigrationError(
            f"migration parent is not a directory: {directory}"
        ) from exc


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while persisting migration state")
        view = view[written:]


def _atomic_private_write(path: Path, payload: bytes, *, replace: bool) -> None:
    """Publish private bytes without ever opening an existing final path."""

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    created = False
    try:
        descriptor = platform_fs.open_nofollow(temporary, flags, PRIVATE_FILE_MODE)
        created = True
        platform_fs.set_private_mode(descriptor, PRIVATE_FILE_MODE)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        if replace:
            existing = _lstat(path)
            if existing is not None:
                _read_regular_nofollow(
                    path,
                    maximum=max(MAX_JOURNAL_BYTES, len(payload)),
                    required_mode=PRIVATE_FILE_MODE,
                )
            os.replace(temporary, path)
            created = False
        else:
            # A hard-link publication supplies rename-like durability without
            # replacing a pre-existing backup or journal of unknown origin.
            platform_fs.hardlink(temporary, path)
            os.unlink(temporary)
            created = False
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _ensure_private_artifact(path: Path, payload: bytes, *, maximum: int) -> bool:
    existing = _lstat(path)
    if existing is None:
        try:
            _atomic_private_write(path, payload, replace=False)
            return True
        except FileExistsError:
            # A concurrent invocation may have published the identical state.
            pass
    snapshot = _read_regular_nofollow(
        path, maximum=maximum, required_mode=PRIVATE_FILE_MODE
    )
    if snapshot.data != payload:
        raise MigrationError(f"private migration artifact does not match: {path}")
    return False


def _parse_yaml(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        raw = yaml.load(text, Loader=_MigrationLoader)
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise MigrationError(f"legacy configuration is not safe YAML: {path}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise MigrationError("legacy configuration root must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise MigrationError("legacy configuration keys must be strings")
    return raw


def _safe_legacy_source(value: Any) -> tuple[bool, int | str | None]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value >= 0, value if value >= 0 else None)
    if not isinstance(value, str) or not value:
        return False, None
    if (
        value.strip() != value
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        return False, None
    if value.isdecimal():
        return True, int(value)
    windows_drive = bool(_WINDOWS_DRIVE_RE.match(value))
    if (
        value.startswith(("//", "\\\\"))
        or "://" in value
        or (_URI_SCHEME_RE.match(value) and not windows_drive)
    ):
        return False, None
    if os.path.isabs(value) or windows_drive:
        return True, value
    return False, None


def _plan_config_migration(raw: dict[str, Any], target_id: str) -> _ConfigPlan:
    if not _TARGET_ID_RE.fullmatch(target_id):
        raise MigrationError("target ID must match [a-z][a-z0-9_-]{0,63}")
    background = raw.get("background", {})
    if not isinstance(background, dict):
        raise MigrationError("legacy background configuration must be a mapping")
    has_legacy = "camera_device" in background and background.get("camera_device") != ""
    if not has_legacy:
        try:
            AppConfig.from_dict(copy.deepcopy(raw))
        except (TypeError, ValueError, RecursionError) as exc:
            raise MigrationError(
                f"configuration is not valid for this release: {format_config_error(exc)}"
            ) from exc
        return _ConfigPlan(
            MigrationStatus.ALREADY_CURRENT,
            "configuration has no legacy live-backdrop source",
        )

    safe, source = _safe_legacy_source(background.get("camera_device"))
    if not safe:
        return _ConfigPlan(
            MigrationStatus.OPERATOR_ACTION_REQUIRED,
            "legacy camera_device is not a numeric index or absolute local path",
        )

    selected = background.get("camera_target", "")
    if selected not in ("", target_id):
        return _ConfigPlan(
            MigrationStatus.OPERATOR_ACTION_REQUIRED,
            "legacy source conflicts with an existing immutable target selection",
        )
    targets = raw.get("backdrop_targets", {})
    if not isinstance(targets, dict):
        raise MigrationError("backdrop_targets must be a mapping")
    existing = targets.get(target_id)
    if existing is not None and existing != {"source": source}:
        return _ConfigPlan(
            MigrationStatus.OPERATOR_ACTION_REQUIRED,
            "the requested immutable target ID is already assigned differently",
        )

    migrated = copy.deepcopy(raw)
    migrated_background = migrated.setdefault("background", {})
    migrated_background.pop("camera_device", None)
    migrated_background["camera_target"] = target_id
    migrated_targets = copy.deepcopy(targets)
    migrated_targets[target_id] = {"source": source}
    if "backdrop_targets" in migrated:
        migrated["backdrop_targets"] = migrated_targets
    else:
        ordered: dict[str, Any] = {}
        for key, value in migrated.items():
            ordered[key] = value
            if key == "background":
                ordered["backdrop_targets"] = migrated_targets
        migrated = ordered

    try:
        AppConfig.from_dict(copy.deepcopy(migrated))
        encoded = yaml.safe_dump(
            migrated,
            sort_keys=False,
            allow_unicode=True,
        ).encode("utf-8")
    except (TypeError, ValueError, yaml.YAMLError, RecursionError) as exc:
        raise MigrationError(
            f"migrated configuration is invalid: {format_config_error(exc)}"
        ) from exc
    return _ConfigPlan(
        MigrationStatus.MIGRATED,
        "legacy live-backdrop source moved to immutable operator authority",
        encoded,
    )


def _journal_payload(journal: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(journal), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _load_journal(
    path: Path, config: Path, artifacts: MigrationArtifacts
) -> dict[str, Any]:
    snapshot = _read_regular_nofollow(
        path,
        maximum=MAX_JOURNAL_BYTES,
        required_mode=PRIVATE_FILE_MODE,
    )
    try:
        journal = json.loads(snapshot.data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"migration journal is invalid: {path}") from exc
    required = {
        "schema",
        "operation",
        "config_name",
        "target_id",
        "source_digest",
        "migrated_digest",
        "backup_name",
        "candidate_name",
        "phase",
    }
    if not isinstance(journal, dict) or set(journal) != required:
        raise MigrationError(f"migration journal has an invalid shape: {path}")
    if (
        journal["schema"] != JOURNAL_SCHEMA
        or journal["operation"] != "legacy-backdrop-config"
        or journal["config_name"] != config.name
        or journal["backup_name"] != artifacts.backup.name
        or journal["candidate_name"] != artifacts.candidate.name
        or not isinstance(journal["target_id"], str)
        or not _TARGET_ID_RE.fullmatch(journal["target_id"])
        or journal["phase"] not in {"prepared", "candidate", "committed"}
        or not isinstance(journal["source_digest"], str)
        or not _DIGEST_RE.fullmatch(journal["source_digest"])
        or not isinstance(journal["migrated_digest"], str)
        or not _DIGEST_RE.fullmatch(journal["migrated_digest"])
    ):
        raise MigrationError(f"migration journal does not match {config}")
    return journal


def _write_journal(path: Path, journal: dict[str, Any], *, replace: bool) -> None:
    payload = _journal_payload(journal)
    if len(payload) > MAX_JOURNAL_BYTES:
        raise MigrationError("migration journal exceeds its size limit")
    _atomic_private_write(path, payload, replace=replace)


def _remove_private(path: Path, *, digest: str | None = None) -> bool:
    if _lstat(path) is None:
        return False
    snapshot = _read_regular_nofollow(
        path,
        maximum=max(MAX_CONFIG_BYTES, MAX_JOURNAL_BYTES),
        required_mode=PRIVATE_FILE_MODE,
    )
    if digest is not None and _digest(snapshot.data) != digest:
        raise MigrationError(f"refusing to remove changed migration artifact: {path}")
    path.unlink()
    return True


def _finish_migration(
    config: Path,
    artifacts: MigrationArtifacts,
    journal: dict[str, Any],
    *,
    hook: BoundaryHook | None,
    recovered: bool,
) -> ConfigMigrationResult:
    if journal["phase"] != "committed":
        journal = {**journal, "phase": "committed"}
        _write_journal(artifacts.journal, journal, replace=True)
        _checkpoint("journal-committed", hook)
    removed = _remove_private(artifacts.candidate, digest=journal["migrated_digest"])
    removed = _remove_private(artifacts.journal) or removed
    if removed:
        _fsync_directory(config.parent)
    _checkpoint("cleanup-durable", hook)
    return ConfigMigrationResult(
        MigrationStatus.MIGRATED,
        config,
        artifacts.backup,
        "configuration migration committed",
        recovered=recovered,
    )


def _resume_config_migration(
    config: Path,
    artifacts: MigrationArtifacts,
    journal: dict[str, Any],
    *,
    target_id: str,
    hook: BoundaryHook | None,
    recovered: bool,
) -> ConfigMigrationResult:
    if journal["target_id"] != target_id:
        raise MigrationError(
            "a pending migration requires the original target ID "
            f"{journal['target_id']!r}"
        )
    backup = _read_regular_nofollow(
        artifacts.backup,
        maximum=MAX_CONFIG_BYTES,
        required_mode=PRIVATE_FILE_MODE,
    )
    if _digest(backup.data) != journal["source_digest"]:
        raise MigrationError("migration backup does not match its journal")
    plan = _plan_config_migration(_parse_yaml(backup.data, artifacts.backup), target_id)
    if plan.status is not MigrationStatus.MIGRATED or plan.migrated is None:
        raise MigrationError(
            "migration backup no longer produces the journaled candidate"
        )
    if _digest(plan.migrated) != journal["migrated_digest"]:
        raise MigrationError("migration candidate digest does not match its journal")

    current = _read_regular_nofollow(config, maximum=MAX_CONFIG_BYTES)
    current_digest = _digest(current.data)
    if current_digest == journal["migrated_digest"]:
        current_plan = _plan_config_migration(
            _parse_yaml(current.data, config), target_id
        )
        if current_plan.status is not MigrationStatus.ALREADY_CURRENT:
            raise MigrationError(
                "published migration is not a valid current configuration"
            )
        return _finish_migration(
            config,
            artifacts,
            journal,
            hook=hook,
            recovered=recovered,
        )
    if current_digest != journal["source_digest"]:
        raise MigrationError("configuration changed while a migration was pending")
    if journal["phase"] == "committed":
        raise MigrationError(
            "committed migration configuration was replaced externally"
        )

    candidate_created = _ensure_private_artifact(
        artifacts.candidate,
        plan.migrated,
        maximum=MAX_CONFIG_BYTES,
    )
    if candidate_created:
        _checkpoint("candidate-durable", hook)
    if journal["phase"] == "prepared":
        journal = {**journal, "phase": "candidate"}
        _write_journal(artifacts.journal, journal, replace=True)
        _checkpoint("journal-candidate-durable", hook)

    # Re-read immediately before publication so a concurrent editor cannot be
    # overwritten using authority obtained from an earlier snapshot.
    before_replace = _read_regular_nofollow(config, maximum=MAX_CONFIG_BYTES)
    if _digest(before_replace.data) != journal["source_digest"]:
        raise MigrationError("configuration changed before atomic replacement")
    candidate = _read_regular_nofollow(
        artifacts.candidate,
        maximum=MAX_CONFIG_BYTES,
        required_mode=PRIVATE_FILE_MODE,
    )
    if _digest(candidate.data) != journal["migrated_digest"]:
        raise MigrationError("migration candidate changed before publication")
    os.replace(artifacts.candidate, config)
    _fsync_directory(config.parent)
    _checkpoint("config-replaced", hook)
    return _finish_migration(
        config,
        artifacts,
        journal,
        hook=hook,
        recovered=recovered,
    )


def migrate_config(
    config_path: str | os.PathLike[str],
    target_id: str,
    *,
    boundary_hook: BoundaryHook | None = None,
) -> ConfigMigrationResult:
    """Migrate one legacy config, or safely report required operator action."""

    if not _TARGET_ID_RE.fullmatch(target_id):
        raise MigrationError("target ID must match [a-z][a-z0-9_-]{0,63}")
    config = _lexical_path(config_path)
    artifacts = migration_artifacts(config)
    if _lstat(artifacts.journal) is not None:
        journal = _load_journal(artifacts.journal, config, artifacts)
        return _resume_config_migration(
            config,
            artifacts,
            journal,
            target_id=target_id,
            hook=boundary_hook,
            recovered=True,
        )

    original = _read_regular_nofollow(config, maximum=MAX_CONFIG_BYTES)
    raw = _parse_yaml(original.data, config)
    plan = _plan_config_migration(raw, target_id)
    if plan.status is MigrationStatus.OPERATOR_ACTION_REQUIRED:
        return ConfigMigrationResult(
            plan.status,
            config,
            None,
            plan.detail,
        )
    if plan.status is MigrationStatus.ALREADY_CURRENT:
        return ConfigMigrationResult(
            plan.status,
            config,
            artifacts.backup if _lstat(artifacts.backup) is not None else None,
            plan.detail,
        )
    assert plan.migrated is not None

    _ensure_private_artifact(
        artifacts.backup,
        original.data,
        maximum=MAX_CONFIG_BYTES,
    )
    _checkpoint("backup-durable", boundary_hook)
    journal = {
        "schema": JOURNAL_SCHEMA,
        "operation": "legacy-backdrop-config",
        "config_name": config.name,
        "target_id": target_id,
        "source_digest": _digest(original.data),
        "migrated_digest": _digest(plan.migrated),
        "backup_name": artifacts.backup.name,
        "candidate_name": artifacts.candidate.name,
        "phase": "prepared",
    }
    try:
        _write_journal(artifacts.journal, journal, replace=False)
    except FileExistsError:
        existing = _load_journal(artifacts.journal, config, artifacts)
        return _resume_config_migration(
            config,
            artifacts,
            existing,
            target_id=target_id,
            hook=boundary_hook,
            recovered=True,
        )
    _checkpoint("journal-prepared", boundary_hook)
    return _resume_config_migration(
        config,
        artifacts,
        journal,
        target_id=target_id,
        hook=boundary_hook,
        recovered=False,
    )


def _normalize_store_roots(
    roots: Iterable[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    normalized = tuple(dict.fromkeys(_lexical_path(root) for root in roots))
    for index, root in enumerate(normalized):
        for other in normalized[index + 1 :]:
            if root in other.parents or other in root.parents:
                raise StorageMigrationError(
                    f"managed storage roots must not overlap: {root} and {other}"
                )
    return normalized


def _storage_issue(
    path: Path,
    reason: str,
    metadata: os.stat_result | None,
    expected: int | None,
) -> StorageIssue:
    return StorageIssue(
        path=path,
        reason=reason,
        expected_mode=expected,
        actual_mode=stat.S_IMODE(metadata.st_mode) if metadata is not None else None,
        device=metadata.st_dev if metadata is not None else None,
        inode=metadata.st_ino if metadata is not None else None,
    )


def _scan_storage_path(path: Path, issues: list[StorageIssue]) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        issues.append(_storage_issue(path, "unreadable", None, None))
        return
    if stat.S_ISLNK(metadata.st_mode):
        issues.append(_storage_issue(path, "symlink", metadata, None))
        return
    is_directory = stat.S_ISDIR(metadata.st_mode)
    is_file = stat.S_ISREG(metadata.st_mode)
    expected = (
        PRIVATE_DIRECTORY_MODE
        if is_directory
        else PRIVATE_FILE_MODE
        if is_file
        else None
    )
    if not (is_directory or is_file):
        issues.append(_storage_issue(path, "type", metadata, expected))
        return
    if not platform_fs.stat_owner_matches(metadata):
        issues.append(_storage_issue(path, "owner", metadata, expected))
        return
    if stat.S_IMODE(metadata.st_mode) != expected:
        issues.append(_storage_issue(path, "mode", metadata, expected))
    if not is_directory:
        return

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = platform_fs.open_nofollow(path, flags, directory=True)
    except OSError:
        issues.append(_storage_issue(path, "unreadable", metadata, expected))
        return
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            issues.append(_storage_issue(path, "changed", metadata, expected))
            return
        try:
            names = sorted(platform_fs.listdir_secure(descriptor, path))
        except OSError:
            issues.append(_storage_issue(path, "unreadable", metadata, expected))
            return
    finally:
        os.close(descriptor)
    for name in names:
        if name in {".", ".."} or os.sep in name:
            issues.append(_storage_issue(path, "changed", metadata, expected))
            continue
        _scan_storage_path(path / name, issues)


def audit_storage(
    roots: Iterable[str | os.PathLike[str]],
) -> StorageAudit:
    """Audit managed payload trees without following or changing any path.

    Ownership ledgers live beside each payload root, not below it.  Restricting
    the walk to the supplied roots guarantees their reservation records and
    lease files are neither opened nor chmodded by this operation.
    """

    normalized = _normalize_store_roots(roots)
    issues: list[StorageIssue] = []
    for root in normalized:
        _scan_storage_path(root, issues)
    return StorageAudit(normalized, tuple(issues))


def _repair_mode(issue: StorageIssue) -> None:
    if issue.expected_mode is None or issue.device is None or issue.inode is None:
        raise StorageMigrationError(f"cannot repair unsafe storage path: {issue.path}")
    before = issue.path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not (stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode))
        or (before.st_dev, before.st_ino) != (issue.device, issue.inode)
        or not platform_fs.stat_owner_matches(before)
    ):
        raise StorageMigrationError(f"storage path changed before repair: {issue.path}")
    platform_fs.chmod_private(issue.path, issue.expected_mode)
    after = issue.path.lstat()
    if stat.S_ISLNK(after.st_mode) or (after.st_dev, after.st_ino) != (
        issue.device,
        issue.inode,
    ):
        raise StorageMigrationError(f"storage path changed during repair: {issue.path}")

    is_directory = stat.S_ISDIR(after.st_mode)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = platform_fs.open_nofollow(issue.path, flags, directory=is_directory)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (issue.device, issue.inode):
            raise StorageMigrationError(
                f"storage path changed while binding repair: {issue.path}"
            )
        # Authoritative owner check on the bound descriptor (SID on Windows).
        if not platform_fs.owner_matches(descriptor):
            raise StorageMigrationError(
                f"storage path is not owned by the current user: {issue.path}"
            )
        platform_fs.set_private_mode(descriptor, issue.expected_mode)
        if is_directory:
            platform_fs.fsync_dir(issue.path)
        else:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def repair_storage(
    roots: Iterable[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    """Apply exact private modes after no-follow safety audits.

    An owner-only execute directory cannot be enumerated until its own mode is
    repaired.  Such a directory appears as both ``mode`` and ``unreadable``;
    repair that known inode, then repeat the complete audit before touching any
    newly visible child.  Other unreadable paths remain hard failures.
    """

    normalized = _normalize_store_roots(roots)
    repaired: list[Path] = []
    repaired_set: set[Path] = set()
    while True:
        audit = audit_storage(normalized)
        mode_issues = {
            issue.path: issue for issue in audit.issues if issue.reason == "mode"
        }
        repairable_unreadable = {
            issue.path
            for issue in audit.issues
            if issue.reason == "unreadable"
            and issue.path in mode_issues
            and mode_issues[issue.path].expected_mode == PRIVATE_DIRECTORY_MODE
        }
        unsafe = tuple(
            issue
            for issue in audit.issues
            if issue.reason != "mode"
            and not (
                issue.reason == "unreadable" and issue.path in repairable_unreadable
            )
        )
        if unsafe:
            first = unsafe[0]
            raise StorageMigrationError(
                f"refusing unsafe managed storage ({first.reason}): {first.path}",
                unsafe,
            )
        if not mode_issues:
            if audit.issues:
                first = audit.issues[0]
                raise StorageMigrationError(
                    f"managed storage is not private after repair: {first.path}",
                    audit.issues,
                )
            return tuple(repaired)
        for issue in sorted(
            mode_issues.values(),
            key=lambda item: len(item.path.parts),
            reverse=True,
        ):
            _repair_mode(issue)
            if issue.path not in repaired_set:
                repaired.append(issue.path)
                repaired_set.add(issue.path)


def build_parser(*, prog: str = "custback migrate") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="migrate legacy configuration and audit managed stores",
    )
    parser.add_argument("--config", metavar="PATH", help="legacy core YAML to migrate")
    parser.add_argument(
        "--target-id",
        help="operator-owned backdrop target ID for a legacy camera_device",
    )
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--audit-storage",
        action="store_true",
        help="audit core and avatar stores without changing them",
    )
    storage.add_argument(
        "--repair-storage",
        action="store_true",
        help="repair safe core and avatar store modes after a full audit",
    )
    parser.add_argument("--core-store", metavar="PATH", help="core background store")
    parser.add_argument("--avatar-rigs-store", metavar="PATH", help="avatar rig store")
    parser.add_argument(
        "--avatar-backgrounds-store",
        metavar="PATH",
        help="avatar scene-background store",
    )
    return parser


def _default_store_roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    from .avatar.config import StorageConfig
    from .backgrounds import DEFAULT_BACKGROUNDS_DIR

    avatar = StorageConfig()
    return (
        _lexical_path(args.core_store or DEFAULT_BACKGROUNDS_DIR),
        _lexical_path(args.avatar_rigs_store or avatar.rigs_dir),
        _lexical_path(args.avatar_backgrounds_store or avatar.backgrounds_dir),
    )


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "custback migrate",
) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.config is None and not (args.audit_storage or args.repair_storage):
        parser.error("provide --config and/or a storage audit/repair action")
    if args.config is not None and args.target_id is None:
        parser.error("--target-id is required with --config")
    if args.config is None and args.target_id is not None:
        parser.error("--target-id requires --config")
    if any(
        (args.core_store, args.avatar_rigs_store, args.avatar_backgrounds_store)
    ) and not (args.audit_storage or args.repair_storage):
        parser.error("store overrides require --audit-storage or --repair-storage")

    try:
        if args.config is not None:
            result = migrate_config(args.config, args.target_id)
            if result.status is MigrationStatus.OPERATOR_ACTION_REQUIRED:
                print(
                    "custback migrate: operator action required; " + result.detail,
                    file=sys.stderr,
                )
                return EXIT_OPERATOR_ACTION_REQUIRED
            if result.status is MigrationStatus.MIGRATED:
                qualifier = "recovered and migrated" if result.recovered else "migrated"
                print(
                    f"custback migrate: {qualifier} {result.config_path}; "
                    f"backup retained at {result.backup_path}"
                )
            else:
                print(f"custback migrate: {result.config_path} is already current")

        if args.audit_storage or args.repair_storage:
            roots = _default_store_roots(args)
            if args.repair_storage:
                repaired = repair_storage(roots)
                print(
                    "custback migrate: storage repair complete "
                    f"({len(repaired)} path(s) secured)"
                )
            else:
                audit = audit_storage(roots)
                if audit.issues:
                    for issue in audit.issues:
                        print(
                            f"custback migrate: {issue.reason}: {issue.path}",
                            file=sys.stderr,
                        )
                    return 2
                print("custback migrate: storage audit clean")
        return 0
    except (MigrationError, OSError) as exc:
        print(f"custback migrate: {format_config_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
