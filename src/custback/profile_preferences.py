"""Secure concrete-value persistence for restart-bound system profiles."""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

from . import _platform as platform_fs
from .config import AppConfig
from .system_profiles import PROFILE_CATALOG, flatten_patch


PREFERENCES_SCHEMA = "custback.profile-preferences"
PREFERENCES_VERSION = 1
PREFERENCES_FILENAME = "profile-preferences.yaml"
MAX_PREFERENCES_BYTES = 64 * 1024
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_LOCK_TIMEOUT_S = 2.0
_MAX_REVISION = 2**63 - 1


class ProfilePreferencesError(ValueError):
    """A preference document or its storage boundary is unsafe."""


class PreferenceRevisionConflict(ProfilePreferencesError):
    def __init__(self, expected: int, current: int):
        self.expected = expected
        self.current = current
        super().__init__(
            f"profile preferences changed concurrently: expected revision "
            f"{expected}, current revision is {current}"
        )


class _NoAliasSafeLoader(yaml.SafeLoader):
    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            raise yaml.YAMLError("aliases are not allowed")
        return super().compose_node(parent, index)

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise yaml.YAMLError("expected a mapping")
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError as exc:
                raise yaml.YAMLError("mapping keys must be scalar") from exc
            if duplicate:
                raise yaml.YAMLError("duplicate mapping keys are not allowed")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class ProfilePreferences:
    revision: int
    values: dict[str, Any]


def default_preferences_path() -> Path:
    return platform_fs.config_dir() / PREFERENCES_FILENAME


def _validate_values(values: object, allowed_paths: frozenset[str]) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ProfilePreferencesError("profile preference values must be a mapping")
    if not values:
        return {}
    try:
        leaves = flatten_patch(values)
    except ValueError as exc:
        raise ProfilePreferencesError(str(exc)) from exc
    disallowed = sorted(set(leaves) - allowed_paths)
    if disallowed:
        raise ProfilePreferencesError(
            f"profile preferences contain a forbidden path: {disallowed[0]}"
        )
    try:
        # A stored document is always a complete sparse profile overlay. This
        # catches paired canvas fields and strict scalar types before startup.
        AppConfig().patched(values)
    except (TypeError, ValueError) as exc:
        raise ProfilePreferencesError("profile preference values are invalid") from exc
    return yaml.safe_load(yaml.safe_dump(values, sort_keys=True)) or {}


def _validate_document(
    raw: object, allowed_paths: frozenset[str]
) -> ProfilePreferences:
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "version",
        "revision",
        "values",
    }:
        raise ProfilePreferencesError(
            "profile preference document does not match schema"
        )
    if raw["schema"] != PREFERENCES_SCHEMA or raw["version"] != PREFERENCES_VERSION:
        raise ProfilePreferencesError(
            "profile preference schema/version is unsupported"
        )
    revision = raw["revision"]
    if type(revision) is not int or revision < 1 or revision > _MAX_REVISION:
        raise ProfilePreferencesError("profile preference revision is invalid")
    return ProfilePreferences(revision, _validate_values(raw["values"], allowed_paths))


def _reject_reparse_ancestors(path: Path) -> None:
    """Reject every existing symlink/reparse component without resolving it."""

    current = path
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(metadata.st_mode) or platform_fs.is_reparse(current):
                raise ProfilePreferencesError(
                    "profile preference storage contains a reparse point"
                )
        parent = current.parent
        if parent == current:
            return
        current = parent


def _secure_directory(path: Path, *, create: bool) -> None:
    _reject_reparse_ancestors(path)
    try:
        path.lstat()
    except FileNotFoundError:
        existed = False
    else:
        existed = True
    if create:
        path.mkdir(parents=True, mode=_DIRECTORY_MODE, exist_ok=True)
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise ProfilePreferencesError(
            "profile preference directory does not exist"
        ) from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ProfilePreferencesError(
            "profile preference directory is not a safe directory"
        )
    if not platform_fs.stat_owner_matches(before):
        raise ProfilePreferencesError("profile preference directory has another owner")
    if not existed:
        # A hostile umask can remove owner permissions from a directory just
        # created by this process. Repair only that new inode; an existing
        # insecure directory is configuration corruption and must fail closed.
        platform_fs.chmod_private(path, _DIRECTORY_MODE)
    descriptor = platform_fs.open_nofollow(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        directory=True,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ProfilePreferencesError(
                "profile preference directory changed while opening"
            )
        if not platform_fs.owner_matches(descriptor):
            raise ProfilePreferencesError(
                "profile preference directory has another owner"
            )
        if existed and not platform_fs.is_private_to_owner(descriptor):
            raise ProfilePreferencesError(
                "profile preference directory is not owner-only"
            )
        if not existed:
            platform_fs.set_private_mode(descriptor, _DIRECTORY_MODE)
            if not platform_fs.is_private_to_owner(descriptor):
                raise ProfilePreferencesError(
                    "profile preference directory could not be made owner-only"
                )
    finally:
        os.close(descriptor)


def _open_private_regular(path: Path, flags: int) -> int:
    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ProfilePreferencesError("profile preference path is not a regular file")
    if not platform_fs.stat_owner_matches(before):
        raise ProfilePreferencesError("profile preference file has another owner")
    descriptor = platform_fs.open_nofollow(path, flags | getattr(os, "O_CLOEXEC", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ProfilePreferencesError(
                "profile preference file changed while opening"
            )
        if not platform_fs.owner_matches(descriptor):
            raise ProfilePreferencesError("profile preference file has another owner")
        if not platform_fs.is_private_to_owner(descriptor):
            raise ProfilePreferencesError("profile preference file is not owner-only")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class ProfilePreferencesStore:
    """Cross-process CAS store for sparse concrete profile values."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        allowed_paths: frozenset[str] | None = None,
    ) -> None:
        selected = default_preferences_path() if path is None else Path(path)
        self.path = Path(os.path.abspath(os.fspath(selected)))
        self.allowed_paths = (
            PROFILE_CATALOG.manageable_paths if allowed_paths is None else allowed_paths
        )

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    def read(self) -> ProfilePreferences:
        try:
            return self._read_unwrapped()
        except ProfilePreferencesError:
            raise
        except OSError as exc:
            raise ProfilePreferencesError(
                "profile preference storage cannot be read securely"
            ) from exc

    def _read_unwrapped(self) -> ProfilePreferences:
        try:
            self.path.lstat()
        except FileNotFoundError:
            _reject_reparse_ancestors(self.path.parent)
            try:
                parent = self.path.parent.lstat()
            except FileNotFoundError:
                return ProfilePreferences(0, {})
            if not stat.S_ISDIR(parent.st_mode):
                raise ProfilePreferencesError(
                    "profile preference parent is not a directory"
                )
            return ProfilePreferences(0, {})
        _secure_directory(self.path.parent, create=False)
        descriptor = _open_private_regular(
            self.path,
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb") as source:
            encoded = source.read(MAX_PREFERENCES_BYTES + 1)
        if len(encoded) > MAX_PREFERENCES_BYTES:
            raise ProfilePreferencesError(
                "profile preference document exceeds its size limit"
            )
        try:
            raw = yaml.load(encoded.decode("utf-8"), Loader=_NoAliasSafeLoader)
        except (UnicodeDecodeError, yaml.YAMLError, RecursionError) as exc:
            raise ProfilePreferencesError(
                "profile preference document is malformed"
            ) from exc
        return _validate_document(raw, self.allowed_paths)

    @contextlib.contextmanager
    def _locked(self):
        _secure_directory(self.path.parent, create=True)
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = platform_fs.open_nofollow(
                self.lock_path,
                flags | os.O_CREAT | os.O_EXCL,
                _FILE_MODE,
            )
            created = True
        except FileExistsError:
            descriptor = _open_private_regular(self.lock_path, flags)
            created = False
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or not platform_fs.owner_matches(
                descriptor
            ):
                raise ProfilePreferencesError("profile preference lock is unsafe")
            if created:
                platform_fs.set_private_mode(descriptor, _FILE_MODE)
                if not platform_fs.is_private_to_owner(descriptor):
                    raise ProfilePreferencesError(
                        "profile preference lock could not be made owner-only"
                    )
            deadline = time.monotonic() + _LOCK_TIMEOUT_S
            while True:
                try:
                    platform_fs.lock_exclusive(descriptor)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ProfilePreferencesError(
                            "profile preference store is busy"
                        ) from None
                    time.sleep(0.02)
            yield
        finally:
            with contextlib.suppress(OSError):
                platform_fs.unlock(descriptor)
            os.close(descriptor)

    def _write_locked(self, preferences: ProfilePreferences) -> None:
        document = {
            "schema": PREFERENCES_SCHEMA,
            "version": PREFERENCES_VERSION,
            "revision": preferences.revision,
            "values": preferences.values,
        }
        encoded = yaml.safe_dump(
            document,
            sort_keys=True,
            allow_unicode=False,
        ).encode("ascii")
        if len(encoded) > MAX_PREFERENCES_BYTES:
            raise ProfilePreferencesError(
                "profile preference document exceeds its size limit"
            )
        temporary = self.path.with_name(
            f".{self.path.name}.{secrets.token_hex(16)}.tmp"
        )
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            descriptor = platform_fs.open_nofollow(temporary, flags, _FILE_MODE)
            platform_fs.set_private_mode(descriptor, _FILE_MODE)
            with os.fdopen(descriptor, "wb") as destination:
                descriptor = None
                destination.write(encoded)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.path)
            platform_fs.fsync_dir(self.path.parent)
        finally:
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    def update(
        self,
        expected_revision: int,
        mutate: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> ProfilePreferences:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ProfilePreferencesError("expected preference revision is invalid")
        try:
            with self._locked():
                current = self.read()
                if current.revision != expected_revision:
                    raise PreferenceRevisionConflict(
                        expected_revision, current.revision
                    )
                if current.revision == _MAX_REVISION:
                    raise ProfilePreferencesError(
                        "profile preference revision is exhausted"
                    )
                candidate = mutate(yaml.safe_load(yaml.safe_dump(current.values)) or {})
                if candidate is None:
                    candidate = current.values
                values = _validate_values(candidate, self.allowed_paths)
                updated = ProfilePreferences(current.revision + 1, values)
                self._write_locked(updated)
                return updated
        except ProfilePreferencesError:
            raise
        except OSError as exc:
            raise ProfilePreferencesError(
                "profile preference storage cannot be updated securely"
            ) from exc


def apply_preferences(base: AppConfig, preferences: ProfilePreferences) -> AppConfig:
    """Apply a validated concrete overlay without mutating its source config."""

    try:
        return (
            base.patched(preferences.values)
            if preferences.values
            else AppConfig.from_dict(base.to_dict())
        )
    except (TypeError, ValueError) as exc:
        raise ProfilePreferencesError(
            "profile preferences conflict with the operator configuration"
        ) from exc


__all__ = [
    "MAX_PREFERENCES_BYTES",
    "PREFERENCES_FILENAME",
    "PREFERENCES_SCHEMA",
    "PREFERENCES_VERSION",
    "PreferenceRevisionConflict",
    "ProfilePreferences",
    "ProfilePreferencesError",
    "ProfilePreferencesStore",
    "apply_preferences",
    "default_preferences_path",
]
