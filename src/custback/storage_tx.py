"""Durable ownership records for rename-based storage transactions.

The stores in :mod:`custback.api.server` and :mod:`custback.avatar.store`
publish files by renaming private staging inodes.  A pathname is not stable
ownership: after a successful rename followed by a failed chmod, rollback, or
unlink, cleanup must follow the inode to its new name.  This module keeps a
small private sidecar record for every owned inode and records both names
*before* a rename.  A crashed process can therefore reconcile the inode by
``(st_dev, st_ino)`` and retry cleanup without glob-deleting user data.

The ledger deliberately does not run cleanup itself.  Stores supply the
file/tree remover so quota and publication locks remain owned by the caller.
"""

from __future__ import annotations

import json
import os
import re
import stat
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import _platform as platform_fs


_RECORD_RE = re.compile(r"\.custback-owned-([0-9a-f]{32})\.json\Z")
_TEMP_RECORD_RE = re.compile(r"\.custback-owned-([0-9a-f]{32})-[0-9a-f]{32}\.tmp\Z")
_MAX_RECORD_BYTES = 16 * 1024
_MAX_RECOVERY_RECORDS = 4096


def sync_directory(path: Path) -> None:
    """Durably order namespace changes below a no-follow directory."""

    platform_fs.fsync_dir(Path(path))


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without ever replacing an unmarked destination."""

    platform_fs.rename_noreplace(Path(source), Path(destination))


def is_ownership_metadata(path: Path) -> bool:
    """Return whether ``path`` is internal transaction metadata."""

    return bool(_RECORD_RE.fullmatch(path.name) or _TEMP_RECORD_RE.fullmatch(path.name))


def _private_open(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = platform_fs.open_nofollow(path, flags, 0o600)
    try:
        platform_fs.set_private_mode(descriptor, 0o600)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _inode(path: Path, kind: str | None = None) -> tuple[int, int] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        return None
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        return None
    if kind == "tree" and not stat.S_ISDIR(metadata.st_mode):
        return None
    return metadata.st_dev, metadata.st_ino


@dataclass
class OwnedPath:
    """One store-owned file or directory and its quota reservation."""

    txid: str
    kind: str
    paths: tuple[Path, ...]
    device: int | None = None
    inode: int | None = None
    reserved_bytes: int = 0
    reserved_slots: int = 0
    state: str = "active"  # active | cleanup | committed
    cleanup_error: OSError | None = None

    @property
    def identity(self) -> tuple[int, int] | None:
        if self.device is None or self.inode is None:
            return None
        return self.device, self.inode


class OwnershipLedger:
    """Private, bounded, restart-readable ownership records below one root."""

    def __init__(self, root: Path, *, recovery_limit: int = _MAX_RECOVERY_RECORDS):
        self.root = Path(root)
        # Keep bookkeeping outside the managed payload directory.  Besides
        # keeping public directory scans clean, this prevents metadata files
        # from becoming part of user byte/file quotas.
        self.metadata_root = self.root.parent / (
            f".{self.root.name}.custback-ownership"
        )
        self.recovery_limit = recovery_limit
        self._lock = threading.RLock()
        self._records: dict[str, OwnedPath] = {}
        self._lease_descriptor: int | None = None
        try:
            if self.metadata_root.exists() or self.metadata_root.is_symlink():
                self._acquire_lease(create=False)
            self._release_lease_if_idle()
        except BaseException:
            self.close()
            raise

    @property
    def records(self) -> tuple[OwnedPath, ...]:
        with self._lock:
            return tuple(self._records.values())

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return sum(
                record.reserved_bytes
                for record in self._records.values()
                if record.state != "committed"
            )

    @property
    def reserved_slots(self) -> int:
        with self._lock:
            return sum(
                record.reserved_slots
                for record in self._records.values()
                if record.state != "committed"
            )

    def _record_path(self, txid: str) -> Path:
        return self.metadata_root / f".custback-owned-{txid}.json"

    def _lease_path(self) -> Path:
        return self.metadata_root / ".custback-owner.lock"

    def _relative_name(self, path: Path) -> str:
        candidate = Path(path)
        if candidate.parent != self.root or candidate.name in {"", ".", ".."}:
            raise ValueError("owned paths must be direct children of the store")
        return candidate.name

    def _payload(self, record: OwnedPath) -> dict[str, object]:
        return {
            "version": 1,
            "txid": record.txid,
            "kind": record.kind,
            "paths": [self._relative_name(path) for path in record.paths],
            "device": record.device,
            "inode": record.inode,
            "reserved_bytes": record.reserved_bytes,
            "reserved_slots": record.reserved_slots,
            "state": record.state,
        }

    def _sync_directory(self) -> None:
        platform_fs.fsync_dir(self.metadata_root)

    def _secure_metadata_root(self, *, create: bool) -> os.stat_result | None:
        created = False
        if create:
            try:
                self.metadata_root.mkdir(mode=0o700)
                created = True
            except FileExistsError:
                pass
        try:
            before = self.metadata_root.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise OSError("storage ownership metadata is not a directory")
        if not platform_fs.stat_owner_matches(before):
            raise PermissionError("storage ownership metadata has another owner")
        if created:
            platform_fs.chmod_private(self.metadata_root, 0o700)
            before = self.metadata_root.lstat()
        elif stat.S_IMODE(before.st_mode) != 0o700:
            raise PermissionError("storage ownership metadata does not have mode 0700")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        descriptor = platform_fs.open_nofollow(self.metadata_root, flags, directory=True)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise OSError("storage ownership metadata changed while opening")
            # Authoritative owner check on the bound descriptor (SID on Windows).
            if not platform_fs.owner_matches(descriptor):
                raise PermissionError("storage ownership metadata has another owner")
            platform_fs.set_private_mode(descriptor, 0o700)
            return opened
        finally:
            os.close(descriptor)

    def _acquire_lease(self, *, create: bool) -> None:
        if self._lease_descriptor is not None:
            return
        metadata = self._secure_metadata_root(create=create)
        if metadata is None:
            return
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        descriptor = platform_fs.open_nofollow(self._lease_path(), flags, 0o600)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or not platform_fs.owner_matches(
                descriptor
            ):
                raise OSError("unsafe storage ownership lease")
            platform_fs.set_private_mode(descriptor, 0o600)
            try:
                platform_fs.lock_exclusive(descriptor)
            except BlockingIOError as exc:
                raise OSError("storage ownership is held by another process") from exc
        except BaseException:
            os.close(descriptor)
            raise
        self._lease_descriptor = descriptor
        try:
            self._load()
        except BaseException:
            self.close()
            raise

    def _release_lease_if_idle(self) -> None:
        if self._records or self._lease_descriptor is None:
            return
        descriptor = self._lease_descriptor
        self._lease_descriptor = None
        os.close(descriptor)

    def close(self) -> None:
        """Release the process lease (normally held until records are terminal)."""

        with self._lock:
            if self._lease_descriptor is not None:
                descriptor = self._lease_descriptor
                self._lease_descriptor = None
                os.close(descriptor)

    def __del__(self):  # pragma: no cover - normal stores live for the process
        try:
            self.close()
        except Exception:
            pass

    def _persist(self, record: OwnedPath) -> None:
        """Atomically replace a record; the old candidates survive a crash."""

        self._acquire_lease(create=True)
        self._secure_metadata_root(create=False)
        encoded = json.dumps(
            self._payload(record), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ValueError("ownership record is too large")
        temporary = self.metadata_root / (
            f".custback-owned-{record.txid}-{uuid.uuid4().hex}.tmp"
        )
        descriptor: int | None = None
        try:
            descriptor = _private_open(temporary)
            with os.fdopen(descriptor, "wb") as destination:
                descriptor = None
                destination.write(encoded)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self._record_path(record.txid))
            self._sync_directory()
        except BaseException:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # The temporary contains metadata only.  It is recognized and
                # ignored by quota scans, and a later recovery pass removes it.
                pass
            raise

    def _load(self) -> None:
        root_metadata = self._secure_metadata_root(create=False)
        if root_metadata is None:
            return
        loaded = 0
        entries = sorted(self.metadata_root.iterdir())
        after_snapshot = self.metadata_root.lstat()
        if stat.S_ISLNK(after_snapshot.st_mode) or (
            after_snapshot.st_dev,
            after_snapshot.st_ino,
        ) != (root_metadata.st_dev, root_metadata.st_ino):
            raise OSError("storage ownership metadata changed during recovery")
        for path in entries:
            if _TEMP_RECORD_RE.fullmatch(path.name):
                # A temp record was never made authoritative.  It owns no data.
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            match = _RECORD_RE.fullmatch(path.name)
            if match is None:
                continue
            loaded += 1
            if loaded > self.recovery_limit:
                raise OSError("too many pending storage ownership records")
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > _MAX_RECORD_BYTES
            ):
                raise OSError(f"unsafe storage ownership record: {path}")
            if not platform_fs.stat_owner_matches(metadata):
                raise PermissionError(
                    f"storage ownership record has another owner: {path}"
                )
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise PermissionError(
                    f"storage ownership record does not have mode 0600: {path}"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            descriptor = platform_fs.open_nofollow(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (metadata.st_dev, metadata.st_ino)
                    or opened.st_size > _MAX_RECORD_BYTES
                ):
                    raise OSError(
                        f"storage ownership record changed while opening: {path}"
                    )
                # Authoritative owner check on the bound descriptor (SID on Windows).
                if not platform_fs.owner_matches(descriptor):
                    raise PermissionError(
                        f"storage ownership record has another owner: {path}"
                    )
                with os.fdopen(descriptor, "rb") as source:
                    descriptor = -1
                    raw = source.read(_MAX_RECORD_BYTES + 1)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            try:
                payload = json.loads(raw)
                txid = str(payload["txid"])
                names = tuple(str(name) for name in payload["paths"])
                kind = str(payload["kind"])
                state = str(payload["state"])
                device = payload.get("device")
                inode = payload.get("inode")
                reserved_bytes = int(payload["reserved_bytes"])
                reserved_slots = int(payload["reserved_slots"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise OSError(f"invalid storage ownership record: {path}") from exc
            if (
                payload.get("version") != 1
                or txid != match.group(1)
                or kind not in {"file", "tree"}
                or state not in {"active", "cleanup", "committed"}
                or not names
                or len(names) > 2
                or any(
                    not name
                    or Path(name).name != name
                    or is_ownership_metadata(self.root / name)
                    for name in names
                )
                or reserved_bytes < 0
                or reserved_slots < 0
                or (device is None) != (inode is None)
            ):
                raise OSError(f"invalid storage ownership record: {path}")
            # Any non-committed transaction from a dead process is cleanup,
            # regardless of the last in-memory phase it persisted.
            recovered_state = "committed" if state == "committed" else "cleanup"
            self._records[txid] = OwnedPath(
                txid=txid,
                kind=kind,
                paths=tuple(self.root / name for name in names),
                device=None if device is None else int(device),
                inode=None if inode is None else int(inode),
                reserved_bytes=reserved_bytes,
                reserved_slots=reserved_slots,
                state=recovered_state,
            )

    def begin(
        self,
        path: Path,
        *,
        kind: str,
        reserved_bytes: int = 0,
        reserved_slots: int = 0,
    ) -> OwnedPath:
        if kind not in {"file", "tree"}:
            raise ValueError("owned path kind must be file or tree")
        with self._lock:
            record = OwnedPath(
                txid=uuid.uuid4().hex,
                kind=kind,
                paths=(Path(path),),
                reserved_bytes=reserved_bytes,
                reserved_slots=reserved_slots,
            )
            self._persist(record)
            self._records[record.txid] = record
            return record

    def bind(self, record: OwnedPath, path: Path) -> None:
        """Bind a newly created inode to its pre-existing ownership record."""

        with self._lock:
            identity = _inode(Path(path), record.kind)
            if identity is None:
                raise FileNotFoundError(path)
            record.paths = (Path(path),)
            record.device, record.inode = identity
            self._persist(record)

    def set_charge(
        self,
        record: OwnedPath,
        *,
        reserved_bytes: int | None = None,
        reserved_slots: int | None = None,
    ) -> None:
        with self._lock:
            old = record.reserved_bytes, record.reserved_slots
            if reserved_bytes is not None:
                if reserved_bytes < 0:
                    raise ValueError("reserved bytes cannot be negative")
                record.reserved_bytes = reserved_bytes
            if reserved_slots is not None:
                if reserved_slots < 0:
                    raise ValueError("reserved slots cannot be negative")
                record.reserved_slots = reserved_slots
            try:
                self._persist(record)
            except BaseException:
                record.reserved_bytes, record.reserved_slots = old
                raise

    def prepare_rename(
        self, record: OwnedPath, source: Path, destination: Path
    ) -> None:
        """Persist both possible inode names before the filesystem rename."""

        with self._lock:
            source = Path(source)
            destination = Path(destination)
            self._relative_name(source)
            self._relative_name(destination)
            old_paths = record.paths
            record.paths = (source, destination)
            try:
                self._persist(record)
            except BaseException:
                record.paths = old_paths
                raise

    def finish_rename(self, record: OwnedPath, destination: Path) -> None:
        with self._lock:
            destination = Path(destination)
            identity = _inode(destination, record.kind)
            if identity is None:
                raise FileNotFoundError(destination)
            if record.identity is not None and identity != record.identity:
                raise OSError("owned inode changed during rename")
            old = record.paths, record.device, record.inode
            record.paths = (destination,)
            record.device, record.inode = identity
            try:
                self._persist(record)
            except BaseException:
                record.paths, record.device, record.inode = old
                raise

    def reconcile(self, record: OwnedPath) -> tuple[Path, ...]:
        """Return current paths which still name the owned inode."""

        with self._lock:
            identity = record.identity
            matches: list[Path] = []
            if identity is None:
                # A crash can happen after the marker is durable but before
                # inode creation.  Never infer ownership from a later inode at
                # that name; it may be unmarked user data created after crash.
                return ()
            else:
                for path in record.paths:
                    if _inode(path, record.kind) == identity:
                        matches.append(path)
            if matches and tuple(matches) != record.paths:
                record.paths = tuple(matches)
                self._persist(record)
            return tuple(matches)

    def owns(self, path: Path) -> bool:
        """Whether a non-committed record owns the inode currently at path."""

        with self._lock:
            candidate = Path(path)
            for record in self._records.values():
                identity = _inode(candidate, record.kind)
                if (
                    identity is not None
                    and record.state != "committed"
                    and candidate in record.paths
                    and record.identity == identity
                ):
                    return True
            return False

    def find(self, *paths: Path) -> OwnedPath | None:
        candidates = {Path(path) for path in paths}
        with self._lock:
            for record in self._records.values():
                if candidates.intersection(record.paths):
                    return record
        return None

    def mark_cleanup(self, record: OwnedPath) -> None:
        with self._lock:
            old_state = record.state
            record.state = "cleanup"
            try:
                self._persist(record)
            except BaseException:
                record.state = old_state
                raise

    def commit(self, record: OwnedPath) -> None:
        """Transfer the inode to committed disk usage before dropping metadata."""

        with self._lock:
            old = record.state, record.reserved_bytes, record.reserved_slots
            record.state = "committed"
            record.reserved_bytes = 0
            record.reserved_slots = 0
            try:
                self._persist(record)
            except BaseException:
                record.state, record.reserved_bytes, record.reserved_slots = old
                raise
            try:
                self._record_path(record.txid).unlink()
                self._sync_directory()
            except OSError:
                # The committed record is safe after restart: recovery removes
                # only the metadata and never the published inode.
                return
            self._records.pop(record.txid, None)
            self._release_lease_if_idle()

    def abandon_unbound(self, record: OwnedPath) -> None:
        """Drop a marker after O_EXCL found an inode we never owned.

        The durable committed state is written before metadata removal, so a
        crash can never reinterpret the colliding user inode as cleanup data.
        """

        with self._lock:
            if record.identity is not None:
                raise ValueError("cannot abandon a bound ownership record")
            self.commit(record)

    def _drop_empty(self, record: OwnedPath) -> None:
        # Persist a terminal, non-deleting state before metadata removal.  If
        # the basename is recreated and even reuses the old inode number after
        # a crash, recovery must only remove this sidecar, never the new data.
        record.state = "committed"
        record.reserved_bytes = 0
        record.reserved_slots = 0
        record.device = None
        record.inode = None
        record.paths = record.paths[:1]
        self._persist(record)
        self._record_path(record.txid).unlink()
        self._sync_directory()
        self._records.pop(record.txid, None)
        self._release_lease_if_idle()

    def _quarantine(self, record: OwnedPath, path: Path) -> Path:
        """Move an owned inode to a private transaction name before removal."""

        quarantine = self.root / f".custback-cleanup-{record.txid}"
        if path == quarantine:
            if _inode(path, record.kind) != record.identity:
                raise OSError("cleanup quarantine no longer names the owned inode")
            return quarantine
        self.prepare_rename(record, path, quarantine)
        try:
            rename_noreplace(path, quarantine)
            self.finish_rename(record, quarantine)
            return quarantine
        except OSError as primary:
            # If a path swap raced the rename, put that unmarked inode back.
            # It is never passed to the remover unless finish_rename proved
            # that the quarantine contains our recorded inode.
            if (
                _inode(quarantine, record.kind) is not None
                and _inode(quarantine, record.kind) != record.identity
                and not path.exists()
                and not path.is_symlink()
            ):
                try:
                    rename_noreplace(quarantine, path)
                except OSError:
                    pass
            try:
                self.reconcile(record)
            except OSError:
                pass
            raise primary

    def cleanup(
        self,
        record: OwnedPath,
        remove: Callable[[Path, str], None],
    ) -> OSError | None:
        """Try removal once, retaining record/charge on every failure."""

        with self._lock:
            try:
                record.state = "cleanup"
                self._persist(record)
                paths = self.reconcile(record)
                while paths:
                    quarantine = self._quarantine(record, paths[0])
                    if _inode(quarantine, record.kind) != record.identity:
                        raise OSError("cleanup quarantine changed before removal")
                    remove(quarantine, record.kind)
                    sync_directory(self.root)
                    paths = self.reconcile(record)
                if self.reconcile(record):
                    raise OSError("owned inode remains after cleanup")
                # Repeat on retries where a prior unlink/rmtree succeeded but
                # the directory fsync failed; metadata cannot be dropped until
                # the namespace deletion itself is durable.
                sync_directory(self.root)
                self._drop_empty(record)
                record.cleanup_error = None
                return None
            except OSError as exc:
                # Reconcile again so a partial recursive removal or raced
                # rollback retains the actual authoritative name.
                try:
                    self.reconcile(record)
                    self._persist(record)
                except OSError:
                    pass
                record.cleanup_error = exc
                return exc

    def retry_cleanup(
        self, remove: Callable[[Path, str], None]
    ) -> tuple[OwnedPath, ...]:
        """Retry all recovered/pending records and return those still pending."""

        with self._lock:
            if self._lease_descriptor is None and (
                self.metadata_root.exists() or self.metadata_root.is_symlink()
            ):
                self._acquire_lease(create=False)
            pending = tuple(self._records.values())
        for record in pending:
            if record.state == "committed":
                try:
                    self._record_path(record.txid).unlink()
                    self._records.pop(record.txid, None)
                    self._release_lease_if_idle()
                except FileNotFoundError:
                    self._records.pop(record.txid, None)
                    self._release_lease_if_idle()
                except OSError:
                    pass
                continue
            if record.state == "cleanup":
                self.cleanup(record, remove)
        return tuple(record for record in self.records if record.state == "cleanup")

    def pending_paths(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(
                path
                for record in self._records.values()
                if record.state == "cleanup"
                for path in record.paths
            )

    def active_identities(self) -> frozenset[tuple[int, int]]:
        with self._lock:
            return frozenset(
                identity
                for record in self._records.values()
                if record.state != "committed"
                if (identity := record.identity) is not None
            )

    def records_for(self, paths: Iterable[Path]) -> tuple[OwnedPath, ...]:
        candidates = {Path(path) for path in paths}
        with self._lock:
            return tuple(
                record
                for record in self._records.values()
                if candidates.intersection(record.paths)
            )
