"""On-host stores for uploaded avatar rigs and scene media, plus tile thumbnails.

Rigs and avatar backgrounds must live on the machine that runs the avatar
service — that is where :class:`~custback.avatar.rig.LayeredRig` loads PNG
layers and where the backdrop providers open media files. Uploads therefore
arrive through the avatar control API (possibly proxied by custback) and
land here.

Rig archives are never extracted with ``ZipFile.extract``: every member is
allow-listed by basename (``<part>.png``, expression variants, ``rig.yaml``),
re-written under our own name, and capped in count and uncompressed size, so
zip-slip and zip-bomb payloads cannot escape or exhaust the host. A staged
rig only becomes visible after :class:`~custback.avatar.rig.LayeredRig`
accepts it.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import secrets
import shutil
import stat
import threading
import warnings
import zipfile
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import yaml
from yaml.events import AliasEvent

from .. import _platform as platform_fs
from ..backgrounds import IMAGE_EXTS, VIDEO_EXTS
from ..storage_tx import (
    OwnedPath,
    OwnershipLedger,
    rename_noreplace,
)
from .config import (
    AVATAR_PARTS,
    AppearanceConfig,
    AvatarFraming,
    StorageConfig,
)
from .renderer import compose_avatar
from .rig import (
    DEFAULT_RIG_LAYER_MAX_PIXELS,
    DEFAULT_RIG_MANIFEST_MAX_BYTES,
    DEFAULT_RIG_TOTAL_MAX_PIXELS,
    RigError,
    create_rig,
)
from .state import FaceState

try:
    import cv2
except ImportError:  # pragma: no cover - required by the package, defensive
    cv2 = None

try:  # Pillow supplies a safe header/dimension check before OpenCV allocates.
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - OpenCV remains a compatibility fallback
    Image = None
    UnidentifiedImageError = OSError

log = logging.getLogger(__name__)

RIG_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}\Z")
_MEDIA_STEM_MAX = 48
_ZIP_READ_CHUNK = 1024 * 1024
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600

_IMAGE_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".webp": "WEBP",
}


class _RigManifestLoader(yaml.SafeLoader):
    """SafeLoader variant rejecting aliases and their recursive/amplified graphs."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            raise yaml.YAMLError("rig.yaml aliases are not allowed")
        return super().compose_node(parent, index)


# The only files a rig archive may carry (optionally under one shared
# top-level directory, which is flattened away).
RIG_MEMBER_NAMES = frozenset(
    {f"{part}.png" for part in AVATAR_PARTS}
    | {"eyes_closed.png", "mouth_open.png", "rig.yaml"}
)

THUMBNAIL_SIZE = (256, 144)  # 16:9, like a meeting tile
_THUMBNAIL_BACKDROP_BGR = (52, 44, 38)
_THUMBNAIL_JPEG_QUALITY = 82


class StoreError(Exception):
    """Storage failure carrying a stable API error code."""

    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        super().__init__(message)


def _owned_by_current_user(metadata: os.stat_result) -> bool:
    """Advisory pre-open owner check from a stat result.

    POSIX compares ``st_uid`` to the effective UID.  A Windows stat result
    carries no usable owner, so this is a pre-filter that defers to the
    authoritative post-open :func:`platform_fs.owner_matches` SID check in
    :func:`_secure_existing` (WIN-2.6 / CC-1).
    """

    return platform_fs.stat_owner_matches(metadata)


def _unsafe_storage(path: Path, message: str) -> StoreError:
    return StoreError(409, "unsafe_storage_path", f"{message}: {path}")


def _open_flags(*, directory: bool = False, writable: bool = False) -> int:
    """Base ``open`` flags; :func:`platform_fs.open_nofollow` adds no-follow and,
    when requested, the directory flag."""

    flags = os.O_WRONLY if writable else os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    if not directory and not writable:
        # Refuse a raced FIFO/device without waiting on it.
        flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _secure_existing(path: Path, mode: int, *, directory: bool) -> None:
    """Apply an exact private mode without following a final symlink."""

    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(before.st_mode):
        raise _unsafe_storage(path, "managed storage cannot be a symlink")
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(before.st_mode):
        kind = "directory" if directory else "regular file"
        raise _unsafe_storage(path, f"managed storage is not a {kind}")
    if not _owned_by_current_user(before):
        raise _unsafe_storage(path, "managed storage is not owned by this user")

    # A restrictive umask can create a mode-000 inode. Its owner may chmod it
    # but cannot bind it with os.open first. Repair by name only after a
    # no-follow lstat/ownership check, then verify the inode identity before
    # opening and fchmod'ing the authoritative descriptor.
    required_owner_bits = stat.S_IRUSR | stat.S_IXUSR if directory else stat.S_IRUSR
    if stat.S_IMODE(before.st_mode) & required_owner_bits != required_owner_bits:
        platform_fs.chmod_private(path, mode)
        after = path.lstat()
        if (
            stat.S_ISLNK(after.st_mode)
            or not expected_type(after.st_mode)
            or not _owned_by_current_user(after)
            or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise _unsafe_storage(path, "managed storage changed while securing")

    descriptor = platform_fs.open_nofollow(
        path, _open_flags(directory=directory), directory=directory
    )
    try:
        current = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise _unsafe_storage(path, "managed storage changed while opening")
        if not expected_type(current.st_mode):
            raise _unsafe_storage(path, "managed storage changed type while opening")
        # Authoritative owner check on the bound descriptor (SID on Windows).
        if not platform_fs.owner_matches(descriptor):
            raise _unsafe_storage(path, "managed storage is not owned by this user")
        platform_fs.set_private_mode(descriptor, mode)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    """Create/repair one managed directory as exactly mode 0700."""

    path.mkdir(parents=True, mode=_PRIVATE_DIRECTORY_MODE, exist_ok=True)
    _secure_existing(path, _PRIVATE_DIRECTORY_MODE, directory=True)


def _managed_directory_exists(path: Path) -> bool:
    """Validate a read root without following or changing it."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        raise _unsafe_storage(path, "managed storage cannot be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise _unsafe_storage(path, "managed storage is not a directory")
    if not _owned_by_current_user(metadata):
        raise _unsafe_storage(path, "managed storage is not owned by this user")
    return True


def _make_private_directory(path: Path) -> None:
    """Exclusively create one private managed child directory."""

    path.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
    # The caller creates a durable ownership marker before reaching here, so
    # any hardening failure is cleaned by that transaction rather than by a
    # best-effort rmdir which could forget the remaining inode.
    _secure_existing(path, _PRIVATE_DIRECTORY_MODE, directory=True)


def _open_private_file(path: Path) -> BinaryIO:
    """Exclusively bind a no-follow mode-0600 file and return it writable."""

    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = platform_fs.open_nofollow(path, flags, _PRIVATE_FILE_MODE)
        # Both permissive and restrictive umasks converge on the exact mode.
        platform_fs.set_private_mode(descriptor, _PRIVATE_FILE_MODE)
        destination = os.fdopen(descriptor, "wb")
        descriptor = None
        return destination
    except BaseException:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            descriptor = None
        # The containing upload/install transaction already owns ``path`` (or
        # its parent tree) and will retain retry metadata if removal fails.
        raise
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _open_regular_file(path: Path) -> BinaryIO:
    """Open an existing regular file for reading without following symlinks."""

    descriptor = platform_fs.open_nofollow(path, _open_flags())
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _unsafe_storage(path, "upload source is not a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _tree_size(path: Path) -> int:
    """Count regular-file bytes below ``path`` without following symlinks."""

    metadata = path.lstat()
    if stat.S_ISREG(metadata.st_mode):
        return metadata.st_size
    if not stat.S_ISDIR(metadata.st_mode):
        # Symlinks and special entries still consume a conservative inode-sized
        # quota charge, without ever following their targets.
        return metadata.st_size
    total = 0
    for child in path.iterdir():
        total += _tree_size(child)
    return total


def _remove_owned_path(path: Path, kind: str) -> None:
    """Remove one ledger-owned artifact without discarding failures."""

    if kind == "tree":
        shutil.rmtree(path)
    else:
        path.unlink()


@dataclass(frozen=True)
class StoragePermissionIssue:
    """One managed path whose privacy contract needs operator attention."""

    path: Path
    expected_mode: int
    actual_mode: int
    reason: str  # mode | symlink | owner | type | unreadable


def _permission_nodes(root: Path):
    """Yield a no-follow snapshot of a managed tree."""

    try:
        metadata = root.lstat()
    except FileNotFoundError:
        return
    yield root, metadata
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return
    try:
        children = list(root.iterdir())
    except OSError:
        return
    for child in children:
        yield from _permission_nodes(child)


def audit_storage_permissions(
    cfg: StorageConfig,
) -> tuple[StoragePermissionIssue, ...]:
    """Audit existing avatar stores without following or changing any path."""

    issues: list[StoragePermissionIssue] = []
    for root in (
        Path(cfg.rigs_dir).expanduser(),
        Path(cfg.backgrounds_dir).expanduser(),
    ):
        for path, metadata in _permission_nodes(root):
            is_directory = stat.S_ISDIR(metadata.st_mode)
            expected = _PRIVATE_DIRECTORY_MODE if is_directory else _PRIVATE_FILE_MODE
            actual = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                reason = "symlink"
            elif not _owned_by_current_user(metadata):
                reason = "owner"
            elif not (is_directory or stat.S_ISREG(metadata.st_mode)):
                reason = "type"
            elif actual != expected:
                reason = "mode"
            else:
                continue
            issues.append(StoragePermissionIssue(path, expected, actual, reason))
    return tuple(issues)


def repair_storage_permissions(cfg: StorageConfig) -> tuple[Path, ...]:
    """Repair user-owned modes, refusing symlinks/non-owned/special paths.

    Snapshots are repeated because a mode-000 directory cannot be traversed
    until its own mode is repaired. Unsafe entries are never followed or
    changed; making an inaccessible parent private may reveal one on the next
    pass, at which point repair stops with ``unsafe_storage_path``.
    """

    repaired: list[Path] = []
    repaired_set: set[Path] = set()
    while True:
        issues = audit_storage_permissions(cfg)
        unsafe = [issue for issue in issues if issue.reason != "mode"]
        if unsafe:
            issue = unsafe[0]
            raise _unsafe_storage(
                issue.path,
                f"cannot repair managed storage ({issue.reason})",
            )
        if not issues:
            break
        # Secure deepest visible entries first, then their containing dirs.
        for issue in sorted(
            issues, key=lambda item: len(item.path.parts), reverse=True
        ):
            _secure_existing(
                issue.path,
                issue.expected_mode,
                directory=issue.expected_mode == _PRIVATE_DIRECTORY_MODE,
            )
            if issue.path not in repaired_set:
                repaired.append(issue.path)
                repaired_set.add(issue.path)
    return tuple(repaired)


class UploadReservation(os.PathLike[str]):
    """Store-owned private staging file with chunk-level quota accounting."""

    def __init__(
        self,
        store: Any,
        path: Path,
        destination: BinaryIO,
        *,
        max_bytes: int,
        ownership: OwnedPath,
    ) -> None:
        self._store = store
        self.path = path
        self._ownership = ownership
        self._destination: BinaryIO | None = destination
        self.max_bytes = max_bytes
        self.reserved_bytes = 0
        self._active = True
        self._io_lock = threading.RLock()

    @property
    def active(self) -> bool:
        return self._active

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def stat(self) -> os.stat_result:
        with self._io_lock:
            if self._destination is not None:
                return os.fstat(self._destination.fileno())
        return self.path.stat()

    def write(self, payload: bytes | bytearray | memoryview) -> int:
        data = memoryview(payload)
        if not data:
            return 0
        with self._io_lock:
            if not self._active or self._destination is None:
                raise ValueError("upload reservation is closed")
            try:
                self._store._reserve_chunk(self, len(data))
                written = self._destination.write(data)
                if written != len(data):
                    raise OSError("short staging-file write")
                return written
            except OSError as exc:
                self._abort_after_primary_failure()
                raise StoreError(
                    507,
                    "insufficient_storage",
                    "cannot write upload staging",
                ) from exc
            except BaseException:
                self._abort_after_primary_failure()
                raise

    def _abort_after_primary_failure(self) -> None:
        """Retain cleanup ownership without replacing the operation error."""

        try:
            self.abort()
        except OSError:
            try:
                with self._store._lock:
                    self._store._detach_failed_upload_locked(self)
            except OSError:
                # The ledger already retained its last durable candidates and
                # charge; the original write/quota error remains authoritative.
                pass

    def write_bytes(self, payload: bytes) -> int:
        """Path-compatible one-shot helper that retains reservation ownership."""

        return self.write(payload)

    def seal(self) -> None:
        """Durably close the staging inode without releasing its reservation."""

        with self._io_lock:
            if not self._active or self._destination is None:
                return
            destination = self._destination
            try:
                destination.flush()
                os.fsync(destination.fileno())
            finally:
                destination.close()
                self._destination = None

    def _close_noexcept(self) -> None:
        with self._io_lock:
            if self._destination is not None:
                with contextlib.suppress(OSError):
                    self._destination.close()
                self._destination = None

    def _finish(self) -> None:
        self._close_noexcept()
        self._active = False

    def abort(self) -> None:
        """Delete staging and release every byte/file reservation, idempotently."""

        with self._io_lock:
            if not self._active:
                return
            self._close_noexcept()
            self._store._abort_reservation(self)

    def unlink(self, missing_ok: bool = False) -> None:
        existed = self.path.exists() or self.path.is_symlink()
        self.abort()
        if not existed and not missing_ok:
            raise FileNotFoundError(self.path)


@dataclass(frozen=True)
class InstalledRig:
    name: str
    parts: tuple[str, ...]
    has_manifest: bool
    size_bytes: int


@dataclass(frozen=True)
class StoredMedia:
    name: str
    kind: str  # "image" | "video"
    size: int
    path: str


def is_rig_directory(path: str | Path) -> bool:
    """Return whether ``path`` is a real directory, never a symlink."""

    try:
        metadata = Path(path).expanduser().lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)


def resolve_rig_selector(selector: str, rigs_dir: str | Path) -> str:
    """Map ``appearance.rig`` to something :func:`create_rig` understands.

    ``builtin`` and existing directories pass through; a bare installed-rig
    name resolves to its directory in ``rigs_dir``. Unknown selectors are
    returned unchanged so rig loading reports its usual error.
    """
    if selector == "builtin":
        return selector
    candidate = Path(selector).expanduser()
    if is_rig_directory(candidate):
        return str(candidate)
    if RIG_NAME_RE.fullmatch(selector):
        root = Path(rigs_dir).expanduser()
        installed = root / selector
        if is_rig_directory(root) and is_rig_directory(installed):
            return str(installed)
    return selector


def sanitize_media_name(name: str, kind: str) -> str:
    """Normalize an upload name to a safe ``stem.ext`` for ``kind``."""
    allowed = IMAGE_EXTS if kind == "image" else VIDEO_EXTS
    base = os.path.basename(name.strip().replace("\\", "/")).lower()
    stem, dot, suffix = base.rpartition(".")
    suffix = f".{suffix}" if dot else ""
    if suffix not in allowed:
        raise StoreError(
            415,
            "unsupported_media_type",
            f"{kind} name must end in one of: {', '.join(sorted(allowed))}",
        )
    stem = re.sub(r"[^a-z0-9._-]+", "-", stem).strip(".-")
    if not stem:
        stem = f"{kind}-{secrets.token_hex(4)}"
    return f"{stem[:_MEDIA_STEM_MAX]}{suffix}"


def _neutral_face_state() -> FaceState:
    """A pleasant resting pose for thumbnails: soft smile, slight turn."""
    state = FaceState.neutral()
    state.yaw = 0.05
    state.set_channel("mouthSmileLeft", 0.22)
    state.set_channel("mouthSmileRight", 0.22)
    return state


def render_avatar_thumbnail(
    selector: str,
    *,
    avatar: str = "casey",
    style: str = "cartoon",
    framing: AvatarFraming = "bust",
    size: tuple[int, int] = THUMBNAIL_SIZE,
    rig_layer_max_pixels: int = DEFAULT_RIG_LAYER_MAX_PIXELS,
    rig_total_max_pixels: int = DEFAULT_RIG_TOTAL_MAX_PIXELS,
    rig_manifest_max_bytes: int = DEFAULT_RIG_MANIFEST_MAX_BYTES,
) -> bytes:
    """Render one avatar tile as JPEG bytes (raises RigError on bad rigs)."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required for thumbnails")
    width, height = size
    rig = create_rig(
        selector,
        avatar=avatar,
        style=style,
        rig_layer_max_pixels=rig_layer_max_pixels,
        rig_total_max_pixels=rig_total_max_pixels,
        rig_manifest_max_bytes=rig_manifest_max_bytes,
    )
    try:
        sprite = rig.render(_neutral_face_state(), frozenset(AVATAR_PARTS))
        backdrop = np.full((height, width, 3), _THUMBNAIL_BACKDROP_BGR, dtype=np.uint8)
        frame = compose_avatar(
            sprite,
            backdrop,
            AppearanceConfig(framing=framing),
            framing_window=rig.framing_window(framing),
        )
    finally:
        rig.close()
    ok, jpeg = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _THUMBNAIL_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("thumbnail JPEG encoding failed")
    return jpeg.tobytes()


def render_media_thumbnail(
    path: Path, kind: str, *, size: tuple[int, int] = THUMBNAIL_SIZE
) -> bytes:
    """First frame (video) or downscaled image as JPEG bytes."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required for thumbnails")
    if kind == "image":
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    else:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            if not ok:
                frame = None
        finally:
            capture.release()
    if frame is None or frame.ndim != 3:
        raise StoreError(422, "invalid_media", "stored media cannot be decoded")
    height, width = frame.shape[:2]
    scale = min(size[0] / width, size[1] / height, 1.0)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, jpeg = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _THUMBNAIL_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("thumbnail JPEG encoding failed")
    return jpeg.tobytes()


class ThumbnailCache:
    """Small keyed JPEG cache; keys should include content identity (mtime)."""

    def __init__(self, capacity: int = 64):
        self._capacity = capacity
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, bytes] = OrderedDict()

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: tuple, value: bytes) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)


class RigStore:
    """Installed PNG-layer rigs under one managed directory."""

    def __init__(self, cfg: StorageConfig):
        self.directory = Path(cfg.rigs_dir).expanduser()
        self._cfg = cfg
        self._lock = threading.RLock()
        self._ledger = OwnershipLedger(self.directory)
        self._external_reserved_bytes = 0
        self._reserved_bytes = self._ledger.reserved_bytes
        self._reserved_rigs = self._ledger.reserved_slots
        self._active_uploads: dict[Path, UploadReservation] = {}
        self._active_extractions: set[Path] = set()
        self._extraction_records: dict[Path, OwnedPath] = {}
        self._retry_pending_cleanup()

    def _sync_reservations_locked(self) -> None:
        self._reserved_bytes = (
            self._ledger.reserved_bytes + self._external_reserved_bytes
        )
        self._reserved_rigs = self._ledger.reserved_slots

    def _retry_pending_cleanup(self) -> None:
        with self._lock:
            self._ledger.retry_cleanup(_remove_owned_path)
            self._sync_reservations_locked()

    @property
    def _cleanup_pending(self) -> tuple[Path, ...]:
        return self._ledger.pending_paths()

    def rig_path(self, name: str) -> Path:
        if not RIG_NAME_RE.fullmatch(name):
            raise StoreError(
                422,
                "invalid_rig_name",
                "rig names are 1-32 lowercase letters, digits, '-' or '_' "
                "and start with a letter or digit",
            )
        return self.directory / name

    def _describe(self, path: Path) -> InstalledRig:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise _unsafe_storage(path, "installed rig is not a regular directory")
        if not _owned_by_current_user(metadata):
            raise _unsafe_storage(path, "installed rig is not owned by this user")
        files: dict[str, os.stat_result] = {}
        for entry in path.iterdir():
            child = entry.lstat()
            if stat.S_ISLNK(child.st_mode) or not stat.S_ISREG(child.st_mode):
                raise _unsafe_storage(
                    entry, "installed rig contains a non-regular asset"
                )
            if not _owned_by_current_user(child):
                raise _unsafe_storage(
                    entry, "installed rig asset is not owned by this user"
                )
            files[entry.name] = child
        parts = tuple(part for part in AVATAR_PARTS if f"{part}.png" in files)
        size = sum(metadata.st_size for metadata in files.values())
        return InstalledRig(
            name=path.name,
            parts=parts,
            has_manifest="rig.yaml" in files,
            size_bytes=size,
        )

    def list(self) -> list[InstalledRig]:
        if not _managed_directory_exists(self.directory):
            return []
        rigs = []
        for entry in sorted(self.directory.iterdir()):
            if self._ledger.owns(entry):
                continue
            try:
                metadata = entry.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(metadata.st_mode) and RIG_NAME_RE.fullmatch(entry.name):
                rigs.append(self._describe(entry))
        return rigs

    def stored_path(self, name: str) -> Path:
        """Return an installed rig directory after a complete no-follow check."""

        path = self.rig_path(name)
        if not _managed_directory_exists(self.directory):
            raise StoreError(404, "rig_not_found", f"no installed rig {name!r}")
        if self._ledger.owns(path):
            raise StoreError(404, "rig_not_found", f"no installed rig {name!r}")
        try:
            self._describe(path)
        except FileNotFoundError as exc:
            raise StoreError(
                404, "rig_not_found", f"no installed rig {name!r}"
            ) from exc
        return path

    def _usage_locked(self) -> tuple[int, int]:
        """Return on-disk bytes and rig slots not represented by reservations."""

        if not _managed_directory_exists(self.directory):
            return 0, 0
        total = count = 0
        active = set(self._active_uploads) | self._active_extractions
        for entry in self.directory.iterdir():
            if entry in active:
                continue
            if self._ledger.owns(entry):
                continue
            try:
                total += _tree_size(entry)
                metadata = entry.lstat()
            except FileNotFoundError:
                continue
            # Installed rigs consume one slot. Crash-left hidden directories
            # also consume a conservative slot until repaired/reclaimed.
            if stat.S_ISDIR(metadata.st_mode):
                count += 1
        return total, count

    def _reserve_chunk(self, reservation: UploadReservation, amount: int) -> None:
        with self._lock:
            if self._active_uploads.get(reservation.path) is not reservation:
                raise ValueError("upload reservation does not belong to this rig store")
            if reservation.reserved_bytes + amount > reservation.max_bytes:
                raise StoreError(
                    413,
                    "rig_too_large",
                    f"rig archive exceeds {reservation.max_bytes} bytes",
                )
            used, _count = self._usage_locked()
            if used + self._reserved_bytes + amount > self._cfg.rig_storage_max_bytes:
                raise StoreError(
                    507,
                    "storage_full",
                    "rig upload does not fit within rig_storage_max_bytes",
                )
            self._reserved_bytes += amount
            reservation.reserved_bytes += amount
            self._ledger.set_charge(
                reservation._ownership,
                reserved_bytes=reservation.reserved_bytes,
            )
            self._sync_reservations_locked()

    def _release_upload_locked(
        self, reservation: UploadReservation, *, remove: bool
    ) -> None:
        if reservation._ownership not in self._ledger.records:
            reservation.reserved_bytes = 0
            reservation._finish()
            return
        if remove:
            error = self._ledger.cleanup(reservation._ownership, _remove_owned_path)
            self._sync_reservations_locked()
            if error is not None:
                for path, active in tuple(self._active_uploads.items()):
                    if active is reservation:
                        self._active_uploads.pop(path, None)
                if reservation._ownership.paths:
                    reservation.path = reservation._ownership.paths[0]
                    self._active_uploads[reservation.path] = reservation
                raise error
        else:
            self._ledger.commit(reservation._ownership)
        for path, active in tuple(self._active_uploads.items()):
            if active is reservation:
                self._active_uploads.pop(path, None)
        reservation.reserved_bytes = 0
        reservation._finish()
        self._sync_reservations_locked()

    def _detach_failed_upload_locked(self, reservation: UploadReservation) -> None:
        """Queue failed cleanup without allowing it to mask the primary error."""

        try:
            self._ledger.mark_cleanup(reservation._ownership)
        except OSError:
            pass
        self._ledger.cleanup(reservation._ownership, _remove_owned_path)
        for path, active in tuple(self._active_uploads.items()):
            if active is reservation:
                self._active_uploads.pop(path, None)
        reservation.reserved_bytes = 0
        reservation._finish()
        self._sync_reservations_locked()

    def _abort_reservation(self, reservation: UploadReservation) -> None:
        with self._lock:
            self._release_upload_locked(reservation, remove=True)

    def open_staging(self) -> UploadReservation:
        """Create a private, quota-owned staging file for one rig archive."""

        try:
            _ensure_private_directory(self.directory)
            with self._lock:
                self._retry_pending_cleanup()
                _used, count = self._usage_locked()
                if count + self._reserved_rigs >= self._cfg.max_rigs:
                    raise StoreError(
                        507,
                        "storage_full",
                        f"the rig store already holds {self._cfg.max_rigs} rigs",
                    )
                while True:
                    path = self.directory / f".upload-{secrets.token_hex(16)}.zip"
                    if path.exists() or path.is_symlink():
                        continue
                    ownership = self._ledger.begin(path, kind="file", reserved_slots=1)
                    try:
                        destination = _open_private_file(path)
                        self._ledger.bind(ownership, path)
                        break
                    except FileExistsError:
                        self._ledger.abandon_unbound(ownership)
                        continue
                    except BaseException:
                        self._ledger.mark_cleanup(ownership)
                        self._ledger.cleanup(ownership, _remove_owned_path)
                        self._sync_reservations_locked()
                        raise
                reservation = UploadReservation(
                    self,
                    path,
                    destination,
                    max_bytes=self._cfg.rig_zip_max_bytes,
                    ownership=ownership,
                )
                self._active_uploads[path] = reservation
                self._sync_reservations_locked()
                return reservation
        except StoreError:
            raise
        except OSError as exc:
            raise StoreError(
                507, "insufficient_storage", "cannot create rig upload staging"
            ) from exc

    @staticmethod
    def _member_basenames(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
        """Map allow-listed basenames to members, flattening one shared root."""
        files = [info for info in archive.infolist() if not info.is_dir()]
        if not files:
            raise StoreError(422, "invalid_rig", "rig archive contains no files")
        names = [info.filename.replace("\\", "/").lstrip("/") for info in files]
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        strip_root = len(roots) == 1 and all("/" in name for name in names)
        members: dict[str, zipfile.ZipInfo] = {}
        for info, name in zip(files, names):
            if strip_root:
                name = name.split("/", 1)[1]
            if "/" in name or name.startswith("."):
                raise StoreError(
                    422, "invalid_rig", f"unexpected archive entry: {info.filename}"
                )
            if name not in RIG_MEMBER_NAMES:
                raise StoreError(
                    422,
                    "invalid_rig",
                    f"unexpected archive entry {info.filename!r}; rigs may only "
                    "contain <part>.png layers, eyes_closed.png, mouth_open.png "
                    "and rig.yaml",
                )
            if name in members:
                raise StoreError(422, "invalid_rig", f"duplicate archive entry: {name}")
            members[name] = info
        return members

    def _reserve_extracted_locked(self, ownership: OwnedPath, amount: int) -> None:
        used, _count = self._usage_locked()
        if used + self._reserved_bytes + amount > self._cfg.rig_storage_max_bytes:
            raise StoreError(
                507,
                "storage_full",
                "compressed and extracted rig staging exceeds rig_storage_max_bytes",
            )
        self._ledger.set_charge(
            ownership,
            reserved_bytes=ownership.reserved_bytes + amount,
        )
        self._sync_reservations_locked()

    def _extract(
        self,
        archive: zipfile.ZipFile,
        destination: Path,
        extracted: list[int],
        ownership: OwnedPath,
    ) -> None:
        # Bound the complete central-directory inventory, not just extracted
        # files. One optional explicit top-level directory record is tolerated
        # for archives that also place every layer beneath that shared root.
        inventory = archive.infolist()
        if len(inventory) > self._cfg.rig_max_entries + 1:
            raise StoreError(
                413,
                "rig_too_large",
                "rig archive central directory exceeds the configured entry limit",
            )
        directory_entries = [info for info in inventory if info.is_dir()]
        if len(directory_entries) > 1:
            raise StoreError(
                422,
                "invalid_rig",
                "rig archive may contain at most one top-level directory entry",
            )
        members = self._member_basenames(archive)
        if len(members) > self._cfg.rig_max_entries:
            raise StoreError(
                413,
                "rig_too_large",
                f"rig archives may contain at most {self._cfg.rig_max_entries} files",
            )
        remaining = self._cfg.rig_max_bytes
        for name, info in members.items():
            member_size = 0
            try:
                source = archive.open(info)
            except (RuntimeError, zipfile.BadZipFile) as exc:
                raise StoreError(
                    422, "invalid_rig", f"cannot read archive entry {name!r}"
                ) from exc
            with source, _open_private_file(destination / name) as out:
                while True:
                    chunk = source.read(min(_ZIP_READ_CHUNK, remaining + 1))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    member_size += len(chunk)
                    if remaining < 0:
                        raise StoreError(
                            413,
                            "rig_too_large",
                            f"uncompressed rig exceeds {self._cfg.rig_max_bytes} bytes",
                        )
                    if (
                        name == "rig.yaml"
                        and member_size > self._cfg.rig_manifest_max_bytes
                    ):
                        raise StoreError(
                            413,
                            "rig_too_large",
                            "rig.yaml exceeds "
                            f"{self._cfg.rig_manifest_max_bytes} bytes",
                        )
                    self._reserve_extracted_locked(ownership, len(chunk))
                    extracted[0] += len(chunk)
                    out.write(chunk)

    @staticmethod
    def _has_alpha(image: Any) -> bool:
        return "A" in image.getbands() or "transparency" in image.info

    def _validate_layer_headers(self, destination: Path) -> None:
        """Fully validate PNG layers with Pillow before OpenCV sees a path."""

        if Image is None:
            raise StoreError(
                503, "decoder_unavailable", "Pillow is required to validate rigs"
            )
        expected_size: tuple[int, int] | None = None
        total_pixels = 0
        layers = sorted(destination.glob("*.png"))
        for path in layers:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with Image.open(path) as image:
                        if image.format != "PNG":
                            raise ValueError("layer header is not PNG")
                        width, height = image.size
                        if width <= 0 or height <= 0:
                            raise ValueError("invalid layer dimensions")
                        pixels = width * height
                        if pixels > self._cfg.rig_layer_max_pixels:
                            raise StoreError(
                                413,
                                "rig_too_large",
                                f"rig layer {path.name} exceeds "
                                f"{self._cfg.rig_layer_max_pixels} pixels",
                            )
                        total_pixels += pixels
                        if total_pixels > self._cfg.rig_total_max_pixels:
                            raise StoreError(
                                413,
                                "rig_too_large",
                                "rig decoded layers exceed "
                                f"{self._cfg.rig_total_max_pixels} pixels",
                            )
                        if expected_size is None:
                            expected_size = (width, height)
                        elif (width, height) != expected_size:
                            raise StoreError(
                                422,
                                "invalid_rig",
                                f"rig layer {path.name} size {width}x{height} "
                                f"does not match {expected_size[0]}x{expected_size[1]}",
                            )
                        if not self._has_alpha(image):
                            raise StoreError(
                                422,
                                "invalid_rig",
                                f"rig layer must be a PNG with alpha: {path.name}",
                            )
                        image.verify()
                    # ``verify`` checks structure/CRC; ``load`` then forces a
                    # complete decompression before cv2.imread can allocate.
                    with Image.open(path) as decoded:
                        decoded.load()
            except StoreError:
                raise
            except (
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
                UnidentifiedImageError,
                OSError,
                ValueError,
                Warning,
            ) as exc:
                raise StoreError(
                    422, "invalid_rig", f"invalid PNG rig layer: {path.name}"
                ) from exc

    def _validate_manifest(self, destination: Path) -> None:
        manifest = destination / "rig.yaml"
        if not manifest.is_file():
            return
        if manifest.stat().st_size > self._cfg.rig_manifest_max_bytes:
            raise StoreError(
                413,
                "rig_too_large",
                f"rig.yaml exceeds {self._cfg.rig_manifest_max_bytes} bytes",
            )
        try:
            raw = yaml.load(
                manifest.read_text(encoding="utf-8"),
                Loader=_RigManifestLoader,
            )
        except (OSError, UnicodeError, yaml.YAMLError, RecursionError) as exc:
            raise StoreError(422, "invalid_rig", "rig.yaml is invalid") from exc

        visited: set[int] = set()
        node_count = 0

        def has_non_finite(value: Any, depth: int = 0) -> bool:
            nonlocal node_count
            node_count += 1
            if depth > 32 or node_count > 2048:
                raise StoreError(
                    422, "invalid_rig", "rig.yaml structure is too complex"
                )
            if isinstance(value, bool) or value is None:
                return False
            if isinstance(value, (int, float)):
                try:
                    return not math.isfinite(float(value))
                except (OverflowError, ValueError):
                    return True
            if isinstance(value, dict):
                identity = id(value)
                if identity in visited:
                    raise StoreError(
                        422, "invalid_rig", "rig.yaml contains a recursive value"
                    )
                visited.add(identity)
                return any(
                    has_non_finite(key, depth + 1) or has_non_finite(item, depth + 1)
                    for key, item in value.items()
                )
            if isinstance(value, (list, tuple)):
                identity = id(value)
                if identity in visited:
                    raise StoreError(
                        422, "invalid_rig", "rig.yaml contains a recursive value"
                    )
                visited.add(identity)
                return any(has_non_finite(item, depth + 1) for item in value)
            return False

        if has_non_finite(raw):
            raise StoreError(422, "invalid_rig", "rig.yaml geometry must be finite")

    def _secure_installed_rig(self, path: Path) -> None:
        _secure_existing(path, _PRIVATE_DIRECTORY_MODE, directory=True)
        for child in path.iterdir():
            _secure_existing(child, _PRIVATE_FILE_MODE, directory=False)

    def install_zip(
        self, name: str, zip_path: Path | UploadReservation
    ) -> InstalledRig:
        """Validate and atomically install an uploaded rig archive."""
        final = self.rig_path(name)
        reservation = zip_path if isinstance(zip_path, UploadReservation) else None
        archive_path = reservation.path if reservation is not None else Path(zip_path)
        archive_file: BinaryIO | None = None
        external_reserved = 0
        extracted = [0]
        staging: Path | None = None
        extraction: OwnedPath | None = None
        installed: InstalledRig | None = None
        primary: BaseException | None = None
        try:
            if reservation is not None:
                if reservation._store is not self:
                    raise ValueError("rig upload reservation belongs to another store")
                reservation.seal()
            archive_file = _open_regular_file(archive_path)
            archive_size = os.fstat(archive_file.fileno()).st_size
            if reservation is not None and archive_size != reservation.reserved_bytes:
                raise StoreError(
                    409,
                    "invalid_reservation",
                    "rig staging size does not match its quota reservation",
                )
            if archive_size > self._cfg.rig_zip_max_bytes:
                raise StoreError(
                    413,
                    "rig_too_large",
                    f"rig archive exceeds {self._cfg.rig_zip_max_bytes} bytes",
                )
            _ensure_private_directory(self.directory)
            with self._lock:
                used, count = self._usage_locked()
                if reservation is not None:
                    if self._active_uploads.get(reservation.path) is not reservation:
                        raise ValueError("rig upload reservation is no longer active")
                    if count + self._reserved_rigs > self._cfg.max_rigs:
                        raise StoreError(
                            507,
                            "storage_full",
                            f"the rig store already holds {self._cfg.max_rigs} rigs",
                        )
                else:
                    if count + self._reserved_rigs >= self._cfg.max_rigs:
                        raise StoreError(
                            507,
                            "storage_full",
                            f"the rig store already holds {self._cfg.max_rigs} rigs",
                        )
                    if (
                        used + self._reserved_bytes + archive_size
                        > self._cfg.rig_storage_max_bytes
                    ):
                        raise StoreError(
                            507,
                            "storage_full",
                            "rig archive does not fit within rig_storage_max_bytes",
                        )
                    self._external_reserved_bytes += archive_size
                    external_reserved = archive_size
                    self._sync_reservations_locked()
                if final.exists() or final.is_symlink():
                    raise StoreError(
                        409,
                        "rig_exists",
                        f"rig {name!r} already exists; delete it first to replace it",
                    )
                while True:
                    staging = self.directory / f".staged-{secrets.token_hex(8)}"
                    if staging.exists() or staging.is_symlink():
                        continue
                    extraction = self._ledger.begin(
                        staging,
                        kind="tree",
                        reserved_slots=1,
                    )
                    try:
                        _make_private_directory(staging)
                        self._ledger.bind(extraction, staging)
                        break
                    except FileExistsError:
                        self._ledger.abandon_unbound(extraction)
                        extraction = None
                        continue
                    except BaseException:
                        self._ledger.mark_cleanup(extraction)
                        self._ledger.cleanup(extraction, _remove_owned_path)
                        extraction = None
                        raise
                assert extraction is not None
                if reservation is not None:
                    # One install consumes one rig slot even though the archive
                    # and extracted tree coexist.  Transfer the slot before any
                    # cleanup can detach the archive reservation.
                    self._ledger.set_charge(reservation._ownership, reserved_slots=0)
                self._sync_reservations_locked()
                self._active_extractions.add(staging)
                self._extraction_records[staging] = extraction
                try:
                    with zipfile.ZipFile(archive_file) as archive:
                        self._extract(archive, staging, extracted, extraction)
                except (
                    zipfile.BadZipFile,
                    zipfile.LargeZipFile,
                    RuntimeError,
                    EOFError,
                    NotImplementedError,
                    zlib.error,
                ) as exc:
                    raise StoreError(
                        422, "invalid_rig", "rig upload is not a valid zip archive"
                    ) from exc
                self._validate_layer_headers(staging)
                self._validate_manifest(staging)
                try:
                    create_rig(
                        str(staging),
                        rig_layer_max_pixels=self._cfg.rig_layer_max_pixels,
                        rig_total_max_pixels=self._cfg.rig_total_max_pixels,
                        rig_manifest_max_bytes=self._cfg.rig_manifest_max_bytes,
                    ).close()
                except RigError as exc:
                    raise StoreError(422, "invalid_rig", str(exc)) from exc
                archive_file.close()
                archive_file = None
                self._ledger.prepare_rename(extraction, staging, final)
                rename_noreplace(staging, final)
                self._ledger.finish_rename(extraction, final)
                self._active_extractions.discard(staging)
                self._extraction_records.pop(staging, None)
                self._active_extractions.add(final)
                self._extraction_records[final] = extraction
                self._secure_installed_rig(final)
                installed = self._describe(final)
                self._ledger.commit(extraction)
                self._active_extractions.discard(final)
                self._extraction_records.pop(final, None)
                extracted[0] = 0
                self._sync_reservations_locked()
        except BaseException as exc:
            primary = exc
        finally:
            if archive_file is not None:
                try:
                    archive_file.close()
                except OSError as exc:
                    if primary is None:
                        primary = exc
            with self._lock:
                if primary is not None and extraction is not None:
                    # If publication moved the inode, first try to restore its
                    # hidden name.  Both candidates were durable before each
                    # rename, so a second failure remains discoverable.
                    try:
                        current = self._ledger.reconcile(extraction)
                    except OSError:
                        current = ()
                    if final in current and staging is not None:
                        try:
                            self._ledger.prepare_rename(extraction, final, staging)
                            rename_noreplace(final, staging)
                            self._ledger.finish_rename(extraction, staging)
                        except OSError:
                            try:
                                self._ledger.reconcile(extraction)
                            except OSError:
                                pass
                    try:
                        self._ledger.mark_cleanup(extraction)
                    except OSError:
                        pass
                    self._ledger.cleanup(extraction, _remove_owned_path)
                    for path, record in tuple(self._extraction_records.items()):
                        if record is extraction:
                            self._extraction_records.pop(path, None)
                            self._active_extractions.discard(path)
                    if (
                        reservation is not None
                        and extraction not in self._ledger.records
                        and reservation._ownership in self._ledger.records
                    ):
                        # Cleanup removed the tree, so the still-owned archive
                        # resumes the failed install's single rig-slot charge.
                        self._ledger.set_charge(
                            reservation._ownership, reserved_slots=1
                        )
                if external_reserved:
                    self._external_reserved_bytes = max(
                        0, self._external_reserved_bytes - external_reserved
                    )
                if reservation is not None:
                    try:
                        self._release_upload_locked(reservation, remove=True)
                    except OSError:
                        self._detach_failed_upload_locked(reservation)
                self._sync_reservations_locked()
        if primary is not None:
            if isinstance(primary, StoreError):
                raise primary
            if isinstance(primary, OSError):
                raise StoreError(
                    507,
                    "insufficient_storage",
                    "cannot install rig archive",
                ) from primary
            raise primary
        assert installed is not None
        return installed

    def remove(self, name: str, active_selector: str) -> None:
        with self._lock:
            path = self.stored_path(name)
            active = resolve_rig_selector(active_selector, self.directory)
            if active != "builtin" and Path(active) == path:
                raise StoreError(
                    409,
                    "rig_in_use",
                    f"rig {name!r} is the active appearance.rig; switch rigs first",
                )
            shutil.rmtree(path)


class MediaStore:
    """Uploaded avatar-scene images and videos under one managed directory."""

    def __init__(self, cfg: StorageConfig):
        self.directory = Path(cfg.backgrounds_dir).expanduser()
        self._cfg = cfg
        self._lock = threading.RLock()
        self._ledger = OwnershipLedger(self.directory)
        self._reserved_bytes = self._ledger.reserved_bytes
        self._reserved_files = self._ledger.reserved_slots
        self._active_uploads: dict[Path, UploadReservation] = {}
        self._retry_pending_cleanup()

    def _sync_reservations_locked(self) -> None:
        self._reserved_bytes = self._ledger.reserved_bytes
        self._reserved_files = self._ledger.reserved_slots

    def _retry_pending_cleanup(self) -> None:
        with self._lock:
            self._ledger.retry_cleanup(_remove_owned_path)
            self._sync_reservations_locked()

    @property
    def _cleanup_pending(self) -> tuple[Path, ...]:
        return self._ledger.pending_paths()

    @staticmethod
    def kind_of(path: Path) -> str | None:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            return "image"
        if suffix in VIDEO_EXTS:
            return "video"
        return None

    def max_bytes(self, kind: str) -> int:
        if kind not in ("image", "video"):
            raise ValueError("media kind must be 'image' or 'video'")
        return (
            self._cfg.image_max_bytes if kind == "image" else self._cfg.video_max_bytes
        )

    def _usage_locked(self) -> tuple[int, int]:
        """Count committed and crash-left data not represented in memory."""

        if not _managed_directory_exists(self.directory):
            return 0, 0
        total = count = 0
        for entry in self.directory.iterdir():
            if entry in self._active_uploads:
                continue
            if self._ledger.owns(entry):
                continue
            try:
                total += _tree_size(entry)
            except FileNotFoundError:
                continue
            count += 1
        return total, count

    def _reserve_chunk(self, reservation: UploadReservation, amount: int) -> None:
        with self._lock:
            if self._active_uploads.get(reservation.path) is not reservation:
                raise ValueError(
                    "upload reservation does not belong to this media store"
                )
            if reservation.reserved_bytes + amount > reservation.max_bytes:
                raise StoreError(
                    413,
                    "upload_too_large",
                    f"media upload exceeds {reservation.max_bytes} bytes",
                )
            used, _count = self._usage_locked()
            if used + self._reserved_bytes + amount > self._cfg.storage_max_bytes:
                raise StoreError(
                    507,
                    "storage_full",
                    "media upload does not fit within storage_max_bytes",
                )
            self._reserved_bytes += amount
            reservation.reserved_bytes += amount
            self._ledger.set_charge(
                reservation._ownership,
                reserved_bytes=reservation.reserved_bytes,
            )
            self._sync_reservations_locked()

    def _release_upload_locked(
        self, reservation: UploadReservation, *, remove: bool
    ) -> None:
        if reservation._ownership not in self._ledger.records:
            reservation.reserved_bytes = 0
            reservation._finish()
            return
        if remove:
            error = self._ledger.cleanup(reservation._ownership, _remove_owned_path)
            self._sync_reservations_locked()
            if error is not None:
                for path, active in tuple(self._active_uploads.items()):
                    if active is reservation:
                        self._active_uploads.pop(path, None)
                if reservation._ownership.paths:
                    reservation.path = reservation._ownership.paths[0]
                    self._active_uploads[reservation.path] = reservation
                raise error
        else:
            self._ledger.commit(reservation._ownership)
        for path, active in tuple(self._active_uploads.items()):
            if active is reservation:
                self._active_uploads.pop(path, None)
        reservation.reserved_bytes = 0
        reservation._finish()
        self._sync_reservations_locked()

    def _detach_failed_upload_locked(self, reservation: UploadReservation) -> None:
        try:
            self._ledger.mark_cleanup(reservation._ownership)
        except OSError:
            pass
        self._ledger.cleanup(reservation._ownership, _remove_owned_path)
        for path, active in tuple(self._active_uploads.items()):
            if active is reservation:
                self._active_uploads.pop(path, None)
        reservation.reserved_bytes = 0
        reservation._finish()
        self._sync_reservations_locked()

    def _abort_reservation(self, reservation: UploadReservation) -> None:
        with self._lock:
            self._release_upload_locked(reservation, remove=True)

    def open_staging(self, kind: str | None = None) -> UploadReservation:
        """Reserve one private file slot and account every subsequent chunk."""

        if kind not in (None, "image", "video"):
            raise ValueError("media staging kind must be 'image' or 'video'")
        maximum = (
            self.max_bytes(kind)
            if kind is not None
            else max(self._cfg.image_max_bytes, self._cfg.video_max_bytes)
        )
        try:
            _ensure_private_directory(self.directory)
            with self._lock:
                self._retry_pending_cleanup()
                _used, count = self._usage_locked()
                if count + self._reserved_files >= self._cfg.max_files:
                    raise StoreError(
                        507,
                        "storage_full",
                        f"the store already holds {self._cfg.max_files} files",
                    )
                while True:
                    path = self.directory / f".upload-{secrets.token_hex(16)}.part"
                    if path.exists() or path.is_symlink():
                        continue
                    ownership = self._ledger.begin(path, kind="file", reserved_slots=1)
                    try:
                        destination = _open_private_file(path)
                        self._ledger.bind(ownership, path)
                        break
                    except FileExistsError:
                        self._ledger.abandon_unbound(ownership)
                        continue
                    except BaseException:
                        self._ledger.mark_cleanup(ownership)
                        self._ledger.cleanup(ownership, _remove_owned_path)
                        self._sync_reservations_locked()
                        raise
                reservation = UploadReservation(
                    self,
                    path,
                    destination,
                    max_bytes=maximum,
                    ownership=ownership,
                )
                self._active_uploads[path] = reservation
                self._sync_reservations_locked()
                return reservation
        except StoreError:
            raise
        except OSError as exc:
            raise StoreError(
                507, "insufficient_storage", "cannot create media upload staging"
            ) from exc

    def _entries(self) -> list[Path]:
        if not _managed_directory_exists(self.directory):
            return []
        entries: list[Path] = []
        for entry in sorted(self.directory.iterdir()):
            if self._ledger.owns(entry):
                continue
            try:
                metadata = entry.lstat()
            except FileNotFoundError:
                continue
            if (
                stat.S_ISREG(metadata.st_mode)
                and not entry.name.startswith(".")
                and self.kind_of(entry) is not None
            ):
                entries.append(entry)
        return entries

    def list(self) -> list[StoredMedia]:
        described = []
        for entry in self._entries():
            metadata = entry.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                continue
            kind = self.kind_of(entry)
            assert kind is not None
            described.append(
                StoredMedia(
                    name=entry.name,
                    kind=kind,
                    size=metadata.st_size,
                    path=str(entry),
                )
            )
        return described

    def stored_path(self, name: str) -> Path:
        """A store entry by exact name, refusing traversal and symlinks."""
        if (
            not name
            or name.startswith(".")
            or os.path.basename(name.replace("\\", "/")) != name
        ):
            raise StoreError(404, "media_not_found", "no such stored file")
        if not _managed_directory_exists(self.directory):
            raise StoreError(404, "media_not_found", "no such stored file")
        path = self.directory / name
        if self._ledger.owns(path):
            raise StoreError(404, "media_not_found", "no such stored file")
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise StoreError(404, "media_not_found", "no such stored file") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or self.kind_of(path) is None
        ):
            raise StoreError(404, "media_not_found", "no such stored file")
        if not _owned_by_current_user(metadata):
            raise _unsafe_storage(path, "stored media is not owned by this user")
        return path

    def _validate_image(self, path: Path, suffix: str) -> None:
        if Image is None:
            raise StoreError(
                503, "decoder_unavailable", "Pillow is required to validate images"
            )
        expected_format = _IMAGE_FORMATS.get(suffix)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(path) as image:
                    if image.format != expected_format:
                        raise ValueError("image header does not match its filename")
                    width, height = image.size
                    if width <= 0 or height <= 0:
                        raise ValueError("invalid image dimensions")
                    if width * height > self._cfg.image_max_pixels:
                        raise StoreError(
                            413,
                            "media_too_large",
                            f"image exceeds {self._cfg.image_max_pixels} pixels",
                        )
                    image.verify()
                with Image.open(path) as decoded:
                    decoded.load()
        except StoreError:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            ValueError,
            Warning,
        ) as exc:
            raise StoreError(
                422, "invalid_media", "image upload cannot be decoded"
            ) from exc
        if cv2 is not None:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if (
                frame is None
                or frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[2] != 3
            ):
                raise StoreError(422, "invalid_media", "image upload cannot be decoded")
            actual_height, actual_width = frame.shape[:2]
            if (actual_width, actual_height) != (width, height):
                raise StoreError(
                    422, "invalid_media", "image decoders disagree on dimensions"
                )

    def _validate_video(self, path: Path, suffix: str) -> None:
        if cv2 is None:  # pragma: no cover - required by the package
            return
        with _open_regular_file(path) as media_file:
            header = media_file.read(16)
        header_matches = {
            ".mp4": len(header) >= 8 and header[4:8] == b"ftyp",
            ".mov": len(header) >= 8 and header[4:8] == b"ftyp",
            ".webm": header.startswith(b"\x1aE\xdf\xa3"),
            ".mkv": header.startswith(b"\x1aE\xdf\xa3"),
            ".gif": header.startswith((b"GIF87a", b"GIF89a")),
            ".avi": header.startswith(b"RIFF") and header[8:12] == b"AVI ",
        }.get(suffix, False)
        if not header_matches:
            raise StoreError(
                422,
                "invalid_media",
                "video container header does not match its filename",
            )
        try:
            capture = cv2.VideoCapture(str(path))
        except Exception as exc:
            raise StoreError(
                422, "invalid_media", "video upload cannot be decoded"
            ) from exc
        try:
            try:
                if not capture.isOpened():
                    raise StoreError(
                        422, "invalid_media", "video upload cannot be decoded"
                    )
                metadata_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (
                    not math.isfinite(metadata_width)
                    or not math.isfinite(metadata_height)
                    or metadata_width < 1
                    or metadata_height < 1
                ):
                    raise StoreError(
                        422, "invalid_media", "video dimensions are missing"
                    )
                if (
                    metadata_width > self._cfg.video_max_width
                    or metadata_height > self._cfg.video_max_height
                ):
                    raise StoreError(
                        413,
                        "media_too_large",
                        f"video frames exceed {self._cfg.video_max_width}"
                        f"x{self._cfg.video_max_height}",
                    )
                decoded = 0
                while True:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    if (
                        not isinstance(frame, np.ndarray)
                        or frame.dtype != np.uint8
                        or frame.ndim != 3
                        or frame.shape[2] != 3
                        or frame.shape[0] < 1
                        or frame.shape[1] < 1
                    ):
                        raise StoreError(
                            422,
                            "invalid_media",
                            "video contains an invalid frame",
                        )
                    height, width = frame.shape[:2]
                    if (
                        width > self._cfg.video_max_width
                        or height > self._cfg.video_max_height
                    ):
                        raise StoreError(
                            413,
                            "media_too_large",
                            f"video frames exceed {self._cfg.video_max_width}"
                            f"x{self._cfg.video_max_height}",
                        )
                    decoded += 1
                if decoded == 0:
                    raise StoreError(
                        422, "invalid_media", "video upload cannot be decoded"
                    )
            except StoreError:
                raise
            except Exception as exc:
                raise StoreError(
                    422, "invalid_media", "video upload cannot be decoded"
                ) from exc
        finally:
            with contextlib.suppress(Exception):
                capture.release()

    def commit(self, staging: UploadReservation, name: str, kind: str) -> StoredMedia:
        """Validate a fully-written staging file and publish it."""
        if not isinstance(staging, UploadReservation) or staging._store is not self:
            raise ValueError("media commit requires this store's reservation")
        final: Path | None = None
        source = staging.path
        primary: BaseException | None = None
        saved: StoredMedia | None = None
        try:
            maximum = self.max_bytes(kind)
            safe_name = sanitize_media_name(name, kind)
            staging.seal()
            size = staging.stat().st_size
            if size != staging.reserved_bytes:
                raise StoreError(
                    409,
                    "invalid_reservation",
                    "media staging size does not match its quota reservation",
                )
            if size == 0:
                raise StoreError(422, "invalid_media", "upload is empty")
            if size > maximum:
                raise StoreError(
                    413,
                    "upload_too_large",
                    f"{kind} exceeds {maximum} bytes",
                )
            suffix = Path(safe_name).suffix
            if kind == "image":
                self._validate_image(staging.path, suffix)
            else:
                self._validate_video(staging.path, suffix)
            with self._lock:
                if self._active_uploads.get(staging.path) is not staging:
                    raise ValueError("media upload reservation is no longer active")
                final = self.directory / safe_name
                stem, suffix = os.path.splitext(safe_name)
                attempt = 2
                while final.exists() or final.is_symlink():
                    final = self.directory / f"{stem}-{attempt}{suffix}"
                    attempt += 1
                self._ledger.prepare_rename(staging._ownership, source, final)
                rename_noreplace(source, final)
                self._ledger.finish_rename(staging._ownership, final)
                self._active_uploads.pop(source, None)
                self._active_uploads[final] = staging
                staging.path = final
                # Atomic rename preserves the staging inode's mode, but the
                # final descriptor is explicitly re-secured as part of commit.
                _secure_existing(final, _PRIVATE_FILE_MODE, directory=False)
                self._release_upload_locked(staging, remove=False)
            kind_checked = self.kind_of(final)
            assert kind_checked == kind
            saved = StoredMedia(name=final.name, kind=kind, size=size, path=str(final))
        except BaseException as exc:
            primary = exc
            with self._lock:
                try:
                    current = self._ledger.reconcile(staging._ownership)
                except OSError:
                    current = ()
                original = source
                # ``prepare_rename`` retains both names.  Prefer the original
                # hidden name when a post-rename hardening check fails.
                if final is not None and final in current and original is not None:
                    try:
                        self._ledger.prepare_rename(staging._ownership, final, original)
                        rename_noreplace(final, original)
                        self._ledger.finish_rename(staging._ownership, original)
                        self._active_uploads.pop(final, None)
                        self._active_uploads[original] = staging
                        staging.path = original
                    except OSError:
                        try:
                            self._ledger.reconcile(staging._ownership)
                        except OSError:
                            pass
                self._detach_failed_upload_locked(staging)
        if primary is not None:
            if isinstance(primary, StoreError):
                raise primary
            if isinstance(primary, OSError):
                raise StoreError(
                    507,
                    "insufficient_storage",
                    "cannot publish media upload",
                ) from primary
            raise primary
        assert saved is not None
        return saved

    def remove(self, name: str, background_cfg) -> None:
        with self._lock:
            path = self.stored_path(name)
            # Like custback's upload store, only the path the current mode
            # displays counts as active; stale paths from other modes don't
            # block deletion.
            if background_cfg.mode == "image":
                active = background_cfg.image_path
            elif background_cfg.mode == "video":
                active = background_cfg.video_path
            else:
                active = ""
            if active and Path(active).expanduser().resolve(
                strict=False
            ) == path.resolve(strict=False):
                raise StoreError(
                    409,
                    "media_in_use",
                    f"{name!r} is the active avatar background; switch first",
                )
            path.unlink()
