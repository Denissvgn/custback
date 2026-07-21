"""POSIX backend for the filesystem-security seam.

Every function here is the historical inline behavior, extracted verbatim so
that Linux and macOS semantics are byte-for-byte unchanged.  On a genuine POSIX
host ``geteuid`` and ``O_NOFOLLOW`` always exist, so these are not degraded
fallbacks -- they are the real guarantees.

Phase 2 (WIN-2.x) widened this seam past the original five primitives so that
the security-sensitive filesystem idioms scattered across the storage, token,
model-cache, log, and migration code have a single home per CC-4.  The POSIX
bodies below reproduce exactly the syscalls those call sites issued before the
seam widened; the matching :mod:`custback._platform.windows` bodies supply the
Win32/NTFS equivalents.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import stat
import sys
from pathlib import Path

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x4


# --- WIN-2.1: exclusive advisory locking -------------------------------------


def lock_exclusive(fd: int, *, blocking: bool = False) -> None:
    """Take an exclusive advisory lock on ``fd``.

    Non-blocking by default: raises :class:`BlockingIOError` if another holder
    owns the lock, matching the callers that poll with a timeout.
    """

    flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    fcntl.flock(fd, flags)


def unlock(fd: int) -> None:
    """Release an advisory lock previously taken with :func:`lock_exclusive`."""

    fcntl.flock(fd, fcntl.LOCK_UN)


# --- WIN-2.3: exact private mode on an open descriptor -----------------------


def set_private_mode(fd: int, mode: int) -> None:
    """Bind an exact private mode to an already-open descriptor."""

    os.fchmod(fd, mode)


def chmod_private(path: Path, mode: int) -> None:
    """Repair an inode's mode by name without following a final symlink.

    A restrictive umask can create a mode-000 inode whose owner cannot open it
    to bind the descriptor.  Repairing by name first (no-follow) lets the
    subsequent :func:`set_private_mode` bind the authoritative descriptor.
    """

    os.chmod(path, mode, follow_symlinks=False)


# --- WIN-2.4: no-follow / reparse-rejecting open -----------------------------


def open_nofollow(
    path: os.PathLike[str] | str,
    flags: int,
    mode: int = 0o777,
    *,
    directory: bool = False,
) -> int:
    """Open ``path`` refusing to follow a final symlink.

    ``flags`` carries the caller's access/creation intent (``O_RDONLY`` ...,
    ``O_CREAT``, ``O_EXCL``, ``O_CLOEXEC``, ``O_NONBLOCK``); this function adds
    ``O_NOFOLLOW`` and, when ``directory`` is set, ``O_DIRECTORY``.  The result
    is byte-for-byte the flag set the call sites composed inline before the seam
    owned it.
    """

    resolved = flags | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        resolved |= getattr(os, "O_DIRECTORY", 0)
    return os.open(path, resolved, mode)


def is_reparse(path: os.PathLike[str] | str) -> bool:
    """Return whether the final component is a symlink (the POSIX reparse)."""

    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


# --- WIN-2.5: stable file identity -------------------------------------------


def file_identity(fd: int) -> tuple[int, int]:
    """Return ``(device, inode)`` identity for an open descriptor.

    On POSIX this is ``(st_dev, st_ino)``; the Windows backend returns the same
    tuple sourced from the volume serial number and NTFS file index, which is
    what CPython already reports through :func:`os.fstat` there.
    """

    metadata = os.fstat(fd)
    return metadata.st_dev, metadata.st_ino


# --- WIN-2.6: ownership verification -----------------------------------------


def owner_matches(fd: int) -> bool:
    """Authoritative post-open owner check for an open descriptor."""

    return os.fstat(fd).st_uid == os.geteuid()


def stat_owner_matches(metadata: os.stat_result) -> bool:
    """Advisory pre-open owner check from a stat result.

    On POSIX this is the real ``st_uid == geteuid()`` comparison.  It is only a
    pre-filter: every flow additionally re-verifies ownership after open through
    :func:`owner_matches`, which is where the Windows SID comparison lives.
    """

    return metadata.st_uid == os.geteuid()


def is_private_to_owner(fd: int) -> bool:
    """Return whether an open descriptor is inaccessible to group/other.

    On POSIX this is the historical ``mode & 0o077 == 0`` privacy check; the
    Windows backend proves the equivalent by confirming the owner SID and an
    owner-only DACL.
    """

    return (stat.S_IMODE(os.fstat(fd).st_mode) & 0o077) == 0


# --- WIN-2.7: directory durability -------------------------------------------


def fsync_dir(path: os.PathLike[str] | str) -> None:
    """Durably order namespace changes below a no-follow directory."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# --- WIN-2.2: atomic no-replace rename / hard link ---------------------------


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without ever replacing an unmarked destination."""

    source = Path(source)
    destination = Path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    result: int | None = None
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), _RENAME_EXCL)
    if result is None:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace rename is unavailable on this platform",
            str(destination),
        )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    fsync_dir(source.parent)


def hardlink(source: Path, destination: Path) -> None:
    """Create a hard link without following a symlink at either path."""

    os.link(source, destination, follow_symlinks=False)


def listdir_secure(fd: int, path: Path) -> list[str]:
    """List a directory that was already opened and identity-checked.

    POSIX lists relative to the open descriptor so the walk cannot be diverted
    by a directory swap between the identity check and the read.
    """

    return os.listdir(fd)
