"""Windows/NTFS backend for the filesystem-security seam.

Phase 2 (WIN-2.x) replaces the fail-closed Phase 1 stubs with real Win32/NTFS
implementations that preserve the POSIX security and crash-recovery contract:

* WIN-2.1 exclusive, bounded-wait file locking via ``LockFileEx``;
* WIN-2.2 atomic no-replace publication via ``MoveFileEx``/``CreateHardLink``;
* WIN-2.3 owner-only, inheritance-stripped DACLs in place of ``chmod 0600/0700``;
* WIN-2.4 reparse-point / symlink rejection on open
  (``FILE_FLAG_OPEN_REPARSE_POINT`` + attribute check);
* WIN-2.5 stable file identity from the volume serial + NTFS file index;
* WIN-2.6 owner-SID comparison in place of ``st_uid``/``geteuid``;
* WIN-2.7 a directory-durability equivalent (``FlushFileBuffers`` with an
  NTFS-journal-ordered fallback).

This module is imported only when ``sys.platform == "win32"`` (see
:mod:`custback._platform`), so importing ``pywin32`` at module scope is safe;
POSIX hosts never load it.  ``pywin32`` is therefore a hard Windows requirement,
declared as the ``custback[windows]`` extra and bundled by the frozen installer
(D5).  The matching :mod:`custback._platform.posix` bodies are the byte-for-byte
behavior these replace.
"""

from __future__ import annotations

import errno
import msvcrt as _msvcrt
import os
import stat
from pathlib import Path
from typing import Any

import ntsecuritycon as _ntsecuritycon  # pyright: ignore[reportMissingModuleSource]
import pywintypes as _pywintypes  # pyright: ignore[reportMissingModuleSource]
import win32api as _win32api  # pyright: ignore[reportMissingModuleSource]
import win32con as _win32con  # pyright: ignore[reportMissingModuleSource]
import win32file as _win32file  # pyright: ignore[reportMissingModuleSource]
import win32security as _win32security  # pyright: ignore[reportMissingModuleSource]
import winerror as _winerror  # pyright: ignore[reportMissingModuleSource]

# pywin32 exposes a native, version-dependent surface with partial stubs. Keep
# the concrete imports visible to the Windows packager, but treat the FFI edge
# as opaque after import so platform-specific stub gaps do not leak into the
# portable filesystem-security contract.
msvcrt: Any = _msvcrt
ntsecuritycon: Any = _ntsecuritycon
pywintypes: Any = _pywintypes
win32api: Any = _win32api
win32con: Any = _win32con
win32file: Any = _win32file
win32security: Any = _win32security
winerror: Any = _winerror

# LockFileEx flags (winbase.h). Defined here rather than pulled from win32con so
# the lock contract does not depend on a particular pywin32 constant table.
_LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
_LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
_LOCK_WHOLE_FILE_LOW = 0xFFFFFFFF
_LOCK_WHOLE_FILE_HIGH = 0xFFFFFFFF

# MoveFileEx flags (winbase.h): fail if the destination exists (no
# MOVEFILE_REPLACE_EXISTING) and flush the rename before returning.
_MOVEFILE_WRITE_THROUGH = 0x00000008

_SHARE_ALL = (
    win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE | win32con.FILE_SHARE_DELETE
)
# Every security-relevant handle also needs to read and rewrite its own DACL so
# set_private_mode/owner_matches can operate on it after open.
_SECURITY_ACCESS = win32con.READ_CONTROL | ntsecuritycon.WRITE_DAC

_current_user_sid_cache: Any | None = None


def _current_user_sid() -> Any:
    """Return (and cache) the current process user's SID."""

    global _current_user_sid_cache
    if _current_user_sid_cache is None:
        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(), win32con.TOKEN_QUERY
        )
        try:
            sid, _attributes = win32security.GetTokenInformation(
                token, ntsecuritycon.TokenUser
            )
        finally:
            win32api.CloseHandle(token)
        _current_user_sid_cache = sid
    return _current_user_sid_cache


def _oserror(exc: Any, path: os.PathLike[str] | str | None = None) -> OSError:
    """Translate a Win32 error into an ``OSError`` with a mapped ``errno``.

    Passing ``winerror`` as the fourth argument lets CPython map it onto the
    closest POSIX ``errno`` so existing ``errno``-based handling still works.
    """

    filename = None if path is None else str(path)
    return OSError(0, exc.strerror, filename, exc.winerror)


# --- WIN-2.1: exclusive advisory locking -------------------------------------


def lock_exclusive(fd: int, *, blocking: bool = False) -> None:
    handle = msvcrt.get_osfhandle(fd)
    flags = _LOCKFILE_EXCLUSIVE_LOCK
    if not blocking:
        flags |= _LOCKFILE_FAIL_IMMEDIATELY
    overlapped = pywintypes.OVERLAPPED()
    try:
        win32file.LockFileEx(
            handle, flags, _LOCK_WHOLE_FILE_LOW, _LOCK_WHOLE_FILE_HIGH, overlapped
        )
    except pywintypes.error as exc:
        if not blocking and exc.winerror in (
            winerror.ERROR_LOCK_VIOLATION,
            winerror.ERROR_IO_PENDING,
        ):
            # Match the POSIX non-blocking contract the pollers depend on.
            raise BlockingIOError(errno.EAGAIN, exc.strerror) from exc
        raise _oserror(exc) from exc


def unlock(fd: int) -> None:
    handle = msvcrt.get_osfhandle(fd)
    overlapped = pywintypes.OVERLAPPED()
    try:
        win32file.UnlockFileEx(
            handle, _LOCK_WHOLE_FILE_LOW, _LOCK_WHOLE_FILE_HIGH, overlapped
        )
    except pywintypes.error as exc:
        raise _oserror(exc) from exc


# --- WIN-2.3: owner-only DACL as the private-mode equivalent -----------------


def set_private_mode(fd: int, mode: int) -> None:
    """Apply an owner-only, inheritance-stripped DACL to the open handle.

    POSIX modes ``0600``/``0700`` become "current-user SID: full control, no
    other ACEs".  ``PROTECTED_DACL_SECURITY_INFORMATION`` strips any inherited
    ACEs so no ancestor grant survives (WIN-2.3).  Directories additionally
    carry inheritable ACEs so children are private by default.
    """

    handle = msvcrt.get_osfhandle(fd)
    is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
    user_sid = _current_user_sid()
    ace_flags = 0
    if is_directory:
        ace_flags = (
            ntsecuritycon.OBJECT_INHERIT_ACE | ntsecuritycon.CONTAINER_INHERIT_ACE
        )
    dacl = win32security.ACL()
    dacl.AddAccessAllowedAceEx(
        win32security.ACL_REVISION, ace_flags, ntsecuritycon.FILE_ALL_ACCESS, user_sid
    )
    try:
        win32security.SetSecurityInfo(
            handle,
            win32security.SE_FILE_OBJECT,
            win32security.DACL_SECURITY_INFORMATION
            | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            dacl,
            None,
        )
    except pywintypes.error as exc:
        raise _oserror(exc) from exc


def chmod_private(path: Path, mode: int) -> None:
    """No-op on Windows.

    The POSIX ``chmod``-by-name step repairs an inode a restrictive umask left
    unopenable.  Windows has no umask; the authoritative owner-only DACL is
    applied to the bound descriptor by :func:`set_private_mode`, and files we
    create inherit their private parent's DACL.
    """

    return None


# --- WIN-2.4: no-follow / reparse-rejecting open -----------------------------


def _translate_open_flags(flags: int) -> tuple[int, int, bool]:
    """Map POSIX ``os.open`` flags to (desiredAccess, creationDisposition, append)."""

    # ``O_ACCMODE`` isn't present in Windows typeshed even though CPython may
    # provide it there; 0x3 is the standard POSIX access-mode mask.
    accmode = flags & getattr(os, "O_ACCMODE", 0x3)
    if accmode == os.O_WRONLY:
        access = win32con.GENERIC_WRITE
    elif accmode == os.O_RDWR:
        access = win32con.GENERIC_READ | win32con.GENERIC_WRITE
    else:
        access = win32con.GENERIC_READ
    appended = bool(flags & os.O_APPEND)
    if appended:
        # Append writes must target EOF; FILE_APPEND_DATA plus the CRT O_APPEND
        # flag on the wrapped descriptor give os.write append semantics.
        access = (access & ~win32con.GENERIC_WRITE) | ntsecuritycon.FILE_APPEND_DATA
        if accmode == os.O_RDWR:
            access |= win32con.GENERIC_READ
    access |= _SECURITY_ACCESS

    creating = bool(flags & os.O_CREAT)
    exclusive = bool(flags & os.O_EXCL)
    truncating = bool(flags & os.O_TRUNC)
    if creating and exclusive:
        disposition = win32con.CREATE_NEW
    elif creating and truncating:
        disposition = win32con.CREATE_ALWAYS
    elif creating:
        disposition = win32con.OPEN_ALWAYS
    elif truncating:
        disposition = win32con.TRUNCATE_EXISTING
    else:
        disposition = win32con.OPEN_EXISTING
    return access, disposition, appended


def _private_file_security() -> Any:
    """Create files for this user, including under an elevated default owner."""
    user_sid = _current_user_sid()
    dacl = win32security.ACL()
    dacl.AddAccessAllowedAceEx(
        win32security.ACL_REVISION, 0, ntsecuritycon.FILE_ALL_ACCESS, user_sid
    )
    attributes = win32security.SECURITY_ATTRIBUTES()
    descriptor = attributes.SECURITY_DESCRIPTOR
    descriptor.SetSecurityDescriptorOwner(user_sid, False)
    descriptor.SetSecurityDescriptorDacl(True, dacl, False)
    descriptor.SetSecurityDescriptorControl(
        win32security.SE_DACL_PROTECTED, win32security.SE_DACL_PROTECTED
    )
    attributes.bInheritHandle = False
    return attributes


def open_nofollow(
    path: os.PathLike[str] | str,
    flags: int,
    mode: int = 0o777,
    *,
    directory: bool = False,
) -> int:
    """Open ``path`` without following a final reparse point.

    ``FILE_FLAG_OPEN_REPARSE_POINT`` opens the reparse point itself rather than
    its target; the attribute check then refuses it, giving ``O_NOFOLLOW``
    semantics for the final component (WIN-2.4).  The Win32 handle is wrapped as
    a CRT descriptor so callers keep using ``os.fstat``/``os.fdopen``/``os.read``
    unchanged.
    """

    access, disposition, appended = _translate_open_flags(flags)
    attributes = win32file.FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        attributes |= win32file.FILE_FLAG_BACKUP_SEMANTICS
    security = _private_file_security() if flags & os.O_CREAT else None
    try:
        handle = win32file.CreateFile(
            str(path), access, _SHARE_ALL, security, disposition, attributes, None
        )
    except pywintypes.error as exc:
        if exc.winerror in (winerror.ERROR_ALREADY_EXISTS, winerror.ERROR_FILE_EXISTS):
            raise FileExistsError(errno.EEXIST, exc.strerror, str(path)) from exc
        raise _oserror(exc, path) from exc
    try:
        file_attributes = win32file.GetFileInformationByHandle(handle)[0]
        if file_attributes & win32con.FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError(errno.ELOOP, "refusing to open a reparse point", str(path))
        is_directory = bool(file_attributes & win32con.FILE_ATTRIBUTE_DIRECTORY)
        if directory and not is_directory:
            raise NotADirectoryError(errno.ENOTDIR, "not a directory", str(path))
        if not directory and is_directory:
            raise IsADirectoryError(errno.EISDIR, "is a directory", str(path))
    except pywintypes.error as exc:
        handle.Close()
        raise _oserror(exc, path) from exc
    except BaseException:
        handle.Close()
        raise
    raw = handle.Detach()
    descriptor_flags = os.O_APPEND if appended else 0
    try:
        return msvcrt.open_osfhandle(raw, descriptor_flags)
    except BaseException:
        win32file.CloseHandle(raw)
        raise


def is_reparse(path: os.PathLike[str] | str) -> bool:
    try:
        attributes = win32file.GetFileAttributes(str(path))
    except pywintypes.error as exc:
        if exc.winerror in (
            winerror.ERROR_FILE_NOT_FOUND,
            winerror.ERROR_PATH_NOT_FOUND,
        ):
            return False
        raise _oserror(exc, path) from exc
    if attributes in (-1, 0xFFFFFFFF):
        error = win32api.GetLastError()
        if error in (winerror.ERROR_FILE_NOT_FOUND, winerror.ERROR_PATH_NOT_FOUND):
            return False
        raise OSError(0, "cannot read path attributes", str(path), error)
    return bool(attributes & win32con.FILE_ATTRIBUTE_REPARSE_POINT)


# --- WIN-2.5: stable file identity -------------------------------------------


def file_identity(fd: int) -> tuple[int, int]:
    """Return ``(volume serial, file index)`` for an open descriptor.

    CPython's :func:`os.fstat` populates ``st_dev`` from the NTFS volume serial
    number and ``st_ino`` from the 128-bit ``FILE_ID_INFO`` file index, so the
    POSIX identity comparison is already correct on NTFS.
    """

    metadata = os.fstat(fd)
    return metadata.st_dev, metadata.st_ino


# --- WIN-2.6: owner-SID ownership verification -------------------------------


def owner_matches(fd: int) -> bool:
    """Authoritative post-open owner check by SID comparison."""

    handle = msvcrt.get_osfhandle(fd)
    try:
        descriptor = win32security.GetSecurityInfo(
            handle,
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION,
        )
    except pywintypes.error as exc:
        raise _oserror(exc) from exc
    owner_sid = descriptor.GetSecurityDescriptorOwner()
    if owner_sid is None:
        return False
    return bool(owner_sid == _current_user_sid())


def stat_owner_matches(metadata: os.stat_result) -> bool:
    """Advisory pre-open owner check.

    A ``stat_result`` carries no usable owner on Windows (``st_uid`` is 0), so
    this pre-filter defers to the authoritative post-open :func:`owner_matches`
    SID comparison that every flow performs on the bound descriptor.
    """

    return True


def is_private_to_owner(fd: int) -> bool:
    """Return whether the handle's DACL grants access only to its owner.

    The NTFS equivalent of ``mode & 0o077 == 0``: the owner must be the current
    user and every DACL ACE must name that same SID.  Files we create carry a
    single protected owner-only ACE (:func:`set_private_mode`), so a foreign or
    inherited grant fails this check.
    """

    handle = msvcrt.get_osfhandle(fd)
    try:
        descriptor = win32security.GetSecurityInfo(
            handle,
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION,
        )
    except pywintypes.error as exc:
        raise _oserror(exc) from exc
    user_sid = _current_user_sid()
    owner_sid = descriptor.GetSecurityDescriptorOwner()
    if owner_sid is None or owner_sid != user_sid:
        return False
    dacl = descriptor.GetSecurityDescriptorDacl()
    if dacl is None:
        # A NULL DACL grants everyone full access; that is never private.
        return False
    for index in range(dacl.GetAceCount()):
        ace_sid = dacl.GetAce(index)[-1]
        if ace_sid != user_sid:
            return False
    return True


# --- WIN-2.7: directory durability -------------------------------------------


def fsync_dir(path: os.PathLike[str] | str) -> None:
    try:
        handle = win32file.CreateFile(
            str(path),
            win32con.GENERIC_READ,
            _SHARE_ALL,
            None,
            win32con.OPEN_EXISTING,
            win32file.FILE_FLAG_BACKUP_SEMANTICS
            | win32file.FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
    except pywintypes.error as exc:
        raise _oserror(exc, path) from exc
    try:
        file_attributes = win32file.GetFileInformationByHandle(handle)[0]
        if file_attributes & win32con.FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError(errno.ELOOP, "refusing to sync a reparse point", str(path))
        if not file_attributes & win32con.FILE_ATTRIBUTE_DIRECTORY:
            raise NotADirectoryError(errno.ENOTDIR, "not a directory", str(path))
        try:
            win32file.FlushFileBuffers(handle)
        except pywintypes.error as exc:
            # A read-only directory handle cannot always be flushed. NTFS orders
            # metadata (create/rename/delete) through its journal, so treat the
            # documented cases as the journal-ordered no-op (WIN-2.7) rather than
            # failing durable publication.
            if exc.winerror not in (
                winerror.ERROR_INVALID_FUNCTION,
                winerror.ERROR_ACCESS_DENIED,
            ):
                raise _oserror(exc, path) from exc
    finally:
        handle.Close()


# --- WIN-2.2: atomic no-replace rename / hard link ---------------------------


def rename_noreplace(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    try:
        win32file.MoveFileEx(str(source), str(destination), _MOVEFILE_WRITE_THROUGH)
    except pywintypes.error as exc:
        if exc.winerror in (winerror.ERROR_ALREADY_EXISTS, winerror.ERROR_FILE_EXISTS):
            raise FileExistsError(errno.EEXIST, exc.strerror, str(destination)) from exc
        raise _oserror(exc, destination) from exc
    fsync_dir(source.parent)


def hardlink(source: Path, destination: Path) -> None:
    try:
        win32file.CreateHardLink(str(destination), str(source))
    except pywintypes.error as exc:
        if exc.winerror in (winerror.ERROR_ALREADY_EXISTS, winerror.ERROR_FILE_EXISTS):
            raise FileExistsError(errno.EEXIST, exc.strerror, str(destination)) from exc
        raise _oserror(exc, destination) from exc


def listdir_secure(fd: int, path: Path) -> list[str]:
    """List a managed directory.

    Windows has no ``fdopendir`` equivalent, so the listing is by path.  The
    caller has already verified the descriptor's identity and the enclosing tree
    is owner-only, which bounds the exposure of the reopen.
    """

    return os.listdir(path)
