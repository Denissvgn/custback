"""Platform filesystem-security seam tests (WIN-2.8, cross-platform contract).

The seam (:mod:`custback._platform`) is the single home for the OS-specific
filesystem security, identity, and durability primitives that protect tokens,
uploads, model caches, logs, and migration state.  These tests assert the
security *contract* against the seam, so they run on the POSIX backend here and
on the Win32/NTFS backend under the ``windows-latest`` CI job (WIN-1.8) without
change.  A few genuinely Windows-only behaviours (junction reparse rejection,
DACL inspection) are guarded with ``skipif``.

The storage/token/migration crash-recovery, concurrency, quota, and privacy
suites (``test_phase5_storage``, ``test_api_security``, ``test_phase6_migration``)
already exercise the seam end to end; this file adds the primitive-level
adversarial coverage the Phase 2 exit gate calls for.
"""

from __future__ import annotations

import os
import stat
import sys
from typing import Any

import pytest

from custback import _platform as platform_fs

WINDOWS = sys.platform == "win32"


# --- WIN-2.4: no-follow / reparse-point rejection ----------------------------


def _make_private_file(path, data=b"payload"):
    fd = platform_fs.open_nofollow(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        platform_fs.set_private_mode(fd, 0o600)
        os.write(fd, data)
    finally:
        os.close(fd)


def test_open_nofollow_opens_and_reads_a_regular_file(tmp_path):
    target = tmp_path / "regular"
    _make_private_file(target, b"secret")
    fd = platform_fs.open_nofollow(target, os.O_RDONLY)
    try:
        assert stat.S_ISREG(os.fstat(fd).st_mode)
        assert os.read(fd, 64) == b"secret"
    finally:
        os.close(fd)


def test_open_nofollow_rejects_a_final_symlink(tmp_path):
    target = tmp_path / "target"
    _make_private_file(target, b"secret")
    link = tmp_path / "link"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable on this host")
    # POSIX: O_NOFOLLOW -> ELOOP. Windows: FILE_FLAG_OPEN_REPARSE_POINT opens the
    # reparse point itself and the attribute check refuses it.
    with pytest.raises(OSError):
        platform_fs.open_nofollow(link, os.O_RDONLY)


def test_open_nofollow_excl_refuses_existing(tmp_path):
    target = tmp_path / "excl"
    _make_private_file(target)
    with pytest.raises(FileExistsError):
        platform_fs.open_nofollow(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)


def test_open_nofollow_directory_flag_opens_a_directory(tmp_path):
    fd = platform_fs.open_nofollow(tmp_path, os.O_RDONLY, directory=True)
    try:
        assert stat.S_ISDIR(os.fstat(fd).st_mode)
    finally:
        os.close(fd)


@pytest.mark.skipif(not WINDOWS, reason="NTFS junction reparse-point rejection")
def test_open_nofollow_rejects_a_junction(tmp_path):
    import subprocess

    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    # Directory junctions do not require elevation, unlike symlinks.
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("could not create an NTFS junction on this host")
    with pytest.raises(OSError):
        platform_fs.open_nofollow(junction, os.O_RDONLY, directory=True)


def test_is_reparse_distinguishes_symlinks(tmp_path):
    regular = tmp_path / "regular"
    _make_private_file(regular)
    assert platform_fs.is_reparse(regular) is False
    assert platform_fs.is_reparse(tmp_path / "missing") is False
    link = tmp_path / "link"
    try:
        os.symlink(regular, link)
    except (OSError, NotImplementedError):
        return
    assert platform_fs.is_reparse(link) is True


# --- WIN-2.6 / WIN-2.3: ownership and privacy --------------------------------


def test_owner_matches_true_for_a_file_we_created(tmp_path):
    target = tmp_path / "owned"
    _make_private_file(target)
    fd = platform_fs.open_nofollow(target, os.O_RDONLY)
    try:
        assert platform_fs.owner_matches(fd) is True
    finally:
        os.close(fd)


def test_stat_owner_matches_is_advisory_but_true_for_own_files(tmp_path):
    target = tmp_path / "owned"
    _make_private_file(target)
    # POSIX compares st_uid == geteuid(); Windows returns True as the advisory
    # pre-filter (the authoritative check is owner_matches on the descriptor).
    assert platform_fs.stat_owner_matches(target.lstat()) is True


def test_is_private_to_owner_true_after_set_private_mode(tmp_path):
    target = tmp_path / "private"
    _make_private_file(target)
    fd = platform_fs.open_nofollow(target, os.O_RDONLY)
    try:
        assert platform_fs.is_private_to_owner(fd) is True
    finally:
        os.close(fd)


@pytest.mark.skipif(WINDOWS, reason="POSIX mode bits; Windows uses DACL enumeration")
def test_is_private_to_owner_false_when_group_or_other_readable(tmp_path):
    target = tmp_path / "leaky"
    _make_private_file(target)
    os.chmod(target, 0o644)
    fd = platform_fs.open_nofollow(target, os.O_RDONLY)
    try:
        assert platform_fs.is_private_to_owner(fd) is False
    finally:
        os.close(fd)


@pytest.mark.skipif(
    not WINDOWS, reason="NTFS owner-only DACL is protected (no inheritance)"
)
def test_private_dacl_is_owner_only_and_protected(tmp_path):
    import win32security as _win32security  # pyright: ignore[reportMissingModuleSource]

    win32security: Any = _win32security

    target = tmp_path / "private"
    _make_private_file(target)
    descriptor = win32security.GetNamedSecurityInfo(
        str(target),
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION,
    )
    owner = descriptor.GetSecurityDescriptorOwner()
    dacl = descriptor.GetSecurityDescriptorDacl()
    assert dacl is not None and dacl.GetAceCount() >= 1
    for index in range(dacl.GetAceCount()):
        assert win32security.EqualSid(dacl.GetAce(index)[-1], owner)


# --- WIN-2.5: stable file identity -------------------------------------------


def test_file_identity_matches_stat_and_is_stable(tmp_path):
    target = tmp_path / "identity"
    _make_private_file(target)
    fd = platform_fs.open_nofollow(target, os.O_RDONLY)
    try:
        metadata = os.fstat(fd)
        assert platform_fs.file_identity(fd) == (metadata.st_dev, metadata.st_ino)
    finally:
        os.close(fd)
    # Identity survives a rename of the same inode (no-replace into a new name).
    moved = tmp_path / "identity-moved"
    before = target.lstat()
    platform_fs.rename_noreplace(target, moved)
    after = moved.lstat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


# --- WIN-2.2: atomic no-replace publication ----------------------------------


def test_rename_noreplace_moves_into_a_free_name(tmp_path):
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    _make_private_file(source, b"body")
    platform_fs.rename_noreplace(source, destination)
    assert not source.exists()
    assert destination.read_bytes() == b"body"


def test_rename_noreplace_refuses_an_existing_destination(tmp_path):
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    _make_private_file(source, b"new")
    _make_private_file(destination, b"original")
    with pytest.raises(FileExistsError):
        platform_fs.rename_noreplace(source, destination)
    # The pre-existing destination is never clobbered.
    assert destination.read_bytes() == b"original"
    assert source.read_bytes() == b"new"


def test_hardlink_publishes_without_replacing(tmp_path):
    source = tmp_path / "src"
    _make_private_file(source, b"body")
    link = tmp_path / "link"
    platform_fs.hardlink(source, link)
    assert link.read_bytes() == b"body"
    occupied = tmp_path / "occupied"
    _make_private_file(occupied, b"present")
    with pytest.raises(FileExistsError):
        platform_fs.hardlink(source, occupied)
    assert occupied.read_bytes() == b"present"


# --- WIN-2.1: exclusive, bounded-wait locking --------------------------------


def test_lock_exclusive_is_exclusive_and_releasable(tmp_path):
    path = tmp_path / "lock"
    first = platform_fs.open_nofollow(path, os.O_CREAT | os.O_RDWR, 0o600)
    second = platform_fs.open_nofollow(path, os.O_RDWR, 0o600)
    try:
        platform_fs.lock_exclusive(first)
        with pytest.raises(BlockingIOError):
            platform_fs.lock_exclusive(second)
        platform_fs.unlock(first)
        # Once released, the contender can take the lock.
        platform_fs.lock_exclusive(second)
        platform_fs.unlock(second)
    finally:
        os.close(first)
        os.close(second)


# --- WIN-2.7: directory durability -------------------------------------------


def test_fsync_dir_orders_a_directory(tmp_path):
    _make_private_file(tmp_path / "child")
    # Must not raise on a real directory (Windows tolerates the journal-ordered
    # no-op when a read-only directory handle cannot be flushed).
    platform_fs.fsync_dir(tmp_path)


def test_fsync_dir_refuses_a_regular_file(tmp_path):
    target = tmp_path / "regular"
    _make_private_file(target)
    with pytest.raises(OSError):
        platform_fs.fsync_dir(target)


# --- chmod_private / listdir_secure ------------------------------------------


def test_chmod_private_does_not_raise(tmp_path):
    target = tmp_path / "file"
    _make_private_file(target)
    # POSIX repairs the mode by name; Windows is a documented no-op.
    platform_fs.chmod_private(target, 0o600)


def test_listdir_secure_lists_children(tmp_path):
    _make_private_file(tmp_path / "a")
    _make_private_file(tmp_path / "b")
    fd = platform_fs.open_nofollow(tmp_path, os.O_RDONLY, directory=True)
    try:
        assert sorted(platform_fs.listdir_secure(fd, tmp_path)) == ["a", "b"]
    finally:
        os.close(fd)
