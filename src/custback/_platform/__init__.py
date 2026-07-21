"""Platform filesystem-security seam.

A single OS-specific boundary for the primitives that protect tokens, uploads,
model caches, and logs.  POSIX keeps the historical behavior byte-for-byte; the
Windows backend supplies the Win32/NTFS equivalents added in Phase 2 (see
:mod:`custback._platform.base` for the contract and WINDOWS_IMPLEMENTATION_PLAN.md
CC-1 / CC-4).

Call sites import this package and use, for example,
``platform_fs.open_nofollow(path, flags)`` in place of composing
``os.open(path, flags | O_NOFOLLOW)`` inline, and
``platform_fs.owner_matches(fd)`` in place of ``st_uid == geteuid()``.  All
OS-specific filesystem security/identity/durability behavior lives here so the
storage, token, and migration code contains no ``sys.platform`` branches.
"""

from __future__ import annotations

import sys

from .base import PlatformSecurityUnsupported

if sys.platform == "win32":  # pragma: no cover - selected only on Windows
    from . import windows as _backend
else:
    from . import posix as _backend

# WIN-2.1 locking
lock_exclusive = _backend.lock_exclusive
unlock = _backend.unlock
# WIN-2.3 private mode / DACL
set_private_mode = _backend.set_private_mode
chmod_private = _backend.chmod_private
# WIN-2.4 no-follow / reparse rejection
open_nofollow = _backend.open_nofollow
is_reparse = _backend.is_reparse
# WIN-2.5 file identity
file_identity = _backend.file_identity
# WIN-2.6 ownership verification
owner_matches = _backend.owner_matches
stat_owner_matches = _backend.stat_owner_matches
is_private_to_owner = _backend.is_private_to_owner
# WIN-2.7 directory durability
fsync_dir = _backend.fsync_dir
# WIN-2.2 atomic no-replace publication
rename_noreplace = _backend.rename_noreplace
hardlink = _backend.hardlink
# Directory listing over an already-verified descriptor
listdir_secure = _backend.listdir_secure

__all__ = [
    "PlatformSecurityUnsupported",
    "lock_exclusive",
    "unlock",
    "set_private_mode",
    "chmod_private",
    "open_nofollow",
    "is_reparse",
    "file_identity",
    "owner_matches",
    "stat_owner_matches",
    "is_private_to_owner",
    "fsync_dir",
    "rename_noreplace",
    "hardlink",
    "listdir_secure",
]
