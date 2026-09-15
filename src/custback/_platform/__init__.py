"""Platform filesystem-security seam.

One OS-specific boundary protects tokens, uploads, model caches, and logs.
The POSIX and Windows backends implement the same contract defined in
:mod:`custback._platform.base`. Call sites use this package for ownership,
private permissions, locking, symlink/reparse-point rejection, and durability.
Unsupported security operations fail closed.
"""

from __future__ import annotations

import sys

from .base import PlatformSecurityUnsupported
from .paths import config_dir

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
    "config_dir",
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
