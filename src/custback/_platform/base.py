"""Contract for platform filesystem security.

Storage, tokens, model caches, logging, and migration use these primitives
instead of calling platform-specific permission and locking APIs directly.
An unsupported security operation raises :class:`PlatformSecurityUnsupported`;
it must never silently become a permissive no-op.

Windows implementations provide Win32/NTFS locking, private DACLs, SID ownership,
reparse-point rejection, atomic no-replace rename, and directory durability.
POSIX implementations provide the corresponding filesystem guarantees.
"""

from __future__ import annotations

import errno


class PlatformSecurityUnsupported(OSError):
    """A filesystem-security primitive has no safe implementation here.

    Subclasses :class:`OSError` (with ``errno.ENOSYS``) so it interoperates
    with the existing ``OSError``-based error handling around these primitives,
    but it must be allowed to propagate: callers must never catch it and
    continue as though the security guarantee held. See CC-1.
    """

    def __init__(self, operation: str) -> None:
        super().__init__(
            errno.ENOSYS,
            f"{operation} is not implemented on this platform; refusing to "
            "continue without the POSIX filesystem-security guarantee",
        )
        self.operation = operation
