"""Contract for the platform filesystem-security seam.

This package is the single boundary for OS-specific filesystem *security*
behavior (WINDOWS_IMPLEMENTATION_PLAN.md, WIN-1.1 / CC-4). Storage, token,
model-cache, logging, and migration code call these primitives instead of
touching ``os.fchmod``/``fcntl``/``geteuid``/``O_NOFOLLOW`` directly.

The invariant that makes this seam worth having is CC-1: on a platform without
a real implementation, a security primitive must **fail closed** -- it raises
:class:`PlatformSecurityUnsupported` rather than silently degrading to a
permissive no-op. Historically several call sites used
``getattr(os, "O_NOFOLLOW", 0)`` and ``getattr(os, "geteuid", lambda: ...)()``
fallbacks that, on a non-POSIX host, silently dropped symlink/reparse-point
protection and turned ownership checks into unconditional passes. That is more
dangerous than crashing, because it looks like it works. This seam replaces
those fallbacks so the POSIX path is byte-for-byte unchanged while any
unimplemented platform refuses to proceed.

Phase 2 (WIN-2.x) supplied the Windows backend's real Win32/NTFS
implementations (LockFileEx locking, owner-only DACLs, reparse-point rejection,
SID ownership, atomic no-replace rename, directory durability) behind the same
surface.  :class:`PlatformSecurityUnsupported` remains for any future primitive
a backend does not yet implement: it must be raised, never silently degraded to
a permissive no-op (CC-1).
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
