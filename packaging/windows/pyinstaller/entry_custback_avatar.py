"""Frozen entry point for the Windows avatar service (WIN-6.4).

The avatar renderer is the product's second process (WIN-5.5): the shell
launches ``custback-avatar.exe`` from the same onedir payload as the engine
and supervises it only when installed.  This shim mirrors ``entry_custback.py``
for :func:`custback.avatar.__main__.main` — same freeze_support rationale, the
vision driver (MediaPipe) spawns helper processes under some delegates.
"""

import multiprocessing
import sys

from custback.avatar.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
