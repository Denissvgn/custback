"""Frozen entry point for the Windows custback engine (WIN-5.1).

PyInstaller freezes a *script*, not a module target, so this thin shim calls the
same :func:`custback.__main__.main` that the ``custback`` console script uses.
Keeping the shim separate from ``src/`` means nothing in the shipped package
depends on the freezing tool.
"""

import multiprocessing
import sys

from custback.__main__ import main

if __name__ == "__main__":
    # OpenCV/MediaPipe/ONNX Runtime may spawn worker processes; under a frozen
    # build each child re-executes this bootstrap, so freeze_support must run
    # before any such process is created or the app forks itself endlessly.
    multiprocessing.freeze_support()
    sys.exit(main())
