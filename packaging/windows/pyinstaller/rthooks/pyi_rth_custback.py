"""PyInstaller runtime hook for the frozen custback engine (WIN-5.1).

Runs inside the frozen process before any custback code imports.  Its only job
is to make native DLLs that ship *inside* the onedir bundle discoverable by late
imports (ONNX Runtime providers, OpenCV, MediaPipe), and to advertise that the
process is frozen so runtime code can branch on it without guessing.

CUDA/cuDNN discovery for the accelerated path is handled later and defensively
by :func:`custback.acceleration.preload_acceleration_dlls`, which reads
``CUDA_PATH`` / ``PATH`` and is a non-fatal best effort; this hook only covers
the DLLs bundled beside the executable.
"""

import os
import sys

# Marker for runtime code and diagnostics ("am I a frozen build?").
os.environ.setdefault("CUSTBACK_FROZEN", "1")

_bundle = getattr(sys, "_MEIPASS", None)
if _bundle and sys.platform == "win32":
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            add_dll_directory(_bundle)
        except OSError:
            # A missing bundle directory should not abort startup; the import
            # that needs the DLL will raise a clear error of its own.
            pass
