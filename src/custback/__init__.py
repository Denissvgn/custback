"""custback — virtual camera with background replacement.

Pipeline:  real camera -> segmentation -> compositor (backdrop) -> virtual camera
                                        \\-> API (MJPEG / WebSocket frame forwarding)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("custback")
except PackageNotFoundError:  # source-tree import without an installed distribution
    __version__ = "0.3.0"
