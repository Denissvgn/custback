"""PyInstaller analysis hook for the ``custback`` package (WIN-5.1).

Ensures the pieces of ``custback`` that are reached by dynamic dispatch rather
than a static ``import`` are pulled into the frozen build:

* the platform-security backends (``custback._platform.*``) — selected at import
  time by ``sys.platform`` (see ``custback/_platform/__init__.py``);
* the segmentation delegates loaded via ``importlib.import_module`` in
  ``custback.segmentation``;
* packaged data resources such as ``custback/avatar/avatar.yaml`` that are read
  through ``importlib.resources``.

Model *weights* (``*.onnx`` / ``*.tflite``) are deliberately excluded — they are
licensing-gated and downloaded on first run (WIN-0.4, D6).
"""

from PyInstaller.utils.hooks import (  # pyright: ignore[reportMissingModuleSource] - build-time-only dependency
    collect_data_files,
    collect_submodules,
)

hiddenimports = collect_submodules("custback")

datas = collect_data_files(
    "custback",
    includes=["**/*.yaml", "**/system-profile-catalog.json"],
    excludes=["**/*.onnx", "**/*.tflite"],
)
