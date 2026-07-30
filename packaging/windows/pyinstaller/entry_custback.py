"""Frozen entry point for the Windows custback engine (WIN-5.1).

PyInstaller freezes a *script*, not a module target, so this thin shim calls the
same :func:`custback.__main__.main` that the ``custback`` console script uses.
Keeping the shim separate from ``src/`` means nothing in the shipped package
depends on the freezing tool.
"""

import multiprocessing
import sys

from custback.__main__ import main

_FROZEN_VIDEO_COLOR_SMOKE_ARG = "--frozen-video-color-smoke"


def _run_frozen_video_color_smoke() -> None:
    """Exercise PyAV's bundled FFmpeg conversion without files or network I/O."""

    import av
    import numpy as np

    from custback import video_decoder

    # Limited-range BT.709 neutral gray should normalize to full-range sRGB
    # neutral gray.  This reaches PyAV's Cython modules, bundled FFmpeg/swscale
    # DLLs, metadata resolution, and custback's final BGR contract entirely in
    # memory; accepting a path/URL here would give a packaging probe needless
    # I/O authority.
    frame = av.VideoFrame.from_ndarray(
        np.stack(
            (
                np.full((2, 2), 126, dtype=np.uint8),
                np.full((2, 2), 128, dtype=np.uint8),
                np.full((2, 2), 128, dtype=np.uint8),
            )
        ),
        format="yuv444p",
    )
    frame.colorspace = 1
    frame.color_range = 1
    frame.color_primaries = 1
    frame.color_trc = 13

    normalized, contract = video_decoder.normalize_video_frame(
        frame,
        frame,
        video_decoder.VideoColorOverrides(),
    )
    expected = np.full((2, 2, 3), 128, dtype=np.uint8)
    max_delta = int(
        np.max(np.abs(normalized.astype(np.int16) - expected.astype(np.int16)))
    )
    if (
        normalized.shape != (2, 2, 3)
        or normalized.dtype != np.uint8
        or not normalized.flags.c_contiguous
        or max_delta > 2
        or contract.declared_input != "bt709/limited/bt709/srgb"
        or contract.status != "tagged"
        or contract.output != "srgb-full-bgr"
        or contract.assumed_fields
        or contract.overridden_fields
    ):
        raise RuntimeError(
            "frozen tagged video normalization smoke failed: "
            f"shape={normalized.shape}, dtype={normalized.dtype}, "
            f"contiguous={normalized.flags.c_contiguous}, max_delta={max_delta}, "
            f"input={contract.declared_input}, status={contract.status}, "
            f"output={contract.output}"
        )


if __name__ == "__main__":
    # OpenCV/MediaPipe/ONNX Runtime may spawn worker processes; under a frozen
    # build each child re-executes this bootstrap, so freeze_support must run
    # before any such process is created or the app forks itself endlessly.
    multiprocessing.freeze_support()
    if sys.argv[1:] == [_FROZEN_VIDEO_COLOR_SMOKE_ARG]:
        _run_frozen_video_color_smoke()
        sys.exit(0)
    sys.exit(main())
