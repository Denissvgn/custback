from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest

import custback.backgrounds as backgrounds_mod
import custback.video_decoder as video_decoder_mod
from custback.backgrounds import VideoBackdrop
from custback.video_decoder import (
    MetadataVideoCapture,
    ResolvedVideoColor,
    VideoColorOverrides,
    VideoDecoderError,
    normalize_video_frame,
    resolve_video_color,
)


def _reference_srgb(height: int = 6, width: int = 8) -> np.ndarray:
    x = np.linspace(0.12, 0.82, width, dtype=np.float32)
    y = np.linspace(0.15, 0.75, height, dtype=np.float32)[:, None]
    red = np.broadcast_to(x, (height, width))
    green = np.broadcast_to(y, (height, width))
    blue = 0.18 + 0.35 * (1.0 - red) * green
    rgb = np.stack((red, green, blue), axis=2)
    return np.rint(rgb[..., ::-1] * 255.0).astype(np.uint8)


def _srgb_bgr_to_yuv444(
    bgr: np.ndarray,
    *,
    matrix: str,
    range_name: str,
) -> np.ndarray:
    rgb = bgr[..., ::-1].astype(np.float64) / 255.0
    kr, kb = (0.299, 0.114) if matrix == "bt601" else (0.2126, 0.0722)
    kg = 1.0 - kr - kb
    y = kr * rgb[..., 0] + kg * rgb[..., 1] + kb * rgb[..., 2]
    cb = (rgb[..., 2] - y) / (2.0 * (1.0 - kb))
    cr = (rgb[..., 0] - y) / (2.0 * (1.0 - kr))
    if range_name == "limited":
        planes = np.stack((16.0 + 219.0 * y, 128.0 + 224.0 * cb, 128.0 + 224.0 * cr))
    else:
        planes = np.stack((255.0 * y, 127.5 + 255.0 * cb, 127.5 + 255.0 * cr))
    return np.rint(np.clip(planes, 0.0, 255.0)).astype(np.uint8)


def _write_tagged_ffv1(
    path: Path,
    frames: list[np.ndarray],
    *,
    matrix: str = "bt709",
    range_name: str = "limited",
    primaries: int = 1,
    transfer: int = 13,
    pts: list[int] | None = None,
    rate: int = 2,
) -> None:
    matrix_tag = 5 if matrix == "bt601" else 1
    range_tag = 1 if range_name == "limited" else 2
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv444p"
        stream.codec_context.colorspace = matrix_tag
        stream.codec_context.color_range = range_tag
        stream.codec_context.color_primaries = primaries
        stream.codec_context.color_trc = transfer
        for index, bgr in enumerate(frames):
            yuv = _srgb_bgr_to_yuv444(
                bgr,
                matrix=matrix,
                range_name=range_name,
            )
            frame = av.VideoFrame.from_ndarray(yuv, format="yuv444p")
            frame.pts = pts[index] if pts is not None else index
            frame.colorspace = matrix_tag
            frame.color_range = range_tag
            frame.color_primaries = primaries
            frame.color_trc = transfer
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_rotated_png_mov(
    path: Path,
    frame: np.ndarray,
    *,
    display_rotation_ccw: int,
    hflip: bool = False,
) -> None:
    height, width = frame.shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("png", rate=1)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "rgb24"
        stream.set_display_rotation(display_rotation_ccw, hflip=hflip)
        stream.codec_context.color_range = 2
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 13
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.color_range = 2
        video_frame.color_primaries = 1
        video_frame.color_trc = 13
        for packet in stream.encode(video_frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("matrix", ("bt601", "bt709"))
@pytest.mark.parametrize("range_name", ("limited", "full"))
def test_tagged_bt601_bt709_limited_full_fixtures_normalize_to_same_srgb(
    tmp_path: Path,
    matrix: str,
    range_name: str,
) -> None:
    reference = _reference_srgb()
    path = tmp_path / f"{matrix}-{range_name}.mkv"
    _write_tagged_ffv1(
        path,
        [reference],
        matrix=matrix,
        range_name=range_name,
    )

    capture = MetadataVideoCapture(str(path))
    try:
        ok, decoded = capture.read()
        contract = capture.color_contract
    finally:
        capture.release()

    assert ok and decoded is not None
    assert contract is not None
    assert contract.matrix == matrix
    assert contract.range == range_name
    assert contract.primaries == "bt709"
    assert contract.transfer == "srgb"
    assert contract.status == "tagged"
    assert contract.assumed_fields == ()
    assert contract.overridden_fields == ()
    assert contract.output == "srgb-full-bgr"
    # Lossless FFV1 preserves YUV samples; the tolerance is only the expected
    # 8-bit RGB<->YUV matrix quantization.
    assert np.max(np.abs(decoded.astype(np.int16) - reference.astype(np.int16))) <= 3


def test_unsupported_tagged_hdr_transfer_is_rejected_during_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pq.mkv"
    _write_tagged_ffv1(path, [_reference_srgb()], transfer=16)

    with pytest.raises(
        VideoDecoderError,
        match=r"unsupported tagged video transfer declaration \(16\)",
    ):
        MetadataVideoCapture(str(path))


def test_operator_override_is_local_only_and_observable(tmp_path: Path) -> None:
    path = tmp_path / "known-asset.mkv"
    reference = _reference_srgb()
    _write_tagged_ffv1(path, [reference])
    capture = MetadataVideoCapture(
        str(path),
        overrides=VideoColorOverrides(matrix="bt601", range="full"),
    )
    try:
        ok, _decoded = capture.read()
        contract = capture.color_contract
    finally:
        capture.release()

    assert ok
    assert contract is not None
    assert contract.status == "operator-override"
    assert contract.overridden_fields == ("matrix", "range")

    with pytest.raises(
        VideoDecoderError,
        match="requires a local operator-owned file",
    ):
        MetadataVideoCapture(
            "https://example.invalid/video.mkv",
            overrides=VideoColorOverrides(matrix="bt709"),
        )


@pytest.mark.parametrize(
    "source",
    (
        "concat:https://example.invalid/a|b",
        "//server/share/video.mkv",
        r"\\server\share\video.mkv",
    ),
)
def test_protocol_or_unc_source_is_rejected_before_filesystem_or_av_io(
    monkeypatch,
    source: str,
) -> None:
    monkeypatch.setattr(
        video_decoder_mod.Path,
        "is_file",
        lambda *_args, **_kwargs: pytest.fail(
            "non-local rejection must happen before filesystem I/O"
        ),
    )
    monkeypatch.setattr(
        av,
        "open",
        lambda *_args, **_kwargs: pytest.fail(
            "protocol rejection must happen before decoder I/O"
        ),
    )

    with pytest.raises(
        VideoDecoderError,
        match="requires a local operator-owned file",
    ):
        MetadataVideoCapture(source)


def test_local_manifest_cannot_open_nested_http_protocol(tmp_path: Path) -> None:
    manifest = tmp_path / "external-segment.m3u8"
    manifest.write_text(
        "\n".join(
            (
                "#EXTM3U",
                "#EXT-X-VERSION:3",
                "#EXT-X-TARGETDURATION:1",
                "#EXT-X-MEDIA-SEQUENCE:0",
                "#EXTINF:1.000000,",
                "http://127.0.0.1:9/segment.ts",
                "#EXT-X-ENDLIST",
                "",
            )
        ),
        encoding="ascii",
    )
    av_logging = cast(Any, av).logging
    previous_level = av_logging.get_level()
    av_logging.set_level(av_logging.DEBUG)
    try:
        with av_logging.Capture(local=True) as logs:
            with pytest.raises(FileNotFoundError, match="cannot open"):
                MetadataVideoCapture(str(manifest))
    finally:
        av_logging.set_level(previous_level)

    # FFmpeg 7/8 word and categorize this diagnostic slightly differently.
    # The stable contract is that the nested HTTP protocol is rejected by the
    # whitelist and no TCP/TLS transport is entered.
    messages = "\n".join(message.lower() for _level, _component, message in logs)
    assert "http" in messages
    assert "whitelist" in messages
    assert all(
        component.lower() not in {"tcp", "tls"} for _level, component, _message in logs
    )


class _OpaqueColorOwner:
    colorspace = 2
    color_range = 0
    color_primaries = 2
    color_trc = 2


class _ColorOwner:
    def __init__(
        self,
        *,
        colorspace: int,
        color_range: int,
        color_primaries: int,
        color_trc: int,
    ) -> None:
        self.colorspace = colorspace
        self.color_range = color_range
        self.color_primaries = color_primaries
        self.color_trc = color_trc


def test_color_precedence_is_override_then_frame_then_codec_per_field() -> None:
    frame = _ColorOwner(
        colorspace=1,
        color_range=0,
        color_primaries=5,
        color_trc=0,
    )
    codec = _ColorOwner(
        colorspace=5,
        color_range=1,
        color_primaries=1,
        color_trc=1,
    )

    tagged = resolve_video_color(frame, codec, VideoColorOverrides())
    assert tagged.declared_input == "bt709/limited/bt470bg/bt709"
    assert tagged.status == "tagged"
    assert tagged.assumed_fields == ()

    overridden = resolve_video_color(
        frame,
        codec,
        VideoColorOverrides(
            matrix="bt601",
            range="full",
            primaries="smpte170m",
            transfer="srgb",
        ),
    )
    assert overridden.declared_input == "bt601/full/smpte170m/srgb"
    assert overridden.status == "operator-override"
    assert overridden.overridden_fields == (
        "matrix",
        "range",
        "primaries",
        "transfer",
    )


def test_untagged_resolution_is_explicit_legacy_assumption_not_histogram() -> None:
    dark = np.full((4, 5, 3), 16, dtype=np.uint8)
    bright = np.full((4, 5, 3), 235, dtype=np.uint8)

    first = resolve_video_color(
        _OpaqueColorOwner(),
        _OpaqueColorOwner(),
        VideoColorOverrides(),
    )
    second = resolve_video_color(
        _OpaqueColorOwner(),
        _OpaqueColorOwner(),
        VideoColorOverrides(),
    )

    # Pixel arrays are intentionally never inputs to resolution; values that
    # resemble studio range cannot trigger expansion.
    assert dark.min() == 16 and bright.max() == 235
    assert first == second
    assert first.status == "legacy-assumption"
    assert first.declared_input == "bt709/full/bt709/srgb"
    assert first.assumed_fields == (
        "matrix",
        "range",
        "primaries",
        "transfer",
    )


def test_bt709_transfer_is_converted_to_srgb_not_only_retagged() -> None:
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
    frame.color_trc = 1
    output, contract = normalize_video_frame(
        frame,
        _OpaqueColorOwner(),
        VideoColorOverrides(),
    )

    assert contract.transfer == "bt709"
    # Limited-range Y=126 converts near encoded 0.5. BT.709 EOTF followed by
    # sRGB OETF is visibly brighter than a metadata-only retag.
    assert 130 < int(output[0, 0, 0]) < 145
    assert np.ptp(output[0, 0]) <= 1


def test_bt470bg_primaries_are_converted_in_linear_light() -> None:
    green_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    green_bgr[..., 1] = 255
    yuv = _srgb_bgr_to_yuv444(
        green_bgr,
        matrix="bt601",
        range_name="full",
    )
    frame = av.VideoFrame.from_ndarray(yuv, format="yuv444p")
    frame.colorspace = 5
    frame.color_range = 2
    frame.color_primaries = 5
    frame.color_trc = 13

    output, contract = normalize_video_frame(
        frame,
        _OpaqueColorOwner(),
        VideoColorOverrides(),
    )

    assert contract.primaries == "bt470bg"
    # EBU green has a small positive linear-sRGB blue component. Conversion
    # therefore produces visible blue that a metadata-only retag would omit.
    assert 15 <= int(output[0, 0, 0]) <= 40
    assert int(output[0, 0, 1]) >= 250
    assert int(output[0, 0, 2]) <= 4


def test_metadata_decoder_preserves_clock_reuse_loop_and_vfr_timing(
    tmp_path: Path,
) -> None:
    frames = [np.full((6, 8, 3), value, dtype=np.uint8) for value in (32, 96, 176)]
    path = tmp_path / "vfr.mkv"
    # At rate=10, these PTS represent 0.0s, 0.5s, and 1.5s.
    _write_tagged_ffv1(path, frames, pts=[0, 5, 15], rate=10)
    now = [0.0]
    backdrop = VideoBackdrop(str(path), clock=lambda: now[0])
    try:
        assert int(backdrop.frame(8, 6).mean()) == pytest.approx(32, abs=2)
        now[0] = 0.25
        assert int(backdrop.frame(8, 6).mean()) == pytest.approx(32, abs=2)
        now[0] = 0.50
        assert int(backdrop.frame(8, 6).mean()) == pytest.approx(96, abs=2)
        now[0] = 1.50
        assert int(backdrop.frame(8, 6).mean()) == pytest.approx(176, abs=2)
        # EOF learning and the last reliable VFR step establish a 2.5-second
        # loop. The real PyAV seek path must return to frame zero at phase zero.
        now[0] = 2.50
        assert int(backdrop.frame(8, 6).mean()) == pytest.approx(32, abs=2)
        stats = backdrop.stats_dict()
    finally:
        backdrop.close()

    assert stats["background_video_timing_mode"] == "container"
    assert cast(int, stats["background_video_frames_reused"]) >= 1
    assert stats["background_video_decoder_backend"] == "PYAV_FFMPEG"
    assert stats["background_video_color_status"] == "tagged"
    assert stats["background_video_input_color"] == "bt709/limited/bt709/srgb"
    assert stats["background_video_output_color"] == "srgb-full-bgr"


def test_real_pyav_display_matrix_rotation_is_applied_once(
    tmp_path: Path,
) -> None:
    reference = _reference_srgb(height=6, width=8)
    path = tmp_path / "rotation-ccw-90.mov"
    _write_rotated_png_mov(
        path,
        reference,
        display_rotation_ccw=90,
    )

    backdrop = VideoBackdrop(str(path), clock=lambda: 0.0)
    try:
        output = backdrop.frame(6, 8)
        stats = backdrop.stats_dict()
    finally:
        backdrop.close()

    expected = np.ascontiguousarray(np.rot90(reference, 1))
    assert output.shape == expected.shape
    assert np.max(np.abs(output.astype(np.int16) - expected.astype(np.int16))) <= 3
    assert stats["background_video_decoder_backend"] == "PYAV_FFMPEG"
    assert stats["background_video_orientation_status"] == ("qualified-manual-metadata")
    # PyAV's +90 degrees counter-clockwise is translated to the clockwise
    # convention exposed by the cv2-compatible capture boundary.
    assert stats["background_video_metadata_rotation"] == 270
    assert stats["background_video_auto_rotation_disabled"] is True


def test_reflected_pyav_display_matrix_is_rejected_explicitly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reflected.mov"
    _write_rotated_png_mov(
        path,
        _reference_srgb(),
        display_rotation_ccw=0,
        hflip=True,
    )

    with pytest.raises(
        VideoDecoderError,
        match="mirrored video display matrices are not supported",
    ):
        MetadataVideoCapture(str(path))


def test_prefetch_does_not_publish_next_frames_color_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed-color-contract.mkv"
    frames = [np.full((6, 8, 3), value, dtype=np.uint8) for value in (48, 160)]
    _write_tagged_ffv1(path, frames, rate=2)
    capture = MetadataVideoCapture(str(path))
    first_contract = ResolvedVideoColor(
        matrix="bt709",
        range="limited",
        primaries="bt709",
        transfer="srgb",
        status="tagged",
        assumed_fields=(),
        overridden_fields=(),
    )
    second_contract = ResolvedVideoColor(
        matrix="bt601",
        range="full",
        primaries="smpte170m",
        transfer="bt709",
        status="operator-override",
        assumed_fields=(),
        overridden_fields=("matrix", "range", "primaries", "transfer"),
    )
    contracts = iter((first_contract, second_contract, first_contract))
    original_convert = capture._convert

    def convert_with_distinct_contract(frame: object) -> np.ndarray:
        output = original_convert(frame)
        capture.color_contract = next(contracts, first_contract)
        return output

    monkeypatch.setattr(capture, "_convert", convert_with_distinct_contract)
    monkeypatch.setattr(
        backgrounds_mod,
        "_open_video_capture",
        lambda *_args, **_kwargs: capture,
    )
    now = [0.0]
    backdrop = VideoBackdrop(str(path), clock=lambda: now[0])
    try:
        backdrop.frame(8, 6)
        # Construction has already prefetched frame two; status must still be
        # atomically coupled to the displayed first frame.
        first_stats = backdrop.stats_dict()
        assert capture.color_contract == second_contract
        assert first_stats["background_video_input_color"] == (
            first_contract.declared_input
        )
        assert first_stats["background_video_color_status"] == "tagged"

        now[0] = 0.5
        backdrop.frame(8, 6)
        second_stats = backdrop.stats_dict()
        assert second_stats["background_video_input_color"] == (
            second_contract.declared_input
        )
        assert second_stats["background_video_color_status"] == ("operator-override")
    finally:
        backdrop.close()


def test_later_frame_unsupported_tag_is_not_swallowed_as_eof(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata-change.mkv"
    reference = _reference_srgb()
    _write_tagged_ffv1(path, [reference, reference])
    capture = MetadataVideoCapture(str(path))
    try:
        ok, _first = capture.read()
        assert ok
        original_decode = capture._decode_next_raw
        later = cast(Any, original_decode())
        assert later is not None
        later.color_trc = 16
        monkeypatch_frame = iter((later,))
        capture._iterator = monkeypatch_frame
        with pytest.raises(
            VideoDecoderError,
            match=r"unsupported tagged video transfer declaration \(16\)",
        ):
            capture.read()
        with pytest.raises(
            VideoDecoderError,
            match=r"unsupported tagged video transfer declaration \(16\)",
        ):
            capture.read()
    finally:
        capture.release()


def test_decoder_exception_is_not_reinterpreted_as_loop_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decode-error.mkv"
    _write_tagged_ffv1(path, [_reference_srgb()])
    capture = MetadataVideoCapture(str(path))

    class BrokenIterator:
        def __next__(self) -> object:
            raise RuntimeError("codec failure")

    try:
        capture._prefetched = None
        capture._iterator = BrokenIterator()
        with pytest.raises(VideoDecoderError, match="decode failed"):
            capture.read()
        with pytest.raises(VideoDecoderError, match="decode failed"):
            capture.read()
    finally:
        capture.release()


def test_oversized_later_frame_quarantines_decoder(tmp_path: Path) -> None:
    path = tmp_path / "dimension-change.mkv"
    _write_tagged_ffv1(path, [_reference_srgb()])
    capture = MetadataVideoCapture(str(path), max_width=8, max_height=6)
    oversized = av.VideoFrame(10, 6, "yuv444p")
    oversized.colorspace = 1
    oversized.color_range = 1
    oversized.color_primaries = 1
    oversized.color_trc = 13
    try:
        with pytest.raises(
            VideoDecoderError,
            match="exceeds configured dimensions",
        ):
            capture._convert(oversized)
        with pytest.raises(
            VideoDecoderError,
            match="exceeds configured dimensions",
        ):
            capture.read()
    finally:
        capture.release()


@pytest.mark.parametrize("route", ("grab-skip", "seek-scan"))
@pytest.mark.parametrize("failure", ("unsupported-color", "oversized"))
def test_every_raw_frame_is_validated_before_grab_or_seek_discard(
    tmp_path: Path,
    route: str,
    failure: str,
) -> None:
    path = tmp_path / f"raw-validation-{route}-{failure}.mkv"
    _write_tagged_ffv1(path, [_reference_srgb()])
    capture = MetadataVideoCapture(str(path), max_width=8, max_height=6)
    bad = av.VideoFrame(10 if failure == "oversized" else 8, 6, "yuv444p")
    bad.colorspace = 1
    bad.color_range = 1
    bad.color_primaries = 1
    bad.color_trc = 16 if failure == "unsupported-color" else 13
    original_contract = capture.color_contract
    capture._prefetched = None
    capture._iterator = iter((bad,))
    if route == "seek-scan":
        capture._seek_target_index = 100

    expected = (
        "unsupported tagged video transfer declaration"
        if failure == "unsupported-color"
        else "exceeds configured dimensions"
    )
    try:
        with pytest.raises(VideoDecoderError, match=expected):
            if route == "grab-skip":
                capture.grab()
            else:
                capture.read()
        with pytest.raises(VideoDecoderError, match=expected):
            capture.read()
        assert capture.color_contract == original_contract
    finally:
        capture.release()


def test_metadata_decoder_seek_scan_decodes_at_most_300_frames(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seek.mkv"
    reference = _reference_srgb()
    _write_tagged_ffv1(path, [reference], rate=2)
    capture = MetadataVideoCapture(str(path))
    frame = av.VideoFrame(8, 6, "yuv444p")
    frame.colorspace = 1
    frame.color_range = 1
    frame.color_primaries = 1
    frame.color_trc = 13

    class CountingIterator:
        def __init__(self) -> None:
            self.calls = 0

        def __next__(self) -> object:
            self.calls += 1
            if self.calls > 400:
                raise StopIteration
            return frame

    iterator = CountingIterator()
    try:
        capture._prefetched = None
        capture._iterator = iterator
        capture._seek_target_index = 10_000
        ok, decoded = capture.read()
    finally:
        capture.release()

    assert not ok and decoded is None
    assert iterator.calls == 300
