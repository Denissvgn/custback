"""Canonical output-sink and constrained native-camera geometry regressions."""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import custback.vcam as vcam
from custback import vcam_native
from custback.capture import SyntheticCapture
from custback.config import CameraConfig, OutputConfig
from custback.vcam import NullOutput, PyVirtualCamOutput, open_output

ROOT = Path(__file__).resolve().parents[1]
VCAM_DIR = ROOT / "packaging" / "windows" / "vcam"


def _frame(width: int = 8, height: int = 6, *, offset: int = 0) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.uint16)
    return np.stack(
        (
            (13 * xx + 3 * yy + offset) % 256,
            (5 * xx + 17 * yy + offset) % 256,
            (29 * xx + 7 * yy + offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _invalid_sink_frames(width: int = 8, height: int = 6):
    yield pytest.param(
        np.zeros((height, width, 3), dtype=np.float32),
        id="float-dtype",
    )
    yield pytest.param(
        np.zeros((height, width, 4), dtype=np.uint8),
        id="four-channels",
    )
    yield pytest.param(
        np.zeros((height, width), dtype=np.uint8),
        id="gray",
    )
    yield pytest.param(
        np.zeros((height + 1, width, 3), dtype=np.uint8),
        id="wrong-height",
    )
    yield pytest.param(
        np.zeros((height, width * 2, 3), dtype=np.uint8)[:, ::2],
        id="non-contiguous",
    )


@pytest.mark.parametrize("malformed", tuple(_invalid_sink_frames()))
def test_null_output_strictly_validates_every_sent_frame(malformed) -> None:
    output = NullOutput(width=8, height=6, fps=47)

    with pytest.raises(ValueError):
        output.send(malformed)

    assert output.frames_sent == 0
    assert (output.width, output.height, output.fps) == (8, 6, 47)


def test_null_output_accepts_only_exact_canvas_and_reports_effective_mode() -> None:
    output = open_output(OutputConfig(backend="null", fps=47), 8, 6)
    frame = _frame()

    output.send(frame)

    assert isinstance(output, NullOutput)
    assert output.frames_sent == 1
    assert (output.width, output.height, output.fps) == (8, 6, 47)


class _FakePyVirtualCamera:
    opened: list["_FakePyVirtualCamera"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.device = "fake-vcam"
        self.sent: list[np.ndarray] = []
        self.sleep_calls = 0
        self.closed = False
        type(self).opened.append(self)

    def send(self, frame: np.ndarray) -> None:
        self.sent.append(frame)

    def sleep_until_next_frame(self) -> None:
        self.sleep_calls += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_pyvirtualcam(monkeypatch):
    _FakePyVirtualCamera.opened.clear()
    module = SimpleNamespace(
        Camera=_FakePyVirtualCamera,
        PixelFormat=SimpleNamespace(BGR="BGR"),
    )
    monkeypatch.setitem(sys.modules, "pyvirtualcam", module)
    return module


def test_pyvirtualcam_reports_effective_mode_and_forwards_exact_bgr(
    fake_pyvirtualcam,
) -> None:
    output = PyVirtualCamOutput(
        OutputConfig(backend="pyvirtualcam", device="chosen", fps=47),
        8,
        6,
    )
    frame = _frame()

    output.send(frame)
    output.close()

    assert (output.width, output.height, output.fps) == (8, 6, 47)
    camera = _FakePyVirtualCamera.opened[-1]
    assert camera.kwargs == {
        "width": 8,
        "height": 6,
        "fps": 47,
        "fmt": "BGR",
        "device": "chosen",
    }
    assert camera.sent == [frame]
    assert camera.sleep_calls == 1
    assert camera.closed


@pytest.mark.parametrize("malformed", tuple(_invalid_sink_frames()))
def test_pyvirtualcam_rejects_malformed_frames_before_backend_send_or_sleep(
    fake_pyvirtualcam,
    malformed,
) -> None:
    output = PyVirtualCamOutput(OutputConfig(backend="pyvirtualcam"), 8, 6)
    camera = _FakePyVirtualCamera.opened[-1]

    with pytest.raises(ValueError):
        output.send(malformed)

    assert camera.sent == []
    assert camera.sleep_calls == 0


@pytest.mark.parametrize("malformed", tuple(_invalid_sink_frames()))
def test_frame_ring_writer_rejects_malformed_frames_without_advancing_counter(
    malformed,
) -> None:
    buffer = bytearray(vcam_native.ring_size(8, 6))
    writer = vcam_native.FrameRingWriter(buffer, 8, 6)
    before = vcam_native.unpack_header(buffer)

    with pytest.raises(ValueError):
        writer.publish(malformed)

    after = vcam_native.unpack_header(buffer)
    assert after["frame_counter"] == before["frame_counter"] == 0
    assert after["seq"] == before["seq"] == 0
    assert not after["flags"] & vcam_native.FLAG_ACTIVE


@pytest.mark.parametrize("malformed", tuple(_invalid_sink_frames()))
def test_native_output_rejects_malformed_frames_and_reports_effective_mode(
    malformed,
) -> None:
    buffer = bytearray(vcam_native.ring_size(8, 6))
    output = vcam_native.NativeVirtualCameraOutput(
        8,
        6,
        fps=47,
        buffer=buffer,
    )

    with pytest.raises(ValueError):
        output.send(malformed)

    assert output.frames_sent == 0
    assert (output.width, output.height, output.fps) == (8, 6, 47)
    assert vcam_native.unpack_header(buffer)["frame_counter"] == 0


def test_frame_ring_bgrx_stride_order_and_counter_are_exact() -> None:
    width, height = 3, 2
    buffer = bytearray(vcam_native.ring_size(width, height))
    writer = vcam_native.FrameRingWriter(buffer, width, height)
    first = _frame(width, height)

    writer.publish(first)
    first_header = vcam_native.unpack_header(buffer)
    first_bgrx = vcam_native.read_latest_frame(buffer)

    assert first_header["stride"] == width * vcam_native.BYTES_PER_PIXEL
    assert first_header["fourcc"] == b"BGRX"
    assert first_header["frame_counter"] == 1
    assert first_header["seq"] == 2
    assert first_bgrx is not None
    assert first_bgrx.flags.c_contiguous
    np.testing.assert_array_equal(first_bgrx[:, :, :3], first)
    np.testing.assert_array_equal(
        first_bgrx[:, :, 3],
        np.full((height, width), 255, dtype=np.uint8),
    )
    assert bytes(
        buffer[
            vcam_native.HEADER_SIZE : vcam_native.HEADER_SIZE
            + vcam_native.BYTES_PER_PIXEL
        ]
    ) == bytes((*first[0, 0], 255))
    second_row = vcam_native.HEADER_SIZE + first_header["stride"]
    assert bytes(buffer[second_row : second_row + 4]) == bytes((*first[1, 0], 255))

    second = _frame(width, height, offset=101)
    writer.publish(second)
    second_header = vcam_native.unpack_header(buffer)
    second_bgrx = vcam_native.read_latest_frame(buffer)

    assert second_header["frame_counter"] == 2
    assert second_header["seq"] == 4
    assert second_bgrx is not None
    np.testing.assert_array_equal(second_bgrx[:, :, :3], second)


def test_writer_precomputes_payload_before_entering_seqlock() -> None:
    source = inspect.getsource(vcam_native.FrameRingWriter.publish)
    payload_index = source.index("payload = bgrx.tobytes()")
    odd_index = source.index("self._seq += 1")
    mapping_copy_index = source.index(
        "self._buffer[HEADER_SIZE : HEADER_SIZE + len(payload)] = payload"
    )

    assert payload_index < odd_index < mapping_copy_index


def test_explicit_native_supported_modes_open_exact_ring(monkeypatch) -> None:
    probes: list[str] = []
    opened: list[tuple[int, int, int]] = []

    class _NativeMarker:
        def __init__(self, width: int, height: int, fps: int) -> None:
            self.width = width
            self.height = height
            self.fps = fps

        def send(self, _frame: np.ndarray) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        vcam_native,
        "require_native_camera_component",
        lambda: probes.append("component"),
    )

    def open_native(width: int, height: int, *, fps: int):
        opened.append((width, height, fps))
        return _NativeMarker(width, height, fps)

    monkeypatch.setattr(vcam_native, "NativeVirtualCameraOutput", open_native)

    for width, height, fps in sorted(vcam.SUPPORTED_NATIVE_MODES):
        output = open_output(OutputConfig(backend="native", fps=fps), width, height)
        assert (output.width, output.height, output.fps) == (width, height, fps)

    assert probes == ["component"] * len(vcam.SUPPORTED_NATIVE_MODES)
    assert opened == sorted(vcam.SUPPORTED_NATIVE_MODES)


@pytest.mark.parametrize(
    ("width", "height", "fps"),
    (
        (640, 480, 30),
        (1281, 721, 30),
        (1280, 720, 29),
        (1280, 720, 60),
        (1920, 1080, 31),
    ),
)
def test_explicit_native_unsupported_mode_fails_before_component_probe_or_ring_open(
    monkeypatch,
    width,
    height,
    fps,
) -> None:
    monkeypatch.setattr(
        vcam_native,
        "require_native_camera_component",
        lambda: pytest.fail("unsupported mode must fail before component probe"),
    )
    monkeypatch.setattr(
        vcam_native,
        "NativeVirtualCameraOutput",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported mode must fail before ring open"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"does not support {width}x{height}@{fps}",
    ):
        open_output(OutputConfig(backend="native", fps=fps), width, height)


def test_auto_skips_unsupported_native_mode_without_component_probe(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        vcam,
        "PyVirtualCamOutput",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("virtual camera unavailable")
        ),
    )
    monkeypatch.setattr(vcam, "_AUTO_NATIVE_ENABLED", True)
    monkeypatch.setattr(vcam.sys, "platform", "win32")
    monkeypatch.setattr(
        vcam_native,
        "native_camera_component_available",
        lambda: pytest.fail("unsupported native mode must not probe component"),
    )
    monkeypatch.setattr(
        vcam_native,
        "NativeVirtualCameraOutput",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported native mode must not open ring"
        ),
    )

    output = open_output(OutputConfig(backend="auto", fps=60), 1280, 720)

    assert isinstance(output, NullOutput)
    assert output.fallback_active
    assert output.fallback_reason == "virtual-camera-unavailable"
    assert (output.width, output.height, output.fps) == (1280, 720, 60)


def test_auto_supported_native_mode_falls_back_when_component_is_absent(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def unavailable_pyvirtualcam(*_args, **_kwargs):
        calls.append("pyvirtualcam")
        raise RuntimeError("virtual camera unavailable")

    def component_available() -> bool:
        calls.append("component")
        return False

    monkeypatch.setattr(vcam, "PyVirtualCamOutput", unavailable_pyvirtualcam)
    monkeypatch.setattr(vcam, "_AUTO_NATIVE_ENABLED", True)
    monkeypatch.setattr(vcam.sys, "platform", "win32")
    monkeypatch.setattr(
        vcam_native,
        "native_camera_component_available",
        component_available,
    )
    monkeypatch.setattr(
        vcam_native,
        "NativeVirtualCameraOutput",
        lambda *_args, **_kwargs: pytest.fail(
            "absent native component must not open ring"
        ),
    )

    output = open_output(OutputConfig(backend="auto", fps=30), 1280, 720)

    assert calls == ["pyvirtualcam", "component"]
    assert isinstance(output, NullOutput)
    assert output.fallback_active
    assert (output.width, output.height, output.fps) == (1280, 720, 30)


def _cpp_function(text: str, signature: str) -> str:
    marker = text.index(signature)
    opening = text.index("{", marker + len(signature))
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise AssertionError(f"unterminated C++ function {signature!r}")


def test_python_and_cpp_native_supported_modes_are_identical() -> None:
    source = (VCAM_DIR / "MediaSource.h").read_text(encoding="utf-8")
    formats = source.split("inline constexpr StreamFormat kStreamFormats[] = {", 1)[
        1
    ].split("};", 1)[0]
    cpp_modes = {
        tuple(int(value) for value in match)
        for match in re.findall(r"\{(\d+),\s*(\d+),\s*(\d+)\}", formats)
    }

    assert cpp_modes == vcam.SUPPORTED_NATIVE_MODES
    assert cpp_modes == {(1280, 720, 30), (1920, 1080, 30)}


def test_cpp_advertises_only_the_exact_ring_mode() -> None:
    source = (VCAM_DIR / "MediaSource.cpp").read_text(encoding="utf-8")
    create = _cpp_function(
        source,
        "HRESULT MediaSource::CreateStreamDescriptor("
        "IMFStreamDescriptor** descriptor) try",
    )

    assert "ring.ReadGeometry(ringWidth, ringHeight)" in create
    assert "if (!ring.ReadGeometry(ringWidth, ringHeight))" in create
    assert "bool matched = false;" in create
    assert "if (!matched)" in create
    assert create.count("winrt::check_hresult(MF_E_INVALIDMEDIATYPE);") == 2
    assert "kStreamFormats[i].width == ringWidth" in create
    assert "kStreamFormats[i].height == ringHeight" in create
    assert "MakeVideoType(kStreamFormats[selectedFormat])" in create
    assert "IMFMediaType* raw[] = {type.get()};" in create
    assert "IMFMediaType* raw[ARRAYSIZE(kStreamFormats)]" not in create

    frame_ring = (VCAM_DIR / "FrameRing.h").read_text(encoding="utf-8")
    read_geometry = _cpp_function(
        frame_ring,
        "bool ReadGeometry(uint32_t& width, uint32_t& height,",
    )
    assert "ValidateGeometry(header, payload)" in read_geometry
    assert "header.flags" not in read_geometry
    assert "int maxAttempts = 100) {" in frame_ring
    assert read_geometry.count("::Sleep(1);") == 3
    assert "width = header.width;" in read_geometry
    assert "height = header.height;" in read_geometry


def test_cpp_mismatch_uses_placeholder_and_contains_no_overlap_copy() -> None:
    source = (VCAM_DIR / "MediaStream.cpp").read_text(encoding="utf-8")
    compose = _cpp_function(
        source,
        "void MediaStream::ComposeFrame(std::vector<uint8_t>& out)",
    )

    assert "ringWidth == m_width && ringHeight == m_height" in compose
    assert "out = m_ringFrame;" in compose
    assert compose.count("FillPlaceholder(out, m_width, m_height);") == 2
    assert "FrameReadStatus::Transient" in compose
    assert "m_ringFrame.size() == expected" in compose
    assert compose.count("m_ringFrame.clear();") == 2
    for forbidden in (
        "copyWidth",
        "copyHeight",
        "srcLeft",
        "srcTop",
        "dstLeft",
        "dstTop",
        "std::memcpy",
        "cv::resize",
        "StretchBlt",
    ):
        assert forbidden not in compose

    frame_ring = (VCAM_DIR / "FrameRing.h").read_text(encoding="utf-8")
    copy_latest = _cpp_function(
        frame_ring,
        "FrameReadStatus CopyLatest(std::vector<uint8_t>& frame,",
    )
    assert "m_candidate.resize" in copy_latest
    assert "frame.swap(m_candidate)" in copy_latest
    assert copy_latest.count("::Sleep(0);") == 2
    assert "return FrameReadStatus::Transient;" in copy_latest
    live_flags_load = copy_latest.index("const uint32_t flagsBefore =")
    inactive_check = copy_latest.index("if ((header.flags & kFlagActive) == 0")
    sequence_retry = copy_latest.index("if (seqBefore != header.seq")
    geometry_validation = copy_latest.index("ValidateGeometry(header, payload)")
    assert geometry_validation < inactive_check < sequence_retry
    assert live_flags_load < inactive_check


def test_cpp_refreshes_consumer_type_before_each_mismatch_decision() -> None:
    source = (VCAM_DIR / "MediaStream.cpp").read_text(encoding="utf-8")
    request_sample = _cpp_function(
        source,
        "IFACEMETHODIMP MediaStream::RequestSample(IUnknown* token) noexcept try",
    )

    refresh = request_sample.index("RefreshNegotiatedType()")
    compose = request_sample.index("ComposeFrame(payload)")
    assert refresh < compose


def test_synthetic_source_normalizes_to_supported_native_canvas() -> None:
    canvas = (1280, 720)
    capture = SyntheticCapture(
        CameraConfig(
            synthetic=True,
            width=64,
            height=48,
            fit_mode="cover",
        ),
        canvas,
    )
    buffer = bytearray(vcam_native.ring_size(*canvas))
    output = vcam_native.NativeVirtualCameraOutput(
        *canvas,
        fps=30,
        buffer=buffer,
    )
    try:
        frame = capture.read()
        assert frame is not None
        output.send(frame.pixels)
        bgrx = vcam_native.read_latest_frame(buffer)
    finally:
        capture.close()
        output.close()

    assert frame.pixels.shape == (720, 1280, 3)
    assert frame.pixels.dtype == np.uint8
    assert frame.pixels.flags.c_contiguous
    assert frame.sequence == 1
    assert frame.generation == 1
    assert frame.geometry_generation == 1
    assert frame.content_rect == (0, 0, 1280, 720)
    assert bgrx is not None
    assert bgrx.shape == (720, 1280, 4)
    np.testing.assert_array_equal(
        bgrx[::120, ::160, :3],
        frame.pixels[::120, ::160],
    )
    assert (bgrx[:, :, 3] == 255).all()
    assert vcam_native.unpack_header(buffer)["frame_counter"] == 1
