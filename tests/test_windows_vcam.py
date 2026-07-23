"""Contract tests for the native Windows virtual camera (WIN-6.1).

The Media Foundation media source itself builds and runs only on Windows 11,
but everything the two sides *agree on* is byte-static and verified here on
any platform:

* the shared frame-ring protocol (``src/custback/vcam_native.py`` is the
  source of truth; ``packaging/windows/vcam/FrameRing.h`` must mirror it),
* the activator CLSID, which must be identical in the C++ project, the C#
  shell, the installer registration, and the Python constant,
* fail-closed behavior of the ``native`` output backend off Windows, and
* the packaging wiring (project file lists, installer staging/registration).
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from custback import vcam_native
from custback.camera_devices import is_virtual_output_name
from custback.config import OutputConfig
from custback.vcam import open_output

ROOT = Path(__file__).resolve().parents[1]
VCAM_DIR = ROOT / "packaging" / "windows" / "vcam"
SHELL_DIR = ROOT / "packaging" / "windows" / "shell"
INSTALLER_DIR = ROOT / "packaging" / "windows" / "installer"

pytestmark = pytest.mark.skipif(
    not (VCAM_DIR / "FrameRing.h").exists(),
    reason="native vcam packaging tree is not present in this layout",
)


def _ring_buffer(width: int = 8, height: int = 6) -> bytearray:
    return bytearray(vcam_native.ring_size(width, height))


def _frame(width: int = 8, height: int = 6, value: int = 17) -> np.ndarray:
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    return frame


# -- frame-ring protocol -----------------------------------------------------
def test_header_layout_is_64_bytes() -> None:
    assert vcam_native.HEADER_SIZE == 64
    assert vcam_native.ring_size(1280, 720) == 64 + 1280 * 720 * 4


def test_writer_publishes_and_reference_reader_roundtrips() -> None:
    buffer = _ring_buffer()
    writer = vcam_native.FrameRingWriter(buffer, 8, 6)
    frame = _frame()
    writer.publish(frame)

    header = vcam_native.unpack_header(buffer)
    assert header["magic"] == vcam_native.MAGIC
    assert header["version"] == vcam_native.PROTOCOL_VERSION
    assert header["fourcc"] == vcam_native.FOURCC
    assert header["seq"] % 2 == 0
    assert header["flags"] & vcam_native.FLAG_ACTIVE
    assert header["frame_counter"] == 1
    assert header["stride"] == 8 * 4

    read = vcam_native.read_latest_frame(buffer)
    assert read is not None
    assert read.shape == (6, 8, 4)
    np.testing.assert_array_equal(read[:, :, :3], frame)
    assert (read[:, :, 3] == 255).all()


def test_reader_rejects_torn_write_and_inactive_ring() -> None:
    buffer = _ring_buffer()
    writer = vcam_native.FrameRingWriter(buffer, 8, 6)
    writer.publish(_frame())

    # Odd seq = write in progress; a bounded reader must give up, not tear.
    seq = vcam_native.unpack_header(buffer)["seq"]
    struct.pack_into("<I", buffer, 24, seq + 1)
    assert vcam_native.read_latest_frame(buffer) is None
    struct.pack_into("<I", buffer, 24, seq)

    writer.mark_inactive()
    assert vcam_native.read_latest_frame(buffer) is None


def test_writer_rejects_geometry_mismatch() -> None:
    writer = vcam_native.FrameRingWriter(_ring_buffer(), 8, 6)
    with pytest.raises(ValueError, match="ring is 8x6"):
        writer.publish(np.zeros((7, 9, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="holds"):
        vcam_native.FrameRingWriter(bytearray(16), 8, 6)


def test_reader_rejects_wrong_magic_or_version() -> None:
    buffer = _ring_buffer()
    writer = vcam_native.FrameRingWriter(buffer, 8, 6)
    writer.publish(_frame())
    buffer[0:4] = b"XXXX"
    assert vcam_native.read_latest_frame(buffer) is None


# -- C++ mirror consistency --------------------------------------------------
def _cpp_constant(text: str, name: str) -> int:
    match = re.search(rf"{name}\s*=\s*(0x[0-9A-Fa-f]+|\d+)u?", text)
    assert match, f"FrameRing.h does not define {name}"
    return int(match.group(1), 0)


def test_frame_ring_header_mirrors_python_protocol() -> None:
    text = (VCAM_DIR / "FrameRing.h").read_text(encoding="utf-8")
    magic_le = struct.unpack("<I", vcam_native.MAGIC)[0]
    fourcc_le = struct.unpack("<I", vcam_native.FOURCC)[0]
    assert _cpp_constant(text, "kMagic") == magic_le
    assert _cpp_constant(text, "kFourcc") == fourcc_le
    assert _cpp_constant(text, "kProtocolVersion") == vcam_native.PROTOCOL_VERSION
    assert _cpp_constant(text, "kHeaderSize") == vcam_native.HEADER_SIZE
    assert _cpp_constant(text, "kFlagActive") == vcam_native.FLAG_ACTIVE
    assert _cpp_constant(text, "kBytesPerPixel") == vcam_native.BYTES_PER_PIXEL
    # Section name: C++ escapes the backslash; compare unescaped.
    cpp_section = re.search(r'kSectionName\[\]\s*=\s*L"([^"]+)"', text)
    assert cpp_section is not None
    assert cpp_section.group(1).replace("\\\\", "\\") == vcam_native.SECTION_NAME
    # The C++ struct must be pinned to the same 64-byte layout.
    assert "static_assert(sizeof(FrameRingHeader) == kHeaderSize" in text


def test_activator_clsid_is_identical_everywhere() -> None:
    clsid = vcam_native.VCAM_CLSID
    assert re.fullmatch(r"\{[0-9A-F-]{36}\}", clsid)
    for relpath, root in (
        ("Activator.h", VCAM_DIR),
        ("VirtualCameraSession.cs", SHELL_DIR),
        ("Package.wxs", INSTALLER_DIR),
    ):
        text = (root / relpath).read_text(encoding="utf-8")
        assert clsid in text, f"{relpath} does not carry the activator CLSID"
    # The C++ GUID struct literal must be the same value, not just the string.
    activator = (VCAM_DIR / "Activator.h").read_text(encoding="utf-8")
    parts = clsid.strip("{}").split("-")
    assert f"0x{parts[0]}" in activator and f"0x{parts[1]}" in activator


def test_friendly_name_is_loop_prevented() -> None:
    session = (SHELL_DIR / "VirtualCameraSession.cs").read_text(encoding="utf-8")
    match = re.search(r'FriendlyName = "([^"]+)"', session)
    assert match is not None
    friendly = match.group(1)
    assert friendly == vcam_native.VCAM_FRIENDLY_NAME
    # WIN-3.4: the native output camera must never be offered as an input.
    assert is_virtual_output_name(friendly)


# -- output backend behavior -------------------------------------------------
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fail-closed check")
def test_native_backend_fails_closed_off_windows() -> None:
    cfg = OutputConfig(backend="native")
    with pytest.raises(RuntimeError, match="Windows 11"):
        open_output(cfg, 8, 6)


def test_native_output_publishes_via_injected_buffer() -> None:
    buffer = _ring_buffer()
    output = vcam_native.NativeVirtualCameraOutput(8, 6, buffer=buffer)
    assert output.paces is False  # the MF source resamples on its own clock
    output.send(_frame())
    output.send(_frame(value=42))
    assert output.frames_sent == 2
    read = vcam_native.read_latest_frame(buffer)
    assert read is not None and read[1, 1, 0] == 42
    output.close()
    assert vcam_native.read_latest_frame(buffer) is None


def test_auto_backend_does_not_select_native() -> None:
    # Guardrail: until the WIN-6.1 clean-machine gate passes, `auto` must keep
    # the OBS/pyvirtualcam-then-null behavior; native is explicit opt-in only.
    output = open_output(OutputConfig(backend="auto"), 8, 6)
    try:
        assert type(output).__name__ != "NativeVirtualCameraOutput"
    finally:
        output.close()


# -- packaging wiring --------------------------------------------------------
def test_vcxproj_lists_exactly_the_sources_on_disk() -> None:
    project = (VCAM_DIR / "CustbackVCam.vcxproj").read_text(encoding="utf-8")
    listed = set(re.findall(r'<Cl(?:Compile|Include) Include="([^"]+)"', project))
    on_disk = {
        path.name for path in VCAM_DIR.iterdir() if path.suffix in (".cpp", ".h")
    }
    assert listed == on_disk
    assert "CustbackVCam.def" in project


def test_def_exports_the_com_surface() -> None:
    exports = (VCAM_DIR / "CustbackVCam.def").read_text(encoding="utf-8")
    for symbol in ("DllGetClassObject", "DllCanUnloadNow", "DllUnregisterServer"):
        assert symbol in exports


def test_installer_stages_and_registers_the_vcam_dll() -> None:
    build = (INSTALLER_DIR / "build.ps1").read_text(encoding="utf-8")
    assert "CustbackVCam.dll" in build
    assert "IncludeNativeVCam=1" in build
    package = (INSTALLER_DIR / "Package.wxs").read_text(encoding="utf-8")
    assert "NativeVCamRegistration" in package
    assert "InProcServer32" in package
    # Uninstall cleanup must target the real CLSID, not a placeholder.
    assert "{00000000-0000-0000-0000-000000000000}" not in package


def test_shell_owns_camera_lifecycle() -> None:
    session = (SHELL_DIR / "VirtualCameraSession.cs").read_text(encoding="utf-8")
    # Session lifetime + current-user access: nothing persists past the shell.
    assert "MFVirtualCameraLifetime_Session" in session
    assert "MFVirtualCameraAccess_CurrentUser" in session
    assert "MFCreateVirtualCamera" in session
    tray = (SHELL_DIR / "TrayApplicationContext.cs").read_text(encoding="utf-8")
    assert "_vcam.TryStart" in tray and "_vcam.Stop" in tray
