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

import custback.vcam as vcam
from custback import vcam_native
from custback.camera_devices import is_virtual_output_name
from custback.config import OutputConfig
from custback.vcam import NullOutput, open_output

ROOT = Path(__file__).resolve().parents[1]
VCAM_DIR = ROOT / "packaging" / "windows" / "vcam"
SHELL_DIR = ROOT / "packaging" / "windows" / "shell"
INSTALLER_DIR = ROOT / "packaging" / "windows" / "installer"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(
    not (VCAM_DIR / "FrameRing.h").exists(),
    reason="native vcam packaging tree is not present in this layout",
)


def _csharp_block_after(text: str, marker: str) -> str:
    """Return the contents of the braced C# block following *marker*."""
    marker_index = text.index(marker)
    block_start = text.index("{", marker_index + len(marker))
    depth = 0
    for index in range(block_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[block_start + 1 : index]
    raise AssertionError(f"unterminated C# block after {marker!r}")


def _ring_buffer(width: int = 8, height: int = 6) -> bytearray:
    return bytearray(vcam_native.ring_size(width, height))


def _frame(width: int = 8, height: int = 6, value: int = 17) -> np.ndarray:
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    return frame


def test_rgb32_media_type_declares_the_proven_canonical_output_color() -> None:
    source = (VCAM_DIR / "MediaSource.cpp").read_text(encoding="utf-8")
    make_type = source[
        source.index("winrt::com_ptr<IMFMediaType> MakeVideoType") : source.index(
            "}  // namespace"
        )
    ]

    assert "MF_MT_VIDEO_PRIMARIES, MFVideoPrimaries_BT709" in make_type
    assert "MF_MT_TRANSFER_FUNCTION, MFVideoTransFunc_sRGB" in make_type
    assert "MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_0_255" in make_type
    assert "MF_MT_YUV_MATRIX" not in make_type


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


def test_reader_rechecks_inactive_flag_after_payload_copy() -> None:
    class InactivatingBuffer(bytearray):
        def __init__(self, size: int) -> None:
            super().__init__(size)
            self.inactivated_during_copy = False

        def __getitem__(self, key):
            value = super().__getitem__(key)
            if (
                isinstance(key, slice)
                and key.start == vcam_native.HEADER_SIZE
                and not self.inactivated_during_copy
            ):
                struct.pack_into("<I", self, 28, 0)
                self.inactivated_during_copy = True
            return value

    buffer = InactivatingBuffer(vcam_native.ring_size(8, 6))
    writer = vcam_native.FrameRingWriter(buffer, 8, 6)
    writer.publish(_frame())

    assert vcam_native.read_latest_frame(buffer) is None
    assert buffer.inactivated_during_copy


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


def test_reader_rejects_truncated_header_without_raising() -> None:
    assert vcam_native.read_latest_frame(bytearray(vcam_native.HEADER_SIZE - 1)) is None


def test_reader_rejects_wrong_fourcc_or_stride() -> None:
    buffer = _ring_buffer()
    writer = vcam_native.FrameRingWriter(buffer, 8, 6)
    writer.publish(_frame())

    buffer[20:24] = b"RGBA"
    assert vcam_native.read_latest_frame(buffer) is None
    buffer[20:24] = vcam_native.FOURCC

    struct.pack_into("<I", buffer, 16, 8 * 4 + 4)
    assert vcam_native.read_latest_frame(buffer) is None


@pytest.mark.parametrize(
    ("present", "expected"),
    [(True, "section present"), (False, "section absent")],
)
def test_native_ring_diagnostic_probes_without_creating(present, expected) -> None:
    calls = []

    def probe(name):
        calls.append(name)
        return present

    assert vcam_native.native_ring_status(platform="win32", probe=probe) == expected
    assert calls == [vcam_native.SECTION_NAME]


def test_native_ring_diagnostic_is_explicitly_unsupported_off_windows() -> None:
    assert (
        vcam_native.native_ring_diagnostic(platform="linux")
        == "native ring: unsupported"
    )


def test_native_component_probe_requires_registered_existing_dll() -> None:
    dll_path = r"C:\Program Files\Custback\CustbackVCam.dll"
    calls = []

    def read_registration(key):
        calls.append(("registry", key))
        return dll_path

    def file_exists(path):
        calls.append(("file", path))
        return True

    assert vcam_native.native_camera_component_available(
        platform="win32",
        read_registration=read_registration,
        file_exists=file_exists,
    )
    assert calls == [
        ("registry", vcam_native.VCAM_INPROC_REGISTRY_KEY),
        ("file", dll_path),
    ]


def test_native_component_probe_fails_closed_for_missing_registration_or_dll() -> None:
    def unexpected_file_probe(_path):
        pytest.fail("a missing registration must not probe a file")

    assert not vcam_native.native_camera_component_available(
        platform="win32",
        read_registration=lambda _key: None,
        file_exists=unexpected_file_probe,
    )
    assert not vcam_native.native_camera_component_available(
        platform="win32",
        read_registration=lambda _key: r"C:\missing\CustbackVCam.dll",
        file_exists=lambda _path: False,
    )


def test_native_component_probe_skips_registry_off_windows() -> None:
    def unexpected_registry_read(_key):
        pytest.fail("non-Windows availability must not read the registry")

    assert not vcam_native.native_camera_component_available(
        platform="linux",
        read_registration=unexpected_registry_read,
    )


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
    # The reader must pair CPython's release-ordered publication with acquire
    # loads/fencing; plain seq loads can admit torn frames on ARM64.
    assert "#include <atomic>" in text
    assert text.count("std::atomic_ref<const uint32_t>") == 6
    assert text.count("std::memory_order_acquire") == 8
    assert "header.fourcc != kFourcc" in text
    assert "header.stride != static_cast<uint32_t>(expectedStride)" in text
    assert "available / expectedStride" in text
    payload_copy = text.index("std::memcpy(m_candidate.data(), m_view + kHeaderSize,")
    acquire_fence = text.index(
        "std::atomic_thread_fence(std::memory_order_acquire);",
        payload_copy,
    )
    seq_recheck = text.index("const uint32_t seqAfter =", acquire_fence)
    active_recheck = text.index("const uint32_t flagsAfter =", seq_recheck)
    inactive_decision = text.index(
        "if ((flagsAfter & kFlagActive) == 0)",
        active_recheck,
    )
    sequence_retry = text.index("if (seqAfter != seqBefore)", inactive_decision)
    candidate_swap = text.index("frame.swap(m_candidate)", sequence_retry)
    assert (
        payload_copy
        < acquire_fence
        < seq_recheck
        < active_recheck
        < inactive_decision
        < sequence_retry
        < candidate_swap
    )


def test_mit_c1_gate_build_reports_frame_ring_open_error() -> None:
    frame_ring = (VCAM_DIR / "FrameRing.h").read_text(encoding="utf-8")
    open_index = frame_ring.index(
        "::OpenFileMappingW(FILE_MAP_READ, FALSE, sectionName)"
    )
    error_index = frame_ring.index("const DWORD error = ::GetLastError()", open_index)
    trace_index = frame_ring.index("TraceOpenFailure(sectionName, error)", error_index)
    assert open_index < error_index < trace_index
    assert "CUSTBACK_VCAM_GATE_DIAGNOSTICS" in frame_ring
    assert "::OutputDebugStringW(message)" in frame_ring
    assert 'L"GetLastError=%lu\\r\\n"' in frame_ring

    project = (VCAM_DIR / "CustbackVCam.vcxproj").read_text(encoding="utf-8")
    assert (
        "<CustbackVcamGateDiagnostics "
        """Condition="'$(CustbackVcamGateDiagnostics)'==''">false""" in project
    )
    assert "CUSTBACK_VCAM_GATE_DIAGNOSTICS;" in project

    build = (VCAM_DIR / "build.ps1").read_text(encoding="utf-8")
    assert "[switch]$GateDiagnostics" in build
    assert "/p:CustbackVcamGateDiagnostics=$gateDiagnosticsValue" in build


def test_mit_c1_checklist_starts_with_live_transport_and_softened_assumption() -> None:
    readme = (VCAM_DIR / "README.md").read_text(encoding="utf-8")
    checklist = readme.split("## WIN-6.1 clean-machine checklist", 1)[1]
    first_row = checklist.split("\n1. ", 1)[1].split("\n2. ", 1)[0]
    assert "engine publishing" in first_row
    assert "Windows Camera app" in first_row
    assert "live processed frames" in first_row
    assert "placeholder" in first_row
    assert "OpenFileMappingW" in first_row
    assert checklist.index(first_row) < checklist.index("Teams, Zoom")

    module_doc = " ".join((vcam_native.__doc__ or "").split())
    assert "namespace visibility remains an assumption" in module_doc
    assert "pending the MIT-C1 clean-machine gate evidence" in module_doc
    assert "Everything security-relevant about the section is local" not in module_doc


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
        open_output(cfg, 1280, 720)


def test_explicit_native_backend_fails_when_component_is_not_installed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(vcam.sys, "platform", "win32")
    monkeypatch.setattr(
        vcam_native,
        "native_camera_component_available",
        lambda **_kwargs: False,
    )

    def unexpected_native_open(*_args):
        pytest.fail("missing native component must fail before opening the ring")

    monkeypatch.setattr(
        vcam_native,
        "NativeVirtualCameraOutput",
        unexpected_native_open,
    )

    with pytest.raises(RuntimeError, match="component is not installed"):
        open_output(OutputConfig(backend="native"), 1280, 720)


def test_native_output_publishes_via_injected_buffer() -> None:
    buffer = _ring_buffer()
    output = vcam_native.NativeVirtualCameraOutput(8, 6, buffer=buffer)
    assert output.paces is False  # the MF source resamples on its own clock
    assert vcam_native.read_latest_frame(buffer) is None
    output.send(_frame())
    first = vcam_native.read_latest_frame(buffer)
    assert first is not None and first[1, 1, 0] == 17
    output.send(_frame(value=42))
    assert output.frames_sent == 2
    read = vcam_native.read_latest_frame(buffer)
    assert read is not None and read[1, 1, 0] == 42
    output.close()
    assert vcam_native.read_latest_frame(buffer) is None


def test_auto_backend_does_not_select_native(monkeypatch) -> None:
    # Guardrail: until the WIN-6.1 clean-machine gate passes, `auto` must keep
    # the OBS/pyvirtualcam-then-null behavior; native is explicit opt-in only.
    def unavailable_pyvirtualcam(*_args):
        raise RuntimeError("OBS unavailable")

    assert vcam._AUTO_NATIVE_ENABLED is False
    monkeypatch.setattr(vcam.sys, "platform", "win32")
    monkeypatch.setattr(vcam, "PyVirtualCamOutput", unavailable_pyvirtualcam)
    monkeypatch.setattr(
        vcam_native,
        "native_camera_component_available",
        lambda: pytest.fail("disabled native rung must not probe the component"),
    )
    output = open_output(OutputConfig(backend="auto"), 1280, 720)
    try:
        assert isinstance(output, NullOutput)
    finally:
        output.close()


def test_auto_backend_uses_native_second_when_gate_flag_is_enabled(
    monkeypatch,
) -> None:
    calls = []

    def unavailable_pyvirtualcam(*_args):
        calls.append("pyvirtualcam")
        raise RuntimeError("OBS unavailable")

    class NativeMarker:
        def close(self):
            calls.append("close")

    def open_native(*_args, **_kwargs):
        calls.append("native")
        return NativeMarker()

    monkeypatch.setattr(vcam, "PyVirtualCamOutput", unavailable_pyvirtualcam)
    monkeypatch.setattr(vcam, "_AUTO_NATIVE_ENABLED", True)
    monkeypatch.setattr(vcam.sys, "platform", "win32")
    monkeypatch.setattr(
        vcam_native,
        "native_camera_component_available",
        lambda: True,
    )
    monkeypatch.setattr(vcam_native, "NativeVirtualCameraOutput", open_native)

    output = open_output(OutputConfig(backend="auto"), 1280, 720)
    assert calls == ["pyvirtualcam", "native"]
    output.close()


def test_auto_backend_falls_through_native_to_null_when_both_fail(
    monkeypatch,
) -> None:
    calls = []

    def unavailable_pyvirtualcam(*_args):
        calls.append("pyvirtualcam")
        raise RuntimeError("OBS unavailable")

    def unavailable_native(*_args, **_kwargs):
        calls.append("native")
        raise RuntimeError("native unavailable")

    monkeypatch.setattr(vcam, "PyVirtualCamOutput", unavailable_pyvirtualcam)
    monkeypatch.setattr(vcam, "_AUTO_NATIVE_ENABLED", True)
    monkeypatch.setattr(vcam.sys, "platform", "win32")
    monkeypatch.setattr(
        vcam_native,
        "native_camera_component_available",
        lambda: True,
    )
    monkeypatch.setattr(vcam_native, "NativeVirtualCameraOutput", unavailable_native)

    output = open_output(OutputConfig(backend="auto"), 1280, 720)
    try:
        assert isinstance(output, NullOutput)
        assert calls == ["pyvirtualcam", "native"]
    finally:
        output.close()


def test_auto_backend_skips_uninstalled_native_component_and_returns_null(
    monkeypatch,
) -> None:
    calls = []

    def unavailable_pyvirtualcam(*_args):
        calls.append("pyvirtualcam")
        raise RuntimeError("OBS unavailable")

    def component_available():
        calls.append("component-probe")
        return False

    def unexpected_native_open(*_args):
        pytest.fail("auto must not open native output when its DLL is unavailable")

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
        unexpected_native_open,
    )

    output = open_output(OutputConfig(backend="auto"), 1280, 720)
    try:
        assert isinstance(output, NullOutput)
        assert calls == ["pyvirtualcam", "component-probe"]
    finally:
        output.close()


# -- packaging wiring --------------------------------------------------------
def test_vcxproj_targets_sdk_installed_on_pinned_windows_runner() -> None:
    project = (VCAM_DIR / "CustbackVCam.vcxproj").read_text(encoding="utf-8")
    assert (
        "<WindowsTargetPlatformVersion>10.0.22621.0</WindowsTargetPlatformVersion>"
    ) in project
    assert (
        "<WindowsTargetPlatformMinVersion>10.0.22000.0"
        "</WindowsTargetPlatformMinVersion>"
    ) in project


def test_vcxproj_lists_exactly_the_sources_on_disk() -> None:
    project = (VCAM_DIR / "CustbackVCam.vcxproj").read_text(encoding="utf-8")
    listed = set(re.findall(r'<Cl(?:Compile|Include) Include="([^"]+)"', project))
    on_disk = {
        path.name for path in VCAM_DIR.iterdir() if path.suffix in (".cpp", ".h")
    }
    assert listed == on_disk
    assert "CustbackVCam.def" in project
    assert "<LanguageStandard>stdcpp20</LanguageStandard>" in project


def test_ci_compiles_native_vcam_on_pinned_windows_runner() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    job = workflow.split("\n  windows-native-vcam:\n", 1)[1].split("\n  node:\n", 1)[0]
    assert "runs-on: windows-2022" in job
    assert "actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955" in job
    assert "./packaging/windows/vcam/build.ps1 -Arch x64" in job

    build = (VCAM_DIR / "build.ps1").read_text(encoding="utf-8")
    assert "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" in build
    assert '-find "VC\\Tools\\MSVC\\**\\bin\\Hostx64\\x64\\dumpbin.exe"' in build
    assert "& $dumpbin /nologo /exports $dll" in build


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

    engine = (SHELL_DIR / "Engine.cs").read_text(encoding="utf-8")
    status_method = _csharp_block_after(
        engine,
        "internal async Task<string> GetOutputBackendAsync("
        "CancellationToken cancellationToken)",
    )
    request_index = status_method.index(
        'new HttpRequestMessage(HttpMethod.Get, $"{BaseUrl}/status")'
    )
    auth_index = status_method.index(
        'request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {Bearer}")'
    )
    send_index = status_method.index("_http.SendAsync(request, cancellationToken)")
    assert request_index < auth_index < send_index
    assert '"output_backend"' in status_method

    tray = (SHELL_DIR / "TrayApplicationContext.cs").read_text(encoding="utf-8")
    boot_method = _csharp_block_after(tray, "private async Task BootAsync()")
    installed_block = _csharp_block_after(
        boot_method, "if (VirtualCameraSession.IsInstalled)"
    )
    assert (
        "outputBackend = await _engine.GetOutputBackendAsync(_cts.Token)"
        in installed_block
    )
    native_block = _csharp_block_after(
        installed_block, 'if (outputBackend == "NativeVirtualCameraOutput")'
    )
    assert tray.count("_vcam.TryStart(") == 1
    assert native_block.count("_vcam.TryStart(Log);") == 1
    assert "_vcam.Stop" in tray
    assert "native vcam DLL installed but engine output.backend is" in tray
    assert "native camera not started" in tray


def test_shell_gate_build_checks_com_projection_before_start() -> None:
    session = (SHELL_DIR / "VirtualCameraSession.cs").read_text(encoding="utf-8")
    try_start = _csharp_block_after(
        session, "internal bool TryStart(Action<string> log)"
    )

    create_index = try_start.index("hr = NativeMethods.MFCreateVirtualCamera(")
    create_check_index = try_start.index(
        "Marshal.ThrowExceptionForHR(hr);", create_index
    )
    ownership_index = try_start.index("_camera = camera;", create_check_index)
    gate_index = try_start.index("#if DEBUG || CUSTBACK_GATE_BUILD", ownership_index)
    get_count_index = try_start.index(
        "hr = camera.GetCount(out uint attributeCount);", gate_index
    )
    result_log_index = try_start.index(
        "IMFAttributes::GetCount HRESULT=0x", get_count_index
    )
    projection_check_index = try_start.index(
        "Marshal.ThrowExceptionForHR(hr);", result_log_index
    )
    start_index = try_start.index("camera.Start(IntPtr.Zero)", projection_check_index)

    assert (
        create_index
        < create_check_index
        < ownership_index
        < gate_index
        < get_count_index
        < result_log_index
        < projection_check_index
        < start_index
    )
    assert "[PreserveSig] int GetCount(out uint count);" in session

    project = (SHELL_DIR / "Custback.Shell.csproj").read_text(encoding="utf-8")
    assert "$(CustbackGateBuild)" in project
    assert "CUSTBACK_GATE_BUILD" in project

    shell_readme = (SHELL_DIR / "README.md").read_text(encoding="utf-8")
    assert "WIN-1.8 shell smoke" in shell_readme
    assert "-p:CustbackGateBuild=true" in shell_readme
    assert "IMFAttributes::GetCount HRESULT=0x00000000" in shell_readme
