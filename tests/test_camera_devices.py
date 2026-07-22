"""Camera enumeration, backend selection, and output-loop prevention tests."""

from __future__ import annotations

import custback.camera_devices as cd
from custback.camera_devices import (
    CameraDevice,
    camera_open_hint,
    enumerate_cameras,
    filter_input_devices,
    is_virtual_output_name,
    linux_video_devices,
    preferred_capture_backends,
    probe_camera_indices,
)


class FakeCV2:
    """Minimal OpenCV stand-in for index probing."""

    CAP_MSMF = 1400
    CAP_DSHOW = 700

    def __init__(self, openable_indices):
        self.openable = set(openable_indices)
        self.opened_with: list[tuple[int, int | None]] = []

    def VideoCapture(self, index, apiPreference=None):
        self.opened_with.append((index, apiPreference))
        return FakeCap(index in self.openable)


class FakeCap:
    def __init__(self, ok):
        self.ok = ok
        self.released = False

    def isOpened(self):
        return self.ok

    def release(self):
        self.released = True


# -- virtual-output detection (WIN-3.4) -------------------------------------
def test_is_virtual_output_name_matches_obs_and_project():
    assert is_virtual_output_name("OBS Virtual Camera")
    assert is_virtual_output_name("obs-camera")
    assert is_virtual_output_name("Custback Output")
    assert is_virtual_output_name("Dummy video device (v4l2loopback)")
    assert not is_virtual_output_name("Logitech BRIO")
    assert not is_virtual_output_name("")
    assert not is_virtual_output_name(None)


def test_filter_input_devices_drops_virtual_outputs():
    devices = [
        CameraDevice(0, "Logitech BRIO", "by-id/usb-brio", "V4L2"),
        CameraDevice(1, "OBS Virtual Camera", "v4l2:1", "V4L2", is_virtual_output=True),
    ]
    kept = filter_input_devices(devices)
    assert [d.index for d in kept] == [0]


# -- backend selection (WIN-3.2) --------------------------------------------
def test_preferred_backends_windows_prefers_msmf_then_dshow():
    order = preferred_capture_backends(FakeCV2([]), platform="win32")
    assert order == [("MSMF", FakeCV2.CAP_MSMF), ("DSHOW", FakeCV2.CAP_DSHOW)]


def test_preferred_backends_empty_off_windows():
    assert preferred_capture_backends(FakeCV2([]), platform="linux") == []
    assert preferred_capture_backends(FakeCV2([]), platform="darwin") == []


def test_preferred_backends_skips_missing_constants():
    class NoMsmf:
        CAP_DSHOW = 700  # MSMF constant absent (older OpenCV build)

    order = preferred_capture_backends(NoMsmf(), platform="win32")
    assert order == [("DSHOW", 700)]


# -- privacy-denial guidance (WIN-3.3) --------------------------------------
def test_camera_open_hint_mentions_windows_privacy_setting():
    hint = camera_open_hint("win32")
    assert "Privacy" in hint and "Camera" in hint


def test_camera_open_hint_is_platform_specific():
    assert "System Settings" in camera_open_hint("darwin")
    assert camera_open_hint("linux")


# -- Linux enumeration (WIN-3.1) --------------------------------------------
def test_linux_video_devices_reads_friendly_names_and_stable_ids(tmp_path):
    sysfs = tmp_path / "sys"
    for index, name in ((0, "Integrated Camera"), (2, "Logitech BRIO")):
        node = sysfs / f"video{index}"
        node.mkdir(parents=True)
        (node / "name").write_text(name + "\n")
    # A non-camera node and a metadata-only node without a name file.
    (sysfs / "video1").mkdir()
    (sysfs / "not-a-video").mkdir()

    by_id = tmp_path / "by-id"
    by_id.mkdir()
    (tmp_path / "video2").write_text("")  # symlink target lives beside by-id
    (by_id / "usb-Logitech_BRIO-video-index0").symlink_to(tmp_path / "video2")

    devices = linux_video_devices(sysfs_root=sysfs, by_id_root=by_id)
    by_index = {d.index: d for d in devices}
    assert by_index[0].name == "Integrated Camera"
    assert by_index[0].stable_id == "v4l2:0"
    assert by_index[2].name == "Logitech BRIO"
    assert by_index[2].stable_id == "by-id/usb-Logitech_BRIO-video-index0"
    assert by_index[1].name == "/dev/video1"  # no name file -> path fallback
    assert [d.index for d in devices] == [0, 1, 2]  # numeric order


def test_linux_video_devices_missing_root_returns_empty(tmp_path):
    assert linux_video_devices(sysfs_root=tmp_path / "absent") == []


def test_linux_video_devices_flags_loopback_as_virtual_output(tmp_path):
    node = tmp_path / "video10"
    node.mkdir()
    (node / "name").write_text("Dummy video device (0x0000)\n")
    (device,) = linux_video_devices(sysfs_root=tmp_path, by_id_root=tmp_path / "none")
    assert device.is_virtual_output


# -- index probing (Windows / fallback) -------------------------------------
def test_probe_camera_indices_keeps_openable_only():
    fake = FakeCV2(openable_indices={0, 2})
    backends = [("MSMF", FakeCV2.CAP_MSMF)]
    devices = probe_camera_indices(fake, max_probe=4, backends=backends)
    assert [d.index for d in devices] == [0, 2]
    assert all(d.backend == "MSMF" for d in devices)
    assert devices[0].stable_id == "msmf:0"
    assert devices[0].name == "Camera 0"


def test_probe_camera_indices_uses_name_provider():
    fake = FakeCV2(openable_indices={0})
    names = {0: "OBS Virtual Camera"}
    devices = probe_camera_indices(
        fake,
        max_probe=1,
        backends=[("DSHOW", FakeCV2.CAP_DSHOW)],
        name_provider=names.get,
    )
    assert devices[0].name == "OBS Virtual Camera"
    assert devices[0].is_virtual_output


def test_probe_falls_back_to_next_backend():
    class PickyCV2(FakeCV2):
        def VideoCapture(self, index, apiPreference=None):
            self.opened_with.append((index, apiPreference))
            # Only DSHOW opens; MSMF never does.
            return FakeCap(apiPreference == self.CAP_DSHOW)

    fake = PickyCV2(openable_indices=set())
    backends = [("MSMF", FakeCV2.CAP_MSMF), ("DSHOW", FakeCV2.CAP_DSHOW)]
    devices = probe_camera_indices(fake, max_probe=1, backends=backends)
    assert [d.backend for d in devices] == ["DSHOW"]


# -- top-level dispatch ------------------------------------------------------
def test_enumerate_cameras_linux_excludes_virtual_output(monkeypatch):
    monkeypatch.setattr(
        cd,
        "linux_video_devices",
        lambda: [
            CameraDevice(0, "BRIO", "v4l2:0", "V4L2"),
            CameraDevice(9, "OBS Virtual Camera", "v4l2:9", "V4L2", True),
        ],
    )
    devices = enumerate_cameras(platform="linux")
    assert [d.index for d in devices] == [0]

    full = enumerate_cameras(platform="linux", include_virtual_output=True)
    assert [d.index for d in full] == [0, 9]


def test_enumerate_cameras_non_linux_probes(monkeypatch):
    fake = FakeCV2(openable_indices={0})
    devices = enumerate_cameras(platform="win32", cv2_module=fake, max_probe=2)
    assert [d.index for d in devices] == [0]


def test_enumerate_cameras_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("sysfs exploded")

    monkeypatch.setattr(cd, "linux_video_devices", boom)
    assert enumerate_cameras(platform="linux") == []
