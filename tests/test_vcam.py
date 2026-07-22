"""Virtual-camera output selection and OBS setup-guidance tests (WIN-3.7)."""

from __future__ import annotations

import custback.vcam as vcam
from custback.config import OutputConfig
from custback.vcam import NullOutput, open_output, virtual_camera_setup_hint


def test_setup_hint_is_platform_specific():
    windows = virtual_camera_setup_hint("win32")
    assert "OBS" in windows and "Virtual Camera" in windows
    assert "OBS" in virtual_camera_setup_hint("darwin")
    assert "v4l2loopback" in virtual_camera_setup_hint("linux")


def test_classify_output_failure_detects_single_instance_contention():
    assert (
        vcam._classify_output_failure(RuntimeError("device is already in use"))
        == "virtual-camera-in-use"
    )
    assert (
        vcam._classify_output_failure(RuntimeError("virtual camera not installed"))
        == "virtual-camera-unavailable"
    )


def test_open_output_null_backend_returns_null():
    out = open_output(OutputConfig(backend="null"), 128, 72)
    assert isinstance(out, NullOutput)
    assert not out.fallback_active


def test_open_output_auto_falls_back_to_null_with_reason(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("no virtual camera device found")

    monkeypatch.setattr(vcam, "PyVirtualCamOutput", boom)
    out = open_output(OutputConfig(backend="auto"), 128, 72)
    assert isinstance(out, NullOutput)
    assert out.fallback_active
    assert out.fallback_reason == "virtual-camera-unavailable"


def test_open_output_auto_reports_in_use(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("OBS Virtual Camera is already in use")

    monkeypatch.setattr(vcam, "PyVirtualCamOutput", boom)
    out = open_output(OutputConfig(backend="auto"), 128, 72)
    assert out.fallback_reason == "virtual-camera-in-use"


def test_open_output_explicit_backend_raises_with_hint(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("cannot connect")

    monkeypatch.setattr(vcam, "PyVirtualCamOutput", boom)
    try:
        open_output(OutputConfig(backend="pyvirtualcam"), 128, 72)
    except RuntimeError as exc:
        assert "cannot connect" in str(exc)
        assert "OBS" in str(exc) or "v4l2loopback" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")
