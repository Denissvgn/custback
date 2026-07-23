"""RVM matting backend tests with a fake onnxruntime — no model download,
no GPU, no network. The fake mimics the RVM ONNX interface: src + 4 recurrent
states + downsample_ratio in, (fgr, pha, 4 states) out."""

import json
import sys
import tempfile
import types
from typing import Any, cast

import numpy as np
import pytest

from custback.config import SegmentationConfig
from custback.segmentation import (
    HeuristicSegmenter,
    MaskRefiner,
    RVMSegmenter,
    create_segmenter,
    refiner_for,
)


def fake_ort(
    available_providers,
    *,
    run_error=None,
    gpu_fail_after=None,
    profile_provider=True,
):
    """Fake onnxruntime.

    ``run_error`` (a callable) is raised from a *production* GPU session's
    ``run`` once more than ``gpu_fail_after`` production GPU inferences have run
    (the warm-up is the first), to exercise inference-time recovery without
    tripping the startup proof or warm-up. ``profile_provider`` controls whether
    the profiling proof shows a node executed on the bound provider — set False
    to simulate a registered-but-not-executing GPU provider.
    """

    mod = cast(Any, types.ModuleType("onnxruntime"))
    mod.get_available_providers = lambda: list(available_providers)

    class SessionOptions:
        def __init__(self):
            self.log_severity_level = 2
            self.enable_profiling = False
            self.profile_file_prefix = ""
            self._entries = {}

        def add_session_config_entry(self, key, value):
            self._entries[key] = value

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            avail = set(available_providers)
            requested = []
            for p in providers or []:
                requested.append(p[0] if isinstance(p, tuple) else p)
            self._providers = [p for p in requested if p in avail] or [
                "CPUExecutionProvider"
            ]
            self._profiling = bool(getattr(sess_options, "enable_profiling", False))
            self._gpu_runs = 0
            self.feeds = []

        def get_providers(self):
            return self._providers

        def run(self, _outputs, feeds):
            self.feeds.append(feeds)
            active = self._providers[0]
            # Only a production (non-profiling) GPU session fails, and only after
            # the warm-up so the startup proof and warm-up both succeed first.
            if active != "CPUExecutionProvider" and not self._profiling:
                self._gpu_runs += 1
                if run_error is not None and (
                    gpu_fail_after is None or self._gpu_runs > gpu_fail_after
                ):
                    raise run_error()
            src = feeds["src"]  # (1, 3, H, W) float32 RGB in [0, 1]
            gray = src[0].mean(axis=0)
            pha = (gray > 0.5).astype(np.float32)[None, None]
            fgr = src  # "clean foreground" prediction = the source itself
            rec = [feeds[k] + 1.0 for k in ("r1i", "r2i", "r3i", "r4i")]
            return [fgr, pha, *rec]

        def end_profiling(self):
            if not self._profiling:
                return ""
            provider = (
                self._providers[0] if profile_provider else "CPUExecutionProvider"
            )
            events = [
                {"cat": "Node", "name": "MatteNode", "args": {"provider": provider}}
            ]
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, prefix="fake-ort-profile-"
            )
            with handle:
                json.dump(events, handle)
            return handle.name

    mod.SessionOptions = SessionOptions
    mod.InferenceSession = InferenceSession
    return mod


@pytest.fixture()
def cpu_ort(monkeypatch):
    mod = fake_ort(["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


def rvm_cfg(**kw):
    # model_path set -> no download attempt
    return SegmentationConfig(backend="rvm", model_path="/fake/model.onnx", **kw)


def bright_center_frame(h=72, w=128):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    img[20:52, 44:84] = 220
    return img


def test_segment_returns_mask_and_clean_foreground(cpu_ort):
    seg = RVMSegmenter(rvm_cfg())
    frame = bright_center_frame()
    mask = seg.segment(frame)
    assert mask.shape == (72, 128)
    assert mask.dtype == np.float32
    assert mask[36, 64] == 1.0  # bright center = person
    assert mask[2, 2] == 0.0
    assert seg.last_foreground is not None
    assert seg.last_foreground.shape == frame.shape
    assert seg.last_foreground.dtype == np.uint8


def test_device_cpu_and_cuda(cpu_ort, monkeypatch):
    assert RVMSegmenter(rvm_cfg()).device == "cpu"
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    assert RVMSegmenter(rvm_cfg()).device == "cuda"


def test_recurrent_state_is_fed_back_and_reset_on_resize(cpu_ort):
    seg = RVMSegmenter(rvm_cfg())
    seg.segment(bright_center_frame())
    seg.segment(bright_center_frame())
    feeds = seg._session.feeds
    assert float(feeds[0]["r1i"].ravel()[0]) == 0.0
    assert float(feeds[1]["r1i"].ravel()[0]) == 1.0  # previous output reused
    # resolution change -> state must reset (it is resolution-bound)
    seg.segment(bright_center_frame(h=36, w=64))
    assert float(feeds[2]["r1i"].ravel()[0]) == 0.0


def test_downsample_ratio_auto_and_explicit(cpu_ort):
    seg = RVMSegmenter(rvm_cfg())
    seg.segment(np.zeros((720, 1280, 3), np.uint8))
    auto = float(seg._session.feeds[0]["downsample_ratio"][0])
    assert auto == pytest.approx(512 / 1280)

    seg = RVMSegmenter(rvm_cfg(rvm_downsample=0.5))
    seg.segment(np.zeros((720, 1280, 3), np.uint8))
    assert float(seg._session.feeds[0]["downsample_ratio"][0]) == pytest.approx(0.5)


def test_auto_backend_prefers_rvm_when_onnxruntime_present(cpu_ort):
    cfg = SegmentationConfig(backend="auto", model_path="/fake/model.onnx")
    assert isinstance(create_segmenter(cfg), RVMSegmenter)


def test_auto_backend_without_onnxruntime_falls_back(monkeypatch):
    # None in sys.modules makes `import x` raise ImportError; block both ML
    # backends so the test never hits the network for model downloads.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    seg = create_segmenter(SegmentationConfig(backend="auto"))
    assert isinstance(seg, HeuristicSegmenter)


def test_explicit_rvm_backend_raises_without_onnxruntime(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError):
        create_segmenter(SegmentationConfig(backend="rvm"))


def test_refiner_skips_redundant_work_for_matting_backends(cpu_ort):
    cfg = rvm_cfg(mask_blur=7, edge_refine=True, temporal_smoothing=0.5, mask_shift=-1)
    matte_refiner = refiner_for(cfg, RVMSegmenter(cfg))
    assert matte_refiner.cfg.mask_blur == 0
    assert matte_refiner.cfg.edge_refine is False
    assert matte_refiner.cfg.temporal_smoothing == 0.0
    assert matte_refiner.cfg.mask_shift == -1  # user's halo control survives

    plain_refiner = refiner_for(cfg, HeuristicSegmenter(cfg))
    assert isinstance(plain_refiner, MaskRefiner)
    assert plain_refiner.cfg.mask_blur == 7


# -- acceleration policy (WIN-4.x) --------------------------------------------

from custback.acceleration import GpuRequiredError  # noqa: E402
from custback.config import AccelerationConfig  # noqa: E402


def gpu_ort(monkeypatch, **kwargs):
    mod = fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"], **kwargs)
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    return mod


def test_cpu_mode_never_touches_gpu(monkeypatch):
    gpu_ort(monkeypatch)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="cpu"))
    assert seg.device == "cpu"
    status = seg.accel.status()
    assert status.active_provider == "cpu"
    assert status.fallback_active is False
    # The production session was asked for CPU only.
    assert seg._session.get_providers() == ["CPUExecutionProvider"]


def test_auto_uses_proven_gpu(monkeypatch):
    gpu_ort(monkeypatch)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig())
    assert seg.device == "cuda"
    assert seg.accel.status().active_provider == "cuda"
    assert seg.accel.on_gpu is True


def test_auto_gpu_registered_but_unprovable_falls_back(monkeypatch):
    gpu_ort(monkeypatch, profile_provider=False)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig())
    assert seg.device == "cpu"
    status = seg.accel.status()
    assert status.fallback_active is True
    assert status.fallback_count == 1


def test_gpu_required_without_gpu_raises(cpu_ort):
    with pytest.raises(GpuRequiredError):
        RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="gpu_required"))


def test_gpu_required_registered_but_unprovable_raises(monkeypatch):
    gpu_ort(monkeypatch, profile_provider=False)
    with pytest.raises(GpuRequiredError):
        RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="gpu_required"))


def test_gpu_required_with_proven_gpu_succeeds(monkeypatch):
    gpu_ort(monkeypatch)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="gpu_required"))
    assert seg.device == "cuda"


def test_create_segmenter_threads_acceleration(monkeypatch):
    gpu_ort(monkeypatch)
    seg = create_segmenter(
        SegmentationConfig(backend="rvm", model_path="/fake/model.onnx"),
        acceleration=AccelerationConfig(mode="cpu"),
    )
    assert isinstance(seg, RVMSegmenter)
    assert seg.device == "cpu"


def test_gpu_required_not_swallowed_by_auto_backend(monkeypatch):
    # backend=auto would normally fall back to another backend on RVM failure,
    # but gpu_required is an explicit demand that must propagate.
    gpu_ort(monkeypatch, profile_provider=False)
    with pytest.raises(GpuRequiredError):
        create_segmenter(
            SegmentationConfig(backend="auto", model_path="/fake/model.onnx"),
            acceleration=AccelerationConfig(mode="gpu_required"),
        )


def test_inference_time_recovery_rebuilds_cpu_and_retries_once(monkeypatch):
    gpu_ort(monkeypatch, run_error=RuntimeError, gpu_fail_after=1)
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig())
    assert seg.device == "cuda"  # warm-up (1st production GPU run) succeeded
    frame = bright_center_frame()
    # First real frame fails on GPU, recovers to CPU, retries once, succeeds.
    mask = seg.segment(frame)
    assert mask.shape == frame.shape[:2]
    assert seg.device == "cpu"
    status = seg.accel.status()
    assert status.fallback_active is True
    assert status.fallback_count == 1
    assert seg.accel.on_gpu is False
    # Recurrent state was cleared on recovery, then advances again on CPU.
    seg.segment(frame)
    assert seg.device == "cpu"


def test_cpu_inference_failure_is_not_retried(cpu_ort):
    # A pure-CPU session that fails is a real error, not a GPU fallback: it must
    # surface rather than loop rebuilding CPU sessions.
    seg = RVMSegmenter(rvm_cfg(), acceleration=AccelerationConfig(mode="cpu"))
    assert seg.device == "cpu"

    def boom(*_a, **_k):
        raise RuntimeError("cpu inference error")

    seg._session.run = boom
    with pytest.raises(RuntimeError, match="cpu inference error"):
        seg.segment(bright_center_frame())
