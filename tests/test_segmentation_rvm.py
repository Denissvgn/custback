"""RVM matting backend tests with a fake onnxruntime — no model download,
no GPU, no network. The fake mimics the RVM ONNX interface: src + 4 recurrent
states + downsample_ratio in, (fgr, pha, 4 states) out."""

import sys
import types

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


def fake_ort(available_providers):
    mod = types.ModuleType("onnxruntime")
    mod.get_available_providers = lambda: list(available_providers)

    class SessionOptions:
        log_severity_level = 2

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            avail = set(available_providers)
            self._providers = [p for p in (providers or []) if p in avail] or [
                "CPUExecutionProvider"
            ]
            self.feeds = []

        def get_providers(self):
            return self._providers

        def run(self, _outputs, feeds):
            self.feeds.append(feeds)
            src = feeds["src"]  # (1, 3, H, W) float32 RGB in [0, 1]
            gray = src[0].mean(axis=0)
            pha = (gray > 0.5).astype(np.float32)[None, None]
            fgr = src  # "clean foreground" prediction = the source itself
            rec = [feeds[k] + 1.0 for k in ("r1i", "r2i", "r3i", "r4i")]
            return [fgr, pha, *rec]

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
