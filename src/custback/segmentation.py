"""Person segmentation backends.

Produces a float32 mask in [0, 1] with the same HxW as the input frame,
where 1.0 = person (keep), 0.0 = background (replace).

Backends:
  - rvm:        Robust Video Matting via onnxruntime. True alpha matting
                (hair-level edges) with a recurrent temporal state, plus a
                clean-foreground prediction used to remove background color
                spill. Runs on NVIDIA GPUs (CUDA) when onnxruntime-gpu is
                installed; CPU otherwise. Install: custback[rvm] or [gpu].
  - mediapipe:  MediaPipe Tasks ImageSegmenter (selfie segmentation model).
                Production quality, real-time on CPU; optional GPU delegate.
  - heuristic:  brightness/center-prior fallback so the pipeline still works
                without ML dependencies (and in tests / CI).
  - none:       full-frame mask (everything is "person") -> passthrough.

"auto" picks the best available: rvm > mediapipe > heuristic.
"""

from __future__ import annotations

import logging
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .config import SegmentationConfig

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

log = logging.getLogger(__name__)

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
RVM_MODEL_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
    "rvm_mobilenetv3_fp32.onnx"
)
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "custback" / "models"


def _download_model(url: str, filename: str) -> Path:
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = DEFAULT_MODEL_DIR / filename
    if not model_path.exists():
        log.info("downloading %s to %s", filename, model_path)
        tmp = model_path.with_suffix(model_path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(model_path)
    return model_path


class Segmenter(ABC):
    #: BGR uint8 clean-foreground prediction for the last frame, when the
    #: backend provides one (rvm). The compositor uses it inside the soft
    #: edge band to remove original-background color spill.
    last_foreground: np.ndarray | None = None
    #: Where inference runs ("cpu", "cuda", "gpu", "coreml") — shown in /status.
    device: str = "cpu"
    #: True when the backend outputs an edge-accurate alpha matte with its own
    #: temporal consistency; the refiner then skips redundant feathering/EMA.
    produces_matte: bool = False

    @abstractmethod
    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return float32 HxW mask in [0, 1]."""

    def close(self) -> None:
        pass


class NullSegmenter(Segmenter):
    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        return np.ones(frame_bgr.shape[:2], dtype=np.float32)


class HeuristicSegmenter(Segmenter):
    """Crude person prior: bright, center-weighted regions.

    Not meant for production visuals — it keeps the pipeline functional when
    no ML backend is available, and drives hardware-free tests.
    """

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._prior: np.ndarray | None = None

    def _center_prior(self, h: int, w: int) -> np.ndarray:
        if self._prior is None or self._prior.shape != (h, w):
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dist = ((xx - w / 2) / (w * 0.45)) ** 2 + ((yy - h / 2) / (h * 0.6)) ** 2
            self._prior = np.clip(1.5 - dist, 0.0, 1.0)
        return self._prior

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = frame_bgr.astype(np.float32).mean(axis=2) / 255.0
        h, w = gray.shape
        score = gray * self._center_prior(h, w)
        mask = (score > self.cfg.threshold * 0.8).astype(np.float32)
        return mask


class MediaPipeSegmenter(Segmenter):
    """MediaPipe Tasks ImageSegmenter with the selfie segmentation model."""

    def __init__(self, cfg: SegmentationConfig):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if cfg.model_path and cfg.model_path.endswith(".tflite"):
            model_path = Path(cfg.model_path)
        else:
            model_path = _download_model(MEDIAPIPE_MODEL_URL, "selfie_segmenter.tflite")

        def make(delegate):
            options = mp_vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path), delegate=delegate
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                output_confidence_masks=True,
            )
            return mp_vision.ImageSegmenter.create_from_options(options)

        self._segmenter = None
        if cfg.delegate == "gpu":
            try:
                self._segmenter = make(mp_python.BaseOptions.Delegate.GPU)
                self.device = "gpu"
            except Exception as exc:
                log.warning("mediapipe GPU delegate unavailable (%s); using CPU", exc)
        if self._segmenter is None:
            self._segmenter = make(None)
        self._mp = mp
        self._ts_ms = 0

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33
        result = self._segmenter.segment_for_video(mp_image, self._ts_ms)
        mask = result.confidence_masks[0].numpy_view().astype(np.float32)
        if mask.shape != frame_bgr.shape[:2]:
            mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return mask

    def close(self) -> None:
        self._segmenter.close()


class RVMSegmenter(Segmenter):
    """Robust Video Matting (https://github.com/PeterL1n/RobustVideoMatting)
    through onnxruntime.

    Outputs a real alpha matte plus a clean foreground prediction, and keeps
    a recurrent state across frames for temporal consistency. onnxruntime-gpu
    (the custback[gpu] extra) enables CUDA inference on NVIDIA cards; plain
    onnxruntime (custback[rvm]) runs on CPU.
    """

    produces_matte = True

    def __init__(self, cfg: SegmentationConfig):
        import onnxruntime as ort

        if cfg.model_path and cfg.model_path.endswith(".onnx"):
            model_path = Path(cfg.model_path)
        else:
            model_path = _download_model(RVM_MODEL_URL, "rvm_mobilenetv3_fp32.onnx")

        available = ort.get_available_providers()
        preferred = [
            p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider")
            if p in available
        ]
        options = ort.SessionOptions()
        options.log_severity_level = 3  # hide per-node provider assignment noise
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=preferred + ["CPUExecutionProvider"],
        )
        active = self._session.get_providers()[0]
        self.device = {
            "CUDAExecutionProvider": "cuda",
            "CoreMLExecutionProvider": "coreml",
        }.get(active, "cpu")
        self._downsample = cfg.rvm_downsample
        self._rec: list[np.ndarray] | None = None
        self._size: tuple[int, int] | None = None

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self._rec is None or self._size != (h, w):
            # Recurrent state is resolution-bound; reset on size changes.
            self._rec = [np.zeros((1, 1, 1, 1), dtype=np.float32)] * 4
            self._size = (h, w)
        if cv2 is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:  # pragma: no cover
            rgb = frame_bgr[..., ::-1]
        src = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        # Internal inference resolution: the model was trained to matte at a
        # reduced size and refine at full size; ~512 px on the long side is
        # the quality/speed sweet spot for webcam framing.
        ratio = self._downsample or min(1.0, max(0.125, 512.0 / max(h, w)))
        fgr, pha, *self._rec = self._session.run(
            None,
            {
                "src": src,
                "r1i": self._rec[0],
                "r2i": self._rec[1],
                "r3i": self._rec[2],
                "r4i": self._rec[3],
                "downsample_ratio": np.asarray([ratio], dtype=np.float32),
            },
        )
        fgr_rgb = np.clip(fgr[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        self.last_foreground = np.ascontiguousarray(fgr_rgb[..., ::-1])
        return np.ascontiguousarray(pha[0, 0].astype(np.float32))

    def close(self) -> None:
        self._session = None
        self._rec = None


def _watershed_edge_snap(
    mask: np.ndarray,
    frame_bgr: np.ndarray,
    radius: int = 8,
) -> np.ndarray:
    """Snap a coarse mask contour to nearby image edges within a bounded band.

    A guided filter can preserve an edge but cannot reliably move a displaced
    contour onto it. Marker watershed uses eroded foreground/background cores
    as immutable seeds and may move the boundary by at most ``radius`` pixels.
    """
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or not np.issubdtype(mask.dtype, np.number)
    ):
        return mask
    clipped = np.array(mask, dtype=np.float32, copy=True)
    np.nan_to_num(
        clipped,
        copy=False,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    np.clip(clipped, 0.0, 1.0, out=clipped)
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.dtype != np.uint8
        or radius <= 0
        or cv2 is None
    ):
        return clipped
    h, w = clipped.shape
    if h < 3 or w < 3 or frame_bgr.shape[:2] != (h, w):
        return clipped
    hard = (clipped >= 0.5).astype(np.uint8)
    if not hard.any() or hard.all():
        return clipped

    radius = min(radius, max(1, (min(h, w) - 1) // 4))
    try:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        sure_fg = cv2.erode(hard, kernel)
        sure_bg = cv2.erode(1 - hard, kernel)
        if not sure_fg.any() or not sure_bg.any():
            return clipped

        unknown = (sure_fg == 0) & (sure_bg == 0)
        # Erosion can remove a small connected component completely even when
        # another, larger component supplies the global foreground/background
        # marker. Watershed would then have no seed representing that component
        # and classify it away. Protect each seedless component and its bounded
        # uncertainty band independently.
        def seedless_components(
            binary: np.ndarray, seeds: np.ndarray
        ) -> np.ndarray:
            contours, hierarchy = cv2.findContours(
                binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if hierarchy is None:
                return np.zeros_like(binary, dtype=bool)
            hierarchy = hierarchy[0]
            components = [
                index for index, relation in enumerate(hierarchy) if relation[3] == -1
            ]
            if len(components) <= 1:
                # The global non-empty seed check above proves the sole
                # component owns a marker.
                return np.zeros_like(binary, dtype=bool)

            result = np.zeros_like(binary, dtype=np.uint8)
            for index in components:
                x, y, width, height = cv2.boundingRect(contours[index])
                seed_crop = seeds[y : y + height, x : x + width]
                component = np.zeros((height, width), dtype=np.uint8)
                offset = np.asarray([[[x, y]]], dtype=contours[index].dtype)
                cv2.drawContours(
                    component, [contours[index] - offset], -1, 1, cv2.FILLED
                )
                child = hierarchy[index][2]
                while child != -1:
                    cv2.drawContours(
                        component, [contours[child] - offset], -1, 0, cv2.FILLED
                    )
                    child = hierarchy[child][0]
                if not np.any(component & seed_crop):
                    result[y : y + height, x : x + width] |= component
            return result != 0

        seedless = seedless_components(hard, sure_fg) | seedless_components(
            1 - hard, sure_bg
        )
        if seedless.any():
            protected = cv2.dilate(seedless.astype(np.uint8), kernel) != 0
            unknown &= ~protected
        if not unknown.any():
            return clipped

        # Watershed only needs the bounded uncertainty band and one marker
        # margin on either side. Cropping avoids a full-frame watershed and
        # keeps the 720p median comfortably inside the real-time budget while
        # preserving every pixel outside the band verbatim.
        active_rows = np.flatnonzero(unknown.any(axis=1))
        active_cols = np.flatnonzero(unknown.any(axis=0))
        if active_rows.size == 0 or active_cols.size == 0:
            return clipped
        margin = 2
        y0 = max(0, int(active_rows[0]) - margin)
        y1 = min(h, int(active_rows[-1]) + margin + 1)
        x0 = max(0, int(active_cols[0]) - margin)
        x1 = min(w, int(active_cols[-1]) + margin + 1)
        unknown_crop = unknown[y0:y1, x0:x1]
        guide = cv2.GaussianBlur(frame_bgr[y0:y1, x0:x1], (3, 3), 0)
        # Watershed on a featureless image degenerates to a distance split
        # between markers. That invents an edge rather than snapping to one,
        # so preserve the segmenter's contour when the uncertainty band has
        # no visible contrast at all.
        if np.ptp(guide[unknown_crop].astype(np.int16), axis=0).max(initial=0) == 0:
            return clipped

        markers = np.zeros(unknown_crop.shape, dtype=np.int32)
        markers[sure_bg[y0:y1, x0:x1] != 0] = 1
        markers[sure_fg[y0:y1, x0:x1] != 0] = 2
        cv2.watershed(guide, markers)
    except cv2.error as exc:
        log.warning("edge watershed failed; retaining segmenter mask: %s", exc)
        return clipped

    refined = clipped.copy()
    band = refined[y0:y1, x0:x1]
    band[unknown_crop & (markers == 1)] = 0.0
    band[unknown_crop & (markers == 2)] = 1.0
    band[unknown_crop & (markers == -1)] = 0.5
    return refined


class MaskRefiner:
    """Post-processing shared by all backends: mask grow/shrink, edge
    feathering, edge-aware refinement and adaptive temporal smoothing."""

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._prev: np.ndarray | None = None

    def refine(self, mask: np.ndarray, frame_bgr: np.ndarray | None = None) -> np.ndarray:
        if cv2 is not None:
            # Snap first. A later user-requested grow/shrink must remain an
            # intentional halo-control offset rather than being undone here.
            if self.cfg.edge_refine and frame_bgr is not None:
                mask = _watershed_edge_snap(mask, frame_bgr)
            if self.cfg.mask_shift:
                r = abs(self.cfg.mask_shift)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)
                )
                op = cv2.dilate if self.cfg.mask_shift > 0 else cv2.erode
                mask = op(mask, kernel)
            if self.cfg.mask_blur:
                k = self.cfg.mask_blur
                mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Temporal smoothing, weighted by local change: static regions get the
        # full EMA (no flicker), moving edges track immediately (no ghosting).
        alpha = self.cfg.temporal_smoothing
        if alpha > 0 and self._prev is not None and self._prev.shape == mask.shape:
            diff = np.abs(mask - self._prev)
            if cv2 is not None:
                diff = cv2.blur(diff, (7, 7))
            keep = alpha * np.clip(1.0 - 4.0 * diff, 0.0, 1.0)
            mask = keep * self._prev + (1.0 - keep) * mask

        mask = np.clip(mask, 0.0, 1.0)
        self._prev = mask
        return mask

    def reset(self) -> None:
        self._prev = None


def refiner_for(cfg: SegmentationConfig, segmenter: Segmenter) -> MaskRefiner:
    """Build a refiner tuned to the segmenter: matting backends (rvm) already
    produce edge-accurate, temporally consistent alphas — re-feathering or
    re-smoothing them would only blur hair detail and add lag. mask_shift
    stays honored as the user's halo control."""
    if getattr(segmenter, "produces_matte", False):
        cfg = cfg.model_copy(
            update={
                "mask_blur": 0,
                "edge_refine": False,
                "temporal_smoothing": 0.0,
            },
            deep=True,
        )
    return MaskRefiner(cfg)


def create_segmenter(cfg: SegmentationConfig) -> Segmenter:
    backend = cfg.backend
    if backend == "none":
        return NullSegmenter()
    if backend in ("auto", "rvm"):
        try:
            seg = RVMSegmenter(cfg)
            log.info("using rvm matting backend on %s", seg.device)
            return seg
        except Exception as exc:
            if backend == "rvm":
                raise
            log.info("rvm backend unavailable (%s)", exc)
    if backend in ("auto", "mediapipe"):
        try:
            seg = MediaPipeSegmenter(cfg)
            log.info("using mediapipe segmentation backend on %s", seg.device)
            return seg
        except Exception as exc:
            if backend == "mediapipe":
                raise
            log.warning("mediapipe unavailable (%s); falling back to heuristic", exc)
    log.info("using heuristic segmentation backend")
    return HeuristicSegmenter(cfg)
