#!/usr/bin/env python3
"""Reproducible VIS-4.2 qualification and fail-closed evidence validation.

The harness has three deliberately distinct authorities:

* ``deterministic`` proves operation counts, visual metrics, pacing, and
  bounded retained state in shared CI without using wall-clock thresholds.
* ``calibrated`` records p50/p95/FPS/deadline/RSS results.  It is release
  evidence only when the caller identifies a pinned reference runner and runs
  the manifest's complete measured-frame/soak contract.
* ``physical`` is never synthesized here.  Operators attach hashed
  contact-sheet, preview/API, and real consumer captures for the exact
  cross-platform matrix.  Release validation rejects pending or absent rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import stat
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterable, NoReturn, TypeGuard

_LIGHTWEIGHT_CLI = __name__ == "__main__" and any(
    argument in ("validate", "template") for argument in sys.argv[1:]
)
if TYPE_CHECKING or not _LIGHTWEIGHT_CLI:
    import cv2
    import numpy as np

    import custback.backgrounds as backgrounds_mod
    import custback.color as color_mod
    import custback.compositor as compositor_mod
    import custback.geometry as geometry_mod
    import custback.pipeline as pipeline_mod
    from custback.backgrounds import BackdropProvider, CameraBackdrop, VideoBackdrop
    from custback.capture import CapturedFrame, CaptureHealth
    from custback.color import (
        ColorEstimate,
        ColorHarmonizer,
        apply_color_transform,
        estimate_color_transform_linear,
        linear_rgb_to_bgr_u8,
    )
    from custback.compositor import (
        composite,
        composite_linear_predecoded,
    )
    from custback.config import AppConfig, RuntimeConfig
    from custback.geometry import (
        apply_transform,
        clear_plan_cache,
        plan_cache_info,
        plan_transform,
    )
    from custback.hub import FrameHub
    from custback.pipeline import Pipeline
    from custback.vcam import NullOutput
else:
    cv2 = np = None
    backgrounds_mod = color_mod = compositor_mod = geometry_mod = pipeline_mod = None

    class _RuntimePlaceholder:
        pass

    BackdropProvider = CameraBackdrop = VideoBackdrop = _RuntimePlaceholder
    CapturedFrame = CaptureHealth = ColorEstimate = ColorHarmonizer = (
        _RuntimePlaceholder
    )
    AppConfig = RuntimeConfig = FrameHub = Pipeline = NullOutput = _RuntimePlaceholder
    apply_color_transform = estimate_color_transform_linear = None
    linear_rgb_to_bgr_u8 = composite = composite_linear_predecoded = None
    apply_transform = clear_plan_cache = plan_cache_info = plan_transform = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("visual-qualification-manifest.json")
REPORT_SCHEMA_VERSION = 1
REPORT_ID = "custback-visual-qualification-report-v1"
SHA256_RE_LENGTH = 64
SOURCE_FILES = (
    "config/default.yaml",
    "src/custback/default.yaml",
    "src/custback/api/server.py",
    "src/custback/backgrounds.py",
    "src/custback/capture.py",
    "src/custback/color.py",
    "src/custback/compositor.py",
    "src/custback/config.py",
    "src/custback/geometry.py",
    "src/custback/hub.py",
    "src/custback/pipeline.py",
    "src/custback/preview.py",
    "src/custback/vcam.py",
    "src/custback/vcam_native.py",
    "src/custback/video_decoder.py",
)
TIMING_STAGES = (
    "geometry_ms",
    "linear_conversion_ms",
    "color_analysis_ms",
    "composite_ms",
    "output_send_ms",
    "frame_total_ms",
)


class QualificationError(RuntimeError):
    """A manifest, qualification, or evidence contract failed."""


def _fail(message: str) -> NoReturn:
    raise QualificationError(message)


def _exact_keys(value: object, expected: Iterable[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        _fail(
            f"{label} has an invalid schema; missing={missing or 'none'}; "
            f"unexpected={unexpected or 'none'}"
        )
    return value


def _is_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_number(value: object, label: str) -> float:
    if not _is_number(value) or float(value) < 0.0:
        _fail(f"{label} must be a finite non-negative number")
    return float(value)


def _finite_number(value: object, label: str) -> float:
    if not _is_number(value):
        _fail(f"{label} must be a finite number")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        _fail(f"{label} must be a positive integer")
    return value


def _required_string(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > 512
    ):
        _fail(f"{label} must be a non-empty bounded string")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        _fail(f"evidence input must be a regular non-symlink file: {path.name}")
    return _sha256_bytes(path.read_bytes())


def _valid_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_RE_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _valid_git_oid(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase Git object id")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(manifest))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label} is invalid JSON: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object")
    return value


def _unique_ids(entries: object, label: str) -> list[str]:
    if not isinstance(entries, list) or not entries:
        _fail(f"{label} must be a non-empty array")
    ids: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            _fail(f"{label}[{index}] must be an object")
        ids.append(_required_string(entry.get("id"), f"{label}[{index}].id"))
    if len(set(ids)) != len(ids):
        _fail(f"{label} contains duplicate ids")
    return ids


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact reviewed VIS-4.2 requirements manifest."""

    _exact_keys(
        manifest,
        (
            "schema_version",
            "manifest_id",
            "baseline",
            "reference_runner",
            "performance_tiers",
            "budgets",
            "deterministic_contracts",
            "physical_matrix",
            "physical_requirements",
        ),
        "visual qualification manifest",
    )
    if manifest["schema_version"] != 1:
        _fail("visual qualification manifest schema_version must be 1")
    _required_string(manifest["manifest_id"], "manifest_id")

    baseline = _exact_keys(
        manifest["baseline"], ("path", "sha256"), "manifest.baseline"
    )
    baseline_path = _required_string(baseline["path"], "manifest.baseline.path")
    if baseline_path.startswith(("/", "\\")) or ".." in Path(baseline_path).parts:
        _fail("manifest.baseline.path must stay inside the source checkout")
    _valid_sha256(baseline["sha256"], "manifest.baseline.sha256")
    reference_runner = _exact_keys(
        manifest["reference_runner"],
        ("os_family", "rss_required"),
        "manifest.reference_runner",
    )
    if reference_runner["os_family"] != "linux":
        _fail("the reviewed reference runner must use Linux")
    if reference_runner["rss_required"] is not True:
        _fail("the reviewed reference runner must require RSS measurements")

    tier_ids = _unique_ids(manifest["performance_tiers"], "performance_tiers")
    if tier_ids != ["720p30", "720p60", "1080p30"]:
        _fail("performance tiers must be the reviewed 720p30/720p60/1080p30 set")
    for tier in manifest["performance_tiers"]:
        _exact_keys(
            tier,
            (
                "id",
                "width",
                "height",
                "fps",
                "minimum_measured_frames",
                "maximum_added_p95_ms",
            ),
            f"performance tier {tier['id']}",
        )
        _positive_int(tier["width"], f"{tier['id']}.width")
        _positive_int(tier["height"], f"{tier['id']}.height")
        _positive_int(tier["fps"], f"{tier['id']}.fps")
        if (
            _positive_int(
                tier["minimum_measured_frames"], f"{tier['id']}.minimum_measured_frames"
            )
            < 300
        ):
            _fail(f"{tier['id']} must measure at least 300 frames")
        _nonnegative_number(
            tier["maximum_added_p95_ms"], f"{tier['id']}.maximum_added_p95_ms"
        )

    budgets = _exact_keys(
        manifest["budgets"],
        (
            "maximum_relative_end_to_end_overhead_percent",
            "maximum_deadline_miss_increase_percentage_points",
            "minimum_fps_attainment_percent",
            "soak_frames",
            "maximum_retained_growth_bytes",
            "maximum_fixed_session_state_bytes",
            "privacy_history_storage_bytes",
            "analysis_long_edge_pixels",
            "maximum_geometry_resizes_common_path",
        ),
        "manifest.budgets",
    )
    for name in (
        "maximum_relative_end_to_end_overhead_percent",
        "maximum_deadline_miss_increase_percentage_points",
        "minimum_fps_attainment_percent",
    ):
        _nonnegative_number(budgets[name], f"manifest.budgets.{name}")
    if _positive_int(budgets["soak_frames"], "manifest.budgets.soak_frames") < 10_000:
        _fail("the reviewed soak must include at least 10,000 frames")
    for name in (
        "maximum_retained_growth_bytes",
        "maximum_fixed_session_state_bytes",
        "privacy_history_storage_bytes",
        "analysis_long_edge_pixels",
        "maximum_geometry_resizes_common_path",
    ):
        _positive_int(budgets[name], f"manifest.budgets.{name}")

    contracts = manifest["deterministic_contracts"]
    if (
        not isinstance(contracts, list)
        or not contracts
        or any(not isinstance(item, str) or not item for item in contracts)
        or len(set(contracts)) != len(contracts)
    ):
        _fail("deterministic_contracts must be a non-empty unique string array")

    row_ids = _unique_ids(manifest["physical_matrix"], "physical_matrix")
    expected_rows = [
        "linux-v4l2-pyvirtualcam",
        "macos-obsvcam-pyvirtualcam",
        "windows-msmf-pyvirtualcam",
        "windows-dshow-pyvirtualcam",
        "windows-native",
    ]
    if row_ids != expected_rows:
        _fail("physical_matrix does not match the reviewed cross-platform rows")
    for row in manifest["physical_matrix"]:
        _exact_keys(
            row,
            ("id", "os", "capture_backend", "output_backend", "transport"),
            f"physical row {row['id']}",
        )
        for name in ("os", "capture_backend", "output_backend", "transport"):
            _required_string(row[name], f"{row['id']}.{name}")

    physical = _exact_keys(
        manifest["physical_requirements"],
        (
            "minimum_distinct_webcams",
            "minimum_distinct_meeting_consumers",
            "require_different_auto_control_profiles",
            "maximum_crop_edge_error_pixels",
            "maximum_orientation_error_degrees",
            "maximum_transport_mean_absolute_error",
            "minimum_luminance_gap_reduction_percent",
            "minimum_neutral_error_reduction_percent",
            "maximum_skin_hue_drift_degrees",
            "maximum_skin_chroma_drift_percent",
            "maximum_clothing_hue_drift_degrees",
            "maximum_clothing_chroma_drift_percent",
            "maximum_steady_state_ev_delta_p95",
            "maximum_steady_state_wb_log2_delta_p95",
            "required_artifacts",
        ),
        "manifest.physical_requirements",
    )
    _positive_int(
        physical["minimum_distinct_webcams"],
        "physical_requirements.minimum_distinct_webcams",
    )
    _positive_int(
        physical["minimum_distinct_meeting_consumers"],
        "physical_requirements.minimum_distinct_meeting_consumers",
    )
    if physical["require_different_auto_control_profiles"] is not True:
        _fail("physical qualification must require different auto-control profiles")
    for name in (
        "maximum_crop_edge_error_pixels",
        "maximum_orientation_error_degrees",
        "maximum_transport_mean_absolute_error",
        "minimum_luminance_gap_reduction_percent",
        "minimum_neutral_error_reduction_percent",
        "maximum_skin_hue_drift_degrees",
        "maximum_skin_chroma_drift_percent",
        "maximum_clothing_hue_drift_degrees",
        "maximum_clothing_chroma_drift_percent",
        "maximum_steady_state_ev_delta_p95",
        "maximum_steady_state_wb_log2_delta_p95",
    ):
        _nonnegative_number(physical[name], f"physical_requirements.{name}")
    required_artifacts = physical["required_artifacts"]
    if (
        not isinstance(required_artifacts, dict)
        or len(required_artifacts) < 4
        or any(
            not isinstance(item, str) or not item
            for item in (*required_artifacts.keys(), *required_artifacts.values())
        )
        or set(required_artifacts.values()) != {"image/png", "application/json"}
    ):
        _fail("physical required_artifacts must map exact ids to reviewed media types")
    return manifest


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = validate_manifest(_read_json(path, "visual qualification manifest"))
    baseline_path = ROOT / manifest["baseline"]["path"]
    actual = _sha256_file(baseline_path)
    if actual != manifest["baseline"]["sha256"]:
        _fail(
            "checked-in VIS-0.2 baseline digest differs from the reviewed "
            "qualification manifest"
        )
    return manifest


def _phase0_module() -> ModuleType:
    path = ROOT / "tests" / "visual_consistency_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "custback_phase0_visual_consistency_evidence", path
    )
    if spec is None or spec.loader is None:
        _fail("cannot load the checked-in VIS-0.2 evidence implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _reduction(before: float, after: float) -> float:
    return 100.0 * (before - after) / max(before, 1e-9)


def _visual_contract(
    manifest: dict[str, Any],
    contact_sheet_path: Path,
) -> dict[str, Any]:
    """Compare production geometry/color against the authoritative Phase-0 corpus."""

    baseline = _read_json(
        ROOT / manifest["baseline"]["path"], "VIS-0.2 baseline evidence"
    )
    phase0 = _phase0_module()

    source = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(source, (160, 120), 40, (255, 255, 255), thickness=-1)
    cover_plan = plan_transform((320, 240), (320, 180), fit="cover")
    cover = apply_transform(source, cover_plan)
    stretch = apply_transform(
        source,
        plan_transform((320, 240), (320, 180), fit="stretch"),
    )
    points = np.argwhere(cover[..., 0] >= 250)
    if points.size == 0:
        _fail("production cover transform lost the visual geometry target")
    height = int(points[:, 0].max() - points[:, 0].min() + 1)
    width = int(points[:, 1].max() - points[:, 1].min() + 1)
    axis_ratio = width / height
    distortion = abs(axis_ratio - 1.0) * 100.0
    baseline_distortion = float(
        baseline["current_baseline"]["geometry"]["distortion_percent"]
    )

    scene = phase0.synthetic_scene()
    estimate = estimate_color_transform_linear(
        scene.foreground,
        scene.backdrop,
        scene.mask,
        mode="image",
        strength=0.5,
        exposure_limit_ev=0.85,
        white_balance_strength=0.5,
    )
    if not isinstance(estimate, ColorEstimate) or not estimate.reliable:
        _fail("production estimator did not produce a reliable Phase-0 estimate")
    corrected = apply_color_transform(scene.foreground, estimate.transform)
    foreground_region, backdrop_region = phase0._metric_regions(scene)
    before_gap = phase0.log_luminance_gap(
        scene.foreground,
        scene.backdrop,
        foreground_region,
        backdrop_region,
    )
    after_gap = phase0.log_luminance_gap(
        corrected,
        scene.backdrop,
        foreground_region,
        backdrop_region,
    )
    before_neutral = phase0.neutral_axis_error(
        scene.foreground,
        scene.backdrop,
        foreground_region,
        backdrop_region,
    )
    after_neutral = phase0.neutral_axis_error(
        corrected,
        scene.backdrop,
        foreground_region,
        backdrop_region,
    )
    skin = phase0.patch_preservation(scene.foreground, corrected, scene.skin_mask)
    clothing = phase0.patch_preservation(
        scene.foreground, corrected, scene.clothing_mask
    )
    luminance_reduction = _reduction(before_gap, after_gap)
    neutral_reduction = _reduction(before_neutral, after_neutral)
    phase0_bounded = baseline["algorithm_comparison"]["bounded_wb"]
    recommendation = baseline["recommendation"]

    foreground = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    backdrop = np.ascontiguousarray(255 - foreground)
    endpoint_mask = np.zeros((12, 16), dtype=np.float32)
    endpoint_mask[:, 8:] = 1.0
    endpoint = composite(
        foreground,
        backdrop,
        endpoint_mask,
        blend_space="linear_srgb",
    )
    foreground_delta = int(
        np.abs(
            endpoint[endpoint_mask == 1.0].astype(np.int16)
            - foreground[endpoint_mask == 1.0].astype(np.int16)
        ).max()
    )
    backdrop_delta = int(
        np.abs(
            endpoint[endpoint_mask == 0.0].astype(np.int16)
            - backdrop[endpoint_mask == 0.0].astype(np.int16)
        ).max()
    )

    checks = {
        "geometry_improves_phase0": distortion < baseline_distortion,
        "luminance_not_worse_than_phase0_selected": (
            luminance_reduction + 0.5
            >= float(phase0_bounded["luminance_gap_ev"]["reduction_percent"])
        ),
        "neutral_axis_not_worse_than_phase0_selected": (
            neutral_reduction + 0.5
            >= float(
                phase0_bounded["neutral_axis_error_delta_e_ab"]["reduction_percent"]
            )
        ),
        "skin_hue_preserved": (
            skin["hue_drift_degrees"]
            <= float(recommendation["skin_hue_drift_max_degrees"])
        ),
        "skin_chroma_preserved": (
            skin["normalized_chroma_drift_percent"]
            <= float(recommendation["skin_normalized_chroma_drift_max_percent"])
        ),
        "clothing_hue_preserved": (
            clothing["hue_drift_degrees"]
            <= float(recommendation["clothing_hue_drift_max_degrees"])
        ),
        "clothing_chroma_preserved": (
            clothing["normalized_chroma_drift_percent"]
            <= float(recommendation["clothing_normalized_chroma_drift_max_percent"])
        ),
        "foreground_endpoint_byte_exact": foreground_delta == 0,
        "backdrop_endpoint_byte_exact": backdrop_delta == 0,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        _fail(f"production visual metrics failed: {', '.join(failed)}")

    foreground_bgr = linear_rgb_to_bgr_u8(scene.foreground)
    backdrop_bgr = linear_rgb_to_bgr_u8(scene.backdrop)
    corrected_bgr = linear_rgb_to_bgr_u8(corrected)
    composited = composite_linear_predecoded(
        foreground_bgr,
        backdrop_bgr,
        scene.mask,
        foreground_linear_rgb=scene.foreground,
        backdrop_linear_rgb=scene.backdrop,
        color_transform=estimate.transform,
    )
    resized_stretch = cv2.resize(stretch, (320, 180), interpolation=cv2.INTER_NEAREST)
    panels = (
        resized_stretch,
        cover,
        foreground_bgr,
        corrected_bgr,
        backdrop_bgr,
        composited,
    )
    contact_sheet = np.ascontiguousarray(np.concatenate(panels, axis=1))
    if contact_sheet_path.exists():
        _fail(f"contact sheet already exists: {contact_sheet_path.name}")
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(contact_sheet_path), contact_sheet):
        _fail("cannot write deterministic contact sheet")
    contact_bytes = contact_sheet_path.read_bytes()

    return {
        "status": "pass",
        "checks": checks,
        "geometry": {
            "phase0_distortion_percent": _round(baseline_distortion, 3),
            "production_axis_ratio": _round(axis_ratio),
            "production_distortion_percent": _round(distortion, 3),
            "crop_rect": [
                cover_plan.crop_rect.left,
                cover_plan.crop_rect.top,
                cover_plan.crop_rect.right,
                cover_plan.crop_rect.bottom,
            ],
        },
        "color": {
            "behavior": estimate.behavior.value,
            "confidence": _round(estimate.confidence),
            "luminance_gap_ev": {
                "before": _round(before_gap),
                "after": _round(after_gap),
                "reduction_percent": _round(luminance_reduction, 3),
                "phase0_selected_reduction_percent": _round(
                    phase0_bounded["luminance_gap_ev"]["reduction_percent"], 3
                ),
            },
            "neutral_axis_error_delta_e_ab": {
                "before": _round(before_neutral),
                "after": _round(after_neutral),
                "reduction_percent": _round(neutral_reduction, 3),
                "phase0_selected_reduction_percent": _round(
                    phase0_bounded["neutral_axis_error_delta_e_ab"][
                        "reduction_percent"
                    ],
                    3,
                ),
            },
            "skin_preservation": {name: _round(value) for name, value in skin.items()},
            "clothing_preservation": {
                name: _round(value) for name, value in clothing.items()
            },
            "foreground_endpoint_max_delta": foreground_delta,
            "backdrop_endpoint_max_delta": backdrop_delta,
        },
        "contact_sheet": {
            "filename": contact_sheet_path.name,
            "sha256": _sha256_bytes(contact_bytes),
            "bytes": len(contact_bytes),
            "width": int(contact_sheet.shape[1]),
            "height": int(contact_sheet.shape[0]),
        },
    }


class _QualificationBackdrop(BackdropProvider):
    def __init__(self, pixels: np.ndarray):
        super().__init__(fit_mode="cover")
        self.pixels = np.ascontiguousarray(pixels)
        self._cached: np.ndarray | None = None

    def frame(self, width: int, height: int) -> np.ndarray:
        if self._cached is None:
            self._cached = self._fit_frame(self.pixels, width, height)
        return self._cached

    def _invalidate_geometry_cache(self) -> None:
        self._cached = None


class _QualificationSegmenter:
    device = "cpu"
    produces_matte = False
    last_foreground = None

    def __init__(self, mask: np.ndarray):
        self.mask = np.ascontiguousarray(mask, dtype=np.float32)

    def segment(self, _frame: np.ndarray) -> np.ndarray:
        return self.mask.copy()

    def close(self) -> None:
        return None


class _IdentityRefiner:
    def refine(self, mask: np.ndarray, _frame: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(mask, dtype=np.float32)


class _SequenceCapture:
    def __init__(
        self,
        frames: list[np.ndarray | None],
        width: int,
        height: int,
        fps: int,
    ):
        self.frames = list(frames)
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_read = 0

    def read(self) -> CapturedFrame | None:
        value = self.frames.pop(0) if self.frames else None
        if value is None:
            return None
        self.frames_read += 1
        return CapturedFrame(
            pixels=value.copy(),
            sequence=self.frames_read,
            captured_at_ns=self.frames_read * 1_000_000_000 // self.fps,
            generation=0,
            geometry_generation=0,
            content_rect=(0, 0, self.width, self.height),
        )

    def health_snapshot(self) -> CaptureHealth:
        return CaptureHealth(
            sequence=self.frames_read,
            captured_monotonic_ns=(
                None
                if self.frames_read == 0
                else self.frames_read * 1_000_000_000 // self.fps
            ),
            generation=0,
            geometry_generation=0,
            content_rect=(0, 0, self.width, self.height),
            backend="qualification",
            width=self.width,
            height=self.height,
            normalized_width=self.width,
            normalized_height=self.height,
            fps_reported=float(self.fps),
            frames_read=self.frames_read,
        )

    def close(self) -> None:
        return None


class _StopAfterOutput:
    paces = True
    fallback_active = False
    fallback_reason = ""

    def __init__(self, pipeline: Pipeline, count: int):
        self.pipeline = pipeline
        self.count = count
        self.frames_sent = 0

    def send(self, _frame: np.ndarray) -> None:
        self.frames_sent += 1
        if self.frames_sent >= self.count:
            self.pipeline._stop.set()

    def close(self) -> None:
        return None


class _BoundedNullOutput(NullOutput):
    paces = False

    def __init__(
        self,
        pipeline: Pipeline,
        count: int,
        width: int,
        height: int,
        fps: int,
    ):
        super().__init__(width, height, fps)
        self.pipeline = pipeline
        self.count = count

    def send(self, frame_bgr: np.ndarray) -> None:
        super().send(frame_bgr)
        if self.frames_sent >= self.count:
            self.pipeline._stop.set()


class _FakeVideoCapture:
    """Small deterministic VideoCapture seam for scheduler qualification."""

    def __init__(self, values: Iterable[int], fps: float = 10.0):
        self.frames = [np.full((4, 6, 3), value, dtype=np.uint8) for value in values]
        self.fps = fps
        self.pos = 0
        self.last_index: int | None = None
        self.grabbed: np.ndarray | None = None
        self.read_calls = 0
        self.grab_calls = 0

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return (
                float("nan")
                if self.last_index is None
                else self.last_index / self.fps * 1000.0
            )
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 6.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 4.0
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = max(0, min(len(self.frames), int(value)))
            return True
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_calls += 1
        if self.pos >= len(self.frames):
            return False, None
        index = self.pos
        self.pos += 1
        self.last_index = index
        return True, self.frames[index].copy()

    def grab(self) -> bool:
        self.grab_calls += 1
        if self.pos >= len(self.frames):
            self.grabbed = None
            return False
        index = self.pos
        self.pos += 1
        self.last_index = index
        self.grabbed = self.frames[index].copy()
        return True

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        if self.grabbed is None:
            return False, None
        return True, self.grabbed.copy()


def _repeat_operation_counts() -> dict[str, Any]:
    config = AppConfig.from_dict(
        {
            "camera": {"width": 32, "height": 24, "fps": 60},
            "background": {
                "mode": "image",
                "image_path": "/qualification/not-opened.png",
            },
            "segmentation": {
                "backend": "heuristic",
                "temporal_smoothing": 0.0,
                "edge_refine": False,
                "mask_blur": 0,
            },
            "compositing": {
                "blend_space": "linear_srgb",
                "light_wrap": 0.0,
                "use_model_foreground": False,
                "color_correction": {"mode": "auto"},
            },
            "output": {"backend": "null", "fps": 60},
            "api": {"enabled": False},
        }
    )
    y, x = np.indices((24, 32))
    raw = np.stack(
        (
            (x * 7 + y * 3) % 251,
            (x * 2 + y * 11) % 251,
            (x * 13 + y * 5) % 251,
        ),
        axis=2,
    ).astype(np.uint8)
    mask = np.full((24, 32), 0.65, dtype=np.float32)
    background = np.full((24, 32, 3), (80, 100, 120), dtype=np.uint8)
    pipeline = Pipeline(RuntimeConfig(config), FrameHub())
    output = _StopAfterOutput(pipeline, 6)
    resources = pipeline_mod._Resources(
        config,
        0,
        _SequenceCapture([raw, None, None, None, None, None], 32, 24, 60),
        _QualificationSegmenter(mask),
        _IdentityRefiner(),
        _QualificationBackdrop(background),
        output,
        harmonizer=ColorHarmonizer(0.8, mode="image"),
    )
    counts = {"analysis": 0, "linear_decode": 0, "composite": 0}
    real_analysis = pipeline_mod.estimate_color_transform_linear
    real_decode = pipeline_mod.bgr_u8_to_linear_rgb
    real_composite = pipeline_mod.composite_linear_predecoded

    def counted_analysis(*args: Any, **kwargs: Any) -> Any:
        counts["analysis"] += 1
        return real_analysis(*args, **kwargs)

    def counted_decode(*args: Any, **kwargs: Any) -> Any:
        counts["linear_decode"] += 1
        return real_decode(*args, **kwargs)

    def counted_composite(*args: Any, **kwargs: Any) -> Any:
        counts["composite"] += 1
        return real_composite(*args, **kwargs)

    pipeline_mod.estimate_color_transform_linear = counted_analysis
    pipeline_mod.bgr_u8_to_linear_rgb = counted_decode
    pipeline_mod.composite_linear_predecoded = counted_composite
    try:
        pipeline._loop(resources)
    finally:
        pipeline_mod.estimate_color_transform_linear = real_analysis
        pipeline_mod.bgr_u8_to_linear_rgb = real_decode
        pipeline_mod.composite_linear_predecoded = real_composite
    stats = pipeline.hub.stats_dict()
    result = {
        "source_frames": 1,
        "output_frames": output.frames_sent,
        "repeated_output_frames": stats["output_repeated_frames"],
        "color_analysis_calls": counts["analysis"],
        "linear_decode_calls": counts["linear_decode"],
        "composite_calls": counts["composite"],
    }
    if result != {
        "source_frames": 1,
        "output_frames": 6,
        "repeated_output_frames": 5,
        "color_analysis_calls": 1,
        "linear_decode_calls": 2,
        "composite_calls": 1,
    }:
        _fail(f"repeated output operation contract changed: {result}")
    return result


def _video_pacing_counts() -> dict[str, Any]:
    capture = _FakeVideoCapture(range(10), fps=10.0)
    now = [0.0]
    original_constructor = backgrounds_mod.cv2.VideoCapture
    backgrounds_mod.cv2.VideoCapture = lambda _path: capture
    try:
        backdrop = VideoBackdrop("qualification.avi", clock=lambda: now[0])
        try:
            first = backdrop.frame(6, 4).copy()
            reads_after_first = capture.read_calls
            for _ in range(4):
                if not np.array_equal(backdrop.frame(6, 4), first):
                    _fail("video pacing did not reuse the frame before its deadline")
            reads_after_repeats = capture.read_calls
            now[0] = 0.1
            second = backdrop.frame(6, 4)
            stats = backdrop.stats_dict()
        finally:
            backdrop.close()
    finally:
        backgrounds_mod.cv2.VideoCapture = original_constructor
    video = {
        "calls_at_same_deadline": 5,
        "decoder_reads_during_early_repeats": (reads_after_repeats - reads_after_first),
        "displayed_frames": stats["background_video_frames_displayed"],
        "reused_frames": stats["background_video_frames_reused"],
        "second_frame_changed": not np.array_equal(second, first),
    }
    if video != {
        "calls_at_same_deadline": 5,
        "decoder_reads_during_early_repeats": 0,
        "displayed_frames": 2,
        "reused_frames": 4,
        "second_frame_changed": True,
    }:
        _fail(f"video pacing operation contract changed: {video}")

    camera_capture = _FakeVideoCapture(range(5), fps=30.0)
    backgrounds_mod.cv2.VideoCapture = lambda _device: camera_capture
    try:
        camera = CameraBackdrop(1)
        try:
            returned = sum(
                camera.frame(6, 4).shape == (4, 6, 3) for _request in range(5)
            )
        finally:
            camera.close()
    finally:
        backgrounds_mod.cv2.VideoCapture = original_constructor
    camera_counts = {
        "pipeline_requests": 5,
        "capture_reads": camera_capture.read_calls,
        "frames_returned": returned,
        "prefetched_frames": camera_capture.read_calls - 5,
    }
    if camera_counts != {
        "pipeline_requests": 5,
        "capture_reads": 5,
        "frames_returned": 5,
        "prefetched_frames": 0,
    }:
        _fail(f"camera pacing operation contract changed: {camera_counts}")
    return {"video": video, "camera": camera_counts}


def _resize_operation_counts(manifest: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    real_resize = geometry_mod._resize

    def counted_resize(frame: np.ndarray, step: Any) -> np.ndarray:
        counts["active"] += 1
        return real_resize(frame, step)

    geometry_mod._resize = counted_resize
    try:
        for label, source_size, target_size in (
            ("720p", (960, 720), (1280, 720)),
            ("1080p", (1440, 1080), (1920, 1080)),
        ):
            counts["active"] = 0
            source = np.zeros((source_size[1], source_size[0], 3), dtype=np.uint8)
            plan = plan_transform(source_size, target_size, fit="cover")
            output = apply_transform(source, plan)
            if output.shape != (target_size[1], target_size[0], 3):
                _fail(f"{label} geometry output has the wrong shape")
            counts[label] = counts["active"]
            if (
                counts[label]
                > manifest["budgets"]["maximum_geometry_resizes_common_path"]
            ):
                _fail(f"{label} common path performs duplicate full-frame resizes")
    finally:
        geometry_mod._resize = real_resize
    counts.pop("active", None)
    return counts


def _analysis_caps(manifest: dict[str, Any]) -> dict[str, list[int]]:
    expected = manifest["budgets"]["analysis_long_edge_pixels"]
    result: dict[str, list[int]] = {}
    for tier in manifest["performance_tiers"]:
        size = color_mod._analysis_size(tier["width"], tier["height"])
        if max(size) > expected:
            _fail(f"{tier['id']} analysis exceeds the reviewed long-edge cap")
        result[tier["id"]] = [int(size[0]), int(size[1])]
    return result


def _process_rss_bytes() -> int | None:
    """Return current RSS without adding a non-project dependency."""

    if sys.platform.startswith("linux"):
        try:
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return value if sys.platform == "darwin" else value * 1024


def _soak_contract(
    manifest: dict[str, Any],
    *,
    frames: int,
) -> dict[str, Any]:
    """Exercise bounded temporal/privacy/geometry state over a long session."""

    if frames < 1:
        _fail("soak frame count must be positive")
    warmup = min(512, max(32, frames // 10))
    config = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 32, "height": 24, "fps": 60},
            "output": {"backend": "null", "fps": 60},
            "api": {"enabled": False},
        }
    )
    # Intentionally use the production default million-frame fail-closed
    # replay capacity.  A smaller qualification-only capacity would hide its
    # fixed 16 MiB Bloom allocation.
    pipeline = Pipeline(RuntimeConfig(config), FrameHub())
    phase0 = _phase0_module()
    scene = phase0.synthetic_scene()
    estimate = estimate_color_transform_linear(
        scene.foreground,
        scene.backdrop,
        scene.mask,
        mode="image",
    )
    if not estimate.reliable:
        _fail("cannot seed the soak harmonizer with a reliable estimate")
    harmonizer = ColorHarmonizer(0.8, mode="image")
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    maximum_tier = max(
        manifest["performance_tiers"],
        key=lambda tier: tier["width"] * tier["height"],
    )
    maximum_frame = np.zeros(
        (maximum_tier["height"], maximum_tier["width"], 3), dtype=np.uint8
    )

    def one(index: int) -> None:
        frame[..., 0] = index % 251
        frame[..., 1] = (index * 7) % 251
        frame[..., 2] = (index * 17) % 251
        pipeline._remember_raw_frame(frame)
        harmonizer.update(estimate, index / 60.0, source_generation=0)
        dimension = 16 + index
        plan_transform((dimension, dimension + 1), (32, 24), fit="cover")

    clear_plan_cache()
    tracemalloc.start(10)
    try:
        initial_traced, _initial_peak = tracemalloc.get_traced_memory()
        initial_rss = _process_rss_bytes()
        pipeline._remember_raw_frame(maximum_frame)
        for index in range(warmup):
            one(index)
        gc.collect()
        before_traced, _before_peak = tracemalloc.get_traced_memory()
        fixed_rss = _process_rss_bytes()
        tracemalloc.reset_peak()
        before_rss = _process_rss_bytes()
        for offset in range(frames):
            one(warmup + offset)
        gc.collect()
        after_traced, peak_traced = tracemalloc.get_traced_memory()
        after_rss = _process_rss_bytes()
    finally:
        tracemalloc.stop()

    history = pipeline._recent_raw_fingerprints
    bits = history._bits
    cache = plan_cache_info()
    fixed_traced_growth = max(0, before_traced - initial_traced)
    fixed_rss_growth = (
        None
        if initial_rss is None or fixed_rss is None
        else max(0, fixed_rss - initial_rss)
    )
    traced_growth = max(0, after_traced - before_traced)
    rss_growth = (
        None
        if before_rss is None or after_rss is None
        else max(0, after_rss - before_rss)
    )
    limit = manifest["budgets"]["maximum_retained_growth_bytes"]
    fixed_limit = manifest["budgets"]["maximum_fixed_session_state_bytes"]
    history_bytes = 0 if bits is None else len(bits)
    analysis_long_edge = color_mod.ANALYSIS_LONG_EDGE
    if analysis_long_edge != manifest["budgets"]["analysis_long_edge_pixels"]:
        _fail("production and manifest color-analysis bounds differ")
    maximum_color_analysis_cache_bytes = (
        analysis_long_edge**2 * 3 * np.dtype(np.float32).itemsize
    )
    fixed_semantic_bytes = (
        history_bytes + maximum_frame.nbytes + maximum_color_analysis_cache_bytes
    )
    checks = {
        "fixed_session_state_within_limit": fixed_semantic_bytes <= fixed_limit,
        "tracemalloc_growth_within_limit": traced_growth <= limit,
        "rss_growth_within_limit": rss_growth is None or rss_growth <= limit,
        "history_storage_is_fixed": (
            bits is not None
            and len(bits) == history._byte_count
            and len(bits) == pipeline_mod._RawReplayHistory._MAX_BYTES
        ),
        "geometry_cache_is_bounded": (
            getattr(cache, "currsize", 513) <= getattr(cache, "maxsize", 512)
        ),
        "latest_frame_is_single_canvas": (
            pipeline._latest_raw_frame is not None
            and pipeline._latest_raw_frame.nbytes == frame.nbytes
        ),
        "harmonizer_has_scalar_snapshot": (
            harmonizer.snapshot().source_generation == 0
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        _fail(f"retained-state soak failed: {', '.join(failed)}")
    return {
        "status": "pass",
        "frames": frames,
        "warmup_frames": warmup,
        "fixed_session_state_semantic_bytes": fixed_semantic_bytes,
        "fixed_session_state_limit_bytes": fixed_limit,
        "fixed_tracemalloc_growth_bytes": fixed_traced_growth,
        "fixed_rss_growth_bytes": fixed_rss_growth,
        "tracemalloc_current_growth_bytes": traced_growth,
        "tracemalloc_peak_bytes": peak_traced,
        "rss_growth_bytes": rss_growth,
        "retained_growth_limit_bytes": limit,
        "privacy_history_entries": len(history),
        "privacy_history_storage_bytes": history_bytes,
        "maximum_canvas_frame_bytes": maximum_frame.nbytes,
        "maximum_color_analysis_cache_bytes": maximum_color_analysis_cache_bytes,
        "latest_frame_bytes": frame.nbytes,
        "geometry_cache_entries": getattr(cache, "currsize", None),
        "geometry_cache_max_entries": getattr(cache, "maxsize", None),
        "checks": checks,
    }


def _deterministic_contract(
    manifest: dict[str, Any],
    *,
    contact_sheet_path: Path,
    soak_frames: int,
) -> dict[str, Any]:
    visual = _visual_contract(manifest, contact_sheet_path)
    resize_counts = _resize_operation_counts(manifest)
    analysis_sizes = _analysis_caps(manifest)
    repeated = _repeat_operation_counts()
    pacing = _video_pacing_counts()
    soak = _soak_contract(manifest, frames=soak_frames)
    results = {
        "phase0-visual-improvement": {
            "status": "pass",
            "metrics": {
                "geometry": visual["geometry"],
                "color": visual["color"],
            },
        },
        "foreground-preservation": {
            "status": "pass",
            "metrics": visual["checks"],
        },
        "common-path-single-resize": {
            "status": "pass",
            "metrics": resize_counts,
        },
        "analysis-resolution-cap": {
            "status": "pass",
            "metrics": analysis_sizes,
        },
        "repeated-output-work-skip": {
            "status": "pass",
            "metrics": repeated,
        },
        "backdrop-source-pacing": {
            "status": "pass",
            "metrics": pacing,
        },
        "bounded-session-history": {
            "status": "pass",
            "metrics": soak,
        },
    }
    if set(results) != set(manifest["deterministic_contracts"]):
        _fail("deterministic result set differs from the reviewed manifest")
    digest_payload = {
        "baseline_sha256": manifest["baseline"]["sha256"],
        "contracts": results,
        "contact_sheet_sha256": visual["contact_sheet"]["sha256"],
    }
    return {
        "status": "pass",
        "authority": "shared-ci-deterministic",
        "contracts": results,
        "contact_sheet": visual["contact_sheet"],
        "soak": soak,
        "deterministic_sha256": _sha256_bytes(_canonical_json(digest_payload)),
    }


def _percentile_value(values: list[float], percentile: float) -> float:
    if not values:
        _fail("timing sample is empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50": _round(_percentile_value(values, 50), 4),
        "p95": _round(_percentile_value(values, 95), 4),
    }


def _ellipse_mask(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.float32)
    normalized = ((x - width * 0.5) / max(1.0, width * 0.23)) ** 2 + (
        (y - height * 0.52) / max(1.0, height * 0.42)
    ) ** 2
    return np.ascontiguousarray(np.clip((1.05 - np.sqrt(normalized)) / 0.08, 0, 1))


def _tier_arrays(width: int, height: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(width * 10_000 + height)
    source_width = width * 3 // 4
    geometry_source = rng.integers(0, 256, (height, source_width, 3), dtype=np.uint8)
    foreground = rng.integers(24, 224, (height, width, 3), dtype=np.uint8)
    backdrop = rng.integers(24, 224, (height, width, 3), dtype=np.uint8)
    mask = _ellipse_mask(height, width).astype(np.float32)
    return geometry_source, foreground, backdrop, mask


def _one_baseline_frame(
    geometry_source: np.ndarray,
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    plan: Any,
    output: NullOutput,
) -> dict[str, float]:
    started = time.perf_counter_ns()
    stage = time.perf_counter_ns()
    apply_transform(geometry_source, plan)
    geometry_done = time.perf_counter_ns()
    rendered = composite(
        foreground,
        backdrop,
        mask,
        blend_space="srgb_legacy",
    )
    composite_done = time.perf_counter_ns()
    output.send(rendered)
    sent = time.perf_counter_ns()
    return {
        "geometry_ms": (geometry_done - stage) / 1_000_000.0,
        "linear_conversion_ms": 0.0,
        "color_analysis_ms": 0.0,
        "composite_ms": (composite_done - geometry_done) / 1_000_000.0,
        "output_send_ms": (sent - composite_done) / 1_000_000.0,
        "frame_total_ms": (sent - started) / 1_000_000.0,
    }


def _one_candidate_frame(
    geometry_source: np.ndarray,
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    plan: Any,
    output: NullOutput,
    *,
    resize_executor: Executor,
    backdrop_analysis_linear_bgr: np.ndarray,
) -> dict[str, float]:
    started = time.perf_counter_ns()
    apply_transform(geometry_source, plan)
    geometry_done = time.perf_counter_ns()
    foreground_linear = color_mod._bgr_u8_to_linear_bgr_prevalidated(foreground)
    backdrop_linear = color_mod._bgr_u8_to_linear_bgr_prevalidated(backdrop)
    conversion_done = time.perf_counter_ns()
    estimate = color_mod._estimate_color_transform_linear_bgr_prevalidated(
        foreground_linear,
        backdrop_linear,
        mask,
        mode="image",
        resize_executor=resize_executor,
        backdrop_analysis_linear_bgr=backdrop_analysis_linear_bgr,
    )
    analysis_done = time.perf_counter_ns()
    rendered = compositor_mod._composite_linear_bgr_prevalidated(
        foreground,
        backdrop,
        mask,
        foreground_linear_bgr=foreground_linear,
        backdrop_linear_bgr=backdrop_linear,
        color_transform=estimate.transform,
    )
    composite_done = time.perf_counter_ns()
    output.send(rendered)
    sent = time.perf_counter_ns()
    return {
        "geometry_ms": (geometry_done - started) / 1_000_000.0,
        "linear_conversion_ms": (conversion_done - geometry_done) / 1_000_000.0,
        "color_analysis_ms": (analysis_done - conversion_done) / 1_000_000.0,
        "composite_ms": (composite_done - analysis_done) / 1_000_000.0,
        "output_send_ms": (sent - composite_done) / 1_000_000.0,
        "frame_total_ms": (sent - started) / 1_000_000.0,
    }


def _production_ewma_run(
    *,
    width: int,
    height: int,
    fps: int,
    frames: int,
    foreground: np.ndarray,
    geometry_source: np.ndarray,
    mask: np.ndarray,
    candidate: bool,
) -> dict[str, Any]:
    config = AppConfig.from_dict(
        {
            "camera": {
                "width": width,
                "height": height,
                "fps": fps,
                "fit_mode": "cover",
            },
            "background": {
                "mode": "image",
                "image_path": "/qualification/not-opened.png",
                "fit_mode": "cover",
            },
            "segmentation": {
                "backend": "heuristic",
                "temporal_smoothing": 0.0,
                "edge_refine": False,
                "mask_blur": 0,
            },
            "compositing": {
                "blend_space": "linear_srgb" if candidate else "srgb_legacy",
                "light_wrap": 0.0,
                "use_model_foreground": False,
                "color_correction": {"mode": "auto" if candidate else "off"},
            },
            "output": {
                "width": width,
                "height": height,
                "backend": "null",
                "fps": fps,
            },
            "api": {"enabled": False},
        }
    )
    pipeline = Pipeline(RuntimeConfig(config), FrameHub())
    output = _BoundedNullOutput(pipeline, frames, width, height, fps)
    resources = pipeline_mod._Resources(
        config,
        0,
        _SequenceCapture([foreground] * frames, width, height, fps),
        _QualificationSegmenter(mask),
        _IdentityRefiner(),
        _QualificationBackdrop(geometry_source),
        output,
        harmonizer=(ColorHarmonizer(0.8, mode="image") if candidate else None),
    )
    try:
        pipeline._loop(resources)
    finally:
        resources.close()
    stats = pipeline.hub.stats_dict()
    fields = (
        "fps",
        "fps_attainment_pct",
        "output_repeated_frames",
        "processing_deadline_misses",
        "segmentation_ms",
        "background_ms",
        "color_correction_ms",
        "composite_ms",
        "output_send_ms",
        "frame_processing_ms",
    )
    return {
        "frames_out": stats["frames_out"],
        **{name: stats[name] for name in fields},
    }


def _production_ewma_snapshot(
    *,
    width: int,
    height: int,
    fps: int,
    foreground: np.ndarray,
    geometry_source: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    frames = 12
    return {
        "source": "FrameHub.stats_dict production EWMA fields",
        "frames": frames,
        "baseline": _production_ewma_run(
            width=width,
            height=height,
            fps=fps,
            frames=frames,
            foreground=foreground,
            geometry_source=geometry_source,
            mask=mask,
            candidate=False,
        ),
        "candidate": _production_ewma_run(
            width=width,
            height=height,
            fps=fps,
            frames=frames,
            foreground=foreground,
            geometry_source=geometry_source,
            mask=mask,
            candidate=True,
        ),
    }


def _benchmark_tier_with_executor(
    tier: dict[str, Any],
    manifest: dict[str, Any],
    *,
    measured_frames: int,
    warmup_frames: int,
    resize_executor: Executor,
) -> dict[str, Any]:
    width, height, fps = tier["width"], tier["height"], tier["fps"]
    geometry_source, foreground, backdrop, mask = _tier_arrays(width, height)
    plan = plan_transform(
        (geometry_source.shape[1], geometry_source.shape[0]),
        (width, height),
        fit="cover",
    )
    baseline_output = NullOutput(width, height, fps)
    candidate_output = NullOutput(width, height, fps)
    # ImageBackdrop is immutable within one resource generation. Production
    # builds this bounded analysis raster once, during warmup, and reuses it on
    # every steady-state frame. Keep that one-time work out of raw budget
    # samples while retaining the per-frame full-resolution backdrop decode
    # needed by the compositor.
    backdrop_analysis_linear_bgr = color_mod._linear_bgr_analysis_raster_prevalidated(
        color_mod._bgr_u8_to_linear_bgr_prevalidated(backdrop)
    )
    if (
        max(backdrop_analysis_linear_bgr.shape[:2])
        > manifest["budgets"]["analysis_long_edge_pixels"]
    ):
        _fail("calibrated backdrop analysis exceeds the reviewed long-edge cap")
    for _ in range(warmup_frames):
        _one_baseline_frame(
            geometry_source, foreground, backdrop, mask, plan, baseline_output
        )
        _one_candidate_frame(
            geometry_source,
            foreground,
            backdrop,
            mask,
            plan,
            candidate_output,
            resize_executor=resize_executor,
            backdrop_analysis_linear_bgr=backdrop_analysis_linear_bgr,
        )

    baseline_samples = {name: [] for name in TIMING_STAGES}
    candidate_samples = {name: [] for name in TIMING_STAGES}
    for _ in range(measured_frames):
        baseline = _one_baseline_frame(
            geometry_source,
            foreground,
            backdrop,
            mask,
            plan,
            baseline_output,
        )
        candidate = _one_candidate_frame(
            geometry_source,
            foreground,
            backdrop,
            mask,
            plan,
            candidate_output,
            resize_executor=resize_executor,
            backdrop_analysis_linear_bgr=backdrop_analysis_linear_bgr,
        )
        for name in TIMING_STAGES:
            baseline_samples[name].append(baseline[name])
            candidate_samples[name].append(candidate[name])

    # Allocation tracing materially changes NumPy/OpenCV wall times.  Keep the
    # memory probe separate from every p50/p95 sample.
    gc.collect()
    rss_before = _process_rss_bytes()
    tracemalloc.start(5)
    try:
        tracemalloc.reset_peak()
        _one_candidate_frame(
            geometry_source,
            foreground,
            backdrop,
            mask,
            plan,
            candidate_output,
            resize_executor=resize_executor,
            backdrop_analysis_linear_bgr=backdrop_analysis_linear_bgr,
        )
        _current, traced_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    gc.collect()
    rss_after = _process_rss_bytes()

    baseline_stats = {
        name: _percentiles(values) for name, values in baseline_samples.items()
    }
    candidate_stats = {
        name: _percentiles(values) for name, values in candidate_samples.items()
    }
    baseline_total = baseline_stats["frame_total_ms"]["p95"]
    candidate_total = candidate_stats["frame_total_ms"]["p95"]
    added_p95 = candidate_total - baseline_total
    overhead = 100.0 * added_p95 / max(baseline_total, 1e-9)
    deadline_ms = 1000.0 / fps
    baseline_miss = (
        100.0
        * sum(value > deadline_ms for value in baseline_samples["frame_total_ms"])
        / measured_frames
    )
    candidate_miss = (
        100.0
        * sum(value > deadline_ms for value in candidate_samples["frame_total_ms"])
        / measured_frames
    )
    mean_total = float(np.mean(candidate_samples["frame_total_ms"]))
    effective_fps = 1000.0 / max(mean_total, 1e-9)
    attainment = min(100.0, effective_fps / fps * 100.0)
    budgets = manifest["budgets"]
    checks = {
        "relative_overhead": (
            overhead <= budgets["maximum_relative_end_to_end_overhead_percent"]
        ),
        "added_p95": added_p95 <= tier["maximum_added_p95_ms"],
        "deadline_miss_delta": (
            candidate_miss - baseline_miss
            <= budgets["maximum_deadline_miss_increase_percentage_points"]
        ),
        "fps_attainment": (attainment >= budgets["minimum_fps_attainment_percent"]),
    }
    production_ewma = _production_ewma_snapshot(
        width=width,
        height=height,
        fps=fps,
        foreground=foreground,
        geometry_source=geometry_source,
        mask=mask,
    )
    samples = {
        "baseline": {
            name: [_round(value, 6) for value in baseline_samples[name]]
            for name in TIMING_STAGES
        },
        "candidate": {
            name: [_round(value, 6) for value in candidate_samples[name]]
            for name in TIMING_STAGES
        },
    }
    return {
        "id": tier["id"],
        "status": "pass" if all(checks.values()) else "fail",
        "width": width,
        "height": height,
        "target_fps": fps,
        "measured_frames": measured_frames,
        "warmup_frames": warmup_frames,
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "samples": samples,
        "samples_sha256": _sha256_bytes(_canonical_json(samples)),
        "added_p95_ms": _round(added_p95, 4),
        "relative_end_to_end_overhead_percent": _round(overhead, 3),
        "baseline_deadline_miss_percent": _round(baseline_miss, 4),
        "candidate_deadline_miss_percent": _round(candidate_miss, 4),
        "deadline_miss_increase_percentage_points": _round(
            candidate_miss - baseline_miss, 4
        ),
        "effective_fps": _round(effective_fps, 3),
        "fps_attainment_percent": _round(attainment, 3),
        "peak_tracemalloc_bytes": traced_peak,
        "rss_growth_bytes": (
            None
            if rss_before is None or rss_after is None
            else max(0, rss_after - rss_before)
        ),
        "production_ewma": production_ewma,
        "checks": checks,
    }


def _benchmark_tier(
    tier: dict[str, Any],
    manifest: dict[str, Any],
    *,
    measured_frames: int,
    warmup_frames: int,
) -> dict[str, Any]:
    resize_executor = ThreadPoolExecutor(
        max_workers=3,
        thread_name_prefix="custback-qualification-color-analysis",
    )
    try:
        return _benchmark_tier_with_executor(
            tier,
            manifest,
            measured_frames=measured_frames,
            warmup_frames=warmup_frames,
            resize_executor=resize_executor,
        )
    finally:
        resize_executor.shutdown(wait=True, cancel_futures=True)


def _calibrated_section(
    manifest: dict[str, Any],
    *,
    measured_frames: int,
    warmup_frames: int,
    pinned_runner: bool,
    runner_id: str,
    reference_cpu: str,
) -> dict[str, Any]:
    tiers = [
        _benchmark_tier(
            tier,
            manifest,
            measured_frames=measured_frames,
            warmup_frames=warmup_frames,
        )
        for tier in manifest["performance_tiers"]
    ]
    complete_frames = all(
        result["measured_frames"] >= definition["minimum_measured_frames"]
        for result, definition in zip(tiers, manifest["performance_tiers"], strict=True)
    )
    authority = (
        "pinned-reference-runner"
        if pinned_runner and runner_id and reference_cpu and complete_frames
        else "local-observation"
    )
    return {
        "status": "pass" if all(row["status"] == "pass" for row in tiers) else "fail",
        "authority": authority,
        "runner": {
            "id": runner_id or platform.node() or "local",
            "pinned": bool(pinned_runner),
            "cpu": reference_cpu or platform.processor() or "unknown",
            "os": platform.platform(),
            "os_family": _os_family(),
        },
        "tiers": tiers,
    }


def _not_run_calibrated() -> dict[str, Any]:
    return {
        "status": "not-run",
        "authority": "none",
        "runner": {
            "id": "not-run",
            "pinned": False,
            "cpu": "not-run",
            "os": "not-run",
            "os_family": "not-run",
        },
        "tiers": [],
    }


def _pending_physical(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "pending",
        "webcams": [],
        "consumers": [],
        "rows": [
            {
                "id": row["id"],
                "os": row["os"],
                "capture_backend": row["capture_backend"],
                "output_backend": row["output_backend"],
                "transport": row["transport"],
                "status": "pending",
                "webcam_ids": [],
                "consumer_ids": [],
                "artifacts": {},
                "measurements": None,
                "notes": "manual physical qualification not supplied",
            }
            for row in manifest["physical_matrix"]
        ],
    }


def _physical_evidence_schemas() -> dict[str, Any]:
    return {
        "webcam": {
            "id": "stable-webcam-id",
            "manufacturer": "exact manufacturer",
            "model": "exact model",
            "auto_white_balance": "enabled|disabled|unavailable",
            "auto_exposure": "enabled|disabled|unavailable",
        },
        "consumer": {
            "id": "stable-consumer-id",
            "name": "Meeting application name",
            "version": "exact version",
            "os": "linux|macos|windows",
        },
        "artifact": {
            "filename": "unique-relative-evidence-file.png",
            "sha256": "replace-with-lowercase-sha256",
            "bytes": 1,
            "media_type": "image/png",
        },
        "required_artifact_ids": {
            "contact_sheet": "image/png",
            "preview_capture": "image/png",
            "api_snapshot": "application/json",
            "consumer_capture": "image/png",
        },
        "measurements": {
            "crop_edge_error_pixels": 0,
            "orientation_error_degrees": 0,
            "transport_mean_absolute_error": 0.0,
            "luminance_gap_reduction_percent": 0.0,
            "neutral_error_reduction_percent": 0.0,
            "skin_hue_drift_degrees": 0.0,
            "skin_chroma_drift_percent": 0.0,
            "clothing_hue_drift_degrees": 0.0,
            "clothing_chroma_drift_percent": 0.0,
            "steady_state_ev_delta_p95": 0.0,
            "steady_state_wb_log2_delta_p95": 0.0,
            "no_temporal_oscillation": False,
            "preview_api_equal": False,
            "all_consumers_match": False,
        },
    }


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _os_family() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform in ("win32", "cygwin"):
        return "windows"
    return "other"


def _source_evidence() -> dict[str, str]:
    return {name: _sha256_file(ROOT / name) for name in SOURCE_FILES}


def _git_value(*arguments: str, cwd: Path = ROOT) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _git_status(*, cwd: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _git_blob_digest(commit: str, filename: str, *, cwd: Path = ROOT) -> str:
    try:
        result = subprocess.run(
            ("git", "show", f"{commit}:{filename}"),
            cwd=cwd,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        _fail(f"cannot read {filename} from qualification source commit")
    return _sha256_bytes(result.stdout)


def _git_is_ancestor(commit: str, *, cwd: Path = ROOT) -> bool:
    try:
        result = subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
            cwd=cwd,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _release_git_context(
    source_commit: str, source_tree: str
) -> tuple[Path, bool, str]:
    """Resolve an optional trusted Git root for a git-archive prepack stage."""

    names = (
        "CUSTBACK_RELEASE_GIT_ROOT",
        "CUSTBACK_RELEASE_SOURCE_COMMIT",
        "CUSTBACK_RELEASE_SOURCE_TREE",
    )
    values = {name: os.environ.get(name) for name in names}
    supplied = [value is not None for value in values.values()]
    if any(supplied) and not all(supplied):
        _fail("trusted release Git bridge variables must be supplied together")
    if not any(supplied):
        return ROOT, False, _git_value("rev-parse", "HEAD")

    root_text = _required_string(values[names[0]], names[0])
    bridge_commit = _valid_git_oid(values[names[1]], names[1])
    bridge_tree = _valid_git_oid(values[names[2]], names[2])
    bridge_path = Path(root_text)
    if not bridge_path.is_absolute():
        _fail("CUSTBACK_RELEASE_GIT_ROOT must be an absolute path")
    try:
        metadata = bridge_path.lstat()
        resolved = bridge_path.resolve(strict=True)
    except OSError as exc:
        _fail(f"trusted release Git root is unavailable: {exc}")
    if not stat.S_ISDIR(metadata.st_mode) or bridge_path.is_symlink():
        _fail("trusted release Git root must be a real non-symlink directory")
    top_level = _git_value("rev-parse", "--show-toplevel", cwd=resolved)
    try:
        actual_top_level = Path(top_level).resolve(strict=True)
    except OSError:
        _fail("trusted release Git root is not a valid repository")
    if actual_top_level != resolved:
        _fail("trusted release Git root must be the repository top level")
    if (
        _git_value("rev-parse", "HEAD", cwd=resolved) != bridge_commit
        or _git_value("rev-parse", "HEAD^{tree}", cwd=resolved) != bridge_tree
    ):
        _fail("trusted release Git bridge does not match its declared HEAD/tree")
    if not _git_is_ancestor(source_commit, cwd=resolved):
        _fail("qualification source is not an ancestor of the trusted release HEAD")
    if (
        _git_value("rev-parse", f"{source_commit}^{{tree}}", cwd=resolved)
        != source_tree
    ):
        _fail("qualification source tree is absent from the trusted repository")
    if _git_status(cwd=resolved) != "":
        _fail("trusted release Git root must be clean")
    return resolved, True, bridge_commit


def _source_binding() -> dict[str, Any]:
    commit = _git_value("rev-parse", "HEAD")
    tree = _git_value("rev-parse", "HEAD^{tree}")
    status = _git_status()
    clean = status == ""
    return {
        "commit": commit,
        "tree": tree,
        "clean": clean,
        "files": _source_evidence(),
    }


def _pending_candidate() -> dict[str, Any]:
    return {
        "status": "pending",
        "manifest_filename": None,
        "manifest_sha256": None,
    }


def build_report(
    manifest: dict[str, Any],
    *,
    contact_sheet_path: Path,
    soak_frames: int,
    calibrated: bool = False,
    measured_frames: int = 300,
    warmup_frames: int = 5,
    pinned_runner: bool = False,
    runner_id: str = "",
    reference_cpu: str = "",
) -> dict[str, Any]:
    if measured_frames < 1 or warmup_frames < 0:
        _fail("measured_frames must be positive and warmup_frames non-negative")
    if pinned_runner and (not runner_id or not reference_cpu):
        _fail("a pinned runner requires explicit runner id and reference CPU")
    deterministic = _deterministic_contract(
        manifest,
        contact_sheet_path=contact_sheet_path,
        soak_frames=soak_frames,
    )
    calibrated_section = (
        _calibrated_section(
            manifest,
            measured_frames=measured_frames,
            warmup_frames=warmup_frames,
            pinned_runner=pinned_runner,
            runner_id=runner_id,
            reference_cpu=reference_cpu,
        )
        if calibrated
        else _not_run_calibrated()
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": _manifest_digest(manifest),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "baseline": {
            "path": manifest["baseline"]["path"],
            "sha256": manifest["baseline"]["sha256"],
        },
        "source": _source_binding(),
        "candidate": _pending_candidate(),
        "environment": _environment(),
        "deterministic": deterministic,
        "calibrated": calibrated_section,
        "physical": _pending_physical(manifest),
        "release_qualified": False,
    }
    validate_report(report, manifest, claim="deterministic")
    return report


def _exact_id_set(
    entries: object, expected: list[str], label: str
) -> list[dict[str, Any]]:
    actual = _unique_ids(entries, label)
    if sorted(actual) != sorted(expected):
        _fail(f"{label} ids do not match the reviewed manifest")
    assert isinstance(entries, list)
    return entries


def _validate_deterministic(
    section: object, manifest: dict[str, Any]
) -> dict[str, Any]:
    value = _exact_keys(
        section,
        (
            "status",
            "authority",
            "contracts",
            "contact_sheet",
            "soak",
            "deterministic_sha256",
        ),
        "report.deterministic",
    )
    if value["status"] != "pass" or value["authority"] != "shared-ci-deterministic":
        _fail("deterministic qualification is not a shared-CI pass")
    contracts = value["contracts"]
    if not isinstance(contracts, dict) or set(contracts) != set(
        manifest["deterministic_contracts"]
    ):
        _fail("deterministic contract results do not match the manifest")
    for contract_id, result in contracts.items():
        _exact_keys(result, ("status", "metrics"), f"contract {contract_id}")
        if result["status"] != "pass" or not isinstance(result["metrics"], dict):
            _fail(f"deterministic contract {contract_id} did not pass")
    visual = _exact_keys(
        contracts["phase0-visual-improvement"]["metrics"],
        ("geometry", "color"),
        "phase0 visual metrics",
    )
    geometry = visual["geometry"]
    if not isinstance(geometry, dict) or _nonnegative_number(
        geometry.get("production_distortion_percent"),
        "production geometry distortion",
    ) >= _nonnegative_number(
        geometry.get("phase0_distortion_percent"), "Phase-0 geometry distortion"
    ):
        _fail("production geometry does not improve the checked-in Phase-0 baseline")
    baseline = _read_json(
        ROOT / manifest["baseline"]["path"], "VIS-0.2 baseline evidence"
    )
    recommendation = baseline["recommendation"]
    color = visual["color"]
    if not isinstance(color, dict):
        _fail("production color metrics are missing")
    luminance = color.get("luminance_gap_ev")
    neutral = color.get("neutral_axis_error_delta_e_ab")
    skin = color.get("skin_preservation")
    clothing = color.get("clothing_preservation")
    for metrics, label in (
        (luminance, "luminance gap"),
        (neutral, "neutral-axis error"),
        (skin, "skin preservation"),
        (clothing, "clothing preservation"),
    ):
        if not isinstance(metrics, dict):
            _fail(f"{label} metrics are missing")
    assert isinstance(luminance, dict)
    assert isinstance(neutral, dict)
    assert isinstance(skin, dict)
    assert isinstance(clothing, dict)
    if _finite_number(
        luminance.get("reduction_percent"), "luminance reduction"
    ) + 0.5 < _finite_number(
        luminance.get("phase0_selected_reduction_percent"),
        "Phase-0 luminance reduction",
    ):
        _fail("production luminance improvement regresses the Phase-0 selection")
    if _finite_number(
        neutral.get("reduction_percent"), "neutral-axis reduction"
    ) + 0.5 < _finite_number(
        neutral.get("phase0_selected_reduction_percent"),
        "Phase-0 neutral-axis reduction",
    ):
        _fail("production neutral-axis improvement regresses the Phase-0 selection")
    for metrics, prefix, hue_limit, chroma_limit in (
        (
            skin,
            "skin",
            recommendation["skin_hue_drift_max_degrees"],
            recommendation["skin_normalized_chroma_drift_max_percent"],
        ),
        (
            clothing,
            "clothing",
            recommendation["clothing_hue_drift_max_degrees"],
            recommendation["clothing_normalized_chroma_drift_max_percent"],
        ),
    ):
        if _nonnegative_number(
            metrics.get("hue_drift_degrees"), f"{prefix} hue drift"
        ) > float(hue_limit) or _nonnegative_number(
            metrics.get("normalized_chroma_drift_percent"),
            f"{prefix} normalized chroma drift",
        ) > float(chroma_limit):
            _fail(f"production {prefix} preservation exceeds the Phase-0 threshold")
    if (
        color.get("foreground_endpoint_max_delta") != 0
        or color.get("backdrop_endpoint_max_delta") != 0
    ):
        _fail("linear composite endpoints are not byte exact")

    preservation = contracts["foreground-preservation"]["metrics"]
    if not preservation or any(value is not True for value in preservation.values()):
        _fail("foreground-preservation checks are not all true")
    resize = contracts["common-path-single-resize"]["metrics"]
    if set(resize) != {"720p", "1080p"} or any(
        type(count) is not int
        or count > manifest["budgets"]["maximum_geometry_resizes_common_path"]
        for count in resize.values()
    ):
        _fail("common-path resize counts exceed the reviewed bound")
    analysis = contracts["analysis-resolution-cap"]["metrics"]
    expected_tiers = {tier["id"] for tier in manifest["performance_tiers"]}
    if set(analysis) != expected_tiers:
        _fail("analysis-size results do not cover every performance tier")
    for tier_id, size in analysis.items():
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(item) is not int or item < 1 for item in size)
            or max(size) > manifest["budgets"]["analysis_long_edge_pixels"]
        ):
            _fail(f"{tier_id} analysis size exceeds the reviewed cap")
    repeated = contracts["repeated-output-work-skip"]["metrics"]
    if repeated != {
        "source_frames": 1,
        "output_frames": 6,
        "repeated_output_frames": 5,
        "color_analysis_calls": 1,
        "linear_decode_calls": 2,
        "composite_calls": 1,
    }:
        _fail("repeated-output operation counts are not the reviewed contract")
    pacing = contracts["backdrop-source-pacing"]["metrics"]
    expected_video = {
        "calls_at_same_deadline": 5,
        "decoder_reads_during_early_repeats": 0,
        "displayed_frames": 2,
        "reused_frames": 4,
        "second_frame_changed": True,
    }
    if not isinstance(pacing, dict) or pacing.get("video") != expected_video:
        _fail("video pacing counts are not the reviewed contract")
    camera_pacing = pacing.get("camera")
    if camera_pacing != {
        "pipeline_requests": 5,
        "capture_reads": 5,
        "frames_returned": 5,
        "prefetched_frames": 0,
    }:
        _fail("live-camera pacing counts are not the reviewed contract")
    contact = _exact_keys(
        value["contact_sheet"],
        ("filename", "sha256", "bytes", "width", "height"),
        "deterministic contact sheet",
    )
    _required_string(contact["filename"], "contact_sheet.filename")
    _valid_sha256(contact["sha256"], "contact_sheet.sha256")
    for name in ("bytes", "width", "height"):
        _positive_int(contact[name], f"contact_sheet.{name}")
    soak = value["soak"]
    if not isinstance(soak, dict) or soak.get("status") != "pass":
        _fail("deterministic soak did not pass")
    soak = _exact_keys(
        soak,
        (
            "status",
            "frames",
            "warmup_frames",
            "fixed_session_state_semantic_bytes",
            "fixed_session_state_limit_bytes",
            "fixed_tracemalloc_growth_bytes",
            "fixed_rss_growth_bytes",
            "tracemalloc_current_growth_bytes",
            "tracemalloc_peak_bytes",
            "rss_growth_bytes",
            "retained_growth_limit_bytes",
            "privacy_history_entries",
            "privacy_history_storage_bytes",
            "maximum_canvas_frame_bytes",
            "maximum_color_analysis_cache_bytes",
            "latest_frame_bytes",
            "geometry_cache_entries",
            "geometry_cache_max_entries",
            "checks",
        ),
        "deterministic.soak",
    )
    frames = _positive_int(soak["frames"], "deterministic.soak.frames")
    warmup = _positive_int(soak["warmup_frames"], "deterministic.soak.warmup_frames")
    maximum_tier = max(
        manifest["performance_tiers"],
        key=lambda tier: tier["width"] * tier["height"],
    )
    expected_canvas_bytes = maximum_tier["width"] * maximum_tier["height"] * 3
    expected_history_bytes = manifest["budgets"]["privacy_history_storage_bytes"]
    expected_analysis_cache_bytes = (
        manifest["budgets"]["analysis_long_edge_pixels"] ** 2 * 3 * 4
    )
    expected_fixed_bytes = (
        expected_history_bytes + expected_canvas_bytes + expected_analysis_cache_bytes
    )
    _positive_int(
        soak["maximum_color_analysis_cache_bytes"],
        "deterministic.soak.maximum_color_analysis_cache_bytes",
    )
    if (
        soak["fixed_session_state_semantic_bytes"] != expected_fixed_bytes
        or soak["fixed_session_state_limit_bytes"]
        != manifest["budgets"]["maximum_fixed_session_state_bytes"]
        or soak["retained_growth_limit_bytes"]
        != manifest["budgets"]["maximum_retained_growth_bytes"]
        or soak["privacy_history_storage_bytes"] != expected_history_bytes
        or soak["maximum_canvas_frame_bytes"] != expected_canvas_bytes
        or soak["maximum_color_analysis_cache_bytes"] != expected_analysis_cache_bytes
        or soak["latest_frame_bytes"] != 24 * 32 * 3
        or soak["privacy_history_entries"] != 1 + warmup + frames
        or soak["geometry_cache_max_entries"] != 512
    ):
        _fail("deterministic soak retained-state accounting is inconsistent")
    if expected_fixed_bytes > manifest["budgets"]["maximum_fixed_session_state_bytes"]:
        _fail("deterministic soak fixed session state exceeds its reviewed limit")
    retained_limit = manifest["budgets"]["maximum_retained_growth_bytes"]
    for name in (
        "fixed_tracemalloc_growth_bytes",
        "tracemalloc_current_growth_bytes",
        "tracemalloc_peak_bytes",
    ):
        _nonnegative_number(soak[name], f"deterministic.soak.{name}")
    for name in ("fixed_rss_growth_bytes", "rss_growth_bytes"):
        if soak[name] is not None:
            _nonnegative_number(soak[name], f"deterministic.soak.{name}")
    if soak["tracemalloc_current_growth_bytes"] > retained_limit or (
        soak["rss_growth_bytes"] is not None
        and soak["rss_growth_bytes"] > retained_limit
    ):
        _fail("deterministic soak retained growth exceeds its reviewed limit")
    if (
        type(soak["geometry_cache_entries"]) is not int
        or soak["geometry_cache_entries"] < 0
        or soak["geometry_cache_entries"] > soak["geometry_cache_max_entries"]
    ):
        _fail("deterministic soak geometry cache accounting is invalid")
    checks = soak.get("checks")
    if not isinstance(checks, dict) or any(
        item is not True for item in checks.values()
    ):
        _fail("deterministic soak checks are not all true")
    if contracts["bounded-session-history"]["metrics"] != soak:
        _fail("bounded-session contract and soak evidence differ")
    digest_payload = {
        "baseline_sha256": manifest["baseline"]["sha256"],
        "contracts": contracts,
        "contact_sheet_sha256": contact["sha256"],
    }
    expected_digest = _sha256_bytes(_canonical_json(digest_payload))
    if value["deterministic_sha256"] != expected_digest:
        _fail("deterministic report digest does not match its contract results")
    return value


def _validate_timing_stats(value: object, label: str) -> None:
    stage = _exact_keys(value, TIMING_STAGES, label)
    for name in TIMING_STAGES:
        percentiles = _exact_keys(stage[name], ("p50", "p95"), f"{label}.{name}")
        p50 = _nonnegative_number(percentiles["p50"], f"{label}.{name}.p50")
        p95 = _nonnegative_number(percentiles["p95"], f"{label}.{name}.p95")
        if p95 + 1e-6 < p50:
            _fail(f"{label}.{name}.p95 must be at least p50")


def _validate_timing_samples(
    value: object, *, measured_frames: int, label: str
) -> dict[str, dict[str, list[float]]]:
    profiles = _exact_keys(value, ("baseline", "candidate"), label)
    normalized: dict[str, dict[str, list[float]]] = {}
    for profile_name in ("baseline", "candidate"):
        stages = _exact_keys(
            profiles[profile_name], TIMING_STAGES, f"{label}.{profile_name}"
        )
        normalized[profile_name] = {}
        for stage_name in TIMING_STAGES:
            samples = stages[stage_name]
            if not isinstance(samples, list) or len(samples) != measured_frames:
                _fail(
                    f"{label}.{profile_name}.{stage_name} must contain exactly "
                    f"{measured_frames} samples"
                )
            normalized[profile_name][stage_name] = [
                _nonnegative_number(
                    sample,
                    f"{label}.{profile_name}.{stage_name}[{index}]",
                )
                for index, sample in enumerate(samples)
            ]
        for index in range(measured_frames):
            stage_sum = sum(
                normalized[profile_name][stage_name][index]
                for stage_name in TIMING_STAGES[:-1]
            )
            total = normalized[profile_name]["frame_total_ms"][index]
            if abs(stage_sum - total) > 0.001:
                _fail(
                    f"{label}.{profile_name} sample {index} has inconsistent "
                    "stage and total timing"
                )
    return normalized


def _validate_calibrated_release(section: object, manifest: dict[str, Any]) -> None:
    value = _exact_keys(
        section, ("status", "authority", "runner", "tiers"), "report.calibrated"
    )
    if value["status"] != "pass" or value["authority"] != "pinned-reference-runner":
        _fail("release qualification requires a passing pinned-runner timing report")
    runner = _exact_keys(
        value["runner"],
        ("id", "pinned", "cpu", "os", "os_family"),
        "calibrated.runner",
    )
    for name in ("id", "cpu", "os", "os_family"):
        _required_string(runner[name], f"calibrated.runner.{name}")
    if runner["pinned"] is not True:
        _fail("calibrated release evidence must identify a pinned runner")
    if runner["os_family"] != manifest["reference_runner"]["os_family"]:
        _fail("calibrated release evidence is not from the reviewed OS family")

    definitions = {tier["id"]: tier for tier in manifest["performance_tiers"]}
    rows = _exact_id_set(value["tiers"], list(definitions), "calibrated.tiers")
    budgets = manifest["budgets"]
    for row in rows:
        tier_id = row["id"]
        definition = definitions[tier_id]
        _exact_keys(
            row,
            (
                "id",
                "status",
                "width",
                "height",
                "target_fps",
                "measured_frames",
                "warmup_frames",
                "baseline",
                "candidate",
                "samples",
                "samples_sha256",
                "added_p95_ms",
                "relative_end_to_end_overhead_percent",
                "baseline_deadline_miss_percent",
                "candidate_deadline_miss_percent",
                "deadline_miss_increase_percentage_points",
                "effective_fps",
                "fps_attainment_percent",
                "peak_tracemalloc_bytes",
                "rss_growth_bytes",
                "production_ewma",
                "checks",
            ),
            f"calibrated tier {tier_id}",
        )
        if (
            row["status"] != "pass"
            or not isinstance(row["checks"], dict)
            or set(row["checks"])
            != {
                "relative_overhead",
                "added_p95",
                "deadline_miss_delta",
                "fps_attainment",
            }
            or not all(check is True for check in row["checks"].values())
        ):
            _fail(f"calibrated tier {tier_id} did not pass every budget")
        if (
            row["width"] != definition["width"]
            or row["height"] != definition["height"]
            or row["target_fps"] != definition["fps"]
        ):
            _fail(f"calibrated tier {tier_id} ran the wrong mode")
        if (
            _positive_int(row["measured_frames"], f"{tier_id}.measured_frames")
            < definition["minimum_measured_frames"]
        ):
            _fail(f"calibrated tier {tier_id} has too few measured frames")
        if type(row["warmup_frames"]) is not int or row["warmup_frames"] < 0:
            _fail(f"calibrated tier {tier_id} has an invalid warmup")
        _validate_timing_stats(row["baseline"], f"{tier_id}.baseline")
        _validate_timing_stats(row["candidate"], f"{tier_id}.candidate")
        samples = _validate_timing_samples(
            row["samples"],
            measured_frames=row["measured_frames"],
            label=f"{tier_id}.samples",
        )
        if row["samples_sha256"] != _sha256_bytes(_canonical_json(row["samples"])):
            _fail(f"{tier_id} raw timing sample digest is inconsistent")
        for profile in ("baseline", "candidate"):
            for stage_name in TIMING_STAGES:
                expected = _percentiles(samples[profile][stage_name])
                reported = row[profile][stage_name]
                if any(
                    abs(float(reported[name]) - expected[name]) > 0.0002
                    for name in ("p50", "p95")
                ):
                    _fail(
                        f"{tier_id}.{profile}.{stage_name} percentiles do not "
                        "match raw samples"
                    )
        added = _finite_number(row["added_p95_ms"], f"{tier_id}.added_p95_ms")
        overhead = _finite_number(
            row["relative_end_to_end_overhead_percent"],
            f"{tier_id}.relative_end_to_end_overhead_percent",
        )
        deadline_delta = _finite_number(
            row["deadline_miss_increase_percentage_points"],
            f"{tier_id}.deadline_miss_increase_percentage_points",
        )
        attainment = _nonnegative_number(
            row["fps_attainment_percent"], f"{tier_id}.fps_attainment_percent"
        )
        baseline_p95 = float(row["baseline"]["frame_total_ms"]["p95"])
        candidate_p95 = float(row["candidate"]["frame_total_ms"]["p95"])
        expected_added = candidate_p95 - baseline_p95
        expected_overhead = 100.0 * expected_added / max(baseline_p95, 1e-9)
        baseline_miss = _nonnegative_number(
            row["baseline_deadline_miss_percent"],
            f"{tier_id}.baseline_deadline_miss_percent",
        )
        candidate_miss = _nonnegative_number(
            row["candidate_deadline_miss_percent"],
            f"{tier_id}.candidate_deadline_miss_percent",
        )
        if baseline_miss > 100.0 or candidate_miss > 100.0 or attainment > 100.0:
            _fail(f"{tier_id} reports an impossible percentage")
        deadline_ms = 1000.0 / definition["fps"]
        expected_baseline_miss = _round(
            100.0
            * sum(
                sample > deadline_ms for sample in samples["baseline"]["frame_total_ms"]
            )
            / row["measured_frames"],
            4,
        )
        expected_candidate_miss = _round(
            100.0
            * sum(
                sample > deadline_ms
                for sample in samples["candidate"]["frame_total_ms"]
            )
            / row["measured_frames"],
            4,
        )
        effective_fps = _nonnegative_number(
            row["effective_fps"], f"{tier_id}.effective_fps"
        )
        expected_effective_fps = 1000.0 / max(
            sum(samples["candidate"]["frame_total_ms"]) / row["measured_frames"],
            1e-9,
        )
        expected_attainment = min(
            100.0, expected_effective_fps / definition["fps"] * 100.0
        )
        if (
            abs(added - expected_added) > 0.01
            or abs(overhead - expected_overhead) > 0.05
            or abs(deadline_delta - (candidate_miss - baseline_miss)) > 0.01
            or abs(baseline_miss - expected_baseline_miss) > 0.0002
            or abs(candidate_miss - expected_candidate_miss) > 0.0002
            or abs(effective_fps - expected_effective_fps) > 0.002
            or abs(attainment - expected_attainment) > 0.002
        ):
            _fail(f"calibrated tier {tier_id} derived metrics are inconsistent")
        expected_checks = {
            "relative_overhead": (
                overhead <= budgets["maximum_relative_end_to_end_overhead_percent"]
            ),
            "added_p95": added <= definition["maximum_added_p95_ms"],
            "deadline_miss_delta": (
                deadline_delta
                <= budgets["maximum_deadline_miss_increase_percentage_points"]
            ),
            "fps_attainment": (attainment >= budgets["minimum_fps_attainment_percent"]),
        }
        if row["checks"] != expected_checks:
            _fail(f"{tier_id} budget checks do not match derived metrics")
        if (
            added > definition["maximum_added_p95_ms"]
            or overhead > budgets["maximum_relative_end_to_end_overhead_percent"]
            or deadline_delta
            > budgets["maximum_deadline_miss_increase_percentage_points"]
            or attainment < budgets["minimum_fps_attainment_percent"]
        ):
            _fail(f"calibrated tier {tier_id} exceeds a reviewed budget")
        _positive_int(
            row["peak_tracemalloc_bytes"], f"{tier_id}.peak_tracemalloc_bytes"
        )
        if row["rss_growth_bytes"] is not None:
            _nonnegative_number(row["rss_growth_bytes"], f"{tier_id}.rss_growth_bytes")
        elif manifest["reference_runner"]["rss_required"]:
            _fail(f"{tier_id} lacks the required RSS measurement")
        ewma = _exact_keys(
            row["production_ewma"],
            ("source", "frames", "baseline", "candidate"),
            f"{tier_id}.production_ewma",
        )
        if ewma["source"] != "FrameHub.stats_dict production EWMA fields":
            _fail(f"{tier_id} EWMA snapshot is not from the production stats path")
        ewma_frames = _positive_int(ewma["frames"], f"{tier_id}.production_ewma.frames")
        for profile in ("baseline", "candidate"):
            snapshot = _exact_keys(
                ewma[profile],
                (
                    "frames_out",
                    "fps",
                    "fps_attainment_pct",
                    "output_repeated_frames",
                    "processing_deadline_misses",
                    "segmentation_ms",
                    "background_ms",
                    "color_correction_ms",
                    "composite_ms",
                    "output_send_ms",
                    "frame_processing_ms",
                ),
                f"{tier_id}.production_ewma.{profile}",
            )
            if snapshot["frames_out"] != ewma_frames:
                _fail(f"{tier_id} EWMA {profile} has the wrong frame count")
            for name in (
                "fps",
                "fps_attainment_pct",
                "segmentation_ms",
                "background_ms",
                "color_correction_ms",
                "composite_ms",
                "output_send_ms",
                "frame_processing_ms",
            ):
                _nonnegative_number(
                    snapshot[name], f"{tier_id}.production_ewma.{profile}.{name}"
                )
            for name in ("output_repeated_frames", "processing_deadline_misses"):
                if type(snapshot[name]) is not int or snapshot[name] < 0:
                    _fail(f"{tier_id} EWMA {profile}.{name} is invalid")


def _validate_pending_calibrated(section: object) -> None:
    value = _exact_keys(
        section, ("status", "authority", "runner", "tiers"), "report.calibrated"
    )
    if value["status"] not in ("not-run", "pass", "fail"):
        _fail("calibrated status is invalid")
    if not isinstance(value["tiers"], list):
        _fail("calibrated.tiers must be an array")
    _exact_keys(
        value["runner"],
        ("id", "pinned", "cpu", "os", "os_family"),
        "calibrated.runner",
    )


def _safe_evidence_file(
    root: Path,
    filename: object,
    expected_sha256: object,
    expected_bytes: object | None,
    *,
    label: str,
    seen: set[Path],
    seen_digests: set[str],
) -> Path:
    name = _required_string(filename, f"{label}.filename")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        _fail(f"{label}.filename must be a safe relative path")
    try:
        root_metadata = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        _fail(f"evidence root is unavailable: {exc}")
    if not stat.S_ISDIR(root_metadata.st_mode) or root.is_symlink():
        _fail("evidence root must be a regular non-symlink directory")
    candidate = root / relative
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label} is unavailable: {exc}")
    if not stat.S_ISREG(metadata.st_mode) or candidate.is_symlink():
        _fail(f"{label} must be a regular non-symlink file")
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        _fail(f"{label} escapes the evidence root")
    if resolved in seen:
        _fail(f"duplicate evidence artifact path: {name}")
    seen.add(resolved)
    if expected_bytes is not None:
        size = _positive_int(expected_bytes, f"{label}.bytes")
        if metadata.st_size != size:
            _fail(f"{label} byte count differs from the evidence report")
    expected = _valid_sha256(expected_sha256, f"{label}.sha256")
    if expected in seen_digests:
        _fail(f"duplicate evidence artifact digest: {name}")
    seen_digests.add(expected)
    if _sha256_file(candidate) != expected:
        _fail(f"{label} SHA-256 differs from the evidence report")
    return resolved


def _validate_artifact_content(path: Path, media_type: str, label: str) -> None:
    payload = path.read_bytes()
    if media_type == "image/png":
        if (
            path.suffix.lower() != ".png"
            or len(payload) < 33
            or payload[:8] != b"\x89PNG\r\n\x1a\n"
            or payload[12:16] != b"IHDR"
            or int.from_bytes(payload[16:20], "big") < 1
            or int.from_bytes(payload[20:24], "big") < 1
        ):
            _fail(f"{label} is not a positive-dimension PNG")
        return
    if media_type == "application/json":
        if path.suffix.lower() != ".json":
            _fail(f"{label} must use a .json filename")
        try:
            value = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            _fail(f"{label} is invalid JSON: {exc}")
        if not isinstance(value, dict) or not value:
            _fail(f"{label} must contain a non-empty JSON object")
        return
    _fail(f"{label} has an unsupported media type")


def _validate_candidate_manifest(path: Path, source_commit: str) -> None:
    candidate = _read_json(path, "candidate manifest evidence")
    source = candidate.get("source")
    if not isinstance(source, dict) or source.get("commit") != source_commit:
        _fail("candidate manifest is not bound to the qualification source commit")
    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        _fail("candidate manifest contains no release artifacts")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            _fail(f"candidate artifact {index} must be an object")
        _required_string(artifact.get("id"), f"candidate artifact {index}.id")
        _required_string(
            artifact.get("filename"), f"candidate artifact {index}.filename"
        )
        _valid_sha256(artifact.get("sha256"), f"candidate artifact {index}.sha256")
        _positive_int(artifact.get("size"), f"candidate artifact {index}.size")


def _validate_physical_release(
    section: object,
    manifest: dict[str, Any],
    *,
    evidence_root: Path,
    seen_artifacts: set[Path],
    seen_digests: set[str],
) -> None:
    value = _exact_keys(
        section, ("status", "webcams", "consumers", "rows"), "report.physical"
    )
    if value["status"] != "pass":
        _fail("release qualification requires a complete physical matrix")
    requirements = manifest["physical_requirements"]

    webcam_ids = _unique_ids(value["webcams"], "physical.webcams")
    if len(webcam_ids) < requirements["minimum_distinct_webcams"]:
        _fail("physical qualification has too few distinct webcams")
    profiles: set[tuple[object, object]] = set()
    webcam_identities: set[tuple[str, str]] = set()
    webcam_by_id: dict[str, dict[str, Any]] = {}
    for webcam in value["webcams"]:
        _exact_keys(
            webcam,
            ("id", "manufacturer", "model", "auto_white_balance", "auto_exposure"),
            f"webcam {webcam['id']}",
        )
        for name in ("manufacturer", "model"):
            _required_string(webcam[name], f"webcam {webcam['id']}.{name}")
        for name in ("auto_white_balance", "auto_exposure"):
            if webcam[name] not in ("enabled", "disabled", "unavailable"):
                _fail(f"webcam {webcam['id']}.{name} has an invalid state")
        profiles.add((webcam["auto_white_balance"], webcam["auto_exposure"]))
        webcam_identities.add((webcam["manufacturer"], webcam["model"]))
        webcam_by_id[webcam["id"]] = webcam
    if len(webcam_identities) != len(webcam_ids):
        _fail("physical webcam ids do not identify distinct hardware models")
    if requirements["require_different_auto_control_profiles"] and len(profiles) < 2:
        _fail("the two webcams do not have different auto-control profiles")

    consumer_ids = _unique_ids(value["consumers"], "physical.consumers")
    if len(consumer_ids) < requirements["minimum_distinct_meeting_consumers"]:
        _fail("physical qualification has too few meeting-app consumers")
    consumer_names: set[str] = set()
    consumer_by_id: dict[str, dict[str, Any]] = {}
    for consumer in value["consumers"]:
        _exact_keys(
            consumer,
            ("id", "name", "version", "os"),
            f"consumer {consumer['id']}",
        )
        for name in ("name", "version", "os"):
            _required_string(consumer[name], f"consumer {consumer['id']}.{name}")
        consumer_names.add(consumer["name"].casefold())
        consumer_by_id[consumer["id"]] = consumer
    if len(consumer_names) < requirements["minimum_distinct_meeting_consumers"]:
        _fail("physical qualification must exercise distinct meeting applications")

    definitions = {row["id"]: row for row in manifest["physical_matrix"]}
    rows = _exact_id_set(value["rows"], list(definitions), "physical.rows")
    required_artifacts = requirements["required_artifacts"]
    used_webcam_ids: set[str] = set()
    used_consumer_ids: set[str] = set()
    for row in rows:
        row_id = row["id"]
        definition = definitions[row_id]
        _exact_keys(
            row,
            (
                "id",
                "os",
                "capture_backend",
                "output_backend",
                "transport",
                "status",
                "webcam_ids",
                "consumer_ids",
                "artifacts",
                "measurements",
                "notes",
            ),
            f"physical row {row_id}",
        )
        if row["status"] != "pass":
            _fail(f"physical row {row_id} did not pass")
        for name in ("os", "capture_backend", "output_backend", "transport"):
            if row[name] != definition[name]:
                _fail(f"physical row {row_id}.{name} differs from the manifest")
        if (
            not isinstance(row["webcam_ids"], list)
            or not row["webcam_ids"]
            or len(set(row["webcam_ids"])) != len(row["webcam_ids"])
            or any(item not in webcam_ids for item in row["webcam_ids"])
        ):
            _fail(f"physical row {row_id} has invalid webcam references")
        if (
            not isinstance(row["consumer_ids"], list)
            or not row["consumer_ids"]
            or len(set(row["consumer_ids"])) != len(row["consumer_ids"])
            or any(item not in consumer_ids for item in row["consumer_ids"])
        ):
            _fail(f"physical row {row_id} has invalid consumer references")
        if any(
            consumer_by_id[item]["os"] != definition["os"]
            for item in row["consumer_ids"]
        ):
            _fail(f"physical row {row_id} references a consumer on another OS")
        used_webcam_ids.update(row["webcam_ids"])
        used_consumer_ids.update(row["consumer_ids"])
        artifacts = row["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(required_artifacts):
            _fail(f"physical row {row_id} lacks exact required artifacts")
        for artifact_id, artifact in artifacts.items():
            metadata = _exact_keys(
                artifact,
                ("filename", "sha256", "bytes", "media_type"),
                f"physical row {row_id} artifact {artifact_id}",
            )
            _required_string(metadata["filename"], f"{row_id}.{artifact_id}.filename")
            _valid_sha256(metadata["sha256"], f"{row_id}.{artifact_id}.sha256")
            _positive_int(metadata["bytes"], f"{row_id}.{artifact_id}.bytes")
            if metadata["media_type"] != required_artifacts[artifact_id]:
                _fail(
                    f"physical row {row_id} artifact {artifact_id} media type differs"
                )
            artifact_path = _safe_evidence_file(
                evidence_root,
                metadata["filename"],
                metadata["sha256"],
                metadata["bytes"],
                label=f"physical row {row_id} artifact {artifact_id}",
                seen=seen_artifacts,
                seen_digests=seen_digests,
            )
            _validate_artifact_content(
                artifact_path,
                metadata["media_type"],
                f"physical row {row_id} artifact {artifact_id}",
            )
        measurements = _exact_keys(
            row["measurements"],
            (
                "crop_edge_error_pixels",
                "orientation_error_degrees",
                "transport_mean_absolute_error",
                "luminance_gap_reduction_percent",
                "neutral_error_reduction_percent",
                "skin_hue_drift_degrees",
                "skin_chroma_drift_percent",
                "clothing_hue_drift_degrees",
                "clothing_chroma_drift_percent",
                "steady_state_ev_delta_p95",
                "steady_state_wb_log2_delta_p95",
                "no_temporal_oscillation",
                "preview_api_equal",
                "all_consumers_match",
            ),
            f"physical row {row_id} measurements",
        )
        if (
            _nonnegative_number(
                measurements["crop_edge_error_pixels"],
                f"{row_id}.crop_edge_error_pixels",
            )
            > requirements["maximum_crop_edge_error_pixels"]
            or _nonnegative_number(
                measurements["orientation_error_degrees"],
                f"{row_id}.orientation_error_degrees",
            )
            > requirements["maximum_orientation_error_degrees"]
            or _nonnegative_number(
                measurements["transport_mean_absolute_error"],
                f"{row_id}.transport_mean_absolute_error",
            )
            > requirements["maximum_transport_mean_absolute_error"]
            or _finite_number(
                measurements["luminance_gap_reduction_percent"],
                f"{row_id}.luminance_gap_reduction_percent",
            )
            < requirements["minimum_luminance_gap_reduction_percent"]
            or _finite_number(
                measurements["neutral_error_reduction_percent"],
                f"{row_id}.neutral_error_reduction_percent",
            )
            < requirements["minimum_neutral_error_reduction_percent"]
            or _nonnegative_number(
                measurements["skin_hue_drift_degrees"],
                f"{row_id}.skin_hue_drift_degrees",
            )
            > requirements["maximum_skin_hue_drift_degrees"]
            or _nonnegative_number(
                measurements["skin_chroma_drift_percent"],
                f"{row_id}.skin_chroma_drift_percent",
            )
            > requirements["maximum_skin_chroma_drift_percent"]
            or _nonnegative_number(
                measurements["clothing_hue_drift_degrees"],
                f"{row_id}.clothing_hue_drift_degrees",
            )
            > requirements["maximum_clothing_hue_drift_degrees"]
            or _nonnegative_number(
                measurements["clothing_chroma_drift_percent"],
                f"{row_id}.clothing_chroma_drift_percent",
            )
            > requirements["maximum_clothing_chroma_drift_percent"]
            or _nonnegative_number(
                measurements["steady_state_ev_delta_p95"],
                f"{row_id}.steady_state_ev_delta_p95",
            )
            > requirements["maximum_steady_state_ev_delta_p95"]
            or _nonnegative_number(
                measurements["steady_state_wb_log2_delta_p95"],
                f"{row_id}.steady_state_wb_log2_delta_p95",
            )
            > requirements["maximum_steady_state_wb_log2_delta_p95"]
            or measurements["no_temporal_oscillation"] is not True
            or measurements["preview_api_equal"] is not True
            or measurements["all_consumers_match"] is not True
        ):
            _fail(f"physical row {row_id} exceeds a transport tolerance")
        _required_string(row["notes"], f"physical row {row_id}.notes")
    if used_webcam_ids != set(webcam_ids):
        _fail("every declared physical webcam must contribute matrix evidence")
    if used_consumer_ids != set(consumer_ids):
        _fail("every declared physical consumer must contribute matrix evidence")
    used_profiles = {
        (
            webcam_by_id[item]["auto_white_balance"],
            webcam_by_id[item]["auto_exposure"],
        )
        for item in used_webcam_ids
    }
    if (
        requirements["require_different_auto_control_profiles"]
        and len(used_profiles) < 2
    ):
        _fail("used physical evidence lacks differing auto-control profiles")


def _validate_pending_physical(section: object, manifest: dict[str, Any]) -> None:
    value = _exact_keys(
        section, ("status", "webcams", "consumers", "rows"), "report.physical"
    )
    if value["status"] not in ("pending", "pass", "fail"):
        _fail("physical status is invalid")
    _exact_id_set(
        value["rows"],
        [row["id"] for row in manifest["physical_matrix"]],
        "physical.rows",
    )
    definitions = {row["id"]: row for row in manifest["physical_matrix"]}
    for row in value["rows"]:
        row_id = row["id"]
        definition = definitions[row_id]
        _exact_keys(
            row,
            (
                "id",
                "os",
                "capture_backend",
                "output_backend",
                "transport",
                "status",
                "webcam_ids",
                "consumer_ids",
                "artifacts",
                "measurements",
                "notes",
            ),
            f"physical row {row_id}",
        )
        for name in ("os", "capture_backend", "output_backend", "transport"):
            if row[name] != definition[name]:
                _fail(f"physical row {row_id}.{name} differs from the manifest")
        if row["status"] != "pending":
            _fail("deterministic-only physical rows must remain pending")


def validate_report(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    claim: str,
    evidence_root: Path | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Validate deterministic evidence or the complete release claim."""

    if claim not in ("deterministic", "release"):
        _fail("claim must be deterministic or release")
    _exact_keys(
        report,
        (
            "schema_version",
            "report_id",
            "manifest_id",
            "manifest_sha256",
            "generated_at_utc",
            "baseline",
            "source",
            "candidate",
            "environment",
            "deterministic",
            "calibrated",
            "physical",
            "release_qualified",
        ),
        "visual qualification report",
    )
    if (
        report["schema_version"] != REPORT_SCHEMA_VERSION
        or report["report_id"] != REPORT_ID
        or report["manifest_id"] != manifest["manifest_id"]
    ):
        _fail("visual qualification report header is invalid")
    if report["manifest_sha256"] != _manifest_digest(manifest):
        _fail("visual qualification report is not bound to the reviewed manifest")
    _required_string(report["generated_at_utc"], "generated_at_utc")
    baseline = _exact_keys(report["baseline"], ("path", "sha256"), "report.baseline")
    if baseline != manifest["baseline"]:
        _fail("visual qualification report is not bound to VIS-0.2 baseline")
    source = _exact_keys(
        report["source"], ("commit", "tree", "clean", "files"), "report.source"
    )
    if not isinstance(source["files"], dict) or set(source["files"]) != set(
        SOURCE_FILES
    ):
        _fail("report source file set is incomplete")
    for filename, digest in source["files"].items():
        _valid_sha256(digest, f"source file {filename}")
    if type(source["clean"]) is not bool:
        _fail("report.source.clean must be a boolean")
    candidate = _exact_keys(
        report["candidate"],
        ("status", "manifest_filename", "manifest_sha256"),
        "report.candidate",
    )
    environment = _exact_keys(
        report["environment"],
        ("python", "numpy", "opencv", "platform", "machine"),
        "report.environment",
    )
    for name, value in environment.items():
        _required_string(value, f"environment.{name}")

    deterministic = _validate_deterministic(report["deterministic"], manifest)
    soak = deterministic["soak"]
    if claim == "release":
        if _canonical_json(manifest) != _canonical_json(
            load_manifest(DEFAULT_MANIFEST)
        ):
            _fail("release qualification must use the reviewed default manifest")
        _valid_git_oid(source["commit"], "report.source.commit")
        _valid_git_oid(source["tree"], "report.source.tree")
        if source["clean"] is not True:
            _fail("release qualification requires a clean source checkout")
        bound_commit = source["commit"] if expected_commit is None else expected_commit
        _valid_git_oid(bound_commit, "expected qualification commit")
        if source["commit"] != bound_commit:
            _fail("qualification source commit differs from the expected commit")
        git_root, bridged_git_root, trusted_head = _release_git_context(
            bound_commit, source["tree"]
        )
        if not _git_is_ancestor(bound_commit, cwd=git_root):
            _fail("qualification source commit is not an ancestor of current HEAD")
        expected_tree = _git_value(
            "rev-parse", f"{bound_commit}^{{tree}}", cwd=git_root
        )
        if source["tree"] != expected_tree:
            _fail("qualification source tree differs from the expected commit")
        for filename, digest in source["files"].items():
            if digest != _git_blob_digest(bound_commit, filename, cwd=git_root):
                _fail(
                    f"source file digest differs from qualification commit: {filename}"
                )
        if _git_status(cwd=git_root) != "":
            _fail("release validation requires a clean current checkout")
        if evidence_root is None:
            _fail("release qualification requires --evidence-root")
        seen_artifacts: set[Path] = set()
        seen_digests: set[str] = set()
        if candidate["status"] != "bound":
            _fail("release qualification is not bound to a candidate manifest")
        candidate_path = _safe_evidence_file(
            evidence_root,
            candidate["manifest_filename"],
            candidate["manifest_sha256"],
            None,
            label="candidate manifest",
            seen=seen_artifacts,
            seen_digests=seen_digests,
        )
        _validate_candidate_manifest(candidate_path, source["commit"])
        contact = deterministic["contact_sheet"]
        contact_path = _safe_evidence_file(
            evidence_root,
            contact["filename"],
            contact["sha256"],
            contact["bytes"],
            label="deterministic contact sheet",
            seen=seen_artifacts,
            seen_digests=seen_digests,
        )
        _validate_artifact_content(
            contact_path, "image/png", "deterministic contact sheet"
        )
        if soak["frames"] < manifest["budgets"]["soak_frames"]:
            _fail("release qualification did not run the 10,000-frame soak")
        limit = manifest["budgets"]["maximum_retained_growth_bytes"]
        if soak["tracemalloc_current_growth_bytes"] > limit or (
            soak["rss_growth_bytes"] is not None and soak["rss_growth_bytes"] > limit
        ):
            _fail("release qualification exceeds retained-growth limits")
        if (
            manifest["reference_runner"]["rss_required"]
            and soak["rss_growth_bytes"] is None
        ):
            _fail("release qualification lacks the required soak RSS measurement")
        _validate_calibrated_release(report["calibrated"], manifest)
        _validate_physical_release(
            report["physical"],
            manifest,
            evidence_root=evidence_root,
            seen_artifacts=seen_artifacts,
            seen_digests=seen_digests,
        )
        if bridged_git_root:
            staged_root = ROOT.resolve(strict=True)
            for artifact_path in seen_artifacts:
                try:
                    relative = artifact_path.resolve(strict=True).relative_to(
                        staged_root
                    )
                except (OSError, ValueError):
                    _fail(
                        "trusted Git bridge requires every evidence artifact "
                        "inside the staged source root"
                    )
                filename = relative.as_posix()
                if _sha256_file(artifact_path) != _git_blob_digest(
                    trusted_head, filename, cwd=git_root
                ):
                    _fail(
                        "staged evidence artifact differs from trusted source "
                        f"blob: {filename}"
                    )
        if report["release_qualified"] is not True:
            _fail("complete evidence must explicitly ratify release_qualified=true")
    else:
        for filename, digest in source["files"].items():
            if digest != _sha256_file(ROOT / filename):
                _fail(f"source file digest differs from this checkout: {filename}")
        if source["commit"] != "unavailable":
            _valid_git_oid(source["commit"], "report.source.commit")
        if source["tree"] != "unavailable":
            _valid_git_oid(source["tree"], "report.source.tree")
        if candidate["status"] != "pending":
            _fail("deterministic-only evidence cannot bind a release candidate")
        if (
            candidate["manifest_filename"] is not None
            or candidate["manifest_sha256"] is not None
        ):
            _fail("pending candidate binding must not name an artifact")
        _validate_pending_calibrated(report["calibrated"])
        _validate_pending_physical(report["physical"], manifest)
        if report["release_qualified"] is not False:
            _fail("deterministic-only evidence cannot claim release qualification")
    return report


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        _fail(f"output already exists: {path.name}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="reviewed visual qualification manifest",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run deterministic/local qualification")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--contact-sheet", required=True, type=Path)
    run.add_argument("--soak-frames", type=int, default=None)
    run.add_argument("--calibrated", action="store_true")
    run.add_argument(
        "--observation-only",
        action="store_true",
        help=(
            "write a failing calibrated report with exit 0 for diagnostics; "
            "never use for a qualification gate"
        ),
    )
    run.add_argument("--measured-frames", type=int, default=300)
    run.add_argument("--warmup-frames", type=int, default=5)
    run.add_argument("--pinned-runner", action="store_true")
    run.add_argument("--runner-id", default="")
    run.add_argument("--reference-cpu", default="")

    validate = subparsers.add_parser("validate", help="validate an evidence report")
    validate.add_argument("--report", required=True, type=Path)
    validate.add_argument(
        "--claim", choices=("deterministic", "release"), required=True
    )
    validate.add_argument("--evidence-root", type=Path)
    validate.add_argument("--expected-commit")

    template = subparsers.add_parser(
        "template", help="write a pending physical-evidence template"
    )
    template.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(sys.argv[1:] if argv is None else argv)
        manifest = load_manifest(args.manifest)
        if args.command == "run":
            if args.observation_only and not args.calibrated:
                _fail("--observation-only requires --calibrated")
            soak_frames = (
                manifest["budgets"]["soak_frames"]
                if args.soak_frames is None
                else args.soak_frames
            )
            report = build_report(
                manifest,
                contact_sheet_path=args.contact_sheet,
                soak_frames=soak_frames,
                calibrated=args.calibrated,
                measured_frames=args.measured_frames,
                warmup_frames=args.warmup_frames,
                pinned_runner=args.pinned_runner,
                runner_id=args.runner_id,
                reference_cpu=args.reference_cpu,
            )
            _write_exclusive(args.output, report)
            calibrated_status = report["calibrated"]["status"]
            if args.calibrated and calibrated_status != "pass":
                print(
                    "[custback visual qualification] calibrated budgets failed; "
                    f"report={args.output}",
                    file=sys.stderr,
                )
                return 0 if args.observation_only else 1
            print(
                "[custback visual qualification] requested contracts passed; "
                f"calibrated={calibrated_status}; "
                f"release_qualified={report['release_qualified']}"
            )
            return 0
        if args.command == "validate":
            validate_report(
                _read_json(args.report, "visual qualification report"),
                manifest,
                claim=args.claim,
                evidence_root=args.evidence_root,
                expected_commit=args.expected_commit,
            )
            print(f"[custback visual qualification] {args.claim} evidence is valid")
            return 0
        if args.command == "template":
            template = {
                "manifest_id": manifest["manifest_id"],
                "manifest_sha256": _manifest_digest(manifest),
                "physical": _pending_physical(manifest),
                "schemas": _physical_evidence_schemas(),
                "instructions": (
                    "Merge this physical object into a deterministic + pinned "
                    "calibrated report; replace every pending row with hashed real "
                    "artifacts, then set release_qualified=true and validate the "
                    "release claim."
                ),
            }
            _write_exclusive(args.output, template)
            return 0
        _fail("unknown command")
    except QualificationError as exc:
        print(f"[custback visual qualification] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
