#!/usr/bin/env python3
"""Deterministic, non-production evidence harness for VIS-0.2 and VIS-0.3.

This module deliberately lives with the test suite, outside ``src/custback``.
It measures the current geometry/compositing baseline and compares candidate
color estimators; it is not a runtime implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageCms

ANALYSIS_LONG_EDGE = 192
DEFAULT_STRENGTH = 0.50
DEFAULT_WHITE_BALANCE_STRENGTH = 0.50
EXPOSURE_CLAMP_EV = 0.85
WB_GAIN_MIN = 0.86
WB_GAIN_MAX = 1.16
MIN_CONFIDENCE = 0.45
MIN_SAMPLES = 96
NEAR_BLACK = 0.02
NEAR_CLIP = 0.98
NEUTRAL_SATURATION_MAX = 0.30

GEOMETRY_SIZES = {
    "square": (201, 201),
    "four_three": (240, 320),
    "sixteen_nine": (180, 320),
    "portrait": (320, 180),
    "ultrawide": (144, 384),
    "odd": (181, 319),
}

METHODS = ("exposure_only", "bounded_wb", "aggressive_moment")

_FONT = {
    " ": ("000", "000", "000", "000", "000"),
    ";": ("000", "010", "000", "010", "100"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("110", "101", "110", "100", "100"),
    "Q": ("111", "101", "101", "111", "001"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}


@dataclass(frozen=True)
class SyntheticScene:
    foreground: np.ndarray
    backdrop: np.ndarray
    mask: np.ndarray
    neutral_mask: np.ndarray
    skin_mask: np.ndarray
    clothing_mask: np.ndarray


@dataclass(frozen=True)
class Sampling:
    foreground: np.ndarray
    backdrop: np.ndarray
    core: np.ndarray
    target: np.ndarray
    target_is_local: bool
    mask_coverage: float


@dataclass(frozen=True)
class Estimate:
    method: str
    behavior: str
    exposure_ev: float
    wb_gains: tuple[float, float, float]
    confidence: float
    exposure_confidence: float
    white_balance_confidence: float
    usable_source: int
    usable_target: int
    neutral_source: int
    neutral_target: int
    target_is_local: bool
    moment_scale: tuple[float, float, float] | None = None
    moment_offset: tuple[float, float, float] | None = None


def srgb_to_linear(value: np.ndarray | float) -> np.ndarray:
    """Decode normalized sRGB values to linear-light RGB."""
    encoded = np.asarray(value, dtype=np.float64)
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(value: np.ndarray | float) -> np.ndarray:
    """Encode normalized linear-light RGB values as sRGB."""
    linear = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def _linear_u8(image: np.ndarray) -> np.ndarray:
    return np.rint(linear_to_srgb(image) * 255.0).astype(np.uint8)


def relative_luminance(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def _draw_label(
    image: np.ndarray,
    text: str,
    *,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: int,
) -> None:
    cursor = x
    for character in text:
        glyph = _FONT[character]
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    y0 = y + row * scale
                    x0 = cursor + column * scale
                    image[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += 4 * scale


def geometry_fixture(height: int, width: int) -> np.ndarray:
    """Create an asymmetric RGB grid with labeled corners and a round target."""
    if height < 24 or width < 24:
        raise ValueError("geometry fixture must be at least 24x24")
    y, x = np.indices((height, width))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = (17 + 3 * x + y) % 71
    image[..., 1] = (29 + x + 5 * y) % 83
    image[..., 2] = (11 + 2 * x + 7 * y) % 67

    grid_colors = ((255, 80, 40), (40, 255, 80), (80, 40, 255))
    for fraction, color in zip((0.17, 0.41, 0.73), grid_colors, strict=True):
        column = min(width - 1, round((width - 1) * fraction))
        image[:, max(0, column - 1) : column + 1] = color
    for fraction, color in zip((0.13, 0.52, 0.81), grid_colors[::-1], strict=True):
        row = min(height - 1, round((height - 1) * fraction))
        image[max(0, row - 1) : row + 1, :] = color

    center_x, center_y = width * 0.56, height * 0.44
    radius = min(height, width) * 0.22
    circle_distance = np.abs(np.hypot(x - center_x, y - center_y) - radius)
    image[circle_distance <= 1.1] = (255, 255, 0)

    scale = max(1, min(height, width) // 80)
    margin = max(2, scale)
    label_width = 7 * scale
    label_height = 5 * scale
    labels = (
        ("TL", margin, margin, (255, 0, 0)),
        ("TR", width - margin - label_width, margin, (0, 255, 0)),
        ("BL", margin, height - margin - label_height, (0, 0, 255)),
        (
            "BR",
            width - margin - label_width,
            height - margin - label_height,
            (255, 0, 255),
        ),
    )
    for text, left, top, color in labels:
        _draw_label(
            image,
            text,
            x=max(0, left),
            y=max(0, top),
            color=color,
            scale=scale,
        )
    return image


def apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Apply the pixel operation represented by EXIF orientation 1..8."""
    operations = {
        1: lambda value: value,
        2: np.fliplr,
        3: lambda value: np.rot90(value, 2),
        4: np.flipud,
        5: lambda value: np.transpose(value, (1, 0, 2)),
        6: lambda value: np.rot90(value, 3),
        7: lambda value: np.flip(np.transpose(value, (1, 0, 2)), axis=(0, 1)),
        8: lambda value: np.rot90(value, 1),
    }
    try:
        return np.ascontiguousarray(operations[orientation](image))
    except KeyError as exc:
        raise ValueError("EXIF orientation must be in 1..8") from exc


def stretch_axis_ratio(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> float:
    """Return horizontal/vertical scale; 1.0 is distortion-free."""
    return (target_width / source_width) / (target_height / source_height)


def cover_crop_coordinates(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    *,
    anchor_x: float = 0.5,
    anchor_y: float = 0.5,
) -> dict[str, float]:
    """Return the source-space crop used by an aspect-preserving cover fit."""
    scale = max(target_width / source_width, target_height / source_height)
    visible_width = target_width / scale
    visible_height = target_height / scale
    left = (source_width - visible_width) * anchor_x
    top = (source_height - visible_height) * anchor_y
    return {
        "left": left,
        "top": top,
        "right": left + visible_width,
        "bottom": top + visible_height,
        "scale": scale,
    }


def _ellipse_alpha(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.float64)
    radius = np.sqrt(
        ((x - width * 0.50) / (width * 0.23)) ** 2
        + ((y - height * 0.52) / (height * 0.42)) ** 2
    )
    return np.clip((1.025 - radius) / 0.05, 0.0, 1.0).astype(np.float32)


def synthetic_scene(
    *,
    target_ev: float = 0.70,
    source_cast: tuple[float, float, float] = (0.86, 0.96, 1.12),
    target_cast: tuple[float, float, float] = (1.10, 0.99, 0.83),
    saturated_backdrop: bool = False,
    seed: int = 741_103,
) -> SyntheticScene:
    """Generate a portrait-like linear-RGB scene with protected color patches."""
    height, width = 180, 320
    rng = np.random.default_rng(seed)
    mask = _ellipse_alpha(height, width)
    core = mask >= 0.995

    y, x = np.indices((height, width))
    neutral_level = 0.13 + 0.07 * (y / max(1, height - 1))
    source_illumination = np.asarray(source_cast, dtype=np.float64)
    foreground = neutral_level[..., None] * source_illumination

    skin_mask = (
        core
        & (x >= int(width * 0.42))
        & (x < int(width * 0.58))
        & (y >= int(height * 0.28))
        & (y < int(height * 0.50))
    )
    clothing_mask = (
        core
        & (x >= int(width * 0.38))
        & (x < int(width * 0.62))
        & (y >= int(height * 0.62))
        & (y < int(height * 0.80))
    )
    highlight_mask = (
        core
        & (x >= int(width * 0.44))
        & (x < int(width * 0.49))
        & (y >= int(height * 0.13))
        & (y < int(height * 0.23))
    )
    shadow_mask = (
        core
        & (x >= int(width * 0.51))
        & (x < int(width * 0.56))
        & (y >= int(height * 0.13))
        & (y < int(height * 0.23))
    )
    foreground[skin_mask] = np.array((0.50, 0.235, 0.135)) * source_illumination
    foreground[clothing_mask] = np.array((0.035, 0.12, 0.64)) * source_illumination
    foreground[highlight_mask] = np.array((1.04, 1.02, 0.99))
    foreground[shadow_mask] = np.array((0.006, 0.008, 0.011))
    foreground += rng.normal(0.0, 0.0015, foreground.shape)
    foreground = np.clip(foreground, 0.0, 1.0).astype(np.float32)
    foreground[~(mask > 0.0)] = 0.0

    neutral_mask = core & ~skin_mask & ~clothing_mask & ~highlight_mask & ~shadow_mask

    if saturated_backdrop:
        backdrop = np.empty_like(foreground)
        backdrop[:] = np.array((0.82, 0.018, 0.72), dtype=np.float32)
    else:
        target_illumination = np.asarray(target_cast, dtype=np.float64)
        target_level = (0.16 + 0.06 * (x / max(1, width - 1))) * (2.0**target_ev)
        backdrop = target_level[..., None] * target_illumination
        backdrop += rng.normal(0.0, 0.0015, backdrop.shape)
        backdrop = np.clip(backdrop, 0.0, 1.0).astype(np.float32)
    return SyntheticScene(
        foreground=foreground,
        backdrop=backdrop,
        mask=mask,
        neutral_mask=neutral_mask,
        skin_mask=skin_mask,
        clothing_mask=clothing_mask,
    )


def exposure_pair_scene(ev: float) -> SyntheticScene:
    """Generate an exact neutral exposure pair for positive or negative EV."""
    if not math.isfinite(ev):
        raise ValueError("exposure EV must be finite")
    height, width = 180, 320
    mask = _ellipse_alpha(height, width)
    foreground = np.full((height, width, 3), 0.20, dtype=np.float32)
    backdrop = np.full(
        (height, width, 3),
        np.clip(0.20 * (2.0**ev), 0.0, 1.0),
        dtype=np.float32,
    )
    core = mask >= 0.995
    empty = np.zeros((height, width), dtype=bool)
    return SyntheticScene(
        foreground=foreground,
        backdrop=backdrop,
        mask=mask,
        neutral_mask=core,
        skin_mask=empty,
        clothing_mask=empty,
    )


def temporal_fixture_sequence() -> list[SyntheticScene | None]:
    """Static noise, slow drift, hard cut, reconnect, and stable recovery."""
    frames: list[SyntheticScene | None] = []
    for index in range(8):
        frames.append(synthetic_scene(target_ev=0.20, seed=80_000 + index))
    for index in range(8):
        frames.append(
            synthetic_scene(
                target_ev=0.20 + 0.025 * index,
                target_cast=(1.10 + 0.008 * index, 0.99, 0.83 - 0.006 * index),
                seed=81_000 + index,
            )
        )
    for index in range(4):
        frames.append(
            synthetic_scene(
                target_ev=-0.35,
                target_cast=(0.82, 0.98, 1.15),
                seed=82_000 + index,
            )
        )
    frames.append(None)
    for index in range(7):
        frames.append(
            synthetic_scene(
                target_ev=-0.35,
                target_cast=(0.82, 0.98, 1.15),
                seed=83_000 + index,
            )
        )
    return frames


def _analysis_inputs(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    *,
    long_edge: int = ANALYSIS_LONG_EDGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = foreground.shape[:2]
    if max(height, width) <= long_edge:
        return foreground.copy(), backdrop.copy(), mask.copy()
    scale = long_edge / max(height, width)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return (
        cv2.resize(foreground, target, interpolation=cv2.INTER_AREA),
        cv2.resize(backdrop, target, interpolation=cv2.INTER_AREA),
        cv2.resize(mask, target, interpolation=cv2.INTER_AREA),
    )


def sampling_regions(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
) -> Sampling:
    """Build eroded foreground-core and local-backdrop analysis regions."""
    small_fg, small_bg, small_mask = _analysis_inputs(foreground, backdrop, mask)
    hard_core = (small_mask >= 0.90).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    core = cv2.erode(hard_core, kernel, iterations=1).astype(bool)

    support = (small_mask > 0.05).astype(np.uint8)
    dilated = cv2.dilate(support, np.ones((19, 19), dtype=np.uint8), iterations=1)
    local_target = (dilated > 0) & (support == 0)
    target_is_local = int(local_target.sum()) >= MIN_SAMPLES
    if target_is_local:
        target = local_target
    else:
        outside = support == 0
        target = outside if int(outside.sum()) >= MIN_SAMPLES else np.ones_like(core)
    return Sampling(
        foreground=small_fg,
        backdrop=small_bg,
        core=core,
        target=target,
        target_is_local=target_is_local,
        mask_coverage=float(np.mean(small_mask >= 0.90)),
    )


def _valid_luminance(pixels: np.ndarray) -> np.ndarray:
    luminance = relative_luminance(pixels)
    return (
        (luminance > NEAR_BLACK)
        & (np.max(pixels, axis=1) < NEAR_CLIP)
        & np.isfinite(pixels).all(axis=1)
    )


def _neutral(pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    maximum = np.max(pixels, axis=1)
    minimum = np.min(pixels, axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    return valid & (saturation <= NEUTRAL_SATURATION_MAX)


def _identity_estimate(method: str, sampling: Sampling) -> Estimate:
    return Estimate(
        method=method,
        behavior="identity",
        exposure_ev=0.0,
        wb_gains=(1.0, 1.0, 1.0),
        confidence=0.0,
        exposure_confidence=0.0,
        white_balance_confidence=0.0,
        usable_source=0,
        usable_target=0,
        neutral_source=0,
        neutral_target=0,
        target_is_local=sampling.target_is_local,
    )


def estimate_transform(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    mask: np.ndarray,
    *,
    method: str,
    strength: float = DEFAULT_STRENGTH,
    white_balance_strength: float = DEFAULT_WHITE_BALANCE_STRENGTH,
) -> Estimate:
    """Estimate one of the three VIS-0.3 candidates on a 192px analysis copy."""
    if method not in METHODS:
        raise ValueError(f"unknown evidence method: {method}")
    sampling = sampling_regions(foreground, backdrop, mask)
    source = sampling.foreground[sampling.core]
    target = sampling.backdrop[sampling.target]
    if (
        len(source) < MIN_SAMPLES
        or len(target) < MIN_SAMPLES
        or sampling.mask_coverage < 0.01
    ):
        return _identity_estimate(method, sampling)

    source_valid = _valid_luminance(source)
    target_valid = _valid_luminance(target)
    usable_source = int(source_valid.sum())
    usable_target = int(target_valid.sum())
    if usable_source < MIN_SAMPLES or usable_target < MIN_SAMPLES:
        return _identity_estimate(method, sampling)

    source_luminance = relative_luminance(source[source_valid])
    target_luminance = relative_luminance(target[target_valid])
    raw_ev = float(
        np.median(np.log2(target_luminance)) - np.median(np.log2(source_luminance))
    )
    bounded_ev = float(np.clip(raw_ev, -EXPOSURE_CLAMP_EV, EXPOSURE_CLAMP_EV))
    exposure_ev = bounded_ev * strength

    count_score = min(1.0, min(usable_source, usable_target) / 512.0)
    valid_score = min(
        usable_source / max(1, len(source)),
        usable_target / max(1, len(target)),
    )
    coverage_score = min(1.0, sampling.mask_coverage / 0.10)
    exposure_confidence = float(count_score * valid_score * coverage_score)
    if exposure_confidence < MIN_CONFIDENCE:
        return _identity_estimate(method, sampling)

    source_neutral = _neutral(source, source_valid)
    target_neutral = _neutral(target, target_valid)
    neutral_source = int(source_neutral.sum())
    neutral_target = int(target_neutral.sum())
    neutral_availability_score = (
        min(1.0, min(neutral_source, neutral_target) / MIN_SAMPLES)
        if sampling.target_is_local
        else 0.0
    )
    white_balance_confidence = float(exposure_confidence * neutral_availability_score)

    def candidate(
        behavior: str,
        *,
        wb_gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
        moment_scale: tuple[float, float, float] | None = None,
        moment_offset: tuple[float, float, float] | None = None,
    ) -> Estimate:
        return Estimate(
            method=method,
            behavior=behavior,
            exposure_ev=exposure_ev,
            wb_gains=wb_gains,
            confidence=(
                white_balance_confidence
                if behavior == "bounded_wb"
                else exposure_confidence
            ),
            exposure_confidence=exposure_confidence,
            white_balance_confidence=white_balance_confidence,
            usable_source=usable_source,
            usable_target=usable_target,
            neutral_source=neutral_source,
            neutral_target=neutral_target,
            target_is_local=sampling.target_is_local,
            moment_scale=moment_scale,
            moment_offset=moment_offset,
        )

    if method == "exposure_only":
        return candidate("exposure_only")

    if method == "aggressive_moment":
        source_selected = source[source_valid]
        target_selected = target[target_valid]
        source_mean = np.mean(source_selected, axis=0)
        target_mean = np.mean(target_selected, axis=0)
        source_std = np.maximum(np.std(source_selected, axis=0), 0.015)
        target_std = np.maximum(np.std(target_selected, axis=0), 0.015)
        moment_scale = target_std / source_std
        moment_offset = target_mean - source_mean * moment_scale
        return candidate(
            "aggressive_moment",
            moment_scale=(
                float(moment_scale[0]),
                float(moment_scale[1]),
                float(moment_scale[2]),
            ),
            moment_offset=(
                float(moment_offset[0]),
                float(moment_offset[1]),
                float(moment_offset[2]),
            ),
        )

    can_adapt_wb = (
        sampling.target_is_local
        and neutral_source >= MIN_SAMPLES
        and neutral_target >= MIN_SAMPLES
        and white_balance_confidence >= MIN_CONFIDENCE
    )
    if not can_adapt_wb:
        return candidate("exposure_only")

    source_rgb = np.median(source[source_neutral], axis=0)
    target_rgb = np.median(target[target_neutral], axis=0)
    raw_gains = target_rgb / np.maximum(source_rgb, 1e-6)
    raw_gains /= np.exp(np.mean(np.log(np.maximum(raw_gains, 1e-6))))
    bounded_gains = np.clip(raw_gains, WB_GAIN_MIN, WB_GAIN_MAX)
    applied_gains = np.exp(np.log(bounded_gains) * white_balance_strength)
    return candidate(
        "bounded_wb",
        wb_gains=(
            float(applied_gains[0]),
            float(applied_gains[1]),
            float(applied_gains[2]),
        ),
    )


def apply_estimate(image: np.ndarray, estimate: Estimate) -> np.ndarray:
    """Apply an evidence estimate to a linear-RGB image."""
    if estimate.behavior == "identity":
        return image.copy()
    if estimate.behavior == "aggressive_moment":
        assert estimate.moment_scale is not None
        assert estimate.moment_offset is not None
        scale = np.asarray(estimate.moment_scale)
        offset = np.asarray(estimate.moment_offset)
        return np.clip(image * scale + offset, 0.0, 1.0).astype(np.float32)
    gains = (2.0**estimate.exposure_ev) * np.asarray(estimate.wb_gains)
    return np.clip(image * gains, 0.0, 1.0).astype(np.float32)


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    matrix = np.array(
        (
            (0.4124564, 0.3575761, 0.1804375),
            (0.2126729, 0.7151522, 0.0721750),
            (0.0193339, 0.1191920, 0.9503041),
        )
    )
    xyz = rgb @ matrix.T
    xyz /= np.array((0.95047, 1.0, 1.08883))
    delta = 6.0 / 29.0
    f = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ),
        axis=-1,
    )


def _median_lab(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    values = _rgb_to_lab(image[region])
    return np.median(values, axis=0)


def _hue_degrees(lab: np.ndarray) -> float:
    return math.degrees(math.atan2(float(lab[2]), float(lab[1]))) % 360.0


def _angular_difference(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def patch_preservation(
    before: np.ndarray,
    after: np.ndarray,
    region: np.ndarray,
) -> dict[str, float]:
    original = _median_lab(before, region)
    corrected = _median_lab(after, region)
    original_chroma = float(np.hypot(original[1], original[2]))
    corrected_chroma = float(np.hypot(corrected[1], corrected[2]))
    original_normalized_chroma = original_chroma / max(float(original[0]), 1e-6)
    corrected_normalized_chroma = corrected_chroma / max(float(corrected[0]), 1e-6)
    return {
        "hue_drift_degrees": _angular_difference(
            _hue_degrees(original), _hue_degrees(corrected)
        ),
        "normalized_chroma_drift_percent": (
            abs(corrected_normalized_chroma - original_normalized_chroma)
            / max(original_normalized_chroma, 1e-6)
            * 100.0
        ),
        "delta_e76": float(np.linalg.norm(corrected - original)),
    }


def _metric_regions(scene: SyntheticScene) -> tuple[np.ndarray, np.ndarray]:
    sampling = sampling_regions(scene.foreground, scene.backdrop, scene.mask)
    # Map analysis masks back with nearest-neighbor only for metric selection.
    size = (scene.foreground.shape[1], scene.foreground.shape[0])
    core = cv2.resize(
        sampling.core.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    target = cv2.resize(
        sampling.target.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return core & scene.neutral_mask, target


def log_luminance_gap(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    foreground_region: np.ndarray,
    backdrop_region: np.ndarray,
) -> float:
    source = relative_luminance(foreground[foreground_region])
    target = relative_luminance(backdrop[backdrop_region])
    return float(
        abs(
            np.median(np.log2(np.maximum(target, 1e-6)))
            - np.median(np.log2(np.maximum(source, 1e-6)))
        )
    )


def neutral_axis_error(
    foreground: np.ndarray,
    backdrop: np.ndarray,
    foreground_region: np.ndarray,
    backdrop_region: np.ndarray,
) -> float:
    source = _median_lab(foreground, foreground_region)
    target = _median_lab(backdrop, backdrop_region)
    return float(np.hypot(source[1] - target[1], source[2] - target[2]))


def method_metrics(scene: SyntheticScene, method: str) -> dict[str, Any]:
    estimate = estimate_transform(
        scene.foreground, scene.backdrop, scene.mask, method=method
    )
    corrected = apply_estimate(scene.foreground, estimate)
    foreground_region, backdrop_region = _metric_regions(scene)
    before_gap = log_luminance_gap(
        scene.foreground,
        scene.backdrop,
        foreground_region,
        backdrop_region,
    )
    after_gap = log_luminance_gap(
        corrected,
        scene.backdrop,
        foreground_region,
        backdrop_region,
    )
    before_neutral = neutral_axis_error(
        scene.foreground,
        scene.backdrop,
        foreground_region,
        backdrop_region,
    )
    after_neutral = neutral_axis_error(
        corrected,
        scene.backdrop,
        foreground_region,
        backdrop_region,
    )
    return {
        "estimate": _rounded_mapping(asdict(estimate)),
        "luminance_gap_ev": {
            "before": round(before_gap, 6),
            "after": round(after_gap, 6),
            "reduction_percent": round(
                100.0 * (before_gap - after_gap) / max(before_gap, 1e-6), 3
            ),
        },
        "neutral_axis_error_delta_e_ab": {
            "before": round(before_neutral, 6),
            "after": round(after_neutral, 6),
            "reduction_percent": round(
                100.0 * (before_neutral - after_neutral) / max(before_neutral, 1e-6),
                3,
            ),
        },
        "skin_preservation": _rounded_mapping(
            patch_preservation(scene.foreground, corrected, scene.skin_mask)
        ),
        "clothing_preservation": _rounded_mapping(
            patch_preservation(scene.foreground, corrected, scene.clothing_mask)
        ),
    }


def temporal_metrics() -> dict[str, Any]:
    estimates: list[Estimate | None] = []
    for scene in temporal_fixture_sequence():
        estimates.append(
            None
            if scene is None
            else estimate_transform(
                scene.foreground,
                scene.backdrop,
                scene.mask,
                method="bounded_wb",
            )
        )

    static = [estimate for estimate in estimates[:8] if estimate is not None]
    static_ev_delta = np.abs(np.diff([estimate.exposure_ev for estimate in static]))
    static_gain_delta = []
    for previous, current in zip(static, static[1:], strict=False):
        static_gain_delta.append(
            max(
                abs(math.log2(b / a))
                for a, b in zip(previous.wb_gains, current.wb_gains, strict=True)
            )
        )

    post_cut = [estimate for estimate in estimates[16:20] if estimate is not None]
    stable_ev = float(np.median([estimate.exposure_ev for estimate in post_cut]))
    stable_gain = np.median(
        np.array([estimate.wb_gains for estimate in post_cut]), axis=0
    )
    settling_frames = None
    for offset, estimate in enumerate(estimates[16:20]):
        if estimate is None:
            continue
        ev_close = abs(estimate.exposure_ev - stable_ev) <= 0.05
        gain_close = (
            max(
                abs(math.log2(value / reference))
                for value, reference in zip(estimate.wb_gains, stable_gain, strict=True)
            )
            <= 0.02
        )
        if ev_close and gain_close:
            settling_frames = offset
            break

    reconnect = estimates[20]
    first_after_reconnect = estimates[21]
    slow_start = estimates[8]
    slow_end = estimates[15]
    assert slow_start is not None
    assert slow_end is not None
    return {
        "static_noise_ev_delta_p95": round(
            float(np.percentile(static_ev_delta, 95)), 6
        ),
        "static_noise_gain_delta_ev_p95": round(
            float(np.percentile(static_gain_delta, 95)), 6
        ),
        "slow_drift_total_applied_ev": round(
            slow_end.exposure_ev - slow_start.exposure_ev,
            6,
        ),
        "hard_cut_instantaneous_settling_frames": settling_frames,
        "reconnect_frame_has_no_estimate": reconnect is None,
        "first_valid_frame_after_reconnect_behavior": (
            None if first_after_reconnect is None else first_after_reconnect.behavior
        ),
    }


def edge_blend_metrics() -> dict[str, float | int]:
    foreground = 200
    alpha = 0.5
    encoded_result = int(np.uint8(foreground * alpha))
    foreground_linear = float(srgb_to_linear(foreground / 255.0))
    linear_result = int(round(float(linear_to_srgb(foreground_linear * alpha)) * 255))
    encoded_result_linear = float(srgb_to_linear(encoded_result / 255.0))
    expected_linear = foreground_linear * alpha
    return {
        "legacy_encoded_u8": encoded_result,
        "linear_light_reference_u8": linear_result,
        "edge_linear_luminance_error_percent": round(
            100.0
            * (encoded_result_linear - expected_linear)
            / max(expected_linear, 1e-12),
            3,
        ),
    }


def exposure_pair_metrics() -> dict[str, Any]:
    """Measure real +1/-1 EV fixtures through the exposure-only estimator."""
    result = {}
    for name, ev in (("plus_one_ev", 1.0), ("minus_one_ev", -1.0)):
        scene = exposure_pair_scene(ev)
        estimate = estimate_transform(
            scene.foreground,
            scene.backdrop,
            scene.mask,
            method="exposure_only",
        )
        corrected = apply_estimate(scene.foreground, estimate)
        foreground_region, backdrop_region = _metric_regions(scene)
        result[name] = {
            "requested_ev": ev,
            "gap_before_ev": round(
                log_luminance_gap(
                    scene.foreground,
                    scene.backdrop,
                    foreground_region,
                    backdrop_region,
                ),
                6,
            ),
            "applied_ev": round(estimate.exposure_ev, 6),
            "gap_after_ev": round(
                log_luminance_gap(
                    corrected,
                    scene.backdrop,
                    foreground_region,
                    backdrop_region,
                ),
                6,
            ),
            "behavior": estimate.behavior,
        }
    return result


def edge_case_metrics() -> dict[str, Any]:
    normal = synthetic_scene()
    saturated = synthetic_scene(saturated_backdrop=True)
    masks = {
        "all_zero": np.zeros_like(normal.mask),
        "all_one": np.ones_like(normal.mask),
        "tiny": np.zeros_like(normal.mask),
    }
    masks["tiny"][88:92, 158:162] = 1.0
    clipped_foreground = normal.foreground.copy()
    clipped_foreground[normal.mask >= 0.9] = 1.0
    full_frame_foreground = normal.foreground.copy()
    full_frame_foreground[normal.mask <= 0.0] = np.array((0.14, 0.16, 0.19))

    cases = {
        "saturated_backdrop": estimate_transform(
            saturated.foreground,
            saturated.backdrop,
            saturated.mask,
            method="bounded_wb",
        ),
        "all_zero_mask": estimate_transform(
            normal.foreground,
            normal.backdrop,
            masks["all_zero"],
            method="bounded_wb",
        ),
        "all_one_mask": estimate_transform(
            full_frame_foreground,
            normal.backdrop,
            masks["all_one"],
            method="bounded_wb",
        ),
        "tiny_mask": estimate_transform(
            normal.foreground,
            normal.backdrop,
            masks["tiny"],
            method="bounded_wb",
        ),
        "clipped_foreground": estimate_transform(
            clipped_foreground,
            normal.backdrop,
            normal.mask,
            method="bounded_wb",
        ),
    }
    return {
        name: {
            "behavior": estimate.behavior,
            "confidence": round(estimate.confidence, 6),
            "exposure_confidence": round(estimate.exposure_confidence, 6),
            "white_balance_confidence": round(estimate.white_balance_confidence, 6),
            "neutral_source": estimate.neutral_source,
            "neutral_target": estimate.neutral_target,
            "target_is_local": estimate.target_is_local,
        }
        for name, estimate in cases.items()
    }


def _fixed(value: float) -> bytes:
    return struct.pack(">i", round(value * 65_536))


def _xyz_tag(values: Iterable[float]) -> bytes:
    return b"XYZ " + b"\0" * 4 + b"".join(_fixed(value) for value in values)


def _text_tag(value: str) -> bytes:
    return b"text" + b"\0" * 4 + value.encode("ascii") + b"\0"


def _description_tag(value: str) -> bytes:
    encoded = value.encode("utf-16-be")
    return (
        b"mluc"
        + b"\0" * 4
        + struct.pack(">II", 1, 12)
        + b"enUS"
        + struct.pack(">II", len(encoded), 28)
        + encoded
    )


def _parametric_srgb_tag() -> bytes:
    values = (2.4, 1.0 / 1.055, 0.055 / 1.055, 0.0, 0.04045, 1 / 12.92, 0.0)
    return (
        b"para"
        + b"\0" * 4
        + struct.pack(">HH", 4, 0)
        + b"".join(_fixed(value) for value in values)
    )


def _gamma_tag(gamma: float) -> bytes:
    return b"curv" + b"\0" * 4 + struct.pack(">IH", 1, round(gamma * 256))


def _xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array((x / y, 1.0, (1.0 - x - y) / y))


def _rgb_colorants(
    primaries: tuple[tuple[float, float], ...],
) -> np.ndarray:
    d65 = _xy_to_xyz(0.3127, 0.3290)
    base = np.stack([_xy_to_xyz(*primary) for primary in primaries], axis=1)
    matrix_d65 = base @ np.diag(np.linalg.solve(base, d65))
    d50 = _xy_to_xyz(0.34567, 0.35850)
    bradford = np.array(
        (
            (0.8951, 0.2664, -0.1614),
            (-0.7502, 1.7135, 0.0367),
            (0.0389, -0.0685, 1.0296),
        )
    )
    adaptation = (
        np.linalg.inv(bradford)
        @ np.diag((bradford @ d50) / (bradford @ d65))
        @ bradford
    )
    return adaptation @ matrix_d65


def _icc_profile(
    *,
    description: str,
    color_space: bytes,
    profile_class: bytes,
    tags: dict[bytes, bytes],
    version: int,
) -> bytes:
    ordered = sorted(tags.items())
    table_length = 4 + len(ordered) * 12
    offset = 128 + table_length
    entries: list[tuple[bytes, int, int]] = []
    payload = bytearray()
    for signature, data in ordered:
        padding = (-offset) % 4
        payload.extend(b"\0" * padding)
        offset += padding
        entries.append((signature, offset, len(data)))
        payload.extend(data)
        offset += len(data)

    profile_size = 128 + table_length + len(payload)
    header = bytearray(128)
    struct.pack_into(">I", header, 0, profile_size)
    header[4:8] = b"CDEX"
    struct.pack_into(">I", header, 8, version)
    header[12:16] = profile_class
    header[16:20] = color_space
    header[20:24] = b"XYZ "
    struct.pack_into(">6H", header, 24, 2026, 7, 30, 12, 0, 0)
    header[36:40] = b"acsp"
    header[40:44] = b"APPL"
    header[68:80] = _xyz_tag((0.9642, 1.0, 0.8249))[8:]
    header[80:84] = b"CDEX"

    table = bytearray(struct.pack(">I", len(entries)))
    for signature, tag_offset, length in entries:
        table.extend(signature + struct.pack(">II", tag_offset, length))
    profile = bytes(header + table + payload)
    if len(profile) != profile_size:
        raise AssertionError(f"invalid generated profile size for {description}")
    return profile


def _rgb_profile(
    name: str,
    primaries: tuple[tuple[float, float], ...],
    *,
    gamma: float | None,
) -> bytes:
    colorants = _rgb_colorants(primaries)
    trc = _parametric_srgb_tag() if gamma is None else _gamma_tag(gamma)
    return _icc_profile(
        description=name,
        color_space=b"RGB ",
        profile_class=b"mntr",
        version=0x04300000,
        tags={
            b"bTRC": trc,
            b"bXYZ": _xyz_tag(colorants[:, 2]),
            b"cprt": _text_tag("CC0 generated VIS-0 evidence profile"),
            b"desc": _description_tag(name),
            b"gTRC": trc,
            b"gXYZ": _xyz_tag(colorants[:, 1]),
            b"rTRC": trc,
            b"rXYZ": _xyz_tag(colorants[:, 0]),
            b"wtpt": _xyz_tag((0.9642, 1.0, 0.8249)),
        },
    )


def _cmyk_lut_tag() -> bytes:
    matrix = b"".join(
        _fixed(value) for value in (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )
    identity = bytes(range(256))
    clut = bytearray()
    rgb_to_xyz = _rgb_colorants(((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)))
    for cyan in (0.0, 1.0):
        for magenta in (0.0, 1.0):
            for yellow in (0.0, 1.0):
                for black in (0.0, 1.0):
                    rgb = np.array(
                        (
                            (1.0 - cyan) * (1.0 - black),
                            (1.0 - magenta) * (1.0 - black),
                            (1.0 - yellow) * (1.0 - black),
                        )
                    )
                    xyz = rgb_to_xyz @ rgb
                    clut.extend(
                        np.clip(np.rint(xyz / 1.999969 * 255.0), 0, 255).astype(
                            np.uint8
                        )
                    )
    return (
        b"mft1"
        + b"\0" * 4
        + bytes((4, 3, 2, 0))
        + matrix
        + identity * 4
        + bytes(clut)
        + identity * 3
    )


def generated_profiles() -> dict[str, bytes]:
    """Return deterministic ICC fixtures; no external profile files are needed."""
    return {
        "sRGB": _rgb_profile(
            "VIS-0 generated sRGB",
            ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
            gamma=None,
        ),
        "Display-P3": _rgb_profile(
            "VIS-0 generated Display-P3",
            ((0.68, 0.32), (0.265, 0.69), (0.15, 0.06)),
            gamma=None,
        ),
        "Adobe-RGB": _rgb_profile(
            "VIS-0 generated Adobe-RGB",
            ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)),
            gamma=2.19921875,
        ),
        "CMYK": _icc_profile(
            description="VIS-0 generated synthetic CMYK",
            color_space=b"CMYK",
            profile_class=b"scnr",
            version=0x02100000,
            tags={
                b"A2B0": _cmyk_lut_tag(),
                b"cprt": _text_tag("CC0 generated VIS-0 evidence profile"),
                b"desc": _description_tag("VIS-0 generated synthetic CMYK"),
                b"wtpt": _xyz_tag((0.9642, 1.0, 0.8249)),
            },
        ),
    }


def tagged_image_fixtures() -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    """Generate profile-tagged images and a provenance/hash manifest."""
    profiles = generated_profiles()
    payloads: dict[str, bytes] = {}
    manifest: dict[str, dict[str, Any]] = {}
    rgb_pixels = geometry_fixture(33, 47)
    cmyk_pixels = Image.fromarray(rgb_pixels, mode="RGB").convert("CMYK")
    for name, profile in profiles.items():
        buffer = io.BytesIO()
        if name == "CMYK":
            cmyk_pixels.save(
                buffer,
                format="TIFF",
                compression="raw",
                icc_profile=profile,
            )
            expected_mode = "CMYK"
            container = "TIFF"
        else:
            Image.fromarray(rgb_pixels, mode="RGB").save(
                buffer,
                format="PNG",
                icc_profile=profile,
            )
            expected_mode = "RGB"
            container = "PNG"
        payload = buffer.getvalue()
        reopened = Image.open(io.BytesIO(payload))
        embedded = reopened.info.get("icc_profile")
        if embedded != profile or reopened.mode != expected_mode:
            raise AssertionError(f"tagged {name} fixture did not round-trip")
        # Parsing proves the embedded bytes are an ICC profile, not a label.
        parsed = ImageCms.ImageCmsProfile(io.BytesIO(profile)).profile
        payloads[name] = payload
        manifest[name] = {
            "container": container,
            "mode": expected_mode,
            "dimensions": [47, 33],
            "profile_description": ImageCms.getProfileDescription(parsed).strip(),
            "profile_sha256": hashlib.sha256(profile).hexdigest(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_bytes": len(payload),
            "generator": "tests/visual_consistency_evidence.py",
            "provenance": "generated in repository; CC0; no external binary",
        }
    return payloads, manifest


def _percentile_timings(
    callback: Any,
    *,
    iterations: int,
    warmup: int = 3,
) -> dict[str, float | int]:
    for _ in range(warmup):
        callback()
    values = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        callback()
        values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "iterations": iterations,
        "median_ms": round(float(np.median(values)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }


def timing_evidence(iterations: int = 20) -> dict[str, Any]:
    """Offline analogues of existing composite_ms/frame_processing_ms stats."""
    from custback.compositor import composite
    from custback.geometry import transform_frame

    rng = np.random.default_rng(91_771)
    camera = rng.integers(0, 256, (720, 960, 3), dtype=np.uint8)
    output_background = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    output_foreground = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    mask = rng.random((720, 1280), dtype=np.float32)
    scene = synthetic_scene()
    return {
        "current_direct_resize_960x720_to_1280x720": _percentile_timings(
            lambda: cv2.resize(camera, (1280, 720), interpolation=cv2.INTER_LINEAR),
            iterations=iterations,
        ),
        "current_background_cover_fit_960x720_to_1280x720": _percentile_timings(
            lambda: transform_frame(camera, (1280, 720), fit="cover")[0],
            iterations=iterations,
        ),
        "current_legacy_composite_720p": _percentile_timings(
            lambda: composite(output_foreground, output_background, mask),
            iterations=iterations,
        ),
        "bounded_estimator_320x180_analysis_to_192px": _percentile_timings(
            lambda: estimate_transform(
                scene.foreground,
                scene.backdrop,
                scene.mask,
                method="bounded_wb",
            ),
            iterations=iterations,
        ),
        "interpretation": (
            "Offline microbenchmarks map to existing background_ms/composite_ms/"
            "frame_processing_ms fields; they are host observations, not CI gates."
        ),
    }


def runtime_stats_evidence(
    *,
    minimum_frames: int = 24,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Capture an actual EWMA timing snapshot from the production stats path."""
    from custback.config import AppConfig, RuntimeConfig
    from custback.hub import FrameHub
    from custback.pipeline import Pipeline

    if minimum_frames < 1 or timeout_s <= 0:
        raise ValueError("runtime timing bounds must be positive")
    config = AppConfig.from_dict(
        {
            "camera": {
                "synthetic": True,
                "width": 320,
                "height": 180,
                "fps": 60,
            },
            "background": {"mode": "color", "color": [18, 100, 32]},
            "segmentation": {
                "backend": "heuristic",
                "temporal_smoothing": 0.0,
            },
            "output": {"backend": "null", "fps": 60},
            "api": {"enabled": False},
        }
    )
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(config), hub)
    pipeline.start(timeout=timeout_s)
    try:
        deadline = time.monotonic() + timeout_s
        while hub.stats_dict()["frames_out"] < minimum_frames:
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out collecting production timing statistics")
            time.sleep(0.01)
        stats = hub.stats_dict()
    finally:
        pipeline.stop(timeout=timeout_s)

    timing_fields = (
        "capture_read_ms",
        "segmentation_ms",
        "background_ms",
        "composite_ms",
        "output_send_ms",
        "frame_processing_ms",
    )
    timings = {name: stats[name] for name in timing_fields}
    if any(value is None for value in timings.values()):
        raise RuntimeError("production timing statistics were not populated")
    return {
        "source": "FrameHub.stats_dict production EWMA fields",
        "configuration": {
            "canvas": [320, 180],
            "camera": "synthetic",
            "segmentation": "heuristic",
            "background": "color",
            "output": "null",
            "target_fps": 60,
        },
        "frames_out": stats["frames_out"],
        "fps": stats["fps"],
        "timings_ms": timings,
    }


def _rounded_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rounded_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded_mapping(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def deterministic_evidence() -> dict[str, Any]:
    scene = synthetic_scene()
    _, profiles = tagged_image_fixtures()
    geometry = {
        "fixture_sizes_height_width": {
            name: list(size) for name, size in GEOMETRY_SIZES.items()
        },
        "stretch_4_3_to_16_9": {
            "axis_ratio": round(stretch_axis_ratio(320, 240, 320, 180), 6),
            "distortion_percent": round(
                abs(stretch_axis_ratio(320, 240, 320, 180) - 1.0) * 100.0, 3
            ),
        },
        "cover_4_3_to_16_9_center_crop": _rounded_mapping(
            cover_crop_coordinates(320, 240, 320, 180)
        ),
        "orientation_output_shapes": {
            str(orientation): list(
                apply_exif_orientation(geometry_fixture(31, 47), orientation).shape[:2]
            )
            for orientation in range(1, 9)
        },
    }
    result = {
        "schema": "custback.visual_consistency.phase0.evidence.v1",
        "constants": {
            "analysis_long_edge": ANALYSIS_LONG_EDGE,
            "default_strength": DEFAULT_STRENGTH,
            "default_white_balance_strength": DEFAULT_WHITE_BALANCE_STRENGTH,
            "exposure_clamp_ev": EXPOSURE_CLAMP_EV,
            "wb_gain_range": [WB_GAIN_MIN, WB_GAIN_MAX],
            "minimum_confidence": MIN_CONFIDENCE,
            "minimum_samples": MIN_SAMPLES,
            "near_black": NEAR_BLACK,
            "near_clip": NEAR_CLIP,
            "neutral_saturation_max": NEUTRAL_SATURATION_MAX,
        },
        "fixtures": {
            "geometry": geometry,
            "profiles": profiles,
            "color_cases": [
                "warm_and_cool_neutral_casts",
                "plus_minus_one_ev",
                "clipped_highlights",
                "deep_shadows",
                "skin_like_patch",
                "saturated_clothing_patch",
                "neutral_and_saturated_backdrops",
                "static_noise",
                "slow_drift",
                "hard_scene_cut",
                "reconnect",
            ],
            "rotation_metadata_video": {
                "generated": False,
                "reason": (
                    "No ffmpeg/ffprobe in the qualification environment; EXIF "
                    "orientations 1-8 are covered in-memory."
                ),
            },
        },
        "current_baseline": {
            "geometry": geometry["stretch_4_3_to_16_9"],
            "alpha_50_percent": edge_blend_metrics(),
            "exposure_pairs": exposure_pair_metrics(),
        },
        "algorithm_comparison": {
            method: method_metrics(scene, method) for method in METHODS
        },
        "edge_cases": edge_case_metrics(),
        "temporal": temporal_metrics(),
        "recommendation": {
            "method": "bounded_wb",
            "mode_default_for_first_compatibility_release": "off",
            "qualified_target_default_after_rollout_gates": "auto",
            "exposure_clamp_ev": EXPOSURE_CLAMP_EV,
            "wb_gain_range": [WB_GAIN_MIN, WB_GAIN_MAX],
            "default_strength": DEFAULT_STRENGTH,
            "default_white_balance_strength": DEFAULT_WHITE_BALANCE_STRENGTH,
            "skin_hue_drift_max_degrees": 5.0,
            "clothing_hue_drift_max_degrees": 8.0,
            "skin_normalized_chroma_drift_max_percent": 12.0,
            "clothing_normalized_chroma_drift_max_percent": 15.0,
            "low_confidence_instantaneous_semantics": "identity_no_update",
            "saturated_target_semantics": "exposure_only",
            "all_one_mask_semantics": "exposure_only",
            "all_zero_or_tiny_mask_semantics": "identity_no_update",
        },
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["deterministic_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def full_evidence(*, timing_iterations: int = 20) -> dict[str, Any]:
    result = deterministic_evidence()
    result["runtime_observation"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": Image.__version__,
        "platform": platform.platform(),
        "pipeline_stats_snapshot": runtime_stats_evidence(),
        "offline_microbenchmarks": timing_evidence(timing_iterations),
    }
    return result


def write_contact_sheet(path: Path) -> None:
    """Write a human-review aid; all decisions remain backed by numeric tests."""
    scene = synthetic_scene()
    candidates = [("source", scene.foreground)]
    for method in METHODS:
        estimate = estimate_transform(
            scene.foreground, scene.backdrop, scene.mask, method=method
        )
        candidates.append((method, apply_estimate(scene.foreground, estimate)))
    candidates.append(("target backdrop", scene.backdrop))

    tile_width, tile_height = 320, 210
    sheet = np.full((tile_height * 2, tile_width * 3, 3), 255, dtype=np.uint8)
    for index, (label, pixels) in enumerate(candidates):
        left = (index % 3) * tile_width
        top = (index // 3) * tile_height
        tile = _linear_u8(pixels)
        sheet[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
        _draw_label(
            sheet,
            label.upper().replace("_", " "),
            x=left + 8,
            y=top + 184,
            color=(0, 0, 0),
            scale=2,
        )
    _draw_label(
        sheet,
        "VISUAL AID ONLY; SEE JSON METRICS",
        x=tile_width * 2 + 8,
        y=tile_height + 196,
        color=(0, 0, 0),
        scale=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, mode="RGB").save(path, format="PNG")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        help="write evidence JSON to this path instead of stdout",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        help="optionally write a generated PNG comparison sheet",
    )
    parser.add_argument(
        "--timing-iterations",
        type=int,
        default=20,
        help="offline timing iterations (default: 20)",
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="omit host-dependent versions and timings",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.timing_iterations < 1:
        raise SystemExit("--timing-iterations must be positive")
    evidence = (
        deterministic_evidence()
        if args.deterministic_only
        else full_evidence(timing_iterations=args.timing_iterations)
    )
    serialized = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.json is None:
        sys.stdout.write(serialized)
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(serialized, encoding="utf-8")
    if args.contact_sheet is not None:
        write_contact_sheet(args.contact_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
