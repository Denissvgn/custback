"""Deterministic production tests for Phase-2 color normalization."""

from __future__ import annotations

import builtins
import io
import math
import sys
import tracemalloc
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

import custback.color as color
from custback.color import (
    ANALYSIS_LONG_EDGE,
    LOW_CONFIDENCE_FREEZE_S,
    MIN_CONFIDENCE,
    STALE_CLEAR_S,
    ColorBehavior,
    ColorError,
    ColorEstimate,
    ColorHarmonizer,
    ColorReason,
    ColorSceneSignature,
    ColorTransform,
    HarmonizerPhase,
    apply_color_transform,
    bgr_u8_to_linear_rgb,
    bounded_color_transform,
    decode_image_to_srgb_bgr,
    estimate_color_transform,
    estimate_color_transform_linear,
    linear_log_luminance,
    linear_luminance,
    linear_rgb_to_bgr_u8,
    masked_median,
    masked_quantile,
    srgb_eotf,
    srgb_oetf,
)
from custback.geometry import Rect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visual_consistency_evidence as evidence


def _ellipse_mask(height: int = 180, width: int = 320) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.float32)
    radius = np.sqrt(
        ((x - width * 0.5) / (width * 0.23)) ** 2
        + ((y - height * 0.52) / (height * 0.42)) ** 2
    )
    return np.ascontiguousarray(np.clip((1.025 - radius) / 0.05, 0.0, 1.0))


def _neutral_pair(
    *,
    source: float = 0.20,
    target: float = 0.40,
    height: int = 180,
    width: int = 320,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    foreground_linear = np.full((height, width, 3), source, np.float32)
    backdrop_linear = np.full((height, width, 3), target, np.float32)
    return (
        linear_rgb_to_bgr_u8(foreground_linear),
        linear_rgb_to_bgr_u8(backdrop_linear),
        _ellipse_mask(height, width),
    )


def _signature(
    *,
    source_luma: float = -2.0,
    target_luma: float = -1.5,
    source_chroma: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_chroma: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ColorSceneSignature:
    return ColorSceneSignature(
        source_log_luminance=source_luma,
        target_log_luminance=target_luma,
        source_chroma_log2=source_chroma,
        target_chroma_log2=target_chroma,
    )


def _reliable_estimate(
    exposure_ev: float = 0.40,
    wb_gains: tuple[float, float, float] = (1.05, 1.0, 0.95),
    *,
    signature: ColorSceneSignature | None = None,
) -> ColorEstimate:
    return ColorEstimate(
        transform=ColorTransform(exposure_ev, wb_gains),
        behavior=ColorBehavior.EXPOSURE_WHITE_BALANCE,
        reason=ColorReason.OK,
        confidence=0.9,
        exposure_confidence=0.95,
        white_balance_confidence=0.9,
        usable_source=512,
        usable_target=512,
        neutral_source=256,
        neutral_target=256,
        target_is_local=True,
        reliable=True,
        signature=signature or _signature(),
    )


def _run_constant_sequence(fps: int, duration_s: float = 3.0) -> ColorTransform:
    harmonizer = ColorHarmonizer()
    estimate = _reliable_estimate()
    for index in range(round(duration_s * fps) + 1):
        harmonizer.update(estimate, index / fps, source_generation=1)
    return harmonizer.transform


def test_srgb_reference_vectors_monotonic_endpoints_and_round_trip():
    encoded = np.array([0.0, 0.04045, 0.5, 1.0], np.float32)
    expected = np.array([0.0, 0.0031308, 0.21404114, 1.0], np.float32)
    decoded = srgb_eotf(encoded)
    assert decoded == pytest.approx(expected, abs=2e-7)
    assert np.all(np.diff(decoded) >= 0.0)
    assert srgb_oetf(decoded) == pytest.approx(encoded, abs=2e-7)


def test_internal_linear_bgr_codec_matches_public_rgb_for_every_u8_code():
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    bgr = np.stack(
        (
            codes,
            np.roll(codes, 37, axis=1),
            np.roll(codes, 83, axis=0),
        ),
        axis=2,
    )

    linear_bgr = color._bgr_u8_to_linear_bgr_prevalidated(bgr)
    linear_rgb = bgr_u8_to_linear_rgb(bgr)

    np.testing.assert_array_equal(linear_bgr[..., ::-1], linear_rgb)
    encoded = color._linear_bgr_to_bgr_u8_prevalidated(linear_bgr)
    assert np.max(np.abs(encoded.astype(np.int16) - bgr.astype(np.int16))) <= 1


def test_internal_linear_bgr_oetf_matches_public_rgb_on_random_values():
    rng = np.random.default_rng(0xB6720)
    linear_bgr = rng.random((127, 193, 3), dtype=np.float32) * np.float32(4.0)
    original = linear_bgr.copy()
    expected = linear_rgb_to_bgr_u8(np.ascontiguousarray(linear_bgr[..., ::-1]))
    actual = color._linear_bgr_to_bgr_u8_prevalidated(linear_bgr)

    assert np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))) <= 1
    np.testing.assert_array_equal(linear_bgr, original)


def test_consuming_linear_bgr_oetf_reuses_owned_buffer_and_matches_reference():
    rng = np.random.default_rng(0xC05E)
    owned_linear_bgr = rng.random((127, 193, 3), dtype=np.float32)
    original = owned_linear_bgr.copy()
    expected = color._linear_bgr_to_bgr_u8_prevalidated(original)

    actual = color._consume_linear_bgr_to_bgr_u8_prevalidated(owned_linear_bgr)

    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(owned_linear_bgr, original)


def test_all_u8_codes_round_trip_exactly_and_channels_are_not_swapped():
    values = np.arange(256, dtype=np.uint8)
    frame = np.zeros((1, 256, 3), np.uint8)
    frame[..., 0] = values
    frame[..., 1] = values[::-1]
    frame[..., 2] = 73
    linear = bgr_u8_to_linear_rgb(frame)
    assert linear.dtype == np.float32
    assert linear.flags.c_contiguous
    assert linear[0, 0, 0] == pytest.approx(float(srgb_eotf(73 / 255.0)))
    assert np.array_equal(linear_rgb_to_bgr_u8(linear), frame)


@pytest.mark.parametrize(
    "callback",
    [
        lambda: srgb_eotf(np.array([np.nan], np.float32)),
        lambda: srgb_eotf(np.array([-0.01], np.float32)),
        lambda: srgb_oetf(np.array([1.01], np.float32)),
        lambda: bgr_u8_to_linear_rgb(np.zeros((2, 2, 4), np.uint8)),
        lambda: linear_rgb_to_bgr_u8(np.full((2, 2, 3), np.inf, dtype=np.float32)),
    ],
)
def test_transfer_and_boundary_primitives_reject_adversarial_values(callback):
    with pytest.raises(ColorError):
        callback()


def test_linear_encode_clips_without_uint8_wrap_and_rounds():
    frame = np.array(
        [[[-1.0, 0.0, 4.0], [0.5, 0.5, 0.5]]],
        dtype=np.float32,
    )
    encoded = linear_rgb_to_bgr_u8(frame)
    assert encoded[0, 0].tolist() == [255, 0, 0]
    assert encoded[0, 1].tolist() == [188, 188, 188]


def test_luminance_and_robust_statistics_are_explicit_and_finite():
    rgb = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], np.float32)
    luminance = linear_luminance(rgb)
    assert luminance == pytest.approx(np.array([[0.2126729, 0.7151522]]))
    doubled = np.stack((rgb, rgb * 2.0), axis=0)
    log_luminance = linear_log_luminance(doubled)
    assert log_luminance[1] - log_luminance[0] == pytest.approx(1.0)
    assert np.isfinite(linear_log_luminance(np.zeros_like(rgb))).all()
    with pytest.raises(ColorError, match="floor"):
        linear_log_luminance(rgb, floor=0.0)
    values = np.array([1.0, 2.0, 100.0], np.float32)
    selected = np.array([True, True, False])
    assert masked_median(values, selected) == 1.5
    assert masked_quantile(values, 0.75, selected) == 1.75
    with pytest.raises(ColorError, match="selects no values"):
        masked_median(values, np.zeros(3, dtype=bool))


def test_bounded_transform_clamps_and_interpolates_strengths_independently():
    transform = bounded_color_transform(
        2.0,
        (1.4, 1.0, 0.7),
        strength=0.5,
        exposure_limit_ev=0.85,
        white_balance_strength=0.5,
    )
    assert transform.exposure_ev == pytest.approx(0.425)
    assert min(transform.wb_gains) >= math.sqrt(0.86) - 1e-12
    assert max(transform.wb_gains) <= math.sqrt(1.16) + 1e-12

    exposure_disabled = bounded_color_transform(
        0.8,
        (1.1, 1.0, 0.9),
        strength=0.0,
        white_balance_strength=0.5,
    )
    wb_disabled = bounded_color_transform(
        0.8,
        (1.1, 1.0, 0.9),
        strength=0.5,
        white_balance_strength=0.0,
    )
    assert exposure_disabled.exposure_ev == 0.0
    assert exposure_disabled.wb_gains != (1.0, 1.0, 1.0)
    assert wb_disabled.exposure_ev == pytest.approx(0.4)
    assert wb_disabled.wb_gains == pytest.approx((1.0, 1.0, 1.0))


def test_apply_transform_uses_linear_rgb_and_bounded_headroom():
    image = np.full((2, 3, 3), 0.25, np.float32)
    transformed = apply_color_transform(
        image,
        ColorTransform(1.0, (1.0, 1.0, 1.0)),
    )
    assert transformed.dtype == np.float32
    assert transformed.flags.c_contiguous
    assert transformed == pytest.approx(0.5)
    assert (
        np.max(apply_color_transform(np.full_like(image, 4.0), ColorTransform(1.0)))
        <= 4
    )


def _save_rgb(
    path: Path,
    *,
    size: tuple[int, int] = (7, 5),
    icc_profile: bytes | None = None,
) -> np.ndarray:
    pixels = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(
        size[1], size[0], 3
    )
    image = Image.fromarray(pixels, "RGB")
    if icc_profile is None:
        image.save(path, format="PNG")
    else:
        image.save(path, format="PNG", icc_profile=icc_profile)
    return pixels


def test_secure_decode_uses_one_descriptor_and_normalizes_tagged_srgb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = tmp_path / "tagged.png"
    pixels = _save_rgb(path, icc_profile=profile)
    real_open = builtins.open
    opened: list[Path] = []

    def tracked_open(name, *args, **kwargs):
        opened.append(Path(name))
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracked_open)
    decoded = decode_image_to_srgb_bgr(path, "PNG", 1000)
    assert opened == [path]
    assert decoded.dtype == np.uint8 and decoded.flags.c_contiguous
    # A tagged sRGB-to-sRGB transform is allowed one code of CMS rounding.
    assert np.max(np.abs(decoded[..., ::-1].astype(int) - pixels.astype(int))) <= 1


@pytest.mark.parametrize("profile_name", ["Display-P3", "Adobe-RGB", "CMYK"])
def test_wide_gamut_and_cmyk_profiles_decode_deterministically_to_srgb_bgr(
    tmp_path: Path,
    profile_name: str,
):
    profiles = evidence.generated_profiles()
    pixels = evidence.geometry_fixture(33, 47)
    profile = profiles[profile_name]
    if profile_name == "CMYK":
        path = tmp_path / "profiled-cmyk.jpg"
        Image.fromarray(pixels, "RGB").convert("CMYK").save(
            path,
            format="JPEG",
            quality=100,
            subsampling=0,
            icc_profile=profile,
        )
        expected_format = "JPEG"
    else:
        path = tmp_path / f"profiled-{profile_name}.png"
        Image.fromarray(pixels, "RGB").save(
            path,
            format="PNG",
            icc_profile=profile,
        )
        expected_format = "PNG"

    first = decode_image_to_srgb_bgr(path, expected_format, 10_000)
    second = decode_image_to_srgb_bgr(path, expected_format, 10_000)
    with Image.open(path) as source:
        expected_rgb = np.asarray(
            ImageCms.profileToProfile(
                source,
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            ),
            dtype=np.uint8,
        )
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.ascontiguousarray(expected_rgb[..., ::-1]))


def test_untagged_decode_assumes_srgb_and_enforces_format_and_pixel_cap(
    tmp_path: Path,
):
    path = tmp_path / "plain.png"
    pixels = _save_rgb(path)
    decoded = decode_image_to_srgb_bgr(path, "PNG", pixels.shape[0] * pixels.shape[1])
    assert np.array_equal(decoded[..., ::-1], pixels)
    with pytest.raises(ColorError, match="does not match expected JPEG"):
        decode_image_to_srgb_bgr(path, "JPEG", 1000)
    with pytest.raises(ColorError, match="image exceeds 34 pixels"):
        decode_image_to_srgb_bgr(path, "PNG", 34)


@pytest.mark.parametrize(
    "bomb_type",
    (Image.DecompressionBombError, Image.DecompressionBombWarning),
)
def test_decoder_normalizes_pillow_bomb_signals_to_pixel_cap_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bomb_type: type[BaseException],
):
    path = tmp_path / "plain.png"
    _save_rgb(path)

    def reject_bomb(*args, **kwargs):
        raise bomb_type("Pillow dimension guard")

    monkeypatch.setattr(color.Image, "open", reject_bomb)
    with pytest.raises(ColorError, match=r"^image exceeds 1000 pixels$"):
        decode_image_to_srgb_bgr(path, "PNG", 1000)


@pytest.mark.parametrize(
    "profile_name",
    ("sRGB", "Display-P3", "Adobe-RGB", "CMYK"),
)
def test_profile_tagged_phase0_fixtures_are_normalized_to_srgb_bgr(
    tmp_path: Path,
    profile_name: str,
):
    payloads, manifest = evidence.tagged_image_fixtures()
    fixture = manifest[profile_name]
    expected_format = str(fixture["container"])
    path = tmp_path / f"profile.{'tiff' if expected_format == 'TIFF' else 'png'}"
    path.write_bytes(payloads[profile_name])

    with Image.open(io.BytesIO(payloads[profile_name])) as tagged:
        source_profile = ImageCms.ImageCmsProfile(
            io.BytesIO(tagged.info["icc_profile"])
        )
        destination_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(
            tagged,
            source_profile,
            destination_profile,
            outputMode="RGB",
        )
        assert converted is not None
        expected_rgb = np.asarray(converted, dtype=np.uint8)

    decoded = decode_image_to_srgb_bgr(
        path,
        expected_format,
        expected_rgb.shape[0] * expected_rgb.shape[1],
    )
    assert decoded.dtype == np.uint8 and decoded.flags.c_contiguous
    assert np.array_equal(decoded[..., ::-1], expected_rgb)


def test_malformed_embedded_profile_is_rejected_not_assumed_srgb(tmp_path: Path):
    path = tmp_path / "bad-profile.png"
    _save_rgb(path, icc_profile=b"not-an-icc-profile")
    with pytest.raises(ColorError, match="invalid embedded ICC profile"):
        decode_image_to_srgb_bgr(path, "PNG", 1000)


def test_present_empty_profile_is_rejected_not_treated_as_untagged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "empty-profile.png"
    _save_rgb(path)
    real_open = color.Image.open
    calls = 0

    def open_with_empty_profile(*args, **kwargs):
        nonlocal calls
        image = real_open(*args, **kwargs)
        calls += 1
        if calls == 2:
            image.info["icc_profile"] = b""
        return image

    monkeypatch.setattr(color.Image, "open", open_with_empty_profile)
    with pytest.raises(ColorError, match="invalid embedded ICC profile"):
        decode_image_to_srgb_bgr(path, "PNG", 1000)


def test_exif_orientation_is_applied_once_before_bgr_boundary(tmp_path: Path):
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (8, 5), (0, 0, 0))
    image.paste((255, 0, 0), (0, 0, 3, 2))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, quality=100, subsampling=0, exif=exif)
    decoded = decode_image_to_srgb_bgr(path, "JPEG", 1000)
    assert decoded.shape == (8, 5, 3)
    # Orientation 6 rotates the source top-left red patch to displayed top-right.
    assert int(decoded[:3, -2:, 2].mean()) > 180
    assert int(decoded[-3:, :2, 2].mean()) < 80


def test_estimator_matches_phase0_selected_math_and_preservation_gates():
    scene = evidence.synthetic_scene()
    foreground = linear_rgb_to_bgr_u8(scene.foreground)
    backdrop = linear_rgb_to_bgr_u8(scene.backdrop)
    foreground_linear = bgr_u8_to_linear_rgb(foreground)
    backdrop_linear = bgr_u8_to_linear_rgb(backdrop)
    estimate = estimate_color_transform_linear(
        foreground_linear,
        backdrop_linear,
        scene.mask,
        mode="image",
    )
    assert estimate.reason is ColorReason.OK
    assert estimate.behavior is ColorBehavior.EXPOSURE_WHITE_BALANCE
    assert estimate.exposure_ev == pytest.approx(0.425, abs=0.003)
    assert estimate.confidence >= MIN_CONFIDENCE
    corrected = apply_color_transform(foreground_linear, estimate.transform)
    skin = evidence.patch_preservation(
        foreground_linear,
        np.clip(corrected, 0.0, 1.0),
        scene.skin_mask,
    )
    clothing = evidence.patch_preservation(
        foreground_linear,
        np.clip(corrected, 0.0, 1.0),
        scene.clothing_mask,
    )
    assert skin["hue_drift_degrees"] <= 5.0
    assert skin["normalized_chroma_drift_percent"] <= 12.0
    assert clothing["hue_drift_degrees"] <= 8.0
    assert clothing["normalized_chroma_drift_percent"] <= 15.0


@pytest.mark.parametrize("ev", [1.0, -1.0])
def test_estimator_recovers_symmetric_exposure_pairs_with_ratified_clamp(ev: float):
    source = 0.20
    target = source * (2.0**ev)
    foreground, backdrop, mask = _neutral_pair(source=source, target=target)
    estimate = estimate_color_transform(
        foreground,
        backdrop,
        mask,
        mode="image",
    )
    assert estimate.reliable
    assert estimate.behavior is ColorBehavior.EXPOSURE_WHITE_BALANCE
    assert estimate.exposure_ev == pytest.approx(math.copysign(0.425, ev), abs=0.005)
    assert estimate.transform.wb_gains == pytest.approx((1.0, 1.0, 1.0), abs=1e-5)


def test_estimator_external_and_predecoded_entrypoints_agree():
    foreground, backdrop, mask = _neutral_pair(target=0.34)
    external = estimate_color_transform(
        foreground,
        backdrop,
        mask,
        mode="video",
    )
    predecoded = estimate_color_transform_linear(
        bgr_u8_to_linear_rgb(foreground),
        bgr_u8_to_linear_rgb(backdrop),
        mask,
        mode="video",
    )
    assert external.transform == predecoded.transform
    assert external.reason is predecoded.reason
    assert external.confidence == predecoded.confidence


def test_internal_linear_bgr_estimator_matches_public_rgb_after_downsampling(
    monkeypatch: pytest.MonkeyPatch,
):
    foreground, backdrop, mask = _neutral_pair(
        source=0.18,
        target=0.37,
        height=720,
        width=1280,
    )
    foreground_linear_bgr = color._bgr_u8_to_linear_bgr_prevalidated(foreground)
    backdrop_linear_bgr = color._bgr_u8_to_linear_bgr_prevalidated(backdrop)
    expected = estimate_color_transform_linear(
        np.ascontiguousarray(foreground_linear_bgr[..., ::-1]),
        np.ascontiguousarray(backdrop_linear_bgr[..., ::-1]),
        mask,
        mode="image",
    )
    resized_shapes: list[tuple[int, int, int]] = []
    real_resize = color._resize_linear_and_mask

    def tracked_resize(
        foreground_value,
        backdrop_value,
        mask_value,
        *,
        executor=None,
        backdrop_analysis=None,
    ):
        # The full rasters reach the resize in native BGR order. Channel reversal
        # happens only after this bounded analysis resize.
        np.testing.assert_array_equal(
            foreground_value[0, 0],
            foreground_linear_bgr[0, 0],
        )
        resized_shapes.append(foreground_value.shape)
        return real_resize(
            foreground_value,
            backdrop_value,
            mask_value,
            executor=executor,
            backdrop_analysis=backdrop_analysis,
        )

    monkeypatch.setattr(color, "_resize_linear_and_mask", tracked_resize)
    actual = color._estimate_color_transform_linear_bgr_prevalidated(
        foreground_linear_bgr,
        backdrop_linear_bgr,
        mask,
        mode="image",
    )

    assert resized_shapes == [(720, 1280, 3)]
    assert actual == expected


def test_cached_linear_bgr_backdrop_analysis_is_bounded_and_equivalent():
    foreground, backdrop, mask = _neutral_pair(
        source=0.18,
        target=0.37,
        height=720,
        width=1280,
    )
    foreground_linear_bgr = color._bgr_u8_to_linear_bgr_prevalidated(foreground)
    backdrop_linear_bgr = color._bgr_u8_to_linear_bgr_prevalidated(backdrop)
    cached_analysis = color._linear_bgr_analysis_raster_prevalidated(
        backdrop_linear_bgr
    )

    uncached = color._estimate_color_transform_linear_bgr_prevalidated(
        foreground_linear_bgr,
        backdrop_linear_bgr,
        mask,
        mode="image",
    )
    cached = color._estimate_color_transform_linear_bgr_prevalidated(
        foreground_linear_bgr,
        backdrop_linear_bgr,
        mask,
        mode="image",
        backdrop_analysis_linear_bgr=cached_analysis,
    )

    assert cached_analysis.dtype == np.float32
    assert cached_analysis.flags.c_contiguous
    assert max(cached_analysis.shape[:2]) == ANALYSIS_LONG_EDGE
    assert cached == uncached


def test_estimator_analysis_is_capped_at_192_and_does_not_mutate_inputs(
    monkeypatch: pytest.MonkeyPatch,
):
    foreground, backdrop, mask = _neutral_pair(height=720, width=1280)
    originals = foreground.copy(), backdrop.copy(), mask.copy()
    sizes: list[tuple[int, int]] = []
    real_resize = color.cv2.resize

    def tracked_resize(source, size, *args, **kwargs):
        sizes.append(size)
        return real_resize(source, size, *args, **kwargs)

    monkeypatch.setattr(color.cv2, "resize", tracked_resize)
    estimate = estimate_color_transform(
        foreground,
        backdrop,
        mask,
        mode="image",
    )
    assert estimate.reliable
    assert sizes and all(max(size) <= ANALYSIS_LONG_EDGE for size in sizes)
    assert np.array_equal(foreground, originals[0])
    assert np.array_equal(backdrop, originals[1])
    assert np.array_equal(mask, originals[2])


def test_valid_content_rectangles_exclude_contain_padding_from_confidence():
    height, width = 180, 320
    foreground_linear = np.full((height, width, 3), 0.001, np.float32)
    backdrop_linear = np.full((height, width, 3), 0.001, np.float32)
    valid = Rect(100, 0, 220, height)
    foreground_linear[:, valid.left : valid.right] = 0.20
    backdrop_linear[:, valid.left : valid.right] = 0.40
    mask = np.ones((height, width), np.float32)
    with_rects = estimate_color_transform_linear(
        foreground_linear,
        backdrop_linear,
        mask,
        mode="image",
        foreground_content_rect=valid,
        backdrop_content_rect=valid,
    )
    without_rects = estimate_color_transform_linear(
        foreground_linear,
        backdrop_linear,
        mask,
        mode="image",
    )
    assert with_rects.reliable
    assert with_rects.exposure_confidence >= MIN_CONFIDENCE
    assert with_rects.exposure_ev == pytest.approx(0.425, abs=1e-6)
    assert (
        not without_rects.reliable
        or without_rects.exposure_confidence < with_rects.exposure_confidence
    )


def test_narrow_content_coverage_uses_valid_content_not_full_canvas():
    height, width = 180, 320
    foreground_linear = np.full((height, width, 3), 0.001, np.float32)
    backdrop_linear = np.full((height, width, 3), 0.001, np.float32)
    narrow_content = Rect(153, 0, 167, height)
    foreground_linear[:, narrow_content.left : narrow_content.right] = 0.20
    backdrop_linear[:, narrow_content.left : narrow_content.right] = 0.40
    mask = np.ones((height, width), np.float32)

    estimate = estimate_color_transform_linear(
        foreground_linear,
        backdrop_linear,
        mask,
        mode="image",
        foreground_content_rect=narrow_content,
        backdrop_content_rect=narrow_content,
    )

    # Content occupies under 5% of the canvas, which made the old full-canvas
    # denominator fail confidence even though the valid content is fully masked.
    assert narrow_content.width / width < 0.05
    assert estimate.reliable
    assert estimate.exposure_confidence >= MIN_CONFIDENCE
    assert estimate.exposure_ev == pytest.approx(0.425, abs=1e-6)


@pytest.mark.parametrize(
    ("case", "reason", "behavior"),
    [
        ("zero", ColorReason.INSUFFICIENT_MASK, ColorBehavior.IDENTITY),
        ("tiny", ColorReason.INSUFFICIENT_MASK, ColorBehavior.IDENTITY),
        ("one", ColorReason.INSUFFICIENT_NEUTRAL, ColorBehavior.EXPOSURE_ONLY),
        ("clipped", ColorReason.CLIPPED, ColorBehavior.IDENTITY),
        ("saturated", ColorReason.SOLID_SATURATED, ColorBehavior.EXPOSURE_ONLY),
    ],
)
def test_estimator_edge_case_reason_precedence(case, reason, behavior):
    foreground, backdrop, mask = _neutral_pair()
    if case == "zero":
        mask[:] = 0.0
    elif case == "tiny":
        mask[:] = 0.0
        mask[88:92, 158:162] = 1.0
    elif case == "one":
        mask[:] = 1.0
    elif case == "clipped":
        foreground[:] = 255
    elif case == "saturated":
        saturated = np.empty_like(backdrop)
        saturated[:] = (188, 26, 233)
        backdrop = saturated
    estimate = estimate_color_transform(
        foreground,
        backdrop,
        mask,
        mode="image",
    )
    assert estimate.reason is reason
    assert estimate.behavior is behavior


def test_invalid_precedes_mode_excluded_and_excluded_modes_do_no_analysis(
    monkeypatch: pytest.MonkeyPatch,
):
    foreground, backdrop, mask = _neutral_pair()
    bad_mask = mask.copy()
    bad_mask[0, 0] = np.nan
    assert (
        estimate_color_transform(foreground, backdrop, bad_mask, mode="blur").reason
        is ColorReason.INVALID
    )
    monkeypatch.setattr(
        color,
        "bgr_u8_to_linear_rgb",
        lambda _frame: pytest.fail("excluded mode must not decode/analyze"),
    )
    for mode in ("passthrough", "blur", "color", "remote"):
        result = estimate_color_transform(foreground, backdrop, mask, mode=mode)
        assert result.reason is ColorReason.MODE_EXCLUDED
        assert result.transform.is_identity


def test_outliers_and_twenty_percent_core_contamination_do_not_dominate():
    foreground, backdrop, mask = _neutral_pair(target=0.34)
    baseline = estimate_color_transform(
        foreground,
        backdrop,
        mask,
        mode="image",
    )
    contaminated = foreground.copy()
    core_indices = np.argwhere(mask >= 0.99)
    count = len(core_indices) // 5
    for row, column in core_indices[:count]:
        contaminated[row, column] = (255, 0, 255)
    result = estimate_color_transform(
        contaminated,
        backdrop,
        mask,
        mode="image",
    )
    assert result.reliable
    assert abs(result.exposure_ev - baseline.exposure_ev) <= 0.05
    assert (
        max(
            abs(math.log2(a / b))
            for a, b in zip(
                result.transform.wb_gains,
                baseline.transform.wb_gains,
                strict=True,
            )
        )
        <= 0.03
    )


def test_resolution_changes_do_not_retune_percentage_thresholds():
    foreground, backdrop, mask = _neutral_pair()
    baseline = estimate_color_transform(
        foreground,
        backdrop,
        mask,
        mode="image",
    )
    small = (
        cv2.resize(foreground, (160, 90), interpolation=cv2.INTER_AREA),
        cv2.resize(backdrop, (160, 90), interpolation=cv2.INTER_AREA),
        cv2.resize(mask, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32),
    )
    resized = estimate_color_transform(*small, mode="image")
    assert resized.reliable
    assert abs(resized.exposure_ev - baseline.exposure_ev) <= 0.02
    assert (
        max(
            abs(math.log2(a / b))
            for a, b in zip(
                resized.transform.wb_gains,
                baseline.transform.wb_gains,
                strict=True,
            )
        )
        <= 0.02
    )


def test_histogram_looking_limited_range_is_not_expanded_or_reinterpreted():
    foreground = np.full((180, 320, 3), 64, np.uint8)
    backdrop = np.full_like(foreground, 128)
    estimate = estimate_color_transform(
        foreground,
        backdrop,
        _ellipse_mask(),
        mode="image",
    )
    source_y = float(linear_luminance(bgr_u8_to_linear_rgb(foreground))[0, 0])
    target_y = float(linear_luminance(bgr_u8_to_linear_rgb(backdrop))[0, 0])
    expected = min(0.85, math.log2(target_y / source_y)) * 0.5
    assert estimate.exposure_ev == pytest.approx(expected, abs=1e-5)


def test_temporal_sequences_are_cadence_equivalent():
    transforms = [_run_constant_sequence(fps) for fps in (15, 30, 60)]
    for first, second in zip(transforms, transforms[1:], strict=False):
        assert abs(first.exposure_ev - second.exposure_ev) <= 0.01
        assert (
            max(
                abs(math.log2(a / b))
                for a, b in zip(first.wb_gains, second.wb_gains, strict=True)
            )
            <= 0.01
        )


def test_temporal_static_noise_stays_below_jitter_gate():
    rng = np.random.default_rng(2401)
    harmonizer = ColorHarmonizer()
    exposures = []
    gains = []
    for index in range(240):
        estimate = _reliable_estimate(
            0.30 + float(rng.normal(0.0, 0.004)),
            (
                1.03 * (2.0 ** float(rng.normal(0.0, 0.001))),
                1.0,
                0.97 * (2.0 ** float(rng.normal(0.0, 0.001))),
            ),
        )
        transform = harmonizer.update(
            estimate,
            index / 60.0,
            source_generation=1,
        )
        exposures.append(transform.exposure_ev)
        gains.append(transform.wb_gains)
    ev_delta = np.abs(np.diff(exposures[120:]))
    gain_delta = [
        max(abs(math.log2(b / a)) for a, b in zip(previous, current, strict=True))
        for previous, current in zip(gains[120:-1], gains[121:], strict=True)
    ]
    assert float(np.percentile(ev_delta, 95)) <= 0.002
    assert float(np.percentile(gain_delta, 95)) <= 0.002


def test_scene_cut_holds_cut_frame_then_fast_acquires_inside_clamps():
    harmonizer = ColorHarmonizer()
    old = _reliable_estimate(0.35, signature=_signature())
    for index in range(91):
        harmonizer.update(old, index / 30.0, source_generation=1)
    before = harmonizer.transform
    cut_signature = _signature(target_luma=-0.3)
    cut = _reliable_estimate(-0.30, (0.95, 1.0, 1.05), signature=cut_signature)
    held = harmonizer.update(cut, 91 / 30.0, source_generation=1)
    assert held == before
    assert harmonizer.snapshot().phase is HarmonizerPhase.SCENE_CUT

    outputs = []
    for index in range(92, 92 + 46):
        outputs.append(harmonizer.update(cut, index / 30.0, source_generation=1))
    assert abs(outputs[-1].exposure_ev - cut.transform.exposure_ev) <= 0.05
    assert (
        max(
            abs(math.log2(a / b))
            for a, b in zip(
                outputs[-1].wb_gains,
                cut.transform.wb_gains,
                strict=True,
            )
        )
        <= 0.02
    )
    assert all(abs(item.exposure_ev) <= 1.0 for item in outputs)


def test_low_confidence_freezes_then_decays_monotonically_to_exact_identity():
    harmonizer = ColorHarmonizer()
    estimate = _reliable_estimate()
    for index in range(91):
        harmonizer.update(estimate, index / 30.0, source_generation=1)
    active = harmonizer.transform
    invalid = ColorEstimate.identity(ColorReason.INSUFFICIENT_MASK)
    start = 91 / 30.0
    assert harmonizer.update(invalid, start, source_generation=1) == active
    assert (
        harmonizer.update(
            invalid,
            start + LOW_CONFIDENCE_FREEZE_S,
            source_generation=1,
        )
        == active
    )
    values = []
    for offset in (1.0, 2.0, 3.0, 4.0):
        values.append(harmonizer.update(invalid, start + offset, source_generation=1))
    assert all(
        abs(second.exposure_ev) <= abs(first.exposure_ev) + 1e-7
        for first, second in zip(values, values[1:], strict=False)
    )
    identity = harmonizer.update(
        invalid,
        start + STALE_CLEAR_S,
        source_generation=1,
    )
    assert identity.is_identity
    assert not harmonizer.snapshot().reliable


def test_sparse_low_confidence_update_does_not_decay_through_freeze_window():
    baseline = ColorHarmonizer()
    estimate = _reliable_estimate()
    for index in range(91):
        baseline.update(estimate, index / 30.0, source_generation=1)
    invalid = ColorEstimate.identity(ColorReason.INSUFFICIENT_MASK)
    start = 91 / 30.0
    dense = baseline.clone()
    sparse = baseline.clone()
    dense.update(invalid, start, source_generation=1)
    sparse.update(invalid, start, source_generation=1)

    for step in range(1, 11):
        dense_result = dense.update(
            invalid,
            start + step / 10.0,
            source_generation=1,
        )
    sparse_result = sparse.update(
        invalid,
        start + 1.0,
        source_generation=1,
    )

    assert sparse_result.exposure_ev == pytest.approx(
        dense_result.exposure_ev,
        abs=1e-7,
    )
    assert sparse_result.wb_gains == pytest.approx(
        dense_result.wb_gains,
        abs=1e-7,
    )


def test_no_prior_reliable_estimate_and_error_are_identity():
    harmonizer = ColorHarmonizer()
    assert harmonizer.on_error(0.0, source_generation=1).is_identity
    assert harmonizer.snapshot().reason is ColorReason.INVALID
    assert harmonizer.snapshot().phase is HarmonizerPhase.IDENTITY


def test_generation_change_and_long_gap_discard_stale_transform():
    harmonizer = ColorHarmonizer()
    estimate = _reliable_estimate()
    for index in range(91):
        harmonizer.update(estimate, index / 30.0, source_generation=1)
    assert not harmonizer.transform.is_identity
    assert harmonizer.update(estimate, 4.0, source_generation=2).is_identity
    reset_snapshot = harmonizer.snapshot()
    assert reset_snapshot.source_generation == 2
    assert reset_snapshot.reliable
    assert reset_snapshot.signature == estimate.signature
    assert reset_snapshot.phase is HarmonizerPhase.WARMING
    assert not harmonizer.update(
        estimate,
        4.0 + 1.0 / 30.0,
        source_generation=2,
    ).is_identity

    for index in range(2, 40):
        harmonizer.update(estimate, 4.0 + index / 30.0, source_generation=2)
    assert not harmonizer.transform.is_identity
    assert harmonizer.update(estimate, 20.0, source_generation=2).is_identity


def test_same_timestamp_is_idempotent_and_backwards_time_is_rejected():
    harmonizer = ColorHarmonizer()
    first = _reliable_estimate(0.2)
    second = _reliable_estimate(-0.2)
    harmonizer.update(first, 1.0, source_generation=1)
    snapshot = harmonizer.snapshot()
    assert harmonizer.update(second, 1.0, source_generation=1) == snapshot.transform
    assert harmonizer.snapshot() == snapshot
    with pytest.raises(ColorError, match="moved backwards"):
        harmonizer.update(first, 0.9, source_generation=1)


def test_clone_is_independent_and_state_contains_no_arrays():
    harmonizer = ColorHarmonizer()
    estimate = _reliable_estimate()
    for index in range(61):
        harmonizer.update(estimate, index / 30.0, source_generation=1)
    clone = harmonizer.clone()
    original_snapshot = harmonizer.snapshot()
    clone.reset(3.0, source_generation=2)
    assert harmonizer.snapshot() == original_snapshot
    assert clone.snapshot() != original_snapshot
    for field in fields(original_snapshot):
        assert not isinstance(getattr(original_snapshot, field.name), np.ndarray)
    assert not any(
        isinstance(value, np.ndarray) for value in harmonizer.__dict__.values()
    )


def test_camera_mode_uses_slower_steady_adaptation():
    image = ColorHarmonizer(mode="image")
    camera = ColorHarmonizer(mode="camera")
    initial = _reliable_estimate(0.15)
    for index in range(91):
        timestamp = index / 30.0
        image.update(initial, timestamp, source_generation=1)
        camera.update(initial, timestamp, source_generation=1)
    changed = _reliable_estimate(0.35)
    image_before = image.transform.exposure_ev
    camera_before = camera.transform.exposure_ev
    for index in range(91, 106):
        timestamp = index / 30.0
        image.update(changed, timestamp, source_generation=1)
        camera.update(changed, timestamp, source_generation=1)
    assert image.transform.exposure_ev - image_before > (
        camera.transform.exposure_ev - camera_before
    )


def test_harmonizer_has_bounded_retained_memory_over_ten_thousand_updates():
    harmonizer = ColorHarmonizer()
    estimate = _reliable_estimate()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for index in range(10_000):
        harmonizer.update(estimate, index / 60.0, source_generation=1)
    after = tracemalloc.take_snapshot()
    growth = sum(
        stat.size_diff
        for stat in after.compare_to(before, "lineno")
        if stat.size_diff > 0
    )
    tracemalloc.stop()
    assert growth < 1024 * 1024
    assert not any(
        isinstance(value, np.ndarray) for value in harmonizer.__dict__.values()
    )


def test_analysis_workspace_is_bounded_not_retained():
    foreground, backdrop, mask = _neutral_pair(height=1080, width=1920)
    estimate = estimate_color_transform(
        foreground,
        backdrop,
        mask,
        mode="image",
    )
    assert estimate.reliable
    assert not any(
        isinstance(value, np.ndarray) for value in estimate.__dict__.values()
    )
