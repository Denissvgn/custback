"""Numeric qualification for the non-production VIS-0.2/VIS-0.3 harness."""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from custback.compositor import composite

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visual_consistency_evidence as evidence


def test_generated_geometry_inventory_is_asymmetric_and_deterministic():
    expected_sizes = {
        "square": (201, 201),
        "four_three": (240, 320),
        "sixteen_nine": (180, 320),
        "portrait": (320, 180),
        "ultrawide": (144, 384),
        "odd": (181, 319),
    }
    assert evidence.GEOMETRY_SIZES == expected_sizes

    for height, width in expected_sizes.values():
        first = evidence.geometry_fixture(height, width)
        second = evidence.geometry_fixture(height, width)
        assert first.shape == (height, width, 3)
        assert np.array_equal(first, second)
        # Corner labels use four distinct primary/secondary colors.
        corner_regions = (
            first[: height // 4, : width // 4],
            first[: height // 4, -width // 4 :],
            first[-height // 4 :, : width // 4],
            first[-height // 4 :, -width // 4 :],
        )
        label_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255))
        for region, color in zip(corner_regions, label_colors, strict=True):
            assert np.any(np.all(region == color, axis=2))
        assert np.any(np.all(first == (255, 255, 0), axis=2))


def test_exif_orientations_1_to_8_have_expected_shape_and_are_invertible():
    source = evidence.geometry_fixture(31, 47)
    expected_shapes = {
        1: (31, 47),
        2: (31, 47),
        3: (31, 47),
        4: (31, 47),
        5: (47, 31),
        6: (47, 31),
        7: (47, 31),
        8: (47, 31),
    }
    inverse = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 7, 8: 6}
    for orientation in range(1, 9):
        transformed = evidence.apply_exif_orientation(source, orientation)
        assert transformed.shape[:2] == expected_shapes[orientation]
        restored = evidence.apply_exif_orientation(transformed, inverse[orientation])
        assert np.array_equal(restored, source)

    with pytest.raises(ValueError, match="1..8"):
        evidence.apply_exif_orientation(source, 0)


def test_current_four_three_to_sixteen_nine_stretch_fails_distortion_gate():
    axis_ratio = evidence.stretch_axis_ratio(320, 240, 320, 180)
    assert axis_ratio == pytest.approx(4.0 / 3.0)

    def assert_distortion_free(value: float, tolerance: float = 0.01) -> None:
        assert abs(value - 1.0) <= tolerance

    with pytest.raises(AssertionError):
        assert_distortion_free(axis_ratio)

    cover = evidence.cover_crop_coordinates(320, 240, 320, 180)
    assert cover == pytest.approx(
        {
            "left": 0.0,
            "top": 30.0,
            "right": 320.0,
            "bottom": 210.0,
            "scale": 1.0,
        }
    )


def test_current_encoded_half_blend_is_100_and_linear_reference_is_146():
    foreground = np.full((3, 5, 3), 200, dtype=np.uint8)
    backdrop = np.zeros_like(foreground)
    mask = np.full((3, 5), 0.5, dtype=np.float32)
    current = composite(foreground, backdrop, mask)
    assert np.unique(current).tolist() == [100]

    metrics = evidence.edge_blend_metrics()
    assert metrics["legacy_encoded_u8"] == 100
    assert metrics["linear_light_reference_u8"] == 146
    assert metrics["edge_linear_luminance_error_percent"] == pytest.approx(
        -55.872, abs=0.001
    )


def test_profile_tagged_fixtures_are_generated_valid_and_byte_stable():
    first_payloads, first_manifest = evidence.tagged_image_fixtures()
    second_payloads, second_manifest = evidence.tagged_image_fixtures()
    profiles = evidence.generated_profiles()
    assert first_manifest == second_manifest
    assert first_payloads == second_payloads
    assert set(first_payloads) == {"sRGB", "Display-P3", "Adobe-RGB", "CMYK"}

    srgb = ImageCms.ImageCmsProfile(io.BytesIO(profiles["sRGB"]))
    for name, payload in first_payloads.items():
        image = Image.open(io.BytesIO(payload))
        embedded = image.info["icc_profile"]
        source = ImageCms.ImageCmsProfile(io.BytesIO(embedded))
        converted = ImageCms.profileToProfile(image, source, srgb, outputMode="RGB")
        assert converted is not None
        assert converted.mode == "RGB"
        assert converted.size == (47, 33)
        assert (
            hashlib.sha256(payload).hexdigest()
            == first_manifest[name]["payload_sha256"]
        )
        assert first_manifest[name]["provenance"].startswith("generated in repository")


def test_estimator_analyzes_only_a_192_pixel_long_edge_and_uses_eroded_core():
    scene = evidence.synthetic_scene()
    sampling = evidence.sampling_regions(scene.foreground, scene.backdrop, scene.mask)
    assert max(sampling.foreground.shape[:2]) == evidence.ANALYSIS_LONG_EDGE
    assert sampling.target_is_local is True
    assert sampling.core.sum() >= evidence.MIN_SAMPLES
    assert np.all(
        evidence._analysis_inputs(scene.foreground, scene.backdrop, scene.mask)[2][
            sampling.core
        ]
        >= 0.90
    )
    assert not np.any(sampling.core & sampling.target)


def test_bounded_wb_reduces_errors_without_breaking_preservation_bounds():
    scene = evidence.synthetic_scene()
    bounded = evidence.method_metrics(scene, "bounded_wb")
    exposure = evidence.method_metrics(scene, "exposure_only")
    aggressive = evidence.method_metrics(scene, "aggressive_moment")

    assert bounded["estimate"]["behavior"] == "bounded_wb"
    assert bounded["luminance_gap_ev"]["reduction_percent"] >= 45.0
    assert bounded["neutral_axis_error_delta_e_ab"]["reduction_percent"] >= 20.0
    assert (
        bounded["neutral_axis_error_delta_e_ab"]["after"]
        < exposure["neutral_axis_error_delta_e_ab"]["after"]
    )

    assert bounded["skin_preservation"]["hue_drift_degrees"] <= 5.0
    assert bounded["clothing_preservation"]["hue_drift_degrees"] <= 8.0
    assert bounded["skin_preservation"]["normalized_chroma_drift_percent"] <= 12.0
    assert bounded["clothing_preservation"]["normalized_chroma_drift_percent"] <= 15.0

    assert aggressive["luminance_gap_ev"]["reduction_percent"] > 90.0
    assert aggressive["skin_preservation"]["hue_drift_degrees"] > 30.0
    assert aggressive["clothing_preservation"]["hue_drift_degrees"] > 60.0
    assert aggressive["clothing_preservation"]["normalized_chroma_drift_percent"] > 90.0


def test_ratified_clamps_and_strength_are_enforced_on_extreme_pairs():
    bright_target = evidence.synthetic_scene(target_ev=2.0)
    estimate = evidence.estimate_transform(
        bright_target.foreground,
        bright_target.backdrop,
        bright_target.mask,
        method="bounded_wb",
    )
    assert estimate.exposure_ev == pytest.approx(
        evidence.EXPOSURE_CLAMP_EV * evidence.DEFAULT_STRENGTH
    )
    applied_full_strength_gains = np.power(
        np.asarray(estimate.wb_gains),
        1.0 / evidence.DEFAULT_WHITE_BALANCE_STRENGTH,
    )
    assert np.all(applied_full_strength_gains >= evidence.WB_GAIN_MIN - 1e-6)
    assert np.all(applied_full_strength_gains <= evidence.WB_GAIN_MAX + 1e-6)
    assert evidence.DEFAULT_STRENGTH == 0.50
    assert evidence.DEFAULT_WHITE_BALANCE_STRENGTH == 0.50


def test_actual_plus_and_minus_one_ev_fixtures_are_clamped_symmetrically():
    metrics = evidence.exposure_pair_metrics()
    for name, sign in (("plus_one_ev", 1.0), ("minus_one_ev", -1.0)):
        pair = metrics[name]
        assert pair["requested_ev"] == sign
        assert pair["gap_before_ev"] == pytest.approx(1.0, abs=1e-6)
        assert pair["applied_ev"] == pytest.approx(
            sign * evidence.EXPOSURE_CLAMP_EV * evidence.DEFAULT_STRENGTH
        )
        assert pair["gap_after_ev"] == pytest.approx(
            1.0 - evidence.EXPOSURE_CLAMP_EV * evidence.DEFAULT_STRENGTH
        )
        assert pair["behavior"] == "exposure_only"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("saturated_backdrop", "exposure_only"),
        ("all_zero_mask", "identity"),
        ("all_one_mask", "exposure_only"),
        ("tiny_mask", "identity"),
        ("clipped_foreground", "identity"),
    ],
)
def test_low_confidence_and_solid_color_behavior(case, expected):
    assert evidence.edge_case_metrics()[case]["behavior"] == expected


def test_temporal_metrics_cover_noise_drift_cut_and_reconnect():
    metrics = evidence.temporal_metrics()
    assert metrics["static_noise_ev_delta_p95"] < 0.002
    assert metrics["static_noise_gain_delta_ev_p95"] < 0.002
    assert 0.08 < metrics["slow_drift_total_applied_ev"] < 0.11
    assert metrics["hard_cut_instantaneous_settling_frames"] == 0
    assert metrics["reconnect_frame_has_no_estimate"] is True
    assert metrics["first_valid_frame_after_reconnect_behavior"] == "bounded_wb"


def test_runtime_timing_snapshot_uses_production_hub_stats():
    observation = evidence.runtime_stats_evidence(minimum_frames=8)

    assert observation["source"] == "FrameHub.stats_dict production EWMA fields"
    assert observation["frames_out"] >= 8
    assert observation["configuration"]["camera"] == "synthetic"
    assert observation["configuration"]["output"] == "null"
    assert set(observation["timings_ms"]) == {
        "capture_read_ms",
        "segmentation_ms",
        "background_ms",
        "composite_ms",
        "output_send_ms",
        "frame_processing_ms",
    }
    assert all(
        isinstance(value, float) and value >= 0.0
        for value in observation["timings_ms"].values()
    )


def test_evidence_payload_is_numerically_reproducible_and_self_fingerprinted():
    first = evidence.deterministic_evidence()
    second = evidence.deterministic_evidence()
    assert first == second
    claimed = first.pop("deterministic_sha256")
    canonical = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == claimed
    assert claimed == second["deterministic_sha256"]


def test_contact_sheet_is_a_generated_aid_not_the_acceptance_oracle(tmp_path):
    path = tmp_path / "contact-sheet.png"
    evidence.write_contact_sheet(path)
    image = Image.open(path)
    assert image.size == (960, 420)
    assert image.mode == "RGB"


def test_wb_confidence_includes_neutral_availability():
    selected = evidence.method_metrics(evidence.synthetic_scene(), "bounded_wb")[
        "estimate"
    ]
    saturated = evidence.edge_case_metrics()["saturated_backdrop"]

    assert selected["exposure_confidence"] >= evidence.MIN_CONFIDENCE
    assert selected["white_balance_confidence"] >= evidence.MIN_CONFIDENCE
    assert selected["confidence"] == selected["white_balance_confidence"]
    assert saturated["behavior"] == "exposure_only"
    assert saturated["exposure_confidence"] >= evidence.MIN_CONFIDENCE
    assert saturated["white_balance_confidence"] == 0.0
    assert saturated["confidence"] == saturated["exposure_confidence"]


def test_committed_baseline_and_contact_sheet_match_the_generator(tmp_path):
    root = Path(__file__).resolve().parents[1]
    baseline_path = (
        root / "tests" / "fixtures" / "visual" / "geometry-color-baseline.json"
    )
    contact_path = (
        root / "tests" / "fixtures" / "visual" / "geometry-color-contact-sheet.png"
    )
    assert json.loads(baseline_path.read_text(encoding="utf-8")) == (
        evidence.deterministic_evidence()
    )

    regenerated = tmp_path / "contact-sheet.png"
    evidence.write_contact_sheet(regenerated)
    with Image.open(contact_path) as committed, Image.open(regenerated) as current:
        assert committed.mode == current.mode == "RGB"
        assert committed.size == current.size == (960, 420)
        assert np.array_equal(np.asarray(committed), np.asarray(current))


def test_log_luminance_metric_reports_exact_one_ev_pair():
    foreground = np.full((5, 7, 3), 0.20, dtype=np.float32)
    backdrop = np.full((5, 7, 3), 0.40, dtype=np.float32)
    region = np.ones((5, 7), dtype=bool)
    assert evidence.log_luminance_gap(
        foreground, backdrop, region, region
    ) == pytest.approx(1.0, abs=1e-6)


def test_linear_transfer_functions_round_trip_and_are_monotonic():
    encoded = np.linspace(0.0, 1.0, 257)
    decoded = evidence.srgb_to_linear(encoded)
    restored = evidence.linear_to_srgb(decoded)
    assert np.all(np.diff(decoded) >= 0.0)
    assert np.max(np.abs(restored - encoded)) < 1e-12
    assert math.isfinite(float(decoded.sum()))
