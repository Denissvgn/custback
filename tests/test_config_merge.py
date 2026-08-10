"""Shared RFC 7396 configuration merge semantics."""

import pytest
from pydantic import ValidationError

from custback.config import (
    AppConfig,
    legacy_matte_policy_patch,
    uses_legacy_matte_policy,
)
from custback.config_merge import merge_patch


def test_merge_patch_recurses_replaces_and_deep_copies_inputs():
    target = {
        "object": {"keep": [1, 2], "replace": {"old": True}},
        "scalar": 1,
    }
    patch = {
        "object": {"replace": ["new"]},
        "scalar": {"nested": 2},
    }

    merged = merge_patch(target, patch)

    assert merged == {
        "object": {"keep": [1, 2], "replace": ["new"]},
        "scalar": {"nested": 2},
    }
    merged["object"]["keep"].append(3)
    merged["object"]["replace"].append("changed")
    assert target["object"]["keep"] == [1, 2]
    assert patch["object"]["replace"] == ["new"]


def test_config_null_resets_known_field_without_resetting_siblings():
    original = AppConfig.from_dict(
        {
            "api": {
                "uploads": {
                    "image_max_bytes": 1234,
                    "max_files": 7,
                }
            }
        }
    )

    candidate = original.patched({"api": {"uploads": {"max_files": None}}})

    assert candidate.api.uploads.max_files == 100
    assert candidate.api.uploads.image_max_bytes == 1234
    assert original.api.uploads.max_files == 7


def test_config_patch_retains_unknown_null_for_strict_validation():
    with pytest.raises(ValidationError) as excinfo:
        AppConfig().patched({"api": {"unknown_limit": None}})

    assert excinfo.value.errors()[0]["loc"] == ("api", "unknown_limit")


def test_visual_patch_null_resets_nested_defaults_without_resetting_siblings():
    original = AppConfig.from_dict(
        {
            "compositing": {
                "blend_space": "linear_srgb",
                "color_correction": {
                    "mode": "auto",
                    "strength": 0.9,
                    "adaptation_time_s": 2.0,
                },
            }
        }
    )

    candidate = original.patched(
        {
            "compositing": {
                "color_correction": {
                    "strength": None,
                }
            }
        }
    )

    assert candidate.compositing.color_correction.strength == 0.5
    assert candidate.compositing.color_correction.mode == "auto"
    assert candidate.compositing.color_correction.adaptation_time_s == 2.0
    assert candidate.compositing.blend_space == "linear_srgb"
    assert original.compositing.color_correction.strength == 0.9


def test_schema_v1_null_resets_use_legacy_matte_semantics_and_keep_siblings():
    original = AppConfig.from_dict(
        {
            "segmentation": {
                "backend": "mediapipe",
                "mask_shift": -2,
                "temporal_smoothing": 0.8,
                "boundary_stabilization": {
                    "mode": "motion_aware",
                    "time_constant_s": 0.2,
                },
            },
            "acceleration": {
                "mode": "auto",
                "provider": "cuda",
                "device_id": 4,
            },
            "compositing": {
                "light_wrap": 0.8,
                "light_wrap_stabilization": {
                    "mode": "temporal_bounded",
                    "time_constant_s": 0.3,
                },
            },
        }
    )

    candidate = original.patched(
        {
            "segmentation": {
                "temporal_smoothing": None,
                "boundary_stabilization": {"mode": None},
            },
            "acceleration": None,
            "compositing": {
                "light_wrap": None,
                "light_wrap_stabilization": {"mode": None},
            },
        }
    )

    assert candidate.segmentation.temporal_smoothing == 0.35
    assert candidate.segmentation.boundary_stabilization.mode == "off"
    assert candidate.segmentation.boundary_stabilization.time_constant_s == 0.2
    assert candidate.segmentation.mask_shift == -2
    assert candidate.acceleration == AppConfig().acceleration
    assert candidate.compositing.light_wrap == 0.25
    assert candidate.compositing.light_wrap_stabilization.mode == "off"
    assert candidate.compositing.light_wrap_stabilization.time_constant_s == 0.3
    assert original.segmentation.temporal_smoothing == 0.8
    assert original.acceleration.device_id == 4


def test_output_dimension_pair_can_be_reset_atomically_but_not_one_side():
    original = AppConfig.from_dict({"output": {"width": 1920, "height": 1080}})

    reset = original.patched({"output": {"width": None, "height": None}})

    assert reset.output.width is None
    assert reset.output.height is None
    with pytest.raises(ValidationError, match="configured together"):
        original.patched({"output": {"width": None}})
    assert (original.output.width, original.output.height) == (1920, 1080)


def test_legacy_matte_policy_patch_is_exact_detached_and_scope_bounded():
    expected = {
        "segmentation": {
            "backend": "auto",
            "delegate": "cpu",
            "rvm_downsample": 0.0,
            "threshold": 0.5,
            "mask_blur": 7,
            "edge_refine": True,
            "mask_shift": 0,
            "temporal_smoothing": 0.35,
            "boundary_stabilization": {
                "mode": "off",
                "time_constant_s": 0.1,
                "max_motion_px_per_s": 720.0,
            },
            "spatial_edge_refinement": {
                "mode": "legacy_watershed",
                "reference_short_edge_px": 720,
                "radius_at_reference_px": 8,
                "min_radius_px": 2,
                "max_radius_px": 12,
            },
        },
        "acceleration": {
            "mode": "auto",
            "provider": "auto",
            "device_id": 0,
        },
        "compositing": {
            "light_wrap": 0.25,
            "use_model_foreground": True,
            "blend_space": "srgb_legacy",
            "light_wrap_stabilization": {
                "mode": "off",
                "time_constant_s": 0.12,
            },
            "color_correction": {
                "mode": "off",
                "strength": 0.5,
                "exposure_limit_ev": 0.85,
                "white_balance_strength": 0.5,
                "adaptation_time_s": 0.8,
            },
        },
    }

    first = legacy_matte_policy_patch()
    second = legacy_matte_policy_patch()

    assert first == second == expected
    assert set(first) == {"segmentation", "acceleration", "compositing"}
    assert "schema_version" not in first
    assert "model_path" not in first["segmentation"]
    first["segmentation"]["boundary_stabilization"]["mode"] = "motion_aware"
    first["compositing"]["color_correction"]["strength"] = 0.9
    assert legacy_matte_policy_patch() == expected


def test_one_patch_matte_rollback_preserves_schema_model_and_unrelated_config():
    candidate = AppConfig.from_dict(
        {
            "schema_version": 1,
            "camera": {"width": 640, "height": 480, "mirror": True},
            "background": {"mode": "color", "color": [1, 2, 3]},
            "segmentation": {
                "backend": "rvm",
                "model_path": "/models/operator-rvm.onnx",
                "rvm_downsample": 0.75,
                "threshold": 0.7,
                "mask_blur": 0,
                "edge_refine": False,
                "mask_shift": -2,
                "temporal_smoothing": 0.0,
                "boundary_stabilization": {"mode": "motion_aware"},
                "spatial_edge_refinement": {"mode": "stable_guided"},
            },
            "acceleration": {
                "mode": "gpu_required",
                "provider": "cuda",
                "device_id": 3,
            },
            "compositing": {
                "light_wrap": 0.8,
                "use_model_foreground": False,
                "blend_space": "linear_srgb",
                "light_wrap_stabilization": {"mode": "temporal_bounded"},
                "color_correction": {"mode": "auto", "strength": 0.9},
            },
            "output": {"width": 1920, "height": 1080, "fps": 60},
            "api": {"port": 9876},
        }
    )
    before = candidate.to_dict()

    rolled_back = candidate.patched(legacy_matte_policy_patch())

    assert uses_legacy_matte_policy(candidate) is False
    assert uses_legacy_matte_policy(rolled_back) is True
    assert candidate.to_dict() == before
    assert rolled_back.schema_version == candidate.schema_version == 1
    assert rolled_back.segmentation.model_path == "/models/operator-rvm.onnx"
    assert rolled_back.camera == candidate.camera
    assert rolled_back.background == candidate.background
    assert rolled_back.output == candidate.output
    assert rolled_back.api == candidate.api
    assert rolled_back.avatar == candidate.avatar
    assert rolled_back.backdrop_targets == candidate.backdrop_targets


def test_legacy_matte_policy_matcher_rejects_non_config_values():
    with pytest.raises(TypeError, match="AppConfig"):
        uses_legacy_matte_policy({})  # type: ignore[arg-type]
