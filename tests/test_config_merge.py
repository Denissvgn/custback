"""Shared RFC 7396 configuration merge semantics."""

import pytest
from pydantic import ValidationError

from custback.config import AppConfig
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


def test_output_dimension_pair_can_be_reset_atomically_but_not_one_side():
    original = AppConfig.from_dict({"output": {"width": 1920, "height": 1080}})

    reset = original.patched({"output": {"width": None, "height": None}})

    assert reset.output.width is None
    assert reset.output.height is None
    with pytest.raises(ValidationError, match="configured together"):
        original.patched({"output": {"width": None}})
    assert (original.output.width, original.output.height) == (1920, 1080)
