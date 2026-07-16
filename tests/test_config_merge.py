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
