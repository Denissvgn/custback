"""Public cadence behavior and configuration contract."""

from __future__ import annotations

from pathlib import Path

from custback.config import AppConfig


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "cadence-observability.md"


def test_public_cadence_contract_distinguishes_repeats_from_new_work() -> None:
    text = GUIDE.read_text("utf-8")
    for required in (
        "output_send_count",
        "segmentation_update_count",
        "base_composite_update_count",
        "repeat",
        "privacy",
    ):
        assert required in text


def test_rejected_feature_has_no_runtime_configuration_surface() -> None:
    def keys(value: object) -> list[str]:
        if isinstance(value, dict):
            result: list[str] = []
            for key, child in value.items():
                result.append(str(key))
                result.extend(keys(child))
            return result
        if isinstance(value, (list, tuple)):
            return [key for child in value for key in keys(child)]
        return []

    config_keys = keys(AppConfig().to_dict())
    assert all(
        "interpolat" not in key.lower() and "synthes" not in key.lower()
        for key in config_keys
    )
