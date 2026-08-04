"""Publication contract for the MATTE-3.3 interpolation decision."""

from __future__ import annotations

from pathlib import Path

from custback.config import AppConfig


ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs" / "adr" / "0002-output-rate-matte-interpolation.md"


def test_adr_rejects_interpolation_and_compares_every_required_strategy() -> None:
    text = ADR.read_text("utf-8")
    normalized = " ".join(text.split())

    for required in (
        "- Status: Accepted",
        "Decision: Reject output-rate matte interpolation for the current runtime",
        "Repeat the last guarded output",
        "Low-latency motion-compensated composite interpolation",
        "Advance the backdrop while holding the subject/matte",
        "Match output FPS to sustained unique-input FPS",
        "Retain as default",
        "Reject: not qualified",
        "manual diagnosis; reject as automatic default",
        "adds no interpolator",
        "frame-history buffer",
        "automatic FPS negotiation",
    ):
        assert required in normalized


def test_adr_fails_closed_on_evidence_budget_and_privacy() -> None:
    normalized = " ".join(ADR.read_text("utf-8").split())

    for required in (
        "MATTE-3.1 physical-camera evidence",
        "MATTE-3.4 fixed-replay service evidence",
        "model-backed MATTE-2.5 profile",
        "owner-only, bounded, digest-bound output-timeline evidence format",
        "actual 30 Hz output artifact",
        "presentation latency",
        "occlusion error",
        "CPU utilization",
        "GPU utilization",
        "final raw-echo guard again",
        "inapplicable in remote mode",
        "proxy-only evidence",
        "privacy failure keeps the outcome rejected",
    ):
        assert required in normalized


def test_future_acceptance_keeps_synthetic_outputs_out_of_temporal_state() -> None:
    normalized = " ".join(ADR.read_text("utf-8").split())

    for required in (
        "`interpolated_output_count`",
        "`interpolated_output_fps`",
        "does not increment capture sequence",
        "segmentation update",
        "model invocation",
        "refiner update",
        "base-composite update counts",
        "never fed back as a capture/model observation",
        "rolls back immediately to exact repeat",
    ):
        assert required in normalized


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
