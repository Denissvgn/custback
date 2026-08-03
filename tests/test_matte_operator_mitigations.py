"""Publication contract for MATTE-0.4 operator guidance."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "matte-operator-mitigations.md"


def test_operator_guide_is_evidence_gated_reversible_and_privacy_preserving():
    text = GUIDE.read_text("utf-8")
    normalized = " ".join(text.split())
    for required in (
        "generated proxy does not qualify `0.67`",
        "model-backed",
        "exact rollback authority",
        "`segmentation_generation`",
        "`effective_rvm_downsample_ratio`",
        "`effective_edge_refine`",
        "`effective_mask_shift`",
        "`effective_use_model_foreground`",
        "`effective_light_wrap`",
        "`light_wrap: 0`",
        "`use_model_foreground: false`",
        "`rvm_downsample: 0.0`",
        "`mask_shift: 0`",
        "`edge_refine: true`",
        "`delegate: cpu`",
        "`output.fps: 30`",
        "rebuilds the segmenter and refiner",
        "does not rebuild the segmenter",
        "affects only `HeuristicSegmenter`",
        "never writes camera controls",
        "does not create alpha",
        "`privacy-slate`",
    ):
        assert required in normalized


def test_every_operator_experiment_names_apply_rollback_and_confirmation():
    text = GUIDE.read_text("utf-8")
    for heading in (
        "Light-wrap isolation",
        "Model-foreground isolation",
        "Reviewed RVM ratio or mask shift",
        "MediaPipe GPU delegate",
        "MediaPipe edge refinement",
        "Output-rate diagnosis",
    ):
        section = text.split(f"### {heading}", 1)[1].split("\n### ", 1)[0]
        assert "**Apply:**" in section
        assert "**Rollback:**" in section
        assert "**Confirm:**" in section
        assert "**Resource effect:**" in section
