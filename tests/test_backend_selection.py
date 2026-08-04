from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import custback.pipeline as pipeline_mod
import custback.segmentation as segmentation_mod
from custback.acceleration import GpuRequiredError
from custback.api.server import _SegmentationSelectionResponse
from custback.config import (
    AccelerationConfig,
    CompositingConfig,
    SegmentationConfig,
)
from custback.hub import FrameHub
from custback.matte_policy import MatteBackendKind, resolve_matte_policy
from custback.segmentation import (
    ActivationResult,
    PreparationResult,
    SegmentationQualityTier,
    SelectionReasonCategory,
    SegmenterBackendName,
    SegmenterPreparation,
    SegmenterSelectionAttempt,
    create_segmenter,
    preacquire_segmenter_model,
    segmenter_selection_status,
)


class _FakeMediaPipe:
    device = "cpu"
    matte_backend_kind = MatteBackendKind.CONFIDENCE_MASK_VIDEO

    def __init__(self, *_args, **_kwargs) -> None:
        pass


class _FakeRVM:
    device = "cuda"
    matte_backend_kind = MatteBackendKind.TRUE_ALPHA_RECURRENT

    def __init__(self, *_args, **_kwargs) -> None:
        self.accel = SimpleNamespace(
            status=lambda: SimpleNamespace(active_provider="cuda")
        )


def _attempt(
    backend: SegmenterBackendName,
    tier: SegmentationQualityTier,
    *,
    preparation_result: PreparationResult,
    activation_result: ActivationResult = "not-attempted",
    reason_category: SelectionReasonCategory = "none",
    reason: str = "",
    guidance: str = "",
) -> SegmenterSelectionAttempt:
    return SegmenterSelectionAttempt(
        backend=backend,
        quality_tier=tier,
        preparation_result=preparation_result,
        activation_result=activation_result,
        reason_category=reason_category,
        reason=reason,
        guidance=guidance,
    )


def _run_a_preparation() -> SegmenterPreparation:
    return SegmenterPreparation(
        frozenset({"mediapipe"}),
        (
            _attempt(
                "rvm",
                SegmentationQualityTier.MATTING,
                preparation_result="unavailable",
                reason_category="runtime-not-installed",
                reason="RVM unavailable: runtime not installed",
                guidance="Install the RVM runtime profile and restart.",
            ),
            _attempt(
                "mediapipe",
                SegmentationQualityTier.SEGMENTATION,
                preparation_result="ready",
            ),
        ),
    )


def _public_policy(kind: MatteBackendKind) -> dict[str, object]:
    return {
        "schema": "custback.matte-policy",
        "version": 1,
        "blend_space": "srgb_legacy",
        **resolve_matte_policy(
            SegmentationConfig(),
            CompositingConfig(),
            kind,
        ).to_dict(),
    }


def test_auto_mediapipe_retains_sanitized_rvm_downgrade(monkeypatch):
    monkeypatch.setattr(
        segmentation_mod,
        "RVMSegmenter",
        lambda *_args, **_kwargs: pytest.fail("unprepared RVM must not be retried"),
    )
    monkeypatch.setattr(segmentation_mod, "MediaPipeSegmenter", _FakeMediaPipe)

    segmenter = create_segmenter(
        SegmentationConfig(backend="auto"),
        preparation=_run_a_preparation(),
    )
    selection = segmenter_selection_status(segmenter, "auto")

    assert selection == {
        "schema": "custback.backend-selection",
        "version": 1,
        "requested_backend": "auto",
        "selected_backend": "mediapipe",
        "quality_tier": "segmentation",
        "selection_mode": "automatic",
        "fallback_active": True,
        "fallback_category": "runtime-not-installed",
        "fallback_reason": "RVM unavailable: runtime not installed",
        "guidance": "Install the RVM runtime profile and restart.",
        "active_device": "cpu",
        "active_provider": "cpu",
        "attempts": [
            {
                "backend": "rvm",
                "quality_tier": "matting",
                "preparation_result": "unavailable",
                "activation_result": "not-attempted",
                "reason_category": "runtime-not-installed",
                "reason": "RVM unavailable: runtime not installed",
                "guidance": "Install the RVM runtime profile and restart.",
            },
            {
                "backend": "mediapipe",
                "quality_tier": "segmentation",
                "preparation_result": "ready",
                "activation_result": "selected",
                "reason_category": "none",
                "reason": "",
                "guidance": "",
            },
        ],
    }
    FrameHub().update_stats(
        segmentation_selection=selection,
        matte_policy=_public_policy(MatteBackendKind.CONFIDENCE_MASK_VIDEO),
    )
    assert (
        _SegmentationSelectionResponse.model_validate(selection).quality_tier
        == "segmentation"
    )
    unsafe = json.loads(json.dumps(selection))
    unsafe["fallback_reason"] = "/private/models/rvm.onnx failed to load"
    with pytest.raises(ValueError, match="not sanitized"):
        FrameHub().update_stats(segmentation_selection=unsafe)


def test_preacquisition_retains_candidate_availability_without_raw_error(
    monkeypatch,
    caplog,
):
    private_detail = "/home/alice/onnxruntime.so failed to load"

    def import_module(name):
        if name == "onnxruntime":
            raise ModuleNotFoundError(private_detail)
        return object()

    monkeypatch.setattr(segmentation_mod.importlib, "import_module", import_module)
    monkeypatch.setattr(
        segmentation_mod,
        "acquire_builtin_model",
        lambda _backend: object(),
    )
    with caplog.at_level("INFO", logger="custback.segmentation"):
        preparation = preacquire_segmenter_model(SegmentationConfig(backend="auto"))

    assert preparation.ready_backends == frozenset({"mediapipe"})
    assert [attempt.to_dict() for attempt in preparation.attempts] == [
        _run_a_preparation().attempts[0].to_dict(),
        _run_a_preparation().attempts[1].to_dict(),
    ]
    assert private_detail not in caplog.text


def test_preacquisition_distinguishes_runtime_load_from_model_failure(monkeypatch):
    def import_module(name):
        if name == "onnxruntime":
            raise ImportError("/private/libonnxruntime.so could not be loaded")
        return object()

    def acquire_model(backend):
        if backend == "mediapipe":
            raise segmentation_mod.ModelAcquisitionError(
                "/private/model.tflite checksum detail"
            )
        return object()

    monkeypatch.setattr(segmentation_mod.importlib, "import_module", import_module)
    monkeypatch.setattr(segmentation_mod, "acquire_builtin_model", acquire_model)

    preparation = preacquire_segmenter_model(SegmentationConfig(backend="auto"))

    assert preparation.ready_backends == frozenset()
    assert [
        (attempt.backend, attempt.reason_category, attempt.reason)
        for attempt in preparation.attempts
    ] == [
        ("rvm", "runtime-unavailable", "RVM unavailable: runtime failed to load"),
        (
            "mediapipe",
            "model-unavailable",
            "MediaPipe unavailable: model is not ready",
        ),
    ]
    segmenter = create_segmenter(
        SegmentationConfig(backend="auto"),
        preparation=preparation,
    )
    selection = segmenter_selection_status(segmenter, "auto")
    FrameHub().update_stats(
        segmentation_selection=selection,
        matte_policy=_public_policy(MatteBackendKind.BINARY_COARSE),
    )
    _SegmentationSelectionResponse.model_validate(selection)


def test_explicit_and_model_constrained_mediapipe_are_not_fallbacks(monkeypatch):
    monkeypatch.setattr(segmentation_mod, "MediaPipeSegmenter", _FakeMediaPipe)

    explicit = create_segmenter(SegmentationConfig(backend="mediapipe"))
    explicit_status = segmenter_selection_status(explicit, "mediapipe")
    assert explicit_status["selection_mode"] == "explicit"
    assert explicit_status["quality_tier"] == "segmentation"
    assert explicit_status["fallback_active"] is False
    assert explicit_status["fallback_reason"] == ""

    constrained = create_segmenter(
        SegmentationConfig(backend="auto", model_path="/private/custom.tflite")
    )
    constrained_status = segmenter_selection_status(constrained, "auto")
    assert constrained_status["selection_mode"] == "model-format"
    assert constrained_status["selected_backend"] == "mediapipe"
    assert constrained_status["fallback_active"] is False
    assert "/private/custom.tflite" not in json.dumps(constrained_status)


def test_auto_rvm_cuda_reports_matting_tier_and_live_provider(monkeypatch):
    monkeypatch.setattr(segmentation_mod, "RVMSegmenter", _FakeRVM)

    segmenter = create_segmenter(SegmentationConfig(backend="auto"))
    selection = segmenter_selection_status(segmenter, "auto")

    assert selection["selected_backend"] == "rvm"
    assert selection["quality_tier"] == "matting"
    assert selection["active_device"] == "cuda"
    assert selection["active_provider"] == "cuda"
    assert selection["fallback_active"] is False
    assert selection["attempts"] == [
        {
            "backend": "rvm",
            "quality_tier": "matting",
            "preparation_result": "not-run",
            "activation_result": "selected",
            "reason_category": "none",
            "reason": "",
            "guidance": "",
        },
        {
            "backend": "mediapipe",
            "quality_tier": "segmentation",
            "preparation_result": "not-run",
            "activation_result": "not-attempted",
            "reason_category": "none",
            "reason": "",
            "guidance": "",
        },
    ]
    assert (
        _SegmentationSelectionResponse.model_validate(selection).quality_tier
        == "matting"
    )


@pytest.mark.parametrize(
    ("backend", "selected", "tier", "device"),
    [
        ("heuristic", "heuristic", "heuristic", "cpu"),
        ("none", "none", "none", "none"),
    ],
)
def test_explicit_non_ml_quality_tiers(backend, selected, tier, device):
    segmenter = create_segmenter(SegmentationConfig(backend=backend))
    selection = segmenter_selection_status(segmenter, backend)

    assert selection["selected_backend"] == selected
    assert selection["quality_tier"] == tier
    assert selection["active_device"] == device
    assert selection["fallback_active"] is False
    assert _SegmentationSelectionResponse.model_validate(selection).quality_tier == tier


def test_prepared_gpu_required_cannot_degrade_below_rvm(monkeypatch):
    monkeypatch.setattr(
        segmentation_mod,
        "MediaPipeSegmenter",
        lambda *_args, **_kwargs: pytest.fail("gpu_required must fail before fallback"),
    )

    with pytest.raises(GpuRequiredError, match="preparation did not succeed"):
        create_segmenter(
            SegmentationConfig(backend="auto"),
            acceleration=AccelerationConfig(mode="gpu_required"),
            preparation=_run_a_preparation(),
        )


def test_gpu_required_generic_rvm_activation_failure_cannot_fall_back(monkeypatch):
    private_detail = "/private/models/rvm.onnx provider activation failed"

    def fail_rvm(*_args, **_kwargs):
        raise RuntimeError(private_detail)

    monkeypatch.setattr(segmentation_mod, "RVMSegmenter", fail_rvm)
    monkeypatch.setattr(
        segmentation_mod,
        "MediaPipeSegmenter",
        lambda *_args, **_kwargs: pytest.fail("gpu_required must not try MediaPipe"),
    )

    with pytest.raises(
        GpuRequiredError,
        match="gpu_required RVM backend activation failed",
    ) as failure:
        create_segmenter(
            SegmentationConfig(backend="auto"),
            acceleration=AccelerationConfig(mode="gpu_required"),
        )

    assert private_detail not in str(failure.value)


def test_raw_backend_exception_never_enters_selection_or_operator_log(
    monkeypatch,
    caplog,
):
    private_detail = "/home/alice/private-model.onnx secret provider traceback"

    def fail_rvm(*_args, **_kwargs):
        raise RuntimeError(private_detail)

    monkeypatch.setattr(segmentation_mod, "RVMSegmenter", fail_rvm)
    monkeypatch.setattr(segmentation_mod, "MediaPipeSegmenter", _FakeMediaPipe)

    with caplog.at_level("INFO", logger="custback.segmentation"):
        segmenter = create_segmenter(SegmentationConfig(backend="auto"))
    selection = segmenter_selection_status(segmenter, "auto")
    rendered = json.dumps(selection)

    assert selection["fallback_category"] == "activation-failed"
    assert selection["fallback_reason"] == "RVM unavailable: backend activation failed"
    assert private_detail not in rendered
    assert private_detail not in caplog.text


def test_hub_rejects_unversioned_or_multiline_selection():
    hub = FrameHub()
    selection = hub.stats_dict()["segmentation_selection"]

    malformed = dict(selection)
    malformed["fallback_reason"] = "private\ntrace"
    with pytest.raises(ValueError, match="one-line"):
        hub.update_stats(segmentation_selection=malformed)

    unversioned = dict(selection)
    unversioned["version"] = 2
    with pytest.raises(ValueError, match="unsupported"):
        hub.update_stats(segmentation_selection=unversioned)

    impossible = json.loads(json.dumps(selection))
    impossible["requested_backend"] = "rvm"
    with pytest.raises(ValueError, match="explicit selection"):
        hub.update_stats(segmentation_selection=impossible)


def test_hub_policy_publication_is_sanitized_cross_bound_and_atomic():
    hub = FrameHub()
    original = hub.stats_dict()
    unsafe_policy = json.loads(json.dumps(original["matte_policy"]))
    unsafe_policy["controls"]["raw_alpha"]["effective"] = "/private/models/rvm.onnx"

    with pytest.raises(ValueError, match="allowed public policy value"):
        hub.update_stats(frames_in=99, matte_policy=unsafe_policy)

    after_rejection = hub.stats_dict()
    assert after_rejection["frames_in"] == original["frames_in"]
    assert after_rejection["matte_policy"] == original["matte_policy"]

    with pytest.raises(ValueError, match="policy kind do not match"):
        hub.update_stats(
            segmentation_selection=original["segmentation_selection"],
            matte_policy=_public_policy(MatteBackendKind.TRUE_ALPHA_RECURRENT),
        )
    assert (
        hub.stats_dict()["segmentation_selection"] == original["segmentation_selection"]
    )


def test_public_acceleration_fallback_reason_is_allowlisted():
    private_detail = "/home/alice/libonnxruntime.so: CUDA provider traceback"
    segmenter = SimpleNamespace(
        accel=SimpleNamespace(
            status=lambda: SimpleNamespace(
                requested_mode="auto",
                requested_provider="cuda",
                device_id=0,
                state="cpu_fallback",
                active_provider="cpu",
                fallback_active=True,
                fallback_reason=private_detail,
                fallback_count=1,
                last_transition_ms=12.0,
            )
        )
    )

    public = pipeline_mod._acceleration_stats(segmenter)

    assert public["acceleration_fallback_reason"] == (
        "requested accelerator unavailable; using CPU"
    )
    assert private_detail not in json.dumps(public)
