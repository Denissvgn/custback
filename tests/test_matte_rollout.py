from __future__ import annotations

import json
import threading

import pytest

from custback.config import AppConfig, RuntimeConfig, legacy_matte_policy_patch
from custback.hub import FrameHub
from custback.matte_rollout import (
    MatteRolloutTelemetry,
    classify_matte_patch,
    empty_matte_rollout_status,
    validate_matte_rollout_status,
)
from custback.pipeline import Pipeline


def test_rollout_status_is_exact_content_free_and_fail_closed() -> None:
    status = empty_matte_rollout_status()

    assert validate_matte_rollout_status(status) == status
    assert status == {
        "schema": "custback.matte-rollout-status",
        "version": 1,
        "stage": "compatibility_hold",
        "decision": "held_pending_physical_qualification",
        "configured_schema_version": 1,
        "config_version": 0,
        "qualified_default_active": False,
        "preset_catalog_version": 1,
        "preset_evidence_status": "not_qualified",
        "legacy_policy_available": True,
        "legacy_policy_active": True,
        "rollback_patch_id": "matte-legacy-v1",
        "patch_attempts": 0,
        "patch_in_flight": 0,
        "patch_successes": 0,
        "patch_failures": 0,
        "legacy_rollbacks": 0,
        "last_outcome": "none",
    }
    serialized = json.dumps(status, sort_keys=True)
    for forbidden in ("model_path", "device", "package", "/", "\\"):
        assert forbidden not in serialized

    altered = dict(status, model_path="/private/models/rvm.onnx")
    with pytest.raises(ValueError, match="exact version-1"):
        validate_matte_rollout_status(altered)

    altered = dict(status, qualified_default_active=True)
    with pytest.raises(ValueError, match="invalid rollout disposition"):
        validate_matte_rollout_status(altered)


def test_patch_classification_is_value_free_and_rollback_is_exact() -> None:
    rollback = legacy_matte_policy_patch()

    assert classify_matte_patch({"background": {"mode": "color"}}) is None
    assert classify_matte_patch({"segmentation": {"threshold": 0.61}}) == (
        "matte_update"
    )
    assert classify_matte_patch(rollback) == "legacy_rollback"
    rollback["segmentation"]["threshold"] = 0.61
    assert classify_matte_patch(rollback) == "matte_update"


def test_rollout_telemetry_preserves_concurrent_in_flight_truth() -> None:
    telemetry = MatteRolloutTelemetry()
    first_attempt_ready = threading.Event()
    allow_completion = threading.Event()

    def worker() -> None:
        telemetry.record_attempt()
        first_attempt_ready.set()
        assert allow_completion.wait(2.0)
        telemetry.record_success("matte_update")

    thread = threading.Thread(target=worker)
    thread.start()
    assert first_attempt_ready.wait(2.0)
    in_flight = telemetry.snapshot(AppConfig())
    assert in_flight["patch_attempts"] == 1
    assert in_flight["patch_in_flight"] == 1
    assert in_flight["last_outcome"] == "attempt"

    allow_completion.set()
    thread.join(2.0)
    assert not thread.is_alive()
    complete = telemetry.snapshot(AppConfig())
    assert complete["patch_attempts"] == complete["patch_successes"] == 1
    assert complete["patch_in_flight"] == complete["patch_failures"] == 0
    assert complete["last_outcome"] == "success"


def test_pipeline_counts_matte_success_failure_and_one_patch_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = RuntimeConfig(AppConfig())
    pipeline = Pipeline(runtime, FrameHub())

    def apply_stub(
        patch,
        timeout=5.0,
        *,
        origin="internal",
        rollout_attempt=None,
    ):
        del timeout, origin, rollout_attempt
        if patch.get("segmentation", {}).get("threshold") == 0.99:
            raise ValueError("private failure /must/not/escape")
        return runtime.read()

    monkeypatch.setattr(pipeline, "_apply_config_patch", apply_stub)

    pipeline.apply_config_patch({"segmentation": {"threshold": 0.61}})
    with pytest.raises(ValueError, match="private failure"):
        pipeline.apply_config_patch({"segmentation": {"threshold": 0.99}})
    pipeline.apply_config_patch(legacy_matte_policy_patch())
    # Unrelated configuration is intentionally outside the matte canary.
    pipeline.apply_config_patch({"background": {"mode": "color"}})

    status = pipeline._matte_rollout.snapshot(runtime.snapshot())
    assert status["patch_attempts"] == 3
    assert status["patch_in_flight"] == 0
    assert status["patch_successes"] == 2
    assert status["patch_failures"] == 1
    assert status["legacy_rollbacks"] == 1
    assert status["last_outcome"] == "rollback"
    assert "private" not in json.dumps(status)


def test_hub_rejects_unknown_or_inconsistent_rollout_status() -> None:
    hub = FrameHub()
    valid = empty_matte_rollout_status()
    hub.update_stats(matte_rollout=valid)
    assert hub.stats_dict()["matte_rollout"] == valid

    with pytest.raises(ValueError, match="attempt counters"):
        hub.update_stats(matte_rollout=dict(valid, patch_attempts=1))
    with pytest.raises(ValueError, match="exact version-1"):
        hub.update_stats(matte_rollout=dict(valid, raw_error="secret"))
    # Generic status writers may advance config state before the next output
    # frame publishes its matching rollout snapshot. Consumers fail closed on
    # this transient mismatch instead of the hub forging policy alignment.
    hub.update_stats(config_version=2)
    assert hub.stats_dict()["matte_rollout"]["config_version"] == 0


def test_pipeline_publishes_custom_prestart_rollout_at_matching_generation() -> None:
    cfg = AppConfig().patched({"segmentation": {"temporal_smoothing": 0.7}})
    runtime = RuntimeConfig(cfg)
    hub = FrameHub()

    Pipeline(runtime, hub)

    status = hub.stats_dict()
    assert status["config_version"] == 0
    assert status["matte_rollout"]["config_version"] == 0
    assert status["matte_rollout"]["legacy_policy_active"] is False
