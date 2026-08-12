"""Status/OpenAPI/WebUI integration for runtime performance v2."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from typing import Any, cast

import pytest

from custback.api.server import _StatusResponse
from custback.api.webui import WEBUI_HTML
from custback.hub import TIMING_FIELD_NAMES, FrameHub
from custback.runtime_performance import (
    RUNTIME_STAGE_NAMES,
    PerformanceEpochKey,
    PublisherTelemetry,
    RuntimePerformanceTracker,
    empty_runtime_performance_status,
)


class _Clock:
    now_ns = 0

    def __call__(self) -> int:
        return self.now_ns


def test_hub_default_and_openapi_expose_strict_runtime_performance_v2() -> None:
    status = FrameHub().stats_dict()
    runtime = cast(dict[str, Any], status["runtime_performance"])
    assert runtime == empty_runtime_performance_status()
    assert runtime["schema_version"] == 2
    assert runtime["state"] == "warming"
    assert set(runtime["stage_p50_ms"]) == set(RUNTIME_STAGE_NAMES)
    assert set(runtime["stage_p95_ms"]) == set(RUNTIME_STAGE_NAMES)
    assert "/" not in json.dumps(runtime, sort_keys=True)
    independent = empty_runtime_performance_status()
    independent_p50 = cast(dict[str, float | None], independent["stage_p50_ms"])
    independent_current = cast(dict[str, Any], independent["current_epoch"])
    independent_current_p50 = cast(
        dict[str, float | None], independent_current["stage_p50_ms"]
    )
    independent_p50["capture.read"] = 1.0
    assert independent_current_p50["capture.read"] is None

    model = _StatusResponse.model_validate({**status, "native_ring": "unsupported"})
    public = model.model_dump(mode="json", by_alias=True)
    assert public["runtime_performance"] == runtime
    assert set(public["timing_ms"]) == set(TIMING_FIELD_NAMES)

    schema = _StatusResponse.model_json_schema()
    assert schema["properties"]["runtime_performance"]["$ref"].endswith(
        "/_RuntimePerformanceResponse"
    )
    runtime_schema = schema["$defs"]["_RuntimePerformanceResponse"]
    assert runtime_schema["additionalProperties"] is False
    assert set(runtime_schema["properties"]) == set(runtime)
    assert set(runtime_schema["required"]) == set(runtime)
    stage_schema = schema["$defs"]["_RuntimeStageMapResponse"]
    assert stage_schema["additionalProperties"] is False
    assert set(stage_schema["properties"]) == set(RUNTIME_STAGE_NAMES)
    assert set(stage_schema["required"]) == set(RUNTIME_STAGE_NAMES)


def test_hub_runtime_registry_validates_consistency_and_defensively_copies() -> None:
    hub = FrameHub()
    runtime = empty_runtime_performance_status()
    hub.update_stats(runtime_performance=runtime)

    stage_p50 = cast(dict[str, float | None], runtime["stage_p50_ms"])
    stage_p50["capture.read"] = 999.0
    public = cast(dict[str, Any], hub.stats_dict()["runtime_performance"])
    assert public["stage_p50_ms"]["capture.read"] is None
    public["stage_p50_ms"]["capture.read"] = 777.0
    assert (
        cast(dict[str, Any], hub.stats_dict()["runtime_performance"])["stage_p50_ms"][
            "capture.read"
        ]
        is None
    )

    extra = empty_runtime_performance_status()
    extra["private_path"] = "/private/operator/model.onnx"
    with pytest.raises(ValueError, match="exactly"):
        hub.update_stats(runtime_performance=extra)

    unsafe = empty_runtime_performance_status()
    unsafe["dominant_stage"] = "/private/operator/model.onnx"
    current = cast(dict[str, object], unsafe["current_epoch"])
    current["dominant_stage"] = "/private/operator/model.onnx"
    with pytest.raises(ValueError, match="dominant stage"):
        hub.update_stats(runtime_performance=unsafe)

    inconsistent = empty_runtime_performance_status()
    inconsistent["output_healthy"] = True
    with pytest.raises(ValueError, match="health flags"):
        hub.update_stats(runtime_performance=inconsistent)

    alias_drift = empty_runtime_performance_status()
    alias_drift["target_fps"] = 30.0
    with pytest.raises(ValueError, match="alias"):
        hub.update_stats(runtime_performance=alias_drift)

    deadline_drift = empty_runtime_performance_status()
    deadline_drift["transport_deadline_ms"] = 33.333
    with pytest.raises(ValueError, match="deadline domains"):
        hub.update_stats(runtime_performance=deadline_drift)

    cadence_drift = empty_runtime_performance_status()
    cadence_drift["cadence_status"] = "matched"
    with pytest.raises(ValueError, match="cadence status"):
        hub.update_stats(runtime_performance=cadence_drift)


def test_tracker_snapshot_round_trips_through_hub_and_status_model() -> None:
    clock = _Clock()
    tracker = RuntimePerformanceTracker(
        30,
        PerformanceEpochKey(9, 2, 3, 4),
        color_correction_mode="auto",
        light_wrap=1.0,
        clock_ns=clock,
    )
    tracker.mark_ready(at_ns=0)
    tracker.update_publisher(
        PublisherTelemetry(
            mode="sink-paced",
            state="running",
            handoff_overwrite_count=7,
            missed_slot_count=2,
            slate_send_count=1,
            pending_depth=1,
            output_base_config_version=9,
        ),
        at_ns=1_000_000_000,
    )
    tracker.snapshot(at_ns=3_000_000_000)
    runtime = tracker.snapshot(at_ns=6_000_000_000)
    assert runtime["state"] == "degraded"

    hub = FrameHub()
    old_timing = copy.deepcopy(hub.stats_dict()["timing_ms"])
    hub.update_stats(runtime_performance=runtime)
    body = hub.stats_dict()
    assert body["timing_ms"] == old_timing
    assert body["output_send_fps"] == 0.0
    assert body["runtime_performance"] == runtime

    public = _StatusResponse.model_validate(
        {**body, "native_ring": "unsupported"}
    ).model_dump(mode="json", by_alias=True)
    published = cast(dict[str, Any], public["runtime_performance"])
    assert published["state"] == "degraded"
    assert published["publisher"]["handoff_overwrite_count"] == 7
    assert published["recommended_mitigation"] == {
        "config_version": 9,
        "kind": "disable-color-and-light-wrap",
        "patch": {
            "compositing": {
                "color_correction": {"mode": "off"},
                "light_wrap": 0.0,
            }
        },
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_webui_renders_truthful_runtime_performance_rows() -> None:
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    title_case = (
        "function titleCase"
        + script.split("function titleCase", 1)[1].split("function setView", 1)[0]
    )
    runtime_rows = (
        "function runtimePerformanceRows"
        + script.split("function runtimePerformanceRows", 1)[1].split(
            "function geometrySummary", 1
        )[0]
    )
    harness = r"""
const rows = runtimePerformanceRows({runtime_performance: {
  schema_version: 2,
  state: "degraded",
  cadence_status: "unexpected-shortfall",
  target_fps: 30,
  transport_target_fps: 30,
  unique_target_fps: 15,
  transport_deadline_ms: 33.333,
  processing_deadline_ms: 66.667,
  output_send_fps: 30,
  sent_unique_base_fps: 8,
  output_attainment: 1,
  unique_attainment: 8 / 15,
  output_healthy: true,
  unique_healthy: false,
  dominant_stage: "compositor.total",
  stage_p95_ms: {"compositor.total": 91.6},
  recommended_mitigation: {
    config_version: 9,
    kind: "disable-color-and-light-wrap",
    patch: {compositing: {color_correction: {mode: "off"}, light_wrap: 0}},
  },
}});
process.stdout.write(JSON.stringify(rows));
"""
    result = subprocess.run(
        ["node"],
        input=title_case + runtime_rows + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        ["Output cadence", "30.0 / 30.0 fps · 100.0% target", "good"],
        ["Unique visual cadence", "8.0 / 15.0 fps · 53.3% target", "warn"],
        [
            "Cadence interpretation",
            "Unexpected shortfall · at least one cadence target is below 90%",
            "warn",
        ],
        [
            "Deadline domains",
            "Processing 66.7 ms @ 15.0 fps · transport 33.3 ms @ 30.0 fps",
            "",
        ],
        ["Dominant stage", "Compositor Total · p95 91.6 ms", "warn"],
        [
            "Suggested patch",
            "Disable Color And Light Wrap · config v9 · "
            '{"compositing":{"color_correction":{"mode":"off"},"light_wrap":0}}',
            "warn",
        ],
    ]
    assert "coreRows.push(...runtimePerformanceRows(status))" in script


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_webui_validates_user_triggered_mitigation_and_explains_video_skips() -> None:
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    mitigation = (
        "function stabilityMitigationAction"
        + script.split("function stabilityMitigationAction", 1)[1].split(
            "function renderStabilityMitigationAction", 1
        )[0]
    )
    video = (
        "function videoCadenceSummary"
        + script.split("function videoCadenceSummary", 1)[1].split(
            "function renderDiagnostics", 1
        )[0]
    )
    cadence = (
        "function cadenceShortfallActive"
        + script.split("function cadenceShortfallActive", 1)[1].split(
            "function visualCadenceRows", 1
        )[0]
    )
    harness = r"""
function plainObject(value) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
const status = {
  config_version: 9,
  base_composite_update_fps: 15,
  background_video_source_fps: 60,
  background_video_frames_skipped: 90,
  background_video_frames_reused: 2,
  background_video_skip_ratio: 0.75,
  runtime_performance: {
    schema_version: 2,
    state: "degraded",
    cadence_status: "unexpected-shortfall",
    current_epoch: {key: {config_version: 9}},
    recommended_mitigation: {
      config_version: 9,
      kind: "disable-color-and-light-wrap",
      patch: {compositing: {color_correction: {mode: "off"}, light_wrap: 0}},
    },
  },
};
const accepted = stabilityMitigationAction(status, 9);
const stale = stabilityMitigationAction(status, 10);
status.runtime_performance.recommended_mitigation.patch.compositing.light_wrap = 0.5;
const altered = stabilityMitigationAction(status, 9);
status.runtime_performance.recommended_mitigation.kind = "__proto__";
status.runtime_performance.recommended_mitigation.patch = {};
const inherited = stabilityMitigationAction(status, 9);
const unexpectedCadence = cadenceShortfallActive(status);
const intentionalCadence = cadenceShortfallActive({
  cadence_mismatch_active: true,
  runtime_performance: {schema_version: 2, cadence_status: "intentional-repeat"},
});
process.stdout.write(JSON.stringify({accepted, stale, altered, inherited,
  unexpectedCadence, intentionalCadence,
  video: videoCadenceSummary(status)}));
"""
    result = subprocess.run(
        ["node"],
        input=mitigation + video + cadence + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "accepted": {
            "configVersion": 9,
            "kind": "disable-color-and-light-wrap",
            "patch": {
                "compositing": {
                    "color_correction": {"mode": "off"},
                    "light_wrap": 0,
                }
            },
        },
        "stale": None,
        "altered": None,
        "inherited": None,
        "unexpectedCadence": True,
        "intentionalCadence": False,
        "video": [
            "Source 60.0 fps · visual updates 15.0 fps · phase preserved; "
            "source frames skipped because visual updates are slower (90, 75.0%) "
            "· 2 visual updates "
            "reused the current decoded frame",
            "",
        ],
    }
