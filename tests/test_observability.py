"""VIS-3.1 geometry/color status and transition-log contract tests."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from custback.__main__ import _log_shutdown_summary
from custback.api.server import _StatusResponse
from custback.capture import (
    CameraControlObservation,
    CameraControlReport,
    CaptureHealth,
)
from custback.color import (
    ColorReason,
    ColorTransform,
    HarmonizerPhase,
    HarmonizerSnapshot,
)
from custback.config import AppConfig, RuntimeConfig
from custback.geometry import plan_transform
from custback.hub import (
    TIMING_FIELD_NAMES,
    TIMING_SCHEMA_VERSION,
    FrameHub,
    PostBaseProvenance,
    Stats,
)
from custback.pipeline import (
    Pipeline,
    _Resources,
    _background_plan_stats,
    _camera_plan_stats,
    _color_stats,
)
from custback.preview import _status_overlay_lines


class _Capture:
    def __init__(self, health: CaptureHealth):
        self.health = health

    def health_snapshot(self) -> CaptureHealth:
        return self.health


class _Segmenter:
    device = "cpu"


class _Backdrop:
    def __init__(self, plan, *, path: str = "/private/operator/room.png"):
        self.plan = plan
        self.path = path

    def transform_plan(self, _width: int, _height: int):
        return self.plan


class _Output:
    width = 1280
    height = 720
    fps = 30
    fallback_active = False
    fallback_reason = ""


def _config(*, correction_mode: str = "auto", background_mode: str = "image"):
    background: dict[str, object] = {
        "mode": background_mode,
        "fit_mode": "cover",
    }
    if background_mode == "image":
        background["image_path"] = "/private/operator/room.png"
    return AppConfig.from_dict(
        {
            "camera": {
                "synthetic": True,
                "width": 640,
                "height": 480,
                "fps": 30,
                "fit_mode": "cover",
            },
            "background": background,
            "compositing": {"color_correction": {"mode": correction_mode}},
            "output": {"backend": "null", "width": 1280, "height": 720, "fps": 30},
            "api": {"enabled": False},
        }
    )


def _snapshot(
    phase: HarmonizerPhase,
    *,
    transform: ColorTransform = ColorTransform(),
    reason: ColorReason = ColorReason.OK,
    confidence: float = 0.82,
    reliable: bool = True,
) -> HarmonizerSnapshot:
    return HarmonizerSnapshot(
        transform=transform,
        phase=phase,
        reason=reason,
        confidence=confidence,
        reliable=reliable,
        last_timestamp_s=2.0,
        last_reliable_s=2.0 if reliable else None,
        low_confidence_since_s=None,
        fast_until_s=None,
        source_generation=4,
        signature=None,
    )


def _resources(
    cfg: AppConfig,
    health: CaptureHealth,
    *,
    backdrop_plan=None,
) -> _Resources:
    plan = backdrop_plan or plan_transform((1920, 1080), (1280, 720), fit="cover")
    return _Resources(
        cfg,
        3,
        _Capture(health),
        _Segmenter(),
        object(),
        _Backdrop(plan),
        _Output(),
    )


def test_output_and_matching_stats_publish_atomically(monkeypatch):
    hub = FrameHub()
    hub.update_stats(frames_out=1, mode="old")
    frame = np.full((2, 3, 3), 17, np.uint8)
    entered_put = threading.Event()
    release_put = threading.Event()
    reader_started = threading.Event()
    reader_done = threading.Event()
    observed: list[dict[str, object]] = []
    original_put = hub.output.put

    def blocked_put(value):
        entered_put.set()
        assert release_put.wait(1.0)
        original_put(value)

    monkeypatch.setattr(hub.output, "put", blocked_put)
    publisher = threading.Thread(
        target=lambda: hub.publish_output(
            frame,
            stats={"frames_out": 2, "mode": "new"},
        )
    )
    publisher.start()
    assert entered_put.wait(1.0)

    def read_status():
        reader_started.set()
        observed.append(hub.stats_dict())
        reader_done.set()

    reader = threading.Thread(target=read_status)
    reader.start()
    assert reader_started.wait(1.0)
    assert not reader_done.wait(0.05)
    release_put.set()
    publisher.join(1.0)
    reader.join(1.0)

    published, _timestamp = hub.output.latest()
    assert np.array_equal(published, frame)
    assert observed[0]["frames_out"] == 2
    assert observed[0]["mode"] == "new"


def test_hot_commit_identity_waits_for_matching_output_publication():
    cfg = _config()
    health = CaptureHealth(
        generation=1,
        delivered_width=640,
        delivered_height=480,
        oriented_width=640,
        oriented_height=480,
    )
    resources = _resources(cfg, health)
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    old_frame = np.full((720, 1280, 3), 17, np.uint8)
    new_frame = np.full((720, 1280, 3), 29, np.uint8)
    hub.publish_output(
        old_frame,
        stats={"mode": "image", "config_version": 3},
    )

    candidate = cfg.patched({"background": {"anchor_x": 0.25}})
    resources.cfg = candidate
    resources.version = 4
    pipeline._post_install_activation(resources, cfg)

    # PATCH acknowledgement/commit may precede rendering, but public identity
    # must continue to describe the output slot until the new frame is ready.
    before = hub.stats_dict()
    published, _timestamp = hub.output.latest()
    assert before["config_version"] == 3
    assert np.array_equal(published, old_frame)

    matching = pipeline._identity_stats(resources, capture_health=health)
    hub.publish_output(new_frame, stats=matching)
    after = hub.stats_dict()
    published, _timestamp = hub.output.latest()
    assert after["config_version"] == 4
    assert np.array_equal(published, new_frame)


def test_geometry_and_color_status_express_matching_transform():
    report = CameraControlReport(
        backend_family="v4l2",
        generation=7,
        properties=(CameraControlObservation("gain", "reported", 2.5),),
    )
    health = CaptureHealth(
        generation=7,
        geometry_generation=3,
        backend="V4L2",
        width=1920,
        height=1080,
        delivered_width=640,
        delivered_height=480,
        oriented_width=640,
        oriented_height=480,
        normalized_width=1280,
        normalized_height=720,
        geometry_transitions=3,
        camera_controls=report,
    )
    cfg = _config()
    resources = _resources(cfg, health)
    transform = ColorTransform(
        exposure_ev=0.35,
        wb_gains=(1.04, 1.0, 0.96),
    )
    color = _color_stats(
        cfg,
        _snapshot(HarmonizerPhase.ACTIVE, transform=transform),
        applied_transform=transform,
    )
    hub = FrameHub()
    status = Pipeline(RuntimeConfig(cfg), hub)._identity_stats(
        resources,
        capture_health=health,
        color_status=color,
    )
    status["output_effective_fps"] = 29.7
    hub.update_stats(**status)
    public = hub.stats_dict()

    # Negotiated-mode dimensions retain their existing meaning and are not
    # silently relabelled as delivered or output geometry.
    assert (public["capture_width"], public["capture_height"]) == (1920, 1080)
    assert (public["capture_delivered_width"], public["capture_delivered_height"]) == (
        640,
        480,
    )
    assert (public["output_width"], public["output_height"]) == (1280, 720)
    assert public["capture_generation"] == 7
    assert public["camera_fit"] == "cover"
    assert public["camera_scale_x"] == public["camera_scale_y"] == 2.0
    assert (
        public["camera_crop_left"],
        public["camera_crop_top"],
        public["camera_crop_right"],
        public["camera_crop_bottom"],
    ) == (0, 120, 1280, 840)
    assert public["camera_pad_left"] == public["camera_pad_right"] == 0
    assert public["background_fit"] == "cover"
    assert public["output_effective_fps"] == 29.7
    assert public["color_correction_state"] == "active"
    assert public["color_correction_active"] is True
    assert public["color_correction_effective_mode"] == "exposure-white-balance"
    assert public["color_correction_reason"] == "ok"
    assert public["color_correction_exposure_ev"] == 0.35
    assert public["color_correction_confidence"] == 0.82
    assert public["color_correction_wb_active"] is True
    assert public["color_input_assumption"] == (
        "display-referred-srgb-bt709-full-range"
    )
    assert public["camera_controls"]["properties"]["gain"]["value"] == 2.5

    # Neither the real frozen report nor a returned nested dict can mutate the
    # hub's internal status after publication.
    public["camera_controls"]["properties"]["gain"]["value"] = 99.0
    assert hub.stats_dict()["camera_controls"]["properties"]["gain"]["value"] == 2.5


@pytest.mark.parametrize(
    (
        "cfg",
        "snapshot",
        "expected_state",
        "expected_effective",
        "expected_warming",
        "expected_stale",
    ),
    [
        (
            _config(correction_mode="off"),
            None,
            "disabled",
            "off",
            False,
            False,
        ),
        (
            _config(background_mode="color"),
            None,
            "mode-excluded",
            "bypass",
            False,
            False,
        ),
        (
            _config(),
            _snapshot(HarmonizerPhase.WARMING),
            "warming",
            "identity",
            True,
            False,
        ),
        (
            _config(),
            _snapshot(
                HarmonizerPhase.FROZEN,
                transform=ColorTransform(exposure_ev=0.2),
                reason=ColorReason.INSUFFICIENT_MASK,
                confidence=0.1,
            ),
            "low-confidence",
            "exposure",
            False,
            False,
        ),
        (
            _config(),
            _snapshot(
                HarmonizerPhase.STALE_DECAY,
                transform=ColorTransform(exposure_ev=0.1),
                reason=ColorReason.CLIPPED,
                confidence=0.2,
            ),
            "stale-decay",
            "exposure",
            False,
            True,
        ),
        (
            _config(),
            _snapshot(
                HarmonizerPhase.ACTIVE,
                transform=ColorTransform(exposure_ev=0.35),
            ),
            "active",
            "exposure",
            False,
            False,
        ),
    ],
)
def test_correction_status_distinguishes_every_operator_state(
    cfg,
    snapshot,
    expected_state,
    expected_effective,
    expected_warming,
    expected_stale,
):
    status = _color_stats(cfg, snapshot)
    assert status["color_correction_state"] == expected_state
    assert status["color_correction_effective_mode"] == expected_effective
    assert status["color_correction_warming"] is expected_warming
    assert status["color_correction_stale"] is expected_stale


def test_geometry_and_color_logs_are_transition_only_and_path_free(caplog):
    cfg = _config()
    health = CaptureHealth(
        generation=1,
        delivered_width=640,
        delivered_height=480,
        oriented_width=640,
        oriented_height=480,
    )
    resources = _resources(cfg, health)
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    camera = _camera_plan_stats(cfg, resources.canvas_size, health)
    background = _background_plan_stats(resources)
    active = _color_stats(
        cfg,
        _snapshot(
            HarmonizerPhase.ACTIVE,
            transform=ColorTransform(exposure_ev=0.2),
        ),
    )

    with caplog.at_level("INFO", logger="custback.pipeline"):
        pipeline._log_geometry_transitions(resources, health, camera, background)
        pipeline._log_geometry_transitions(resources, health, camera, background)
        pipeline._log_color_transition(resources, active)
        pipeline._log_color_transition(resources, active)

        reconnected = SimpleNamespace(**{**health.__dict__, "generation": 2})
        pipeline._log_geometry_transitions(
            resources,
            reconnected,
            camera,
            {**background, "background_fit": "contain"},
        )
        pipeline._log_color_transition(
            resources,
            _color_stats(
                cfg,
                _snapshot(
                    HarmonizerPhase.FROZEN,
                    transform=ColorTransform(exposure_ev=0.2),
                    reason=ColorReason.INSUFFICIENT_MASK,
                    confidence=0.1,
                ),
            ),
        )
        pipeline._log_color_transition(
            resources,
            _color_stats(cfg, _snapshot(HarmonizerPhase.WARMING)),
        )
        pipeline._log_color_transition(
            resources,
            _color_stats(cfg, _snapshot(HarmonizerPhase.SCENE_CUT)),
        )

    messages = [record.getMessage() for record in caplog.records]
    assert sum(message.startswith("camera transform") for message in messages) == 2
    assert sum(message.startswith("background transform") for message in messages) == 2
    phase_messages = [
        message for message in messages if message.startswith("color correction phase")
    ]
    reason_messages = [
        message
        for message in messages
        if message.startswith("color correction estimator reason")
    ]
    assert len(phase_messages) == 4
    assert len(reason_messages) == 1
    assert all("reason=" not in message for message in phase_messages)
    assert all("state=" not in message for message in reason_messages)
    assert any("state=low-confidence" in message for message in messages)
    assert any("state=warming" in message for message in messages)
    assert any("state=scene-cut" in message for message in messages)
    assert resources.color_correction_transitions == 4
    assert resources.color_correction_reason_transitions == 1
    assert "/private/operator" not in "\n".join(messages)
    assert not any(
        message.startswith("camera geometry mismatch") for message in messages
    )
    assert all(
        record.levelname == "INFO"
        for record in caplog.records
        if record.getMessage().startswith("camera transform")
    )


def test_geometry_warns_only_for_mismatch_or_material_stretch_distortion(caplog):
    cfg = _config().patched({"camera": {"fit_mode": "stretch"}})
    health = CaptureHealth(
        generation=1,
        delivered_width=640,
        delivered_height=480,
        oriented_width=640,
        oriented_height=480,
    )
    resources = _resources(cfg, health)
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())

    with caplog.at_level("INFO", logger="custback.pipeline"):
        pipeline._log_geometry_transitions(
            resources,
            health,
            _camera_plan_stats(cfg, resources.canvas_size, health),
            _background_plan_stats(resources),
        )

    transform = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("camera transform")
    )
    mismatch = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("camera geometry mismatch")
    )
    assert transform.levelname == "INFO"
    assert mismatch.levelname == "WARNING"
    assert "negotiation=False" in mismatch.getMessage()
    assert "stretch_aspect_distortion_pct=25.000" in mismatch.getMessage()


def test_color_estimator_reason_changes_are_independently_debounced(caplog):
    cfg = _config()
    resources = _resources(cfg, CaptureHealth(generation=1))
    pipeline = Pipeline(RuntimeConfig(cfg), FrameHub())
    ok = _color_stats(cfg, _snapshot(HarmonizerPhase.ACTIVE))
    low_confidence = _color_stats(
        cfg,
        _snapshot(
            HarmonizerPhase.ACTIVE,
            reason=ColorReason.INSUFFICIENT_MASK,
            confidence=0.1,
        ),
    )

    with caplog.at_level("INFO", logger="custback.pipeline"):
        pipeline._log_color_transition(resources, ok, now_s=0.0)
        pipeline._log_color_transition(resources, low_confidence, now_s=0.1)
        pipeline._log_color_transition(resources, low_confidence, now_s=0.4)
        pipeline._log_color_transition(resources, ok, now_s=0.45)
        pipeline._log_color_transition(resources, low_confidence, now_s=1.0)
        pipeline._log_color_transition(resources, low_confidence, now_s=1.49)
        pipeline._log_color_transition(resources, low_confidence, now_s=1.5)

    reason_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("color correction estimator reason")
    ]
    phase_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("color correction phase")
    ]
    assert reason_messages == [
        "color correction estimator reason=ok confidence=0.820",
        "color correction estimator reason=insufficient-mask confidence=0.100",
    ]
    assert len(phase_messages) == 1
    assert resources.color_correction_transitions == 1
    assert resources.color_correction_reason_transitions == 2


def test_color_clamp_status_is_false_for_bypass_or_application_failure():
    cfg = _config()
    snapshot = _snapshot(HarmonizerPhase.ACTIVE)

    active = _color_stats(
        cfg,
        snapshot,
        exposure_clamped=True,
        white_balance_clamped=True,
    )
    failed = _color_stats(
        cfg,
        snapshot,
        application_failed=True,
        exposure_clamped=True,
        white_balance_clamped=True,
    )
    disabled = _color_stats(
        _config(correction_mode="off"),
        snapshot,
        exposure_clamped=True,
        white_balance_clamped=True,
    )

    assert active["color_correction_exposure_clamped"] is True
    assert active["color_correction_wb_clamped"] is True
    assert failed["color_correction_exposure_clamped"] is False
    assert failed["color_correction_wb_clamped"] is False
    assert disabled["color_correction_exposure_clamped"] is False
    assert disabled["color_correction_wb_clamped"] is False


def test_status_model_and_openapi_schema_have_exact_hub_key_parity():
    hub_status = FrameHub().stats_dict()
    hub_keys = set(hub_status)
    model_keys = set(_StatusResponse.model_fields)
    model_schema = _StatusResponse.model_json_schema()
    schema_keys = set(model_schema["properties"])
    assert {
        "capture_sequence",
        "capture_sequence_gap_count",
        "capture_missing_input_count",
        "matte_reset_count",
        "matte_last_reset_reason",
    } <= hub_keys
    assert model_keys == schema_keys == hub_keys | {"native_ring"}
    assert set(model_schema["required"]) == schema_keys

    timing_schema = model_schema["$defs"]["_TimingFieldsResponse"]
    timing_keys = set(TIMING_FIELD_NAMES)
    assert set(hub_status["timing_ms"]) == timing_keys
    assert set(timing_schema["properties"]) == timing_keys
    assert set(timing_schema["required"]) == timing_keys
    assert timing_schema["additionalProperties"] is False

    extensions = hub_status["extensions"]
    assert set(extensions) == {"post_base"}
    post_base = extensions["post_base"]
    assert set(post_base) == {"schema", "version", "stages"}
    assert post_base == {
        "schema": "custback.post-base-cadence",
        "version": 1,
        "stages": [],
    }
    extensions_schema = model_schema["$defs"]["_StatusExtensionsResponse"]
    assert set(extensions_schema["properties"]) == {"post_base"}
    assert set(extensions_schema["required"]) == {"post_base"}
    assert extensions_schema["additionalProperties"] is False
    post_base_schema = model_schema["$defs"]["_PostBaseResponse"]
    assert set(post_base_schema["properties"]) == {"schema", "version", "stages"}
    assert set(post_base_schema["required"]) == {"schema", "version", "stages"}
    assert post_base_schema["properties"]["stages"]["maxItems"] == 8
    assert post_base_schema["additionalProperties"] is False
    stage_fields = {
        "namespace",
        "update_count",
        "update_fps",
        "base_reuse_update_count",
        "base_reuse_update_fps",
    }
    stage_schema = model_schema["$defs"]["_PostBaseStageResponse"]
    assert set(stage_schema["properties"]) == stage_fields
    assert set(stage_schema["required"]) == stage_fields
    assert stage_schema["additionalProperties"] is False
    _StatusResponse.model_validate({**hub_status, "native_ring": "unsupported"})


def test_color_clamp_telemetry_is_strictly_validated_and_projected():
    hub = FrameHub()
    hub.update_stats(
        color_correction_exposure_clamped=True,
        color_correction_exposure_clamp_count=7,
        color_correction_exposure_clamp_time_s=1.23456,
        color_correction_wb_clamped=False,
        color_correction_wb_clamp_count=3,
        color_correction_wb_clamp_time_s=0.75,
        color_correction_reason_transitions=5,
    )

    status = hub.stats_dict()
    assert status["color_correction_exposure_clamped"] is True
    assert status["color_correction_exposure_clamp_count"] == 7
    assert status["color_correction_exposure_clamp_time_s"] == 1.235
    assert status["color_correction_wb_clamped"] is False
    assert status["color_correction_wb_clamp_count"] == 3
    assert status["color_correction_wb_clamp_time_s"] == 0.75
    assert status["color_correction_reason_transitions"] == 5

    with pytest.raises(TypeError, match="must be boolean"):
        hub.update_stats(color_correction_exposure_clamped=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        hub.update_stats(color_correction_wb_clamp_count=-1)
    with pytest.raises(ValueError, match="finite duration"):
        hub.update_stats(color_correction_exposure_clamp_time_s=float("nan"))


def test_timing_registry_validates_exact_keys_bounds_and_copies_values():
    hub = FrameHub()
    timing = dict.fromkeys(TIMING_FIELD_NAMES)
    timing["capture.read"] = 12.34567
    timing["pipeline.new_frame_serialized_loop"] = 3_600_000

    hub.update_stats(
        timing_schema_version=TIMING_SCHEMA_VERSION,
        timing_ms=timing,
    )
    timing["capture.read"] = 999.0
    public = hub.stats_dict()
    assert public["timing_schema_version"] == TIMING_SCHEMA_VERSION
    assert public["timing_ms"]["capture.read"] == 12.346
    assert public["timing_ms"]["pipeline.new_frame_serialized_loop"] == 3_600_000.0

    public["timing_ms"]["capture.read"] = 777.0
    assert hub.stats_dict()["timing_ms"]["capture.read"] == 12.346

    missing = dict.fromkeys(TIMING_FIELD_NAMES)
    missing.pop("capture.read")
    with pytest.raises(ValueError, match="exactly"):
        hub.update_stats(timing_ms=missing)
    extra: dict[str, object] = {str(key): None for key in TIMING_FIELD_NAMES}
    extra["private.stage"] = 1.0
    with pytest.raises(ValueError, match="exactly"):
        hub.update_stats(timing_ms=extra)
    with pytest.raises(TypeError, match="mapping"):
        hub.update_stats(timing_ms=[])
    with pytest.raises(ValueError, match="timing_schema_version"):
        hub.update_stats(timing_schema_version=TIMING_SCHEMA_VERSION + 1)


@pytest.mark.parametrize(
    "sample",
    [
        True,
        -0.001,
        float("nan"),
        float("inf"),
        3_600_000.001,
        "12.3",
    ],
)
def test_timing_registry_rejects_unbounded_or_non_numeric_samples(sample):
    hub = FrameHub()
    timing = dict.fromkeys(TIMING_FIELD_NAMES)
    timing["capture.read"] = sample
    with pytest.raises(ValueError, match="bounded nonnegative finite duration"):
        hub.update_stats(timing_ms=timing)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("update_count", True),
        ("update_count", -1),
        ("update_count", 2**63),
        ("base_reuse_update_count", 1.5),
        ("update_fps", float("nan")),
        ("update_fps", 10_000.001),
        ("base_reuse_update_fps", True),
    ],
)
def test_post_base_provenance_rejects_invalid_typed_values(field, value):
    with pytest.raises(ValueError, match=f"post-base {field}"):
        PostBaseProvenance(**{field: value})


@pytest.mark.parametrize(
    "namespace",
    ["", "Upper", "bad_name", "a" * 33, 3],
)
def test_post_base_provenance_rejects_invalid_namespaces(namespace):
    with pytest.raises(ValueError, match="namespace"):
        FrameHub().publish_post_base_provenance(namespace, PostBaseProvenance())


def test_post_base_stage_limit_and_core_cadence_isolation():
    hub = FrameHub()
    hub.update_stats(
        segmentation_update_count=11,
        base_composite_update_count=10,
        base_composite_reuse_count=4,
        exact_final_output_repeat_count=3,
        output_send_count=14,
    )
    core_fields = {
        "segmentation_update_count",
        "base_composite_update_count",
        "base_composite_reuse_count",
        "exact_final_output_repeat_count",
        "output_send_count",
    }
    core_before = {key: hub.stats_dict()[key] for key in core_fields}

    for index in range(8):
        hub.publish_post_base_provenance(
            f"stage-{index}",
            PostBaseProvenance(
                update_count=index,
                update_fps=index + 0.12345,
                base_reuse_update_count=index // 2,
                base_reuse_update_fps=index / 3,
            ),
        )
    hub.publish_post_base_provenance(
        "stage-0",
        PostBaseProvenance(update_count=99, update_fps=1.25),
    )
    with pytest.raises(ValueError, match="stage limit"):
        hub.publish_post_base_provenance("stage-8", PostBaseProvenance())

    with pytest.raises(TypeError, match="PostBaseProvenance"):
        hub.publish_post_base_provenance(
            "reaction",
            cast(PostBaseProvenance, {"output_send_count": 999}),
        )
    with pytest.raises(KeyError, match="extensions"):
        hub.update_stats(
            extensions={
                "post_base": {
                    "stages": [{"namespace": "reaction", "output_send_count": 999}]
                }
            }
        )

    public = hub.stats_dict()
    assert {key: public[key] for key in core_fields} == core_before
    stages = public["extensions"]["post_base"]["stages"]
    assert [stage["namespace"] for stage in stages] == [
        f"stage-{index}" for index in range(8)
    ]
    assert stages[0] == {
        "namespace": "stage-0",
        "update_count": 99,
        "update_fps": 1.25,
        "base_reuse_update_count": 0,
        "base_reuse_update_fps": 0.0,
    }
    stages[0]["update_count"] = 1_000
    assert (
        hub.stats_dict()["extensions"]["post_base"]["stages"][0]["update_count"] == 99
    )


def test_temporal_status_defaults_and_assigned_values_round_trip():
    defaults = {
        "capture_sequence": 0,
        "capture_sequence_gap_count": 0,
        "capture_missing_input_count": 0,
        "matte_reset_count": 0,
        "matte_last_reset_reason": "",
    }
    stats = Stats()
    assert {key: getattr(stats, key) for key in defaults} == defaults

    hub = FrameHub()
    assert {key: hub.stats_dict()[key] for key in defaults} == defaults

    assigned = {
        "capture_sequence": 17,
        "capture_sequence_gap_count": 2,
        "capture_missing_input_count": 5,
        "matte_reset_count": 3,
        "matte_last_reset_reason": "timestamp-gap",
    }
    hub.update_stats(**assigned)
    public = hub.stats_dict()
    assert {key: public[key] for key in assigned} == assigned


def test_spatial_edge_refinement_status_defaults_and_values_round_trip():
    defaults = {
        "effective_edge_refinement_mode": "off",
        "effective_edge_refinement_radius_px": 0,
    }
    stats = Stats()
    assert {key: getattr(stats, key) for key in defaults} == defaults

    hub = FrameHub()
    assert {key: hub.stats_dict()[key] for key in defaults} == defaults

    assigned = {
        "effective_edge_refinement_mode": "stable_guided",
        "effective_edge_refinement_radius_px": 12,
    }
    hub.update_stats(**assigned)
    public = hub.stats_dict()
    assert {key: public[key] for key in assigned} == assigned


def test_preview_hud_renders_geometry_color_and_confidence_warnings():
    status, warnings = _status_overlay_lines(
        {
            "capture_delivered_width": 640,
            "capture_delivered_height": 480,
            "output_width": 1280,
            "output_height": 720,
            "camera_fit": "cover",
            "color_correction_state": "low-confidence",
            "color_correction_effective_mode": "exposure",
            "color_correction_reason": "insufficient-mask",
            "color_correction_confidence": 0.21,
            "color_correction_exposure_ev": 0.35,
        }
    )
    rendered = "\n".join(status)
    assert "GEOMETRY 640x480 -> cover -> 1280x720" in rendered
    assert "COLOR low-confidence/exposure  +0.35 EV" in rendered
    assert warnings == ["COLOR LOW CONFIDENCE: insufficient-mask"]

    _status, warnings = _status_overlay_lines(
        {
            "color_correction_state": "stale-decay",
            "color_correction_effective_mode": "exposure",
            "color_correction_reason": "clipped",
        }
    )
    assert warnings == ["COLOR CORRECTION STALE: decaying to identity"]


def test_shutdown_summary_includes_geometry_correction_and_cadence_totals(caplog):
    hub = FrameHub()
    hub.update_stats(
        capture_generation=4,
        capture_geometry_transitions=3,
        background_geometry_transitions=2,
        color_correction_applied_frames=11,
        color_correction_bypassed_frames=5,
        color_correction_scene_cuts=2,
        color_correction_transitions=6,
        base_composite_update_count=541,
        segmentation_update_count=541,
        output_send_count=1105,
        base_composite_reuse_count=564,
        base_composite_reuse_ratio=564 / 1105,
        exact_final_output_repeat_count=563,
        capture_sequence_gap_count=3,
        capture_missing_input_count=7,
        capture_dropped_frames=9,
        processing_deadline_misses=11,
        serialized_new_frame_deadline_misses=13,
        output_sink_pacing_events=17,
        output_sink_recovery_events=19,
        application_pacing_events=23,
        output_schedule_late_events=29,
        matte_reset_count=31,
        matte_last_reset_reason="capture-gap",
    )
    with caplog.at_level("INFO", logger="custback"):
        _log_shutdown_summary(hub, "test", 0)
    assert "capture_generation=4 camera_geometry=3 background_geometry=2" in caplog.text
    assert "corrections_applied=11 corrections_bypassed=5" in caplog.text
    assert "color_scene_cuts=2 color_transitions=6" in caplog.text
    assert (
        "unique_updates=541 segmentation_updates=541 output_sends=1105" in caplog.text
    )
    assert "safe_base_reuses=564 safe_base_reuse_pct=51.0" in caplog.text
    assert "exact_final_repeats=563 capture_gaps=3 capture_missing=7" in caplog.text
    assert "capture_slot_overwrites=9 processing_deadline_misses=11" in caplog.text
    assert "serialized_deadline_misses=13 sink_pacing_events=17" in caplog.text
    assert "sink_recovery_events=19 application_pacing_events=23" in caplog.text
    assert "schedule_late_events=29 matte_resets=31" in caplog.text
    assert "matte_last_reset=capture-gap" in caplog.text


def test_shutdown_summary_reports_backend_quality_and_effective_policy(caplog):
    hub = FrameHub()
    selection = {
        "schema": "custback.backend-selection",
        "version": 1,
        "requested_backend": "auto",
        "selected_backend": "rvm",
        "quality_tier": "matting",
        "selection_mode": "automatic",
        "fallback_active": False,
        "fallback_category": "none",
        "fallback_reason": "",
        "guidance": "",
        "active_device": "cuda",
        "active_provider": "cuda",
        "attempts": [
            {
                "backend": "rvm",
                "quality_tier": "matting",
                "preparation_result": "ready",
                "activation_result": "selected",
                "reason_category": "none",
                "reason": "",
                "guidance": "",
            },
            {
                "backend": "mediapipe",
                "quality_tier": "segmentation",
                "preparation_result": "ready",
                "activation_result": "not-attempted",
                "reason_category": "none",
                "reason": "",
                "guidance": "",
            },
        ],
    }
    matte_policy = hub.stats_dict()["matte_policy"]
    matte_policy["selected_backend_kind"] = "true_alpha_recurrent"
    matte_policy["backend_kind"] = "true_alpha_recurrent"
    matte_policy["blend_space"] = "linear_srgb"
    matte_policy["effective"]["rvm_downsample_ratio"] = 0.4
    matte_policy["effective"]["raw_alpha_mode"] = "native_soft_alpha"
    matte_policy["effective"]["residual_temporal_mode"] = "model_only"
    matte_policy["effective"]["light_wrap"] = 0.1
    hub.update_stats(
        segmentation_selection=selection,
        matte_policy=matte_policy,
    )

    with caplog.at_level("INFO", logger="custback"):
        _log_shutdown_summary(hub, "test", 0)

    assert "backend=auto->rvm tier=matting device=cuda provider=cuda" in caplog.text
    assert "backend_fallback=False backend_fallback_category=none" in caplog.text
    assert "rvm_ratio=0.4 alpha_policy=native_soft_alpha" in caplog.text
    assert "temporal_policy=model_only light_wrap=0.1" in caplog.text
    assert "blend_space=linear_srgb" in caplog.text
