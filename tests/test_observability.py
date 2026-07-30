"""VIS-3.1 geometry/color status and transition-log contract tests."""

from __future__ import annotations

import threading
from types import SimpleNamespace

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
from custback.hub import FrameHub
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
    assert sum(message.startswith("color correction") for message in messages) == 4
    assert any("state=low-confidence" in message for message in messages)
    assert any("state=warming" in message for message in messages)
    assert any("state=scene-cut" in message for message in messages)
    assert "/private/operator" not in "\n".join(messages)


def test_status_model_and_openapi_schema_have_exact_hub_key_parity():
    hub_keys = set(FrameHub().stats_dict())
    model_keys = set(_StatusResponse.model_fields)
    schema_keys = set(_StatusResponse.model_json_schema()["properties"])
    assert model_keys == schema_keys == hub_keys | {"native_ring"}
    _StatusResponse.model_validate(
        {**FrameHub().stats_dict(), "native_ring": "unsupported"}
    )


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


def test_shutdown_summary_includes_geometry_and_correction_totals(caplog):
    hub = FrameHub()
    hub.update_stats(
        capture_generation=4,
        capture_geometry_transitions=3,
        background_geometry_transitions=2,
        color_correction_applied_frames=11,
        color_correction_bypassed_frames=5,
        color_correction_scene_cuts=2,
        color_correction_transitions=6,
    )
    with caplog.at_level("INFO", logger="custback"):
        _log_shutdown_summary(hub, "test", 0)
    assert "capture_generation=4 camera_geometry=3 background_geometry=2" in caplog.text
    assert "corrections_applied=11 corrections_bypassed=5" in caplog.text
    assert "color_scene_cuts=2 color_transitions=6" in caplog.text
