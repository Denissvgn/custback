"""Cumulative background-video telemetry across provider generations."""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import pytest

import custback.pipeline as pipeline_mod
from custback.__main__ import _log_shutdown_summary
from custback.api.server import _StatusResponse
from custback.config import AppConfig, RuntimeConfig
from custback.hub import FrameHub
from custback.pipeline import Pipeline, _BackgroundVideoLifetimeCounters


class _StatsProvider:
    def __init__(self, **stats: object) -> None:
        self.stats = stats
        self.closed = False

    def stats_dict(self) -> dict[str, object]:
        return dict(self.stats)

    def frame(self, width: int, height: int) -> np.ndarray:
        return np.zeros((height, width, 3), np.uint8)

    def close(self) -> None:
        self.closed = True


def _video_stats(
    displayed: int,
    skipped: int,
    reused: int,
    seeks: int,
    failures: int,
) -> dict[str, int]:
    return {
        "background_video_frames_displayed": displayed,
        "background_video_frames_skipped": skipped,
        "background_video_frames_reused": reused,
        "background_video_seek_count": seeks,
        "background_video_decode_failures": failures,
    }


def test_lifetime_accumulator_projects_active_and_retired_generations() -> None:
    counters = _BackgroundVideoLifetimeCounters()
    first = _StatsProvider(**_video_stats(20, 5, 2, 1, 0))
    second = _StatsProvider(**_video_stats(7, 3, 4, 2, 1))

    assert counters.project(first.stats_dict()) == {
        "background_video_lifetime_frames_displayed": 20,
        "background_video_lifetime_frames_skipped": 5,
        "background_video_lifetime_frames_reused": 2,
        "background_video_lifetime_seek_count": 1,
        "background_video_lifetime_decode_failures": 0,
    }
    counters.retire(first)
    assert counters.project({}) == {
        "background_video_lifetime_frames_displayed": 20,
        "background_video_lifetime_frames_skipped": 5,
        "background_video_lifetime_frames_reused": 2,
        "background_video_lifetime_seek_count": 1,
        "background_video_lifetime_decode_failures": 0,
    }
    assert counters.project(second.stats_dict()) == {
        "background_video_lifetime_frames_displayed": 27,
        "background_video_lifetime_frames_skipped": 8,
        "background_video_lifetime_frames_reused": 6,
        "background_video_lifetime_seek_count": 3,
        "background_video_lifetime_decode_failures": 1,
    }
    counters.retire(second)
    assert counters.project({})["background_video_lifetime_frames_skipped"] == 8


def test_lifetime_accumulator_ignores_malformed_and_saturates() -> None:
    counters = _BackgroundVideoLifetimeCounters()
    counters.retire(
        _StatsProvider(
            background_video_frames_displayed=-1,
            background_video_frames_skipped=True,
            background_video_frames_reused=1.5,
            background_video_seek_count=2**63 + 100,
            background_video_decode_failures=3,
        )
    )
    projected = counters.project({"background_video_decode_failures": 2**63 - 1})
    assert projected == {
        "background_video_lifetime_frames_displayed": 0,
        "background_video_lifetime_frames_skipped": 0,
        "background_video_lifetime_frames_reused": 0,
        "background_video_lifetime_seek_count": 2**63 - 1,
        "background_video_lifetime_decode_failures": 2**63 - 1,
    }


def test_lifetime_fields_are_strict_hub_and_openapi_status() -> None:
    hub = FrameHub()
    values = {
        "background_video_lifetime_frames_displayed": 27,
        "background_video_lifetime_frames_skipped": 8,
        "background_video_lifetime_frames_reused": 6,
        "background_video_lifetime_seek_count": 3,
        "background_video_lifetime_decode_failures": 1,
    }
    hub.update_stats(**values)
    body = hub.stats_dict()
    assert {name: body[name] for name in values} == values
    public = _StatusResponse.model_validate(
        {**body, "native_ring": "unsupported"}
    ).model_dump(mode="json", by_alias=True)
    assert {name: public[name] for name in values} == values

    schema = _StatusResponse.model_json_schema()
    for name in values:
        field_schema = schema["properties"][name]
        assert field_schema["minimum"] == 0
        assert field_schema["maximum"] == 2**63 - 1

    for invalid in (True, -1, 1.5, 2**63):
        with pytest.raises(ValueError, match="bounded nonnegative integer"):
            hub.update_stats(background_video_lifetime_frames_skipped=invalid)


def test_shutdown_summary_distinguishes_current_and_lifetime_video_counts(
    caplog,
) -> None:
    hub = FrameHub()
    hub.update_stats(
        background_video_frames_skipped=0,
        background_video_lifetime_frames_displayed=27,
        background_video_lifetime_frames_skipped=8,
        background_video_lifetime_frames_reused=6,
        background_video_lifetime_seek_count=3,
        background_video_lifetime_decode_failures=1,
    )
    with caplog.at_level("INFO", logger="custback.__main__"):
        _log_shutdown_summary(hub, "test", 0)
    assert "video_skips_current=0" in caplog.text
    assert "video_displayed_lifetime=27" in caplog.text
    assert "video_skips_lifetime=8" in caplog.text
    assert "video_reuses_lifetime=6" in caplog.text
    assert "video_seeks_lifetime=3" in caplog.text
    assert "video_failures_lifetime=1" in caplog.text


def test_live_provider_replacement_preserves_lifetime_counters(monkeypatch) -> None:
    real_create_backdrop = pipeline_mod.create_backdrop
    providers: list[_StatsProvider] = []
    generations = (
        _video_stats(20, 5, 2, 1, 0),
        _video_stats(7, 3, 4, 2, 1),
    )

    def create_backdrop(cfg, **kwargs):
        if cfg.mode != "video":
            return real_create_backdrop(cfg, **kwargs)
        provider = _StatsProvider(**generations[len(providers)])
        providers.append(provider)
        return provider

    monkeypatch.setattr(pipeline_mod, "create_backdrop", create_backdrop)
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 64, "height": 36, "fps": 30},
            "background": {"mode": "video", "video_path": "first.mp4"},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null", "fps": 30},
            "api": {"enabled": False},
        }
    )
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    pipeline.start(timeout=5.0)
    try:
        initial = hub.stats_dict()
        assert initial["background_video_frames_skipped"] == 5
        assert initial["background_video_lifetime_frames_skipped"] == 5

        first_commit = pipeline.apply_config_patch(
            {"background": {"mode": "color", "color": [0, 0, 0]}},
            timeout=5.0,
        )
        assert first_commit.version == 1
        deadline = time.monotonic() + 1.0
        while hub.stats_dict()["config_version"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        after_first = hub.stats_dict()
        assert after_first["background_video_frames_skipped"] == 0
        assert after_first["background_video_lifetime_frames_skipped"] == 5
        deadline = time.monotonic() + 1.0
        while not providers[0].closed and time.monotonic() < deadline:
            time.sleep(0.005)
        assert providers[0].closed is True

        second_commit = pipeline.apply_config_patch(
            {"background": {"mode": "video", "video_path": "second.mp4"}},
            timeout=5.0,
        )
        assert second_commit.version == 2
        deadline = time.monotonic() + 1.0
        while hub.stats_dict()["config_version"] != 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        active_second = hub.stats_dict()
        assert active_second["background_video_frames_skipped"] == 3
        assert active_second["background_video_lifetime_frames_skipped"] == 8

        third_commit = pipeline.apply_config_patch(
            {"background": {"mode": "color", "color": [1, 2, 3]}},
            timeout=5.0,
        )
        assert third_commit.version == 3
        deadline = time.monotonic() + 1.0
        while hub.stats_dict()["config_version"] != 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        final = cast(dict[str, Any], hub.stats_dict())
        assert final["background_video_frames_displayed"] == 0
        assert final["background_video_frames_skipped"] == 0
        assert final["background_video_lifetime_frames_displayed"] == 27
        assert final["background_video_lifetime_frames_skipped"] == 8
        assert final["background_video_lifetime_frames_reused"] == 6
        assert final["background_video_lifetime_seek_count"] == 3
        assert final["background_video_lifetime_decode_failures"] == 1
        deadline = time.monotonic() + 1.0
        while not providers[1].closed and time.monotonic() < deadline:
            time.sleep(0.005)
        assert providers[1].closed is True
    finally:
        pipeline.stop()
