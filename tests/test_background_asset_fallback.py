"""Fail-closed startup and recovery for unavailable persisted backdrops."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import custback.pipeline as pipeline_mod
from custback.api.server import _StatusResponse
from custback.api.webui import WEBUI_HTML
from custback.config import AppConfig, RuntimeConfig
from custback.hub import FrameHub
from custback.pipeline import Pipeline


def _config(mode: str, path: Path) -> AppConfig:
    background = {"mode": mode, f"{mode}_path": str(path)}
    return AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 64, "height": 36, "fps": 30},
            "background": background,
            "segmentation": {"backend": "heuristic", "temporal_smoothing": 0.0},
            "output": {"backend": "null", "fps": 30},
            "api": {"enabled": False},
        }
    )


def _wait_for_status(hub: FrameHub, predicate, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = hub.stats_dict()
        if predicate(status):
            return status
        time.sleep(0.01)
    raise AssertionError("status did not reach the expected state")


@pytest.mark.parametrize(
    ("mode", "asset_kind"),
    (
        ("image", "missing"),
        ("image", "unreadable"),
        ("video", "missing"),
    ),
)
def test_unavailable_persisted_asset_starts_on_fixed_slate(
    tmp_path: Path,
    mode: str,
    asset_kind: str,
) -> None:
    suffix = ".png" if mode == "image" else ".mp4"
    asset = tmp_path / f"persisted-{asset_kind}{suffix}"
    if asset_kind == "unreadable":
        asset.write_bytes(b"not-an-image")
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(_config(mode, asset)), hub)

    pipeline.start(timeout=5.0)
    try:
        status = _wait_for_status(
            hub,
            lambda value: value["background_fallback_active"] is True,
        )
        output, _sequence = hub.output.latest()
        raw, _raw_sequence = hub.raw.latest()
        expected = Pipeline._privacy_slate((36, 64, 3))

        assert output is not None
        assert raw is not None
        assert np.array_equal(output, expected)
        assert not np.array_equal(output, raw)
        assert status["mode"] == mode
        assert status["background_fallback_reason"] == "asset-unavailable"
        assert status["remote_fallback_active"] is False
        assert pipeline.running
    finally:
        pipeline.stop()


def test_valid_live_background_patch_clears_asset_fallback(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    recovered = tmp_path / "recovered.png"
    image = np.full((36, 64, 3), (13, 71, 149), dtype=np.uint8)
    assert cv2.imwrite(str(recovered), image)
    hub = FrameHub()
    runtime = RuntimeConfig(_config("image", missing))
    pipeline = Pipeline(runtime, hub)

    pipeline.start(timeout=5.0)
    try:
        initial = _wait_for_status(
            hub,
            lambda value: value["background_fallback_active"] is True,
        )
        sequence = -1
        committed = pipeline.apply_config_patch(
            {"background": {"image_path": str(recovered)}},
            timeout=5.0,
        )
        assert committed.version == initial["config_version"] + 1

        recovered_status = _wait_for_status(
            hub,
            lambda value: (
                value["config_version"] == committed.version
                and value["background_fallback_active"] is False
            ),
        )
        deadline = time.monotonic() + 5.0
        output = None
        slate = Pipeline._privacy_slate((36, 64, 3))
        while time.monotonic() < deadline:
            candidate, sequence = hub.output.get(sequence, timeout=0.2)
            if candidate is not None and not np.array_equal(candidate, slate):
                output = candidate
                break

        assert output is not None
        assert recovered_status["background_fallback_reason"] == ""
        assert recovered_status["remote_fallback_active"] is False
        assert committed.config.background.mode == "image"
        assert committed.config.background.image_path == str(recovered)
        assert pipeline.running
    finally:
        pipeline.stop()


def test_video_first_decode_failure_is_startup_asset_unavailability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnreadableVideo:
        closed = False

        def frame(self, _width: int, _height: int) -> np.ndarray:
            raise RuntimeError("first frame cannot be decoded")

        def stats_dict(self) -> dict[str, int]:
            return {"background_video_decode_failures": 1}

        def close(self) -> None:
            self.closed = True

    provider = UnreadableVideo()
    monkeypatch.setattr(pipeline_mod, "create_backdrop", lambda *_args, **_kw: provider)
    cfg = _config("video", tmp_path / "persisted.mp4")
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)

    pipeline.start(timeout=5.0)
    try:
        status = _wait_for_status(
            hub,
            lambda value: value["background_fallback_active"] is True,
        )
        output, _timestamp = hub.output.latest()
        assert output is not None
        assert np.array_equal(output, Pipeline._privacy_slate((36, 64, 3)))
        assert status["background_fallback_reason"] == "asset-unavailable"
        assert status["background_video_lifetime_decode_failures"] == 1
        assert provider.closed is True
    finally:
        pipeline.stop()


def test_background_fallback_status_is_strict_and_documented() -> None:
    hub = FrameHub()
    default = hub.stats_dict()
    assert default["background_fallback_active"] is False
    assert default["background_fallback_reason"] == ""

    hub.update_stats(
        background_fallback_active=True,
        background_fallback_reason="asset-unavailable",
    )
    public = _StatusResponse.model_validate(
        {**hub.stats_dict(), "native_ring": "unsupported"}
    ).model_dump(mode="json", by_alias=True)
    assert public["background_fallback_active"] is True
    assert public["background_fallback_reason"] == "asset-unavailable"

    schema = _StatusResponse.model_json_schema()
    assert schema["properties"]["background_fallback_reason"]["enum"] == [
        "",
        "asset-unavailable",
    ]
    with pytest.raises(TypeError, match="must be boolean"):
        hub.update_stats(background_fallback_active=1)
    with pytest.raises(ValueError, match="not sanitized"):
        hub.update_stats(background_fallback_reason="/private/operator/image.png")
    with pytest.raises(ValueError, match="inconsistent"):
        hub.update_stats(
            background_fallback_active=False,
            background_fallback_reason="asset-unavailable",
        )

    assert '"Background fallback"' in WEBUI_HTML
    assert '"Background fallback reason"' in WEBUI_HTML
    assert '" · fixed privacy slate"' in WEBUI_HTML
