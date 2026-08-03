"""End-to-end canonical-canvas regressions across pipeline and API boundaries."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

import numpy as np
import pytest

pytest.importorskip("fastapi")
cv2 = pytest.importorskip("cv2")

import starlette

if int(starlette.__version__.split(".", 1)[0]) >= 1:
    from httpx2 import ASGITransport as _ASGITransport
    from httpx2 import AsyncClient as _AsyncClient
else:  # Starlette < 1 uses the original httpx client contract.
    from httpx import ASGITransport as _ASGITransport
    from httpx import AsyncClient as _AsyncClient

import custback.pipeline as pipeline_mod
from custback.api.security import SecurityPolicy
from custback.api.server import create_app
from custback.capture import CapturedFrame, CaptureHealth
from custback.config import AppConfig, ConfigState, RuntimeConfig
from custback.hub import FrameHub
from custback.pipeline import ActivationError, Pipeline, _Resources


ACQUISITION_SIZE = (64, 48)
CANVAS_SIZE = (80, 45)
CANVAS_SHAPE = (CANVAS_SIZE[1], CANVAS_SIZE[0], 3)
OUTPUT_FPS = 47
TOKEN = "canonical-canvas-test-token-at-least-32-characters"
RENDERER_TOKEN = "canonical-renderer-test-token-at-least-32-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
RENDERER_AUTH = {"Authorization": f"Bearer {RENDERER_TOKEN}"}
ORIGIN = "http://testserver"


def _config(
    mode: str = "passthrough",
    *,
    segmentation: str = "none",
    camera_fps: int = 60,
    output_fps: int = OUTPUT_FPS,
) -> AppConfig:
    return AppConfig.from_dict(
        {
            "camera": {
                "synthetic": True,
                "width": ACQUISITION_SIZE[0],
                "height": ACQUISITION_SIZE[1],
                "fps": camera_fps,
                "fit_mode": "cover",
            },
            "background": {
                "mode": mode,
                "color": [173, 31, 11],
                "blur_strength": 15,
                "remote_fallback_mode": "color",
            },
            "segmentation": {
                "backend": segmentation,
                "temporal_smoothing": 0.0,
                "edge_refine": False,
                "mask_blur": 0,
            },
            "output": {
                "backend": "null",
                "width": CANVAS_SIZE[0],
                "height": CANVAS_SIZE[1],
                "fps": output_fps,
            },
            "api": {
                "enabled": False,
                "ws_max_bytes": 256 * 1024,
            },
        }
    )


def _pattern(
    size: tuple[int, int] = CANVAS_SIZE,
    *,
    offset: int = 0,
) -> np.ndarray:
    width, height = size
    yy, xx = np.indices((height, width), dtype=np.uint16)
    return np.stack(
        (
            (13 * xx + 3 * yy + offset) % 256,
            (5 * xx + 17 * yy + offset) % 256,
            (29 * xx + 7 * yy + offset) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _wait_until(predicate, *, timeout: float = 3.0, message: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {message}")


def _wait_for_output(hub: FrameHub, *, timeout: float = 3.0) -> np.ndarray:
    deadline = time.monotonic() + timeout
    sequence = -1
    while time.monotonic() < deadline:
        frame, sequence = hub.output.get(sequence, 0.1)
        if frame is not None:
            return frame
    raise AssertionError("pipeline produced no output frame")


class _ScriptedCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        repeat_last: bool = False,
        normalized_size: tuple[int, int] = CANVAS_SIZE,
    ) -> None:
        self._frames = deque(frames)
        self._last = frames[-1] if frames else None
        self._repeat_last = repeat_last
        self._normalized_size = normalized_size
        self.frames_read = 0
        self.closed = False

    def read(self) -> CapturedFrame | None:
        if self.closed:
            return None
        if self._frames:
            frame = self._frames.popleft()
        elif self._repeat_last:
            frame = self._last
        else:
            return None
        self.frames_read += 1
        assert frame is not None
        height, width = frame.shape[:2]
        return CapturedFrame(
            pixels=frame,
            sequence=self.frames_read,
            captured_at_ns=self.frames_read * 1_000_000,
            generation=1,
            geometry_generation=1,
            content_rect=(0, 0, width, height),
        )

    def health_snapshot(self) -> CaptureHealth:
        return CaptureHealth(
            sequence=self.frames_read,
            captured_monotonic_ns=(
                self.frames_read * 1_000_000 if self.frames_read else None
            ),
            generation=1 if self.frames_read else 0,
            geometry_generation=1 if self.frames_read else 0,
            content_rect=(
                (0, 0, self._normalized_size[0], self._normalized_size[1])
                if self.frames_read
                else None
            ),
            backend="scripted",
            width=ACQUISITION_SIZE[0],
            height=ACQUISITION_SIZE[1],
            delivered_width=ACQUISITION_SIZE[0],
            delivered_height=ACQUISITION_SIZE[1],
            oriented_width=ACQUISITION_SIZE[0],
            oriented_height=ACQUISITION_SIZE[1],
            normalized_width=self._normalized_size[0],
            normalized_height=self._normalized_size[1],
            geometry_transitions=1 if self.frames_read else 0,
            fps_reported=60.0,
            frames_read=self.frames_read,
        )

    def close(self) -> None:
        self.closed = True


class _RecordingOutput:
    paces = False
    fallback_active = False
    fallback_reason = ""

    def __init__(
        self,
        width: int = CANVAS_SIZE[0],
        height: int = CANVAS_SIZE[1],
        fps: int = OUTPUT_FPS,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self.closed = False

    def send(self, frame: np.ndarray) -> None:
        with self._lock:
            self.frames.append(frame)

    def snapshots(self) -> list[np.ndarray]:
        with self._lock:
            return list(self.frames)

    def close(self) -> None:
        self.closed = True


class _TrackingSegmenter:
    device = "cpu"
    last_foreground = None

    def __init__(self, mask_value: float = 0.0) -> None:
        self.mask_value = mask_value
        self.frames: list[np.ndarray] = []
        self.closed = False

    def segment(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return np.full(frame.shape[:2], self.mask_value, dtype=np.float32)

    def close(self) -> None:
        self.closed = True


class _TrackingRefiner:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def refine(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return mask


def _install_pipeline_fakes(
    monkeypatch: pytest.MonkeyPatch,
    capture: _ScriptedCapture,
    output: _RecordingOutput,
) -> None:
    monkeypatch.setattr(
        pipeline_mod,
        "open_capture",
        lambda _cfg, _canvas_size: capture,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "open_output",
        lambda _cfg, _width, _height: output,
    )


def test_open_resources_separates_acquisition_request_from_canonical_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    capture = _ScriptedCapture([_pattern()])
    segmenter = _TrackingSegmenter()
    refiner = _TrackingRefiner()
    output = _RecordingOutput()
    calls: dict[str, Any] = {}

    def open_capture(camera_cfg, canvas_size):
        calls["capture"] = (camera_cfg.width, camera_cfg.height, canvas_size)
        return capture

    def open_output(output_cfg, width, height):
        calls["output"] = (width, height, output_cfg.fps)
        return output

    monkeypatch.setattr(pipeline_mod, "open_capture", open_capture)
    monkeypatch.setattr(pipeline_mod, "create_segmenter", lambda *_a, **_k: segmenter)
    monkeypatch.setattr(pipeline_mod, "refiner_for", lambda *_a, **_k: refiner)
    monkeypatch.setattr(pipeline_mod, "_build_backdrop", lambda _cfg: None)
    monkeypatch.setattr(pipeline_mod, "open_output", open_output)

    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    resources = pipeline._open_resources(ConfigState(cfg, 0))
    try:
        assert calls["capture"] == (*ACQUISITION_SIZE, CANVAS_SIZE)
        assert calls["output"] == (*CANVAS_SIZE, OUTPUT_FPS)
        assert resources.canvas_size == CANVAS_SIZE
        assert resources.canvas_shape == CANVAS_SHAPE
        assert hub._canvas_size == CANVAS_SIZE
    finally:
        resources.close()


def test_preflight_rejects_wrong_size_before_hub_segmentation_or_output() -> None:
    cfg = _config()
    wrong = _pattern(ACQUISITION_SIZE)
    capture = _ScriptedCapture([wrong], normalized_size=ACQUISITION_SIZE)
    segmenter = _TrackingSegmenter()
    refiner = _TrackingRefiner()
    output = _RecordingOutput()
    hub = FrameHub()
    hub.configure_canvas(CANVAS_SIZE)
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    resources = _Resources(cfg, 0, capture, segmenter, refiner, None, output)

    with pytest.raises(ActivationError, match="canonical canvas"):
        pipeline._preflight(resources)

    assert segmenter.frames == []
    assert refiner.frames == []
    assert output.snapshots() == []
    assert hub.raw.latest()[0] is None
    assert hub.output.latest()[0] is None


def test_steady_state_rejects_wrong_size_before_hub_segmentation_or_output() -> None:
    cfg = _config()
    wrong = _pattern(ACQUISITION_SIZE)
    capture = _ScriptedCapture([wrong], normalized_size=ACQUISITION_SIZE)
    segmenter = _TrackingSegmenter()
    refiner = _TrackingRefiner()
    output = _RecordingOutput()
    hub = FrameHub()
    hub.configure_canvas(CANVAS_SIZE)
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    resources = _Resources(cfg, 0, capture, segmenter, refiner, None, output)

    with pytest.raises(ValueError, match="canonical canvas"):
        pipeline._loop(resources)

    assert segmenter.frames == []
    assert refiner.frames == []
    assert output.snapshots() == []
    assert hub.raw.latest()[0] is None
    assert hub.output.latest()[0] is None


@pytest.mark.parametrize(
    ("mode", "segmentation"),
    (
        ("passthrough", "none"),
        ("color", "heuristic"),
        ("blur", "heuristic"),
    ),
)
def test_live_local_modes_publish_only_the_non_default_canonical_canvas(
    mode: str,
    segmentation: str,
) -> None:
    cfg = _config(mode, segmentation=segmentation)
    runtime = RuntimeConfig(cfg)
    hub = FrameHub()
    pipeline = Pipeline(runtime, hub)
    pipeline.start(timeout=3.0)
    try:
        output = _wait_for_output(hub)
        _wait_until(
            lambda: hub.raw.latest()[0] is not None,
            message="canonical raw frame",
        )
        raw = hub.raw.latest()[0]
        assert raw is not None
        assert raw.shape == output.shape == CANVAS_SHAPE
        assert raw.dtype == output.dtype == np.uint8
        assert raw.flags.c_contiguous and output.flags.c_contiguous

        status = hub.stats_dict()
        assert (status["capture_width"], status["capture_height"]) == ACQUISITION_SIZE
        assert (
            status["capture_delivered_width"],
            status["capture_delivered_height"],
        ) == ACQUISITION_SIZE
        assert (
            status["capture_normalized_width"],
            status["capture_normalized_height"],
        ) == CANVAS_SIZE
        assert (status["output_width"], status["output_height"]) == CANVAS_SIZE
        assert status["output_fps"] == OUTPUT_FPS
    finally:
        pipeline.stop()


def test_passthrough_keeps_hub_raw_output_and_sink_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pattern(offset=19)
    capture = _ScriptedCapture([source], repeat_last=True)
    output = _RecordingOutput()
    _install_pipeline_fakes(monkeypatch, capture, output)
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(_config("passthrough")), hub)
    pipeline.start(timeout=2.0)
    try:
        published = _wait_for_output(hub)
        raw = hub.raw.latest()[0]
        assert raw is source
        assert published is source
        np.testing.assert_array_equal(raw, published)
        assert all(frame.shape == CANVAS_SHAPE for frame in output.snapshots())
        assert all(np.array_equal(frame, source) for frame in output.snapshots())
    finally:
        pipeline.stop()


def test_remote_startup_fallback_and_valid_success_keep_exact_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pattern(offset=7)
    capture = _ScriptedCapture([source], repeat_last=True)
    output = _RecordingOutput()
    _install_pipeline_fakes(monkeypatch, capture, output)
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(_config("remote")), hub)
    slate = Pipeline._privacy_slate(CANVAS_SHAPE)
    pipeline.start(timeout=2.0)
    session = None
    try:
        _wait_for_output(hub)
        sent = output.snapshots()
        assert sent
        np.testing.assert_array_equal(sent[0], slate)
        assert all(frame.shape == CANVAS_SHAPE for frame in sent)
        np.testing.assert_array_equal(hub.output.latest()[0], slate)

        session = hub.remote_client_connected()
        rendered = np.full(CANVAS_SHAPE, (231, 17, 93), dtype=np.uint8)
        assert hub.push_remote_frame(rendered, session)
        _wait_until(
            lambda: (
                hub.output.latest()[0] is not None
                and np.array_equal(hub.output.latest()[0], rendered)
            ),
            message="accepted remote renderer frame",
        )
        np.testing.assert_array_equal(hub.output.latest()[0], rendered)
        assert any(np.array_equal(frame, rendered) for frame in output.snapshots())
        assert hub.stats_dict()["remote_fallback_active"] is False
        assert hub.stats_dict()["remote_frames_used"] >= 1
    finally:
        if session is not None:
            hub.remote_client_disconnected(session)
        pipeline.stop()


def test_wrong_size_remote_output_is_not_fitted_and_emits_privacy_slate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pattern(offset=29)
    capture = _ScriptedCapture([source], repeat_last=True)
    output = _RecordingOutput()
    _install_pipeline_fakes(monkeypatch, capture, output)
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(_config("remote")), hub)
    pipeline.start(timeout=2.0)
    session = hub.remote_client_connected()
    try:
        wrong = np.full(
            (ACQUISITION_SIZE[1], ACQUISITION_SIZE[0], 3),
            (1, 222, 19),
            dtype=np.uint8,
        )
        assert hub.push_remote_frame(wrong, session)
        _wait_until(
            lambda: hub.stats_dict()["remote_fallback_reason"] == "wrong-size",
            message="wrong-size remote fallback",
        )
        published = hub.output.latest()[0]
        assert published is not None
        np.testing.assert_array_equal(
            published,
            Pipeline._privacy_slate(CANVAS_SHAPE),
        )
        assert published.shape == CANVAS_SHAPE
        assert not np.any(np.all(published == wrong[0, 0], axis=2))
        assert all(frame.shape == CANVAS_SHAPE for frame in output.snapshots())
    finally:
        hub.remote_client_disconnected(session)
        pipeline.stop()


def test_remote_raw_echo_is_replaced_by_canonical_privacy_slate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pattern(offset=41)
    capture = _ScriptedCapture([source], repeat_last=True)
    output = _RecordingOutput()
    _install_pipeline_fakes(monkeypatch, capture, output)
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(_config("remote")), hub)
    pipeline.start(timeout=2.0)
    session = hub.remote_client_connected()
    try:
        assert hub.push_remote_frame(source.copy(), session)
        _wait_until(
            lambda: hub.stats_dict()["remote_fallback_reason"] == "privacy-raw-echo",
            message="raw-echo privacy fallback",
        )
        published = hub.output.latest()[0]
        assert published is not None
        np.testing.assert_array_equal(
            published,
            Pipeline._privacy_slate(CANVAS_SHAPE),
        )
        assert not np.array_equal(published, source)
    finally:
        hub.remote_client_disconnected(session)
        pipeline.stop()


def test_repeats_reuse_last_guarded_frame_without_reprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pattern(offset=53)
    capture = _ScriptedCapture([source])
    output = _RecordingOutput()
    segmenter = _TrackingSegmenter(mask_value=0.0)
    refiner = _TrackingRefiner()
    _install_pipeline_fakes(monkeypatch, capture, output)
    monkeypatch.setattr(pipeline_mod, "create_segmenter", lambda *_a, **_k: segmenter)
    monkeypatch.setattr(pipeline_mod, "refiner_for", lambda *_a, **_k: refiner)
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(_config("remote")), hub)
    slate = Pipeline._privacy_slate(CANVAS_SHAPE)
    pipeline.start(timeout=2.0)
    try:
        _wait_until(
            lambda: hub.stats_dict()["output_repeated_frames"] >= 2,
            message="repeated guarded output",
        )
        sent = output.snapshots()
        assert len(sent) >= 3
        assert len(segmenter.frames) == 1
        assert len(refiner.frames) == 1
        assert hub.raw.latest()[0] is None
        assert all(frame.shape == CANVAS_SHAPE for frame in sent)
        assert all(np.array_equal(frame, slate) for frame in sent)
        np.testing.assert_array_equal(hub.output.latest()[0], slate)
    finally:
        pipeline.stop()


def test_raw_fingerprint_uses_same_normalized_frame_published_to_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pattern(offset=67)
    capture = _ScriptedCapture([source], repeat_last=True)
    output = _RecordingOutput()
    _install_pipeline_fakes(monkeypatch, capture, output)
    fingerprint_inputs: list[np.ndarray] = []
    original = Pipeline._raw_fingerprint

    def fingerprint(frame: np.ndarray):
        fingerprint_inputs.append(frame)
        return original(frame)

    monkeypatch.setattr(Pipeline, "_raw_fingerprint", staticmethod(fingerprint))
    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(_config("remote")), hub)
    pipeline.start(timeout=2.0)
    session = hub.remote_client_connected()
    try:
        _wait_until(
            lambda: any(frame is source for frame in fingerprint_inputs),
            message="raw fingerprint",
        )
        published = hub.raw.latest()[0]
        assert published is source
        assert any(frame is published for frame in fingerprint_inputs)
        np.testing.assert_array_equal(published, source)
    finally:
        hub.remote_client_disconnected(session)
        pipeline.stop()


def test_status_reads_effective_mode_from_output_resource() -> None:
    cfg = _config()
    capture = _ScriptedCapture([_pattern()])
    segmenter = _TrackingSegmenter()
    reads: list[str] = []

    class EffectiveOutput:
        paces = False
        fallback_active = False
        fallback_reason = ""

        @property
        def width(self) -> int:
            reads.append("width")
            return 79

        @property
        def height(self) -> int:
            reads.append("height")
            return 44

        @property
        def fps(self) -> int:
            reads.append("fps")
            return 43

        def send(self, _frame: np.ndarray) -> None:
            pass

        def close(self) -> None:
            pass

    hub = FrameHub()
    pipeline = Pipeline(RuntimeConfig(cfg), hub)
    resources = _Resources(
        cfg,
        0,
        capture,
        segmenter,
        _TrackingRefiner(),
        None,
        cast(Any, EffectiveOutput()),
    )
    pipeline._update_identity_stats(resources)

    status = hub.stats_dict()
    assert {"width", "height", "fps"} <= set(reads)
    assert (status["output_width"], status["output_height"]) == (79, 44)
    assert status["output_target_fps"] == OUTPUT_FPS
    assert status["output_fps"] == 43


def _async_client(app: object) -> Any:
    transport = cast(Any, _ASGITransport)(app=app)
    return cast(Any, _AsyncClient)(
        transport=transport,
        base_url="http://testserver",
    )


async def _with_event_loop_heartbeat(awaitable):
    async def heartbeat():
        while True:
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    try:
        return await awaitable
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _run_async(awaitable):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_with_event_loop_heartbeat(awaitable))
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        asyncio.set_event_loop(None)


class _WebSocketClosed(Exception):
    def __init__(self, code: int, reason: str = "") -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"WebSocket closed with {code}: {reason}")


class _ASGIWebSocket:
    def __init__(self, app: Any, url: str, headers: dict[str, str]) -> None:
        self.app = app
        self.url = url
        self.headers = headers
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.outgoing: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.accept_headers: dict[str, str] = {}

    async def _receive(self):
        return await self.incoming.get()

    async def _send(self, message):
        await self.outgoing.put(message)

    async def connect(self):
        parsed = urlsplit(self.url)
        headers = {"host": "testserver", **self.headers}
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 12345),
            "root_path": "",
            "path": parsed.path,
            "raw_path": parsed.path.encode(),
            "query_string": parsed.query.encode(),
            "headers": [
                (name.lower().encode(), value.encode())
                for name, value in headers.items()
            ],
            "subprotocols": [],
            "state": {},
            "extensions": {},
        }
        self.task = asyncio.create_task(self.app(scope, self._receive, self._send))
        await self.incoming.put({"type": "websocket.connect"})
        message = await asyncio.wait_for(self.outgoing.get(), 2.0)
        if message["type"] == "websocket.close":
            await self._finish()
            raise _WebSocketClosed(
                message.get("code", 1000),
                message.get("reason", ""),
            )
        assert message["type"] == "websocket.accept", message
        self.accept_headers = {
            key.decode("ascii").lower(): value.decode("ascii")
            for key, value in message.get("headers", ())
        }
        return self

    async def send_bytes(self, data: bytes) -> None:
        await self.incoming.put({"type": "websocket.receive", "bytes": data})

    async def receive_bytes(self) -> bytes:
        message = await asyncio.wait_for(self.outgoing.get(), 3.0)
        if message["type"] == "websocket.close":
            raise _WebSocketClosed(
                message.get("code", 1000),
                message.get("reason", ""),
            )
        assert message["type"] == "websocket.send", message
        return message["bytes"]

    async def receive_close(self) -> _WebSocketClosed:
        for _ in range(50):
            try:
                await self.receive_bytes()
            except _WebSocketClosed as closed:
                return closed
        raise AssertionError("server did not close after invalid renderer frame")

    async def _finish(self) -> None:
        if self.task is None:
            return
        if not self.task.done():
            await self.incoming.put({"type": "websocket.disconnect", "code": 1000})
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(self.task, 2.0)

    async def close(self) -> None:
        await self._finish()


@dataclass
class _ApiStack:
    app: Any
    runtime: RuntimeConfig
    hub: FrameHub
    pipeline: Pipeline

    async def arequest(self, method: str, path: str, **kwargs):
        async with _async_client(self.app) as client:
            return await client.request(method, path, **kwargs)

    def get(self, path: str, **kwargs):
        return _run_async(self.arequest("GET", path, **kwargs))


@pytest.fixture
def canonical_api_stack(tmp_path):
    runtime = RuntimeConfig(_config("passthrough"))
    hub = FrameHub()
    pipeline = Pipeline(runtime, hub)
    pipeline.start(timeout=3.0)
    _wait_for_output(hub)
    _wait_until(lambda: hub.raw.latest()[0] is not None, message="API raw frame")
    security = SecurityPolicy.for_bind(
        TOKEN,
        "testserver",
        80,
        allowed_origins=[ORIGIN],
        extra_hosts=["testserver"],
        renderer_token=RENDERER_TOKEN,
    )
    app = create_app(
        runtime,
        hub,
        pipeline,
        security=security,
        upload_dir=tmp_path / "uploads",
    )
    yield _ApiStack(app, runtime, hub, pipeline)
    pipeline.stop()


def _decode(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    return frame


def test_snapshot_and_status_report_resolved_canvas(canonical_api_stack) -> None:
    snapshot = canonical_api_stack.get("/video/snapshot.jpg", headers=AUTH)
    assert snapshot.status_code == 200
    assert snapshot.headers["x-frame-width"] == str(CANVAS_SIZE[0])
    assert snapshot.headers["x-frame-height"] == str(CANVAS_SIZE[1])
    assert _decode(snapshot.content).shape == CANVAS_SHAPE

    status = canonical_api_stack.get("/status", headers=AUTH)
    assert status.status_code == 200
    body = status.json()
    assert (body["capture_width"], body["capture_height"]) == ACQUISITION_SIZE
    assert (
        body["capture_normalized_width"],
        body["capture_normalized_height"],
    ) == CANVAS_SIZE
    assert (body["output_width"], body["output_height"]) == CANVAS_SIZE
    assert body["output_target_fps"] == OUTPUT_FPS
    assert body["output_fps"] == OUTPUT_FPS


def test_mjpeg_part_metadata_and_pixels_use_resolved_canvas(
    canonical_api_stack,
) -> None:
    route = next(
        route
        for route in canonical_api_stack.app.routes
        if getattr(route, "path", None) == "/video/mjpeg"
    )

    async def scenario() -> bytes:
        response = await route.endpoint()
        try:
            return await asyncio.wait_for(anext(response.body_iterator), 3.0)
        finally:
            await response.body_iterator.aclose()

    part = _run_async(scenario())
    headers, payload = part.split(b"\r\n\r\n", 1)
    jpeg = payload.removesuffix(b"\r\n")
    assert f"X-Frame-Width: {CANVAS_SIZE[0]}".encode() in headers
    assert f"X-Frame-Height: {CANVAS_SIZE[1]}".encode() in headers
    assert _decode(jpeg).shape == CANVAS_SHAPE
    assert canonical_api_stack.app.state.stream_connections.active == 0


def test_raw_websocket_advertises_canvas_and_renderer_accepts_only_that_size(
    canonical_api_stack,
) -> None:
    async def scenario() -> None:
        websocket = await _ASGIWebSocket(
            canonical_api_stack.app,
            "/ws/frames?stream=raw",
            headers={**RENDERER_AUTH, "Origin": ORIGIN},
        ).connect()
        try:
            assert websocket.accept_headers == {
                "x-custback-frame-width": str(CANVAS_SIZE[0]),
                "x-custback-frame-height": str(CANVAS_SIZE[1]),
            }
            assert _decode(await websocket.receive_bytes()).shape == CANVAS_SHAPE
            rendered = np.full(CANVAS_SHAPE, (27, 109, 233), dtype=np.uint8)
            ok, jpeg = cv2.imencode(".jpg", rendered)
            assert ok
            await websocket.send_bytes(jpeg.tobytes())
            deadline = time.monotonic() + 3.0
            while (
                canonical_api_stack.hub.remote_in.latest()[0] is None
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.01)
            accepted = canonical_api_stack.hub.remote_in.latest()[0]
            assert accepted is not None
            assert accepted.shape == CANVAS_SHAPE
        finally:
            await websocket.close()

        websocket = await _ASGIWebSocket(
            canonical_api_stack.app,
            "/ws/frames?stream=raw",
            headers={**RENDERER_AUTH, "Origin": ORIGIN},
        ).connect()
        try:
            await websocket.receive_bytes()
            wrong = np.zeros(
                (ACQUISITION_SIZE[1], ACQUISITION_SIZE[0], 3),
                dtype=np.uint8,
            )
            ok, jpeg = cv2.imencode(".jpg", wrong)
            assert ok
            await websocket.send_bytes(jpeg.tobytes())
            closed = await websocket.receive_close()
            assert closed.code == 1007
            assert closed.reason == "invalid JPEG or frame dimensions"
        finally:
            await websocket.close()

    _run_async(scenario())


def test_output_websocket_advertises_and_sends_resolved_canvas(
    canonical_api_stack,
) -> None:
    async def scenario() -> None:
        websocket = await _ASGIWebSocket(
            canonical_api_stack.app,
            "/ws/frames?stream=output",
            headers={**AUTH, "Origin": ORIGIN},
        ).connect()
        try:
            assert websocket.accept_headers == {
                "x-custback-frame-width": str(CANVAS_SIZE[0]),
                "x-custback-frame-height": str(CANVAS_SIZE[1]),
            }
            assert _decode(await websocket.receive_bytes()).shape == CANVAS_SHAPE
            assert canonical_api_stack.hub.stats_dict()["remote_connected"] is False
        finally:
            await websocket.close()

    _run_async(scenario())
