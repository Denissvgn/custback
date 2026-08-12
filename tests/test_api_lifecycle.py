import contextlib
import http.client
import logging
import signal
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import custback.__main__ as main_mod
import custback.api.server as server_mod
from custback.api.security import SecurityPolicy
from custback.config import AppConfig, RuntimeConfig
from custback.hub import FrameHub


class FakePipeline:
    instances = []
    start_error: BaseException | None = None

    def __init__(self, runtime, hub, **_kwargs):
        self.running = False
        self.stopped = False
        self.options = _kwargs
        type(self).instances.append(self)

    def start(self):
        if self.start_error:
            raise self.start_error
        self.running = True

    def stop(self):
        self.running = False
        self.stopped = True


def _cfg():
    return AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 64, "height": 36},
            "background": {"mode": "passthrough"},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
        }
    )


def _uninitialized_api_runner() -> Any:
    """Build a runner whose lifecycle state is supplied by the test."""

    runner = object.__new__(main_mod._ApiRunner)
    runner._stop_lock = threading.Lock()
    runner._stream_lifecycle = None
    return runner


def _patch_common(monkeypatch):
    FakePipeline.instances.clear()
    FakePipeline.start_error = None
    monkeypatch.setattr(main_mod, "Pipeline", FakePipeline)
    monkeypatch.setattr(main_mod, "_security_policy", lambda _cfg: object())
    monkeypatch.setattr(server_mod, "create_app", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main_mod.signal, "signal", lambda *_args: None)


def test_api_startup_failure_returns_3_and_stops_preflighted_pipeline(monkeypatch):
    _patch_common(monkeypatch)

    class FailedRunner:
        def __init__(self, *_args):
            pass

        def start(self):
            raise main_mod.ApiStartupError("occupied")

    monkeypatch.setattr(main_mod, "_ApiRunner", FailedRunner)
    assert main_mod.run(_cfg()) == main_mod.EXIT_API
    assert len(FakePipeline.instances) == 1
    assert not FakePipeline.instances[0].running
    assert FakePipeline.instances[0].stopped


def test_pipeline_start_exception_does_not_bind_api(monkeypatch):
    _patch_common(monkeypatch)
    stopped = []

    class Runner:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def stop(self):
            stopped.append(True)
            return True

    FakePipeline.start_error = RuntimeError("capture failed")
    monkeypatch.setattr(main_mod, "_ApiRunner", Runner)
    assert main_mod.run(_cfg()) == main_mod.EXIT_RUNTIME
    assert stopped == []


def test_recorded_pipeline_start_failure_has_one_authoritative_traceback(
    monkeypatch, caplog
):
    _patch_common(monkeypatch)

    class WorkerFailedPipeline(FakePipeline):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.error: BaseException | None = None

        def start(self):
            try:
                raise RuntimeError("worker failed")
            except RuntimeError as exc:
                self.error = exc
                logging.getLogger("custback.pipeline").exception("pipeline crashed")
                raise RuntimeError("startup propagation") from exc

        def stop(self):
            self.stopped = True
            assert self.error is not None
            raise self.error

    monkeypatch.setattr(main_mod, "Pipeline", WorkerFailedPipeline)

    with caplog.at_level("ERROR"):
        assert main_mod.run(_cfg()) == main_mod.EXIT_RUNTIME

    traceback_records = [record for record in caplog.records if record.exc_info]
    assert [record.getMessage() for record in traceback_records] == ["pipeline crashed"]
    assert caplog.text.count("pipeline startup failed") == 1
    assert "worker traceback logged" in caplog.text
    assert "pipeline shutdown failed" not in caplog.text


def test_signal_during_pipeline_start_unwinds_as_clean_exit(monkeypatch, caplog):
    _patch_common(monkeypatch)
    handlers = {}

    def install(sig, handler):
        previous = handlers.get(sig, signal.SIG_DFL)
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(main_mod.signal, "signal", install)

    def interrupted_start(self):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(FakePipeline, "start", interrupted_start)
    with caplog.at_level("INFO"):
        assert main_mod.run(_cfg()) == 0
    assert FakePipeline.instances[0].stopped
    assert "shutdown reason=sigterm exit=0" in caplog.text


def test_unexpected_api_exit_is_latched_as_exit_3(monkeypatch):
    _patch_common(monkeypatch)

    class Runner:
        def __init__(self, *_args):
            self.error = RuntimeError("server died")
            self.failed = True

        def start(self):
            pass

        def stop(self):
            return True

    monkeypatch.setattr(main_mod, "_ApiRunner", Runner)
    assert main_mod.run(_cfg()) == main_mod.EXIT_API
    assert FakePipeline.instances[0].stopped


def test_unexpected_api_exit_during_preview_is_classified_and_logged(
    monkeypatch, caplog
):
    _patch_common(monkeypatch)
    import custback.preview as preview_mod

    cfg = _cfg()
    cfg.output.preview = True
    monkeypatch.setattr(preview_mod, "preview_available", lambda: (True, ""))
    monkeypatch.setattr(
        preview_mod,
        "run_preview",
        lambda *_args, **_kwargs: False,
    )

    class Runner:
        def __init__(self, *_args):
            self.error = RuntimeError("preview API died")
            self.failed = True

        def start(self):
            pass

        def stop(self):
            return True

    monkeypatch.setattr(main_mod, "_ApiRunner", Runner)
    assert main_mod.run(cfg) == main_mod.EXIT_API
    assert "preview API died" in caplog.text
    assert FakePipeline.instances[0].stopped


def test_ready_and_shutdown_records_are_ordered_and_complete(monkeypatch, caplog):
    _patch_common(monkeypatch)
    import custback.preview as preview_mod

    cfg = _cfg()
    cfg.output.preview = True
    monkeypatch.setattr(preview_mod, "preview_available", lambda: (True, ""))
    monkeypatch.setattr(preview_mod, "run_preview", lambda *_args, **_kwargs: True)

    class Runner:
        error = None
        failed = False

        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def stop(self):
            return True

    monkeypatch.setattr(main_mod, "_ApiRunner", Runner)
    with caplog.at_level("INFO"):
        assert main_mod.run(cfg, run_id="lifecycle") == 0

    ready = caplog.text.index("ready api=")
    shutdown = caplog.text.index("shutdown reason=preview-quit")
    assert ready < shutdown
    assert "camera_requested=auto/64x36@30" in caplog.text
    assert "camera_negotiated=" in caplog.text
    assert "visual_policy=schema_version=1" in caplog.text
    assert "camera.fit_mode=stretch" in caplog.text
    assert "background.fit_mode=cover" in caplog.text
    assert "compositing.blend_space=srgb_legacy" in caplog.text
    assert "compositing.color_correction.mode=off" in caplog.text
    assert caplog.text.count("backend=none->none tier=none") == 2
    assert caplog.text.count("provider=none") == 2
    assert caplog.text.count("backend_fallback=False") == 2
    assert caplog.text.count("backend_fallback_category=none") == 2
    assert caplog.text.count("alpha_policy=opaque_passthrough") == 2
    assert caplog.text.count("blend_space=srgb_legacy") >= 2
    for field in (
        "capture_read_ms=",
        "segmentation_ms=",
        "background_ms=",
        "color_correction_ms=",
        "composite_ms=",
        "output_send_ms=",
        "frame_processing_ms=",
    ):
        assert field in caplog.text
    for field in (
        "unique_updates=",
        "segmentation_updates=",
        "output_sends=",
        "safe_base_reuses=",
        "exact_final_repeats=",
        "capture_gaps=",
        "capture_missing=",
        "capture_slot_overwrites=",
        "processing_deadline_misses=",
        "serialized_deadline_misses=",
        "sink_pacing_events=",
        "sink_recovery_events=",
        "application_pacing_events=",
        "schedule_late_events=",
        "matte_resets=",
        "matte_last_reset=",
    ):
        assert caplog.text.count(field) >= 2
    assert "model_preparation" in FakePipeline.instances[0].options


def test_api_runner_failure_latch_survives_shutdown():
    runner = _uninitialized_api_runner()
    runner.server = SimpleNamespace(should_exit=False, force_exit=False)
    runner._thread = SimpleNamespace(is_alive=lambda: False, join=lambda _timeout: None)
    runner._socket = None
    runner._error = None
    runner._shutdown_requested = False
    runner._unexpected_exit = False
    assert runner.failed
    assert runner.stop()
    assert runner.failed


def test_api_runner_passes_configured_websocket_size_to_uvicorn():
    cfg = _cfg().api.model_copy(update={"ws_max_bytes": 123456})
    runner = main_mod._ApiRunner(object(), cfg)
    assert runner.server.config.ws_max_size == 123456
    assert (
        runner.server.config.timeout_graceful_shutdown
        == main_mod.API_GRACEFUL_SHUTDOWN_TIMEOUT_S
    )


def test_api_runner_bind_failure_is_api_startup_error():
    runner = _uninitialized_api_runner()
    runner.server = SimpleNamespace(
        config=SimpleNamespace(
            bind_socket=lambda: (_ for _ in ()).throw(OSError("occupied"))
        )
    )
    runner._thread = None
    runner._socket = None
    runner._error = None
    runner._shutdown_requested = False
    runner._unexpected_exit = False
    with pytest.raises(main_mod.ApiStartupError, match="bind failed"):
        runner.start()


def test_api_runner_thread_start_failure_closes_prebound_socket(monkeypatch):
    class Socket:
        closed = False

        def close(self):
            self.closed = True

    sock = Socket()
    runner = _uninitialized_api_runner()
    runner.server = SimpleNamespace(
        config=SimpleNamespace(bind_socket=lambda: sock),
        run=lambda **_kwargs: None,
    )
    runner._thread = None
    runner._socket = None
    runner._error = None
    runner._shutdown_requested = False
    runner._unexpected_exit = False

    class FailedThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(main_mod.threading, "Thread", FailedThread)
    with pytest.raises(main_mod.ApiStartupError, match="thread startup failed"):
        runner.start()
    assert sock.closed
    assert runner._thread is None


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_real_api_runner_closes_active_core_streams_and_is_idempotent(tmp_path):
    from websockets.sync.client import connect
    from websockets.typing import Origin

    from custback.pipeline import Pipeline as RealPipeline

    token = "real-api-lifecycle-token-which-is-long-enough"
    renderer_token = "renderer-lifecycle-token-which-is-long-enough"
    port = _reserve_loopback_port()
    origin = f"http://127.0.0.1:{port}"
    websocket_origin = Origin(origin)
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 64, "height": 36},
            "background": {"mode": "passthrough"},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
            "api": {"host": "127.0.0.1", "port": port},
        }
    )
    runtime = RuntimeConfig(cfg)
    hub = FrameHub()
    pipeline = RealPipeline(runtime, hub)
    security = SecurityPolicy.for_bind(
        token,
        "127.0.0.1",
        port,
        allowed_origins=[origin],
        renderer_token=renderer_token,
    )
    app = server_mod.create_app(
        runtime,
        hub,
        pipeline,
        security=security,
        upload_dir=tmp_path / "uploads",
    )
    runner = main_mod._ApiRunner(app, cfg.api)
    http_stream = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    management_ws = None
    renderer_ws = None
    try:
        runner.start()
        http_stream.request(
            "GET",
            "/video/mjpeg",
            headers={"Authorization": f"Bearer {token}"},
        )
        response = http_stream.getresponse()
        assert response.status == 200

        management_ws = connect(
            f"ws://127.0.0.1:{port}/ws/frames?stream=output",
            origin=websocket_origin,
            additional_headers={"Authorization": f"Bearer {token}"},
            proxy=None,
            open_timeout=2.0,
            close_timeout=1.0,
        )
        renderer_ws = connect(
            f"ws://127.0.0.1:{port}/ws/frames?stream=raw",
            origin=websocket_origin,
            additional_headers={"Authorization": f"Bearer {renderer_token}"},
            proxy=None,
            open_timeout=2.0,
            close_timeout=1.0,
        )

        deadline = time.monotonic() + 2.0
        while app.state.stream_lifecycle.snapshot()["active"] != 3:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert app.state.stream_connections.active == 3
        assert hub.stats_dict()["remote_connected"] is True

        started = time.monotonic()
        assert runner.stop()
        assert time.monotonic() - started < main_mod.API_STOP_TIMEOUT_S
        assert runner._thread is not None and not runner._thread.is_alive()
        assert runner.server.force_exit is False
        assert app.state.stream_lifecycle.snapshot()["active"] == 0
        assert app.state.stream_connections.active == 0
        assert hub.stats_dict()["remote_connected"] is False

        repeated = time.monotonic()
        assert runner.stop()
        assert time.monotonic() - repeated < 0.2
    finally:
        if management_ws is not None:
            with contextlib.suppress(Exception):
                management_ws.close()
        if renderer_ws is not None:
            with contextlib.suppress(Exception):
                renderer_ws.close()
        http_stream.close()
        runner.stop()


def test_real_api_runner_closes_active_avatar_proxy_stream(monkeypatch, tmp_path):
    import asyncio

    import httpx

    from custback.pipeline import Pipeline as RealPipeline

    class ObservedUpstream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.started = threading.Event()
            self.closed = threading.Event()
            self._lock = threading.Lock()
            self._read_polls = 0
            self._close_calls = 0

        async def __aiter__(self):
            self.started.set()
            while not self.closed.is_set():
                with self._lock:
                    self._read_polls += 1
                await asyncio.sleep(0.01)
            if False:  # pragma: no cover - keeps this an async byte iterator
                yield b""

        async def aclose(self) -> None:
            with self._lock:
                self._close_calls += 1
            self.closed.set()

        def snapshot(self) -> tuple[int, int]:
            with self._lock:
                return self._read_polls, self._close_calls

    token = "avatar-proxy-lifecycle-token-which-is-long-enough"
    avatar_token = "avatar-upstream-lifecycle-token-long-enough"
    monkeypatch.setenv("CUSTBACK_AVATAR_API_TOKEN", avatar_token)
    port = _reserve_loopback_port()
    cfg = AppConfig.from_dict(
        {
            "camera": {"synthetic": True, "width": 64, "height": 36},
            "background": {"mode": "passthrough"},
            "segmentation": {"backend": "heuristic"},
            "output": {"backend": "null"},
            "api": {"host": "127.0.0.1", "port": port},
            "avatar": {"url": "https://avatar.example:8711"},
        }
    )
    runtime = RuntimeConfig(cfg)
    hub = FrameHub()
    pipeline = RealPipeline(runtime, hub)
    body = ObservedUpstream()
    clients: list[httpx.AsyncClient] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "multipart/x-mixed-replace"},
            stream=body,
        )

    def client_factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    security = SecurityPolicy.for_bind(token, "127.0.0.1", port)
    app = server_mod.create_app(
        runtime,
        hub,
        pipeline,
        security=security,
        upload_dir=tmp_path / "uploads",
        avatar_client_factory=client_factory,
    )
    runner = main_mod._ApiRunner(app, cfg.api)
    stream = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    try:
        runner.start()
        stream.request(
            "GET",
            "/avatar/video/mjpeg",
            headers={"Authorization": f"Bearer {token}"},
        )
        response = stream.getresponse()
        assert response.status == 200
        assert body.started.wait(2.0)

        deadline = time.monotonic() + 2.0
        while app.state.stream_lifecycle.snapshot()["active"] != 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert app.state.stream_connections.active == 1
        assert hub.active_remote_session() is None

        started = time.monotonic()
        assert runner.stop()
        assert time.monotonic() - started < 5.0
        assert runner._thread is not None and not runner._thread.is_alive()
        assert app.state.stream_lifecycle.snapshot()["active"] == 0
        assert app.state.stream_lifecycle.snapshot()["active_tasks"] == 0
        assert app.state.stream_connections.active == 0
        assert hub.active_remote_session() is None
        assert body.closed.is_set()
        assert clients and all(client.is_closed for client in clients)

        reads_after_stop, close_calls = body.snapshot()
        assert close_calls >= 1
        time.sleep(0.05)
        assert body.snapshot()[0] == reads_after_stop
    finally:
        stream.close()
        runner.stop()


def test_real_api_runner_closes_active_standalone_avatar_mjpeg(tmp_path):
    import numpy as np

    from custback.avatar.api import create_avatar_app
    from custback.avatar.config import AvatarConfig, AvatarRuntime
    from custback.avatar.service import AvatarService

    token = "standalone-avatar-lifecycle-token-which-is-long-enough"
    port = _reserve_loopback_port()
    avatar_runtime = AvatarRuntime(
        AvatarConfig.from_dict(
            {
                "driver": {"backend": "idle"},
                "storage": {
                    "rigs_dir": str(tmp_path / "rigs"),
                    "backgrounds_dir": str(tmp_path / "backgrounds"),
                },
                "api": {"host": "127.0.0.1", "port": port},
            }
        )
    )
    service = AvatarService(avatar_runtime)
    security = SecurityPolicy.for_bind(token, "127.0.0.1", port)
    app = create_avatar_app(avatar_runtime, service, security=security)
    runner = main_mod._ApiRunner(app, avatar_runtime.read().config.api)
    stream = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)

    def subscriber_count() -> int:
        with service.output._cond:
            return len(service.output._subscribers)

    try:
        runner.start()
        stream.request(
            "GET",
            "/video/mjpeg",
            headers={"Authorization": f"Bearer {token}"},
        )
        response = stream.getresponse()
        assert response.status == 200

        deadline = time.monotonic() + 2.0
        while (
            app.state.stream_lifecycle.snapshot()["active"] != 1
            or subscriber_count() != 1
        ):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert app.state.stream_connections.active == 1
        assert service.stats_dict()["connected"] is False

        started = time.monotonic()
        assert runner.stop()
        assert time.monotonic() - started < 5.0
        assert runner._thread is not None and not runner._thread.is_alive()
        assert app.state.stream_lifecycle.snapshot()["active"] == 0
        assert app.state.stream_lifecycle.snapshot()["active_tasks"] == 0
        assert app.state.stream_connections.active == 0
        assert subscriber_count() == 0
        assert service.stats_dict()["connected"] is False

        # A publication after API teardown must have no surviving stream
        # subscriber and therefore cannot touch an exited event loop.
        service.output.put(np.zeros((2, 2, 3), dtype=np.uint8))
        assert subscriber_count() == 0
    finally:
        stream.close()
        runner.stop()
        service.close()


def test_api_runner_timeout_log_contains_only_stream_counts(caplog):
    class AliveThread:
        def is_alive(self):
            return True

        def join(self, timeout):
            threading.Event().wait(timeout)

    runner = _uninitialized_api_runner()
    runner.server = SimpleNamespace(should_exit=False, force_exit=False)
    runner._thread = AliveThread()
    runner._socket = None
    runner._error = None
    runner._shutdown_requested = False
    runner._unexpected_exit = False
    runner._stream_lifecycle = SimpleNamespace(
        request_shutdown=lambda _timeout: None,
        snapshot=lambda: {
            "active": 2,
            "active_tasks": 2,
            "request_path": "/private/operator/stream",
            "credential": "must-not-appear",
            "by_kind": {
                "core_mjpeg": 1,
                "management_websocket": 0,
                "renderer_websocket": 1,
                "avatar_proxy": 0,
            },
        },
    )

    with caplog.at_level("ERROR"):
        started = time.monotonic()
        assert runner.stop(timeout=0.05) is False
        assert time.monotonic() - started < 0.2

    assert "API shutdown fallback phase=force-exit reason=forced-timeout" in caplog.text
    assert "streams_active=2" in caplog.text
    assert "tasks_active=2" in caplog.text
    assert "core_mjpeg=1" in caplog.text
    assert "renderer_websocket=1" in caplog.text
    assert "/private/operator/stream" not in caplog.text
    assert "must-not-appear" not in caplog.text


def test_api_runner_forced_exit_is_diagnosed_and_latched(caplog):
    class ForcedThread:
        def __init__(self):
            self.alive = True
            self.joins = 0

        def is_alive(self):
            return self.alive

        def join(self, _timeout):
            self.joins += 1
            if self.joins == 2:
                self.alive = False

    runner = _uninitialized_api_runner()
    runner.server = SimpleNamespace(should_exit=False, force_exit=False)
    runner._thread = ForcedThread()
    runner._socket = None
    runner._error = None
    runner._shutdown_requested = False
    runner._unexpected_exit = False

    with caplog.at_level("ERROR"):
        assert runner.stop(timeout=0.05) is False
        assert runner.stop(timeout=0.05) is False

    assert runner.server.force_exit is True
    assert runner._thread.is_alive() is False
    assert caplog.text.count("reason=graceful-timeout") == 1
