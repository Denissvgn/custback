from types import SimpleNamespace
import signal
from typing import Any

import pytest

import custback.__main__ as main_mod
import custback.api.server as server_mod
from custback.config import AppConfig


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

    return object.__new__(main_mod._ApiRunner)


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
