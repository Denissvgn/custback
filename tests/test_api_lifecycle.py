from types import SimpleNamespace

import pytest

import custback.__main__ as main_mod
import custback.api.server as server_mod
from custback.config import AppConfig


class FakePipeline:
    instances = []
    start_error = None

    def __init__(self, runtime, hub):
        self.running = False
        self.stopped = False
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


def _patch_common(monkeypatch):
    FakePipeline.instances.clear()
    FakePipeline.start_error = None
    monkeypatch.setattr(main_mod, "Pipeline", FakePipeline)
    monkeypatch.setattr(main_mod, "_security_policy", lambda _cfg: object())
    monkeypatch.setattr(server_mod, "create_app", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main_mod.signal, "signal", lambda *_args: None)


def test_api_startup_failure_returns_3_without_starting_capture(monkeypatch):
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


def test_pipeline_start_exception_stops_already_started_api(monkeypatch):
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
    assert stopped == [True]


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


def test_api_runner_failure_latch_survives_shutdown():
    runner = object.__new__(main_mod._ApiRunner)
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
    runner = object.__new__(main_mod._ApiRunner)
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
    runner = object.__new__(main_mod._ApiRunner)
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
