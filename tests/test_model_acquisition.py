from __future__ import annotations

import hashlib
import io
import stat
import threading
import time
from pathlib import Path

import pytest

import custback.segmentation as segmentation_mod
from custback.config import SegmentationConfig
from custback.segmentation import (
    MEDIAPIPE_MODEL,
    RVM_MODEL,
    ModelAcquisitionError,
    ModelSpec,
    SegmenterPreparation,
    acquire_builtin_model,
    acquire_model,
    create_segmenter,
    preacquire_segmenter_model,
)


class Response(io.BytesIO):
    def __init__(self, data: bytes, content_length: str | None = None):
        super().__init__(data)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def spec_for(data: bytes, filename: str = "model.bin") -> ModelSpec:
    return ModelSpec(
        backend="test",
        url="https://models.invalid/model.bin",
        filename=filename,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def test_builtin_model_registry_is_immutable_and_version_pinned():
    assert RVM_MODEL.size == 14_975_696
    assert RVM_MODEL.sha256 == (
        "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"
    )
    assert "/v1.0.0/" in RVM_MODEL.url
    assert MEDIAPIPE_MODEL.size == 249_537
    assert MEDIAPIPE_MODEL.sha256 == (
        "191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b"
    )
    assert "/float16/1/" in MEDIAPIPE_MODEL.url
    assert "/latest/" not in MEDIAPIPE_MODEL.url
    with pytest.raises(Exception):
        RVM_MODEL.size = 1


def test_acquire_model_streams_verifies_and_atomically_promotes(tmp_path):
    payload = b"verified-model"
    observed = {}

    def opener(request, timeout):
        observed["url"] = request.full_url
        observed["agent"] = request.headers["User-agent"]
        observed["timeout"] = timeout
        return Response(payload, str(len(payload)))

    path = acquire_model(spec_for(payload), tmp_path, opener=opener)

    assert path.read_bytes() == payload
    assert observed == {
        "url": "https://models.invalid/model.bin",
        "agent": "custback-model-fetch/1",
        "timeout": 15.0,
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert not list(tmp_path.glob("*.part"))


def test_cached_model_is_hashed_on_every_use_and_corruption_is_replaced(tmp_path):
    payload = b"valid-model"
    model = tmp_path / "model.bin"
    tmp_path.mkdir(exist_ok=True)
    model.write_bytes(b"x" * len(payload))
    calls = 0

    def opener(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 15.0
        return Response(payload)

    assert acquire_model(spec_for(payload), tmp_path, opener=opener).read_bytes() == payload
    assert calls == 1
    assert acquire_model(
        spec_for(payload),
        tmp_path,
        opener=lambda *_args, **_kwargs: pytest.fail("valid cache redownloaded"),
    ) == model


@pytest.mark.parametrize(
    ("body", "header", "message"),
    [
        (b"short", None, "truncated download"),
        (b"bad-data", None, "checksum mismatch"),
        (b"too-long-body", None, "exceeded expected size"),
        (b"expected", "999", "server advertised"),
        (b"expected", "invalid", "invalid Content-Length"),
    ],
)
def test_failed_download_preserves_existing_cache_and_cleans_temporary(
    tmp_path, body, header, message
):
    expected = b"expected"
    old = b"old-data"
    model = tmp_path / "model.bin"
    model.write_bytes(old)

    with pytest.raises(ModelAcquisitionError, match=message):
        acquire_model(
            spec_for(expected),
            tmp_path,
            opener=lambda *_args, **_kwargs: Response(body, header),
        )

    assert model.read_bytes() == old
    assert not list(tmp_path.glob("*.part"))


def test_concurrent_acquisition_downloads_once(tmp_path):
    payload = b"one-writer"
    model_spec = spec_for(payload)
    calls = 0
    calls_lock = threading.Lock()
    start = threading.Barrier(2)
    results: list[Path] = []

    def opener(_request, timeout):
        nonlocal calls
        assert timeout == 15.0
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return Response(payload)

    def acquire():
        start.wait()
        results.append(acquire_model(model_spec, tmp_path, opener=opener))

    threads = [threading.Thread(target=acquire) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert calls == 1
    assert results == [tmp_path / "model.bin", tmp_path / "model.bin"]


def test_unknown_builtin_backend_fails_without_network(tmp_path):
    with pytest.raises(ValueError, match="does not have"):
        acquire_builtin_model("unknown", tmp_path)


def test_network_timeout_preserves_cache_and_is_actionable(tmp_path):
    payload = b"expected"
    model = tmp_path / "model.bin"
    model.write_bytes(b"old-data")

    def timeout(*_args, **_kwargs):
        raise TimeoutError("socket timed out")

    with pytest.raises(ModelAcquisitionError, match="socket timed out"):
        acquire_model(spec_for(payload), tmp_path, opener=timeout)
    assert model.read_bytes() == b"old-data"
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize(
    "bad_spec",
    [
        ModelSpec("test", "http://models.invalid/x", "model.bin", 1, "0" * 64),
        ModelSpec("test", "https://models.invalid/x", "../model.bin", 1, "0" * 64),
        ModelSpec("test", "https://models.invalid/x", "model.bin", 0, "0" * 64),
        ModelSpec("test", "https://models.invalid/x", "model.bin", 1, "not-a-hash"),
    ],
)
def test_invalid_model_spec_is_rejected_before_io(tmp_path, bad_spec):
    with pytest.raises(ValueError):
        acquire_model(
            bad_spec,
            tmp_path,
            opener=lambda *_args, **_kwargs: pytest.fail("network should not run"),
        )


def test_runtime_use_of_preacquired_model_never_starts_a_download(tmp_path):
    payload = b"verified"
    with pytest.raises(ModelAcquisitionError, match="pre-acquired model"):
        acquire_model(
            spec_for(payload),
            tmp_path,
            allow_download=False,
            opener=lambda *_args, **_kwargs: pytest.fail("network should not run"),
        )


def test_auto_preacquires_every_installed_fallback(monkeypatch):
    acquired = []
    monkeypatch.setattr(
        segmentation_mod.importlib,
        "import_module",
        lambda name: object(),
    )
    monkeypatch.setattr(
        segmentation_mod,
        "acquire_builtin_model",
        lambda backend: acquired.append(backend),
    )

    preparation = preacquire_segmenter_model(SegmentationConfig(backend="auto"))

    assert acquired == ["rvm", "mediapipe"]
    assert preparation.ready_backends == frozenset({"rvm", "mediapipe"})


def test_preparation_prevents_retry_of_an_unavailable_higher_backend(monkeypatch):
    class PreparedMediaPipe:
        device = "cpu"

    monkeypatch.setattr(
        segmentation_mod,
        "RVMSegmenter",
        lambda *_args, **_kwargs: pytest.fail("RVM must not be retried"),
    )
    monkeypatch.setattr(
        segmentation_mod,
        "MediaPipeSegmenter",
        lambda *_args, **_kwargs: PreparedMediaPipe(),
    )
    segmenter = create_segmenter(
        SegmentationConfig(backend="auto"),
        preparation=SegmenterPreparation(frozenset({"mediapipe"})),
    )
    assert isinstance(segmenter, PreparedMediaPipe)


def test_total_download_deadline_is_checked_after_a_blocking_read(monkeypatch):
    payload = b"verified"
    ticks = iter((0.0, 0.0, 0.1, 2.0))
    monkeypatch.setattr(segmentation_mod, "MODEL_DOWNLOAD_TIMEOUT_S", 1.0)
    monkeypatch.setattr(segmentation_mod.time, "monotonic", lambda: next(ticks))

    with pytest.raises(ModelAcquisitionError, match="download timed out"):
        segmentation_mod._stream_model(
            spec_for(payload),
            io.BytesIO(),
            opener=lambda *_args, **_kwargs: Response(payload),
        )
