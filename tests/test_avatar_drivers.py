"""Animation drivers: idle determinism, smoothing, Audio2Face mapping."""

import builtins
import math
import ssl
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import custback.avatar.drivers as drivers_mod
from custback.avatar.audio2face import (
    Audio2FaceDriver,
    WavAudioSource,
    arkit_channel_name,
    map_blendshape_frame,
)
from custback.avatar.config import Audio2FaceConfig, DriverConfig
from custback.api.security import SecurityConfigurationError
from custback.avatar.drivers import (
    FACE_LANDMARKER_MODEL,
    DriverStartupError,
    DriverUnavailableError,
    IdleDriver,
    ThreadSafeLatestState,
    create_driver,
    idle_blink,
    prepare_driver,
)
from custback.avatar.state import ARKIT_BLENDSHAPES, FaceState, StateSmoother


def test_idle_driver_is_deterministic_and_present():
    driver = IdleDriver()
    first = driver.update(None, 2.5)
    second = driver.update(None, 2.5)
    assert first.present and second.present
    assert first.blendshapes == second.blendshapes
    assert (first.yaw, first.pitch, first.roll) == (second.yaw, second.pitch, second.roll)
    assert abs(first.yaw) < 0.1  # gentle sway only


def test_idle_driver_blinks_periodically():
    driver = IdleDriver()
    blinks = [
        driver.update(None, t / 100.0).channel("eyeBlinkLeft")
        for t in range(0, 800)
    ]
    assert max(blinks) > 0.9  # a full blink happens
    assert min(blinks) == 0.0  # and the eyes reopen
    assert idle_blink(0.11) == pytest.approx(1.0, abs=0.01)


def test_state_smoother_damps_pose_but_keeps_blinks_sharp():
    smoother = StateSmoother(0.5)
    first = FaceState.neutral()
    first.yaw = 1.0
    smoother.apply(first)
    second = FaceState.neutral()
    second.set_channel("eyeBlinkLeft", 1.0)
    smoothed = smoother.apply(second)
    assert smoothed.yaw == pytest.approx(0.5)
    assert smoothed.channel("eyeBlinkLeft") == 1.0  # instant upward blink
    third = FaceState.neutral()
    relaxed = smoother.apply(third)
    assert 0 < relaxed.channel("eyeBlinkLeft") < 1.0  # decays smoothly


def test_state_smoother_validates_factor():
    with pytest.raises(ValueError):
        StateSmoother(0.99)


def test_face_state_rejects_unknown_channels():
    state = FaceState.neutral()
    with pytest.raises(KeyError):
        state.set_channel("noSuchChannel", 1.0)
    with pytest.raises(KeyError):
        state.channel("noSuchChannel")
    state.set_channel("jawOpen", 7.0)
    assert state.channel("jawOpen") == 1.0  # clamped
    state.set_channel("jawOpen", math.nan)
    assert state.channel("jawOpen") == 0.0


def test_create_driver_idle_and_auto_fallback():
    assert create_driver(DriverConfig(backend="idle")).name == "idle"
    pytest.importorskip("cv2")
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        # auto degrades to idle instead of failing the service.
        assert create_driver(DriverConfig(backend="auto")).name == "idle"
        with pytest.raises(DriverUnavailableError, match="mediapipe"):
            create_driver(DriverConfig(backend="vision"))


def test_face_landmarker_model_is_pinned():
    assert FACE_LANDMARKER_MODEL.url.startswith("https://")
    assert FACE_LANDMARKER_MODEL.filename == "face_landmarker.task"
    assert FACE_LANDMARKER_MODEL.size > 0
    assert len(FACE_LANDMARKER_MODEL.sha256) == 64


def _fake_vision_bindings():
    captured = {}

    class BaseOptions:
        def __init__(self, *, model_asset_path):
            captured["model_asset_path"] = model_asset_path

    class FaceLandmarkerOptions:
        def __init__(self, **kwargs):
            captured["options"] = kwargs

    class Landmarker:
        def close(self):
            captured["closed"] = True

    class FaceLandmarker:
        @staticmethod
        def create_from_options(options):
            captured["created_with"] = options
            return Landmarker()

    bindings = drivers_mod._VisionBindings(
        mediapipe=SimpleNamespace(),
        tasks=SimpleNamespace(BaseOptions=BaseOptions),
        vision=SimpleNamespace(
            FaceLandmarkerOptions=FaceLandmarkerOptions,
            FaceLandmarker=FaceLandmarker,
            RunningMode=SimpleNamespace(VIDEO="video"),
        ),
    )
    return bindings, captured


def test_prepared_managed_model_is_acquired_once_and_reused(
    monkeypatch, tmp_path
):
    model = tmp_path / "managed.task"
    model.write_bytes(b"model")
    bindings, captured = _fake_vision_bindings()
    acquisitions = []

    def acquire(spec, *, allow_download):
        acquisitions.append((spec, allow_download))
        return model

    monkeypatch.setattr(drivers_mod, "_load_vision_bindings", lambda: bindings)
    monkeypatch.setattr(drivers_mod, "acquire_model", acquire)
    cfg = DriverConfig(backend="vision")

    preparation = prepare_driver(cfg, allow_model_download=False)
    assert preparation.vision_model_path == model
    assert acquisitions == [(FACE_LANDMARKER_MODEL, False)]

    monkeypatch.setattr(
        drivers_mod,
        "acquire_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepared construction must not reacquire the model")
        ),
    )
    driver = create_driver(cfg, preparation=preparation)
    assert captured["model_asset_path"] == str(model)
    driver.close()
    assert captured["closed"] is True


def test_prepared_custom_model_is_not_revalidated_on_render_lane(
    monkeypatch, tmp_path
):
    model = tmp_path / "custom.task"
    model.write_bytes(b"custom")
    bindings, captured = _fake_vision_bindings()
    monkeypatch.setattr(drivers_mod, "_load_vision_bindings", lambda: bindings)
    monkeypatch.setattr(
        drivers_mod,
        "acquire_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a custom model must not use managed acquisition")
        ),
    )
    cfg = DriverConfig(
        backend="vision", vision={"model_path": str(model)}
    )

    preparation = prepare_driver(cfg)
    model.unlink()  # construction consumes the prepared result without stat I/O
    driver = create_driver(cfg, preparation=preparation)
    assert captured["model_asset_path"] == str(model)
    driver.close()


def test_prepare_driver_preserves_auto_fallback_for_missing_custom_model(
    monkeypatch, tmp_path
):
    missing = tmp_path / "missing.task"
    monkeypatch.setattr(
        drivers_mod,
        "_load_vision_bindings",
        lambda: (_ for _ in ()).throw(
            AssertionError("custom path validation must happen before imports")
        ),
    )
    auto = DriverConfig(
        backend="auto", vision={"model_path": str(missing)}
    )
    preparation = prepare_driver(auto)
    assert create_driver(auto, preparation=preparation).name == "idle"

    vision = DriverConfig(
        backend="vision", vision={"model_path": str(missing)}
    )
    with pytest.raises(DriverUnavailableError, match="does not exist"):
        prepare_driver(vision)


def test_prepare_auto_does_not_acquire_managed_model_without_bindings(monkeypatch):
    monkeypatch.setattr(
        drivers_mod,
        "_load_vision_bindings",
        lambda: (_ for _ in ()).throw(
            DriverUnavailableError("mediapipe unavailable")
        ),
    )
    monkeypatch.setattr(
        drivers_mod,
        "acquire_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unusable backends must not trigger a model download")
        ),
    )
    cfg = DriverConfig(backend="auto")

    preparation = prepare_driver(cfg)
    assert create_driver(cfg, preparation=preparation).name == "idle"


def test_create_driver_rejects_preparation_for_a_different_candidate(
    monkeypatch, tmp_path
):
    first = tmp_path / "first.task"
    second = tmp_path / "second.task"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    bindings, _captured = _fake_vision_bindings()
    monkeypatch.setattr(drivers_mod, "_load_vision_bindings", lambda: bindings)
    original = DriverConfig(
        backend="vision", vision={"model_path": str(first)}
    )
    preparation = prepare_driver(original)
    changed = DriverConfig(
        backend="vision", vision={"model_path": str(second)}
    )

    with pytest.raises(ValueError, match="does not match"):
        create_driver(changed, preparation=preparation)


def test_arkit_channel_name_normalizes_audio2face_names():
    assert arkit_channel_name("EyeBlinkLeft") == "eyeBlinkLeft"
    assert arkit_channel_name("JawOpen") == "jawOpen"
    assert arkit_channel_name("TongueOut") == "tongueOut"
    assert arkit_channel_name("eyeBlinkLeft") == "eyeBlinkLeft"
    assert arkit_channel_name("EmotionJoy") is None
    assert arkit_channel_name("") is None


def test_map_blendshape_frame_filters_and_clamps():
    weights = map_blendshape_frame(
        ["JawOpen", "Unknown", "EyeBlinkRight"], [1.7, 0.5, -0.2]
    )
    assert weights == {"jawOpen": 1.0, "eyeBlinkRight": 0.0}
    assert set(weights) <= set(ARKIT_BLENDSHAPES)


def test_thread_safe_latest_state_snapshot():
    shared = ThreadSafeLatestState()
    assert shared.snapshot() == ({}, None)
    shared.update_blendshapes({"jawOpen": 0.4}, 12.0)
    shared.update_blendshapes({"eyeBlinkLeft": 1.0}, 13.0)
    weights, updated_at = shared.snapshot()
    assert weights == {"jawOpen": 0.4, "eyeBlinkLeft": 1.0}
    assert updated_at == 13.0


def _write_wav(path, *, rate=16000, channels=1, width=2, frames=1600):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        tone = (
            np.sin(np.linspace(0, 40 * np.pi, frames)) * 12_000
        ).astype("<i2")
        if channels == 2:
            tone = np.repeat(tone, 2)
        wav.writeframes(tone.tobytes())


def test_wav_audio_source_reads_and_loops(tmp_path):
    path = tmp_path / "voice.wav"
    _write_wav(path, frames=100)
    source = WavAudioSource(path, 16000)
    chunk = source.read(60)
    assert len(chunk) == 120
    looped = source.read(80)  # crosses the end of the file
    assert len(looped) == 160
    source.close()


def test_wav_audio_source_rejects_wrong_formats(tmp_path):
    stereo = tmp_path / "stereo.wav"
    _write_wav(stereo, channels=2)
    with pytest.raises(DriverUnavailableError, match="mono"):
        WavAudioSource(stereo, 16000)
    slow = tmp_path / "slow.wav"
    _write_wav(slow, rate=8000)
    with pytest.raises(DriverUnavailableError, match="16000"):
        WavAudioSource(slow, 16000)


def test_audio2face_driver_requires_bindings(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def without_service_bindings(name, *args, **kwargs):
        if name.startswith("nvidia_audio2face_3d"):
            raise ImportError("simulated missing Audio2Face service bindings")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_service_bindings)
    wav = tmp_path / "voice.wav"
    _write_wav(wav)
    driver = Audio2FaceDriver(
        Audio2FaceConfig(url="grpc://127.0.0.1:52000", audio_source=str(wav))
    )
    with pytest.raises(DriverUnavailableError, match="nvidia-audio2face-3d"):
        driver.start()


def test_audio2face_start_failure_closes_the_prepared_source(monkeypatch):
    import custback.avatar.audio2face as a2f

    class Source:
        closes = 0

        def close(self):
            self.closes += 1

    source = Source()

    class FailedThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread quota exhausted")

    monkeypatch.setattr(a2f, "_load_protocol", lambda: object())
    monkeypatch.setattr(a2f, "create_audio_source", lambda _cfg: source)
    monkeypatch.setattr(a2f.threading, "Thread", FailedThread)
    driver = Audio2FaceDriver(Audio2FaceConfig(url="grpc://127.0.0.1:52000"))

    with pytest.raises(RuntimeError, match="thread quota"):
        driver.start()

    assert source.closes == 1
    assert driver._source is None
    assert driver._worker is None
    driver.close()
    assert source.closes == 1


def test_audio2face_thread_construction_failure_closes_source(monkeypatch):
    import custback.avatar.audio2face as a2f

    class Source:
        closes = 0

        def close(self):
            self.closes += 1

    source = Source()

    class FailedThread:
        def __init__(self, **_kwargs):
            raise RuntimeError("cannot allocate worker")

    monkeypatch.setattr(a2f, "_load_protocol", lambda: object())
    monkeypatch.setattr(a2f, "create_audio_source", lambda _cfg: source)
    monkeypatch.setattr(a2f.threading, "Thread", FailedThread)
    driver = Audio2FaceDriver(Audio2FaceConfig(url="grpc://127.0.0.1:52000"))

    with pytest.raises(RuntimeError, match="allocate worker"):
        driver.start()
    assert source.closes == 1
    driver.close()
    assert source.closes == 1


def test_audio2face_factory_transfers_failed_start_cleanup_ownership(monkeypatch):
    import custback.avatar.audio2face as a2f

    allow_close = threading.Event()

    class Source:
        closes = 0

        def close(self):
            self.closes += 1
            if not allow_close.is_set():
                raise RuntimeError("capture device still owns native state")

    source = Source()

    class FailedThread:
        def __init__(self, **_kwargs):
            raise RuntimeError("cannot allocate worker")

    monkeypatch.setattr(a2f, "_load_protocol", lambda: object())
    monkeypatch.setattr(a2f, "create_audio_source", lambda _cfg: source)
    monkeypatch.setattr(a2f.threading, "Thread", FailedThread)
    cfg = DriverConfig(
        backend="audio2face",
        audio2face=Audio2FaceConfig(url="grpc://127.0.0.1:52000"),
    )

    with pytest.raises(DriverStartupError) as caught:
        create_driver(cfg)

    # start() and the factory both tried; the concrete failed driver remains
    # available to the transactional caller until native teardown succeeds.
    assert source.closes == 2
    failed_driver = caught.value.driver
    with pytest.raises(DriverUnavailableError, match="interruption failed"):
        failed_driver.close()
    allow_close.set()
    failed_driver.close()
    assert source.closes == 4
    assert failed_driver._survivors == []


def test_audio2face_concurrent_close_waits_for_one_terminal_operation():
    entered = threading.Event()
    release = threading.Event()
    returns = []

    class Source:
        closes = 0

        def close(self):
            self.closes += 1
            entered.set()
            release.wait()

    source = Source()
    driver = Audio2FaceDriver(Audio2FaceConfig(url="grpc://127.0.0.1:52000"))
    driver._source = source

    def close(label):
        driver.close()
        returns.append(label)

    first = threading.Thread(target=close, args=("first",))
    second = threading.Thread(target=close, args=("second",))
    try:
        first.start()
        assert entered.wait(1.0)
        second.start()
        time.sleep(0.03)
        assert returns == []
        release.set()
        first.join(1.0)
        second.join(1.0)
        assert not first.is_alive() and not second.is_alive()
        assert returns == ["first", "second"]
        assert source.closes == 1
    finally:
        release.set()
        first.join(1.0)
        second.join(1.0)


def test_audio2face_close_bounds_and_retains_blocked_native_interrupt(monkeypatch):
    import custback.avatar.audio2face as a2f

    entered = threading.Event()
    release = threading.Event()

    class Source:
        closes = 0

        def close(self):
            self.closes += 1
            entered.set()
            release.wait()

    source = Source()
    driver = Audio2FaceDriver(Audio2FaceConfig(url="grpc://127.0.0.1:52000"))
    driver._source = source
    monkeypatch.setattr(a2f, "_CLOSE_TIMEOUT_S", 0.05)

    started = time.monotonic()
    try:
        with pytest.raises(DriverUnavailableError, match="did not stop"):
            driver.close()
        assert entered.is_set()
        assert time.monotonic() - started < 0.5
        assert len(driver._interruptions) == 1
        assert driver._interruptions[0].resource is source

        release.set()
        driver.close()
        assert source.closes == 1
        assert driver._interruptions == []
    finally:
        release.set()


def test_audio2face_interruption_completion_race_retains_identity():
    import custback.avatar.audio2face as audio2face_mod

    pending = audio2face_mod._PendingInterruption("source", object())

    class CompletingEvent:
        def __init__(self):
            self.calls = 0
            self.completed = False

        def is_set(self):
            self.calls += 1
            if self.calls == 2:
                # Completion lands immediately after the classifier's state
                # observation. A second read would see True and could drop it.
                pending.error = RuntimeError("native close failed")
                self.completed = True
                return False
            return self.completed

        def wait(self, _timeout=None):
            return self.completed

    pending.done = CompletingEvent()

    failures, active = Audio2FaceDriver._settle_interruptions(
        [pending], time.monotonic()
    )

    assert failures == []
    assert active == [pending]


def test_audio2face_close_interrupts_pacing_wait():
    entered = threading.Event()
    finished = threading.Event()

    class AudioHeader:
        AUDIO_FORMAT_PCM = 1

        def __init__(self, **_kwargs):
            pass

    protocol = SimpleNamespace(
        audio_header=AudioHeader,
        audio_stream_header=lambda **kwargs: kwargs,
        audio_stream=lambda **kwargs: kwargs,
        audio_with_emotion=lambda **kwargs: kwargs,
    )

    class EmptySource:
        def read(self, _frames):
            entered.set()
            return b""

    driver = Audio2FaceDriver(
        Audio2FaceConfig(url="grpc://127.0.0.1:52000", chunk_ms=1000)
    )
    requests = driver._requests(protocol, EmptySource())
    next(requests)  # protocol header

    def request_audio():
        try:
            next(requests)
        except StopIteration:
            pass
        finally:
            finished.set()

    worker = threading.Thread(target=request_audio)
    worker.start()
    assert entered.wait(1.0)
    started = time.monotonic()
    driver.close()
    worker.join(0.75)

    assert finished.is_set()
    assert time.monotonic() - started < 0.75


def test_audio2face_close_interrupts_source_before_rpc_registration(monkeypatch):
    import custback.avatar.audio2face as a2f

    read_entered = threading.Event()
    read_released = threading.Event()
    requests_stopped = threading.Event()
    events = []

    class Source:
        def read(self, _frames):
            read_entered.set()
            read_released.wait()
            return b""

        def close(self):
            events.append("source.close")
            read_released.set()

    source = Source()

    class AudioHeader:
        AUDIO_FORMAT_PCM = 1

        def __init__(self, **_kwargs):
            pass

    class Channel:
        def close(self):
            events.append("channel.close")

    class Stub:
        def __init__(self, _channel):
            pass

        def ProcessAudioStream(self, requests):
            next(requests)  # protocol header
            try:
                next(requests)  # blocked source read, interrupted by close()
            except StopIteration:
                requests_stopped.set()
            return ()

    protocol = SimpleNamespace(
        grpc=SimpleNamespace(
            insecure_channel=lambda _url, **_kwargs: Channel()
        ),
        stub_class=Stub,
        audio_header=AudioHeader,
        audio_stream_header=lambda **kwargs: kwargs,
        audio_stream=lambda **kwargs: kwargs,
        audio_with_emotion=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(a2f, "_load_protocol", lambda: protocol)
    monkeypatch.setattr(a2f, "create_audio_source", lambda _cfg: source)
    driver = Audio2FaceDriver(Audio2FaceConfig(url="grpc://127.0.0.1:52000"))
    driver.start()
    assert read_entered.wait(1.0)
    assert driver._worker is not None
    worker = driver._worker

    driver.close()

    assert not worker.is_alive()
    assert requests_stopped.is_set()
    assert events == ["source.close", "channel.close"]
    driver.close()
    assert events == ["source.close", "channel.close"]


def test_audio2face_release_failure_is_terminal_and_retryable():
    class Channel:
        closes = 0

        def close(self):
            self.closes += 1
            if self.closes == 1:
                raise RuntimeError("native channel survived")

    channel = Channel()
    driver = Audio2FaceDriver(
        Audio2FaceConfig(url="grpc://127.0.0.1:52000")
    )
    driver._channel = channel

    driver._release_channel(channel)

    assert driver._stop.is_set()  # reconnect is terminal after teardown failure
    assert driver._channel is None
    assert [(kind, resource) for kind, resource, _error in driver._survivors] == [
        ("channel", channel)
    ]
    driver.close()  # retries the retained native identity
    assert channel.closes == 2
    assert driver._survivors == []
    driver.close()
    assert channel.closes == 2


def test_audio2face_rejected_call_cancel_failure_is_reclaimed_on_close():
    class Channel:
        closes = 0

        def close(self):
            self.closes += 1

    channel = Channel()

    class Call:
        cancels = 0

        def cancel(self):
            self.cancels += 1
            if self.cancels == 1:
                raise RuntimeError("native call survived")

    call = Call()
    driver = Audio2FaceDriver(
        Audio2FaceConfig(url="grpc://127.0.0.1:52000")
    )

    class Stub:
        def __init__(self, _channel):
            pass

        def ProcessAudioStream(self, _requests):
            # Force the registration rejection after the native call exists.
            driver._stop.set()
            return call

    protocol = SimpleNamespace(
        grpc=SimpleNamespace(
            insecure_channel=lambda _url, **_kwargs: channel
        ),
        stub_class=Stub,
    )

    driver._run_session(protocol, object())

    assert call.cancels == 1
    assert channel.closes == 1
    assert [(kind, resource) for kind, resource, _error in driver._survivors] == [
        ("call", call)
    ]
    driver.close()
    assert call.cancels == 2
    assert driver._survivors == []


def test_audio2face_driver_renders_fresh_weights_and_idles_when_stale(monkeypatch):
    driver = Audio2FaceDriver(Audio2FaceConfig(url="grpc://127.0.0.1:52000"))
    state = driver.update(None, 5.0)
    assert state.present  # neutral avatar rather than absence
    assert state.channel("jawOpen") == 0.0

    import custback.avatar.audio2face as a2f

    monkeypatch.setattr(a2f.time, "monotonic", lambda: 100.0)
    driver._state.update_blendshapes({"jawOpen": 0.8}, 100.0)
    fresh = driver.update(None, 5.0)
    assert fresh.channel("jawOpen") == 0.8
    assert driver.healthy

    monkeypatch.setattr(a2f.time, "monotonic", lambda: 200.0)
    stale = driver.update(None, 6.0)
    assert stale.channel("jawOpen") == 0.0
    assert not driver.healthy


def test_audio2face_remote_session_uses_verified_secure_channel():
    calls = []

    class Channel:
        def close(self):
            calls.append(("close",))

    class Grpc:
        @staticmethod
        def ssl_channel_credentials(**kwargs):
            calls.append(("credentials", kwargs))
            return "verified-credentials"

        @staticmethod
        def secure_channel(target, credentials, *, options):
            calls.append(("secure", target, credentials, options))
            return Channel()

        @staticmethod
        def insecure_channel(_target, **_kwargs):
            raise AssertionError("remote Audio2Face must never downgrade to plaintext")

    class Stub:
        def __init__(self, _channel):
            pass

        def ProcessAudioStream(self, _requests):
            return ()

    protocol = SimpleNamespace(grpc=Grpc, stub_class=Stub)
    driver = Audio2FaceDriver(Audio2FaceConfig(url="grpcs://a2f.example:52000"))
    driver._run_session(protocol, SimpleNamespace())

    assert (
        "secure",
        "a2f.example:52000",
        "verified-credentials",
        (("grpc.enable_http_proxy", 0),),
    ) in calls
    credential_call = next(call for call in calls if call[0] == "credentials")
    assert credential_call[1] == {
        "root_certificates": None,
        "private_key": None,
        "certificate_chain": None,
    }


def test_audio2face_loopback_channel_disables_environment_proxying():
    calls = []

    class Channel:
        def close(self):
            pass

    class Grpc:
        @staticmethod
        def insecure_channel(target, *, options):
            calls.append((target, options))
            return Channel()

        @staticmethod
        def secure_channel(*_args, **_kwargs):
            raise AssertionError("numeric loopback grpc must remain local")

    class Stub:
        def __init__(self, _channel):
            pass

        def ProcessAudioStream(self, _requests):
            return ()

    driver = Audio2FaceDriver(
        Audio2FaceConfig(url="grpc://127.0.0.1:52000")
    )
    driver._run_session(SimpleNamespace(grpc=Grpc, stub_class=Stub), object())

    assert calls == [
        ("127.0.0.1:52000", (("grpc.enable_http_proxy", 0),))
    ]


def test_audio2face_rejects_malformed_ca_eagerly_without_disclosing_path(tmp_path):
    ca_file = tmp_path / "operator-private-location" / "broken-ca.pem"
    ca_file.parent.mkdir()
    ca_file.write_text("not a certificate")
    cfg = Audio2FaceConfig(
        url="grpcs://a2f.example:52000",
        tls_ca_file=str(ca_file),
    )

    with pytest.raises(SecurityConfigurationError) as caught:
        Audio2FaceDriver(cfg)

    assert "could not be loaded or verified" in str(caught.value)
    assert str(ca_file) not in str(caught.value)

    system_ca = ssl.get_default_verify_paths().cafile
    if not system_ca:
        return
    cert_file = tmp_path / "client-certificate.pem"
    cert_file.write_bytes(Path(system_ca).read_bytes())
    key_file = tmp_path / "client-private-key.pem"
    key_file.write_text("not the certificate private key")
    mismatched = Audio2FaceConfig(
        url="grpcs://a2f.example:52000",
        tls_certfile=str(cert_file),
        tls_keyfile=str(key_file),
    )
    with pytest.raises(SecurityConfigurationError) as caught:
        Audio2FaceDriver(mismatched)
    assert str(cert_file) not in str(caught.value)
    assert str(key_file) not in str(caught.value)


def test_audio2face_snapshots_tls_trust_before_reconnects(tmp_path):
    system_ca = ssl.get_default_verify_paths().cafile
    if not system_ca:
        pytest.skip("the test interpreter has no default CA bundle")
    ca_bytes = Path(system_ca).read_bytes()
    ca_file = tmp_path / "a2f-ca.pem"
    ca_file.write_bytes(ca_bytes)
    driver = Audio2FaceDriver(
        Audio2FaceConfig(
            url="grpcs://a2f.example:52000",
            tls_ca_file=str(ca_file),
        )
    )
    ca_file.unlink()
    calls = []

    class Channel:
        def close(self):
            calls.append(("close",))

    class Grpc:
        @staticmethod
        def ssl_channel_credentials(**kwargs):
            calls.append(("credentials", kwargs))
            return "snapshotted-credentials"

        @staticmethod
        def secure_channel(target, credentials, *, options):
            calls.append(("secure", target, credentials, options))
            return Channel()

        @staticmethod
        def insecure_channel(_target, **_kwargs):
            raise AssertionError("remote Audio2Face must never downgrade")

    class Stub:
        def __init__(self, _channel):
            pass

        def ProcessAudioStream(self, _requests):
            return ()

    driver._run_session(SimpleNamespace(grpc=Grpc, stub_class=Stub), object())

    credential_call = next(call for call in calls if call[0] == "credentials")
    assert credential_call[1]["root_certificates"] == ca_bytes
    assert (
        "secure",
        "a2f.example:52000",
        "snapshotted-credentials",
        (("grpc.enable_http_proxy", 0),),
    ) in calls
