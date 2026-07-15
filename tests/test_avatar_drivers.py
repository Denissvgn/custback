"""Animation drivers: idle determinism, smoothing, Audio2Face mapping."""

import math
import ssl
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
    DriverUnavailableError,
    IdleDriver,
    ThreadSafeLatestState,
    create_driver,
    idle_blink,
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


def test_audio2face_driver_requires_bindings(tmp_path):
    try:
        import nvidia_ace  # noqa: F401
        pytest.skip("nvidia-ace installed; unavailability path not testable")
    except ImportError:
        pass
    wav = tmp_path / "voice.wav"
    _write_wav(wav)
    driver = Audio2FaceDriver(
        Audio2FaceConfig(url="grpc://127.0.0.1:52000", audio_source=str(wav))
    )
    with pytest.raises(DriverUnavailableError, match="nvidia-ace"):
        driver.start()


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
