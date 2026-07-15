"""Avatar-service configuration: strict validation and patch semantics."""

import pytest

from custback.avatar.config import (
    AVATAR_PARTS,
    BUILTIN_AVATARS,
    AvatarConfig,
    AvatarRuntime,
    RestartRequiredError,
)


def test_defaults_are_valid_and_local_first():
    cfg = AvatarConfig()
    assert cfg.source.url == "ws://127.0.0.1:8710"
    assert cfg.api.host == "127.0.0.1"
    assert cfg.api.port == 8711
    assert cfg.source.token_file == "~/.config/custback/renderer-token"
    assert cfg.api.token_file != cfg.source.token_file
    assert cfg.appearance.parts == AVATAR_PARTS
    assert cfg.appearance.avatar == "casey"
    assert cfg.appearance.style == "cartoon"
    assert cfg.appearance.framing == "bust"  # head and chest for meeting tiles
    assert cfg.driver.backend == "auto"


def test_appearance_choices_are_validated():
    for name in BUILTIN_AVATARS:
        assert AvatarConfig.from_dict(
            {"appearance": {"avatar": name}}
        ).appearance.avatar == name
    with pytest.raises(ValueError, match="unknown builtin avatar"):
        AvatarConfig.from_dict({"appearance": {"avatar": "zorp"}})
    with pytest.raises(ValueError):
        AvatarConfig.from_dict({"appearance": {"style": "anime"}})
    with pytest.raises(ValueError):
        AvatarConfig.from_dict({"appearance": {"framing": "waist"}})


def test_avatar_style_and_framing_patch_hot():
    runtime = AvatarRuntime(AvatarConfig())
    state = runtime.apply_patch(
        {"appearance": {"avatar": "nova", "style": "realistic", "framing": "closeup"}}
    )
    assert state.version == 1
    assert state.config.appearance.avatar == "nova"
    assert state.config.appearance.style == "realistic"
    assert state.config.appearance.framing == "closeup"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8710",
        "ws://",
        "ws://host:8710/path",
        "ws://host:8710?x=1",
        "ws://user:pass@host:8710",
        " ws://host:8710",
    ],
)
def test_source_url_rejects_non_websocket_forms(url):
    with pytest.raises(ValueError):
        AvatarConfig.from_dict({"source": {"url": url}})


def test_source_url_accepts_wss_and_strips_trailing_slash():
    cfg = AvatarConfig.from_dict({"source": {"url": "wss://gpu-host:8710/"}})
    assert cfg.source.url == "wss://gpu-host:8710"


@pytest.mark.parametrize(
    "url",
    [
        "ws://localhost:8710",
        "ws://camera.example:8710",
        "ws://192.168.1.10:8710",
        "ws://127.0.0.1:nope",
        "ws://127.0.0.1:65536",
        "wss://[::1]:nope",
        "wss://0.0.0.0:8710",
    ],
)
def test_source_url_rejects_non_numeric_loopback_plaintext_and_bad_addresses(url):
    with pytest.raises(ValueError):
        AvatarConfig.from_dict({"source": {"url": url}})


def test_source_tls_files_must_exist_and_client_identity_is_paired(tmp_path):
    missing = tmp_path / "missing.pem"
    with pytest.raises(ValueError, match="does not exist"):
        AvatarConfig.from_dict(
            {"source": {"url": "wss://gpu.example:8710", "tls_ca_file": str(missing)}}
        )
    cert = tmp_path / "client.pem"
    cert.write_text("certificate")
    with pytest.raises(ValueError, match="configured together"):
        AvatarConfig.from_dict(
            {
                "source": {
                    "url": "wss://gpu.example:8710",
                    "tls_certfile": str(cert),
                }
            }
        )


def test_reconnect_window_must_be_ordered():
    with pytest.raises(ValueError):
        AvatarConfig.from_dict(
            {"source": {"reconnect_min_s": 5.0, "reconnect_max_s": 1.0}}
        )


@pytest.mark.parametrize(
    "parts,message",
    [
        ([], "at least one"),
        (["head", "wings"], "unknown avatar parts"),
        (["head", "head"], "unique"),
    ],
)
def test_parts_validation(parts, message):
    with pytest.raises(ValueError, match=message):
        AvatarConfig.from_dict({"appearance": {"parts": parts}})


def test_background_requires_active_paths():
    with pytest.raises(ValueError, match="image_path"):
        AvatarConfig.from_dict({"background": {"mode": "image"}})
    with pytest.raises(ValueError, match="video_path"):
        AvatarConfig.from_dict({"background": {"mode": "video"}})


def test_background_blur_kernel_normalized_to_odd():
    cfg = AvatarConfig.from_dict({"background": {"blur_strength": 30}})
    assert cfg.background.blur_strength == 31


def test_audio2face_backend_requires_url():
    with pytest.raises(ValueError, match="audio2face.url"):
        AvatarConfig.from_dict({"driver": {"backend": "audio2face"}})
    cfg = AvatarConfig.from_dict(
        {
            "driver": {
                "backend": "audio2face",
                "audio2face": {"url": "grpcs://a2f.example:52000"},
            }
        }
    )
    assert cfg.driver.audio2face.url == "grpcs://a2f.example:52000"


def test_audio2face_url_requires_explicit_secure_remote_transport():
    local = AvatarConfig.from_dict(
        {"driver": {"audio2face": {"url": "127.0.0.1:52000"}}}
    )
    assert local.driver.audio2face.url == "grpc://127.0.0.1:52000"
    with pytest.raises(ValueError):
        AvatarConfig.from_dict(
            {"driver": {"audio2face": {"url": "grpc://a2f.example:52000"}}}
        )
    with pytest.raises(ValueError):
        AvatarConfig.from_dict(
            {"driver": {"audio2face": {"url": "grpcs://a2f.example:bad"}}}
        )


def test_vision_model_path_extension():
    with pytest.raises(ValueError, match=".task"):
        AvatarConfig.from_dict({"driver": {"vision": {"model_path": "model.onnx"}}})


def test_unknown_sections_and_fields_are_rejected():
    with pytest.raises(ValueError):
        AvatarConfig.from_dict({"avatarx": {}})
    with pytest.raises(ValueError):
        AvatarConfig.from_dict({"appearance": {"scal": 1.0}})


def test_yaml_roundtrip(tmp_path):
    cfg = AvatarConfig.from_dict({"appearance": {"scale": 0.75}})
    path = tmp_path / "avatar.yaml"
    cfg.save(path)
    assert AvatarConfig.load(path).to_dict() == cfg.to_dict()


def test_runtime_hot_patch_bumps_version():
    runtime = AvatarRuntime(AvatarConfig())
    state = runtime.apply_patch({"appearance": {"scale": 0.5, "parts": ["head"]}})
    assert state.version == 1
    assert state.config.appearance.scale == 0.5
    assert state.config.appearance.parts == ("head",)


def test_runtime_noop_patch_preserves_version():
    runtime = AvatarRuntime(AvatarConfig())
    state = runtime.apply_patch({"appearance": {"scale": 1.0}})
    assert state.version == 0


def test_runtime_restart_sections_are_rejected():
    runtime = AvatarRuntime(AvatarConfig())
    with pytest.raises(RestartRequiredError) as excinfo:
        runtime.apply_patch({"api": {"port": 9000}})
    assert "api.port" in excinfo.value.fields
    with pytest.raises(RestartRequiredError):
        runtime.apply_patch({"source": {"url": "wss://other.example:8710"}})


def test_audio2face_destination_and_trust_are_restart_only():
    runtime = AvatarRuntime(AvatarConfig())
    with pytest.raises(RestartRequiredError) as excinfo:
        runtime.apply_patch(
            {"driver": {"audio2face": {"url": "grpcs://a2f.example:52000"}}}
        )
    assert excinfo.value.fields == ("driver.audio2face.url",)


def test_runtime_mixed_patch_applies_nothing():
    runtime = AvatarRuntime(AvatarConfig())
    with pytest.raises(RestartRequiredError):
        runtime.apply_patch(
            {"appearance": {"scale": 0.5}, "api": {"port": 9000}}
        )
    state = runtime.read()
    assert state.version == 0
    assert state.config.appearance.scale == 1.0


def test_runtime_invalid_patch_changes_nothing():
    runtime = AvatarRuntime(AvatarConfig())
    with pytest.raises(ValueError):
        runtime.apply_patch({"appearance": {"scale": 99}})
    assert runtime.version == 0
