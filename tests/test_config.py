import threading
import time

import pytest

from custback.config import (
    AVATAR_PROXY_RESTART_ONLY_FIELDS,
    AppConfig,
    ConfigVersionConflictError,
    RuntimeConfig,
)


def test_avatar_proxy_security_boundary_fields_are_restart_only():
    assert AVATAR_PROXY_RESTART_ONLY_FIELDS == {
        "avatar.url",
        "avatar.token_file",
        "avatar.tls_ca_file",
        "avatar.tls_certfile",
        "avatar.tls_keyfile",
    }


def test_defaults_valid():
    cfg = AppConfig()
    cfg.validate()
    assert cfg.background.mode == "blur"
    assert cfg.camera.pixel_format == "auto"
    assert cfg.camera.mode_mismatch == "warn"
    assert cfg.camera.recovery_timeout_s == 10.0
    assert cfg.api.renderer_token_file == "~/.config/custback/renderer-token"


def test_round_trip_yaml(tmp_path):
    cfg = AppConfig()
    cfg.background.mode = "color"
    cfg.background.color = (1, 2, 3)
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    loaded = AppConfig.load(path)
    assert loaded.background.mode == "color"
    assert tuple(loaded.background.color) == (1, 2, 3)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        AppConfig.from_dict({"background": {"mode": "hologram"}})


@pytest.mark.parametrize(
    "background",
    [
        {"mode": "image"},
        {"mode": "video"},
        {"mode": "camera"},
        {"mode": "remote", "remote_fallback_mode": "image"},
        {"mode": "remote", "remote_fallback_mode": "video"},
        {"mode": "remote", "remote_fallback_mode": "camera"},
    ],
)
def test_active_background_source_is_required(background):
    with pytest.raises(ValueError, match="required"):
        AppConfig.from_dict({"background": background})


def test_failed_cross_field_assignment_rolls_back_object_state():
    cfg = AppConfig()
    with pytest.raises(ValueError):
        cfg.background.mode = "image"
    assert cfg.background.mode == "blur"
    assert cfg.background.image_path == ""

    with pytest.raises(ValueError):
        cfg.api.tls_certfile = "cert.pem"
    assert cfg.api.tls_certfile == ""
    assert cfg.api.tls_keyfile == ""
    cfg.validate()


def test_active_background_paths_are_structural_not_existence_checks():
    cfg = AppConfig.from_dict(
        {"background": {"mode": "image", "image_path": "/not-created/office.jpg"}}
    )
    assert cfg.background.image_path == "/not-created/office.jpg"

    remote = AppConfig.from_dict(
        {
            "background": {
                "mode": "remote",
                "remote_fallback_mode": "camera",
                "camera_device": 2,
            }
        }
    )
    assert remote.background.camera_device == 2


def test_even_blur_made_odd():
    cfg = AppConfig.from_dict({"background": {"blur_strength": 20}})
    assert cfg.background.blur_strength % 2 == 1
    assert AppConfig.from_dict({"background": {"blur_strength": 2}}).background.blur_strength == 3
    with pytest.raises(ValueError):
        AppConfig.from_dict({"background": {"blur_strength": 0}})


def test_runtime_public_mutations_are_disabled_before_they_can_split_resources():
    runtime = RuntimeConfig(AppConfig())
    with pytest.raises(RuntimeError, match="pipeline reconfiguration coordinator"):
        runtime.update({"background": {"mode": "passthrough"}})
    base = runtime.read()
    candidate = base.config.patched({"background": {"mode": "color"}})
    with pytest.raises(RuntimeError, match="pipeline reconfiguration coordinator"):
        runtime.commit(candidate, base.version)
    with pytest.raises(RuntimeError, match="pipeline reconfiguration coordinator"):
        runtime.commit_with_activation(candidate, base.version, lambda _version: None)
    assert runtime.version == 0
    assert runtime.snapshot().background.mode == "blur"


def test_new_quality_fields_defaults():
    cfg = AppConfig()
    assert cfg.segmentation.edge_refine is True
    assert cfg.segmentation.mask_shift == 0
    assert cfg.compositing.light_wrap == 0.25
    assert cfg.compositing.use_model_foreground is True


def test_invalid_delegate_rejected():
    with pytest.raises(ValueError):
        AppConfig.from_dict({"segmentation": {"delegate": "tpu"}})


@pytest.mark.parametrize(
    "data",
    [
        {"camera": {"device": -1}},
        {"camera": {"pixel_format": "h264"}},
        {"camera": {"mode_mismatch": "ignore"}},
        {"camera": {"recovery_timeout_s": 2.0}},
        {"background": {"image_path": " office.jpg"}},
        {"background": {"video_path": "clip.mp4\x00ignored"}},
        {"output": {"device": "/dev/video10\n"}},
        {"api": {"token_file": "  "}},
        {"api": {"renderer_token_file": ""}},
        {"api": {"tls_certfile": "cert.pem\t", "tls_keyfile": "key.pem"}},
        {"avatar": {"tls_ca_file": "ca.pem\n"}},
        {"segmentation": {"backend": "rvm", "model_path": "model.tflite"}},
        {"segmentation": {"backend": "mediapipe", "model_path": "model.onnx"}},
        {"segmentation": {"backend": "none", "model_path": "model.onnx"}},
        {"segmentation": {"backend": "rvm", "delegate": "gpu"}},
        {"background": {"image_path": "x" * 4097}},
    ],
)
def test_invalid_devices_paths_and_model_extensions_are_rejected(data):
    with pytest.raises(ValueError):
        AppConfig.from_dict(data)


def test_api_host_and_origins_are_validated_and_canonicalized():
    cfg = AppConfig.from_dict(
        {
            "api": {
                "host": "[0:0:0:0:0:0:0:1]",
                "allowed_origins": ["HTTP://LOCALHOST:80/", "http://[::1]:80"],
            }
        }
    )
    assert cfg.api.host == "::1"
    assert cfg.api.allowed_origins == ("http://localhost", "http://[::1]")

    for api in (
        {"host": "localhost:8710"},
        {"host": "https://localhost"},
        {"allowed_origins": ["https://example.com/path"]},
        {"allowed_origins": ["https://example.com:70000"]},
        {
            "allowed_origins": [
                "https://EXAMPLE.com:443",
                "https://example.com",
            ]
        },
    ):
        with pytest.raises(ValueError):
            AppConfig.from_dict({"api": api})


def test_falsey_non_mapping_config_roots_are_rejected(tmp_path):
    with pytest.raises(TypeError, match="mapping"):
        AppConfig.from_dict([])  # type: ignore[arg-type]
    path = tmp_path / "bad.yaml"
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="mapping"):
        AppConfig.load(path)


@pytest.mark.parametrize(
    "patch",
    [
        {"segmentation": {"rvm_downsample": 0.01}},
        {"segmentation": {"mask_shift": 100}},
        {"compositing": {"light_wrap": 5.0}},
        {"camera": {"width": "1280"}},  # strict: no string coercion
        {"segmentation": {"backend": "medipipe"}},
        {"unknown": {}},
    ],
)
def test_invalid_or_coerced_quality_fields_rejected(patch):
    with pytest.raises(ValueError):
        AppConfig.from_dict(patch)


def test_new_security_upload_and_remote_fallback_defaults():
    cfg = AppConfig()
    assert cfg.background.remote_fallback_mode == "blur"
    assert cfg.api.allow_non_loopback is False
    assert cfg.api.ws_max_bytes == 16 * 1024 * 1024
    assert cfg.api.uploads.video_max_bytes == 256 * 1024 * 1024
    with pytest.raises(ValueError):
        AppConfig.from_dict(
            {"background": {"remote_fallback_mode": "passthrough"}}
        )


def test_atomic_read_and_compare_and_swap_commit():
    runtime = RuntimeConfig(AppConfig())
    writer = runtime._coordinator_writer()
    base = runtime.read()
    candidate = base.config.patched({"background": {"mode": "color"}})
    committed = writer.commit(candidate, base.version)
    assert committed.version == 1
    assert committed.config.background.mode == "color"
    with pytest.raises(ConfigVersionConflictError):
        writer.commit(candidate, base.version)


def test_commit_with_activation_hides_candidate_until_resource_swap():
    runtime = RuntimeConfig(AppConfig())
    writer = runtime._coordinator_writer()
    base = runtime.read()
    candidate = base.config.patched({"background": {"mode": "color"}})
    entered = threading.Event()
    release = threading.Event()
    effective = {"mode": "blur"}

    def activate(_version):
        entered.set()
        release.wait(1.0)
        effective["mode"] = "color"

    committer = threading.Thread(
        target=lambda: writer.commit_with_activation(
            candidate, base.version, activate
        )
    )
    committer.start()
    assert entered.wait(1.0)

    observed = []
    reader = threading.Thread(target=lambda: observed.append(runtime.read()))
    reader.start()
    time.sleep(0.03)
    assert reader.is_alive()  # read is excluded during the effective swap
    assert effective["mode"] == "blur"

    release.set()
    committer.join(1.0)
    reader.join(1.0)
    assert effective["mode"] == "color"
    assert observed[0].config.background.mode == "color"
    assert observed[0].version == 1
