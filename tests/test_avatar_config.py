"""Avatar-service configuration: strict validation and patch semantics."""

import io
import json
import os
import ssl
import threading
import time
from pathlib import Path

import pytest

from custback.avatar.__main__ import (
    EXIT_CONFIG,
    _storage_permission_command,
    avatar_config_bytes,
    build_parser,
    config_from_args,
    export_avatar_config,
)
from custback.avatar.config import (
    AVATAR_PARTS,
    BUILTIN_AVATARS,
    AvatarConfig,
    AvatarConfigVersionConflictError,
    AvatarRuntime,
    RestartRequiredError,
    StorageConfig,
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


@pytest.mark.parametrize(
    "layout",
    ["identical", "rig-parent", "media-parent", "dotdot", "symlink-alias"],
)
def test_storage_roots_reject_canonical_overlap(tmp_path, layout):
    rigs = tmp_path / "rigs"
    media = tmp_path / "media"
    if layout == "identical":
        media = rigs
    elif layout == "rig-parent":
        media = rigs / "media"
    elif layout == "media-parent":
        rigs = media / "rigs"
    elif layout == "dotdot":
        media = tmp_path / "child" / ".." / "rigs"
    else:
        rigs.mkdir()
        alias = tmp_path / "rigs-alias"
        alias.symlink_to(rigs, target_is_directory=True)
        media = alias

    with pytest.raises(ValueError, match="separate, non-overlapping"):
        StorageConfig.model_validate(
            {"rigs_dir": str(rigs), "backgrounds_dir": str(media)}
        )

    valid = StorageConfig.model_validate(
        {
            "rigs_dir": str(tmp_path / "valid-rigs"),
            "backgrounds_dir": str(tmp_path / "valid-media"),
        }
    )
    assert valid.rigs_dir != valid.backgrounds_dir


def test_appearance_choices_are_validated():
    for name in BUILTIN_AVATARS:
        assert (
            AvatarConfig.from_dict({"appearance": {"avatar": name}}).appearance.avatar
            == name
        )
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


def test_cli_driver_override_is_optional_and_explicit(tmp_path):
    config_path = tmp_path / "avatar.yaml"
    config_path.write_text(
        "driver:\n"
        "  backend: audio2face\n"
        "  audio2face:\n"
        "    url: grpc://127.0.0.1:52000\n"
    )
    parser = build_parser()

    config_args = parser.parse_args(["--config", str(config_path)])
    assert config_args.driver is None
    from_config = config_from_args(config_args)
    assert from_config.driver.backend == "audio2face"

    from_cli = config_from_args(
        parser.parse_args(["--config", str(config_path), "--driver", "idle"])
    )
    assert from_cli.driver.backend == "idle"


def test_packaged_cli_preserves_driver_but_owns_plaintext_loopback_transports(
    tmp_path,
):
    ca_file = ssl.get_default_verify_paths().cafile
    if not ca_file:
        pytest.skip("the test interpreter has no default CA bundle")
    config_path = tmp_path / "avatar.yaml"
    config_path.write_text(
        "source:\n"
        "  url: wss://engine.example:8710\n"
        f"  tls_ca_file: {json.dumps(ca_file)}\n"
        "driver:\n"
        "  backend: audio2face\n"
        "  audio2face:\n"
        "    url: grpc://127.0.0.1:52000\n"
        "api:\n"
        "  enabled: false\n"
        "  host: remote.example\n"
        "  tls_certfile: /operator/server.crt\n"
        "  tls_keyfile: /operator/server.key\n"
    )

    cfg = config_from_args(
        build_parser().parse_args(
            [
                "--config",
                str(config_path),
                "--source",
                "ws://127.0.0.1:28710",
                "--source-plaintext",
                "--enable-api",
                "--api-host",
                "127.0.0.1",
                "--api-port",
                "28711",
                "--api-plaintext",
            ]
        )
    )

    assert cfg.driver.backend == "audio2face"
    assert cfg.source.url == "ws://127.0.0.1:28710"
    assert (
        cfg.source.tls_ca_file,
        cfg.source.tls_certfile,
        cfg.source.tls_keyfile,
    ) == ("", "", "")
    assert cfg.api.enabled is True
    assert (cfg.api.host, cfg.api.port) == ("127.0.0.1", 28711)
    assert (cfg.api.tls_certfile, cfg.api.tls_keyfile) == ("", "")


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--source",
            "wss://renderer.example:8710",
            "--source-plaintext",
        ],
        [
            "--api-host",
            "remote.example",
            "--api-plaintext",
        ],
        [
            "--api-plaintext",
            "--api-tls-cert",
            "/operator/server.crt",
            "--api-tls-key",
            "/operator/server.key",
        ],
    ],
)
def test_plaintext_cli_flags_cannot_silently_weaken_remote_tls(arguments):
    with pytest.raises(ValueError, match="plaintext"):
        config_from_args(build_parser().parse_args(arguments))


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
        runtime.apply_patch({"appearance": {"scale": 0.5}, "api": {"port": 9000}})
    state = runtime.read()
    assert state.version == 0
    assert state.config.appearance.scale == 1.0


def test_runtime_invalid_patch_changes_nothing():
    runtime = AvatarRuntime(AvatarConfig())
    with pytest.raises(ValueError):
        runtime.apply_patch({"appearance": {"scale": 99}})
    assert runtime.version == 0


def test_runtime_prepare_patch_returns_cas_base_and_marks_noop():
    runtime = AvatarRuntime(AvatarConfig())
    base, candidate = runtime.prepare_patch({"appearance": {"scale": 0.5}})
    assert base.version == 0
    assert candidate is not None
    assert candidate.appearance.scale == 0.5
    assert runtime.version == 0

    unchanged, noop = runtime.prepare_patch({"appearance": {"scale": 1.0}})
    assert unchanged.version == 0
    assert noop is None


def test_runtime_coordinator_commit_is_cas_and_activation_atomic():
    runtime = AvatarRuntime(AvatarConfig())
    writer = runtime._coordinator_writer()
    base, candidate = runtime.prepare_patch({"appearance": {"scale": 0.5}})
    assert candidate is not None
    entered = threading.Event()
    release = threading.Event()
    effective = {"scale": 1.0}
    result = []

    def activate(_version):
        entered.set()
        assert release.wait(1.0)
        effective["scale"] = 0.5

    committer = threading.Thread(
        target=lambda: result.append(
            writer.commit_with_activation(candidate, base.version, activate)
        )
    )
    committer.start()
    assert entered.wait(1.0)

    observed = []
    reader = threading.Thread(target=lambda: observed.append(runtime.read()))
    reader.start()
    time.sleep(0.03)
    assert reader.is_alive()
    assert effective["scale"] == 1.0

    release.set()
    committer.join(1.0)
    reader.join(1.0)
    assert not committer.is_alive()
    assert not reader.is_alive()
    assert result[0].version == 1
    assert observed[0].version == 1
    assert observed[0].config.appearance.scale == 0.5
    assert effective["scale"] == 0.5

    with pytest.raises(AvatarConfigVersionConflictError) as excinfo:
        writer.commit_with_activation(candidate, base.version, lambda _version: None)
    assert excinfo.value.expected_version == 0
    assert excinfo.value.current_version == 1


def test_runtime_failed_activation_does_not_publish_candidate():
    runtime = AvatarRuntime(AvatarConfig())
    writer = runtime._coordinator_writer()
    base, candidate = runtime.prepare_patch({"appearance": {"scale": 0.5}})
    assert candidate is not None

    def fail(_version):
        raise RuntimeError("candidate failed")

    with pytest.raises(RuntimeError, match="candidate failed"):
        writer.commit_with_activation(candidate, base.version, fail)
    assert runtime.read() == base


def test_bound_runtime_apply_patch_delegates_to_service_coordinator():
    runtime = AvatarRuntime(AvatarConfig())
    writer = runtime._coordinator_writer()
    calls = []

    def coordinate(patch):
        calls.append(patch)
        base, candidate = runtime.prepare_patch(patch)
        if candidate is None:
            return base
        return writer.commit_with_activation(
            candidate, base.version, lambda _version: None
        )

    runtime.bind_coordinator(coordinate)
    state = runtime.apply_patch({"appearance": {"scale": 0.5}})
    assert calls == [{"appearance": {"scale": 0.5}}]
    assert state.version == 1
    assert state.config.appearance.scale == 0.5
    with pytest.raises(RuntimeError, match="already bound"):
        runtime.bind_coordinator(coordinate)


def test_storage_permission_cli_check_and_fix(tmp_path, capsys):
    rigs = tmp_path / "rigs"
    media = tmp_path / "media"
    rigs.mkdir(mode=0o755)
    media.mkdir(mode=0o755)
    cfg = AvatarConfig.from_dict(
        {
            "storage": {
                "rigs_dir": str(rigs),
                "backgrounds_dir": str(media),
            }
        }
    )

    assert _storage_permission_command(cfg, fix=False) == EXIT_CONFIG
    assert "--fix-storage-permissions" in capsys.readouterr().err
    assert _storage_permission_command(cfg, fix=True) == 0
    assert "storage permissions repaired" in capsys.readouterr().out
    assert (Path(rigs).stat().st_mode & 0o777) == 0o700
    assert (Path(media).stat().st_mode & 0o777) == 0o700


def test_PKG_01_installed_avatar_template_exports_privately_without_overwrite(
    tmp_path,
    monkeypatch,
):
    canonical = Path("config/avatar.yaml").read_bytes()
    assert avatar_config_bytes() == canonical

    output = io.BytesIO()
    assert export_avatar_config(None, output=output) == 0
    assert output.getvalue() == canonical

    destination = tmp_path / "avatar.yaml"
    assert export_avatar_config(destination) == 0
    assert destination.read_bytes() == canonical
    assert os.stat(destination).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        export_avatar_config(destination)

    calls = []

    def fake_avatar_main(argv, *, prog):
        calls.append((argv, prog))
        return 17

    monkeypatch.setattr("custback.avatar.__main__.main", fake_avatar_main)
    from custback.__main__ import main as custback_main

    assert custback_main(["avatar", "--help"]) == 17
    assert calls == [(["--help"], "custback avatar")]
