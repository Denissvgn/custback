"""Phase 6 golden migration, crash recovery, and storage safety contracts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
import yaml

import custback.migration as migration
from custback.config import AppConfig
from custback.storage_tx import OwnershipLedger


FIXTURES = Path(__file__).parent / "fixtures" / "migration"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _private_mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def test_migration_fixtures_are_bound_to_the_0_3_0_source_commit():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())

    assert provenance["release"] == "0.3.0"
    assert provenance["source_commit"] == ("f01baadfa3b1e2a1ef19eceda315eedf06fbe883")
    assert provenance["source_path"] == "config/default.yaml"
    assert provenance["publication_status"] == "unpublished-reference-commit"
    assert provenance["source_sha256"] == (
        "d060c287b3cd1b0ca23b379b641e9b1df511ead5212c80e90d56502804e93043"
    )
    for name, record in provenance["fixtures"].items():
        assert hashlib.sha256(_fixture(name)).hexdigest() == record["sha256"]


def test_golden_config_migration_is_private_atomic_and_idempotent(tmp_path):
    original = _fixture("legacy-0.3.0-local-camera.yaml")
    expected = _fixture("expected-0.4.0-local-camera.yaml")
    config = tmp_path / "config.yaml"
    config.write_bytes(original)
    config.chmod(0o644)

    result = migration.migrate_config(config, "legacy-camera")
    artifacts = migration.migration_artifacts(config)

    assert result.status is migration.MigrationStatus.MIGRATED
    assert result.changed
    assert not result.recovered
    assert config.read_bytes() == expected
    assert artifacts.backup.read_bytes() == original
    assert _private_mode(config) == 0o600
    assert _private_mode(artifacts.backup) == 0o600
    assert not artifacts.candidate.exists()
    assert not artifacts.journal.exists()
    loaded = AppConfig.load(config)
    assert loaded.background.camera_device == ""
    assert loaded.background.camera_target == "legacy-camera"
    assert loaded.backdrop_targets["legacy-camera"].source == "/dev/video2"

    before = {
        path: (path.lstat().st_ino, path.lstat().st_mtime_ns, path.read_bytes())
        for path in (config, artifacts.backup)
    }
    again = migration.migrate_config(config, "legacy-camera")
    assert again.status is migration.MigrationStatus.ALREADY_CURRENT
    assert not again.changed
    assert {
        path: (path.lstat().st_ino, path.lstat().st_mtime_ns, path.read_bytes())
        for path in (config, artifacts.backup)
    } == before


@pytest.mark.parametrize("boundary", migration.DURABLE_BOUNDARIES)
def test_retry_converges_after_every_durable_boundary(tmp_path, boundary):
    original = _fixture("legacy-0.3.0-local-camera.yaml")
    expected = _fixture("expected-0.4.0-local-camera.yaml")
    config = tmp_path / f"{boundary}.yaml"
    config.write_bytes(original)
    injected = False

    def interrupt(observed: str) -> None:
        nonlocal injected
        if observed == boundary and not injected:
            injected = True
            raise RuntimeError(f"interrupted at {boundary}")

    with pytest.raises(RuntimeError, match=boundary):
        migration.migrate_config(
            config,
            "legacy-camera",
            boundary_hook=interrupt,
        )
    assert injected
    # Atomic replacement means the authoritative path is always exactly old
    # or new, never a partial serialization.
    assert config.read_bytes() in {original, expected}

    recovered = migration.migrate_config(config, "legacy-camera")
    assert recovered.status in {
        migration.MigrationStatus.MIGRATED,
        migration.MigrationStatus.ALREADY_CURRENT,
    }
    assert config.read_bytes() == expected
    artifacts = migration.migration_artifacts(config)
    assert artifacts.backup.read_bytes() == original
    assert not artifacts.candidate.exists()
    assert not artifacts.journal.exists()


@pytest.mark.parametrize(
    "source",
    [
        "rtsp://camera.example/live",
        "https://user:secret@camera.example/live",
        "file:///run/secrets/token",
        "//camera.example/share",
        "relative-camera-name",
        "../camera",
    ],
)
def test_unsafe_legacy_source_requires_operator_action_with_zero_writes(
    tmp_path, source
):
    raw = yaml.safe_load(_fixture("legacy-0.3.0-local-camera.yaml"))
    raw["background"]["camera_device"] = source
    config = tmp_path / "unsafe.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    before = {
        path.name: (path.lstat().st_mode, path.read_bytes())
        for path in tmp_path.iterdir()
    }

    result = migration.migrate_config(config, "legacy-camera")

    assert result.status is migration.MigrationStatus.OPERATOR_ACTION_REQUIRED
    assert not result.changed
    assert result.backup_path is None
    assert {
        path.name: (path.lstat().st_mode, path.read_bytes())
        for path in tmp_path.iterdir()
    } == before


def test_config_symlink_and_duplicate_keys_are_rejected_without_writes(tmp_path):
    real = tmp_path / "real.yaml"
    real.write_bytes(_fixture("legacy-0.3.0-local-camera.yaml"))
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(real)

    with pytest.raises(migration.MigrationError, match="symbolic link"):
        migration.migrate_config(linked, "legacy-camera")
    assert set(tmp_path.iterdir()) == {real, linked}

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "background:\n  mode: camera\n  camera_device: 2\n  camera_device: 3\n"
    )
    snapshot = duplicate.read_bytes()
    with pytest.raises(migration.MigrationError, match="safe YAML"):
        migration.migrate_config(duplicate, "legacy-camera")
    assert duplicate.read_bytes() == snapshot
    assert set(tmp_path.iterdir()) == {real, linked, duplicate}


def _metadata_snapshot(root: Path) -> dict[str, tuple[int, int, bytes]]:
    return {
        str(path.relative_to(root)): (
            path.lstat().st_ino,
            stat.S_IMODE(path.lstat().st_mode),
            path.read_bytes() if path.is_file() else b"",
        )
        for path in sorted(root.rglob("*"))
    }


def test_store_repair_preserves_data_and_durable_quota_ownership(tmp_path):
    core = tmp_path / "backgrounds"
    rigs = tmp_path / "rigs"
    avatar_backgrounds = tmp_path / "avatar-backgrounds"
    for root in (core, rigs, avatar_backgrounds):
        root.mkdir(mode=0o755)
        root.chmod(0o755)
    rig = rigs / "office-rig"
    rig.mkdir(mode=0o755)
    rig.chmod(0o755)
    assets = {
        core / "committed.png": b"core-image",
        rig / "body.png": b"rig-layer",
        avatar_backgrounds / "scene.mp4": b"scene-video",
    }
    for path, contents in assets.items():
        path.write_bytes(contents)
        path.chmod(0o644)

    pending = core / ".upload-owned.part"
    pending.write_bytes(b"reserved-upload")
    pending.chmod(0o640)
    ledger = OwnershipLedger(core)
    record = ledger.begin(pending, kind="file", reserved_bytes=15, reserved_slots=1)
    ledger.bind(record, pending)
    ledger.close()
    metadata_root = core.parent / f".{core.name}.custback-ownership"
    metadata_before = _metadata_snapshot(metadata_root)

    audit = migration.audit_storage((core, rigs, avatar_backgrounds))
    assert audit.issues
    assert {issue.reason for issue in audit.issues} == {"mode"}
    repaired = set(migration.repair_storage(audit.roots))

    assert {core, rigs, rig, avatar_backgrounds, *assets, pending} <= repaired
    assert migration.audit_storage(audit.roots).clean
    for root in (core, rigs, rig, avatar_backgrounds):
        assert _private_mode(root) == 0o700
    for path, contents in {**assets, pending: b"reserved-upload"}.items():
        assert _private_mode(path) == 0o600
        assert path.read_bytes() == contents
    # The sibling ownership tree, its record bytes, and inode identities are
    # outside the repair walk and must remain untouched.
    assert _metadata_snapshot(metadata_root) == metadata_before
    reloaded = OwnershipLedger(core)
    try:
        assert reloaded.reserved_bytes == 15
        assert reloaded.reserved_slots == 1
        assert len(reloaded.records) == 1
        assert reloaded.records[0].paths == (pending,)
    finally:
        reloaded.close()


def test_store_repair_preflights_symlinks_and_special_files_before_chmod(tmp_path):
    root = tmp_path / "backgrounds"
    root.mkdir()
    root.chmod(0o755)
    ordinary = root / "ordinary.png"
    ordinary.write_bytes(b"unchanged")
    ordinary.chmod(0o644)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    outside.chmod(0o644)
    (root / "linked.png").symlink_to(outside)
    fifo = root / "special"
    os.mkfifo(fifo, 0o600)

    audit = migration.audit_storage((root,))
    assert {issue.reason for issue in audit.issues} == {"mode", "symlink", "type"}
    with pytest.raises(migration.StorageMigrationError, match="refusing unsafe"):
        migration.repair_storage((root,))

    assert _private_mode(root) == 0o755
    assert _private_mode(ordinary) == 0o644
    assert ordinary.read_bytes() == b"unchanged"
    assert outside.read_bytes() == b"outside"


def test_store_repair_refuses_foreign_owners_before_mutation(tmp_path, monkeypatch):
    root = tmp_path / "rigs"
    root.mkdir()
    root.chmod(0o755)
    # Simulate a foreign owner through the platform seam. Ownership now routes
    # through platform_fs (st_uid == geteuid() on POSIX; an owner-SID comparison
    # on Windows), so forcing the advisory check to report "not ours" drives the
    # audit's "owner" issue without a file actually owned by another principal.
    monkeypatch.setattr(
        migration.platform_fs, "stat_owner_matches", lambda metadata: False
    )

    audit = migration.audit_storage((root,))
    assert [issue.reason for issue in audit.issues] == ["owner"]
    with pytest.raises(migration.StorageMigrationError, match="owner"):
        migration.repair_storage((root,))
    assert _private_mode(root) == 0o755


def test_store_repair_reaudits_after_securing_an_unreadable_parent(tmp_path):
    root = tmp_path / "rigs"
    nested = root / "installed-rig"
    root.mkdir()
    nested.mkdir()
    layer = nested / "body.png"
    layer.write_bytes(b"layer")
    layer.chmod(0o644)
    nested.chmod(0o100)
    root.chmod(0o100)

    repaired = set(migration.repair_storage((root,)))

    assert repaired == {root, nested, layer}
    assert migration.audit_storage((root,)).clean
    assert _private_mode(root) == 0o700
    assert _private_mode(nested) == 0o700
    assert _private_mode(layer) == 0o600
    assert layer.read_bytes() == b"layer"


def test_core_cli_dispatches_migration_before_the_runtime_parser(monkeypatch):
    observed = {}

    def fake_migration_main(argv, *, prog):
        observed["argv"] = argv
        observed["prog"] = prog
        return 23

    monkeypatch.setattr(migration, "main", fake_migration_main)
    monkeypatch.setattr(
        "custback.__main__.build_parser",
        lambda: pytest.fail("normal runtime parser must not run"),
    )
    from custback.__main__ import main as core_main

    assert core_main(["migrate", "--config", "legacy.yaml"]) == 23
    assert observed == {
        "argv": ["--config", "legacy.yaml"],
        "prog": "custback migrate",
    }


def test_migrate_cli_returns_distinct_operator_action_status_without_leaking_source(
    tmp_path, capsys
):
    raw = yaml.safe_load(_fixture("legacy-0.3.0-local-camera.yaml"))
    raw["background"]["camera_device"] = (
        "https://operator:super-secret@camera.example/live"
    )
    config = tmp_path / "unsafe.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    before = config.read_bytes()

    from custback.__main__ import main as core_main

    assert (
        core_main(
            [
                "migrate",
                "--config",
                str(config),
                "--target-id",
                "legacy-camera",
            ]
        )
        == migration.EXIT_OPERATOR_ACTION_REQUIRED
    )
    output = capsys.readouterr()
    assert "operator action required" in output.err
    assert "super-secret" not in output.err
    assert config.read_bytes() == before
    assert set(tmp_path.iterdir()) == {config}
