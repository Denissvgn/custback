"""Phase 5 STOR-02 rename-aware ownership regressions."""

from __future__ import annotations

import os
import stat
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import custback.api.server as server_mod
import custback.avatar.store as avatar_store_mod
from custback.api.server import _UploadLimits, _UploadStore
from custback.avatar.config import StorageConfig
from custback.avatar.store import MediaStore, RigStore, StoreError
from custback.config import AppConfig
from custback.storage_tx import OwnershipLedger


def _avatar_storage(tmp_path: Path) -> StorageConfig:
    return StorageConfig.model_validate(
        {
            "rigs_dir": str(tmp_path / "rigs"),
            "backgrounds_dir": str(tmp_path / "media"),
        }
    )


def _png(*, alpha: bool = False) -> bytes:
    shape = (32, 32, 4 if alpha else 3)
    image = np.full(shape, 127, dtype=np.uint8)
    if alpha:
        image[..., 3] = 255
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _zip_layer(payload: bytes) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("head.png", payload)
    return output.getvalue()


def _core_staging(store: _UploadStore, payload: bytes) -> tuple[Path, int]:
    store.ensure_directory()
    temporary = store.directory / (".upload-" + "0" * 32 + ".part")
    store._reserve_file(temporary)
    destination = store._open_private_file(temporary)
    store._bind_created(temporary)
    store._reserve_bytes(len(payload), temporary)
    destination.write(payload)
    destination.close()
    return temporary, len(payload)


def test_STOR_02_post_rename_failures_retain_cleanup_ownership(tmp_path, monkeypatch):
    """Every store follows the inode after hardening + rollback failures."""

    real_replace = os.replace
    real_unlink = Path.unlink
    real_core_rename = server_mod.rename_noreplace
    real_avatar_rename = avatar_store_mod.rename_noreplace

    # Core upload: temporary -> hidden staged succeeded, hardening failed,
    # rollback failed, and all immediate unlink attempts failed.
    core = _UploadStore(tmp_path / "core", _UploadLimits())
    temporary, size = _core_staging(core, b"core-owned")
    staged = core.directory / (".upload-" + "0" * 32 + ".png")

    def core_replace(source, destination):
        if Path(source) == staged and Path(destination) == temporary:
            raise OSError("core rollback failed")
        return real_core_rename(source, destination)

    def core_unlink(path, *args, **kwargs):
        if Path(path).parent == core.directory and Path(path).name.startswith(
            ".custback-cleanup-"
        ):
            raise OSError("core cleanup failed")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(server_mod, "rename_noreplace", core_replace)
    monkeypatch.setattr(
        core,
        "_secure_private_file",
        lambda _path: (_ for _ in ()).throw(PermissionError("core hardening failed")),
    )
    monkeypatch.setattr(Path, "unlink", core_unlink)
    with pytest.raises(PermissionError, match="core hardening failed"):
        core._commit(temporary, staged, size)
    core._cleanup_save(None, temporary, None, size, True)

    core_pending = core._ledger.pending_paths()
    assert len(core_pending) == 1 and core_pending[0].exists()
    assert core_pending[0].name.startswith(".custback-cleanup-")
    assert not staged.exists() and not temporary.exists()
    assert core._reserved_bytes == size
    assert core._reserved_files == 1
    assert core._usage() == (0, 0)

    monkeypatch.setattr(server_mod, "rename_noreplace", real_core_rename)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    core._retry_pending_cleanup()
    assert not core_pending[0].exists()
    assert core._ledger.records == ()
    assert (core._reserved_bytes, core._reserved_files) == (0, 0)

    # Avatar media: the public name is never listed while cleanup owns it.
    cfg = _avatar_storage(tmp_path)
    media = MediaStore(cfg)
    reservation = media.open_staging("image")
    payload = _png()
    reservation.write(payload)
    source = reservation.path
    final_media = media.directory / "failed.png"
    real_secure = avatar_store_mod._secure_existing

    def media_secure(path, mode, *, directory):
        if Path(path) == final_media:
            raise PermissionError("media hardening failed")
        return real_secure(path, mode, directory=directory)

    def media_replace(source_path, destination):
        if Path(source_path) == final_media and Path(destination) == source:
            raise OSError("media rollback failed")
        return real_avatar_rename(source_path, destination)

    def media_unlink(path, *args, **kwargs):
        if Path(path).parent == media.directory and Path(path).name.startswith(
            ".custback-cleanup-"
        ):
            raise OSError("media cleanup failed")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(avatar_store_mod, "_secure_existing", media_secure)
    monkeypatch.setattr(avatar_store_mod, "rename_noreplace", media_replace)
    monkeypatch.setattr(Path, "unlink", media_unlink)
    with pytest.raises(StoreError, match="cannot publish media") as media_error:
        media.commit(reservation, "failed.png", "image")
    assert isinstance(media_error.value.__cause__, PermissionError)
    media_pending = media._cleanup_pending
    assert len(media_pending) == 1 and media_pending[0].exists()
    assert media_pending[0].name.startswith(".custback-cleanup-")
    assert stat.S_IMODE(media_pending[0].stat().st_mode) == 0o600
    assert not final_media.exists() and not source.exists()
    assert media.list() == []
    assert media._cleanup_pending == media_pending
    assert media._reserved_bytes == len(payload)
    assert media._reserved_files == 1
    assert media._usage_locked() == (0, 0)

    monkeypatch.setattr(avatar_store_mod, "_secure_existing", real_secure)
    monkeypatch.setattr(avatar_store_mod, "rename_noreplace", real_avatar_rename)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    media._retry_pending_cleanup()
    assert not media_pending[0].exists()
    assert media._cleanup_pending == ()
    assert (media._reserved_bytes, media._reserved_files) == (0, 0)

    # Rig directory: recursive cleanup has the same transaction semantics.
    rig = RigStore(cfg)
    archive = tmp_path / "rig.zip"
    layer = _png(alpha=True)
    archive.write_bytes(_zip_layer(layer))
    final_rig = rig.directory / "failed-rig"
    real_rmtree = avatar_store_mod.shutil.rmtree

    def rig_replace(source_path, destination):
        if Path(source_path) == final_rig and Path(destination).name.startswith(
            ".staged-"
        ):
            raise OSError("rig rollback failed")
        return real_avatar_rename(source_path, destination)

    def rig_rmtree(path, *args, **kwargs):
        if Path(path).parent == rig.directory and Path(path).name.startswith(
            ".custback-cleanup-"
        ):
            raise OSError("rig cleanup failed")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        rig,
        "_secure_installed_rig",
        lambda _path: (_ for _ in ()).throw(PermissionError("rig hardening failed")),
    )
    monkeypatch.setattr(avatar_store_mod, "rename_noreplace", rig_replace)
    monkeypatch.setattr(avatar_store_mod.shutil, "rmtree", rig_rmtree)
    with pytest.raises(StoreError, match="cannot install rig") as rig_error:
        rig.install_zip("failed-rig", archive)
    assert isinstance(rig_error.value.__cause__, PermissionError)
    rig_pending = rig._cleanup_pending
    assert len(rig_pending) == 1 and rig_pending[0].exists()
    assert rig_pending[0].name.startswith(".custback-cleanup-")
    assert stat.S_IMODE(rig_pending[0].stat().st_mode) == 0o700
    assert not final_rig.exists()
    assert rig.list() == []
    assert rig._cleanup_pending == rig_pending
    assert rig._reserved_bytes == len(layer)
    assert rig._reserved_rigs == 1
    assert rig._usage_locked() == (0, 0)

    monkeypatch.setattr(avatar_store_mod, "rename_noreplace", real_avatar_rename)
    monkeypatch.setattr(avatar_store_mod.shutil, "rmtree", real_rmtree)
    rig._retry_pending_cleanup()
    assert not rig_pending[0].exists()
    assert rig._cleanup_pending == ()
    assert (rig._reserved_bytes, rig._reserved_rigs) == (0, 0)


def test_storage_restart_recovers_marked_data_and_preserves_unmarked_lookalikes(
    tmp_path,
):
    core_root = tmp_path / "core-restart"
    core = _UploadStore(core_root, _UploadLimits())
    temporary, _size = _core_staging(core, b"marked-core")
    core_record = core._ledger.find(temporary)
    assert core_record is not None
    core._ledger.mark_cleanup(core_record)
    unmarked_core = core_root / (".upload-" + "a" * 32 + ".png")
    unmarked_core.write_bytes(b"user core data")

    core._ledger.close()  # simulate process death releasing the OS lease
    restarted_core = _UploadStore(core_root, _UploadLimits())
    restarted_core.cleanup_staged(AppConfig())
    assert not temporary.exists()
    assert unmarked_core.read_bytes() == b"user core data"
    assert restarted_core._ledger.records == ()

    cfg = _avatar_storage(tmp_path)
    media = MediaStore(cfg)
    media_upload = media.open_staging("image")
    media_upload.write(b"marked media")
    media_upload.seal()
    media._ledger.mark_cleanup(media_upload._ownership)
    unmarked_media = media.directory / ".upload-user.part"
    unmarked_media.write_bytes(b"user media data")

    media._ledger.close()
    restarted_media = MediaStore(cfg)
    assert not media_upload.path.exists()
    assert unmarked_media.read_bytes() == b"user media data"
    assert restarted_media._ledger.records == ()

    rig = RigStore(cfg)
    rig.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    staged_rig = rig.directory / ".staged-owned"
    rig_record = rig._ledger.begin(
        staged_rig, kind="tree", reserved_bytes=9, reserved_slots=1
    )
    staged_rig.mkdir(mode=0o700)
    (staged_rig / "head.png").write_bytes(b"owned rig")
    rig._ledger.bind(rig_record, staged_rig)
    rig._ledger.mark_cleanup(rig_record)
    unmarked_rig = rig.directory / ".staged-user"
    unmarked_rig.mkdir(mode=0o700)
    (unmarked_rig / "note").write_bytes(b"user rig data")

    rig._ledger.close()
    restarted_rig = RigStore(cfg)
    assert not staged_rig.exists()
    assert (unmarked_rig / "note").read_bytes() == b"user rig data"
    assert restarted_rig._ledger.records == ()


def test_cleanup_primary_error_is_not_masked_by_staging_unlink_failure(
    tmp_path, monkeypatch
):
    store = MediaStore(_avatar_storage(tmp_path))
    reservation = store.open_staging("image")
    reservation.write(b"not an image")
    source = reservation.path
    real_unlink = Path.unlink

    def fail_owned_unlink(path, *args, **kwargs):
        if Path(path).parent == store.directory and Path(path).name.startswith(
            ".custback-cleanup-"
        ):
            raise OSError("secondary cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_owned_unlink)
    with pytest.raises(StoreError) as caught:
        store.commit(reservation, "bad.png", "image")
    assert caught.value.code == "invalid_media"
    pending = store._cleanup_pending
    assert len(pending) == 1 and pending[0].exists()
    assert not source.exists()
    assert store._reserved_bytes == len(b"not an image")

    monkeypatch.setattr(Path, "unlink", real_unlink)
    store._retry_pending_cleanup()
    assert not pending[0].exists()
    assert store._cleanup_pending == ()


def test_publication_rename_never_replaces_a_concurrent_destination(
    tmp_path, monkeypatch
):
    real_core_rename = server_mod.rename_noreplace
    core = _UploadStore(tmp_path / "core-collision", _UploadLimits())
    temporary, size = _core_staging(core, b"owned core bytes")
    staged = core.directory / (".upload-" + "0" * 32 + ".png")

    def collide_core(source, destination):
        if Path(destination) == staged and not staged.exists():
            staged.write_bytes(b"unmarked core sentinel")
        return real_core_rename(source, destination)

    monkeypatch.setattr(server_mod, "rename_noreplace", collide_core)
    with pytest.raises(FileExistsError):
        core._commit(temporary, staged, size)
    core._cleanup_save(None, temporary, None, size, True)
    assert staged.read_bytes() == b"unmarked core sentinel"

    cfg = _avatar_storage(tmp_path)
    media = MediaStore(cfg)
    media_upload = media.open_staging("image")
    media_upload.write(_png())
    final_media = media.directory / "collision.png"
    real_avatar_rename = avatar_store_mod.rename_noreplace

    def collide_media(source, destination):
        if Path(destination) == final_media and not final_media.exists():
            final_media.write_bytes(b"unmarked media sentinel")
        return real_avatar_rename(source, destination)

    monkeypatch.setattr(avatar_store_mod, "rename_noreplace", collide_media)
    with pytest.raises(StoreError) as media_error:
        media.commit(media_upload, "collision.png", "image")
    assert media_error.value.code == "insufficient_storage"
    assert final_media.read_bytes() == b"unmarked media sentinel"

    rig = RigStore(cfg)
    archive = tmp_path / "collision-rig.zip"
    archive.write_bytes(_zip_layer(_png(alpha=True)))
    final_rig = rig.directory / "collision-rig"

    def collide_rig(source, destination):
        if Path(destination) == final_rig and not final_rig.exists():
            final_rig.mkdir(mode=0o700)
            (final_rig / "sentinel").write_bytes(b"unmarked rig sentinel")
        return real_avatar_rename(source, destination)

    monkeypatch.setattr(avatar_store_mod, "rename_noreplace", collide_rig)
    with pytest.raises(StoreError) as rig_error:
        rig.install_zip("collision-rig", archive)
    assert rig_error.value.code == "insufficient_storage"
    assert (final_rig / "sentinel").read_bytes() == b"unmarked rig sentinel"


def test_recovery_never_deletes_an_inode_that_no_longer_matches_its_record(
    tmp_path,
):
    root = tmp_path / "inode-mismatch"
    store = _UploadStore(root, _UploadLimits())
    temporary, _size = _core_staging(store, b"original owned inode")
    record = store._ledger.find(temporary)
    assert record is not None
    store._ledger.mark_cleanup(record)

    displaced = root / "user-renamed-owned-lookalike.part"
    os.replace(temporary, displaced)
    temporary.write_bytes(b"unmarked replacement")

    store._ledger.close()
    restarted = _UploadStore(root, _UploadLimits())
    restarted.cleanup_staged(AppConfig())
    assert temporary.read_bytes() == b"unmarked replacement"
    assert displaced.read_bytes() == b"original owned inode"
    assert restarted._ledger.records == ()


def test_recovery_never_claims_a_later_inode_for_an_unbound_marker(tmp_path):
    root = tmp_path / "unbound-marker"
    store = _UploadStore(root, _UploadLimits())
    store.ensure_directory()
    candidate = root / (".upload-" + "d" * 32 + ".part")
    store._reserve_file(candidate)
    # Crash before O_EXCL creation/bind, then unrelated user data appears at
    # the same marked candidate name.
    store._ledger.close()
    candidate.write_bytes(b"later unmarked data")

    restarted = _UploadStore(root, _UploadLimits())
    restarted.cleanup_staged(AppConfig())
    assert candidate.read_bytes() == b"later unmarked data"
    assert restarted._ledger.records == ()
    assert restarted._usage() == (len(b"later unmarked data"), 1)


def test_live_store_lease_prevents_a_second_instance_from_reaping_upload(
    tmp_path,
):
    cfg = _avatar_storage(tmp_path)
    first = MediaStore(cfg)
    upload = first.open_staging("image")
    upload.write(b"live owned bytes")
    upload.seal()

    with pytest.raises(OSError, match="held by another process"):
        MediaStore(cfg)
    assert upload.path.read_bytes() == b"live owned bytes"
    assert upload.active

    upload.abort()


def test_ownership_recovery_refuses_symlinked_metadata_and_records(tmp_path):
    root = tmp_path / "payloads"
    victim_directory = tmp_path / "victim-directory"
    victim_directory.mkdir(mode=0o755)
    metadata_root = tmp_path / ".payloads.custback-ownership"
    metadata_root.symlink_to(victim_directory, target_is_directory=True)

    with pytest.raises(OSError, match="metadata is not a directory"):
        OwnershipLedger(root)
    assert stat.S_IMODE(victim_directory.stat().st_mode) == 0o755

    metadata_root.unlink()
    metadata_root.mkdir(mode=0o700)
    victim_record = tmp_path / "victim-record"
    victim_record.write_text("not transaction metadata", encoding="utf-8")
    record_link = metadata_root / (".custback-owned-" + "b" * 32 + ".json")
    record_link.symlink_to(victim_record)

    with pytest.raises(OSError, match="unsafe storage ownership record"):
        OwnershipLedger(root)
    assert victim_record.read_text(encoding="utf-8") == "not transaction metadata"
