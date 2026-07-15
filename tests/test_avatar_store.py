"""Rig/media stores: safe archive handling, quotas, and thumbnails."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from custback.avatar.config import AvatarBackgroundConfig, StorageConfig
from custback.avatar.store import (
    MediaStore,
    RigStore,
    StoreError,
    ThumbnailCache,
    render_avatar_thumbnail,
    render_media_thumbnail,
    resolve_rig_selector,
    sanitize_media_name,
)


def storage(tmp_path: Path, **overrides) -> StorageConfig:
    values = {
        "rigs_dir": str(tmp_path / "rigs"),
        "backgrounds_dir": str(tmp_path / "media"),
    }
    values.update(overrides)
    return StorageConfig.model_validate(values)


def layer_png(width: int = 64, height: int = 96) -> bytes:
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    layer[8:-8, 8:-8] = (90, 140, 200, 255)
    ok, data = cv2.imencode(".png", layer)
    assert ok
    return data.tobytes()


def rig_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def write_zip(tmp_path: Path, members: dict[str, bytes]) -> Path:
    path = tmp_path / "upload.zip"
    path.write_bytes(rig_zip(members))
    return path


def jpeg_size(data: bytes) -> tuple[int, int]:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    return frame.shape[1], frame.shape[0]


# -- selector resolution and naming ----------------------------------------


def test_resolve_rig_selector(tmp_path):
    rigs = tmp_path / "rigs"
    (rigs / "myrig").mkdir(parents=True)
    assert resolve_rig_selector("builtin", rigs) == "builtin"
    assert resolve_rig_selector("myrig", rigs) == str(rigs / "myrig")
    explicit = tmp_path / "elsewhere"
    explicit.mkdir()
    assert resolve_rig_selector(str(explicit), rigs) == str(explicit)
    assert resolve_rig_selector("missing", rigs) == "missing"


def test_sanitize_media_name_slugs_and_checks_extension():
    assert sanitize_media_name("My Beach Photo.JPG", "image") == "my-beach-photo.jpg"
    assert sanitize_media_name("../../etc/passwd.png", "image") == "passwd.png"
    with pytest.raises(StoreError) as excinfo:
        sanitize_media_name("notes.txt", "image")
    assert excinfo.value.code == "unsupported_media_type"
    with pytest.raises(StoreError):
        sanitize_media_name("clip.png", "video")


# -- rig store ---------------------------------------------------------------


def test_install_zip_valid_rig(tmp_path):
    store = RigStore(storage(tmp_path))
    members = {
        "torso.png": layer_png(),
        "head.png": layer_png(),
        "rig.yaml": b"pivot: [32, 40]\n",
    }
    installed = store.install_zip("casey-two", write_zip(tmp_path, members))
    assert installed.name == "casey-two"
    assert installed.parts == ("torso", "head")
    assert installed.has_manifest
    listed = store.list()
    assert [rig.name for rig in listed] == ["casey-two"]
    assert resolve_rig_selector("casey-two", store.directory).endswith("casey-two")


def test_install_zip_flattens_single_root(tmp_path):
    store = RigStore(storage(tmp_path))
    members = {"myrig/head.png": layer_png()}
    installed = store.install_zip("nested", write_zip(tmp_path, members))
    assert installed.parts == ("head",)


def test_install_zip_rejects_unexpected_and_duplicate_entries(tmp_path):
    store = RigStore(storage(tmp_path))
    with pytest.raises(StoreError) as excinfo:
        store.install_zip(
            "bad", write_zip(tmp_path, {"torso.png": layer_png(), "evil.sh": b"x"})
        )
    assert excinfo.value.code == "invalid_rig"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("head.png", layer_png())
        archive.writestr("sub/head.png", layer_png())
    dupe = tmp_path / "dupe.zip"
    dupe.write_bytes(buffer.getvalue())
    with pytest.raises(StoreError) as excinfo:
        store.install_zip("dupe", dupe)
    assert excinfo.value.code == "invalid_rig"
    assert store.list() == []


def test_install_zip_traversal_names_cannot_escape(tmp_path):
    store = RigStore(storage(tmp_path))
    installed = store.install_zip(
        "traversal", write_zip(tmp_path, {"../head.png": layer_png()})
    )
    assert installed.parts == ("head",)
    assert (store.directory / "traversal" / "head.png").is_file()
    assert not (store.directory.parent / "head.png").exists()


def test_install_zip_enforces_entry_and_byte_limits(tmp_path):
    cfg = storage(tmp_path, rig_max_entries=1, rig_max_bytes=1024)
    store = RigStore(cfg)
    with pytest.raises(StoreError) as excinfo:
        store.install_zip(
            "many",
            write_zip(tmp_path, {"torso.png": layer_png(), "head.png": layer_png()}),
        )
    assert excinfo.value.code == "rig_too_large"
    noisy = np.random.default_rng(7).integers(
        0, 255, size=(96, 64, 4), dtype=np.uint8
    )
    ok, big_layer = cv2.imencode(".png", noisy)
    assert ok and big_layer.size > 1024
    with pytest.raises(StoreError) as excinfo:
        store.install_zip(
            "big", write_zip(tmp_path, {"head.png": big_layer.tobytes()})
        )
    assert excinfo.value.code == "rig_too_large"
    assert store.list() == []


def test_install_zip_rejects_invalid_archives_and_rigs(tmp_path):
    store = RigStore(storage(tmp_path))
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip")
    with pytest.raises(StoreError) as excinfo:
        store.install_zip("notzip", bad)
    assert excinfo.value.code == "invalid_rig"
    # A PNG without an alpha channel fails LayeredRig validation.
    opaque = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, data = cv2.imencode(".png", opaque)
    assert ok
    with pytest.raises(StoreError) as excinfo:
        store.install_zip(
            "noalpha", write_zip(tmp_path, {"head.png": data.tobytes()})
        )
    assert excinfo.value.code == "invalid_rig"
    assert store.list() == []
    assert not any(store.directory.glob(".staged-*"))


def test_install_zip_name_rules_and_conflicts(tmp_path):
    store = RigStore(storage(tmp_path))
    with pytest.raises(StoreError) as excinfo:
        store.install_zip("Bad Name", write_zip(tmp_path, {"head.png": layer_png()}))
    assert excinfo.value.code == "invalid_rig_name"
    store.install_zip("taken", write_zip(tmp_path, {"head.png": layer_png()}))
    with pytest.raises(StoreError) as excinfo:
        store.install_zip("taken", write_zip(tmp_path, {"head.png": layer_png()}))
    assert excinfo.value.code == "rig_exists"


def test_remove_rig_guards_active_selector(tmp_path):
    store = RigStore(storage(tmp_path))
    store.install_zip("active", write_zip(tmp_path, {"head.png": layer_png()}))
    with pytest.raises(StoreError) as excinfo:
        store.remove("active", "active")
    assert excinfo.value.code == "rig_in_use"
    with pytest.raises(StoreError) as excinfo:
        store.remove("missing", "builtin")
    assert excinfo.value.code == "rig_not_found"
    store.remove("active", "builtin")
    assert store.list() == []


# -- media store -------------------------------------------------------------


def image_bytes(width: int = 320, height: int = 200) -> bytes:
    frame = np.full((height, width, 3), (30, 90, 160), dtype=np.uint8)
    ok, data = cv2.imencode(".png", frame)
    assert ok
    return data.tobytes()


def staged(store: MediaStore, payload: bytes) -> Path:
    path = store.open_staging()
    path.write_bytes(payload)
    return path


def test_media_commit_lists_and_removes(tmp_path):
    store = MediaStore(storage(tmp_path))
    saved = store.commit(staged(store, image_bytes()), "Beach Day.PNG", "image")
    assert saved.name == "beach-day.png"
    assert saved.kind == "image"
    listed = store.list()
    assert [media.name for media in listed] == ["beach-day.png"]
    assert listed[0].path == saved.path
    active = AvatarBackgroundConfig(mode="image", image_path=saved.path)
    with pytest.raises(StoreError) as excinfo:
        store.remove(saved.name, active)
    assert excinfo.value.code == "media_in_use"
    store.remove(saved.name, AvatarBackgroundConfig())
    assert store.list() == []


def test_media_commit_deduplicates_names(tmp_path):
    store = MediaStore(storage(tmp_path))
    first = store.commit(staged(store, image_bytes()), "wall.png", "image")
    second = store.commit(staged(store, image_bytes()), "wall.png", "image")
    assert first.name == "wall.png"
    assert second.name == "wall-2.png"


def test_media_commit_rejects_bad_and_oversized_uploads(tmp_path):
    store = MediaStore(storage(tmp_path))
    with pytest.raises(StoreError) as excinfo:
        store.commit(staged(store, b"not an image"), "x.png", "image")
    assert excinfo.value.code == "invalid_media"
    with pytest.raises(StoreError) as excinfo:
        store.commit(staged(store, b""), "x.png", "image")
    assert excinfo.value.code == "invalid_media"
    small = MediaStore(storage(tmp_path, image_max_pixels=1024))
    with pytest.raises(StoreError) as excinfo:
        small.commit(staged(small, image_bytes(64, 64)), "big.png", "image")
    assert excinfo.value.code == "media_too_large"


def test_media_quota_limits(tmp_path):
    store = MediaStore(storage(tmp_path, max_files=1))
    store.commit(staged(store, image_bytes()), "one.png", "image")
    with pytest.raises(StoreError) as excinfo:
        store.commit(staged(store, image_bytes()), "two.png", "image")
    assert excinfo.value.code == "storage_full"


def test_stored_path_refuses_traversal(tmp_path):
    store = MediaStore(storage(tmp_path))
    store.commit(staged(store, image_bytes()), "ok.png", "image")
    for name in ("../ok.png", ".hidden.png", "sub/ok.png", ""):
        with pytest.raises(StoreError) as excinfo:
            store.stored_path(name)
        assert excinfo.value.code == "media_not_found"


# -- thumbnails ---------------------------------------------------------------


def test_render_avatar_thumbnail_builtin():
    data = render_avatar_thumbnail("builtin", avatar="nova", style="cartoon")
    assert jpeg_size(data) == (256, 144)


def test_render_media_thumbnail_downscales(tmp_path):
    store = MediaStore(storage(tmp_path))
    saved = store.commit(
        staged(store, image_bytes(1280, 720)), "wide.png", "image"
    )
    data = render_media_thumbnail(Path(saved.path), "image")
    width, height = jpeg_size(data)
    assert (width, height) == (256, 144)


def test_thumbnail_cache_evicts_oldest():
    cache = ThumbnailCache(capacity=2)
    cache.put(("a",), b"1")
    cache.put(("b",), b"2")
    assert cache.get(("a",)) == b"1"
    cache.put(("c",), b"3")
    assert cache.get(("b",)) is None
    assert cache.get(("a",)) == b"1"
    assert cache.get(("c",)) == b"3"