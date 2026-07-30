"""Rig/media stores: safe archive handling, quotas, and thumbnails."""

import io
import os
import stat
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

cv2 = pytest.importorskip("cv2")

import custback.avatar.store as store_mod
from custback.avatar.config import AvatarBackgroundConfig, StorageConfig
from custback.avatar.store import (
    MediaStore,
    RigStore,
    StoreError,
    ThumbnailCache,
    UploadReservation,
    audit_storage_permissions,
    is_rig_directory,
    repair_storage_permissions,
    render_avatar_thumbnail,
    render_media_thumbnail,
    resolve_rig_selector,
    sanitize_media_name,
)
from custback.backgrounds import ImageBackdrop


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
    linked = rigs / "linked"
    linked.symlink_to(explicit, target_is_directory=True)
    assert not is_rig_directory(linked)
    assert resolve_rig_selector("linked", rigs) == "linked"


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


def test_install_zip_forwards_configured_decode_limits(monkeypatch, tmp_path):
    cfg = storage(
        tmp_path,
        rig_layer_max_pixels=20_000_000,
        rig_total_max_pixels=160_000_000,
        rig_manifest_max_bytes=128 * 1024,
    )
    store = RigStore(cfg)
    observed = []
    original = store_mod.create_rig

    def recording_create_rig(selector, **kwargs):
        observed.append(dict(kwargs))
        return original(selector, **kwargs)

    monkeypatch.setattr(store_mod, "create_rig", recording_create_rig)
    store.install_zip(
        "large-policy",
        write_zip(tmp_path, {"head.png": layer_png()}),
    )

    assert observed == [
        {
            "rig_layer_max_pixels": 20_000_000,
            "rig_total_max_pixels": 160_000_000,
            "rig_manifest_max_bytes": 128 * 1024,
        }
    ]


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
    noisy = np.random.default_rng(7).integers(0, 255, size=(96, 64, 4), dtype=np.uint8)
    ok, big_layer = cv2.imencode(".png", noisy)
    assert ok and big_layer.size > 1024
    with pytest.raises(StoreError) as excinfo:
        store.install_zip("big", write_zip(tmp_path, {"head.png": big_layer.tobytes()}))
    assert excinfo.value.code == "rig_too_large"
    assert store.list() == []


def test_rig_entry_limit_counts_empty_directory_records(tmp_path):
    store = RigStore(storage(tmp_path, rig_max_entries=2))
    members = {f"empty-{index}/": b"" for index in range(3)}
    members["head.png"] = layer_png()
    with pytest.raises(StoreError) as excinfo:
        store.install_zip("directory-spam", write_zip(tmp_path, members))
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
        store.install_zip("noalpha", write_zip(tmp_path, {"head.png": data.tobytes()}))
    assert excinfo.value.code == "invalid_rig"
    assert store.list() == []
    assert not any(store.directory.glob(".staged-*"))


def test_rig_headers_are_validated_with_pillow_before_opencv(tmp_path, monkeypatch):
    store = RigStore(
        storage(
            tmp_path,
            rig_layer_max_pixels=1024,
            rig_total_max_pixels=2048,
        )
    )
    monkeypatch.setattr(
        store_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCV must not decode an oversized layer")
        ),
    )
    with pytest.raises(StoreError) as excinfo:
        store.install_zip(
            "bomb",
            write_zip(tmp_path, {"head.png": layer_png(width=64, height=64)}),
        )
    assert excinfo.value.code == "rig_too_large"
    assert store.list() == []


def test_rig_rejects_pillow_decompression_warning_before_opencv(tmp_path, monkeypatch):
    store = RigStore(
        storage(
            tmp_path,
            rig_layer_max_pixels=10_000,
            rig_total_max_pixels=10_000,
        )
    )
    monkeypatch.setattr(store_mod.Image, "MAX_IMAGE_PIXELS", 3_000)
    monkeypatch.setattr(
        store_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCV must not decode a Pillow bomb warning")
        ),
    )
    with pytest.raises(StoreError) as excinfo:
        store.install_zip(
            "pillow-bomb",
            write_zip(tmp_path, {"head.png": layer_png(width=64, height=64)}),
        )
    assert excinfo.value.code == "invalid_rig"
    assert store.list() == []


def test_rig_total_decoded_pixel_limit_counts_every_layer(tmp_path):
    store = RigStore(
        storage(
            tmp_path,
            rig_layer_max_pixels=1024,
            rig_total_max_pixels=1500,
        )
    )
    with pytest.raises(StoreError) as excinfo:
        store.install_zip(
            "too-many-pixels",
            write_zip(
                tmp_path,
                {
                    "head.png": layer_png(width=32, height=32),
                    "torso.png": layer_png(width=32, height=32),
                },
            ),
        )
    assert excinfo.value.code == "rig_too_large"
    assert store.list() == []


def test_rig_rejects_disguised_png_and_inconsistent_dimensions(tmp_path):
    store = RigStore(storage(tmp_path))
    opaque = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", opaque)
    assert ok
    with pytest.raises(StoreError, match="invalid PNG") as excinfo:
        store.install_zip("not-png", write_zip(tmp_path, {"head.png": jpeg.tobytes()}))
    assert excinfo.value.code == "invalid_rig"

    with pytest.raises(StoreError, match="does not match") as excinfo:
        store.install_zip(
            "mismatch",
            write_zip(
                tmp_path,
                {
                    "head.png": layer_png(32, 32),
                    "torso.png": layer_png(48, 32),
                },
            ),
        )
    assert excinfo.value.code == "invalid_rig"


def test_rig_rejects_oversized_manifest_and_nonfinite_geometry(tmp_path):
    limited = RigStore(storage(tmp_path, rig_manifest_max_bytes=256))
    with pytest.raises(StoreError) as excinfo:
        limited.install_zip(
            "manifest-big",
            write_zip(
                tmp_path,
                {"head.png": layer_png(), "rig.yaml": b"#" * 257},
            ),
        )
    assert excinfo.value.code == "rig_too_large"

    store = RigStore(storage(tmp_path))
    with pytest.raises(StoreError, match="finite") as excinfo:
        store.install_zip(
            "geometry-nan",
            write_zip(
                tmp_path,
                {"head.png": layer_png(), "rig.yaml": b"pivot: [.nan, 1]\n"},
            ),
        )
    assert excinfo.value.code == "invalid_rig"
    with pytest.raises(StoreError, match="finite") as excinfo:
        store.install_zip(
            "geometry-overflow",
            write_zip(
                tmp_path,
                {
                    "head.png": layer_png(),
                    "rig.yaml": ("pivot: [" + "9" * 1000 + ", 1]\n").encode(),
                },
            ),
        )
    assert excinfo.value.code == "invalid_rig"
    with pytest.raises(StoreError) as excinfo:
        store.install_zip(
            "geometry-cycle",
            write_zip(
                tmp_path,
                {
                    "head.png": layer_png(),
                    "rig.yaml": b"pivot: &loop [*loop, 1]\n",
                },
            ),
        )
    assert excinfo.value.code == "invalid_rig"
    assert store.list() == []


def test_rig_quota_counts_compressed_and_extracted_staging_together(tmp_path):
    noisy = np.random.default_rng(19).integers(0, 255, size=(64, 64, 4), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", noisy)
    assert ok and len(encoded) > 1024
    payload = encoded.tobytes()
    archive = write_zip(tmp_path, {"head.png": payload})
    extracted_size = len(payload)
    # Either representation fits alone; their simultaneous installation peak
    # does not, so the aggregate staging quota must reject it.
    limit = max(archive.stat().st_size, extracted_size)
    store = RigStore(storage(tmp_path, rig_storage_max_bytes=limit))
    with pytest.raises(StoreError) as excinfo:
        store.install_zip("peak", archive)
    assert excinfo.value.code == "storage_full"
    assert store.list() == []
    assert store._reserved_bytes == 0
    assert store._active_extractions == set()


def test_rig_aggregate_byte_quota_includes_committed_rigs(tmp_path):
    noisy = np.random.default_rng(29).integers(0, 255, size=(64, 64, 4), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", noisy)
    assert ok
    payload = encoded.tobytes()
    archive = write_zip(tmp_path, {"head.png": payload})
    limit = archive.stat().st_size + len(payload)
    store = RigStore(storage(tmp_path, max_rigs=2, rig_storage_max_bytes=limit))
    store.install_zip("first", archive)
    with pytest.raises(StoreError) as excinfo:
        store.install_zip("second", archive)
    assert excinfo.value.code == "storage_full"
    assert [rig.name for rig in store.list()] == ["first"]
    assert store._reserved_bytes == 0
    assert store._active_extractions == set()


def test_concurrent_rig_archive_reservations_cannot_exceed_quota(tmp_path):
    noisy = np.random.default_rng(31).integers(0, 255, size=(64, 64, 4), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", noisy)
    assert ok
    archive = rig_zip({"head.png": encoded.tobytes()})
    store = RigStore(
        storage(
            tmp_path,
            rig_zip_max_bytes=len(archive),
            rig_storage_max_bytes=len(archive),
        )
    )
    reservations = [store.open_staging() for _ in range(2)]
    barrier = threading.Barrier(2)

    def attempt(reservation):
        barrier.wait()
        try:
            reservation.write(archive)
            return reservation
        except StoreError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result(2.0)
            for future in (
                pool.submit(attempt, reservations[0]),
                pool.submit(attempt, reservations[1]),
            )
        ]
    failures = [result for result in results if isinstance(result, StoreError)]
    winners = [result for result in results if not isinstance(result, StoreError)]
    assert len(failures) == len(winners) == 1
    assert failures[0].code == "storage_full"
    winners[0].abort()
    assert store._reserved_bytes == 0
    assert store._active_uploads == {}
    assert list(store.directory.iterdir()) == []


def test_pending_rig_uploads_reserve_aggregate_rig_slots(tmp_path):
    store = RigStore(storage(tmp_path, max_rigs=1))
    first = store.open_staging()
    assert store._reserved_rigs == 1
    with pytest.raises(StoreError) as excinfo:
        store.open_staging()
    assert excinfo.value.code == "storage_full"
    first.abort()
    assert store._reserved_rigs == 0
    replacement = store.open_staging()
    replacement.abort()
    assert store._reserved_rigs == 0


def test_post_rename_rig_permission_failure_removes_publication(tmp_path, monkeypatch):
    store = RigStore(storage(tmp_path))

    def fail_secure(_path):
        raise PermissionError("cannot secure installed rig")

    monkeypatch.setattr(store, "_secure_installed_rig", fail_secure)
    with pytest.raises(StoreError) as excinfo:
        store.install_zip(
            "not-published", write_zip(tmp_path, {"head.png": layer_png()})
        )
    assert excinfo.value.code == "insufficient_storage"
    assert not (store.directory / "not-published").exists()
    assert store._reserved_bytes == 0
    assert store._active_extractions == set()
    assert not list(store.directory.glob(".staged-*"))


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


def profiled_oriented_image_bytes() -> bytes:
    rgb = np.empty((80, 120, 3), dtype=np.uint8)
    rgb[:40, :60] = (220, 40, 30)
    rgb[:40, 60:] = (20, 190, 60)
    rgb[40:, :60] = (30, 60, 220)
    rgb[40:, 60:] = (180, 170, 35)
    exif = Image.Exif()
    exif[274] = 6
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    payload = io.BytesIO()
    Image.fromarray(rgb).save(
        payload,
        format="PNG",
        exif=exif,
        icc_profile=profile,
    )
    return payload.getvalue()


def malformed_profile_image_bytes() -> bytes:
    payload = io.BytesIO()
    Image.new("RGB", (32, 24), (40, 90, 160)).save(
        payload,
        format="PNG",
        icc_profile=b"not-an-icc-profile",
    )
    return payload.getvalue()


def staged(store: MediaStore, payload: bytes) -> UploadReservation:
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


def test_media_header_mismatch_is_rejected_before_opencv(tmp_path, monkeypatch):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame)
    assert ok
    store = MediaStore(storage(tmp_path))
    monkeypatch.setattr(
        store_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCV must not decode a mismatched image header")
        ),
    )
    with pytest.raises(StoreError) as excinfo:
        store.commit(staged(store, jpeg.tobytes()), "forged.png", "image")
    assert excinfo.value.code == "invalid_media"
    assert store.list() == []


def test_media_image_validation_uses_only_shared_color_decoder(
    tmp_path,
    monkeypatch,
) -> None:
    store = MediaStore(storage(tmp_path))
    monkeypatch.setattr(
        store_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: pytest.fail(
            "MediaStore must not decode a Pillow-validated path through OpenCV"
        ),
    )

    saved = store.commit(staged(store, image_bytes()), "shared.png", "image")

    assert saved.name == "shared.png"
    assert store.image_max_pixels == 16_777_216


def test_media_store_rejects_malformed_embedded_icc(tmp_path) -> None:
    store = MediaStore(storage(tmp_path))

    with pytest.raises(StoreError) as excinfo:
        store.commit(
            staged(store, malformed_profile_image_bytes()),
            "malformed-profile.png",
            "image",
        )

    assert excinfo.value.code == "invalid_media"
    assert store.list() == []


def test_video_validation_rejects_malformed_later_frame_and_releases_capture(
    tmp_path, monkeypatch
):
    released = False

    class Capture:
        def __init__(self, _path):
            self.frames = iter(
                [
                    (True, np.zeros((4, 4, 3), dtype=np.uint8)),
                    (True, np.zeros((4, 4), dtype=np.uint8)),
                ]
            )

        def isOpened(self):
            return True

        def get(self, _property):
            return 4

        def read(self):
            return next(self.frames, (False, None))

        def release(self):
            nonlocal released
            released = True

    monkeypatch.setattr(store_mod.cv2, "VideoCapture", Capture)
    store = MediaStore(storage(tmp_path))
    avi_header = b"RIFF" + b"\x00" * 4 + b"AVI " + b"payload"
    with pytest.raises(StoreError) as excinfo:
        store.commit(staged(store, avi_header), "malformed.avi", "video")
    assert excinfo.value.code == "invalid_media"
    assert released
    assert store.list() == []


def test_media_quota_limits(tmp_path):
    store = MediaStore(storage(tmp_path, max_files=1))
    store.commit(staged(store, image_bytes()), "one.png", "image")
    with pytest.raises(StoreError) as excinfo:
        store.commit(staged(store, image_bytes()), "two.png", "image")
    assert excinfo.value.code == "storage_full"


def test_media_reservations_make_concurrent_chunk_quota_atomic(tmp_path):
    noisy = np.random.default_rng(23).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", noisy)
    assert ok
    payload = encoded.tobytes()
    cfg = storage(
        tmp_path,
        image_max_bytes=max(1024, len(payload)),
        video_max_bytes=max(1024, len(payload)),
        storage_max_bytes=max(1024, len(payload)),
        max_files=2,
    )
    store = MediaStore(cfg)
    reservations = [store.open_staging("image") for _ in range(2)]
    barrier = threading.Barrier(2)

    def attempt(reservation):
        barrier.wait()
        try:
            reservation.write(payload)
            return reservation
        except StoreError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result(2.0)
            for future in (
                pool.submit(attempt, reservations[0]),
                pool.submit(attempt, reservations[1]),
            )
        ]
    failures = [result for result in results if isinstance(result, StoreError)]
    winners = [result for result in results if not isinstance(result, StoreError)]
    assert len(failures) == len(winners) == 1
    assert failures[0].code == "storage_full"
    saved = store.commit(winners[0], "winner.png", "image")
    assert Path(saved.path).stat().st_size == len(payload)
    assert store._reserved_bytes == 0
    assert store._reserved_files == 0
    assert not list(store.directory.glob(".upload-*"))


def test_failed_media_validation_consumes_and_cleans_reservation(tmp_path):
    store = MediaStore(storage(tmp_path))
    reservation = store.open_staging("image")
    reservation.write(b"not an image")
    with pytest.raises(StoreError) as excinfo:
        store.commit(reservation, "bad.png", "image")
    assert excinfo.value.code == "invalid_media"
    assert not reservation.active
    assert store._reserved_bytes == 0
    assert store._reserved_files == 0
    assert list(store.directory.iterdir()) == []


@pytest.mark.parametrize("kind", ["media", "rig"])
def test_failed_staging_unlink_remains_reserved_and_retryable(
    tmp_path,
    monkeypatch,
    kind,
):
    cfg = storage(tmp_path)
    if kind == "media":
        store = MediaStore(cfg)
        reservation = store.open_staging("image")
    else:
        store = RigStore(cfg)
        reservation = store.open_staging()
    reservation.write(b"reserved bytes")
    original_unlink = Path.unlink
    failed = False

    def fail_once(path, *args, **kwargs):
        nonlocal failed
        if path.name.startswith(".custback-cleanup-") and not failed:
            failed = True
            raise OSError("staging unlink failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(OSError, match="staging unlink failed"):
        reservation.abort()
    assert reservation.active
    assert store._active_uploads[reservation.path] is reservation
    assert store._reserved_bytes == len(b"reserved bytes")
    assert (
        getattr(
            store,
            "_reserved_files" if kind == "media" else "_reserved_rigs",
        )
        == 1
    )
    assert reservation.path.exists()

    reservation.abort()
    assert not reservation.active
    assert reservation.path not in store._active_uploads
    assert store._reserved_bytes == 0
    assert (
        getattr(
            store,
            "_reserved_files" if kind == "media" else "_reserved_rigs",
        )
        == 0
    )
    assert not reservation.path.exists()


def test_media_commit_rejects_bytes_written_outside_reservation(tmp_path):
    store = MediaStore(storage(tmp_path))
    reservation = staged(store, image_bytes())
    reservation.seal()
    with reservation.path.open("ab") as bypass:
        bypass.write(b"unreserved")
    with pytest.raises(StoreError) as excinfo:
        store.commit(reservation, "tampered.png", "image")
    assert excinfo.value.code == "invalid_reservation"
    assert not reservation.active
    assert store._reserved_bytes == 0
    assert store._reserved_files == 0
    assert list(store.directory.iterdir()) == []


def test_crash_left_media_staging_counts_toward_quota(tmp_path):
    cfg = storage(
        tmp_path,
        image_max_bytes=1024,
        video_max_bytes=1024,
        storage_max_bytes=1024,
    )
    directory = Path(cfg.backgrounds_dir)
    directory.mkdir()
    orphan = directory / ".upload-crash.part"
    orphan.write_bytes(b"x" * 1024)
    store = MediaStore(cfg)
    reservation = store.open_staging("image")
    with pytest.raises(StoreError) as excinfo:
        reservation.write(b"y")
    assert excinfo.value.code == "storage_full"
    assert orphan.exists()
    assert not reservation.active
    assert store._reserved_bytes == 0
    assert store._reserved_files == 0


def test_private_modes_hold_under_restrictive_umask(tmp_path):
    cfg = storage(tmp_path)
    media_store = MediaStore(cfg)
    rig_store = RigStore(cfg)
    archive = write_zip(tmp_path, {"head.png": layer_png()})
    previous = os.umask(0o777)
    try:
        media = media_store.commit(
            staged(media_store, image_bytes()), "private.png", "image"
        )
        rig_store.install_zip("private", archive)
    finally:
        os.umask(previous)
    expected = {
        media_store.directory: 0o700,
        Path(media.path): 0o600,
        rig_store.directory: 0o700,
        rig_store.directory / "private": 0o700,
        rig_store.directory / "private" / "head.png": 0o600,
    }
    assert {path: stat.S_IMODE(path.stat().st_mode) for path in expected} == expected


def test_permission_doctor_repairs_modes_and_refuses_symlinks(tmp_path):
    cfg = storage(tmp_path)
    rigs = Path(cfg.rigs_dir)
    media = Path(cfg.backgrounds_dir)
    (rigs / "old-rig").mkdir(parents=True)
    old_layer = rigs / "old-rig" / "head.png"
    old_layer.write_bytes(layer_png())
    media.mkdir()
    old_media = media / "old.png"
    old_media.write_bytes(image_bytes())
    rigs.chmod(0o755)
    (rigs / "old-rig").chmod(0o755)
    old_layer.chmod(0o000)
    media.chmod(0o755)
    old_media.chmod(0o644)

    issues = audit_storage_permissions(cfg)
    assert {issue.path for issue in issues} == {
        rigs,
        rigs / "old-rig",
        old_layer,
        media,
        old_media,
    }
    repaired = set(repair_storage_permissions(cfg))
    assert repaired == {issue.path for issue in issues}
    assert audit_storage_permissions(cfg) == ()

    target = tmp_path / "outside.png"
    target.write_bytes(b"outside")
    target.chmod(0o644)
    link = media / "linked.png"
    link.symlink_to(target)
    assert any(issue.reason == "symlink" for issue in audit_storage_permissions(cfg))
    with pytest.raises(StoreError) as excinfo:
        repair_storage_permissions(cfg)
    assert excinfo.value.code == "unsafe_storage_path"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_permission_doctor_reaudits_after_execute_only_parent(tmp_path):
    cfg = storage(tmp_path)
    rigs = Path(cfg.rigs_dir)
    rig = rigs / "hidden-rig"
    rig.mkdir(parents=True)
    layer = rig / "head.png"
    layer.write_bytes(layer_png())
    rig.chmod(0o777)
    layer.chmod(0o666)
    rigs.chmod(0o100)

    repaired = set(repair_storage_permissions(cfg))
    assert repaired == {rigs, rig, layer}
    assert audit_storage_permissions(cfg) == ()


def test_permission_repair_detects_inode_swap_before_fchmod(tmp_path, monkeypatch):
    managed = tmp_path / "managed.png"
    managed.write_bytes(b"original")
    managed.chmod(0o644)
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(b"replacement")
    replacement.chmod(0o644)
    displaced = tmp_path / "displaced.png"
    real_open = store_mod.os.open
    swapped = False

    def swap_then_open(path, flags, *args):
        nonlocal swapped
        if Path(path) == managed and not swapped:
            swapped = True
            managed.rename(displaced)
            replacement.rename(managed)
        return real_open(path, flags, *args)

    monkeypatch.setattr(store_mod.os, "open", swap_then_open)
    with pytest.raises(StoreError) as excinfo:
        store_mod._secure_existing(managed, 0o600, directory=False)
    assert excinfo.value.code == "unsafe_storage_path"
    assert managed.read_bytes() == b"replacement"
    assert stat.S_IMODE(managed.stat().st_mode) == 0o644
    assert stat.S_IMODE(displaced.stat().st_mode) == 0o644


def test_stored_path_refuses_traversal(tmp_path):
    store = MediaStore(storage(tmp_path))
    store.commit(staged(store, image_bytes()), "ok.png", "image")
    for name in ("../ok.png", ".hidden.png", "sub/ok.png", ""):
        with pytest.raises(StoreError) as excinfo:
            store.stored_path(name)
        assert excinfo.value.code == "media_not_found"


def test_store_reads_refuse_symlinked_roots(tmp_path):
    real_rigs = tmp_path / "real-rigs"
    rig = real_rigs / "linked-rig"
    rig.mkdir(parents=True)
    (rig / "head.png").write_bytes(layer_png())
    real_media = tmp_path / "real-media"
    real_media.mkdir()
    (real_media / "linked.png").write_bytes(image_bytes())
    rigs_link = tmp_path / "rigs-link"
    media_link = tmp_path / "media-link"
    rigs_link.symlink_to(real_rigs, target_is_directory=True)
    media_link.symlink_to(real_media, target_is_directory=True)
    cfg = StorageConfig.model_validate(
        {
            "rigs_dir": str(rigs_link),
            "backgrounds_dir": str(media_link),
        }
    )
    rig_store = RigStore(cfg)
    media_store = MediaStore(cfg)
    assert resolve_rig_selector("linked-rig", rigs_link) == "linked-rig"
    for operation in (
        rig_store.list,
        lambda: rig_store.stored_path("linked-rig"),
        media_store.list,
        lambda: media_store.stored_path("linked.png"),
    ):
        with pytest.raises(StoreError) as excinfo:
            operation()
        assert excinfo.value.code == "unsafe_storage_path"
    assert (rig / "head.png").read_bytes() == layer_png()
    assert (real_media / "linked.png").read_bytes() == image_bytes()


# -- thumbnails ---------------------------------------------------------------


def test_render_avatar_thumbnail_builtin():
    data = render_avatar_thumbnail("builtin", avatar="nova", style="cartoon")
    assert jpeg_size(data) == (256, 144)


def test_render_media_thumbnail_downscales(tmp_path):
    store = MediaStore(storage(tmp_path))
    saved = store.commit(staged(store, image_bytes(1280, 720)), "wide.png", "image")
    data = render_media_thumbnail(Path(saved.path), "image")
    width, height = jpeg_size(data)
    assert (width, height) == (256, 144)


def test_color_managed_thumbnail_matches_full_oriented_backdrop(tmp_path) -> None:
    path = tmp_path / "profiled-oriented.png"
    path.write_bytes(profiled_oriented_image_bytes())

    full = ImageBackdrop(str(path)).frame(80, 120)
    thumbnail_jpeg = render_media_thumbnail(
        path,
        "image",
        max_pixels=80 * 120,
    )
    thumbnail = cv2.imdecode(
        np.frombuffer(thumbnail_jpeg, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    assert thumbnail is not None
    assert thumbnail.shape == full.shape == (120, 80, 3)
    difference = np.abs(thumbnail.astype(np.int16) - full.astype(np.int16))
    assert float(np.mean(difference)) < 3.0
    assert float(np.quantile(difference, 0.95)) < 8.0


def test_thumbnail_rejects_malformed_icc_and_enforces_pixel_limit(tmp_path) -> None:
    malformed = tmp_path / "malformed.png"
    malformed.write_bytes(malformed_profile_image_bytes())
    with pytest.raises(StoreError) as excinfo:
        render_media_thumbnail(malformed, "image")
    assert excinfo.value.code == "invalid_media"

    valid = tmp_path / "oversized.png"
    valid.write_bytes(image_bytes(40, 30))
    with pytest.raises(StoreError) as excinfo:
        render_media_thumbnail(valid, "image", max_pixels=1000)
    assert excinfo.value.code == "media_too_large"


def test_thumbnail_cache_evicts_oldest():
    cache = ThumbnailCache(capacity=2)
    cache.put(("a",), b"1")
    cache.put(("b",), b"2")
    assert cache.get(("a",)) == b"1"
    cache.put(("c",), b"3")
    assert cache.get(("b",)) is None
    assert cache.get(("a",)) == b"1"
    assert cache.get(("c",)) == b"3"
