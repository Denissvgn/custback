"""On-host stores for uploaded avatar rigs and scene media, plus tile thumbnails.

Rigs and avatar backgrounds must live on the machine that runs the avatar
service — that is where :class:`~custback.avatar.rig.LayeredRig` loads PNG
layers and where the backdrop providers open media files. Uploads therefore
arrive through the avatar control API (possibly proxied by custback) and
land here.

Rig archives are never extracted with ``ZipFile.extract``: every member is
allow-listed by basename (``<part>.png``, expression variants, ``rig.yaml``),
re-written under our own name, and capped in count and uncompressed size, so
zip-slip and zip-bomb payloads cannot escape or exhaust the host. A staged
rig only becomes visible after :class:`~custback.avatar.rig.LayeredRig`
accepts it.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
import threading
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..backgrounds import IMAGE_EXTS, VIDEO_EXTS
from .config import (
    AVATAR_PARTS,
    AppearanceConfig,
    StorageConfig,
)
from .renderer import compose_avatar
from .rig import RigError, create_rig
from .state import FaceState

try:
    import cv2
except ImportError:  # pragma: no cover - required by the package, defensive
    cv2 = None

try:  # Pillow supplies a safe header/dimension check before OpenCV allocates.
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - OpenCV remains a compatibility fallback
    Image = None
    UnidentifiedImageError = OSError

log = logging.getLogger(__name__)

RIG_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}\Z")
_MEDIA_STEM_MAX = 48
_ZIP_READ_CHUNK = 1024 * 1024

# The only files a rig archive may carry (optionally under one shared
# top-level directory, which is flattened away).
RIG_MEMBER_NAMES = frozenset(
    {f"{part}.png" for part in AVATAR_PARTS}
    | {"eyes_closed.png", "mouth_open.png", "rig.yaml"}
)

THUMBNAIL_SIZE = (256, 144)  # 16:9, like a meeting tile
_THUMBNAIL_BACKDROP_BGR = (52, 44, 38)
_THUMBNAIL_JPEG_QUALITY = 82


class StoreError(Exception):
    """Storage failure carrying a stable API error code."""

    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class InstalledRig:
    name: str
    parts: tuple[str, ...]
    has_manifest: bool
    size_bytes: int


@dataclass(frozen=True)
class StoredMedia:
    name: str
    kind: str  # "image" | "video"
    size: int
    path: str


def resolve_rig_selector(selector: str, rigs_dir: str | Path) -> str:
    """Map ``appearance.rig`` to something :func:`create_rig` understands.

    ``builtin`` and existing directories pass through; a bare installed-rig
    name resolves to its directory in ``rigs_dir``. Unknown selectors are
    returned unchanged so rig loading reports its usual error.
    """
    if selector == "builtin":
        return selector
    candidate = Path(selector).expanduser()
    if candidate.is_dir():
        return str(candidate)
    if RIG_NAME_RE.fullmatch(selector):
        installed = Path(rigs_dir).expanduser() / selector
        if installed.is_dir():
            return str(installed)
    return selector


def sanitize_media_name(name: str, kind: str) -> str:
    """Normalize an upload name to a safe ``stem.ext`` for ``kind``."""
    allowed = IMAGE_EXTS if kind == "image" else VIDEO_EXTS
    base = os.path.basename(name.strip().replace("\\", "/")).lower()
    stem, dot, suffix = base.rpartition(".")
    suffix = f".{suffix}" if dot else ""
    if suffix not in allowed:
        raise StoreError(
            415,
            "unsupported_media_type",
            f"{kind} name must end in one of: {', '.join(sorted(allowed))}",
        )
    stem = re.sub(r"[^a-z0-9._-]+", "-", stem).strip(".-")
    if not stem:
        stem = f"{kind}-{secrets.token_hex(4)}"
    return f"{stem[:_MEDIA_STEM_MAX]}{suffix}"


def _neutral_face_state() -> FaceState:
    """A pleasant resting pose for thumbnails: soft smile, slight turn."""
    state = FaceState.neutral()
    state.yaw = 0.05
    state.set_channel("mouthSmileLeft", 0.22)
    state.set_channel("mouthSmileRight", 0.22)
    return state


def render_avatar_thumbnail(
    selector: str,
    *,
    avatar: str = "casey",
    style: str = "cartoon",
    framing: str = "bust",
    size: tuple[int, int] = THUMBNAIL_SIZE,
) -> bytes:
    """Render one avatar tile as JPEG bytes (raises RigError on bad rigs)."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required for thumbnails")
    width, height = size
    rig = create_rig(selector, avatar=avatar, style=style)
    try:
        sprite = rig.render(_neutral_face_state(), frozenset(AVATAR_PARTS))
        backdrop = np.full(
            (height, width, 3), _THUMBNAIL_BACKDROP_BGR, dtype=np.uint8
        )
        frame = compose_avatar(
            sprite,
            backdrop,
            AppearanceConfig(framing=framing),
            framing_window=rig.framing_window(framing),
        )
    finally:
        rig.close()
    ok, jpeg = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _THUMBNAIL_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("thumbnail JPEG encoding failed")
    return jpeg.tobytes()


def render_media_thumbnail(
    path: Path, kind: str, *, size: tuple[int, int] = THUMBNAIL_SIZE
) -> bytes:
    """First frame (video) or downscaled image as JPEG bytes."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required for thumbnails")
    if kind == "image":
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    else:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            if not ok:
                frame = None
        finally:
            capture.release()
    if frame is None or frame.ndim != 3:
        raise StoreError(422, "invalid_media", "stored media cannot be decoded")
    height, width = frame.shape[:2]
    scale = min(size[0] / width, size[1] / height, 1.0)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, jpeg = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _THUMBNAIL_JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("thumbnail JPEG encoding failed")
    return jpeg.tobytes()


class ThumbnailCache:
    """Small keyed JPEG cache; keys should include content identity (mtime)."""

    def __init__(self, capacity: int = 64):
        self._capacity = capacity
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, bytes] = OrderedDict()

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: tuple, value: bytes) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)


class RigStore:
    """Installed PNG-layer rigs under one managed directory."""

    def __init__(self, cfg: StorageConfig):
        self.directory = Path(cfg.rigs_dir).expanduser()
        self._cfg = cfg
        self._lock = threading.Lock()

    def rig_path(self, name: str) -> Path:
        if not RIG_NAME_RE.fullmatch(name):
            raise StoreError(
                422,
                "invalid_rig_name",
                "rig names are 1-32 lowercase letters, digits, '-' or '_' "
                "and start with a letter or digit",
            )
        return self.directory / name

    def _describe(self, path: Path) -> InstalledRig:
        parts = tuple(
            part for part in AVATAR_PARTS if (path / f"{part}.png").is_file()
        )
        size = sum(
            entry.stat().st_size for entry in path.iterdir() if entry.is_file()
        )
        return InstalledRig(
            name=path.name,
            parts=parts,
            has_manifest=(path / "rig.yaml").is_file(),
            size_bytes=size,
        )

    def list(self) -> list[InstalledRig]:
        if not self.directory.is_dir():
            return []
        rigs = []
        for entry in sorted(self.directory.iterdir()):
            if (
                entry.is_dir()
                and not entry.is_symlink()
                and RIG_NAME_RE.fullmatch(entry.name)
            ):
                rigs.append(self._describe(entry))
        return rigs

    @staticmethod
    def _member_basenames(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
        """Map allow-listed basenames to members, flattening one shared root."""
        files = [info for info in archive.infolist() if not info.is_dir()]
        if not files:
            raise StoreError(422, "invalid_rig", "rig archive contains no files")
        names = [info.filename.replace("\\", "/").lstrip("/") for info in files]
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        strip_root = (
            len(roots) == 1 and all("/" in name for name in names)
        )
        members: dict[str, zipfile.ZipInfo] = {}
        for info, name in zip(files, names):
            if strip_root:
                name = name.split("/", 1)[1]
            if "/" in name or name.startswith("."):
                raise StoreError(
                    422, "invalid_rig", f"unexpected archive entry: {info.filename}"
                )
            if name not in RIG_MEMBER_NAMES:
                raise StoreError(
                    422,
                    "invalid_rig",
                    f"unexpected archive entry {info.filename!r}; rigs may only "
                    "contain <part>.png layers, eyes_closed.png, mouth_open.png "
                    "and rig.yaml",
                )
            if name in members:
                raise StoreError(
                    422, "invalid_rig", f"duplicate archive entry: {name}"
                )
            members[name] = info
        return members

    def _extract(self, archive: zipfile.ZipFile, destination: Path) -> None:
        members = self._member_basenames(archive)
        if len(members) > self._cfg.rig_max_entries:
            raise StoreError(
                413,
                "rig_too_large",
                f"rig archives may contain at most {self._cfg.rig_max_entries} files",
            )
        remaining = self._cfg.rig_max_bytes
        for name, info in members.items():
            with archive.open(info) as source, open(destination / name, "wb") as out:
                while True:
                    chunk = source.read(min(_ZIP_READ_CHUNK, remaining + 1))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise StoreError(
                            413,
                            "rig_too_large",
                            "uncompressed rig exceeds "
                            f"{self._cfg.rig_max_bytes} bytes",
                        )
                    out.write(chunk)

    def install_zip(self, name: str, zip_path: Path) -> InstalledRig:
        """Validate and atomically install an uploaded rig archive."""
        final = self.rig_path(name)
        with self._lock:
            if final.exists():
                raise StoreError(
                    409,
                    "rig_exists",
                    f"rig {name!r} already exists; delete it first to replace it",
                )
            self.directory.mkdir(parents=True, exist_ok=True)
            staging = self.directory / f".staged-{secrets.token_hex(8)}"
            staging.mkdir()
            try:
                try:
                    with zipfile.ZipFile(zip_path) as archive:
                        self._extract(archive, staging)
                except zipfile.BadZipFile as exc:
                    raise StoreError(
                        422, "invalid_rig", "rig upload is not a valid zip archive"
                    ) from exc
                try:
                    create_rig(str(staging)).close()
                except RigError as exc:
                    raise StoreError(422, "invalid_rig", str(exc)) from exc
                os.replace(staging, final)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return self._describe(final)

    def remove(self, name: str, active_selector: str) -> None:
        path = self.rig_path(name)
        with self._lock:
            if not path.is_dir() or path.is_symlink():
                raise StoreError(404, "rig_not_found", f"no installed rig {name!r}")
            active = resolve_rig_selector(active_selector, self.directory)
            if active != "builtin" and Path(active) == path:
                raise StoreError(
                    409,
                    "rig_in_use",
                    f"rig {name!r} is the active appearance.rig; switch rigs first",
                )
            shutil.rmtree(path)


class MediaStore:
    """Uploaded avatar-scene images and videos under one managed directory."""

    def __init__(self, cfg: StorageConfig):
        self.directory = Path(cfg.backgrounds_dir).expanduser()
        self._cfg = cfg
        self._lock = threading.Lock()

    @staticmethod
    def kind_of(path: Path) -> str | None:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            return "image"
        if suffix in VIDEO_EXTS:
            return "video"
        return None

    def max_bytes(self, kind: str) -> int:
        return (
            self._cfg.image_max_bytes
            if kind == "image"
            else self._cfg.video_max_bytes
        )

    def open_staging(self) -> Path:
        """Reserve a staging file inside the store (same filesystem)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f".upload-{secrets.token_hex(16)}.part"

    def _entries(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return [
            entry
            for entry in sorted(self.directory.iterdir())
            if entry.is_file()
            and not entry.is_symlink()
            and not entry.name.startswith(".")
            and self.kind_of(entry) is not None
        ]

    def list(self) -> list[StoredMedia]:
        described = []
        for entry in self._entries():
            kind = self.kind_of(entry)
            assert kind is not None
            described.append(
                StoredMedia(
                    name=entry.name,
                    kind=kind,
                    size=entry.stat().st_size,
                    path=str(entry),
                )
            )
        return described

    def stored_path(self, name: str) -> Path:
        """A store entry by exact name, refusing traversal and symlinks."""
        if (
            not name
            or name.startswith(".")
            or os.path.basename(name.replace("\\", "/")) != name
        ):
            raise StoreError(404, "media_not_found", "no such stored file")
        path = self.directory / name
        if not path.is_file() or path.is_symlink() or self.kind_of(path) is None:
            raise StoreError(404, "media_not_found", "no such stored file")
        return path

    def _validate_image(self, path: Path) -> None:
        if Image is not None:
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except (UnidentifiedImageError, OSError) as exc:
                raise StoreError(
                    422, "invalid_media", "image upload cannot be decoded"
                ) from exc
            if width * height > self._cfg.image_max_pixels:
                raise StoreError(
                    413,
                    "media_too_large",
                    f"image exceeds {self._cfg.image_max_pixels} pixels",
                )
        if cv2 is not None and cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
            raise StoreError(422, "invalid_media", "image upload cannot be decoded")

    def _validate_video(self, path: Path) -> None:
        if cv2 is None:  # pragma: no cover - required by the package
            return
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise StoreError(422, "invalid_media", "video upload cannot be decoded")
        height, width = frame.shape[:2]
        if width > self._cfg.video_max_width or height > self._cfg.video_max_height:
            raise StoreError(
                413,
                "media_too_large",
                f"video frames exceed {self._cfg.video_max_width}"
                f"x{self._cfg.video_max_height}",
            )

    def commit(self, staging: Path, name: str, kind: str) -> StoredMedia:
        """Validate a fully-written staging file and publish it."""
        safe_name = sanitize_media_name(name, kind)
        size = staging.stat().st_size
        if size == 0:
            raise StoreError(422, "invalid_media", "upload is empty")
        if kind == "image":
            self._validate_image(staging)
        else:
            self._validate_video(staging)
        with self._lock:
            entries = self._entries()
            if len(entries) + 1 > self._cfg.max_files:
                raise StoreError(
                    507,
                    "storage_full",
                    f"the store already holds {self._cfg.max_files} files",
                )
            used = sum(entry.stat().st_size for entry in entries)
            if used + size > self._cfg.storage_max_bytes:
                raise StoreError(
                    507,
                    "storage_full",
                    "the upload does not fit within storage_max_bytes",
                )
            final = self.directory / safe_name
            stem, suffix = os.path.splitext(safe_name)
            attempt = 2
            while final.exists() or final.is_symlink():
                final = self.directory / f"{stem}-{attempt}{suffix}"
                attempt += 1
            os.replace(staging, final)
        kind_checked = self.kind_of(final)
        assert kind_checked == kind
        return StoredMedia(
            name=final.name, kind=kind, size=size, path=str(final)
        )

    def remove(self, name: str, background_cfg) -> None:
        with self._lock:
            path = self.stored_path(name)
            # Like custback's upload store, only the path the current mode
            # displays counts as active; stale paths from other modes don't
            # block deletion.
            if background_cfg.mode == "image":
                active = background_cfg.image_path
            elif background_cfg.mode == "video":
                active = background_cfg.video_path
            else:
                active = ""
            if active and Path(active).expanduser().resolve(
                strict=False
            ) == path.resolve(strict=False):
                raise StoreError(
                    409,
                    "media_in_use",
                    f"{name!r} is the active avatar background; switch first",
                )
            path.unlink()
