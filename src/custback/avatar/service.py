"""The avatar service loop: custback frames in, avatar frames out.

Connects to custback's ``WS /ws/frames?stream=raw`` (locally or across
hosts), animates the rig from the configured driver, composites it over the
selected background at exactly the incoming frame size, and returns JPEG
frames on the same socket. With ``background.mode: remote`` on the custback
side these frames become the virtual camera output; custback's privacy-safe
custback's fixed privacy slate covers every stall or disconnect of this service.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import websockets

from ..api.security import (
    create_client_ssl_context,
    resolve_renderer_token,
    validate_outbound_endpoint,
)
from ..backgrounds import BackdropProvider
from ..hub import _Slot
from .config import AvatarConfig, AvatarRuntime
from .drivers import FaceDriver, create_driver
from .renderer import blurred_room, compose_avatar, create_avatar_backdrop
from .rig import Rig, create_rig
from .state import StateSmoother
from .store import resolve_rig_selector

try:
    import cv2
except ImportError:  # pragma: no cover - required by the package, defensive
    cv2 = None

log = logging.getLogger(__name__)

RAW_STREAM_PATH = "/ws/frames?stream=raw"


class _LatestBytes:
    """Latest-value slot for the newest raw frame (drops the backlog)."""

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._value: bytes | None = None
        self._seq = 0

    async def put(self, value: bytes) -> None:
        async with self._cond:
            self._value = value
            self._seq += 1
            self._cond.notify_all()

    async def get(self, last_seq: int, timeout: float) -> tuple[bytes | None, int]:
        async with self._cond:
            try:
                await asyncio.wait_for(
                    self._cond.wait_for(
                        lambda: self._value is not None and self._seq != last_seq
                    ),
                    timeout,
                )
            except asyncio.TimeoutError:
                return None, last_seq
            return self._value, self._seq


@dataclass
class _Components:
    """Live render components matching one configuration version."""

    version: int = -1
    driver: FaceDriver | None = None
    rig: Rig | None = None
    backdrop: BackdropProvider | None = None
    smoother: StateSmoother | None = None
    driver_key: tuple | None = None
    rig_key: tuple | None = None
    background_key: tuple | None = None

    def close(self) -> None:
        for resource in (self.driver, self.rig, self.backdrop):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    log.debug("avatar component close failed", exc_info=True)


class AvatarService:
    """Renders avatar frames for one custback source until stopped."""

    def __init__(
        self,
        runtime: AvatarRuntime,
        *,
        driver_factory: Callable[..., FaceDriver] = create_driver,
        allow_model_download: bool = True,
    ):
        if cv2 is None:
            raise RuntimeError("opencv-python is required for the avatar service")
        self.runtime = runtime
        self._driver_factory = driver_factory
        self._allow_model_download = allow_model_download
        source = runtime.read().config.source
        source_endpoint = validate_outbound_endpoint(
            source.url,
            kind="websocket",
            label="source.url",
        )
        assert source_endpoint is not None
        self._source_ssl = create_client_ssl_context(
            source_endpoint,
            ca_file=source.tls_ca_file,
            certfile=source.tls_certfile,
            keyfile=source.tls_keyfile,
            label="source",
        )
        self.output = _Slot()  # latest rendered BGR frame, for snapshot/MJPEG
        self._components = _Components()
        self._stats_lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "connected": False,
            "frames_received": 0,
            "frames_rendered": 0,
            "frames_sent": 0,
            "render_failures": 0,
            "connect_attempts": 0,
            "reconnects": 0,
            "driver_backend": "",
            "driver_device": "",
            "face_present": False,
            "driver_ms": None,
            "render_ms": None,
            "output_width": None,
            "output_height": None,
            "last_error": "",
            "started_at": time.time(),
        }
        self._epoch = time.monotonic()

    # -- observability -----------------------------------------------------
    def _update_stats(self, **kwargs: Any) -> None:
        with self._stats_lock:
            self._stats.update(kwargs)

    def _count(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] += amount

    def stats_dict(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        started_at = stats.pop("started_at")
        stats["uptime_s"] = round(time.time() - started_at, 1)
        stats["config_version"] = self.runtime.version
        for key in ("driver_ms", "render_ms"):
            if stats[key] is not None:
                stats[key] = round(stats[key], 1)
        return stats

    # -- rendering (worker thread; one call in flight at a time) -----------
    def _ensure_components(self, cfg: AvatarConfig, version: int) -> _Components:
        components = self._components
        if components.version == version:
            return components
        driver_key = (
            cfg.driver.backend,
            cfg.driver.vision.model_path,
            cfg.driver.audio2face.url,
            cfg.driver.audio2face.tls_ca_file,
            cfg.driver.audio2face.tls_certfile,
            cfg.driver.audio2face.tls_keyfile,
            cfg.driver.audio2face.audio_source,
            cfg.driver.audio2face.sample_rate,
            cfg.driver.audio2face.chunk_ms,
        )
        background_key = (
            cfg.background.mode,
            cfg.background.color,
            cfg.background.image_path,
            cfg.background.video_path,
        )
        if components.driver is None or components.driver_key != driver_key:
            driver = self._driver_factory(
                cfg.driver, allow_model_download=self._allow_model_download
            )
            if components.driver is not None:
                components.driver.close()
            components.driver = driver
            components.driver_key = driver_key
            components.smoother = StateSmoother(cfg.driver.smoothing)
            self._update_stats(
                driver_backend=driver.name, driver_device=driver.device
            )
        if components.smoother is None or components.smoother.factor != cfg.driver.smoothing:
            components.smoother = StateSmoother(cfg.driver.smoothing)
        rig_selector = resolve_rig_selector(cfg.appearance.rig, cfg.storage.rigs_dir)
        rig_key = (rig_selector, cfg.appearance.avatar, cfg.appearance.style)
        if components.rig is None or components.rig_key != rig_key:
            rig = create_rig(
                rig_selector,
                avatar=cfg.appearance.avatar,
                style=cfg.appearance.style,
            )
            if components.rig is not None:
                components.rig.close()
            components.rig = rig
            components.rig_key = rig_key
        if components.backdrop is None or components.background_key != background_key:
            backdrop = create_avatar_backdrop(cfg.background)
            if components.backdrop is not None:
                components.backdrop.close()
            components.backdrop = backdrop
            components.background_key = background_key
        components.version = version
        return components

    def _process(self, data: bytes) -> bytes | None:
        state = self.runtime.read()
        cfg = state.config
        try:
            components = self._ensure_components(cfg, state.version)
        except Exception as exc:
            # Keep rendering with the previous components; a bad hot patch
            # must not take the avatar (and the meeting) down.
            log.warning("avatar reconfiguration failed: %s", exc)
            self._update_stats(last_error=f"reconfigure: {type(exc).__name__}")
            components = self._components
            if components.driver is None or components.rig is None:
                raise
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            self._count("render_failures")
            return None
        height, width = frame.shape[:2]
        timestamp = time.monotonic() - self._epoch
        started = time.perf_counter()
        face = components.driver.update(frame, timestamp)
        assert components.smoother is not None
        face = components.smoother.apply(face)
        driver_ms = (time.perf_counter() - started) * 1000.0
        sprite = components.rig.render(face, frozenset(cfg.appearance.parts))
        if components.backdrop is not None:
            backdrop = components.backdrop.frame(width, height)
        else:
            backdrop = blurred_room(frame, cfg.background.blur_strength)
        rendered = compose_avatar(
            sprite,
            backdrop,
            cfg.appearance,
            framing_window=components.rig.framing_window(cfg.appearance.framing),
        )
        render_ms = (time.perf_counter() - started) * 1000.0
        ok, jpeg = cv2.imencode(
            ".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, cfg.render.jpeg_quality]
        )
        if not ok:
            self._count("render_failures")
            return None
        self.output.put(rendered)
        self._count("frames_rendered")
        self._update_stats(
            driver_ms=driver_ms,
            render_ms=render_ms,
            face_present=face.present,
            output_width=width,
            output_height=height,
        )
        return jpeg.tobytes()

    # -- session -----------------------------------------------------------
    async def _session(self, ws, stop: asyncio.Event) -> None:
        slot = _LatestBytes()

        async def receiver() -> None:
            while True:
                data = await ws.recv()
                if isinstance(data, bytes):
                    self._count("frames_received")
                    await slot.put(data)

        async def renderer() -> None:
            seq = -1
            last_sent = 0.0
            while True:
                data, seq = await slot.get(seq, 0.5)
                if data is None:
                    continue
                min_interval = 1.0 / self.runtime.read().config.render.max_fps
                now = asyncio.get_running_loop().time()
                if now - last_sent < min_interval:
                    await asyncio.sleep(min_interval - (now - last_sent))
                    # Render the newest frame available after pacing.
                    newest, seq = await slot.get(seq - 1, 0.01)
                    if newest is not None:
                        data = newest
                payload = await asyncio.to_thread(self._process, data)
                if payload is not None:
                    await ws.send(payload)
                    last_sent = asyncio.get_running_loop().time()
                    self._count("frames_sent")

        stopper = asyncio.create_task(stop.wait())
        tasks = {asyncio.create_task(receiver()), asyncio.create_task(renderer())}
        try:
            done, _pending = await asyncio.wait(
                tasks | {stopper}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task is not stopper:
                    task.result()  # propagate the session failure
        finally:
            stopper.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, stopper, return_exceptions=True)

    async def run(self, stop: asyncio.Event) -> None:
        """Connect, render, and reconnect with backoff until ``stop`` is set."""
        connect_parameters = inspect.signature(websockets.connect).parameters
        header_arg = (
            "additional_headers"
            if "additional_headers" in connect_parameters
            else "extra_headers"
        )
        proxy_args = {"proxy": None} if "proxy" in connect_parameters else {}
        had_session = False
        backoff = self.runtime.read().config.source.reconnect_min_s
        try:
            while not stop.is_set():
                source = self.runtime.read().config.source
                try:
                    token = await asyncio.to_thread(
                        resolve_renderer_token, source.token_file
                    )
                    self._count("connect_attempts")
                    tls_args = (
                        {"ssl": self._source_ssl}
                        if self._source_ssl is not None
                        else {}
                    )
                    async with websockets.connect(
                        source.url + RAW_STREAM_PATH,
                        max_size=source.frame_max_bytes,
                        open_timeout=source.connect_timeout_s,
                        **proxy_args,
                        **tls_args,
                        **{header_arg: {"Authorization": f"Bearer {token.value}"}},
                    ) as ws:
                        if had_session:
                            self._count("reconnects")
                        had_session = True
                        self._update_stats(connected=True, last_error="")
                        log.info("connected to custback frame stream")
                        try:
                            await self._session(ws, stop)
                        finally:
                            self._update_stats(connected=False)
                    backoff = source.reconnect_min_s
                    if stop.is_set():
                        return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # Never log the URL or token; the type name is enough to
                    # distinguish refused/timeout/handshake/closed cases.
                    self._update_stats(
                        connected=False, last_error=type(exc).__name__
                    )
                    log.warning(
                        "custback stream unavailable (%s); retrying",
                        type(exc).__name__,
                    )
                delay = backoff
                while delay > 0 and not stop.is_set():
                    step = min(delay, 0.2)
                    try:
                        await asyncio.wait_for(stop.wait(), step)
                    except asyncio.TimeoutError:
                        pass
                    delay -= step
                source_now = self.runtime.read().config.source
                backoff = min(backoff * 2.0, source_now.reconnect_max_s)
        finally:
            self._components.close()
