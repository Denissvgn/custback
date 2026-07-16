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
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np
import websockets

from ..api.security import (
    create_client_ssl_context,
    resolve_renderer_token,
    validate_outbound_endpoint,
)
from ..backgrounds import BackdropProvider
from ..hub import _Slot
from .config import (
    AvatarConfig,
    AvatarConfigState,
    AvatarConfigVersionConflictError,
    AvatarRuntime,
)
from .drivers import (
    DriverPreparation,
    DriverStartupError,
    FaceDriver,
    create_driver,
    prepare_driver,
)
from .renderer import blurred_room, compose_avatar, create_avatar_backdrop
from .rig import Rig, create_rig
from .state import FaceState, StateSmoother
from .store import is_rig_directory, resolve_rig_selector

try:
    import cv2
except ImportError:  # pragma: no cover - required by the package, defensive
    cv2 = None

log = logging.getLogger(__name__)

RAW_STREAM_PATH = "/ws/frames?stream=raw"
_ACTIVATION_TIMEOUT_S = 5.0
_TRIAL_FRAME_SHAPE = (96, 96, 3)
_T = TypeVar("_T")


class ActivationError(RuntimeError):
    """A validated avatar candidate could not activate its resources."""


class ReconfigurationUnavailable(RuntimeError):
    """The owned render lane cannot acknowledge a lifecycle request."""


class ConfigConflictError(RuntimeError):
    """An avatar patch was prepared from a generation that is now stale."""

    def __init__(self, expected_version: int, current_version: int):
        self.expected_version = expected_version
        self.current_version = current_version
        super().__init__(
            f"configuration changed concurrently: expected version "
            f"{expected_version}, current version is {current_version}"
        )


def _close_error(resource: Any, label: str) -> BaseException | None:
    if resource is None:
        return None
    try:
        resource.close()
    except BaseException as exc:
        log.exception("cannot close avatar %s", label)
        return exc
    return None
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
    """One complete, immutable-by-convention render generation."""

    config: AvatarConfig | None = None
    version: int = -1
    driver: FaceDriver | None = None
    rig: Rig | None = None
    backdrop: BackdropProvider | None = None
    smoother: StateSmoother | None = None
    driver_key: tuple | None = None
    rig_key: tuple | None = None
    background_key: tuple | None = None
    _closed: bool = False

    def close(self) -> list[tuple[Any, str, BaseException]]:
        if self._closed:
            return []
        failures: list[tuple[Any, str, BaseException]] = []
        for attribute, label in (
            ("driver", "driver"),
            ("rig", "rig"),
            ("backdrop", "backdrop"),
        ):
            resource = getattr(self, attribute)
            if resource is None:
                continue
            error = _close_error(resource, label)
            if error is None:
                setattr(self, attribute, None)
            else:
                failures.append((resource, label, error))
        self._closed = not failures
        return failures

    def close_replaced_by(
        self, current: "_Components"
    ) -> list[tuple[Any, str, BaseException]]:
        """Close only identities not transferred into ``current``."""

        if self._closed:
            return []
        failures: list[tuple[Any, str, BaseException]] = []
        for attribute, label in (
            ("driver", "replaced driver"),
            ("rig", "replaced rig"),
            ("backdrop", "replaced backdrop"),
        ):
            resource = getattr(self, attribute)
            if resource is None:
                continue
            if resource is getattr(current, attribute):
                # Ownership moved to the complete current generation.
                setattr(self, attribute, None)
                continue
            error = _close_error(resource, label)
            if error is None:
                setattr(self, attribute, None)
            else:
                failures.append((resource, label, error))
        self._closed = not failures
        return failures


@dataclass(frozen=True)
class _GenerationSnapshot:
    version: int
    storage_epoch: int
    driver_key: tuple | None
    rig_key: tuple | None
    background_key: tuple | None

    @property
    def active(self) -> bool:
        return self.version >= 0


@dataclass(frozen=True)
class _PreparedActivation:
    candidate: AvatarConfig
    expected_version: int
    snapshot: _GenerationSnapshot
    driver_key: tuple
    rig_key: tuple
    background_key: tuple
    rig_selector: str
    driver_preparation: DriverPreparation | None


@dataclass
class _ActivationRequest:
    prepared: _PreparedActivation
    publish: bool = True
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: bool = False
    result: AvatarConfigState | None = None
    error: BaseException | None = None

    def cancel(self) -> bool:
        with self.lock:
            if self.result is not None or self.error is not None:
                return False
            self.cancelled = True
            return True


@dataclass
class _MutationRequest:
    mutate: Callable[[AvatarConfig], None]
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: bool = False
    result: AvatarConfigState | None = None
    error: BaseException | None = None

    def cancel(self) -> bool:
        with self.lock:
            if self.result is not None or self.error is not None:
                return False
            self.cancelled = True
            return True


@dataclass(frozen=True)
class _RenderPublication:
    frame: np.ndarray
    version: int
    driver_ms: float
    render_ms: float
    face_present: bool
    width: int
    height: int


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
        self._runtime_writer = runtime._coordinator_writer()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="custback-avatar-render"
        )
        self._executor_lock = threading.Lock()
        self._lane_ident: int | None = None
        self._closing = False
        self._closed = False
        self._close_future: Future[None] | None = None
        self._deferred_closes: list[tuple[Any, str, BaseException]] = []
        self._lifecycle_error: ReconfigurationUnavailable | None = None
        self._asset_slot = threading.Lock()
        self._asset_threads_lock = threading.Lock()
        self._asset_threads: set[threading.Thread] = set()
        self._storage_epoch = 0
        self._defer_render_publication = False
        self._pending_render_publication: _RenderPublication | None = None
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
        service_ref = weakref.ref(self)

        def coordinate(patch: dict[str, Any]) -> AvatarConfigState:
            service = service_ref()
            if service is None:
                raise ReconfigurationUnavailable("avatar service is unavailable")
            return service.apply_config_patch(patch)

        runtime.bind_coordinator(coordinate)

    # -- observability -----------------------------------------------------
    def _update_stats(self, **kwargs: Any) -> None:
        with self._stats_lock:
            self._stats.update(kwargs)

    def _count(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] += amount

    def stats_dict(self) -> dict[str, Any]:
        # A generation swap happens under the runtime write lock. If the first
        # read races the assignment-only swap, the second read blocks until the
        # matching config/version publication completes.
        while True:
            state = self.runtime.read()
            components = self._components
            if components.version in (-1, state.version):
                break
        with self._stats_lock:
            stats = dict(self._stats)
        started_at = stats.pop("started_at")
        stats["uptime_s"] = round(time.time() - started_at, 1)
        stats["config_version"] = state.version
        if components.version == state.version and components.driver is not None:
            stats["driver_backend"] = components.driver.name
            stats["driver_device"] = components.driver.device
        for key in ("driver_ms", "render_ms"):
            if stats[key] is not None:
                stats[key] = round(stats[key], 1)
        return stats

    # -- owned render/activation lane -------------------------------------
    @staticmethod
    def _driver_key(cfg: AvatarConfig) -> tuple:
        audio = cfg.driver.audio2face
        return (
            cfg.driver.backend,
            cfg.driver.vision.model_path,
            audio.url,
            audio.tls_ca_file,
            audio.tls_certfile,
            audio.tls_keyfile,
            audio.audio_source,
            audio.sample_rate,
            audio.chunk_ms,
        )

    @staticmethod
    def _background_key(cfg: AvatarConfig) -> tuple:
        return (
            cfg.background.mode,
            cfg.background.color,
            cfg.background.image_path,
            cfg.background.video_path,
            cfg.storage.image_max_pixels,
            cfg.storage.video_max_width,
            cfg.storage.video_max_height,
        )

    @staticmethod
    def _rig_key(cfg: AvatarConfig, selector: str) -> tuple:
        return (
            selector,
            cfg.appearance.avatar,
            cfg.appearance.style,
            cfg.storage.rig_layer_max_pixels,
            cfg.storage.rig_total_max_pixels,
            cfg.storage.rig_manifest_max_bytes,
        )

    def _lane_entry(self, callback: Callable[..., _T], *args: Any) -> _T:
        ident = threading.get_ident()
        if self._lane_ident is None:
            self._lane_ident = ident
        elif self._lane_ident != ident:  # pragma: no cover - executor invariant
            raise RuntimeError("avatar render executor changed worker identity")
        return callback(*args)

    def _on_lane(self) -> bool:
        return self._lane_ident == threading.get_ident()

    def _submit_lane(self, callback: Callable[..., _T], *args: Any) -> Future[_T]:
        with self._executor_lock:
            if self._closing or self._closed:
                raise ReconfigurationUnavailable("avatar service is closing")
            return cast(
                Future[_T], self._executor.submit(self._lane_entry, callback, *args)
            )

    def _snapshot_generation(self) -> _GenerationSnapshot:
        if self._lifecycle_error is not None:
            raise self._lifecycle_error
        components = self._components
        return _GenerationSnapshot(
            components.version,
            self._storage_epoch,
            components.driver_key,
            components.rig_key,
            components.background_key,
        )

    def _remember_close_failure(
        self, resource: Any, label: str, error: BaseException
    ) -> None:
        if not any(resource is pending[0] for pending in self._deferred_closes):
            self._deferred_closes.append((resource, label, error))

    def _close_or_defer(self, resource: Any, label: str) -> None:
        error = _close_error(resource, label)
        if error is not None:
            self._remember_close_failure(resource, label, error)

    def _poison_if_close_failed(self) -> None:
        """Make a failed backend teardown terminal before anything can overlap."""

        if not self._deferred_closes or self._lifecycle_error is not None:
            return
        self._lifecycle_error = ReconfigurationUnavailable(
            "avatar resource teardown failed; the render lane is unavailable"
        )
        current = self._components
        self._components = _Components()
        for resource, label, error in current.close():
            self._remember_close_failure(resource, label, error)
        try:
            self._update_stats(last_error="resource_teardown_failed")
        except Exception:
            log.exception("cannot publish avatar teardown failure stats")

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReconfigurationUnavailable(
                "avatar activation deadline expired"
            )
        return remaining

    def _future_result(self, future: Future[_T], deadline: float) -> _T:
        try:
            return future.result(timeout=self._remaining(deadline))
        except FutureTimeout as exc:
            future.cancel()
            raise ReconfigurationUnavailable(
                "avatar render lane did not acknowledge the request"
            ) from exc

    def _prepare_driver_off_lane(
        self, candidate: AvatarConfig, deadline: float
    ) -> DriverPreparation:
        """Bound slow dependency/model preparation without blocking rendering."""

        future: Future[DriverPreparation] = Future()
        driver_cfg = candidate.driver.model_copy(deep=True)
        allow_download = self._allow_model_download
        slot = self._asset_slot
        threads = self._asset_threads
        threads_lock = self._asset_threads_lock
        worker: threading.Thread

        def prepare() -> None:
            try:
                result = prepare_driver(
                    driver_cfg,
                    allow_model_download=allow_download,
                )
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                slot.release()

        worker = threading.Thread(
            target=prepare,
            name="custback-avatar-assets",
            daemon=True,
        )
        # Concurrent writers wait within their common activation deadline. This
        # lets one of them commit and the other observe a deterministic CAS
        # conflict instead of failing merely because preparation overlapped.
        if not slot.acquire(timeout=self._remaining(deadline)):
            raise ReconfigurationUnavailable(
                "avatar asset preparation did not complete before the activation deadline"
            )

        # Register and start preparation atomically with respect to shutdown.
        # Completed threads stay in the set until the next snapshot so close()
        # cannot observe an empty set during the final instructions of a worker.
        with self._executor_lock:
            if self._closing or self._closed:
                slot.release()
                raise ReconfigurationUnavailable("avatar service is closing")
            with threads_lock:
                threads.add(worker)
            try:
                worker.start()
            except BaseException:
                with threads_lock:
                    threads.discard(worker)
                slot.release()
                raise
        try:
            return future.result(timeout=self._remaining(deadline))
        except FutureTimeout as exc:
            raise ReconfigurationUnavailable(
                "avatar asset preparation exceeded the activation deadline"
            ) from exc

    def _prepare_activation(
        self,
        candidate: AvatarConfig,
        expected_version: int,
        deadline: float,
    ) -> _PreparedActivation:
        """Pre-acquire fallible assets without touching live components."""

        snapshot = self._future_result(
            self._submit_lane(self._snapshot_generation), deadline
        )
        if snapshot.active and snapshot.version != expected_version:
            raise ConfigConflictError(expected_version, snapshot.version)

        rig_selector = resolve_rig_selector(
            candidate.appearance.rig, candidate.storage.rigs_dir
        )
        if (
            rig_selector != "builtin"
            and not is_rig_directory(rig_selector)
        ):
            raise FileNotFoundError("candidate rig is unavailable")
        active_background = (
            candidate.background.image_path
            if candidate.background.mode == "image"
            else candidate.background.video_path
            if candidate.background.mode == "video"
            else ""
        )
        if (
            active_background
            and not Path(active_background).expanduser().is_file()
        ):
            raise FileNotFoundError("candidate background is unavailable")
        if (
            candidate.driver.backend == "vision"
            and candidate.driver.vision.model_path
            and not Path(candidate.driver.vision.model_path).expanduser().is_file()
        ):
            raise FileNotFoundError("candidate vision model is unavailable")
        driver_key = self._driver_key(candidate)
        rig_key = self._rig_key(candidate, rig_selector)
        background_key = self._background_key(candidate)
        driver_preparation = None
        if (
            self._driver_factory is create_driver
            and (not snapshot.active or snapshot.driver_key != driver_key)
        ):
            driver_preparation = self._prepare_driver_off_lane(
                candidate, deadline
            )
        return _PreparedActivation(
            candidate.model_copy(deep=True),
            expected_version,
            snapshot,
            driver_key,
            rig_key,
            background_key,
            rig_selector,
            driver_preparation,
        )

    def _validate_candidate_assets(self, prepared: _PreparedActivation) -> None:
        # Filesystem probing and model acquisition happened before this lane.
        # A successful serialized deletion advances this pure in-memory token,
        # invalidating candidates prepared against the old storage view.
        if self._storage_epoch != prepared.snapshot.storage_epoch:
            raise FileNotFoundError("candidate assets changed during activation")

    def _stage_generation(
        self,
        current: _Components,
        prepared: _PreparedActivation,
        version: int,
    ) -> tuple[_Components, ExitStack]:
        """Construct and trial changed resources with deterministic rollback."""

        cfg = prepared.candidate
        staged = ExitStack()
        try:
            if current.driver is None or current.driver_key != prepared.driver_key:
                kwargs: dict[str, Any] = {
                    "allow_model_download": self._allow_model_download
                }
                if self._driver_factory is create_driver:
                    kwargs["preparation"] = prepared.driver_preparation
                try:
                    driver = self._driver_factory(cfg.driver, **kwargs)
                except DriverStartupError as exc:
                    staged.callback(
                        self._close_or_defer,
                        exc.driver,
                        "failed-start driver",
                    )
                    raise
                staged.callback(self._close_or_defer, driver, "staged driver")
            else:
                driver = current.driver

            if current.rig is None or current.rig_key != prepared.rig_key:
                rig = create_rig(
                    prepared.rig_selector,
                    avatar=cfg.appearance.avatar,
                    style=cfg.appearance.style,
                    rig_layer_max_pixels=cfg.storage.rig_layer_max_pixels,
                    rig_total_max_pixels=cfg.storage.rig_total_max_pixels,
                    rig_manifest_max_bytes=cfg.storage.rig_manifest_max_bytes,
                )
                staged.callback(self._close_or_defer, rig, "staged rig")
            else:
                rig = current.rig

            if (
                current.version < 0
                or current.background_key != prepared.background_key
            ):
                backdrop = create_avatar_backdrop(
                    cfg.background,
                    image_max_pixels=cfg.storage.image_max_pixels,
                    video_max_width=cfg.storage.video_max_width,
                    video_max_height=cfg.storage.video_max_height,
                )
                if backdrop is not None:
                    staged.callback(
                        self._close_or_defer, backdrop, "staged backdrop"
                    )
            else:
                backdrop = current.backdrop

            if (
                current.smoother is not None
                and current.driver is driver
                and current.smoother.factor == cfg.driver.smoothing
            ):
                smoother = current.smoother
            else:
                smoother = StateSmoother(cfg.driver.smoothing)

            generation = _Components(
                config=cfg.model_copy(deep=True),
                version=version,
                driver=driver,
                rig=rig,
                backdrop=backdrop,
                smoother=smoother,
                driver_key=prepared.driver_key,
                rig_key=prepared.rig_key,
                background_key=prepared.background_key,
            )
            self._trial_generation(current, generation)
            return generation, staged
        except BaseException:
            staged.close()
            raise

    def _trial_generation(
        self, current: _Components, candidate: _Components
    ) -> None:
        """Exercise staged identities without advancing reused live state."""

        cfg = candidate.config
        assert cfg is not None and candidate.driver is not None and candidate.rig is not None
        frame = np.full(_TRIAL_FRAME_SHAPE, 48, dtype=np.uint8)
        if candidate.driver is not current.driver:
            face = candidate.driver.update(frame.copy(), 0.0)
        else:
            face = FaceState.neutral(timestamp=0.0)

        if candidate.rig is not current.rig:
            sprite = candidate.rig.render(
                face,
                frozenset(cfg.appearance.parts),
                follow_pose=cfg.appearance.follow_pose,
            )
            framing = candidate.rig.framing_window(cfg.appearance.framing)
        else:
            sprite = np.zeros((64, 48, 4), dtype=np.uint8)
            sprite[4:-4, 4:-4] = (80, 140, 210, 220)
            framing = (0.0, 1.0)

        if candidate.backdrop is not current.backdrop:
            if candidate.backdrop is None:
                backdrop = blurred_room(frame.copy(), cfg.background.blur_strength)
            else:
                backdrop = candidate.backdrop.frame(frame.shape[1], frame.shape[0])
        else:
            backdrop = np.full_like(frame, cfg.background.color)

        rendered = compose_avatar(
            sprite,
            backdrop,
            cfg.appearance,
            framing_window=framing,
        )
        if rendered.dtype != np.uint8 or rendered.shape != frame.shape:
            raise ValueError("candidate produced an invalid trial frame")
        ok, _jpeg = cv2.imencode(
            ".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, cfg.render.jpeg_quality]
        )
        if not ok:
            raise ValueError("candidate trial JPEG encoding failed")

    def _handle_activation(self, request: _ActivationRequest) -> None:
        prepared = request.prepared
        with request.lock:
            if request.cancelled:
                request.done.set()
                return

        state = self.runtime.read()
        current = self._components
        if state.version != prepared.expected_version or (
            current.version not in (-1, prepared.expected_version)
        ):
            request.error = ConfigConflictError(
                prepared.expected_version, state.version
            )
            request.done.set()
            return

        ownership: ExitStack | None = None
        old: _Components | None = None
        generation: _Components | None = None
        driver_backend = ""
        driver_device = ""
        try:
            self._validate_candidate_assets(prepared)
            target_version = (
                prepared.expected_version + 1
                if request.publish
                else prepared.expected_version
            )
            generation, ownership = self._stage_generation(
                current, prepared, target_version
            )
            assert generation.driver is not None
            # Read custom backend metadata before the CAS. Nothing after a
            # successful pointer swap may be allowed to roll it back.
            driver_backend = generation.driver.name
            driver_device = generation.driver.device
            with request.lock:
                if request.cancelled:
                    ownership.close()
                    self._poison_if_close_failed()
                    request.done.set()
                    return
                if request.publish:
                    def activate(next_version: int) -> None:
                        nonlocal old
                        generation.version = next_version
                        old = self._components
                        self._components = generation

                    try:
                        committed = self._runtime_writer.commit_with_activation(
                            prepared.candidate,
                            prepared.expected_version,
                            activate,
                        )
                    except AvatarConfigVersionConflictError as exc:
                        raise ConfigConflictError(
                            exc.expected_version, exc.current_version
                        ) from exc
                else:
                    latest = self.runtime.read()
                    if latest.version != prepared.expected_version:
                        raise ConfigConflictError(
                            prepared.expected_version, latest.version
                        )
                    old = self._components
                    self._components = generation
                    committed = latest
                ownership.pop_all()
                ownership = None
                request.result = committed
                request.done.set()
        except BaseException as exc:
            if ownership is not None:
                ownership.close()
            error = exc if isinstance(
                exc, (ConfigConflictError, ReconfigurationUnavailable)
            ) else ActivationError(str(exc))
            with request.lock:
                if request.result is None:
                    request.error = error
                request.done.set()
            try:
                self._update_stats(last_error=f"activation: {type(exc).__name__}")
            except Exception:
                log.exception("cannot publish avatar activation failure stats")
            self._poison_if_close_failed()
            return

        try:
            self._update_stats(
                driver_backend=driver_backend,
                driver_device=driver_device,
                last_error="",
            )
        except Exception:
            log.exception("cannot publish avatar activation stats")
        # The acknowledgement is visible before any replaced backend is asked
        # to close. This lane cannot render with the old generation again.
        if old is not None and generation is not None:
            for resource, label, error in old.close_replaced_by(generation):
                self._remember_close_failure(resource, label, error)
            self._poison_if_close_failed()

    def _wait_activation(
        self, request: _ActivationRequest, deadline: float
    ) -> AvatarConfigState:
        try:
            remaining = self._remaining(deadline)
        except ReconfigurationUnavailable:
            if request.cancel():
                raise
            # The assignment-only commit won concurrently with the deadline;
            # return its authoritative result rather than reporting ambiguity.
            request.done.wait()
        else:
            if not request.done.wait(remaining):
                if request.cancel():
                    raise ReconfigurationUnavailable(
                        "avatar activation was not acknowledged before the deadline"
                    )
                request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise ReconfigurationUnavailable("avatar activation returned no result")
        return request.result

    def apply_config_patch(
        self,
        patch: dict[str, Any],
        timeout: float = _ACTIVATION_TIMEOUT_S,
    ) -> AvatarConfigState:
        """Validate, trial, CAS-activate, and acknowledge an avatar patch."""

        if self._on_lane():
            raise ReconfigurationUnavailable(
                "cannot synchronously patch from the avatar render lane"
            )
        deadline = time.monotonic() + timeout
        while True:
            base, candidate = self.runtime.prepare_patch(patch)
            if candidate is not None:
                break
            current = self.runtime.read()
            if current.version == base.version:
                return self.activate_initial(self._remaining(deadline))
        try:
            prepared = self._prepare_activation(candidate, base.version, deadline)
        except (ConfigConflictError, ReconfigurationUnavailable):
            raise
        except BaseException as exc:
            raise ActivationError(str(exc)) from exc
        request = _ActivationRequest(prepared)
        self._submit_lane(self._handle_activation, request)
        return self._wait_activation(request, deadline)

    def activate_initial(
        self, timeout: float = _ACTIVATION_TIMEOUT_S
    ) -> AvatarConfigState:
        """Activate the startup snapshot before frames or the API use it."""

        deadline = time.monotonic() + timeout
        while True:
            state = self.runtime.read()
            snapshot = self._future_result(
                self._submit_lane(self._snapshot_generation), deadline
            )
            if snapshot.version == state.version:
                return state
            try:
                prepared = self._prepare_activation(
                    state.config, state.version, deadline
                )
                request = _ActivationRequest(prepared, publish=False)
                self._submit_lane(self._handle_activation, request)
                return self._wait_activation(request, deadline)
            except ConfigConflictError:
                self._remaining(deadline)
            except (ActivationError, ReconfigurationUnavailable):
                raise
            except BaseException as exc:
                raise ActivationError(str(exc)) from exc

    def _handle_mutation(self, request: _MutationRequest) -> None:
        # Hold the request lock across an irreversible filesystem mutation. A
        # timeout can cancel a queued request, but never report an ambiguous
        # result for a deletion that has already begun.
        with request.lock:
            if request.cancelled:
                request.done.set()
                return
            try:
                if self._lifecycle_error is not None:
                    raise self._lifecycle_error
                state = self.runtime.read()
                components = self._components
                if components.version not in (-1, state.version):
                    raise ConfigConflictError(components.version, state.version)
                effective = components.config or state.config
                request.mutate(effective.model_copy(deep=True))
                self._storage_epoch += 1
                request.result = state
            except BaseException as exc:
                request.error = exc
            request.done.set()

    def apply_storage_mutation(
        self,
        mutate: Callable[[AvatarConfig], None],
        timeout: float = _ACTIVATION_TIMEOUT_S,
    ) -> AvatarConfigState:
        """Serialize active rig/background deletion with generation swaps."""

        if self._on_lane():
            raise ReconfigurationUnavailable(
                "cannot synchronously mutate storage from the avatar render lane"
            )
        request = _MutationRequest(mutate)
        self._submit_lane(self._handle_mutation, request)
        deadline = time.monotonic() + timeout
        try:
            remaining = self._remaining(deadline)
        except ReconfigurationUnavailable:
            if request.cancel():
                raise
            request.done.wait()
        else:
            if not request.done.wait(remaining):
                if request.cancel():
                    raise ReconfigurationUnavailable(
                        "avatar storage mutation was not acknowledged before the deadline"
                    )
                request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise ReconfigurationUnavailable("avatar storage mutation returned no result")
        return request.result

    # -- rendering (the same lane; one call in flight at a time) -----------

    def _publish_render(self, publication: _RenderPublication) -> None:
        self.output.put(publication.frame)
        self._count("frames_rendered")
        self._update_stats(
            driver_ms=publication.driver_ms,
            render_ms=publication.render_ms,
            face_present=publication.face_present,
            output_width=publication.width,
            output_height=publication.height,
        )

    def _publish_render_if_current(
        self, publication: _RenderPublication
    ) -> bool:
        if (
            self._lifecycle_error is not None
            or self._components.version != publication.version
        ):
            return False
        self._publish_render(publication)
        return True

    def _process_deferred(
        self, data: bytes
    ) -> tuple[bytes | None, _RenderPublication | None]:
        """Render on the lane while deferring externally visible publication."""

        if not self._on_lane():  # pragma: no cover - private lane contract
            raise RuntimeError("deferred avatar processing must run on its lane")
        self._defer_render_publication = True
        self._pending_render_publication = None
        try:
            payload = self._process(data)
            return payload, self._pending_render_publication
        finally:
            self._pending_render_publication = None
            self._defer_render_publication = False

    def _process(self, data: bytes) -> bytes | None:
        if not self._on_lane():
            self.activate_initial()
            return self._future_result(
                self._submit_lane(self._process, data),
                time.monotonic() + _ACTIVATION_TIMEOUT_S,
            )
        components = self._components
        if self._lifecycle_error is not None:
            raise self._lifecycle_error
        cfg = components.config
        if cfg is None or components.driver is None or components.rig is None:
            raise ReconfigurationUnavailable(
                "avatar generation is not activated"
            )
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
        sprite = components.rig.render(
            face,
            frozenset(cfg.appearance.parts),
            follow_pose=cfg.appearance.follow_pose,
        )
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
        publication = _RenderPublication(
            rendered,
            version=components.version,
            driver_ms=driver_ms,
            render_ms=render_ms,
            face_present=face.present,
            width=width,
            height=height,
        )
        if self._defer_render_publication:
            self._pending_render_publication = publication
        else:
            self._publish_render(publication)
        return jpeg.tobytes()

    # -- session -----------------------------------------------------------
    async def _await_lane_future(self, future: Future[_T]) -> _T:
        """Preserve lane ownership across cancellation and discard its result."""

        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError as cancellation:
            task = asyncio.current_task()
            uncancel = getattr(task, "uncancel", None)
            if callable(uncancel):
                uncancel()
            # Poll the authoritative concurrent Future directly. Recreating an
            # asyncio shield after cancellation can itself remain cancelled,
            # and repeated Task.cancel() calls must not release lane ownership.
            while not future.done():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    if callable(uncancel):
                        uncancel()
            try:
                future.result()
            except BaseException:
                pass
            raise cancellation

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
                payload, publication = await self._await_lane_future(
                    self._submit_lane(self._process_deferred, data)
                )
                if stop.is_set():
                    return
                if payload is not None:
                    await ws.send(payload)
                    if publication is not None:
                        await self._await_lane_future(
                            self._submit_lane(
                                self._publish_render_if_current, publication
                            )
                        )
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
        try:
            await asyncio.to_thread(self.activate_initial)
        except BaseException:
            await self.aclose()
            raise
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
                except ReconfigurationUnavailable:
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
            await self.aclose()

    def _close_components_on_lane(self) -> None:
        components = self._components
        self._components = _Components()
        prior = self._deferred_closes
        self._deferred_closes = []
        remaining = components.close()
        for resource, label, _previous_error in prior:
            error = _close_error(resource, f"deferred {label}")
            if error is not None:
                remaining.append((resource, label, error))
        for resource, label, error in remaining:
            self._remember_close_failure(resource, label, error)
        if self._deferred_closes:
            raise ReconfigurationUnavailable(
                f"{len(self._deferred_closes)} avatar resource(s) survived teardown"
            )

    def _begin_close(self) -> Future[None]:
        with self._executor_lock:
            if self._close_future is not None:
                if not self._close_future.done():
                    return self._close_future
                if self._close_future.exception() is None:
                    return self._close_future
                # A backend reported a surviving worker. Keep ownership and
                # allow an explicit retry to join it after its interrupt lands.
                self._close_future = None
            if self._closed:
                completed: Future[None] = Future()
                completed.set_result(None)
                return completed
            self._closing = True
            self._close_future = cast(
                Future[None],
                self._executor.submit(
                    self._lane_entry, self._close_components_on_lane
                ),
            )
            return self._close_future

    def _watch_close_after_timeout(self, future: Future[None]) -> None:
        def finish_when_terminal() -> None:
            try:
                future.result()
            except BaseException:
                return
            self._join_asset_threads()
            self._finish_close()

        watcher = threading.Thread(
            target=finish_when_terminal,
            name="custback-avatar-shutdown",
            daemon=True,
        )
        try:
            watcher.start()
        except BaseException:
            log.exception("cannot start avatar shutdown watcher")

            def stop_executor_when_terminal(done: Future[None]) -> None:
                if done.cancelled() or done.exception() is not None:
                    return
                # A callback runs on the render worker itself, so it cannot
                # join that executor. It can still restore the allocation-
                # failure fallback when no separately owned asset worker lives.
                if self._live_asset_threads():
                    return
                with self._executor_lock:
                    if not self._closed:
                        self._executor.shutdown(
                            wait=False, cancel_futures=False
                        )
                        self._closed = True

            future.add_done_callback(stop_executor_when_terminal)

    def _live_asset_threads(self) -> tuple[threading.Thread, ...]:
        with self._asset_threads_lock:
            live = tuple(
                worker for worker in self._asset_threads if worker.is_alive()
            )
            self._asset_threads.intersection_update(live)
            return live

    def _join_asset_threads(self, deadline: float | None = None) -> bool:
        """Join every registered asset worker, optionally within a deadline."""

        while workers := self._live_asset_threads():
            for worker in workers:
                if deadline is None:
                    worker.join()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                worker.join(remaining)
        return True

    async def _await_close_condition(
        self,
        pending: Callable[[], bool],
        deadline: float,
    ) -> tuple[bool, asyncio.CancelledError | None]:
        """Poll owned teardown state within a deadline, retaining cancellation."""

        cancellation: asyncio.CancelledError | None = None
        task = asyncio.current_task()
        uncancel = getattr(task, "uncancel", None)
        while pending():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, cancellation
            try:
                await asyncio.sleep(min(0.01, remaining))
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                if callable(uncancel):
                    uncancel()
        return True, cancellation

    def _finish_close(self) -> None:
        with self._executor_lock:
            if self._closed:
                return
            if self._live_asset_threads():
                raise ReconfigurationUnavailable(
                    "avatar asset preparation worker survived teardown"
                )
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._closed = True

    def close(self, timeout: float = _ACTIVATION_TIMEOUT_S) -> None:
        """Synchronously drain the lane, close its generation, and join it."""

        if self._on_lane():
            raise ReconfigurationUnavailable(
                "cannot close the avatar service from its render lane"
            )
        deadline = time.monotonic() + max(timeout, 0.0)
        future = self._begin_close()
        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeout as exc:
            self._watch_close_after_timeout(future)
            raise ReconfigurationUnavailable(
                f"avatar render lane did not stop within {timeout:.1f}s"
            ) from exc
        if not self._join_asset_threads(deadline):
            self._watch_close_after_timeout(future)
            raise ReconfigurationUnavailable(
                f"avatar asset preparation worker did not stop within {timeout:.1f}s"
            )
        self._finish_close()

    async def aclose(self, timeout: float = _ACTIVATION_TIMEOUT_S) -> None:
        """Bounded async close; a watcher retains ownership after a timeout."""

        deadline = time.monotonic() + max(timeout, 0.0)
        future = self._begin_close()
        cancellation: asyncio.CancelledError | None = None
        lane_error: BaseException | None = None
        lane_terminal, cancellation = await self._await_close_condition(
            lambda: not future.done(), deadline
        )
        if not lane_terminal:
            self._watch_close_after_timeout(future)
            raise ReconfigurationUnavailable(
                f"avatar render lane did not stop within {timeout:.1f}s"
            )
        if future.done():
            if future.cancelled():
                lane_error = ReconfigurationUnavailable(
                    "avatar render lane shutdown was cancelled"
                )
            else:
                lane_error = future.exception()
        assets_terminal, asset_cancellation = await self._await_close_condition(
            lambda: bool(self._live_asset_threads()), deadline
        )
        if cancellation is None:
            cancellation = asset_cancellation
        if not assets_terminal:
            if lane_error is None:
                self._watch_close_after_timeout(future)
            raise ReconfigurationUnavailable(
                f"avatar asset preparation worker did not stop within {timeout:.1f}s"
            )
        if lane_error is not None:
            raise lane_error
        self._finish_close()
        if cancellation is not None:
            raise cancellation

    def __del__(self) -> None:  # pragma: no cover - defensive fixture cleanup
        try:
            future = self._begin_close()
            self._executor.shutdown(wait=False, cancel_futures=False)
            # Keep the local reference until shutdown has queued its sentinel.
            _ = future
        except Exception:
            pass
