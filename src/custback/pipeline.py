"""Main processing loop and transactional live reconfiguration."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeout,
)
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import cv2

from .backgrounds import BlurBackdrop, create_backdrop
from .capture import open_capture
from .compositor import composite
from .config import (
    AVATAR_PROXY_RESTART_ONLY_FIELDS,
    AppConfig,
    ConfigState,
    ConfigVersionConflictError,
    RuntimeConfig,
)
from .diagnostics import sanitized_config_summary
from .hub import FrameHub
from .segmentation import (
    HeuristicSegmenter,
    NullSegmenter,
    SegmenterPreparation,
    create_segmenter,
    refiner_for,
)
from .vcam import open_output

log = logging.getLogger(__name__)

_VIDEO_STATS_DEFAULTS: dict[str, object] = {
    "background_video_source_fps": None,
    "background_video_timing_mode": None,
    "background_video_frames_displayed": 0,
    "background_video_frames_skipped": 0,
    "background_video_frames_reused": 0,
    "background_video_skip_ratio": 0.0,
    "background_video_seek_count": 0,
    "background_video_decode_failures": 0,
}

# Retain compact downsampled fingerprints for the complete pipeline session.
# The bound is a fail-closed admission limit, never an eviction policy: once it
# is reached, the renderer session is invalidated and output remains the slate.
_RAW_FINGERPRINT_HISTORY = 1_000_000
_RAW_FINGERPRINT_SIZE = (16, 12)
_RAW_ECHO_PIXEL_TOLERANCE = 3
_RAW_ECHO_CHANGED_FRACTION = 0.02
_RAW_ECHO_MEAN_DELTA = 4.0
_RAW_ECHO_LOWRES_MEAN_DELTA = 20.0
_RAW_ECHO_LOWRES_CORRELATION = 0.93


def _backdrop_key(cfg: AppConfig) -> tuple[object, ...]:
    """Only inputs consumed by the currently selected backdrop resource."""

    background = cfg.background
    mode = (
        background.remote_fallback_mode
        if background.mode == "remote"
        else background.mode
    )
    if mode == "passthrough":
        return ("passthrough",)
    if mode == "blur":
        return ("blur", background.blur_strength)
    if mode == "color":
        return ("color", background.color)
    if mode == "image":
        return ("image", background.image_path, cfg.api.uploads.image_max_pixels)
    if mode == "video":
        return (
            "video",
            background.video_path,
            cfg.api.uploads.video_max_width,
            cfg.api.uploads.video_max_height,
        )
    if mode == "camera":
        target = cfg.resolved_backdrop_target()
        if target is None:
            raise ValueError("camera backdrop target is not configured")
        return ("camera", target.identifier, target.source)
    raise ValueError(f"unknown background mode: {mode!r}")


def _build_backdrop(cfg: AppConfig) -> Any:
    kwargs: dict[str, Any] = {
        "image_max_pixels": cfg.api.uploads.image_max_pixels,
        "video_max_width": cfg.api.uploads.video_max_width,
        "video_max_height": cfg.api.uploads.video_max_height,
    }
    if _backdrop_key(cfg)[0] == "camera":
        kwargs["camera_target"] = cfg.resolved_backdrop_target()
    return create_backdrop(cfg.background, **kwargs)


def _ewma(previous: float | None, sample: float, alpha: float = 0.1) -> float:
    return sample if previous is None else previous + alpha * (sample - previous)


class RestartRequiredError(RuntimeError):
    """The patch is valid but changes resources that cannot be swapped live."""

    status_code = 409

    def __init__(self, fields: list[str] | tuple[str, ...], current_version: int):
        self.fields = tuple(sorted(fields))
        self.current_version = current_version
        super().__init__(
            "restart required for configuration fields: " + ", ".join(self.fields)
        )


class ActivationError(RuntimeError):
    """A validated hot configuration could not activate its resources."""

    status_code = 422


class ReconfigurationUnavailable(RuntimeError):
    """The pipeline cannot currently acknowledge a reconfiguration request."""

    status_code = 503


class ConfigConflictError(RuntimeError):
    """The patch was based on a configuration version that is no longer current."""

    status_code = 409

    def __init__(self, expected_version: int, current_version: int):
        self.expected_version = expected_version
        self.current_version = current_version
        super().__init__(
            f"configuration changed concurrently: expected version {expected_version}, "
            f"current version is {current_version}"
        )


class _PrivacyViolation(ValueError):
    """A frame or mask violated the remote-output privacy boundary."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class _RawFingerprint:
    jpeg_features: tuple[int, ...]
    thumbnail: np.ndarray


class _RawReplayHistory:
    """Compact, bounded Bloom-style index with no permissive eviction."""

    _MAX_BYTES = 16 * 1024 * 1024
    _MIN_BYTES = 4 * 1024
    _JPEG_QUANTIZATION_WIDTH = 64
    _JPEG_QUANTIZATION_STEP = 4
    _MATCHING_SIGNATURES = 8

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._byte_count = min(
            self._MAX_BYTES,
            max(self._MIN_BYTES, capacity * 64),
        )
        self._bits: bytearray | None = None
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def clear(self) -> None:
        self._bits = None
        self._count = 0

    @staticmethod
    def _keys(fingerprint: _RawFingerprint):
        # JPEG preserves low-frequency block/DC values even when edge-based
        # perceptual hashes flip near equal comparisons.  Hash the complete
        # coarse feature vector under sixteen overlapping quantizers.  A small
        # JPEG drift crosses only a subset of their staggered boundaries, so
        # multiple whole-frame keys remain identical without retaining a
        # reversible historical thumbnail.
        width = _RawReplayHistory._JPEG_QUANTIZATION_WIDTH
        step = _RawReplayHistory._JPEG_QUANTIZATION_STEP
        for phase_index, offset in enumerate(range(0, width, step)):
            key = phase_index
            for value in fingerprint.jpeg_features:
                quantized = min(3, (value + offset) // width)
                key = (key << 2) | quantized
            yield key

    @staticmethod
    def _positions(key: int, bit_count: int):
        mask = (1 << 64) - 1
        for salt in (
            0x9E3779B97F4A7C15,
            0xD1B54A32D192ED03,
            0x94D049BB133111EB,
        ):
            value = (key + salt) & mask
            value ^= value >> 30
            value = (value * 0xBF58476D1CE4E5B9) & mask
            value ^= value >> 27
            value = (value * 0x94D049BB133111EB) & mask
            value ^= value >> 31
            yield value % bit_count

    def append(self, fingerprint: _RawFingerprint) -> None:
        if self._bits is None:
            self._bits = bytearray(self._byte_count)
        bit_count = len(self._bits) * 8
        for key in self._keys(fingerprint):
            for position in self._positions(key, bit_count):
                self._bits[position >> 3] |= 1 << (position & 7)
        self._count += 1

    def matches(self, fingerprint: _RawFingerprint) -> bool:
        if self._bits is None:
            return False
        bit_count = len(self._bits) * 8
        matches = 0
        for key in self._keys(fingerprint):
            if all(
                self._bits[position >> 3] & (1 << (position & 7))
                for position in self._positions(key, bit_count)
            ):
                matches += 1
                if matches >= self._MATCHING_SIGNATURES:
                    return True
        return False


def _safe_close(resource: Any, label: str) -> None:
    if resource is None:
        return
    try:
        resource.close()
    except Exception:
        log.exception("cannot close %s", label)


@dataclass
class _Resources:
    cfg: AppConfig
    version: int
    capture: Any
    segmenter: Any
    refiner: Any
    backdrop: Any
    output: Any

    def close(self) -> None:
        # Close independently so one faulty backend cannot strand the others.
        _safe_close(self.output, "video output")
        _safe_close(self.backdrop, "backdrop")
        _safe_close(self.segmenter, "segmenter")
        _safe_close(self.capture, "capture")


@dataclass
class _Activation:
    candidate: AppConfig
    replace_segmenter: bool = False
    segmenter: Any = None
    refiner: Any = None
    replace_backdrop: bool = False
    backdrop: Any = None

    def discard(self) -> None:
        if self.replace_backdrop:
            _safe_close(self.backdrop, "staged backdrop")
        if self.replace_segmenter:
            _safe_close(self.segmenter, "staged segmenter")


@dataclass
class _PatchRequest:
    candidate: AppConfig
    expected_version: int
    origin: str = "internal"
    activation_candidate: AppConfig | None = None
    before_activate: Callable[[], None] | None = None
    rollback_activate: Callable[[], None] | None = None
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: bool = False
    result: ConfigState | None = None
    error: BaseException | None = None
    storage_epoch: int = 0
    prepared_activation: _Activation | None = None

    def cancel_and_take_activation(self) -> tuple[bool, _Activation | None]:
        """Cancel and reclaim an activation not yet claimed by the frame lane."""

        with self.lock:
            if self.result is not None or self.error is not None:
                return False, None
            self.cancelled = True
            activation = self.prepared_activation
            self.prepared_activation = None
            return True, activation

    def fail(self, error: BaseException) -> None:
        with self.lock:
            if self.result is None and self.error is None:
                self.error = error
        self.done.set()


@dataclass
class _MutationRequest:
    """Serialized side effect tied to the effective config generation."""

    mutate: Callable[[AppConfig], None]
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: bool = False
    result: ConfigState | None = None
    error: BaseException | None = None

    def cancel(self) -> bool:
        with self.lock:
            if self.result is not None or self.error is not None:
                return False
            self.cancelled = True
            return True

    def fail(self, error: BaseException) -> None:
        with self.lock:
            if self.result is None and self.error is None:
                self.error = error
        self.done.set()


def _changed_paths(old: Any, new: Any, prefix: str = "") -> list[str]:
    if hasattr(old, "model_dump"):
        old = old.model_dump(mode="python")
    if hasattr(new, "model_dump"):
        new = new.model_dump(mode="python")
    if isinstance(old, dict) and isinstance(new, dict):
        changed: list[str] = []
        for key in sorted(old.keys() | new.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in old or key not in new:
                changed.append(path)
            else:
                changed.extend(_changed_paths(old[key], new[key], path))
        return changed
    return [] if old == new else [prefix]


def _restart_only_changes(old: AppConfig, new: AppConfig) -> list[str]:
    changed = _changed_paths(old, new)
    return [
        path
        for path in changed
        if path.startswith("camera.")
        or path.startswith("output.")
        or (path.startswith("api.") and path != "api.remote_timeout_ms")
        or path == "background.camera_device"
        or path.startswith("backdrop_targets.")
        or path in AVATAR_PROXY_RESTART_ONLY_FIELDS
    ]


class Pipeline:
    def __init__(
        self,
        runtime: RuntimeConfig,
        hub: FrameHub,
        *,
        model_preparation: SegmenterPreparation | None = None,
        raw_fingerprint_capacity: int = _RAW_FINGERPRINT_HISTORY,
    ):
        if (
            not isinstance(raw_fingerprint_capacity, int)
            or isinstance(raw_fingerprint_capacity, bool)
            or raw_fingerprint_capacity < 1
        ):
            raise ValueError("raw fingerprint capacity must be a positive integer")
        self.runtime = runtime
        self.hub = hub
        self._stop = threading.Event()
        self._startup_done = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._requests: queue.Queue[_PatchRequest | _MutationRequest] = queue.Queue()
        self._request_enqueue_lock = threading.Lock()
        # A preparation stays registered until its result has either moved to
        # request ownership or all abandoned-result cleanup has finished.  The
        # condition lets stop() enforce its deadline without calling an
        # unbounded executor join while a backend's close() is still running in
        # a Future callback.  RLock is intentional: cancelling queued futures
        # during executor.shutdown() can invoke their callbacks synchronously.
        self._preparation_lock = threading.RLock()
        self._preparation_condition = threading.Condition(self._preparation_lock)
        self._preparation_executor: ThreadPoolExecutor | None = None
        self._preparation_futures: set[Future[_Activation]] = set()
        self._storage_epoch_lock = threading.Lock()
        self._storage_epoch = 0
        self._lifecycle_lock = threading.Lock()
        self._active_state: ConfigState | None = None
        self._teardown_lock = threading.Lock()
        self._teardown_threads: set[threading.Thread] = set()
        self._deferred_closes: list[tuple[Any, str]] = []
        self._runtime_writer = runtime._coordinator_writer()
        self._model_preparation = model_preparation
        self._fallback_log_states: dict[str, tuple[bool, str]] = {}
        self._raw_fingerprint_capacity = raw_fingerprint_capacity
        self._recent_raw_fingerprints = _RawReplayHistory(raw_fingerprint_capacity)
        self._privacy_history_exhausted = False
        self._privacy_invalidated_session: int | None = None
        self._latest_raw_frame: np.ndarray | None = None

    def start(self, timeout: float | None = None) -> None:
        """Start and synchronously acknowledge resource activation."""
        effective_timeout = (
            max(30.0, self.runtime.snapshot().camera.recovery_timeout_s + 5.0)
            if timeout is None
            else timeout
        )
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                raise ReconfigurationUnavailable("pipeline is already running")
            self._stop.clear()
            self._startup_done.clear()
            self._error = None
            self._active_state = None
            self._fallback_log_states.clear()
            with self._preparation_lock:
                if self._preparation_executor is not None:
                    raise ReconfigurationUnavailable(
                        "pipeline candidate preparation is still shutting down"
                    )
                self._preparation_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="custback-segmentation-prepare",
                )
            self._thread = threading.Thread(
                target=self._run, name="pipeline", daemon=True
            )
            self._thread.start()
        if not self._startup_done.wait(effective_timeout):
            self._stop.set()
            thread = self._thread
            if thread is not None:
                thread.join(timeout=max(0.1, min(5.0, effective_timeout)))
            if thread is not None and thread.is_alive():
                raise ReconfigurationUnavailable(
                    "pipeline startup timed out; worker is still running after "
                    "shutdown was requested"
                )
            if self._error is not None:
                raise ActivationError(str(self._error)) from self._error
            raise ReconfigurationUnavailable(
                "pipeline startup timed out; worker stopped during shutdown"
            )
        if self._error is not None:
            thread = self._thread
            if thread is not None:
                thread.join(timeout=1.0)
                if thread.is_alive():
                    raise ReconfigurationUnavailable(
                        "pipeline startup failed and its worker survived teardown"
                    ) from self._error
            raise ActivationError(str(self._error)) from self._error

    def stop(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        self._stop.set()
        with self._request_enqueue_lock:
            self._fail_pending(ReconfigurationUnavailable("pipeline is stopping"))
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
        frame_survived = thread is not None and thread.is_alive()
        with self._teardown_lock:
            teardown_threads = tuple(self._teardown_threads)
        for teardown in teardown_threads:
            teardown.join(max(0.0, deadline - time.monotonic()))
        with self._teardown_lock:
            survivors = [
                worker for worker in self._teardown_threads if worker.is_alive()
            ]
        preparation_error: ReconfigurationUnavailable | None = None
        try:
            # Always initiate executor shutdown, even if another worker has
            # survived its deadline. A later stop() can finish joining any
            # model constructor that could not be interrupted on this pass.
            self._shutdown_preparation_executor(deadline, timeout)
        except ReconfigurationUnavailable as exc:
            preparation_error = exc
        if frame_survived:
            raise ReconfigurationUnavailable(
                f"pipeline worker did not stop within {timeout:.1f}s"
            )
        if survivors:
            raise ReconfigurationUnavailable(
                f"{len(survivors)} resource teardown worker(s) did not stop "
                f"within {timeout:.1f}s"
            )
        if preparation_error is not None:
            raise preparation_error
        if self._error is not None:
            raise self._error

    @property
    def running(self) -> bool:
        return (
            self._error is None and self._thread is not None and self._thread.is_alive()
        )

    def apply_config_patch(
        self,
        patch: dict[str, Any],
        timeout: float = 5.0,
        *,
        origin: str = "internal",
    ) -> ConfigState:
        """Validate, activate, commit, and acknowledge a hot configuration patch."""
        if isinstance(patch, dict):
            restart_fields = []
            background_patch = patch.get("background")
            if (
                isinstance(background_patch, dict)
                and "camera_device" in background_patch
            ):
                restart_fields.append("background.camera_device")
            if "backdrop_targets" in patch:
                restart_fields.append("backdrop_targets")
            if restart_fields:
                raise RestartRequiredError(restart_fields, self.runtime.version)
        while True:
            base = self.runtime.read()
            candidate = base.config.patched(patch)
            if candidate != base.config:
                break
            # Establish a linearization point for a no-op response. If a
            # commit won between validation and this read, reapply the patch
            # to that generation instead of returning a stale header/body.
            current = self.runtime.read()
            if current.version == base.version:
                log.info(
                    "config update accepted origin=%s version=%d fields=none "
                    "summary=none (no change)",
                    origin,
                    current.version,
                )
                return current
        restart_fields = _restart_only_changes(base.config, candidate)
        if restart_fields:
            raise RestartRequiredError(restart_fields, base.version)
        if not self.running or not self._startup_done.is_set():
            raise ReconfigurationUnavailable("pipeline is not running")
        if threading.current_thread() is self._thread:
            raise ReconfigurationUnavailable(
                "cannot synchronously reconfigure from the pipeline worker"
            )

        deadline = time.monotonic() + timeout
        request = _PatchRequest(
            candidate,
            base.version,
            origin=origin,
            storage_epoch=self._read_storage_epoch(),
        )
        self._prepare_patch_request(request, base.config, deadline)
        return self._submit_prepared_patch(request, deadline)

    def apply_staged_config_patch(
        self,
        patch: dict[str, Any],
        staging_patch: dict[str, Any],
        before_activate: Callable[[], None],
        rollback_activate: Callable[[], None],
        timeout: float = 5.0,
    ) -> ConfigState:
        """Preflight hidden resources, then promote and commit at one boundary.

        ``patch`` describes the final effective configuration while
        ``staging_patch`` points constructors at a hidden candidate asset.
        The promotion callback runs only after construction and trial succeed,
        inside the same compare-and-swap activation that publishes config.
        """
        base = self.runtime.read()
        candidate = base.config.patched(patch)
        staging_candidate = base.config.patched(staging_patch)
        final_paths = set(_changed_paths(base.config, candidate))
        staging_paths = set(_changed_paths(base.config, staging_candidate))
        if not final_paths or final_paths != staging_paths:
            raise ValueError(
                "staged and final patches must change the same configuration fields"
            )
        staged_differences = set(_changed_paths(candidate, staging_candidate))
        allowed_asset_path = {
            "image": "background.image_path",
            "video": "background.video_path",
        }.get(candidate.background.mode)
        if candidate.background.mode == "remote":
            allowed_asset_path = {
                "image": "background.image_path",
                "video": "background.video_path",
            }.get(candidate.background.remote_fallback_mode)
        if allowed_asset_path is None or staged_differences != {allowed_asset_path}:
            raise ValueError(
                "staged and final configurations may differ only in the active "
                "image/video asset path"
            )
        restart_fields = _restart_only_changes(base.config, candidate)
        if restart_fields:
            raise RestartRequiredError(restart_fields, base.version)
        if not self.running or not self._startup_done.is_set():
            raise ReconfigurationUnavailable("pipeline is not running")
        if threading.current_thread() is self._thread:
            raise ReconfigurationUnavailable(
                "cannot synchronously reconfigure from the pipeline worker"
            )
        request = _PatchRequest(
            candidate,
            base.version,
            origin="api-upload",
            activation_candidate=staging_candidate,
            before_activate=before_activate,
            rollback_activate=rollback_activate,
            storage_epoch=self._read_storage_epoch(),
        )
        deadline = time.monotonic() + timeout
        self._prepare_patch_request(request, base.config, deadline)
        return self._submit_prepared_patch(request, deadline)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise ReconfigurationUnavailable(
                "pipeline candidate preparation exceeded the reconfiguration deadline"
            )
        return remaining

    def _read_storage_epoch(self) -> int:
        with self._storage_epoch_lock:
            return self._storage_epoch

    def _track_preparation_future(self, future: Future[_Activation]) -> None:
        with self._preparation_condition:
            self._preparation_futures.add(future)

    def _untrack_preparation_future(self, future: Future[_Activation]) -> None:
        with self._preparation_condition:
            self._preparation_futures.discard(future)
            self._preparation_condition.notify_all()

    def _discard_preparation_result(self, future: Future[_Activation]) -> None:
        """Close an abandoned result before releasing its lifecycle token."""

        try:
            if future.cancelled():
                return
            activation = future.result()
        except BaseException:
            return
        else:
            activation.discard()
        finally:
            # Future.result() becomes observable before its callbacks finish.
            # Removing this token only here prevents stop() from mistaking a
            # blocked candidate close for a terminal preparation worker.
            self._untrack_preparation_future(future)

    def _abandon_preparation_future(self, future: Future[_Activation]) -> None:
        """Transfer a not-yet-returned future to deterministic cleanup."""

        if future.cancel():
            self._untrack_preparation_future(future)
            return
        # The constructor is already running or completed concurrently.
        # add_done_callback also runs immediately for a completed future.
        future.add_done_callback(self._discard_preparation_result)

    def _prepare_patch_request(
        self,
        request: _PatchRequest,
        current: AppConfig,
        deadline: float,
    ) -> None:
        """Build fallible candidate resources on the dedicated executor."""

        candidate = request.activation_candidate or request.candidate
        with self._preparation_lock:
            executor = self._preparation_executor
            if executor is None:
                raise ReconfigurationUnavailable(
                    "pipeline candidate preparation is unavailable"
                )
            try:
                future = executor.submit(
                    self._prepare_activation_off_lane,
                    current.model_copy(deep=True),
                    candidate.model_copy(deep=True),
                )
                # Register under the same lock as submit so shutdown cannot
                # snapshot the executor in the handoff gap between the two.
                self._track_preparation_future(future)
            except RuntimeError as exc:
                raise ReconfigurationUnavailable(
                    "pipeline candidate preparation is shutting down"
                ) from exc
        cleanup_deferred = False
        try:
            try:
                remaining = self._remaining(deadline)
            except BaseException:
                cleanup_deferred = True
                self._abandon_preparation_future(future)
                raise
            try:
                activation = future.result(timeout=remaining)
            except FutureTimeout as exc:
                cleanup_deferred = True
                self._abandon_preparation_future(future)
                raise ReconfigurationUnavailable(
                    "pipeline candidate preparation exceeded the reconfiguration deadline"
                ) from exc
            except ActivationError:
                raise
            except BaseException as exc:
                raise ActivationError(str(exc)) from exc
            if not self.running or not self._startup_done.is_set():
                activation.discard()
                raise ReconfigurationUnavailable("pipeline stopped during preparation")
            request.prepared_activation = activation
        finally:
            if not cleanup_deferred:
                self._untrack_preparation_future(future)

    def _submit_prepared_patch(
        self, request: _PatchRequest, deadline: float
    ) -> ConfigState:
        """Transfer a prepared activation to the frame queue or discard it."""

        try:
            remaining = self._remaining(deadline)
        except BaseException:
            activation = request.prepared_activation
            request.prepared_activation = None
            self._schedule_discard_activation(activation)
            raise
        return self._submit_patch(request, remaining)

    def _shutdown_preparation_executor(self, deadline: float, timeout: float) -> None:
        with self._preparation_condition:
            executor = self._preparation_executor
            if executor is None:
                return
            executor.shutdown(wait=False, cancel_futures=True)
            while self._preparation_futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise ReconfigurationUnavailable(
                        "pipeline candidate preparation worker did not stop "
                        f"within {timeout:.1f}s"
                    )
                self._preparation_condition.wait(remaining)
        # All running callbacks have reached terminal state, so this wait is
        # now bounded to executor bookkeeping and worker exit.
        executor.shutdown(wait=True, cancel_futures=True)
        with self._preparation_condition:
            if self._preparation_executor is executor:
                self._preparation_executor = None

    def _submit_patch(self, request: _PatchRequest, timeout: float) -> ConfigState:
        try:
            self._enqueue_request(request)
        except BaseException:
            _cancelled, activation = request.cancel_and_take_activation()
            # The request never entered the owned frame queue, so the caller
            # remains the sole owner and closes it before returning.
            if activation is not None:
                activation.discard()
            raise
        if not request.done.wait(timeout):
            cancelled, activation = request.cancel_and_take_activation()
            if cancelled:
                self._schedule_discard_activation(activation)
                raise ReconfigurationUnavailable(
                    f"pipeline did not acknowledge reconfiguration within {timeout:.1f}s"
                )
            # Completion won the lock concurrently; wait for its event publish.
            request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:  # defensive invariant
            raise ReconfigurationUnavailable(
                "pipeline returned no configuration result"
            )
        return request.result

    def _enqueue_request(self, request: _PatchRequest | _MutationRequest) -> None:
        """Atomically enqueue only while the frame worker still accepts work."""

        with self._request_enqueue_lock:
            if (
                self._stop.is_set()
                or not self.running
                or not self._startup_done.is_set()
            ):
                raise ReconfigurationUnavailable("pipeline is not accepting requests")
            self._requests.put(request)

    def apply_storage_mutation(
        self,
        mutate: Callable[[AppConfig], None],
        timeout: float = 5.0,
    ) -> ConfigState:
        """Serialize an asset mutation with frame-boundary reconfiguration.

        Storage changes are ordered in the same queue as PATCH activation, so
        a candidate is either preflighted before deletion (and makes the asset
        active) or afterward (and fails to open it). Because effective config
        does not change, a successful mutation must not advance its version.
        """
        if not self.running or not self._startup_done.is_set():
            raise ReconfigurationUnavailable("pipeline is not running")
        if threading.current_thread() is self._thread:
            raise ReconfigurationUnavailable(
                "cannot synchronously mutate storage from the pipeline worker"
            )
        request = _MutationRequest(mutate)
        self._enqueue_request(request)
        if not request.done.wait(timeout):
            if request.cancel():
                raise ReconfigurationUnavailable(
                    f"pipeline did not acknowledge storage mutation within {timeout:.1f}s"
                )
            request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise ReconfigurationUnavailable("pipeline returned no storage result")
        return request.result

    # -- lifecycle -----------------------------------------------------
    def _open_resources(self, state: ConfigState) -> _Resources:
        cfg = state.config
        # ExitStack protects every successfully opened backend if a later
        # constructor fails. Once complete, _Resources owns deterministic close.
        with ExitStack() as startup:
            capture = open_capture(cfg.camera)
            startup.callback(_safe_close, capture, "capture")
            if self._model_preparation is not None:
                segmenter = create_segmenter(
                    cfg.segmentation,
                    preparation=self._model_preparation,
                )
            else:
                segmenter = create_segmenter(cfg.segmentation)
            startup.callback(_safe_close, segmenter, "segmenter")
            refiner = refiner_for(cfg.segmentation, segmenter)
            backdrop = _build_backdrop(cfg)
            if backdrop is not None:
                startup.callback(_safe_close, backdrop, "backdrop")
            output = open_output(cfg.output, cfg.camera.width, cfg.camera.height)
            startup.callback(_safe_close, output, "video output")
            resources = _Resources(
                cfg, state.version, capture, segmenter, refiner, backdrop, output
            )
            startup.pop_all()
            return resources

    def _run(self) -> None:
        resources: _Resources | None = None
        try:
            # Reset replay evidence only after every stale renderer lease and
            # queued remote frame have been invalidated at one hub boundary.
            self.hub.reset_remote_session(self._reset_raw_replay_history)
            state = self.runtime.read()
            resources = self._open_resources(state)
            self._active_state = state
            initial_output = self._preflight(resources)
            self._update_identity_stats(resources)
            self._startup_done.set()
            self._loop(resources, initial_output=initial_output)
        except BaseException as exc:
            log.exception("pipeline crashed")
            self._error = exc
        finally:
            self._stop.set()
            with self._request_enqueue_lock:
                self._fail_pending(
                    ReconfigurationUnavailable("pipeline worker stopped")
                )
            if resources is not None:
                resources.close()
            self._drain_deferred_closes()
            # On startup failure, readiness is not published until teardown
            # finishes. A blocked close is therefore observed as a surviving
            # startup worker rather than being hidden behind the root error.
            self._startup_done.set()

    def _fail_pending(self, error: BaseException) -> None:
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            if isinstance(request, _PatchRequest):
                with request.lock:
                    activation = request.prepared_activation
                    request.prepared_activation = None
                    if request.result is None and request.error is None:
                        request.error = error
                request.done.set()
                self._schedule_discard_activation(activation)
            else:
                request.fail(error)

    # -- reconfiguration ----------------------------------------------
    @staticmethod
    def _prepare_activation_off_lane(
        current: AppConfig,
        candidate: AppConfig,
    ) -> _Activation:
        """Construct changed resources without occupying the frame worker."""

        activation = _Activation(candidate=candidate)
        try:
            if candidate.segmentation != current.segmentation:
                activation.segmenter = create_segmenter(candidate.segmentation)
                activation.replace_segmenter = True
                activation.refiner = refiner_for(
                    candidate.segmentation, activation.segmenter
                )
            if _backdrop_key(candidate) != _backdrop_key(current):
                activation.backdrop = _build_backdrop(candidate)
                activation.replace_backdrop = True
        except Exception as exc:
            activation.discard()
            raise ActivationError(str(exc)) from exc
        return activation

    def _stage_activation(
        self,
        resources: _Resources,
        candidate: AppConfig,
        prepared: _Activation | None,
    ) -> _Activation:
        """Attach reused live pointers to an already constructed candidate."""

        old_cfg = resources.cfg
        if prepared is None:
            raise ActivationError("candidate resources were not prepared off-lane")
        activation = prepared
        activation.candidate = candidate
        segmentation_changed = candidate.segmentation != old_cfg.segmentation
        background_changed = _backdrop_key(candidate) != _backdrop_key(old_cfg)
        if segmentation_changed != activation.replace_segmenter:
            raise ActivationError("prepared segmentation candidate is stale")
        if background_changed != activation.replace_backdrop:
            raise ActivationError("prepared background candidate is stale")
        if not segmentation_changed:
            activation.refiner = resources.refiner
        if not background_changed:
            activation.backdrop = resources.backdrop
        return activation

    def _trial_activation(
        self,
        resources: _Resources,
        activation: _Activation,
        frame: np.ndarray,
    ) -> None:
        """Exercise only staged/synthetic state before committing.

        Existing working segmenter, refiner, and backdrop objects are never
        called here, so a failed trial preserves recurrent state, temporal mask,
        video position, and blur caches exactly.
        """
        old_cfg = resources.cfg
        segmentation_changed = activation.candidate.segmentation != old_cfg.segmentation
        background_changed = _backdrop_key(activation.candidate) != _backdrop_key(
            old_cfg
        )
        compositing_changed = activation.candidate.compositing != old_cfg.compositing
        if not (segmentation_changed or background_changed or compositing_changed):
            return
        try:
            if segmentation_changed:
                mask = self._segment_and_refine_mask(
                    activation.segmenter,
                    activation.refiner,
                    frame,
                    privacy_safe=activation.candidate.background.mode == "remote",
                )
            else:
                # A deterministic synthetic edge exercises backdrop/compositor
                # contracts without touching the working processing state.
                mask = np.full(frame.shape[:2], 0.5, dtype=np.float32)

            if background_changed and activation.backdrop is not None:
                if isinstance(activation.backdrop, BlurBackdrop):
                    activation.backdrop.set_source_frame(frame, mask)
                bg = activation.backdrop.frame(frame.shape[1], frame.shape[0])
            else:
                bg = np.zeros_like(frame)

            edge_fg = (
                activation.segmenter.last_foreground
                if segmentation_changed
                and activation.candidate.compositing.use_model_foreground
                else None
            )
            out = composite(
                frame,
                bg,
                mask,
                light_wrap=activation.candidate.compositing.light_wrap,
                edge_foreground=edge_fg,
            )
            self._validate_output_frame(out, frame)
        except Exception as exc:
            raise ActivationError(str(exc)) from exc

    def _install_activation(
        self,
        resources: _Resources,
        activation: _Activation,
        version: int,
    ) -> tuple[Any, Any, AppConfig]:
        """Swap prepared pointers using only precomputed, reversible operations."""
        old_cfg = resources.cfg
        old_segmenter = resources.segmenter
        old_refiner = resources.refiner
        old_backdrop = resources.backdrop
        old_version = resources.version
        old_active_state = self._active_state
        # Allocate/copy before the first effective pointer changes. The actual
        # swap below is assignment-only and has an explicit rollback guard.
        new_active_state = ConfigState(
            activation.candidate.model_copy(deep=True), version
        )
        try:
            if activation.replace_segmenter:
                resources.segmenter = activation.segmenter
            resources.refiner = activation.refiner
            if activation.replace_backdrop:
                resources.backdrop = activation.backdrop
            resources.cfg = activation.candidate
            resources.version = version
            self._active_state = new_active_state
        except BaseException:
            resources.segmenter = old_segmenter
            resources.refiner = old_refiner
            resources.backdrop = old_backdrop
            resources.cfg = old_cfg
            resources.version = old_version
            self._active_state = old_active_state
            raise
        return (
            old_backdrop if activation.replace_backdrop else None,
            old_segmenter if activation.replace_segmenter else None,
            old_cfg,
        )

    def _post_install_activation(
        self, resources: _Resources, old_cfg: AppConfig
    ) -> None:
        """Run non-critical hub side effects without invalidating a commit."""
        if old_cfg.background.mode != resources.cfg.background.mode:
            try:
                self.hub.invalidate_remote_session()
            except Exception:
                log.exception("cannot invalidate remote session after mode switch")
                self._privacy_history_exhausted = True
        try:
            self._update_identity_stats(resources)
        except Exception:
            log.exception("cannot update pipeline identity statistics")

    def _handle_patch_request(
        self,
        resources: _Resources,
        request: _PatchRequest,
        trial_frame: np.ndarray,
    ) -> None:
        with request.lock:
            if request.cancelled:
                activation = request.prepared_activation
                request.prepared_activation = None
                request.done.set()
                self._schedule_discard_activation(activation)
                return
            # Ownership leaves the request before any fallible trial work.
            # A concurrent timeout can only reclaim an activation that the
            # frame lane has not claimed yet.
            activation = request.prepared_activation
            request.prepared_activation = None
        current = self.runtime.read()
        storage_epoch = self._read_storage_epoch()
        if (
            current.version != request.expected_version
            or resources.version != request.expected_version
        ):
            self._schedule_discard_activation(activation)
            request.fail(ConfigConflictError(request.expected_version, current.version))
            return
        if storage_epoch != request.storage_epoch:
            self._schedule_discard_activation(activation)
            request.fail(ActivationError("candidate assets changed during preparation"))
            return
        try:
            activation = self._stage_activation(
                resources,
                request.activation_candidate or request.candidate,
                activation,
            )
            self._trial_activation(resources, activation, trial_frame)
            if activation.replace_backdrop and hasattr(
                activation.backdrop, "reset_stats"
            ):
                # Candidate trials are deliberately unsent.  The provider
                # object is installed after the trial, so reset only its public
                # counters (not playback state) before publishing the new
                # current-provider telemetry generation.
                activation.backdrop.reset_stats()
            # Resources were exercised through the hidden staging path, but
            # the effective snapshot published below must contain only the
            # promoted final path.
            activation.candidate = request.candidate
        except BaseException as exc:
            self._schedule_discard_activation(activation)
            request.fail(exc)
            return

        old_backdrop = old_segmenter = None
        old_cfg: AppConfig | None = None
        with request.lock:
            if request.cancelled:
                self._schedule_discard_activation(activation)
                request.done.set()
                return
            try:

                def activate(next_version: int) -> None:
                    nonlocal old_backdrop, old_segmenter, old_cfg
                    promoted = False
                    if request.before_activate is not None:
                        request.before_activate()
                        promoted = True
                    try:
                        old_backdrop, old_segmenter, old_cfg = self._install_activation(
                            resources, activation, next_version
                        )
                    except BaseException:
                        if promoted and request.rollback_activate is not None:
                            request.rollback_activate()
                        raise

                committed = self._runtime_writer.commit_with_activation(
                    request.candidate, request.expected_version, activate
                )
            except ConfigVersionConflictError as exc:
                self._schedule_discard_activation(activation)
                request.error = ConfigConflictError(
                    exc.expected_version, exc.current_version
                )
            except BaseException as exc:
                self._schedule_discard_activation(activation)
                request.error = exc
            else:
                if old_cfg is not None:
                    self._post_install_activation(resources, old_cfg)
                request.result = committed
                changed = _changed_paths(old_cfg or resources.cfg, request.candidate)
                log.info(
                    "config update accepted origin=%s version=%d fields=%s summary=%s",
                    request.origin,
                    committed.version,
                    ",".join(changed) or "none",
                    sanitized_config_summary(request.candidate, changed),
                )
            # Publish success/failure before teardown. A blocking or faulty
            # old backend must not delay or invalidate an already-committed ack.
            request.done.set()
        self._schedule_close(old_backdrop, "replaced backdrop")
        self._schedule_close(old_segmenter, "replaced segmenter")

    def _schedule_discard_activation(self, activation: _Activation | None) -> None:
        if activation is None:
            return
        if activation.replace_backdrop:
            backdrop = activation.backdrop
            activation.backdrop = None
            activation.replace_backdrop = False
            self._schedule_close(backdrop, "discarded staged backdrop")
        if activation.replace_segmenter:
            segmenter = activation.segmenter
            activation.segmenter = None
            activation.replace_segmenter = False
            self._schedule_close(segmenter, "discarded staged segmenter")

    def _handle_mutation_request(
        self, resources: _Resources, request: _MutationRequest
    ) -> None:
        with request.lock:
            if request.cancelled:
                request.done.set()
                return
            try:
                # Give the callback a read-only-by-convention copy so a
                # storage operation cannot mutate the live resource config.
                request.mutate(resources.cfg.model_copy(deep=True))
                request.result = self.runtime.read()
                if request.result.version != resources.version:
                    raise ConfigConflictError(resources.version, request.result.version)
                with self._storage_epoch_lock:
                    self._storage_epoch += 1
            except BaseException as exc:
                request.error = exc
            request.done.set()

    def _schedule_close(self, resource: Any, label: str) -> None:
        """Close replaced resources off the frame worker and track liveness."""
        if resource is None:
            return

        worker: threading.Thread

        def close_resource() -> None:
            try:
                _safe_close(resource, label)
            finally:
                with self._teardown_lock:
                    self._teardown_threads.discard(worker)

        worker = threading.Thread(
            target=close_resource,
            name=f"teardown-{label.replace(' ', '-')}",
            daemon=True,
        )
        with self._teardown_lock:
            self._teardown_threads.add(worker)
        try:
            worker.start()
        except BaseException:
            with self._teardown_lock:
                self._teardown_threads.discard(worker)
                self._deferred_closes.append((resource, label))
            # The new config/resource pair was committed before teardown was
            # scheduled. Thread exhaustion must not revoke that success or kill
            # the frame worker; retain the old resource for shutdown cleanup.
            log.exception(
                "cannot schedule %s teardown; deferring until shutdown", label
            )

    def _drain_deferred_closes(self) -> None:
        """Close resources whose post-commit teardown thread could not start."""
        with self._teardown_lock:
            deferred = tuple(self._deferred_closes)
            self._deferred_closes.clear()
        for resource, label in deferred:
            _safe_close(resource, f"deferred {label}")

    def _update_identity_stats(self, resources: _Resources) -> None:
        cfg = resources.cfg
        capture_health = (
            resources.capture.health_snapshot()
            if hasattr(resources.capture, "health_snapshot")
            else None
        )
        output_fallback = bool(getattr(resources.output, "fallback_active", False))
        segmentation_fallback = cfg.segmentation.backend == "auto" and isinstance(
            resources.segmenter, HeuristicSegmenter
        )
        video_stats = (
            resources.backdrop.stats_dict()
            if resources.backdrop is not None
            and hasattr(resources.backdrop, "stats_dict")
            else dict(_VIDEO_STATS_DEFAULTS)
        )
        self.hub.update_stats(
            mode=cfg.background.mode,
            segmentation_backend=type(resources.segmenter).__name__,
            segmentation_device=resources.segmenter.device,
            output_backend=type(resources.output).__name__,
            output_target_fps=cfg.output.fps,
            output_fallback_active=output_fallback,
            output_fallback_reason=(
                getattr(resources.output, "fallback_reason", "")
                if output_fallback
                else ""
            ),
            segmentation_fallback_active=segmentation_fallback,
            segmentation_fallback_reason=(
                "ml-backend-unavailable" if segmentation_fallback else ""
            ),
            capture_backend=(
                getattr(capture_health, "backend", type(resources.capture).__name__)
                if capture_health is not None
                else type(resources.capture).__name__
            ),
            capture_fourcc=getattr(capture_health, "fourcc", None),
            capture_width=getattr(capture_health, "width", cfg.camera.width),
            capture_height=getattr(capture_health, "height", cfg.camera.height),
            capture_fps_reported=getattr(
                capture_health, "fps_reported", float(cfg.camera.fps)
            ),
            capture_target_fps=cfg.camera.fps,
            remote_fallback_mode=(
                "privacy-slate" if cfg.background.mode == "remote" else ""
            ),
            config_version=resources.version,
            **video_stats,
        )
        self._log_fallback_transition(
            "output",
            output_fallback,
            getattr(resources.output, "fallback_reason", "") if output_fallback else "",
        )
        self._log_fallback_transition(
            "segmentation",
            segmentation_fallback,
            "ml-backend-unavailable" if segmentation_fallback else "",
        )

    def _log_fallback_transition(self, kind: str, active: bool, reason: str) -> None:
        """Emit one record only when a fallback state or reason changes."""

        state = (active, reason if active else "")
        previous = self._fallback_log_states.get(kind)
        if previous == state:
            return
        self._fallback_log_states[kind] = state
        if active:
            if kind == "output":
                log.warning(
                    "output fallback active reason=%s; frames are API-only. "
                    "Run the platform setup script to restore the virtual camera",
                    reason or "unknown",
                )
            else:
                log.warning(
                    "%s fallback active reason=%s",
                    kind,
                    reason or "unknown",
                )
        elif previous is not None and previous[0]:
            log.info("%s fallback recovered", kind)

    # -- frame processing ---------------------------------------------
    @staticmethod
    def _validate_output_frame(out: np.ndarray, source: np.ndarray) -> None:
        if (
            not isinstance(out, np.ndarray)
            or out.dtype != np.uint8
            or out.ndim != 3
            or out.shape != source.shape
            or out.shape[2] != 3
        ):
            raise ValueError(
                f"processed frame must be uint8 BGR {source.shape}, got "
                f"{getattr(out, 'dtype', None)} {getattr(out, 'shape', None)}"
            )

    @staticmethod
    def _privacy_slate(shape: tuple[int, ...]) -> np.ndarray:
        """Return a fixed opaque slate containing no camera-derived pixels."""

        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"privacy slate requires an HxWx3 shape, got {shape}")
        height, width, _channels = shape
        if height <= 0 or width <= 0:
            raise ValueError(f"privacy slate requires a non-empty shape, got {shape}")
        # A two-tone neutral checker remains visibly a privacy fallback and,
        # unlike a mean-color or blurred fallback, cannot reveal source color,
        # silhouettes, text, or other spatial detail. The pattern also avoids
        # becoming byte-identical to an ordinary uniform camera frame.
        tile = max(2, min(height, width) // 8)
        yy, xx = np.indices((height, width), dtype=np.int32)
        checker = ((yy // tile) + (xx // tile)) & 1
        slate = np.empty((height, width, 3), dtype=np.uint8)
        slate[checker == 0] = (24, 27, 32)
        slate[checker == 1] = (36, 40, 48)
        return slate

    @staticmethod
    def _emergency_blur(frame: np.ndarray) -> np.ndarray:
        """Compatibility name for the input-independent privacy slate."""

        return Pipeline._privacy_slate(frame.shape)

    @staticmethod
    def _validate_mask(
        mask: np.ndarray,
        frame: np.ndarray,
        *,
        privacy_safe: bool,
    ) -> np.ndarray:
        """Validate a mask before any compositor or backdrop can consume it."""

        if (
            not isinstance(mask, np.ndarray)
            or mask.ndim != 2
            or mask.shape != frame.shape[:2]
            or mask.dtype != np.float32
            or mask.size == 0
        ):
            raise _PrivacyViolation(
                "segmentation-invalid-mask",
                "segmenter returned a mask with an invalid type or shape",
            )
        if not np.isfinite(mask).all():
            raise _PrivacyViolation(
                "segmentation-invalid-mask",
                "segmenter returned a non-finite mask",
            )
        minimum = float(np.min(mask))
        maximum = float(np.max(mask))
        if minimum < 0.0 or maximum > 1.0:
            raise _PrivacyViolation(
                "segmentation-invalid-mask",
                "segmenter returned a mask outside [0, 1]",
            )
        validated = np.ascontiguousarray(mask, dtype=np.float32)
        if privacy_safe and bool(np.all(validated >= (1.0 - 1e-6))):
            raise _PrivacyViolation(
                "segmentation-all-foreground",
                "remote fallback mask exposes the entire camera frame",
            )
        return validated

    @classmethod
    def _segment_and_refine_mask(
        cls,
        segmenter: Any,
        refiner: Any,
        frame: np.ndarray,
        *,
        privacy_safe: bool,
    ) -> np.ndarray:
        """Validate the backend contract both before and after refinement."""

        raw = cls._validate_mask(
            segmenter.segment(frame),
            frame,
            privacy_safe=False,
        )
        refined = refiner.refine(raw, frame)
        return cls._validate_mask(
            refined,
            frame,
            privacy_safe=privacy_safe,
        )

    @staticmethod
    def _raw_fingerprint(frame: np.ndarray) -> _RawFingerprint:
        """Build a compact low-pass fingerprint without retaining raw pixels."""

        thumbnail = cv2.resize(
            frame,
            _RAW_FINGERPRINT_SIZE,
            interpolation=cv2.INTER_AREA,
        )
        coarse = cv2.resize(
            frame,
            (4, 3),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint16)
        luminance = (
            coarse[..., 0] * 29 + coarse[..., 1] * 150 + coarse[..., 2] * 77
        ) >> 8
        means = tuple(
            int(value)
            for value in np.rint(thumbnail.mean(axis=(0, 1), dtype=np.float64))
        )
        jpeg_features = tuple(int(value) for value in luminance.reshape(-1)) + means
        thumbnail.setflags(write=False)
        return _RawFingerprint(
            jpeg_features=jpeg_features,
            thumbnail=thumbnail,
        )

    def _reset_raw_replay_history(self) -> None:
        """Begin a new pipeline privacy session after renderer invalidation."""

        self._recent_raw_fingerprints.clear()
        self._privacy_history_exhausted = False
        self._privacy_invalidated_session = None
        self._latest_raw_frame = None

    def _remember_raw_frame(self, frame: np.ndarray) -> bool:
        # Keep one current copy for the precise near-raw comparison. Historical
        # frames are represented only by small, non-reversible fingerprints.
        self._latest_raw_frame = frame.copy()
        if self._privacy_history_exhausted:
            return False
        if len(self._recent_raw_fingerprints) >= self._raw_fingerprint_capacity:
            # Never evict evidence into a permissive state.  The caller revokes
            # the renderer lease before any subsequent publication can proceed.
            self._privacy_history_exhausted = True
            return False
        self._recent_raw_fingerprints.append(self._raw_fingerprint(frame))
        return True

    def _record_remote_raw_frame(self, frame: np.ndarray) -> None:
        """Remember frames exposed to a renderer and revoke on exhaustion."""

        session = self.hub.active_remote_session()
        if session is None:
            self._latest_raw_frame = frame.copy()
            return
        if self._remember_raw_frame(frame):
            return
        if self._privacy_invalidated_session == session:
            return
        self.hub.invalidate_remote_session(session)
        self._privacy_invalidated_session = session

    @staticmethod
    def _lowres_raw_similarity(
        candidate: np.ndarray,
        raw_thumbnail: np.ndarray,
    ) -> bool:
        candidate_thumbnail = cv2.resize(
            candidate,
            _RAW_FINGERPRINT_SIZE,
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        reference = raw_thumbnail.astype(np.float32)
        mean_delta = float(np.abs(candidate_thumbnail - reference).mean())
        if mean_delta > _RAW_ECHO_LOWRES_MEAN_DELTA:
            return False
        candidate_centered = candidate_thumbnail - float(candidate_thumbnail.mean())
        reference_centered = reference - float(reference.mean())
        denominator = float(
            np.sqrt(
                np.sum(candidate_centered * candidate_centered, dtype=np.float64)
                * np.sum(reference_centered * reference_centered, dtype=np.float64)
            )
        )
        if denominator <= 1e-9:
            # Uniform/near-uniform frames have no stable correlation; a small
            # low-resolution delta is sufficient to treat them as an echo.
            return mean_delta <= _RAW_ECHO_MEAN_DELTA
        correlation = float(
            np.sum(
                candidate_centered * reference_centered,
                dtype=np.float64,
            )
            / denominator
        )
        return correlation >= _RAW_ECHO_LOWRES_CORRELATION

    @staticmethod
    def _is_near_raw(candidate: np.ndarray, raw: np.ndarray) -> bool:
        if candidate.shape != raw.shape:
            return False
        delta = np.abs(candidate.astype(np.int16) - raw.astype(np.int16))
        if not np.any(delta):
            return True
        changed = np.any(delta > _RAW_ECHO_PIXEL_TOLERANCE, axis=2)
        changed_fraction = float(np.count_nonzero(changed)) / changed.size
        pixel_near = (
            changed_fraction <= _RAW_ECHO_CHANGED_FRACTION
            or float(delta.mean()) <= _RAW_ECHO_MEAN_DELTA
        )
        if pixel_near:
            return True
        return Pipeline._lowres_raw_similarity(
            candidate,
            Pipeline._raw_fingerprint(raw).thumbnail,
        )

    def _matches_recent_raw(self, candidate: np.ndarray) -> bool:
        if not self._recent_raw_fingerprints:
            return False
        candidate_fingerprint = self._raw_fingerprint(candidate)
        return self._recent_raw_fingerprints.matches(candidate_fingerprint)

    def _guard_remote_output(
        self,
        candidate: np.ndarray,
        raw: np.ndarray,
        *,
        privacy_safe: bool,
    ) -> tuple[np.ndarray, str]:
        """Apply the final fail-closed gate shared by every output sink."""

        if not privacy_safe:
            return candidate, ""
        if self._privacy_history_exhausted:
            return self._privacy_slate(raw.shape), "privacy-history-exhausted"
        try:
            self._validate_output_frame(candidate, raw)
        except Exception:
            return self._privacy_slate(raw.shape), "privacy-invalid-output"
        if self._is_near_raw(candidate, raw):
            return self._privacy_slate(raw.shape), "privacy-raw-echo"
        if self._matches_recent_raw(candidate):
            return self._privacy_slate(raw.shape), "privacy-delayed-raw-echo"
        return candidate, ""

    def _render_local_mode(
        self, resources: _Resources, frame: np.ndarray
    ) -> np.ndarray:
        mode = resources.cfg.background.mode
        if mode == "passthrough":
            return frame
        if mode == "remote" and isinstance(resources.segmenter, NullSegmenter):
            return self._emergency_blur(frame)
        out, _ = self._local_composite(
            resources,
            frame,
            privacy_safe=mode == "remote",
        )
        return out

    def _preflight(self, resources: _Resources) -> np.ndarray:
        """Read and process a real frame before reporting startup readiness."""
        camera_wait = (
            2.0
            if resources.cfg.camera.synthetic
            else resources.cfg.camera.recovery_timeout_s + 2.0
        )
        deadline = time.monotonic() + camera_wait
        frame: np.ndarray | None = None
        while frame is None and time.monotonic() < deadline and not self._stop.is_set():
            frame = resources.capture.read()
            if frame is None:
                self._stop.wait(0.05)
        if frame is None:
            raise ActivationError("capture returned no frame during startup preflight")
        remote_mode = resources.cfg.background.mode == "remote"
        privacy_reason = ""
        try:
            if resources.cfg.background.mode == "passthrough":
                # Passthrough does not need a mask to render, but the segmenter
                # is already part of the hot-swappable resource generation.
                # Exercise it so a later background-only PATCH cannot reveal a
                # backend failure for the first time.
                self._segment_and_refine_mask(
                    resources.segmenter,
                    resources.refiner,
                    frame,
                    privacy_safe=False,
                )
            if remote_mode:
                self._record_remote_raw_frame(frame)
                # Exercise the configured fallback without publishing it. A
                # remote startup probes the real output backend only with the
                # fixed slate, never with a camera-derived composite.
                _candidate, privacy_reason = self._local_composite(
                    resources,
                    frame,
                    privacy_safe=True,
                )
                out = self._privacy_slate(frame.shape)
                privacy_reason = privacy_reason or "startup-slate"
            else:
                self._latest_raw_frame = frame.copy()
                out = self._render_local_mode(resources, frame)
            self._validate_output_frame(out, frame)
        except _PrivacyViolation as exc:
            raise ActivationError(f"invalid mask: {exc}") from exc
        # The privacy gate sits at the final publication boundary. Startup uses
        # the same fail-closed path as the steady-state loop, so a bad mask or
        # raw-looking preflight result can never reach the real output backend.
        out, gate_reason = self._guard_remote_output(
            out,
            frame,
            privacy_safe=remote_mode,
        )
        privacy_reason = gate_reason or privacy_reason
        resources.output.send(out.copy() if remote_mode else out)
        health = (
            resources.capture.health_snapshot()
            if hasattr(resources.capture, "health_snapshot")
            else None
        )
        self.hub.update_stats(
            frames_in=1,
            frames_out=1,
            capture_frames_read=getattr(health, "frames_read", 1),
            capture_dropped_frames=getattr(health, "dropped_frames", 0),
            capture_read_failures=getattr(health, "read_failures", 0),
            capture_restarts=getattr(health, "restarts", 0),
            capture_stalled=getattr(health, "stalled", False),
            capture_frame_age_ms=getattr(health, "frame_age_ms", None),
            capture_read_ms=getattr(health, "read_ms", None),
            capture_fps=getattr(health, "capture_fps", 0.0),
            capture_target_met=getattr(health, "target_met", None),
            remote_fallback_active=bool(privacy_reason),
            remote_fallback_count=1 if privacy_reason else 0,
            remote_fallback_reason=privacy_reason,
        )
        if privacy_reason:
            log.warning("remote privacy fallback active reason=%s", privacy_reason)
        return out

    def _local_composite(
        self,
        resources: _Resources,
        frame: np.ndarray,
        *,
        privacy_safe: bool,
        timings: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, str]:
        cfg = resources.cfg
        backdrop = resources.backdrop
        if privacy_safe and isinstance(resources.segmenter, NullSegmenter):
            return self._emergency_blur(frame), "segmentation-none"

        try:
            started = time.monotonic_ns()
            mask = self._segment_and_refine_mask(
                resources.segmenter,
                resources.refiner,
                frame,
                privacy_safe=privacy_safe,
            )
            if timings is not None:
                timings["segmentation_ms"] = (
                    time.monotonic_ns() - started
                ) / 1_000_000.0
            started = time.monotonic_ns()
            if isinstance(backdrop, BlurBackdrop):
                backdrop.set_source_frame(frame, mask)
            if backdrop is None:
                if timings is not None:
                    timings["background_ms"] = (
                        time.monotonic_ns() - started
                    ) / 1_000_000.0
                if privacy_safe:
                    return self._emergency_blur(frame), "local-failure"
                return frame, ""
            bg = backdrop.frame(frame.shape[1], frame.shape[0])
            if timings is not None:
                timings["background_ms"] = (time.monotonic_ns() - started) / 1_000_000.0
            edge_fg = (
                resources.segmenter.last_foreground
                if cfg.compositing.use_model_foreground
                else None
            )
            started = time.monotonic_ns()
            rendered = composite(
                frame,
                bg,
                mask,
                light_wrap=cfg.compositing.light_wrap,
                edge_foreground=edge_fg,
            )
            self._validate_output_frame(rendered, frame)
            if timings is not None:
                timings["composite_ms"] = (time.monotonic_ns() - started) / 1_000_000.0
            return (
                rendered,
                "",
            )
        except _PrivacyViolation as exc:
            if not privacy_safe:
                raise
            log.warning("local remote-mode privacy fallback reason=%s", exc.reason)
            return self._privacy_slate(frame.shape), exc.reason
        except Exception:
            if not privacy_safe:
                raise
            log.exception("local remote-mode fallback failed")
            return self._privacy_slate(frame.shape), "local-failure"

    def _privacy_checked(
        self, candidate: np.ndarray, frame: np.ndarray, privacy_safe: bool
    ) -> np.ndarray:
        """Compatibility wrapper for direct privacy-gate callers and tests."""

        guarded, _reason = self._guard_remote_output(
            candidate,
            frame,
            privacy_safe=privacy_safe,
        )
        return guarded

    def _loop(
        self, resources: _Resources, *, initial_output: np.ndarray | None = None
    ) -> None:
        frame_interval = 1.0 / resources.cfg.output.fps
        fps_window: deque[float] = deque()
        initial_sends = 1 if initial_output is not None else 0
        frames_in = frames_out = initial_sends
        initial_remote_slate = (
            initial_output is not None and resources.cfg.background.mode == "remote"
        )
        remote_used = 0
        fallback_count = 1 if initial_remote_slate else 0
        repeated_frames = deadline_misses = 0
        last_output = initial_output
        fallback_active = initial_remote_slate
        fallback_reason = "startup-slate" if initial_remote_slate else ""
        previous_remote_fallback: tuple[bool, str] = (False, "")
        stage_ewma: dict[str, float | None] = {
            "segmentation_ms": None,
            "background_ms": None,
            "composite_ms": None,
            "output_send_ms": None,
            "frame_processing_ms": None,
        }

        while not self._stop.is_set():
            loop_start = time.monotonic()
            frame = resources.capture.read()
            used_remote_candidate = False
            timings = {
                "segmentation_ms": 0.0,
                "background_ms": 0.0,
                "composite_ms": 0.0,
            }
            processed = frame is not None
            if processed:
                assert frame is not None
                frames_in += 1
                self.hub.publish_raw(frame)

                # A real current frame is the activation trial input. No candidate
                # is committed until this preflight succeeds.
                try:
                    request = self._requests.get_nowait()
                except queue.Empty:
                    request = None
                if request is not None:
                    if isinstance(request, _PatchRequest):
                        self._handle_patch_request(resources, request, frame)
                    else:
                        self._handle_mutation_request(resources, request)

                process_started = time.monotonic_ns()
                cfg = resources.cfg
                mode = cfg.background.mode
                fallback_active = False
                fallback_reason = ""
                if mode == "remote":
                    self._record_remote_raw_frame(frame)
                else:
                    self._latest_raw_frame = frame.copy()

                if mode == "remote":
                    remote, fallback_reason = self.hub.remote_frame_status(
                        max_age_s=cfg.api.remote_timeout_ms / 1000.0
                    )
                    if remote is not None:
                        if (
                            remote.dtype != np.uint8
                            or remote.ndim != 3
                            or remote.shape[2] != 3
                        ):
                            remote = None
                            fallback_reason = "invalid"
                        elif remote.shape != frame.shape:
                            remote = None
                            fallback_reason = "wrong-size"
                    if remote is not None:
                        out_frame = remote
                        used_remote_candidate = True
                        fallback_reason = ""
                    else:
                        # A segmented local composite necessarily preserves the
                        # foreground and is therefore not a privacy boundary.
                        # Missing, stale, or malformed remote output always
                        # fails closed to the input-independent slate.
                        out_frame = self._privacy_slate(frame.shape)
                        fallback_active = True
                        fallback_count += 1
                elif mode == "passthrough" or resources.backdrop is None:
                    out_frame = frame
                else:
                    out_frame, _ = self._local_composite(
                        resources,
                        frame,
                        privacy_safe=False,
                        timings=timings,
                    )
                frame_processing_ms = (
                    time.monotonic_ns() - process_started
                ) / 1_000_000.0
                if frame_processing_ms / 1000.0 > frame_interval:
                    deadline_misses += 1
                samples = (
                    *timings.items(),
                    ("frame_processing_ms", frame_processing_ms),
                )
                for name, sample in samples:
                    stage_ewma[name] = _ewma(stage_ewma[name], sample)
            else:
                if last_output is None:
                    self._stop.wait(min(0.01, frame_interval))
                    continue
                out_frame = last_output
                repeated_frames += 1

            mode = resources.cfg.background.mode
            guard_source = frame if frame is not None else self._latest_raw_frame
            if guard_source is None:
                # Defensive invariant: startup preflight always establishes the
                # source associated with ``initial_output`` before this loop.
                raise RuntimeError("output publication has no associated source frame")
            out_frame, privacy_reason = self._guard_remote_output(
                out_frame,
                guard_source,
                privacy_safe=mode == "remote",
            )
            if privacy_reason:
                if not fallback_active:
                    fallback_count += 1
                fallback_active = True
                fallback_reason = privacy_reason
                used_remote_candidate = False
            if used_remote_candidate:
                remote_used += 1
            # Repeats retain only the already-guarded output, so loss of camera
            # input cannot resurrect an unsafe pre-gate candidate.
            last_output = out_frame

            send_started = time.monotonic_ns()
            resources.output.send(out_frame.copy() if mode == "remote" else out_frame)
            output_send_ms = (time.monotonic_ns() - send_started) / 1_000_000.0
            stage_ewma["output_send_ms"] = _ewma(
                stage_ewma["output_send_ms"], output_send_ms
            )
            frames_out += 1

            now = time.monotonic()
            fps_window.append(now)
            while fps_window and now - fps_window[0] > 2.0:
                fps_window.popleft()
            if len(fps_window) >= 2:
                span = fps_window[-1] - fps_window[0]
                measured_fps = (len(fps_window) - 1) / span if span > 0 else 0.0
            else:
                measured_fps = 0.0
            attainment = (
                min(100.0, measured_fps / resources.cfg.output.fps * 100.0)
                if len(fps_window) >= 2
                else None
            )
            capture_health = (
                resources.capture.health_snapshot()
                if hasattr(resources.capture, "health_snapshot")
                else None
            )
            video_stats = (
                resources.backdrop.stats_dict()
                if resources.backdrop is not None
                and hasattr(resources.backdrop, "stats_dict")
                else dict(_VIDEO_STATS_DEFAULTS)
            )
            remote_state = (fallback_active, fallback_reason)
            if remote_state != previous_remote_fallback:
                if fallback_active:
                    log.warning(
                        "remote fallback active mode=%s reason=%s",
                        "privacy-slate",
                        fallback_reason,
                    )
                elif previous_remote_fallback[0]:
                    log.info("remote renderer recovered; privacy fallback inactive")
                previous_remote_fallback = remote_state
            self.hub.update_stats(
                frames_in=frames_in,
                frames_out=frames_out,
                remote_frames_used=remote_used,
                remote_fallback_active=fallback_active,
                remote_fallback_count=fallback_count,
                remote_fallback_reason=fallback_reason,
                fps=measured_fps,
                fps_attainment_pct=attainment,
                output_repeated_frames=repeated_frames,
                processing_deadline_misses=deadline_misses,
                capture_fps=getattr(capture_health, "capture_fps", 0.0),
                capture_target_met=getattr(capture_health, "target_met", None),
                capture_backend=getattr(
                    capture_health, "backend", type(resources.capture).__name__
                ),
                capture_fourcc=getattr(capture_health, "fourcc", None),
                capture_width=getattr(capture_health, "width", None),
                capture_height=getattr(capture_health, "height", None),
                capture_fps_reported=getattr(capture_health, "fps_reported", None),
                capture_frames_read=getattr(capture_health, "frames_read", frames_in),
                capture_dropped_frames=getattr(capture_health, "dropped_frames", 0),
                capture_read_failures=getattr(capture_health, "read_failures", 0),
                capture_restarts=getattr(capture_health, "restarts", 0),
                capture_stalled=getattr(capture_health, "stalled", False),
                capture_frame_age_ms=getattr(capture_health, "frame_age_ms", None),
                capture_read_ms=getattr(capture_health, "read_ms", None),
                segmentation_ms=stage_ewma["segmentation_ms"],
                background_ms=stage_ewma["background_ms"],
                composite_ms=stage_ewma["composite_ms"],
                output_send_ms=stage_ewma["output_send_ms"],
                frame_processing_ms=stage_ewma["frame_processing_ms"],
                config_version=resources.version,
                **video_stats,
            )
            # Publish only after the matching counters/fallback state are
            # visible, so consumers never observe a frame with stale status.
            self.hub.publish_output(out_frame)

            if not resources.output.paces:
                elapsed = time.monotonic() - loop_start
                if elapsed < frame_interval:
                    self._stop.wait(frame_interval - elapsed)
