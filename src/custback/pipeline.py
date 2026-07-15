"""Main processing loop and transactional live reconfiguration."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .backgrounds import BlurBackdrop, create_backdrop
from .capture import open_capture
from .compositor import composite
from .config import (
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

    def cancel(self) -> bool:
        """Cancel unless completion already won the request lock."""
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
    ]


class Pipeline:
    def __init__(
        self,
        runtime: RuntimeConfig,
        hub: FrameHub,
        *,
        model_preparation: SegmenterPreparation | None = None,
    ):
        self.runtime = runtime
        self.hub = hub
        self._stop = threading.Event()
        self._startup_done = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._requests: queue.Queue[_PatchRequest | _MutationRequest] = queue.Queue()
        self._lifecycle_lock = threading.Lock()
        self._active_state: ConfigState | None = None
        self._teardown_lock = threading.Lock()
        self._teardown_threads: set[threading.Thread] = set()
        self._deferred_closes: list[tuple[Any, str]] = []
        self._runtime_writer = runtime._coordinator_writer()
        self._model_preparation = model_preparation
        self._fallback_log_states: dict[str, tuple[bool, str]] = {}

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
        self._fail_pending(ReconfigurationUnavailable("pipeline is stopping"))
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                raise ReconfigurationUnavailable(
                    f"pipeline worker did not stop within {timeout:.1f}s"
                )
        with self._teardown_lock:
            teardown_threads = tuple(self._teardown_threads)
        for teardown in teardown_threads:
            teardown.join(max(0.0, deadline - time.monotonic()))
        with self._teardown_lock:
            survivors = [worker for worker in self._teardown_threads if worker.is_alive()]
        if survivors:
            raise ReconfigurationUnavailable(
                f"{len(survivors)} resource teardown worker(s) did not stop "
                f"within {timeout:.1f}s"
            )
        if self._error is not None:
            raise self._error

    @property
    def running(self) -> bool:
        return (
            self._error is None
            and self._thread is not None
            and self._thread.is_alive()
        )

    def apply_config_patch(
        self,
        patch: dict[str, Any],
        timeout: float = 5.0,
        *,
        origin: str = "internal",
    ) -> ConfigState:
        """Validate, activate, commit, and acknowledge a hot configuration patch."""
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

        request = _PatchRequest(candidate, base.version, origin=origin)
        return self._submit_patch(request, timeout)

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
        )
        return self._submit_patch(request, timeout)

    def _submit_patch(
        self, request: _PatchRequest, timeout: float
    ) -> ConfigState:
        self._requests.put(request)
        if not request.done.wait(timeout):
            if request.cancel():
                raise ReconfigurationUnavailable(
                    f"pipeline did not acknowledge reconfiguration within {timeout:.1f}s"
                )
            # Completion won the lock concurrently; wait for its event publish.
            request.done.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:  # defensive invariant
            raise ReconfigurationUnavailable("pipeline returned no configuration result")
        return request.result

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
        self._requests.put(request)
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
            backdrop = create_backdrop(
                cfg.background,
                video_max_width=cfg.api.uploads.video_max_width,
                video_max_height=cfg.api.uploads.video_max_height,
            )
            if backdrop is not None:
                startup.callback(_safe_close, backdrop, "backdrop")
            output = open_output(
                cfg.output, cfg.camera.width, cfg.camera.height
            )
            startup.callback(_safe_close, output, "video output")
            resources = _Resources(
                cfg, state.version, capture, segmenter, refiner, backdrop, output
            )
            startup.pop_all()
            return resources

    def _run(self) -> None:
        resources: _Resources | None = None
        try:
            state = self.runtime.read()
            resources = self._open_resources(state)
            self._active_state = state
            if state.config.background.mode == "remote":
                self.hub.clear_remote_frames()
            initial_output = self._preflight(resources)
            self._update_identity_stats(resources)
            self._startup_done.set()
            self._loop(resources, initial_output=initial_output)
        except BaseException as exc:
            log.exception("pipeline crashed")
            self._error = exc
        finally:
            self._fail_pending(ReconfigurationUnavailable("pipeline worker stopped"))
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
            request.fail(error)

    # -- reconfiguration ----------------------------------------------
    def _stage_activation(
        self, resources: _Resources, candidate: AppConfig
    ) -> _Activation:
        activation = _Activation(candidate=candidate)
        old_cfg = resources.cfg
        try:
            # Every segmentation patch receives an isolated backend/refiner.
            # Candidate trials therefore cannot advance the working backend's
            # recurrent state or temporal mask on failure.
            if candidate.segmentation != old_cfg.segmentation:
                activation.segmenter = create_segmenter(candidate.segmentation)
                activation.replace_segmenter = True
                activation.refiner = refiner_for(
                    candidate.segmentation, activation.segmenter
                )
            else:
                activation.refiner = resources.refiner

            if candidate.background != old_cfg.background:
                activation.backdrop = create_backdrop(
                    candidate.background,
                    video_max_width=candidate.api.uploads.video_max_width,
                    video_max_height=candidate.api.uploads.video_max_height,
                )
                activation.replace_backdrop = True
            else:
                activation.backdrop = resources.backdrop
        except Exception as exc:
            activation.discard()
            raise ActivationError(str(exc)) from exc
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
        background_changed = activation.candidate.background != old_cfg.background
        compositing_changed = activation.candidate.compositing != old_cfg.compositing
        if not (segmentation_changed or background_changed or compositing_changed):
            return
        try:
            if segmentation_changed:
                mask = activation.refiner.refine(
                    activation.segmenter.segment(frame), frame
                )
                if (
                    not isinstance(mask, np.ndarray)
                    or mask.shape != frame.shape[:2]
                    or not np.isfinite(mask).all()
                ):
                    raise ValueError("candidate segmenter returned an invalid mask")
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
                self.hub.clear_remote_frames()
            except Exception:
                log.exception("cannot clear remote frames after mode switch")
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
                request.done.set()
                return
        current = self.runtime.read()
        if current.version != request.expected_version or resources.version != request.expected_version:
            request.fail(ConfigConflictError(request.expected_version, current.version))
            return
        try:
            activation = self._stage_activation(
                resources, request.activation_candidate or request.candidate
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
            if "activation" in locals():
                activation.discard()
            request.fail(exc)
            return

        old_backdrop = old_segmenter = None
        old_cfg: AppConfig | None = None
        with request.lock:
            if request.cancelled:
                activation.discard()
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
                activation.discard()
                request.error = ConfigConflictError(
                    exc.expected_version, exc.current_version
                )
            except BaseException as exc:
                activation.discard()
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
                    raise ConfigConflictError(
                        resources.version, request.result.version
                    )
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
            log.exception("cannot schedule %s teardown; deferring until shutdown", label)

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
        output_fallback = bool(
            getattr(resources.output, "fallback_active", False)
        )
        segmentation_fallback = (
            cfg.segmentation.backend == "auto"
            and isinstance(resources.segmenter, HeuristicSegmenter)
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
                cfg.background.remote_fallback_mode
                if cfg.background.mode == "remote"
                else ""
            ),
            config_version=resources.version,
            **video_stats,
        )
        self._log_fallback_transition(
            "output",
            output_fallback,
            getattr(resources.output, "fallback_reason", "")
            if output_fallback
            else "",
        )
        self._log_fallback_transition(
            "segmentation",
            segmentation_fallback,
            "ml-backend-unavailable" if segmentation_fallback else "",
        )

    def _log_fallback_transition(
        self, kind: str, active: bool, reason: str
    ) -> None:
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
    def _emergency_blur(frame: np.ndarray) -> np.ndarray:
        """Produce a privacy-safe full-frame fallback without segmentation."""
        try:
            emergency = BlurBackdrop(101)
            emergency.set_source_frame(frame)
            blurred = emergency.frame(frame.shape[1], frame.shape[0])
            if (
                blurred.shape == frame.shape
                and blurred.dtype == np.uint8
                and not np.array_equal(blurred, frame)
            ):
                return blurred
        except Exception:
            log.exception("emergency full-frame blur failed; using solid frame")
        # Destroy all spatial detail if even the blur backend is unavailable
        # or if a uniform source made the blur byte-identical to raw capture.
        mean_bgr = frame.astype(np.float32).mean(axis=(0, 1)).astype(np.uint8)
        solid = np.full_like(frame, mean_bgr)
        if np.array_equal(solid, frame):
            # XOR by the high bit is deterministic and guarantees a different
            # value for every uint8 channel, including all-black/all-white.
            solid = np.full_like(frame, np.bitwise_xor(mean_bgr, 0x80))
        return solid

    def _render_local_mode(
        self, resources: _Resources, frame: np.ndarray, *, preflight: bool = False
    ) -> np.ndarray:
        mode = resources.cfg.background.mode
        if mode == "passthrough":
            return frame
        if mode == "remote" and isinstance(resources.segmenter, NullSegmenter):
            return self._emergency_blur(frame)
        out, _ = self._local_composite(
            resources,
            frame,
            # Startup and activation must expose a broken candidate rather than
            # silently accepting it through the runtime emergency path.
            privacy_safe=mode == "remote" and not preflight,
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
        if resources.cfg.background.mode == "passthrough":
            # Passthrough does not need a mask to render, but the segmenter is
            # already part of the hot-swappable resource generation. Exercise
            # it now so a later background-only PATCH cannot acknowledge and
            # then discover that the never-used backend is broken.
            mask = resources.refiner.refine(
                resources.segmenter.segment(frame), frame
            )
            if (
                not isinstance(mask, np.ndarray)
                or mask.shape != frame.shape[:2]
                or not np.isfinite(mask).all()
            ):
                raise ActivationError("segmenter returned an invalid mask")
        out = self._render_local_mode(resources, frame, preflight=True)
        self._validate_output_frame(out, frame)
        # Sending verifies the actual output backend contract. This frame is
        # deliberately not published, but it is a real consumed/sent frame and
        # therefore participates in the public counter meanings.
        resources.output.send(out)
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
        )
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
            mask = resources.refiner.refine(
                resources.segmenter.segment(frame), frame
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
                timings["background_ms"] = (
                    time.monotonic_ns() - started
                ) / 1_000_000.0
            edge_fg = (
                resources.segmenter.last_foreground
                if cfg.compositing.use_model_foreground
                else None
            )
            started = time.monotonic_ns()
            rendered = self._privacy_checked(
                composite(
                    frame,
                    bg,
                    mask,
                    light_wrap=cfg.compositing.light_wrap,
                    edge_foreground=edge_fg,
                ),
                frame,
                privacy_safe,
            )
            if timings is not None:
                timings["composite_ms"] = (
                    time.monotonic_ns() - started
                ) / 1_000_000.0
            return (
                rendered,
                "",
            )
        except Exception:
            if not privacy_safe:
                raise
            log.exception("local remote-mode fallback failed")
            return self._emergency_blur(frame), "local-failure"

    def _privacy_checked(
        self, candidate: np.ndarray, frame: np.ndarray, privacy_safe: bool
    ) -> np.ndarray:
        """Ensure a remote fallback can never be the raw capture verbatim."""
        if not privacy_safe:
            return candidate
        self._validate_output_frame(candidate, frame)
        if np.array_equal(candidate, frame):
            return self._emergency_blur(frame)
        return candidate

    def _loop(
        self, resources: _Resources, *, initial_output: np.ndarray | None = None
    ) -> None:
        frame_interval = 1.0 / resources.cfg.output.fps
        fps_window: deque[float] = deque()
        initial_sends = 1 if initial_output is not None else 0
        frames_in = frames_out = initial_sends
        remote_used = fallback_count = 0
        repeated_frames = deadline_misses = 0
        last_output = initial_output
        fallback_active = False
        fallback_reason = ""
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
                        remote_used += 1
                        fallback_reason = ""
                    else:
                        out_frame, local_reason = self._local_composite(
                            resources,
                            frame,
                            privacy_safe=True,
                            timings=timings,
                        )
                        if local_reason:
                            fallback_reason = local_reason
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
                last_output = out_frame
            else:
                if last_output is None:
                    self._stop.wait(min(0.01, frame_interval))
                    continue
                out_frame = last_output
                repeated_frames += 1

            send_started = time.monotonic_ns()
            resources.output.send(out_frame)
            output_send_ms = (
                time.monotonic_ns() - send_started
            ) / 1_000_000.0
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
                        resources.cfg.background.remote_fallback_mode,
                        fallback_reason,
                    )
                elif previous_remote_fallback[0]:
                    log.info("remote renderer recovered; local fallback inactive")
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
                capture_fps_reported=getattr(
                    capture_health, "fps_reported", None
                ),
                capture_frames_read=getattr(capture_health, "frames_read", frames_in),
                capture_dropped_frames=getattr(
                    capture_health, "dropped_frames", 0
                ),
                capture_read_failures=getattr(
                    capture_health, "read_failures", 0
                ),
                capture_restarts=getattr(capture_health, "restarts", 0),
                capture_stalled=getattr(capture_health, "stalled", False),
                capture_frame_age_ms=getattr(
                    capture_health, "frame_age_ms", None
                ),
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
