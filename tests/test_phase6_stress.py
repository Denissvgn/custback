"""Deterministic, bounded Phase 6 lifecycle and storage stress gates."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
from fastapi import HTTPException

import custback.avatar.audio2face as audio2face_mod
from custback.api.server import _UploadLimits, _UploadStore
from custback.api.streaming import ConnectionLimiter
from custback.avatar.audio2face import Audio2FaceDriver
from custback.avatar.config import Audio2FaceConfig, AvatarConfig, AvatarRuntime
from custback.avatar.service import AvatarService, _RenderPublication
from custback.config import (
    AppConfig,
    ConfigVersionConflictError,
    RuntimeConfig,
)
from custback.storage_tx import OwnershipLedger


STRESS_FAMILY_IDS = (
    "concurrent-patch-activation",
    "session-cancellation-reconnect",
    "stream-connection-caps",
    "upload-reservations",
    "cleanup-retries",
    "audio2face-shutdown",
)
DEFAULT_STRESS_ITERATIONS = 100
DEFAULT_STRESS_SEED = 0xC057BAC6


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw, 0)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


STRESS_ITERATIONS = _positive_environment_integer(
    "CUSTBACK_STRESS_ITERATIONS", DEFAULT_STRESS_ITERATIONS
)
STRESS_SEED = _positive_environment_integer("CUSTBACK_STRESS_SEED", DEFAULT_STRESS_SEED)


def _iteration_seed(family_id: str, iteration: int) -> int:
    material = f"{STRESS_SEED}:{family_id}:{iteration}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@contextmanager
def _stress_iteration(family_id: str, iteration: int) -> Iterator[random.Random]:
    seed = _iteration_seed(family_id, iteration)
    try:
        yield random.Random(seed)
    except Exception as exc:
        raise AssertionError(
            f"stress family {family_id!r} failed at iteration {iteration}; "
            f"replay with CUSTBACK_STRESS_SEED={STRESS_SEED} "
            f"CUSTBACK_STRESS_ITERATIONS={STRESS_ITERATIONS} "
            f"(iteration_seed={seed})"
        ) from exc


def _join_workers(workers: list[threading.Thread]) -> None:
    for worker in workers:
        worker.join(1.0)
    assert not [worker.name for worker in workers if worker.is_alive()]


def test_stress_manifest_contract() -> None:
    """The executable families exactly match the reviewed Phase 6 manifest."""

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "scripts" / "release" / "required-gates.json").read_text()
    )
    definitions = manifest["stress_families"]
    assert tuple(entry["id"] for entry in definitions) == STRESS_FAMILY_IDS
    assert all(
        entry["minimum_iterations"] <= DEFAULT_STRESS_ITERATIONS
        for entry in definitions
    )


def test_stress_concurrent_patch_activation() -> None:
    """concurrent-patch-activation: one CAS wins and every worker terminates."""

    family_id = STRESS_FAMILY_IDS[0]
    for iteration in range(STRESS_ITERATIONS):
        with _stress_iteration(family_id, iteration) as rng:
            runtime = RuntimeConfig(AppConfig())
            writer = runtime._coordinator_writer()
            base = runtime.read()
            colors = [
                tuple(rng.randrange(256) for _channel in range(3))
                for _candidate in range(2)
            ]
            if colors[0] == colors[1]:
                colors[1] = ((colors[1][0] + 1) % 256, *colors[1][1:])
            candidates = [
                base.config.patched(
                    {"background": {"mode": "color", "color": list(color)}}
                )
                for color in colors
            ]
            winner_index = rng.randrange(2)
            loser_index = 1 - winner_index
            activation_entered = threading.Event()
            loser_started = threading.Event()
            release_activation = threading.Event()
            committed = []
            errors: list[BaseException] = []
            result_lock = threading.Lock()

            def winner() -> None:
                try:

                    def activate(version: int) -> None:
                        assert version == 1
                        activation_entered.set()
                        assert release_activation.wait(1.0)

                    state = writer.commit_with_activation(
                        candidates[winner_index], base.version, activate
                    )
                    with result_lock:
                        committed.append(state)
                except BaseException as exc:
                    with result_lock:
                        errors.append(exc)

            def loser() -> None:
                try:
                    assert activation_entered.wait(1.0)
                    loser_started.set()
                    writer.commit(candidates[loser_index], base.version)
                except BaseException as exc:
                    with result_lock:
                        errors.append(exc)

            workers = [
                threading.Thread(
                    target=winner,
                    name=f"stress-patch-winner-{iteration}",
                ),
                threading.Thread(
                    target=loser,
                    name=f"stress-patch-loser-{iteration}",
                ),
            ]
            for worker in workers:
                worker.start()
            assert activation_entered.wait(1.0)
            assert loser_started.wait(1.0)
            release_activation.set()
            _join_workers(workers)

            conflicts = [
                error
                for error in errors
                if isinstance(error, ConfigVersionConflictError)
            ]
            unexpected = [error for error in errors if error not in conflicts]
            assert not unexpected
            assert len(conflicts) == 1
            assert len(committed) == 1
            state = runtime.read()
            assert state.version == committed[0].version == 1
            assert state.config.background.color == colors[winner_index]


def test_stress_session_cancellation_reconnect() -> None:
    """session-cancellation-reconnect: stale epochs never publish."""

    family_id = STRESS_FAMILY_IDS[1]
    runtime = AvatarRuntime(AvatarConfig.from_dict({"driver": {"backend": "idle"}}))
    service = AvatarService(runtime)
    expected_publications = 0
    try:
        for iteration in range(STRESS_ITERATIONS):
            with _stress_iteration(family_id, iteration) as rng:
                first = service._begin_render_session()
                assert service._account_completed_send(first, 0)
                stale = _RenderPublication(
                    frame=np.full((1, 1, 3), iteration % 256, dtype=np.uint8),
                    version=service._components.version,
                    driver_ms=0.0,
                    render_ms=0.0,
                    face_present=True,
                    width=1,
                    height=1,
                    session_epoch=first.epoch,
                    frame_sequence=0,
                )
                cancel_first = bool(rng.randrange(2))
                start = threading.Event()
                first_operation_done = threading.Event()
                publication_results: list[bool] = []

                def cancel() -> None:
                    assert start.wait(1.0)
                    if not cancel_first:
                        assert first_operation_done.wait(1.0)
                    service._invalidate_render_session(first)
                    if cancel_first:
                        first_operation_done.set()

                def publish() -> None:
                    assert start.wait(1.0)
                    if cancel_first:
                        assert first_operation_done.wait(1.0)
                    publication_results.append(
                        service._publish_render_if_current(stale)
                    )
                    if not cancel_first:
                        first_operation_done.set()

                workers = [
                    threading.Thread(
                        target=cancel,
                        name=f"stress-session-cancel-{iteration}",
                    ),
                    threading.Thread(
                        target=publish,
                        name=f"stress-session-publish-{iteration}",
                    ),
                ]
                for worker in workers:
                    worker.start()
                start.set()
                _join_workers(workers)
                assert publication_results == [not cancel_first]
                expected_publications += int(not cancel_first)
                assert service._active_session_lease is None
                assert not first.active

                second = service._begin_render_session()
                assert second.epoch > first.epoch
                assert not service._publish_render_if_current(stale)
                assert service._account_completed_send(second, 0)
                current = replace(
                    stale,
                    session_epoch=second.epoch,
                    frame_sequence=0,
                )
                assert service._publish_render_if_current(current)
                expected_publications += 1
                service._invalidate_render_session(second)
                assert service._active_session_lease is None
                assert not second.active

        stats = service.stats_dict()
        assert stats["frames_rendered"] == expected_publications
        assert stats["frames_sent"] == STRESS_ITERATIONS * 2
    finally:
        active = service._active_session_lease
        if active is not None:
            service._invalidate_render_session(active)
        service.close()
    assert service._closed
    assert service._live_asset_threads() == ()
    assert service._active_session_lease is None


def test_stress_stream_connection_caps() -> None:
    """stream-connection-caps: admission never exceeds the configured cap."""

    family_id = STRESS_FAMILY_IDS[2]
    for iteration in range(STRESS_ITERATIONS):
        with _stress_iteration(family_id, iteration) as rng:
            maximum = rng.randrange(1, 6)
            attempts = maximum + rng.randrange(1, 5)
            limiter = ConnectionLimiter(maximum)
            start = threading.Event()
            leases = []
            denied = 0
            result_lock = threading.Lock()

            def acquire() -> None:
                nonlocal denied
                assert start.wait(1.0)
                lease = limiter.try_acquire()
                with result_lock:
                    if lease is None:
                        denied += 1
                    else:
                        leases.append(lease)

            workers = [
                threading.Thread(
                    target=acquire,
                    name=f"stress-stream-{iteration}-{attempt}",
                )
                for attempt in range(attempts)
            ]
            for worker in workers:
                worker.start()
            start.set()
            _join_workers(workers)
            assert len(leases) == maximum
            assert denied == attempts - maximum
            assert limiter.active == maximum

            rng.shuffle(leases)
            for lease in leases:
                lease.release()
                lease.release()
            assert limiter.active == 0


def test_stress_upload_reservations(tmp_path: Path) -> None:
    """upload-reservations: contended quota charges leave no ownership."""

    family_id = STRESS_FAMILY_IDS[3]
    charge = 64
    store = _UploadStore(
        tmp_path / "uploads",
        _UploadLimits(storage_max_bytes=charge, max_files=2),
    )
    store.ensure_directory()
    try:
        for iteration in range(STRESS_ITERATIONS):
            with _stress_iteration(family_id, iteration):
                paths = [
                    store.directory / f".upload-{iteration:030x}{index:x}.part"
                    for index in range(2)
                ]
                reserved = threading.Barrier(2, timeout=1.0)
                charged = threading.Barrier(2, timeout=1.0)
                accepted: list[Path] = []
                rejected: list[Path] = []
                errors: list[BaseException] = []
                result_lock = threading.Lock()

                def reserve(path: Path) -> None:
                    owns_record = False
                    try:
                        store._reserve_file(path)
                        owns_record = True
                        reserved.wait()
                        try:
                            store._reserve_bytes(charge, path)
                        except HTTPException as exc:
                            assert exc.status_code == 507
                            with result_lock:
                                rejected.append(path)
                        else:
                            with result_lock:
                                accepted.append(path)
                        charged.wait()
                    except BaseException as exc:
                        with result_lock:
                            errors.append(exc)
                    finally:
                        if owns_record:
                            store._abandon_unbound(path)

                workers = [
                    threading.Thread(
                        target=reserve,
                        args=(path,),
                        name=f"stress-upload-{iteration}-{index}",
                    )
                    for index, path in enumerate(paths)
                ]
                for worker in workers:
                    worker.start()
                _join_workers(workers)
                assert not errors
                assert len(accepted) == len(rejected) == 1
                assert store._reserved_bytes == 0
                assert store._reserved_files == 0
                assert store._active_temps == set()
                assert store._cleanup_pending == {}
                assert store._ledger.records == ()
                assert list(store.directory.iterdir()) == []
    finally:
        store._ledger.close()


def test_stress_cleanup_retries(tmp_path: Path) -> None:
    """cleanup-retries: transient failures retain charge until deletion."""

    family_id = STRESS_FAMILY_IDS[4]
    root = tmp_path / "cleanup"
    root.mkdir(mode=0o700)
    ledger = OwnershipLedger(root)
    try:
        for iteration in range(STRESS_ITERATIONS):
            with _stress_iteration(family_id, iteration) as rng:
                path = root / f"owned-{iteration}"
                payload = bytes([iteration % 256]) * rng.randrange(1, 65)
                record = ledger.begin(
                    path,
                    kind="file",
                    reserved_bytes=len(payload),
                    reserved_slots=1,
                )
                path.write_bytes(payload)
                ledger.bind(record, path)
                ledger.mark_cleanup(record)
                injected_failures = rng.randrange(4)
                failures_remaining = injected_failures
                remove_calls = 0

                def remove(owned: Path, kind: str) -> None:
                    nonlocal failures_remaining, remove_calls
                    assert kind == "file"
                    remove_calls += 1
                    if failures_remaining:
                        failures_remaining -= 1
                        raise PermissionError("deterministic transient cleanup")
                    owned.unlink()

                ledger.cleanup(record, remove)
                if injected_failures:
                    assert ledger.records == (record,)
                    assert ledger.reserved_bytes == len(payload)
                    assert ledger.reserved_slots == 1
                    assert len(ledger.pending_paths()) == 1
                for _retry in range(4):
                    if not ledger.records:
                        break
                    pending = ledger.retry_cleanup(remove)
                    assert len(pending) <= 1

                assert failures_remaining == 0
                assert remove_calls <= 4
                assert not path.exists()
                assert ledger.pending_paths() == ()
                assert ledger.records == ()
                assert ledger.reserved_bytes == 0
                assert ledger.reserved_slots == 0
                assert ledger.active_identities() == frozenset()
                assert not list(root.glob(".custback-cleanup-*"))
    finally:
        ledger.close()


def test_stress_audio2face_shutdown(monkeypatch) -> None:
    """audio2face-shutdown: repeated close joins every owned native handle."""

    family_id = STRESS_FAMILY_IDS[5]
    current: dict[str, object] = {}
    lifecycle_threads: list[threading.Thread] = []
    real_thread_factory = audio2face_mod._LIFECYCLE_THREAD_FACTORY

    def tracked_thread(*args, **kwargs):
        worker = real_thread_factory(*args, **kwargs)
        lifecycle_threads.append(worker)
        return worker

    monkeypatch.setattr(audio2face_mod, "_LIFECYCLE_THREAD_FACTORY", tracked_thread)
    monkeypatch.setattr(audio2face_mod, "_CLOSE_TIMEOUT_S", 0.5)
    monkeypatch.setattr(audio2face_mod, "_load_protocol", lambda: current["protocol"])
    monkeypatch.setattr(
        audio2face_mod,
        "create_audio_source",
        lambda _config: current["source"],
    )

    for iteration in range(STRESS_ITERATIONS):
        with _stress_iteration(family_id, iteration) as rng:
            entered = threading.Event()
            released = threading.Event()

            class Source:
                close_calls = 0

                def close(self) -> None:
                    self.close_calls += 1

            class Call:
                cancel_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    entered.set()
                    released.wait()
                    raise StopIteration

                def cancel(self) -> bool:
                    self.cancel_calls += 1
                    released.set()
                    return True

            class Channel:
                close_calls = 0

                def close(self) -> None:
                    self.close_calls += 1

            source = Source()
            call = Call()
            channel = Channel()

            class Stub:
                def __init__(self, _channel) -> None:
                    pass

                def ProcessAudioStream(self, _requests):
                    return call

            current["source"] = source
            current["protocol"] = SimpleNamespace(
                grpc=SimpleNamespace(
                    insecure_channel=lambda _target, **_kwargs: channel
                ),
                stub_class=Stub,
            )
            driver = Audio2FaceDriver(Audio2FaceConfig(url="grpc://127.0.0.1:52000"))
            worker: threading.Thread | None = None
            first_lifecycle_thread = len(lifecycle_threads)
            try:
                driver.start()
                assert entered.wait(1.0)
                worker = driver._worker
                assert worker is not None
                driver.close()
                for _duplicate in range(rng.randrange(1, 4)):
                    driver.close()
            finally:
                released.set()
                if worker is not None:
                    worker.join(1.0)
                if driver._worker is not None:
                    with contextlib.suppress(Exception):
                        driver.close()

            owned_threads = lifecycle_threads[first_lifecycle_thread:]
            _join_workers(owned_threads)
            assert len(owned_threads) == 3
            assert worker is not None and not worker.is_alive()
            assert source.close_calls == 1
            assert call.cancel_calls == 1
            assert channel.close_calls == 1
            assert driver._worker is None
            assert driver._source is None
            assert driver._call is None
            assert driver._channel is None
            assert driver._owned_generation is None
            assert driver._interruptions == []
            assert driver._survivors == []
